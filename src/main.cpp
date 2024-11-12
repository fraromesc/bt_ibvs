#include <iostream>
#include <winsock2.h>
#include <string>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#pragma comment(lib,"ws2_32.lib") // Link Winsock

cv::Point findCentroid(const std::vector<cv::Point> &points) {
  cv::Point center(0, 0);
  for(const auto &point : points) {
    center.x += point.x;
    center.y += point.y;
  }
  center.x /= points.size();
  center.y /= points.size();
  return center;
}


double calculateAngle(const cv::Point& point, const cv::Point& center) {
    return atan2(point.y - center.y, point.x - center.x);
}

void sortPointsClockwise(std::vector<cv::Point> &points) {
  auto topLeft = *std::min_element(points.begin(), points.end(), [](const cv::Point &a, const cv::Point &b) {
    return (a.y < b.y) || (a.y == b.y && a.x < b.x);
  });

  cv::Point center = findCentroid(points);

  std::sort(points.begin(), points.end(), [&center, &topLeft](const cv::Point &a, const cv::Point &b) {
    double angleA = calculateAngle(a, center);
    double angleB = calculateAngle(b, center);
    if(a == topLeft) angleA = -1;
    if(b == topLeft) angleB = -1;

    return angleA < angleB;
  });
}

enum class Mode : uint8_t {
    WAITING,
    APPROACH,
    FINE
};

class AutoLanding {

    public:
        double velRef_[6];
        double status_[3];
        Mode currentMode_;
        double fx_, fy_, cx_, cy_;
        double altitude_;
        AutoLanding()
        {

            velRef_[0] = 0.0;
            velRef_[1] = 0.0;
            velRef_[2] = 0.0;
            velRef_[3] = 0.0;
            velRef_[4] = 0.0;
            velRef_[5] = 0.0;

            status_[0] = 0.0;
            status_[1] = 0.0;
            status_[2] = 0.0;

            currentMode_ = Mode::WAITING;

            altitude_ = 0;

            fx_ = 1; 
            fy_ = 1; 
            cx_ = 1; 
            cy_ = 1; 

        }
        int imageCallback(std::vector<char> imageData, int& imgHeight, int& imgWidth){

            /***--- In ---***/
            cv::Mat img(cv::Size(imgWidth, imgHeight), CV_8UC3, (void *)imageData.data());

            /***--- Checks ---***/
            if (currentMode_ == Mode::WAITING) {
                std::array<cv::Mat, 2> masks = getMasks(img);
                status_[0] = static_cast<double>(cv::countNonZero(masks[0])) / (masks[0].rows * masks[0].cols);
                status_[1] = static_cast<double>(cv::countNonZero(masks[1])) / (masks[1].rows * masks[1].cols);
                status_[2] = 0;
                return 2;
            }

            /***--- Corner detection ---***/
            std::array<cv::Mat, 2> masks = getMasks(img);
            std::vector<cv::Point> corners = getSquareCorners(img, currentMode_ == Mode::APPROACH ? masks[0] : masks[1]);
            
            if(corners.size() != 4) 
                return 3;

            sortPointsClockwise(corners);

            /***--- Compute reference ---***/
            if (ref_.empty()) {
                const double offsetWidth = 0; //0.1 * imgWidth; 
                const double offsetHeight = 0; //- 0.25 * imgHeight; 
                const double square_size = 0.3 * imgHeight;
                ref_.emplace_back(imgWidth / 2.0 - square_size - offsetWidth, imgHeight / 2.0 - square_size - offsetHeight);
                ref_.emplace_back(imgWidth / 2.0 + square_size - offsetWidth, imgHeight / 2.0 - square_size - offsetHeight);
                ref_.emplace_back(imgWidth / 2.0 + square_size - offsetWidth, imgHeight / 2.0 + square_size - offsetHeight);
                ref_.emplace_back(imgWidth / 2.0 - square_size - offsetWidth, imgHeight / 2.0 + square_size - offsetHeight);
            }
            for(int i = 0; i < ref_.size(); i++) {
                cv::circle(img, ref_[i], 5, cv::Scalar(255, 255, 0), -1);
                cv::line(img, ref_[i], corners[i], cv::Scalar(255, 255, 0), 2);
            }


            /***--- IBVS ---***/
            Eigen::Matrix<double, 2 * 4, 6> J1;
            Eigen::Matrix<double, 2 * 4, 6> J2;
            Eigen::Matrix<double, 2 * 4, 1> e;
            J1 << computeJacobian(corners[0], altitude_, fx_, fy_, cx_, cy_),
                computeJacobian(corners[1], altitude_, fx_, fy_, cx_, cy_),
                computeJacobian(corners[2], altitude_, fx_, fy_, cx_, cy_),
                computeJacobian(corners[3], altitude_, fx_, fy_, cx_, cy_);
            J2 << computeJacobian(ref_[0], 0.5, fx_, fy_, cx_, cy_),
                computeJacobian(ref_[1], 0.5, fx_, fy_, cx_, cy_),
                computeJacobian(ref_[2], 0.5, fx_, fy_, cx_, cy_),
                computeJacobian(ref_[3], 0.5, fx_, fy_, cx_, cy_);
            e << ref_[0].x - corners[0].x,
                ref_[0].y - corners[0].y,
                ref_[1].x - corners[1].x,
                ref_[1].y - corners[1].y,
                ref_[2].x - corners[2].x,
                ref_[2].y - corners[2].y,
                ref_[3].x - corners[3].x,
                ref_[3].y - corners[3].y;

            Eigen::Matrix<double, 6, 1> nu = 1 * (J1.completeOrthogonalDecomposition().pseudoInverse() + J2.completeOrthogonalDecomposition().pseudoInverse()) * e;

            // Reference velocity on UAV system
            velRef_[0] = nu[0];
            velRef_[1] = nu[1];
            velRef_[2] = -nu[2];
            velRef_[3] = -nu[4];
            velRef_[4] = -nu[3];
            velRef_[5] = -nu[5];

            // Saturation
            float limSat = 0.05; 
            if (velRef_[0] > limSat) {
                velRef_[0] = limSat;
            }
            if (velRef_[1] > limSat) {
                velRef_[1] = limSat;
            }
            if (velRef_[2] > 0.75) {
                velRef_[2] = 0.75;
            }
            if (velRef_[0] < -limSat) {
                velRef_[0] = -limSat;
            }
            if (velRef_[1] < -limSat) {
                velRef_[1] = -limSat;
            }
            if (velRef_[2] < -0.75) {
                velRef_[2] = -0.75;
            }

            /***--- Status ---***/
            status_[0] = static_cast<double>(cv::countNonZero(masks[0])) / (masks[0].rows * masks[0].cols);
            status_[1] = static_cast<double>(cv::countNonZero(masks[1])) / (masks[1].rows * masks[1].cols);
            status_[2] = cv::norm(ref_[0] - corners[0]) + cv::norm(ref_[1] - corners[1]) + cv::norm(ref_[2] - corners[2]) + cv::norm(ref_[3] - corners[3]);

            /***--- Image ---***/
            cv::imshow("Autolanding ibvs", img);
            cv::waitKey(1);

            return 0;

        }


    private:
        std::vector<cv::Point> ref_;
        std::array<cv::Mat, 2> getMasks(cv::Mat& img){
            std::array<cv::Mat, 2> mask;
            cv::Mat hsv;

            cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);

            cv::inRange(hsv, cv::Scalar(90, 50, 50), cv::Scalar(150, 255, 255), mask[0]);

            cv::Mat mask1, mask2;
            cv::inRange(hsv, cv::Scalar(0, 80, 80), cv::Scalar(30, 255, 255), mask1);
            cv::inRange(hsv, cv::Scalar(150, 80, 80), cv::Scalar(180, 255, 255), mask2);
            cv::bitwise_or(mask1, mask2, mask[1]);

            return mask;
        }
        std::vector<cv::Point> getSquareCorners(cv::Mat &img, cv::Mat &mask) {
           cv::Mat overlay;
           cv::cvtColor(mask, overlay, cv::COLOR_GRAY2BGR);
           overlay.setTo(cv::Scalar(255, 255, 255), mask);
           const double alpha = 0.5;
           cv::addWeighted(overlay, alpha, img, 1 - alpha, 0, img);

           std::vector<std::vector<cv::Point>> contours;
           cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

           std::vector<cv::Point> approx;
           for(const std::vector<cv::Point> &contour : contours) {
             cv::approxPolyDP(contour, approx, cv::arcLength(contour, true) * 0.02, true);
             if(approx.size() == 4) {
               for(const cv::Point &point : approx) {
                 cv::circle(img, point, 10, cv::Scalar(127, 0, 255), -1);
               }
               for(size_t i = 0; i < 4; ++i) {
                 cv::line(img, approx[i], approx[(i + 1) % 4], cv::Scalar(255, 128, 0), 2);
               }
               return approx;
             }
           }
           return {};
         }


         inline Eigen::Matrix<double, 2, 6> computeJacobian(const cv::Point &p, const double z, const double fx, const double fy, const double cx, const double cy) 
         {
            const double u = p.x - cx;
            const double v = p.y - cy;
            return (Eigen::Matrix<double, 2, 6>() << -fx / z, 0, u / z, u * v / fy, -(fx * fx + u * u) / fx, (fx / fy) * v,
                    0, -fy / z, v / z, (fy * fy + v * v) / fy, -u * v / fx, -(fy / fx) * u)
                .finished();
        }
};

class TCPServer {
private:
    SOCKET serverSocket;
    SOCKET clientSocket;
    struct sockaddr_in serverAddr, clientAddr;
    int addrLen;

public:
    bool clientConnected_ = false; 
    TCPServer(int port) {
        // Initialize Winsock
        WSADATA wsaData;
        if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
            std::cerr << "Error initializing Winsock: " << WSAGetLastError() << std::endl;
            exit(EXIT_FAILURE);
        }

        // Create server socket
        serverSocket = socket(AF_INET, SOCK_STREAM, 0);
        if (serverSocket == INVALID_SOCKET) {
            std::cerr << "Error creating socket: " << WSAGetLastError() << std::endl;
            WSACleanup();
            exit(EXIT_FAILURE);
        }

        // Configure server structure
        serverAddr.sin_family = AF_INET;
        serverAddr.sin_addr.s_addr = INADDR_ANY; // Listen all interfaces
        serverAddr.sin_port = htons(port); // Port

        // Link socket to port and IP address
        if (bind(serverSocket, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) == SOCKET_ERROR) {
            std::cerr << "Error linking socket: " << WSAGetLastError() << std::endl;
            closesocket(serverSocket);
            WSACleanup();
            exit(EXIT_FAILURE);
        }

        // Server on listening mode
        if (listen(serverSocket, 3) == SOCKET_ERROR) {
            std::cerr << "Error listening socket: " << WSAGetLastError() << std::endl;
            closesocket(serverSocket);
            WSACleanup();
            exit(EXIT_FAILURE);
        }

        std::cout << "TCP Server listening port " << port << std::endl;
    }

    // Method to accept connections
    void AcceptConnection() {
        addrLen = sizeof(struct sockaddr_in);
        clientSocket = accept(serverSocket, (struct sockaddr*)&clientAddr, &addrLen);
        if (clientSocket == INVALID_SOCKET) {
            std::cerr << "Error accepting connection: " << WSAGetLastError() << std::endl;
            closesocket(serverSocket);
            WSACleanup();
            exit(EXIT_FAILURE);
        }
        clientConnected_ = true;
        std::cout << "Connection accepted from: " << inet_ntoa(clientAddr.sin_addr) << std::endl;
    }

    // Method to receive clients data
    std::string ReceiveData() {
        char buffer[1024];
        int bytesReceived = recv(clientSocket, buffer, sizeof(buffer), 0);
        if (bytesReceived == SOCKET_ERROR) {
            std::cerr << "Error receiving data (std::string): " << WSAGetLastError() << std::endl;
            clientConnected_ = false;
            return "";
        }

        buffer[bytesReceived] = '\0'; // Check the end of string
        std::string data(buffer);
        return data;
    }

    void ReceiveData(int& data)
    {
        
        int bytesReceived = recv(clientSocket, (char*)&data, sizeof(data), 0);
        if (bytesReceived == SOCKET_ERROR || bytesReceived == 0) {
            std::cerr << "Error receiving data (int): " << WSAGetLastError() << std::endl;
            clientConnected_ = false;
            return;
        }
    }   
    void ReceiveData(double& data)
    {
        
        int bytesReceived = recv(clientSocket, (char*)&data, sizeof(data), 0);
        if (bytesReceived == SOCKET_ERROR || bytesReceived == 0) {
            std::cerr << "Error receiving data (double): " << WSAGetLastError() << std::endl;
            clientConnected_ = false;
            return;
        }
    }

    void ReceiveData( std::vector<char>& imageBuffer) {
        // At first, receive size of image
        int imageSize = 0;
        int bytesReceived = recv(clientSocket, (char*)&imageSize, sizeof(imageSize), 0);
        if (bytesReceived == SOCKET_ERROR || bytesReceived == 0) {
            std::cerr << "Error receiving data (size image): " << WSAGetLastError() << std::endl;
            clientConnected_ = false;
            return;
        }


        // Resize vector to storage all images
        imageBuffer.resize(imageSize);

        // Receive fragments of 1024 bytes
        int totalReceived = 0;
        int bufferSize = 1024;
        
        while (totalReceived < imageSize) {
            int bytesToReceive = std::min(bufferSize, imageSize - totalReceived);
            char buffer[1024];
            int received = recv(clientSocket, buffer, bytesToReceive, 0);

            if (received == SOCKET_ERROR) {
                std::cerr << "Error receiving data (std::vector<char>):" << WSAGetLastError() << std::endl;
                clientConnected_ = false;
                return;
            }

            // Copy data to image vector
            memcpy(&imageBuffer[totalReceived], buffer, received);
            totalReceived += received;
        }

    }

    void ReceiveData(Mode& data) {
        char dataByte; 
        int bytesReceived = recv(clientSocket, (char*)&dataByte, sizeof(dataByte), 0);
        if (bytesReceived == SOCKET_ERROR || bytesReceived == 0) {
            std::cerr << "Error receiving data (int): " << WSAGetLastError() << std::endl;
            clientConnected_ = false;
            return;
        }
        switch(dataByte){
            case 0: 
                data = Mode::WAITING; 
                break; 
            case 1: 
                data = Mode::APPROACH; 
                break; 
            case 2: 
                data = Mode::FINE; 
                break; 
        }
    }
    // Methods to send data to client
    void SendData(const std::string& data) {
        int sendResult = send(clientSocket, data.c_str(), data.length(), 0);
        if (sendResult == SOCKET_ERROR) {
            std::cerr << "Error sending data (std::string): " << WSAGetLastError() << std::endl;
            clientConnected_ = false;
        }
    }
    void SendData(const double* data, int size) {
        int sendSizeResult = send(clientSocket, (char*)(&size), sizeof(int), 0);
        if (sendSizeResult == SOCKET_ERROR) {
            std::cerr << "Error sending size data (double[]): " << WSAGetLastError() << std::endl;
            clientConnected_ = false;
            return;
        }
        
        int sendDataResult = send(clientSocket, (char*)(data), size * sizeof(double), 0);
        if (sendDataResult == SOCKET_ERROR) {
            std::cerr << "Error sending data (double[]): " << WSAGetLastError() << std::endl;
            clientConnected_ = false;
        }
    }
    
    // Close socket client
    void CloseClientConnection() {
        closesocket(clientSocket);
    }

    // Close server
    void CloseServer() {
        closesocket(serverSocket);
        WSACleanup();
    }
};

int main(int argc, char* argv[]) {
    // Set default port
    int port = 4404;

    // Check if a port argument is provided
    if (argc > 1) {
        port = std::stoi(argv[1]); // Convert argument to an integer
    }

    // Create server at specified port
    TCPServer server(port);

    // Accept connection
    server.AcceptConnection();

    AutoLanding al;
    int imgHeight, imgWidth;

    server.ReceiveData(al.fx_); 
    server.ReceiveData(al.fy_); 
    server.ReceiveData(al.cx_); 
    server.ReceiveData(al.cy_); 
    server.ReceiveData(imgHeight); 
    server.ReceiveData(imgWidth); 

    std::vector<char> imageData; 

    al.status_[0] = 1.1; 
    al.status_[1] = 2.1;
    al.status_[2] = 3.1; 

    al.velRef_[0] = 0; 
    al.velRef_[1] = 0; 
    al.velRef_[2] = 0; 
    al.velRef_[3] = 0; 
    al.velRef_[4] = 0; 
    al.velRef_[5] = 0; 
 
     while (server.clientConnected_) {
        server.ReceiveData(imageData); 
        server.ReceiveData(al.altitude_); 
        server.ReceiveData(al.currentMode_); 
        al.imageCallback(imageData, imgHeight, imgWidth);
        
        server.SendData(al.status_, 3); 
        server.SendData(al.velRef_, 6);       
      }


    // Close connections
    server.CloseClientConnection();
    server.CloseServer();

    return 0;
}

   