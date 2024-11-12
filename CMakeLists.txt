cmake_minimum_required(VERSION 3.1.0)
project(server_ibvs)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find Eigen3
find_package(Eigen3 REQUIRED)

# Include directories for OpenCV and Eigen3
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIRS})

# Add the executable
add_executable(server_ibvs src/main.cpp)


# Link the libraries (OpenCV, Eigen3, and ws2_32)
target_link_libraries(server_ibvs ${OpenCV_LIBS} Eigen3::Eigen ws2_32)
