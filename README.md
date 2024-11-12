# Fork of bt_ibvs with TCP Communication

This repository is a fork of [raultapia/bt_ibvs](https://github.com/raultapia/bt_ibvs) adapted to use TCP communication instead of ROS. This adaptation maintains the core functionality of the original repository but replaces the ROS-dependent components with TCP-based communication.

## Usage Instructions

To build and run this project, follow these steps:

1. **Create the build directory**:
   ```bash
   mkdir ./build
   ```
2. **Configure the build with CMake**:
   ```bash
   cd build 
   cmake ...
   ```
3. **Compile the project**:
    ```bash
    cmake --build . --config Release
    ```

Please make sure you have CMake installed and configured in your development environment.    
