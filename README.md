# rk3588 Detect Accelerate

[English](README.md) | [简体中文](README.zh.md)

## Project Overview

This project is an improved version of [rknn-cpp-Multithreading](https://github.com/leafqycc/rknn-cpp-Multithreading?tab=readme-ov-file), utilizing a thread pool to accelerate the processing and adding detailed comments to help beginners learn and use it more effectively.

## Main Features

- **Thread Pool Acceleration**: Uses thread pool technology to enhance model processing speed.
- **Educational Comments**: Adds detailed comments to key parts of the code to facilitate understanding and learning for beginners.
- **Open Source Foundation**: Based on an open-source project, inheriting and extending its functionality.

## Instructions

To successfully build and run this project, you need to meet the following requirements:

- **System Dependencies**: OpenCV must be installed on your system.
- **Build Tools**: Use CMake to build the project.

## Model Information

This project uses the official model and converts it using the following tools:

- **Model Conversion Tool**: Uses the official [rknn-toolkit2](https://github.com/rockchip-linux/rknn-toolkit2/tree/master) for model conversion.

## Getting Started

1. Ensure OpenCV and CMake are installed.
2. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/wzxzhuxi/rknn-3588-npu-yolo-accelerate
3. Navigate to the project directory and create a build directory:
   ```bash
   cd rknn-3588-npu-yolo-accelerate-master
   mkdir build && cd build
4. Build the project using CMake:
   ```bash
   cmake ..
   make
5. Run the project:
   ```bash
   ./yolov5_detect <model> <video_source> <num_threads>
