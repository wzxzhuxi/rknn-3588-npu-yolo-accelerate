# 项目简介

本项目是基于 [rknn-cpp-Multithreading](https://github.com/leafqycc/rknn-cpp-Multithreading?tab=readme-ov-file) 的改进版本，使用线程池来加速处理过程，并增加了详细的注释，帮助初学者更好地学习和使用。

## 主要特性

- **线程池加速**：利用线程池技术，提高模型处理速度。
- **教育性注释**：为代码中的关键部分添加了详细注释，方便初学者理解和学习。
- **开源基础**：基于开源项目改进，继承并扩展其功能。

## 使用说明

要成功构建和运行此项目，您需要满足以下条件：

- **系统依赖**：系统中必须安装有 OpenCV。
- **构建工具**：使用 CMake 来构建项目。

## 模型信息

本项目使用了官方提供的模型，并通过以下工具进行转换：

- **模型转换工具**：使用官方的 [rknn-toolkit2](https://github.com/rockchip-linux/rknn-toolkit2/tree/master) 进行模型转换。

## 开始使用

1. 确保已安装 OpenCV 和 CMake。
2. 克隆仓库到本地：
   ```bash
   git clone https://github.com/wzxzhuxi/rknn-3588-npu-yolo-accelerate
3. 进入项目目录，并创建构建目录：
   ```bash
   cd rknn-3588-npu-yolo-accelerate-master
   mkdir build && cd build
4. 使用CMake构建项目：
   ```bash
   cmake ..
   make
6. 运行项目：
   ```bash
   ./yolov5_detect 模型 视频源 线程数
