# 设置最低版本号
cmake_minimum_required(VERSION 3.11 FATAL_ERROR)
# 设置项目名称
project(rk3588-demo VERSION 0.0.1 LANGUAGES CXX)

# 输出系统信息
message(STATUS "System: ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_VERSION}")

# 设置编译器
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置库架构
set(LIB_ARCH "aarch64")
set(DEVICE_NAME "RK3588")

#  rknn_api 文件夹路径
set(RKNN_API_PATH ${CMAKE_CURRENT_SOURCE_DIR}/librknn_api)
#  rknn_api include 路径
set(RKNN_API_INCLUDE_PATH ${RKNN_API_PATH}/include)
#  rknn_api lib 路径
set(RKNN_API_LIB_PATH ${RKNN_API_PATH}/${LIB_ARCH}/librknnrt.so)

# 所有依赖均在仓库内 — 不依赖任何系统/上层目录, 便于独立 clone 后直接构建.
# 唯一例外是 pthread (glibc 的一部分, 系统提供, 无法本地化).
set(3RDPARTY_PATH ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty)
set(RGA_DIR ${3RDPARTY_PATH}/rga/${DEVICE_NAME})
set(RGA_LIB ${RGA_DIR}/lib/Linux/${LIB_ARCH}/librga.so)
set(MPP_PATH ${3RDPARTY_PATH}/mpp)
set(MPP_LIBS ${MPP_PATH}/Linux/${LIB_ARCH}/librockchip_mpp.so)
set(OpenCV_DIR ${3RDPARTY_PATH}/opencv/opencv-linux-${LIB_ARCH}/share/OpenCV)
find_package(OpenCV REQUIRED)
# 输出OpenCV信息
message(STATUS "include path: ${OpenCV_INCLUDE_DIRS}")


# 用来搜索头文件的目录
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${RKNN_API_INCLUDE_PATH}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${RGA_DIR}/include
    ${MPP_PATH}/include
)

# 构建预处理和后处理库
add_library(nn_process SHARED
            src/preprocess/rga_nv12_to_rgb.cpp
            src/postprocess/yolov5_nms.cpp
)
# 链接库
target_link_libraries(nn_process
    ${OpenCV_LIBS}
    ${RGA_LIB}
)

# MPP 硬解封装(vendor demo 的 mpp_decoder.cpp 一份复制, 见 src/decoder/)
add_library(mpp_decoder SHARED src/decoder/mpp_decoder.cpp)
target_link_libraries(mpp_decoder ${MPP_LIBS})

# 构建自定义封装API库
add_library(rknn_engine SHARED src/engine/rknn_engine.cpp)
# 链接库
target_link_libraries(rknn_engine
    ${RKNN_API_LIB_PATH}
)
# yolov5_lib
add_library(yolov5_lib SHARED src/inference/yolov5_detector.cpp)
# 链接库
target_link_libraries(yolov5_lib
    rknn_engine
    nn_process
)

# 主可执行文件: 真实视频检测全流水 (~175 FPS)
add_executable(yolov5_thread_pool
    src/main.cpp
    src/inference/worker_pool.cpp
    )

target_link_libraries(yolov5_thread_pool
        yolov5_lib
        mpp_decoder
)

# NPU-only naked benchmark — 等量对比参考项目的 main.cc 测法
# 加 OpenCV 仅用于一次性图片加载 (resize / BGR2RGB), 不进 benchmark 循环
add_executable(npu_only_bench src/bench/npu_throughput.cpp)
target_link_libraries(npu_only_bench
    ${RKNN_API_LIB_PATH}
    ${OpenCV_LIBS}
    pthread
)

