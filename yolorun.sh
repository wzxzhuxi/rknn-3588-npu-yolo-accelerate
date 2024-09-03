#!/bin/bash

# 定义模型文件路径和视频文件路径
MODEL_FILE1="./weights/yolov5s.rknn"
VIDEO_FILE1="./720p60hz.mp4"

# 启动程序处理一个视频，默认程序开启4个线程
./build/yolov5_thread_pool $MODEL_FILE1 $VIDEO_FILE1 4 
# 等待所有程序执行完毕
wait

echo "All YOLOv5 video processing completed."
