// 包含OpenCV库的头文件，这是进行图像处理的基础库
#include <opencv2/opencv.hpp>

// 包含自定义的YOLOv5模型的头文件，用于物体检测
#include "task/yolov5.h"
// 包含日志记录功能的头文件，用于输出日志信息
#include "utils/logging.h"
// 包含绘图相关功能的头文件，用于在图像上绘制结果
#include "draw/cv_draw.h"

// 包含一个管理YOLOv5模型的线程池的头文件，用于并行处理视频帧
#include "task/yolov5_thread_pool.h"

// 定义两个静态全局变量用来跟踪视频帧的处理状态
static int g_frame_start_id = 0; // 读取视频帧的索引
static int g_frame_end_id = 0;   // 模型处理完的帧的索引

// 定义一个指向YOLOv5线程池的全局指针，用来管理线程池
static Yolov5ThreadPool *g_pool = nullptr;
// 定义一个布尔变量，用于标记处理过程何时结束
bool end = false;

// 函数：获取处理结果并统计处理性能
void get_results(int width = 1280, int height = 720, int fps = 30)
{
    // 记录处理开始的时间点，用于计算处理时间和帧率
    auto start_all = std::chrono::high_resolution_clock::now();
    int frame_count = 0;  // 用于统计处理的帧数

    // 循环直到处理结束
    while (true)
    {
        cv::Mat img;  // 创建一个空的图像矩阵用来存放获取的结果
        auto ret = g_pool->getTargetImgResult(img, g_frame_end_id++);  // 从线程池获取处理结果
        // 如果标记结束且没有成功获取到结果，停止线程池并退出循环
        if (end && ret != NN_SUCCESS)
        {
            g_pool->stopAll();
            break;
        }

        // 计算从开始到现在的总处理时间，并每隔1秒输出一次处理性能
        frame_count++;
        auto end_all = std::chrono::high_resolution_clock::now();
        auto elapsed_all_2 = std::chrono::duration_cast<std::chrono::microseconds>(end_all - start_all).count() / 1000.f;
        if (elapsed_all_2 > 1000)
        {
            NN_LOG_INFO("Method2 Time:%fms, FPS:%f, Frame Count:%d", elapsed_all_2, frame_count / (elapsed_all_2 / 1000.0f), frame_count);
            frame_count = 0;
            start_all = std::chrono::high_resolution_clock::now();
        }
    }
    g_pool->stopAll();  // 结束线程池的所有线程
    NN_LOG_INFO("Get results end.");  // 输出结束日志
}

// 函数：读取视频流并将帧提交给线程池进行处理
void read_stream(const char *video_file)
{
    cv::VideoCapture cap(video_file);  // 打开视频文件
    if (!cap.isOpened())
    {
        NN_LOG_ERROR("Failed to open video file: %s", video_file);  // 如果视频打不开，记录错误日志
    }

    // 获取视频的宽度、高度和帧率
    int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int fps = cap.get(cv::CAP_PROP_FPS);
    NN_LOG_INFO("Video size: %d x %d, fps: %d", width, height, fps);  // 输出视频属性

    cv::Mat img;  // 创建一个用于存放每一帧的图像矩阵
    while (true)
    {
        cap >> img;  // 读取一帧
        if (img.empty())  // 如果读取的帧为空，说明视频结束
        {
            NN_LOG_INFO("Video end.");  // 记录视频结束的信息
            end = true;  // 设置结束标志为真
            break;
        }

        // 将读取的帧克隆一份并提交到线程池处理，克隆是必要的以防数据在内存中不连续
        g_pool->submitTask(img.clone(), g_frame_start_id++);
    }
    cap.release();  // 释放视频文件相关资源
}

// 主函数
int main(int argc, char **argv)
{
    // 从命令行参数获取模型文件路径和视频文件路径
    std::string model_file = argv[1];  // 模型文件路径
    const char *video_file = argv[2];  // 视频文件路径
    const int num_threads = (argc > 3) ? atoi(argv[3]) : 12;  // 获取线程数，如果未指定，默认为12

    // 创建线程池实例并设置线程池
    g_pool = new Yolov5ThreadPool();
    g_pool->setUp(model_file, num_threads);

    // 创建并启动读取视频流的线程
    std::thread read_stream_thread(read_stream, video_file);
    // 创建并启动获取处理结果的线程
    std::thread result_thread(get_results, 1280, 720, 25);

    // 等待上述两个线程执行完毕
    read_stream_thread.join();
    result_thread.join();

    return 0;  // 程序结束
}
