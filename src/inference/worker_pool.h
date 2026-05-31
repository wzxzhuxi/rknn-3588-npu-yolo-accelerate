// N 个 Yolov5Detector 实例的线程池. 每个 worker 绑一个 NPU 核 (i % 3 round-robin),
// 生产者 (MPP 回调) submit, 消费者 (主线程) get, 帧 id 维持原序.

#ifndef RK3588_DEMO_WORKER_POOL_H
#define RK3588_DEMO_WORKER_POOL_H

#include "yolov5_detector.h"

#include <opencv2/core/core.hpp>   // cv::Mat 在 FrameItem 里
#include <atomic>
#include <string>
#include <vector>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>

// 队列元素 — 已在 MPP 回调里 RGA-resize 好的 640x640 RGB888 数据.
// src_w/src_h 是原图尺寸, 给 postprocess 反算 bbox scale 用.
struct FrameItem
{
    cv::Mat rgb_640;   // CV_8UC3, 640x640, RGB888 顺序, refcount 共享数据
    int src_w;
    int src_h;
};

class WorkerPool
{
public:
    WorkerPool();
    ~WorkerPool();

    // 启动: 加载 num_threads 份模型实例 (每个绑一个 NPU 核 round-robin), 起 worker 线程
    nn_error_e Start(std::string &model_path, int num_threads = 12);

    // 生产端: 提交一帧 (队列满则阻塞, 防内存爆炸)
    nn_error_e SubmitFrame(const FrameItem &frame, int id);

    // 消费端: 按 id 取检测结果, 5s 超时
    nn_error_e GetResult(std::vector<Detection> &objects, int id);

    // 停所有 worker 线程 (析构里会自动调一次, 也可显式提前停)
    void Stop();

    // 累计 Run 失败次数 (NPU 短暂错误等). 主程序据此判断 exit code.
    long RunErrors() const { return run_errors_.load(std::memory_order_relaxed); }

private:
    void Worker(int id);

    std::queue<std::pair<int, FrameItem>>          tasks;            // <id, frame> 任务队列
    std::vector<std::shared_ptr<Yolov5Detector>>   yolov5_instances; // 每 worker 一份模型实例
    std::map<int, std::vector<Detection>>          results;          // <id, objects> 结果
    std::vector<std::thread>                       threads;
    std::mutex              mtx1;   // 保护 tasks
    std::mutex              mtx2;   // 保护 results
    std::condition_variable cv_task, cv_result;
    bool stop;
    std::atomic<long>       run_errors_{0};
};

#endif // RK3588_DEMO_WORKER_POOL_H
