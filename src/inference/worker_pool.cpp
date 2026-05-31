#include "worker_pool.h"

#include <chrono>

#include "utils/logging.h"

WorkerPool::WorkerPool() { stop = false; }

WorkerPool::~WorkerPool()
{
    stop = true;
    cv_task.notify_all();
    for (auto &thread : threads) {
        if (thread.joinable()) thread.join();
    }
}

nn_error_e WorkerPool::Start(std::string &model_path, int num_threads)
{
    // 每个 worker 显式绑到 RK3588 三个 NPU 核之一 (core0/1/2), round-robin.
    // 不绑则 librknnrt 走 RKNN_NPU_CORE_AUTO, N 个 context 可能挤同一个核, 失去多核加速.
    for (int i = 0; i < num_threads; ++i) {
        auto inst = std::make_shared<Yolov5Detector>();
        nn_error_e ret = inst->LoadModel(model_path.c_str());
        if (ret != NN_SUCCESS) {
            NN_LOG_ERROR("WorkerPool::Start: LoadModel failed for worker %d (ret=%d, model=%s)",
                         i, ret, model_path.c_str());
            // 已加载的 instance 在 WorkerPool 析构时随 vector 一起释放, 不需要在这里手动清理
            return ret;
        }
        inst->SetCoreMask(i % 3);   // 必须在 LoadModel 之后
        yolov5_instances.push_back(inst);
    }
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(&WorkerPool::Worker, this, i);
    }
    return NN_SUCCESS;
}

// worker 单帧路径: dequeue → memcpy + rknn_run + postprocess → results[id]
// 直接在 worker 里做 postprocess (尝试过解耦到独立 pool 但 2.2MB heap copy 反而 -12 FPS, 已弃).
//
// 错误处理策略: 单帧 Run 失败不致命 (NPU 可能短暂 hang), worker 继续下一帧.
// 但要 push 空 detections 以维持帧 id 单调递增, 否则 main 线程 GetResult 会卡住.
// 累计错误计数对外暴露给 main, 用于 exit code 判断.
void WorkerPool::Worker(int id)
{
    auto instance = yolov5_instances[id];
    while (!stop) {
        std::pair<int, FrameItem> task;
        {
            std::unique_lock<std::mutex> lock(mtx1);
            cv_task.wait(lock, [&] { return !tasks.empty() || stop; });
            if (stop) return;
            task = tasks.front();
            tasks.pop();
        }

        std::vector<Detection> detections;
        const FrameItem &frame = task.second;
        nn_error_e ret = instance->Run(frame.rgb_640.data, frame.src_w, frame.src_h, detections);
        if (ret != NN_SUCCESS) {
            NN_LOG_ERROR("worker %d: Run failed on frame %d (ret=%d) — pushing empty detections",
                         id, task.first, ret);
            run_errors_.fetch_add(1, std::memory_order_relaxed);
            // detections 此时是空 vector, 下面正常入 results 维持帧序
        }

        {
            std::lock_guard<std::mutex> lock(mtx2);
            results.insert({task.first, std::move(detections)});
            cv_result.notify_one();
        }
    }
}

// 队列长度上限 10 帧, 满则 1ms 退避 — 防止解码器跑太快把内存撑爆
nn_error_e WorkerPool::SubmitFrame(const FrameItem &frame, int id)
{
    while (true) {
        {
            std::lock_guard<std::mutex> lock(mtx1);
            if (tasks.size() <= 10) break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    {
        std::lock_guard<std::mutex> lock(mtx1);
        tasks.push({id, frame});
    }
    cv_task.notify_one();
    return NN_SUCCESS;
}

nn_error_e WorkerPool::GetResult(std::vector<Detection> &objects, int id)
{
    int loop_cnt = 0;
    while (true) {
        {
            std::lock_guard<std::mutex> lock(mtx2);
            auto it = results.find(id);
            if (it != results.end()) {
                objects = std::move(it->second);
                results.erase(it);
                return NN_SUCCESS;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        if (++loop_cnt > 5000) {  // 5s 超时
            NN_LOG_ERROR("GetResult timeout id=%d", id);
            return NN_TIMEOUT;
        }
    }
}

void WorkerPool::Stop()
{
    stop = true;
    cv_task.notify_all();
}
