
#include "yolov5_thread_pool.h"
#include "draw/cv_draw.h"
// 构造函数
Yolov5ThreadPool::Yolov5ThreadPool() { stop = false; }

// 析构函数
Yolov5ThreadPool::~Yolov5ThreadPool()
{
    // stop all threads
    stop = true;
    cv_task.notify_all();
    for (auto &thread : threads)
    {
        if (thread.joinable())
        {
            thread.join();
        }
    }
}

// 初始化：加载模型，创建线程，参数：模型路径，线程数量
nn_error_e Yolov5ThreadPool::setUp(std::string &model_path, int num_threads)
{
    // 遍历线程数量，创建模型实例，放入vector
    // 这些线程加载的模型是同一个
    for (size_t i = 0; i < num_threads; ++i)
    {
        // 创建一个Yolov5模型实例
        std::shared_ptr<Yolov5> yolov5 = std::make_shared<Yolov5>();
        // 调用Yolov5的LoadModel方法加载模型，传入模型路径
        yolov5->LoadModel(model_path.c_str());
        // 将模型实例添加到yolov5_instances向量中
        yolov5_instances.push_back(yolov5);
    }
    // 遍历线程数量，创建线程
    for (size_t i = 0; i < num_threads; ++i)
    {
        // 为每个工作线程创建一个新线程，执行worker方法，传入当前线程ID
        threads.emplace_back(&Yolov5ThreadPool::worker, this, i);
    }
    // 返回成功状态
    return NN_SUCCESS;
}

// 线程函数。参数：线程id
void Yolov5ThreadPool::worker(int id)
{
    // 当前线程循环运行，直到接收到停止信号
    while (!stop)
    {
        // 定义一个用于存放任务的变量
        std::pair<int, cv::Mat> task;
        // 获取当前线程对应的Yolov5模型实例
        std::shared_ptr<Yolov5> instance = yolov5_instances[id]; // 获取模型实例
        
        {
            // 锁定任务队列
            std::unique_lock<std::mutex> lock(mtx1);
            // 等待条件变量，直到有任务加入队列或接收到停止信号
            cv_task.wait(lock, [&]
                         { return !tasks.empty() || stop; });
            // 如果接收到停止信号，则退出循环，结束线程
            if (stop)
            {
                return;
            }
            // 从任务队列中取出一个任务
            task = tasks.front();
            // 弹出已取出的任务
            tasks.pop();
        }
        
        // 运行模型进行推理
        std::vector<Detection> detections;
        // 使用取出的任务中的图像进行推理，并将结果保存在detections中
        instance->Run(task.second, detections);

        {
            // 锁定用于存储结果的部分
            std::lock_guard<std::mutex> lock(mtx2);
            // 将检测结果保存到结果集合中
            results.insert({task.first, detections});
            // 使用检测结果对图像进行绘制
            DrawDetections(task.second, detections);
            // 将绘制后的图像保存到img_results中
            img_results.insert({task.first, task.second});
            // 通知等待结果的线程
            cv_result.notify_one();
        }
    }
}

// 提交任务，参数：图片，id（帧号）
nn_error_e Yolov5ThreadPool::submitTask(const cv::Mat &img, int id)
{
    // 如果任务队列中的任务数量大于10，等待，避免内存占用过多
    while (tasks.size() > 10)
    {
        // sleep 1ms
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    {
        // 保存任务
        std::lock_guard<std::mutex> lock(mtx1);  // 使用锁保证线程安全
        tasks.push({id, img});  // 将任务（帧号和对应的图片）添加到任务队列中
    }
    cv_task.notify_one();  // 通知一个正在等待的工作线程有新的任务到来
    return NN_SUCCESS;  // 返回成功状态
}

// 获取结果，参数：检测框，id（帧号）
nn_error_e Yolov5ThreadPool::getTargetResult(std::vector<Detection> &objects, int id)
{
    // 如果没有结果，等待
    while (results.find(id) == results.end())
    {
        // sleep 1ms
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    std::lock_guard<std::mutex> lock(mtx2);
    objects = results[id];
    // remove from map
    results.erase(id);

    return NN_SUCCESS;
}

// 获取结果（图片），参数：图片，id（帧号）
nn_error_e Yolov5ThreadPool::getTargetImgResult(cv::Mat &img, int id)
{
    int loop_cnt = 0;
    // 如果没有结果，等待
    while (img_results.find(id) == img_results.end())
    {
        // 等待 5ms x 1000 = 5s
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        loop_cnt++;
        if (loop_cnt > 1000)
        {
            NN_LOG_ERROR("getTargetImgResult timeout");
            return NN_TIMEOUT;
        }
    }
    std::lock_guard<std::mutex> lock(mtx2);
    img = img_results[id];
    // remove from map
    img_results.erase(id);

    return NN_SUCCESS;
}
// 停止所有线程
void Yolov5ThreadPool::stopAll()
{
    stop = true;
    cv_task.notify_all();
}