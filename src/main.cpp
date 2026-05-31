// 实时 YOLOv5 视频检测主程序
// 流水线: MPP 硬解 → RGA 单步 NV12→RGB888 → WorkerPool 多核 NPU 推理 → 结果按帧序回收
// 端到端峰值 ~175 FPS / NPU 99% 满载 @ yolov5s_relu_pt int8 / 720p60 H.264 / RK3588 三核 NPU

#include <atomic>
#include <chrono>
#include <fstream>
#include <vector>
#include <cstdint>
#include <thread>

#include <opencv2/opencv.hpp>   // cv::Mat 持有 RGA 解出的 RGB888

#include "inference/yolov5_detector.h"
#include "inference/worker_pool.h"
#include "decoder/mpp_decoder.h"
#include "preprocess/rga_nv12_to_rgb.h"
#include "utils/logging.h"

// 帧 id 计数器 — 解码端递增 submit, 收集端按序 get, 维持帧序
static int g_frame_start_id = 0;
static int g_frame_end_id = 0;

static WorkerPool *g_pool = nullptr;
static bool g_end = false;
// 流水线级失败标志 — 由 read_stream 在 fatal 错误时 set, main 据此决定 exit code.
// worker 级错误 (单帧 Run 失败) 不通过这里, 用 WorkerPool::RunErrors() 单独统计.
static std::atomic<bool> g_pipeline_failed{false};

// 结果收集线程: 按帧序取 detection, 每秒打印 FPS + AvgDet/Frame
// AvgDet 是精度信号 — 突然掉到 0 或飙到 64 都意味着 postprocess 出错
void get_results()
{
    auto start_all = std::chrono::high_resolution_clock::now();
    int frame_count = 0;
    long total_detections = 0;

    while (true)
    {
        std::vector<Detection> dets;
        auto ret = g_pool->GetResult(dets, g_frame_end_id);
        // 读流结束 + 取不到结果 → 真正完成
        if (g_end && ret != NN_SUCCESS)
        {
            g_pool->Stop();
            break;
        }
        g_frame_end_id++;
        frame_count++;
        total_detections += static_cast<long>(dets.size());

        auto end_all = std::chrono::high_resolution_clock::now();
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(end_all - start_all).count() / 1000.f;
        if (elapsed_ms > 1000)
        {
            const double fps = frame_count / (elapsed_ms / 1000.0f);
            const double avg_det = frame_count > 0 ? (double)total_detections / frame_count : 0.0;
            NN_LOG_INFO("Time:%.2fms FPS:%.2f Frames:%d AvgDet/Frame:%.2f",
                        elapsed_ms, fps, frame_count, avg_det);
            frame_count = 0;
            total_detections = 0;
            start_all = std::chrono::high_resolution_clock::now();
        }
    }
    g_pool->Stop();
    NN_LOG_INFO("Get results end.");
}

// MPP 解码回调 — MPP 在自己线程里同步调用. NV12 buffer 回调返回后会被复用,
// 必须在回调内完成 RGA 转换并把结果拷到独立 cv::Mat (cv::Mat refcount 跨线程安全).
void on_mpp_frame(void * /*userdata*/,
                  int width_stride, int height_stride,
                  int width, int height,
                  int /*format*/, int /*fd*/, void *data)
{
    constexpr int MODEL_SIZE = 640;
    cv::Mat rgb;
    nv12_to_rgb_mat_rga(static_cast<uint8_t *>(data),
                        width, height,
                        width_stride, height_stride,
                        MODEL_SIZE, MODEL_SIZE,
                        rgb);
    if (rgb.empty()) return;  // RGA 失败已在内部 NN_LOG_ERROR

    FrameItem item;
    item.rgb_640 = rgb;
    item.src_w = width;   // 原图尺寸传给 postprocess 反算 bbox scale
    item.src_h = height;
    g_pool->SubmitFrame(item, g_frame_start_id++);
}

// 把裸 H.264/H.265 码流整文件喂给 MPP 硬解 (chunked, 8KB 一片)
void read_stream(const char *video_file, int video_type)
{
    std::ifstream f(video_file, std::ios::binary);
    if (!f) {
        NN_LOG_ERROR("Failed to open video file: %s", video_file);
        g_pipeline_failed = true;
        g_end = true;
        return;
    }
    std::vector<uint8_t> bs((std::istreambuf_iterator<char>(f)),
                            std::istreambuf_iterator<char>());
    f.close();
    NN_LOG_INFO("Loaded raw H.%d stream: %zu bytes", video_type, bs.size());

    // fps_limit=1000 等同于关闭节流, 让 MPP 尽快出帧 — 测真实吞吐用
    MppDecoder dec;
    if (dec.Init(video_type, 1000, nullptr) < 0) {
        NN_LOG_ERROR("MppDecoder.Init failed (video_type=%d)", video_type);
        g_pipeline_failed = true;
        g_end = true;
        return;
    }
    dec.SetCallback(on_mpp_frame);

    const size_t CHUNK = 8192;
    uint8_t *p = bs.data();
    uint8_t *e = p + bs.size();
    while (p < e) {
        int pkt_eos = 0;
        size_t sz = CHUNK;
        if (p + sz >= e) {
            pkt_eos = 1;
            sz = e - p;
        }
        dec.Decode(p, sz, pkt_eos);
        p += sz;
    }

    NN_LOG_INFO("Video end.");
    g_end = true;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        printf("Usage: %s <rknn_model> <raw_h264_or_h265> [video_type 264|265, default 264] [num_threads, default 12]\n", argv[0]);
        return -1;
    }
    std::string model_file = argv[1];
    const char *video_file = argv[2];
    int video_type = (argc > 3) ? atoi(argv[3]) : 264;
    const int num_threads = (argc > 4) ? atoi(argv[4]) : 12;

    NN_LOG_INFO("yolo_detect: model=%s video=%s type=H.%d threads=%d",
                model_file.c_str(), video_file, video_type, num_threads);

    g_pool = new WorkerPool();
    nn_error_e start_ret = g_pool->Start(model_file, num_threads);
    if (start_ret != NN_SUCCESS) {
        NN_LOG_ERROR("WorkerPool::Start failed (ret=%d) — aborting", start_ret);
        delete g_pool;
        return -1;
    }

    std::thread read_stream_thread(read_stream, video_file, video_type);
    std::thread result_thread(get_results);

    read_stream_thread.join();
    result_thread.join();

    const long run_errs = g_pool->RunErrors();
    delete g_pool;

    // 退出码语义:
    //   0   正常: 流水线初始化通过, 输入读完整, 中途无 NPU 错误
    //   -1  fatal: model 加载失败 / 视频读失败 / MPP 初始化失败 / 中途有 NPU 错误
    if (g_pipeline_failed.load() || run_errs > 0) {
        NN_LOG_ERROR("pipeline finished with errors (pipeline_failed=%d run_errors=%ld)",
                     (int)g_pipeline_failed.load(), run_errs);
        return -1;
    }
    return 0;
}
