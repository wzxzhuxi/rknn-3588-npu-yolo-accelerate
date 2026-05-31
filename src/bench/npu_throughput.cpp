// 裸 NPU 吞吐 benchmark — 剥掉所有流水线 (decode/RGA/memcpy/postprocess), 只死循环 rknn_run.
// 用于标定 NPU 算力上限 (~180 FPS for yolov5s_relu_pt @ RK3588), 跟 yolov5_thread_pool 的
// 端到端峰值 (~175 FPS) 对比 → 差距就是真实流水线的 CPU 端开销 (~5 FPS = memcpy + postprocess + 调度).
//
// 配置:
//   - 可选 N 个 rknn_context (绑核 i % 3), 默认 3 个
//   - 可选 M 个 worker threads (i % N 共享 ctx), 默认 8
//   - pass_through=1, flag=0 (默认就是 PRIOR_HIGH)
//   - 输入数据可选: 全零 / bus.jpg / 模拟主流水线 memcpy 负载
//   - 不读 output, 不算 postprocess

#include <rknn_api.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <thread>
#include <vector>

// OpenCV 仅用于一次性加载 + resize 输入图 (不进 benchmark 循环).
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

static constexpr int NUM_CORES = 3;          // RK3588 物理 NPU 核数, 固定
// num_ctxs 在 main 里可配置 — 关键实验变量

static std::atomic<long> g_total_runs{0};
static std::atomic<bool> g_stop{false};

static unsigned char *load_model_file(const char *path, int *out_size)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return nullptr;
    int sz = static_cast<int>(f.tellg());
    f.seekg(0);
    auto *buf = static_cast<unsigned char *>(malloc(sz));
    if (!buf) return nullptr;
    f.read(reinterpret_cast<char *>(buf), sz);
    *out_size = sz;
    return buf;
}

// 是否在每次 rknn_run 前做一次 XOR-copy 模拟主流水线的 input 写入开销.
// 用来诊断: DDR 带宽争夺到底是不是 175→180 那 5 FPS 缺口的元凶 (实验证否, 只 -0.8 FPS).
static bool g_simulate_memcpy = false;

static void worker_loop(rknn_context ctx, uint8_t *input_virt, const uint8_t *src, size_t bytes)
{
    while (!g_stop.load(std::memory_order_relaxed)) {
        if (g_simulate_memcpy) {
            // 跟 yolov5_detector.cpp::copy_uint8_to_int8_shift128 同行为 (NEON XOR 0x80)
#if defined(__ARM_NEON)
            const uint8x16_t bias = vdupq_n_u8(0x80);
            size_t i = 0;
            for (; i + 64 <= bytes; i += 64) {
                vst1q_u8(input_virt + i +  0, veorq_u8(vld1q_u8(src + i +  0), bias));
                vst1q_u8(input_virt + i + 16, veorq_u8(vld1q_u8(src + i + 16), bias));
                vst1q_u8(input_virt + i + 32, veorq_u8(vld1q_u8(src + i + 32), bias));
                vst1q_u8(input_virt + i + 48, veorq_u8(vld1q_u8(src + i + 48), bias));
            }
            for (; i < bytes; ++i) input_virt[i] = src[i] ^ 0x80;
#else
            for (size_t i = 0; i < bytes; ++i) input_virt[i] = src[i] ^ 0x80;
#endif
        }
        int ret = rknn_run(ctx, nullptr);
        if (ret < 0) break;
        g_total_runs.fetch_add(1, std::memory_order_relaxed);
    }
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        printf("Usage: %s <rknn_model> [num_threads=8] [seconds=20] [num_ctxs=3] [image_path] [sim_memcpy=0|1]\n", argv[0]);
        printf("  num_ctxs:    rknn_init context 数量 (绑核 i %% 3)\n");
        printf("  image_path:  可选, 不传则用全零输入\n");
        printf("  sim_memcpy:  每次 rknn_run 前是否 XOR-copy 1.17MB 模拟主流水线 input 写入开销\n");
        return 1;
    }
    const char *model_path = argv[1];
    int num_threads = (argc > 2) ? atoi(argv[2]) : 8;
    int seconds = (argc > 3) ? atoi(argv[3]) : 20;
    int num_ctxs = (argc > 4) ? atoi(argv[4]) : 3;
    const char *image_path = (argc > 5) ? argv[5] : nullptr;
    if (argc > 6 && atoi(argv[6]) != 0) g_simulate_memcpy = true;
    if (num_ctxs < 1) num_ctxs = 1;

    int model_size = 0;
    unsigned char *model_data = load_model_file(model_path, &model_size);
    if (!model_data) {
        printf("Failed to load model: %s\n", model_path);
        return 1;
    }
    printf("Model: %s (%d bytes)\n", model_path, model_size);
    printf("Input: %s\n", image_path ? image_path : "<zeros>");

    std::vector<rknn_context> ctxs(num_ctxs);
    rknn_core_mask core_masks[NUM_CORES] = {
        RKNN_NPU_CORE_0, RKNN_NPU_CORE_1, RKNN_NPU_CORE_2
    };
    std::vector<rknn_tensor_mem *> input_mems(num_ctxs, nullptr);
    // 每个 ctx 拥有自己的 output mems, 跟踪以便 destroy 时调对应 ctx
    std::vector<std::vector<rknn_tensor_mem *>> output_mems(num_ctxs);

    // 一次性预备输入数据 — 不进 benchmark 循环.
    // pass_through=1 要求 int8 layout (模型 native), 用 XOR 0x80 等价 uint8-128 转换.
    // 全零输入也是合法的 int8, 跟 reference 测法等价.
    std::vector<uint8_t> input_data;
    int img_w = 0, img_h = 0;  // 仅用于打印

    for (int i = 0; i < num_ctxs; ++i) {
        // flag = 0 = RKNN_FLAG_PRIOR_HIGH (默认值, 注意不是 high priority 的真正"标志"!)
        int ret = rknn_init(&ctxs[i], model_data, model_size, 0, NULL);
        if (ret < 0) {
            printf("rknn_init ctx %d failed: %d\n", i, ret);
            return 1;
        }
        // 绑核: i % 3 round-robin 到三个物理 NPU 核
        rknn_set_core_mask(ctxs[i], core_masks[i % NUM_CORES]);

        rknn_input_output_num io_num;
        rknn_query(ctxs[i], RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));

        // Input: pass_through=1 (跟参考项目一样, 跳过 driver 的 normalize/quant)
        rknn_tensor_attr in_attr{};
        in_attr.index = 0;
        rknn_query(ctxs[i], RKNN_QUERY_INPUT_ATTR, &in_attr, sizeof(in_attr));
        input_mems[i] = rknn_create_mem(ctxs[i], in_attr.size_with_stride);
        if (!input_mems[i]) {
            printf("rknn_create_mem(input) ctx %d failed\n", i);
            return 1;
        }

        // 第一次进 ctx 0 时一次性准备好输入数据 (image or zeros), 后两个 ctx 复用
        if (i == 0) {
            const uint32_t W = in_attr.dims[2];
            const uint32_t H = in_attr.dims[1];
            const uint32_t C = in_attr.dims[3];
            const uint32_t bytes = in_attr.size_with_stride;
            input_data.assign(bytes, 0);
            if (image_path) {
                cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);  // BGR
                if (img.empty()) {
                    printf("imread failed: %s — falling back to zeros\n", image_path);
                } else {
                    img_w = img.cols; img_h = img.rows;
                    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
                    cv::resize(img, img, cv::Size(W, H));
                    // pass_through=1 配合 model native INT8: 需要 uint8 → int8 (XOR 0x80 等价减 128).
                    // 一次性, 不进 benchmark 循环.
                    const uint32_t plain = W * H * C;
                    for (uint32_t k = 0; k < plain; ++k) {
                        input_data[k] = img.data[k] ^ 0x80;
                    }
                }
            }
        }
        memcpy(input_mems[i]->virt_addr, input_data.data(), in_attr.size_with_stride);

        in_attr.pass_through = 1;
        if (rknn_set_io_mem(ctxs[i], input_mems[i], &in_attr) < 0) {
            printf("rknn_set_io_mem(input) ctx %d failed\n", i);
            return 1;
        }

        // Output mems (跟参考项目不一样, 它根本没绑 output)
        // 但 rknn 在 zero-copy 模式下 *必须* 绑 output mem, 不然 rknn_run 返回错误
        for (uint32_t o = 0; o < io_num.n_output; ++o) {
            rknn_tensor_attr out_attr{};
            out_attr.index = o;
            rknn_query(ctxs[i], RKNN_QUERY_OUTPUT_ATTR, &out_attr, sizeof(out_attr));
            auto *m = rknn_create_mem(ctxs[i], out_attr.size_with_stride);
            if (!m) {
                printf("rknn_create_mem(output[%u]) ctx %d failed\n", o, i);
                return 1;
            }
            rknn_set_io_mem(ctxs[i], m, &out_attr);
            output_mems[i].push_back(m);
        }
    }

    printf("\n===== Bench config =====\n");
    printf("  contexts: %d (each bound to core %% %d)\n", num_ctxs, NUM_CORES);
    printf("  threads:  %d (dispatched via i %% num_ctxs)\n", num_threads);
    printf("  window:   %d s\n", seconds);
    printf("  pass_through: 1\n");
    printf("  sim_memcpy:   %s\n", g_simulate_memcpy ? "YES (XOR-copy 1.17MB per iter)" : "NO");
    printf("=========================\n\n");

    // sim_memcpy 用的源数据 — 跟 input_data 同内容, 每个 worker 都读它. 一份 ~1.17MB.
    std::vector<uint8_t> shared_src = input_data;
    const size_t copy_bytes = input_data.size();

    // 启线程, 计时, 等, 停, 收
    std::vector<std::thread> workers;
    workers.reserve(num_threads);

    auto t_start = std::chrono::steady_clock::now();
    for (int i = 0; i < num_threads; ++i) {
        int cidx = i % num_ctxs;
        workers.emplace_back(worker_loop,
                             ctxs[cidx],
                             static_cast<uint8_t *>(input_mems[cidx]->virt_addr),
                             shared_src.data(),
                             copy_bytes);
    }

    std::this_thread::sleep_for(std::chrono::seconds(seconds));
    g_stop.store(true, std::memory_order_relaxed);
    for (auto &t : workers) t.join();
    auto t_end = std::chrono::steady_clock::now();

    double elapsed = std::chrono::duration<double>(t_end - t_start).count();
    long total = g_total_runs.load();
    double fps = total / elapsed;
    double per_core_ms = (1000.0 * num_threads) / (fps * NUM_CORES);

    printf("\n===== Result =====\n");
    printf("  total rknn_run: %ld\n", total);
    printf("  elapsed:        %.3f s\n", elapsed);
    printf("  THROUGHPUT:     \033[1;32m%.2f FPS\033[0m\n", fps);
    printf("  per-core avg:   %.2f ms / inference\n", per_core_ms);
    printf("==================\n");

    for (int i = 0; i < num_ctxs; ++i) {
        for (auto *m : output_mems[i]) rknn_destroy_mem(ctxs[i], m);
        rknn_destroy_mem(ctxs[i], input_mems[i]);
        rknn_destroy(ctxs[i]);
    }
    free(model_data);
    return 0;
}
