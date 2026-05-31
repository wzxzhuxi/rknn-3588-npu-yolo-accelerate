# RK3588 YOLOv5 实时检测 · 触达 NPU 物理极限

[English](README_en.md) | 简体中文

端到端 **峰值 175 FPS** YOLOv5s 实时视频检测,跑在 RK3588 三核 NPU 上 — **NPU 99% 满载,距裸算力上限 180.5 FPS 仅差 ~3%**。剩余差距是 CPU 端 postprocess + 调度 + memcpy 的**不可压缩开销**(已实验排除 DDR 争夺、ctx 数、cpuidle 等优化因素), **本架构已无显著加速空间**。

源头是 [rknn-cpp-Multithreading](https://github.com/leafqycc/rknn-cpp-Multithreading),本仓库做了完整的性能改造 + 架构重构,直至**触达 NPU 算力天花板**。

---

## 性能数字

| 测试 | FPS | NPU 利用率 |
|---|---|---|
| `yolov5_thread_pool` (端到端,真实视频流水线) | **175** (峰值) / 173 (典型) | 99% |
| `npu_only_bench` (裸 `rknn_run` 循环) | **180.5** | 99% |
| 理论上限 (3 核 × 1000/16.6ms/帧) | 180.7 | — |

测试条件: yolov5s_relu (per-tensor int8) / 640×640 / 720p60 H.264 视频 / 12 推理 worker / 硬定频.
精度信号: AvgDet/Frame 稳定在 8-16 之间(yolov5s_relu_pt.rknn @ COCO 720p 城市交通画面).

---

## 一键运行

```sh
./yolorun.sh
```

自动: 检 adb 连接 → 交叉编译 → push binary+lib+model → 锁 CPU/NPU/DDR → 跑 → 实时打 FPS.

环境变量覆盖默认:
```sh
DURATION=30 ./yolorun.sh                       # 跑 30 秒后自动停
THREADS=24 ./yolorun.sh                        # 24 worker (默认 12)
MODEL=yolov5s.rknn ./yolorun.sh                # 换模型
TOOLCHAIN=/path/to/aarch64-... ./yolorun.sh    # 自定义交叉工具链前缀
```

视频文件 `720p60hz.h264` (裸 H.264 Annex-B, ~55 MB) 已 ship 在仓库根, **clone 即跑**, 不需要任何转码准备.
换用自己的视频:
```sh
VIDEO=/path/to/your.h264 ./yolorun.sh
```
mp4 转裸 H.264 (如果用自己的视频):
```sh
ffmpeg -i <你的.mp4> -c:v copy -bsf:v h264_mp4toannexb -f h264 720p60hz.h264
```

---

## 流水线架构

```
H.264 文件 → MPP 硬解 (回调线程) → RGA 单步 NV12→RGB888 + resize
                                                ↓
                                  cv::Mat (refcount, 跨线程安全)
                                                ↓
                                      [生产-消费队列, max 10]
                                                ↓
              ┌─────────────────────────────────┴─────────────────────────────────┐
              ↓                                                                   ↓
       worker[0] (core 0)                                                worker[11] (core 2)
        ├─ NEON XOR-128 → input_mem  (零拷贝 DMA-BUF)                     ...
        ├─ rknn_run                                                       ...
        └─ NEON-skip NMS                                                  ...
              ↓                                                                   ↓
              └────────────────── 结果按 frame_id 入序 ──────────────────────────┘
                                                ↓
                                       主线程 GetResult + FPS 统计
```

**关键加速点**:

| 优化 | 实现 | 收益 |
|---|---|---|
| 多核 NPU 绑定 | 每个 worker `rknn_set_core_mask(i % 3)` | 19 → 100 FPS |
| MPP 硬解 + 单步 RGA | `nv12_to_rgb_mat_rga()` 一次 imresize | 100 → 163 FPS |
| 零拷贝 DMA-BUF | `rknn_create_mem` + `rknn_set_io_mem` 一次性绑定, 之后只 `rknn_run` | 163 → 175 FPS |
| `pass_through=1` + NEON XOR-128 | worker 端 uint8→int8 转换, driver 不做 normalize | 持平(已撞 NPU 算力上限) |

---

## 源码结构

```
src/
├── main.cpp                    # 主程序入口
├── bench/
│   └── npu_throughput.cpp      # 裸 NPU 吞吐诊断工具 (yolov5_thread_pool 的对照组)
├── engine/
│   ├── engine.h                # NNEngine 抽象基类
│   ├── rknn_engine.{h,cpp}     # RKNN 零拷贝后端
│   └── rknn_helpers.h          # 模型加载 / tensor_attr 互转 / 打印
├── inference/
│   ├── yolov5_detector.{h,cpp} # 单个 NPU context 的推理实例
│   └── worker_pool.{h,cpp}     # 12 worker 线程池, 帧 id 维序
├── decoder/
│   └── mpp_decoder.{h,cpp}     # MPP H.264/H.265 硬件解码 (vendor demo 同款)
├── preprocess/
│   └── rga_nv12_to_rgb.{h,cpp} # 单步 RGA NV12→RGB888+resize
├── postprocess/
│   └── yolov5_nms.{h,cpp}      # box 解码 + NMS (NEON fast-skip)
├── types/
│   ├── tensor_types.h
│   ├── yolo_types.h
│   └── error_codes.h
└── utils/
    └── logging.h
```

---

## 构建

需要 **buildroot aarch64 交叉工具链** (本项目默认 RK3588 板子的 Buildroot SDK 产出).
除交叉工具链 + 系统 libc/pthread 外, **所有依赖均在仓库内** (`3rdparty/opencv` + `3rdparty/rga` + `3rdparty/mpp` + `librknn_api`), 独立 clone 即可构建, 不依赖父目录或 vendor SDK 上下文.

```sh
mkdir build && cd build

TOOL=/path/to/buildroot/output/.../host/bin/aarch64-buildroot-linux-gnu-
cmake -DCMAKE_C_COMPILER=${TOOL}gcc -DCMAKE_CXX_COMPILER=${TOOL}g++ ..
make -j$(nproc)
```

产物:
- `yolov5_thread_pool` — 主程序
- `npu_only_bench`    — 诊断工具
- `lib*.so`           — 共享库 (engine / image_process / decoder / inference)

---

## 部署到板子(手动模式)

> **大多数情况直接用 `./yolorun.sh` 一键搞定**(见上文). 下面是手动流程, 给想理解每一步的人.

板上目录约定:
```
/userdata/yolo_phase1/
├── yolov5_thread_pool     # 主程序 binary
├── npu_only_bench         # 诊断工具
├── lib/                   # 所有 .so (含 vendor: librknnrt.so librga.so librockchip_mpp.so)
├── model/                 # *.rknn
├── 720p60hz.h264          # 裸 H.264 (注意: 不是 mp4!)
└── bus.jpg                # bench 用一次性输入图
```

推送 + 跑:
```sh
adb push build/yolov5_thread_pool build/npu_only_bench /userdata/yolo_phase1/
adb push build/lib*.so                                  /userdata/yolo_phase1/lib/
adb push weights/yolov5s_relu_pt.rknn                   /userdata/yolo_phase1/model/
adb push 720p60hz.h264                                  /userdata/yolo_phase1/
adb push media/bus.jpg                                  /userdata/yolo_phase1/   # npu_only_bench 用

# 1. 先定频 — 直接把下文 [硬件定频](#硬件定频) 段的指令贴一次到 adb shell 跑.
#    默认 governor 让 DDR 停在 528 MHz / 1848 MHz 顶频, NPU 吞吐掉 ~7%.

# 2. 跑端到端检测
adb shell "cd /userdata/yolo_phase1 && export LD_LIBRARY_PATH=lib && \
    ./yolov5_thread_pool model/yolov5s_relu_pt.rknn 720p60hz.h264 264 12"
```

参数: `<model.rknn> <raw_h264_or_h265> [264|265] [num_threads]`

---

## 诊断工具: `npu_only_bench`

剥掉所有 CPU 端流水线, 只死循环 `rknn_run()` — 用来标定 **NPU 算力上限**.

```sh
adb shell "cd /userdata/yolo_phase1 && export LD_LIBRARY_PATH=lib && \
    ./npu_only_bench model/yolov5s_relu_pt.rknn <threads> <seconds> <ctxs> <image_path?> <sim_memcpy?>"

# 示例 — 12 线程 12 ctx, 跑 10s, 用 bus.jpg 作输入
./npu_only_bench model/yolov5s_relu_pt.rknn 12 10 12 bus.jpg
```

它和 `yolov5_thread_pool` 的 FPS 差距 ≈ 真实流水线 CPU 端开销 (~6 FPS = memcpy + postprocess + 调度).

---

## 硬件定频

**单一真相源在 [yolorun.sh](yolorun.sh) 里 inline** — 一键模式自动跑. 手动跑 `yolov5_thread_pool` 时, 把下面这段贴到一次 `adb shell` 里执行即可 (不需要在板上创建任何脚本文件):

```sh
#!/bin/sh
# 禁 CPU idle state1 (避免短时空闲进深度睡眠, 拖累 worker 唤醒)
for i in 0 1 2 3 4 5 6 7; do
  echo 1 > /sys/devices/system/cpu/cpu$i/cpuidle/state1/disable 2>/dev/null
done
# CPU 顶频
for spec in 'policy0 1800000' 'policy4 2304000' 'policy6 2304000'; do
  p=$(echo $spec | cut -d' ' -f1); f=$(echo $spec | cut -d' ' -f2)
  echo userspace > /sys/devices/system/cpu/cpufreq/$p/scaling_governor
  echo $f        > /sys/devices/system/cpu/cpufreq/$p/scaling_setspeed
done
# NPU / DDR 顶频
echo userspace  > /sys/class/devfreq/fdab0000.npu/governor
echo 1000000000 > /sys/class/devfreq/fdab0000.npu/userspace/set_freq
echo userspace  > /sys/class/devfreq/dmc/governor
echo 1848000000 > /sys/class/devfreq/dmc/userspace/set_freq
```

定频是 **运行时状态**, 重启丢失. 生产部署用 systemd service 或 init.d 脚本固化.

---

## 已知陷阱

1. **DDR 默认 528 MHz**: 不定频前 NPU 满载但 DDR 是瓶颈, FPS 跌到 162.
2. **`pass_through=1` 要求 int8 输入**: 我们用 NEON XOR-128 (等价 `uint8 - 128`) 在 worker 端做转换. 不写就喂错数据.
3. **`AvgDet/Frame = 64.00`**: 这是 `OBJ_NUMB_MAX_SIZE` 上限. 如果换模型后每帧都 64, 通常是 **sigmoid 头不匹配** — airockchip 的 yolov5s_relu 自带 sigmoid head, 不要在 postprocess 再 sigmoid 一次.
4. **H.264 必须是裸流 (Annex B)**, mp4 容器 MPP 解不动. 用 `ffmpeg -bsf:v h264_mp4toannexb -f h264` 转出来.

---

## 退出码

| 退出码 | 语义 |
|---|---|
| `0` | 正常 — 流水线初始化通过, 输入读完整, 中途无 NPU 错误 |
| `255` (即 main `return -1`) | model 加载失败 / 视频读失败 / MPP 初始化失败 / 中途任意 worker 累计 NPU 错误 |

CI 脚本直接 `$?` 判定即可. 之前版本 (refactor 前) main 永远返回 0, 错误只打 log — 已修复.

---

## 模型转换 (PC 端)

ONNX → RKNN 走 `rknn-toolkit2`. 注意 **Python 3.13+ 没有对应 wheel** (`cp36..cp312` 为止),
本仓库测试用 Python 3.10 venv (`onnx==1.14.1`).

`rknn.config()` 关键参数:
- `target_platform='rk3588'` (默认 `rk3566`, 必改)
- `quantized_method='layer'` (per-tensor, vendor 默认) 或 `'channel'` (per-channel, 略高 mAP, RK3588 NPU 上**运行时性能一致**)
- `do_quantization=True` 需要 `DATASET` 指向 ~128 张校准图 (coco128 即可)

airockchip 的 yolov5 model zoo:
https://ftrg.zbox.filez.com/v2/delivery/data/... (CN 从 GitHub raw 拉很慢, 用这个镜像)

---

## License

继承上游项目, 见 `LICENSE.txt`.
