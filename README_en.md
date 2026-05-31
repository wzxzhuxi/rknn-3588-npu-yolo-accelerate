# RK3588 YOLOv5 Real-time Detection Accelerator

English | [简体中文](README.md)

End-to-end **up to 175 FPS** YOLOv5s real-time video detection on the RK3588 tri-core NPU —
**99% NPU utilization** against the bare NPU compute ceiling of 180.5 FPS.

Forked from [rknn-cpp-Multithreading](https://github.com/leafqycc/rknn-cpp-Multithreading).
This repo adds a full performance overhaul, architectural refactor.

---

## Performance

| Workload | FPS | NPU util |
|---|---|---|
| `yolov5_thread_pool` (end-to-end, real video pipeline) | **175** (peak) / 173 (typical) | 99% |
| `npu_only_bench` (bare `rknn_run` loop) | **180.5** | 99% |
| Theoretical ceiling (3 cores × 1000/16.6ms/frame) | 180.7 | — |

Conditions: yolov5s_relu (per-tensor int8) / 640×640 / 720p60 H.264 / 12 inference workers / hardware locked at max freq.
Precision signal: AvgDet/Frame steady at 8-16 (yolov5s_relu_pt.rknn @ COCO 720p urban-traffic footage).

---

## Quick Start

```sh
./yolorun.sh
```

Does it all: adb check → cross-compile → push binary+libs+model → lock CPU/NPU/DDR → run → stream FPS in real time.

Override defaults via env vars:
```sh
DURATION=30 ./yolorun.sh                       # auto-stop after 30s
THREADS=24 ./yolorun.sh                        # 24 workers (default 12)
MODEL=yolov5s.rknn ./yolorun.sh                # different model
TOOLCHAIN=/path/to/aarch64-... ./yolorun.sh    # custom cross toolchain prefix
```

The test video `720p60hz.h264` (raw H.264 Annex-B, ~55 MB) is **shipped in repo root**, so it's clone-and-run — no transcoding needed.
Use your own video:
```sh
VIDEO=/path/to/your.h264 ./yolorun.sh
```
Demux any mp4 to raw H.264 (if using your own video):
```sh
ffmpeg -i <your.mp4> -c:v copy -bsf:v h264_mp4toannexb -f h264 720p60hz.h264
```

---

## Pipeline Architecture

```
H.264 file → MPP HW decode (its own thread) → single-pass RGA NV12→RGB888 + resize
                                                       ↓
                                    cv::Mat (refcounted, thread-safe handoff)
                                                       ↓
                                       [producer-consumer queue, cap 10]
                                                       ↓
              ┌────────────────────────────────────────┴────────────────────────────────────────┐
              ↓                                                                                  ↓
       worker[0] (NPU core 0)                                                          worker[11] (NPU core 2)
        ├─ NEON XOR-128 → input_mem  (zero-copy DMA-BUF)                                ...
        ├─ rknn_run                                                                     ...
        └─ NEON-skip NMS                                                                ...
              ↓                                                                                  ↓
              └─────────────────── results inserted by frame_id (ordered) ───────────────────────┘
                                                       ↓
                                         main thread GetResult + FPS counter
```

**Key optimizations**:

| Optimization | How | Gain |
|---|---|---|
| Multi-core NPU binding | each worker calls `rknn_set_core_mask(i % 3)` | 19 → 100 FPS |
| MPP HW decode + single-pass RGA | `nv12_to_rgb_mat_rga()` in one imresize | 100 → 163 FPS |
| Zero-copy DMA-BUF I/O | `rknn_create_mem` + `rknn_set_io_mem` once, then only `rknn_run` | 163 → 175 FPS |
| `pass_through=1` + NEON XOR-128 | uint8→int8 conversion in worker; driver skips normalize | flat (already at NPU compute ceiling) |

---

## Source Layout

```
src/
├── main.cpp                    # entry point
├── bench/
│   └── npu_throughput.cpp      # bare rknn_run benchmark (control for yolov5_thread_pool)
├── engine/
│   ├── engine.h                # NNEngine abstract base
│   ├── rknn_engine.{h,cpp}     # zero-copy RKNN backend
│   └── rknn_helpers.h          # model loader / tensor_attr convert / debug print
├── inference/
│   ├── yolov5_detector.{h,cpp} # single per-NPU-context inference instance
│   └── worker_pool.{h,cpp}     # 12-worker thread pool, frame-id ordered
├── decoder/
│   └── mpp_decoder.{h,cpp}     # MPP H.264/H.265 hardware decoder (vendor demo)
├── preprocess/
│   └── rga_nv12_to_rgb.{h,cpp} # single-step RGA NV12→RGB888+resize
├── postprocess/
│   └── yolov5_nms.{h,cpp}      # box decode + NMS (NEON fast-skip)
├── types/
│   ├── tensor_types.h
│   ├── yolo_types.h
│   └── error_codes.h
└── utils/
    └── logging.h
```

---

## Build

Requires the **buildroot aarch64 cross toolchain** (this project assumes the standard RK3588 Buildroot SDK output).
Apart from the cross toolchain + system libc/pthread, **all dependencies are vendored in-tree** (`3rdparty/opencv` + `3rdparty/rga` + `3rdparty/mpp` + `librknn_api`). Clone this repo standalone and it builds — no parent directory / vendor SDK context required.

```sh
mkdir build && cd build

TOOL=/path/to/buildroot/output/.../host/bin/aarch64-buildroot-linux-gnu-
cmake -DCMAKE_C_COMPILER=${TOOL}gcc -DCMAKE_CXX_COMPILER=${TOOL}g++ ..
make -j$(nproc)
```

Outputs:
- `yolov5_thread_pool` — main binary
- `npu_only_bench`    — diagnostic tool
- `lib*.so`           — shared libs (engine / image_process / decoder / inference)

---

## Deploy & Run on Board (manual mode)

> **For most use cases just run `./yolorun.sh`** (see above). The following is the manual flow for those who want to understand each step.

Convention:
```
/userdata/yolo_phase1/
├── yolov5_thread_pool     # main binary
├── npu_only_bench         # diagnostic tool
├── lib/                   # all .so (vendor blobs: librknnrt.so librga.so librockchip_mpp.so)
├── model/                 # *.rknn
├── 720p60hz.h264          # raw H.264 (NOT mp4!)
└── bus.jpg                # one-shot input for bench
```

Push + run:
```sh
adb push build/yolov5_thread_pool build/npu_only_bench /userdata/yolo_phase1/
adb push build/lib*.so                                  /userdata/yolo_phase1/lib/
adb push weights/yolov5s_relu_pt.rknn                   /userdata/yolo_phase1/model/
adb push 720p60hz.h264                                  /userdata/yolo_phase1/
adb push media/bus.jpg                                  /userdata/yolo_phase1/   # for npu_only_bench

# 1. Lock frequencies FIRST — paste the block from [Frequency Lock](#frequency-lock)
#    below into a one-shot adb shell. Default governors keep DDR at 528 MHz of
#    1848 MHz peak, costing ~7% NPU throughput.

# 2. Run end-to-end detection
adb shell "cd /userdata/yolo_phase1 && export LD_LIBRARY_PATH=lib && \
    ./yolov5_thread_pool model/yolov5s_relu_pt.rknn 720p60hz.h264 264 12"
```

Args: `<model.rknn> <raw_h264_or_h265> [264|265] [num_threads]`

---

## Diagnostic Tool: `npu_only_bench`

Strips all CPU-side pipeline and just spams `rknn_run()` — used to calibrate the **NPU compute ceiling**.

```sh
adb shell "cd /userdata/yolo_phase1 && export LD_LIBRARY_PATH=lib && \
    ./npu_only_bench model/yolov5s_relu_pt.rknn <threads> <seconds> <ctxs> <image_path?> <sim_memcpy?>"

# Example: 12 threads × 12 ctx, 10s, bus.jpg as input
./npu_only_bench model/yolov5s_relu_pt.rknn 12 10 12 bus.jpg
```

The FPS gap between this and `yolov5_thread_pool` ≈ real pipeline CPU-side cost
(~6 FPS = memcpy + postprocess + scheduling).

---

## Frequency Lock

**Single source of truth is inlined in [yolorun.sh](yolorun.sh)** — one-click mode runs it automatically. When running `yolov5_thread_pool` manually, paste the block below into a one-shot `adb shell` (no need to create any script file on the board):

```sh
#!/bin/sh
# Disable CPU idle state1 (no deep sleep — eliminates wake latency)
for i in 0 1 2 3 4 5 6 7; do
  echo 1 > /sys/devices/system/cpu/cpu$i/cpuidle/state1/disable 2>/dev/null
done
# CPU at max
for spec in 'policy0 1800000' 'policy4 2304000' 'policy6 2304000'; do
  p=$(echo $spec | cut -d' ' -f1); f=$(echo $spec | cut -d' ' -f2)
  echo userspace > /sys/devices/system/cpu/cpufreq/$p/scaling_governor
  echo $f        > /sys/devices/system/cpu/cpufreq/$p/scaling_setspeed
done
# NPU / DDR at max
echo userspace  > /sys/class/devfreq/fdab0000.npu/governor
echo 1000000000 > /sys/class/devfreq/fdab0000.npu/userspace/set_freq
echo userspace  > /sys/class/devfreq/dmc/governor
echo 1848000000 > /sys/class/devfreq/dmc/userspace/set_freq
```

These are **runtime sysfs writes**, lost on reboot. For production, wrap in a systemd
service or init.d script.

---

## Known Gotchas

1. **DDR defaults to 528 MHz** (28% of 1848 MHz peak). Without locking frequencies
   the NPU saturates but DDR starves it, costing FPS down to ~162.
2. **`pass_through=1` requires int8 input.** We do NEON XOR-128 (equivalent to
   `uint8 - 128`) in the worker. Skip it and the model sees garbage.
3. **`AvgDet/Frame = 64.00`** means hitting `OBJ_NUMB_MAX_SIZE`. If a new model
   reports 64 every frame, it's almost certainly a **sigmoid-head mismatch** —
   airockchip's yolov5s_relu has sigmoid fused into the model graph; do NOT apply
   sigmoid again in postprocess.
4. **H.264 input must be raw Annex-B**, not an mp4 container. MPP won't decode mp4
   wrapping. Demux with `ffmpeg -bsf:v h264_mp4toannexb -f h264`.

---

## Exit Codes

| Code | Meaning |
|---|---|
| `0` | success — pipeline init OK, input fully consumed, no NPU errors during run |
| `255` (i.e. main `return -1`) | model load failed / video open failed / MPP init failed / any worker accumulated NPU errors |

CI scripts can just check `$?`. Previously (before refactor) main returned 0 unconditionally — fixed.

---

## Model Conversion (PC side)

ONNX → RKNN goes through `rknn-toolkit2`. Note: **no wheels exist for Python 3.13+**
(latest is `cp312`). Tested with a Python 3.10 venv (`onnx==1.14.1`).

Key `rknn.config()` knobs:
- `target_platform='rk3588'` (defaults to `rk3566`, MUST change)
- `quantized_method='layer'` (per-tensor, vendor default) or `'channel'` (per-channel,
   slightly higher mAP, **identical runtime cost on RK3588 NPU**)
- `do_quantization=True` needs `DATASET` pointing at ~128 calibration images
  (coco128 works fine)

Airockchip yolov5 model zoo mirror (GitHub raw is slow/404 from CN):
https://ftrg.zbox.filez.com/v2/delivery/data/...

---

## License

Inherits from the upstream project. See `LICENSE.txt`.
