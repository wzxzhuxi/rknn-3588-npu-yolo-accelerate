#!/usr/bin/env bash
# yolorun.sh — PC 端一键极限性能驱动
# 流程: 交叉编译 → adb push → 锁 CPU/NPU/DDR 频率 → 跑 yolov5_thread_pool → 实时 grep FPS
#
# 用法:
#   ./yolorun.sh                  跑完整段视频, 默认极限配置
#   DURATION=30 ./yolorun.sh      只跑 30 秒
#   THREADS=24 ./yolorun.sh       改 24 worker
#   MODEL=yolov5s.rknn ./yolorun.sh  换模型
#
# 默认产出: 端到端 ~175 FPS (峰值), NPU 三核 99% 利用率 (yolov5s_relu_pt + 720p60 H.264).

set -euo pipefail

# === 实测极限性能配置 ===
: ${TOOLCHAIN:=/home/fuck/Lubuncat_sdk/buildroot/output/rockchip_rk3588_lubancat/host/bin/aarch64-buildroot-linux-gnu-}
: ${BOARD_DIR:=/userdata/yolo_phase1}
: ${MODEL:=yolov5s_relu_pt.rknn}   # per-tensor int8 ReLU, 实测 ~175 FPS 端到端
: ${VIDEO:=720p60hz.h264}          # 裸 H.264 Annex-B (mp4 容器 MPP 解不动)
: ${VIDEO_TYPE:=264}               # 264 / 265
: ${THREADS:=12}                   # NPU 三核 × 4 worker 队列深度 = 吞吐甜点
: ${DURATION:=0}                   # 0 = 跑完整段; >0 = timeout 秒数

green()  { printf '\033[1;32m%s\033[0m\n' "$*"; }
yellow() { printf '\033[1;33m%s\033[0m\n' "$*"; }
red()    { printf '\033[1;31m%s\033[0m\n' "$*"; }

# === 1. 检查 adb ===
if ! adb devices 2>/dev/null | grep -q 'device$'; then
    red "[!] No adb device. Check USB cable / power / 'adb start-server'."
    exit 1
fi
green "[+] adb device detected"

# === 2. 检查交叉编译器 ===
if [[ ! -x "${TOOLCHAIN}gcc" ]]; then
    red "[!] Cross compiler not found: ${TOOLCHAIN}gcc"
    red "    Override with TOOLCHAIN=/path/to/aarch64-...- (note trailing dash)"
    exit 1
fi

# === 3. 增量构建 ===
mkdir -p build
pushd build >/dev/null
if [[ ! -f CMakeCache.txt ]]; then
    green "[+] First-time cmake (cross-toolchain)"
    cmake -DCMAKE_C_COMPILER=${TOOLCHAIN}gcc \
          -DCMAKE_CXX_COMPILER=${TOOLCHAIN}g++ .. >/dev/null
fi
green "[+] make -j$(nproc)"
make -j$(nproc) 2>&1 | tail -5
popd >/dev/null

# === 4. 推送产物到板子 ===
green "[+] Pushing binaries + libs to ${BOARD_DIR}"
adb shell "mkdir -p ${BOARD_DIR}/lib ${BOARD_DIR}/model"
adb push build/yolov5_thread_pool ${BOARD_DIR}/ 2>&1 | tail -1
adb push build/npu_only_bench    ${BOARD_DIR}/ 2>&1 | tail -1   # 顺路推, 方便板上手动跑 bench
adb push build/lib*.so           ${BOARD_DIR}/lib/ 2>&1 | tail -1
# bus.jpg: npu_only_bench 的图片输入. 一次性推, 后续 bench 直接用.
if ! adb shell "test -f ${BOARD_DIR}/bus.jpg" 2>/dev/null && [[ -f media/bus.jpg ]]; then
    adb push media/bus.jpg ${BOARD_DIR}/ 2>&1 | tail -1
fi

# 模型: 缺则推 (避免重复传 8MB)
if ! adb shell "test -f ${BOARD_DIR}/model/${MODEL}" 2>/dev/null; then
    if [[ -f weights/${MODEL} ]]; then
        green "[+] Pushing model ${MODEL}"
        adb push weights/${MODEL} ${BOARD_DIR}/model/ 2>&1 | tail -1
    else
        red "[!] Model missing on PC (weights/${MODEL}) AND on board"
        exit 1
    fi
fi

# 视频: 缺则推. 仓库默认 ship 了 720p60hz.h264, 正常 clone 直接命中.
if ! adb shell "test -f ${BOARD_DIR}/${VIDEO}" 2>/dev/null; then
    if [[ -f ${VIDEO} ]]; then
        green "[+] Pushing video ${VIDEO}"
        adb push ${VIDEO} ${BOARD_DIR}/ 2>&1 | tail -1
    else
        red "[!] Video '${VIDEO}' missing from PC. 应该是仓库里 720p60hz.h264, 检查是否被误删."
        red "    用自己的视频: VIDEO=/path/to/your.h264 ./yolorun.sh"
        red "    mp4 转裸 H.264: ffmpeg -i your.mp4 -c:v copy -bsf:v h264_mp4toannexb -f h264 ${VIDEO}"
        exit 1
    fi
fi

# === 5. 锁 CPU/NPU/DDR 到最高频率 (inline, 自带) ===
# 默认 governor 会让 DDR 跌到 528 MHz (顶频 1848), 拖 ~7% FPS.
# 也禁 cpuidle state1, 避免 worker 短暂阻塞时 CPU 进深度睡眠拉高唤醒延迟.
green "[+] Locking CPU/NPU/DDR to max frequencies"
adb shell '
for i in 0 1 2 3 4 5 6 7; do
    echo 1 > /sys/devices/system/cpu/cpu$i/cpuidle/state1/disable 2>/dev/null
done
for spec in "policy0 1800000" "policy4 2304000" "policy6 2304000"; do
    p=$(echo $spec | cut -d" " -f1); f=$(echo $spec | cut -d" " -f2)
    echo userspace > /sys/devices/system/cpu/cpufreq/$p/scaling_governor
    echo $f        > /sys/devices/system/cpu/cpufreq/$p/scaling_setspeed
done
echo userspace  > /sys/class/devfreq/fdab0000.npu/governor
echo 1000000000 > /sys/class/devfreq/fdab0000.npu/userspace/set_freq
echo userspace  > /sys/class/devfreq/dmc/governor
echo 1848000000 > /sys/class/devfreq/dmc/userspace/set_freq
echo userspace  > /sys/class/devfreq/fb000000.gpu/governor 2>/dev/null
echo 1000000000 > /sys/class/devfreq/fb000000.gpu/userspace/set_freq 2>/dev/null
' >/dev/null

# 打印实际生效频率
yellow "[i] Effective frequencies:"
adb shell '
    echo -n "    A55(p0):     "; cat /sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq
    echo -n "    A76(p4):     "; cat /sys/devices/system/cpu/cpufreq/policy4/scaling_cur_freq
    echo -n "    A76(p6):     "; cat /sys/devices/system/cpu/cpufreq/policy6/scaling_cur_freq
    echo -n "    NPU:         "; cat /sys/class/devfreq/fdab0000.npu/cur_freq
    echo -n "    DDR:         "; cat /sys/class/devfreq/dmc/cur_freq
'

# === 6. run ===
green "[+] Running: ${MODEL} × ${THREADS} threads on ${VIDEO} (H.${VIDEO_TYPE})"
echo "=================================================================="

if [[ ${DURATION} -gt 0 ]]; then
    CMD="cd ${BOARD_DIR} && export LD_LIBRARY_PATH=lib && timeout ${DURATION} ./yolov5_thread_pool model/${MODEL} ${VIDEO} ${VIDEO_TYPE} ${THREADS}"
else
    CMD="cd ${BOARD_DIR} && export LD_LIBRARY_PATH=lib && ./yolov5_thread_pool model/${MODEL} ${VIDEO} ${VIDEO_TYPE} ${THREADS}"
fi

# MPP decoder 用 printf 不换行, 把日志粘成一长行 — 用 grep -oE 只 extract 我们要的字段.
# tr 把超长一行拆成多行可读.
adb shell "${CMD}" 2>&1 | \
    grep --line-buffered -oE 'FPS:[0-9.]+|AvgDet/Frame:[0-9.]+|\[NN_ERROR\][^\[]*|\[STAGE[^]]*\][^[]*'
ADB_RC=${PIPESTATUS[0]}

echo "=================================================================="
if [[ ${ADB_RC} -eq 0 ]]; then
    green "[+] Run finished successfully (exit 0)"
elif [[ ${ADB_RC} -eq 124 ]]; then
    yellow "[i] Stopped by DURATION timer (${DURATION}s)"
else
    red "[!] Run failed with adb exit code ${ADB_RC}"
    red "    See log via: adb shell 'cd ${BOARD_DIR} && ./yolov5_thread_pool ...' for full output"
    exit ${ADB_RC}
fi
