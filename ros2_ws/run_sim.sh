#!/usr/bin/env bash
# =============================================================================
# 一键仿真启动脚本：Gazebo 3D 物理场景（小车/墙壁）+ 建图 + 探索 + matplotlib 建图可视化
#
# 做的顺序：
#   1) 清理上一轮残留进程（关键：防止 gzserver 因端口 11345 被占而 exit 255）
#   2) 载入 ROS2 Foxy 环境，设置 HOME / Gazebo 模型库规避项
#   3) 启动 exploration_demo（software_gl:=true）：
#        gzserver(物理仿真) + gzclient(Gazebo 图形窗口：3D 场景/小车/墙壁)
#        + Cartographer 建图 + 前沿探索 + 路径跟随 + matplotlib 建图可视化
#   4) 自检 gzclient 是否存活；并把 Gazebo 窗口自动提到最前
#      （GNOME 默认有焦点防抢：新窗口只会在任务栏标记“需要关注”，不会前置，
#        若从最大化的 IDE 终端启动，窗口会被 IDE 完全盖住而“看不到”）
#
# 用法：
#   bash <本目录>/run_sim.sh
#   可选追加 launch 参数，例如：
#   bash run_sim.sh viz:=false        # 不弹 matplotlib，只留 Gazebo 窗口
#   bash run_sim.sh headless:=true    # 无显示环境：只跑物理仿真，不弹 gzclient
#   Ctrl-C 退出时自动清理本次全部进程。
#
# 重要：本脚本必须在**能直接访问显卡/GLX 的终端**里运行。
#       Trae/VSCode 等 IDE 内置终端通常处于沙箱中，X 客户端能连上（matplotlib 窗口
#       能弹出来），但拿不到 GLX → gzclient 立即退出 → 看不到 Gazebo。
#       这种情况请改用系统原生终端（GNOME Terminal / Konsole）运行。
# =============================================================================
set -u

# 工作区根目录 = 本脚本所在目录（不再写死绝对路径）。
# 这样把整个 ros2_ws 拷到别的机器/路径后仍可直接运行
# （前提：已在目标机执行过 colcon build，install/ 里的路径是构建时生成的）。
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="${WS}/run_sim.log"

# 需要清理/关停的进程名特征（pkill -f，进程名或命令行命中即可）
PATTERNS=(
  "gzserver"
  "gzclient"
  "cartographer_node"
  "occupancy_grid_node"
  "scan_no_return_filter"
  "frontier_explorer_node"
  "path_follower_node"
  "exploration_viz"
  "spawn_entity.py"
  "robot_state_publisher"
  "static_transform_publisher"
  "ros2 launch"
)

kill_all() {
  for pat in "${PATTERNS[@]}"; do
    pkill -9 -f -- "${pat}" 2>/dev/null || true
  done
}

# 把 Gazebo 主窗口提到最前：
#   GNOME 有焦点防抢策略，默认不会把新窗口前置（只标记 _NET_WM_STATE_DEMANDS_ATTENTION），
#   若从最大化的 IDE 终端启动，Gazebo 会被 IDE 完全盖住而"看不到"。
#   这里用 XRaiseWindow + XSetInputFocus + EWMH(_NET_ACTIVE_WINDOW/_NET_RESTACK_WINDOW) 强行前置。
raise_gazebo() {
  command -v xwininfo >/dev/null 2>&1 || return 1
  local wid act
  wid=$(xwininfo -root -tree 2>/dev/null | awk '/"Gazebo"/ {print $1; exit}')
  [ -n "${wid:-}" ] || return 1
  python3 - "$wid" <<'PY' >/dev/null 2>&1
import ctypes, sys
wid = int(sys.argv[1], 16)
x11 = ctypes.CDLL("libX11.so.6")
x11.XOpenDisplay.restype = ctypes.c_void_p
x11.XInternAtom.restype = ctypes.c_ulong
x11.XDefaultRootWindow.restype = ctypes.c_ulong
x11.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
d = x11.XOpenDisplay(None)
if not d:
    sys.exit(1)
root = x11.XDefaultRootWindow(d)

x11.XRaiseWindow.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
x11.XSetInputFocus.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
x11.XRaiseWindow(d, ctypes.c_ulong(wid))
x11.XSetInputFocus(d, ctypes.c_ulong(wid), 1, 0)  # RevertToParent

class Ev(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("serial", ctypes.c_ulong), ("send_event", ctypes.c_int),
                ("display", ctypes.c_void_p), ("window", ctypes.c_ulong),
                ("message_type", ctypes.c_ulong), ("format", ctypes.c_int), ("data", ctypes.c_long * 5)]

mask = (1 << 20) | (1 << 19)  # SubstructureRedirectMask | SubstructureNotifyMask
for name in (b"_NET_ACTIVE_WINDOW", b"_NET_RESTACK_WINDOW"):
    atom = x11.XInternAtom(d, name, False)
    ev = Ev()
    ev.type = 33          # ClientMessage
    ev.display = d
    ev.window = ctypes.c_ulong(wid)
    ev.message_type = atom
    ev.format = 32
    ev.data[0] = 2        # source indication: 2 = pager（可绕过焦点防抢）
    ev.data[1] = 0        # timestamp: CurrentTime
    if name.endswith(b"RESTACK_WINDOW"):
        ev.data[3] = 0    # detail: 0 = Above
    x11.XSendEvent(d, ctypes.c_ulong(root), False, ctypes.c_long(mask), ctypes.byref(ev))
x11.XFlush(d)
PY
  # 验证是否真的成了活动窗口
  act=$(xprop -root _NET_ACTIVE_WINDOW 2>/dev/null | awk '{print $NF}')
  [ -n "${act:-}" ] && [ "${act}" = "${wid}" ]
}

echo "[1/4] 清理上一轮残留进程 ..."
kill_all
sleep 1
if ss -lnt 2>/dev/null | grep -q ':11345'; then
  echo "[警告] 端口 11345 仍被占用（可能不是本轮进程），请检查："
  ss -lntp 2>/dev/null | grep ':11345' || true
else
  echo "      端口 11345 已空闲，可安全启动 gzserver。"
fi

echo "[2/4] 载入 ROS2 Foxy 环境 ..."
set +u
# shellcheck disable=SC1091
source /opt/ros/foxy/setup.bash
# shellcheck disable=SC1091
source "${WS}/install/setup.bash"
set -u
export HOME=/tmp/roshome
export GAZEBO_MODEL_DATABASE_URI=""
cd "${WS}" || exit 1

# 退出（Ctrl-C / 结束）时统一关停，避免又留下占端口的进程
cleanup() {
  echo ""
  echo "[清理] 正在关停本次启动的所有进程 ..."
  kill_all
  echo "[清理] 完成。"
}
trap cleanup EXIT INT TERM

echo "[3/4] 启动 Gazebo 仿真 + 建图 + 探索 闭环 ..."
echo "      提示：日志同时写入 ${LOG}"
set -o pipefail
ros2 launch bringup exploration_demo.launch.py software_gl:=true "$@" 2>&1 | tee "${LOG}" &
LAUNCH_PID=$!

echo "[4/4] 自检 Gazebo 图形窗口(gzclient) ..."
GZ_OK=0
for _ in $(seq 1 30); do
  if pgrep -f -- gzclient >/dev/null 2>&1; then GZ_OK=1; break; fi
  # gzclient 若因 GLX/X 失败退出，日志里会立刻出现这些关键字
  if grep -qE "X Error|GLX|X_GLXCreateContext|BadValue|process has died.*gzclient" "${LOG}" 2>/dev/null; then
    break
  fi
  sleep 1
done

if [[ "${GZ_OK}" == "1" ]]; then
  echo "      [OK] gzclient 进程存活。正在把 Gazebo 窗口提到最前 ..."
  RAISED=0
  for _ in $(seq 1 15); do
    if raise_gazebo; then RAISED=1; break; fi
    sleep 1
  done
  if [[ "${RAISED}" == "1" ]]; then
    echo "      [OK] Gazebo 窗口已置于最前。"
  else
    echo "      [提示] 未能自动前置 Gazebo 窗口（GNOME 焦点防抢）。"
    echo "             ▶ 手动切过去：按 Super 键 / Alt+Tab，或点任务栏(左侧)的 Gazebo 图标。"
  fi
else
  echo "      [失败] gzclient 未存活，Gazebo 窗口起不来。"
  if grep -qE "X Error|GLX|X_GLXCreateContext|BadValue" "${LOG}" 2>/dev/null; then
    echo "      日志中检测到 GLX/X 错误 —— 说明当前终端拿不到 OpenGL/GLX 上下文。"
    echo "      ▶ 对策：不要在 IDE 内置终端（沙箱）里跑，改用系统原生终端（GNOME Terminal）运行："
    echo "          bash ${WS}/run_sim.sh"
  else
    echo "      未检测到 GLX 错误。请检查："
    echo "        1) DISPLAY 是否有效（当前 DISPLAY='${DISPLAY:-<空>}'）"
    echo "        2) 端口 11345 是否被其它 gzserver 占用"
    echo "        3) 查看 ${LOG} 中 gzclient 相关行"
  fi
fi

echo ""
echo "按 Ctrl-C 结束并自动清理。"

# 跟随主栈生命周期；Ctrl-C 会触发上面的 trap 统一清理
wait "${LAUNCH_PID}"
