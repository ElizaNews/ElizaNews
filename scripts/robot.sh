#!/usr/bin/env bash
set -euo pipefail

# AiNex robot management over SSH
# Usage: scripts/robot.sh <command> [args]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/robot-config.sh"

# Resolve robot IP
get_ip() {
    if [[ -n "$ROBOT_IP" ]]; then
        echo "$ROBOT_IP"
    elif [[ -f "$SCRIPT_DIR/.robot-ip" ]]; then
        cat "$SCRIPT_DIR/.robot-ip"
    else
        echo "ERROR: Robot IP not set. Run: scripts/setup-ssh.sh <ip>" >&2
        exit 1
    fi
}

IP="$(get_ip)"
SSH_CMD="ssh $SSH_OPTS ${ROBOT_USER}@${IP}"

# Check if we have key auth or need sshpass
if ! ssh $SSH_OPTS -o BatchMode=yes "${ROBOT_USER}@${IP}" "true" 2>/dev/null; then
    if command -v sshpass &>/dev/null; then
        SSH_CMD="sshpass -p '$ROBOT_PASSWORD' ssh $SSH_OPTS ${ROBOT_USER}@${IP}"
    else
        echo "SSH key auth not set up. Run: scripts/setup-ssh.sh $IP"
        exit 1
    fi
fi

cmd_status() {
    echo "=== AiNex Robot Status ==="
    echo "IP: $IP"
    echo ""
    eval $SSH_CMD bash <<'REMOTE'
echo "--- System ---"
echo "Hostname:    $(hostname)"
echo "Uptime:      $(uptime -p)"
echo "Load:        $(cat /proc/loadavg | awk '{print $1, $2, $3}')"
echo "Memory:      $(free -h | awk '/Mem/{printf "%s / %s (%s used)", $3, $2, $5}')"
echo "Disk:        $(df -h / | awk 'NR==2{printf "%s / %s (%s)", $3, $2, $5}')"
echo "Temperature: $(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null | awk '{printf "%.1f°C", $1/1000}' || echo 'N/A')"
echo ""

echo "--- ROS Stack ---"
if systemctl is-active --quiet start_app_node 2>/dev/null; then
    echo "Service:     RUNNING"
else
    echo "Service:     STOPPED"
fi

# Check if ROS master is reachable
if command -v rostopic &>/dev/null; then
    source /home/ubuntu/ros_ws/devel/setup.bash 2>/dev/null
    TOPICS=$(rostopic list 2>/dev/null | wc -l || echo 0)
    echo "ROS topics:  $TOPICS"

    # Check key nodes
    for node in /ainex_controller /ros_robot_controller /sensor_node; do
        if rosnode ping -c1 $node &>/dev/null; then
            echo "  $node: OK"
        else
            echo "  $node: DOWN"
        fi
    done
else
    echo "ROS:         not sourced"
fi

echo ""
echo "--- Battery ---"
if command -v rostopic &>/dev/null; then
    source /home/ubuntu/ros_ws/devel/setup.bash 2>/dev/null
    BATT=$(timeout 2 rostopic echo -n1 /ros_robot_controller/battery 2>/dev/null | grep "data:" | awk '{printf "%.2fV", $2/1000}' || echo "N/A")
    echo "Voltage:     $BATT"
fi

echo ""
echo "--- Network ---"
echo "WiFi:        $(iwgetid -r 2>/dev/null || echo 'N/A')"
echo "IP:          $(hostname -I | awk '{print $1}')"
REMOTE
}

cmd_logs() {
    local LINES="${1:-50}"
    echo "=== ROS Logs (last $LINES lines) ==="
    eval $SSH_CMD "journalctl -u start_app_node --no-pager -n $LINES"
}

cmd_logs_follow() {
    echo "=== Following ROS Logs (Ctrl+C to stop) ==="
    eval $SSH_CMD "journalctl -u start_app_node -f"
}

cmd_restart() {
    echo "Restarting ROS stack on $IP..."
    eval $SSH_CMD "sudo systemctl restart start_app_node"
    echo "Waiting for startup..."
    sleep 5
    eval $SSH_CMD bash <<'REMOTE'
if systemctl is-active --quiet start_app_node 2>/dev/null; then
    echo "ROS stack: RUNNING"
else
    echo "ROS stack: FAILED - check logs with: scripts/robot.sh logs"
fi
REMOTE
}

cmd_stop() {
    echo "Stopping ROS stack on $IP..."
    eval $SSH_CMD "sudo systemctl stop start_app_node"
    echo "ROS stack stopped."
}

cmd_start() {
    echo "Starting ROS stack on $IP..."
    eval $SSH_CMD "sudo systemctl start start_app_node"
    sleep 3
    eval $SSH_CMD bash <<'REMOTE'
if systemctl is-active --quiet start_app_node 2>/dev/null; then
    echo "ROS stack: RUNNING"
else
    echo "ROS stack: FAILED"
fi
REMOTE
}

cmd_reboot() {
    echo "Rebooting robot at $IP..."
    eval $SSH_CMD "sudo reboot" || true
    echo "Robot is rebooting. Wait ~30s then check: scripts/robot.sh status"
}

cmd_shell() {
    echo "Opening shell on $IP..."
    eval $SSH_CMD
}

cmd_topics() {
    echo "=== ROS Topics ==="
    eval $SSH_CMD bash <<'REMOTE'
source /home/ubuntu/ros_ws/devel/setup.bash 2>/dev/null
rostopic list 2>/dev/null || echo "ERROR: ROS not running"
REMOTE
}

cmd_imu() {
    echo "=== IMU Data (Ctrl+C to stop) ==="
    eval $SSH_CMD bash <<'REMOTE'
source /home/ubuntu/ros_ws/devel/setup.bash 2>/dev/null
rostopic echo /imu 2>/dev/null || echo "ERROR: IMU topic not available"
REMOTE
}

cmd_servos() {
    echo "=== Servo Positions ==="
    eval $SSH_CMD bash <<'REMOTE'
source /home/ubuntu/ros_ws/devel/setup.bash 2>/dev/null
rosservice call /ros_robot_controller/bus_servo/get_position "{}" 2>/dev/null || echo "ERROR: Servo service not available"
REMOTE
}

cmd_camera() {
    echo "Camera stream: http://${IP}:${CAMERA_PORT}/stream?topic=/camera/image_raw&type=mjpeg"
    echo "Snapshot:      http://${IP}:${CAMERA_PORT}/snapshot?topic=/camera/image_raw"
    echo ""
    echo "Open in browser or: mpv http://${IP}:${CAMERA_PORT}/stream?topic=/camera/image_raw"

    # Try to open in browser
    if command -v xdg-open &>/dev/null; then
        read -rp "Open camera stream in browser? [y/N] " yn
        if [[ "$yn" =~ ^[Yy] ]]; then
            xdg-open "http://${IP}:${CAMERA_PORT}/stream?topic=/camera/image_raw&type=mjpeg"
        fi
    fi
}

cmd_walk() {
    local DIRECTION="${1:-forward}"
    local DURATION="${2:-3}"
    echo "Walking $DIRECTION for ${DURATION}s..."
    eval $SSH_CMD bash <<REMOTE
source /home/ubuntu/ros_ws/devel/setup.bash 2>/dev/null
rosservice call /walking/command "command: '$DIRECTION'" 2>/dev/null
sleep $DURATION
rosservice call /walking/command "command: 'stop'" 2>/dev/null
echo "Stopped."
REMOTE
}

cmd_head() {
    local PAN="${1:-0}"
    local TILT="${2:-0}"
    echo "Moving head to pan=$PAN tilt=$TILT..."
    eval $SSH_CMD bash <<REMOTE
source /home/ubuntu/ros_ws/devel/setup.bash 2>/dev/null
rostopic pub -1 /head_pan_controller/command std_msgs/Float64 "data: $PAN" 2>/dev/null &
rostopic pub -1 /head_tilt_controller/command std_msgs/Float64 "data: $TILT" 2>/dev/null &
wait
echo "Done."
REMOTE
}

usage() {
    cat <<EOF
AiNex Robot Management

Usage: scripts/robot.sh <command> [args]

Connection:
  status              Full system status (battery, ROS, sensors)
  shell               Open SSH shell on robot
  reboot              Reboot the robot

ROS Stack:
  start               Start ROS stack
  stop                Stop ROS stack
  restart             Restart ROS stack
  logs [N]            Show last N log lines (default: 50)
  logs-follow         Follow logs in real-time
  topics              List all ROS topics

Sensors:
  imu                 Stream IMU data
  servos              Read all servo positions
  camera              Show camera stream URL

Control:
  walk [dir] [secs]   Walk in direction (forward/backward/left/right) for N seconds
  head [pan] [tilt]   Move head (radians)

Environment:
  ROBOT_IP=x.x.x.x   Override robot IP
  ROBOT_USER=ubuntu   Override SSH user
EOF
}

# Dispatch
case "${1:-help}" in
    status)     cmd_status ;;
    logs)       cmd_logs "${2:-50}" ;;
    logs-follow|logf) cmd_logs_follow ;;
    restart)    cmd_restart ;;
    stop)       cmd_stop ;;
    start)      cmd_start ;;
    reboot)     cmd_reboot ;;
    shell|ssh)  cmd_shell ;;
    topics)     cmd_topics ;;
    imu)        cmd_imu ;;
    servos)     cmd_servos ;;
    camera)     cmd_camera ;;
    walk)       cmd_walk "${2:-forward}" "${3:-3}" ;;
    head)       cmd_head "${2:-0}" "${3:-0}" ;;
    help|--help|-h) usage ;;
    *)          echo "Unknown command: $1"; usage; exit 1 ;;
esac
