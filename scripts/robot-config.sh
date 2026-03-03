#!/usr/bin/env bash
# Robot connection configuration
# Edit these values to match your robot's network setup

# Robot SSH connection
ROBOT_USER="${ROBOT_USER:-ubuntu}"
ROBOT_IP="${ROBOT_IP:-}"  # Set via env or auto-discover
ROBOT_SSH_PORT="${ROBOT_SSH_PORT:-22}"
ROBOT_PASSWORD="${ROBOT_PASSWORD:-ubuntu}"  # Default Hiwonder password

# Paths on robot
ROBOT_ROS_WS="/home/ubuntu/ros_ws"
ROBOT_CODE_DIR="/home/ubuntu/ainex-robot-code"

# Paths on host
HOST_CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Network discovery
SCAN_SUBNET="${SCAN_SUBNET:-192.168.1.0/24}"

# Bridge ports
BRIDGE_PORT="${BRIDGE_PORT:-9100}"
ROSBRIDGE_PORT="${ROSBRIDGE_PORT:-9090}"
CAMERA_PORT="${CAMERA_PORT:-8080}"

# SSH options (no host key checking for robot, connection timeout)
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=5 -o LogLevel=ERROR"
