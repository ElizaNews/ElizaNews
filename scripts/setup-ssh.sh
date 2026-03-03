#!/usr/bin/env bash
set -euo pipefail

# Set up SSH key-based auth to the AiNex robot
# Usage: scripts/setup-ssh.sh [robot-ip]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/robot-config.sh"

ROBOT_IP="${1:-$ROBOT_IP}"

if [[ -z "$ROBOT_IP" ]]; then
    echo "Usage: scripts/setup-ssh.sh <robot-ip>"
    echo "   or: ROBOT_IP=192.168.x.x scripts/setup-ssh.sh"
    exit 1
fi

echo "=== AiNex SSH Setup ==="
echo "Robot: ${ROBOT_USER}@${ROBOT_IP}"
echo ""

# 1. Generate SSH key if needed
SSH_KEY="$HOME/.ssh/ainex_robot"
if [[ ! -f "$SSH_KEY" ]]; then
    echo "[1/5] Generating SSH key..."
    ssh-keygen -t ed25519 -f "$SSH_KEY" -N "" -C "ainex-robot-dev"
    echo "  Created: $SSH_KEY"
else
    echo "[1/5] SSH key exists: $SSH_KEY"
fi

# 2. Copy key to robot
echo "[2/5] Copying SSH key to robot (you may need to enter password: '$ROBOT_PASSWORD')..."
ssh-copy-id -i "$SSH_KEY.pub" $SSH_OPTS -p "$ROBOT_SSH_PORT" "${ROBOT_USER}@${ROBOT_IP}" 2>/dev/null || {
    echo "  ssh-copy-id failed, trying manual copy..."
    cat "$SSH_KEY.pub" | ssh $SSH_OPTS -p "$ROBOT_SSH_PORT" "${ROBOT_USER}@${ROBOT_IP}" \
        'mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys'
}

# 3. Test connection
echo "[3/5] Testing SSH connection..."
if ssh $SSH_OPTS -i "$SSH_KEY" -p "$ROBOT_SSH_PORT" "${ROBOT_USER}@${ROBOT_IP}" "echo 'SSH OK'" 2>/dev/null; then
    echo "  SSH key auth working!"
else
    echo "  ERROR: SSH key auth failed"
    exit 1
fi

# 4. Add SSH config entry
SSH_CONFIG="$HOME/.ssh/config"
if ! grep -q "Host ainex" "$SSH_CONFIG" 2>/dev/null; then
    echo "[4/5] Adding SSH config entry..."
    cat >> "$SSH_CONFIG" <<EOF

# AiNex Robot
Host ainex
    HostName $ROBOT_IP
    User $ROBOT_USER
    Port $ROBOT_SSH_PORT
    IdentityFile $SSH_KEY
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    LogLevel ERROR
EOF
    chmod 600 "$SSH_CONFIG"
    echo "  Added 'ainex' to ~/.ssh/config"
    echo "  You can now use: ssh ainex"
else
    echo "[4/5] SSH config entry 'ainex' already exists"
    # Update the IP in case it changed
    sed -i "s/HostName .*/HostName $ROBOT_IP/" <(awk '/^Host ainex$/,/^Host /{print}' "$SSH_CONFIG") 2>/dev/null || true
fi

# 5. Cache the IP
echo "$ROBOT_IP" > "$SCRIPT_DIR/.robot-ip"

echo "[5/5] Verifying robot status..."
ssh $SSH_OPTS -i "$SSH_KEY" -p "$ROBOT_SSH_PORT" "${ROBOT_USER}@${ROBOT_IP}" bash <<'REMOTE'
echo "  Hostname: $(hostname)"
echo "  Uptime: $(uptime -p)"
echo "  Kernel: $(uname -r)"
echo "  ROS workspace: $(ls /home/ubuntu/ros_ws/src/ 2>/dev/null | wc -l) packages"
if systemctl is-active --quiet start_app_node 2>/dev/null; then
    echo "  ROS stack: RUNNING"
else
    echo "  ROS stack: STOPPED"
fi
REMOTE

echo ""
echo "=== Setup complete ==="
echo "Quick commands:"
echo "  ssh ainex                         # SSH into robot"
echo "  scripts/robot.sh status           # Check robot status"
echo "  scripts/robot.sh logs             # View ROS logs"
echo "  scripts/robot.sh restart          # Restart ROS stack"
echo "  scripts/deploy.sh                 # Deploy bridge code"
echo "  scripts/connect.sh                # Start bridge connection"
