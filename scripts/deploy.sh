#!/usr/bin/env bash
set -euo pipefail

# Deploy code to the AiNex robot over SSH
# Syncs bridge code and configs, optionally restarts services
#
# Usage: scripts/deploy.sh [--restart] [--full]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/robot-config.sh"

RESTART=false
FULL=false

for arg in "$@"; do
    case "$arg" in
        --restart) RESTART=true ;;
        --full)    FULL=true ;;
        --help|-h)
            echo "Usage: scripts/deploy.sh [--restart] [--full]"
            echo ""
            echo "  --restart  Restart ROS stack after deploy"
            echo "  --full     Deploy everything (bridge + ROS configs + walking params)"
            echo ""
            echo "Default: deploy bridge code only (runs on host, not robot)"
            exit 0
            ;;
    esac
done

# Resolve robot IP
if [[ -n "$ROBOT_IP" ]]; then
    IP="$ROBOT_IP"
elif [[ -f "$SCRIPT_DIR/.robot-ip" ]]; then
    IP="$(cat "$SCRIPT_DIR/.robot-ip")"
else
    echo "ERROR: Robot IP not set. Run: scripts/setup-ssh.sh <ip>"
    exit 1
fi

echo "=== Deploying to AiNex Robot ==="
echo "Robot: ${ROBOT_USER}@${IP}"
echo "Mode:  $(if $FULL; then echo 'FULL'; else echo 'bridge only'; fi)"
echo ""

# Test connection
if ! ssh $SSH_OPTS "${ROBOT_USER}@${IP}" "true" 2>/dev/null; then
    echo "ERROR: Cannot connect to robot. Check SSH setup."
    exit 1
fi

# 1. Always deploy bridge code to robot
echo "[1/4] Syncing bridge code..."
rsync -avz --delete \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.git' \
    --exclude 'node_modules' \
    -e "ssh $SSH_OPTS" \
    "$HOST_CODE_DIR/bridge/" \
    "${ROBOT_USER}@${IP}:${ROBOT_CODE_DIR}/bridge/"

echo "  Bridge code synced."

# 2. Sync training interfaces (needed by bridge)
echo "[2/4] Syncing training interfaces..."
rsync -avz \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    -e "ssh $SSH_OPTS" \
    "$HOST_CODE_DIR/training/__init__.py" \
    "$HOST_CODE_DIR/training/interfaces.py" \
    "${ROBOT_USER}@${IP}:${ROBOT_CODE_DIR}/training/"

echo "  Interfaces synced."

# 3. If full deploy, sync ROS configs
if $FULL; then
    echo "[3/4] Syncing ROS configs..."

    # Walking parameters
    rsync -avz \
        -e "ssh $SSH_OPTS" \
        "$HOST_CODE_DIR/ros_ws_src/ainex_driver/ainex_kinematics/config/" \
        "${ROBOT_USER}@${IP}:${ROBOT_ROS_WS}/src/ainex_driver/ainex_kinematics/config/"

    echo "  Walking configs synced."
else
    echo "[3/4] Skipping ROS configs (use --full to include)"
fi

# 4. Install bridge systemd service
echo "[4/4] Updating bridge service..."
ssh $SSH_OPTS "${ROBOT_USER}@${IP}" bash <<REMOTE
# Create code directory if needed
mkdir -p ${ROBOT_CODE_DIR}

# Install systemd service for bridge auto-start
sudo cp ${ROBOT_CODE_DIR}/bridge/systemd/ainex-bridge.service /etc/systemd/system/ 2>/dev/null || true
sudo systemctl daemon-reload 2>/dev/null || true
sudo systemctl enable ainex-bridge 2>/dev/null || true
echo "  Bridge service installed."
REMOTE

# Restart if requested
if $RESTART; then
    echo ""
    echo "Restarting services..."
    ssh $SSH_OPTS "${ROBOT_USER}@${IP}" bash <<'REMOTE'
sudo systemctl restart ainex-bridge 2>/dev/null || true
sudo systemctl restart start_app_node 2>/dev/null || true
echo "Services restarted."
REMOTE
    sleep 3
    "$SCRIPT_DIR/robot.sh" status
fi

echo ""
echo "=== Deploy complete ==="
echo ""
echo "Next steps:"
echo "  scripts/robot.sh status    # Check robot status"
echo "  scripts/connect.sh         # Start bridge connection"
