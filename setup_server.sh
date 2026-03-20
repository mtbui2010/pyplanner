#!/bin/bash
# =============================================================================
# setup_server.sh
# Full setup for PyPlanner demo server on Ubuntu + NVIDIA GPU
#
# Project layout expected (flat pyplanner):
#   <project_root>/
#     pyplanner/          ← package (pip install -e .)
#       __init__.py
#       base.py  cot.py  react.py  ...
#     thor_app/           ← demo application
#       app.py  thor_server.py  thor_client.py  ...
#     demo_app.py
#     setup_server.sh     ← this file
#
# What this does:
#   1. Install system dependencies (Xvfb, nginx, certbot)
#   2. Install Python deps (ai2thor, pyzmq, streamlit)
#   3. pip install -e .  (pyplanner from project root)
#   4. Configure Xvfb virtual display for AI2-THOR headless
#   5. Create systemd services (auto-start on reboot)
#   6. Configure nginx reverse proxy for:
#        ollama.aistations.org  → :11434
#        demo-planner.aistations.org    → :8501  (Streamlit UI, optional)
#   7. SSL via Let's Encrypt
#
# Usage:
#   chmod +x setup_server.sh
#
#   If using conda (recommended):
#     conda activate <your-env>
#     sudo env PATH=$PATH bash setup_server.sh
#
#   Or specify Python explicitly:
#     sudo PYTHON_BIN_OVERRIDE=$(which python) bash setup_server.sh
#
#   Bare sudo (uses system python — may need --break-system-packages):
#     sudo bash setup_server.sh
#
# After running:
#   Streamlit Cloud secrets:
#     OLLAMA_HOST      = "https://ollama.aistations.org"
#     THOR_SERVER_URL  = "tcp://<public-ip>:5555"   ← after port forwarding
# =============================================================================

set -e
DOMAIN="aistations.org"
DEMO_SUBDOMAIN="demo-planner"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
THOR_APP_DIR="$PROJECT_DIR/thor_app"
USER_NAME="$(logname 2>/dev/null || echo $SUDO_USER)"
HOME_DIR="/home/$USER_NAME"

echo ""
echo "======================================================"
echo "  PyPlanner Demo Server Setup"
echo "======================================================"

# ── 0. Python binary detection ──────────────────────────────────────
# sudo loses conda PATH — we find the right python explicitly
echo "▶ Detecting Python binary..."

# Try candidates in order: conda env of calling user, then system
PYTHON_BIN=""
SUDO_USER_HOME=$(eval echo "~${SUDO_USER:-$USER}")

# Candidate list: conda envs, pyenv, system python3.10+
CANDIDATES=(
    # Conda base and named envs of the calling user
    "$SUDO_USER_HOME/miniconda3/bin/python"
    "$SUDO_USER_HOME/miniconda3/envs/base/bin/python"
    "$SUDO_USER_HOME/anaconda3/bin/python"
    "$SUDO_USER_HOME/anaconda3/envs/base/bin/python"
    # Common conda env names for this project
    "$SUDO_USER_HOME/miniconda3/envs/pyplanner/bin/python"
    "$SUDO_USER_HOME/anaconda3/envs/pyplanner/bin/python"
    # pyenv
    "$SUDO_USER_HOME/.pyenv/shims/python3"
    # System python3.10+
    "/usr/bin/python3.10"
    "/usr/local/bin/python3.10"
    # Fallback
    "$(which python3 2>/dev/null)"
)

for candidate in "${CANDIDATES[@]}"; do
    [ -z "$candidate" ] && continue
    [ ! -x "$candidate" ] && continue
    VER=$("$candidate" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
    MAJOR=$(echo "$VER" | cut -d. -f1)
    MINOR=$(echo "$VER" | cut -d. -f2)
    if [ "${MAJOR:-0}" -ge 3 ] && [ "${MINOR:-0}" -ge 8 ]; then
        PYTHON_BIN="$candidate"
        echo "  ✅ Found Python $VER at $candidate"
        break
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "  ❌ No suitable Python found. Activate your conda env and re-run:"
    echo "     conda activate <your-env>"
    echo "     sudo env PATH=\$PATH bash setup_server.sh"
    exit 1
fi

# Allow caller to override via PYTHON_BIN env var
# e.g.: sudo PYTHON_BIN=/path/to/python bash setup_server.sh
if [ -n "$PYTHON_BIN_OVERRIDE" ]; then
    PYTHON_BIN="$PYTHON_BIN_OVERRIDE"
    echo "  🔧 Using override: $PYTHON_BIN"
fi

echo "  ✅ Using: $PYTHON_BIN ($("$PYTHON_BIN" --version 2>&1))"

echo ""
echo "======================================================"
echo "  Setup (continued)"
echo "  Domain : $DOMAIN"
echo "  Project: $PROJECT_DIR"
echo "  User   : $USER_NAME"
echo "======================================================"
echo ""

# ── 1. System packages ──────────────────────────────────────────────
echo "▶ Installing system packages..."
apt-get update -qq
apt-get install -y \
    xvfb \
    x11-utils \
    libgl1-mesa-glx \
    libglu1-mesa \
    nginx \
    certbot \
    python3-certbot-nginx \
    python3-pip \
    python3-venv \
    curl \
    git \
    supervisor

# ── 2. Python virtual environment ───────────────────────────────────
echo "▶ Setting up Python environment..."
VENV="$HOME_DIR/pyplanner-env"

# If conda env is active, use it directly instead of creating a new venv
# This avoids venv-inside-conda issues
CONDA_PREFIX_DETECTED=$("$PYTHON_BIN" -c "import sys; print(sys.prefix)" 2>/dev/null)
if echo "$CONDA_PREFIX_DETECTED" | grep -qiE "conda|miniconda|anaconda"; then
    echo "  ✅ Using conda env directly: $CONDA_PREFIX_DETECTED"
    # Use conda python's pip directly — no venv needed
    VENV_PYTHON="$PYTHON_BIN"
else
    # Not conda — create a venv as before
    if [ ! -d "$VENV" ]; then
        "$PYTHON_BIN" -m venv "$VENV"
    fi
    source "$VENV/bin/activate"
    PIP_BIN="pip"
    VENV_PYTHON="$VENV/bin/python"
fi

echo "  ▸ Installing Python packages..."
$PYTHON_BIN -m pip install --upgrade pip -q
$PYTHON_BIN -m pip install     ai2thor     pyzmq     pillow     streamlit     pandas     requests     ollama     openai     anthropic     -q

# Install pyplanner package
# Supports both layouts:
#   flat:   pyplanner/__init__.py at root  → pip install -e .
#   nested: pyplanner/pyplanner/__init__.py → pip install -e ./pyplanner
if [ -f "$PROJECT_DIR/pyproject.toml" ] || [ -f "$PROJECT_DIR/setup.py" ]; then
    $PYTHON_BIN -m pip install -e "$PROJECT_DIR" -q
    echo "  ✅ pyplanner installed from project root (flat layout)"
elif [ -f "$PROJECT_DIR/pyplanner/pyproject.toml" ]; then
    $PYTHON_BIN -m pip install -e "$PROJECT_DIR/pyplanner" -q
    echo "  ✅ pyplanner installed from pyplanner/ subdirectory"
else
    echo "  ❌ Cannot find pyproject.toml — check project structure"
    exit 1
fi

echo "  ✅ Python environment ready: $VENV"

# ── 3. Xvfb virtual display ─────────────────────────────────────────
echo "▶ Configuring Xvfb virtual display..."

cat > /etc/systemd/system/xvfb.service << EOF
[Unit]
Description=Xvfb virtual display for AI2-THOR
After=network.target

[Service]
ExecStart=/usr/bin/Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX +render -noreset
Restart=always
RestartSec=3
User=root

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable xvfb
systemctl restart xvfb
sleep 2

if systemctl is-active --quiet xvfb; then
    echo "  ✅ Xvfb running on :99"
else
    echo "  ❌ Xvfb failed to start — check: journalctl -u xvfb"
    exit 1
fi

# ── 4. ThorServer systemd service ───────────────────────────────────
echo "▶ Configuring ThorServer service..."

cat > /etc/systemd/system/thor-server.service << EOF
[Unit]
Description=PyPlanner AI2-THOR Server
After=network.target xvfb.service
Requires=xvfb.service

[Service]
Type=simple
User=$USER_NAME
WorkingDirectory=$THOR_APP_DIR
Environment=DISPLAY=:99
Environment=PATH=$VENV/bin:/usr/local/bin:/usr/bin:/bin
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONPATH=$PROJECT_DIR
ExecStart=$VENV_PYTHON thor_server.py
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable thor-server
systemctl restart thor-server
sleep 5

if systemctl is-active --quiet thor-server; then
    echo "  ✅ ThorServer running on :5555"
else
    echo "  ⚠  ThorServer not yet ready (AI2-THOR first run may take ~60s to download)"
    echo "     Check: journalctl -u thor-server -f"
fi

# ── 5. Streamlit demo service (optional — use Streamlit Cloud instead) ──
echo "▶ Configuring Streamlit service..."

cat > /etc/systemd/system/streamlit-demo.service << EOF
[Unit]
Description=PyPlanner Streamlit Demo
After=network.target thor-server.service

[Service]
Type=simple
User=$USER_NAME
WorkingDirectory=$PROJECT_DIR
Environment=PATH=$VENV/bin:/usr/local/bin:/usr/bin:/bin
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONPATH=$PROJECT_DIR
Environment=OLLAMA_HOST=https://ollama.${DOMAIN}
Environment=DEMO_HOST=https://demo-planner.aistations.org
Environment=THOR_SERVER_URL=tcp://localhost:5555
ExecStart=$VENV_PYTHON -m streamlit run demo_app.py --server.port 8501 --server.headless true
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable streamlit-demo
systemctl restart streamlit-demo
sleep 3
echo "  ✅ Streamlit running on :8501"

# ── 6. Nginx configuration ──────────────────────────────────────────
echo "▶ Configuring nginx..."

# ollama.aistations.org → :11434
cat > /etc/nginx/sites-available/ollama << EOF
server {
    listen 80;
    server_name ollama.${DOMAIN};

    location / {
        proxy_pass         http://127.0.0.1:11434;
        proxy_http_version 1.1;
        proxy_set_header   Host              \$host;
        proxy_set_header   X-Real-IP         \$remote_addr;
        proxy_set_header   X-Forwarded-For   \$proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto \$scheme;
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
        # Allow large model payloads
        client_max_body_size 100m;
    }
}
EOF

# demo-planner.aistations.org → :8501  (Streamlit — optional if using Streamlit Cloud)
cat > /etc/nginx/sites-available/demo << EOF
server {
    listen 80;
    server_name ${DEMO_SUBDOMAIN}.${DOMAIN};

    location / {
        proxy_pass         http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header   Host              \$host;
        proxy_set_header   X-Real-IP         \$remote_addr;
        proxy_set_header   Upgrade           \$http_upgrade;
        proxy_set_header   Connection        "upgrade";
        proxy_read_timeout 86400;
    }

    # Streamlit websocket endpoint
    location /_stcore/stream {
        proxy_pass         http://127.0.0.1:8501/_stcore/stream;
        proxy_http_version 1.1;
        proxy_set_header   Upgrade    \$http_upgrade;
        proxy_set_header   Connection "upgrade";
        proxy_read_timeout 86400;
    }
}
EOF

# Enable sites
ln -sf /etc/nginx/sites-available/ollama /etc/nginx/sites-enabled/
ln -sf /etc/nginx/sites-available/demo   /etc/nginx/sites-enabled/

nginx -t && systemctl reload nginx
echo "  ✅ nginx configured"

# ── 7. SSL certificates ─────────────────────────────────────────────
echo "▶ Requesting SSL certificates..."
echo "  (requires DNS records to point to this machine)"
echo ""

read -p "  Issue SSL certificates now? [y/N] " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    certbot --nginx \
        -d "ollama.${DOMAIN}" \
        -d "${DEMO_SUBDOMAIN}.${DOMAIN}" \
        --non-interactive \
        --agree-tos \
        --email "admin@${DOMAIN}" \
        --redirect
    echo "  ✅ SSL certificates issued"
else
    echo "  ⏭  Skipped. Two options to get SSL:"
    echo ""
    echo "  Option A — HTTP challenge (requires port 80 open, Cloudflare proxy OFF):"
    echo "    certbot --nginx \\"
    echo "      -d ollama.${DOMAIN} \\"
    echo "      -d ${DEMO_SUBDOMAIN}.${DOMAIN} \\"
    echo "      --agree-tos --email admin@${DOMAIN}"
    echo ""
    echo "  Option B — DNS challenge via Cloudflare (proxy can stay ON, port 80 not needed):"
    echo "    pip install certbot-dns-cloudflare"
    echo "    # Get API token: Cloudflare → My Profile → API Tokens"
    echo "    # Permissions: Zone → DNS → Edit → your domain"
    echo "    mkdir -p ~/.secrets"
    echo "    echo 'dns_cloudflare_api_token = YOUR_CF_TOKEN' > ~/.secrets/cloudflare.ini"
    echo "    chmod 600 ~/.secrets/cloudflare.ini"
    echo "    certbot certonly --dns-cloudflare \\"
    echo "      --dns-cloudflare-credentials ~/.secrets/cloudflare.ini \\"
    echo "      -d ollama.${DOMAIN} \\"
    echo "      -d ${DEMO_SUBDOMAIN}.${DOMAIN} \\"
    echo "      --agree-tos --email admin@${DOMAIN}"
    echo "    # Then link certs to nginx:"
    echo "    certbot install --nginx \\"
    echo "      -d ollama.${DOMAIN} \\"
    echo "      -d ${DEMO_SUBDOMAIN}.${DOMAIN}"
fi

# ── 8. NVIDIA GPU check for AI2-THOR ────────────────────────────────
echo ""
echo "▶ Checking GPU for AI2-THOR..."
if command -v nvidia-smi &> /dev/null; then
    GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    echo "  ✅ GPU detected: $GPU"
    # Make sure VirtualGL or proper OpenGL is available
    if dpkg -l | grep -q "nvidia-driver"; then
        echo "  ✅ NVIDIA driver installed"
    else
        echo "  ⚠  Install NVIDIA driver if AI2-THOR has rendering issues:"
        echo "     sudo apt install nvidia-driver-535"
    fi
else
    echo "  ⚠  nvidia-smi not found — AI2-THOR will use CPU rendering (slower)"
fi

# ── 9. Test ThorServer connection ───────────────────────────────────
echo ""
echo "▶ Testing ThorServer..."
"$PYTHON_BIN" - << 'PYEOF'
import sys, time
sys.path.insert(0, "thor_app")   # for thor_client
sys.path.insert(0, ".")          # for pyplanner (flat layout)
try:
    from thor_client import ThorClient
    client = ThorClient(host="localhost", port=5555)
    if client.connected:
        print("  ✅ ThorServer responding on localhost:5555")
    else:
        print("  ⚠  ThorServer not yet responding (may still be loading AI2-THOR)")
        print("     Check: journalctl -u thor-server -f")
except Exception as e:
    print(f"  ⚠  {e}")
PYEOF

# ── Summary ─────────────────────────────────────────────────────────
echo ""
echo "======================================================"
echo "  Setup complete!"
echo "======================================================"
echo ""
echo "  Services:"
echo "    systemctl status xvfb           # virtual display"
echo "    systemctl status thor-server    # AI2-THOR server"
echo "    systemctl status streamlit-demo # Streamlit UI"
echo ""
echo "  DNS records required (Cloudflare → aistations.org):"
echo "    Type  Name                          Content"
echo "    A     ollama                        192.168.1.11"
echo "    A     ${DEMO_SUBDOMAIN}             192.168.1.18"
echo "    (Proxy: OFF ⬜ for HTTP challenge  /  ON 🟠 then use DNS challenge)"
echo ""
echo "  Endpoints (after DNS + SSL):"
echo "    https://ollama.${DOMAIN}                    → Ollama LLM"
echo "    https://${DEMO_SUBDOMAIN}.${DOMAIN}         → Streamlit UI"
echo "    tcp://192.168.1.18:5555                     → ThorServer (needs router port forward)"
echo ""
echo "  Streamlit Cloud secrets to set:"
echo "    OLLAMA_HOST     = \"https://ollama.${DOMAIN}\""
echo "    THOR_SERVER_URL = \"tcp://$(hostname -I | awk '{print $1}'):5555\""
echo ""
echo "  Logs:"
echo "    journalctl -u thor-server -f"
echo "    journalctl -u streamlit-demo -f"
echo ""