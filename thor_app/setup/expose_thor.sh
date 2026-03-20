#!/bin/bash
# =============================================================================
# expose_thor.sh
# Expose ThorServer (ZMQ TCP :5555) to the internet
#
# ZMQ uses raw TCP — cannot go through nginx HTTP proxy.
# Two options:
#
#   Option A: Direct TCP via router port forwarding (simplest)
#     Router: forward external port 5555 → 192.168.1.11:5555
#     Streamlit secret: THOR_SERVER_URL = "tcp://<your-public-ip>:5555"
#
#   Option B: stunnel SSL wrapper (secure, works through firewalls)
#     Wraps ZMQ TCP in TLS → accessible as thor.aistations.org:5556
#     Streamlit secret: THOR_SERVER_URL = "tcp://thor.aistations.org:5556"
#
#   Option C: socat TCP relay on same machine (for testing)
#
# =============================================================================

set -e
DOMAIN="aistations.org"

echo ""
echo "  ThorServer (ZMQ :5555) exposure options"
echo ""
echo "  ZMQ is raw TCP — nginx HTTP proxy cannot forward it directly."
echo "  Choose your method:"
echo ""
echo "  [A] Router port forwarding (requires router admin access)"
echo "  [B] stunnel SSL wrapper    (recommended for security)"
echo "  [C] Just use local IP      (Streamlit Cloud → your public IP)"
echo ""
read -p "  Choice [A/B/C]: " -n 1 choice
echo ""

case "${choice^^}" in

# ── Option A: Router port forwarding ────────────────────────────────
A)
    echo ""
    echo "  Router port forwarding setup:"
    echo "  ─────────────────────────────"
    PUBLIC_IP=$(curl -s ifconfig.me 2>/dev/null || echo "<your-public-ip>")
    echo ""
    echo "  1. Login to your router (usually 192.168.1.1)"
    echo "  2. Add port forwarding rule:"
    echo "       External port : 5555"
    echo "       Internal IP   : 192.168.1.11"
    echo "       Internal port : 5555"
    echo "       Protocol      : TCP"
    echo ""
    echo "  3. Set in Streamlit Cloud secrets:"
    echo "       THOR_SERVER_URL = \"tcp://$PUBLIC_IP:5555\""
    echo ""
    echo "  4. Or add DNS record:"
    echo "       thor.$DOMAIN  A  $PUBLIC_IP"
    echo "     Then use:"
    echo "       THOR_SERVER_URL = \"tcp://thor.$DOMAIN:5555\""
    echo ""
    echo "  ⚠  Make sure port 5555 is not blocked by your ISP."
    echo "     Test with: nc -zv $PUBLIC_IP 5555"
    ;;

# ── Option B: stunnel SSL wrapper ───────────────────────────────────
B)
    echo ""
    echo "  Installing stunnel..."
    apt-get install -y stunnel4 -q

    # Generate self-signed cert for stunnel
    SSL_DIR="/etc/stunnel"
    mkdir -p "$SSL_DIR"

    if [ ! -f "$SSL_DIR/thor.pem" ]; then
        openssl req -new -x509 -days 3650 -nodes \
            -out "$SSL_DIR/thor.pem" \
            -keyout "$SSL_DIR/thor.pem" \
            -subj "/CN=thor.$DOMAIN" 2>/dev/null
        chmod 600 "$SSL_DIR/thor.pem"
        echo "  ✅ Self-signed certificate generated"
    fi

    # stunnel server config (on this machine — wraps :5555 in TLS on :5556)
    cat > /etc/stunnel/thor-server.conf << EOF
[thor-zmq-server]
accept  = 0.0.0.0:5556
connect = 127.0.0.1:5555
cert    = /etc/stunnel/thor.pem
EOF

    # Enable stunnel
    sed -i 's/ENABLED=0/ENABLED=1/' /etc/default/stunnel4 2>/dev/null || true
    systemctl enable stunnel4 2>/dev/null || true
    service stunnel4 restart

    echo "  ✅ stunnel wrapping :5555 → TLS :5556"
    echo ""

    # stunnel client config (on Streamlit Cloud side)
    # We need a different approach — update thor_client.py to support TLS
    echo "  Note: ThorClient also needs stunnel on the client side."
    echo "  Simpler alternative: use the SSH tunnel approach below."
    echo ""
    echo "  Set in Streamlit Cloud secrets:"
    PUBLIC_IP=$(curl -s ifconfig.me 2>/dev/null || echo "<your-public-ip>")
    echo "    THOR_SERVER_URL = \"tcp://$PUBLIC_IP:5556\""
    echo ""
    echo "  ⚠  Also need to forward port 5556 on your router."
    ;;

# ── Option C: Use public IP directly ────────────────────────────────
C)
    PUBLIC_IP=$(curl -s ifconfig.me 2>/dev/null || echo "<your-public-ip>")
    echo ""
    echo "  Your public IP: $PUBLIC_IP"
    echo ""
    echo "  Steps:"
    echo "  1. Forward port 5555 on your router:"
    echo "       External: 5555  →  Internal: 192.168.1.11:5555"
    echo ""
    echo "  2. Open firewall on this machine:"
    echo "       sudo ufw allow 5555/tcp"
    echo ""
    echo "  3. Set in Streamlit Cloud secrets:"
    echo "       THOR_SERVER_URL = \"tcp://$PUBLIC_IP:5555\""
    echo ""
    ufw allow 5555/tcp 2>/dev/null && echo "  ✅ Firewall rule added for :5555" || echo "  (ufw not available, check your firewall manually)"
    ;;

*)
    echo "  Invalid choice."
    exit 1
    ;;
esac

echo ""
echo "  ── Quick test after setup ──────────────────────────────"
echo "  python3 thor_app/thor_client.py  # should connect"
echo "  journalctl -u thor-server -f     # watch logs"
echo ""