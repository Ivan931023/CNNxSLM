#!/bin/bash
# Linux launcher for the Flattop Beam Generator (Zernike aberrations).
# Usage: bash launch_generator.sh   (or chmod +x and ./launch_generator.sh)

cd "$(dirname "$0")"

PORT=8766

# Kill any previous instance on this port (ignore errors if lsof not installed)
if command -v lsof >/dev/null 2>&1; then
    lsof -ti tcp:$PORT 2>/dev/null | xargs -r kill -9 2>/dev/null
elif command -v fuser >/dev/null 2>&1; then
    fuser -k $PORT/tcp 2>/dev/null
fi

# Pick the project venv if available, otherwise system python3
if [ -x ".venv/bin/python3" ]; then
    PY=".venv/bin/python3"
else
    PY="python3"
fi

echo "Starting Flattop Beam Generator server..."
echo "Loading tensors.mat — this takes several seconds (longer on CPU-only)."
echo ""

$PY app/generator_server.py &
SERVER_PID=$!

# Wait for the server to come up before opening the browser.
URL="http://localhost:$PORT/Flattop_beam_with_Zernike_aberrations.html"
for i in $(seq 1 60); do
    sleep 0.5
    if curl -s -o /dev/null --max-time 1 "$URL" 2>/dev/null; then
        break
    fi
done

# Open in default browser (xdg-open is the Linux standard)
if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$URL" >/dev/null 2>&1 &
else
    echo "(xdg-open not found — please open this URL manually:)"
fi

echo ""
echo "✅ Generator running at: $URL"
echo ""
echo "Press Ctrl+C to stop the server."

trap "kill $SERVER_PID 2>/dev/null; echo 'Server stopped.'; exit 0" INT TERM
wait $SERVER_PID
