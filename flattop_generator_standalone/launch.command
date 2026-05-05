#!/bin/bash
# macOS launcher — double-click to start the Flattop Beam Generator.

cd "$(dirname "$0")"

PORT=8766

# Kill any previous instance on this port
lsof -ti tcp:$PORT 2>/dev/null | xargs kill -9 2>/dev/null

# Pick a local venv if present, otherwise system python3
if [ -x ".venv/bin/python3" ]; then
    PY=".venv/bin/python3"
elif [ -x "../.venv/bin/python3" ]; then
    PY="../.venv/bin/python3"
else
    PY="python3"
fi

echo "Starting Flattop Beam Generator..."
echo "Loading tensors.mat — this takes several seconds (longer on CPU-only)."
echo ""

$PY generator_server.py &
SERVER_PID=$!

URL="http://localhost:$PORT/Flattop_beam_with_Zernike_aberrations.html"
for i in {1..60}; do
    sleep 0.5
    if curl -s -o /dev/null --max-time 1 "$URL"; then
        break
    fi
done

open "$URL"

echo ""
echo "✅ Running at: $URL"
echo "Press Ctrl+C or close this window to stop the server."

trap "kill $SERVER_PID 2>/dev/null; echo 'Server stopped.'; exit 0" INT TERM
wait $SERVER_PID
