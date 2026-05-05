#!/bin/bash
# Double-click to launch the Flattop Beam Generator (Zernike aberrations).

cd "$(dirname "$0")"

PORT=8766

# Kill any previous instance on this port
lsof -ti tcp:$PORT | xargs kill -9 2>/dev/null

# Pick the project venv if available, otherwise system python3
if [ -x ".venv/bin/python3" ]; then
    PY=".venv/bin/python3"
else
    PY="python3"
fi

echo "Starting Flattop Beam Generator server..."
echo "Loading tensors.mat — this takes a few seconds the first time."
echo ""

$PY app/generator_server.py &
SERVER_PID=$!

# Wait for the server to come up before opening the browser.
URL="http://localhost:$PORT/Flattop_beam_with_Zernike_aberrations.html"
for i in {1..40}; do
    sleep 0.5
    if curl -s -o /dev/null --max-time 1 "$URL"; then
        break
    fi
done

open "$URL"

echo ""
echo "✅ Generator running at: $URL"
echo ""
echo "Press Ctrl+C or close this window to stop the server."

trap "kill $SERVER_PID 2>/dev/null; echo 'Server stopped.'; exit 0" INT TERM
wait $SERVER_PID
