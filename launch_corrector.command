#!/bin/bash
# Double-click to launch the Zernike Aberrations Corrector (closed-loop iteration page).
# Reuses the same backend server as launch_generator.command — if that server is
# already running, this just opens the corrector page in the browser.

cd "$(dirname "$0")"

PORT=8766
URL="http://localhost:$PORT/Zernike_aberrations_corrector.html"

# If a server is already up on this port, just open the page and exit.
if curl -s -o /dev/null --max-time 1 "$URL"; then
    echo "Backend already running on port $PORT — opening corrector page."
    open "$URL"
    exit 0
fi

# Pick the project venv if available, otherwise system python3
if [ -x ".venv/bin/python3" ]; then
    PY=".venv/bin/python3"
else
    PY="python3"
fi

echo "Starting backend server for the corrector..."
echo "Loading tensors.mat — this takes a few seconds the first time."
echo ""

$PY app/generator_server.py &
SERVER_PID=$!

# Wait for the server to come up before opening the browser.
for i in {1..40}; do
    sleep 0.5
    if curl -s -o /dev/null --max-time 1 "$URL"; then
        break
    fi
done

open "$URL"

echo ""
echo "✅ Corrector running at: $URL"
echo ""
echo "Press Ctrl+C or close this window to stop the server."

trap "kill $SERVER_PID 2>/dev/null; echo 'Server stopped.'; exit 0" INT TERM
wait $SERVER_PID
