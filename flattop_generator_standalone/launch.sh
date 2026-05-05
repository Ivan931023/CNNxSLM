#!/bin/bash
# Linux launcher — `bash launch.sh` (or chmod +x and ./launch.sh).

cd "$(dirname "$0")"

PORT=8766

if command -v lsof >/dev/null 2>&1; then
    lsof -ti tcp:$PORT 2>/dev/null | xargs -r kill -9 2>/dev/null
elif command -v fuser >/dev/null 2>&1; then
    fuser -k $PORT/tcp 2>/dev/null
fi

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
for i in $(seq 1 60); do
    sleep 0.5
    if curl -s -o /dev/null --max-time 1 "$URL" 2>/dev/null; then
        break
    fi
done

if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$URL" >/dev/null 2>&1 &
else
    echo "(xdg-open not found — open this URL manually:)"
fi

echo ""
echo "✅ Running at: $URL"
echo "Press Ctrl+C to stop the server."

trap "kill $SERVER_PID 2>/dev/null; echo 'Server stopped.'; exit 0" INT TERM
wait $SERVER_PID
