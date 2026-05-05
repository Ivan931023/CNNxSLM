#!/bin/bash
# Double-click this file to launch Zernike Flattop Analyzer in your browser.

cd "$(dirname "$0")"

PORT=8765

# Kill any existing server on this port
lsof -ti tcp:$PORT | xargs kill -9 2>/dev/null

echo "Starting local server on http://localhost:$PORT ..."
python3 -m http.server $PORT &
SERVER_PID=$!

sleep 1

# Open browser
open "http://localhost:$PORT/Zernike_predictor.html"

echo ""
echo "✅ Zernike Flattop Analyzer is running."
echo "   URL: http://localhost:$PORT/Zernike_predictor.html"
echo ""
echo "Press Ctrl+C or close this window to stop the server."

# Wait for Ctrl+C
trap "kill $SERVER_PID 2>/dev/null; echo 'Server stopped.'; exit 0" INT TERM
wait $SERVER_PID
