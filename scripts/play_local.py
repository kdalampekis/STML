"""Play Wugs and Daxes on your own computer, with every move saved to logs/.

Usage:
    python scripts/play_local.py            # then open http://127.0.0.1:8000
    python scripts/play_local.py --port 8080

The game notices the server and saves as you play: one JSON file per world,
in logs/<participant code>/<session>_<world>.json, plus the session start.
Those files are what scripts/analyze_log.py and the training code read.
No internet connection is needed; nothing leaves your computer.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GAME = ROOT / "game" / "index.html"
LOGS = ROOT / "logs"
SAFE = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")
MAX_BODY = 5 * 1024 * 1024

PAGE_HEAD = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
             '<meta name="viewport" content="width=device-width,initial-scale=1"></head><body>')


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/index.html"):
            if not GAME.exists():
                self._send(500, b"Run python scripts/build_game.py first.", "text/plain")
                return
            html = PAGE_HEAD + GAME.read_text(encoding="utf-8") + "</body></html>"
            self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path == "/api/ping":
            self._send(200, json.dumps({"stml": True, "logs": str(LOGS)}).encode(), "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/log":
            self._send(404, b"not found", "text/plain")
            return
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0 or length > MAX_BODY:
            self._send(413, b"bad size", "text/plain")
            return
        try:
            body = json.loads(self.rfile.read(length))
            participant, doc_id, doc = body["participant"], body["doc_id"], body["doc"]
            if not (SAFE.match(participant) and SAFE.match(doc_id)) or doc.get("schema") != "stml-log/1":
                raise ValueError("bad fields")
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            self._send(400, b"bad request", "text/plain")
            return
        folder = LOGS / participant
        folder.mkdir(parents=True, exist_ok=True)
        tmp = folder / f".{doc_id}.tmp"
        tmp.write_text(json.dumps(doc, indent=1), encoding="utf-8")
        tmp.replace(folder / f"{doc_id}.json")  # atomic: a half-written file is never read
        self._send(200, b'{"ok":true}', "application/json")

    def log_message(self, fmt: str, *args) -> None:  # quieter console
        if self.path.startswith("/api/log"):
            return
        sys.stderr.write("%s\n" % (fmt % args))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Wugs and Daxes: {url}\nLogs are saved to {LOGS}\nPress Ctrl+C to stop.")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped. Your logs are in", LOGS)


if __name__ == "__main__":
    main()
