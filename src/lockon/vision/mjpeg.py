"""Minimal MJPEG-over-HTTP streamer so a headless Pi can show its live
camera feed (and overlays) in a browser on a laptop. Shared by
tools/web_preview.py and tools/bench_integration_runner.py so there is one
camera owner per run - the browser just views whatever JPEGs that one
process publishes.

Endpoints:  /  (viewer page)   /stream.mjpg  (MJPEG)   /snapshot.jpg
"""

from __future__ import annotations

import socket
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FrameHub:
    """Holds the most recent JPEG; stream readers wait for the next one."""

    def __init__(self) -> None:
        self._jpeg = b""
        self._seq = 0
        self._cv = threading.Condition()
        self.stopped = False

    def publish(self, jpeg: bytes) -> None:
        with self._cv:
            self._jpeg = jpeg
            self._seq += 1
            self._cv.notify_all()

    def latest(self) -> bytes:
        with self._cv:
            return self._jpeg

    def wait_next(self, last_seq: int, timeout: float = 5.0):
        with self._cv:
            self._cv.wait_for(lambda: self._seq != last_seq or self.stopped, timeout=timeout)
            return self._jpeg, self._seq

    def stop(self) -> None:
        with self._cv:
            self.stopped = True
            self._cv.notify_all()


PAGE = b"""<!doctype html><html><head><title>lock-on camera</title>
<style>body{margin:0;background:#111;color:#ddd;font-family:sans-serif;text-align:center}
img{max-width:100vw;max-height:95vh}</style></head>
<body><img src="/stream.mjpg"></body></html>"""


def make_handler(hub: FrameHub):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # quiet
            pass

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(PAGE)))
                self.end_headers()
                self.wfile.write(PAGE)
            elif self.path == "/snapshot.jpg":
                jpeg = hub.latest()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                self.end_headers()
                self.wfile.write(jpeg)
            elif self.path == "/stream.mjpg":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                seq = -1
                try:
                    while not hub.stopped:
                        jpeg, seq = hub.wait_next(seq)
                        if not jpeg:
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                         + f"Content-Length: {len(jpeg)}\r\n\r\n".encode() + jpeg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_error(404)

    return Handler


def local_ip() -> str:
    try:
        out = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=2).stdout.split()
        if out:
            return out[0]
    except Exception:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "<pi-ip>"


def serve(hub: FrameHub, http_port: int) -> ThreadingHTTPServer:
    """Start the HTTP server in a background thread and return it. Prints
    the URL to open on the laptop."""
    server = ThreadingHTTPServer(("0.0.0.0", http_port), make_handler(hub))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"\n  Live view on your laptop browser:  http://{local_ip()}:{http_port}/\n")
    return server
