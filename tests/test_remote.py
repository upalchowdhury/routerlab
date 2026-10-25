"""Remote checkpoints over HTTP range requests, tested against local servers (no network)."""

import http.server
import json
import socketserver
import threading

import numpy as np
import pytest

from routerlab import Checkpoint, quantize, save_safetensors
from routerlab.remote import RemoteFile, is_remote, parse_hf_uri


def test_parse_hf_uri():
    assert parse_hf_uri("hf://deepseek-ai/DeepSeek-V3") == ("deepseek-ai/DeepSeek-V3", "main")
    assert parse_hf_uri("hf://org/name@v1.0") == ("org/name", "v1.0")
    assert parse_hf_uri("https://huggingface.co/org/name/tree/abc123") == ("org/name", "abc123")
    assert is_remote("hf://a/b") and not is_remote("/local/path")
    with pytest.raises(ValueError):
        parse_hf_uri("hf://just-a-name")


class _FastServer(http.server.ThreadingHTTPServer):
    def server_bind(self):  # skip HTTPServer's reverse-DNS getfqdn(), which can take ~20 s
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]


class _Server:
    """Serves files from a dict with Range support; records the headers it saw."""

    def __init__(self, files, redirect_to=None, fail_first=0):
        self.files, self.redirect_to, self.seen, self.fail = files, redirect_to, [], fail_first
        server = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                server.seen.append({**dict(self.headers), "path": self.path})
                if server.fail > 0:
                    server.fail -= 1
                    self.send_error(503)
                    return
                if server.redirect_to:
                    self.send_response(302)
                    self.send_header("Location", server.redirect_to() + self.path)
                    self.end_headers()
                    return
                path = self.path.split("/resolve/main/")[-1]
                if path == "gated":
                    self.send_error(403)
                    return
                if path not in server.files:
                    self.send_error(404)
                    return
                data = server.files[path]
                status, body = 200, data
                if "Range" in self.headers:
                    start, end = self.headers["Range"].removeprefix("bytes=").split("-")
                    body = data[int(start):int(end) + 1]
                    status = 206
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.httpd = _FastServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def close(self):
        self.httpd.shutdown()


@pytest.fixture
def checkpoint_files(tmp_path):
    rng = np.random.default_rng(0)
    weight = rng.normal(size=(8, 16)).astype(np.float32)
    save_safetensors(tmp_path / "model.safetensors", {
        "model.layers.0.mlp.gate.weight": weight,
        "model.layers.0.mlp.gate.e_score_correction_bias": rng.normal(size=8).astype(np.float32)},
        bf16=["model.layers.0.mlp.gate.weight"])
    config = {"model_type": "deepseek_v3", "n_routed_experts": 8, "num_experts_per_tok": 2, "n_group": 1,
              "topk_group": 1, "hidden_size": 16}
    files = {"model.safetensors": (tmp_path / "model.safetensors").read_bytes(),
             "config.json": json.dumps(config).encode()}
    return files, weight


def test_remote_checkpoint_matches_local(checkpoint_files, monkeypatch):
    files, weight = checkpoint_files
    cdn = _Server(files)
    hub = _Server(files, redirect_to=lambda: cdn.url)  # the hub redirects every file to the "CDN"
    monkeypatch.setenv("HF_ENDPOINT", hub.url)
    monkeypatch.setenv("HF_TOKEN", "secret-token")
    try:
        ckpt = Checkpoint("hf://org/model")
        (router,) = ckpt.find_routers()
        loaded, bias, _ = ckpt.load_router(router)
        np.testing.assert_array_equal(loaded, quantize(weight, "bf16"))
        assert bias.shape == (8,) and ckpt.router_spec().use_correction_bias
        assert all(h.get("Authorization") == "Bearer secret-token" for h in hub.seen)
        assert cdn.seen and not any("Authorization" in h for h in cdn.seen)  # never leaked cross-host
        weights = [h for h in cdn.seen if h["path"].endswith("model.safetensors")]
        assert weights and all("Range" in h for h in weights)  # safetensors read by range, not whole file
    finally:
        cdn.close()
        hub.close()


def test_remote_errors_and_retries(checkpoint_files):
    files, _ = checkpoint_files
    server = _Server(files, fail_first=2)  # two transient 503s, then success
    try:
        data = RemoteFile(f"{server.url}/resolve/main/config.json", retries=3).read_all()
        assert json.loads(data)["model_type"] == "deepseek_v3"
        with pytest.raises(FileNotFoundError):
            RemoteFile(f"{server.url}/resolve/main/missing.json", retries=0).read_all()
        with pytest.raises(PermissionError, match="HF_TOKEN"):
            RemoteFile(f"{server.url}/resolve/main/gated", retries=0).read_all()
        chunk = RemoteFile(f"{server.url}/resolve/main/model.safetensors").read(8, 16)
        assert chunk == files["model.safetensors"][8:24]
    finally:
        server.close()
