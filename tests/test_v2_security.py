"""全方位测试方案 v2 · F 组：安全绕过矩阵（对 F6/F8 做攻击面测试，非功能重测）。

对照方案矩阵：伪 Origin / Origin null / Referer 伪造 / Referer about:blank /
Host 伪造（DNS rebinding）/ IPv6 本机放行 / Content-Type 分号解析放行 /
空 Origin 放行 / F8 路径绕过（B 组 TestOutsideWorkspaceMatrix 已覆盖）。
"""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

from conftest import FakeLLM

from vgent.config import Config
from vgent.store import SessionStore
from vgent.web.server import HubManager, make_server


def _server(tmp_path):
    cfg = Config(data_dir=tmp_path)
    store = SessionStore(tmp_path / "t.db")
    manager = HubManager(cfg, store, FakeLLM())
    httpd = make_server(manager, port=0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}", store


def _req(url: str, method: str = "POST", headers: dict | None = None, body: bytes | None = None):
    req = urllib.request.Request(url, data=body, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8") or "{}")


# -- 拒绝面 -----------------------------------------------------------------


def test_attack_fake_origin_post_403(tmp_path) -> None:
    """伪 Origin（https://evil.com）POST → 403。"""
    httpd, base, store = _server(tmp_path)
    try:
        code, _ = _req(
            base + "/api/sessions",
            headers={"Origin": "https://evil.com", "Content-Type": "application/json"},
            body=b"{}",
        )
        assert code == 403
    finally:
        httpd.shutdown()
        httpd.server_close()
        store.close()


def test_attack_origin_null_403(tmp_path) -> None:
    """Origin: null（sandboxed iframe）→ 403。"""
    httpd, base, store = _server(tmp_path)
    try:
        code, _ = _req(
            base + "/api/sessions",
            headers={"Origin": "null", "Content-Type": "application/json"},
            body=b"{}",
        )
        assert code == 403
    finally:
        httpd.shutdown()
        httpd.server_close()
        store.close()


def test_attack_referer_spoof_without_origin_403(tmp_path) -> None:
    """Referer 伪造（无 Origin 头，Referer 指向 evil.com）→ 403。"""
    httpd, base, store = _server(tmp_path)
    try:
        code, _ = _req(
            base + "/api/sessions",
            headers={"Referer": "https://evil.com/x", "Content-Type": "application/json"},
            body=b"{}",
        )
        assert code == 403
    finally:
        httpd.shutdown()
        httpd.server_close()
        store.close()


def test_attack_referer_about_blank_403(tmp_path) -> None:
    """Referer: about:blank（hostname 解析为空）→ 403（复审跟进语义）。"""
    httpd, base, store = _server(tmp_path)
    try:
        code, _ = _req(
            base + "/api/sessions",
            headers={"Referer": "about:blank", "Content-Type": "application/json"},
            body=b"{}",
        )
        assert code == 403
    finally:
        httpd.shutdown()
        httpd.server_close()
        store.close()


def test_attack_host_spoof_get_and_post_403(tmp_path) -> None:
    """Host 伪造（DNS rebinding）GET/POST → 403。"""
    httpd, base, store = _server(tmp_path)
    try:
        code, _ = _req(base + "/api/sessions", method="GET", headers={"Host": "evil.com"})
        assert code == 403
        code, _ = _req(
            base + "/api/sessions",
            method="POST",
            headers={"Host": "evil.com", "Content-Type": "application/json"},
            body=b"{}",
        )
        assert code == 403
    finally:
        httpd.shutdown()
        httpd.server_close()
        store.close()


# -- 放行面（勿误杀）---------------------------------------------------------


def test_ipv6_loopback_host_allowed(tmp_path) -> None:
    """IPv6 本机 Host: [::1]:<port> → 放行（不误杀 IPv6 loopback）。"""
    httpd, base, store = _server(tmp_path)
    port = httpd.server_address[1]
    try:
        code, body = _req(base + "/api/sessions", method="GET", headers={"Host": f"[::1]:{port}"})
        assert code == 200 and "sessions" in body
    finally:
        httpd.shutdown()
        httpd.server_close()
        store.close()


def test_ipv6_loopback_origin_allowed(tmp_path) -> None:
    """Origin: http://[::1]:<port> POST → 放行。"""
    httpd, base, store = _server(tmp_path)
    port = httpd.server_address[1]
    try:
        code, _ = _req(
            base + "/api/sessions",
            headers={"Origin": f"http://[::1]:{port}", "Content-Type": "application/json"},
            body=b"{}",
        )
        assert code == 200
    finally:
        httpd.shutdown()
        httpd.server_close()
        store.close()


def test_content_type_with_charset_allowed(tmp_path) -> None:
    """Content-Type: application/json; charset=utf-8 → 放行（分号解析）。"""
    httpd, base, store = _server(tmp_path)
    try:
        code, _ = _req(
            base + "/api/sessions",
            headers={"Content-Type": "application/json; charset=utf-8"},
            body=b"{}",
        )
        assert code == 200
    finally:
        httpd.shutdown()
        httpd.server_close()
        store.close()


def test_empty_origin_header_allowed(tmp_path) -> None:
    """空 Origin 头（值空串）→ 等同无头，非浏览器客户端放行。"""
    httpd, base, store = _server(tmp_path)
    try:
        code, _ = _req(
            base + "/api/sessions",
            headers={"Origin": "", "Content-Type": "application/json"},
            body=b"{}",
        )
        assert code == 200
    finally:
        httpd.shutdown()
        httpd.server_close()
        store.close()


def test_text_plain_still_rejected_with_local_origin(tmp_path) -> None:
    """本机 Origin + text/plain 体 → 仍 400（Content-Type 闸与 Origin 闸独立）。"""
    httpd, base, store = _server(tmp_path)
    port = httpd.server_address[1]
    try:
        code, _ = _req(
            base + "/api/sessions",
            headers={"Origin": f"http://127.0.0.1:{port}", "Content-Type": "text/plain"},
            body=b"{}",
        )
        assert code == 400
    finally:
        httpd.shutdown()
        httpd.server_close()
        store.close()
