"""VR_RequestBanner.log_file 必须把后续 vr_log 重绑到独立文件，
不传则维持默认 vr_debug.log（不破坏 layered 行为）。"""
import importlib

import pytest

dp = importlib.import_module("comfyui_vector_ready.nodes.debug_probe")


@pytest.fixture
def isolate_plugin_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(dp, "_PLUGIN_DIR", tmp_path, raising=True)
    monkeypatch.setattr(dp, "_DEFAULT_LOG_PATH", tmp_path / "vr_debug.log", raising=True)
    dp.set_log_path("")  # 复位到默认
    yield tmp_path
    dp.set_log_path("")  # 收尾复位


def test_named_log_file_routes_next_to_plugin(isolate_plugin_dir):
    dp.set_log_path("vr_ip_consistent.log")
    dp.vr_log("T", "hello-ipc")
    target = isolate_plugin_dir / "vr_ip_consistent.log"
    assert target.exists()
    assert "hello-ipc" in target.read_text()
    assert not (isolate_plugin_dir / "vr_debug.log").exists()


def test_empty_resets_to_default(isolate_plugin_dir):
    dp.set_log_path("vr_ip_consistent.log")
    dp.set_log_path("")
    dp.vr_log("T", "back-to-default")
    assert "back-to-default" in (isolate_plugin_dir / "vr_debug.log").read_text()


class _ImgStub:
    """torch-free stand-in: banner() only reads image.shape for the log line."""
    shape = (1, 4, 4, 3)


def test_banner_sets_log_file_before_first_line(isolate_plugin_dir):
    banner = dp.VR_RequestBanner()
    banner.banner(_ImgStub(), tag="ip_consistent", log_file="vr_ip_consistent.log")
    text = (isolate_plugin_dir / "vr_ip_consistent.log").read_text()
    assert "REQUEST START" in text and "ip_consistent" in text
