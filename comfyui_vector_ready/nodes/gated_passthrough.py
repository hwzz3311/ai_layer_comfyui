"""Conditionally pass a value through, or emit ComfyUI's ExecutionBlocker.

When `enable` is True the input is forwarded unchanged. When False, an
ExecutionBlocker(None) is emitted; ComfyUI's execution engine then skips
every downstream node that receives it (including KSamplers, VAEDecode,
SaveImage). This is the canonical way to short-circuit one branch of an
A/B workflow without touching mute state."""

from __future__ import annotations

try:
    # Available in ComfyUI 2024+ — the official execution-graph hook.
    from comfy_execution.graph import ExecutionBlocker  # type: ignore
except ImportError:  # pragma: no cover - allows unit tests outside ComfyUI
    class ExecutionBlocker:  # noqa: D401
        def __init__(self, msg=None):
            self.msg = msg


_ANY = ("*",)  # ComfyUI accepts the wildcard tuple as "any socket type"


class _AnyType(str):
    """str subclass that equals any other type string — used for wildcard sockets."""

    def __ne__(self, other):
        return False


ANY_TYPE = _AnyType("*")


class VR_GatedPassthrough:
    """Forward `value` if `enable` is True, else block downstream execution."""

    CATEGORY = "VectorReady/flow"
    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("value",)
    FUNCTION = "gate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": (ANY_TYPE,),
                "enable": ("BOOLEAN", {"default": True}),
                "invert": ("BOOLEAN", {"default": False}),
            }
        }

    def gate(self, value, enable, invert):
        active = (not enable) if invert else enable
        if active:
            return (value,)
        return (ExecutionBlocker(None),)
