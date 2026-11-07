"""Harness package built for openJiuwen integration."""
try:
    import openjiuwen  # noqa: F401
    OPENJIUWEN_AVAILABLE = True
except ImportError:
    openjiuwen = None
    OPENJIUWEN_AVAILABLE = False
