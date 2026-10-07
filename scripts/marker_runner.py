"""
Marker persistent worker — chạy trong marker_venv riêng biệt.
Giao tiếp qua stdin/stdout với protocol:

  stdin  → "PROCESS <image_path>\n"  hoặc  "QUIT\n"
  stdout ← "LOADING <msg>\n"         (khi đang download/load models)
           "READY\n"                 (sau khi load xong models)
           "RESULT\n<json>\nEND\n"   (sau mỗi ảnh)
           "ERROR\n<message>\nEND\n" (nếu lỗi)

Chạy: marker_venv\Scripts\python.exe scripts\marker_runner.py
"""
import json
import os
import sys

# Ưu tiên GPU để giảm áp lực RAM — nếu không có GPU thì dùng CPU
os.environ.setdefault("TORCH_DEVICE", "cpu")

# Giảm batch size tối đa để tiết kiệm RAM khi chạy CPU
os.environ.setdefault("RECOGNITION_BATCH_SIZE", "4")
os.environ.setdefault("LAYOUT_BATCH_SIZE", "1")
os.environ.setdefault("DETECTION_BATCH_SIZE", "1")
os.environ.setdefault("ORDER_BATCH_SIZE", "1")
os.environ.setdefault("MARKER_WORKERS", "1")


def _status(msg: str) -> None:
    print(f"LOADING {msg}", flush=True)


_status("importing marker...")

try:
    _status("loading layout model...")
    from marker.converters.pdf import PdfConverter
    _status("loading OCR + detection models...")
    from marker.models import create_model_dict
    from marker.output import text_from_rendered
except ImportError as e:
    print(f"ERROR\n{e}\nEND", flush=True)
    sys.exit(1)

_status("initialising converter (may take 1-3 min first run)...")
try:
    converter = PdfConverter(artifact_dict=create_model_dict())
except Exception as e:
    print(f"ERROR\n{e}\nEND", flush=True)
    sys.exit(1)

print("READY", flush=True)

for raw_line in sys.stdin:
    cmd = raw_line.strip()

    if cmd == "QUIT":
        break

    if cmd.startswith("PROCESS "):
        image_path = cmd[8:].strip()
        try:
            rendered = converter(image_path)
            text, _, _ = text_from_rendered(rendered)
            payload = json.dumps({"ok": True, "text": text.strip()})
            print(f"RESULT\n{payload}\nEND", flush=True)
        except Exception as e:
            payload = json.dumps({"ok": False, "error": str(e), "text": ""})
            print(f"RESULT\n{payload}\nEND", flush=True)
