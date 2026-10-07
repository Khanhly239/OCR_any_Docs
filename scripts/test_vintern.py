"""
Test Vintern-1B-v2 OCR trực tiếp trên một ảnh.
Model: 5CD-AI/Vintern-1B-v2 (InternVL2 base, ~1B params, ~2GB VRAM)
Chạy: python scripts/test_vintern.py "path/to/image.jpg"
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

print("[test_vintern] Script bắt đầu chạy...", flush=True)


def load_image(image_path: str, max_num: int = 12):
    """Tải và tile ảnh theo chuẩn InternVL (dynamic resolution)."""
    import torch
    import torchvision.transforms as T
    from PIL import Image
    from torchvision.transforms.functional import InterpolationMode

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)
    IMG_SIZE = 448

    def build_transform():
        return T.Compose([
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def find_closest_aspect_ratio(ratio, target_ratios, width, height, img_size):
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio_candidate in target_ratios:
            target_ratio = ratio_candidate[0] / ratio_candidate[1]
            ratio_diff = abs(ratio - target_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio_candidate
            elif ratio_diff == best_ratio_diff and area > 0.5 * img_size ** 2 * ratio_candidate[0] * ratio_candidate[1]:
                best_ratio = ratio_candidate
        return best_ratio

    def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
        target_ratios = {(i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if min_num <= i * j <= max_num}
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
        resized = image.resize((target_width, target_height))
        processed = []
        for i in range(blocks):
            box = ((i % target_aspect_ratio[0]) * image_size,
                   (i // target_aspect_ratio[0]) * image_size,
                   ((i % target_aspect_ratio[0]) + 1) * image_size,
                   ((i // target_aspect_ratio[0]) + 1) * image_size)
            processed.append(resized.crop(box))
        if use_thumbnail and blocks != 1:
            processed.append(image.resize((image_size, image_size)))
        return processed

    image = Image.open(image_path).convert("RGB")
    transform = build_transform()
    images = dynamic_preprocess(image, image_size=IMG_SIZE, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    return torch.stack(pixel_values)


def main():
    import torch
    from transformers import AutoModel, AutoTokenizer
    from loguru import logger

    image_path = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\Adm\Downloads\hd.jpg"

    if not Path(image_path).exists():
        logger.error(f"File không tồn tại: {image_path}")
        sys.exit(1)

    model_id = "5CD-AI/Vintern-1B-v2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device} | Model: {model_id}")
    logger.info(f"Ảnh: {image_path}")

    # Resolve local cache path — offline nếu đã có, download nếu chưa có
    from huggingface_hub import snapshot_download
    try:
        local_model_dir = snapshot_download(model_id, local_files_only=True)
        logger.info(f"Model dir (cache): {local_model_dir}")
    except Exception:
        logger.info("Chưa có cache, đang download...")
        local_model_dir = snapshot_download(model_id)
        logger.info(f"Model dir (downloaded): {local_model_dir}")

    # Load model
    logger.info("Đang load model (~2GB)...")
    t0 = time.perf_counter()
    try:
        model = AutoModel.from_pretrained(
            local_model_dir,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            local_files_only=True,
        ).eval().to(device)
        tokenizer = AutoTokenizer.from_pretrained(local_model_dir, trust_remote_code=True, local_files_only=True)
    except Exception as e:
        import traceback
        logger.error(f"Load model thất bại: {e}")
        traceback.print_exc()
        sys.exit(1)

    load_time = time.perf_counter() - t0
    logger.success(f"Model loaded trong {load_time:.1f}s")

    logger.info("Đang xử lý ảnh...")
    pixel_values = load_image(image_path, max_num=12).to(device, dtype=torch.bfloat16)

    generation_config = dict(
        max_new_tokens=2048,
        do_sample=False,
    )

    # Prompt OCR
    question = (
        "<image>\n"
        "Hãy đọc và trích xuất toàn bộ văn bản trong ảnh này. "
        "Giữ nguyên cấu trúc, định dạng và dấu tiếng Việt đầy đủ. "
        "Nếu có bảng, hãy trình bày dạng bảng markdown."
    )

    logger.info("Đang chạy OCR inference...")
    t1 = time.perf_counter()
    response = model.chat(tokenizer, pixel_values, question, generation_config)
    infer_time = time.perf_counter() - t1

    logger.success(f"Inference hoàn tất trong {infer_time:.1f}s")

    vram_info = ""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**2)
        reserved  = torch.cuda.memory_reserved() / (1024**2)
        vram_info = f"{allocated:.0f}MB allocated / {reserved:.0f}MB reserved"
        logger.info(f"VRAM: {vram_info}")

    # Lưu HTML và mở browser
    _save_html_and_open(image_path, response, load_time, infer_time, vram_info)


def _save_html_and_open(image_path: str, ocr_text: str, load_s: float, infer_s: float, vram: str) -> None:
    import base64, webbrowser, tempfile, re, html as _html
    from pathlib import Path as _P

    # Chuyển markdown đơn giản → HTML
    def md_to_html(text: str) -> str:
        lines, out = text.splitlines(), []
        for ln in lines:
            ln_e = _html.escape(ln)
            if ln_e.startswith("## "):
                out.append(f"<h3>{ln_e[3:]}</h3>")
            elif ln_e.startswith("# "):
                out.append(f"<h2>{ln_e[2:]}</h2>")
            elif re.match(r"^\|.+\|$", ln_e):
                cells = [c.strip() for c in ln_e.strip("|").split("|")]
                row = "".join(f"<td>{c}</td>" for c in cells)
                out.append(f"<tr>{row}</tr>")
            elif re.match(r"^\|[-| :]+\|$", ln_e):
                pass  # bỏ dòng phân cách bảng markdown
            elif ln_e.strip() == "":
                out.append("<br>")
            else:
                out.append(f"<p>{ln_e}</p>")
        # bọc các <tr> vào <table>
        html_str = "\n".join(out)
        html_str = re.sub(r"(<tr>.*?</tr>\n?)+", lambda m: f"<table>{m.group()}</table>", html_str, flags=re.S)
        return html_str

    # Encode ảnh gốc để nhúng vào HTML
    img_b64 = ""
    try:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        ext = _P(image_path).suffix.lstrip(".").lower() or "jpeg"
        img_tag = f'<img src="data:image/{ext};base64,{img_b64}" style="max-width:100%;border-radius:6px;">'
    except Exception:
        img_tag = f"<p><i>{_html.escape(image_path)}</i></p>"

    ocr_html = md_to_html(ocr_text)

    page = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8">
<title>Vintern OCR — {_html.escape(_P(image_path).name)}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; padding: 24px; }}
  h1 {{ font-size: 1.3rem; color: #818cf8; margin-bottom: 16px; }}
  .meta {{ font-size: .8rem; color: #64748b; margin-bottom: 20px; }}
  .meta span {{ margin-right: 16px; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  .card {{ background: #1e293b; border-radius: 10px; padding: 16px; }}
  .card h2 {{ font-size: .85rem; color: #94a3b8; text-transform: uppercase; letter-spacing: .05em; margin-bottom: 12px; }}
  .ocr-output p {{ margin: 4px 0; line-height: 1.6; font-size: .95rem; }}
  .ocr-output h2 {{ color: #818cf8; margin: 12px 0 4px; font-size: 1.1rem; }}
  .ocr-output h3 {{ color: #a5b4fc; margin: 10px 0 4px; font-size: 1rem; }}
  table {{ border-collapse: collapse; width: 100%; margin: 8px 0; font-size: .85rem; }}
  td {{ border: 1px solid #334155; padding: 6px 10px; }}
  tr:nth-child(even) td {{ background: #0f172a55; }}
  @media (max-width: 700px) {{ .grid {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<h1>🇻🇳 Vintern-1B OCR — {_html.escape(_P(image_path).name)}</h1>
<div class="meta">
  <span>⏱ Load: {load_s:.1f}s</span>
  <span>⚡ Inference: {infer_s:.1f}s</span>
  {"<span>🎮 VRAM: " + _html.escape(vram) + "</span>" if vram else ""}
</div>
<div class="grid">
  <div class="card"><h2>Ảnh gốc</h2>{img_tag}</div>
  <div class="card"><h2>Kết quả OCR</h2><div class="ocr-output">{ocr_html}</div></div>
</div>
</body></html>"""

    out_path = _P(tempfile.gettempdir()) / f"vintern_ocr_{_P(image_path).stem}.html"
    out_path.write_text(page, encoding="utf-8")
    print(f"\n[test_vintern] Kết quả lưu tại: {out_path}", flush=True)
    webbrowser.open(out_path.as_uri())


if __name__ == "__main__":
    main()
