"""
Chandra OCR 2 — Web Demo + REST API
  Gradio UI  : http://localhost:7860/
  API docs   : http://localhost:7860/docs
  OCR endpoint: POST /api/ocr  (multipart: file=<image|pdf>)

Colab : !python demo_chandra.py   → share link tự động
Local : python demo_chandra.py
Docker: CMD trong Dockerfile
"""
import io, time, base64, re
from io import BytesIO

import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import gradio as gr


# ── Env detection ────────────────────────────────────────────────────────────
def _in_colab():
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


# ── Load model ───────────────────────────────────────────────────────────────
def _build_manager():
    free_mb = torch.cuda.mem_get_info()[0] // (1024 ** 2) if torch.cuda.is_available() else 0
    use_4bit = free_mb < 10_000
    print(f"VRAM: {free_mb} MB → {'4-bit NF4' if use_4bit else 'fp16'}")

    from chandra.model import InferenceManager

    if not use_4bit:
        return InferenceManager(method='hf')

    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    import chandra.model as _cm
    from chandra.model import settings as _cs

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        llm_int8_enable_fp32_cpu_offload=True,
    )

    try:
        from bitsandbytes.nn.modules import Params4bit
        _orig = Params4bit.__new__
        def _patch(cls, *a, **kw):
            kw.pop('_is_hf_initialized', None)
            return _orig(cls, *a, **kw)
        Params4bit.__new__ = staticmethod(_patch)
    except Exception:
        pass

    def _load_4bit():
        model = AutoModelForImageTextToText.from_pretrained(
            _cs.MODEL_CHECKPOINT,
            quantization_config=bnb,
            device_map='auto',
            max_memory={0: f'{free_mb - 500}MiB', 'cpu': '16GiB'},
            trust_remote_code=True,
        ).eval()
        proc = AutoProcessor.from_pretrained(_cs.MODEL_CHECKPOINT)
        proc.tokenizer.padding_side = 'left'
        model.processor = proc
        return model

    orig = _cm.load_model
    _cm.load_model = _load_4bit
    mgr = InferenceManager(method='hf')
    _cm.load_model = orig
    return mgr


print("Đang load Chandra OCR 2...")
_t0 = time.perf_counter()
_manager = _build_manager()
print(f"✅ Sẵn sàng ({time.perf_counter() - _t0:.1f}s)")


# ── OCR core ─────────────────────────────────────────────────────────────────
def _embed_images(md, imgs):
    if not imgs:
        return md
    def _sub(m):
        img = imgs.get(m.group(2))
        if img is None:
            return m.group(0)
        buf = BytesIO()
        img.save(buf, format='PNG')
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f'![{m.group(1)}](data:image/png;base64,{b64})'
    return re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', _sub, md)


def _run_ocr(pil_img: Image.Image):
    from chandra.model.schema import BatchInputItem
    t0 = time.perf_counter()
    results = _manager.generate([BatchInputItem(image=pil_img, prompt_type='ocr_layout')])
    elapsed = time.perf_counter() - t0
    r = results[0]
    md = getattr(r, 'markdown', '') or ''
    md = _embed_images(md, getattr(r, 'images', {}) or {})
    return md, elapsed


def _pdf_to_pages(data: bytes) -> list:
    import fitz
    doc = fitz.open(stream=data, filetype="pdf")
    pages = []
    for page in doc:
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        pages.append(Image.frombytes("RGB", [pix.w, pix.h], pix.samples))
    doc.close()
    return pages


# ── NER core ─────────────────────────────────────────────────────────────────
_NER_LABELS = {
    'PER':  ('#dbeafe', '#1d4ed8', 'Người'),
    'ORG':  ('#dcfce7', '#15803d', 'Tổ chức'),
    'LOC':  ('#ffedd5', '#c2410c', 'Địa điểm'),
    'TIME': ('#f3e8ff', '#7e22ce', 'Thời gian'),
    'MISC': ('#f1f5f9', '#475569', 'Khác'),
}


def _extract_entities(raw) -> list[dict]:
    entities, buf, tag = [], [], None
    for item in raw:
        word, ner_tag = item[0], item[-1]
        if ner_tag.startswith('B-'):
            if buf:
                entities.append({"text": ' '.join(buf), "type": tag})
            buf, tag = [word], ner_tag[2:]
        elif ner_tag.startswith('I-') and tag == ner_tag[2:]:
            buf.append(word)
        else:
            if buf:
                entities.append({"text": ' '.join(buf), "type": tag})
            buf, tag = [], None
    if buf:
        entities.append({"text": ' '.join(buf), "type": tag})
    return entities


def _run_ner(plain_text: str) -> tuple[list[dict], str]:
    """Returns (entities_list, html_str)."""
    try:
        from underthesea import ner
        raw = ner(plain_text)
        entities = _extract_entities(raw)
        return entities, _ner_html(entities)
    except ImportError:
        return [], (
            '<p style="color:#999;font-size:13px">'
            'Cài underthesea để dùng NER: <code>pip install underthesea</code></p>'
        )
    except Exception as e:
        return [], f'<p style="color:#c00;font-size:13px">Lỗi NER: {e}</p>'


def _ner_html(entities: list[dict]) -> str:
    if not entities:
        return '<p style="color:#aaa;font-style:italic;margin:0">Không tìm thấy thực thể nào.</p>'
    by_type: dict = {}
    for e in entities:
        if e["type"] in _NER_LABELS:
            by_type.setdefault(e["type"], []).append(e["text"])
    if not by_type:
        return '<p style="color:#aaa;font-style:italic;margin:0">Không tìm thấy thực thể nào.</p>'
    parts = []
    for etype in ('PER', 'ORG', 'LOC', 'TIME', 'MISC'):
        words = by_type.get(etype)
        if not words:
            continue
        bg, color, label = _NER_LABELS[etype]
        chips = ''.join(
            f'<span style="display:inline-block;background:{bg};color:{color};'
            f'border-radius:20px;padding:3px 12px;margin:3px 4px 3px 0;'
            f'font-size:13px;font-weight:500">{w}</span>'
            for w in dict.fromkeys(words)
        )
        badge = (
            f'<span style="display:inline-block;background:{color};color:#fff;'
            f'border-radius:4px;padding:1px 7px;font-size:11px;font-weight:600;'
            f'margin-right:8px;vertical-align:middle">{label}</span>'
        )
        parts.append(
            f'<div style="margin-bottom:10px">'
            f'<div style="margin-bottom:5px">{badge}</div>{chips}</div>'
        )
    return ''.join(parts)


def _md_to_plain(md: str) -> str:
    text = re.sub(r'!\[[^\]]*\]\([^)]*\)', '', md)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'[#*`_~>|\\]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


# ── FastAPI ───────────────────────────────────────────────────────────────────
_api = FastAPI(
    title="Chandra OCR 2 API",
    version="1.0.0",
    description="OCR chữ viết tay và tài liệu tiếng Việt — Chandra OCR 2 (Qwen3.5-VL)",
)


@_api.post(
    "/api/ocr",
    summary="Nhận dạng văn bản từ ảnh hoặc PDF",
    response_description="Markdown text + NER entities",
)
async def api_ocr(file: UploadFile = File(..., description="Ảnh (jpg/png/webp) hoặc PDF")):
    data = await file.read()
    ext = (file.filename or "").rsplit(".", 1)[-1].lower()

    # Parse file → danh sách PIL pages
    if ext == "pdf":
        try:
            pages = _pdf_to_pages(data)
        except ImportError:
            return JSONResponse({"error": "Cài pymupdf: pip install pymupdf"}, status_code=422)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=422)
    else:
        try:
            pages = [Image.open(io.BytesIO(data)).convert("RGB")]
        except Exception as e:
            return JSONResponse({"error": f"Không đọc được ảnh: {e}"}, status_code=422)

    # OCR + NER từng trang
    page_results = []
    for i, pil in enumerate(pages):
        md, elapsed = _run_ocr(pil)
        plain = _md_to_plain(md)
        entities, _ = _run_ner(plain)
        page_results.append({
            "page": i + 1,
            "markdown": md,
            "plain_text": plain,
            "elapsed_s": round(elapsed, 2),
            "entities": entities,
        })

    return {
        "filename": file.filename,
        "total_pages": len(page_results),
        "pages": page_results,
    }


# ── Gradio UI ─────────────────────────────────────────────────────────────────
CSS = """
footer { display: none !important; }
body, .gradio-container { background: #f0efeb !important; }
#result-box {
    background: #fff;
    border: 1px solid #e0dfd9 !important;
    border-radius: 10px !important;
    min-height: 320px;
    max-height: 380px;
    padding: 20px !important;
    font-size: 15px;
    line-height: 1.8;
    color: #1c1c1c;
    overflow-y: auto;
}
#result-box p  { margin-bottom: .75em; }
#result-box h1, #result-box h2, #result-box h3 { margin: 1em 0 .4em; font-weight: 600; }
#result-box table { border-collapse: collapse; width: 100%; font-size: 14px; }
#result-box th, #result-box td { border: 1px solid #e0dfd9; padding: 6px 10px; }
#ner-box {
    background: #fff;
    border: 1px solid #e0dfd9 !important;
    border-radius: 10px !important;
    min-height: 80px;
    padding: 16px !important;
}
#ner-label {
    font-size: 11px !important; font-weight: 600 !important;
    letter-spacing: .5px !important; color: #999 !important;
    text-transform: uppercase; margin-bottom: 6px;
}
#run-btn {
    background: #1c1c1c !important; border: none !important;
    border-radius: 7px !important; color: #fff !important;
    font-size: 15px !important; font-weight: 500 !important;
    height: 46px !important;
}
#run-btn:hover { background: #3a3a3a !important; }
#timing textarea {
    border: none !important; background: transparent !important;
    box-shadow: none !important; padding: 0 !important;
    font-size: 12px !important; font-family: ui-monospace,monospace !important;
    color: #999 !important; text-align: right !important; resize: none !important;
}
"""


def _gradio_process(image):
    if image is None:
        return "", gr.update(visible=False), gr.update(visible=False)
    pil = image.convert('RGB')
    md, elapsed = _run_ocr(pil)
    plain = _md_to_plain(md)
    _, ner_html_str = _run_ner(plain)
    ner_elapsed = 0.0
    timing = f"⏱ OCR {elapsed:.1f}s  ·  NER {ner_elapsed:.1f}s"
    return (
        md,
        gr.update(value=timing, visible=True),
        gr.update(value=ner_html_str, visible=True),
    )


with gr.Blocks(title="OCR Vietnamese") as _demo:
    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            img_in = gr.Image(
                type="pil", show_label=False, height=460,
                sources=["upload", "webcam", "clipboard"],
            )
            btn = gr.Button("Nhận dạng", variant="primary", elem_id="run-btn")
        with gr.Column(scale=1):
            result_md = gr.Markdown(value="", elem_id="result-box", show_label=False)
            timing_box = gr.Textbox(
                value="", interactive=False, show_label=False,
                visible=False, max_lines=1, lines=1, elem_id="timing",
            )
            gr.HTML('<div id="ner-label">Thực thể nhận dạng (NER)</div>')
            ner_out = gr.HTML(value="", visible=False, elem_id="ner-box")

    btn.click(_gradio_process, inputs=[img_in], outputs=[result_md, timing_box, ner_out])


# Mount Gradio vào FastAPI
app = gr.mount_gradio_app(_api, _demo, path="/")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if _in_colab():
        _demo.launch(server_name="0.0.0.0", server_port=7860, share=True, quiet=False,
                     css=CSS, theme=gr.themes.Base())
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")
