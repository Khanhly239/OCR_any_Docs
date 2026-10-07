"""
OCR Viewer — localhost Gradio UI để test từng engine.
Chạy: python scripts/ocr_viewer.py
Mở:   http://localhost:7861
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import gradio as gr
import numpy as np
from PIL import Image

# ── Lazy engine loader ──────────────────────────────────────────────────────

_engines: dict = {}


def _get_engine(name: str):
    if name in _engines:
        return _engines[name]
    from config.settings import settings
    from ocrdoc.core.vram_manager import VRAMManager
    vram = VRAMManager()
    if name == "vintern":
        from ocrdoc.engines.vintern_engine import VinternEngine
        eng = VinternEngine(settings, vram)
    elif name == "chandra":
        from ocrdoc.engines.chandra_engine import ChandraEngine
        eng = ChandraEngine(settings, vram)
    elif name == "easyocr":
        from ocrdoc.engines.easyocr_engine import EasyOCREngine
        eng = EasyOCREngine(settings, vram)
    elif name == "openocr":
        from ocrdoc.engines.openocr_engine import OpenOCREngine
        eng = OpenOCREngine(settings, vram)
    elif name == "marker":
        from ocrdoc.engines.marker_engine import MarkerEngine
        eng = MarkerEngine(settings, vram)
    else:
        from ocrdoc.engines.paddle_engine import PaddleEngine
        eng = PaddleEngine(settings, vram)
    eng.warm_up()
    _engines[name] = eng
    return eng


# ── Markdown → HTML (inline styles, không phụ thuộc CSS) ───────────────────

def md_to_html(text: str) -> str:
    import html as _h, re

    WRAP = (
        'style="font-family:\'Segoe UI\',sans-serif;font-size:14px;'
        'line-height:1.7;color:#e2e8f0;background:#1e293b;'
        'border-radius:8px;padding:16px 20px;'
        'max-height:560px;overflow-y:auto"'
    )
    H1  = 'style="color:#f1f5f9;font-size:1.15rem;margin:14px 0 6px;font-weight:700"'
    H2  = 'style="color:#818cf8;font-size:1.05rem;margin:10px 0 4px;font-weight:600"'
    H3  = 'style="color:#a5b4fc;font-size:.97rem;margin:8px 0 4px;font-weight:600"'
    P   = 'style="margin:2px 0"'
    TBL = ('style="border-collapse:collapse;width:100%;margin:10px 0;font-size:13px"')
    TD  = 'style="border:1px solid #334155;padding:6px 12px;vertical-align:top;color:#e2e8f0"'
    TH  = ('style="border:1px solid #334155;padding:6px 12px;'
           'background:#334155;font-weight:600;color:#cbd5e1"')
    HR  = '<hr style="border:none;border-top:1px solid #334155;margin:10px 0">'
    UL  = 'style="margin:4px 0 4px 1.4rem;padding:0;color:#e2e8f0"'
    OL  = 'style="margin:4px 0 4px 1.4rem;padding:0;color:#e2e8f0"'
    LI  = 'style="margin:1px 0"'
    PRE = ('style="background:#0f172a;border-radius:6px;padding:10px 14px;'
           'font-family:\'Consolas\',monospace;font-size:12px;color:#7dd3fc;'
           'white-space:pre-wrap;margin:8px 0;overflow-x:auto"')

    def _inline(s: str) -> str:
        """Escape HTML rồi áp dụng inline markdown: bold, italic, code, LaTeX."""
        s = _h.escape(s)
        # LaTeX $$...$$
        s = re.sub(r'\$\$(.+?)\$\$',
                   lambda m: f'<span style="color:#fcd34d;font-family:monospace">{_h.escape(m.group(1))}</span>',
                   s)
        # Inline code `...`
        s = re.sub(r'`([^`]+)`',
                   lambda m: f'<code style="background:#0f172a;border-radius:3px;padding:1px 5px;'
                              f'font-family:monospace;font-size:12px;color:#7dd3fc">{_h.escape(m.group(1))}</code>',
                   s)
        # Bold **...**
        s = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', s)
        # Italic *...* (không phải **)
        s = re.sub(r'\*([^*\n]+)\*', r'<em style="color:#c4b5fd">\1</em>', s)
        return s

    lines = text.splitlines()
    out = []
    in_table = False
    first_row = False
    in_code  = False
    code_buf: list[str] = []
    in_ul    = False
    in_ol    = False

    def _close_table():
        nonlocal in_table, first_row
        if in_table:
            out.append("</tbody></table>")
            in_table = first_row = False

    def _close_list():
        nonlocal in_ul, in_ol
        if in_ul:
            out.append("</ul>")
            in_ul = False
        if in_ol:
            out.append("</ol>")
            in_ol = False

    def _flush_code():
        nonlocal in_code, code_buf
        code_html = _h.escape("\n".join(code_buf))
        out.append(f'<pre {PRE}>{code_html}</pre>')
        in_code = False
        code_buf = []

    for ln in lines:
        raw = ln.strip()

        # ── Code fence ──────────────────────────────────────────────────────
        if raw.startswith("```"):
            if in_code:
                _flush_code()
            else:
                _close_table(); _close_list()
                in_code = True
                code_buf = []
            continue
        if in_code:
            code_buf.append(ln)
            continue

        # ── Empty line ──────────────────────────────────────────────────────
        if not raw:
            _close_table(); _close_list()
            out.append("<br>")
            continue

        # ── Horizontal rule ─────────────────────────────────────────────────
        if re.match(r"^[-*_]{3,}$", raw):
            _close_table(); _close_list()
            out.append(HR)
            continue

        # ── Table separator |---|---| ────────────────────────────────────────
        if re.match(r"^\|[-| :]+\|$", raw):
            continue

        # ── Table row | cell | cell | ────────────────────────────────────────
        if re.match(r"^\|.+\|$", raw):
            _close_list()
            cells = [_inline(c.strip()) for c in raw.strip("|").split("|")]
            if not in_table:
                out.append(f'<table {TBL}><tbody>')
                in_table, first_row = True, True
            tag   = "th" if first_row else "td"
            style = TH   if first_row else TD
            out.append("<tr>" + "".join(f"<{tag} {style}>{c}</{tag}>" for c in cells) + "</tr>")
            first_row = False
            continue

        _close_table()

        # ── Headings ─────────────────────────────────────────────────────────
        if raw.startswith("#### "):
            _close_list(); out.append(f"<h5 {H3}>{_inline(raw[5:])}</h5>"); continue
        if raw.startswith("### "):
            _close_list(); out.append(f"<h4 {H3}>{_inline(raw[4:])}</h4>"); continue
        if raw.startswith("## "):
            _close_list(); out.append(f"<h3 {H2}>{_inline(raw[3:])}</h3>"); continue
        if raw.startswith("# "):
            _close_list(); out.append(f"<h2 {H1}>{_inline(raw[2:])}</h2>"); continue

        # ── Bullet list ──────────────────────────────────────────────────────
        if re.match(r"^[-*+] ", raw):
            if not in_ul:
                if in_ol:
                    out.append("</ol>"); in_ol = False
                out.append(f"<ul {UL}>"); in_ul = True
            out.append(f"<li {LI}>{_inline(raw[2:])}</li>")
            continue

        # ── Numbered list ────────────────────────────────────────────────────
        m = re.match(r"^(\d+)\. (.+)$", raw)
        if m:
            if not in_ol:
                if in_ul:
                    out.append("</ul>"); in_ul = False
                out.append(f"<ol {OL}>"); in_ol = True
            out.append(f"<li {LI}>{_inline(m.group(2))}</li>")
            continue

        # ── Plain paragraph ──────────────────────────────────────────────────
        _close_list()
        out.append(f"<p {P}>{_inline(raw)}</p>")

    # flush leftovers
    _close_table(); _close_list()
    if in_code and code_buf:
        _flush_code()

    return f"<div {WRAP}>{''.join(out)}</div>"


# ── Render PageResult → HTML + plain text ──────────────────────────────────

def _table_to_html(table) -> str:
    """Chuyển Table model → HTML table với inline styles."""
    TD = 'style="border:1px solid #334155;padding:6px 12px;vertical-align:top;color:#e2e8f0"'
    TH = ('style="border:1px solid #334155;padding:6px 12px;'
          'background:#334155;font-weight:600;color:#cbd5e1"')
    TBL = ('style="border-collapse:collapse;width:100%;'
           'margin:10px 0;font-size:13px;background:#0f172a"')

    grid = table.to_2d()
    if not grid:
        return ""
    rows_html = []
    for i, row in enumerate(grid):
        tag, style = ("th", TH) if i == 0 else ("td", TD)
        cells = "".join(f"<{tag} {style}>{c}</{tag}>" for c in row)
        rows_html.append(f"<tr>{cells}</tr>")
    return f'<table {TBL}><tbody>{"".join(rows_html)}</tbody></table>'


def _table_to_md(table) -> str:
    """Chuyển Table model → markdown table cho plain text output."""
    grid = table.to_2d()
    if not grid:
        return ""
    lines = []
    for i, row in enumerate(grid):
        lines.append("| " + " | ".join(row) + " |")
        if i == 0:
            lines.append("|" + "|".join(["---"] * len(row)) + "|")
    return "\n".join(lines)


def result_to_outputs(result):
    """Kết hợp text_blocks + tables theo thứ tự Y, trả về (plain_text, html)."""
    # Marker trả về full markdown trong 1 block — render toàn bộ qua md_to_html
    if result.engine_used == "marker":
        raw = "\n".join(b.text for b in result.text_blocks if b.text.strip())
        return raw, md_to_html(raw)

    WRAP = (
        'style="font-family:\'Segoe UI\',sans-serif;font-size:14px;'
        'line-height:1.7;color:#e2e8f0;background:#1e293b;'
        'border-radius:8px;padding:16px 20px;'
        'max-height:540px;overflow-y:auto"'
    )
    BLK_TYPE_STYLE = {
        "heading": 'style="color:#818cf8;font-size:1.05rem;font-weight:600;margin:10px 0 4px"',
        "header":  'style="color:#94a3b8;font-size:.85rem;margin:2px 0"',
        "footer":  'style="color:#94a3b8;font-size:.85rem;margin:2px 0"',
        "caption": 'style="color:#a5b4fc;font-style:italic;margin:2px 0"',
    }
    P = 'style="margin:2px 0"'

    # Gộp text_blocks và tables vào 1 list, sort theo Y (top của bbox)
    items: list[tuple[int, str, object]] = []  # (y, kind, obj)
    for blk in result.text_blocks:
        y = blk.bbox.y if blk.bbox else 0
        items.append((y, "text", blk))
    for tbl in result.tables:
        y = tbl.bbox.y if tbl.bbox else 0
        items.append((y, "table", tbl))
    items.sort(key=lambda x: x[0])

    html_parts, text_parts = [], []
    import html as _h

    for _, kind, obj in items:
        if kind == "table":
            html_parts.append(_table_to_html(obj))
            text_parts.append(_table_to_md(obj))
        else:
            t = obj.text.strip()
            if not t:
                continue
            text_parts.append(t)
            # Nếu text trông như markdown table (Vintern output)
            import re
            if re.match(r"^\|.+\|$", t):
                html_parts.append(md_to_html(t))
            else:
                bstyle = BLK_TYPE_STYLE.get(obj.block_type, P)
                html_parts.append(f"<p {bstyle}>{_h.escape(t)}</p>")

    full_html = f"<div {WRAP}>{''.join(html_parts)}</div>"
    full_text = "\n".join(text_parts)
    return full_text, full_html


# ── OCR function ────────────────────────────────────────────────────────────

def _load_pages(file_obj) -> list[np.ndarray]:
    """Nhận gr.File object (hoặc PIL) → trả về list[BGR ndarray]."""
    from pathlib import Path as _P
    from ocrdoc.preprocessing.pdf_converter import PDFConverter

    if file_obj is None:
        return []

    # gr.File trả về object có .name (path) hoặc PIL Image
    if hasattr(file_obj, "name"):
        path = _P(file_obj.name)
        if path.suffix.lower() == ".pdf":
            return PDFConverter(dpi=200).convert(path, max_pages=50)
        # Ảnh thường
        pil = Image.open(path).convert("RGB")
        return [cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)]

    # PIL Image (từ gr.Image)
    if hasattr(file_obj, "convert"):
        return [cv2.cvtColor(np.array(file_obj.convert("RGB")), cv2.COLOR_RGB2BGR)]

    return []


def run_ocr(file_obj, engine_name: str, doc_type: str):
    if file_obj is None:
        yield "", md_to_html("_Chưa có file_"), "Chưa chọn file"
        return

    try:
        pages = _load_pages(file_obj)
    except Exception as e:
        msg = f"❌ Lỗi đọc file: {e}"
        yield "", md_to_html(f"**{msg}**"), msg
        return

    if not pages:
        yield "", md_to_html("_Không đọc được file_"), "Lỗi đọc file"
        return

    is_pdf = len(pages) > 1
    page_label = f"{len(pages)} trang PDF" if is_pdf else "ảnh"
    yield "", _loading_html(engine_name), f"⏳ Đang load {engine_name}... ({page_label})"

    t0 = time.perf_counter()
    try:
        engine = _get_engine(engine_name)
        from ocrdoc.core.models import PageResult

        all_results: list[PageResult] = []
        for i, bgr in enumerate(pages):
            if is_pdf:
                yield "", _loading_html(engine_name), f"⏳ OCR trang {i+1}/{len(pages)}..."
            r = engine.process(bgr, lang=["vi", "en"], doc_type=doc_type)
            r.page_number = i + 1
            all_results.append(r)

        # Gộp nhiều trang thành 1 output
        if len(all_results) == 1:
            result = all_results[0]
        else:
            # Tạo PageResult tổng hợp để result_to_outputs xử lý
            from ocrdoc.core.models import PageResult as _PR
            combined_blocks = [b for p in all_results for b in p.text_blocks]
            combined_tables = [t for p in all_results for t in p.tables]
            result = _PR(
                page_number=0,
                width=all_results[0].width,
                height=all_results[0].height,
                text_blocks=combined_blocks,
                tables=combined_tables,
                engine_used=engine_name,
                processing_time_ms=sum(p.processing_time_ms for p in all_results),
            )

    except Exception as e:
        msg = f"❌ Lỗi: {e}"
        yield "", md_to_html(f"**{msg}**"), msg
        return

    elapsed = time.perf_counter() - t0
    full_text, full_html = result_to_outputs(result)
    stats = (
        f"✅  {engine_name} | {page_label} | ⏱ {elapsed:.1f}s | "
        f"📊 {len(result.tables)} bảng | "
        f"📝 {len(result.text_blocks)} blocks | "
        f"🔤 {len(full_text)} ký tự"
    )
    yield full_text, full_html, stats


def _loading_html(name: str) -> str:
    return (
        '<div style="font-family:\'Segoe UI\',sans-serif;color:#fbbf24;'
        'background:#1e293b;border-radius:8px;padding:20px;font-size:14px">'
        f'⏳ Đang load engine <strong>{name}</strong>...<br>'
        '<span style="color:#64748b;font-size:12px">'
        + (
            'Marker lần đầu cần download ~2GB models (10-20 phút). '
            'Xem tiến trình trong terminal.'
            if name == "marker" else
            'Lần đầu mất 1-3 phút để load model'
        )
        + '</span></div>'
    )


# ── Gradio UI ───────────────────────────────────────────────────────────────

ENGINE_OPTIONS = [
    ("Vintern 1B  — VLM tiếng Việt có dấu (~2GB VRAM) ⭐",        "vintern"),
    ("EasyOCR  — Tiếng Việt có dấu, ảnh chụp",                    "easyocr"),
    ("Paddle OCR  — Form, invoice, bảng biểu, tiếng Việt có dấu", "paddle"),
    ("Chandra 7B  — VLM lớn (cần ≥8GB VRAM, RTX 3050 NG!)",       "chandra"),
    ("OpenOCR SVTRv2  — Scene text",                               "openocr"),
    ("Marker  — Surya layout+OCR → Markdown (~4GB VRAM)",          "marker"),
]

CSS = """
.gr-button-primary { background: #6366f1 !important; border-color: #6366f1 !important; }
footer { display: none !important; }
"""

with gr.Blocks(title="OCR Viewer", css=CSS, theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "## 🔍 OCR Viewer\n"
        "Upload ảnh hoặc PDF → chọn engine → bấm **Chạy OCR**"
    )

    with gr.Row(equal_height=False):
        # ── Cột trái: input controls ────────────────────────────────────────
        with gr.Column(scale=1, min_width=300):
            image_input = gr.File(
                label="Ảnh hoặc PDF",
                file_types=["image", ".pdf"],
                height=120,
            )
            engine_radio = gr.Radio(
                choices=ENGINE_OPTIONS,
                value="vintern",
                label="Engine OCR",
            )
            doc_type_radio = gr.Radio(
                choices=[
                    ("📋  Có bảng / form / hóa đơn  (markdown table)", "structured"),
                    ("📄  Văn bản thường / thư / báo cáo",              "unstructured"),
                ],
                value="structured",
                label="Loại tài liệu",
            )
            with gr.Row():
                ocr_btn  = gr.Button("▶  Chạy OCR", variant="primary", size="lg")
                clear_btn = gr.Button("🗑", size="lg", variant="secondary")
            status_box = gr.Textbox(
                label="", lines=1, interactive=False,
                placeholder="Trạng thái sẽ hiện ở đây...",
            )

        # ── Cột phải: output ────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=340):
            with gr.Tabs():
                with gr.Tab("🎨 Render  (bảng + định dạng)"):
                    html_out = gr.HTML(
                        value='<div style="color:#475569;padding:20px;'
                              'font-family:\'Segoe UI\',sans-serif">'
                              'Kết quả sẽ hiển thị ở đây...</div>'
                    )
                with gr.Tab("📄 Văn bản thuần"):
                    text_out = gr.Textbox(
                        label="",
                        lines=24,
                        show_copy_button=True,
                        autoscroll=False,
                        placeholder="OCR output...",
                    )

    def _run(img, eng, dtype):
        for text, html, status in run_ocr(img, eng, dtype):
            yield gr.update(value=html), gr.update(value=text), gr.update(value=status)

    def _clear():
        return (
            gr.update(value=None),
            gr.update(value='<div style="color:#475569;padding:20px">Kết quả sẽ hiển thị ở đây...</div>'),
            gr.update(value=""),
            gr.update(value=""),
        )

    ocr_btn.click(
        fn=_run,
        inputs=[image_input, engine_radio, doc_type_radio],
        outputs=[html_out, text_out, status_box],
    )
    clear_btn.click(
        fn=_clear,
        outputs=[image_input, html_out, text_out, status_box],
    )

demo.queue(max_size=2)   # bắt buộc để generator yield hoạt động

if __name__ == "__main__":
    print("OCR Viewer: http://localhost:7861", flush=True)
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False)
