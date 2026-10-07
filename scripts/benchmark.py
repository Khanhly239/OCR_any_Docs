"""
OCR Benchmark — so sánh nhiều engine trên cùng tài liệu.
Chạy: python scripts/benchmark.py
Mở:   http://localhost:7862

Tính năng:
  • So sánh tốc độ xử lý (ms) từng engine
  • Xem output side-by-side trên cùng ảnh
  • Tính CER / WER nếu có file ground truth (.txt cùng tên với ảnh)
  • Batch mode: chạy cả thư mục, xuất báo cáo tổng hợp
"""
from __future__ import annotations

import sys
import time
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import gradio as gr
import numpy as np
from PIL import Image

# ── Engine loader ─────────────────────────────────────────────────────────────

_engines: dict = {}

ENGINES = {
    "vintern": "🇻🇳 Vintern 1B",
    "easyocr": "👁 EasyOCR",
    "paddle":  "📊 Paddle",
    "chandra": "🤖 Chandra 7B",
    "openocr": "🔧 OpenOCR",
}


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
    else:
        from ocrdoc.engines.paddle_engine import PaddleEngine
        eng = PaddleEngine(settings, vram)
    eng.warm_up()
    _engines[name] = eng
    return eng


def _result_to_text(result) -> str:
    items = []
    for blk in result.text_blocks:
        items.append((blk.bbox.y if blk.bbox else 0, "text", blk))
    for tbl in result.tables:
        items.append((tbl.bbox.y if tbl.bbox else 0, "table", tbl))
    items.sort(key=lambda x: x[0])
    parts = []
    for _, kind, obj in items:
        if kind == "table":
            grid = obj.to_2d()
            for i, row in enumerate(grid):
                parts.append("| " + " | ".join(row) + " |")
                if i == 0:
                    parts.append("|" + "|".join(["---"] * len(row)) + "|")
        elif obj.text.strip():
            parts.append(obj.text.strip())
    return "\n".join(parts)


# ── Accuracy metrics ──────────────────────────────────────────────────────────

def _norm(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text.lower()).split())


def _edit_dist(a: list, b: list) -> int:
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        nd = [i + 1]
        for j, cb in enumerate(b):
            nd.append(min(d[j] + (0 if ca == cb else 1), d[j+1]+1, nd[-1]+1))
        d = nd
    return d[-1]


def cer(ref: str, hyp: str) -> float:
    r = list(_norm(ref))
    h = list(_norm(hyp))
    if not r:
        return 0.0 if not h else 1.0
    return min(_edit_dist(r, h) / len(r), 1.0)


def wer(ref: str, hyp: str) -> float:
    r = _norm(ref).split()
    h = _norm(hyp).split()
    if not r:
        return 0.0 if not h else 1.0
    return min(_edit_dist(r, h) / len(r), 1.0)


# ── HTML helpers ──────────────────────────────────────────────────────────────

def _wrap_msg(msg: str) -> str:
    return (f'<div style="font-family:\'Segoe UI\',sans-serif;color:#fbbf24;'
            f'background:#1e293b;border-radius:8px;padding:20px;font-size:14px">{msg}</div>')


def _text_to_html_col(text: str) -> str:
    """Render text (kể cả markdown table) thành HTML cho cột so sánh."""
    import html as _h, re
    lines_html = []
    in_tbl = False
    first_row = True
    for ln in text.splitlines():
        e = _h.escape(ln.strip())
        if not e:
            if in_tbl:
                lines_html.append("</table>"); in_tbl = False
            lines_html.append("<br>"); continue
        if re.match(r"^\|[-| :]+\|$", e):
            continue
        if re.match(r"^\|.+\|$", e):
            cells = [c.strip() for c in e.strip("|").split("|")]
            if not in_tbl:
                lines_html.append(
                    '<table style="border-collapse:collapse;width:100%;'
                    'font-size:12px;margin:6px 0">')
                in_tbl = True; first_row = True
            tag = "th" if first_row else "td"
            bg = "background:#1e3a5f;" if first_row else ""
            row = "".join(
                f'<{tag} style="border:1px solid #334155;padding:4px 8px;{bg}">'
                f'{c}</{tag}>' for c in cells)
            lines_html.append(f"<tr>{row}</tr>"); first_row = False; continue
        if in_tbl:
            lines_html.append("</table>"); in_tbl = False; first_row = True
        lines_html.append(f'<p style="margin:2px 0;font-size:13px">{e}</p>')
    if in_tbl:
        lines_html.append("</table>")
    return "".join(lines_html) or '<span style="color:#475569">Không có kết quả</span>'


# ── Single image benchmark ────────────────────────────────────────────────────

def run_single(pil_image, engine_names: list[str], doc_type: str, gt_text: str):
    if pil_image is None:
        yield _wrap_msg("Chưa có ảnh"), ""; return
    if not engine_names:
        yield _wrap_msg("Chưa chọn engine nào"), ""; return

    bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    has_gt = bool(gt_text.strip())
    results: dict[str, dict] = {}

    for eng_name in engine_names:
        yield _wrap_msg(f"⏳ Đang chạy <b>{ENGINES.get(eng_name, eng_name)}</b>..."), ""
        t0 = time.perf_counter()
        try:
            engine = _get_engine(eng_name)
            result = engine.process(bgr, lang=["vi", "en"], doc_type=doc_type)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            text = _result_to_text(result)
            results[eng_name] = {
                "text": text, "time_ms": elapsed_ms,
                "tables": len(result.tables), "blocks": len(result.text_blocks),
                "cer": cer(gt_text, text) if has_gt else None,
                "wer": wer(gt_text, text) if has_gt else None,
            }
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"\n[benchmark] LỖI {eng_name}:\n{tb}", flush=True)
            results[eng_name] = {
                "text": f"❌ {type(e).__name__}: {e}\n\n{tb}",
                "time_ms": -1, "tables": 0, "blocks": 0,
                "cer": None, "wer": None,
            }

    yield _build_single_report(results, engine_names, has_gt, gt_text), \
          _build_plain(results, engine_names, has_gt)


def _build_single_report(results, order, has_gt, gt_text="") -> str:
    import html as _h
    TH  = 'style="background:#1e3a5f;color:#93c5fd;padding:8px 14px;text-align:left;font-size:13px"'
    TD  = 'style="padding:8px 14px;border-bottom:1px solid #1e293b;font-size:13px"'
    TDR = 'style="padding:8px 14px;border-bottom:1px solid #1e293b;font-size:13px;text-align:right;font-variant-numeric:tabular-nums"'
    TBL = 'style="border-collapse:collapse;width:100%;background:#0f172a;border-radius:8px;overflow:hidden"'

    valid = {k: v for k, v in results.items() if v["time_ms"] > 0}
    fastest  = min(valid, key=lambda k: valid[k]["time_ms"]) if valid else None
    best_cer = (min(valid, key=lambda k: valid[k]["cer"] or 1.0)
                if has_gt and any(v["cer"] is not None for v in valid.values()) else None)

    rows = []
    for name in order:
        r = results.get(name, {})
        label = ENGINES.get(name, name)
        t = r.get("time_ms", -1)
        time_str = f"{t:.0f} ms" if t > 0 else "Lỗi"
        badges = ""
        if name == fastest:
            badges += ' <span style="color:#4ade80;font-size:11px">⚡ fastest</span>'
        if name == best_cer:
            badges += ' <span style="color:#f59e0b;font-size:11px">🎯 best CER</span>'
        cer_val = r.get("cer")
        cer_str = f"{cer_val*100:.1f}%" if cer_val is not None else "—"
        wer_val = r.get("wer")
        wer_str = f"{wer_val*100:.1f}%" if wer_val is not None else "—"
        cer_color = ("#4ade80" if cer_val is not None and cer_val < 0.1
                     else "#fbbf24" if cer_val is not None and cer_val < 0.3
                     else "#f87171")
        rows.append(f"""
        <tr>
          <td {TD}>{label}{badges}</td>
          <td {TDR}>{time_str}</td>
          <td {TDR}>{r.get("tables", 0)}</td>
          <td {TDR}>{r.get("blocks", 0)}</td>
          <td {TDR} style="color:{cer_color}">{cer_str}</td>
          <td {TDR}>{wer_str}</td>
        </tr>""")

    gt_hint = '' if has_gt else '<span style="color:#475569;font-size:11px"> (cần ground truth)</span>'
    summary = f"""
    <table {TBL}>
      <thead><tr>
        <th {TH}>Engine</th><th {TH}>Thời gian</th>
        <th {TH}>Bảng</th><th {TH}>Blocks</th>
        <th {TH}>CER ↓{gt_hint}</th><th {TH}>WER ↓</th>
      </tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table>"""

    # Side-by-side output
    col_w = max(260, 860 // max(len(results), 1))
    cols = []
    for name in order:
        r = results.get(name, {})
        label = ENGINES.get(name, name)
        t = r.get("time_ms", -1)
        time_badge = (f'<span style="font-size:11px;color:#64748b"> {t:.0f}ms</span>'
                      if t > 0 else "")
        inner = _text_to_html_col(r.get("text", ""))
        cols.append(f"""
        <div style="min-width:{col_w}px;flex:1;background:#1e293b;
                    border-radius:8px;padding:12px;display:flex;flex-direction:column">
          <div style="color:#94a3b8;font-size:12px;font-weight:600;margin-bottom:8px">
            {label}{time_badge}
          </div>
          <div style="flex:1;overflow-y:auto;max-height:380px;color:#e2e8f0">{inner}</div>
        </div>""")

    side = f'<div style="display:flex;gap:10px;overflow-x:auto">{"".join(cols)}</div>'

    gt_section = ""
    if has_gt:
        gt_section = f"""
        <h4 style="color:#94a3b8;margin:14px 0 6px;font-size:13px">Ground Truth</h4>
        <div style="background:#0f172a;border-radius:6px;padding:10px;font-size:12px;
                    color:#94a3b8;max-height:100px;overflow-y:auto;white-space:pre-wrap"
        >{_h.escape(gt_text[:600])}{"…" if len(gt_text) > 600 else ""}</div>"""

    WRAP = ('style="font-family:\'Segoe UI\',sans-serif;color:#e2e8f0;'
            'background:#0f172a;padding:16px;border-radius:10px"')
    return f"""
    <div {WRAP}>
      <h3 style="color:#818cf8;margin:0 0 12px">📊 Kết quả Benchmark</h3>
      {summary}
      <h4 style="color:#94a3b8;margin:14px 0 8px;font-size:13px">So sánh Output</h4>
      {side}
      {gt_section}
    </div>"""


def _build_plain(results, order, has_gt) -> str:
    lines = ["=== BENCHMARK SUMMARY ===\n"]
    for name in order:
        r = results.get(name, {})
        label = ENGINES.get(name, name)
        t = r.get("time_ms", -1)
        lines += [
            f"[{label}]",
            f"  Thời gian : {'%.0f ms' % t if t > 0 else 'Lỗi'}",
            f"  Bảng      : {r.get('tables', 0)}",
            f"  Blocks    : {r.get('blocks', 0)}",
        ]
        if has_gt:
            cer_v = r.get("cer")
            wer_v = r.get("wer")
            lines += [
                f"  CER       : {'%.1f%%' % (cer_v*100) if cer_v is not None else '—'}",
                f"  WER       : {'%.1f%%' % (wer_v*100) if wer_v is not None else '—'}",
            ]
        lines.append("")
    return "\n".join(lines)


# ── Batch benchmark ───────────────────────────────────────────────────────────

def run_batch(folder: str, engine_names: list[str], doc_type: str):
    folder_path = Path(folder.strip()) if folder.strip() else None
    if not folder_path or not folder_path.exists():
        yield _wrap_msg("Thư mục không tồn tại"), ""; return
    images = sorted(p for p in folder_path.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"})
    if not images:
        yield _wrap_msg("Không tìm thấy ảnh trong thư mục"), ""; return
    if not engine_names:
        yield _wrap_msg("Chưa chọn engine"), ""; return

    agg: dict[str, list] = {e: [] for e in engine_names}
    total = len(images)

    for idx, img_path in enumerate(images):
        gt_path = img_path.with_suffix(".txt")
        gt_text = gt_path.read_text(encoding="utf-8") if gt_path.exists() else ""
        has_gt = bool(gt_text.strip())
        try:
            bgr = cv2.imread(str(img_path))
            if bgr is None: continue
        except Exception: continue

        yield _wrap_msg(f"⏳ [{idx+1}/{total}] {img_path.name}..."), ""

        for eng_name in engine_names:
            t0 = time.perf_counter()
            try:
                engine = _get_engine(eng_name)
                result = engine.process(bgr, lang=["vi", "en"], doc_type=doc_type)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                text = _result_to_text(result)
                agg[eng_name].append({
                    "file": img_path.name, "time_ms": elapsed_ms, "chars": len(text),
                    "tables": len(result.tables),
                    "cer": cer(gt_text, text) if has_gt else None,
                    "wer": wer(gt_text, text) if has_gt else None,
                })
            except Exception as e:
                agg[eng_name].append({
                    "file": img_path.name, "time_ms": -1,
                    "chars": 0, "tables": 0, "cer": None, "wer": None,
                })

    yield _build_batch_report(agg, engine_names, total)


def _build_batch_report(agg, order, total):
    TH  = 'style="background:#1e3a5f;color:#93c5fd;padding:8px 14px;text-align:left;font-size:13px"'
    TD  = 'style="padding:8px 14px;border-bottom:1px solid #1e293b;font-size:13px"'
    TDR = 'style="padding:8px 14px;border-bottom:1px solid #1e293b;font-size:13px;text-align:right;font-variant-numeric:tabular-nums"'
    TBL = 'style="border-collapse:collapse;width:100%;background:#0f172a;border-radius:8px;overflow:hidden;margin-bottom:16px"'

    rows, plain = [], ["=== BATCH BENCHMARK ===", f"Tổng: {total} ảnh\n"]
    for name in order:
        items = agg.get(name, [])
        valid = [x for x in items if x["time_ms"] > 0]
        if not valid: continue
        avg_t = sum(x["time_ms"] for x in valid) / len(valid)
        min_t = min(x["time_ms"] for x in valid)
        max_t = max(x["time_ms"] for x in valid)
        n_gt  = sum(1 for x in valid if x["cer"] is not None)
        avg_cer = sum(x["cer"] for x in valid if x["cer"] is not None)
        avg_wer = sum(x["wer"] for x in valid if x["wer"] is not None)
        cer_str = f"{(avg_cer/n_gt)*100:.1f}%" if n_gt else "—"
        wer_str = f"{(avg_wer/n_gt)*100:.1f}%" if n_gt else "—"
        cer_color = ("#4ade80" if n_gt and avg_cer/n_gt < 0.1
                     else "#fbbf24" if n_gt and avg_cer/n_gt < 0.3 else "#f87171")
        label = ENGINES.get(name, name)
        rows.append(f"""
        <tr>
          <td {TD}>{label}</td>
          <td {TDR}>{len(valid)}/{total}</td>
          <td {TDR}>{avg_t:.0f} ms</td>
          <td {TDR}>{min_t:.0f} ms</td>
          <td {TDR}>{max_t:.0f} ms</td>
          <td {TDR} style="color:{cer_color}">{cer_str}</td>
          <td {TDR}>{wer_str}</td>
        </tr>""")
        plain += [f"[{label}]",
                  f"  Thành công : {len(valid)}/{total}",
                  f"  Avg time   : {avg_t:.0f} ms",
                  f"  CER avg    : {cer_str}",
                  f"  WER avg    : {wer_str}", ""]

    # Per-file details
    all_files = sorted({x["file"] for items in agg.values() for x in items})
    detail_rows = []
    for fname in all_files:
        for name in order:
            item = next((x for x in agg.get(name, []) if x["file"] == fname), None)
            if not item: continue
            t = item["time_ms"]
            cer_s = f"{item['cer']*100:.1f}%" if item.get("cer") is not None else "—"
            detail_rows.append(f"""
            <tr>
              <td {TD}>{fname}</td>
              <td {TD}>{ENGINES.get(name, name)}</td>
              <td {TDR}>{"%.0f ms" % t if t > 0 else "Lỗi"}</td>
              <td {TDR}>{item.get("tables", 0)}</td>
              <td {TDR}>{cer_s}</td>
            </tr>""")

    detail = f"""
    <details style="margin-top:8px">
      <summary style="cursor:pointer;color:#94a3b8;font-size:13px;padding:4px 0">
        ▶ Chi tiết từng file ({len(all_files)} ảnh)
      </summary>
      <table {TBL} style="margin-top:8px">
        <thead><tr>
          <th {TH}>File</th><th {TH}>Engine</th>
          <th {TH}>Time</th><th {TH}>Bảng</th><th {TH}>CER</th>
        </tr></thead>
        <tbody>{"".join(detail_rows)}</tbody>
      </table>
    </details>"""

    WRAP = ('style="font-family:\'Segoe UI\',sans-serif;color:#e2e8f0;'
            'background:#0f172a;padding:16px;border-radius:10px"')
    html = f"""
    <div {WRAP}>
      <h3 style="color:#818cf8;margin:0 0 12px">📊 Batch Benchmark — {total} ảnh</h3>
      <p style="color:#64748b;font-size:12px;margin-bottom:12px">
        CER/WER chỉ tính khi có file <code>.txt</code> ground truth cùng tên với ảnh.
      </p>
      <table {TBL}>
        <thead><tr>
          <th {TH}>Engine</th><th {TH}>Thành công</th>
          <th {TH}>Avg (ms)</th><th {TH}>Min (ms)</th><th {TH}>Max (ms)</th>
          <th {TH}>CER avg ↓</th><th {TH}>WER avg ↓</th>
        </tr></thead>
        <tbody>{"".join(rows)}</tbody>
      </table>
      {detail}
    </div>"""
    return html, "\n".join(plain)


# ── Gradio UI ─────────────────────────────────────────────────────────────────

ENGINE_CHOICES = [(label, key) for key, label in ENGINES.items()]
DEFAULT_ENGINES = ["vintern", "easyocr", "paddle"]

with gr.Blocks(title="OCR Benchmark", css="footer{display:none!important}",
               theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 📊 OCR Benchmark\nSo sánh accuracy và tốc độ giữa các engine")

    with gr.Tabs():
        # Tab 1: Một ảnh ──────────────────────────────────────────────────────
        with gr.Tab("🖼 Một ảnh"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=280):
                    img_in = gr.Image(type="pil", label="Ảnh đầu vào",
                                      sources=["upload", "clipboard"], height=280)
                    eng_cb = gr.CheckboxGroup(choices=ENGINE_CHOICES,
                                              value=DEFAULT_ENGINES,
                                              label="Chọn engine để so sánh")
                    dtype_r = gr.Radio(
                        choices=[("📋 Có bảng / form", "structured"),
                                 ("📄 Văn bản thường", "unstructured")],
                        value="structured", label="Loại tài liệu")
                    gt_in = gr.Textbox(
                        label="Ground truth (tùy chọn — để tính CER/WER)",
                        lines=3,
                        placeholder="Dán văn bản chuẩn ở đây...")
                    run_btn = gr.Button("▶ Chạy Benchmark", variant="primary", size="lg")

                with gr.Column(scale=2):
                    with gr.Tabs():
                        with gr.Tab("📊 Báo cáo"):
                            html_out = gr.HTML('<div style="padding:20px;color:#475569">Kết quả sẽ hiện ở đây...</div>')
                        with gr.Tab("📄 Text"):
                            txt_out = gr.Textbox(label="", lines=20, show_copy_button=True)

            run_btn.click(fn=run_single,
                          inputs=[img_in, eng_cb, dtype_r, gt_in],
                          outputs=[html_out, txt_out])

        # Tab 2: Batch ────────────────────────────────────────────────────────
        with gr.Tab("📁 Batch (thư mục)"):
            gr.Markdown(
                "Chạy benchmark trên tất cả ảnh trong một thư mục.  \n"
                "Đặt file `.txt` cùng tên với ảnh để tính CER/WER tự động.  \n"
                "Ví dụ: `phieu.jpg` + `phieu.txt` (nội dung chuẩn)")
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=280):
                    folder_in = gr.Textbox(label="Thư mục ảnh",
                                           value=r"C:\Lily\OCRdoc\image",
                                           placeholder=r"C:\Lily\OCRdoc\image")
                    eng_cb2 = gr.CheckboxGroup(choices=ENGINE_CHOICES,
                                               value=DEFAULT_ENGINES, label="Engine")
                    dtype_r2 = gr.Radio(
                        choices=[("📋 Có bảng / form", "structured"),
                                 ("📄 Văn bản thường", "unstructured")],
                        value="structured", label="Loại tài liệu")
                    batch_btn = gr.Button("▶ Chạy Batch", variant="primary", size="lg")
                with gr.Column(scale=2):
                    with gr.Tabs():
                        with gr.Tab("📊 Báo cáo"):
                            html_out2 = gr.HTML('<div style="padding:20px;color:#475569">Kết quả sẽ hiện ở đây...</div>')
                        with gr.Tab("📄 Text"):
                            txt_out2 = gr.Textbox(label="", lines=20, show_copy_button=True)

            batch_btn.click(fn=run_batch,
                            inputs=[folder_in, eng_cb2, dtype_r2],
                            outputs=[html_out2, txt_out2])

        # Tab 3: Hướng dẫn ────────────────────────────────────────────────────
        with gr.Tab("📖 Hướng dẫn"):
            gr.Markdown("""
### Cách đọc kết quả

| Metric | Ý nghĩa | Tốt |
|--------|---------|-----|
| **Time (ms)** | Thời gian xử lý 1 ảnh | Càng nhỏ càng tốt |
| **CER** | Character Error Rate — % ký tự sai | < 5% = tốt, < 15% = chấp nhận |
| **WER** | Word Error Rate — % từ sai | < 10% = tốt, < 25% = chấp nhận |
| **Bảng** | Số bảng detect được | Phụ thuộc tài liệu |

### Đặc điểm từng engine

| Engine | Mạnh nhất | Hạn chế | VRAM |
|--------|-----------|---------|------|
| 🇻🇳 **Vintern 1B** | Tiếng Việt tự do, văn bản scan | Chậm (60-80s) | ~2GB |
| 👁 **EasyOCR** | Ảnh chụp rõ nét | Dấu tiếng Việt đôi khi sai | ~0.5GB |
| 📊 **Paddle** | Form, bảng biểu, hóa đơn | Ảnh mờ / nghiêng | ~1GB |
| 🤖 **Chandra 7B** | Layout phức tạp | Rất chậm, cần 8GB VRAM | ~8GB |
| 🔧 **OpenOCR** | Scene text, biển hiệu | Không tốt với tài liệu in | ~1GB |

### Tính CER/WER tự động (Batch mode)

```
image/
  phieu.jpg
  phieu.txt        ← nội dung văn bản chuẩn (ground truth)
  hop_dong.png
  hop_dong.txt
```

CER < 5% = rất tốt | CER 5-15% = chấp nhận | CER > 15% = cần cải thiện
            """)

if __name__ == "__main__":
    print("OCR Benchmark: http://localhost:7862", flush=True)
    demo.launch(server_name="0.0.0.0", server_port=7862, share=False)
