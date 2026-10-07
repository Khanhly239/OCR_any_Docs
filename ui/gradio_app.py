from __future__ import annotations

import base64
import json
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Generator

import gradio as gr

if TYPE_CHECKING:
    from ocrdoc.orchestrator import OCROrchestrator

# ── Engine catalog — hiển thị trong UI ────────────────────────────────────────
ENGINE_CHOICES = [
    ("🔍  Tự động  —  AI phát hiện loại tài liệu",                     "auto"),
    ("📊  Paddle + PPStructure  —  Form, invoice, bảng biểu",           "structured"),
    ("🇻🇳  Vintern 1B  ⭐  —  VLM tiếng Việt, khuyên dùng",            "vintern"),
    ("👁  EasyOCR  —  Tiếng Việt rõ nét, ảnh chụp",                    "easyocr"),
    ("📄  Paddle plain  —  Văn bản thường, không bảng",                  "unstructured"),
    ("🤖  Chandra 7B  —  VLM lớn (cần ≥ 8 GB VRAM)",                   "chandra"),
    ("🔧  OpenOCR SVTRv2  —  Scene text, biển quảng cáo",               "openocr"),
]

ENGINE_LABEL = {v: k.split("—")[0].strip().lstrip("🔍📊🇻🇳👁📄🤖🔧⭐ ") for k, v in ENGINE_CHOICES}

CSS = """
/* Engine radio: mỗi option thành card */
.engine-radio .wrap { gap: 6px !important; }
.engine-radio label {
    display: block;
    padding: 10px 14px;
    border: 1px solid #334155;
    border-radius: 10px;
    cursor: pointer;
    transition: border-color .15s, background .15s;
}
.engine-radio label:has(input:checked) {
    border-color: #6366f1;
    background: #1e1b4b22;
}
/* Status box color */
.status-ok  textarea { color: #22c55e !important; }
.status-err textarea { color: #ef4444 !important; }
/* Wider output */
.output-col { min-width: 0; }
"""


def build_ui(orchestrator: "OCROrchestrator | None" = None) -> gr.Blocks:
    if orchestrator is None:
        from ocrdoc.orchestrator import OCROrchestrator
        orchestrator = OCROrchestrator()

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _resolve_path(file_obj) -> Path | None:
        if file_obj is None:
            return None
        if isinstance(file_obj, str):
            p = Path(file_obj)
        elif isinstance(file_obj, dict):
            p = Path(file_obj.get("path") or file_obj.get("name", ""))
        else:
            p = Path(getattr(file_obj, "name", str(file_obj)))
        return p if p.exists() else None

    def _save_tmp(content, stem: str, suffix: str) -> str | None:
        if not content:
            return None
        is_bytes = isinstance(content, bytes)
        tmp = tempfile.NamedTemporaryFile(
            suffix=suffix, delete=False, prefix=f"{stem}_",
            mode="wb" if is_bytes else "w",
            **({"encoding": "utf-8"} if not is_bytes else {}),
        )
        tmp.write(content)
        tmp.close()
        return tmp.name

    # 7-tuple helpers — keeps yields readable
    def _st(msg: str, text="", md="", js="{}"):
        """Status-only yield (no downloads yet)."""
        return (text, md, js, None, None, None, msg)

    def _done(text, md, js, txt_f, md_f, xlsx_f, msg):
        return (text, md, js, txt_f, md_f, xlsx_f, msg)

    # ── Main generator ─────────────────────────────────────────────────────────

    def process_document(
        file_obj,
        mode: str,
        formats: list[str],
    ) -> Generator:
        t0 = time.perf_counter()

        # ── 1. Instant feedback ──────────────────────────────────────
        yield _st("⏳ Đang chuẩn bị...")

        file_path = _resolve_path(file_obj)
        if file_path is None:
            yield _st("❌ Vui lòng upload file trước")
            return

        filename = file_path.name
        stem = file_path.stem
        fmt_list = list(formats) if formats else ["txt"]
        for f in ("txt",):
            if f not in fmt_list:
                fmt_list.append(f)

        # ── 2. Vintern — true token streaming ────────────────────────
        if mode == "vintern":
            yield _st(f"⏳ Vintern: đang load ảnh {filename}...")
            try:
                pages = orchestrator._preprocessor.process_path(file_path)
                doc_type = orchestrator._classify(pages[0], mode)

                with orchestrator._vram.acquire("vintern"):
                    if not orchestrator._vintern.is_ready:
                        yield _st("⏳ Vintern: đang khởi động model (~30s lần đầu)...")
                        orchestrator._vintern.warm_up()

                    yield _st("⏳ Vintern: đang nhận dạng văn bản...")

                    full_text = ""
                    for partial in orchestrator._vintern.stream_ocr(pages[0], doc_type):
                        full_text = partial
                        elapsed = time.perf_counter() - t0
                        yield _st(
                            f"⏳ Đang nhận dạng... {len(full_text)} ký tự | {elapsed:.1f}s",
                            text=full_text,
                        )

                # Build downloads from streamed text
                txt_f  = _save_tmp(full_text, stem, ".txt")
                md_f   = _save_tmp(full_text, stem, ".md")
                elapsed = time.perf_counter() - t0
                yield _done(
                    full_text, full_text, "{}",
                    txt_f, md_f, None,
                    f"✓ Hoàn tất  |  Vintern 1B  |  {elapsed:.1f}s  |  {len(full_text)} ký tự",
                )

            except Exception as exc:
                import traceback
                msg = f"❌ Lỗi Vintern: {exc}\n{traceback.format_exc()}"
                print(msg, flush=True)
                yield _st(msg)
            return

        # ── 3. Batch engines (Paddle / EasyOCR / Chandra / auto …) ───
        label = ENGINE_LABEL.get(mode, mode)
        yield _st(f"⏳ {label}: đang xử lý {filename}...")
        print(f"[UI] OCR start: {filename}, mode={mode}", flush=True)

        try:
            for f in ("markdown",):
                if f not in fmt_list:
                    fmt_list.append(f)

            outputs = orchestrator.process(
                source=file_path,
                filename=filename,
                mode=mode,
                output_formats=fmt_list,
                lang=["vi", "en"],
            )
            print(f"[UI] OCR done: keys={list(outputs.keys())}", flush=True)

        except Exception as exc:
            import traceback
            msg = f"❌ Lỗi {label}: {exc}\n{traceback.format_exc()}"
            print(msg, flush=True)
            yield _st(msg)
            return

        txt  = outputs.get("txt", "")
        md   = outputs.get("markdown", outputs.get("md", ""))
        js   = outputs.get("json", "{}")
        if not isinstance(js, str):
            js = json.dumps(js, ensure_ascii=False, indent=2)

        txt_f  = _save_tmp(txt, stem, ".txt")
        md_f   = _save_tmp(md,  stem, ".md")
        xlsx_b = outputs.get("xlsx")
        xlsx_f = _save_tmp(base64.b64decode(xlsx_b) if xlsx_b else None, stem, ".xlsx")

        elapsed = time.perf_counter() - t0
        try:
            meta    = json.loads(js)
            n_pages  = len(meta.get("pages", []))
            n_tables = sum(len(p.get("tables", [])) for p in meta.get("pages", []))
            status = (
                f"✓ Hoàn tất  |  {label}  |  "
                f"{n_pages} trang  |  {n_tables} bảng  |  {elapsed:.1f}s"
            )
        except Exception:
            status = f"✓ Hoàn tất  |  {label}  |  {elapsed:.1f}s"

        yield _done(txt, md, js, txt_f, md_f, xlsx_f, status)

    # ── UI Layout ──────────────────────────────────────────────────────────────
    with gr.Blocks(
        title="OCRdoc — OCR Tiếng Việt + Tiếng Anh",
        theme=gr.themes.Soft(),
        css=CSS,
    ) as demo:

        gr.Markdown(
            "# 🔍 OCRdoc\n"
            "**OCR Tiếng Việt + Tiếng Anh — chạy hoàn toàn local**"
        )

        with gr.Row(equal_height=False):

            # ── Left panel ─────────────────────────────────────────────
            with gr.Column(scale=1, min_width=320):

                file_input = gr.File(
                    label="📁 Upload tài liệu",
                    file_types=[
                        ".png", ".jpg", ".jpeg",
                        ".tiff", ".tif", ".bmp", ".pdf",
                    ],
                    type="filepath",
                )

                engine_radio = gr.Radio(
                    choices=ENGINE_CHOICES,
                    value="auto",
                    label="🤖 Chọn Engine / Model",
                    elem_classes=["engine-radio"],
                )

                format_check = gr.CheckboxGroup(
                    choices=[
                        ("📝 Plain text (.txt)", "txt"),
                        ("📋 Markdown (.md)",    "markdown"),
                        ("🔧 JSON (đầy đủ)",     "json"),
                        ("📊 Excel (.xlsx)",      "xlsx"),
                    ],
                    value=["txt", "json"],
                    label="📄 Định dạng output",
                )

                run_btn = gr.Button(
                    "▶  Chạy OCR",
                    variant="primary",
                    size="lg",
                )

                gr.Markdown(
                    "---\n"
                    "**Gợi ý chọn engine:**\n"
                    "- Form / hóa đơn → **Paddle + PPStructure**\n"
                    "- Tài liệu scan tiếng Việt → **Vintern 1B** ⭐\n"
                    "- Ảnh chụp nhanh → **EasyOCR**\n"
                    "- Tự động (không chắc) → **Tự động**"
                )

            # ── Right panel ────────────────────────────────────────────
            with gr.Column(scale=2, elem_classes=["output-col"]):

                status_box = gr.Textbox(
                    label="Trạng thái",
                    interactive=False,
                    lines=2,
                    max_lines=5,
                    placeholder="Upload file và nhấn ▶ Chạy OCR...",
                )

                with gr.Tabs():
                    with gr.Tab("📝 Text"):
                        text_out = gr.Textbox(
                            label="Văn bản trích xuất",
                            lines=24,
                            max_lines=80,
                            show_copy_button=True,
                            placeholder="Kết quả sẽ xuất hiện ở đây...",
                        )
                    with gr.Tab("📋 Markdown"):
                        md_out = gr.Textbox(
                            label="Markdown (bảng, heading)",
                            lines=24,
                            max_lines=80,
                            show_copy_button=True,
                        )
                    with gr.Tab("🔧 JSON"):
                        json_out = gr.Textbox(
                            label="Structured JSON",
                            lines=24,
                            max_lines=80,
                            show_copy_button=True,
                        )

                with gr.Row():
                    dl_txt  = gr.File(label="⬇ .txt",  interactive=False)
                    dl_md   = gr.File(label="⬇ .md",   interactive=False)
                    dl_xlsx = gr.File(label="⬇ .xlsx", interactive=False)

        # ── Event binding ──────────────────────────────────────────────
        run_btn.click(
            fn=process_document,
            inputs=[file_input, engine_radio, format_check],
            outputs=[text_out, md_out, json_out, dl_txt, dl_md, dl_xlsx, status_box],
            concurrency_limit=1,
        )

    demo.queue(max_size=2)
    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="0.0.0.0", server_port=7860, share=False)
