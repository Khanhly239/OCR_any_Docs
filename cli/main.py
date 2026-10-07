from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="ocrdoc",
    help="OCR Tiếng Việt + Tiếng Anh — chạy hoàn toàn local",
    no_args_is_help=True,
)
console = Console()


@app.command()
def process(
    input_path: Path = typer.Argument(..., help="File ảnh hoặc PDF cần OCR"),
    output_dir: Path = typer.Option(Path("."), "--output", "-o", help="Thư mục lưu kết quả"),
    mode: str = typer.Option("auto", "--mode", "-m", help="auto | structured | unstructured | chandra | openocr | easyocr | vintern"),
    formats: str = typer.Option("json,txt", "--formats", "-f", help="Danh sách format, phân tách bằng dấu phẩy"),
    lang: str = typer.Option("vi,en", "--lang", "-l", help="Ngôn ngữ (vi,en)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Hiện log chi tiết"),
) -> None:
    """Xử lý một file ảnh hoặc PDF và lưu kết quả."""
    if not input_path.exists():
        console.print(f"[red]File không tồn tại: {input_path}[/red]")
        raise typer.Exit(1)

    if verbose:
        logger.enable("ocrdoc")
    else:
        logger.remove()

    fmt_list = [f.strip() for f in formats.split(",") if f.strip()]
    lang_list = [l.strip() for l in lang.split(",") if l.strip()]
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[cyan]OCRdoc[/cyan] đang xử lý: [bold]{input_path.name}[/bold]")
    console.print(f"  Mode: {mode} | Formats: {fmt_list} | Lang: {lang_list}")

    import time
    from ocrdoc.orchestrator import OCROrchestrator
    import base64

    orchestrator = OCROrchestrator()
    t0 = time.perf_counter()

    with console.status("Đang OCR..."):
        outputs = orchestrator.process(
            source=input_path,
            filename=input_path.name,
            mode=mode,  # type: ignore
            output_formats=fmt_list,
            lang=lang_list,
        )

    elapsed = time.perf_counter() - t0
    console.print(f"[green]✓ Hoàn tất[/green] trong {elapsed:.1f}s")

    # Lưu files
    stem = input_path.stem
    for fmt, content in outputs.items():
        if fmt == "xlsx" or fmt.endswith("xlsx"):
            out_path = output_dir / f"{stem}.xlsx"
            out_path.write_bytes(base64.b64decode(content))
        else:
            ext = {"json": "json", "txt": "txt", "markdown": "md", "md": "md"}.get(fmt, fmt)
            out_path = output_dir / f"{stem}.{ext}"
            out_path.write_text(content, encoding="utf-8")
        console.print(f"  → [blue]{out_path}[/blue]")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="Host để lắng nghe"),
    port: int = typer.Option(8000, "--port", "-p", help="Port"),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload (chỉ dùng khi dev)"),
) -> None:
    """Khởi động FastAPI + Gradio server."""
    import uvicorn
    console.print(f"[cyan]OCRdoc Server[/cyan] đang khởi động tại http://{host}:{port}")
    console.print(f"  API docs: http://localhost:{port}/docs")
    console.print(f"  Gradio UI: http://localhost:{port}/ui")
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        workers=1,
        reload=reload,
    )


@app.command()
def info() -> None:
    """Hiển thị thông tin hệ thống và trạng thái GPU."""
    table = Table(title="OCRdoc System Info")
    table.add_column("Mục", style="cyan")
    table.add_column("Giá trị", style="white")

    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        if cuda_ok:
            gpu_name = torch.cuda.get_device_name(0)
            free, total = torch.cuda.mem_get_info()
            vram_info = f"{free // (1024**2)}MB / {total // (1024**2)}MB free"
        else:
            gpu_name = "N/A"
            vram_info = "N/A"
        table.add_row("PyTorch", torch.__version__)
        table.add_row("CUDA", "✓ " + gpu_name if cuda_ok else "✗ Không có GPU")
        table.add_row("VRAM", vram_info)
    except ImportError:
        table.add_row("PyTorch", "Chưa cài — chạy scripts\\install_gpu.ps1")

    try:
        import paddleocr
        table.add_row("PaddleOCR", getattr(paddleocr, "__version__", "installed"))
    except ImportError:
        table.add_row("PaddleOCR", "Chưa cài")

    try:
        import chandra
        table.add_row("Chandra OCR 2", getattr(chandra, "__version__", "installed"))
    except ImportError:
        table.add_row("Chandra OCR 2", "Chưa cài — pip install \"chandra-ocr[hf]\"")

    try:
        import gradio as gr
        table.add_row("Gradio", gr.__version__)
    except ImportError:
        table.add_row("Gradio", "Chưa cài")

    console.print(table)


if __name__ == "__main__":
    app()
