# 🔍 OCRdoc

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

**Hệ thống OCR tiếng Việt + tiếng Anh, chạy hoàn toàn local — không gửi tài liệu ra ngoài.**

OCRdoc gom 6 OCR engine (truyền thống + Vision-Language Model) vào một pipeline duy nhất: tự động phân loại tài liệu, chọn engine phù hợp, chuẩn hoá dấu tiếng Việt, rồi xuất ra `txt` / `markdown` / `json` / `xlsx`. Dùng được qua **CLI**, **REST API** hoặc **web UI**.

```
File / PDF ──▶ Preprocessing ──▶ Classifier ──▶ Engine ──▶ Postprocessing ──▶ Exporter
               deskew, denoise    structured?    1 trong 6   chuẩn hoá tiếng Việt,  txt/md/json/xlsx
               enhance, pdf2img   unstructured?              reading order
```

---

## ✨ Điểm chính

- **6 engine, một API thống nhất** — mỗi engine implement cùng interface `BaseEngine`, chuyển engine chỉ bằng một tham số `mode`.
- **Chạy 100% offline** — model tải về máy một lần, không có call ra cloud.
- **Tối ưu cho tiếng Việt** — Vintern-1B (VLM fine-tune tiếng Việt), EasyOCR vi, PaddleOCR `vi_PP-OCRv4` rec weights, cộng thêm bước chuẩn hoá Unicode NFC và sửa lỗi dấu thanh.
- **VRAM manager** — lock singleton đảm bảo chỉ một engine chiếm GPU tại một thời điểm, tránh OOM khi chạy nhiều model nặng trên cùng một card.
- **Tự phân loại tài liệu** — classifier dựa trên Hough line + contour (OpenCV, CPU, <50ms) nhận ra form/hoá đơn/bảng biểu (`structured`) so với văn bản chạy chữ (`unstructured`), rồi tự chọn engine.
- **Streaming progress** — endpoint SSE báo tiến độ từng trang cho file PDF dài.

## 🤖 Các engine hỗ trợ

| Engine | Model | VRAM | Hợp với |
|---|---|---|---|
| `structured` | PaddleOCR + PP-StructureV3 | ~2 GB | Form, hoá đơn, bảng biểu (trích xuất cả cấu trúc bảng) |
| `vintern` ⭐ | Vintern-1B-v2 (InternVL2 + Qwen2) | ~2 GB | Tài liệu scan tiếng Việt — khuyên dùng |
| `easyocr` | EasyOCR vi + en | ~1 GB | Ảnh chụp, text rõ nét, đủ dấu tiếng Việt |
| `unstructured` | PaddleOCR | ~2 GB | Văn bản thường, không bảng |
| `chandra` | Chandra OCR 2 (~7B, Qwen2-VL) | ~8 GB | Layout + OCR + table trong một model |
| `openocr` | OpenOCR SVTRv2 (det) + Paddle vi (rec) | ~3 GB | Scene text, biển hiệu, ảnh ngoài trời |
| `marker` | Marker / Surya (venv riêng) | ~4 GB | Tài liệu học thuật → Markdown, LaTeX, bảng phức tạp |

Để `mode=auto` thì classifier tự quyết định.

## 📦 Định dạng

**Input:** `.png` `.jpg` `.jpeg` `.tiff` `.tif` `.bmp` `.webp` `.pdf` (tối đa 100 trang, cấu hình được)
**Output:** `.txt` · `.md` · `.json` (đầy đủ bbox + confidence + block type) · `.xlsx` (bảng)

---

## 🚀 Cài đặt

Yêu cầu: **Python 3.10+**, NVIDIA GPU với CUDA 12.6+ (chạy CPU được nhưng chậm hơn nhiều).

```bash
git clone <repo-url> && cd OCRdoc
python -m venv .venv && .venv\Scripts\activate     # Windows
pip install -r requirements.txt
pip install -e .                                   # đăng ký lệnh `ocrdoc`

# PyTorch + PaddlePaddle bản CUDA (Windows)
powershell scripts\install_gpu.ps1

cp .env.example .env        # chỉnh device, VRAM limit, đường dẫn model
python scripts/download_models.py
```

Marker chạy trong venv riêng (xung đột `transformers` 5.x): `scripts\setup_marker_venv.ps1`.
Chandra có thể chạy tách biệt bằng Docker: `docker compose -f docker-compose.chandra.yml up`.

## 💻 Sử dụng

**CLI**

```bash
ocrdoc process invoice.pdf -o ./out -m auto -f json,txt,xlsx   # hoặc: python -m cli.main process ...
ocrdoc serve --port 8000
ocrdoc info                      # kiểm tra GPU / VRAM / phiên bản engine
```

**Server** — FastAPI + Gradio UI trong cùng một process:

```bash
python -m api.main
```

| URL | |
|---|---|
| `http://localhost:8000/ui` | Web UI (Gradio) — upload, chọn engine, xem kết quả, tải file |
| `http://localhost:8000/docs` | OpenAPI / Swagger |

**REST API**

```bash
curl -X POST http://localhost:8000/api/v1/ocr/process \
  -F "file=@document.pdf" \
  -F "mode=auto" \
  -F "formats=json,txt" \
  -F "lang=vi,en"
```

| Endpoint | Mô tả |
|---|---|
| `POST /api/v1/ocr/process` | OCR một file, trả về tất cả format đã chọn |
| `POST /api/v1/ocr/process/stream` | Như trên, nhưng stream tiến độ từng trang (SSE) |
| `GET /api/v1/health` | Trạng thái service + VRAM còn trống |
| `GET /api/v1/info` | Danh sách engine, ngôn ngữ, phiên bản |

**Python**

```python
from ocrdoc.orchestrator import OCROrchestrator

orchestrator = OCROrchestrator()
outputs = orchestrator.process(
    source=Path("invoice.pdf"),
    mode="auto",
    output_formats=["json", "markdown"],
)
print(outputs["markdown"])
```

---

## 🏗 Kiến trúc

```
ocrdoc/
├── core/            models (Pydantic), classifier, vram_manager
├── preprocessing/   deskew, denoise, enhance, pdf_converter, pipeline
├── engines/         base + paddle, chandra, easyocr, openocr, vintern, marker
├── postprocessing/  vietnamese (chuẩn hoá dấu), reading_order, merger
├── exporters/       json, markdown, text, excel
└── orchestrator.py  điều phối toàn bộ pipeline

api/    FastAPI routers + schemas        cli/     Typer CLI
ui/     Gradio app                       config/  pydantic-settings
scripts/ download_models, benchmark, install_gpu, ocr_viewer
```

Thêm engine mới = kế thừa `BaseEngine` (`warm_up` / `process` / `release` / `is_ready`), trả về `PageResult`, rồi đăng ký trong `OCROrchestrator`.

**Lưu ý:** server phải chạy `workers=1` — `VRAMManager` là singleton theo process, nhiều worker sẽ phá vỡ cơ chế khoá GPU.

## ⚙️ Cấu hình

Qua `.env` (xem `.env.example`):

| Biến | Mặc định | |
|---|---|---|
| `PADDLE_DEVICE` | `gpu` | `gpu` hoặc `cpu` |
| `VRAM_LIMIT_MB` | `7000` | Giới hạn VRAM, chừa headroom |
| `DEFAULT_DPI` | `300` | DPI khi render PDF |
| `MAX_PDF_PAGES` | `100` | Trần số trang |
| `OPENOCR_MODE` | `server` | `mobile` (nhanh) / `server` (chính xác) |
| `POPPLER_PATH` | — | Windows; để trống thì fallback sang PyMuPDF |

## 🧪 Test

```bash
pytest                      # toàn bộ
pytest -m "not gpu"         # bỏ qua test cần CUDA
python scripts/benchmark.py # so sánh tốc độ/độ chính xác giữa các engine
```

## 🛠 Stack

PaddleOCR · Chandra OCR 2 · EasyOCR · OpenOCR · Vintern-1B · Marker/Surya · PyTorch · FastAPI · Gradio · Typer · Pydantic · OpenCV · PyMuPDF

---

## 📜 License

Mã nguồn OCRdoc phát hành theo giấy phép **[Apache License 2.0](LICENSE)**.

### Giấy phép của các thành phần bên thứ ba

Phần lớn dependency đều dùng giấy phép permissive (Apache-2.0 / MIT / BSD), nhưng có hai ngoại lệ bạn cần lưu ý khi tự deploy:

| Thành phần | Giấy phép | Ảnh hưởng |
|---|---|---|
| **PyMuPDF** (`pymupdf`) | **AGPL-3.0** hoặc giấy phép thương mại từ Artifex | Dùng làm PDF renderer dự phòng. AGPL ràng buộc cả khi **phục vụ qua mạng** — nếu bạn chạy OCRdoc thành dịch vụ web công khai, bạn phải mở mã nguồn toàn bộ dịch vụ, **hoặc** mua giấy phép thương mại từ Artifex, **hoặc** gỡ `pymupdf` và chỉ dùng `pdf2image` + Poppler. |
| **marker-pdf**, **surya-ocr** | **GPL-3.0-or-later** | Chỉ dùng cho engine `marker`, cài trong venv riêng và gọi qua subprocess nên tách biệt khỏi mã OCRdoc. Nếu bạn **không** chạy `setup_marker_venv.ps1` thì không dính GPL. |

Các engine còn lại — PaddleOCR, EasyOCR, OpenOCR, Chandra OCR 2 — đều Apache-2.0.

> ⚠️ Phần trên chỉ là tóm tắt kỹ thuật, không phải tư vấn pháp lý. Nếu dùng cho mục đích thương mại, hãy tự kiểm tra lại giấy phép của từng dependency và của trọng số model (model weights có giấy phép riêng, tách biệt với mã nguồn thư viện).
