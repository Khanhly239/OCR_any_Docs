from __future__ import annotations

import gc
import time
from html.parser import HTMLParser
from typing import ClassVar

import numpy as np
from loguru import logger

from ocrdoc.core.models import BoundingBox, PageResult, Table, TableCell, TextBlock
from ocrdoc.core.vram_manager import VRAMManager
from ocrdoc.engines.base import BaseEngine
from config.settings import Settings


class _TableHTMLParser(HTMLParser):
    """Parse HTML table từ PP-StructureV3 thành list[list[str]]."""

    def __init__(self):
        super().__init__()
        self.rows: list[list[str]] = []
        self._current_row: list[str] = []
        self._current_cell: str = ""
        self._in_cell = False
        self._current_attrs: dict = {}

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._current_row = []
        elif tag in ("td", "th"):
            self._in_cell = True
            self._current_cell = ""
            self._current_attrs = dict(attrs)

    def handle_endtag(self, tag):
        if tag == "tr":
            if self._current_row:
                self.rows.append(self._current_row)
        elif tag in ("td", "th"):
            self._in_cell = False
            self._current_row.append(self._current_cell.strip())

    def handle_data(self, data):
        if self._in_cell:
            self._current_cell += data


class PaddleEngine(BaseEngine):
    name: ClassVar[str] = "paddle"

    def __init__(self, settings: Settings, vram_manager: VRAMManager):
        self._settings = settings
        self._vram = vram_manager
        self._ocr = None
        self._structure = None
        self._ready = False

    # PaddleOCR 2.9.x bỏ qua model_storage_directory, lưu vào ~/.paddleocr/whl/
    # Vi rec model cần được đặt trong thư mục PaddleOCR thực sự tìm kiếm.
    _VI_REC_URLS: ClassVar[list[tuple[str, str]]] = [
        # v4 có thể không tồn tại cho multilingual, thử v3 trước
        ("vi_PP-OCRv3_rec_infer", "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/vi_PP-OCRv3_rec_infer.tar"),
        ("vi_PP-OCRv4_rec_infer", "https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/vi_PP-OCRv4_rec_infer.tar"),
    ]

    @staticmethod
    def _paddle_cache_rec_dir() -> "Path":
        """Trả về thư mục PaddleOCR thực sự lưu rec model: ~/.paddleocr/whl/rec/latin/"""
        from pathlib import Path as _P
        return _P.home() / ".paddleocr" / "whl" / "rec" / "latin"

    def _build_vi_rec_kwargs(self, model_dir: str, paddleocr_pkg) -> dict:
        """
        Tìm vi_PP-OCRv3/v4 rec model và trả về kwargs để override PaddleOCR.
        PaddleOCR 2.9.x map lang='vi' → latin_PP-OCRv3 (không có dấu tiếng Việt).
        Giải pháp: tải vi_PP-OCRv3_rec_infer vào ~/.paddleocr/whl/rec/latin/
        (đúng chỗ PaddleOCR tìm), rồi override rec_model_dir + rec_char_dict_path.
        """
        from pathlib import Path as _P

        vi_dict = _P(paddleocr_pkg.__file__).parent / "ppocr" / "utils" / "dict" / "vi_dict.txt"
        if not vi_dict.exists():
            logger.warning("  vi_dict.txt không tìm thấy — không thể dùng vi rec model")
            return {}

        # Thư mục PaddleOCR thực sự tìm models (bỏ qua model_storage_directory)
        paddle_cache = self._paddle_cache_rec_dir()
        paddle_cache.mkdir(parents=True, exist_ok=True)

        # Thư mục dự phòng (models/paddle/ trong project)
        project_base = _P(model_dir).resolve()
        project_base.mkdir(parents=True, exist_ok=True)

        # 1. Tìm model đã có — ưu tiên: custom → project → paddle cache
        candidates = []
        if self._settings.vi_rec_model_dir:
            candidates.append(_P(self._settings.vi_rec_model_dir).resolve())
        for name, _ in self._VI_REC_URLS:
            candidates.append(project_base / name)
            candidates.append(paddle_cache / name)

        for path in candidates:
            if path.exists() and any(path.iterdir()):
                logger.info(f"  Vietnamese rec model: {path}")
                return {
                    "rec_model_dir": str(path),
                    "rec_char_dict_path": str(vi_dict),
                }

        # 2. Chưa có → tải về vào paddle cache (đúng chỗ PaddleOCR tìm)
        logger.info(
            "  vi_PP-OCRv3 chưa có.  Đang tải về (~30-60s, chỉ lần đầu)..."
        )
        downloaded = self._download_vi_rec(paddle_cache)
        if downloaded:
            return {
                "rec_model_dir": str(downloaded),
                "rec_char_dict_path": str(vi_dict),
            }

        # 3. Tải thất bại → cảnh báo rõ ràng
        logger.error(
            "  KHÔNG tải được vi rec model → PaddleOCR dùng latin_PP-OCRv3\n"
            "  → Output SẼ THIẾU DẤU tiếng Việt!\n"
            "  Kiểm tra mạng hoặc chạy thủ công:\n"
            "    python scripts/download_models.py"
        )
        return {}

    def _download_vi_rec(self, dest_dir) -> "Path | None":
        """
        Tải vi rec model dùng maybe_download() của PaddleOCR.
        maybe_download extract trực tiếp inference.pdmodel/.pdiparams vào out_dir
        (không phải subdir) — đây là format PaddleOCR expect khi check 'inference.pdmodel'.
        """
        from pathlib import Path as _P
        try:
            from paddleocr.paddleocr import maybe_download
        except ImportError:
            logger.warning("    Không import được paddleocr.maybe_download")
            return None

        dest_dir = _P(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        for model_name, url in self._VI_REC_URLS:
            out_dir = dest_dir / model_name
            out_dir.mkdir(parents=True, exist_ok=True)
            try:
                logger.info(f"    Đang tải {model_name} ({url}) ...")
                maybe_download(str(out_dir), url)
                # maybe_download extract inference.pdmodel trực tiếp vào out_dir
                if (out_dir / "inference.pdmodel").exists():
                    logger.success(f"    {model_name} sẵn sàng tại {out_dir}")
                    return out_dir
                logger.warning(f"    {model_name}: inference.pdmodel không thấy sau extract")
            except Exception as exc:
                logger.warning(f"    {model_name} thất bại: {exc}")
                # Dọn dẹp thư mục rỗng để tránh PaddleOCR nghĩ model đã có
                try:
                    if out_dir.exists() and not any(out_dir.iterdir()):
                        out_dir.rmdir()
                except Exception:
                    pass

        return None

    def warm_up(self) -> None:
        if self._ready:
            return
        logger.info("PaddleEngine: đang load models...")
        self._vram.free_vram()

        use_gpu = self._settings.paddle_device.lower() == "gpu"
        model_dir = self._settings.resolve_paddle_model_dir()

        from paddleocr import PaddleOCR, PPStructure
        import paddleocr as _poc

        # Ưu tiên: vi_PP-OCRv4 → vi_PP-OCRv3 → latin (fallback)
        rec_kwargs = self._build_vi_rec_kwargs(model_dir, _poc)

        self._ocr = PaddleOCR(
            use_gpu=use_gpu,
            lang="vi",
            use_angle_cls=True,
            det_db_thresh=0.3,
            det_db_unclip_ratio=1.6,
            rec_batch_num=6,
            det_limit_side_len=960,
            model_storage_directory=model_dir,
            show_log=False,
            **rec_kwargs,
        )

        self._structure = PPStructure(
            table=True,
            layout=True,
            lang="en",  # layout model chỉ hỗ trợ 'en'/'ch'; text recognition vẫn dùng 'vi' qua self._ocr
            use_gpu=use_gpu,
            image_orientation=False,  # paddleclas không bắt buộc cài
            recovery=True,
            return_word_box=False,
            model_storage_directory=model_dir,
            show_log=False,
        )

        self._ready = True
        logger.success("PaddleEngine: models loaded")

    def process(
        self,
        image: np.ndarray,
        lang: list[str] | None = None,
        doc_type: str = "structured",
    ) -> PageResult:
        if not self._ready:
            self.warm_up()

        h, w = image.shape[:2]
        t0 = time.perf_counter()

        # Unstructured: bỏ qua PPStructure (nặng, chậm), chạy PaddleOCR thuần
        if doc_type == "unstructured":
            logger.info("PaddleEngine: unstructured → plain OCR (bỏ qua PPStructure)")
            text_blocks = self._fallback_ocr(image, w, h)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return PageResult(
                page_number=1,
                width=w,
                height=h,
                text_blocks=text_blocks,
                tables=[],
                engine_used=self.name,
                processing_time_ms=elapsed_ms,
            )

        # Structured: dùng PPStructure để detect layout + tables
        text_blocks: list[TextBlock] = []
        tables: list[Table] = []

        try:
            result = self._structure(image)
        except Exception as e:
            logger.error(f"PP-StructureV3 lỗi: {e}, fallback sang PaddleOCR thuần")
            result = []

        for region in result:
            region_type = region.get("type", "").lower()
            bbox_raw = region.get("bbox", [0, 0, w, h])
            bbox = BoundingBox.from_xyxy(*[int(v) for v in bbox_raw])

            if region_type == "table":
                table = self._parse_table_region(region, bbox)
                if table:
                    tables.append(table)
            else:
                blocks = self._ocr_region_vi(image, bbox, region_type)
                text_blocks.extend(blocks)

        if not text_blocks and not tables:
            text_blocks = self._fallback_ocr(image, w, h)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return PageResult(
            page_number=1,
            width=w,
            height=h,
            text_blocks=text_blocks,
            tables=tables,
            engine_used=self.name,
            processing_time_ms=elapsed_ms,
        )

    def _parse_table_region(self, region: dict, bbox: BoundingBox) -> Table | None:
        try:
            res = region.get("res", {})
            html_str = res.get("html", "") if isinstance(res, dict) else ""
            if not html_str:
                return None

            parser = _TableHTMLParser()
            parser.feed(html_str)
            rows = parser.rows
            if not rows:
                return None

            n_rows = len(rows)
            n_cols = max(len(r) for r in rows) if rows else 0
            cells = []
            for r_idx, row in enumerate(rows):
                for c_idx, cell_text in enumerate(row):
                    cells.append(TableCell(row=r_idx, col=c_idx, text=cell_text))

            return Table(bbox=bbox, rows=n_rows, cols=n_cols, cells=cells)
        except Exception as e:
            logger.warning(f"Parse table HTML thất bại: {e}")
            return None

    def _ocr_region_vi(self, image: np.ndarray, bbox: BoundingBox, region_type: str) -> list[TextBlock]:
        """Crop vùng từ ảnh gốc, chạy PaddleOCR lang='vi' để nhận dạng tiếng Việt đúng."""
        block_type_map = {
            "title": "heading",
            "header": "header",
            "footer": "footer",
            "figure_caption": "caption",
        }
        block_type = block_type_map.get(region_type, "paragraph")

        img_h, img_w = image.shape[:2]
        x1 = max(0, bbox.x)
        y1 = max(0, bbox.y)
        x2 = min(img_w, bbox.x2)
        y2 = min(img_h, bbox.y2)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0 or min(crop.shape[:2]) < 4:
            return []

        try:
            result = self._ocr.ocr(crop, cls=True)
        except Exception as e:
            logger.warning(f"Vietnamese OCR region lỗi: {e}")
            return []

        if not result or not result[0]:
            return []

        blocks = []
        for line in result[0]:
            if not line or len(line) < 2:
                continue
            bbox_points, (text, conf) = line[0], line[1]
            if not text or not text.strip():
                continue
            rel = BoundingBox.from_paddle(bbox_points)
            abs_bbox = BoundingBox(x=x1 + rel.x, y=y1 + rel.y, w=rel.w, h=rel.h)
            blocks.append(TextBlock(
                text=text.strip(),
                confidence=float(conf),
                bbox=abs_bbox,
                block_type=block_type,
            ))
        return blocks

    def _parse_text_region(self, region: dict, bbox: BoundingBox, region_type: str) -> list[TextBlock]:
        blocks = []
        block_type_map = {
            "title": "heading",
            "header": "header",
            "footer": "footer",
            "figure_caption": "caption",
        }
        block_type = block_type_map.get(region_type, "paragraph")

        res = region.get("res", [])
        if not res:
            return blocks

        for line in res:
            if not isinstance(line, (list, tuple)) or len(line) < 2:
                continue
            bbox_points, (text, conf) = line[0], line[1]
            if not text or not text.strip():
                continue
            blocks.append(TextBlock(
                text=text.strip(),
                confidence=float(conf),
                bbox=BoundingBox.from_paddle(bbox_points),
                block_type=block_type,
            ))
        return blocks

    def _fallback_ocr(self, image: np.ndarray, w: int, h: int) -> list[TextBlock]:
        blocks = []
        result = self._ocr.ocr(image, cls=True)
        if not result or not result[0]:
            return blocks
        for line in result[0]:
            if not line:
                continue
            bbox_points, (text, conf) = line[0], line[1]
            if not text or not text.strip():
                continue
            blocks.append(TextBlock(
                text=text.strip(),
                confidence=float(conf),
                bbox=BoundingBox.from_paddle(bbox_points),
                block_type="paragraph",
            ))
        return blocks

    def release(self) -> None:
        self._ocr = None
        self._structure = None
        self._ready = False
        gc.collect()
        try:
            import paddle
            paddle.device.cuda.empty_cache()
        except Exception:
            pass
        logger.debug("PaddleEngine: released")

    @property
    def is_ready(self) -> bool:
        return self._ready
