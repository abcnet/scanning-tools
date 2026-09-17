#!/usr/bin/env python3
"""Extract PDF pages and create conservatively cleaned page images.

Outputs beside input.pdf:
  input/001.png, 002.png, ...
  input-processed/001.png, 002.png, ...

The processing is deterministic and non-generative. Dark text, colorful
content, edges, photographs, and drawings are protected before paper-like
background pixels are whitened.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
from PIL import Image, ImageOps, PngImagePlugin
from scipy import ndimage


PROCESSOR_VERSION = "5-edge-strips"


def select_pdf_with_dialog() -> Path | None:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.update()
        selected = filedialog.askopenfilename(
            title="选择需要处理的 PDF",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
        )
        root.destroy()
        return Path(selected) if selected else None
    except Exception:
        return None


def normalize_image(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    if image.mode in {"1", "L", "I", "I;16"}:
        return image.convert("L")
    if image.mode in {"RGBA", "LA"}:
        background = Image.new("RGB", image.size, "white")
        alpha = image.getchannel("A")
        background.paste(image.convert("RGB"), mask=alpha)
        return background
    rgb = image.convert("RGB")
    sample = np.asarray(rgb.resize((min(512, rgb.width), min(512, rgb.height))))
    if int(np.max(sample, axis=2).astype(np.int16).max() - np.min(sample, axis=2).astype(np.int16).min()) == 0:
        return rgb.convert("L")
    if int(np.max(np.ptp(sample.astype(np.int16), axis=2))) <= 1:
        return rgb.convert("L")
    return rgb


def dominant_embedded_image(document: fitz.Document, page: fitz.Page) -> Image.Image | None:
    """Return a full-page scan image when the page clearly contains one.

    Complex PDFs, masked images, or pages with no dominant raster image fall
    back to rendering, which preserves their visible page appearance.
    """
    page_area = max(page.rect.width * page.rect.height, 1.0)
    candidates: list[tuple[float, int, int, int]] = []

    for info in page.get_images(full=True):
        xref, smask, width, height = info[0], info[1], info[2], info[3]
        if smask:
            continue
        rects = page.get_image_rects(xref)
        coverage = max(
            (max(rect.width, 0) * max(rect.height, 0) / page_area for rect in rects),
            default=0.0,
        )
        candidates.append((coverage, width * height, xref, width))

    if not candidates:
        return None

    coverage, _pixel_area, xref, _width = max(candidates)
    if coverage < 0.82:
        return None

    try:
        payload = document.extract_image(xref)["image"]
        with Image.open(io.BytesIO(payload)) as opened:
            image = normalize_image(opened)
            image.load()

        page_ratio = page.rect.width / max(page.rect.height, 1.0)
        image_ratio = image.width / max(image.height, 1)
        rotated_ratio = image.height / max(image.width, 1)
        if abs(rotated_ratio - page_ratio) < abs(image_ratio - page_ratio):
            # The placement transform determines whether this is 90 or 270
            # degrees. Render the visible page instead of guessing.
            return None
        return image
    except Exception:
        return None


def page_to_image(document: fitz.Document, page: fitz.Page, dpi: int) -> Image.Image:
    image = dominant_embedded_image(document, page)
    if image is not None:
        return image

    pixmap = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB, alpha=False, annots=True)
    return normalize_image(Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples))


def clean_grayscale_borders(gray: np.ndarray) -> np.ndarray:
    """Whiten scanner-bed strips connected to an outer image edge.

    Some scanners include a 5-10 mm gray band beyond the paper plus a dark
    paper/bed boundary line. Detection starts at each outermost row/column and
    stops after several consecutive paper-like lines, so internal content and
    graphics near (but not touching) the edge are not treated as borders.
    """
    result = gray.copy()
    height, width = result.shape
    center = result[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4]
    paper = float(np.percentile(center, 70))
    bright_cutoff = max(210.0, paper - 17.0)

    def strip_depth(lines: np.ndarray, max_depth: int) -> int:
        medians = np.median(lines[:max_depth], axis=1)
        bright_share = np.mean(lines[:max_depth] >= bright_cutoff, axis=1)
        bad = (medians < paper - 20.0) | (bright_share < 0.65)
        # Never remove a border unless the physical outer edge itself looks
        # unlike paper. This prevents a nearby heading or illustration from
        # triggering cleanup.
        if not bool(bad[0]):
            return 0
        clean_run = 0
        for index, is_bad in enumerate(bad):
            clean_run = 0 if is_bad else clean_run + 1
            if clean_run >= 6:
                # Include a small inward safety margin for the dark paper/bed
                # boundary line that commonly precedes the uniform gray band.
                return min(max_depth, index - clean_run + 1 + 8)
        return max_depth if bool(np.mean(bad[-6:]) > 0.5) else 0

    max_y = max(1, int(height * 0.06))
    max_x = max(1, int(width * 0.06))
    top = strip_depth(result, max_y)
    bottom = strip_depth(result[::-1, :], max_y)
    left = strip_depth(result.T, max_x)
    right = strip_depth(result[:, ::-1].T, max_x)

    if top:
        result[:top, :] = 255
    if bottom:
        result[height - bottom :, :] = 255
    if left:
        result[:, :left] = 255
    if right:
        result[:, width - right :] = 255
    return result


def process_grayscale(image: Image.Image, mode: str) -> Image.Image:
    """Clean scanned paper while keeping genuine gray shapes neutral.

    A pixel is treated as document content when it belongs to a component that
    contains a genuinely dark core, or to a large continuous gray region.  A
    light isolated component is paper dirt and is lifted toward white.  The
    result deliberately remains one-channel grayscale, so downstream OCR
    software cannot introduce yellow/blue chroma speckles.
    """
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    gray = clean_grayscale_borders(gray)
    result = gray.copy()

    # Content components are built from mid/dark pixels. Text antialiasing is
    # connected to a dark core; large continuous gray artwork is kept even
    # when it has no black core.
    component_mask = gray < (224 if mode == "strong" else 218)
    labels, count = ndimage.label(component_mask, structure=np.ones((3, 3), dtype=np.uint8))
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    dark_core = ndimage.sum(gray < (105 if mode == "strong" else 115), labels, range(count + 1)) > 0
    dark_content = dark_core & (sizes >= (8 if mode == "strong" else 6))

    # Pencil and light-ink handwriting may have no near-black pixels at all.
    # A grayscale morphological black-hat response detects thin strokes by
    # comparing them with their local paper background. Preserve a component
    # when several of its pixels form such locally dark strokes. Broad stains
    # and uneven paper shading have a much weaker response and remain cleanable.
    local_paper = ndimage.grey_closing(gray, size=(15, 15), mode="nearest")
    stroke_response = local_paper.astype(np.int16) - gray.astype(np.int16)
    stroke_seed = (stroke_response >= (7 if mode == "strong" else 5)) & (gray < 246)
    stroke_pixels = ndimage.sum(stroke_seed, labels, range(count + 1))
    handwriting_stroke = (stroke_pixels >= (5 if mode == "strong" else 3)) & (
        sizes >= (10 if mode == "strong" else 7)
    )
    # On severely mottled photocopies, dirt itself consists of thousands of
    # short stroke-like fragments. Do not enable the pencil heuristic there;
    # the conservative mode still remains available when notes take priority.
    foreground_density = float(np.mean(component_mask))
    handwriting_stroke &= foreground_density < (0.035 if mode == "strong" else 0.060)
    large_gray = sizes >= (20000 if mode == "strong" else 12000)
    keep_label = dark_content | handwriting_stroke | large_gray
    keep_label[0] = False
    protected = keep_label[labels]

    # Include a narrow antialiased fringe around real content, but do not
    # expand into unrelated stains farther away.
    protected |= ndimage.binary_dilation(protected, iterations=1) & (gray < 238)

    # Once a component has neither a dark document core nor the area of a
    # genuine gray illustration, it is scanner dirt/background—not content.
    if mode == "strong":
        result[~protected] = 255
    else:
        work = gray.astype(np.float32)
        lifted = np.where(work >= 235.0, 255.0, np.minimum(255.0, work + 42.0))
        result[~protected] = np.rint(lifted[~protected]).astype(np.uint8)

    # Remove remaining tiny isolated gray flecks that have no dark core.
    flecks, fleck_count = ndimage.label(result < 190, structure=np.ones((3, 3), dtype=np.uint8))
    fleck_sizes = np.bincount(flecks.ravel(), minlength=fleck_count + 1)
    fleck_dark = ndimage.sum(result < 80, flecks, range(fleck_count + 1)) > 0
    erase_label = (fleck_sizes <= (30 if mode == "strong" else 16)) & ~fleck_dark
    erase_label[0] = False
    result[erase_label[flecks] & ~protected] = 255
    return Image.fromarray(result, "L")


def clean_uniform_borders(rgb: np.ndarray) -> np.ndarray:
    """Whiten only narrow, nearly uniform neutral strips touching page edges."""
    result = rgb.copy()
    height, width = result.shape[:2]
    gray = (
        0.2126 * result[:, :, 0]
        + 0.7152 * result[:, :, 1]
        + 0.0722 * result[:, :, 2]
    )
    hsv = np.asarray(Image.fromarray(result, "RGB").convert("HSV"))
    saturation = hsv[:, :, 1]

    max_x = max(1, int(width * 0.04))
    max_y = max(1, int(height * 0.04))

    def neutral_uniform(values: np.ndarray, sats: np.ndarray) -> bool:
        return float(np.std(values)) < 18.0 and float(np.median(sats)) < 48.0

    left = 0
    for x in range(max_x):
        if neutral_uniform(gray[:, x], saturation[:, x]):
            left = x + 1
        else:
            break

    right = width
    for x in range(width - 1, width - max_x - 1, -1):
        if neutral_uniform(gray[:, x], saturation[:, x]):
            right = x
        else:
            break

    top = 0
    for y in range(max_y):
        if neutral_uniform(gray[y, :], saturation[y, :]):
            top = y + 1
        else:
            break

    bottom = height
    for y in range(height - 1, height - max_y - 1, -1):
        if neutral_uniform(gray[y, :], saturation[y, :]):
            bottom = y
        else:
            break

    if left:
        result[:, :left] = 255
    if right < width:
        result[:, right:] = 255
    if top:
        result[:top, :] = 255
    if bottom < height:
        result[bottom:, :] = 255
    return result


def process_image(image: Image.Image, mode: str) -> Image.Image:
    """Whiten high-confidence paper background without generating details."""
    image = normalize_image(image)
    if image.mode == "L":
        return process_grayscale(image, mode)

    rgb = np.asarray(image, dtype=np.uint8)
    rgb = clean_uniform_borders(rgb)

    hsv = np.asarray(Image.fromarray(rgb, "RGB").convert("HSV"))
    saturation = hsv[:, :, 1].astype(np.float32)
    gray = (
        0.2126 * rgb[:, :, 0]
        + 0.7152 * rgb[:, :, 1]
        + 0.0722 * rgb[:, :, 2]
    ).astype(np.float32)

    # Local texture and edge density separate paper from photos and drawings.
    local_mean = ndimage.uniform_filter(gray, size=31, mode="nearest")
    local_sq_mean = ndimage.uniform_filter(gray * gray, size=31, mode="nearest")
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean * local_mean, 0.0))
    grad_x = ndimage.sobel(gray, axis=1, mode="nearest")
    grad_y = ndimage.sobel(gray, axis=0, mode="nearest")
    gradient = np.hypot(grad_x, grad_y)
    edge_density = ndimage.uniform_filter((gradient > 18).astype(np.float32), size=31)

    if mode == "strong":
        min_luma, max_sat, max_std, max_edges = 180.0, 65.0, 22.0, 0.075
    else:
        min_luma, max_sat, max_std, max_edges = 205.0, 48.0, 17.0, 0.055

    paper = (
        (gray >= min_luma)
        & (saturation <= max_sat)
        & (local_std <= max_std)
        & (edge_density <= max_edges)
    )

    # Protect whole color-rich regions, not only their most saturated pixels.
    # This keeps covers, photographs and yellow illustrations pixel-for-pixel
    # even when highlights inside them have relatively low saturation.
    color_density = ndimage.uniform_filter(
        (saturation > 45.0).astype(np.float32), size=61, mode="nearest"
    )
    color_region = ndimage.binary_dilation(color_density > 0.055, iterations=5)

    # Protect original text strokes, gray graphics, colored content and their
    # immediate antialiased edges. Protected pixels are copied unchanged.
    protected_seed = (gray < 205.0) | (saturation > 70.0) | (gradient > 28.0)
    protected = ndimage.binary_dilation(protected_seed, iterations=2) | color_region
    paper &= ~protected

    # Feather only outward into paper; never blur the underlying page image.
    alpha = ndimage.gaussian_filter(paper.astype(np.float32), sigma=0.8)
    alpha[protected] = 0.0
    alpha = np.clip(alpha, 0.0, 1.0)[:, :, None]
    cleaned = np.rint(rgb.astype(np.float32) + alpha * (255.0 - rgb)).astype(np.uint8)
    cleaned[protected] = rgb[protected]
    return Image.fromarray(cleaned, "RGB")


def save_png(
    image: Image.Image, path: Path, dpi: int, processor_version: str | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pnginfo = None
    if processor_version is not None:
        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("pdf_image_processor_version", processor_version)
    image.save(path, "PNG", compress_level=6, dpi=(dpi, dpi), pnginfo=pnginfo)


def valid_existing_png(path: Path, expected_version: str | None = None) -> bool:
    if not path.is_file():
        return False
    try:
        with Image.open(path) as image:
            if expected_version is not None and image.info.get(
                "pdf_image_processor_version"
            ) != expected_version:
                return False
            image.verify()
        return True
    except Exception:
        return False


def process_pdf(pdf_path: Path, dpi: int, mode: str, overwrite: bool) -> None:
    pdf_path = pdf_path.expanduser().resolve()
    if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
        raise FileNotFoundError(f"找不到 PDF：{pdf_path}")

    original_dir = pdf_path.with_suffix("")
    processed_dir = pdf_path.parent / f"{pdf_path.stem}-processed"
    original_dir.mkdir(exist_ok=True)
    processed_dir.mkdir(exist_ok=True)

    with fitz.open(pdf_path) as document:
        total = document.page_count
        if total == 0:
            raise RuntimeError("PDF 没有页面。")
        digits = max(3, len(str(total)))

        print(f"PDF：{pdf_path}")
        print(f"页数：{total}")
        print(f"原始图片：{original_dir}")
        print(f"处理图片：{processed_dir}")
        print(f"模式：{mode}\n")

        for index, page in enumerate(document):
            filename = f"{index + 1:0{digits}d}.png"
            original_path = original_dir / filename
            processed_path = processed_dir / filename

            if overwrite or not valid_existing_png(original_path):
                original = page_to_image(document, page, dpi)
                save_png(original, original_path, dpi)
                original_status = "提取"
            else:
                with Image.open(original_path) as opened:
                    original = normalize_image(opened)
                    original.load()
                original_status = "沿用"

            if overwrite or not valid_existing_png(processed_path, PROCESSOR_VERSION):
                processed = process_image(original, mode)
                save_png(processed, processed_path, dpi, PROCESSOR_VERSION)
                processed_status = "处理"
            else:
                processed_status = "跳过"

            print(
                f"[{index + 1:0{digits}d}/{total}] "
                f"原图:{original_status}  处理图:{processed_status}",
                flush=True,
            )

    print("\n完成。请先抽查文字、手写内容、灰色图形和黄色插图，再导入 Epson DCP。")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="逐页提取 PDF 图片并进行非生成式、保真的纸张背景清理。"
    )
    parser.add_argument("pdf", nargs="?", type=Path, help="需要处理的 PDF 路径")
    parser.add_argument("--dpi", type=int, default=300, help="复杂页面回退渲染 DPI，默认300")
    parser.add_argument(
        "--mode",
        choices=("safe", "strong"),
        default="strong",
        help="strong清理灰斑更彻底；safe更保守，默认strong",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已存在的逐页图片；默认跳过完整的现有页面，便于断点续跑",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    pdf_path = args.pdf
    if pdf_path is None:
        pdf_path = select_pdf_with_dialog()
    if pdf_path is None:
        print("未选择 PDF。也可以把 PDF 路径作为命令行参数传入。")
        return 1

    if args.dpi < 72 or args.dpi > 1200:
        print("DPI 必须在72到1200之间。", file=sys.stderr)
        return 2

    try:
        process_pdf(pdf_path, args.dpi, args.mode, args.overwrite)
        return 0
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
