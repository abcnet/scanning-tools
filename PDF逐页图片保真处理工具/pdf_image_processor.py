#!/usr/bin/env python3
"""Extract PDF pages and create conservatively cleaned page images.

Outputs beside input.pdf:
  input/001.jpg, 002.jpg, ... (or the actual embedded image format)
  input-processed/001.png, 002.png, ...

The processing is deterministic and non-generative. Dark text, colorful
content, edges, photographs, and drawings are protected before paper-like
background pixels are whitened.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
from PIL import Image, ImageOps, PngImagePlugin
from scipy import ndimage


PROCESSOR_VERSION = "13-complete-border-transition-cropping"


def select_pdfs_with_dialog() -> list[Path]:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.update()
        selected = filedialog.askopenfilenames(
            title="选择一个或多个需要处理的 PDF",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
        )
        root.destroy()
        return [Path(item) for item in selected]
    except Exception:
        pass

    if sys.platform == "darwin":
        script = '''
set chosenFiles to choose file with prompt "选择一个或多个需要处理的 PDF" of type {"com.adobe.pdf"} with multiple selections allowed
set outputText to ""
repeat with chosenFile in chosenFiles
    set outputText to outputText & POSIX path of chosenFile & linefeed
end repeat
return outputText
'''
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return [Path(line) for line in result.stdout.splitlines() if line]
            return []
        except Exception:
            pass

    return []


def parse_dragged_paths(raw: str) -> list[Path]:
    try:
        return [Path(item.strip('"')) for item in shlex.split(raw, posix=os.name != "nt")]
    except ValueError:
        return [Path(raw.strip('"'))]


def request_pdf_batch() -> list[Path] | None:
    """Wait for dragged paths, or optionally use the multi-file picker.

    Returning None is an explicit quit request. Cancelling the picker merely
    returns to this prompt so the long-running launcher remains available.
    """
    while True:
        print("\n请把一个或多个 PDF 拖入此窗口，然后按回车。")
        print("直接按回车：打开多选窗口    Q：退出程序")
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None

        if raw.lower() in {"q", "quit", "exit"}:
            return None
        if raw:
            return parse_dragged_paths(raw)

        selected = select_pdfs_with_dialog()
        if selected:
            return selected
        print("已取消文件选择。程序仍在运行，可继续拖入 PDF；输入 Q 才会退出。")


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


def dominant_embedded_image_data(
    document: fitz.Document, page: fitz.Page
) -> tuple[bytes, str] | None:
    """Return untouched bytes and the extension of a full-page scan image.

    Complex PDFs, masked images, or pages with no dominant raster image fall
    back to rendering, which preserves their visible page appearance.
    """
    page_area = max(page.rect.width * page.rect.height, 1.0)
    candidates: list[tuple[float, int, int, int, int]] = []

    for info in page.get_images(full=True):
        xref, smask, width, height = info[0], info[1], info[2], info[3]
        if smask:
            continue
        rects = page.get_image_rects(xref)
        coverage = max(
            (max(rect.width, 0) * max(rect.height, 0) / page_area for rect in rects),
            default=0.0,
        )
        candidates.append((coverage, width * height, xref, width, height))

    if not candidates:
        return None

    coverage, _pixel_area, xref, width, height = max(candidates)
    if coverage < 0.82:
        return None

    try:
        page_ratio = page.rect.width / max(page.rect.height, 1.0)
        image_ratio = width / max(height, 1)
        rotated_ratio = height / max(width, 1)
        if abs(rotated_ratio - page_ratio) < abs(image_ratio - page_ratio):
            # The placement transform determines whether this is 90 or 270
            # degrees. Render the visible page instead of guessing.
            return None
        extracted = document.extract_image(xref)
        payload = extracted["image"]
        extension = str(extracted.get("ext", "bin")).lower().lstrip(".")
        if extension == "jpeg":
            extension = "jpg"
        if not payload or not extension or extension == "bin":
            return None
        return payload, extension
    except Exception:
        return None


def render_page_image(page: fitz.Page, dpi: int) -> Image.Image:
    pixmap = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB, alpha=False, annots=True)
    return normalize_image(Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples))


def detect_photo_regions(
    gray: np.ndarray, saturation: np.ndarray | None = None
) -> np.ndarray:
    """Detect photographic blocks and protect their complete rectangular area.

    Text has many hard edges but little continuously varying midtone texture.
    Photos normally contain both.  Color variation is additional evidence but
    is not required, so monochrome photographs are protected as well.  Once a
    sufficiently large photographic component is found, its bounding rectangle
    is protected; this intentionally includes flat sky, walls, and other
    low-texture backgrounds inside the photograph.
    """
    gray_f = gray.astype(np.float32)
    height, width = gray.shape
    short_side = min(height, width)
    window = max(31, min(81, int(short_side * 0.018) | 1))

    local_mean = ndimage.uniform_filter(gray_f, size=window, mode="nearest")
    local_sq = ndimage.uniform_filter(gray_f * gray_f, size=window, mode="nearest")
    local_std = np.sqrt(np.maximum(local_sq - local_mean * local_mean, 0.0))
    grad_x = ndimage.sobel(gray_f, axis=1, mode="nearest")
    grad_y = ndimage.sobel(gray_f, axis=0, mode="nearest")
    gradient = np.hypot(grad_x, grad_y)

    midtone_density = ndimage.uniform_filter(
        ((gray_f > 35.0) & (gray_f < 238.0)).astype(np.float32),
        size=window,
        mode="nearest",
    )
    texture_density = ndimage.uniform_filter(
        (gradient > 20.0).astype(np.float32), size=window, mode="nearest"
    )
    evidence = (
        (midtone_density > 0.26)
        & (texture_density > 0.075)
        & (local_std > 12.0)
    )

    if saturation is not None:
        color_density = ndimage.uniform_filter(
            (saturation > 28.0).astype(np.float32), size=window, mode="nearest"
        )
        evidence |= (
            (color_density > 0.10)
            & (midtone_density > 0.14)
            & (local_std > 8.0)
        )

    # Join nearby evidence within the same photograph without allowing sparse
    # text lines to grow into a page-sized protected region.
    join_size = max(5, window // 3)
    joined = ndimage.maximum_filter(evidence, size=join_size, mode="nearest")
    joined = ndimage.minimum_filter(joined, size=join_size, mode="nearest")
    labels, count = ndimage.label(joined)
    objects = ndimage.find_objects(labels)

    photo_mask = np.zeros(gray.shape, dtype=bool)
    min_area = max(2500, int(height * width * 0.00045))
    min_dimension = max(35, int(short_side * 0.025))

    for label_id, bounds in enumerate(objects, 1):
        if bounds is None:
            continue
        ys, xs = bounds
        box_height = ys.stop - ys.start
        box_width = xs.stop - xs.start
        box_area = box_height * box_width
        component_area = int(np.count_nonzero(labels[ys, xs] == label_id))
        if (
            component_area < min_area
            or box_height < min_dimension
            or box_width < min_dimension
            or component_area / max(box_area, 1) < 0.16
        ):
            continue

        # A pale, low-texture corner may lie just outside the evidence box.
        # Use a proportional safety margin for large photo candidates instead
        # of globally joining distant components (which can absorb body text).
        padding = max(
            3,
            window // 4,
            min(120, int(min(box_height, box_width) * 0.085)),
        )
        y0 = max(0, ys.start - padding)
        y1 = min(height, ys.stop + padding)
        x0 = max(0, xs.start - padding)
        x1 = min(width, xs.stop + padding)
        photo_mask[y0:y1, x0:x1] = True

    return photo_mask


def grayscale_border_depths(gray: np.ndarray) -> tuple[int, int, int, int]:
    """Return top, bottom, left and right scanner-border depths.

    Some scanners include a 5-10 mm gray band beyond the paper plus a dark
    paper/bed boundary line. Detection starts at each outermost row/column and
    stops after several consecutive paper-like lines, so internal content and
    graphics near (but not touching) the edge are not treated as borders.
    """
    height, width = gray.shape
    center = gray[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4]
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
    top = strip_depth(gray, max_y)
    bottom = strip_depth(gray[::-1, :], max_y)
    left = strip_depth(gray.T, max_x)
    right = strip_depth(gray[:, ::-1].T, max_x)
    return top, bottom, left, right


def clean_grayscale_borders(gray: np.ndarray) -> np.ndarray:
    """Whiten detected scanner borders without changing the canvas size."""
    result = gray.copy()
    height, width = result.shape
    top, bottom, left, right = grayscale_border_depths(result)

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
    original = np.asarray(image.convert("L"), dtype=np.uint8)
    top, bottom, left, right = grayscale_border_depths(original)
    height, width = original.shape
    if top or bottom or left or right:
        original = original[top : height - bottom, left : width - right]
    photo_mask = detect_photo_regions(original)
    gray = clean_grayscale_borders(original)
    border_changed = gray != original
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

    # Chinese full stops (。), degree signs, circled marks, and similar small
    # glyphs are closed loops with a white center.  Their thin antialiased ring
    # may contain no very dark pixel, so a generic speck filter can otherwise
    # mistake them for dust.  Detect enclosed holes topologically and preserve
    # the complete small loop regardless of foreground density.
    filled_foreground = ndimage.binary_fill_holes(component_mask)
    enclosed_holes = filled_foreground & ~component_mask
    hole_boundary = (
        ndimage.binary_dilation(
            enclosed_holes, structure=np.ones((3, 3), dtype=np.uint8)
        )
        & component_mask
    )
    loop_label = np.zeros(count + 1, dtype=bool)
    loop_label[np.unique(labels[hole_boundary])] = True
    loop_label[0] = False
    typographic_loop = loop_label & (sizes >= 6) & (sizes <= 500)

    keep_label = dark_content | handwriting_stroke | large_gray | typographic_loop
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
    # Scanner-bed strips and physical page borders have higher priority than
    # photo protection.  Otherwise a photo mask reaching an outer edge would
    # restore the gray/black strip that clean_grayscale_borders just removed.
    restore_photo = photo_mask & ~border_changed
    result[restore_photo] = original[restore_photo]
    return Image.fromarray(result, "L")


def uniform_border_depths(rgb: np.ndarray) -> tuple[int, int, int, int]:
    """Return top, bottom, left and right neutral scanner-strip depths."""
    height, width = rgb.shape[:2]
    gray = (
        0.2126 * rgb[:, :, 0]
        + 0.7152 * rgb[:, :, 1]
        + 0.0722 * rgb[:, :, 2]
    )
    hsv = np.asarray(Image.fromarray(rgb, "RGB").convert("HSV"))
    saturation = hsv[:, :, 1]

    max_x = max(1, int(width * 0.04))
    max_y = max(1, int(height * 0.04))

    def strip_depth(values: np.ndarray, sats: np.ndarray, max_depth: int) -> int:
        """Find a neutral scanner strip despite a few damaged outer lines.

        JPEG ringing, a scanner lamp highlight, or a torn corner can make the
        physical first row/column much less uniform than the rest of the same
        border.  Requiring line zero to pass therefore leaves the whole strip
        behind.  Robust percentiles ignore sparse outliers, and the short
        look-ahead tolerates at most two anomalous outer lines.  A colored or
        textured header still stops the scan immediately after the gray band.
        """
        values = values[:max_depth]
        sats = sats[:max_depth]
        p10 = np.percentile(values, 10, axis=1)
        med = np.median(values, axis=1)
        p90 = np.percentile(values, 90, axis=1)
        sat_med = np.median(sats, axis=1)
        candidate = (
            ((p90 - p10) < 22.0)
            & (sat_med < 48.0)
            & (med < 246.0)
        )

        # A real removable strip must be visible and must dominate the first
        # few lines.  This prevents an isolated neutral line beside edge-touching
        # artwork from starting border removal.
        visible = med < 238.0
        probe = min(5, len(candidate))
        if probe == 0 or np.count_nonzero(candidate[:probe] & visible[:probe]) < 3:
            return 0

        last_good = -1
        misses = 0
        for index, is_candidate in enumerate(candidate):
            if is_candidate:
                last_good = index
                misses = 0
            else:
                misses += 1
                if misses >= 2:
                    break
        depth = last_good + 1
        # A scanner border can blend gradually into edge-touching colored
        # artwork over a few antialiased rows.  Do not leave that gray/color
        # transition behind: advance to the first unmistakably saturated
        # content line, while preserving that line itself.
        transition_limit = min(len(candidate), depth + 8)
        for index in range(depth, transition_limit):
            if sat_med[index] >= 160.0:
                return index
        return depth

    left = strip_depth(gray.T, saturation.T, max_x)
    right_depth = strip_depth(gray[:, ::-1].T, saturation[:, ::-1].T, max_x)
    top = strip_depth(gray, saturation, max_y)
    bottom_depth = strip_depth(gray[::-1, :], saturation[::-1, :], max_y)
    return top, bottom_depth, left, right_depth


def clean_uniform_borders(rgb: np.ndarray) -> np.ndarray:
    """Whiten detected scanner borders without changing the canvas size."""
    result = rgb.copy()
    height, width = result.shape[:2]
    top, bottom_depth, left, right_depth = uniform_border_depths(result)
    right = width - right_depth
    bottom = height - bottom_depth

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

    original_rgb = np.asarray(image, dtype=np.uint8)
    top, bottom, left, right = uniform_border_depths(original_rgb)
    height, width = original_rgb.shape[:2]
    if top or bottom or left or right:
        original_rgb = original_rgb[top : height - bottom, left : width - right]
    original_hsv = np.asarray(Image.fromarray(original_rgb, "RGB").convert("HSV"))
    original_gray = (
        0.2126 * original_rgb[:, :, 0]
        + 0.7152 * original_rgb[:, :, 1]
        + 0.0722 * original_rgb[:, :, 2]
    ).astype(np.float32)
    photo_mask = detect_photo_regions(original_gray, original_hsv[:, :, 1])

    rgb = clean_uniform_borders(original_rgb)
    border_changed = np.any(rgb != original_rgb, axis=2)

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
    protected = (
        ndimage.binary_dilation(protected_seed, iterations=2)
        | color_region
        | photo_mask
    )
    paper &= ~protected

    # Feather only outward into paper; never blur the underlying page image.
    alpha = ndimage.gaussian_filter(paper.astype(np.float32), sigma=0.8)
    alpha[protected] = 0.0
    alpha = np.clip(alpha, 0.0, 1.0)[:, :, None]
    cleaned = np.rint(rgb.astype(np.float32) + alpha * (255.0 - rgb)).astype(np.uint8)
    cleaned[protected] = rgb[protected]
    # Keep detected photos pixel-for-pixel except where the border detector
    # has positively identified a physical scanner/page-edge strip.
    restore_photo = photo_mask & ~border_changed
    cleaned[restore_photo] = original_rgb[restore_photo]
    return Image.fromarray(cleaned, "RGB")


def save_png(
    image: Image.Image,
    path: Path,
    dpi: int,
    processor_version: str | None = None,
    source_sha256: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pnginfo = None
    if processor_version is not None:
        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("pdf_image_processor_version", processor_version)
        if source_sha256 is not None:
            pnginfo.add_text("source_sha256", source_sha256)
    image.save(path, "PNG", compress_level=6, dpi=(dpi, dpi), pnginfo=pnginfo)


def valid_existing_png(
    path: Path,
    expected_version: str | None = None,
    expected_source_sha256: str | None = None,
) -> bool:
    if not path.is_file():
        return False
    try:
        with Image.open(path) as image:
            if expected_version is not None and image.info.get(
                "pdf_image_processor_version"
            ) != expected_version:
                return False
            if expected_source_sha256 is not None and image.info.get(
                "source_sha256"
            ) != expected_source_sha256:
                return False
            image.verify()
        return True
    except Exception:
        return False


ORIGINAL_IMAGE_SUFFIXES = {
    ".jpg", ".jpeg", ".jp2", ".jpx", ".png", ".tif", ".tiff", ".bmp", ".pbm"
}


def remove_other_page_images(directory: Path, stem: str, keep: Path) -> None:
    """Remove obsolete generated variants such as both 001.png and 001.jpg."""
    for candidate in directory.glob(f"{stem}.*"):
        if (
            candidate != keep
            and candidate.is_file()
            and candidate.suffix.lower() in ORIGINAL_IMAGE_SUFFIXES
        ):
            candidate.unlink()


def process_saved_page(
    task: tuple[int, Path, Path, int, str, bool]
) -> tuple[int, str]:
    index, original_path, processed_path, dpi, mode, overwrite = task

    if not valid_existing_png(original_path):
        raise RuntimeError(f"原始页面图片无效或缺失：{original_path}")

    source_sha256 = hashlib.sha256(original_path.read_bytes()).hexdigest()
    if not overwrite and valid_existing_png(
        processed_path, PROCESSOR_VERSION, source_sha256
    ):
        return index, "跳过"

    with Image.open(original_path) as opened:
        original = normalize_image(opened)
        original.load()
    processed = process_image(original, mode)
    save_png(
        processed,
        processed_path,
        dpi,
        PROCESSOR_VERSION,
        source_sha256,
    )
    return index, "处理"


def process_pdf(
    pdf_path: Path, dpi: int, mode: str, overwrite: bool, workers: int
) -> None:
    pdf_path = pdf_path.expanduser().resolve()
    if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
        raise FileNotFoundError(f"找不到 PDF：{pdf_path}")

    original_dir = pdf_path.with_suffix("")
    processed_dir = pdf_path.parent / f"{pdf_path.stem}-processed"
    original_dir.mkdir(exist_ok=True)
    processed_dir.mkdir(exist_ok=True)

    original_paths: list[Path] = []

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
        print("阶段 1/2：提取并保存全部原始页面", flush=True)

        for index, page in enumerate(document):
            stem = f"{index + 1:0{digits}d}"
            embedded = dominant_embedded_image_data(document, page)

            if embedded is not None:
                payload, extension = embedded
                original_path = original_dir / f"{stem}.{extension}"
                if overwrite or not original_path.is_file() or original_path.read_bytes() != payload:
                    original_path.write_bytes(payload)
                    original_status = f"原始流({extension})"
                else:
                    original_status = f"沿用原始流({extension})"
            else:
                original_path = original_dir / f"{stem}.png"
                if overwrite or not valid_existing_png(original_path):
                    rendered = render_page_image(page, dpi)
                    save_png(rendered, original_path, dpi)
                    original_status = "渲染PNG"
                else:
                    original_status = "沿用渲染PNG"

            remove_other_page_images(original_dir, stem, original_path)
            original_paths.append(original_path)

            print(
                f"[提取 {index + 1:0{digits}d}/{total}] 原图:{original_status}",
                flush=True,
            )

    # Do not begin cleanup until every source page has been saved.  Processing
    # from the saved images also makes the two stages independently resumable.
    print("\n全部原始页面已经保存。", flush=True)
    active_workers = min(workers, total)
    print(
        f"阶段 2/2：并行处理已保存的页面（{active_workers}个工作线程）",
        flush=True,
    )

    tasks = [
        (
            index,
            original_paths[index],
            processed_dir / f"{index + 1:0{digits}d}.png",
            dpi,
            mode,
            overwrite,
        )
        for index in range(total)
    ]
    with ThreadPoolExecutor(max_workers=active_workers) as executor:
        results = executor.map(process_saved_page, tasks)
        for index, processed_status in results:
            print(
                f"[处理 {index + 1:0{digits}d}/{total}] 处理图:{processed_status}",
                flush=True,
            )

    print("\n完成。请先抽查文字、手写内容、灰色图形和黄色插图，再导入 Epson DCP。")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="逐页提取 PDF 图片并进行非生成式、保真的纸张背景清理。"
    )
    parser.add_argument("pdf", nargs="*", type=Path, help="一个或多个需要处理的 PDF 路径")
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
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, max(1, os.cpu_count() or 1)),
        help="并行处理图片的线程数，默认自动选择且最多8个；内存不足可设为1、2或4",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="处理命令行中的文件后立即退出，不继续弹出选择窗口",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.dpi < 72 or args.dpi > 1200:
        print("DPI 必须在72到1200之间。", file=sys.stderr)
        return 2
    if args.workers < 1 or args.workers > 32:
        print("工作线程数必须在1到32之间。", file=sys.stderr)
        return 2

    pending = list(args.pdf)
    had_error = False
    first_batch = True

    while True:
        if not pending:
            if args.once and not first_batch:
                break
            requested = request_pdf_batch()
            if requested is None:
                print("\n已退出程序。")
                break
            pending = requested

        print(f"\n本批次共 {len(pending)} 个 PDF。")
        for batch_index, pdf_path in enumerate(pending, 1):
            print("\n" + "=" * 72)
            print(f"批次文件 [{batch_index}/{len(pending)}]")
            try:
                process_pdf(
                    pdf_path,
                    args.dpi,
                    args.mode,
                    args.overwrite,
                    args.workers,
                )
            except Exception as exc:
                had_error = True
                print(f"错误：{exc}", file=sys.stderr)
                print("已跳过此文件，继续处理本批次的其他 PDF。", file=sys.stderr)

        pending = []
        first_batch = False
        if args.once:
            break
        print("\n本批次处理完毕。程序继续等待下一批 PDF。")

    return 2 if had_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
