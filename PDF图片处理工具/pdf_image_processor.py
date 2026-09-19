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


PROCESSOR_VERSION = "22-contained-scanner-seam-crop"


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


def detect_physical_crop_box(image: Image.Image) -> tuple[int, int, int, int]:
    """Return a conservative rectangular crop for scanner-bed borders.

    A side is cropped only when its outer strip is clearly darker than the
    central paper and nearly every scan line finds a consistent transition to
    bright paper near that same edge.  The inward high quantile removes a
    slightly ragged/torn paper edge completely without creating a jagged crop.
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    gray = (
        0.2126 * rgb[:, :, 0]
        + 0.7152 * rgb[:, :, 1]
        + 0.0722 * rgb[:, :, 2]
    ).astype(np.float32)
    height, width = gray.shape
    if height < 200 or width < 200:
        return 0, 0, width, height

    center = gray[height // 5 : 4 * height // 5, width // 5 : 4 * width // 5]
    paper = float(np.percentile(center, 70))
    bright_cutoff = max(220.0, paper - 22.0)

    def boundary_depth(lines: np.ndarray, maximum: int) -> tuple[int, int]:
        # lines shape: scan lines x distance from the candidate outer edge.
        outer_width = max(3, min(12, maximum // 8))
        outer_by_line = np.median(lines[:, :outer_width], axis=1)
        outer_median = float(np.median(outer_by_line))
        full_edge = outer_median < paper - 16.0

        # A scanner lid/bed strip can cover only part of one physical edge.
        # Accept that case only when the darker outer samples form one long,
        # flat, contiguous run.  Sparse text or isolated marks do not qualify.
        active = outer_by_line < paper - 16.0
        padded = np.pad(active.astype(np.int8), (1, 1))
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        longest_run = int(np.max(ends - starts)) if starts.size else 0
        partial_edge = (
            float(np.mean(active)) >= 0.30
            and longest_run >= int(lines.shape[0] * 0.25)
            and float(np.std(outer_by_line[active])) <= 10.0
        )
        if not full_edge and not partial_edge:
            return 0, 0

        selected_lines = lines if full_edge else lines[active]

        smooth = ndimage.uniform_filter1d(
            selected_lines.astype(np.float32), size=7, axis=1, mode="nearest"
        )
        bright = smooth >= bright_cutoff
        sustained = ndimage.uniform_filter1d(
            bright.astype(np.float32), size=13, axis=1, mode="constant"
        ) >= 0.92

        depths: list[int] = []
        for row in sustained:
            found = np.flatnonzero(row[:maximum])
            if found.size:
                depths.append(int(found[0]))
        if len(depths) < int(selected_lines.shape[0] * 0.90):
            return 0, 0

        values = np.asarray(depths, dtype=np.float32)
        # A genuine page edge stays in a narrow band. Content or shadows near
        # an otherwise normal edge produce a much less consistent estimate.
        if float(np.percentile(values, 95) - np.percentile(values, 5)) > maximum * 0.45:
            return 0, 0
        crop_depth = min(maximum, int(np.ceil(np.percentile(values, 99.5))) + 1)
        # Nearly every scan line is still outside the paper before this
        # conservative lower quantile.  Neutral marks ending wholly before it
        # are scanner-bed seams, not writing on the page.
        trusted_outer_depth = max(0, int(np.floor(np.percentile(values, 1))))
        return crop_depth, trusted_outer_depth

    y_margin = max(10, int(height * 0.01))
    x_margin = max(10, int(width * 0.01))
    vertical = gray[y_margin : height - y_margin, :]
    horizontal = gray[:, x_margin : width - x_margin]
    max_x = max(1, int(width * 0.14))
    max_y = max(1, int(height * 0.10))

    left, left_trusted = boundary_depth(vertical, max_x)
    right_depth, right_trusted = boundary_depth(vertical[:, ::-1], max_x)
    top, top_trusted = boundary_depth(horizontal.T, max_y)
    bottom_depth, bottom_trusted = boundary_depth(horizontal[::-1, :].T, max_y)

    def content_limited_depth(
        strip: np.ndarray,
        color_strip: np.ndarray,
        inward_axis: int,
        trusted_outer_depth: int,
    ) -> int | None:
        """Return a safe depth before dark or colored writing/artwork."""
        if strip.size == 0:
            return None
        # A scanner-bed border may itself be gray.  Ink is identified by
        # multiple connected strokes substantially darker than the paper;
        # isolated dust pixels do not veto a useful crop.
        ink_cutoff = min(170.0, paper - 55.0)
        dark_ink = strip < ink_cutoff
        color_i16 = color_strip.astype(np.int16)
        chroma = np.max(color_i16, axis=2) - np.min(color_i16, axis=2)
        # Red/blue/green annotations can be visually strong but relatively
        # bright in grayscale.  Protect saturated strokes separately.
        color_ink = (chroma >= 25) & (np.min(color_i16, axis=2) <= 225)
        ink = dark_ink | color_ink
        dark_evidence = float(np.mean(dark_ink)) >= 0.001
        color_evidence = float(np.mean(color_ink)) >= 0.0003
        if not dark_evidence and not color_evidence:
            return None
        labels, count = ndimage.label(ink)
        if count == 0:
            return None
        sizes = np.bincount(labels.ravel())[1:]
        significant = np.flatnonzero(sizes >= 15) + 1
        objects = ndimage.find_objects(labels)

        # A scanner-bed seam or the shadow under a curled page edge can form
        # one dark component running almost the full width/height of the side
        # being cropped.  It is part of the physical border, not handwriting.
        # Ignore only components that both touch the outer edge and span most
        # of the perpendicular dimension; ordinary writing, page numbers and
        # colored annotations remain local and continue to limit the crop.
        cross_axis = 1 - inward_axis
        cross_extent = strip.shape[cross_axis]
        outer_tolerance = max(8, int(strip.shape[inward_axis] * 0.12))
        content_components: list[int] = []
        for label_id in significant:
            obj = objects[label_id - 1]
            inward_slice = obj[inward_axis]
            cross_slice = obj[cross_axis]
            spans_side = (cross_slice.stop - cross_slice.start) >= 0.70 * cross_extent
            touches_outer = inward_slice.start <= outer_tolerance
            component = labels[obj] == label_id
            component_color_share = float(np.mean(color_ink[obj][component]))
            neutral_seam = component_color_share <= 0.05
            # For a very shallow candidate strip, retain the old conservative
            # behavior: a full-width rule may genuinely sit at the page edge.
            # The seam exception is reserved for a substantial scanner-bed
            # band such as the 98-pixel strip in the reported failure.
            meaningful_border_depth = strip.shape[inward_axis] >= 40
            contained_outside_paper = inward_slice.stop <= max(
                0, trusted_outer_depth - 3
            )
            if neutral_seam and meaningful_border_depth and (
                (spans_side and touches_outer) or contained_outside_paper
            ):
                continue
            content_components.append(int(label_id))
        significant = np.asarray(content_components, dtype=np.int32)
        if significant.size == 0:
            return None
        # Several separate strokes identify ordinary text.  A single large
        # connected component also matters: faint pencil notes can merge into
        # only one to three detectable components at the strict dark cutoff.
        content_sizes = sizes[significant - 1]
        dark_strokes = dark_evidence and (
            significant.size >= 4 or bool(np.any(content_sizes >= 200))
        )
        color_stroke = color_evidence and bool(np.any(content_sizes >= 20))
        if not dark_strokes and not color_stroke:
            return None
        first = min(objects[label_id - 1][inward_axis].start for label_id in significant)
        return max(0, int(first) - 4)

    # Remove the pure scanner-bed portion, then stop just before page content.
    # Far-side strips are reversed so index zero is always the outer edge.
    if left:
        limit = content_limited_depth(
            gray[:, :left], rgb[:, :left], inward_axis=1,
            trusted_outer_depth=left_trusted,
        )
        if limit is not None:
            left = min(left, limit)
    if right_depth:
        limit = content_limited_depth(
            gray[:, width - right_depth :][:, ::-1],
            rgb[:, width - right_depth :][:, ::-1],
            inward_axis=1,
            trusted_outer_depth=right_trusted,
        )
        if limit is not None:
            right_depth = min(right_depth, limit)
    if top:
        limit = content_limited_depth(
            gray[:top, :], rgb[:top, :], inward_axis=0,
            trusted_outer_depth=top_trusted,
        )
        if limit is not None:
            top = min(top, limit)
    if bottom_depth:
        limit = content_limited_depth(
            gray[height - bottom_depth :, :][::-1, :],
            rgb[height - bottom_depth :, :][::-1, :],
            inward_axis=0,
            trusted_outer_depth=bottom_trusted,
        )
        if limit is not None:
            bottom_depth = min(bottom_depth, limit)

    right = width - right_depth
    bottom = height - bottom_depth

    if right - left < width * 0.70 or bottom - top < height * 0.75:
        return 0, 0, width, height
    return left, top, right, bottom


def crop_physical_scanner_borders(image: Image.Image) -> Image.Image:
    box = detect_physical_crop_box(image)
    if box == (0, 0, image.width, image.height):
        return image
    return image.crop(box)


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


def detect_visual_layout_regions(
    gray: np.ndarray, saturation: np.ndarray | None = None
) -> np.ndarray:
    """Detect photographs and structured layout panels as complete rectangles.

    The same midtone/texture evidence that identifies photographs also appears
    in designed text cards: avatars or icons, several text lines, dotted rules,
    and a continuous pale rectangular fill.  Both are intentional page layout,
    not paper dirt.  Once a sufficiently large visual component is found, its
    complete bounding rectangle is protected.  This deliberately retains flat
    sky and walls inside photos as well as gray/beige fills behind text blocks,
    preventing cleanup from cutting white holes through a designed panel.
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

    layout_mask = np.zeros(gray.shape, dtype=bool)
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
        layout_mask[y0:y1, x0:x1] = True

    return layout_mask


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
    original = np.asarray(image.convert("L"), dtype=np.uint8)
    layout_mask = detect_visual_layout_regions(original)
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
    # Physical scanner borders have already been cropped before this stage.
    # A confirmed layout region must now be restored unconditionally; later
    # pixel cleanup may never punch holes through an intentional background.
    restore_layout = layout_mask
    result[restore_layout] = original[restore_layout]
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


def detect_aged_paper_stains(
    rgb: np.ndarray, gray: np.ndarray, saturation: np.ndarray
) -> tuple[np.ndarray, bool]:
    """Find widespread yellow-brown aging on otherwise document-like pages.

    A few yellow objects must not switch on this cleanup.  The rule therefore
    activates only when warm paper-colored pixels cover a substantial part of
    the page and dark content occupies a document-like minority.  The returned
    mask deliberately includes textured foxing and broad edge stains; those
    are exactly the regions that the normal flat-paper test cannot remove.
    """
    work = rgb.astype(np.int16)
    red = work[:, :, 0]
    green = work[:, :, 1]
    blue = work[:, :, 2]

    warm = (
        (red - green >= 3)
        & (green - blue >= 3)
        & (red - blue >= 12)
        & (saturation >= 8)
        & (gray >= 95.0)
    )
    warm_share = float(np.mean(warm))
    dark_share = float(np.mean(gray < 175.0))
    active = warm_share >= 0.12 and dark_share <= 0.16
    if not active:
        return np.zeros(gray.shape, dtype=bool), False

    # Include the pale fringe around brown foxing without crossing into
    # neutral gray/black document strokes.
    pale_warm = (
        (red - green >= 2)
        & (green - blue >= 2)
        & (red - blue >= 8)
        & (saturation >= 5)
        & (gray >= 105.0)
    )
    stain = ndimage.binary_dilation(warm, iterations=1) & pale_warm
    stain |= warm
    return stain, True


def process_image(image: Image.Image, mode: str) -> Image.Image:
    """Whiten high-confidence paper background without generating details."""
    if mode == "border-only":
        return crop_physical_scanner_borders(image)
    image = normalize_image(image)
    image = crop_physical_scanner_borders(image)
    if image.mode == "L":
        return process_grayscale(image, mode)

    original_rgb = np.asarray(image, dtype=np.uint8)
    original_hsv = np.asarray(Image.fromarray(original_rgb, "RGB").convert("HSV"))
    original_gray = (
        0.2126 * original_rgb[:, :, 0]
        + 0.7152 * original_rgb[:, :, 1]
        + 0.0722 * original_rgb[:, :, 2]
    ).astype(np.float32)
    layout_mask = detect_visual_layout_regions(
        original_gray, original_hsv[:, :, 1]
    )

    rgb = clean_uniform_borders(original_rgb)
    border_changed = np.any(rgb != original_rgb, axis=2)

    hsv = np.asarray(Image.fromarray(rgb, "RGB").convert("HSV"))
    saturation = hsv[:, :, 1].astype(np.float32)
    gray = (
        0.2126 * rgb[:, :, 0]
        + 0.7152 * rgb[:, :, 1]
        + 0.0722 * rgb[:, :, 2]
    ).astype(np.float32)

    # Decide whether this is an aged page from the untouched scan.  Running
    # this after border cleanup can push a borderline page just below the
    # activation threshold and incorrectly disable broad-stain removal.
    stain_mask, aged_page = detect_aged_paper_stains(
        original_rgb, original_gray, original_hsv[:, :, 1].astype(np.float32)
    )
    if aged_page and np.any(layout_mask):
        # Foxing can form one textured, page-sized false "photograph".  Drop
        # only huge edge-touching photo components; genuine smaller embedded
        # photographs remain protected.
        labels, count = ndimage.label(layout_mask)
        sizes = np.bincount(labels.ravel(), minlength=count + 1)
        edge_labels = np.unique(
            np.concatenate((labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]))
        )
        false_photo = np.zeros(count + 1, dtype=bool)
        false_photo[edge_labels] = sizes[edge_labels] >= int(gray.size * 0.30)
        stain_overlap = ndimage.sum(stain_mask, labels, range(count + 1))
        false_photo |= stain_overlap >= np.maximum(1, sizes * 0.12)

        # Aged beige/gray text panels overlap the warm-stain mask by design.
        # Exempt compact, wide rectangles containing substantial dark text;
        # irregular page-edge foxing lacks this geometry/content combination.
        objects = ndimage.find_objects(labels)
        structured_panel = np.zeros(count + 1, dtype=bool)
        for label_id, bounds in enumerate(objects, 1):
            if bounds is None:
                continue
            ys, xs = bounds
            box_height = ys.stop - ys.start
            box_width = xs.stop - xs.start
            box_area = box_height * box_width
            area_share = box_area / max(gray.size, 1)
            aspect = box_width / max(box_height, 1)
            fill = sizes[label_id] / max(box_area, 1)
            dark_share = float(np.mean(gray[ys, xs] < 190.0))
            touches_edge = (
                ys.start == 0
                or xs.start == 0
                or ys.stop == gray.shape[0]
                or xs.stop == gray.shape[1]
            )
            if (
                not touches_edge
                and 0.005 <= area_share <= 0.20
                and aspect >= 2.5
                and fill >= 0.70
                and 0.04 <= dark_share <= 0.40
            ):
                structured_panel[label_id] = True
        false_photo &= ~structured_panel
        false_photo[0] = False
        layout_mask &= ~false_photo[labels]

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
    if aged_page:
        color_region &= ~ndimage.binary_dilation(stain_mask, iterations=2)

    # Protect original text strokes, gray graphics, colored content and their
    # immediate antialiased edges. Protected pixels are copied unchanged.
    protected_seed = (gray < 205.0) | (saturation > 70.0) | (gradient > 28.0)
    if aged_page:
        # Dark neutral cores anchor printed text, dotted rules and gray logos.
        # Preserve their antialiased surroundings even where JPEG blending
        # gives an edge pixel a slight warm cast.  Warm stain fragments alone
        # are not allowed to become protected content.
        document_core = gray < 205.0
        non_warm_color = (saturation > 70.0) & ~stain_mask
        document_content = ndimage.binary_dilation(
            document_core | non_warm_color, iterations=2
        )
        protected_seed &= ~(stain_mask & (gray >= 205.0))
    protected = (
        ndimage.binary_dilation(protected_seed, iterations=2)
        | color_region
        | layout_mask
    )
    if aged_page:
        protected |= document_content
    paper &= ~protected

    if aged_page:
        # Widespread foxing is paper damage even when it is textured or fairly
        # dark, so it bypasses the ordinary flat-background requirement.
        paper |= stain_mask & ~protected

    # Feather only outward into paper; never blur the underlying page image.
    alpha = ndimage.gaussian_filter(paper.astype(np.float32), sigma=0.8)
    alpha[protected] = 0.0
    alpha = np.clip(alpha, 0.0, 1.0)[:, :, None]
    cleaned = np.rint(rgb.astype(np.float32) + alpha * (255.0 - rgb)).astype(np.uint8)
    cleaned[protected] = rgb[protected]
    # Keep detected photos pixel-for-pixel except where the border detector
    # has positively identified a physical scanner/page-edge strip.
    # Cropping owns physical-border removal. Confirmed layout has the final
    # word over whitening and stain cleanup, including its complete backdrop.
    restore_layout = layout_mask & ~border_changed
    cleaned[restore_layout] = original_rgb[restore_layout]
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
    output_version = (
        f"{PROCESSOR_VERSION}-border-only"
        if mode == "border-only"
        else PROCESSOR_VERSION
    )
    if not overwrite and valid_existing_png(
        processed_path, output_version, source_sha256
    ):
        return index, "跳过"

    with Image.open(original_path) as opened:
        if mode == "border-only":
            original = ImageOps.exif_transpose(opened)
        else:
            original = normalize_image(opened)
        original.load()
    processed = process_image(original, mode)
    save_png(
        processed,
        processed_path,
        dpi,
        output_version,
        source_sha256,
    )
    return index, "处理"


def process_pdf(
    pdf_path: Path,
    dpi: int,
    mode: str,
    overwrite: bool,
    workers: int,
    extract_only: bool = False,
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
        if extract_only:
            print("模式：仅提取原图，不处理图片\n")
            print("提取并保存原始页面", flush=True)
        else:
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

    if extract_only:
        print("\n完成：只提取原图；-processed 目录保持不处理。", flush=True)
        return

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
        choices=("safe", "strong", "border-only"),
        default="strong",
        help="strong清理灰斑更彻底；safe更保守；border-only只裁边，默认strong",
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
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="只提取PDF逐页原图并创建-processed目录，不处理任何图片",
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
                    args.extract_only,
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
