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
import struct
import subprocess
import sys
import tempfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
from PIL import Image, ImageOps, PngImagePlugin
from scipy import ndimage


PROCESSOR_VERSION = "26-independent-page-crop"


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


def detect_nested_neutral_frame_box(
    rgb: np.ndarray,
) -> tuple[int, int, int, int] | None:
    """Detect a white/gray scanner frame around a full-bleed colorful page.

    This handles nested borders where a narrow white outer rim hides a much
    wider gray scanner-bed band.  It is deliberately limited to colorful,
    textured pages with a strong, nearly full-length rectangular transition;
    ordinary white document margins must not be cropped to their text block.
    """
    height, width = rgb.shape[:2]
    if height < 200 or width < 200:
        return None

    gray = (
        0.2126 * rgb[:, :, 0]
        + 0.7152 * rgb[:, :, 1]
        + 0.0722 * rgb[:, :, 2]
    ).astype(np.float32)
    y_trim = max(5, int(height * 0.03))
    x_trim = max(5, int(width * 0.03))
    column_texture = np.std(gray[y_trim : height - y_trim, :], axis=0)
    row_texture = np.std(gray[:, x_trim : width - x_trim], axis=1)

    def candidate(profile: np.ndarray, maximum: int) -> int:
        smooth = ndimage.uniform_filter1d(
            profile.astype(np.float32), size=5, mode="nearest"
        )
        textured = smooth >= 12.0
        sustained = ndimage.uniform_filter1d(
            textured.astype(np.float32), size=11, mode="constant"
        ) >= 0.82
        found = np.flatnonzero(sustained[5:maximum])
        if not found.size:
            return 0
        depth = int(found[0] + 5)
        # The region before the page must itself be a low-texture scanner
        # frame, not ordinary page content followed by a photograph.
        outer = smooth[max(0, int(depth * 0.08)) : max(1, depth - 5)]
        if outer.size < 5 or float(np.median(outer)) > 7.0:
            return 0
        return depth

    max_x = max(1, int(width * 0.28))
    max_y = max(1, int(height * 0.18))
    left = candidate(column_texture, max_x)
    right_depth = candidate(column_texture[::-1], max_x)
    top = candidate(row_texture, max_y)
    bottom_depth = candidate(row_texture[::-1], max_y)

    def strong_transition(position: int, axis: int, reversed_side: bool) -> bool:
        if not position:
            return False
        coordinate = (width - position if axis == 1 else height - position) if reversed_side else position
        limit = width if axis == 1 else height
        if coordinate < 5 or coordinate > limit - 5:
            return False
        if axis == 1:
            before = rgb[:, coordinate - 5 : coordinate].astype(np.float32).mean(axis=1)
            after = rgb[:, coordinate : coordinate + 5].astype(np.float32).mean(axis=1)
        else:
            before = rgb[coordinate - 5 : coordinate].astype(np.float32).mean(axis=0)
            after = rgb[coordinate : coordinate + 5].astype(np.float32).mean(axis=0)
        difference = np.linalg.norm(after - before, axis=1)
        return float(np.mean(difference >= 25.0)) >= 0.62

    def nested_transition_depth(axis: int, reversed_side: bool, maximum: int) -> int:
        """Find the inner edge of a neutral frame, even if the page starts flat."""
        work = rgb[:, ::-1] if axis == 1 and reversed_side else rgb
        if axis == 0:
            work = work[::-1, :] if reversed_side else work
        smooth = ndimage.uniform_filter1d(
            work.astype(np.float32), size=5, axis=axis, mode="nearest"
        )
        if axis == 1:
            delta = np.linalg.norm(smooth[:, 6:] - smooth[:, :-6], axis=2)
            shares = np.mean(delta >= 25.0, axis=0)
        else:
            delta = np.linalg.norm(smooth[6:] - smooth[:-6], axis=2)
            shares = np.mean(delta >= 25.0, axis=1)
        possible = np.flatnonzero(shares[: max(0, maximum - 6)] >= 0.62) + 3
        if not possible.size:
            return 0
        work_i16 = work.astype(np.int16)
        chroma = np.max(work_i16, axis=2) - np.min(work_i16, axis=2)
        # Work inward from the deepest transition.  Internal design rules are
        # rejected because the entire region before them is no longer a
        # low-chroma white/gray scanner frame.
        for depth in possible[::-1]:
            outer_start = max(0, int(depth * 0.05))
            outer_stop = max(outer_start + 1, int(depth) - 4)
            outer = (
                chroma[:, outer_start:outer_stop]
                if axis == 1
                else chroma[outer_start:outer_stop, :]
            )
            if outer.size and float(np.mean(outer <= 15)) >= 0.90:
                return int(depth)
        return 0

    if left and not strong_transition(left, 1, False):
        left = 0
    if right_depth and not strong_transition(right_depth, 1, True):
        right_depth = 0
    if top and not strong_transition(top, 0, False):
        top = 0
    if bottom_depth and not strong_transition(bottom_depth, 0, True):
        bottom_depth = 0

    if not left:
        left = nested_transition_depth(1, False, max_x)
    if not right_depth:
        right_depth = nested_transition_depth(1, True, max_x)
    if not top:
        top = nested_transition_depth(0, False, max_y)
    if not bottom_depth:
        bottom_depth = nested_transition_depth(0, True, max_y)

    hsv = np.asarray(Image.fromarray(rgb, "RGB").convert("HSV"))
    central_sat = hsv[
        height // 8 : 7 * height // 8,
        width // 8 : 7 * width // 8,
        1,
    ]
    colorful_page = float(np.mean(central_sat >= 45)) >= 0.22
    if not colorful_page:
        outside_regions = []
        if left:
            outside_regions.append(gray[:, :left])
        if right_depth:
            outside_regions.append(gray[:, width - right_depth :])
        if top:
            outside_regions.append(gray[:top, :])
        if bottom_depth:
            outside_regions.append(gray[height - bottom_depth :, :])
        dark_frame_sides = sum(
            float(np.mean(region < 200.0)) >= 0.25
            for region in outside_regions
            if region.size
        )
        if dark_frame_sides < 2:
            return None

    detected_sides = sum(bool(value) for value in (left, right_depth, top, bottom_depth))
    if detected_sides < 3:
        return None
    right = width - right_depth
    bottom = height - bottom_depth
    if right - left < width * 0.60 or bottom - top < height * 0.65:
        return None
    return left, top, right, bottom


def detect_single_sided_neutral_strip_box(
    rgb: np.ndarray,
) -> tuple[int, int, int, int]:
    """Detect an isolated gray/black scanner-bed strip on any one side.

    Some Epson pages have a scanner-bed band on only one physical edge.  The
    older rectangular-frame detector deliberately required three sides and
    therefore rejected these obvious one-sided cases.  A single side is safe
    to remove only when its whole outer region is neutral and low-texture and
    its inner edge is a strong transition along most of the page.
    """
    height, width = rgb.shape[:2]
    if height < 200 or width < 200:
        return 0, 0, width, height

    work = rgb.astype(np.float32)
    work_i16 = rgb.astype(np.int16)
    base_chroma = np.max(work_i16, axis=2) - np.min(work_i16, axis=2)
    base_gray = (
        0.2126 * work[:, :, 0]
        + 0.7152 * work[:, :, 1]
        + 0.0722 * work[:, :, 2]
    )

    def transition_profile(axis: int) -> np.ndarray:
        smooth = ndimage.uniform_filter1d(work, size=5, axis=axis, mode="nearest")
        if axis == 1:
            delta = np.linalg.norm(smooth[:, 6:] - smooth[:, :-6], axis=2)
            return np.mean(delta >= 20.0, axis=0)
        delta = np.linalg.norm(smooth[6:] - smooth[:-6], axis=2)
        return np.mean(delta >= 20.0, axis=1)

    def depth_for_side(
        axis: int,
        reverse: bool,
        maximum: int,
        transition_share: np.ndarray,
    ) -> int:
        candidates = np.flatnonzero(
            transition_share[: max(0, maximum - 6)] >= 0.50
        ) + 3
        if not candidates.size:
            return 0

        chroma = np.flip(base_chroma, axis=axis) if reverse else base_chroma
        gray = np.flip(base_gray, axis=axis) if reverse else base_gray

        # Prefer the deepest valid transition.  This removes a white outer
        # hairline together with the gray bed rather than stopping between
        # those two scanner artifacts.
        for depth in candidates[::-1]:
            depth = int(depth)
            if depth < 6:
                continue
            outer_stop = max(2, depth - 5)
            outer_chroma = (
                chroma[:, :outer_stop] if axis == 1 else chroma[:outer_stop, :]
            )
            outer_gray = gray[:, :outer_stop] if axis == 1 else gray[:outer_stop, :]
            if not outer_gray.size:
                continue

            neutral_share = float(np.mean(outer_chroma <= 15))
            line_texture = (
                np.std(outer_gray, axis=0)
                if axis == 1
                else np.std(outer_gray, axis=1)
            )
            low_texture_share = float(np.mean(line_texture <= 10.0))
            # Scanner-bed illumination may change gradually along a long gray
            # strip, making its full-column standard deviation look large.
            # Local gradients still remain almost entirely flat, unlike text,
            # photographs, or a designed sidebar.
            gradient_inward = np.abs(np.diff(outer_gray, axis=axis))
            gradient_cross = np.abs(np.diff(outer_gray, axis=1 - axis))
            smooth_gradient_share = min(
                float(np.mean(gradient_inward < 12.0))
                if gradient_inward.size else 1.0,
                float(np.mean(gradient_cross < 12.0))
                if gradient_cross.size else 1.0,
            )
            outer_level = float(np.median(outer_gray))

            inner_start = min(
                (width if axis == 1 else height) - 1,
                depth + 8,
            )
            inner_stop = min(
                width if axis == 1 else height,
                inner_start + max(12, depth // 3),
            )
            inner_gray = (
                gray[:, inner_start:inner_stop]
                if axis == 1
                else gray[inner_start:inner_stop, :]
            )
            if not inner_gray.size:
                continue
            inner_level = float(np.median(inner_gray))
            edge_share = float(transition_share[depth - 3])
            axis_size = width if axis == 1 else height
            deep_relaxed_band = depth >= max(80, int(axis_size * 0.05))
            dark_shallow_band = (
                outer_level <= 190.0
                and edge_share >= 0.60
                and smooth_gradient_share >= 0.96
            )
            texture_evidence = (
                (edge_share >= 0.72 and low_texture_share >= 0.82)
                or (
                    deep_relaxed_band
                    and edge_share >= 0.50
                    and smooth_gradient_share >= 0.96
                )
                or dark_shallow_band
            )

            # A gray/black scanner band is materially darker than the page
            # just inside it.  The absolute ceiling prevents an ordinary
            # white document margin and text-column boundary from qualifying.
            if (
                neutral_share >= 0.90
                and texture_evidence
                and outer_level <= 238.0
                and inner_level - outer_level >= 12.0
            ):
                return depth
        return 0

    max_x = max(1, int(width * 0.28))
    max_y = max(1, int(height * 0.18))
    x_transitions = transition_profile(1)
    left = depth_for_side(1, False, max_x, x_transitions)
    right_depth = depth_for_side(1, True, max_x, x_transitions[::-1])
    del x_transitions
    y_transitions = transition_profile(0)
    top = depth_for_side(0, False, max_y, y_transitions)
    bottom_depth = depth_for_side(0, True, max_y, y_transitions[::-1])
    return left, top, width - right_depth, height - bottom_depth


def refine_corner_residuals(
    rgb: np.ndarray,
    box: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Remove a short gray corner sliver left after a main edge crop.

    A skewed or curled sheet can expose the scanner bed only near one corner.
    Such a run is too short for a safe stand-alone side detection.  Once the
    adjoining top/bottom scanner band is independently confirmed, however, a
    narrow neutral sliver at that same corner can safely refine the rectangle.
    """
    height, width = rgb.shape[:2]
    left, top, right, bottom = box
    if top == 0 and bottom == height:
        return box

    work = rgb.astype(np.float32)
    gray = 0.2126 * work[:, :, 0] + 0.7152 * work[:, :, 1] + 0.0722 * work[:, :, 2]
    rgb_i16 = rgb.astype(np.int16)
    chroma = np.max(rgb_i16, axis=2) - np.min(rgb_i16, axis=2)
    retained_height = bottom - top
    retained_width = right - left
    span_y = max(40, int(retained_height * 0.10))
    max_x = max(8, int(retained_width * 0.05))

    def corner_depth(region_gray: np.ndarray, region_chroma: np.ndarray) -> int:
        profile = np.median(region_gray, axis=0)
        paper = float(np.percentile(profile, 80))
        cutoff = max(225.0, paper - 15.0)
        bright = profile >= cutoff
        sustained = ndimage.uniform_filter1d(
            bright.astype(np.float32), size=5, mode="constant"
        ) >= 0.80
        found = np.flatnonzero(sustained[3:max_x])
        if not found.size:
            return 0
        depth = int(found[0] + 3)
        if depth < 3:
            return 0
        outer_gray = region_gray[:, :depth]
        outer_chroma = region_chroma[:, :depth]
        inner_gray = region_gray[:, depth + 3 : min(max_x, depth + 12)]
        if not inner_gray.size:
            return 0
        if (
            float(np.mean(outer_chroma <= 15)) >= 0.92
            and float(np.mean(outer_gray < cutoff)) >= 0.60
            and float(np.median(inner_gray) - np.median(outer_gray)) >= 18.0
        ):
            return depth
        return 0

    left_extra = 0
    right_extra = 0
    if bottom < height:
        y0 = max(top, bottom - span_y)
        left_extra = max(
            left_extra,
            corner_depth(
                gray[y0:bottom, left : min(right, left + max_x)],
                chroma[y0:bottom, left : min(right, left + max_x)],
            ),
        )
        right_extra = max(
            right_extra,
            corner_depth(
                gray[y0:bottom, max(left, right - max_x) : right][:, ::-1],
                chroma[y0:bottom, max(left, right - max_x) : right][:, ::-1],
            ),
        )
    if top > 0:
        y1 = min(bottom, top + span_y)
        left_extra = max(
            left_extra,
            corner_depth(
                gray[top:y1, left : min(right, left + max_x)],
                chroma[top:y1, left : min(right, left + max_x)],
            ),
        )
        right_extra = max(
            right_extra,
            corner_depth(
                gray[top:y1, max(left, right - max_x) : right][:, ::-1],
                chroma[top:y1, max(left, right - max_x) : right][:, ::-1],
            ),
        )
    if right - right_extra - (left + left_extra) < retained_width * 0.90:
        return box
    return left + left_extra, top, right - right_extra, bottom


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

    single_sided = detect_single_sided_neutral_strip_box(rgb)

    nested_frame = detect_nested_neutral_frame_box(rgb)
    if nested_frame is not None:
        # A rectangular frame and a much deeper isolated scanner strip may
        # coexist on the same page.  Keep the stable frame result for tiny
        # disagreements, but accept a single-side result when it identifies a
        # materially wider outer band from this page's own pixels.
        left, top, right, bottom = nested_frame
        if single_sided[0] >= left + 20:
            left = single_sided[0]
        if single_sided[1] >= top + 20:
            top = single_sided[1]
        if single_sided[2] <= right - 20:
            right = single_sided[2]
        if single_sided[3] <= bottom - 20:
            bottom = single_sided[3]
        return refine_corner_residuals(rgb, (left, top, right, bottom))

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
        left, top, right, bottom = 0, 0, width, height
    combined = (
        max(left, single_sided[0]),
        max(top, single_sided[1]),
        min(right, single_sided[2]),
        min(bottom, single_sided[3]),
    )
    return refine_corner_residuals(rgb, combined)


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


def process_image(
    image: Image.Image,
    mode: str,
    crop_box: tuple[int, int, int, int] | None = None,
) -> Image.Image:
    """Whiten high-confidence paper background without generating details."""
    if mode == "border-only":
        if crop_box is None:
            return crop_physical_scanner_borders(image)
        return image.crop(crop_box)
    image = normalize_image(image)
    if crop_box is None:
        image = crop_physical_scanner_borders(image)
    else:
        image = image.crop(crop_box)
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
    # Encode to a unique file beside the destination, validate it, and only
    # then replace the destination.  An interruption or filesystem hiccup can
    # therefore never turn a previously complete page into a half-written PNG.
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{path.stem}-", suffix=".png", dir=path.parent, delete=False
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        image.save(
            temporary,
            "PNG",
            compress_level=6,
            dpi=(dpi, dpi),
            pnginfo=pnginfo,
        )
        with Image.open(temporary) as check:
            check.verify()
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def set_jpeg_dpi_metadata(path: Path, dpi: int) -> None:
    """Set JFIF density without decoding or recompressing JPEG pixels."""
    data = bytearray(path.read_bytes())
    if data[:2] != b"\xff\xd8":
        raise RuntimeError(f"JPEG文件头无效：{path}")
    density = struct.pack(">H", dpi)
    offset = 2
    while offset + 4 <= len(data) and data[offset] == 0xFF:
        marker = data[offset + 1]
        if marker == 0xDA:
            break
        if marker in {0x01, 0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
            offset += 2
            continue
        length = int.from_bytes(data[offset + 2 : offset + 4], "big")
        if length < 2 or offset + 2 + length > len(data):
            raise RuntimeError(f"JPEG段结构无效：{path}")
        if marker == 0xE0 and data[offset + 4 : offset + 9] == b"JFIF\x00":
            if length < 16:
                raise RuntimeError(f"JPEG JFIF段过短：{path}")
            data[offset + 11] = 1
            data[offset + 12 : offset + 14] = density
            data[offset + 14 : offset + 16] = density
            path.write_bytes(data)
            return
        offset += 2 + length

    jfif_payload = b"JFIF\x00\x01\x01\x01" + density + density + b"\x00\x00"
    jfif_segment = b"\xff\xe0" + struct.pack(">H", len(jfif_payload) + 2) + jfif_payload
    path.write_bytes(data[:2] + jfif_segment + data[2:])


def set_png_dpi_metadata(path: Path, dpi: int) -> None:
    """Insert or replace PNG pHYs without decoding or recompressing pixels."""
    data = path.read_bytes()
    signature = b"\x89PNG\r\n\x1a\n"
    if not data.startswith(signature):
        raise RuntimeError(f"PNG文件头无效：{path}")
    pixels_per_meter = round(dpi / 0.0254)
    chunk_type = b"pHYs"
    chunk_data = struct.pack(">IIB", pixels_per_meter, pixels_per_meter, 1)
    replacement = (
        struct.pack(">I", len(chunk_data))
        + chunk_type
        + chunk_data
        + struct.pack(">I", zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF)
    )
    output = bytearray(signature)
    offset = len(signature)
    inserted = False
    while offset + 12 <= len(data):
        length = int.from_bytes(data[offset : offset + 4], "big")
        end = offset + 12 + length
        if end > len(data):
            raise RuntimeError(f"PNG块结构无效：{path}")
        current_type = data[offset + 4 : offset + 8]
        if current_type == b"pHYs":
            if not inserted:
                output.extend(replacement)
                inserted = True
        else:
            output.extend(data[offset:end])
            if current_type == b"IHDR" and not inserted:
                output.extend(replacement)
                inserted = True
        offset = end
    if offset != len(data):
        raise RuntimeError(f"PNG尾部结构无效：{path}")
    path.write_bytes(output)


def set_image_dpi_metadata(path: Path, dpi: int) -> None:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        set_jpeg_dpi_metadata(path, dpi)
    elif suffix == ".png":
        set_png_dpi_metadata(path, dpi)
    else:
        raise RuntimeError(
            f"无法在不重新编码像素的前提下为{suffix or '未知格式'}写入DPI：{path}"
        )


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


def analyze_document_crop_boxes(
    original_paths: list[Path],
) -> tuple[list[tuple[int, int, int, int]], list[str]]:
    """Analyze the whole document and infer weak edges from peer pages.

    Pages are grouped by orientation and portrait canvas width.  Only groups
    with at least four four-sided direct detections establish a consensus.
    Directly detected sides always win; consensus fills missing sides only.
    """
    sizes: list[tuple[int, int]] = []
    direct: list[tuple[int, int, int, int]] = []
    keys: list[str] = []
    for path in original_paths:
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened)
            image.load()
        width, height = image.size
        sizes.append((width, height))
        direct.append(detect_physical_crop_box(image))
        if width > height:
            keys.append("landscape")
        elif width / max(height, 1) < 0.70:
            keys.append("portrait-narrow")
        else:
            keys.append("portrait-wide")

    boxes = list(direct)
    sources = ["直接检测" for _ in boxes]
    for key in sorted(set(keys)):
        members = [i for i, item in enumerate(keys) if item == key]
        references = []
        for i in members:
            width, height = sizes[i]
            left, top, right, bottom = direct[i]
            margins = (left, top, width - right, height - bottom)
            if all(value > 0 for value in margins):
                references.append(i)
        if len(references) < 4:
            continue

        target_width = int(round(float(np.median([
            direct[i][2] - direct[i][0] for i in references
        ]))))
        target_height = int(round(float(np.median([
            direct[i][3] - direct[i][1] for i in references
        ]))))
        typical_left = int(round(float(np.median([direct[i][0] for i in references]))))
        typical_top = int(round(float(np.median([direct[i][1] for i in references]))))
        typical_right_margin = int(round(float(np.median([
            sizes[i][0] - direct[i][2] for i in references
        ]))))
        typical_bottom_margin = int(round(float(np.median([
            sizes[i][1] - direct[i][3] for i in references
        ]))))

        def nearby_long_edge(
            path: Path,
            axis: int,
            expected: int,
            span: int,
        ) -> int | None:
            with Image.open(path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
                rgb = np.asarray(image, dtype=np.float32)
            smooth = ndimage.uniform_filter1d(rgb, size=5, axis=axis, mode="nearest")
            if axis == 1:
                delta = np.linalg.norm(smooth[:, 6:] - smooth[:, :-6], axis=2)
                shares = np.mean(delta >= 25.0, axis=0)
                limit = rgb.shape[1]
            else:
                delta = np.linalg.norm(smooth[6:] - smooth[:-6], axis=2)
                shares = np.mean(delta >= 25.0, axis=1)
                limit = rgb.shape[0]
            lo = max(3, expected - span)
            hi = min(limit - 4, expected + span)
            if hi <= lo:
                return None
            window = shares[lo - 3 : hi - 3]
            if not window.size:
                return None
            relative = int(np.argmax(window))
            if float(window[relative]) < 0.35:
                return None
            return lo + relative

        def infer_axis(
            size: int,
            start: int,
            end: int,
            target: int,
            typical_start: int,
            typical_end_margin: int,
        ) -> tuple[int, int, bool]:
            start_known = start > 0
            end_known = end < size
            if start_known and end_known:
                return start, end, False
            if target <= 0 or target > size:
                return start, end, False
            if start_known:
                tolerance = max(80, int(size * 0.06))
                if abs(start - typical_start) > tolerance:
                    return start, end, False
                inferred_end = start + target
                if inferred_end <= size:
                    return start, inferred_end, True
            elif end_known:
                tolerance = max(80, int(size * 0.06))
                if abs((size - end) - typical_end_margin) > tolerance:
                    return start, end, False
                inferred_start = end - target
                if inferred_start >= 0:
                    return inferred_start, end, True
            else:
                inferred_start = min(max(0, typical_start), size - target)
                return inferred_start, inferred_start + target, True
            return start, end, False

        for i in members:
            width, height = sizes[i]
            left, top, right, bottom = direct[i]
            refined_any = False
            x_tolerance = max(80, int(width * 0.06))
            y_tolerance = max(80, int(height * 0.06))
            if not left or abs(left - typical_left) > x_tolerance:
                refined = nearby_long_edge(
                    original_paths[i], 1, typical_left, x_tolerance
                )
                if refined is not None:
                    left = refined
                    refined_any = True
            expected_right = width - typical_right_margin
            if right == width or abs(right - expected_right) > x_tolerance:
                refined = nearby_long_edge(
                    original_paths[i], 1, expected_right, x_tolerance
                )
                if refined is not None:
                    right = refined
                    refined_any = True
            if not top or abs(top - typical_top) > y_tolerance:
                refined = nearby_long_edge(
                    original_paths[i], 0, typical_top, y_tolerance
                )
                if refined is not None:
                    top = refined
                    refined_any = True
            expected_bottom = height - typical_bottom_margin
            if bottom == height or abs(bottom - expected_bottom) > y_tolerance:
                refined = nearby_long_edge(
                    original_paths[i], 0, expected_bottom, y_tolerance
                )
                if refined is not None:
                    bottom = refined
                    refined_any = True
            new_left, new_right, inferred_x = infer_axis(
                width,
                left,
                right,
                target_width,
                typical_left,
                typical_right_margin,
            )
            new_top, new_bottom, inferred_y = infer_axis(
                height,
                top,
                bottom,
                target_height,
                typical_top,
                typical_bottom_margin,
            )
            if inferred_x or inferred_y or refined_any:
                # Reject implausibly aggressive inference.  The page rectangle
                # must retain most of each canvas dimension and a normal area.
                retained_width = new_right - new_left
                retained_height = new_bottom - new_top
                if (
                    retained_width >= width * 0.60
                    and retained_height >= height * 0.75
                    and retained_width > 0
                    and retained_height > 0
                ):
                    boxes[i] = (new_left, new_top, new_right, new_bottom)
                    sources[i] = "整本PDF尺寸共识修正"
    return boxes, sources


def process_saved_page(
    task: tuple[
        int,
        Path,
        Path,
        int,
        str,
        bool,
    ]
) -> tuple[int, str, tuple[int, int, int, int] | None]:
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
        return index, "跳过", None

    with Image.open(original_path) as opened:
        if mode == "border-only":
            original = ImageOps.exif_transpose(opened)
        else:
            original = normalize_image(opened)
        original.load()
    crop_box = detect_physical_crop_box(original)
    processed = process_image(original, mode, crop_box)
    save_png(
        processed,
        processed_path,
        dpi,
        output_version,
        source_sha256,
    )
    return index, "处理", crop_box


def process_pdf(
    pdf_path: Path,
    dpi: int,
    mode: str,
    overwrite: bool,
    workers: int,
    extract_only: bool = False,
    extract_dpi: int | None = None,
) -> None:
    pdf_path = pdf_path.expanduser().resolve()
    if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
        raise FileNotFoundError(f"找不到 PDF：{pdf_path}")

    original_dir = pdf_path.with_suffix("")
    processed_dir = pdf_path.parent / f"{pdf_path.stem}-processed"
    original_dir.mkdir(exist_ok=True)
    processed_dir.mkdir(exist_ok=True)
    # Remove only our own abandoned atomic-write files from an earlier
    # interruption. Numbered page outputs and every other user file remain.
    for temporary in processed_dir.glob(".[0-9]*-*.png"):
        temporary.unlink(missing_ok=True)

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
            if extract_dpi is not None:
                print(f"模式：仅提取原图，并写入{extract_dpi} DPI元数据；不处理图片\n")
            else:
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

            if extract_dpi is not None:
                set_image_dpi_metadata(original_path, extract_dpi)
                original_status += f"；DPI={extract_dpi}"

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
        f"阶段 2/2：每页独立并行检测并处理（{active_workers}个工作线程）",
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
        for index, processed_status, crop_box in results:
            crop_text = f"；裁切框:{crop_box}" if crop_box is not None else ""
            print(
                f"[处理 {index + 1:0{digits}d}/{total}] "
                f"处理图:{processed_status}{crop_text}",
                flush=True,
            )
    for temporary in processed_dir.glob(".[0-9]*-*.png"):
        temporary.unlink(missing_ok=True)

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
    parser.add_argument(
        "--extract-dpi",
        type=int,
        default=None,
        help="仅提取时写入指定DPI元数据；JPEG/PNG不重新编码像素",
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
    if args.extract_dpi is not None:
        if args.extract_dpi < 1 or args.extract_dpi > 65535:
            print("提取图片DPI必须在1到65535之间。", file=sys.stderr)
            return 2
        args.extract_only = True

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
                    args.extract_dpi,
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
