from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRESENTATION_DIR = PROJECT_ROOT / "outputs" / "PRESENTATION MATERIAL"
ASSET_DIR = PRESENTATION_DIR / "v6_v8_visual_assets"

WEEK1_DIR = PRESENTATION_DIR / "week1_v6_chart_data"
CONTROLLED_AREA_CSVS = (
    PRESENTATION_DIR
    / "v4_area_dancetrack0065_B-224-L-224-g-224_A128-100000_n16_frames120_20260601_123552"
    / "area_stress_summary.csv",
    PRESENTATION_DIR
    / "v4_area_dancetrack0065_L-224-g-224_A1000-100000_n11_frames120_20260601_130731"
    / "area_stress_summary.csv",
)


COLORS = {
    "ink": "#1f2933",
    "muted": "#52606d",
    "grid": "#d9e2ec",
    "blue": "#1f5f99",
    "blue_light": "#e5f1fb",
    "green": "#247a5a",
    "green_light": "#e2f3ea",
    "gold": "#b7791f",
    "gold_light": "#fff4d6",
    "red": "#b83a2f",
    "red_light": "#fde8e4",
    "purple": "#5b4b8a",
    "purple_light": "#ece8f6",
}


VERSION_COLORS = {
    "V4 serial baseline": "#7b8794",
    "V5 shared wrapper": "#b7791f",
    "V6 gated SOT memory": "#1f5f99",
}


CONFIG_COLORS = {
    "B-224": "#1f5f99",
    "L-224": "#247a5a",
    "g-224": "#b7791f",
}


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(row: Dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in {"", None} else float("nan")


def as_int(row: Dict[str, str], key: str) -> int:
    return int(float(row[key]))


def setup_slide(title: str, subtitle: str = ""):
    fig, ax = plt.subplots(figsize=(16, 9), dpi=120)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.text(0.6, 8.55, title, fontsize=26, fontweight="bold", color=COLORS["ink"], va="top")
    if subtitle:
        ax.text(0.62, 8.12, subtitle, fontsize=13.5, color=COLORS["muted"], va="top")
    return fig, ax


def add_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    lines: Sequence[str] = (),
    fill: str = "white",
    edge: str = COLORS["grid"],
    title_color: str = COLORS["ink"],
    body_color: str = COLORS["muted"],
    title_size: float = 12.5,
    body_size: float = 10.5,
    lw: float = 1.8,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        facecolor=fill,
        edgecolor=edge,
        linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(x + 0.18, y + h - 0.22, title, fontsize=title_size, fontweight="bold", color=title_color, va="top")
    for index, line in enumerate(lines):
        ax.text(x + 0.2, y + h - 0.62 - (index * 0.31), line, fontsize=body_size, color=body_color, va="top")
    return patch


def add_arrow(
    ax,
    start: Tuple[float, float],
    end: Tuple[float, float],
    color: str = COLORS["ink"],
    lw: float = 1.8,
    curve: float = 0.0,
    label: str = "",
    label_offset: Tuple[float, float] = (0, 0),
):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={curve}",
    )
    ax.add_patch(patch)
    if label:
        mx = (start[0] + end[0]) / 2 + label_offset[0]
        my = (start[1] + end[1]) / 2 + label_offset[1]
        ax.text(mx, my, label, fontsize=10.5, color=color, ha="center", va="center")
    return patch


def add_badge(ax, x: float, y: float, text: str, fill: str, color: str = "white", size: float = 11.5):
    ax.text(
        x,
        y,
        text,
        fontsize=size,
        fontweight="bold",
        color=color,
        va="center",
        ha="center",
        bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.15", facecolor=fill, edgecolor=fill),
    )


def save_figure(fig, stem: str) -> List[Path]:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    paths = [ASSET_DIR / f"{stem}.png", ASSET_DIR / f"{stem}.svg"]
    fig.savefig(paths[0], dpi=180, facecolor="white")
    fig.savefig(paths[1], facecolor="white")
    plt.close(fig)
    return paths


def make_v6_architecture() -> List[Path]:
    fig, ax = setup_slide(
        "V6 architecture: gated whole-LoRAT calls",
        "V6 does not split LoRAT apart: every selected memory slot still runs the full LoRAT SOT tracker.",
    )

    add_badge(ax, 2.2, 7.55, "normal frame: 1 primary slot", COLORS["blue"])
    add_badge(ax, 6.4, 7.55, "uncertain frame: up to 5 recovery slots", COLORS["gold"])
    add_badge(ax, 12.5, 7.55, "cost: objects x selected slots", COLORS["red"])

    add_box(
        ax,
        0.55,
        5.85,
        2.5,
        1.25,
        "Frame + tracks",
        ("current frame", "active tracks"),
        fill=COLORS["blue_light"],
        edge=COLORS["blue"],
    )
    add_box(
        ax,
        0.55,
        3.95,
        2.5,
        1.25,
        "Per-track state",
        ("confidence + margin", "lost/occlusion state"),
        fill="#f8fafc",
        edge=COLORS["grid"],
    )
    add_arrow(ax, (1.8, 5.85), (1.8, 5.22), COLORS["muted"])

    add_box(
        ax,
        3.65,
        5.85,
        2.85,
        1.25,
        "Primary slot path",
        ("fresh active slot", "or freshest recent slot"),
        fill=COLORS["blue_light"],
        edge=COLORS["blue"],
    )
    add_box(
        ax,
        3.65,
        3.45,
        2.85,
        1.65,
        "Recovery slot path",
        ("initial anchor", "active + recent slots", "low conf / stale / occlusion"),
        fill=COLORS["gold_light"],
        edge=COLORS["gold"],
        body_size=9.6,
    )

    add_arrow(ax, (3.05, 6.48), (3.65, 6.48), COLORS["blue"])
    add_arrow(ax, (3.05, 4.58), (3.65, 4.28), COLORS["gold"], curve=-0.18)

    add_box(
        ax,
        7.1,
        4.65,
        2.35,
        1.6,
        "Selected LoRAT calls",
        ("whole SOT task per slot", "template/search pair"),
        fill=COLORS["purple_light"],
        edge=COLORS["purple"],
        title_size=11.6,
        body_size=9.9,
    )
    add_arrow(ax, (6.5, 6.48), (7.1, 5.63), COLORS["blue"], curve=-0.12)
    add_arrow(ax, (6.5, 4.28), (7.1, 5.1), COLORS["gold"], curve=0.12)

    add_box(
        ax,
        10.05,
        4.65,
        2.75,
        1.6,
        "Whole LoRAT evaluator",
        ("ViT body + original head", "template/search fused", "box + score"),
        fill="#f8fafc",
        edge=COLORS["muted"],
    )
    add_arrow(ax, (9.45, 5.45), (10.05, 5.45), COLORS["ink"])

    add_box(
        ax,
        13.35,
        4.85,
        2.1,
        1.2,
        "Slot outputs",
        ("candidate boxes", "candidate scores"),
        fill=COLORS["green_light"],
        edge=COLORS["green"],
    )
    add_arrow(ax, (12.8, 5.45), (13.35, 5.45), COLORS["ink"])

    add_box(
        ax,
        9.25,
        2.55,
        3.0,
        1.25,
        "Coordinator gates",
        ("identity arbitration", "motion/path/ReID/IoU"),
        fill="#f8fafc",
        edge=COLORS["grid"],
    )
    add_box(
        ax,
        13.0,
        2.55,
        2.45,
        1.25,
        "Final update",
        ("accept candidate", "hold Kalman box"),
        fill=COLORS["green_light"],
        edge=COLORS["green"],
    )
    add_arrow(ax, (14.4, 4.85), (14.4, 3.8), COLORS["green"])
    add_arrow(ax, (13.0, 3.17), (12.25, 3.17), COLORS["muted"])
    add_arrow(ax, (12.25, 3.17), (13.0, 3.17), COLORS["green"])

    ax.add_patch(Rectangle((0.55, 0.55), 14.9, 0.72, facecolor=COLORS["red_light"], edgecolor=COLORS["red"], linewidth=1.3))
    ax.text(
        8.0,
        0.91,
        "V6 boundary: slot gating around whole LoRAT; no LoRAT head replacement yet.",
        fontsize=12.2,
        fontweight="bold",
        color=COLORS["red"],
        ha="center",
        va="center",
    )
    return save_figure(fig, "v6_architecture_gated_sot_memory")


def make_v8_architecture() -> List[Path]:
    fig, ax = setup_slide(
        "V8 architecture: LoRAT body + replacement head",
        "V8 keeps the LoRAT ViT body/backbone, bypasses the original SOT head, and scores objects with a new batched tracker head.",
    )

    add_badge(ax, 2.0, 7.55, "1 ViT frame pass", COLORS["green"])
    add_badge(ax, 6.0, 7.55, "N batched head items", COLORS["blue"])
    add_badge(ax, 11.0, 7.55, "quality path: rerank + feature ReID", COLORS["purple"])

    add_box(
        ax,
        0.45,
        5.75,
        2.15,
        1.25,
        "Frame t",
        ("search tensor", "LoRAT normalize"),
        fill=COLORS["green_light"],
        edge=COLORS["green"],
    )
    add_box(
        ax,
        3.15,
        5.55,
        2.7,
        1.65,
        "LoRAT ViT body kept",
        ("calls _x_feat", "ViT blocks + norm", "shared grid tokens"),
        fill=COLORS["green_light"],
        edge=COLORS["green"],
    )
    add_box(
        ax,
        6.45,
        5.75,
        2.6,
        1.25,
        "Shared feature map",
        ("grid_h x grid_w tokens", "one tensor for all objects"),
        fill="#f8fafc",
        edge=COLORS["grid"],
        title_size=11.8,
        body_size=9.8,
    )
    add_arrow(ax, (2.6, 6.36), (3.15, 6.36), COLORS["green"])
    add_arrow(ax, (5.85, 6.36), (6.45, 6.36), COLORS["green"])

    add_box(
        ax,
        9.45,
        5.75,
        2.75,
        1.15,
        "Original LoRAT SOT head",
        ("bypassed in V8 update",),
        fill=COLORS["red_light"],
        edge=COLORS["red"],
        title_color=COLORS["red"],
        body_color=COLORS["red"],
        title_size=11.0,
        body_size=9.0,
    )
    ax.plot([9.55, 12.1], [5.85, 6.8], color=COLORS["red"], linewidth=2.5)
    ax.plot([9.55, 12.1], [6.8, 5.85], color=COLORS["red"], linewidth=2.5)

    add_box(
        ax,
        0.45,
        3.35,
        2.9,
        1.45,
        "Per-object head bank",
        ("initial feature vector", "recent trusted features", "primary/recovery selection"),
        fill=COLORS["blue_light"],
        edge=COLORS["blue"],
    )
    add_box(
        ax,
        4.0,
        3.15,
        3.0,
        1.85,
        "New V8 tracker head",
        ("object embedding", "score map + box deltas", "batched across objects"),
        fill=COLORS["blue_light"],
        edge=COLORS["blue"],
    )
    add_arrow(ax, (3.35, 4.05), (4.0, 4.05), COLORS["blue"])
    add_arrow(ax, (7.6, 5.75), (6.85, 5.0), COLORS["green"], curve=0.1, label="shared tokens", label_offset=(0.12, -0.06))

    add_box(
        ax,
        7.65,
        3.15,
        2.55,
        1.85,
        "Candidate decode",
        ("mask to local ROI", "top-5 candidate cells", "center + size deltas"),
        fill=COLORS["gold_light"],
        edge=COLORS["gold"],
    )
    add_arrow(ax, (7.0, 4.05), (7.65, 4.05), COLORS["blue"])

    add_box(
        ax,
        10.85,
        4.05,
        2.35,
        1.45,
        "Rescue path",
        ("template match", "fuse with head bbox", "prefer margin gain"),
        fill=COLORS["purple_light"],
        edge=COLORS["purple"],
        body_size=9.8,
    )
    add_box(
        ax,
        10.85,
        2.2,
        2.35,
        1.35,
        "Feature ReID",
        ("ViT appearance", "motion/path/IoU"),
        fill=COLORS["purple_light"],
        edge=COLORS["purple"],
        body_size=9.8,
    )
    add_arrow(ax, (10.2, 4.15), (10.85, 4.75), COLORS["gold"], curve=0.1)
    add_arrow(ax, (10.2, 3.65), (10.85, 2.88), COLORS["purple"], curve=-0.1)

    add_box(
        ax,
        13.85,
        3.0,
        1.85,
        1.6,
        "Track update",
        ("accept", "hold", "refresh memory"),
        fill=COLORS["green_light"],
        edge=COLORS["green"],
    )
    add_arrow(ax, (13.2, 4.75), (13.85, 4.05), COLORS["purple"], curve=-0.1)
    add_arrow(ax, (13.2, 2.88), (13.85, 3.45), COLORS["purple"], curve=0.1)

    ax.add_patch(Rectangle((0.55, 0.55), 14.9, 0.72, facecolor=COLORS["green_light"], edgecolor=COLORS["green"], linewidth=1.3))
    ax.text(
        8.0,
        0.91,
        "Week 2 proof counters: one LoRAT-body call per frame; V8 tracker-head items match active tracked objects.",
        fontsize=12.6,
        fontweight="bold",
        color=COLORS["green"],
        ha="center",
        va="center",
    )
    return save_figure(fig, "v8_architecture_shared_frame_batched_heads")


def make_v6_to_v8_changes() -> List[Path]:
    fig, ax = setup_slide(
        "What changes between V6 and V8",
        "The refactor moves the expensive object-specific work out of the ViT pass and into a small batched head layer.",
    )

    col_x = [0.6, 5.75, 11.0]
    widths = [4.4, 4.1, 4.4]
    headers = [
        ("V6", "Gated SOT memory wrapper", COLORS["blue"], COLORS["blue_light"]),
        ("Change", "Split model internals", COLORS["gold"], COLORS["gold_light"]),
        ("V8", "Shared-frame + trained head", COLORS["green"], COLORS["green_light"]),
    ]
    for x, w, (title, subtitle, color, fill) in zip(col_x, widths, headers):
        add_box(ax, x, 6.85, w, 0.92, title, (subtitle,), fill=fill, edge=color, title_size=16, body_size=10.5)

    rows = [
        (
            "Backbone work",
            "selected slot -> SOT evaluator",
            "split frame tokens",
            "one ViT pass per frame",
        ),
        (
            "Object memory",
            "template/search slots",
            "cache features",
            "initial + recent feature bank",
        ),
        (
            "Recovery",
            "more SOT slots",
            "reuse trigger logic",
            "more head-bank items",
        ),
        (
            "Box prediction",
            "upstream SOT box",
            "train new head",
            "score map + box deltas",
        ),
        (
            "Quality controls",
            "motion/ReID/IoU gates",
            "carry gates forward",
            "feature ReID + rerank",
        ),
    ]

    y = 5.95
    for index, (label, v6, change, v8) in enumerate(rows):
        fill = "#ffffff" if index % 2 == 0 else "#f8fafc"
        ax.add_patch(Rectangle((0.55, y - 0.85), 14.95, 0.92, facecolor=fill, edgecolor=COLORS["grid"], linewidth=0.8))
        ax.text(0.72, y - 0.1, label, fontsize=11.2, fontweight="bold", color=COLORS["ink"], va="top")
        ax.text(3.0, y - 0.1, v6, fontsize=10.8, color=COLORS["muted"], va="top")
        ax.text(7.0, y - 0.1, change, fontsize=10.8, color=COLORS["muted"], va="top")
        ax.text(11.0, y - 0.1, v8, fontsize=10.8, color=COLORS["muted"], va="top")
        y -= 0.97

    add_arrow(ax, (4.97, 7.31), (5.74, 7.31), COLORS["gold"])
    add_arrow(ax, (9.85, 7.31), (11.0, 7.31), COLORS["green"])

    ax.add_patch(Rectangle((0.55, 0.55), 14.95, 0.74, facecolor=COLORS["purple_light"], edgecolor=COLORS["purple"], linewidth=1.3))
    ax.text(
        8.0,
        0.92,
        "Slide takeaway: V6 controls how many SOT calls happen; V8 changes what one frame update is.",
        fontsize=13.0,
        fontweight="bold",
        color=COLORS["purple"],
        ha="center",
        va="center",
    )
    return save_figure(fig, "v6_to_v8_architecture_changes")


def make_v6_v8_same_vs_different() -> List[Path]:
    fig, ax = setup_slide(
        "V6 vs V8: same tracker shell, different execution engine",
        "Both versions keep the MOT coordinator and safety checks; V8 changes how object evidence is produced.",
    )

    add_badge(ax, 3.0, 7.55, "V6-specific", COLORS["blue"])
    add_badge(ax, 8.0, 7.55, "same in both", COLORS["green"])
    add_badge(ax, 13.0, 7.55, "V8-specific", COLORS["purple"])

    add_box(
        ax,
        0.7,
        5.8,
        4.2,
        1.45,
        "V6 evidence engine",
        ("LoRAT memory slots", "slot -> SOT evaluator", "template/search fusion"),
        fill=COLORS["blue_light"],
        edge=COLORS["blue"],
        title_size=13.2,
        body_size=10.5,
    )
    add_box(
        ax,
        0.7,
        3.82,
        4.2,
        1.38,
        "V6 cost shape",
        ("objects x selected slots", "large SOT path per slot"),
        fill="#f8fafc",
        edge=COLORS["blue"],
        title_size=13.2,
        body_size=10.5,
    )
    add_box(
        ax,
        0.7,
        2.05,
        4.2,
        1.15,
        "V6 memory meaning",
        ("memory = LoRAT task slots", "recovery = try more SOT slots"),
        fill="#f8fafc",
        edge=COLORS["grid"],
        title_size=13.2,
        body_size=10.5,
    )

    add_box(
        ax,
        5.55,
        5.9,
        4.9,
        1.35,
        "Preserved tracker shell",
        ("selected objects + TrackState", "Kalman hold + outputs/status"),
        fill=COLORS["green_light"],
        edge=COLORS["green"],
        title_size=13.2,
        body_size=10.5,
    )
    add_box(
        ax,
        5.55,
        3.86,
        4.9,
        1.45,
        "Preserved safety gates",
        ("identity arbitration", "motion / path / IoU checks", "occlusion + shrink guards"),
        fill=COLORS["green_light"],
        edge=COLORS["green"],
        title_size=13.2,
        body_size=10.5,
    )
    add_box(
        ax,
        5.55,
        2.02,
        4.9,
        1.22,
        "Preserved recovery idea",
        ("stable frames use a small primary set", "uncertain frames expand the memory set"),
        fill="#f8fafc",
        edge=COLORS["green"],
        title_size=13.2,
        body_size=10.5,
    )

    add_box(
        ax,
        11.1,
        5.8,
        4.2,
        1.45,
        "V8 evidence engine",
        ("shared frame encoder", "shared feature map", "batched object heads"),
        fill=COLORS["purple_light"],
        edge=COLORS["purple"],
        title_size=13.2,
        body_size=10.5,
    )
    add_box(
        ax,
        11.1,
        3.82,
        4.2,
        1.38,
        "V8 cost shape",
        ("1 shared ViT frame pass", "+ N smaller head items"),
        fill="#f8fafc",
        edge=COLORS["purple"],
        title_size=13.2,
        body_size=10.5,
    )
    add_box(
        ax,
        11.1,
        2.05,
        4.2,
        1.15,
        "V8 memory meaning",
        ("memory = feature head bank", "recovery = score more head items"),
        fill="#f8fafc",
        edge=COLORS["grid"],
        title_size=13.2,
        body_size=10.5,
    )

    add_arrow(ax, (4.9, 6.45), (5.55, 6.45), COLORS["green"], lw=2.2)
    add_arrow(ax, (10.45, 6.45), (11.1, 6.45), COLORS["green"], lw=2.2)
    add_arrow(ax, (4.9, 4.52), (5.55, 4.52), COLORS["green"], lw=2.2)
    add_arrow(ax, (10.45, 4.52), (11.1, 4.52), COLORS["green"], lw=2.2)
    add_arrow(ax, (4.9, 2.6), (5.55, 2.6), COLORS["green"], lw=2.2)
    add_arrow(ax, (10.45, 2.6), (11.1, 2.6), COLORS["green"], lw=2.2)

    ax.add_patch(Rectangle((0.7, 0.62), 14.6, 0.72, facecolor=COLORS["green_light"], edgecolor=COLORS["green"], linewidth=1.3))
    ax.text(
        8.0,
        0.98,
        "Takeaway: the MOT coordinator is preserved; V6 changes slot selection, while V8 changes the execution engine.",
        fontsize=11.8,
        fontweight="bold",
        color=COLORS["green"],
        ha="center",
        va="center",
    )

    return save_figure(fig, "v6_v8_same_vs_different_map")


def make_lorat_head_replacement_visual() -> List[Path]:
    fig, ax = setup_slide(
        "V6 whole LoRAT vs V8 replacement head",
        "V6 calls complete LoRAT; V8 keeps the body and swaps the head.",
    )

    # Panel backgrounds
    ax.add_patch(Rectangle((0.5, 1.15), 7.15, 6.45, facecolor="#f8fafc", edgecolor=COLORS["blue"], linewidth=1.8))
    ax.add_patch(Rectangle((8.35, 1.15), 7.15, 6.45, facecolor="#f8fafc", edgecolor=COLORS["purple"], linewidth=1.8))
    ax.text(0.78, 7.25, "V6: LoRAT intact", fontsize=17, fontweight="bold", color=COLORS["blue"])
    ax.text(8.63, 7.25, "V8: LoRAT body kept, head replaced", fontsize=17, fontweight="bold", color=COLORS["purple"])

    # V6 left panel
    add_box(
        ax,
        0.9,
        5.95,
        2.0,
        0.9,
        "LoRAT memory slot",
        ("template/crop state",),
        fill=COLORS["blue_light"],
        edge=COLORS["blue"],
        title_size=11.0,
        body_size=9.0,
    )
    add_box(
        ax,
        0.9,
        4.65,
        2.0,
        1.08,
        "Current frame",
        ("same video frame", "used as search image"),
        fill=COLORS["blue_light"],
        edge=COLORS["blue"],
        title_size=11.8,
        body_size=8.3,
    )

    add_box(
        ax,
        3.45,
        4.65,
        3.55,
        2.15,
        "Whole LoRAT tracker",
        ("ViT body + original SOT head", "called together as one tracker"),
        fill="#ffffff",
        edge=COLORS["blue"],
        title_size=13.4,
        body_size=9.8,
    )
    ax.add_patch(Rectangle((3.78, 5.15), 1.32, 0.45, facecolor=COLORS["green_light"], edgecolor=COLORS["green"], linewidth=1.2))
    ax.text(4.44, 5.375, "ViT body", fontsize=9.2, color=COLORS["green"], ha="center", va="center", fontweight="bold")
    ax.add_patch(Rectangle((5.32, 5.15), 1.32, 0.45, facecolor=COLORS["gold_light"], edgecolor=COLORS["gold"], linewidth=1.2))
    ax.text(5.98, 5.375, "SOT head", fontsize=9.2, color=COLORS["gold"], ha="center", va="center", fontweight="bold")

    add_arrow(ax, (2.9, 6.38), (3.45, 5.92), COLORS["blue"])
    add_arrow(ax, (2.9, 5.28), (3.45, 5.42), COLORS["blue"])
    add_box(
        ax,
        3.45,
        3.15,
        3.55,
        0.88,
        "V6 wrapper logic",
        ("gates whole-LoRAT calls",),
        fill=COLORS["blue_light"],
        edge=COLORS["blue"],
        title_size=11.8,
        body_size=9.2,
    )
    add_arrow(ax, (5.22, 4.65), (5.22, 4.03), COLORS["blue"])
    add_box(
        ax,
        3.45,
        1.78,
        3.55,
        0.78,
        "Output",
        ("candidate bbox + confidence score",),
        fill=COLORS["green_light"],
        edge=COLORS["green"],
        title_size=11.8,
        body_size=9.5,
    )
    add_arrow(ax, (5.22, 3.15), (5.22, 2.56), COLORS["green"])
    add_badge(ax, 4.98, 1.23, "cost grows with selected SOT calls", COLORS["red"], size=10.4)

    # V8 right panel
    add_box(
        ax,
        8.75,
        5.55,
        2.05,
        1.1,
        "Current frame",
        ("same video frame", "encoded once"),
        fill=COLORS["green_light"],
        edge=COLORS["green"],
        title_size=11.8,
        body_size=8.7,
    )
    add_box(
        ax,
        11.25,
        5.35,
        2.75,
        1.55,
        "LoRAT ViT body kept",
        ("_x_feat + ViT blocks", "shared feature map", "one pass per frame"),
        fill=COLORS["green_light"],
        edge=COLORS["green"],
        title_size=13.0,
        body_size=9.7,
    )
    add_arrow(ax, (10.8, 6.12), (11.25, 6.12), COLORS["green"])

    add_box(
        ax,
        11.25,
        3.72,
        2.75,
        0.92,
        "Original LoRAT SOT head",
        ("removed from update path",),
        fill=COLORS["red_light"],
        edge=COLORS["red"],
        title_color=COLORS["red"],
        body_color=COLORS["red"],
        title_size=11.7,
        body_size=9.2,
    )
    ax.plot([11.35, 13.9], [3.8, 4.56], color=COLORS["red"], linewidth=3)
    ax.plot([11.35, 13.9], [4.56, 3.8], color=COLORS["red"], linewidth=3)

    add_box(
        ax,
        8.75,
        2.32,
        2.05,
        1.25,
        "V8 head bank",
        ("feature memories", "initial + recent"),
        fill=COLORS["purple_light"],
        edge=COLORS["purple"],
        title_size=11.8,
        body_size=9.2,
    )
    add_box(
        ax,
        11.25,
        2.05,
        3.65,
        1.45,
        "New V8 tracker head",
        ("batched object-conditioned LoRA head", "score maps + box deltas", "template rescue + feature ReID"),
        fill=COLORS["purple_light"],
        edge=COLORS["purple"],
        title_size=12.3,
        body_size=9.4,
    )
    add_arrow(ax, (12.62, 5.35), (12.62, 3.5), COLORS["green"], curve=0.0)
    add_arrow(ax, (10.8, 2.85), (11.25, 2.85), COLORS["purple"])
    add_box(
        ax,
        11.25,
        1.28,
        3.65,
        0.55,
        "Output: multi-object candidates",
        (),
        fill=COLORS["green_light"],
        edge=COLORS["green"],
        title_size=11.3,
        body_size=8.8,
    )
    add_arrow(ax, (13.05, 2.05), (13.05, 1.83), COLORS["green"])
    add_badge(ax, 12.9, 1.04, "cost = 1 ViT frame pass + N head items", COLORS["purple"], size=10.1)

    # Shared coordinator note
    ax.add_patch(Rectangle((0.5, 0.28), 15.0, 0.55, facecolor=COLORS["green_light"], edgecolor=COLORS["green"], linewidth=1.2))
    ax.text(
        8.0,
        0.555,
        "Same downstream MOT coordinator in both: Kalman prediction, identity/motion/path/IoU gates, occlusion/shrink holds, accept-or-hold updates.",
        fontsize=10.8,
        color=COLORS["green"],
        fontweight="bold",
        ha="center",
        va="center",
    )

    return save_figure(fig, "v6_whole_lorat_v8_head_replacement")


def make_iou_explainer() -> List[Path]:
    fig, ax = setup_slide(
        "Metric explainer: IoU, IoU@0.50, and reliability bins",
        "Use this instead of a text-only metric slide: the boxes show the calculation, the dots show how it becomes a benchmark.",
    )

    ax.add_patch(Rectangle((0.75, 1.25), 6.55, 5.65, facecolor="#f8fafc", edgecolor=COLORS["grid"], linewidth=1.2))
    ax.text(1.05, 6.48, "Single sampled frame", fontsize=15, fontweight="bold", color=COLORS["ink"])

    gt = Rectangle((1.35, 2.3), 3.75, 2.55, facecolor="#d7ecfb", edgecolor=COLORS["blue"], linewidth=2.5)
    pred = Rectangle((3.15, 3.2), 3.35, 2.25, facecolor="#dff3e9", edgecolor=COLORS["green"], linewidth=2.5, alpha=0.78)
    inter = Rectangle((3.15, 3.2), 1.95, 1.65, facecolor=COLORS["gold_light"], edgecolor=COLORS["gold"], linewidth=2.0, alpha=0.95)
    ax.add_patch(gt)
    ax.add_patch(pred)
    ax.add_patch(inter)
    ax.text(1.45, 4.98, "Ground truth box", fontsize=11.5, color=COLORS["blue"], fontweight="bold")
    ax.text(4.05, 5.62, "Predicted box", fontsize=11.5, color=COLORS["green"], fontweight="bold")
    ax.text(3.35, 3.96, "intersection", fontsize=11, color=COLORS["gold"], fontweight="bold")

    ax.text(0.98, 1.92, "IoU = intersection area / union area", fontsize=13.5, color=COLORS["ink"], fontweight="bold")
    ax.text(0.98, 1.55, "Union = all pixels covered by either box", fontsize=11.5, color=COLORS["muted"])

    add_box(
        ax,
        8.0,
        5.35,
        3.1,
        1.45,
        "IoU@0.50",
        ("Count sampled boxes with", "IoU >= 0.50, then divide", "by total sampled boxes."),
        fill=COLORS["blue_light"],
        edge=COLORS["blue"],
    )
    add_box(
        ax,
        12.0,
        5.35,
        3.1,
        1.45,
        "Mean IoU",
        ("Average IoU over all", "sampled frames in an", "object-area bin."),
        fill=COLORS["green_light"],
        edge=COLORS["green"],
    )
    add_box(
        ax,
        8.0,
        3.25,
        7.1,
        1.45,
        "Small-object area bin",
        ("Sample every 10 frames. Area = current GT width x current GT height.", "Group samples into pixel-area bins before calculating reliability."),
        fill="#f8fafc",
        edge=COLORS["grid"],
        title_size=13.2,
    )

    sample_ious = [0.82, 0.74, 0.65, 0.58, 0.54, 0.49, 0.43, 0.35, 0.22, 0.10]
    x0, y0 = 8.25, 2.35
    for index, value in enumerate(sample_ious):
        color = COLORS["green"] if value >= 0.5 else COLORS["red"]
        ax.scatter(x0 + index * 0.64, y0, s=260, color=color, edgecolor="white", linewidth=1.5, zorder=3)
        ax.text(x0 + index * 0.64, y0 - 0.42, f"{value:.2f}", fontsize=9.5, color=COLORS["muted"], ha="center")
    ax.text(8.1, 2.92, "sampled IoU values", fontsize=11.5, color=COLORS["ink"], fontweight="bold")
    add_box(
        ax,
        8.0,
        0.88,
        7.1,
        0.95,
        "Reliable bin",
        ("IoU@0.50 >= 0.80 | mean IoU >= 0.50 | samples >= 10",),
        fill=COLORS["red_light"],
        edge=COLORS["red"],
        title_color=COLORS["red"],
        body_color=COLORS["red"],
        title_size=11.8,
        body_size=10.8,
    )

    return save_figure(fig, "iou_metric_explainer")


def group_by(rows: Iterable[Dict[str, str]], key: str) -> Dict[str, List[Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row[key], []).append(row)
    return grouped


def chart_week1_v6_ms_per_box() -> List[Path]:
    rows = [
        row
        for row in read_csv(WEEK1_DIR / "week1_v6_timing_all_versions_chart.csv")
        if row["version_label"] == "V6 gated SOT memory"
    ]
    fig, ax = plt.subplots(figsize=(16, 9), dpi=120)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for config, group in group_by(rows, "config").items():
        group = sorted(group, key=lambda row: as_int(row, "target_objects"))
        ax.plot(
            [as_int(row, "target_objects") for row in group],
            [as_float(row, "tracking_ms_per_box") for row in group],
            marker="o",
            linewidth=3,
            markersize=8,
            label=config,
            color=CONFIG_COLORS.get(config),
        )
    ax.set_title("Week 1 V6: tracking time per produced box", loc="left", fontsize=24, fontweight="bold", color=COLORS["ink"], pad=18)
    ax.text(0.01, 0.94, "Direct replacement for the ms/box table on the Week 1 benchmark slide.", transform=ax.transAxes, fontsize=13, color=COLORS["muted"])
    ax.set_xlabel("Tracked objects (N)", fontsize=13)
    ax.set_ylabel("Tracking ms per box", fontsize=13)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.grid(True, axis="y", color=COLORS["grid"], linewidth=1)
    ax.legend(title="Config", frameon=False, fontsize=12, title_fontsize=12, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.99, 0.03, "Source: week1_v6_timing_all_versions_chart.csv", transform=ax.transAxes, fontsize=9.5, color=COLORS["muted"], ha="right")
    return save_figure(fig, "week1_v6_ms_per_box_graph")


def chart_week1_fps_versions() -> List[Path]:
    rows = read_csv(WEEK1_DIR / "week1_v6_timing_all_versions_chart.csv")
    configs = ["B-224", "L-224", "g-224"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 9), dpi=120, sharey=True)
    fig.patch.set_facecolor("white")
    fig.suptitle("Week 1 benchmark: FPS versus object count", x=0.04, y=0.96, ha="left", fontsize=24, fontweight="bold", color=COLORS["ink"])
    fig.text(0.04, 0.91, "Shows why V6 gating helps runtime but still declines as more objects are tracked.", fontsize=13, color=COLORS["muted"])
    for ax, config in zip(axes, configs):
        config_rows = [row for row in rows if row["config"] == config]
        for version, group in group_by(config_rows, "version_label").items():
            group = sorted(group, key=lambda row: as_int(row, "target_objects"))
            ax.plot(
                [as_int(row, "target_objects") for row in group],
                [as_float(row, "tracking_fps") for row in group],
                marker="o",
                linewidth=2.8,
                markersize=6.5,
                label=version,
                color=VERSION_COLORS.get(version),
            )
        ax.set_title(config, fontsize=15, fontweight="bold", color=CONFIG_COLORS.get(config, COLORS["ink"]))
        ax.set_xlabel("Tracked objects (N)", fontsize=12)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.grid(True, axis="y", color=COLORS["grid"], linewidth=1)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Tracking FPS", fontsize=12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=3, fontsize=12)
    fig.text(0.96, 0.035, "Source: week1_v6_timing_all_versions_chart.csv", fontsize=9.5, color=COLORS["muted"], ha="right")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.82, bottom=0.16, wspace=0.18)
    return save_figure(fig, "week1_fps_vs_objects_by_version")


def chart_week1_speedup() -> List[Path]:
    rows = read_csv(WEEK1_DIR / "week1_v6_speedup_vs_serial_chart.csv")
    configs = ["B-224", "L-224", "g-224"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 9), dpi=120, sharey=True)
    fig.patch.set_facecolor("white")
    fig.suptitle("Week 1 benchmark: wrapper speedup versus serial baseline", x=0.04, y=0.96, ha="left", fontsize=24, fontweight="bold", color=COLORS["ink"])
    fig.text(0.04, 0.91, "V6 gains come from evaluating fewer memory slots on stable frames.", fontsize=13, color=COLORS["muted"])
    for ax, config in zip(axes, configs):
        group = sorted([row for row in rows if row["config"] == config], key=lambda row: as_int(row, "target_objects"))
        xs = [as_int(row, "target_objects") for row in group]
        v5 = [as_float(row, "v5_speedup_vs_serial") for row in group]
        v6 = [as_float(row, "v6_speedup_vs_serial") for row in group]
        width = 0.34
        ax.bar([x - width / 2 for x in xs], v5, width=width, label="V5", color=COLORS["gold"], alpha=0.76)
        ax.bar([x + width / 2 for x in xs], v6, width=width, label="V6", color=CONFIG_COLORS.get(config), alpha=0.9)
        ax.axhline(1.0, color=COLORS["muted"], linewidth=1.2)
        ax.set_title(config, fontsize=15, fontweight="bold", color=CONFIG_COLORS.get(config, COLORS["ink"]))
        ax.set_xlabel("Tracked objects (N)", fontsize=12)
        ax.set_xticks(xs)
        ax.grid(True, axis="y", color=COLORS["grid"], linewidth=1)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("FPS speedup vs V4 serial", fontsize=12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=2, fontsize=12)
    fig.text(0.96, 0.035, "Source: week1_v6_speedup_vs_serial_chart.csv", fontsize=9.5, color=COLORS["muted"], ha="right")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.82, bottom=0.16, wspace=0.18)
    return save_figure(fig, "week1_v6_speedup_vs_serial_graph")


def load_controlled_area_rows() -> List[Dict[str, str]]:
    by_key: Dict[Tuple[str, int], Dict[str, str]] = {}
    for path in CONTROLLED_AREA_CSVS:
        if not path.exists():
            continue
        for row in read_csv(path):
            key = (row["lorat_config"], as_int(row, "target_area_px"))
            by_key[key] = row
    return list(by_key.values())


def chart_controlled_small_object_reliability() -> List[Path]:
    rows = load_controlled_area_rows()
    fig, ax = plt.subplots(figsize=(16, 9), dpi=120)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for config, group in group_by(rows, "lorat_config").items():
        group = sorted(group, key=lambda row: as_float(row, "target_area_px"))
        color = CONFIG_COLORS.get(config, COLORS["ink"])
        x = [as_float(row, "target_area_px") for row in group]
        ax.plot(x, [as_float(row, "mean_iou") for row in group], marker="o", linewidth=2.7, label=f"{config} mean IoU", color=color)
        ax.plot(x, [as_float(row, "iou50") for row in group], marker="s", linewidth=2.2, linestyle="--", label=f"{config} IoU@0.50", color=color, alpha=0.72)
        reliable_x = [as_float(row, "target_area_px") for row in group if row.get("reliable") == "True"]
        reliable_y = [as_float(row, "iou50") for row in group if row.get("reliable") == "True"]
        if reliable_x:
            ax.scatter(reliable_x, reliable_y, s=120, facecolors="white", edgecolors=color, linewidths=2.4, zorder=5)

    ax.axhline(0.50, color=COLORS["green"], linewidth=1.6, alpha=0.8)
    ax.axhline(0.80, color=COLORS["red"], linewidth=1.6, alpha=0.8)
    ax.text(132, 0.515, "mean IoU threshold", fontsize=10.5, color=COLORS["green"], va="bottom")
    ax.text(132, 0.815, "IoU@0.50 threshold", fontsize=10.5, color=COLORS["red"], va="bottom")
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 1.05)
    ax.set_title("Controlled small-object reliability", loc="left", fontsize=24, fontweight="bold", color=COLORS["ink"], pad=18)
    ax.text(0.01, 0.94, "Graph replacement for the small-object reliability tables. Open markers are reliable rows.", transform=ax.transAxes, fontsize=13, color=COLORS["muted"])
    ax.set_xlabel("Forced target area in pixels, log scale", fontsize=13)
    ax.set_ylabel("Score", fontsize=13)
    ax.grid(True, which="both", axis="y", color=COLORS["grid"], linewidth=1)
    ax.grid(True, which="major", axis="x", color="#eef2f7", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=10.5, ncol=2, loc="lower right")
    ax.text(0.99, 0.03, "Sources: controlled area-stress summary CSVs", transform=ax.transAxes, fontsize=9.5, color=COLORS["muted"], ha="right")
    return save_figure(fig, "controlled_small_object_reliability_graph")


def make_transition_gif(png_paths: Sequence[Path]) -> Path:
    frames: List[Image.Image] = []
    for path in png_paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((1280, 720), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (1280, 720), "white")
        canvas.paste(image, ((1280 - image.width) // 2, (720 - image.height) // 2))
        frames.append(canvas)
    output = ASSET_DIR / "v6_to_v8_architecture_transition.gif"
    if frames:
        frames[0].save(output, save_all=True, append_images=frames[1:], duration=1500, loop=0)
    return output


def make_contact_sheet(png_paths: Sequence[Path]) -> Path:
    thumbs: List[Tuple[Path, Image.Image]] = []
    for path in png_paths:
        if path.suffix.lower() != ".png":
            continue
        image = Image.open(path).convert("RGB")
        image.thumbnail((440, 248), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (460, 305), "white")
        canvas.paste(image, ((460 - image.width) // 2, 14))
        draw = ImageDraw.Draw(canvas)
        draw.text((18, 268), path.name, fill=(31, 41, 51))
        thumbs.append((path, canvas))

    columns = 2
    rows = (len(thumbs) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * 460, max(1, rows) * 305), "white")
    for index, (_, thumb) in enumerate(thumbs):
        x = (index % columns) * 460
        y = (index // columns) * 305
        sheet.paste(thumb, (x, y))
    output = ASSET_DIR / "contact_sheet.png"
    sheet.save(output)
    return output


def write_readme(paths: Sequence[Path], transition_gif: Path, contact_sheet: Path) -> Path:
    lines = [
        "# V6/V8 Visual Asset Pack",
        "",
        "Generated by `scripts/make_v6_v8_presentation_visuals.py`.",
        "",
        "## Architecture Visuals",
        "",
        "- `v6_architecture_gated_sot_memory.png`: explains the V6 gated SOT-memory path.",
        "- `v8_architecture_shared_frame_batched_heads.png`: explains the V8 shared-frame ViT and batched object head path.",
        "- `v6_whole_lorat_v8_head_replacement.png`: clearest main comparison slide for the core architectural change: V6 keeps LoRAT whole; V8 keeps the LoRAT body and replaces the original tracking head.",
        "- `v6_v8_same_vs_different_map.png`: supporting comparison slide for what stayed the same and what changed.",
        "- `v6_to_v8_architecture_changes.png`: slide-ready comparison of what changed between V6 and V8.",
        "- `v6_to_v8_architecture_transition.gif`: lightweight three-frame animation built from the architecture slides.",
        "",
        "## Metric And Graph Replacements",
        "",
        "- `iou_metric_explainer.png`: visual explanation of IoU, IoU@0.50, mean IoU, and reliability bins.",
        "- `week1_v6_ms_per_box_graph.png`: graph replacement for the Week 1 ms/box table.",
        "- `week1_fps_vs_objects_by_version.png`: FPS by object count for V4, V5, and V6.",
        "- `week1_v6_speedup_vs_serial_graph.png`: V5/V6 FPS speedup against the V4 serial baseline.",
        "- `controlled_small_object_reliability_graph.png`: graph replacement for controlled small-object reliability tables.",
        "",
        "## Suggested Placement In The Current PDF Deck",
        "",
        "- Slide 4: replace most metric bullets with `iou_metric_explainer.png`.",
        "- Slide 5: replace the ms/box table with `week1_v6_ms_per_box_graph.png`; use `controlled_small_object_reliability_graph.png` if the slide focuses on small-object limits.",
        "- Slide 6: use `week1_fps_vs_objects_by_version.png` or `week1_v6_speedup_vs_serial_graph.png` for the benchmark trend instead of dense tables.",
        "- Slide 7: use `v6_whole_lorat_v8_head_replacement.png` as the main proof object. Use `v6_v8_same_vs_different_map.png` or `v6_to_v8_architecture_changes.png` as supporting backup visuals.",
        "- Follow with either the V6 or V8 architecture detail slide if time allows.",
        "",
        "## Files Generated",
        "",
    ]
    for path in paths:
        lines.append(f"- `{path.name}`")
    lines.append(f"- `{transition_gif.name}`")
    lines.append(f"- `{contact_sheet.name}`")
    output = ASSET_DIR / "README.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def main() -> int:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    generated: List[Path] = []
    generated.extend(make_v6_architecture())
    generated.extend(make_v8_architecture())
    generated.extend(make_lorat_head_replacement_visual())
    generated.extend(make_v6_v8_same_vs_different())
    generated.extend(make_v6_to_v8_changes())
    generated.extend(make_iou_explainer())
    generated.extend(chart_week1_v6_ms_per_box())
    generated.extend(chart_week1_fps_versions())
    generated.extend(chart_week1_speedup())
    generated.extend(chart_controlled_small_object_reliability())

    pngs = [path for path in generated if path.suffix.lower() == ".png"]
    transition = make_transition_gif(
        [
            ASSET_DIR / "v6_architecture_gated_sot_memory.png",
            ASSET_DIR / "v6_whole_lorat_v8_head_replacement.png",
            ASSET_DIR / "v8_architecture_shared_frame_batched_heads.png",
        ]
    )
    contact_sheet = make_contact_sheet(pngs)
    readme = write_readme(generated, transition, contact_sheet)

    print(f"Wrote {len(generated)} static assets")
    print(f"Wrote {transition}")
    print(f"Wrote {contact_sheet}")
    print(f"Wrote {readme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
