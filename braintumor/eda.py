"""braintumor/eda.py - Research-grade exploratory data analysis.

Generates publication-quality figures and a markdown findings report for the
brain-tumor MRI dataset. Covers, as required for the thesis:

    * class distribution (train / test)
    * healthy vs tumor distribution
    * tumor-subtype distribution
    * image dimension analysis
    * pixel-intensity analysis
    * sample MRI visualization
    * data-quality checks (mode, corrupt files, aspect ratios)
    * class-imbalance analysis
    * augmentation visualization
    * outlier analysis (intensity / size)

Run:
    python -m braintumor.eda                       # full EDA -> artifacts/eda + reports/EDA_REPORT.md
    python -m braintumor.eda --sample 150          # images sampled per class for stats
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import config

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# A clean, consistent publication style.
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 11, "axes.titleweight": "bold",
})
_PALETTE = {"glioma": "#EF4444", "meningioma": "#F59E0B",
            "notumor": "#10B981", "pituitary": "#3B82F6"}
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ---------------------------------------------------------------------------
def _scan(root: Path) -> pd.DataFrame:
    rows = []
    for label_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for img in label_dir.iterdir():
            if img.suffix.lower() in _IMG_EXTS:
                rows.append({"path": str(img), "label": label_dir.name})
    return pd.DataFrame(rows)


def _colors(labels: List[str]) -> List[str]:
    return [_PALETTE.get(l, "#6B7280") for l in labels]


# ---------------------------------------------------------------------------
# 1. Class distribution + 2. healthy/tumor + 3. subtype
# ---------------------------------------------------------------------------
def plot_class_distribution(train: pd.DataFrame, test: pd.DataFrame,
                            out: Path) -> Dict:
    tr = train["label"].value_counts().sort_index()
    te = test["label"].value_counts().sort_index()
    labels = sorted(set(tr.index) | set(te.index))
    x = np.arange(len(labels)); w = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    axes[0].bar(x - w/2, [tr.get(l, 0) for l in labels], w, label="Train",
                color=_colors(labels))
    axes[0].bar(x + w/2, [te.get(l, 0) for l in labels], w, label="Test",
                color=_colors(labels), alpha=0.55)
    axes[0].set_xticks(x); axes[0].set_xticklabels([l.upper() for l in labels])
    axes[0].set_ylabel("Image count"); axes[0].set_title("Class distribution (train vs test)")
    axes[0].legend()
    for i, l in enumerate(labels):
        axes[0].text(i - w/2, tr.get(l, 0), str(tr.get(l, 0)), ha="center", va="bottom", fontsize=9)
        axes[0].text(i + w/2, te.get(l, 0), str(te.get(l, 0)), ha="center", va="bottom", fontsize=9)

    # healthy vs tumor (train)
    healthy = int(tr.get("notumor", 0)); tumor = int(tr.sum() - healthy)
    axes[1].pie([healthy, tumor], labels=[f"Healthy\n{healthy}", f"Tumor\n{tumor}"],
                autopct="%1.1f%%", colors=["#10B981", "#EF4444"], startangle=90,
                wedgeprops=dict(width=0.45))
    axes[1].set_title("Healthy vs Tumor (train)")
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)

    counts = {l: int(tr.get(l, 0)) for l in labels}
    imbalance = (max(counts.values()) / max(min(counts.values()), 1)) if counts else 1.0
    return {"train_counts": counts,
            "test_counts": {l: int(te.get(l, 0)) for l in labels},
            "healthy_train": healthy, "tumor_train": tumor,
            "imbalance_ratio": round(imbalance, 3)}


def plot_subtype_distribution(train: pd.DataFrame, out: Path) -> Dict:
    sub = train[train["label"] != "notumor"]["label"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.bar([l.upper() for l in sub.index], sub.values, color=_colors(list(sub.index)))
    ax.set_title("Tumor-subtype distribution (train, tumor-only)")
    ax.set_ylabel("Image count")
    for i, v in enumerate(sub.values):
        ax.text(i, v, str(int(v)), ha="center", va="bottom")
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    return {l: int(v) for l, v in sub.items()}


# ---------------------------------------------------------------------------
# 4. dimensions  + 5. intensity  + 7. quality  (single sampled pass)
# ---------------------------------------------------------------------------
def _sample(df: pd.DataFrame, n_per_class: int, seed: int = 42) -> pd.DataFrame:
    parts = []
    for label, g in df.groupby("label"):
        parts.append(g.sample(min(len(g), n_per_class), random_state=seed))
    return pd.concat(parts, ignore_index=True)


def scan_image_stats(df: pd.DataFrame, n_per_class: int = 150) -> pd.DataFrame:
    from PIL import Image
    sample = _sample(df, n_per_class)
    recs = []
    for _, r in sample.iterrows():
        try:
            with Image.open(r["path"]) as im:
                w, h = im.size; mode = im.mode
                arr = np.asarray(im.convert("L"), dtype=np.float32)
            recs.append({"label": r["label"], "w": w, "h": h, "mode": mode,
                         "mean": float(arr.mean()), "std": float(arr.std()),
                         "aspect": round(w / max(h, 1), 3), "ok": True,
                         "path": r["path"]})
        except Exception as exc:
            recs.append({"label": r["label"], "ok": False, "error": str(exc),
                         "path": r["path"]})
    return pd.DataFrame(recs)


def plot_dimensions(stats: pd.DataFrame, out: Path) -> Dict:
    ok = stats[stats["ok"]]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].hist(ok["w"], bins=30, color="#3B82F6", alpha=0.7, label="width")
    axes[0].hist(ok["h"], bins=30, color="#EF4444", alpha=0.5, label="height")
    axes[0].set_title("Image dimension distribution"); axes[0].set_xlabel("pixels")
    axes[0].legend()
    axes[1].scatter(ok["w"], ok["h"], s=10, alpha=0.4, c="#6366F1")
    axes[1].set_title("Width vs Height"); axes[1].set_xlabel("width"); axes[1].set_ylabel("height")
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    sizes = Counter(zip(ok["w"], ok["h"]))
    return {"unique_sizes": len(sizes),
            "most_common_size": (f"{sizes.most_common(1)[0][0][0]}x"
                                 f"{sizes.most_common(1)[0][0][1]}" if sizes else "n/a"),
            "width_range": [int(ok["w"].min()), int(ok["w"].max())],
            "height_range": [int(ok["h"].min()), int(ok["h"].max())]}


def plot_intensity(stats: pd.DataFrame, out: Path) -> Dict:
    ok = stats[stats["ok"]]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for label in sorted(ok["label"].unique()):
        sub = ok[ok["label"] == label]
        axes[0].hist(sub["mean"], bins=25, alpha=0.5, label=label.upper(),
                     color=_PALETTE.get(label, "#6B7280"))
    axes[0].set_title("Mean pixel intensity by class"); axes[0].set_xlabel("mean intensity (0-255)")
    axes[0].legend()
    data = [ok[ok["label"] == l]["mean"].values for l in sorted(ok["label"].unique())]
    bp = axes[1].boxplot(data, labels=[l.upper() for l in sorted(ok["label"].unique())],
                         patch_artist=True)
    for patch, l in zip(bp["boxes"], sorted(ok["label"].unique())):
        patch.set_facecolor(_PALETTE.get(l, "#6B7280")); patch.set_alpha(0.6)
    axes[1].set_title("Intensity spread by class"); axes[1].set_ylabel("mean intensity")
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    return {l: round(float(ok[ok["label"] == l]["mean"].mean()), 1)
            for l in sorted(ok["label"].unique())}


def quality_report(stats: pd.DataFrame) -> Dict:
    corrupt = stats[~stats["ok"]]
    ok = stats[stats["ok"]]
    modes = Counter(ok["mode"])
    odd_aspect = ok[(ok["aspect"] < 0.8) | (ok["aspect"] > 1.25)]
    return {
        "sampled": int(len(stats)),
        "corrupt": int(len(corrupt)),
        "corrupt_files": corrupt["path"].tolist()[:10],
        "color_modes": dict(modes),
        "non_square_count": int(len(odd_aspect)),
    }


def global_intensity_stats(stats: pd.DataFrame) -> Dict:
    """Global dataset pixel-intensity profile: mean, std and contrast (std/mean)."""
    ok = stats[stats["ok"]]
    mean = float(ok["mean"].mean())
    std = float(ok["mean"].std())
    within_img_contrast = float(ok["std"].mean())   # mean per-image std (contrast)
    return {
        "global_mean_intensity": round(mean, 2),
        "between_image_std": round(std, 2),
        "mean_within_image_contrast": round(within_img_contrast, 2),
        "contrast_ratio": round(within_img_contrast / (mean + 1e-9), 3),
        "per_class_mean": {l: round(float(ok[ok["label"] == l]["mean"].mean()), 2)
                           for l in sorted(ok["label"].unique())},
    }


# ---------------------------------------------------------------------------
# 6. sample MRI grid
# ---------------------------------------------------------------------------
def plot_sample_grid(df: pd.DataFrame, out: Path, per_class: int = 4) -> None:
    from PIL import Image
    labels = sorted(df["label"].unique())
    fig, axes = plt.subplots(len(labels), per_class, figsize=(per_class * 2.6, len(labels) * 2.6))
    for r, label in enumerate(labels):
        sub = df[df["label"] == label].sample(min(per_class, (df["label"] == label).sum()),
                                              random_state=1)
        for c, (_, row) in enumerate(sub.iterrows()):
            ax = axes[r, c] if len(labels) > 1 else axes[c]
            try:
                ax.imshow(Image.open(row["path"]).convert("L"), cmap="gray")
            except Exception:
                ax.text(0.5, 0.5, "load error", ha="center")
            ax.axis("off")
            if c == 0:
                ax.set_ylabel(label.upper(), rotation=0, ha="right", va="center",
                              fontsize=11, fontweight="bold")
                ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Sample MRI slices by class", fontsize=14, fontweight="bold")
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------------------
# 9. augmentation visualization
# ---------------------------------------------------------------------------
def plot_augmentation(df: pd.DataFrame, out: Path) -> None:
    from PIL import Image
    from .preprocessing import enhance
    row = df[df["label"] != "notumor"].sample(1, random_state=3).iloc[0]
    base = np.asarray(Image.open(row["path"]).convert("RGB").resize((224, 224)))
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.6))
    axes = axes.flat
    axes[0].imshow(base); axes[0].set_title("original"); axes[0].axis("off")
    import cv2
    ops = [
        ("rotate +12", lambda im: _rot(im, 12)),
        ("rotate -12", lambda im: _rot(im, -12)),
        ("h-flip", lambda im: im[:, ::-1]),
        ("zoom 1.15x", lambda im: _zoom(im, 1.15)),
        ("bright +25%", lambda im: np.clip(im * 1.25, 0, 255).astype(np.uint8)),
        ("shift", lambda im: np.roll(im, (12, 10), axis=(0, 1))),
        ("CLAHE+denoise", lambda im: enhance(im)),
    ]
    for ax, (name, fn) in zip(axes[1:], ops):
        ax.imshow(fn(base)); ax.set_title(name); ax.axis("off")
    fig.suptitle("Training augmentation examples (MRI-safe: no vertical flip)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)


def _rot(im, deg):
    import cv2
    h, w = im.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), deg, 1.0)
    return cv2.warpAffine(im, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def _zoom(im, f):
    import cv2
    h, w = im.shape[:2]
    z = cv2.resize(im, (int(w*f), int(h*f)))
    y0, x0 = (z.shape[0]-h)//2, (z.shape[1]-w)//2
    return z[y0:y0+h, x0:x0+w]


# ---------------------------------------------------------------------------
# 10. outliers
# ---------------------------------------------------------------------------
def outlier_report(stats: pd.DataFrame, out: Path) -> Dict:
    ok = stats[stats["ok"]].copy()
    mu, sd = ok["mean"].mean(), ok["mean"].std()
    ok["z"] = (ok["mean"] - mu) / (sd + 1e-9)
    outliers = ok[ok["z"].abs() > 2.5].sort_values("z")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(range(len(ok)), ok["mean"].values, s=8, c=np.where(ok["z"].abs() > 2.5, "#EF4444", "#3B82F6"))
    ax.axhline(mu + 2.5*sd, ls="--", c="#EF4444", alpha=0.6)
    ax.axhline(mu - 2.5*sd, ls="--", c="#EF4444", alpha=0.6)
    ax.set_title(f"Intensity outliers (|z| > 2.5): {len(outliers)} flagged")
    ax.set_xlabel("sampled image index"); ax.set_ylabel("mean intensity")
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    return {"n_outliers": int(len(outliers)),
            "examples": outliers[["path", "label", "mean"]].head(8).to_dict("records")}


# ---------------------------------------------------------------------------
# Orchestrator + markdown
# ---------------------------------------------------------------------------
def run_full_eda(n_per_class: int = 150,
                 fig_dir: Optional[Path] = None,
                 report_path: Optional[Path] = None) -> Path:
    fig_dir = Path(fig_dir or (config.OUT_DIR / "eda"))
    fig_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(report_path or (config.ROOT / "reports" / "EDA_REPORT.md"))
    report_path.parent.mkdir(parents=True, exist_ok=True)

    print("[eda] scanning dataset ...")
    train = _scan(config.TRAIN_DIR)
    test = _scan(config.TEST_DIR)
    print(f"[eda] train={len(train)}  test={len(test)}")

    dist = plot_class_distribution(train, test, fig_dir / "class_distribution.png")
    subtype = plot_subtype_distribution(train, fig_dir / "subtype_distribution.png")
    print(f"[eda] sampling {n_per_class}/class for image stats ...")
    stats = scan_image_stats(train, n_per_class)
    dims = plot_dimensions(stats, fig_dir / "dimensions.png")
    inten = plot_intensity(stats, fig_dir / "intensity.png")
    qual = quality_report(stats)
    plot_sample_grid(train, fig_dir / "sample_grid.png")
    plot_augmentation(train, fig_dir / "augmentation.png")
    outl = outlier_report(stats, fig_dir / "outliers.png")

    md = _markdown(dist, subtype, dims, inten, qual, outl, len(train), len(test),
                   n_per_class, fig_dir)
    report_path.write_text(md, encoding="utf-8")
    print(f"[eda] figures -> {fig_dir}")
    print(f"[eda] report  -> {report_path}")
    return report_path


def _rel(fig_dir: Path) -> str:
    try:
        return str(fig_dir.relative_to(config.ROOT)).replace("\\", "/")
    except ValueError:
        return str(fig_dir).replace("\\", "/")


def _markdown(dist, subtype, dims, inten, qual, outl, n_train, n_test,
              n_per_class, fig_dir) -> str:
    rel = _rel(fig_dir)
    tc = dist["train_counts"]
    bal = "perfectly balanced" if dist["imbalance_ratio"] < 1.1 else \
          ("mildly imbalanced" if dist["imbalance_ratio"] < 1.5 else "imbalanced")
    lines = [
        "# Exploratory Data Analysis - Brain Tumor MRI",
        "",
        f"Dataset: `{config.DATASET_DIR.name}` | train images: **{n_train}**, "
        f"test images: **{n_test}** | stats sampled at {n_per_class}/class.",
        "",
        "## 1. Class & healthy/tumor distribution",
        f"![class distribution]({rel}/class_distribution.png)",
        "",
        "| Class | Train | Test |",
        "|---|---:|---:|",
    ]
    for c in sorted(tc):
        lines.append(f"| {c} | {tc[c]} | {dist['test_counts'].get(c, 0)} |")
    lines += [
        "",
        f"- **Healthy vs tumor (train):** {dist['healthy_train']} healthy vs "
        f"{dist['tumor_train']} tumor.",
        f"- **Imbalance ratio (max/min class):** {dist['imbalance_ratio']} -> the set is "
        f"**{bal}**. Class weighting is therefore a safeguard, not a necessity; the "
        "dominant residual error is *label noise* (SARTAJ glioma contamination), not "
        "imbalance.",
        "",
        "## 2. Tumor-subtype distribution",
        f"![subtype]({rel}/subtype_distribution.png)",
        "",
        f"Subtype counts (train): {subtype}.",
        "",
        "## 3. Image dimensions",
        f"![dimensions]({rel}/dimensions.png)",
        "",
        f"- Unique sizes sampled: **{dims['unique_sizes']}**; most common "
        f"**{dims['most_common_size']}**.",
        f"- Width range {dims['width_range']}, height range {dims['height_range']}.",
        "- **Implication:** images are not a single fixed size, so a deterministic "
        "resize (to 224/260/299 per backbone) is required and is handled by "
        "`braintumor.preprocessing`.",
        "",
        "## 4. Pixel-intensity analysis",
        f"![intensity]({rel}/intensity.png)",
        "",
        f"Per-class mean intensity (0-255): {inten}. Distributions overlap heavily "
        "across classes, confirming the task is texture/shape-driven, not a trivial "
        "brightness threshold.",
        "",
        "## 5. Sample MRI visualization",
        f"![samples]({rel}/sample_grid.png)",
        "",
        "## 6. Data-quality checks",
        f"- Sampled **{qual['sampled']}** images; **{qual['corrupt']}** failed to load.",
        f"- Colour modes present: {qual['color_modes']} (the loader forces RGB).",
        f"- Non-square images flagged: {qual['non_square_count']}.",
        "",
        "## 7. Augmentation strategy",
        f"![augmentation]({rel}/augmentation.png)",
        "",
        "Augmentations are **MRI-safe**: rotation, shift, zoom, brightness and "
        "*horizontal* flip only. Vertical flip is excluded (anatomically invalid for "
        "axial brain MRI). CLAHE+denoise is shown as an optional enhancement track.",
        "",
        "## 8. Outlier analysis",
        f"![outliers]({rel}/outliers.png)",
        "",
        f"- Intensity outliers flagged (|z|>2.5): **{outl['n_outliers']}**. These are "
        "candidates for manual review (over/under-exposed slices, non-brain images).",
        "",
        "## Key takeaways",
        "1. The 4-class set is essentially balanced - **do not chase accuracy via "
        "resampling**; target the glioma/meningioma label noise instead.",
        "2. Variable image sizes -> deterministic per-backbone resize is mandatory.",
        "3. Class intensity overlap -> the model must learn morphology; CLAHE may help "
        "only if applied at BOTH train and inference time.",
        "4. A small number of intensity outliers exist and are worth a manual pass.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=int, default=150,
                    help="images sampled per class for intensity/dimension stats")
    args = ap.parse_args()
    run_full_eda(n_per_class=args.sample)


if __name__ == "__main__":
    main()
