from __future__ import annotations

import ast
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
REPORT_DIR = ROOT / "docs" / "final_report" / "my_report"
GRAPHICS_DIR = REPORT_DIR / "graphics"

COLORS = {
    "native BFGS": "#1f77b4",
    "SciPy BFGS": "#ff7f0e",
    "SGD": "#d62728",
    "Momentum": "#9467bd",
    "Adam": "#2ca02c",
    "Muon pad sqrt": "#e377c2",
    "Muon factorization": "#17becf",
    "MTP": "#1f77b4",
    "ETN": "#2ca02c",
}

ORDER = [
    "native BFGS",
    "SciPy BFGS",
    "SGD",
    "Momentum",
    "Adam",
    "Muon pad sqrt",
    "Muon factorization",
]

MINIBATCH_ORDER = ["SGD", "Momentum", "Adam", "Muon pad sqrt", "Muon factorization"]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 260,
            "font.family": "DejaVu Sans",
            "axes.titlesize": 17,
            "axes.titleweight": "bold",
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
            "axes.edgecolor": "#9aa4ad",
            "axes.linewidth": 1.05,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "grid.color": "#d9dee3",
            "grid.linewidth": 0.9,
        }
    )


def save(fig: plt.Figure, filename: str) -> None:
    GRAPHICS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(GRAPHICS_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(
        columns={
            "train_energy_atom_rmse": "train_epa_rmse",
            "val_energy_atom_rmse": "val_epa_rmse",
        }
    )


def load_csv(path: str | Path, model: str, dataset: str, optimizer: str) -> pd.DataFrame:
    df = normalize_columns(pd.read_csv(ROOT / path))
    df["model"] = model
    df["dataset"] = dataset
    df["optimizer"] = optimizer
    return df


def read_param_csv(path: str | Path, param_name: str) -> pd.DataFrame:
    cols = [
        "train_energy_rmse",
        "train_energy_atom_rmse",
        "train_forces_rmse",
        "val_energy_rmse",
        "val_energy_atom_rmse",
        "val_forces_rmse",
        "train_time",
    ]
    rows = []
    with (ROOT / path).open(encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            if not line.strip():
                continue
            left, *metrics = line.strip().rsplit(",", 7)
            pot_num, param_value = left.split(",", 1)
            row = {"pot_num": int(pot_num), param_name: param_value}
            row.update({key: float(value) for key, value in zip(cols, metrics)})
            rows.append(row)
    return normalize_columns(pd.DataFrame(rows))


def summary(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "train_epa_rmse",
        "train_forces_rmse",
        "val_epa_rmse",
        "val_forces_rmse",
        "train_time",
        "nit",
        "success",
    ]
    rows = []
    for optimizer, part in df.groupby("optimizer", sort=False):
        row = {"optimizer": optimizer}
        for metric in metrics:
            if metric in part.columns:
                values = pd.to_numeric(part[metric], errors="coerce").dropna()
                if not values.empty:
                    row[f"{metric}_mean"] = float(values.mean())
                    row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    out = pd.DataFrame(rows)
    order_map = {name: i for i, name in enumerate(ORDER)}
    out["_order"] = out["optimizer"].map(lambda name: order_map.get(str(name), len(order_map)))
    return out.sort_values(["_order", "optimizer"]).drop(columns="_order").reset_index(drop=True)


def parse_history(value) -> list[float]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return [float(v) for v in value if np.isfinite(float(v))]
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = ast.literal_eval(value)
    except Exception:
        return []
    out = []
    for item in parsed:
        try:
            val = float(item)
        except Exception:
            continue
        if np.isfinite(val):
            out.append(val)
    return out


def rolling(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) == 0:
        return values
    if window <= 1:
        return values
    return pd.Series(values).rolling(window=window, min_periods=1, center=True).mean().to_numpy()


def history_stats(df: pd.DataFrame, column: str = "losses", window: int = 151) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    runs = [parse_history(v) for v in df[column].dropna()] if column in df.columns else []
    runs = [r for r in runs if r]
    if not runs:
        return np.array([]), np.array([]), np.array([])
    length = min(len(r) for r in runs)
    arr = np.array([r[:length] for r in runs], dtype=float)
    mean = rolling(arr.mean(axis=0), window)
    std = rolling(arr.std(axis=0, ddof=1), window) if arr.shape[0] > 1 else np.zeros(length)
    return np.arange(length), mean, std


def ordered(summary_df: pd.DataFrame) -> pd.DataFrame:
    present = [m for m in ORDER if m in set(summary_df["optimizer"].astype(str))]
    return summary_df.set_index("optimizer").loc[present].reset_index()


def fmt_value(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) < 0.001 or abs(value) >= 1000:
        return f"{value:.2e}"
    if abs(value) < 0.01:
        return f"{value:.4f}"
    if abs(value) < 1:
        return f"{value:.3f}"
    return f"{value:.1f}"


def add_grid(ax: plt.Axes) -> None:
    ax.grid(True, which="major", axis="both", alpha=0.85)
    ax.grid(True, which="minor", axis="x", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def lollipop_validation(summary_df: pd.DataFrame, title: str, filename: str) -> None:
    data = ordered(summary_df)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2), sharey=True)
    specs = [
        ("val_epa_rmse", "validation EPA RMSE (eV/atom)"),
        ("val_forces_rmse", "validation forces RMSE (eV/Angstrom)"),
    ]
    y = np.arange(len(data))[::-1]
    labels = data["optimizer"].astype(str).tolist()

    for ax, (metric, label) in zip(axes, specs):
        means = data[f"{metric}_mean"].to_numpy(float)
        colors = [COLORS.get(m, "#333333") for m in labels]
        xmin = max(means.min() * 0.65, 1e-12)
        xmax = means.max() * 1.9
        for yi, mean, color in zip(y, means, colors):
            ax.hlines(yi, xmin, mean, color=color, lw=7, alpha=0.32)
            ax.scatter(mean, yi, s=90, color=color, edgecolor="white", linewidth=1.1, zorder=3)
            ax.text(mean * 1.08, yi, fmt_value(mean), va="center", ha="left", fontsize=12, color="#222222")
        ax.set_xscale("log")
        ax.set_xlim(xmin, xmax)
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("RMSE, log scale")
        ax.set_title(label)
        ax.set_yticks(y, labels)
        add_grid(ax)
    fig.suptitle(title, fontsize=20, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, filename)


def spread_bars(summary_df: pd.DataFrame, title: str, filename: str) -> None:
    data = ordered(summary_df)
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 8.4))
    specs = [
        ("val_epa_rmse_std", "validation EPA RMSE std (eV/atom)"),
        ("val_forces_rmse_std", "validation forces RMSE std (eV/Angstrom)"),
    ]
    labels = data["optimizer"].astype(str).tolist()
    display_labels = [
        {
            "native BFGS": "native\nBFGS",
            "SciPy BFGS": "SciPy\nBFGS",
            "Muon pad sqrt": "Muon pad\nsqrt",
            "Muon factorization": "Muon\nfactorization",
        }.get(label, label)
        for label in labels
    ]
    x = np.arange(len(labels))
    colors = [COLORS.get(m, "#333333") for m in labels]
    for ax, (metric, label) in zip(axes, specs):
        vals = data.get(metric, pd.Series(np.zeros(len(data)))).fillna(0).to_numpy(float)
        positive = vals[vals > 0]
        floor = max(positive.min() * 0.35, 1e-14) if len(positive) else 1e-12
        plot_vals = np.where(vals > 0, vals, floor)
        ax.bar(x, plot_vals, color=colors, alpha=0.82)
        ymax = max(plot_vals.max() * 4.5, floor * 10)
        for idx, (xi, val, plot_val) in enumerate(zip(x, vals, plot_vals)):
            multiplier = 1.28 if idx % 2 == 0 else 1.85
            ax.text(xi, min(plot_val * multiplier, ymax / 1.15), fmt_value(val), va="bottom", ha="center", fontsize=12)
        ax.set_yscale("log")
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_ylabel("std, log scale")
        ax.set_title(label)
        ax.set_xticks(x, display_labels, rotation=0, ha="center")
        ax.tick_params(axis="x", labelsize=12)
        ax.set_ylim(floor, ymax)
        add_grid(ax)
    fig.suptitle(title, fontsize=20, fontweight="bold", y=1.0)
    fig.tight_layout()
    save(fig, filename)


def quality_time(summary_df: pd.DataFrame, title: str, filename: str) -> None:
    data = ordered(summary_df)
    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    data = data.sort_values(["train_time_mean", "val_epa_rmse_mean"]).reset_index(drop=True)
    yvals = data["val_epa_rmse_mean"].to_numpy(float)

    for _, row in data.iterrows():
        opt = str(row["optimizer"])
        x = float(row["train_time_mean"])
        y = float(row["val_epa_rmse_mean"])
        ax.scatter(
            x,
            y,
            s=115,
            color=COLORS.get(opt, "#333333"),
            edgecolor="white",
            linewidth=1.1,
            zorder=3,
            label=opt,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Train time (s), log scale")
    ax.set_ylabel("validation EPA RMSE (eV/atom), log scale")
    ax.set_title(title)
    xvals = data["train_time_mean"].to_numpy(float)
    ax.set_xlim(xvals.min() * 0.65, xvals.max() * 1.35)
    ax.set_ylim(yvals.min() * 0.55, yvals.max() * 1.7)
    handles, labels = ax.get_legend_handles_labels()
    handle_by_label = dict(zip(labels, handles))
    legend_labels = [name for name in ORDER if name in handle_by_label]
    ax.legend(
        [handle_by_label[name] for name in legend_labels],
        legend_labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        title="Optimizer",
    )
    add_grid(ax)
    fig.tight_layout()
    save(fig, filename)


def loss_band(df: pd.DataFrame, optimizer: str, title: str, filename: str, window: int = 151) -> None:
    part = df[df["optimizer"] == optimizer]
    x, mean, std = history_stats(part, "losses", window=window)
    if len(x) == 0:
        return
    color = COLORS.get(optimizer, "#d62728")
    lower = np.maximum(mean - std, np.maximum(mean * 0.05, 1e-12))
    upper = mean + std
    fig, ax = plt.subplots(figsize=(10.6, 5.4))
    ax.fill_between(x, lower, upper, color=color, alpha=0.18, lw=0)
    ax.plot(x, mean, color=color, lw=3.0)
    ax.set_yscale("log")
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Loss, log scale")
    ax.set_title(title)
    ax.text(0.97, 0.93, "mean ± std, 5 runs", transform=ax.transAxes, ha="right", va="top", fontsize=12, color="#333333")
    add_grid(ax)
    fig.tight_layout()
    save(fig, filename)


def loss_overview(df: pd.DataFrame, title: str, filename: str, window: int = 151) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 6.0))
    for opt in MINIBATCH_ORDER:
        if opt not in set(df["optimizer"].astype(str)):
            continue
        x, mean, _ = history_stats(df[df["optimizer"] == opt], "losses", window=window)
        if len(x) == 0:
            continue
        ax.plot(x, mean, lw=2.8, label=opt, color=COLORS.get(opt))
    ax.set_yscale("log")
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Loss, log scale")
    ax.set_title(title)
    ax.legend(loc="best", frameon=False)
    add_grid(ax)
    fig.tight_layout()
    save(fig, filename)


def simple_error_bars(df: pd.DataFrame, title: str, filename: str, methods: list[str] | None = None) -> None:
    data = summary(df)
    if methods is not None:
        data = data[data["optimizer"].astype(str).isin(methods)]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.6))
    specs = [
        ("val_epa_rmse", "validation EPA RMSE (eV/atom)"),
        ("val_forces_rmse", "validation forces RMSE (eV/Angstrom)"),
    ]
    labels = data["optimizer"].astype(str).tolist()
    x = np.arange(len(labels))
    colors = [COLORS.get(m, "#333333") for m in labels]
    for ax, (metric, label) in zip(axes, specs):
        vals = data[f"{metric}_mean"].to_numpy(float)
        ax.bar(x, vals, color=colors, alpha=0.78)
        ymax = vals.max() * 1.22 if vals.max() > 0 else 1.0
        for xi, val in zip(x, vals):
            ax.text(xi, val + ymax * 0.025, fmt_value(val), va="bottom", ha="center", fontsize=12)
        ax.set_xticks(x, labels, rotation=18, ha="right")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.set_ylim(0, ymax)
        add_grid(ax)
    fig.suptitle(title, fontsize=20, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, filename)


def mtp_etn_quality_time(mtp_native: pd.DataFrame, etn_native: pd.DataFrame) -> None:
    data = pd.concat([mtp_native, etn_native], ignore_index=True)
    data["optimizer"] = data["model"]
    sm = summary(data)
    sm["optimizer"] = sm["optimizer"].astype(str)
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 5.4))
    specs = [
        ("val_epa_rmse", "validation EPA RMSE\n(eV/atom)"),
        ("val_forces_rmse", "validation forces RMSE\n(eV/Angstrom)"),
        ("train_time", "train time\n(s)"),
    ]
    labels = sm["optimizer"].tolist()
    x = np.arange(len(labels))
    colors = [COLORS.get(m, "#333333") for m in labels]
    for ax, (metric, label) in zip(axes, specs):
        vals = sm[f"{metric}_mean"].to_numpy(float)
        ax.bar(x, vals, color=colors, alpha=0.78)
        ymax = vals.max() * 1.22 if vals.max() > 0 else 1.0
        for xi, val in zip(x, vals):
            ax.text(xi, val + ymax * 0.025, fmt_value(val), va="bottom", ha="center", fontsize=12)
        ax.set_xticks(x, labels)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.set_ylim(0, ymax)
        add_grid(ax)
    fig.suptitle("MTP and ETN native BFGS on AgPd", fontsize=20, fontweight="bold", y=1.04)
    fig.tight_layout()
    save(fig, "fig_05_01_mtp_etn_quality_time_agpd.png")


def etn_tuning() -> None:
    ranks = read_param_csv("task1/results/results_etn_ranks.csv", "rank")
    shapes = read_param_csv("task1/results/results_etn_shape.csv", "shape")

    for df, param, title, filename in [
        (ranks, "rank", "ETN rank tuning on AgPd", "fig_05_02_etn_rank_tuning_agpd.png"),
        (shapes, "shape", "ETN shape tuning on AgPd", "fig_05_03_etn_shape_tuning_agpd.png"),
    ]:
        sm = df.groupby(param).agg(
            val_epa_rmse_mean=("val_epa_rmse", "mean"),
            val_epa_rmse_std=("val_epa_rmse", "std"),
            val_forces_rmse_mean=("val_forces_rmse", "mean"),
            val_forces_rmse_std=("val_forces_rmse", "std"),
        ).reset_index()
        fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.7))
        labels = sm[param].astype(str).tolist()
        x = np.arange(len(labels))
        for ax, metric, label in [
            (axes[0], "val_epa_rmse", "validation EPA RMSE (eV/atom)"),
            (axes[1], "val_forces_rmse", "validation forces RMSE (eV/Angstrom)"),
        ]:
            vals = sm[f"{metric}_mean"].to_numpy(float)
            ax.bar(x, vals, color="#1f77b4", alpha=0.78)
            ymax = vals.max() * 1.22 if vals.max() > 0 else 1.0
            for xi, val in zip(x, vals):
                ax.text(xi, val + ymax * 0.025, fmt_value(val), va="bottom", ha="center", fontsize=12)
            ax.set_xticks(x, labels, rotation=16, ha="right")
            ax.set_ylabel(label)
            ax.set_title(label)
            ax.set_ylim(0, ymax)
            add_grid(ax)
        fig.suptitle(title, fontsize=20, fontweight="bold", y=1.04)
        fig.tight_layout()
        save(fig, filename)


def bfgs_diagnostics(df: pd.DataFrame, title: str, filename: str) -> None:
    simple_error_bars(df, title, filename, methods=["native BFGS", "SciPy BFGS"])


def make_all() -> None:
    setup_style()

    mtp_native = load_csv("all_mtp_5000/results/AgPd_results_mtp_native_bfgs.csv", "MTP", "AgPd", "native BFGS")
    mtp_scipy = load_csv("all_mtp_5000/results/AgPd_results_mtp_scipy_bfgs.csv", "MTP", "AgPd", "SciPy BFGS")
    mtp_adam = load_csv("all_mtp_8000/results/AgPd_results_mtp_my_adam.csv", "MTP", "AgPd", "Adam")
    mtp_momentum = load_csv("all_mtp_8000/results/AgPd_results_mtp_my_momentum.csv", "MTP", "AgPd", "Momentum")
    mtp_sgd = load_csv("all_mtp_8000/results/AgPd_results_mtp_my_sgd.csv", "MTP", "AgPd", "SGD")
    mtp_muon_pad = load_csv("all_mtp_8000/results/AgPd_results_mtp_my_muon_pad_sqrt.csv", "MTP", "AgPd", "Muon pad sqrt")
    mtp_muon_factor = load_csv("all_mtp_8000/results/AgPd_results_mtp_my_muon_factorization.csv", "MTP", "AgPd", "Muon factorization")
    mtp = pd.concat([mtp_native, mtp_scipy, mtp_sgd, mtp_momentum, mtp_adam, mtp_muon_pad, mtp_muon_factor], ignore_index=True)
    mtp_sm = summary(mtp)

    etn_agpd_native = load_csv("all_etn_5000/results/AgPd_results_etn_native_bfgs.csv", "ETN", "AgPd", "native BFGS")
    etn_agpd_scipy = load_csv("all_etn_5000/results/AgPd_results_etn_scipy_bfgs.csv", "ETN", "AgPd", "SciPy BFGS")
    etn_agpd_sgd = load_csv("all_etn_5000/results/AgPd_results_etn_my_sgd.csv", "ETN", "AgPd", "SGD")
    etn_agpd_momentum = load_csv("all_etn_5000/results/AgPd_results_etn_my_momentum.csv", "ETN", "AgPd", "Momentum")
    etn_agpd_adam = load_csv("all_etn_10000/results/AgPd_results_etn_my_adam.csv", "ETN", "AgPd", "Adam")
    etn_agpd_muon_pad = load_csv("all_etn_10000/results/AgPd_results_etn_my_muon_pad_sqrt.csv", "ETN", "AgPd", "Muon pad sqrt")
    etn_agpd_muon_factor = load_csv("all_etn_10000/results/AgPd_results_etn_my_muon_factorization.csv", "ETN", "AgPd", "Muon factorization")
    etn_agpd = pd.concat(
        [etn_agpd_native, etn_agpd_scipy, etn_agpd_sgd, etn_agpd_momentum, etn_agpd_adam, etn_agpd_muon_pad, etn_agpd_muon_factor],
        ignore_index=True,
    )
    etn_agpd_sm = summary(etn_agpd)

    etn_mo = pd.concat(
        [
            load_csv("all_etn_10000/results/MoNbTaVW_results_etn_native_bfgs.csv", "ETN", "MoNbTaVW", "native BFGS"),
            load_csv("all_etn_10000/results/MoNbTaVW_results_etn_scipy_bfgs.csv", "ETN", "MoNbTaVW", "SciPy BFGS"),
            load_csv("all_etn_10000/results/MoNbTaVW_results_etn_my_sgd.csv", "ETN", "MoNbTaVW", "SGD"),
            load_csv("all_etn_10000/results/MoNbTaVW_results_etn_my_momentum.csv", "ETN", "MoNbTaVW", "Momentum"),
            load_csv("all_etn_10000/results/MoNbTaVW_results_etn_my_adam.csv", "ETN", "MoNbTaVW", "Adam"),
            load_csv("all_etn_10000/results/MoNbTaVW_results_etn_my_muon_pad_sqrt.csv", "ETN", "MoNbTaVW", "Muon pad sqrt"),
            load_csv("all_etn_10000/results/MoNbTaVW_results_etn_my_muon_factorization.csv", "ETN", "MoNbTaVW", "Muon factorization"),
        ],
        ignore_index=True,
    )
    etn_mo_sm = summary(etn_mo)

    mtp_etn_quality_time(mtp_native, etn_agpd_native)
    etn_tuning()

    bfgs_diagnostics(mtp[mtp["optimizer"].isin(["native BFGS", "SciPy BFGS"])], "MTP BFGS comparison on AgPd", "fig_05_04_mtp_bfgs_native_vs_scipy_agpd.png")
    bfgs_diagnostics(mtp[mtp["optimizer"].isin(["native BFGS", "SciPy BFGS"])], "MTP BFGS diagnostics on AgPd", "fig_05_05_mtp_bfgs_diagnostics_agpd.png")

    loss_band(mtp, "Adam", "MTP Adam: smoothed loss history", "fig_05_06_mtp_adam_loss_agpd.png")
    loss_band(mtp, "Momentum", "MTP Momentum: smoothed loss history", "fig_05_07_mtp_momentum_loss_agpd.png")
    loss_band(mtp, "SGD", "MTP SGD: smoothed loss history", "fig_05_08_mtp_sgd_loss_agpd.png")
    loss_band(mtp, "Muon pad sqrt", "MTP Muon pad sqrt: smoothed loss history", "fig_05_09_mtp_muon_pad_sqrt_loss_agpd.png")
    loss_band(mtp, "Muon factorization", "MTP Muon factorization: smoothed loss history", "fig_05_10_mtp_muon_factorization_loss_agpd.png")
    lollipop_validation(mtp_sm, "MTP optimizer comparison on AgPd", "fig_05_11_mtp_validation_errors_agpd.png")
    quality_time(mtp_sm, "MTP quality vs train time on AgPd", "fig_05_12_mtp_time_quality_agpd.png")
    spread_bars(mtp_sm, "MTP ensemble spread on AgPd", "fig_05_13_mtp_ensemble_spread_agpd.png")
    loss_overview(mtp, "MTP smoothed loss histories on AgPd", "fig_05_14_mtp_loss_overview_agpd.png")

    bfgs_diagnostics(etn_agpd[etn_agpd["optimizer"].isin(["native BFGS", "SciPy BFGS"])], "ETN BFGS comparison on AgPd", "fig_05_15_etn_bfgs_native_vs_scipy_agpd.png")
    loss_band(etn_agpd, "SGD", "ETN SGD: smoothed loss history on AgPd", "fig_05_16_etn_sgd_loss_agpd.png")
    loss_band(etn_agpd, "Momentum", "ETN Momentum: smoothed loss history on AgPd", "fig_05_17_etn_momentum_loss_agpd.png")
    loss_band(etn_agpd, "Adam", "ETN Adam: smoothed loss history on AgPd", "fig_05_18_etn_adam_loss_agpd.png")
    loss_band(etn_agpd, "Muon pad sqrt", "ETN Muon pad sqrt: smoothed loss history on AgPd", "fig_05_19_etn_muon_pad_sqrt_loss_agpd.png")
    loss_band(etn_agpd, "Muon factorization", "ETN Muon factorization: smoothed loss history on AgPd", "fig_05_20_etn_muon_factorization_loss_agpd.png")
    lollipop_validation(etn_agpd_sm, "ETN optimizer comparison on AgPd", "fig_05_21_etn_validation_errors_agpd.png")
    loss_overview(etn_agpd, "ETN smoothed loss histories on AgPd", "fig_05_22_etn_loss_overview_agpd.png")
    quality_time(etn_agpd_sm, "ETN quality vs train time on AgPd", "fig_05_23_etn_time_quality_agpd.png")
    spread_bars(etn_agpd_sm, "ETN ensemble spread on AgPd", "fig_05_24_etn_ensemble_spread_agpd.png")

    bfgs_diagnostics(etn_mo[etn_mo["optimizer"].isin(["native BFGS", "SciPy BFGS"])], "ETN BFGS comparison on MoNbTaVW", "fig_05_25_etn_bfgs_diagnostics_monbtavw.png")
    loss_band(etn_mo, "SGD", "ETN SGD: smoothed loss history on MoNbTaVW", "fig_05_26_etn_sgd_loss_monbtavw.png")
    loss_band(etn_mo, "Momentum", "ETN Momentum: smoothed loss history on MoNbTaVW", "fig_05_27_etn_momentum_loss_monbtavw.png")
    loss_band(etn_mo, "Adam", "ETN Adam: smoothed loss history on MoNbTaVW", "fig_05_28_etn_adam_loss_monbtavw.png")
    loss_band(etn_mo, "Muon pad sqrt", "ETN Muon pad sqrt: smoothed loss history on MoNbTaVW", "fig_05_29_etn_muon_pad_sqrt_loss_monbtavw.png")
    loss_band(etn_mo, "Muon factorization", "ETN Muon factorization: smoothed loss history on MoNbTaVW", "fig_05_30_etn_muon_factorization_loss_monbtavw.png")
    lollipop_validation(etn_mo_sm, "ETN optimizer comparison on MoNbTaVW", "fig_05_31_etn_validation_errors_monbtavw.png")
    quality_time(etn_mo_sm, "ETN quality vs train time on MoNbTaVW", "fig_05_32_etn_time_quality_monbtavw.png")
    spread_bars(etn_mo_sm, "ETN ensemble spread on MoNbTaVW", "fig_05_33_etn_ensemble_spread_monbtavw.png")
    loss_overview(etn_mo, "ETN smoothed loss histories on MoNbTaVW", "fig_05_34_etn_loss_overview_monbtavw.png")


if __name__ == "__main__":
    make_all()
