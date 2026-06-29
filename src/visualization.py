"""Visualization helpers for EDA and reporting."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import NUMERIC_FEATURES, REPORTS_DIR, TARGET_COLUMN
from src.utils import ensure_dir


def plot_feature_distributions(df: pd.DataFrame, save: bool = True) -> None:
    """Plot histograms for numeric features."""
    cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    n = len(cols)
    fig, axes = plt.subplots((n + 2) // 3, 3, figsize=(12, 4 * ((n + 2) // 3)))
    axes = axes.flatten()

    for ax, col in zip(axes, cols):
        sns.histplot(df[col], kde=True, ax=ax, color="#1e88e5")
        ax.set_title(col.replace("_", " ").title())

    for ax in axes[len(cols) :]:
        ax.axis("off")

    plt.tight_layout()
    if save:
        ensure_dir(REPORTS_DIR)
        plt.savefig(REPORTS_DIR / "feature_distributions.png", dpi=150)
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, save: bool = True) -> None:
    """Plot correlation matrix for numeric columns."""
    cols = [c for c in NUMERIC_FEATURES + [TARGET_COLUMN] if c in df.columns]
    corr = df[cols].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()

    if save:
        ensure_dir(REPORTS_DIR)
        plt.savefig(REPORTS_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()


def plot_target_vs_feature(df: pd.DataFrame, feature: str, save: bool = True) -> None:
    """Scatter plot of target vs a single feature."""
    if feature not in df.columns:
        raise ValueError(f"Feature '{feature}' not in dataset")

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x=feature, y=TARGET_COLUMN, alpha=0.6)
    plt.title(f"{TARGET_COLUMN} vs {feature}")
    plt.tight_layout()

    if save:
        ensure_dir(REPORTS_DIR)
        plt.savefig(REPORTS_DIR / f"target_vs_{feature}.png", dpi=150)
    plt.close()
