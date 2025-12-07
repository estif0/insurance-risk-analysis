"""
Visualization Engine Module.

This module provides comprehensive visualization capabilities for insurance
risk analysis, including distributions, correlations, time series, and more.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path


class Visualizer:
    """
    Engine for creating comprehensive visualizations for EDA and analysis.

    This class provides methods for plotting distributions, relationships,
    correlations, and insurance-specific metrics with consistent styling.

    Attributes:
        style: Matplotlib/Seaborn style theme.
        color_palette: Color palette for plots.
        figure_size: Default figure size for matplotlib plots.
        logger: Logger instance for tracking operations.
    """

    def __init__(
        self,
        style: str = "whitegrid",
        color_palette: str = "Set2",
        figure_size: Tuple[int, int] = (12, 6),
    ):
        """
        Initialize the Visualizer.

        Args:
            style: Seaborn style theme ('whitegrid', 'darkgrid', 'white', 'dark', 'ticks').
            color_palette: Color palette name for plots.
            figure_size: Default figure size as (width, height).

        Raises:
            ValueError: If invalid style provided.
        """
        valid_styles = ["whitegrid", "darkgrid", "white", "dark", "ticks"]
        if style not in valid_styles:
            raise ValueError(f"Style must be one of {valid_styles}")

        self.style = style
        self.color_palette = color_palette
        self.figure_size = figure_size

        # Set style
        sns.set_style(self.style)
        sns.set_palette(self.color_palette)

        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"Visualizer initialized with style='{style}', "
            f"palette='{color_palette}', size={figure_size}"
        )

    def plot_distribution(
        self,
        data: pd.DataFrame,
        column: str,
        bins: int = 30,
        kde: bool = True,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: str = "Frequency",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot distribution of a numeric column with histogram and optional KDE.

        Args:
            data: DataFrame containing the data.
            column: Column name to plot.
            bins: Number of bins for histogram.
            kde: Whether to overlay kernel density estimate.
            title: Plot title. If None, uses column name.
            xlabel: X-axis label. If None, uses column name.
            ylabel: Y-axis label.
            save_path: Path to save figure. If None, doesn't save.

        Returns:
            Matplotlib Figure object.

        Raises:
            ValueError: If column doesn't exist or is not numeric.
        """
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in data")

        if not pd.api.types.is_numeric_dtype(data[column]):
            raise ValueError(f"Column '{column}' must be numeric")

        self.logger.info(f"Plotting distribution for: {column}")

        fig, ax = plt.subplots(figsize=self.figure_size)

        # Plot histogram with KDE
        sns.histplot(
            data=data,
            x=column,
            bins=bins,
            kde=kde,
            ax=ax,
            edgecolor="black",
            alpha=0.7,
        )

        # Set labels and title
        ax.set_xlabel(xlabel if xlabel else column, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(
            title if title else f"Distribution of {column}",
            fontsize=14,
            fontweight="bold",
        )

        # Add mean and median lines
        mean_val = data[column].mean()
        median_val = data[column].median()

        ax.axvline(
            mean_val,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_val:.2f}",
        )
        ax.axvline(
            median_val,
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Median: {median_val:.2f}",
        )

        ax.legend()
        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        return fig

    def plot_categorical(
        self,
        data: pd.DataFrame,
        column: str,
        top_n: Optional[int] = None,
        horizontal: bool = False,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot bar chart for categorical column.

        Args:
            data: DataFrame containing the data.
            column: Column name to plot.
            top_n: Show only top N categories. If None, shows all.
            horizontal: Whether to plot horizontal bars.
            title: Plot title. If None, uses column name.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            save_path: Path to save figure.

        Returns:
            Matplotlib Figure object.

        Raises:
            ValueError: If column doesn't exist.
        """
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in data")

        self.logger.info(f"Plotting categorical distribution for: {column}")

        # Get value counts
        value_counts = data[column].value_counts()

        if top_n:
            value_counts = value_counts.head(top_n)

        fig, ax = plt.subplots(figsize=self.figure_size)

        if horizontal:
            value_counts.plot(kind="barh", ax=ax, edgecolor="black", alpha=0.8)
            ax.set_xlabel(xlabel if xlabel else "Count", fontsize=12)
            ax.set_ylabel(ylabel if ylabel else column, fontsize=12)
        else:
            value_counts.plot(kind="bar", ax=ax, edgecolor="black", alpha=0.8)
            ax.set_xlabel(xlabel if xlabel else column, fontsize=12)
            ax.set_ylabel(ylabel if ylabel else "Count", fontsize=12)

        # Title
        title_text = title if title else f"Distribution of {column}"
        if top_n:
            title_text += f" (Top {top_n})"
        ax.set_title(title_text, fontsize=14, fontweight="bold")

        # Rotate x labels if vertical bars
        if not horizontal:
            plt.xticks(rotation=45, ha="right")

        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        return fig

    def plot_correlation_matrix(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = "pearson",
        annot: bool = True,
        fmt: str = ".2f",
        cmap: str = "coolwarm",
        title: str = "Correlation Matrix",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot correlation matrix heatmap.

        Args:
            data: DataFrame containing the data.
            columns: Columns to include. If None, uses all numeric columns.
            method: Correlation method ('pearson', 'spearman', 'kendall').
            annot: Whether to annotate cells with correlation values.
            fmt: String format for annotations.
            cmap: Colormap for heatmap.
            title: Plot title.
            save_path: Path to save figure.

        Returns:
            Matplotlib Figure object.

        Raises:
            ValueError: If invalid method or no numeric columns.
        """
        valid_methods = ["pearson", "spearman", "kendall"]
        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")

        # Get numeric columns
        if columns is None:
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        else:
            numeric_cols = [
                col for col in columns if pd.api.types.is_numeric_dtype(data[col])
            ]

        if len(numeric_cols) < 2:
            raise ValueError("Need at least 2 numeric columns for correlation matrix")

        self.logger.info(f"Plotting correlation matrix for {len(numeric_cols)} columns")

        # Calculate correlation
        corr = data[numeric_cols].corr(method=method)

        # Create figure
        fig, ax = plt.subplots(
            figsize=(min(len(numeric_cols), 12), min(len(numeric_cols), 10))
        )

        # Plot heatmap
        sns.heatmap(
            corr,
            annot=annot,
            fmt=fmt,
            cmap=cmap,
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax,
        )

        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        return fig

    def plot_scatter(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        hue: Optional[str] = None,
        size: Optional[str] = None,
        alpha: float = 0.6,
        add_regression: bool = False,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot scatter plot for two numeric columns.

        Args:
            data: DataFrame containing the data.
            x: Column name for x-axis.
            y: Column name for y-axis.
            hue: Column name for color coding.
            size: Column name for point sizing.
            alpha: Transparency of points (0-1).
            add_regression: Whether to add regression line.
            title: Plot title.
            save_path: Path to save figure.

        Returns:
            Matplotlib Figure object.

        Raises:
            ValueError: If columns don't exist or are not numeric.
        """
        if x not in data.columns or y not in data.columns:
            raise ValueError(f"One or both columns not found: {x}, {y}")

        if not pd.api.types.is_numeric_dtype(data[x]):
            raise ValueError(f"Column '{x}' must be numeric")
        if not pd.api.types.is_numeric_dtype(data[y]):
            raise ValueError(f"Column '{y}' must be numeric")

        self.logger.info(f"Plotting scatter: {x} vs {y}")

        fig, ax = plt.subplots(figsize=self.figure_size)

        if add_regression:
            sns.regplot(
                data=data,
                x=x,
                y=y,
                ax=ax,
                scatter_kws={"alpha": alpha},
                line_kws={"color": "red", "linewidth": 2},
            )
        else:
            sns.scatterplot(
                data=data,
                x=x,
                y=y,
                hue=hue,
                size=size,
                alpha=alpha,
                ax=ax,
            )

        ax.set_xlabel(x, fontsize=12)
        ax.set_ylabel(y, fontsize=12)
        ax.set_title(title if title else f"{y} vs {x}", fontsize=14, fontweight="bold")

        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        return fig

    def plot_boxplot(
        self,
        data: pd.DataFrame,
        column: str,
        by: Optional[str] = None,
        horizontal: bool = False,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot box plot to show distribution and outliers.

        Args:
            data: DataFrame containing the data.
            column: Numeric column to plot.
            by: Categorical column to group by.
            horizontal: Whether to plot horizontal boxplot.
            title: Plot title.
            save_path: Path to save figure.

        Returns:
            Matplotlib Figure object.

        Raises:
            ValueError: If column doesn't exist or is not numeric.
        """
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in data")

        if not pd.api.types.is_numeric_dtype(data[column]):
            raise ValueError(f"Column '{column}' must be numeric")

        self.logger.info(f"Plotting boxplot for: {column}")

        fig, ax = plt.subplots(figsize=self.figure_size)

        if horizontal:
            if by:
                sns.boxplot(data=data, y=by, x=column, ax=ax)
            else:
                sns.boxplot(data=data, x=column, ax=ax)
        else:
            if by:
                sns.boxplot(data=data, x=by, y=column, ax=ax)
                plt.xticks(rotation=45, ha="right")
            else:
                sns.boxplot(data=data, y=column, ax=ax)

        title_text = title if title else f"Boxplot of {column}"
        if by:
            title_text += f" by {by}"

        ax.set_title(title_text, fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        return fig

    def plot_time_series(
        self,
        data: pd.DataFrame,
        date_column: str,
        value_column: str,
        aggregation: str = "sum",
        freq: str = "ME",
        title: Optional[str] = None,
        xlabel: str = "Date",
        ylabel: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot time series data.

        Args:
            data: DataFrame containing the data.
            date_column: Column containing dates.
            value_column: Column with values to plot.
            aggregation: Aggregation method ('sum', 'mean', 'count', 'median').
            freq: Frequency for grouping ('D', 'W', 'ME', 'Q', 'Y').
            title: Plot title.
            xlabel: X-axis label.
            ylabel: Y-axis label. If None, uses value_column.
            save_path: Path to save figure.

        Returns:
            Matplotlib Figure object.

        Raises:
            ValueError: If columns don't exist or invalid aggregation.
        """
        if date_column not in data.columns or value_column not in data.columns:
            raise ValueError(
                f"One or both columns not found: {date_column}, {value_column}"
            )

        valid_agg = ["sum", "mean", "count", "median", "min", "max"]
        if aggregation not in valid_agg:
            raise ValueError(f"Aggregation must be one of {valid_agg}")

        self.logger.info(f"Plotting time series: {value_column} over {date_column}")

        # Prepare data
        df = data.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.set_index(date_column)

        # Resample and aggregate
        if aggregation == "sum":
            ts_data = df[value_column].resample(freq).sum()
        elif aggregation == "mean":
            ts_data = df[value_column].resample(freq).mean()
        elif aggregation == "count":
            ts_data = df[value_column].resample(freq).count()
        elif aggregation == "median":
            ts_data = df[value_column].resample(freq).median()
        elif aggregation == "min":
            ts_data = df[value_column].resample(freq).min()
        else:  # max
            ts_data = df[value_column].resample(freq).max()

        # Plot
        fig, ax = plt.subplots(figsize=self.figure_size)
        ts_data.plot(ax=ax, linewidth=2, marker="o")

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel if ylabel else value_column, fontsize=12)
        ax.set_title(
            title if title else f"{value_column} over Time ({aggregation.title()})",
            fontsize=14,
            fontweight="bold",
        )

        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        return fig

    def plot_loss_ratio(
        self,
        data: pd.DataFrame,
        group_by: str,
        claims_column: str = "TotalClaims",
        premium_column: str = "TotalPremium",
        top_n: Optional[int] = None,
        title: str = "Loss Ratio by Group",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot loss ratios by category (insurance-specific).

        Loss ratio = TotalClaims / TotalPremium
        A ratio > 1.0 indicates loss, < 1.0 indicates profit.

        Args:
            data: DataFrame containing the data.
            group_by: Column to group by.
            claims_column: Column containing total claims.
            premium_column: Column containing total premiums.
            top_n: Show only top N groups. If None, shows all.
            title: Plot title.
            save_path: Path to save figure.

        Returns:
            Matplotlib Figure object.

        Raises:
            ValueError: If columns don't exist or are not numeric.
        """
        required_cols = [group_by, claims_column, premium_column]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Columns not found: {missing_cols}")

        self.logger.info(f"Plotting loss ratios by: {group_by}")

        # Calculate loss ratios
        grouped = data.groupby(group_by).agg(
            {
                claims_column: "sum",
                premium_column: "sum",
            }
        )

        grouped["loss_ratio"] = grouped[claims_column] / grouped[premium_column]
        grouped = grouped.sort_values("loss_ratio", ascending=False)

        if top_n:
            grouped = grouped.head(top_n)

        # Plot
        fig, ax = plt.subplots(figsize=self.figure_size)

        colors = ["red" if x > 1.0 else "green" for x in grouped["loss_ratio"]]

        grouped["loss_ratio"].plot(
            kind="barh", ax=ax, color=colors, edgecolor="black", alpha=0.7
        )

        # Add reference line at 1.0
        ax.axvline(
            1.0, color="black", linestyle="--", linewidth=2, label="Break-even (1.0)"
        )

        ax.set_xlabel("Loss Ratio", fontsize=12)
        ax.set_ylabel(group_by, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend()

        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        return fig

    def plot_interactive_scatter(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        color: Optional[str] = None,
        size: Optional[str] = None,
        hover_data: Optional[List[str]] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """
        Create interactive scatter plot using Plotly.

        Args:
            data: DataFrame containing the data.
            x: Column name for x-axis.
            y: Column name for y-axis.
            color: Column name for color coding.
            size: Column name for point sizing.
            hover_data: Additional columns to show on hover.
            title: Plot title.
            save_path: Path to save figure as HTML.

        Returns:
            Plotly Figure object.

        Raises:
            ValueError: If columns don't exist.
        """
        required_cols = [x, y]
        if color:
            required_cols.append(color)
        if size:
            required_cols.append(size)

        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Columns not found: {missing_cols}")

        self.logger.info(f"Creating interactive scatter: {x} vs {y}")

        fig = px.scatter(
            data,
            x=x,
            y=y,
            color=color,
            size=size,
            hover_data=hover_data,
            title=title if title else f"{y} vs {x}",
            template="plotly_white",
        )

        fig.update_traces(marker=dict(line=dict(width=0.5, color="DarkSlateGrey")))
        fig.update_layout(
            font=dict(size=12),
            title_font=dict(size=16, family="Arial Black"),
        )

        if save_path:
            self._save_plotly_figure(fig, save_path)

        return fig

    def create_dashboard(
        self,
        data: pd.DataFrame,
        numeric_cols: List[str],
        categorical_cols: List[str],
        title: str = "Data Overview Dashboard",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create a multi-panel dashboard with key visualizations.

        Args:
            data: DataFrame containing the data.
            numeric_cols: List of numeric columns to visualize (max 3).
            categorical_cols: List of categorical columns to visualize (max 3).
            title: Dashboard title.
            save_path: Path to save figure.

        Returns:
            Matplotlib Figure object.

        Raises:
            ValueError: If too many columns specified or columns don't exist.
        """
        if len(numeric_cols) > 3 or len(categorical_cols) > 3:
            raise ValueError("Maximum 3 numeric and 3 categorical columns allowed")

        all_cols = numeric_cols + categorical_cols
        missing_cols = [col for col in all_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Columns not found: {missing_cols}")

        self.logger.info(f"Creating dashboard with {len(all_cols)} plots")

        # Create figure with subplots
        n_plots = len(all_cols)
        n_rows = (n_plots + 1) // 2
        fig, axes = plt.subplots(n_rows, 2, figsize=(14, n_rows * 4))

        # Flatten axes for easy iteration
        if n_rows == 1:
            axes = [axes] if n_plots == 1 else axes
        else:
            axes = axes.flatten()

        # Plot numeric columns
        for idx, col in enumerate(numeric_cols):
            sns.histplot(
                data=data, x=col, kde=True, ax=axes[idx], edgecolor="black", alpha=0.7
            )
            axes[idx].set_title(f"Distribution of {col}", fontweight="bold")

        # Plot categorical columns
        for idx, col in enumerate(categorical_cols, start=len(numeric_cols)):
            top_10 = data[col].value_counts().head(10)
            top_10.plot(kind="bar", ax=axes[idx], edgecolor="black", alpha=0.8)
            axes[idx].set_title(f"Top 10 {col}", fontweight="bold")
            axes[idx].tick_params(axis="x", rotation=45)

        # Hide empty subplots
        for idx in range(n_plots, len(axes)):
            axes[idx].axis("off")

        fig.suptitle(title, fontsize=16, fontweight="bold", y=1.00)
        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        return fig

    def _save_figure(self, fig: plt.Figure, path: str) -> None:
        """
        Save matplotlib figure to file.

        Args:
            fig: Figure object to save.
            path: File path to save to.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        self.logger.info(f"Figure saved to: {path}")

    def _save_plotly_figure(self, fig: go.Figure, path: str) -> None:
        """
        Save Plotly figure to HTML file.

        Args:
            fig: Plotly Figure object to save.
            path: File path to save to.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(path)
        self.logger.info(f"Interactive figure saved to: {path}")
