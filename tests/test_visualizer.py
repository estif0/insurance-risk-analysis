"""
Unit tests for the Visualizer module.
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from core.visualizer import Visualizer
import tempfile
from pathlib import Path


class TestVisualizer:
    """Test suite for Visualizer class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "Age": np.random.randint(18, 80, 100),
                "Premium": np.random.uniform(1000, 5000, 100),
                "Claims": np.random.uniform(0, 3000, 100),
                "Province": np.random.choice(["GP", "WC", "KZN", "EC"], 100),
                "VehicleType": np.random.choice(["Sedan", "SUV", "Truck"], 100),
                "Date": pd.date_range("2024-01-01", periods=100, freq="D"),
            }
        )

    @pytest.fixture
    def visualizer(self):
        """Create Visualizer instance."""
        plt.close("all")  # Clean up any existing plots
        return Visualizer()

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for saving plots."""
        with tempfile.TemporaryDirectory() as tmpdirname:
            yield tmpdirname

    def test_initialization_default(self):
        """Test default initialization."""
        viz = Visualizer()
        assert viz.style == "whitegrid"
        assert viz.color_palette == "Set2"
        assert viz.figure_size == (12, 6)

    def test_initialization_custom(self):
        """Test custom initialization."""
        viz = Visualizer(style="darkgrid", color_palette="Set1", figure_size=(10, 5))
        assert viz.style == "darkgrid"
        assert viz.color_palette == "Set1"
        assert viz.figure_size == (10, 5)

    def test_initialization_invalid_style(self):
        """Test initialization with invalid style."""
        with pytest.raises(ValueError, match="Style must be one of"):
            Visualizer(style="invalid_style")

    def test_plot_distribution(self, visualizer, sample_data):
        """Test distribution plot."""
        fig = visualizer.plot_distribution(sample_data, "Age")

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_plot_distribution_with_options(self, visualizer, sample_data):
        """Test distribution plot with custom options."""
        fig = visualizer.plot_distribution(
            sample_data,
            "Premium",
            bins=20,
            kde=False,
            title="Premium Distribution",
            xlabel="Premium Amount",
            ylabel="Count",
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_distribution_invalid_column(self, visualizer, sample_data):
        """Test distribution plot with invalid column."""
        with pytest.raises(ValueError, match="not found"):
            visualizer.plot_distribution(sample_data, "InvalidColumn")

    def test_plot_distribution_non_numeric(self, visualizer, sample_data):
        """Test distribution plot with non-numeric column."""
        with pytest.raises(ValueError, match="must be numeric"):
            visualizer.plot_distribution(sample_data, "Province")

    def test_plot_distribution_save(self, visualizer, sample_data, temp_dir):
        """Test distribution plot saving."""
        save_path = Path(temp_dir) / "distribution.png"
        fig = visualizer.plot_distribution(sample_data, "Age", save_path=str(save_path))

        assert save_path.exists()
        plt.close(fig)

    def test_plot_categorical(self, visualizer, sample_data):
        """Test categorical plot."""
        fig = visualizer.plot_categorical(sample_data, "Province")

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_categorical_top_n(self, visualizer, sample_data):
        """Test categorical plot with top_n."""
        fig = visualizer.plot_categorical(sample_data, "Province", top_n=3)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_categorical_horizontal(self, visualizer, sample_data):
        """Test categorical plot with horizontal bars."""
        fig = visualizer.plot_categorical(sample_data, "VehicleType", horizontal=True)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_categorical_invalid_column(self, visualizer, sample_data):
        """Test categorical plot with invalid column."""
        with pytest.raises(ValueError, match="not found"):
            visualizer.plot_categorical(sample_data, "InvalidColumn")

    def test_plot_correlation_matrix(self, visualizer, sample_data):
        """Test correlation matrix plot."""
        fig = visualizer.plot_correlation_matrix(sample_data)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_correlation_matrix_specific_columns(self, visualizer, sample_data):
        """Test correlation matrix with specific columns."""
        fig = visualizer.plot_correlation_matrix(
            sample_data, columns=["Age", "Premium"]
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_correlation_matrix_spearman(self, visualizer, sample_data):
        """Test correlation matrix with Spearman method."""
        fig = visualizer.plot_correlation_matrix(sample_data, method="spearman")

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_correlation_matrix_invalid_method(self, visualizer, sample_data):
        """Test correlation matrix with invalid method."""
        with pytest.raises(ValueError, match="Method must be one of"):
            visualizer.plot_correlation_matrix(sample_data, method="invalid")

    def test_plot_correlation_matrix_insufficient_columns(self, visualizer):
        """Test correlation matrix with insufficient numeric columns."""
        data = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})
        with pytest.raises(ValueError, match="at least 2 numeric columns"):
            visualizer.plot_correlation_matrix(data)

    def test_plot_scatter(self, visualizer, sample_data):
        """Test scatter plot."""
        fig = visualizer.plot_scatter(sample_data, "Age", "Premium")

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_scatter_with_hue(self, visualizer, sample_data):
        """Test scatter plot with color coding."""
        fig = visualizer.plot_scatter(sample_data, "Age", "Premium", hue="Province")

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_scatter_with_regression(self, visualizer, sample_data):
        """Test scatter plot with regression line."""
        fig = visualizer.plot_scatter(
            sample_data, "Age", "Premium", add_regression=True
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_scatter_invalid_columns(self, visualizer, sample_data):
        """Test scatter plot with invalid columns."""
        with pytest.raises(ValueError, match="not found"):
            visualizer.plot_scatter(sample_data, "Invalid1", "Invalid2")

    def test_plot_scatter_non_numeric(self, visualizer, sample_data):
        """Test scatter plot with non-numeric column."""
        with pytest.raises(ValueError, match="must be numeric"):
            visualizer.plot_scatter(sample_data, "Province", "Premium")

    def test_plot_boxplot(self, visualizer, sample_data):
        """Test boxplot."""
        fig = visualizer.plot_boxplot(sample_data, "Premium")

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_boxplot_grouped(self, visualizer, sample_data):
        """Test boxplot grouped by category."""
        fig = visualizer.plot_boxplot(sample_data, "Premium", by="Province")

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_boxplot_horizontal(self, visualizer, sample_data):
        """Test horizontal boxplot."""
        fig = visualizer.plot_boxplot(sample_data, "Claims", horizontal=True)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_boxplot_invalid_column(self, visualizer, sample_data):
        """Test boxplot with invalid column."""
        with pytest.raises(ValueError, match="not found"):
            visualizer.plot_boxplot(sample_data, "InvalidColumn")

    def test_plot_boxplot_non_numeric(self, visualizer, sample_data):
        """Test boxplot with non-numeric column."""
        with pytest.raises(ValueError, match="must be numeric"):
            visualizer.plot_boxplot(sample_data, "Province")

    def test_plot_time_series(self, visualizer, sample_data):
        """Test time series plot."""
        fig = visualizer.plot_time_series(sample_data, "Date", "Premium")

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_time_series_aggregations(self, visualizer, sample_data):
        """Test time series plot with different aggregations."""
        for agg in ["sum", "mean", "count", "median"]:
            fig = visualizer.plot_time_series(
                sample_data, "Date", "Premium", aggregation=agg
            )
            assert isinstance(fig, plt.Figure)
            plt.close(fig)

    def test_plot_time_series_invalid_aggregation(self, visualizer, sample_data):
        """Test time series with invalid aggregation."""
        with pytest.raises(ValueError, match="Aggregation must be one of"):
            visualizer.plot_time_series(
                sample_data, "Date", "Premium", aggregation="invalid"
            )

    def test_plot_time_series_invalid_columns(self, visualizer, sample_data):
        """Test time series with invalid columns."""
        with pytest.raises(ValueError, match="not found"):
            visualizer.plot_time_series(sample_data, "InvalidDate", "Premium")

    def test_plot_loss_ratio(self, visualizer):
        """Test loss ratio plot."""
        data = pd.DataFrame(
            {
                "Province": ["GP", "WC", "KZN", "EC"] * 25,
                "TotalClaims": np.random.uniform(1000, 5000, 100),
                "TotalPremium": np.random.uniform(1000, 5000, 100),
            }
        )

        fig = visualizer.plot_loss_ratio(data, "Province")

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_loss_ratio_top_n(self, visualizer):
        """Test loss ratio plot with top_n."""
        data = pd.DataFrame(
            {
                "Province": ["GP", "WC", "KZN", "EC", "MP"] * 20,
                "TotalClaims": np.random.uniform(1000, 5000, 100),
                "TotalPremium": np.random.uniform(1000, 5000, 100),
            }
        )

        fig = visualizer.plot_loss_ratio(data, "Province", top_n=3)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_loss_ratio_invalid_columns(self, visualizer, sample_data):
        """Test loss ratio plot with missing columns."""
        with pytest.raises(ValueError, match="not found"):
            visualizer.plot_loss_ratio(sample_data, "Province")

    def test_plot_interactive_scatter(self, visualizer, sample_data):
        """Test interactive scatter plot."""
        fig = visualizer.plot_interactive_scatter(sample_data, "Age", "Premium")

        assert fig is not None
        # Can't easily assert type due to Plotly structure

    def test_plot_interactive_scatter_with_color(self, visualizer, sample_data):
        """Test interactive scatter plot with color."""
        fig = visualizer.plot_interactive_scatter(
            sample_data, "Age", "Premium", color="Province", hover_data=["Claims"]
        )

        assert fig is not None

    def test_plot_interactive_scatter_invalid_columns(self, visualizer, sample_data):
        """Test interactive scatter with invalid columns."""
        with pytest.raises(ValueError, match="not found"):
            visualizer.plot_interactive_scatter(sample_data, "Invalid", "Premium")

    def test_plot_interactive_scatter_save(self, visualizer, sample_data, temp_dir):
        """Test interactive scatter plot saving."""
        save_path = Path(temp_dir) / "interactive.html"
        fig = visualizer.plot_interactive_scatter(
            sample_data, "Age", "Premium", save_path=str(save_path)
        )

        assert save_path.exists()

    def test_create_dashboard(self, visualizer, sample_data):
        """Test dashboard creation."""
        fig = visualizer.create_dashboard(
            sample_data,
            numeric_cols=["Age", "Premium"],
            categorical_cols=["Province", "VehicleType"],
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_create_dashboard_max_columns(self, visualizer, sample_data):
        """Test dashboard with maximum columns."""
        fig = visualizer.create_dashboard(
            sample_data,
            numeric_cols=["Age", "Premium", "Claims"],
            categorical_cols=["Province", "VehicleType"],
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_create_dashboard_too_many_columns(self, visualizer, sample_data):
        """Test dashboard with too many columns."""
        with pytest.raises(ValueError, match="Maximum 3"):
            visualizer.create_dashboard(
                sample_data,
                numeric_cols=["Age", "Premium", "Claims", "Age"],
                categorical_cols=["Province"],
            )

    def test_create_dashboard_invalid_columns(self, visualizer, sample_data):
        """Test dashboard with invalid columns."""
        with pytest.raises(ValueError, match="not found"):
            visualizer.create_dashboard(
                sample_data, numeric_cols=["Invalid"], categorical_cols=["Province"]
            )

    def test_save_figure_creates_directory(self, visualizer, sample_data, temp_dir):
        """Test that save_figure creates nested directories."""
        save_path = Path(temp_dir) / "nested" / "folder" / "plot.png"
        fig = visualizer.plot_distribution(sample_data, "Age", save_path=str(save_path))

        assert save_path.exists()
        assert save_path.parent.exists()
        plt.close(fig)
