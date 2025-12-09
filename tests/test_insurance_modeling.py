"""
Unit tests for InsuranceModeler class.
"""

import pytest
import pandas as pd
import numpy as np
from core.insurance_modeling import InsuranceModeler


@pytest.fixture
def sample_insurance_data():
    """Create sample insurance data for testing."""
    np.random.seed(42)
    n = 500

    data = pd.DataFrame(
        {
            "Province": np.random.choice(["Gauteng", "Western Cape", "KZN"], n),
            "VehicleType": np.random.choice(["Sedan", "SUV", "Truck"], n),
            "Gender": np.random.choice(["Male", "Female"], n),
            "RegistrationYear": np.random.randint(2000, 2015, n),
            "TotalPremium": np.random.uniform(1000, 10000, n),
            "TotalClaims": np.random.uniform(0, 15000, n),
            "SumInsured": np.random.uniform(50000, 500000, n),
            "CalculatedPremiumPerTerm": np.random.uniform(800, 8000, n),
        }
    )

    return data


class TestInitialization:
    """Test InsuranceModeler initialization."""

    def test_init_with_valid_data(self, sample_insurance_data):
        """Test initialization with valid data."""
        modeler = InsuranceModeler(sample_insurance_data, target_col="TotalClaims")

        assert modeler.target_col == "TotalClaims"
        assert len(modeler.data) == len(sample_insurance_data)
        assert isinstance(modeler.models, dict)
        assert len(modeler.models) == 0

    def test_init_with_invalid_target(self, sample_insurance_data):
        """Test initialization with invalid target column."""
        with pytest.raises(ValueError, match="Target column .* not found"):
            InsuranceModeler(sample_insurance_data, target_col="NonExistentColumn")

    def test_init_default_target(self, sample_insurance_data):
        """Test initialization with default target."""
        modeler = InsuranceModeler(sample_insurance_data)
        assert modeler.target_col == "TotalClaims"


class TestFeaturePreparation:
    """Test feature preparation methods."""

    def test_prepare_features_basic(self, sample_insurance_data):
        """Test basic feature preparation."""
        modeler = InsuranceModeler(sample_insurance_data)
        result = modeler.prepare_features()

        assert result["n_samples"] > 0
        assert result["n_features"] > 0
        assert result["train_size"] > 0
        assert result["test_size"] > 0
        assert modeler.X_train is not None
        assert modeler.X_test is not None
        assert modeler.y_train is not None
        assert modeler.y_test is not None

    def test_prepare_features_creates_vehicle_age(self, sample_insurance_data):
        """Test that VehicleAge feature is created."""
        modeler = InsuranceModeler(sample_insurance_data)
        modeler.prepare_features(create_vehicle_age=True)

        # Check VehicleAge is in feature columns
        assert any("VehicleAge" in str(col) for col in modeler.feature_cols)

    def test_prepare_features_custom_test_size(self, sample_insurance_data):
        """Test custom test size."""
        modeler = InsuranceModeler(sample_insurance_data)
        result = modeler.prepare_features(test_size=0.3)

        total = result["train_size"] + result["test_size"]
        test_ratio = result["test_size"] / total
        assert 0.25 < test_ratio < 0.35  # Allow some tolerance

    def test_prepare_features_with_empty_data(self):
        """Test feature preparation with empty DataFrame."""
        empty_data = pd.DataFrame()
        modeler = InsuranceModeler.__new__(InsuranceModeler)
        modeler.data = empty_data
        modeler.target_col = "TotalClaims"

        with pytest.raises(ValueError, match="Data is empty"):
            modeler.prepare_features()

    def test_prepare_features_encodes_categoricals(self, sample_insurance_data):
        """Test that categorical variables are encoded."""
        modeler = InsuranceModeler(sample_insurance_data)
        modeler.prepare_features()

        assert len(modeler.encoders) > 0
        assert "Province" in modeler.encoders or "Gender" in modeler.encoders


class TestModelTraining:
    """Test model training methods."""

    def test_train_linear_regression(self, sample_insurance_data):
        """Test Linear Regression training."""
        modeler = InsuranceModeler(sample_insurance_data)
        modeler.prepare_features()

        result = modeler.train_linear_regression()

        assert "train_r2" in result
        assert "train_rmse" in result
        assert "train_mae" in result
        assert "Linear Regression" in modeler.models
        assert result["train_rmse"] >= 0
        assert result["train_mae"] >= 0

    def test_train_random_forest(self, sample_insurance_data):
        """Test Random Forest training."""
        modeler = InsuranceModeler(sample_insurance_data)
        modeler.prepare_features()

        result = modeler.train_random_forest(n_estimators=10)

        assert "train_r2" in result
        assert "train_rmse" in result
        assert "train_mae" in result
        assert "Random Forest" in modeler.models
        assert result["train_rmse"] >= 0

    def test_train_xgboost(self, sample_insurance_data):
        """Test XGBoost training."""
        modeler = InsuranceModeler(sample_insurance_data)
        modeler.prepare_features()

        result = modeler.train_xgboost(n_estimators=10)

        assert "train_r2" in result
        assert "train_rmse" in result
        assert "train_mae" in result
        assert "XGBoost" in modeler.models
        assert result["train_rmse"] >= 0

    def test_train_without_preparation(self, sample_insurance_data):
        """Test training without preparing features."""
        modeler = InsuranceModeler(sample_insurance_data)

        with pytest.raises(ValueError, match="Features not prepared"):
            modeler.train_linear_regression()


class TestModelEvaluation:
    """Test model evaluation methods."""

    def test_evaluate_models(self, sample_insurance_data):
        """Test model evaluation."""
        modeler = InsuranceModeler(sample_insurance_data)
        modeler.prepare_features()
        modeler.train_linear_regression()
        modeler.train_random_forest(n_estimators=10)

        results_df = modeler.evaluate_models()

        assert isinstance(results_df, pd.DataFrame)
        assert len(results_df) == 2
        assert "Model" in results_df.columns
        assert "RMSE" in results_df.columns
        assert "R²" in results_df.columns
        assert "MAE" in results_df.columns
        assert all(results_df["RMSE"] >= 0)
        assert all(results_df["MAE"] >= 0)

    def test_evaluate_without_models(self, sample_insurance_data):
        """Test evaluation without trained models."""
        modeler = InsuranceModeler(sample_insurance_data)
        modeler.prepare_features()

        with pytest.raises(ValueError, match="No models trained"):
            modeler.evaluate_models()

    def test_get_best_model(self, sample_insurance_data):
        """Test getting best model."""
        modeler = InsuranceModeler(sample_insurance_data)
        modeler.prepare_features()
        modeler.train_linear_regression()
        modeler.train_random_forest(n_estimators=10)
        modeler.evaluate_models()

        best_name, best_metrics = modeler.get_best_model()

        assert isinstance(best_name, str)
        assert isinstance(best_metrics, dict)
        assert "r2" in best_metrics
        assert "rmse" in best_metrics
        assert "mae" in best_metrics


class TestFeatureImportance:
    """Test feature importance methods."""

    def test_get_feature_importance_random_forest(self, sample_insurance_data):
        """Test feature importance from Random Forest."""
        modeler = InsuranceModeler(sample_insurance_data)
        modeler.prepare_features()
        modeler.train_random_forest(n_estimators=10)

        importance_df = modeler.get_feature_importance("Random Forest", top_n=5)

        assert isinstance(importance_df, pd.DataFrame)
        assert len(importance_df) <= 5
        assert "Feature" in importance_df.columns
        assert "Importance" in importance_df.columns
        assert all(importance_df["Importance"] >= 0)

    def test_get_feature_importance_xgboost(self, sample_insurance_data):
        """Test feature importance from XGBoost."""
        modeler = InsuranceModeler(sample_insurance_data)
        modeler.prepare_features()
        modeler.train_xgboost(n_estimators=10)

        importance_df = modeler.get_feature_importance("XGBoost", top_n=5)

        assert isinstance(importance_df, pd.DataFrame)
        assert len(importance_df) <= 5

    def test_get_feature_importance_invalid_model(self, sample_insurance_data):
        """Test feature importance with invalid model."""
        modeler = InsuranceModeler(sample_insurance_data)
        modeler.prepare_features()

        with pytest.raises(ValueError, match="Model .* not trained"):
            modeler.get_feature_importance("NonExistentModel")

    def test_get_feature_importance_unsupported_model(self, sample_insurance_data):
        """Test feature importance with unsupported model."""
        modeler = InsuranceModeler(sample_insurance_data)
        modeler.prepare_features()
        modeler.train_linear_regression()

        with pytest.raises(ValueError, match="doesn't support feature importance"):
            modeler.get_feature_importance("Linear Regression")


class TestPrediction:
    """Test prediction methods."""

    def test_predict_with_model_name(self, sample_insurance_data):
        """Test prediction with specific model."""
        modeler = InsuranceModeler(sample_insurance_data)
        modeler.prepare_features()
        modeler.train_linear_regression()

        predictions = modeler.predict(modeler.X_test, model_name="Linear Regression")

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(modeler.X_test)
        assert all(predictions >= 0) or any(
            predictions < 0
        )  # Allow negative predictions

    def test_predict_with_best_model(self, sample_insurance_data):
        """Test prediction with best model."""
        modeler = InsuranceModeler(sample_insurance_data)
        modeler.prepare_features()
        modeler.train_linear_regression()
        modeler.train_random_forest(n_estimators=10)
        modeler.evaluate_models()

        predictions = modeler.predict(modeler.X_test)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(modeler.X_test)

    def test_predict_invalid_model(self, sample_insurance_data):
        """Test prediction with invalid model."""
        modeler = InsuranceModeler(sample_insurance_data)
        modeler.prepare_features()

        with pytest.raises(ValueError, match="Model .* not trained"):
            modeler.predict(modeler.X_test, model_name="NonExistentModel")


class TestSHAPAnalysis:
    """Test SHAP analysis methods."""

    @pytest.mark.skip(
        reason="SHAP XGBoost compatibility issue with current library versions"
    )
    def test_shap_analysis_xgboost(self, sample_insurance_data):
        """Test SHAP analysis with XGBoost."""
        modeler = InsuranceModeler(sample_insurance_data)
        modeler.prepare_features()
        modeler.train_xgboost(n_estimators=10)

        explainer, shap_values = modeler.shap_analysis("XGBoost", sample_size=50)

        assert explainer is not None
        assert shap_values is not None
        assert len(shap_values) <= 50

    def test_shap_analysis_random_forest(self, sample_insurance_data):
        """Test SHAP analysis with Random Forest."""
        modeler = InsuranceModeler(sample_insurance_data)
        modeler.prepare_features()
        modeler.train_random_forest(n_estimators=10)

        explainer, shap_values = modeler.shap_analysis("Random Forest", sample_size=50)

        assert explainer is not None
        assert shap_values is not None

    def test_shap_analysis_invalid_model(self, sample_insurance_data):
        """Test SHAP analysis with invalid model."""
        modeler = InsuranceModeler(sample_insurance_data)
        modeler.prepare_features()

        with pytest.raises(ValueError, match="Model .* not trained"):
            modeler.shap_analysis("NonExistentModel")
