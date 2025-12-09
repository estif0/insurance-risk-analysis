"""
Insurance Modeling Module

This module provides the InsuranceModeler class for building and evaluating
predictive models for insurance claims analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import shap
import warnings

warnings.filterwarnings("ignore")


class InsuranceModeler:
    """
    A class for building and evaluating predictive models for insurance claims.

    This class handles feature preparation, model training (Linear Regression,
    Random Forest, XGBoost), evaluation, and interpretation using SHAP.

    Attributes:
        data (pd.DataFrame): The insurance dataset.
        target_col (str): Name of the target column to predict.
        feature_cols (List[str]): List of feature column names.
        models (Dict): Dictionary storing trained models.
        results (Dict): Dictionary storing evaluation results.
        X_train, X_test, y_train, y_test: Train-test split data.
        encoders (Dict): Dictionary of label encoders for categorical features.
        scaler (StandardScaler): Scaler for numerical features.
    """

    def __init__(self, data: pd.DataFrame, target_col: str = "TotalClaims"):
        """
        Initialize the InsuranceModeler.

        Args:
            data: Insurance dataset as pandas DataFrame.
            target_col: Name of the target column to predict (default: "TotalClaims").

        Raises:
            ValueError: If target_col not in data columns.
        """
        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")

        self.data = data.copy()
        self.target_col = target_col
        self.feature_cols = []
        self.models = {}
        self.results = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []

    def prepare_features(
        self,
        categorical_cols: Optional[List[str]] = None,
        numerical_cols: Optional[List[str]] = None,
        create_vehicle_age: bool = True,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Dict[str, any]:
        """
        Prepare features for modeling: encoding, scaling, train-test split.

        Args:
            categorical_cols: List of categorical columns to encode.
            numerical_cols: List of numerical columns to use.
            create_vehicle_age: Whether to create VehicleAge feature.
            test_size: Proportion of data for test set (default: 0.2).
            random_state: Random seed for reproducibility.

        Returns:
            Dictionary with keys: 'n_samples', 'n_features', 'train_size', 'test_size'.

        Raises:
            ValueError: If data is empty or columns not found.
        """
        if self.data.empty:
            raise ValueError("Data is empty")

        df = self.data.copy()

        # Create VehicleAge feature if requested
        if create_vehicle_age and "RegistrationYear" in df.columns:
            current_year = 2015  # Data ends in August 2015
            df["VehicleAge"] = current_year - df["RegistrationYear"]
            df["VehicleAge"] = df["VehicleAge"].clip(lower=0)  # No negative ages

        # Default columns if not specified
        if categorical_cols is None:
            categorical_cols = ["Province", "VehicleType", "Gender"]

        if numerical_cols is None:
            numerical_cols = ["TotalPremium", "SumInsured", "CalculatedPremiumPerTerm"]
            if "VehicleAge" in df.columns:
                numerical_cols.append("VehicleAge")

        # Filter to existing columns
        categorical_cols = [col for col in categorical_cols if col in df.columns]
        numerical_cols = [col for col in numerical_cols if col in df.columns]

        if not categorical_cols and not numerical_cols:
            raise ValueError("No valid feature columns found")

        # Remove target from features if present
        if self.target_col in categorical_cols:
            categorical_cols.remove(self.target_col)
        if self.target_col in numerical_cols:
            numerical_cols.remove(self.target_col)

        # Handle missing values (simple imputation)
        for col in numerical_cols:
            df[col] = df[col].fillna(df[col].median())

        for col in categorical_cols:
            df[col] = df[col].fillna(
                df[col].mode()[0] if not df[col].mode().empty else "Unknown"
            )

        # Encode categorical variables
        encoded_features = []
        for col in categorical_cols:
            encoder = LabelEncoder()
            df[f"{col}_encoded"] = encoder.fit_transform(df[col].astype(str))
            self.encoders[col] = encoder
            encoded_features.append(f"{col}_encoded")

        # Combine features
        self.feature_cols = numerical_cols + encoded_features
        self.feature_names = (
            numerical_cols + categorical_cols
        )  # Original names for interpretation

        # Prepare X and y
        X = df[self.feature_cols].copy()
        y = df[self.target_col].copy()

        # Handle missing target values
        valid_idx = ~y.isna()
        X = X[valid_idx]
        y = y[valid_idx]

        # Scale numerical features
        if numerical_cols:
            X[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])

        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        return {
            "n_samples": len(X),
            "n_features": len(self.feature_cols),
            "train_size": len(self.X_train),
            "test_size": len(self.X_test),
            "feature_names": self.feature_cols,
        }

    def train_linear_regression(self) -> Dict[str, float]:
        """
        Train a Linear Regression model.

        Returns:
            Dictionary with training metrics: 'train_r2', 'train_rmse', 'train_mae'.

        Raises:
            ValueError: If features not prepared.
        """
        if self.X_train is None:
            raise ValueError("Features not prepared. Call prepare_features() first.")

        model = LinearRegression()
        model.fit(self.X_train, self.y_train)
        self.models["Linear Regression"] = model

        # Training metrics
        y_train_pred = model.predict(self.X_train)
        train_r2 = r2_score(self.y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
        train_mae = mean_absolute_error(self.y_train, y_train_pred)

        return {"train_r2": train_r2, "train_rmse": train_rmse, "train_mae": train_mae}

    def train_random_forest(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        random_state: int = 42,
    ) -> Dict[str, float]:
        """
        Train a Random Forest Regressor model.

        Args:
            n_estimators: Number of trees in the forest.
            max_depth: Maximum depth of trees (None = unlimited).
            random_state: Random seed for reproducibility.

        Returns:
            Dictionary with training metrics: 'train_r2', 'train_rmse', 'train_mae'.

        Raises:
            ValueError: If features not prepared.
        """
        if self.X_train is None:
            raise ValueError("Features not prepared. Call prepare_features() first.")

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(self.X_train, self.y_train)
        self.models["Random Forest"] = model

        # Training metrics
        y_train_pred = model.predict(self.X_train)
        train_r2 = r2_score(self.y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
        train_mae = mean_absolute_error(self.y_train, y_train_pred)

        return {"train_r2": train_r2, "train_rmse": train_rmse, "train_mae": train_mae}

    def train_xgboost(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        random_state: int = 42,
    ) -> Dict[str, float]:
        """
        Train an XGBoost Regressor model.

        Args:
            n_estimators: Number of boosting rounds.
            max_depth: Maximum tree depth.
            learning_rate: Learning rate for boosting.
            random_state: Random seed for reproducibility.

        Returns:
            Dictionary with training metrics: 'train_r2', 'train_rmse', 'train_mae'.

        Raises:
            ValueError: If features not prepared.
        """
        if self.X_train is None:
            raise ValueError("Features not prepared. Call prepare_features() first.")

        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(self.X_train, self.y_train)
        self.models["XGBoost"] = model

        # Training metrics
        y_train_pred = model.predict(self.X_train)
        train_r2 = r2_score(self.y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
        train_mae = mean_absolute_error(self.y_train, y_train_pred)

        return {"train_r2": train_r2, "train_rmse": train_rmse, "train_mae": train_mae}

    def evaluate_models(self) -> pd.DataFrame:
        """
        Evaluate all trained models on test set.

        Returns:
            DataFrame with evaluation metrics (RMSE, R², MAE) for each model.

        Raises:
            ValueError: If no models trained or test set not available.
        """
        if not self.models:
            raise ValueError("No models trained. Train models first.")

        if self.X_test is None or self.y_test is None:
            raise ValueError("Test set not available. Call prepare_features() first.")

        results = []

        for model_name, model in self.models.items():
            # Predictions
            y_pred = model.predict(self.X_test)

            # Metrics
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            r2 = r2_score(self.y_test, y_pred)
            mae = mean_absolute_error(self.y_test, y_pred)

            results.append({"Model": model_name, "RMSE": rmse, "R²": r2, "MAE": mae})

            # Store results
            self.results[model_name] = {
                "rmse": rmse,
                "r2": r2,
                "mae": mae,
                "predictions": y_pred,
            }

        return pd.DataFrame(results).sort_values("R²", ascending=False)

    def get_feature_importance(
        self, model_name: str = "Random Forest", top_n: int = 10
    ) -> pd.DataFrame:
        """
        Get feature importance from tree-based models.

        Args:
            model_name: Name of the model ('Random Forest' or 'XGBoost').
            top_n: Number of top features to return.

        Returns:
            DataFrame with features and their importance scores (sorted).

        Raises:
            ValueError: If model not trained or doesn't support feature importance.
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not trained")

        model = self.models[model_name]

        if not hasattr(model, "feature_importances_"):
            raise ValueError(f"Model '{model_name}' doesn't support feature importance")

        # Get importance
        importance = model.feature_importances_

        # Create DataFrame
        feature_importance = pd.DataFrame(
            {"Feature": self.feature_cols, "Importance": importance}
        )

        # Sort and get top N
        feature_importance = feature_importance.sort_values(
            "Importance", ascending=False
        ).head(top_n)

        return feature_importance

    def shap_analysis(
        self, model_name: str = "XGBoost", sample_size: int = 100
    ) -> Tuple[shap.Explainer, np.ndarray]:
        """
        Perform SHAP analysis for model interpretability.

        Args:
            model_name: Name of the model to analyze.
            sample_size: Number of samples to use for SHAP (for speed).

        Returns:
            Tuple of (SHAP explainer, SHAP values).

        Raises:
            ValueError: If model not trained.
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not trained")

        model = self.models[model_name]

        # Sample data for efficiency
        if len(self.X_test) > sample_size:
            sample_idx = np.random.choice(len(self.X_test), sample_size, replace=False)
            X_sample = self.X_test.iloc[sample_idx]
        else:
            X_sample = self.X_test

        # Create SHAP explainer
        if model_name == "Linear Regression":
            explainer = shap.LinearExplainer(model, self.X_train)
            shap_values = explainer.shap_values(X_sample)
        elif model_name == "Random Forest":
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        elif model_name == "XGBoost":
            # Use KernelExplainer for XGBoost to avoid compatibility issues
            explainer = shap.KernelExplainer(
                model.predict, shap.sample(self.X_train, min(50, len(self.X_train)))
            )
            shap_values = explainer.shap_values(X_sample)
        else:
            explainer = shap.Explainer(model, self.X_train)
            shap_values = explainer.shap_values(X_sample)

        return explainer, shap_values

    def get_best_model(self) -> Tuple[str, Dict[str, float]]:
        """
        Get the best performing model based on R² score.

        Returns:
            Tuple of (model_name, metrics_dict).

        Raises:
            ValueError: If no models evaluated.
        """
        if not self.results:
            raise ValueError("No models evaluated. Call evaluate_models() first.")

        best_model_name = max(self.results.items(), key=lambda x: x[1]["r2"])[0]
        best_metrics = self.results[best_model_name]

        return best_model_name, best_metrics

    def predict(self, X: pd.DataFrame, model_name: Optional[str] = None) -> np.ndarray:
        """
        Make predictions using a trained model.

        Args:
            X: Feature matrix for prediction.
            model_name: Name of model to use (None = best model).

        Returns:
            Array of predictions.

        Raises:
            ValueError: If model not trained.
        """
        if model_name is None:
            model_name, _ = self.get_best_model()

        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not trained")

        model = self.models[model_name]
        return model.predict(X)
