# Insurance Risk Analysis - AlphaCare Insurance Solutions

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 📋 Project Overview

This project delivers cutting-edge risk and predictive analytics for **AlphaCare Insurance Solutions (ACIS)**, a car insurance provider in South Africa. The objective is to analyze historical insurance claim data to optimize marketing strategy and identify "low-risk" customer segments for premium reduction, creating opportunities to attract new clients.

**Project Timeline:** December 3-9, 2025  
**Data Period:** February 2014 - August 2015

## 🎯 Business Objectives

- Analyze historical insurance claim data to uncover risk patterns
- Perform A/B hypothesis testing to validate risk drivers
- Build predictive models for claim severity and premium optimization
- Identify low-risk segments for competitive pricing strategies
- Provide data-driven recommendations for marketing and pricing

## 🏗️ Project Architecture

```
insurance-risk-analysis/
├── core/                    # Reusable OOP modules
│   ├── data_loader.py       # Load .txt data files
│   ├── data_quality.py      # Data quality checks
│   ├── data_processor.py    # Data cleaning & processing
│   ├── eda.py              # EDA engine
│   ├── visualizer.py        # Visualization engine
│   ├── dvc_manager.py       # DVC operations
│   ├── metrics.py           # Business metrics
│   ├── statistical_tests.py # Statistical testing
│   ├── ab_testing.py        # A/B test framework
│   ├── feature_engineering.py
│   ├── model_trainer.py
│   ├── model_evaluator.py
│   └── model_interpreter.py
├── data/
│   ├── raw/                 # Original data (DVC tracked)
│   ├── processed/           # Intermediate data (DVC tracked)
│   └── clean/               # Final clean data (DVC tracked)
├── notebooks/               # Analysis notebooks (one per task)
│   ├── task_1_eda.ipynb
│   ├── task_2_dvc.ipynb
│   ├── task_3_hypothesis_testing.ipynb
│   └── task_4_modeling.ipynb
├── models/                  # Saved ML models
├── reports/                 # Figures and outputs
├── tests/                   # Unit tests
└── scripts/                 # Utility scripts
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd insurance-risk-analysis
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up DVC (Data Version Control)**
   ```bash
   dvc init
   dvc remote add -d localstorage /path/to/your/dvc-storage
   dvc pull
   ```

## 📊 Data Structure

The dataset (`MachineLearningRating_v3.txt`) contains insurance policy data with the following categories:

- **Policy Information:** UnderwrittenCoverID, PolicyID, TransactionDate, TransactionMonth
- **Client Demographics:** Gender, MaritalStatus, Citizenship, Language, Bank, AccountType
- **Location Data:** Province, PostalCode, MainCrestaZone, SubCrestaZone
- **Vehicle Details:** VehicleType, Make, Model, RegistrationYear, Cylinders, BodyType, CustomValueEstimate
- **Insurance Plan:** SumInsured, CalculatedPremiumPerTerm, CoverCategory, CoverType, Product
- **Financial Metrics:** TotalPremium, TotalClaims

## 🔬 Key Analysis Areas

### Task 1: Exploratory Data Analysis (EDA)
- Data quality assessment and cleaning
- Descriptive statistics and distributions
- Loss ratio analysis by Province, VehicleType, Gender
- Temporal trends and seasonality
- Outlier detection and treatment
- Creative visualizations for insights

### Task 2: Data Version Control (DVC)
- Implement DVC for data reproducibility
- Track data versions (raw → processed → clean)
- Ensure audit trail for regulatory compliance

### Task 3: A/B Hypothesis Testing
Test the following null hypotheses:
- **H₀₁:** No risk differences across provinces
- **H₀₂:** No risk differences between zip codes
- **H₀₃:** No margin differences between zip codes
- **H₀₄:** No risk differences between genders

**Key Metrics:**
- **Claim Frequency:** Proportion of policies with claims
- **Claim Severity:** Average claim amount (given claim occurred)
- **Margin:** TotalPremium - TotalClaims
- **Loss Ratio:** TotalClaims / TotalPremium

### Task 4: Statistical Modeling
Build predictive models for:
1. **Claim Severity Prediction** - Predict TotalClaims for policies with claims
2. **Premium Optimization** - Predict optimal CalculatedPremiumPerTerm
3. **Claim Probability** - Binary classification for claim occurrence

**Models Implemented:**
- Linear Regression
- Decision Trees
- Random Forests
- XGBoost

**Model Interpretation:**
- SHAP (SHapley Additive exPlanations) for feature importance
- LIME (Local Interpretable Model-agnostic Explanations)
- Top influential features with business interpretation

## 💻 Usage

### Loading Data
```python
from core.data_loader import DataLoader

loader = DataLoader()
df = loader.load_data('data/clean/MachineLearningRating_v3.txt', delimiter='|')
```

### Running EDA
```python
from core.eda import EDAEngine
from core.visualizer import Visualizer

eda = EDAEngine(df)
stats = eda.descriptive_statistics()
loss_ratio = eda.calculate_loss_ratio()

viz = Visualizer()
viz.plot_correlation_matrix(df, save_path='reports/correlation_matrix.png')
```

### Hypothesis Testing
```python
from core.metrics import MetricsCalculator
from core.statistical_tests import StatisticalTester
from core.ab_testing import ABTestFramework

metrics = MetricsCalculator()
claim_freq = metrics.calculate_claim_frequency(df)

tester = StatisticalTester()
result = tester.t_test(group_a, group_b)
```

### Model Training
```python
from core.model_trainer import ModelTrainer
from core.model_evaluator import ModelEvaluator
from core.model_interpreter import ModelInterpreter

trainer = ModelTrainer()
model = trainer.train_random_forest(X_train, y_train)

evaluator = ModelEvaluator()
metrics = evaluator.calculate_rmse(y_test, predictions)

interpreter = ModelInterpreter()
shap_values = interpreter.shap_analysis(model, X_test)
```

## 🧪 Testing

Run all tests:
```bash
pytest tests/ -v
```

Run tests with coverage:
```bash
pytest --cov=core tests/
```

## 📈 Key Business Metrics

```python
# Loss Ratio - Primary profitability indicator
loss_ratio = TotalClaims / TotalPremium

# Claim Frequency - Risk likelihood
claim_frequency = (policies_with_claims / total_policies) * 100

# Claim Severity - Risk magnitude
claim_severity = TotalClaims / number_of_claims

# Margin - Profit per policy
margin = TotalPremium - TotalClaims
```

## 📝 Development Workflow

1. **Create feature branch**
   ```bash
   git checkout -b task-X
   ```

2. **Make changes and commit frequently**
   ```bash
   git add .
   git commit -m "Descriptive commit message"
   ```

3. **Run tests before pushing**
   ```bash
   pytest tests/
   ```

4. **Push changes**
   ```bash
   git push origin task-X
   ```

5. **Create Pull Request to merge into main**

## 🎓 Learning Outcomes

- Exploratory Data Analysis (EDA) techniques
- Statistical hypothesis testing (Chi-squared, t-test, z-test, ANOVA)
- Machine learning model development and evaluation
- Model interpretability (SHAP, LIME)
- Data version control with DVC
- Modular OOP Python programming
- Insurance domain knowledge (risk metrics, pricing strategies)

## 📚 Key Dependencies

- **Data Processing:** pandas, numpy
- **Visualization:** matplotlib, seaborn, plotly
- **Statistical Analysis:** scipy, statsmodels
- **Machine Learning:** scikit-learn, xgboost
- **Model Interpretation:** shap, lime
- **Version Control:** dvc
- **Testing:** pytest, pytest-cov
- **Code Quality:** black, flake8

## 🤝 Contributing

1. Follow PEP 8 style guidelines
2. Write comprehensive docstrings (Google style)
3. Include type hints for all functions
4. Write unit tests for all modules
5. Commit frequently with descriptive messages
6. Use meaningful variable and function names

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
