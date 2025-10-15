# ML WSL Boilerplate

> **A modern, production-ready machine learning project template for WSL environments**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: Ruff](https://img.shields.io/badge/Code%20style-Ruff-black.svg)](https://github.com/astral-sh/ruff)
[![Testing: pytest](https://img.shields.io/badge/Testing-pytest-orange.svg)](https://pytest.org)

A comprehensive, senior-level machine learning boilerplate that eliminates setup friction and provides a robust foundation for ML projects. Built specifically for WSL environments with modern development tools and best practices.

## Features

### Project Structure
- **Modular architecture** with clear separation of concerns
- **Source code organization** (`src/`, `tests/`, `config/`, `scripts/`)
- **Data pipeline management** with validation and preprocessing
- **Model training framework** with multiple algorithm support

### Configuration Management
- **Hierarchical YAML configuration** with OmegaConf
- **Environment variable overrides** for flexible deployment
- **Type-safe configuration access** with validation
- **Multiple environment support** (dev, staging, prod)

### Data Science Stack
- **Modern ML libraries**: XGBoost, LightGBM, scikit-learn
- **Data processing**: Pandas, NumPy with custom DataLoader
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Experiment tracking**: MLflow integration

### Testing & Quality
- **Comprehensive test suite** with pytest and coverage
- **Code quality tools**: Ruff (linting + formatting)
- **Pre-commit hooks** for automated quality checks
- **Type checking** with proper type hints

### Developer Experience
- **VS Code optimization** with extensions and settings
- **Jupyter Lab integration** with auto-reload
- **Rich logging** with beautiful console output
- **Makefile automation** for common tasks
- **Zsh + Powerlevel10k** terminal enhancement

### Production Ready
- **Model persistence** and versioning
- **Cross-validation** and hyperparameter optimization
- **MLflow experiment tracking** and model registry
- **Structured logging** with file rotation
- **Error handling** and validation

## Quick Start

### Prerequisites
- **WSL2** (Windows Subsystem for Linux)
- **Conda/Mamba** (Miniforge recommended)
- **VS Code** with Remote-WSL extension

### 1. Clone and Setup
```bash
git clone https://github.com/yourusername/ml-wsl-boilerplate.git
cd ml-wsl-boilerplate

# Run automated setup
./setup.sh
```

### 2. Activate Environment
```bash
conda activate ml-wsl-base
```

### 3. Quick Commands
```bash
# See all available commands
make help

# Run tests
make test

# Train a model
make train

# Start Jupyter Lab
make notebook

# Code quality checks
make lint

# Auto-format code
make format
```

## Project Structure

```
ml-wsl-boilerplate/
├── src/                     # Source code
│   ├── data/                # Data loading and processing
│   ├── models/              # Model training and evaluation
│   └── utils/               # Utilities (config, logging)
├── tests/                   # Test suite
├── notebooks/               # Jupyter notebooks
├── config/                  # Configuration files
├── scripts/                 # Training and deployment scripts
├── data/                    # Data storage
│   ├── raw/                 # Raw data
│   ├── processed/           # Processed data
│   └── features/            # Feature stores
├── models/                  # Trained models
├── logs/                    # Application logs
├── outputs/                 # Results and reports
├── .vscode/                 # VS Code settings
├── Makefile                 # Automation commands
├── pyproject.toml           # Python project config
└── environment.yml          # Conda environment
```

## 🔧 Configuration

The project uses **hierarchical YAML configuration** with environment variable overrides:

```yaml
# config/config.yaml
project:
  name: "ml-wsl-boilerplate"
  version: "0.1.0"

data:
  train_test_split: 0.8
  validation_split: 0.1
  random_state: 42

model:
  type: "xgboost"  # xgboost, lightgbm, sklearn
  xgboost:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1
```

### Environment Variables
Override any config with environment variables:
```bash
export ML_MODEL_TYPE=lightgbm
export ML_LOG_LEVEL=DEBUG
export ML_RANDOM_SEED=123
```

## Model Training

### Command Line Training
```bash
# Basic training
python scripts/train.py

# With specific model
python scripts/train.py --model xgboost

# With custom config
python scripts/train.py --config config/experiment1.yaml

# Skip cross-validation for speed
python scripts/train.py --no-cv

# Save model after training
python scripts/train.py --save-model
```

### Programmatic Training
```python
from src.models.trainer import get_trainer
from src.utils.config import get_config

# Load configuration
config = get_config()

# Get trainer for specific model
trainer = get_trainer("xgboost")

# Train with cross-validation
cv_results = trainer.cross_validate(X_train, y_train)

# Train final model
metrics = trainer.fit(X_train, y_train, validation_data=(X_val, y_val))

# Evaluate
test_metrics = trainer.evaluate(X_test, y_test)

# Save model
model_path = trainer.save_model()
```

## Data Processing

### Using the DataLoader
```python
from src.data.loader import DataLoader

# Initialize loader
loader = DataLoader(data_dir="data/raw")

# Load and validate data
df = loader.load_csv("dataset.csv")
validation = loader.validate_data(df)

# Clean data
df_clean = loader.clean_data(
    df,
    drop_duplicates=True,
    fill_missing="mean"
)

# Temporal split for time series
train_df, test_df = train_test_split_temporal(
    df, "date_column", test_size=0.2
)
```

## Testing

### Running Tests
```bash
# All tests with coverage
make test

# Fast tests (no coverage)
make test-fast

# Unit tests only
make test-unit

# Integration tests
make test-integration

# Watch mode (requires entr)
make watch-test
```

### Writing Tests
```python
# tests/test_my_feature.py
import pytest
from src.data.loader import DataLoader

def test_data_loader_csv(sample_csv_file):
    loader = DataLoader()
    df = loader.load_csv(sample_csv_file.name)
    assert len(df) > 0
    assert "target" in df.columns
```

## Logging

### Console Logging
```python
from src.utils.logging_config import get_logger

logger = get_logger(component="my_module")

# Rich formatted logging
logger.info("[bold blue]Starting training[/bold blue]")
logger.log_metrics({"accuracy": 0.95, "f1": 0.92})
logger.log_timer("Model training", 45.2)
```

### Structured File Logging
All logs are automatically saved to `logs/` with rotation and structured formatting.

## VS Code Integration

### Optimized Settings
- **Python environment** auto-detection
- **Ruff integration** for linting and formatting
- **Jupyter support** with auto-reload
- **Debugging configurations** for training scripts
- **Testing integration** with pytest discovery

### Debug Configurations
- Python: Current File - Debug any Python file
- Train Model - Debug the training pipeline
- Run Tests - Debug specific tests
- Data Loader Debug - Debug data processing

## 📊 MLflow Tracking

### Automatic Experiment Tracking
```python
# MLflow is automatically configured
# All training runs are tracked with:
# - Parameters (model config)
# - Metrics (accuracy, f1, etc.)
# - Artifacts (model files)
# - Logs (training progress)
```

### View Experiments
```bash
# Start MLflow UI
mlflow ui --backend-store-uri file:./mlruns

# Open browser to http://localhost:5000
```

## 🛠️ Development Workflow

### 1. Development Cycle
```bash
# Quick development cycle
make dev  # format + fast tests

# Full quality check
make check  # lint + test

# CI pipeline
make ci  # clean + lint + test
```

### 2. Adding New Models
```python
# src/models/my_model.py
from .trainer import BaseTrainer

class MyModelTrainer(BaseTrainer):
    def create_model(self):
        # Your model implementation
        return MyModel(**self.config.get_model_config())
```

### 3. Custom Configuration
```yaml
# config/my_experiment.yaml
model:
  type: "my_model"
  my_model:
    param1: value1
    param2: value2
```

## 🚀 Deployment

### Model Artifacts
Trained models include:
- **Serialized model** (joblib/pickle)
- **Configuration snapshot** (experiment reproducibility)
- **Training metrics** (performance tracking)
- **Feature metadata** (preprocessing info)

### Production Checklist
- [ ] Model validation tests pass
- [ ] Performance meets requirements
- [ ] Configuration is environment-specific
- [ ] Logging and monitoring setup
- [ ] Error handling implemented
- [ ] Documentation updated

## 🤝 Contributing

### Code Quality Standards
- **Type hints** for all functions
- **Docstrings** for classes and methods
- **Unit tests** for new functionality
- **Ruff formatting** (automatic with pre-commit)
- **Configuration-driven** behavior

### Commit Process
```bash
# Pre-commit hooks run automatically
git add .
git commit -m "feat: add new model trainer"

# Or manually run checks
make lint
make test
```

## 📚 Advanced Features

### Hyperparameter Optimization
```python
# Enable in configuration
training:
  hyperopt:
    enabled: true
    n_trials: 100
    timeout: 3600
```

### Custom Metrics
```python
# Add to training configuration
training:
  metrics:
    - "accuracy"
    - "precision"
    - "recall"
    - "custom_metric"
```

### Model Ensemble
```python
# Combine multiple models
ensemble = ModelEnsemble([
    get_trainer("xgboost"),
    get_trainer("lightgbm"),
    get_trainer("sklearn")
])
```

## Troubleshooting

### Common Issues

#### Environment Issues
```bash
# Recreate environment
conda env remove -n ml-wsl-base
make setup
```

#### Import Errors
```bash
# Check Python path
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
```

#### Performance Issues
```bash
# Check GPU availability
python -c "import xgboost as xgb; print(xgb.build_info())"
```

### Getting Help
1. Check the [Issues](https://github.com/yourusername/ml-wsl-boilerplate/issues)
2. Review configuration in `config/config.yaml`
3. Check logs in `logs/` directory
4. Run `make help` for available commands

## Performance Tips

### Speed Optimization
- Use **Mamba** instead of Conda (automatically detected)
- Enable **parallel processing** (`n_jobs=-1`)
- **GPU support** for XGBoost/LightGBM (if available)
- **Feature selection** for large datasets

### Memory Optimization
- **Chunked processing** for large datasets
- **Memory mapping** for persistent data
- **Garbage collection** in training loops
- **Efficient data types** (int32 vs int64)

## Roadmap

### Planned Features
- [ ] **Docker support** for containerized deployment
- [ ] **Kubernetes manifests** for scalable deployment
- [ ] **Data versioning** with DVC integration
- [ ] **Model monitoring** and drift detection
- [ ] **A/B testing** framework
- [ ] **Automated retraining** pipelines
- [ ] **SHAP integration** for model explainability
- [ ] **FastAPI deployment** template

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Ruff** for fast Python tooling
- **MLflow** for experiment tracking
- **OmegaConf** for configuration management
- **Rich** for formatted terminal output
- **pytest** for testing framework
- **Miniforge** for Python environment management

---

Built with care for the ML community

Ready to accelerate your machine learning projects? Star this repository and start building!
