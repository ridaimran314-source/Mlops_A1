# .PHONY: all setup venv-create venv-delete venv-info download-data preprocess features train predict evaluate clean help

# Python settings
PYTHON := python
VENV_DIR := venv
VENV_PYTHON := $(VENV_DIR)\Scripts\python.exe
VENV_PIP := $(VENV_DIR)\Scripts\pip.exe

# Directories
DATA_RAW := data/raw
DATA_PROCESSED := data/processed
FEATURES_DIR := features
MODELS_DIR := models
RESULTS_DIR := results

# File dependencies - BOTH TRAIN AND TEST
TRAIN_RAW := $(DATA_RAW)/train.csv
TEST_RAW := $(DATA_RAW)/test.csv
TRAIN_PROCESSED := $(DATA_PROCESSED)/train_processed.csv
TEST_PROCESSED := $(DATA_PROCESSED)/test_processed.csv
TRAIN_FEATURES := $(FEATURES_DIR)/train_features.csv
TEST_FEATURES := $(FEATURES_DIR)/test_features.csv
MODEL_FILE := $(MODELS_DIR)/model.pkl
SCALER_FILE := $(MODELS_DIR)/scaler.pkl
PREDICTIONS := $(RESULTS_DIR)/predictions.csv
METRICS := $(RESULTS_DIR)/metrics.txt

# Use venv Python if exists, otherwise system Python
ifeq ($(shell if exist $(VENV_PYTHON) echo 1),1)
    PY := $(VENV_PYTHON)
else
    PY := $(PYTHON)
endif

# ========================================
# VIRTUAL ENVIRONMENT TARGETS
# ========================================

# Create virtual environment
venv-create:
	@if not exist $(VENV_DIR) (echo Creating virtual environment... && $(PYTHON) -m venv $(VENV_DIR)) else (echo Virtual environment already exists)


# Delete virtual environment
venv-delete:
	@if exist $(VENV_DIR) ( \
		echo Deleting virtual environment... && \
		rmdir /S /Q $(VENV_DIR) && \
		echo Virtual environment deleted \
	) else ( \
		echo Virtual environment does not exist \
	)


# ========================================
# PIPELINE TARGETS
# ========================================

# Default target - runs complete pipeline
all: setup download-data preprocess features train predict evaluate
	@echo.
	@echo ========================================
	@echo    Pipeline Completed Successfully!
	@echo ========================================

# Install dependencies (auto-creates venv)
setup: venv-create
	@echo Installing dependencies...
	@$(VENV_PYTHON) -m pip install --upgrade pip
	@$(VENV_PYTHON) -m pip install -r requirements.txt
	@echo Dependencies installed in virtual environment

# Download Titanic dataset (creates both train and test)
download-data: $(TRAIN_RAW) $(TEST_RAW)

$(TRAIN_RAW) $(TEST_RAW): scripts/download_data.py
	@echo Downloading Titanic dataset...
	@$(PY) scripts/download_data.py 

# Preprocess data (processes both train and test)
preprocess: $(TRAIN_PROCESSED) $(TEST_PROCESSED)

$(TRAIN_PROCESSED) $(TEST_PROCESSED): $(TRAIN_RAW) $(TEST_RAW) scripts/preprocessed.py
	@echo Preprocessing data...
	@$(PY) scripts/preprocessed.py

# Feature engineering (engineers both train and test)
features: $(TRAIN_FEATURES) $(TEST_FEATURES)

$(TRAIN_FEATURES) $(TEST_FEATURES): $(TRAIN_PROCESSED) $(TEST_PROCESSED) scripts/feature_engineering.py
	@echo Engineering features...
	@$(PY) scripts/feature_engineering.py

# Train model (only uses train features)
train: $(MODEL_FILE)

$(MODEL_FILE): $(TRAIN_FEATURES) scripts/train.py
	@echo Training model...
	@$(PY) scripts/train.py

# Generate predictions (uses REAL test features)
predict: $(PREDICTIONS)

$(PREDICTIONS): $(MODEL_FILE) $(SCALER_FILE) $(TEST_FEATURES) scripts/predict.py
	@echo Generating predictions on test data...
	@$(PY) scripts/predict.py

# Evaluate model
evaluate: $(METRICS)

$(METRICS): $(PREDICTIONS) scripts/evaluate.py
	@echo Evaluating model...
	@$(PY) scripts/evaluate.py
	@echo.
	@type "$(METRICS)"

# Clean generated files
clean:
	@echo Cleaning generated files...
	-@if exist "$(DATA_RAW)\*.csv" del /Q "$(DATA_RAW)\*.csv"
	-@if exist "$(DATA_PROCESSED)\*.csv" del /Q "$(DATA_PROCESSED)\*.csv"
	-@if exist "$(FEATURES_DIR)\*.csv" del /Q "$(FEATURES_DIR)\*.csv"
	-@if exist "$(MODELS_DIR)\*.pkl" del /Q "$(MODELS_DIR)\*.pkl"
	-@if exist "$(RESULTS_DIR)\*.csv" del /Q "$(RESULTS_DIR)\*.csv"
	-@if exist "$(RESULTS_DIR)\*.txt" del /Q "$(RESULTS_DIR)\*.txt"
	-@if exist "__pycache__" rmdir /S /Q "__pycache__"
	-@if exist "scripts\__pycache__" rmdir /S /Q "scripts\__pycache__"
	@echo Cleaned all generated artifacts



# Help target
help:
	@echo.
	@echo ========================================
	@echo   Titanic MLOps Pipeline - Make Targets
	@echo ========================================
	@echo.
	@echo Virtual Environment:
	@echo   make venv-create     - Create virtual environment
	@echo   make venv-delete     - Delete virtual environment
	@echo   make venv-info       - Show venv information
	@echo.
	@echo Pipeline Stages:
	@echo   make setup           - Install dependencies (auto-creates venv)
	@echo   make download-data   - Download Titanic dataset
	@echo   make preprocess      - Preprocess train and test data
	@echo   make features        - Engineer features for both datasets
	@echo   make train           - Train the ML model
	@echo   make predict         - Generate predictions on test data
	@echo   make evaluate        - Evaluate model performance
	@echo   make all             - Run entire pipeline (recommended)
	@echo   make clean           - Remove generated files
	@echo   make help            - Show this help message
	@echo.
	@echo Note: 'make all' automatically creates venv if needed!
	@echo.