"""
Main Orchestration Module for Trading Predictor with Baseline Comparison
====================================================================

it includes baseline strategy evaluation and comparison
with machine learning models for comprehensive performance analysis.
"""

import os
import sys
import json
import pickle
import logging
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from baselines import macd_crossover_strategy, rsi_threshold_strategy, random_strategy

# Import our custom modules
from features import comprehensive_feature_engineering
from labeling import create_actionable_targets
from models import AdvancedPreprocessor, create_actionable_models, time_series_cross_validation
from evaluate import (evaluate_model_actionable, plot_confusion_matrix, 
                     print_detailed_classification_report, plot_feature_importance,
                     plot_model_comparison, plot_prediction_distribution, 
                     generate_evaluation_summary)

# Configuration and constants
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
sns.set_palette("husl")

# Global configuration for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Initialize logger as None initially
logger = None

# Optional advanced ML libraries availability check
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM not available. Install with: pip install lightgbm")

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up comprehensive logging for the trading predictor.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path. If None, logs to console only.
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger('TradingPredictor')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def validate_input_data(df: pd.DataFrame, required_columns: List[str],
                       allow_null_columns: bool = True, min_rows: int = 100) -> None:
    """
    Validate input data for the trading predictor with flexible validation rules.
    
    Args:
        df: Input DataFrame
        required_columns: List of required column names
        allow_null_columns: Whether to allow columns with all null values
        min_rows: Minimum number of rows required
    
    Raises:
        ValueError: If validation fails
    """
    if df is None or df.empty:
        raise ValueError("Input DataFrame is None or empty")
    
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    # Check for required columns
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check if required columns have all null values
    required_null_columns = []
    for col in required_columns:
        if col in df.columns and df[col].isnull().all():
            required_null_columns.append(col)
    
    if required_null_columns:
        raise ValueError(f"Required columns with all null values: {required_null_columns}")
    
    # Check minimum rows
    if len(df) < min_rows:
        raise ValueError(f"Dataset too small. Has {len(df)} rows, minimum {min_rows} required.")
    
    # Check for completely empty dataset
    if df.select_dtypes(include=[np.number]).empty:
        raise ValueError("No numeric columns found in dataset")
    
    if logger:
        logger.info(f"Input validation passed. Shape: {df.shape}")

def set_random_seeds(seed: int = RANDOM_SEED) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    
    # Set seeds for optional libraries if available
    try:
        import random
        random.seed(seed)
    except ImportError:
        pass
    
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
    
    # Log the seed setting
    if logger:
        logger.info(f"Random seeds set to {seed}")
    else:
        print(f"Random seeds set to {seed}")

class ExperimentLogger:
    """
    Automated experiment logging system for reproducibility and analysis.
    
    This class handles:
    - Parameter logging and serialization
    - Metrics tracking across experiments
    - Model artifact saving
    - Visualization export
    - Experiment comparison utilities
    """
    
    def __init__(self, experiment_name: str, base_dir: str = "experiments"):
        """
        Initialize experiment logger.
        
        Args:
            experiment_name: Name of the current experiment
            base_dir: Base directory for experiment storage
        """
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"{experiment_name}_{self.timestamp}"
        
        # Create experiment directory
        self.experiment_dir = Path(base_dir) / self.experiment_id
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize experiment metadata
        self.metadata = {
            'experiment_name': experiment_name,
            'experiment_id': self.experiment_id,
            'timestamp': self.timestamp,
            'random_seed': RANDOM_SEED,
            'python_version': sys.version,
            'numpy_version': np.__version__,
            'pandas_version': pd.__version__,
            'xgboost_available': XGBOOST_AVAILABLE,
            'lightgbm_available': LIGHTGBM_AVAILABLE,
            'parameters': {},
            'metrics': {},
            'artifacts': []
        }
        
        if logger:
            logger.info(f"Initialized experiment: {self.experiment_id}")
    
    def log_parameters(self, params: Dict[str, Any]) -> None:
        """Log experiment parameters."""
        self.metadata['parameters'].update(params)
        if logger:
            logger.info(f"Logged parameters: {list(params.keys())}")
    
    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log experiment metrics."""
        self.metadata['metrics'].update(metrics)
        if logger:
            logger.info(f"Logged metrics: {list(metrics.keys())}")
    
    def save_artifact(self, obj: Any, filename: str, artifact_type: str = "pickle") -> str:
        """
        Save experiment artifact.
        
        Args:
            obj: Object to save
            filename: Filename for the artifact
            artifact_type: Type of artifact (pickle, json, csv, etc.)
        
        Returns:
            Path to saved artifact
        """
        filepath = self.experiment_dir / filename
        
        try:
            if artifact_type == "pickle":
                with open(filepath, 'wb') as f:
                    pickle.dump(obj, f)
            elif artifact_type == "json":
                with open(filepath, 'w') as f:
                    json.dump(obj, f, indent=2, default=str)
            elif artifact_type == "csv" and hasattr(obj, 'to_csv'):
                obj.to_csv(filepath, index=True)
            else:
                raise ValueError(f"Unsupported artifact type: {artifact_type}")
            
            self.metadata['artifacts'].append({
                'filename': filename,
                'type': artifact_type,
                'timestamp': datetime.now().isoformat()
            })
            
            if logger:
                logger.info(f"Saved artifact: {filename}")
            return str(filepath)
            
        except Exception as e:
            if logger:
                logger.error(f"Failed to save artifact {filename}: {e}")
            raise
    
    def save_plot(self, fig: plt.Figure, filename: str, dpi: int = 300) -> str:
        """Save matplotlib figure as artifact."""
        filepath = self.experiment_dir / filename
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        
        self.metadata['artifacts'].append({
            'filename': filename,
            'type': 'plot',
            'timestamp': datetime.now().isoformat()
        })
        
        if logger:
            logger.info(f"Saved plot: {filename}")
        return str(filepath)
    
    def finalize_experiment(self) -> str:
        """Finalize experiment and save metadata."""
        metadata_path = self.experiment_dir / "experiment_metadata.json"
        
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        
        if logger:
            logger.info(f"Experiment finalized: {self.experiment_id}")
        return str(metadata_path)

class TradingPredictorTests:
    """
    Unit tests for the trading predictor components.
    """
    
    def __init__(self):
        """Initialize test suite."""
        self.logger = logging.getLogger(f'{__name__}.TradingPredictorTests')
        self.test_results = {}
    
    def create_sample_data(self, n_samples: int = 1000, n_tickers: int = 5) -> pd.DataFrame:
        """
        Create synthetic financial data for testing.
        
        Args:
            n_samples: Number of data points per ticker
            n_tickers: Number of different tickers
            
        Returns:
            DataFrame with synthetic financial data
        """
        np.random.seed(RANDOM_SEED)
        
        data = []
        tickers = [f'STOCK_{i:02d}' for i in range(n_tickers)]
        
        for ticker in tickers:
            # Generate synthetic price data with realistic patterns
            dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
            
            # Random walk with drift for price
            returns = np.random.normal(0.0005, 0.02, n_samples)  # Daily returns
            prices = 100 * np.exp(np.cumsum(returns))  # Price from returns
            
            # Generate OHLC data
            high = prices * (1 + np.abs(np.random.normal(0, 0.01, n_samples)))
            low = prices * (1 - np.abs(np.random.normal(0, 0.01, n_samples)))
            volume = np.random.lognormal(10, 1, n_samples)
            
            ticker_data = pd.DataFrame({
                'Date': dates,
                'Ticker': ticker,
                'Open': prices * (1 + np.random.normal(0, 0.005, n_samples)),
                'High': high,
                'Low': low,
                'Close': prices,
                'Volume': volume
            })
            
            data.append(ticker_data)
        
        return pd.concat(data, ignore_index=True)
    
    def test_data_validation(self) -> bool:
        """Test input data validation functionality."""
        if logger:
            logger.info("Testing data validation...")
        else:
            print("Testing data validation...")
        
        try:
            # Test valid data
            valid_data = self.create_sample_data(100, 2)
            required_cols = ['Date', 'Ticker', 'Close', 'Volume']
            validate_input_data(valid_data, required_cols)
            
            # Test invalid data cases
            test_cases = [
                (None, "None input"),
                (pd.DataFrame(), "Empty DataFrame"),
                (valid_data.drop('Close', axis=1), "Missing required column"),
                (valid_data.head(50), "Too few samples")
            ]
            
            for invalid_data, description in test_cases:
                try:
                    validate_input_data(invalid_data, required_cols)
                    if logger:
                        logger.error(f"Validation should have failed for: {description}")
                    else:
                        print(f"ERROR: Validation should have failed for: {description}")
                    return False
                except ValueError:
                    pass  # Expected behavior
            
            if logger:
                logger.info("Data validation tests passed")
            else:
                print("Data validation tests passed")
            return True
            
        except Exception as e:
            if logger:
                logger.error(f"Data validation test failed: {e}")
            else:
                print(f"ERROR: Data validation test failed: {e}")
            return False
    
    def test_feature_engineering(self) -> bool:
        """Test feature engineering functionality."""
        if logger:
            logger.info("Testing feature engineering...")
        else:
            print("Testing feature engineering...")
        
        try:
            # Create test data
            test_data = self.create_sample_data(200, 2)
            
            # Test feature engineering
            features_df = comprehensive_feature_engineering(test_data)
            
            # Validate results
            assert features_df.shape[0] == test_data.shape[0], "Row count mismatch"
            assert features_df.shape[1] > test_data.shape[1], "No new features created"
            
            # Check for expected feature types
            expected_features = ['Return_1D', 'SMA_20', 'RSI_14', 'MACD', 'BB_Position_20']
            for feature in expected_features:
                assert feature in features_df.columns, f"Missing expected feature: {feature}"
            
            # Check for no infinite values
            assert not np.isinf(features_df.select_dtypes(include=[np.number])).any().any(), "Infinite values found"
            
            if logger:
                logger.info("Feature engineering tests passed")
            else:
                print("Feature engineering tests passed")
            return True
            
        except Exception as e:
            if logger:
                logger.error(f"Feature engineering test failed: {e}")
            else:
                print(f"ERROR: Feature engineering test failed: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """
        Run all unit tests and return results.
        
        Returns:
            Dictionary with test names and their pass/fail status
        """
        if logger:
            logger.info("Starting comprehensive unit tests...")
        else:
            print("Starting comprehensive unit tests...")
        
        tests = [
            ('data_validation', self.test_data_validation),
            ('feature_engineering', self.test_feature_engineering)
        ]
        
        results = {}
        for test_name, test_func in tests:
            try:
                results[test_name] = test_func()
            except Exception as e:
                if logger:
                    logger.error(f"Test {test_name} failed with exception: {e}")
                else:
                    print(f"ERROR: Test {test_name} failed with exception: {e}")
                results[test_name] = False
        
        # Summary
        passed = sum(results.values())
        total = len(results)
        if logger:
            logger.info(f"Unit tests completed: {passed}/{total} passed")
        else:
            print(f"Unit tests completed: {passed}/{total} passed")
        
        if passed == total:
            if logger:
                logger.info("🎉 All unit tests passed!")
            else:
                print("🎉 All unit tests passed!")
        else:
            if logger:
                logger.warning(f"⚠️ {total - passed} unit tests failed")
            else:
                print(f"⚠️ {total - passed} unit tests failed")
        
        return results

def convert_target_to_baseline_format(y: pd.Series) -> pd.Series:
    """
    Convert numeric target labels to baseline format.
    
    Args:
        y: Target series with numeric labels (0, 1, 2)
    
    Returns:
        Target series with string labels ('HOLD', 'BUY', 'SELL')
    """
    mapping = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
    return y.map(mapping)

def convert_baseline_to_target_format(predictions: np.ndarray) -> np.ndarray:
    """
    Convert baseline predictions to numeric format.
    
    Args:
        predictions: Array with string predictions ('BUY', 'SELL', 'HOLD')
    
    Returns:
        Array with numeric predictions (0, 1, 2)
    """
    mapping = {'HOLD': 0, 'BUY': 1, 'SELL': 2}
    return np.array([mapping[pred] for pred in predictions])

def evaluate_baseline_strategies(df: pd.DataFrame, y_true: pd.Series) -> Dict[str, Dict]:
    """
    Evaluate baseline trading strategies.
    
    Args:
        df: DataFrame with features needed for baseline strategies
        y_true: True target values (numeric format)
    
    Returns:
        Dictionary with baseline strategy results
    """
    print("📊 Evaluating baseline strategies...")
    baseline_results = {}
    
    # Check required columns for each baseline
    baseline_strategies = []
    
    # MACD Crossover Strategy
    if 'MACD' in df.columns and 'MACD_signal' in df.columns:
        baseline_strategies.append(('MACD_Crossover', macd_crossover_strategy))
    else:
        print("⚠️ MACD columns not found, skipping MACD strategy")
    
    # RSI Strategy
    rsi_cols = [col for col in df.columns if 'RSI' in col]
    if rsi_cols:
        # Use the first RSI column found
        rsi_col = rsi_cols[0]
        df_rsi = df.copy()
        df_rsi['RSI'] = df_rsi[rsi_col]  # Rename for baseline function
        baseline_strategies.append(('RSI_Threshold', lambda x: rsi_threshold_strategy(x)))
    else:
        print("⚠️ RSI column not found, skipping RSI strategy")
        df_rsi = None
    
    # Random Strategy (always available)
    baseline_strategies.append(('Random', random_strategy))
    
    # Evaluate each baseline strategy
    for strategy_name, strategy_func in baseline_strategies:
        try:
            print(f"  🔄 Evaluating {strategy_name}...")
            
            if strategy_name == 'RSI_Threshold' and df_rsi is not None:
                predictions_str = strategy_func(df_rsi)
            else:
                predictions_str = strategy_func(df)
            
            # Convert predictions to numeric format
            predictions_numeric = convert_baseline_to_target_format(predictions_str)
            
            # Evaluate using the same function as ML models
            metrics = evaluate_model_actionable(y_true, predictions_numeric, strategy_name)
            baseline_results[strategy_name] = metrics
            
            print(f"    ✅ {strategy_name}: Acc={metrics['accuracy']:.4f}, "
                  f"F1={metrics['f1_macro']:.4f}, "
                  f"Actionable={metrics['actionability_score']:.2f}")
            
        except Exception as e:
            print(f"    ⚠️ Failed to evaluate {strategy_name}: {e}")
            baseline_results[strategy_name] = {
                'accuracy': 0.0, 'f1_macro': 0.0, 'actionability_score': 0.0,
                'error': str(e)
            }
    
    return baseline_results

def train_and_evaluate_with_baselines(df, target_buy_pct=20, target_sell_pct=20, test_size=0.2, validation_size=0.1):
    """Enhanced training and evaluation pipeline with baseline comparison"""
    print("🚀 Starting enhanced actionable training pipeline with baselines...")
    
    # Prepare features and target
    exclude_cols = ['Date', 'Ticker', 'Target', 'Action_Score', 'Market_Vol_Regime',
                   'Adjusted_Buy_Pct', 'Adjusted_Sell_Pct']
    
    # Add all future return and risk-adjusted return columns to exclude
    future_cols = [col for col in df.columns if col.startswith(('Future_Return_', 'Risk_Adj_Return_'))]
    exclude_cols.extend(future_cols)
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols].copy()
    y = df['Target'].copy()
    dates = df['Date'].copy() if 'Date' in df.columns else None
    
    # Remove rows with missing targets
    valid_mask = ~y.isna()
    X = X[valid_mask]
    y = y[valid_mask]
    if dates is not None:
        dates = dates[valid_mask]
    
    print(f"📊 Dataset shape: {X.shape}")
    print(f"🎯 Target distribution: {dict(pd.Series(y).value_counts().sort_index())}")
    
    # Enhanced time-based splits
    total_size = len(X)
    train_size = int(total_size * (1 - test_size - validation_size))
    val_size = int(total_size * validation_size)
    
    # Create splits
    X_train = X.iloc[:train_size]
    X_val = X.iloc[train_size:train_size + val_size]
    X_test = X.iloc[train_size + val_size:]
    
    y_train = y.iloc[:train_size]
    y_val = y.iloc[train_size:train_size + val_size]
    y_test = y.iloc[train_size + val_size:]
    
    print(f"📊 Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")
    
    # Check class distribution in each split
    print("📊 Class distribution by split:")
    for split_name, split_y in [("Train", y_train), ("Validation", y_val), ("Test", y_test)]:
        dist = pd.Series(split_y).value_counts(normalize=True).sort_index() * 100
        print(f"   {split_name}: " + " | ".join([f"{int(k)}:{v:.1f}%" for k, v in dist.items()]))
    
    # EVALUATE BASELINE STRATEGIES FIRST
    print("\n🎯 BASELINE STRATEGY EVALUATION")
    print("-" * 50)
    baseline_results = evaluate_baseline_strategies(X_test, y_test)
    
    # Preprocessing for ML models
    print("\n🔧 Applying enhanced preprocessing for ML models...")
    preprocessor = AdvancedPreprocessor(random_state=RANDOM_SEED)
    X_train_processed = preprocessor.advanced_preprocessing(X_train, y_train, fit=True)
    X_val_processed = preprocessor.advanced_preprocessing(X_val, fit=False)
    X_test_processed = preprocessor.advanced_preprocessing(X_test, fit=False)
    
    print(f"📊 Processed shapes - Train: {X_train_processed.shape}, Val: {X_val_processed.shape}, Test: {X_test_processed.shape}")
    
    # Create and train ML models
    models = create_actionable_models(random_state=RANDOM_SEED)
    
    print("\n🎓 Training ML models with validation...")
    trained_models = {}
    validation_results = {}
    
    for name, model in models.items():
        print(f"  🔄 Training {name}...")
        try:
            # Train model
            model.fit(X_train_processed, y_train)
            trained_models[name] = model
            
            # Validate on validation set
            y_val_pred = model.predict(X_val_processed)
            val_metrics = evaluate_model_actionable(y_val, y_val_pred, name)
            validation_results[name] = val_metrics
            
            print(f"    ✅ {name} - Val Acc: {val_metrics['accuracy']:.4f}, "
                  f"F1: {val_metrics['f1_macro']:.4f}, "
                  f"Actionable: {val_metrics['actionability_score']:.2f}")
            
        except Exception as e:
            print(f"    ⚠️ Failed to train {name}: {e}")
    
    # Time series cross-validation for top ML models
    print("\n🔄 Performing time series cross-validation for ML models...")
    cv_results = {}
    
    # Select top 5 models based on validation performance
    val_df = pd.DataFrame(validation_results).T
    if not val_df.empty:
        val_df['val_combined_score'] = (val_df['accuracy'] * 0.6) + (val_df['actionability_score'] * 0.4)
        top_models = val_df.nlargest(5, 'val_combined_score').index.tolist()
        
        for name in top_models:
            if name in trained_models:
                print(f"  🔄 CV for {name}...")
                try:
                    cv_result = time_series_cross_validation(
                        X_train_processed, y_train, trained_models[name], n_splits=5
                    )
                    cv_results[name] = cv_result
                    print(f"    ✅ {name} - CV Acc: {cv_result['accuracy_mean']:.4f}±{cv_result['accuracy_std']:.4f}")
                except Exception as e:
                    print(f"    ⚠️ CV failed for {name}: {e}")
    
    # Final evaluation on test set for ML models
    print("\n📊 Final evaluation on test set for ML models...")
    test_results = {}
    best_model_name = None
    best_model_obj = None
    best_y_pred = None
    
    for name, model in trained_models.items():
        print(f"  📈 Testing {name}...")
        try:
            y_test_pred = model.predict(X_test_processed)
            test_metrics = evaluate_model_actionable(y_test, y_test_pred, name)
            test_results[name] = test_metrics
            
            # Store best model for visualization
            if best_model_name is None or (
                'accuracy' in test_metrics and 'actionability_score' in test_metrics and
                (test_metrics['accuracy'] * 0.6 + test_metrics['actionability_score'] * 0.4) >
                (test_results[best_model_name]['accuracy'] * 0.6 + test_results[best_model_name]['actionability_score'] * 0.4)
            ):
                best_model_name = name
                best_model_obj = model
                best_y_pred = y_test_pred
            
            print(f"    ✅ {name}: Acc={test_metrics['accuracy']:.4f}, "
                  f"F1={test_metrics['f1_macro']:.4f}, "
                  f"Actionable={test_metrics['actionability_score']:.2f} "
                  f"(Buy={test_metrics['pred_buy_pct']:.1f}%, Sell={test_metrics['pred_sell_pct']:.1f}%)")
            
        except Exception as e:
            print(f"    ⚠️ Failed to evaluate {name}: {e}")
    
    # Combine all results (ML + Baselines)
    print("\n📊 Combining ML and baseline results...")
    final_results = {}
    
    # Add ML model results
    for name in trained_models.keys():
        final_results[name] = {}
        
        # Add validation results
        if name in validation_results:
            for key, value in validation_results[name].items():
                final_results[name][f'val_{key}'] = value
        
        # Add CV results
        if name in cv_results:
            for key, value in cv_results[name].items():
                final_results[name][f'cv_{key}'] = value
        
        # Add test results
        if name in test_results:
            for key, value in test_results[name].items():
                final_results[name][f'test_{key}'] = value
    
    # Add baseline results (they only have test results)
    for name, metrics in baseline_results.items():
        final_results[f'Baseline_{name}'] = {}
        for key, value in metrics.items():
            if key != 'error':  # Skip error messages
                final_results[f'Baseline_{name}'][f'test_{key}'] = value
    
    # Create comprehensive results DataFrame
    results_df = pd.DataFrame(final_results).T
    
    # Calculate final combined scores
    if 'test_accuracy' in results_df.columns and 'test_actionability_score' in results_df.columns:
        results_df['final_combined_score'] = (results_df['test_accuracy'] * 0.6) + (results_df['test_actionability_score'] * 0.4)
        results_df = results_df.sort_values('final_combined_score', ascending=False)
    elif 'val_accuracy' in results_df.columns and 'val_actionability_score' in results_df.columns:
        results_df['final_combined_score'] = (results_df['val_accuracy'] * 0.6) + (results_df['val_actionability_score'] * 0.4)
        results_df = results_df.sort_values('final_combined_score', ascending=False)
    
    # Generate visualizations and detailed analysis for best model
    if best_model_obj is not None and best_y_pred is not None:
        print(f"\n📊 Generating detailed analysis for best model: {best_model_name}")
        print("=" * 60)
        
        try:
            # Generate confusion matrix
            cm = plot_confusion_matrix(y_test, best_y_pred,
                                      save_path=f'confusion_matrix_{best_model_name}.png')
            
            # Print detailed classification report
            classification_report_dict = print_detailed_classification_report(y_test, best_y_pred)
            
            # Generate feature importance plot (if model supports it)
            if hasattr(best_model_obj, 'feature_importances_'):
                # Create feature names (simplified for processed features)
                feature_names = [f'Feature_{i}' for i in range(X_test_processed.shape[1])]
                feature_importance_df = plot_feature_importance(
                    best_model_obj, feature_names, top_n=20,
                    save_path=f'feature_importance_{best_model_name}.png'
                )
            else:
                print("⚠️ Best model does not support feature importance extraction")
                feature_importance_df = None
            
            # Generate additional plots
            plot_model_comparison(results_df, save_path='model_comparison_with_baselines.png')
            plot_prediction_distribution(y_test, best_y_pred, save_path='prediction_distribution.png')
            generate_evaluation_summary(results_df, best_model_name, save_path='evaluation_summary_with_baselines.txt')
            
        except Exception as e:
            print(f"⚠️ Error generating visualizations: {e}")
            cm = None
            classification_report_dict = None
            feature_importance_df = None
    else:
        print("⚠️ No valid model found for detailed analysis")
        cm = None
        classification_report_dict = None
        feature_importance_df = None
    
    print("✅ Enhanced pipeline with baselines completed!")
    
    return {
        'results': results_df,
        'models': trained_models,
        'baseline_results': baseline_results,
        'validation_results': validation_results,
        'cv_results': cv_results,
        'test_results': test_results,
        'best_model_name': best_model_name,
        'best_model': best_model_obj,
        'confusion_matrix': cm,
        'classification_report': classification_report_dict,
        'feature_importance': feature_importance_df,
        'preprocessing_info': {
            'original_features': len(feature_cols),
            'processed_features': X_train_processed.shape[1],
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test)
        }
    }

def analyze_baseline_vs_ml_performance(results_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze performance differences between baseline strategies and ML models.
    
    Args:
        results_df: DataFrame containing results for all models and baselines
        
    Returns:
        Dictionary with analysis results
    """
    analysis = {}
    
    # Separate baseline and ML results
    baseline_rows = results_df.index.str.startswith('Baseline_')
    ml_rows = ~baseline_rows
    
    baseline_results = results_df[baseline_rows]
    ml_results = results_df[ml_rows]
    
    if baseline_results.empty or ml_results.empty:
        print("⚠️ Cannot perform comparison: missing baseline or ML results")
        return analysis
    
    # Compare key metrics
    key_metrics = ['test_accuracy', 'test_f1_macro', 'test_actionability_score']
    available_metrics = [m for m in key_metrics if m in results_df.columns]
    
    for metric in available_metrics:
        baseline_values = baseline_results[metric].dropna()
        ml_values = ml_results[metric].dropna()
        
        if not baseline_values.empty and not ml_values.empty:
            analysis[metric] = {
                'baseline_best': baseline_values.max(),
                'baseline_avg': baseline_values.mean(),
                'baseline_worst': baseline_values.min(),
                'ml_best': ml_values.max(),
                'ml_avg': ml_values.mean(),
                'ml_worst': ml_values.min(),
                'improvement_best': ml_values.max() - baseline_values.max(),
                'improvement_avg': ml_values.mean() - baseline_values.mean(),
            }
    
    # Find best performers in each category
    if 'final_combined_score' in results_df.columns:
        analysis['best_baseline'] = {
            'name': baseline_results['final_combined_score'].idxmax(),
            'score': baseline_results['final_combined_score'].max()
        }
        analysis['best_ml'] = {
            'name': ml_results['final_combined_score'].idxmax(),
            'score': ml_results['final_combined_score'].max()
        }
        analysis['overall_best'] = {
            'name': results_df['final_combined_score'].idxmax(),
            'score': results_df['final_combined_score'].max()
        }
    
    return analysis

def main():
    """
    Enhanced main function with baseline comparison for IEEE publication-ready stock prediction system.
    
    This function demonstrates the complete workflow including:
    - Comprehensive data exploration and validation
    - Advanced feature engineering with 100+ indicators
    - Sophisticated target creation with risk adjustment
    - Baseline strategy evaluation (MACD, RSI, Random)
    - Robust preprocessing and feature selection
    - Advanced model training with ensemble methods
    - Time-series cross-validation
    - Comprehensive evaluation and visualization
    - Performance comparison between baselines and ML models
    - Automated experiment logging and artifact saving
    """
    print("🚀 ENHANCED ACTIONABLE STOCK PREDICTION SYSTEM WITH BASELINE COMPARISON")
    print("=" * 80)
    print("🎯 IEEE Publication-Ready Trading Signal Prediction Framework")
    print("📊 Features: Advanced ML, Risk-Adjusted Targets, Baseline Comparison")
    print("🔬 Includes: MACD, RSI, Random Baselines + ML Models")
    print("🎓 Comprehensive Validation & Performance Analysis")
    print("=" * 80)
    
    # Initialize logger FIRST
    global logger
    logger = setup_logging("INFO", f"trading_predictor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # Set random seeds for reproducibility (logger is now available)
    set_random_seeds(RANDOM_SEED)
    
    # Run unit tests first
    print("\n🧪 RUNNING UNIT TESTS")
    print("-" * 50)
    test_suite = TradingPredictorTests()
    test_results = test_suite.run_all_tests()
    
    if not all(test_results.values()):
        print("⚠️ Some unit tests failed. Proceeding with caution...")
    
    # Initialize experiment logging
    experiment_name = f"baseline_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    experiment_logger = ExperimentLogger(experiment_name)
    
    # Log experiment parameters
    experiment_logger.log_parameters({
        'unit_tests_passed': all(test_results.values()),
        'failed_tests': [k for k, v in test_results.items() if not v],
        'baselines_included': ['MACD_Crossover', 'RSI_Threshold', 'Random'],
        'system_info': {
            'python_version': sys.version,
            'numpy_version': np.__version__,
            'pandas_version': pd.__version__,
            'xgboost_available': XGBOOST_AVAILABLE,
            'lightgbm_available': LIGHTGBM_AVAILABLE
        }
    })
    
    # Load and validate data
    print("\n📂 LOADING AND VALIDATING DATA")
    print("-" * 50)
    
    data_file = "us_stocks_5years_FIXED_SORTED.csv"
    
    if not os.path.exists(data_file):
        print(f"❌ Data file '{data_file}' not found!")
        print("📝 Creating synthetic data for demonstration...")
        
        # Create synthetic data for demonstration
        test_suite = TradingPredictorTests()
        df = test_suite.create_sample_data(n_samples=5000, n_tickers=10)
        print(f"✅ Created synthetic data. Shape: {df.shape}")
        
        # Save synthetic data
        df.to_csv("synthetic_stock_data.csv", index=False)
        data_file = "synthetic_stock_data.csv"
    else:
        print(f"📂 Loading data from {data_file}...")
        try:
            df = pd.read_csv(data_file)
            print(f"✅ Data loaded successfully. Shape: {df.shape}")
            
            # Sample data if too large
            if len(df) > 100000:
                print("📊 Sampling data for processing...")
                tickers = df['Ticker'].unique()
                sampled_tickers = np.random.choice(tickers, size=min(10, len(tickers)), replace=False)
                df = df[df['Ticker'].isin(sampled_tickers)].copy()
                print(f"📊 Sampled data shape: {df.shape}")
                
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return
    
    # Validate input data
    try:
        required_columns = ['Date', 'Ticker', 'Close', 'Volume']
        validate_input_data(df, required_columns)
        print("✅ Data validation passed")
    except ValueError as e:
        print(f"❌ Data validation failed: {e}")
        return
    
    # Feature engineering
    print("\n🔧 COMPREHENSIVE FEATURE ENGINEERING")
    print("-" * 50)
    df_features = comprehensive_feature_engineering(df)
    
    # Create actionable targets
    print("\n🎯 ACTIONABLE TARGET CREATION")
    print("-" * 50)
    df_with_targets = create_actionable_targets(
        df_features,
        target_buy_pct=20,
        target_sell_pct=20
    )
    
    # Train and evaluate with baselines
    print("\n🎓 ENHANCED TRAINING & EVALUATION WITH BASELINES")
    print("-" * 50)
    pipeline_results = train_and_evaluate_with_baselines(df_with_targets)
    
    # Analyze baseline vs ML performance
    print("\n🔍 BASELINE VS ML PERFORMANCE ANALYSIS")
    print("-" * 50)
    comparison_analysis = analyze_baseline_vs_ml_performance(pipeline_results['results'])
    
    if comparison_analysis:
        print("📊 Performance Comparison Summary:")
        for metric, stats in comparison_analysis.items():
            if isinstance(stats, dict) and 'baseline_best' in stats:
                improvement = stats['improvement_best']
                print(f"  {metric}: ML Best vs Baseline Best = {improvement:+.4f} "
                      f"({improvement*100:+.2f}%)")
    
    # Save all results and artifacts
    print("\n💾 SAVING EXPERIMENT ARTIFACTS")
    print("-" * 50)
    
    # Save results
    experiment_logger.save_artifact(
        pipeline_results['results'], 'model_results_with_baselines.csv', 'csv'
    )
    
    # Save baseline results separately
    experiment_logger.save_artifact(
        pipeline_results['baseline_results'], 'baseline_results.json', 'json'
    )
    
    # Save comparison analysis
    experiment_logger.save_artifact(
        comparison_analysis, 'baseline_vs_ml_analysis.json', 'json'
    )
    
    # Save best model
    if 'best_model' in pipeline_results and pipeline_results['best_model']:
        experiment_logger.save_artifact(
            pipeline_results['best_model'], 'best_model.pkl', 'pickle'
        )
    
    # Save preprocessing artifacts
    experiment_logger.save_artifact(
        pipeline_results['preprocessing_info'], 'preprocessing_info.json', 'json'
    )
    
    # Log final metrics
    if not pipeline_results['results'].empty:
        best_metrics = pipeline_results['results'].iloc[0].to_dict()
        experiment_logger.log_metrics(best_metrics)
    
    # Finalize experiment
    metadata_path = experiment_logger.finalize_experiment()
    print(f"📋 Experiment metadata saved: {metadata_path}")
    
    # Display comprehensive results summary
    print("\n📊 COMPREHENSIVE RESULTS SUMMARY WITH BASELINES")
    print("=" * 70)
    results_df = pipeline_results['results']
    
    if not results_df.empty:
        # Separate baseline and ML results for display
        baseline_mask = results_df.index.str.startswith('Baseline_')
        ml_mask = ~baseline_mask
        
        baseline_results = results_df[baseline_mask]
        ml_results = results_df[ml_mask]
        
        # Show baseline results
        if not baseline_results.empty:
            print("\n🎯 BASELINE STRATEGY RESULTS:")
            baseline_display_cols = []
            for col in ['test_accuracy', 'test_f1_macro', 'test_actionability_score']:
                if col in baseline_results.columns:
                    baseline_display_cols.append(col)
            
            if baseline_display_cols:
                baseline_display = baseline_results[baseline_display_cols].round(4)
                baseline_display.index = baseline_display.index.str.replace('Baseline_', '')
                print(baseline_display)
        
        # Show ML results
        if not ml_results.empty:
            print("\n🤖 MACHINE LEARNING MODEL RESULTS:")
            ml_display_cols = []
            for col in ['test_accuracy', 'test_f1_macro', 'test_actionability_score', 'val_accuracy']:
                if col in ml_results.columns:
                    ml_display_cols.append(col)
            
            if ml_display_cols:
                print(ml_results[ml_display_cols].round(4))
        
        # Overall best performer
        if 'final_combined_score' in results_df.columns:
            best_overall = results_df.iloc[0]
            best_name = results_df.index[0]
            is_baseline = best_name.startswith('Baseline_')
            
            print(f"\n🏆 OVERALL BEST PERFORMER: {best_name}")
            print(f"🎯 Type: {'Baseline Strategy' if is_baseline else 'Machine Learning Model'}")
            
            # Display best metrics
            for metric_type in ['test_', 'val_', '']:
                accuracy_key = f"{metric_type}accuracy"
                if accuracy_key in best_overall and pd.notna(best_overall[accuracy_key]):
                    print(f"📊 Accuracy: {best_overall[accuracy_key]:.4f} ({best_overall[accuracy_key]*100:.2f}%)")
                    break
            
            for metric_type in ['test_', 'val_', '']:
                f1_key = f"{metric_type}f1_macro"
                if f1_key in best_overall and pd.notna(best_overall[f1_key]):
                    print(f"📊 F1-Score: {best_overall[f1_key]:.4f}")
                    break
            
            for metric_type in ['test_', 'val_', '']:
                actionability_key = f"{metric_type}actionability_score"
                if actionability_key in best_overall and pd.notna(best_overall[actionability_key]):
                    print(f"📊 Actionability: {best_overall[actionability_key]:.4f}")
                    break
            
            print(f"📊 Combined Score: {best_overall['final_combined_score']:.4f}")
    
    # Performance insights
    if comparison_analysis:
        print(f"\n💡 KEY INSIGHTS:")
        if 'best_baseline' in comparison_analysis and 'best_ml' in comparison_analysis:
            best_baseline = comparison_analysis['best_baseline']
            best_ml = comparison_analysis['best_ml']
            
            if best_ml['score'] > best_baseline['score']:
                improvement = ((best_ml['score'] - best_baseline['score']) / best_baseline['score']) * 100
                print(f"✅ ML models outperform baselines by {improvement:.2f}%")
                print(f"   Best ML: {best_ml['name']} (Score: {best_ml['score']:.4f})")
                print(f"   Best Baseline: {best_baseline['name']} (Score: {best_baseline['score']:.4f})")
            else:
                print(f"⚠️ Best baseline strategy outperforms ML models!")
                print(f"   Consider: Feature engineering, model tuning, or data quality issues")
    
    print(f"\n🎉 ENHANCED PIPELINE WITH BASELINE COMPARISON COMPLETED!")
    print("📊 All artifacts saved to experiment directory")
    print("🔬 Ready for IEEE publication analysis")
    print("📈 Baseline vs ML comparison available for research insights")
    
    return pipeline_results

if __name__ == "__main__":

    results = main()
