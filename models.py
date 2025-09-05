"""
Enhanced Models and Preprocessing Module for Trading Predictor
============================================================

This module contains multiple parameter configurations for each model type,
allowing for comprehensive experimentation and optimization.

Features include:
- Multiple parameter sets for each algorithm
- Conservative, balanced, and aggressive configurations
- Hyperparameter variations for better ensemble diversity
- Advanced preprocessing with adaptive scaling
"""

import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import logging
from typing import Dict, Any, Optional, List

# Scikit-learn imports
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold, RFE
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                            ExtraTreesClassifier, VotingClassifier, AdaBoostClassifier)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
# Add these with the other imports
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.base import BaseEstimator, ClassifierMixin
    import tensorflow as tf
    tf.random.set_seed(42)  # For reproducibility
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
# Optional advanced ML libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# Initialize logger
logger = logging.getLogger(__name__)

class AdvancedPreprocessor:
    """Advanced preprocessing pipeline with adaptive methods"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.outlier_bounds = {}
        self.outlier_stats = {}
        self.scalers = {}
        self.scaling_methods = {}
        self.variance_selector = None
        self.correlation_mask = None
        self.feature_selector = None
        self.rfe_selector = None
        self.use_rfe = False
        
    def advanced_preprocessing(self, X, y=None, fit=True):
        """Enhanced preprocessing with better outlier handling and feature selection"""
        print("🔧 Advanced preprocessing...")
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        if fit:
            print("📊 Analyzing data distribution and outliers...")
            # Enhanced outlier detection using multiple methods
            self.outlier_bounds = {}
            self.outlier_stats = {}
            
            for col in numeric_cols:
                col_data = X[col].dropna()
                if len(col_data) == 0:
                    continue
                
                # Method 1: IQR-based (more conservative)
                Q1 = col_data.quantile(0.05)
                Q3 = col_data.quantile(0.95)
                IQR = Q3 - Q1
                iqr_lower = Q1 - 2.0 * IQR
                iqr_upper = Q3 + 2.0 * IQR
                
                # Method 2: Z-score based (for normal distributions)
                mean_val = col_data.mean()
                std_val = col_data.std()
                z_lower = mean_val - 3 * std_val
                z_upper = mean_val + 3 * std_val
                
                # Method 3: Percentile-based (most robust)
                perc_lower = col_data.quantile(0.01)
                perc_upper = col_data.quantile(0.99)
                
                # Use the most conservative bounds
                final_lower = max(iqr_lower, z_lower, perc_lower)
                final_upper = min(iqr_upper, z_upper, perc_upper)
                
                self.outlier_bounds[col] = (final_lower, final_upper)
                self.outlier_stats[col] = {
                    'mean': mean_val,
                    'std': std_val,
                    'median': col_data.median(),
                    'skew': skew(col_data),
                    'kurt': kurtosis(col_data)
                }
        
        # Apply outlier bounds with smart handling
        print("🧹 Handling outliers...")
        X_processed = X.copy()
        outlier_counts = {}
        
        for col, (lower, upper) in self.outlier_bounds.items():
            if col in X_processed.columns:
                original_outliers = ((X_processed[col] < lower) | (X_processed[col] > upper)).sum()
                outlier_counts[col] = original_outliers
                
                # Smart clipping: use median for extreme outliers, gradual adjustment for mild ones
                col_data = X_processed[col]
                median_val = self.outlier_stats[col]['median']
                
                # For extreme outliers, replace with median
                extreme_mask = (col_data < lower * 0.5) | (col_data > upper * 1.5)
                X_processed.loc[extreme_mask, col] = median_val
                
                # For mild outliers, use clipping
                mild_mask = ~extreme_mask
                X_processed.loc[mild_mask, col] = X_processed.loc[mild_mask, col].clip(lower, upper)
        
        if fit:
            print("📏 Selecting optimal scaling method...")
            # Choose scaling method based on data distribution
            self.scalers = {}
            self.scaling_methods = {}
            
            for col in numeric_cols:
                if col in self.outlier_stats:
                    skew_val = abs(self.outlier_stats[col]['skew'])
                    kurt_val = abs(self.outlier_stats[col]['kurt'])
                    
                    # Choose scaler based on distribution characteristics
                    if skew_val > 2 or kurt_val > 7:  # Highly skewed or heavy-tailed
                        scaler = RobustScaler()
                        method = 'robust'
                    elif skew_val > 1:  # Moderately skewed
                        scaler = RobustScaler()
                        method = 'robust'
                    else:  # Approximately normal
                        scaler = StandardScaler()
                        method = 'standard'
                    
                    self.scalers[col] = scaler
                    self.scaling_methods[col] = method
            
            # Fit scalers
            for col, scaler in self.scalers.items():
                if col in X_processed.columns:
                    scaler.fit(X_processed[[col]])
            
            if y is not None:
                print("🎯 Enhanced feature selection...")
                
                # Apply scaling
                X_scaled = X_processed.copy()
                for col, scaler in self.scalers.items():
                    if col in X_scaled.columns:
                        X_scaled[col] = scaler.transform(X_scaled[[col]]).flatten()
                
                # Step 1: Remove low-variance features
                print("   📊 Removing low-variance features...")
                self.variance_selector = VarianceThreshold(threshold=0.005)
                X_var = self.variance_selector.fit_transform(X_scaled[numeric_cols])
                remaining_features = numeric_cols[self.variance_selector.get_support()]
                
                # Step 2: Remove highly correlated features
                print("   🔗 Removing highly correlated features...")
                if len(remaining_features) > 1:
                    corr_matrix = pd.DataFrame(X_var, columns=remaining_features).corr().abs()
                    upper_triangle = corr_matrix.where(
                        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                    )
                    
                    # Find features with correlation > 0.95
                    high_corr_features = [column for column in upper_triangle.columns
                                        if any(upper_triangle[column] > 0.95)]
                    
                    # Remove highly correlated features
                    features_to_keep = [f for f in remaining_features if f not in high_corr_features]
                    self.correlation_mask = np.array([f in features_to_keep for f in remaining_features])
                    X_var = X_var[:, self.correlation_mask]
                    remaining_features = np.array(features_to_keep)
                else:
                    self.correlation_mask = np.ones(len(remaining_features), dtype=bool)
                
                # Step 3: Mutual information feature selection
                print("   🧠 Mutual information feature selection...")
                n_features = min(50, max(20, X_var.shape[1] // 2))
                self.feature_selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
                X_selected = self.feature_selector.fit_transform(X_var, y)
                
                # Step 4: Recursive feature elimination with cross-validation (optional for top features)
                if X_selected.shape[1] > 30:
                    print("   🔄 Recursive feature elimination...")
                    rf_selector = RandomForestClassifier(n_estimators=50, random_state=self.random_state, n_jobs=-1)
                    
                    final_n_features = min(30, X_selected.shape[1])
                    self.rfe_selector = RFE(estimator=rf_selector, n_features_to_select=final_n_features, step=0.1)
                    self.rfe_selector.fit(X_selected, y)
                    self.use_rfe = True
                else:
                    self.use_rfe = False
                
                expected_features = final_n_features if hasattr(self, 'rfe_selector') else X_selected.shape[1]
                print(f"📊 Feature selection pipeline: {len(numeric_cols)} -> {X_selected.shape[1]} -> {expected_features} features")
        
        # Apply preprocessing pipeline
        print("⚙️ Applying preprocessing pipeline...")
        
        # Apply scaling
        X_scaled = X_processed.copy()
        for col, scaler in self.scalers.items():
            if col in X_scaled.columns:
                X_scaled[col] = scaler.transform(X_scaled[[col]]).flatten()
        
        # Apply feature selection steps
        X_var = self.variance_selector.transform(X_scaled[numeric_cols])
        X_var = X_var[:, self.correlation_mask]
        X_selected = self.feature_selector.transform(X_var)
        
        if hasattr(self, 'use_rfe') and self.use_rfe:
            X_final = self.rfe_selector.transform(X_selected)
        else:
            X_final = X_selected
        
        return X_final
class LSTMClassifier(BaseEstimator, ClassifierMixin):
    """LSTM Classifier wrapper to work with sklearn interface"""
    
    def __init__(self, sequence_length=10, units=50, dropout=0.2, 
                 epochs=100, batch_size=32, learning_rate=0.001, 
                 patience=10, layers=1, random_state=42):
        self.sequence_length = sequence_length
        self.units = units
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patience = patience
        self.layers = layers
        self.random_state = random_state
        self.model = None
        self.label_encoder = None
        
    def _create_sequences(self, X, y=None):
        """Create sequences for LSTM input"""
        sequences = []
        targets = []
        
        for i in range(self.sequence_length, len(X)):
            sequences.append(X[i-self.sequence_length:i])
            if y is not None:
                targets.append(y.iloc[i])
        
        return np.array(sequences), np.array(targets) if y is not None else None
    
    def fit(self, X, y):
        """Fit LSTM model"""
        from sklearn.preprocessing import LabelEncoder
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        n_classes = len(self.label_encoder.classes_)
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(X, y)
        
        if len(X_seq) == 0:
            raise ValueError("Not enough data to create sequences")
        
        # Build model
        self.model = Sequential()
        
        if self.layers == 1:
            self.model.add(LSTM(self.units, input_shape=(X_seq.shape[1], X_seq.shape[2])))
        else:
            self.model.add(LSTM(self.units, return_sequences=True, 
                              input_shape=(X_seq.shape[1], X_seq.shape[2])))
            for i in range(self.layers - 2):
                self.model.add(LSTM(self.units, return_sequences=True))
                self.model.add(Dropout(self.dropout))
            self.model.add(LSTM(self.units))
        
        self.model.add(Dropout(self.dropout))
        self.model.add(Dense(n_classes, activation='softmax'))
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=self.patience, 
            restore_best_weights=True
        )
        
        # Train model
        self.model.fit(
            X_seq, self.label_encoder.transform(y_seq),
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        
        return self
    
    def predict(self, X):
        """Predict using LSTM model"""
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        X_seq, _ = self._create_sequences(X)
        
        if len(X_seq) == 0:
            # If not enough data for sequences, return majority class
            return np.full(len(X), self.label_encoder.classes_[0])
        
        predictions = self.model.predict(X_seq, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Pad predictions for the initial sequence_length samples
        padded_predictions = np.full(len(X), predicted_classes[0])
        padded_predictions[self.sequence_length:] = predicted_classes
        
        return self.label_encoder.inverse_transform(padded_predictions)
    
    def predict_proba(self, X):
        """Predict probabilities using LSTM model"""
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        X_seq, _ = self._create_sequences(X)
        
        if len(X_seq) == 0:
            # Return uniform probabilities if not enough data
            n_classes = len(self.label_encoder.classes_)
            return np.full((len(X), n_classes), 1.0/n_classes)
        
        probabilities = self.model.predict(X_seq, verbose=0)
        
        # Pad probabilities for initial samples
        padded_probs = np.full((len(X), probabilities.shape[1]), 1.0/probabilities.shape[1])
        padded_probs[self.sequence_length:] = probabilities
        
        return padded_probs
def create_parameter_variations():
    """Create multiple parameter configurations for each model type"""
    
    param_configs = {
        'random_forest': [
            # Conservative RF - Lower complexity, stable
            {
                'name': 'rf_conservative',
                'params': {
                    'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 10,
                    'min_samples_leaf': 5, 'max_features': 'sqrt', 'class_weight': 'balanced',
                    'max_samples': 0.7, 'bootstrap': True, 'oob_score': True
                }
            },
            # Balanced RF - Good trade-off
            {
                'name': 'rf_balanced',
                'params': {
                    'n_estimators': 300, 'max_depth': 15, 'min_samples_split': 5,
                    'min_samples_leaf': 3, 'max_features': 'sqrt', 'class_weight': 'balanced_subsample',
                    'max_samples': 0.8, 'bootstrap': True, 'oob_score': True
                }
            },
            # Aggressive RF - High complexity, potentially higher variance
            {
                'name': 'rf_aggressive',
                'params': {
                    'n_estimators': 500, 'max_depth': 25, 'min_samples_split': 2,
                    'min_samples_leaf': 1, 'max_features': 'log2', 'class_weight': 'balanced_subsample',
                    'max_samples': 0.9, 'bootstrap': True, 'oob_score': True
                }
            },
            # Deep RF - Focus on depth
            {
                'name': 'rf_deep',
                'params': {
                    'n_estimators': 150, 'max_depth': 30, 'min_samples_split': 3,
                    'min_samples_leaf': 2, 'max_features': 'sqrt', 'class_weight': 'balanced',
                    'max_samples': 0.85, 'bootstrap': True, 'oob_score': True
                }
            }
        ],
        
        'extra_trees': [
            # Conservative ET
            {
                'name': 'et_conservative',
                'params': {
                    'n_estimators': 200, 'max_depth': 12, 'min_samples_split': 8,
                    'min_samples_leaf': 4, 'max_features': 'sqrt', 'class_weight': 'balanced',
                    'max_samples': 0.75, 'bootstrap': True, 'oob_score': True
                }
            },
            # Balanced ET
            {
                'name': 'et_balanced',
                'params': {
                    'n_estimators': 300, 'max_depth': 20, 'min_samples_split': 4,
                    'min_samples_leaf': 2, 'max_features': 'log2', 'class_weight': 'balanced_subsample',
                    'max_samples': 0.8, 'bootstrap': True, 'oob_score': True
                }
            },
            # Aggressive ET
            {
                'name': 'et_aggressive',
                'params': {
                    'n_estimators': 400, 'max_depth': 30, 'min_samples_split': 2,
                    'min_samples_leaf': 1, 'max_features': 'log2', 'class_weight': 'balanced_subsample',
                    'max_samples': 0.9, 'bootstrap': True, 'oob_score': True
                }
            },
            # Wide ET - More estimators, shallower
            {
                'name': 'et_wide',
                'params': {
                    'n_estimators': 600, 'max_depth': 8, 'min_samples_split': 6,
                    'min_samples_leaf': 3, 'max_features': 'sqrt', 'class_weight': 'balanced',
                    'max_samples': 0.7, 'bootstrap': True, 'oob_score': True
                }
            }
        ],
        
        'gradient_boosting': [
            # Conservative GB
            {
                'name': 'gb_conservative',
                'params': {
                    'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 4,
                    'min_samples_split': 8, 'min_samples_leaf': 4, 'subsample': 0.7,
                    'max_features': 'sqrt', 'validation_fraction': 0.1, 'n_iter_no_change': 10
                }
            },
            # Balanced GB
            {
                'name': 'gb_balanced',
                'params': {
                    'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 6,
                    'min_samples_split': 4, 'min_samples_leaf': 2, 'subsample': 0.8,
                    'max_features': 'sqrt', 'validation_fraction': 0.1, 'n_iter_no_change': 15
                }
            },
            # Aggressive GB
            {
                'name': 'gb_aggressive',
                'params': {
                    'n_estimators': 300, 'learning_rate': 0.15, 'max_depth': 8,
                    'min_samples_split': 2, 'min_samples_leaf': 1, 'subsample': 0.9,
                    'max_features': 'log2', 'validation_fraction': 0.15, 'n_iter_no_change': 20
                }
            },
            # Fast GB - High learning rate, fewer estimators
            {
                'name': 'gb_fast',
                'params': {
                    'n_estimators': 80, 'learning_rate': 0.2, 'max_depth': 5,
                    'min_samples_split': 6, 'min_samples_leaf': 3, 'subsample': 0.75,
                    'max_features': 'sqrt', 'validation_fraction': 0.1, 'n_iter_no_change': 8
                }
            }
        ],
        
        'logistic_regression': [
            # L1 regularization (Lasso)
            {
                'name': 'lr_l1',
                'params': {
                    'C': 0.1, 'penalty': 'l1', 'class_weight': 'balanced',
                    'solver': 'liblinear', 'max_iter': 2000
                }
            },
            # L2 regularization (Ridge)
            {
                'name': 'lr_l2',
                'params': {
                    'C': 1.0, 'penalty': 'l2', 'class_weight': 'balanced',
                    'solver': 'lbfgs', 'max_iter': 3000
                }
            },
            # Elastic Net
            {
                'name': 'lr_elastic',
                'params': {
                    'C': 0.5, 'penalty': 'elasticnet', 'l1_ratio': 0.5, 'class_weight': 'balanced',
                    'solver': 'saga', 'max_iter': 3000
                }
            },
            # Strong regularization
            {
                'name': 'lr_strong_reg',
                'params': {
                    'C': 0.01, 'penalty': 'l2', 'class_weight': 'balanced',
                    'solver': 'lbfgs', 'max_iter': 5000
                }
            },
            # Weak regularization
            {
                'name': 'lr_weak_reg',
                'params': {
                    'C': 10.0, 'penalty': 'l2', 'class_weight': 'balanced',
                    'solver': 'lbfgs', 'max_iter': 2000
                }
            }
        ],
        
        'mlp': [
            # Small MLP
            {
                'name': 'mlp_small',
                'params': {
                    'hidden_layer_sizes': (50, 25), 'activation': 'relu', 'solver': 'adam',
                    'alpha': 0.001, 'learning_rate': 'adaptive', 'max_iter': 500,
                    'early_stopping': True, 'validation_fraction': 0.1, 'n_iter_no_change': 10
                }
            },
            # Medium MLP
            {
                'name': 'mlp_medium',
                'params': {
                    'hidden_layer_sizes': (100, 50, 25), 'activation': 'relu', 'solver': 'adam',
                    'alpha': 0.0001, 'learning_rate': 'adaptive', 'max_iter': 800,
                    'early_stopping': True, 'validation_fraction': 0.15, 'n_iter_no_change': 15
                }
            },
            # Large MLP
            {
                'name': 'mlp_large',
                'params': {
                    'hidden_layer_sizes': (200, 100, 50, 25), 'activation': 'relu', 'solver': 'adam',
                    'alpha': 0.00001, 'learning_rate': 'adaptive', 'max_iter': 1000,
                    'early_stopping': True, 'validation_fraction': 0.2, 'n_iter_no_change': 20
                }
            },
            # Tanh activation
            {
                'name': 'mlp_tanh',
                'params': {
                    'hidden_layer_sizes': (150, 75), 'activation': 'tanh', 'solver': 'adam',
                    'alpha': 0.0001, 'learning_rate': 'adaptive', 'max_iter': 800,
                    'early_stopping': True, 'validation_fraction': 0.15, 'n_iter_no_change': 15
                }
            },
            # High regularization
            {
                'name': 'mlp_high_reg',
                'params': {
                    'hidden_layer_sizes': (100, 50), 'activation': 'relu', 'solver': 'adam',
                    'alpha': 0.01, 'learning_rate': 'adaptive', 'max_iter': 600,
                    'early_stopping': True, 'validation_fraction': 0.15, 'n_iter_no_change': 12
                }
            }
        ],
        
        'svm': [
            # Linear SVM
            {
                'name': 'svm_linear',
                'params': {
                    'kernel': 'linear', 'C': 1.0, 'class_weight': 'balanced',
                    'probability': True, 'max_iter': 3000
                }
            },
            # RBF SVM - Conservative
            {
                'name': 'svm_rbf_conservative',
                'params': {
                    'kernel': 'rbf', 'C': 0.1, 'gamma': 'scale', 'class_weight': 'balanced',
                    'probability': True, 'max_iter': 3000
                }
            },
            # RBF SVM - Aggressive
            {
                'name': 'svm_rbf_aggressive',
                'params': {
                    'kernel': 'rbf', 'C': 10.0, 'gamma': 'auto', 'class_weight': 'balanced',
                    'probability': True, 'max_iter': 5000
                }
            },
            # Polynomial SVM
            {
                'name': 'svm_poly',
                'params': {
                    'kernel': 'poly', 'degree': 3, 'C': 1.0, 'gamma': 'scale', 
                    'class_weight': 'balanced', 'probability': True, 'max_iter': 3000
                }
            }
        ],
        
        'knn': [
            # Small K
            {
                'name': 'knn_small',
                'params': {
                    'n_neighbors': 3, 'weights': 'distance', 'algorithm': 'auto',
                    'leaf_size': 30, 'p': 2
                }
            },
            # Medium K
            {
                'name': 'knn_medium',
                'params': {
                    'n_neighbors': 7, 'weights': 'distance', 'algorithm': 'auto',
                    'leaf_size': 30, 'p': 2
                }
            },
            # Large K
            {
                'name': 'knn_large',
                'params': {
                    'n_neighbors': 15, 'weights': 'uniform', 'algorithm': 'auto',
                    'leaf_size': 30, 'p': 2
                }
            },
            # Manhattan distance
            {
                'name': 'knn_manhattan',
                'params': {
                    'n_neighbors': 5, 'weights': 'distance', 'algorithm': 'auto',
                    'leaf_size': 30, 'p': 1
                }
            }
        ],
        
        'adaboost': [
            # Conservative AdaBoost
            {
                'name': 'ada_conservative',
                'params': {
                    'n_estimators': 50, 'learning_rate': 0.5, 'algorithm': 'SAMME.R'
                }
            },
            # Balanced AdaBoost
            {
                'name': 'ada_balanced',
                'params': {
                    'n_estimators': 100, 'learning_rate': 1.0, 'algorithm': 'SAMME.R'
                }
            },
            # Aggressive AdaBoost
            {
                'name': 'ada_aggressive',
                'params': {
                    'n_estimators': 200, 'learning_rate': 1.5, 'algorithm': 'SAMME'
                }
            }
        ],
        
        'decision_tree': [
            # Shallow tree
            {
                'name': 'dt_shallow',
                'params': {
                    'max_depth': 5, 'min_samples_split': 10, 'min_samples_leaf': 5,
                    'class_weight': 'balanced', 'criterion': 'gini'
                }
            },
            # Medium tree
            {
                'name': 'dt_medium',
                'params': {
                    'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 3,
                    'class_weight': 'balanced', 'criterion': 'entropy'
                }
            },
            # Deep tree
            {
                'name': 'dt_deep',
                'params': {
                    'max_depth': 20, 'min_samples_split': 2, 'min_samples_leaf': 1,
                    'class_weight': 'balanced', 'criterion': 'gini'
                }
            }
        ]
    }
    
    # XGBoost variations
    if XGBOOST_AVAILABLE:
        param_configs['xgboost'] = [
            # Conservative XGB
            {
                'name': 'xgb_conservative',
                'params': {
                    'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.05,
                    'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_alpha': 0.1,
                    'reg_lambda': 1.0, 'eval_metric': 'mlogloss', 'use_label_encoder': False
                }
            },
            # Balanced XGB
            {
                'name': 'xgb_balanced',
                'params': {
                    'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.05,
                    'reg_lambda': 0.5, 'eval_metric': 'mlogloss', 'use_label_encoder': False
                }
            },
            # Aggressive XGB
            {
                'name': 'xgb_aggressive',
                'params': {
                    'n_estimators': 300, 'max_depth': 10, 'learning_rate': 0.15,
                    'subsample': 0.9, 'colsample_bytree': 0.9, 'reg_alpha': 0.01,
                    'reg_lambda': 0.1, 'eval_metric': 'mlogloss', 'use_label_encoder': False
                }
            },
            # Fast XGB
            {
                'name': 'xgb_fast',
                'params': {
                    'n_estimators': 80, 'max_depth': 5, 'learning_rate': 0.2,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.1,
                    'reg_lambda': 1.0, 'eval_metric': 'mlogloss', 'use_label_encoder': False
                }
            },
            # Deep XGB
            {
                'name': 'xgb_deep',
                'params': {
                    'n_estimators': 150, 'max_depth': 12, 'learning_rate': 0.08,
                    'subsample': 0.75, 'colsample_bytree': 0.75, 'reg_alpha': 0.2,
                    'reg_lambda': 2.0, 'eval_metric': 'mlogloss', 'use_label_encoder': False
                }
            }
        ]
    
    # LightGBM variations
    if LIGHTGBM_AVAILABLE:
        param_configs['lightgbm'] = [
            # Conservative LGB
            {
                'name': 'lgb_conservative',
                'params': {
                    'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.05,
                    'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_alpha': 0.1,
                    'reg_lambda': 1.0, 'objective': 'multiclass', 'metric': 'multi_logloss',
                    'verbose': -1, 'num_leaves': 15
                }
            },
            # Balanced LGB
            {
                'name': 'lgb_balanced',
                'params': {
                    'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.05,
                    'reg_lambda': 0.5, 'objective': 'multiclass', 'metric': 'multi_logloss',
                    'verbose': -1, 'num_leaves': 31
                }
            },
            # Aggressive LGB
            {
                'name': 'lgb_aggressive',
                'params': {
                    'n_estimators': 300, 'max_depth': 10, 'learning_rate': 0.15,
                    'subsample': 0.9, 'colsample_bytree': 0.9, 'reg_alpha': 0.01,
                    'reg_lambda': 0.1, 'objective': 'multiclass', 'metric': 'multi_logloss',
                    'verbose': -1, 'num_leaves': 63
                }
            },
            # DART LGB
            {
                'name': 'lgb_dart',
                'params': {
                    'n_estimators': 200, 'max_depth': 8, 'learning_rate': 0.08,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.1,
                    'reg_lambda': 1.0, 'objective': 'multiclass', 'metric': 'multi_logloss',
                    'verbose': -1, 'boosting_type': 'dart', 'drop_rate': 0.1,
                    'skip_drop': 0.5, 'num_leaves': 31
                }
            },
            # GBDT LGB
            {
                'name': 'lgb_gbdt',
                'params': {
                    'n_estimators': 250, 'max_depth': 7, 'learning_rate': 0.12,
                    'subsample': 0.85, 'colsample_bytree': 0.85, 'reg_alpha': 0.05,
                    'reg_lambda': 0.8, 'objective': 'multiclass', 'metric': 'multi_logloss',
                    'verbose': -1, 'boosting_type': 'gbdt', 'num_leaves': 50
                }
            }
        ]
    
    # CatBoost variations
    if CATBOOST_AVAILABLE:
        param_configs['catboost'] = [
            # Conservative CatBoost
            {
                'name': 'cat_conservative',
                'params': {
                    'iterations': 100, 'depth': 4, 'learning_rate': 0.05,
                    'l2_leaf_reg': 3.0, 'bootstrap_type': 'Bayesian',
                    'bagging_temperature': 1.0, 'od_type': 'Iter', 'od_wait': 20,
                    'random_seed': 42, 'verbose': False
                }
            },
            # Balanced CatBoost
            {
                'name': 'cat_balanced',
                'params': {
                    'iterations': 200, 'depth': 6, 'learning_rate': 0.1,
                    'l2_leaf_reg': 1.0, 'bootstrap_type': 'Poisson',
                    'bagging_temperature': 1.0, 'od_type': 'Iter', 'od_wait': 15,
                    'random_seed': 42, 'verbose': False
                }
            },
            # Aggressive CatBoost
            {
                'name': 'cat_aggressive',
                'params': {
                    'iterations': 300, 'depth': 8, 'learning_rate': 0.15,
                    'l2_leaf_reg': 0.5, 'bootstrap_type': 'MVS',
                    'bagging_temperature': 0.5, 'od_type': 'IncToDec', 'od_wait': 10,
                    'random_seed': 42, 'verbose': False
                }
            }
        ]
    if TENSORFLOW_AVAILABLE:
        param_configs['lstm'] = [
        # Conservative LSTM - More aggressive than before
        {
            'name': 'lstm_conservative',
            'params': {
                'sequence_length': 8, 'units': 64, 'dropout': 0.2,
                'epochs': 80, 'batch_size': 16, 'learning_rate': 0.002,
                'patience': 12, 'layers': 2
            }
        },
        # Balanced LSTM - Significantly more aggressive
        {
            'name': 'lstm_balanced',
            'params': {
                'sequence_length': 20, 'units': 128, 'dropout': 0.25,
                'epochs': 200, 'batch_size': 8, 'learning_rate': 0.003,
                'patience': 15, 'layers': 3
            }
        },
        # Deep LSTM - Very aggressive architecture
        {
            'name': 'lstm_deep',
            'params': {
                'sequence_length': 30, 'units': 256, 'dropout': 0.3,
                'epochs': 300, 'batch_size': 4, 'learning_rate': 0.005,
                'patience': 20, 'layers': 4
            }
        }
    ]
    return param_configs

def create_actionable_models(random_state: int = 42):
    """Create advanced models with multiple parameter configurations"""
    print("🤖 Creating advanced actionable models with parameter variations...")
    
    models = {}
    param_configs = create_parameter_variations()
    
    # Create models for each configuration
    for model_type, configs in param_configs.items():
        print(f"   🔧 Creating {model_type} variations...")
        
        for config in configs:
            name = config['name']
            params = config['params'].copy()
            
            # Add random state where applicable
            if model_type in ['random_forest', 'extra_trees', 'gradient_boosting', 
                             'mlp', 'decision_tree', 'adaboost']:
                params['random_state'] = random_state
            elif model_type in ['logistic_regression']:
                params['random_state'] = random_state
            elif model_type in ['xgboost', 'lightgbm']:
                if 'random_seed' not in params:
                    params['random_seed'] = random_state
                if model_type == 'lightgbm' and 'random_state' not in params:
                    params['random_state'] = random_state
            
            # Add n_jobs where applicable
            if model_type in ['random_forest', 'extra_trees', 'xgboost', 'lightgbm']:
                if 'n_jobs' not in params:
                    params['n_jobs'] = -1
            
            try:
                if model_type == 'random_forest':
                    models[name] = RandomForestClassifier(**params)
                elif model_type == 'extra_trees':
                    models[name] = ExtraTreesClassifier(**params)
                elif model_type == 'gradient_boosting':
                    models[name] = GradientBoostingClassifier(**params)
                elif model_type == 'logistic_regression':
                    models[name] = LogisticRegression(**params)
                elif model_type == 'mlp':
                    models[name] = MLPClassifier(**params)
                elif model_type == 'svm':
                    models[name] = SVC(**params)
                elif model_type == 'knn':
                    models[name] = KNeighborsClassifier(**params)
                elif model_type == 'adaboost':
                    models[name] = AdaBoostClassifier(**params)
                elif model_type == 'decision_tree':
                    models[name] = DecisionTreeClassifier(**params)
                elif model_type == 'xgboost' and XGBOOST_AVAILABLE:
                    models[name] = xgb.XGBClassifier(**params)
                elif model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
                    models[name] = lgb.LGBMClassifier(**params)
                elif model_type == 'catboost' and CATBOOST_AVAILABLE:
                    models[name] = CatBoostClassifier(**params)
                elif model_type == 'lstm' and TENSORFLOW_AVAILABLE:
                    models[name] = LSTMClassifier(**params)
                    
            except Exception as e:
                print(f"   ⚠️ Warning: Failed to create {name}: {str(e)}")
                continue
    
    # Add some additional classic models
    print("   📚 Adding classic models...")
    
    # Ridge Classifier variations
    models['ridge_weak'] = RidgeClassifier(alpha=0.1, class_weight='balanced', random_state=random_state)
    models['ridge_medium'] = RidgeClassifier(alpha=1.0, class_weight='balanced', random_state=random_state)
    models['ridge_strong'] = RidgeClassifier(alpha=10.0, class_weight='balanced', random_state=random_state)
    
    # Gaussian Naive Bayes
    models['nb_gaussian'] = GaussianNB()
    
    print(f"   🎭 Creating ensemble combinations...")
    
    # Create diverse ensemble combinations
    available_models = list(models.keys())
    
    # Ensemble 1: Tree-based models
    tree_models = [name for name in available_models if any(x in name for x in ['rf_', 'et_', 'gb_', 'xgb_', 'lgb_', 'cat_'])]
    if len(tree_models) >= 3:
        tree_estimators = [(name, models[name]) for name in tree_models[:5]]  # Take first 5
        models['ensemble_trees_diverse'] = VotingClassifier(estimators=tree_estimators, voting='soft')
    
    # Ensemble 2: Linear models
    linear_models = [name for name in available_models if any(x in name for x in ['lr_', 'ridge_', 'svm_linear'])]
    if len(linear_models) >= 3:
        linear_estimators = [(name, models[name]) for name in linear_models[:4]]
        models['ensemble_linear'] = VotingClassifier(estimators=linear_estimators, voting='soft')
    
    # Ensemble 3: Balanced mix (conservative models)
    conservative_models = [name for name in available_models if 'conservative' in name]
    if len(conservative_models) >= 3:
        conservative_estimators = [(name, models[name]) for name in conservative_models[:5]]
        models['ensemble_conservative'] = VotingClassifier(estimators=conservative_estimators, voting='soft')
    
    # Ensemble 4: Aggressive mix
    aggressive_models = [name for name in available_models if 'aggressive' in name]
    if len(aggressive_models) >= 3:
        aggressive_estimators = [(name, models[name]) for name in aggressive_models[:5]]
        models['ensemble_aggressive'] = VotingClassifier(estimators=aggressive_estimators, voting='soft')
    
    # Ensemble 5: Balanced mix
    balanced_models = [name for name in available_models if 'balanced' in name]
    if len(balanced_models) >= 3:
        balanced_estimators = [(name, models[name]) for name in balanced_models[:5]]
        models['ensemble_balanced'] = VotingClassifier(estimators=balanced_estimators, voting='soft')
    
    # Ensemble 6: Best of each algorithm type
    best_models = []
    model_types_seen = set()
    
    for name in available_models:
        model_prefix = name.split('_')[0]
        if model_prefix not in model_types_seen and 'ensemble' not in name:
            best_models.append(name)
            model_types_seen.add(model_prefix)
        if len(best_models) >= 7:  # Limit ensemble size
            break
    
    if len(best_models) >= 3:
        best_estimators = [(name, models[name]) for name in best_models]
        models['ensemble_best_of_each'] = VotingClassifier(estimators=best_estimators, voting='soft')
    
    # Ensemble 7: Top performers (you would select these based on validation results)
    # For now, we'll create a mixed ensemble with different algorithm types
    mixed_models = []
    priorities = ['xgb_balanced', 'rf_balanced', 'lgb_balanced', 'gb_balanced', 
                  'lr_elastic', 'mlp_medium', 'et_balanced']
    
    for priority in priorities:
        if priority in available_models:
            mixed_models.append(priority)
    
    # Fill remaining slots with other good models
    for name in available_models:
        if name not in mixed_models and 'ensemble' not in name and len(mixed_models) < 6:
            mixed_models.append(name)
    
    if len(mixed_models) >= 3:
        mixed_estimators = [(name, models[name]) for name in mixed_models]
        models['ensemble_mixed_algorithms'] = VotingClassifier(estimators=mixed_estimators, voting='soft')
    
    print(f"✅ Created {len(models)} models total")
    
    # Print detailed model summary
    model_types = {}
    ensemble_count = 0
    
    for name, model in models.items():
        if 'ensemble' in name:
            ensemble_count += 1
        else:
            model_type = type(model).__name__
            if model_type not in model_types:
                model_types[model_type] = 0
            model_types[model_type] += 1
    
    print("📊 Model distribution:")
    for model_type, count in sorted(model_types.items()):
        print(f"   {model_type}: {count}")
    print(f"   VotingClassifier (Ensembles): {ensemble_count}")
    
    # Print parameter variation summary
    print("\n🔧 Parameter variations created:")
    for model_type, configs in param_configs.items():
        print(f"   {model_type}: {len(configs)} variations")
    
    return models

def get_model_recommendations(models: Dict[str, Any]) -> Dict[str, List[str]]:
    """Get model recommendations based on different scenarios"""
    
    recommendations = {
        'fast_training': [
            'lr_l2', 'lr_elastic', 'nb_gaussian', 'ridge_medium', 
            'dt_shallow', 'knn_small', 'ada_conservative'
        ],
        'high_accuracy': [
            'xgb_balanced', 'lgb_balanced', 'rf_aggressive', 'gb_aggressive',
            'ensemble_balanced', 'ensemble_mixed_algorithms'
        ],
        'interpretable': [
            'lr_l1', 'lr_elastic', 'dt_medium', 'ridge_medium', 
            'nb_gaussian', 'ada_conservative'
        ],
        'robust_to_overfitting': [
            'rf_conservative', 'gb_conservative', 'lr_l1', 'ridge_strong',
            'ensemble_conservative', 'xgb_conservative'
        ],
        'handling_imbalanced_data': [
            'rf_balanced', 'xgb_balanced', 'lgb_balanced', 'lr_elastic',
            'ensemble_balanced', 'gb_balanced'
        ],
        'feature_selection': [
            'lr_l1', 'rf_aggressive', 'et_aggressive', 'xgb_aggressive',
            'ridge_weak'
        ],
        'ensemble_diversity': [
            'ensemble_trees_diverse', 'ensemble_mixed_algorithms', 
            'ensemble_best_of_each', 'ensemble_balanced'
        ]
    }
    
    # Filter recommendations to only include models that exist
    filtered_recommendations = {}
    available_models = set(models.keys())
    
    for scenario, model_list in recommendations.items():
        filtered_recommendations[scenario] = [
            model for model in model_list if model in available_models
        ]
    
    return filtered_recommendations

def time_series_cross_validation(X, y, model, n_splits=5):
    """Perform time series cross-validation"""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = []
    
    # Import evaluation function (assuming it exists)
    try:
        from evaluate import evaluate_model_actionable
    except ImportError:
        print("Warning: evaluate_model_actionable not found, using basic metrics")
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        def evaluate_model_actionable(y_true, y_pred):
            return {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
            }
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        # Fit model on fold
        try:
            model_fold = type(model)(**model.get_params())
            model_fold.fit(X_train_fold, y_train_fold)
            
            # Predict and evaluate
            y_pred_fold = model_fold.predict(X_val_fold)
            fold_metrics = evaluate_model_actionable(y_val_fold, y_pred_fold)
            cv_scores.append(fold_metrics)
        except Exception as e:
            print(f"Warning: Fold {fold} failed: {str(e)}")
            continue
    
    if not cv_scores:
        return {}
    
    # Calculate mean and std of CV scores
    cv_results = {}
    for metric in cv_scores[0].keys():
        scores = [fold[metric] for fold in cv_scores if metric in fold]
        if scores:
            cv_results[f'{metric}_mean'] = np.mean(scores)
            cv_results[f'{metric}_std'] = np.std(scores)
    
    return cv_results

def print_model_analysis():
    """Print analysis of created models"""
    print("\n" + "="*80)
    print("MODEL PARAMETER ANALYSIS")
    print("="*80)
    
    models = create_actionable_models()
    recommendations = get_model_recommendations(models)
    
    print(f"\n📊 TOTAL MODELS CREATED: {len(models)}")
    print("-" * 40)
    
    # Group models by type
    model_groups = {}
    for name, model in models.items():
        if 'ensemble' in name:
            model_type = 'Ensemble'
        else:
            model_type = type(model).__name__
        
        if model_type not in model_groups:
            model_groups[model_type] = []
        model_groups[model_type].append(name)
    
    for model_type, model_names in sorted(model_groups.items()):
        print(f"\n{model_type} ({len(model_names)} models):")
        for name in sorted(model_names):
            print(f"  • {name}")
    
    print("\n" + "="*60)
    print("SCENARIO-BASED RECOMMENDATIONS")
    print("="*60)
    
    for scenario, model_list in recommendations.items():
        print(f"\n🎯 {scenario.replace('_', ' ').title()}:")
        for model in model_list[:3]:  # Show top 3
            print(f"  • {model}")
    
    print("\n" + "="*60)
    print("PARAMETER VARIATION HIGHLIGHTS")
    print("="*60)
    
    param_configs = create_parameter_variations()
    
    for model_type, configs in param_configs.items():
        print(f"\n🔧 {model_type.upper()}:")
        print(f"  Variations: {len(configs)}")
        
        # Show parameter ranges
        if configs:
            sample_config = configs[0]['params']
            print("  Key parameters:")
            for param, value in list(sample_config.items())[:3]:
                print(f"    • {param}: varies across configs")

# Example usage function
def run_parameter_experiment():
    """Example function showing how to use the enhanced models"""
    print("🚀 Starting Parameter Experiment...")
    
    # Create models with different parameters
    models = create_actionable_models(random_state=42)
    
    # Get recommendations
    recommendations = get_model_recommendations(models)
    
    print("\n✅ Experiment Setup Complete!")
    print(f"Total models available: {len(models)}")
    print(f"Recommendation categories: {len(recommendations)}")
    
    return models, recommendations

if __name__ == "__main__":
    print_model_analysis()