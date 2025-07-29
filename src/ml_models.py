"""
Machine Learning Models for Cryptocurrency Analysis
Author: [pacific]
Course: INSY 8413 | Introduction to Big Data Analytics

This module implements various ML models for price prediction, direction classification, and volatility forecasting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

from utils import setup_logging, save_data

class CryptoPricePredictor:
    """
    Cryptocurrency price prediction using multiple ML models
    """
    
    def __init__(self):
        """Initialize the price predictor"""
        self.logger = setup_logging()
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
    def prepare_features(self, df: pd.DataFrame, target_col: str = 'target_price') -> tuple:
        """
        Prepare features for machine learning
        
        Args:
            df (pd.DataFrame): Feature data
            target_col (str): Target column name
            
        Returns:
            tuple: (X, y, feature_names)
        """
        # Remove rows with missing target values
        df_clean = df.dropna(subset=[target_col])
        
        # Define feature columns (exclude target variables and non-numeric columns)
        exclude_cols = [
            'target_price', 'target_direction', 'target_direction_3class', 
            'target_volatility', 'target_high_volatility', 'symbol',
            'doji', 'hammer', 'shooting_star', 'gap_up', 'gap_down',
            'local_high', 'local_low', 'volatility_breakout'
        ]
        
        feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
        
        # Remove columns with too many missing values
        missing_threshold = 0.5
        for col in feature_cols.copy():
            if df_clean[col].isnull().sum() / len(df_clean) > missing_threshold:
                feature_cols.remove(col)
                self.logger.warning(f"Removed feature {col} due to high missing values")
        
        # Prepare features and target
        X = df_clean[feature_cols].fillna(method='ffill').fillna(0)
        y = df_clean[target_col]
        
        # Remove infinite values
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        self.logger.info(f"Prepared {len(feature_cols)} features for {len(X)} samples")
        
        return X, y, feature_cols
    
    def train_linear_regression(self, X_train, y_train, X_test, y_test, model_name: str):
        """Train Linear Regression model"""
        self.logger.info(f"Training Linear Regression for {model_name}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # Store model and scaler
        self.models[f"{model_name}_lr"] = model
        self.scalers[f"{model_name}_lr"] = scaler
        
        # Calculate metrics
        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test)
        }
        
        return metrics, y_pred_test
    
    def train_random_forest(self, X_train, y_train, X_test, y_test, model_name: str):
        """Train Random Forest model"""
        self.logger.info(f"Training Random Forest for {model_name}")
        
        # Train model with hyperparameter tuning
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        rf = RandomForestRegressor(random_state=42)
        
        # Use TimeSeriesSplit for time series data
        tscv = TimeSeriesSplit(n_splits=3)
        grid_search = GridSearchCV(rf, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        model = grid_search.best_estimator_
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Store model
        self.models[f"{model_name}_rf"] = model
        
        # Store feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.feature_importance[f"{model_name}_rf"] = feature_importance
        
        # Calculate metrics
        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'best_params': grid_search.best_params_
        }
        
        return metrics, y_pred_test
    
    def train_neural_network(self, X_train, y_train, X_test, y_test, model_name: str):
        """Train Neural Network model"""
        try:
            from sklearn.neural_network import MLPRegressor
            
            self.logger.info(f"Training Neural Network for {model_name}")
            
            # Scale features
            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = MLPRegressor(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.001,
                batch_size='auto',
                learning_rate='constant',
                learning_rate_init=0.001,
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)
            
            # Store model and scaler
            self.models[f"{model_name}_nn"] = model
            self.scalers[f"{model_name}_nn"] = scaler
            
            # Calculate metrics
            metrics = {
                'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'train_mae': mean_absolute_error(y_train, y_pred_train),
                'test_mae': mean_absolute_error(y_test, y_pred_test),
                'train_r2': r2_score(y_train, y_pred_train),
                'test_r2': r2_score(y_test, y_pred_test),
                'n_iterations': model.n_iter_
            }
            
            return metrics, y_pred_test
            
        except ImportError:
            self.logger.error("Neural network training requires scikit-learn with MLPRegressor")
            return None, None
    
    def create_ensemble_model(self, predictions_dict: dict, y_test, model_name: str):
        """Create ensemble model from multiple predictions"""
        self.logger.info(f"Creating ensemble model for {model_name}")
        
        # Simple average ensemble
        predictions_array = np.array(list(predictions_dict.values()))
        ensemble_pred = np.mean(predictions_array, axis=0)
        
        # Calculate ensemble metrics
        metrics = {
            'test_rmse': np.sqrt(mean_squared_error(y_test, ensemble_pred)),
            'test_mae': mean_absolute_error(y_test, ensemble_pred),
            'test_r2': r2_score(y_test, ensemble_pred)
        }
        
        return metrics, ensemble_pred
    
    def train_all_models(self, df: pd.DataFrame, symbol: str, target_col: str = 'target_price'):
        """
        Train all models for a cryptocurrency
        
        Args:
            df (pd.DataFrame): Feature data
            symbol (str): Cryptocurrency symbol
            target_col (str): Target column name
            
        Returns:
            dict: Model results
        """
        self.logger.info(f"Training all models for {symbol}")
        
        # Prepare features
        X, y, feature_names = self.prepare_features(df, target_col)
        
        if len(X) < 100:
            self.logger.warning(f"Insufficient data for {symbol}: {len(X)} samples")
            return None
        
        # Split data (time series split)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        self.logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        
        results = {}
        predictions = {}
        
        # Train Linear Regression
        lr_metrics, lr_pred = self.train_linear_regression(X_train, y_train, X_test, y_test, symbol)
        results['linear_regression'] = lr_metrics
        predictions['linear_regression'] = lr_pred
        
        # Train Random Forest
        rf_metrics, rf_pred = self.train_random_forest(X_train, y_train, X_test, y_test, symbol)
        results['random_forest'] = rf_metrics
        predictions['random_forest'] = rf_pred
        
        # Train Neural Network
        nn_metrics, nn_pred = self.train_neural_network(X_train, y_train, X_test, y_test, symbol)
        if nn_metrics is not None:
            results['neural_network'] = nn_metrics
            predictions['neural_network'] = nn_pred
        
        # Create Ensemble
        if len(predictions) > 1:
            ensemble_metrics, ensemble_pred = self.create_ensemble_model(predictions, y_test, symbol)
            results['ensemble'] = ensemble_metrics
            predictions['ensemble'] = ensemble_pred
        
        # Store test data for evaluation
        results['test_data'] = {
            'y_true': y_test.values,
            'predictions': predictions,
            'feature_names': feature_names
        }
        
        return results

class CryptoDirectionClassifier:
    """
    Cryptocurrency direction classification
    """
    
    def __init__(self):
        """Initialize the direction classifier"""
        self.logger = setup_logging()
        self.models = {}
        self.scalers = {}
    
    def train_classification_models(self, df: pd.DataFrame, symbol: str, target_col: str = 'target_direction'):
        """
        Train classification models for direction prediction
        
        Args:
            df (pd.DataFrame): Feature data
            symbol (str): Cryptocurrency symbol
            target_col (str): Target column name
            
        Returns:
            dict: Classification results
        """
        self.logger.info(f"Training classification models for {symbol}")
        
        # Prepare features (reuse the same preparation logic)
        predictor = CryptoPricePredictor()
        X, y, feature_names = predictor.prepare_features(df, target_col)
        
        if len(X) < 100:
            self.logger.warning(f"Insufficient data for {symbol}: {len(X)} samples")
            return None
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        results = {}
        
        # Logistic Regression
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(X_train_scaled, y_train)
        lr_pred = lr_model.predict(X_test_scaled)
        
        # Random Forest Classifier
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        
        # Calculate metrics
        results['logistic_regression'] = {
            'accuracy': accuracy_score(y_test, lr_pred),
            'precision': precision_score(y_test, lr_pred, average='weighted'),
            'recall': recall_score(y_test, lr_pred, average='weighted'),
            'f1': f1_score(y_test, lr_pred, average='weighted')
        }
        
        results['random_forest'] = {
            'accuracy': accuracy_score(y_test, rf_pred),
            'precision': precision_score(y_test, rf_pred, average='weighted'),
            'recall': recall_score(y_test, rf_pred, average='weighted'),
            'f1': f1_score(y_test, rf_pred, average='weighted')
        }
        
        # Store models
        self.models[f"{symbol}_lr"] = lr_model
        self.models[f"{symbol}_rf"] = rf_model
        self.scalers[f"{symbol}_lr"] = scaler
        
        results['test_data'] = {
            'y_true': y_test.values,
            'lr_pred': lr_pred,
            'rf_pred': rf_pred
        }
        
        return results

def main():
    """Main function to run ML model training"""
    from utils import load_data
    
    print("🤖 Starting Machine Learning Model Training")
    print("=" * 50)
    
    # Initialize models
    price_predictor = CryptoPricePredictor()
    direction_classifier = CryptoDirectionClassifier()
    
    # Define symbols and intervals
    symbols = ['BTC', 'ETH', 'BNB', 'ADA', 'SOL']
    intervals = ['1h']  # Focus on hourly data for ML
    
    all_results = {}
    
    for symbol in symbols:
        all_results[symbol] = {}
        
        for interval in intervals:
            print(f"\n🔧 Training models for {symbol} {interval}")
            
            # Load feature data
            filename = f"{symbol}_{interval}_features.csv"
            df = load_data(filename, 'data/processed')
            
            if df is not None and not df.empty:
                # Convert datetime index
                if 'datetime' in df.columns:
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    df.set_index('datetime', inplace=True)
                
                # Train price prediction models
                price_results = price_predictor.train_all_models(df, symbol)
                if price_results:
                    all_results[symbol]['price_prediction'] = price_results
                    print(f"✅ Price prediction models trained")
                
                # Train direction classification models
                direction_results = direction_classifier.train_classification_models(df, symbol)
                if direction_results:
                    all_results[symbol]['direction_classification'] = direction_results
                    print(f"✅ Direction classification models trained")
            else:
                print(f"❌ No feature data found for {symbol} {interval}")
    
    # Save results
    import json
    with open('data/processed/ml_results.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj
        
        json.dump(all_results, f, indent=2, default=convert_numpy)
    
    print(f"\n🎯 Machine learning training completed!")
    print(f"📁 Results saved to: data/processed/ml_results.json")

if __name__ == "__main__":
    main()
