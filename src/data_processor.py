"""
Data Processing and Cleaning Module for Cryptocurrency Analysis
Author: [Your Name]
Course: INSY 8413 | Introduction to Big Data Analytics

This module handles data cleaning, preprocessing, and preparation for ML models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from utils import (
    validate_dataframe, 
    save_data, 
    load_data, 
    print_data_summary,
    setup_logging
)

class CryptoDataProcessor:
    """
    Processes and cleans cryptocurrency data for analysis
    """
    
    def __init__(self):
        """Initialize the data processor"""
        self.logger = setup_logging()
        self.required_columns = ['open', 'high', 'low', 'close', 'volume', 'symbol']
        
    def load_raw_data(self, symbol: str, interval: str) -> pd.DataFrame:
        """
        Load raw data for a specific cryptocurrency and interval
        
        Args:
            symbol (str): Cryptocurrency symbol
            interval (str): Time interval
            
        Returns:
            pd.DataFrame: Raw data
        """
        filename = f"{symbol}_{interval}_data.csv"
        df = load_data(filename, 'data/raw')
        
        if df is not None:
            # Convert datetime column to index if it exists
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df.set_index('datetime', inplace=True)
            elif df.index.name != 'datetime':
                # Try to convert index to datetime
                try:
                    df.index = pd.to_datetime(df.index)
                except:
                    self.logger.warning(f"Could not convert index to datetime for {symbol} {interval}")
            
            self.logger.info(f"Loaded {len(df)} records for {symbol} {interval}")
        else:
            self.logger.error(f"Could not load data for {symbol} {interval}")
            
        return df
    
    def clean_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Clean and preprocess cryptocurrency data
        
        Args:
            df (pd.DataFrame): Raw data
            symbol (str): Cryptocurrency symbol
            
        Returns:
            pd.DataFrame: Cleaned data
        """
        if df is None or df.empty:
            self.logger.error(f"No data to clean for {symbol}")
            return pd.DataFrame()
        
        self.logger.info(f"Cleaning data for {symbol}")
        
        # Create a copy to avoid modifying original data
        cleaned_df = df.copy()
        
        # 1. Remove duplicates
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        duplicates_removed = initial_rows - len(cleaned_df)
        if duplicates_removed > 0:
            self.logger.info(f"Removed {duplicates_removed} duplicate rows")
        
        # 2. Handle missing values
        missing_before = cleaned_df.isnull().sum().sum()
        if missing_before > 0:
            self.logger.info(f"Found {missing_before} missing values")
            
            # Forward fill for price data (carry last observation forward)
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                if col in cleaned_df.columns:
                    cleaned_df[col] = cleaned_df[col].fillna(method='ffill')
            
            # Fill volume with 0 if missing
            if 'volume' in cleaned_df.columns:
                cleaned_df['volume'] = cleaned_df['volume'].fillna(0)
            
            # Drop rows that still have missing values in critical columns
            critical_columns = ['open', 'high', 'low', 'close']
            cleaned_df = cleaned_df.dropna(subset=critical_columns)
            
            missing_after = cleaned_df.isnull().sum().sum()
            self.logger.info(f"Missing values after cleaning: {missing_after}")
        
        # 3. Data validation and outlier detection
        cleaned_df = self._validate_price_data(cleaned_df, symbol)
        cleaned_df = self._detect_outliers(cleaned_df, symbol)
        
        # 4. Ensure data types
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in cleaned_df.columns:
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
        
        # 5. Sort by datetime index
        cleaned_df = cleaned_df.sort_index()
        
        # 6. Add symbol column if not present
        if 'symbol' not in cleaned_df.columns:
            cleaned_df['symbol'] = symbol
        
        self.logger.info(f"Cleaning completed for {symbol}. Final shape: {cleaned_df.shape}")
        return cleaned_df
    
    def _validate_price_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Validate price data for logical consistency
        
        Args:
            df (pd.DataFrame): Data to validate
            symbol (str): Cryptocurrency symbol
            
        Returns:
            pd.DataFrame: Validated data
        """
        initial_rows = len(df)
        
        # Remove rows where high < low (impossible)
        invalid_high_low = df['high'] < df['low']
        df = df[~invalid_high_low]
        
        # Remove rows where close/open is outside high/low range
        invalid_close = (df['close'] > df['high']) | (df['close'] < df['low'])
        df = df[~invalid_close]
        
        invalid_open = (df['open'] > df['high']) | (df['open'] < df['low'])
        df = df[~invalid_open]
        
        # Remove rows with negative prices
        negative_prices = (df[['open', 'high', 'low', 'close']] <= 0).any(axis=1)
        df = df[~negative_prices]
        
        # Remove rows with negative volume
        if 'volume' in df.columns:
            negative_volume = df['volume'] < 0
            df = df[~negative_volume]
        
        removed_rows = initial_rows - len(df)
        if removed_rows > 0:
            self.logger.info(f"Removed {removed_rows} invalid rows for {symbol}")
        
        return df
    
    def _detect_outliers(self, df: pd.DataFrame, symbol: str, method: str = 'iqr') -> pd.DataFrame:
        """
        Detect and handle outliers in price data
        
        Args:
            df (pd.DataFrame): Data to process
            symbol (str): Cryptocurrency symbol
            method (str): Outlier detection method ('iqr' or 'zscore')
            
        Returns:
            pd.DataFrame: Data with outliers handled
        """
        initial_rows = len(df)
        
        if method == 'iqr':
            # Use IQR method for outlier detection
            for col in ['open', 'high', 'low', 'close']:
                if col in df.columns:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    # Define outlier bounds (more lenient for crypto due to volatility)
                    lower_bound = Q1 - 3 * IQR
                    upper_bound = Q3 + 3 * IQR
                    
                    # Cap outliers instead of removing them
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        elif method == 'zscore':
            # Use Z-score method
            from scipy import stats
            for col in ['open', 'high', 'low', 'close']:
                if col in df.columns:
                    z_scores = np.abs(stats.zscore(df[col]))
                    # Cap extreme outliers (z-score > 4)
                    outlier_mask = z_scores > 4
                    if outlier_mask.any():
                        median_val = df[col].median()
                        df.loc[outlier_mask, col] = median_val
        
        processed_rows = len(df)
        if initial_rows != processed_rows:
            self.logger.info(f"Processed outliers for {symbol} using {method} method")
        
        return df
    
    def add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add basic derived features to the dataset
        
        Args:
            df (pd.DataFrame): Cleaned data
            
        Returns:
            pd.DataFrame: Data with additional features
        """
        if df.empty:
            return df
        
        # Price-based features
        df['price_range'] = df['high'] - df['low']
        df['price_change'] = df['close'] - df['open']
        df['price_change_pct'] = (df['close'] - df['open']) / df['open'] * 100
        df['high_low_pct'] = (df['high'] - df['low']) / df['low'] * 100
        
        # Volume-based features
        if 'volume' in df.columns:
            df['volume_usd'] = df['volume'] * df['close']
            df['volume_ma_7'] = df['volume'].rolling(window=7).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma_7']
        
        # Price moving averages
        for window in [7, 14, 30]:
            df[f'close_ma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'close_std_{window}'] = df['close'].rolling(window=window).std()
        
        # Returns
        df['returns_1d'] = df['close'].pct_change(1)
        df['returns_7d'] = df['close'].pct_change(7)
        df['returns_30d'] = df['close'].pct_change(30)
        
        # Volatility (rolling standard deviation of returns)
        df['volatility_7d'] = df['returns_1d'].rolling(window=7).std()
        df['volatility_30d'] = df['returns_1d'].rolling(window=30).std()
        
        # Price position within range
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Time-based features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        
        return df
    
    def process_all_data(self, symbols: List[str], intervals: List[str]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Process all cryptocurrency data
        
        Args:
            symbols (list): List of cryptocurrency symbols
            intervals (list): List of time intervals
            
        Returns:
            dict: Processed data for all symbols and intervals
        """
        processed_data = {}
        
        for symbol in symbols:
            processed_data[symbol] = {}
            
            for interval in intervals:
                self.logger.info(f"Processing {symbol} {interval} data")
                
                # Load raw data
                raw_df = self.load_raw_data(symbol, interval)
                
                if raw_df is not None and not raw_df.empty:
                    # Clean data
                    cleaned_df = self.clean_data(raw_df, symbol)
                    
                    if not cleaned_df.empty:
                        # Add basic features
                        processed_df = self.add_basic_features(cleaned_df)
                        
                        # Save processed data
                        filename = f"{symbol}_{interval}_processed.csv"
                        save_data(processed_df, filename, 'data/processed')
                        
                        processed_data[symbol][interval] = processed_df
                        
                        self.logger.info(f"Successfully processed {symbol} {interval}: {len(processed_df)} records")
                    else:
                        self.logger.warning(f"No data after cleaning for {symbol} {interval}")
                else:
                    self.logger.warning(f"No raw data found for {symbol} {interval}")
        
        return processed_data
    
    def create_analysis_summary(self, processed_data: Dict) -> pd.DataFrame:
        """
        Create a summary of processed data
        
        Args:
            processed_data (dict): Processed data dictionary
            
        Returns:
            pd.DataFrame: Summary statistics
        """
        summary_data = []
        
        for symbol in processed_data:
            for interval in processed_data[symbol]:
                df = processed_data[symbol][interval]
                
                if not df.empty:
                    summary = {
                        'symbol': symbol,
                        'interval': interval,
                        'records': len(df),
                        'date_start': df.index.min(),
                        'date_end': df.index.max(),
                        'avg_price': df['close'].mean(),
                        'min_price': df['close'].min(),
                        'max_price': df['close'].max(),
                        'price_volatility': df['close'].std(),
                        'avg_volume': df['volume'].mean() if 'volume' in df.columns else 0,
                        'total_return': ((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100,
                        'missing_values': df.isnull().sum().sum()
                    }
                    summary_data.append(summary)
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary
        save_data(summary_df, 'processing_summary.csv', 'data/processed')
        
        return summary_df

def main():
    """Main function to run data processing"""
    processor = CryptoDataProcessor()
    
    print("🧹 Starting Data Processing Pipeline")
    print("=" * 50)
    
    # Define symbols and intervals
    symbols = ['BTC', 'ETH', 'BNB', 'ADA', 'SOL']
    intervals = ['5m', '1h']
    
    # Process all data
    processed_data = processor.process_all_data(symbols, intervals)
    
    # Create summary
    summary_df = processor.create_analysis_summary(processed_data)
    
    print("\n📊 PROCESSING SUMMARY")
    print("=" * 30)
    print(summary_df.to_string(index=False))
    
    print(f"\n✅ Data processing completed!")
    print(f"📁 Processed data saved to: data/processed/")

if __name__ == "__main__":
    main()
