"""
Feature Engineering Module for Cryptocurrency Analysis
Author: [Your Name]
Course: INSY 8413 | Introduction to Big Data Analytics

This module creates advanced technical indicators and features for ML models
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from utils import setup_logging, save_data

class TechnicalIndicators:
    """
    Creates technical indicators for cryptocurrency analysis
    """
    
    def __init__(self):
        """Initialize the technical indicators calculator"""
        self.logger = setup_logging()
    
    def calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI)
        
        Args:
            prices (pd.Series): Price series
            window (int): RSI period
            
        Returns:
            pd.Series: RSI values
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Args:
            prices (pd.Series): Price series
            fast (int): Fast EMA period
            slow (int): Slow EMA period
            signal (int): Signal line EMA period
            
        Returns:
            dict: MACD line, signal line, and histogram
        """
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'macd_signal': signal_line,
            'macd_histogram': histogram
        }
    
    def calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands
        
        Args:
            prices (pd.Series): Price series
            window (int): Moving average period
            num_std (float): Number of standard deviations
            
        Returns:
            dict: Upper band, lower band, and middle band
        """
        middle_band = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        
        upper_band = middle_band + (std * num_std)
        lower_band = middle_band - (std * num_std)
        
        return {
            'bb_upper': upper_band,
            'bb_middle': middle_band,
            'bb_lower': lower_band,
            'bb_width': upper_band - lower_band,
            'bb_position': (prices - lower_band) / (upper_band - lower_band)
        }
    
    def calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, k_window: int = 14, d_window: int = 3) -> Dict[str, pd.Series]:
        """
        Calculate Stochastic Oscillator
        
        Args:
            high (pd.Series): High prices
            low (pd.Series): Low prices
            close (pd.Series): Close prices
            k_window (int): %K period
            d_window (int): %D period
            
        Returns:
            dict: %K and %D values
        """
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        
        return {
            'stoch_k': k_percent,
            'stoch_d': d_percent
        }
    
    def calculate_williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculate Williams %R
        
        Args:
            high (pd.Series): High prices
            low (pd.Series): Low prices
            close (pd.Series): Close prices
            window (int): Period
            
        Returns:
            pd.Series: Williams %R values
        """
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        
        return williams_r
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR)
        
        Args:
            high (pd.Series): High prices
            low (pd.Series): Low prices
            close (pd.Series): Close prices
            window (int): Period
            
        Returns:
            pd.Series: ATR values
        """
        high_low = high - low
        high_close_prev = np.abs(high - close.shift(1))
        low_close_prev = np.abs(low - close.shift(1))
        
        true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
        atr = true_range.rolling(window=window).mean()
        
        return atr
    
    def calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Calculate On-Balance Volume (OBV)
        
        Args:
            close (pd.Series): Close prices
            volume (pd.Series): Volume
            
        Returns:
            pd.Series: OBV values
        """
        price_change = close.diff()
        obv = volume.copy()
        
        obv[price_change < 0] = -volume[price_change < 0]
        obv[price_change == 0] = 0
        
        return obv.cumsum()

class CryptoFeatureEngineer:
    """
    Advanced feature engineering for cryptocurrency data
    """
    
    def __init__(self):
        """Initialize the feature engineer"""
        self.logger = setup_logging()
        self.indicators = TechnicalIndicators()
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all technical indicators to the dataset
        
        Args:
            df (pd.DataFrame): OHLCV data
            
        Returns:
            pd.DataFrame: Data with technical indicators
        """
        if df.empty:
            return df
        
        self.logger.info("Adding technical indicators")
        
        # Create a copy to avoid modifying original data
        enhanced_df = df.copy()
        
        # RSI
        enhanced_df['rsi_14'] = self.indicators.calculate_rsi(enhanced_df['close'], 14)
        enhanced_df['rsi_7'] = self.indicators.calculate_rsi(enhanced_df['close'], 7)
        
        # MACD
        macd_data = self.indicators.calculate_macd(enhanced_df['close'])
        for key, values in macd_data.items():
            enhanced_df[key] = values
        
        # Bollinger Bands
        bb_data = self.indicators.calculate_bollinger_bands(enhanced_df['close'])
        for key, values in bb_data.items():
            enhanced_df[key] = values
        
        # Stochastic Oscillator
        stoch_data = self.indicators.calculate_stochastic(
            enhanced_df['high'], enhanced_df['low'], enhanced_df['close']
        )
        for key, values in stoch_data.items():
            enhanced_df[key] = values
        
        # Williams %R
        enhanced_df['williams_r'] = self.indicators.calculate_williams_r(
            enhanced_df['high'], enhanced_df['low'], enhanced_df['close']
        )
        
        # ATR
        enhanced_df['atr'] = self.indicators.calculate_atr(
            enhanced_df['high'], enhanced_df['low'], enhanced_df['close']
        )
        
        # OBV (if volume data is available)
        if 'volume' in enhanced_df.columns:
            enhanced_df['obv'] = self.indicators.calculate_obv(enhanced_df['close'], enhanced_df['volume'])
        
        return enhanced_df
    
    def add_price_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add price pattern features
        
        Args:
            df (pd.DataFrame): OHLCV data
            
        Returns:
            pd.DataFrame: Data with price pattern features
        """
        if df.empty:
            return df
        
        # Candlestick patterns
        df['doji'] = np.abs(df['close'] - df['open']) <= (df['high'] - df['low']) * 0.1
        df['hammer'] = ((df['close'] - df['low']) > 2 * (df['open'] - df['close'])) & \
                       ((df['high'] - df['close']) < (df['close'] - df['low']) * 0.3)
        df['shooting_star'] = ((df['high'] - df['close']) > 2 * (df['close'] - df['open'])) & \
                              ((df['close'] - df['low']) < (df['high'] - df['close']) * 0.3)
        
        # Price gaps
        df['gap_up'] = df['low'] > df['high'].shift(1)
        df['gap_down'] = df['high'] < df['low'].shift(1)
        
        # Support and resistance levels
        df['local_high'] = (df['high'] > df['high'].shift(1)) & (df['high'] > df['high'].shift(-1))
        df['local_low'] = (df['low'] < df['low'].shift(1)) & (df['low'] < df['low'].shift(-1))
        
        return df
    
    def add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add momentum-based features
        
        Args:
            df (pd.DataFrame): Price data
            
        Returns:
            pd.DataFrame: Data with momentum features
        """
        if df.empty:
            return df
        
        # Rate of Change (ROC)
        for period in [1, 5, 10, 20]:
            df[f'roc_{period}'] = ((df['close'] / df['close'].shift(period)) - 1) * 100
        
        # Momentum
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['close'] - df['close'].shift(period)
        
        # Price acceleration
        df['price_acceleration'] = df['close'].diff().diff()
        
        # Velocity (rate of price change)
        df['velocity_5'] = df['close'].diff(5) / 5
        df['velocity_10'] = df['close'].diff(10) / 10
        
        return df
    
    def add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility-based features
        
        Args:
            df (pd.DataFrame): Price data
            
        Returns:
            pd.DataFrame: Data with volatility features
        """
        if df.empty:
            return df
        
        # Historical volatility
        returns = df['close'].pct_change()
        for window in [5, 10, 20, 30]:
            df[f'volatility_{window}'] = returns.rolling(window=window).std() * np.sqrt(252)
        
        # Volatility ratio
        df['volatility_ratio'] = df['volatility_5'] / df['volatility_20']
        
        # True Range
        df['true_range'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                np.abs(df['high'] - df['close'].shift(1)),
                np.abs(df['low'] - df['close'].shift(1))
            )
        )
        
        # Volatility breakout
        df['volatility_breakout'] = df['true_range'] > df['true_range'].rolling(20).mean() * 2
        
        return df
    
    def create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create target variables for machine learning
        
        Args:
            df (pd.DataFrame): Feature data
            
        Returns:
            pd.DataFrame: Data with target variables
        """
        if df.empty:
            return df
        
        # Next day closing price (regression target)
        df['target_price'] = df['close'].shift(-1)
        
        # Price direction (classification target)
        df['target_direction'] = np.where(df['target_price'] > df['close'], 1, 0)
        
        # Multi-class direction (up, down, stable)
        price_change_pct = ((df['target_price'] / df['close']) - 1) * 100
        df['target_direction_3class'] = np.where(
            price_change_pct > 1, 2,  # Up (>1% increase)
            np.where(price_change_pct < -1, 0, 1)  # Down (<-1% decrease), Stable (between -1% and 1%)
        )
        
        # Volatility target (next day volatility)
        returns = df['close'].pct_change()
        df['target_volatility'] = returns.rolling(window=5).std().shift(-1)
        
        # High volatility binary target
        volatility_threshold = df['target_volatility'].quantile(0.75)
        df['target_high_volatility'] = (df['target_volatility'] > volatility_threshold).astype(int)
        
        return df
    
    def engineer_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Apply all feature engineering steps
        
        Args:
            df (pd.DataFrame): Processed data
            symbol (str): Cryptocurrency symbol
            
        Returns:
            pd.DataFrame: Fully engineered features
        """
        if df.empty:
            self.logger.warning(f"No data to engineer features for {symbol}")
            return df
        
        self.logger.info(f"Engineering features for {symbol}")
        
        # Apply all feature engineering steps
        df = self.add_technical_indicators(df)
        df = self.add_price_patterns(df)
        df = self.add_momentum_features(df)
        df = self.add_volatility_features(df)
        df = self.create_target_variables(df)
        
        # Remove rows with NaN values in target variables
        df = df.dropna(subset=['target_price', 'target_direction'])
        
        self.logger.info(f"Feature engineering completed for {symbol}. Shape: {df.shape}")
        
        return df

def main():
    """Main function to run feature engineering"""
    from utils import load_data

    print("⚙️ Starting Feature Engineering")
    print("=" * 40)

    # Initialize engineer
    engineer = CryptoFeatureEngineer()

    # Define symbols and intervals
    symbols = ['BTC', 'ETH', 'BNB', 'ADA', 'SOL']
    intervals = ['5m', '1h']

    for symbol in symbols:
        for interval in intervals:
            print(f"\n🔧 Processing {symbol} {interval}")

            # Load processed data
            filename = f"{symbol}_{interval}_processed.csv"
            df = load_data(filename, 'data/processed')

            if df is not None and not df.empty:
                # Convert datetime column to index if needed
                if 'datetime' in df.columns:
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    df.set_index('datetime', inplace=True)
                elif df.index.name != 'datetime':
                    try:
                        df.index = pd.to_datetime(df.index)
                    except:
                        print(f"⚠️ Could not convert index to datetime for {symbol} {interval}")

                # Engineer features
                engineered_df = engineer.engineer_features(df, symbol)

                if not engineered_df.empty:
                    # Save engineered features
                    filename = f"{symbol}_{interval}_features.csv"
                    save_data(engineered_df, filename, 'data/processed')

                    print(f"✅ Saved {len(engineered_df)} records with {len(engineered_df.columns)} features")
                else:
                    print(f"❌ No features created for {symbol} {interval}")
            else:
                print(f"❌ No processed data found for {symbol} {interval}")

    print(f"\n🎯 Feature engineering completed!")

if __name__ == "__main__":
    main()
