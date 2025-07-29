"""
Utility functions for cryptocurrency market analysis
Author: [Your Name]
Course: INSY 8413 | Introduction to Big Data Analytics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style for consistent plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def setup_logging():
    """Setup logging configuration"""
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('crypto_analysis.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def validate_dataframe(df, required_columns):
    """
    Validate if dataframe has required columns
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
    Returns:
        bool: True if valid, False otherwise
    """
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        print(f"Missing columns: {missing_columns}")
        return False
    return True

def calculate_returns(prices):
    """
    Calculate percentage returns from price series
    
    Args:
        prices (pd.Series): Price series
    
    Returns:
        pd.Series: Percentage returns
    """
    return prices.pct_change().dropna()

def calculate_volatility(returns, window=30):
    """
    Calculate rolling volatility
    
    Args:
        returns (pd.Series): Return series
        window (int): Rolling window size
    
    Returns:
        pd.Series: Rolling volatility
    """
    return returns.rolling(window=window).std() * np.sqrt(365)  # Annualized

def format_currency(value, currency='USD'):
    """
    Format value as currency
    
    Args:
        value (float): Value to format
        currency (str): Currency symbol
    
    Returns:
        str: Formatted currency string
    """
    if currency == 'USD':
        return f"${value:,.2f}"
    return f"{value:,.2f} {currency}"

def save_data(data, filename, folder='data/processed'):
    """
    Save data to specified folder
    
    Args:
        data (pd.DataFrame): Data to save
        filename (str): Filename
        folder (str): Folder path
    """
    import os
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)
    data.to_csv(filepath, index=False)
    print(f"Data saved to: {filepath}")

def load_data(filename, folder='data/processed'):
    """
    Load data from specified folder
    
    Args:
        filename (str): Filename
        folder (str): Folder path
    
    Returns:
        pd.DataFrame: Loaded data
    """
    import os
    filepath = os.path.join(folder, filename)
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    else:
        print(f"File not found: {filepath}")
        return None

def plot_price_comparison(data_dict, title="Cryptocurrency Price Comparison"):
    """
    Plot price comparison for multiple cryptocurrencies
    
    Args:
        data_dict (dict): Dictionary with crypto names as keys and price series as values
        title (str): Plot title
    """
    plt.figure(figsize=(15, 8))
    
    for crypto, prices in data_dict.items():
        plt.plot(prices.index, prices.values, label=crypto, linewidth=2)
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (USD)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def calculate_correlation_matrix(data_dict):
    """
    Calculate correlation matrix for multiple cryptocurrencies
    
    Args:
        data_dict (dict): Dictionary with crypto names as keys and price series as values
    
    Returns:
        pd.DataFrame: Correlation matrix
    """
    # Combine all price series into a single DataFrame
    combined_df = pd.DataFrame(data_dict)
    
    # Calculate correlation matrix
    correlation_matrix = combined_df.corr()
    
    return correlation_matrix

def plot_correlation_heatmap(correlation_matrix, title="Cryptocurrency Correlation Matrix"):
    """
    Plot correlation heatmap
    
    Args:
        correlation_matrix (pd.DataFrame): Correlation matrix
        title (str): Plot title
    """
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap='coolwarm',
        center=0,
        square=True,
        fmt='.2f',
        cbar_kws={'label': 'Correlation Coefficient'}
    )
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def get_date_range(months_back=6):
    """
    Get date range for data collection
    
    Args:
        months_back (int): Number of months to go back
    
    Returns:
        tuple: (start_date, end_date) as timestamps
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=months_back * 30)
    
    return int(start_date.timestamp() * 1000), int(end_date.timestamp() * 1000)

def print_data_summary(df, name="Dataset"):
    """
    Print comprehensive data summary
    
    Args:
        df (pd.DataFrame): DataFrame to summarize
        name (str): Dataset name
    """
    print(f"\n{'='*50}")
    print(f"{name.upper()} SUMMARY")
    print(f"{'='*50}")
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"\nColumn Info:")
    print(df.info())
    print(f"\nMissing Values:")
    print(df.isnull().sum())
    print(f"\nBasic Statistics:")
    print(df.describe())

# Constants for cryptocurrency analysis
CRYPTO_SYMBOLS = {
    'BTC': 'Bitcoin',
    'ETH': 'Ethereum', 
    'BNB': 'Binance Coin',
    'ADA': 'Cardano',
    'SOL': 'Solana'
}

TIMEFRAMES = {
    '1m': '1 minute',
    '5m': '5 minutes', 
    '1h': '1 hour',
    '1d': '1 day'
}

# API Configuration
BINANCE_BASE_URL = "https://api.binance.com/api/v3"
RATE_LIMIT_CALLS = 1200  # Binance rate limit per minute
RATE_LIMIT_PERIOD = 60   # seconds
