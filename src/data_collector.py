"""
Binance API Data Collector for Cryptocurrency Market Analysis
Author: [Your Name]
Course: INSY 8413 | Introduction to Big Data Analytics

This module collects historical cryptocurrency data from Binance Public API
"""

import requests
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import os
from utils import (
    CRYPTO_SYMBOLS, 
    TIMEFRAMES, 
    BINANCE_BASE_URL,
    get_date_range,
    save_data,
    setup_logging
)

class BinanceDataCollector:
    """
    Collects cryptocurrency data from Binance Public API
    """
    
    def __init__(self):
        """Initialize the data collector"""
        self.base_url = BINANCE_BASE_URL
        self.logger = setup_logging()
        self.session = requests.Session()
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        
    def _rate_limit(self):
        """Implement rate limiting to avoid API limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """
        Make API request with error handling
        
        Args:
            endpoint (str): API endpoint
            params (dict): Request parameters
            
        Returns:
            dict: API response data
        """
        self._rate_limit()
        
        try:
            url = f"{self.base_url}/{endpoint}"
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            return None
    
    def get_klines(self, symbol: str, interval: str, start_time: int, end_time: int, limit: int = 1000) -> List:
        """
        Get kline/candlestick data from Binance
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTCUSDT')
            interval (str): Kline interval (e.g., '1m', '5m', '1h')
            start_time (int): Start timestamp in milliseconds
            end_time (int): End timestamp in milliseconds
            limit (int): Number of records to retrieve (max 1000)
            
        Returns:
            list: Kline data
        """
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_time,
            'endTime': end_time,
            'limit': limit
        }
        
        return self._make_request('klines', params)
    
    def get_24hr_ticker(self, symbol: str) -> Dict:
        """
        Get 24hr ticker price change statistics
        
        Args:
            symbol (str): Trading pair symbol
            
        Returns:
            dict: 24hr ticker data
        """
        params = {'symbol': symbol}
        return self._make_request('ticker/24hr', params)
    
    def collect_historical_data(self, symbol: str, interval: str, months_back: int = 6) -> pd.DataFrame:
        """
        Collect historical data for a cryptocurrency
        
        Args:
            symbol (str): Cryptocurrency symbol (e.g., 'BTC')
            interval (str): Time interval
            months_back (int): Number of months to collect
            
        Returns:
            pd.DataFrame: Historical price data
        """
        trading_pair = f"{symbol}USDT"
        start_time, end_time = get_date_range(months_back)
        
        self.logger.info(f"Collecting {symbol} data for interval {interval}")
        
        all_data = []
        current_start = start_time
        
        while current_start < end_time:
            # Calculate end time for this batch (max 1000 records)
            if interval == '1m':
                batch_end = min(current_start + (1000 * 60 * 1000), end_time)
            elif interval == '5m':
                batch_end = min(current_start + (1000 * 5 * 60 * 1000), end_time)
            elif interval == '1h':
                batch_end = min(current_start + (1000 * 60 * 60 * 1000), end_time)
            else:
                batch_end = end_time
            
            # Get data for this batch
            klines = self.get_klines(trading_pair, interval, current_start, batch_end)
            
            if klines:
                all_data.extend(klines)
                self.logger.info(f"Collected {len(klines)} records for {symbol}")
            else:
                self.logger.warning(f"No data received for {symbol} at {current_start}")
                break
            
            current_start = batch_end + 1
            time.sleep(0.1)  # Small delay between batches
        
        if not all_data:
            self.logger.error(f"No data collected for {symbol}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = self._process_klines_data(all_data, symbol)
        return df
    
    def _process_klines_data(self, klines_data: List, symbol: str) -> pd.DataFrame:
        """
        Process raw klines data into structured DataFrame
        
        Args:
            klines_data (list): Raw klines data from API
            symbol (str): Cryptocurrency symbol
            
        Returns:
            pd.DataFrame: Processed data
        """
        columns = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ]
        
        df = pd.DataFrame(klines_data, columns=columns)
        
        # Convert data types
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                          'quote_asset_volume', 'number_of_trades',
                          'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
        
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert timestamp to datetime
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('datetime', inplace=True)
        
        # Add symbol column
        df['symbol'] = symbol
        
        # Calculate additional metrics
        df['price_change'] = df['close'] - df['open']
        df['price_change_pct'] = (df['close'] - df['open']) / df['open'] * 100
        df['high_low_pct'] = (df['high'] - df['low']) / df['low'] * 100
        df['volume_usd'] = df['volume'] * df['close']
        
        # Remove unnecessary columns
        df.drop(['timestamp', 'close_time', 'ignore'], axis=1, inplace=True)
        
        return df
    
    def collect_all_cryptocurrencies(self, intervals: List[str] = ['1m', '5m', '1h'], months_back: int = 6) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Collect data for all cryptocurrencies and intervals
        
        Args:
            intervals (list): List of time intervals
            months_back (int): Number of months to collect
            
        Returns:
            dict: Nested dictionary with crypto -> interval -> DataFrame
        """
        all_data = {}
        
        for symbol in CRYPTO_SYMBOLS.keys():
            all_data[symbol] = {}
            
            for interval in intervals:
                self.logger.info(f"Collecting {symbol} data for {interval} interval")
                
                try:
                    df = self.collect_historical_data(symbol, interval, months_back)
                    
                    if not df.empty:
                        all_data[symbol][interval] = df
                        
                        # Save individual files
                        filename = f"{symbol}_{interval}_data.csv"
                        save_data(df, filename, 'data/raw')
                        
                        self.logger.info(f"Successfully collected {len(df)} records for {symbol} {interval}")
                    else:
                        self.logger.warning(f"No data collected for {symbol} {interval}")
                        
                except Exception as e:
                    self.logger.error(f"Error collecting {symbol} {interval}: {e}")
                    continue
                
                # Delay between different intervals
                time.sleep(1)
            
            # Delay between different cryptocurrencies
            time.sleep(2)
        
        return all_data
    
    def get_current_prices(self) -> Dict[str, float]:
        """
        Get current prices for all cryptocurrencies
        
        Returns:
            dict: Current prices
        """
        current_prices = {}
        
        for symbol in CRYPTO_SYMBOLS.keys():
            trading_pair = f"{symbol}USDT"
            ticker_data = self.get_24hr_ticker(trading_pair)
            
            if ticker_data:
                current_prices[symbol] = float(ticker_data['lastPrice'])
                self.logger.info(f"{symbol}: ${current_prices[symbol]:,.2f}")
            
            time.sleep(0.1)
        
        return current_prices

def main():
    """Main function to run data collection"""
    collector = BinanceDataCollector()
    
    print("🚀 Starting Cryptocurrency Data Collection")
    print("=" * 50)
    
    # Collect current prices
    print("\n📊 Current Prices:")
    current_prices = collector.get_current_prices()
    
    # Collect historical data
    print("\n📈 Collecting Historical Data...")
    intervals = ['5m', '1h']  # Start with these intervals to avoid rate limits
    all_data = collector.collect_all_cryptocurrencies(intervals, months_back=6)
    
    # Save summary
    summary = {
        'collection_date': datetime.now().isoformat(),
        'cryptocurrencies': list(CRYPTO_SYMBOLS.keys()),
        'intervals': intervals,
        'months_collected': 6,
        'current_prices': current_prices,
        'data_summary': {}
    }
    
    for symbol in all_data:
        summary['data_summary'][symbol] = {}
        for interval in all_data[symbol]:
            df = all_data[symbol][interval]
            summary['data_summary'][symbol][interval] = {
                'records': len(df),
                'date_range': f"{df.index.min()} to {df.index.max()}",
                'avg_price': float(df['close'].mean()),
                'price_volatility': float(df['close'].std())
            }
    
    # Save summary
    with open('data/raw/collection_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print("\n✅ Data collection completed!")
    print(f"📁 Data saved to: data/raw/")
    print(f"📋 Summary saved to: data/raw/collection_summary.json")

if __name__ == "__main__":
    main()
