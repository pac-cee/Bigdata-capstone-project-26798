"""
Tableau Data Preparation Script
Author: [Your Name]
Course: INSY 8413 | Introduction to Big Data Analytics

This script prepares cryptocurrency data for Tableau dashboard creation
"""

import sys
import os
sys.path.append('../src')

import pandas as pd
import numpy as np
import json
from datetime import datetime
from utils import load_data, CRYPTO_SYMBOLS

def prepare_main_dashboard_data():
    """Prepare the main dataset for Tableau dashboard"""
    print("📊 Preparing main dashboard data...")
    
    symbols = ['BTC', 'ETH', 'BNB', 'ADA', 'SOL']
    combined_data = []
    
    for symbol in symbols:
        print(f"Processing {symbol}...")
        
        # Load hourly features data
        df = load_data(f"{symbol}_1h_features.csv", "../data/processed")
        
        if df is not None and not df.empty:
            # Convert datetime index to column
            if 'datetime' not in df.columns:
                if df.index.name == 'datetime' or 'datetime' in str(df.index.name):
                    df = df.reset_index()
                else:
                    # If index is datetime but not named, create datetime column
                    df['datetime'] = df.index
                    df = df.reset_index(drop=True)

            # Ensure datetime is properly formatted
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
            
            # Add symbol information
            df['Symbol'] = symbol
            df['Crypto_Name'] = CRYPTO_SYMBOLS[symbol]
            
            # Select key columns for dashboard
            dashboard_columns = [
                'datetime', 'Symbol', 'Crypto_Name',
                'open', 'high', 'low', 'close', 'volume',
                'price_change', 'price_change_pct', 'returns_1d',
                'volatility_7d', 'volatility_30d',
                'rsi_14', 'macd', 'macd_signal', 'macd_histogram',
                'bb_upper', 'bb_middle', 'bb_lower', 'bb_position',
                'stoch_k', 'stoch_d', 'williams_r', 'atr',
                'close_ma_7', 'close_ma_14', 'close_ma_30',
                'volume_ma_7', 'volume_ratio'
            ]
            
            # Keep only available columns
            available_columns = [col for col in dashboard_columns if col in df.columns]
            df_filtered = df[available_columns].copy()
            
            # Add calculated fields for Tableau
            df_filtered['Price_Change_Direction'] = df_filtered['price_change_pct'].apply(
                lambda x: 'Up' if x > 0 else ('Down' if x < 0 else 'Neutral')
            )
            
            df_filtered['Volatility_Level'] = df_filtered['volatility_30d'].apply(
                lambda x: 'High' if x > 0.05 else ('Medium' if x > 0.02 else 'Low')
            )
            
            df_filtered['RSI_Signal'] = df_filtered['rsi_14'].apply(
                lambda x: 'Overbought' if x > 70 else ('Oversold' if x < 30 else 'Neutral')
            )
            
            df_filtered['Trend_Direction'] = np.where(
                df_filtered['close'] > df_filtered['close_ma_30'], 'Bullish', 'Bearish'
            )
            
            # Add market cap approximation (price * volume)
            df_filtered['Market_Activity'] = df_filtered['close'] * df_filtered['volume']
            
            combined_data.append(df_filtered)
            print(f"✅ {symbol}: {len(df_filtered)} records processed")
        else:
            print(f"❌ No data found for {symbol}")
    
    if combined_data:
        # Combine all data
        dashboard_data = pd.concat(combined_data, ignore_index=True)
        
        # Sort by datetime (if column exists)
        if 'datetime' in dashboard_data.columns:
            dashboard_data = dashboard_data.sort_values(['datetime', 'Symbol'])
        else:
            dashboard_data = dashboard_data.sort_values(['Symbol'])
        
        # Save for Tableau
        dashboard_data.to_csv("crypto_dashboard_data.csv", index=False)
        print(f"✅ Main dashboard data saved: {len(dashboard_data)} records")
        return dashboard_data
    else:
        print("❌ No data to combine")
        return None

def prepare_correlation_data():
    """Prepare correlation matrix data for Tableau"""
    print("\n🔗 Preparing correlation data...")
    
    symbols = ['BTC', 'ETH', 'BNB', 'ADA', 'SOL']
    price_data = {}
    returns_data = {}
    
    for symbol in symbols:
        df = load_data(f"{symbol}_1h_features.csv", "../data/processed")
        if df is not None and not df.empty:
            price_data[symbol] = df['close']
            returns_data[symbol] = df['returns_1d']
    
    if price_data:
        # Calculate correlation matrices
        price_df = pd.DataFrame(price_data)
        returns_df = pd.DataFrame(returns_data)
        
        price_corr = price_df.corr()
        returns_corr = returns_df.corr()
        
        # Convert to long format for Tableau
        corr_data = []
        
        for i, symbol1 in enumerate(price_corr.columns):
            for j, symbol2 in enumerate(price_corr.columns):
                corr_data.append({
                    'Symbol_1': symbol1,
                    'Symbol_2': symbol2,
                    'Symbol_1_Name': CRYPTO_SYMBOLS[symbol1],
                    'Symbol_2_Name': CRYPTO_SYMBOLS[symbol2],
                    'Price_Correlation': price_corr.iloc[i, j],
                    'Returns_Correlation': returns_corr.iloc[i, j],
                    'Correlation_Strength': 'Strong' if abs(price_corr.iloc[i, j]) > 0.7 else 
                                          ('Medium' if abs(price_corr.iloc[i, j]) > 0.3 else 'Weak')
                })
        
        corr_df = pd.DataFrame(corr_data)
        corr_df.to_csv("correlation_matrix.csv", index=False)
        print(f"✅ Correlation data saved: {len(corr_df)} records")
        return corr_df
    else:
        print("❌ No price data for correlation analysis")
        return None

def prepare_ml_results_data():
    """Prepare ML results data for Tableau"""
    print("\n🤖 Preparing ML results data...")
    
    try:
        with open('../data/processed/ml_results.json', 'r') as f:
            ml_results = json.load(f)
        
        ml_data = []
        
        for symbol in ml_results:
            # Price prediction results
            if 'price_prediction' in ml_results[symbol]:
                results = ml_results[symbol]['price_prediction']
                
                for model in ['linear_regression', 'random_forest', 'neural_network', 'ensemble']:
                    if model in results:
                        model_data = results[model]
                        ml_data.append({
                            'Symbol': symbol,
                            'Crypto_Name': CRYPTO_SYMBOLS.get(symbol, symbol),
                            'Model_Type': model.replace('_', ' ').title(),
                            'Task': 'Price Prediction',
                            'RMSE': model_data.get('test_rmse', 0),
                            'MAE': model_data.get('test_mae', 0),
                            'R2_Score': model_data.get('test_r2', 0),
                            'Accuracy': None,
                            'F1_Score': None
                        })
            
            # Direction classification results
            if 'direction_classification' in ml_results[symbol]:
                results = ml_results[symbol]['direction_classification']
                
                for model in ['logistic_regression', 'random_forest']:
                    if model in results:
                        model_data = results[model]
                        ml_data.append({
                            'Symbol': symbol,
                            'Crypto_Name': CRYPTO_SYMBOLS.get(symbol, symbol),
                            'Model_Type': model.replace('_', ' ').title(),
                            'Task': 'Direction Classification',
                            'RMSE': None,
                            'MAE': None,
                            'R2_Score': None,
                            'Accuracy': model_data.get('accuracy', 0),
                            'F1_Score': model_data.get('f1', 0)
                        })
        
        if ml_data:
            ml_df = pd.DataFrame(ml_data)
            
            # Add performance categories
            ml_df['Performance_Category'] = ml_df.apply(
                lambda row: 'Excellent' if (row['R2_Score'] and row['R2_Score'] > 0.8) or 
                                         (row['Accuracy'] and row['Accuracy'] > 0.8)
                           else 'Good' if (row['R2_Score'] and row['R2_Score'] > 0.6) or 
                                        (row['Accuracy'] and row['Accuracy'] > 0.7)
                           else 'Fair' if (row['R2_Score'] and row['R2_Score'] > 0.4) or 
                                        (row['Accuracy'] and row['Accuracy'] > 0.6)
                           else 'Poor', axis=1
            )
            
            ml_df.to_csv("ml_predictions.csv", index=False)
            print(f"✅ ML results data saved: {len(ml_df)} records")
            return ml_df
        else:
            print("❌ No ML results to process")
            return None
            
    except FileNotFoundError:
        print("❌ ML results file not found")
        return None

def prepare_summary_statistics():
    """Prepare summary statistics for Tableau"""
    print("\n📈 Preparing summary statistics...")
    
    symbols = ['BTC', 'ETH', 'BNB', 'ADA', 'SOL']
    summary_data = []
    
    for symbol in symbols:
        df = load_data(f"{symbol}_1h_features.csv", "../data/processed")
        
        if df is not None and not df.empty:
            # Calculate summary statistics
            summary = {
                'Symbol': symbol,
                'Crypto_Name': CRYPTO_SYMBOLS[symbol],
                'Current_Price': df['close'].iloc[-1],
                'Price_Change_24h': df['price_change_pct'].iloc[-1],
                'Total_Return_6m': ((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100,
                'Average_Volume': df['volume'].mean(),
                'Volatility_30d': df['volatility_30d'].mean(),
                'Max_Price': df['close'].max(),
                'Min_Price': df['close'].min(),
                'Average_RSI': df['rsi_14'].mean(),
                'Current_RSI': df['rsi_14'].iloc[-1],
                'Days_Above_MA30': (df['close'] > df['close_ma_30']).sum(),
                'Total_Days': len(df),
                'Bullish_Percentage': (df['close'] > df['close_ma_30']).mean() * 100,
                'High_Volatility_Days': (df['volatility_7d'] > df['volatility_7d'].quantile(0.75)).sum(),
                'Market_Cap_Proxy': df['close'].iloc[-1] * df['volume'].mean()
            }
            
            # Add risk categories
            if summary['Volatility_30d'] > 0.05:
                summary['Risk_Level'] = 'High'
            elif summary['Volatility_30d'] > 0.02:
                summary['Risk_Level'] = 'Medium'
            else:
                summary['Risk_Level'] = 'Low'
            
            # Add performance categories
            if summary['Total_Return_6m'] > 20:
                summary['Performance'] = 'Excellent'
            elif summary['Total_Return_6m'] > 0:
                summary['Performance'] = 'Good'
            elif summary['Total_Return_6m'] > -20:
                summary['Performance'] = 'Fair'
            else:
                summary['Performance'] = 'Poor'
            
            summary_data.append(summary)
            print(f"✅ {symbol}: Summary calculated")
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv("crypto_summary.csv", index=False)
        print(f"✅ Summary statistics saved: {len(summary_df)} records")
        return summary_df
    else:
        print("❌ No summary data to create")
        return None

def main():
    """Main function to prepare all Tableau data"""
    print("🚀 PREPARING TABLEAU DASHBOARD DATA")
    print("=" * 50)
    
    # Create dashboard directory if it doesn't exist
    os.makedirs(".", exist_ok=True)
    
    # Prepare all datasets
    main_data = prepare_main_dashboard_data()
    corr_data = prepare_correlation_data()
    ml_data = prepare_ml_results_data()
    summary_data = prepare_summary_statistics()
    
    print(f"\n✅ TABLEAU DATA PREPARATION COMPLETED!")
    print("=" * 50)
    print("📁 Files created:")
    print("  • crypto_dashboard_data.csv - Main dataset")
    print("  • correlation_matrix.csv - Correlation analysis")
    print("  • ml_predictions.csv - ML model results")
    print("  • crypto_summary.csv - Summary statistics")
    print("\n📊 Ready for Tableau Public!")
    print("💡 Follow the dashboard_guide.md for step-by-step instructions")

if __name__ == "__main__":
    main()
