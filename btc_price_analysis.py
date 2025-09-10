"""
BTC Price Analysis with Moving Averages and Autocorrelation

This script analyzes Bitcoin price data using moving averages,
autocorrelation analysis, and transformation techniques to
identify trends and significant periods in the price movements.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf


def download_btc_data():
    """
    Download historical Bitcoin price data from Yahoo Finance.
    
    Returns:
        pandas.DataFrame: DataFrame containing OHLCV data for BTC-USD
    """
    btc_data = yf.download(
        "BTC-USD", 
        start="2020-01-01", 
        end="2024-09-12", 
        progress=False
    )
    return btc_data


def calculate_moving_averages(data):
    """
    Calculate fast and slow simple moving averages for price data.
    
    Args:
        data (pandas.DataFrame): DataFrame containing price data
        
    Returns:
        pandas.DataFrame: DataFrame with added SMA columns
    """
    # Calculate 20-day simple moving average (fast)
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    
    # Calculate 50-day simple moving average (slow)
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    
    return data


def plot_price_with_moving_averages(data):
    """
    Plot BTC price with moving averages.
    
    Args:
        data (pandas.DataFrame): DataFrame containing price and SMA data
    """
    plt.figure(figsize=(14, 7))
    plt.plot(data['Close'], label='BTC Price', color='blue')
    plt.plot(data['SMA_20'], label='20-day SMA', color='red')
    plt.plot(data['SMA_50'], label='50-day SMA', color='green')
    plt.title('BTC Price with Moving Averages')
    plt.legend()
    plt.show()


def plot_autocorrelation_function(data):
    """
    Plot autocorrelation function for BTC closing prices.
    
    Args:
        data (pandas.DataFrame): DataFrame containing price data
    """
    plt.figure(figsize=(12, 6))
    plot_acf(data['Close'], lags=50)
    plt.title('Autocorrelation Function for BTC Price')
    plt.show()


def find_significant_autocorrelation_lags(data, threshold=0.2):
    """
    Identify significant autocorrelation lags above a threshold.
    
    Args:
        data (pandas.DataFrame): DataFrame containing price data
        threshold (float): Minimum autocorrelation value to consider significant
        
    Returns:
        list: List of significant lag values
    """
    # Calculate autocorrelation values
    autocorr_values = acf(data['Close'], nlags=50)
    
    # Find lags with autocorrelation above threshold
    significant_lags = [lag for lag, value in enumerate(autocorr_values) 
                       if value > threshold]
    
    print("Significant lags:", significant_lags)
    return significant_lags


def analyze_log_transformation(data):
    """
    Apply logarithmic transformation and calculate growth rate.
    
    Args:
        data (pandas.DataFrame): DataFrame containing price data
        
    Returns:
        pandas.DataFrame: DataFrame with added log and growth columns
    """
    # Apply logarithmic transformation to closing prices
    data['Log_Close'] = np.log(data['Close'])
    
    # Calculate daily growth rate as difference of log prices
    data['Growth'] = data['Log_Close'].diff()
    
    return data


def plot_log_transformation(data):
    """
    Plot logarithmic transformation of BTC price.
    
    Args:
        data (pandas.DataFrame): DataFrame containing log price data
    """
    plt.figure(figsize=(14, 7))
    plt.plot(data['Log_Close'], label='Log of BTC Price', color='purple')
    plt.title('Logarithmic Transformation of BTC Price')
    plt.legend()
    plt.show()


def plot_growth_rate(data):
    """
    Plot growth rate of BTC price.
    
    Args:
        data (pandas.DataFrame): DataFrame containing growth rate data
    """
    plt.figure(figsize=(14, 7))
    plt.plot(data['Growth'], label='Growth Rate', color='orange')
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Growth Rate of BTC Price')
    plt.legend()
    plt.show()


def main():
    """Main function to execute the BTC price analysis."""
    # Download BTC price data
    btc_data = download_btc_data()
    
    # Calculate moving averages
    btc_data = calculate_moving_averages(btc_data)
    
    # Visualize price with moving averages
    plot_price_with_moving_averages(btc_data)
    
    # Analyze autocorrelation
    plot_autocorrelation_function(btc_data)
    
    # Find significant autocorrelation lags
    significant_lags = find_significant_autocorrelation_lags(btc_data)
    
    # Apply logarithmic transformation and calculate growth
    btc_data = analyze_log_transformation(btc_data)
    
    # Visualize log transformation
    plot_log_transformation(btc_data)
    
    # Visualize growth rate
    plot_growth_rate(btc_data)


if __name__ == "__main__":
    main()

