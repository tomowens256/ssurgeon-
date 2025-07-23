import sys
import os
import time
import threading
import logging
import pytz
import numpy as np
import pandas as pd
import pandas_ta as ta
import joblib
import requests
import re
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, jsonify
from collections import deque
from oandapyV20 import API
from oandapyV20.exceptions import V20Error
import oandapyV20.endpoints.instruments as instruments

# ========================
# CONSTANTS & CONFIG
# ========================
NY_TZ = pytz.timezone("America/New_York")
MODEL_PATH = os.getenv("MODEL_PATH", "./ml_models")
FEATURES = [
    'adj close', 'garman_klass_vol', 'rsi_20', 'bb_low', 'bb_mid', 'bb_high',
    'atr_z', 'macd_z', 'dollar_volume', 'ma_10', 'ma_100', 'vwap', 'vwap_std',
    'rsi', 'ma_20', 'ma_30', 'ma_40', 'ma_60', 'trend_strength_up',
    'trend_strength_down', 'sl_price', 'tp_price', 'prev_volume', 'sl_distance',
    'tp_distance', 'rrr', 'log_sl', 'prev_body_size', 'prev_wick_up',
    'prev_wick_down', 'is_bad_combo', 'price_div_vol', 'rsi_div_macd',
    'price_div_vwap', 'sl_div_atr', 'tp_div_atr', 'rrr_div_rsi',
    'day_Friday', 'day_Monday', 'day_Sunday', 'day_Thursday', 'day_Tuesday',
    'day_Wednesday', 'session_q1', 'session_q2', 'session_q3', 'session_q4',
    'rsi_zone_neutral', 'rsi_zone_overbought', 'rsi_zone_oversold',
    'rsi_zone_unknown', 'trend_direction_downtrend', 'trend_direction_sideways',
    'trend_direction_uptrend', 'crt_BUY', 'crt_SELL', 'trade_type_BUY',
    'trade_type_SELL', 'combo_flag_dead', 'combo_flag_fair', 'combo_flag_fine',
    'combo_flag2_dead', 'combo_flag2_fair', 'combo_flag2_fine',
    'minutes,closed_0', 'minutes,closed_15', 'minutes,closed_30', 'minutes,closed_45'
]

# Oanda configuration
ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
API_KEY = os.getenv("OANDA_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Only needed instrument and timeframe
INSTRUMENT = "XAU_USD"
TIMEFRAME = "M15"

# Global variables
GLOBAL_LOCK = threading.Lock()
CRT_SIGNAL_COUNT = 0
LAST_SIGNAL_TIME = 0
SIGNALS = deque(maxlen=100)
TRADE_JOURNAL = deque(maxlen=50)
PERF_CACHE = {"updated": 0, "data": None}

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)

# Initialize Oanda API
oanda_api = API(access_token=API_KEY, environment="practice")

# Initialize Flask app
app = Flask(__name__)

# ========================
# UTILITY FUNCTIONS
# ========================
def parse_oanda_time(time_str):
    """Parse Oanda's timestamp with variable fractional seconds"""
    if '.' in time_str and len(time_str.split('.')[1]) > 7:
        time_str = re.sub(r'\.(\d{6})\d+', r'.\1', time_str)
    return datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=pytz.utc).astimezone(NY_TZ)

def send_telegram(message):
    """Send formatted message to Telegram with detailed error handling"""
    if len(message) > 4000:
        message = message[:4000] + "... [TRUNCATED]"
    
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        response = requests.post(url, json={
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': 'Markdown'
        }, timeout=10)
        
        logging.info(f"Telegram response: {response.status_code} - {response.text}")
        
        if response.status_code != 200:
            logging.error(f"Telegram error: {response.status_code} - {response.text}")
            return False
            
        if not response.json().get('ok'):
            logging.error(f"Telegram API error: {response.json()}")
            return False
            
        return True
    except Exception as e:
        logging.error(f"Telegram connection failed: {e}")
        return False

def fetch_candles():
    """Fetch candles for XAU_USD M15 with robust error handling"""
    params = {
        "granularity": TIMEFRAME,
        "count": 200,
        "price": "M"  # Midpoint prices
    }
    
    sleep_time = 10  # Initial sleep time for backoff
    max_attempts = 3
    
    for attempt in range(max_attempts):
        try:
            request = instruments.InstrumentsCandles(instrument=INSTRUMENT, params=params)
            response = oanda_api.request(request)
            candles = response.get('candles', [])
            
            if not candles:
                logging.warning("No candles received from Oanda")
                return pd.DataFrame()
            
            data = []
            for candle in candles:
                if not candle.get('complete', False):
                    continue
                    
                price_data = candle.get('mid', {})
                if not price_data:
                    logging.warning(f"Skipping candle with missing mid price data: {candle}")
                    continue
                
                data.append({
                    'time': parse_oanda_time(candle['time']),
                    'open': float(price_data['o']),
                    'high': float(price_data['h']),
                    'low': float(price_data['l']),
                    'close': float(price_data['c']),
                    'volume': int(candle.get('volume', 0))
                })
            
            if not data:
                logging.warning("No complete candles found in response")
                return pd.DataFrame()
                
            return pd.DataFrame(data)
            
        except V20Error as e:
            if "rate" in str(e).lower():
                logging.warning(f"Rate limit hit, sleeping {sleep_time}s (attempt {attempt+1}/{max_attempts})")
                time.sleep(sleep_time)
                sleep_time *= 2  # Exponential backoff
            else:
                logging.error(f"Oanda API error: {e}")
                return pd.DataFrame()
        except KeyError as e:
            logging.error(f"Key error in candle data: {e}")
            return pd.DataFrame()
        except Exception as e:
            logging.error(f"Unexpected error fetching candles: {e}")
            return pd.DataFrame()
    
    logging.error(f"Failed to fetch candles after {max_attempts} attempts")
    return pd.DataFrame()

# ========================
# FEATURE ENGINEER
# ========================
class FeatureEngineer:
    def __init__(self, history_size=200):
        self.history_size = history_size
        # Hardcoded combo flags from combo_flags.csv as tuples
        self.combo_flags = (
            ("SELL_sideways_nan", "dead"),
            ("BUY_sideways_nan", "dead"),
            ("SELL_sideways_(70, 80]", "dead"),
            ("BUY_sideways_(70, 80]", "fine"),
            ("SELL_sideways_(60, 70]", "fair"),
            ("SELL_sideways_(50, 60]", "fair"),
            ("BUY_sideways_(50, 60]", "fine"),
            ("BUY_sideways_(60, 70]", "fine"),
            ("BUY_sideways_(40, 50]", "fair"),
            ("SELL_sideways_(40, 50]", "fine"),
            ("SELL_sideways_(30, 40]", "fine"),
            ("SELL_uptrend_(50, 60]", "fine"),
            ("BUY_uptrend_(50, 60]", "fair"),
            ("SELL_uptrend_(40, 50]", "fine"),
            ("BUY_downtrend_(40, 50]", "fair"),
            ("BUY_uptrend_(60, 70]", "fine"),
            ("SELL_uptrend_(60, 70]", "fair"),
            ("BUY_uptrend_(40, 50]", "fair"),
            ("SELL_downtrend_(40, 50]", "fair"),
            ("BUY_uptrend_(30, 40]", "dead"),
            ("BUY_downtrend_(30, 40]", "fair"),
            ("BUY_downtrend_(50, 60]", "fine"),
            ("SELL_downtrend_(50, 60]", "fair"),
            ("SELL_downtrend_(30, 40]", "fine"),
            ("BUY_uptrend_(70, 80]", "fine"),
            ("SELL_uptrend_(70, 80]", "fair"),
            ("SELL_uptrend_(80, 100]", "dead"),
            ("SELL_downtrend_(20, 30]", "fine"),
            ("SELL_sideways_(80, 100]", "dead"),
            ("BUY_sideways_(30, 40]", "fair"),
            ("SELL_downtrend_(60, 70]", "dead"),
            ("SELL_uptrend_(30, 40]", "fine"),
            ("SELL_downtrend_(70, 80]", "dead"),
            ("BUY_downtrend_(20, 30]", "fair"),
            ("BUY_downtrend_(0, 20]", "dead"),
            ("BUY_sideways_(20, 30]", "dead"),
            ("SELL_sideways_(20, 30]", "fine"),
            ("BUY_downtrend_(60, 70]", "fine"),
            ("BUY_sideways_(0, 20]", "dead"),
            ("SELL_downtrend_(0, 20]", "fine"),
            ("BUY_uptrend_(80, 100]", "fine"),
            ("SELL_sideways_(0, 20]", "fine"),
            ("BUY_sideways_(80, 100]", "fine"),
            ("SELL_uptrend_(20, 30]", "fine"),
            ("BUY_downtrend_(70, 80]", "fine"),
            ("BUY_uptrend_(20, 30]", "dead"),
            ("SELL_downtrend_(80, 100]", "dead"),
            ("BUY_uptrend_(0, 20]", "dead"),
            ("SELL_uptrend_(0, 20]", "dead"),
            ("nan_sideways_(50, 60]", "dead"),
            ("nan_sideways_(40, 50]", "dead")
        )
        # Hardcoded combo flags2 from combo_flags2.csv as tuples
        self.combo_flags2 = (
            ("SELL_nan_nan", "dead"),
            ("BUY_nan_nan", "dead"),
            ("SELL_(70, 80]_nan", "dead"),
            ("BUY_(70, 80]_nan", "dead"),
            ("SELL_(70, 80]_(0.527, 9.246]", "fair"),
            ("SELL_(60, 70]_(0.527, 9.246]", "fair"),
            ("SELL_(50, 60]_(0.527, 9.246]", "fine"),
            ("BUY_(50, 60]_(0.527, 9.246]", "fair"),
            ("BUY_(60, 70]_(0.527, 9.246]", "fine"),
            ("BUY_(40, 50]_(0.134, 0.527]", "fair"),
            ("SELL_(40, 50]_(0.134, 0.527]", "fine"),
            ("SELL_(30, 40]_(0.134, 0.527]", "fine"),
            ("BUY_(40, 50]_(-0.138, 0.134]", "fair"),
            ("SELL_(50, 60]_(-0.138, 0.134]", "fair"),
            ("BUY_(50, 60]_(-0.138, 0.134]", "fine"),
            ("SELL_(40, 50]_(-0.138, 0.134]", "fine"),
            ("BUY_(40, 50]_(-0.496, -0.138]", "fair"),
            ("BUY_(60, 70]_(-0.138, 0.134]", "fine"),
            ("SELL_(60, 70]_(0.134, 0.527]", "fair"),
            ("BUY_(50, 60]_(0.134, 0.527]", "fair"),
            ("SELL_(50, 60]_(0.134, 0.527]", "fine"),
            ("SELL_(40, 50]_(-0.496, -0.138]", "fair"),
            ("SELL_(30, 40]_(-0.496, -0.138]", "fine"),
            ("BUY_(40, 50]_(-12.386, -0.496]", "fine"),
            ("BUY_(60, 70]_(0.134, 0.527]", "fine"),
            ("BUY_(30, 40]_(-0.496, -0.138]", "fair"),
            ("BUY_(50, 60]_(-0.496, -0.138]", "fine"),
            ("BUY_(30, 40]_(-0.138, 0.134]", "dead"),
            ("SELL_(50, 60]_(-0.496, -0.138]", "fair"),
            ("BUY_(70, 80]_(0.134, 0.527]", "fine"),
            ("SELL_(80, 100]_(0.527, 9.246]", "dead"),
            ("BUY_(70, 80]_(0.527, 9.246]", "fine"),
            ("SELL_(40, 50]_(0.527, 9.246]", "fine"),
            ("SELL_(40, 50]_(-12.386, -0.496]", "fair"),
            ("BUY_(30, 40]_(-12.386, -0.496]", "fair"),
            ("SELL_(30, 40]_(-12.386, -0.496]", "fine"),
            ("SELL_(20, 30]_(-12.386, -0.496]", "fine"),
            ("BUY_(50, 60]_(-12.386, -0.496]", "fine"),
            ("SELL_(80, 100]_(-0.496, -0.138]", "dead"),
            ("SELL_(30, 40]_(-0.138, 0.134]", "fine"),
            ("SELL_(70, 80]_(-0.138, 0.134]", "dead"),
            ("SELL_(50, 60]_(-12.386, -0.496]", "dead"),
            ("BUY_(40, 50]_(0.527, 9.246]", "dead"),
            ("SELL_(20, 30]_(-0.496, -0.138]", "fine"),
            ("BUY_(20, 30]_(-12.386, -0.496]", "fair"),
            ("BUY_(0, 20]_(-12.386, -0.496]", "dead"),
            ("SELL_(60, 70]_(-0.138, 0.134]", "dead"),
            ("BUY_(20, 30]_(-0.496, -0.138]", "dead"),
            ("BUY_(60, 70]_(-0.496, -0.138]", "fine"),
            ("BUY_(70, 80]_(-0.138, 0.134]", "fine"),
            ("SELL_(70, 80]_(0.134, 0.527]", "dead"),
            ("SELL_(0, 20]_(-12.386, -0.496]", "fine"),
            ("BUY_(80, 100]_(0.527, 9.246]", "fine"),
            ("SELL_(60, 70]_(-0.496, -0.138]", "dead"),
            ("SELL_(30, 40]_(0.527, 9.246]", "fine"),
            ("BUY_(30, 40]_(0.134, 0.527]", "dead"),
            ("SELL_(60, 70]_(-12.386, -0.496]", "dead"),
            ("BUY_(60, 70]_(-12.386, -0.496]", "fine"),
            ("BUY_(80, 100]_(-0.496, -0.138]", "fine"),
            ("BUY_(80, 100]_(0.134, 0.527]", "fine"),
            ("SELL_(20, 30]_(-0.138, 0.134]", "fine"),
            ("SELL_(0, 20]_(-0.496, -0.138]", "fair"),
            ("BUY_(30, 40]_(0.527, 9.246]", "dead"),
            ("BUY_(20, 30]_(-0.138, 0.134]", "dead"),
            ("SELL_(70, 80]_(-0.496, -0.138]", "dead"),
            ("BUY_(80, 100]_(-0.138, 0.134]", "fine"),
            ("SELL_(20, 30]_(0.134, 0.527]", "fine"),
            ("BUY_(0, 20]_(-0.496, -0.138]", "dead"),
            ("SELL_(80, 100]_(0.134, 0.527]", "fair"),
            ("BUY_(0, 20]_(-0.138, 0.134]", "dead"),
            ("BUY_(0, 20]_(0.134, 0.527]", "dead"),
            ("SELL_(0, 20]_(-0.138, 0.134]", "fine"),
            ("BUY_(70, 80]_(-0.496, -0.138]", "fine"),
            ("SELL_(70, 80]_(-12.386, -0.496]", "dead"),
            ("SELL_(20, 30]_(0.527, 9.246]", "fine"),
            ("BUY_(20, 30]_(0.134, 0.527]", "dead"),
            ("SELL_(80, 100]_(-0.138, 0.134]", "dead"),
            ("BUY_(20, 30]_(0.527, 9.246]", "dead"),
            ("BUY_(0, 20]_(0.527, 9.246]", "dead"),
            ("nan_(50, 60]_(-12.386, -0.496]", "dead"),
            ("nan_(40, 50]_(-0.138, 0.134]", "dead"),
            ("nan_(40, 50]_(-0.496, -0.138]", "dead"),
            ("nan_(50, 60]_(0.134, 0.527]", "dead"),
            ("nan_(50, 60]_(0.527, 9.246]", "dead")
        )

    def calculate_technical_indicators(self, df):
        df = df.copy()
        
        # Adjust close
        df['adj close'] = df['open']
        
        # Garman-Klass Volatility
        df['garman_klass_vol'] = (
            ((np.log(df['high']) - np.log(df['low'])) ** 2) / 2 -
            (2 * np.log(2) - 1) * ((np.log(df['adj close']) - np.log(df['open'])) ** 2)
        )
        
        # RSI
        df['rsi_20'] = ta.rsi(df['adj close'], length=20)
        df['rsi'] = ta.rsi(df['close'], length=14)
        
        # Bollinger Bands
        bb = ta.bbands(np.log1p(df['adj close']), length=20)
        df['bb_low'] = bb['BBL_20_2.0']
        df['bb_mid'] = bb['BBM_20_2.0']
        df['bb_high'] = bb['BBU_20_2.0']
        
        # ATR (z-scored)
        atr = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['atr_z'] = (atr - atr.mean()) / atr.std()
        
        # MACD (z-scored)
        macd = ta.macd(df['adj close'], fast=12, slow=26, signal=9)
        df['macd_z'] = (macd['MACD_12_26_9'] - macd['MACD_12_26_9'].mean()) / macd['MACD_12_26_9'].std()
        
        # Dollar Volume
        df['dollar_volume'] = (df['adj close'] * df['volume']) / 1e6
        
        # Moving Averages
        df['ma_10'] = df['adj close'].rolling(window=10).mean()
        df['ma_100'] = df['adj close'].rolling(window=100).mean()
        df['ma_20'] = df['close'].rolling(window=20).mean()
        df['ma_30'] = df['close'].rolling(window=30).mean()
        df['ma_40'] = df['close'].rolling(window=40).mean()
        df['ma_60'] = df['close'].rolling(window=60).mean()
        
        # VWAP and VWAP STD
        vwap_num = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum()
        vwap_den = df['volume'].cumsum()
        df['vwap'] = vwap_num / vwap_den
        df['vwap_std'] = df['vwap'].rolling(window=20).std()
        
        return df

    def calculate_crt_vectorized(self, df):
        df = df.copy()
        df['crt'] = None
        
        df['c1_low'] = df['low'].shift(2)
        df['c1_high'] = df['high'].shift(2)
        df['c2_low'] = df['low'].shift(1)
        df['c2_high'] = df['high'].shift(1)
        df['c2_close'] = df['close'].shift(1)
        
        df['c2_range'] = df['c2_high'] - df['c2_low']
        df['c2_mid'] = df['c2_low'] + (0.5 * df['c2_range'])
        
        buy_mask = (
            (df['c2_low'] < df['c1_low']) &
            (df['c2_close'] > df['c1_low']) &
            (df['open'] > df['c2_mid'])
        )
        sell_mask = (
            (df['c2_high'] > df['c1_high']) &
            (df['c2_close'] < df['c1_high']) &
            (df['open'] < df['c2_mid'])
        )
        
        df.loc[buy_mask, 'crt'] = 'BUY'
        df.loc[sell_mask, 'crt'] = 'SELL'
        
        df.drop(columns=['c1_low', 'c1_high', 'c2_low', 'c2_high', 'c2_close', 'c2_range', 'c2_mid'], inplace=True)
        return df

    def calculate_trade_features(self, df, signal_type):
        df = df.copy()
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2] if len(df) > 1 else last_row
        
        entry = last_row['open']
        if signal_type == 'SELL':
            df['sl_price'] = prev_row['high']
            risk = abs(entry - df['sl_price'])
            df['tp_price'] = entry - 4 * risk
        else:  # BUY
            df['sl_price'] = prev_row['low']
            risk = abs(entry - df['sl_price'])
            df['tp_price'] = entry + 4 * risk
        
        df['sl_distance'] = abs(entry - df['sl_price']) * 10
        df['tp_distance'] = abs(df['tp_price'] - entry) * 10
        df['rrr'] = df['tp_distance'] / df['sl_distance'].replace(0, np.nan)
        df['log_sl'] = np.log1p(df['sl_price'])
        
        return df

    def calculate_categorical_features(self, df):
        df = df.copy()
        
        # Day of week
        df['day'] = df['time'].dt.day_name()
        df = pd.get_dummies(df, columns=['day'], prefix='day', drop_first=False)
        
        # Session
        def get_session(hour):
            if 0 <= hour < 6:
                return 'q2'
            elif 6 <= hour < 12:
                return 'q3'
            elif 12 <= hour < 18:
                return 'q4'
            else:
                return 'q1'
        df['session'] = df['time'].dt.hour.apply(get_session)
        df = pd.get_dummies(df, columns=['session'], prefix='session', drop_first=False)
        
        # RSI Zone
        def rsi_zone(rsi):
            if pd.isna(rsi):
                return 'unknown'
            elif rsi < 30:
                return 'oversold'
            elif rsi > 70:
                return 'overbought'
            else:
                return 'neutral'
        df['rsi_zone'] = df['rsi'].apply(rsi_zone)
        df = pd.get_dummies(df, columns=['rsi_zone'], prefix='rsi_zone', drop_first=False)
        
        # Trend Direction
        def is_bullish_stack(row):
            return int(row['ma_20'] > row['ma_30'] > row['ma_40'] > row['ma_60'])
        def is_bearish_stack(row):
            return int(row['ma_20'] < row['ma_30'] < row['ma_40'] < row['ma_60'])
        
        df['trend_strength_up'] = df.apply(is_bullish_stack, axis=1).astype(float)
        df['trend_strength_down'] = df.apply(is_bearish_stack, axis=1).astype(float)
        
        def get_trend(row):
            if row['trend_strength_up'] > row['trend_strength_down']:
                return 'uptrend'
            elif row['trend_strength_down'] > row['trend_strength_up']:
                return 'downtrend'
            else:
                return 'sideways'
        df['trend_direction'] = df.apply(get_trend, axis=1)
        df = pd.get_dummies(df, columns=['trend_direction'], prefix='trend_direction', drop_first=False)
        
        return df

    def calculate_combo_flags(self, df, signal_type):
        df = df.copy()
        
        # RSI and MACD bins
        df['rsi_bin'] = pd.cut(df['rsi'], bins=[0, 20, 30, 40, 50, 60, 70, 80, 100])
        df['macd_z_bin'] = pd.qcut(df['macd_z'], q=5, duplicates='drop', labels=[
            '(-12.386, -0.496]', '(-0.496, -0.138]', '(-0.138, 0.134]', '(0.134, 0.527]', '(0.527, 9.246]'
        ])
        
        # Combo keys calculated from features
        df['trade_type'] = signal_type
        df['combo_key'] = df['trade_type'] + '_' + df['trend_direction'] + '_' + df['rsi_bin'].astype(str)
        df['combo_key2'] = df[['trade_type', 'rsi_bin', 'macd_z_bin']].astype(str).agg('_'.join, axis=1)
        
        # Convert tuple lists to dictionaries for mapping
        combo_flag_dict = dict(self.combo_flags)
        combo_flag2_dict = dict(self.combo_flags2)
        
        # Map calculated combo keys to their flags
        df['combo_flag'] = df['combo_key'].map(combo_flag_dict).fillna('dead')
        df['combo_flag2'] = df['combo_key2'].map(combo_flag2_dict).fillna('dead')
        
        # is_bad_combo (dead combo_flag or combo_flag2 indicates bad combo)
        df['is_bad_combo'] = ((df['combo_flag'] == 'dead') | (df['combo_flag2'] == 'dead')).astype(int)
        
        # Create dummy variables
        df = pd.get_dummies(df, columns=['combo_flag'], prefix='combo_flag', drop_first=False)
        df = pd.get_dummies(df, columns=['combo_flag2'], prefix='combo_flag2', drop_first=False)
        
        return df

    def calculate_minutes_closed(self, df, minutes_closed):
        df = df.copy()
        minute = minutes_closed
        if 0 <= minute < 15:
            minute_col = 'minutes,closed_15'
        elif 15 <= minute < 30:
            minute_col = 'minutes,closed_30'
        elif 30 <= minute < 45:
            minute_col = 'minutes,closed_45'
        else:
            minute_col = 'minutes,closed_0'
        
        for col in ['minutes,closed_0', 'minutes,closed_15', 'minutes,closed_30', 'minutes,closed_45']:
            df[col] = 1 if col == minute_col else 0
        
        return df

    def transform(self, df_history, signal_type, minutes_closed):
        try:
            if len(df_history) < self.history_size:
                logging.warning(f"Insufficient data: {len(df_history)} rows, need {self.history_size}")
                return None
            
            df = df_history.copy()
            
            # Ensure time is in correct format
            df['time'] = pd.to_datetime(df['time']).dt.tz_localize('UTC').dt.tz_convert('America/New_York')
            
            # Calculate technical indicators
            df = self.calculate_technical_indicators(df)
            
            # Calculate CRT signals
            df = self.calculate_crt_vectorized(df)
            
            # Calculate trade-related features
            df = self.calculate_trade_features(df, signal_type)
            
            # Calculate categorical features
            df = self.calculate_categorical_features(df)
            
            # Calculate combo flags
            df = self.calculate_combo_flags(df, signal_type)
            
            # Calculate minutes closed
            df = self.calculate_minutes_closed(df, minutes_closed)
            
            # Candle and volume features
            df['prev_volume'] = df['volume'].shift(1)
            df['body_size'] = abs(df['close'] - df['open'])
            df['wick_up'] = df['high'] - df[['close', 'open']].max(axis=1)
            df['wick_down'] = df[['close', 'open']].min(axis=1) - df['low']
            df['prev_body_size'] = df['body_size'].shift(1)
            df['prev_wick_up'] = df['wick_up'].shift(1)
            df['prev_wick_down'] = df['wick_down'].shift(1)
            
            # Derived metrics
            df['price_div_vol'] = df['adj close'] / (df['garman_klass_vol'] + 1e-6)
            df['rsi_div_macd'] = df['rsi'] / (df['macd_z'] + 1e-6)
            df['price_div_vwap'] = df['adj close'] / (df['vwap'] + 1e-6)
            df['sl_div_atr'] = df['sl_distance'] / (df['atr_z'] + 1e-6)
            df['tp_div_atr'] = df['tp_distance'] / (df['atr_z'] + 1e-6)
            df['rrr_div_rsi'] = df['rrr'] / (df['rsi'] + 1e-6)
            
            # Ensure CRT and trade type encoding
            df['crt_BUY'] = (df['crt'] == 'BUY').astype(int)
            df['crt_SELL'] = (df['crt'] == 'SELL').astype(int)
            df['trade_type_BUY'] = (signal_type == 'BUY').astype(int)
            df['trade_type_SELL'] = (signal_type == 'SELL').astype(int)
            
            # Select features in the correct order
            features = df.iloc[-1][FEATURES].astype(float)
            
            # Handle NaN values
            if features.isna().any():
                logging.warning(f"Missing features: {features[features.isna()].index.tolist()}")
                return None
            
            return features
        except Exception as e:
            logging.error(f"Feature engineering error: {e}")
            return None

# ========================
# CANDLE SCHEDULER
# ========================
class CandleScheduler(threading.Thread):
    def __init__(self, timeframe=15):
        super().__init__(daemon=True)
        self.timeframe = timeframe
        self.callback = None
        self.active = True
        self.next_candle = None
        self.event = threading.Event()
    
    def register_callback(self, callback):
        self.callback = callback
        
    def calculate_next_candle(self):
        now = datetime.now(NY_TZ)
        current_minute = now.minute
        remainder = current_minute % self.timeframe
        if remainder == 0:
            return now.replace(second=0, microsecond=0) + timedelta(minutes=self.timeframe)
        next_minute = current_minute - remainder + self.timeframe
        if next_minute >= 60:
            return now.replace(hour=now.hour + 1, minute=0, second=0, microsecond=0)
        return now.replace(minute=next_minute, second=0, microsecond=0)
    
    def calculate_minutes_closed(self, latest_time):
        if not latest_time:
            return 0
        now = datetime.now(NY_TZ)
        elapsed = (now - latest_time).total_seconds() / 60
        if elapsed < 15:
            return 15
        elif elapsed < 30:
            return 30
        elif elapsed < 45:
            return 45
        return 0
    
    def run(self):
        while self.active:
            try:
                self.next_candle = self.calculate_next_candle()
                now = datetime.now(NY_TZ)
                sleep_seconds = (self.next_candle - now).total_seconds()
                if sleep_seconds > 0:
                    logging.info(f"Sleeping {sleep_seconds:.1f}s until next candle")
                    time.sleep(sleep_seconds)
                
                # Fetch latest candle to pass to callback
                latest_candle_df = fetch_candles().iloc[-1:] if fetch_candles() is not None else pd.DataFrame()
                latest_time = latest_candle_df['time'].iloc[-1] if not latest_candle_df.empty else None
                minutes_closed = self.calculate_minutes_closed(latest_time)
                
                if self.callback:
                    self.callback(minutes_closed, latest_candle_df)
                
            except Exception as e:
                logging.error(f"Scheduler error: {e}")
                time.sleep(60)

# ========================
# TRADING DETECTOR
# ========================
class TradingDetector:
    def __init__(self, model_path='/home/runner/work/surgeon-/surgeon-/ml_models', scaler_path='/home/runner/work/surgeon-/surgeon-/ml_models/scaler_oversample.joblib'):
        self.data = pd.DataFrame()
        self.feature_engineer = FeatureEngineer(history_size=200)  # 200 candles (50 hours) for features
        self.models = []
        self.scaler = None
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.scheduler = CandleScheduler(timeframe=15)
        self.pending_signals = deque(maxlen=100)
        self.load_resources()
        
        # Fetch initial 200 candles upfront
        self.data = self.fetch_initial_candles()
        self.scheduler.register_callback(self.process_pending_signals)
        self.scheduler.start()

    def fetch_initial_candles(self):
        import oandapyV20.endpoints.instruments as instruments
        from oandapyV20 import API
        import pandas as pd
        api = API(environment="practice", access_token=os.getenv("OANDA_API_KEY"))
        params = {
            "count": 200,
            "granularity": "M15",
            "price": "M"
        }
        r = instruments.InstrumentsCandles(instrument="XAU/USD", params=params)
        api.request(r)
        candles = r.response.get("candles", [])
        if not candles:
            raise ValueError("Failed to fetch initial candles")
        df = pd.DataFrame([{
            "time": c["time"],
            "open": float(c["mid"]["o"]),
            "high": float(c["mid"]["h"]),
            "low": float(c["mid"]["l"]),
            "close": float(c["mid"]["c"])
        } for c in candles])
        df["time"] = pd.to_datetime(df["time"])
        return df

    def process_pending_signals(self, minutes_closed, latest_candles):
        import time
        global CRT_SIGNAL_COUNT, LAST_SIGNAL_TIME, SIGNALS, GLOBAL_LOCK
        
        # Update data with latest candles
        if not latest_candles.empty:
            self.data = pd.concat([self.data, latest_candles]).drop_duplicates(subset=["time"]).sort_values("time").tail(200)
        
        if self.data.empty or len(self.data) < self.feature_engineer.history_size:
            logging.warning(f"Insufficient data: {len(self.data)} rows")
            return
        
        start_time = time.time()
        df_history = self.data.tail(self.feature_engineer.history_size)
        for signal in list(self.pending_signals):
            features = self.feature_engineer.transform(df_history, signal['signal'], minutes_closed)
            if features is None:
                logging.warning(f"Feature extraction failed for signal: {signal['signal']}")
                self.pending_signals.remove(signal)
                continue
                
            val_start = time.time()
            signal_candle_time = self.data.iloc[-1]['time'].strftime('%Y-%m-%d %H:%M:%S')  # Add signal candle time
            validation_result = self.validate(features)
            print(f"Feature generation took {(val_start - start_time):.2f}s")
            print(f"Validation took {(time.time() - val_start):.2f}s")
            print(f"Model validation outcome: {int(validation_result)}")  # Add 1 or 0 outcome
            print(f"Signal candle time: {signal_candle_time}")  # Add signal candle time
            
            if validation_result:
                with GLOBAL_LOCK:
                    CRT_SIGNAL_COUNT += 1
                    LAST_SIGNAL_TIME = time.time()
                    SIGNALS.append({
                        "time": time.time(),
                        "pair": "XAU_USD",
                        "timeframe": "M15",
                        "signal": signal['signal'],
                        "outcome": "pending",
                        "rrr": None
                    })
                
                alert_time = signal['time'].astimezone(NY_TZ)
                send_telegram(
                    f"🚀 *VALIDATED CRT* XAU/USD {signal['signal']}\n"
                    f"Timeframe: M15\n"
                    f"Time: {alert_time.strftime('%Y-%m-%d %H:%M')} NY\n"
                    f"RSI Zone: {signal['rsi_zone']}\n"
                    f"Confidence: High"
                )
                logging.info(f"Alert triggered for signal: {signal['signal']}")
            
            self.pending_signals.remove(signal)

    def load_resources(self):
        try:
            self.scaler = joblib.load(os.path.join(self.model_path, self.scaler_path))
            model_files = [
                'model_f1_0.0000_20250719_090727.keras',
                'model_f1_0.0000_20250719_092134.keras',
                'model_f1_0.0000_20250719_093712.keras',
                'model_f1_0.0000_20250719_095056.keras',
                'model_f1_0.0000_20250719_100411.keras',
                'model_f1_0.0000_20250719_102457.keras',
                'model_f1_0.0000_20250719_104011.keras',
                'model_f1_0.0000_20250719_110914.keras'
            ]
            for model_file in model_files:
                model = load_model(os.path.join(self.model_path, model_file))
                self.models.append(model)
            logging.info("ML models and scaler loaded successfully")
        except Exception as e:
            logging.error(f"Model loading failed: {e}")
            self.models = []
            self.scaler = None

    def validate(self, features):
        try:
            if isinstance(features, pd.Series):
                features = features.values.reshape(1, -1)
            
            scaled = self.scaler.transform(features)
            reshaped = scaled.reshape(scaled.shape[0], 1, scaled.shape[1])
            
            predictions = []
            for model in self.models:
                pred = model.predict(reshaped, verbose=0).flatten()
                predictions.append(pred)
            
            avg_prob = np.mean(predictions, axis=0)[0]
            return avg_prob >= 0.55
        except Exception as e:
            logging.error(f"Validation error: {e}")
            return False

    def update_data(self, df_new):
        if df_new.empty:
            logging.warning("Received empty dataframe in update_data")
            return
        
        try:
            if self.data.empty:
                self.data = df_new
            else:
                df_combined = pd.concat([self.data, df_new])
                df_combined = df_combined.drop_duplicates(subset=['time'], keep='last')
                self.data = df_combined.sort_values('time').reset_index(drop=True)
            
            if len(self.data) >= self.feature_engineer.history_size:
                self.check_signals()
        except Exception as e:
            logging.error(f"Error in update_data: {e}")

    def check_signals(self):
        df_history = self.data.tail(self.feature_engineer.history_size)
        if len(df_history) < self.feature_engineer.history_size:
            logging.warning(f"Insufficient history data: {len(df_history)} rows")
            return
        
        df_history = self.feature_engineer.calculate_crt_vectorized(df_history)
        last_row = df_history.iloc[-1]
        
        if last_row['crt'] in ['BUY', 'SELL']:
            # Calculate RSI zone for the signal
            rsi = last_row['rsi'] if 'rsi' in last_row else ta.rsi(df_history['close'], length=14).iloc[-1]
            rsi_zone = (
                'unknown' if pd.isna(rsi) else
                'oversold' if rsi < 30 else
                'overbought' if rsi > 70 else
                'neutral'
            )
            
            signal_info = {
                'signal': last_row['crt'],
                'time': last_row['time'],
                'rsi_zone': rsi_zone
            }
            self.pending_signals.append(signal_info)
            logging.info(f"Signal queued for validation: {signal_info['signal']} at {last_row['time'].strftime('%Y-%m-%d %H:%M:%S')}")

# ========================
# FLASK UI ROUTES
# ========================
@app.route('/')
def home():
    return render_template('dashboard.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/journal')
def journal():
    return render_template('journal.html')

@app.route('/metrics')
def metrics():
    return jsonify(calculate_performance_metrics())

@app.route('/signals')
def signals():
    with GLOBAL_LOCK:
        return jsonify(list(SIGNALS)[-20:])

@app.route('/journal/entries')
def journal_entries():
    with GLOBAL_LOCK:
        return jsonify(list(TRADE_JOURNAL))

@app.route('/journal/add', methods=['POST'])
def add_entry():
    data = request.json
    add_journal_entry(
        data.get('type', 'note'),
        data.get('content', ''),
        data.get('image', None)
    )
    return jsonify({"status": "success"})

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'time': datetime.now(NY_TZ).isoformat()})

# ========================
# SUPPORT FUNCTIONS FOR UI
# ========================
def calculate_performance_metrics():
    global PERF_CACHE
    
    if time.time() - PERF_CACHE["updated"] < 300 and PERF_CACHE["data"]:
        return PERF_CACHE["data"]
    
    with GLOBAL_LOCK:
        recent_signals = list(SIGNALS)[-100:]
        
        if not recent_signals:
            return {
                "win_rate": 0,
                "avg_rrr": 0,
                "hourly_dist": {},
                "asset_dist": {}
            }
        
        wins = sum(1 for s in recent_signals if s.get('outcome') == 'win')
        win_rate = round((wins / len(recent_signals)) * 100, 1) if recent_signals else 0
        
        rrr_values = [s.get('rrr', 0) for s in recent_signals if s.get('rrr') is not None]
        avg_rrr = round(np.mean(rrr_values), 2) if rrr_values else 0
        
        hourly_dist = {}
        for signal in recent_signals:
            hour = datetime.fromtimestamp(signal['time']).hour
            hourly_dist[hour] = hourly_dist.get(hour, 0) + 1
        
        asset_dist = {}
        for signal in recent_signals:
            pair = signal['pair'].split('_')[0]
            asset_dist[pair] = asset_dist.get(pair, 0) + 1
        
        metrics = {
            "win_rate": win_rate,
            "avg_rrr": avg_rrr,
            "hourly_dist": hourly_dist,
            "asset_dist": asset_dist
        }
        
        PERF_CACHE = {"updated": time.time(), "data": metrics}
        return metrics

def add_journal_entry(entry_type, content, image_url=None):
    with GLOBAL_LOCK:
        TRADE_JOURNAL.append({
            "timestamp": time.time(),
            "type": entry_type,
            "content": content,
            "image": image_url
        })

# ========================
# MAIN BOT OPERATION
# ========================
def run_bot():
    send_telegram(f"🚀 *Bot Started*\nInstrument: XAU/USD\nTimeframe: M15\nTime: {datetime.now(NY_TZ)}")
    
    detector = TradingDetector(model_path='/home/runner/work/surgeon-/surgeon-/ml_models', scaler_path='/home/runner/work/surgeon-/surgeon-/ml_models/scaler_oversample.joblib')
    refresh_interval = 300  # 5 minutes
    
    while True:
        try:
            df = fetch_candles()
            if not df.empty:
                detector.update_data(df)
            time.sleep(refresh_interval)
        except Exception as e:
            logging.error(f"Main loop error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()
    app.run(host='0.0.0.0', port=5000)
