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
import traceback
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
MODEL_PATH = os.getenv("MODEL_PATH", "/home/runner/work/surgeon-/surgeon-/ml_models")
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
    'minutes_closed_0', 'minutes_closed_15', 'minutes_closed_30', 'minutes_closed_45'
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
logger = logging.getLogger(__name__)

# Initialize Oanda API
oanda_api = API(access_token=API_KEY, environment="practice")

# Initialize Flask app
app = Flask(__name__)

# ========================
# UTILITY FUNCTIONS
# ========================
def parse_oanda_time(time_str):
    """Parse Oanda's timestamp with variable fractional seconds"""
    logger.debug(f"Parsing Oanda timestamp: {time_str}")
    try:
        if '.' in time_str and len(time_str.split('.')[1]) > 7:
            time_str = re.sub(r'\.(\d{6})\d+', r'.\1', time_str)
        result = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=pytz.utc).astimezone(NY_TZ)
        logger.debug(f"Parsed time: {result}")
        return result
    except Exception as e:
        logger.error(f"Error parsing time {time_str}: {str(e)}")
        return datetime.now(NY_TZ)

def send_telegram(message):
    """Send formatted message to Telegram with detailed error handling"""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram credentials not set, skipping message")
        return False
        
    logger.info(f"Attempting to send Telegram message: {message}")
    if len(message) > 4000:
        message = message[:4000] + "... [TRUNCATED]"
    
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        response = requests.post(url, json={
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': 'Markdown'
        }, timeout=10)
        
        logger.info(f"Telegram response: {response.status_code} - {response.text}")
        
        if response.status_code != 200:
            logger.error(f"Telegram error: {response.status_code} - {response.text}")
            return False
            
        if not response.json().get('ok'):
            logger.error(f"Telegram API error: {response.json()}")
            return False
            
        logger.info("Telegram message sent successfully")
        return True
    except Exception as e:
        logger.error(f"Telegram connection failed: {str(e)}")
        return False

def fetch_candles():
    """Fetch candles for XAU_USD M15 with robust error handling"""
    logger.info(f"Fetching candles for {INSTRUMENT} with timeframe {TIMEFRAME}")
    params = {
        "granularity": TIMEFRAME,
        "count": 201,
        "price": "M"
    }
    
    sleep_time = 10
    max_attempts = 3
    
    for attempt in range(max_attempts):
        try:
            request = instruments.InstrumentsCandles(instrument=INSTRUMENT, params=params)
            logger.debug(f"API request: {request}")
            response = oanda_api.request(request)
            logger.debug(f"API response: {response}")
            candles = response.get('candles', [])
            
            if not candles:
                logger.warning("No candles received from Oanda")
                continue
            
            data = []
            for candle in candles:
                price_data = candle.get('mid', {})
                if not price_data:
                    logger.warning(f"Skipping candle with missing mid price data: {candle}")
                    continue
                
                try:
                    parsed_time = parse_oanda_time(candle['time'])
                    data.append({
                        'time': parsed_time,
                        'open': float(price_data['o']),
                        'high': float(price_data['h']),
                        'low': float(price_data['l']),
                        'close': float(price_data['c']),
                        'volume': int(candle.get('volume', 0)),
                        'complete': candle.get('complete', False)
                    })
                except Exception as e:
                    logger.error(f"Error processing candle: {str(e)}")
            
            if not data:
                logger.warning("No valid candles found in response")
                continue
                
            df = pd.DataFrame(data)
            logger.info(f"Successfully fetched {len(df)} candles")
            return df
            
        except V20Error as e:
            if "rate" in str(e).lower():
                logger.warning(f"Rate limit hit, sleeping {sleep_time}s (attempt {attempt+1}/{max_attempts})")
                time.sleep(sleep_time)
                sleep_time *= 2
            else:
                logger.error(f"Oanda API error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error fetching candles: {str(e)}")
            logger.error(traceback.format_exc())
    
    logger.error(f"Failed to fetch candles after {max_attempts} attempts")
    return pd.DataFrame()

# ========================
# FEATURE ENGINEER
# ========================
class FeatureEngineer:
    def __init__(self, history_size=200):
        self.history_size = history_size
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
            ("SELL_(60, 70]_(-0.138, 0.134]", "dead"),
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
        logger.info("Calculating technical indicators")
        df = df.copy()
        
        df['adj close'] = df['open']
        logger.debug("Adjusted close calculated")
        
        df['garman_klass_vol'] = (
            ((np.log(df['high']) - np.log(df['low'])) ** 2) / 2 -
            (2 * np.log(2) - 1) * ((np.log(df['adj close']) - np.log(df['open'])) ** 2)
        )
        logger.debug("Garman-Klass volatility calculated")
        
        df['rsi_20'] = ta.rsi(df['adj close'], length=20)
        df['rsi'] = ta.rsi(df['close'], length=14)
        logger.debug("RSI calculated")
        
        bb = ta.bbands(np.log1p(df['adj close']), length=20)
        df['bb_low'] = bb['BBL_20_2.0']
        df['bb_mid'] = bb['BBM_20_2.0']
        df['bb_high'] = bb['BBU_20_2.0']
        logger.debug("Bollinger Bands calculated")
        
        atr = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['atr_z'] = (atr - atr.mean()) / atr.std()
        logger.debug("ATR z-score calculated")
        
        macd = ta.macd(df['adj close'], fast=12, slow=26, signal=9)
        df['macd_z'] = (macd['MACD_12_26_9'] - macd['MACD_12_26_9'].mean()) / macd['MACD_12_26_9'].std()
        logger.debug("MACD z-score calculated")
        
        df['dollar_volume'] = (df['adj close'] * df['volume']) / 1e6
        logger.debug("Dollar volume calculated")
        
        df['ma_10'] = df['adj close'].rolling(window=10).mean()
        df['ma_100'] = df['adj close'].rolling(window=100).mean()
        df['ma_20'] = df['close'].rolling(window=20).mean()
        df['ma_30'] = df['close'].rolling(window=30).mean()
        df['ma_40'] = df['close'].rolling(window=40).mean()
        df['ma_60'] = df['close'].rolling(window=60).mean()
        logger.debug("Moving averages calculated")
        
        vwap_num = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum()
        vwap_den = df['volume'].cumsum()
        df['vwap'] = vwap_num / vwap_den
        df['vwap_std'] = df['vwap'].rolling(window=20).std()
        logger.debug("VWAP and VWAP STD calculated")
        
        return df

    def calculate_crt_vectorized(self, df):
        """Vectorized implementation of CRT signal calculation"""
        logger.info("Calculating CRT signals")
        df = df.copy()

        # Initialize crt column
        df['crt'] = None

        # Create shifted columns for previous candles
        df['c1_low'] = df['low'].shift(2)
        df['c1_high'] = df['high'].shift(2)
        df['c2_low'] = df['low'].shift(1)
        df['c2_high'] = df['high'].shift(1)
        df['c2_close'] = df['close'].shift(1)

        # Calculate candle metrics
        df['c2_range'] = df['c2_high'] - df['c2_low']
        df['c2_mid'] = df['c2_low'] + (0.5 * df['c2_range'])

        # Vectorized conditions
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

        # Apply masks
        df.loc[buy_mask, 'crt'] = 'BUY'
        df.loc[sell_mask, 'crt'] = 'SELL'

        # Cleanup intermediate columns
        df.drop(columns=['c1_low', 'c1_high', 'c2_low', 'c2_high',
                        'c2_close', 'c2_range', 'c2_mid'], inplace=True)

        logger.debug(f"CRT calculation complete, last row crt: {df['crt'].iloc[-1]}")
        return df

    def calculate_trade_features(self, df, signal_type):
        logger.info(f"Calculating trade features for signal type: {signal_type}")
        df = df.copy()
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2] if len(df) > 1 else last_row
        logger.debug(f"Last row: {last_row}, Prev row: {prev_row}")
        
        entry = last_row['open']
        if signal_type == 'SELL':
            df['sl_price'] = prev_row['high']
            risk = abs(entry - df['sl_price'])
            df['tp_price'] = entry - 4 * risk
        else:  # BUY
            df['sl_price'] = prev_row['low']
            risk = abs(entry - df['sl_price'])
            df['tp_price'] = entry + 4 * risk
        logger.debug(f"SL: {df['sl_price'].iloc[-1]}, TP: {df['tp_price'].iloc[-1]}")
        
        df['sl_distance'] = abs(entry - df['sl_price']) * 10
        df['tp_distance'] = abs(df['tp_price'] - entry) * 10
        df['rrr'] = df['tp_distance'] / df['sl_distance'].replace(0, np.nan)
        df['log_sl'] = np.log1p(df['sl_price'])
        logger.debug(f"SL Distance: {df['sl_distance'].iloc[-1]}, TP Distance: {df['tp_distance'].iloc[-1]}, RRR: {df['rrr'].iloc[-1]}")
        
        return df

    def calculate_categorical_features(self, df):
        logger.info("Calculating categorical features")
        df = df.copy()
        
        df['day'] = df['time'].dt.day_name()
        df = pd.get_dummies(df, columns=['day'], prefix='day', drop_first=False)
        logger.debug("Day of week dummies created")
        
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
        logger.debug("Session dummies created")
        
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
        logger.debug("RSI zone dummies created")
        
        def is_bullish_stack(row):
            return int(row['ma_20'] > row['ma_30'] > row['ma_40'] > row['ma_60'])
        def is_bearish_stack(row):
            return int(row['ma_20'] < row['ma_30'] < row['ma_40'] < row['ma_60'])
        
        df['trend_strength_up'] = df.apply(is_bullish_stack, axis=1).astype(float)
        df['trend_strength_down'] = df.apply(is_bearish_stack, axis=1).astype(float)
        logger.debug("Trend strength calculated")
        
        def get_trend(row):
            if row['trend_strength_up'] > row['trend_strength_down']:
                return 'uptrend'
            elif row['trend_strength_down'] > row['trend_strength_up']:
                return 'downtrend'
            else:
                return 'sideways'
        df['trend_direction'] = df.apply(get_trend, axis=1)
        df = pd.get_dummies(df, columns=['trend_direction'], prefix='trend_direction', drop_first=False)
        logger.debug("Trend direction dummies created")
        
        return df

    def calculate_combo_flags(self, df, signal_type):
        logger.info(f"Calculating combo flags for signal type: {signal_type}")
        df = df.copy()
        
        df['rsi_bin'] = pd.cut(df['rsi'], bins=[0, 20, 30, 40, 50, 60, 70, 80, 100])
        df['macd_z_bin'] = pd.qcut(df['macd_z'], q=5, duplicates='drop', labels=[
            '(-12.386, -0.496]', '(-0.496, -0.138]', '(-0.138, 0.134]', '(0.134, 0.527]', '(0.527, 9.246]'
        ])
        logger.debug("RSI and MACD bins calculated")
        
        df['trade_type'] = signal_type
        df['combo_key'] = df['trade_type'] + '_' + df['trend_direction'] + '_' + df['rsi_bin'].astype(str)
        df['combo_key2'] = df[['trade_type', 'rsi_bin', 'macd_z_bin']].astype(str).agg('_'.join, axis=1)
        logger.debug("Combo keys generated")
        
        combo_flag_dict = dict(self.combo_flags)
        combo_flag2_dict = dict(self.combo_flags2)
        logger.debug("Combo flag dictionaries created")
        
        df['combo_flag'] = df['combo_key'].map(combo_flag_dict).fillna('dead')
        df['combo_flag2'] = df['combo_key2'].map(combo_flag2_dict).fillna('dead')
        logger.debug("Mapped combo flags")
        
        df['is_bad_combo'] = ((df['combo_flag'] == 'dead') | (df['combo_flag2'] == 'dead')).astype(int)
        logger.debug("is_bad_combo calculated")
        
        df = pd.get_dummies(df, columns=['combo_flag'], prefix='combo_flag', drop_first=False)
        df = pd.get_dummies(df, columns=['combo_flag2'], prefix='combo_flag2', drop_first=False)
        logger.debug("Combo flag dummies created")
        
        return df

    def calculate_minutes_closed(self, df, minutes_closed):
        logger.info(f"Calculating minutes closed: {minutes_closed}")
        df = df.copy()
        minute_cols = ['minutes_closed_0', 'minutes_closed_15', 'minutes_closed_30', 'minutes_closed_45']
        
        # Clear all columns first
        for col in minute_cols:
            df[col] = 0
            
        # Set the appropriate column
        if 0 <= minutes_closed < 15:
            df['minutes_closed_15'] = 1
        elif 15 <= minutes_closed < 30:
            df['minutes_closed_30'] = 1
        elif 30 <= minutes_closed < 45:
            df['minutes_closed_45'] = 1
        else:
            df['minutes_closed_0'] = 1
            
        logger.debug(f"Minutes closed set: {dict(zip(minute_cols, df[minute_cols].iloc[0].tolist()))}")
        return df

    def transform(self, df_history, signal_type, minutes_closed):
        logger.info(f"Transforming data for signal type: {signal_type}, minutes closed: {minutes_closed}")
        try:
            if df_history is None or len(df_history) < self.history_size:
                logger.warning(f"Insufficient data: {len(df_history) if df_history is not None else 0} rows, need {self.history_size}")
                return None
            
            df = df_history.copy()
            logger.debug(f"Data copy created with {len(df)} rows")
            
            if df['time'].dt.tz is not None:
                df['time'] = df['time'].dt.tz_convert('America/New_York')
            else:
                df['time'] = pd.to_datetime(df['time']).dt.tz_localize('UTC').dt.tz_convert('America/New_York')
            logger.debug("Time converted to NY timezone")
            
            df = self.calculate_technical_indicators(df)
            logger.debug("Technical indicators calculated")
            
            df = self.calculate_trade_features(df, signal_type)
            logger.debug("Trade features calculated")
            
            df = self.calculate_categorical_features(df)
            logger.debug("Categorical features calculated")
            
            df = self.calculate_combo_flags(df, signal_type)
            logger.debug("Combo flags calculated")
            
            df = self.calculate_minutes_closed(df, minutes_closed)
            logger.debug("Minutes closed calculated")
            
            df['prev_volume'] = df['volume'].shift(1)
            df['body_size'] = abs(df['close'] - df['open'])
            df['wick_up'] = df['high'] - df[['close', 'open']].max(axis=1)
            df['wick_down'] = df[['close', 'open']].min(axis=1) - df['low']
            df['prev_body_size'] = df['body_size'].shift(1)
            df['prev_wick_up'] = df['wick_up'].shift(1)
            df['prev_wick_down'] = df['wick_down'].shift(1)
            logger.debug("Candle and volume features calculated")
            
            df['price_div_vol'] = df['adj close'] / (df['garman_klass_vol'] + 1e-6)
            df['rsi_div_macd'] = df['rsi'] / (df['macd_z'] + 1e-6)
            df['price_div_vwap'] = df['adj close'] / (df['vwap'] + 1e-6)
            df['sl_div_atr'] = df['sl_distance'] / (df['atr_z'] + 1e-6)
            df['tp_div_atr'] = df['tp_distance'] / (df['atr_z'] + 1e-6)
            df['rrr_div_rsi'] = df['rrr'] / (df['rsi'] + 1e-6)
            logger.debug("Derived metrics calculated")
            
            df['crt_BUY'] = (signal_type == 'BUY').astype(int)
            df['crt_SELL'] = (signal_type == 'SELL').astype(int)
            df['trade_type_BUY'] = (signal_type == 'BUY').astype(int)
            df['trade_type_SELL'] = (signal_type == 'SELL').astype(int)
            logger.debug("CRT and trade type encoding applied")
            
            features = df.iloc[-1][FEATURES].astype(float)
            logger.debug(f"Selected features: {features}")
            
            if features.isna().any():
                missing = features[features.isna()].index.tolist()
                logger.warning(f"Missing features: {missing}")
                # Fill with mean for missing values
                for col in missing:
                    if col in df.columns:
                        features[col] = df[col].mean()
                        logger.warning(f"Filled missing feature {col} with mean value")
            
            logger.info("Feature transformation completed successfully")
            return features
        except Exception as e:
            logger.error(f"Feature engineering error: {str(e)}")
            logger.error(traceback.format_exc())
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
        if latest_time is None:
            return 0
            
        logger.info(f"Calculating minutes closed for latest time: {latest_time}")
        now = datetime.now(NY_TZ)
        elapsed = (now - latest_time).total_seconds() / 60
        logger.debug(f"Elapsed time since last candle: {elapsed:.2f} minutes")
        
        # Return actual minutes closed instead of fixed values
        return min(44, max(0, int(elapsed)))  # Cap at 44 minutes
    
    def run(self):
        logger.info("Starting CandleScheduler thread")
        while self.active:
            try:
                logger.info("Fetching latest candle data")
                df_candles = fetch_candles()
                
                if df_candles.empty:
                    logger.warning("No candles fetched")
                    time.sleep(60)
                    continue
                
                latest_candle = df_candles.iloc[-1]
                latest_time = latest_candle['time']
                minutes_closed = self.calculate_minutes_closed(latest_time)
                
                if self.callback:
                    logger.info(f"Calling callback with minutes closed: {minutes_closed}")
                    self.callback(minutes_closed, df_candles.tail(1))
                
                # Sleep until next 15-minute interval
                now = datetime.now(NY_TZ)
                next_run = now + timedelta(minutes=15 - (now.minute % 15), seconds=0, microseconds=0)
                sleep_seconds = (next_run - now).total_seconds()
                logger.info(f"Sleeping {sleep_seconds:.1f} seconds until next candle")
                time.sleep(max(1, sleep_seconds))
                
            except Exception as e:
                logger.error(f"Scheduler error: {str(e)}")
                logger.error(traceback.format_exc())
                time.sleep(60)

# ========================
# TRADING DETECTOR
# ========================
class TradingDetector:
    def __init__(self, model_path='/home/runner/work/surgeon-/surgeon-/ml_models', scaler_path='/home/runner/work/surgeon-/surgeon-/ml_models/scaler_oversample.joblib'):
        logger.info("Initializing TradingDetector")
        self.data = pd.DataFrame()
        self.feature_engineer = FeatureEngineer(history_size=200)
        self.models = []
        self.scaler = None
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.scheduler = CandleScheduler(timeframe=15)
        self.pending_signals = deque(maxlen=100)
        
        # Validate paths
        if not os.path.exists(model_path):
            logger.error(f"Model path does not exist: {model_path}")
            raise FileNotFoundError(f"Model path not found: {model_path}")
            
        self.load_resources()
        logger.info("Loading initial candles")
        self.data = self.fetch_initial_candles()
        
        if self.data.empty:
            logger.error("Failed to load initial candles")
            raise RuntimeError("Initial candle fetch failed")
            
        logger.info(f"Initial data loaded with {len(self.data)} rows")
        self.scheduler.register_callback(self.process_pending_signals)
        logger.info("Starting scheduler thread")
        self.scheduler.start()
        logger.info("TradingDetector initialized")

    def fetch_initial_candles(self):
        logger.info("Fetching initial 201 candles")
        for attempt in range(5):  # Increased attempts
            df = fetch_candles()
            if not df.empty and len(df) >= 200:
                logger.info(f"Successfully fetched {len(df)} initial candles")
                return df
            logger.warning(f"Attempt {attempt+1} failed, retrying in 10s")
            time.sleep(10)
        logger.error("Failed to fetch initial candles after 5 attempts")
        return pd.DataFrame()

    def process_pending_signals(self, minutes_closed, latest_candles):
        logger.info(f"Processing pending signals, minutes closed: {minutes_closed}")
        global CRT_SIGNAL_COUNT, LAST_SIGNAL_TIME, SIGNALS, GLOBAL_LOCK
        
        if not latest_candles.empty:
            logger.info(f"Updating data with {len(latest_candles)} new candles")
            # Convert to timestamp for comparison
            latest_time = latest_candles.iloc[0]['time']
            if not self.data.empty and latest_time > self.data['time'].max():
                self.data = pd.concat([self.data, latest_candles]).drop_duplicates(subset=["time"], keep="last").sort_values("time").tail(200)
                logger.debug(f"Updated data shape: {self.data.shape}, latest time: {self.data['time'].max()}")
            else:
                logger.info("No new data to add, checking latest candle")
                # Force update with latest candle even if duplicate time
                self.data = pd.concat([self.data, latest_candles]).sort_values("time").tail(200)
                logger.debug(f"Forced update, new shape: {self.data.shape}")
        else:
            logger.warning("No new candles to update")

        if self.data.empty or len(self.data) < self.feature_engineer.history_size:
            logger.warning(f"Insufficient data: {len(self.data)} rows, need {self.feature_engineer.history_size}")
            return
        
        start_time = time.time()
        logger.debug(f"Starting signal processing at {start_time}")
        df_history = self.data.tail(self.feature_engineer.history_size)
        logger.debug(f"Using history of {len(df_history)} rows")
        
        for signal in list(self.pending_signals):
            logger.info(f"Processing signal: {signal['signal']}")
            features = self.feature_engineer.transform(df_history, signal['signal'], minutes_closed)
            if features is None:
                logger.warning(f"Feature extraction failed for signal: {signal['signal']}")
                self.pending_signals.remove(signal)
                continue
                
            val_start = time.time()
            signal_candle_time = self.data.iloc[-1]['time'].strftime('%Y-%m-%d %H:%M:%S')
            logger.debug(f"Signal candle time: {signal_candle_time}")
            validation_result = self.validate(features)
            logger.info(f"Feature generation took {(val_start - start_time):.2f}s")
            logger.info(f"Validation took {(time.time() - val_start):.2f}s")
            logger.info(f"Model validation outcome: {int(validation_result)}")
            logger.info(f"Signal candle time: {signal_candle_time}")
            
            if validation_result:
                with GLOBAL_LOCK:
                    logger.info("Acquiring lock for signal count update")
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
                    logger.info(f"Signal count updated to {CRT_SIGNAL_COUNT}")
                
                alert_time = signal['time'].astimezone(NY_TZ)
                send_telegram(
                    f"🚀 *VALIDATED CRT* XAU/USD {signal['signal']}\n"
                    f"Timeframe: M15\n"
                    f"Time: {alert_time.strftime('%Y-%m-%d %H:%M')} NY"
                )
                logger.info(f"Alert triggered for signal: {signal['signal']} at {signal_candle_time}")
            
            self.pending_signals.remove(signal)
            logger.debug(f"Removed processed signal: {signal['signal']}")

    def load_resources(self):
        logger.info("Loading ML models and scaler")
        try:
            # Validate scaler path
            if not os.path.exists(self.scaler_path):
                raise FileNotFoundError(f"Scaler file not found: {self.scaler_path}")
                
            self.scaler = joblib.load(self.scaler_path)
            
            # Load models
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
            
            self.models = []
            for model_file in model_files:
                model_path = os.path.join(self.model_path, model_file)
                if not os.path.exists(model_path):
                    logger.error(f"Model file not found: {model_path}")
                    continue
                    
                model = load_model(model_path)
                self.models.append(model)
                logger.info(f"Loaded model: {model_file}")
                
            if not self.models:
                raise RuntimeError("No models loaded successfully")
                
            logger.info(f"Loaded {len(self.models)} models and scaler")
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def validate(self, features):
        logger.info("Validating features")
        try:
            if isinstance(features, pd.Series):
                features = features.values.reshape(1, -1)
                logger.debug("Reshaped features from Series")
            
            scaled = self.scaler.transform(features)
            logger.debug("Features scaled")
            reshaped = scaled.reshape(scaled.shape[0], 1, scaled.shape[1])
            logger.debug("Features reshaped for model")
            
            predictions = []
            for model in self.models:
                pred = model.predict(reshaped, verbose=0).flatten()
                predictions.append(pred)
                logger.debug(f"Model prediction: {pred}")
            
            avg_prob = np.mean(predictions, axis=0)[0]
            logger.debug(f"Average prediction probability: {avg_prob}")
            return avg_prob >= 0.5  # Adjusted threshold to 0.5 for binary validation
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def update_data(self, df_new):
        logger.info(f"Updating data with new dataframe of size {len(df_new)}")
        if df_new.empty:
            logger.warning("Received empty dataframe in update_data")
            return
        
        try:
            if self.data.empty:
                self.data = df_new.dropna(subset=['time', 'open', 'high', 'low', 'close'])
                logger.debug("Initialized data with new dataframe")
            else:
                # Only add new data if it's newer than existing data, or force latest
                last_existing_time = self.data['time'].max()
                new_data = df_new[df_new['time'] > last_existing_time]
                
                if not new_data.empty:
                    df_combined = pd.concat([self.data, new_data]).drop_duplicates(subset=['time'], keep='last')
                    self.data = df_combined.sort_values('time').reset_index(drop=True).tail(200)
                    logger.debug(f"Combined data shape: {self.data.shape}, latest time: {self.data['time'].max()}")
                else:
                    # Force update with the latest candle
                    latest_new = df_new.iloc[-1]
                    if latest_new['time'] >= self.data['time'].max():
                        self.data = pd.concat([self.data, df_new.tail(1)]).drop_duplicates(subset=['time'], keep='last')
                        self.data = self.data.sort_values('time').reset_index(drop=True).tail(200)
                        logger.debug(f"Forced update with latest candle, new shape: {self.data.shape}, latest time: {self.data['time'].max()}")
                    else:
                        logger.info("No new data to add, data unchanged")
            
            if len(self.data) >= self.feature_engineer.history_size:
                logger.info("Data sufficient, checking signals")
                self.check_signals()
        except Exception as e:
            logger.error(f"Error in update_data: {e}, Dataframe shape: {self.data.shape if 'self.data' in locals() else 'N/A'}")

    def check_signals(self):
        logger.info("Checking for signals")
        if len(self.data) < 3:  # Need at least 3 candles (C1, C2, C3)
            logger.warning(f"Insufficient data: {len(self.data)} rows, need at least 3")
            return
        
        # Use only the last 3 candles for CRT calculation
        df_last_three = self.data.tail(3)
        df_history = self.feature_engineer.calculate_crt_vectorized(df_last_three)
        logger.debug(f"CRT vectorized on last 3 candles, last row crt: {df_history['crt'].iloc[-1]}")
        last_row = df_history.iloc[-1]
        
        if last_row['crt'] in ['BUY', 'SELL']:
            logger.info(f"Detected signal: {last_row['crt']} on current candle")
            features = self.feature_engineer.transform(df_history, last_row['crt'], 0)  # Use 0 for minutes_closed as placeholder
            if features is not None:
                logger.info("Generating features for validation")
                validation_result = self.validate(features)
                logger.info(f"Model validation result: {validation_result}")
                if validation_result:
                    signal_info = {
                        'signal': last_row['crt'],
                        'time': last_row['time'],
                    }
                    self.pending_signals.append(signal_info)
                    logger.info(f"Signal queued for validation: {signal_info['signal']} at {last_row['time'].strftime('%Y-%m-%d %H:%M:%S')}")

# ========================
# FLASK UI ROUTES
# ========================
@app.route('/')
def home():
    logger.info("Serving home page")
    return render_template('dashboard.html')

@app.route('/dashboard')
def dashboard():
    logger.info("Serving dashboard page")
    return render_template('dashboard.html')

@app.route('/journal')
def journal():
    logger.info("Serving journal page")
    return render_template('journal.html')

@app.route('/metrics')
def metrics():
    logger.info("Serving metrics")
    return jsonify(calculate_performance_metrics())

@app.route('/signals')
def signals():
    logger.info("Serving signals")
    with GLOBAL_LOCK:
        return jsonify(list(SIGNALS)[-20:])

@app.route('/journal/entries')
def journal_entries():
    logger.info("Serving journal entries")
    with GLOBAL_LOCK:
        return jsonify(list(TRADE_JOURNAL))

@app.route('/journal/add', methods=['POST'])
def add_entry():
    logger.info("Adding journal entry")
    data = request.json
    add_journal_entry(
        data.get('type', 'note'),
        data.get('content', ''),
        data.get('image', None)
    )
    return jsonify({"status": "success"})

@app.route('/health', methods=['GET'])
def health_check():
    logger.info("Serving health check")
    return jsonify({'status': 'healthy', 'time': datetime.now(NY_TZ).isoformat()})

# ========================
# SUPPORT FUNCTIONS FOR UI
# ========================
def calculate_performance_metrics():
    logger.info("Calculating performance metrics")
    global PERF_CACHE
    
    if time.time() - PERF_CACHE["updated"] < 300 and PERF_CACHE["data"]:
        logger.debug("Using cached performance metrics")
        return PERF_CACHE["data"]
    
    with GLOBAL_LOCK:
        recent_signals = list(SIGNALS)[-100:]
        logger.debug(f"Processing {len(recent_signals)} recent signals")
        
        if not recent_signals:
            logger.info("No recent signals for metrics")
            return {
                "win_rate": 0,
                "avg_rrr": 0,
                "hourly_dist": {},
                "asset_dist": {}
            }
        
        wins = sum(1 for s in recent_signals if s.get('outcome') == 'win')
        win_rate = round((wins / len(recent_signals)) * 100, 1) if recent_signals else 0
        logger.debug(f"Wins: {wins}, Win rate: {win_rate}%")
        
        rrr_values = [s.get('rrr', 0) for s in recent_signals if s.get('rrr') is not None]
        avg_rrr = round(np.mean(rrr_values), 2) if rrr_values else 0
        logger.debug(f"Average RRR: {avg_rrr}")
        
        hourly_dist = {}
        for signal in recent_signals:
            hour = datetime.fromtimestamp(signal['time']).hour
            hourly_dist[hour] = hourly_dist.get(hour, 0) + 1
        logger.debug(f"Hourly distribution: {hourly_dist}")
        
        asset_dist = {}
        for signal in recent_signals:
            pair = signal['pair'].split('_')[0]
            asset_dist[pair] = asset_dist.get(pair, 0) + 1
        logger.debug(f"Asset distribution: {asset_dist}")
        
        metrics = {
            "win_rate": win_rate,
            "avg_rrr": avg_rrr,
            "hourly_dist": hourly_dist,
            "asset_dist": asset_dist
        }
        logger.info("Performance metrics calculated")
        PERF_CACHE = {"updated": time.time(), "data": metrics}
        return metrics

def add_journal_entry(entry_type, content, image_url=None):
    logger.info(f"Adding journal entry of type: {entry_type}")
    with GLOBAL_LOCK:
        TRADE_JOURNAL.append({
            "timestamp": time.time(),
            "type": entry_type,
            "content": content,
            "image": image_url
        })
        logger.debug(f"Journal entry added: {content}")

# ========================
# MAIN BOT OPERATION
# ========================
def run_bot():
    logger.info("Starting trading bot")
    send_telegram(f"🚀 *Bot Started*\nInstrument: XAU/USD\nTimeframe: M15\nTime: {datetime.now(NY_TZ)}")
    
    try:
        detector = TradingDetector(
            model_path='/home/runner/work/surgeon-/surgeon-/ml_models',
            scaler_path='/home/runner/work/surgeon-/surgeon-/ml_models/scaler_oversample.joblib'
        )
    except Exception as e:
        logger.error(f"Detector initialization failed: {str(e)}")
        send_telegram(f"❌ *Bot Failed to Start*:\n{str(e)}")
        return
        
    logger.info("Bot started successfully")
    
    while True:
        try:
            logger.info("Running bot cycle")
            df = fetch_candles()
            if not df.empty:
                logger.info(f"Fetched {len(df)} candles, updating data")
                detector.update_data(df)
            else:
                logger.warning("No candles fetched in this cycle")
            time.sleep(60)  # Check every minute
        except Exception as e:
            logger.error(f"Main loop error: {e}")
            logger.error(traceback.format_exc())
            time.sleep(60)

if __name__ == "__main__":
    logger.info("Launching main application")
    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()
    logger.info("Bot thread started")
    
    try:
        app.run(host='0.0.0.0', port=5000, use_reloader=False)  # Disable reloader for thread safety
    except Exception as e:
        logger.error(f"Flask app failed: {str(e)}")
        send_telegram(f"❌ *Flask App Crashed*:\n{str(e)}")
