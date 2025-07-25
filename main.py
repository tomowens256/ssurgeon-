import sys
import os
import time
import threading
import logging
import pytz
import numpy as np
import pandas as pd
import pandas_ta as ta
import requests
import re
import traceback
from datetime import datetime, timedelta
from oandapyV20 import API
from oandapyV20.exceptions import V20Error
import oandapyV20.endpoints.instruments as instruments
import joblib
from tensorflow.keras.models import load_model

# ========================
# CONSTANTS & CONFIG
# ========================
NY_TZ = pytz.timezone("America/New_York")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Oanda configuration
ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
API_KEY = os.getenv("OANDA_API_KEY")

# Only needed instrument and timeframe
INSTRUMENT = "XAU_USD"
TIMEFRAME = "M15"

# Global variables
GLOBAL_LOCK = threading.Lock()
CRT_SIGNAL_COUNT = 0
LAST_SIGNAL_TIME = 0
SIGNALS = []

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

# Load scaler
scaler_path = os.path.join("ml_models", "scaler_oversample.joblib")
scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

# Load models
models_dir = "ml_models"
models = [load_model(os.path.join(models_dir, f)) for f in os.listdir(models_dir) if f.endswith(".keras")]

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
    """Send formatted message to Telegram with detailed error handling and retries"""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram credentials not set, skipping message")
        return False
        
    logger.info(f"Attempting to send Telegram message: {message}")
    if len(message) > 4000:
        message = message[:4000] + "... [TRUNCATED]"
    
    message = message.replace('_', '\_')  # Escape underscores for Markdown
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json={
                'chat_id': TELEGRAM_CHAT_ID,
                'text': message,
                'parse_mode': 'Markdown'
            }, timeout=10)
            
            logger.info(f"Telegram response: {response.status_code} - {response.text}")
            
            if response.status_code == 200 and response.json().get('ok'):
                logger.info("Telegram message sent successfully")
                return True
            else:
                logger.error(f"Telegram error: {response.status_code} - {response.text}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                continue
        except Exception as e:
            logger.error(f"Telegram connection failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            continue
    logger.error(f"Failed to send Telegram message after {max_retries} attempts")
    return False

def fetch_candles(last_time=None):
    """Fetch new candles since last_time or 201 candles for XAU_USD M15"""
    logger.info(f"Fetching candles for {INSTRUMENT} with timeframe {TIMEFRAME}")
    params = {
        "granularity": TIMEFRAME,
        "count": 201,
        "price": "B"  # Bid prices
    }
    if last_time:
        params["from"] = last_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    sleep_time = 10
    max_attempts = 5
    
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
                price_data = candle.get('bid', {})
                if not price_data:
                    logger.warning(f"Skipping candle with missing bid price data: {candle}")
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
                
            df = pd.DataFrame(data).drop_duplicates(subset=['time'], keep='last')
            if last_time:
                df = df[df['time'] > last_time].sort_values('time')
            logger.info(f"Successfully fetched {len(df)} candles")
            return df
            
        except V20Error as e:
            if "rate" in str(e).lower():
                logger.warning(f"Rate limit hit, sleeping {sleep_time}s (attempt {attempt+1}/{max_attempts})")
                time.sleep(sleep_time)
                sleep_time *= 2
            elif e.status == 502:
                wait_time = sleep_time * (2 ** attempt)
                logger.error(f"Oanda API error: 502 Bad Gateway (Attempt {attempt+1}/{max_attempts}), retrying in {wait_time}s")
                time.sleep(wait_time)
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
    def __init__(self):
        self.features = [
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

    def calculate_crt_signal(self, df):
        """Calculate CRT signal on the current candle using the last 3 candles with detailed logging"""
        logger.info("Calculating CRT signals on last 3 candles")
        if len(df) < 3:
            logger.warning(f"Insufficient data: {len(df)} rows, need at least 3")
            return None, None

        c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
        logger.debug(f"Candle data: c1={c1}, c2={c2}, c3={c3}")

        signal_type = None
        entry = c3['open']
        sl = None
        tp = None
        
        # SELL Signal (looser condition: close slightly below high)
        if (c2['high'] > c1['high']) and (c2['close'] < c1['high'] + 0.1):  # Adjusted threshold
            signal_type = 'SELL'
            sl = c2['high']
            risk = abs(entry - sl)
            tp = entry - 4 * risk
            logger.info(f"SELL signal detected: c2 high={c2['high']}, c1 high={c1['high']}, c2 close={c2['close']}")
        
        # BUY Signal
        elif (c2['low'] < c1['low']) and (c2['close'] > c1['low'] - 0.1):  # Adjusted threshold
            signal_type = 'BUY'
            sl = c2['low']
            risk = abs(entry - sl)
            tp = entry + 4 * risk
            logger.info(f"BUY signal detected: c2 low={c2['low']}, c1 low={c1['low']}, c2 close={c2['close']}")
        
        if signal_type:
            logger.info(f"Detected signal: {signal_type} on current candle at {c3['time']}")
        else:
            logger.debug("No signal detected")
        return signal_type, {'entry': entry, 'sl': sl, 'tp': tp, 'time': c3['time']} if signal_type else (None, None)

    def calculate_technical_indicators(self, df):
        logger.info("Calculating technical indicators")
        df = df.copy().drop_duplicates(subset=['time'], keep='last')
        
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

    def calculate_trade_features(self, df, signal_type, entry):
        logger.info(f"Calculating trade features for signal_type: {signal_type}")
        df = df.copy()
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2] if len(df) > 1 else last_row
        logger.debug(f"Last row: {last_row}, Prev row: {prev_row}")
        
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
        all_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Sunday']
        for day in all_days:
            df[f'day_{day}'] = 0
        today = datetime.now(NY_TZ).strftime('%A')
        df[f'day_{today}'] = 1
        logger.debug(f"Day dummies set for today: {today}")
        
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

    def calculate_minutes_closed(self, df, minutes_closed):
        logger.info(f"Calculating minutes closed: {minutes_closed}")
        df = df.copy()
        minute_cols = ['minutes,closed_0', 'minutes,closed_15', 'minutes,closed_30', 'minutes,closed_45']
        
        for col in minute_cols:
            df[col] = 0
            
        if 0 <= minutes_closed < 15:
            df['minutes,closed_15'] = 1
        elif 15 <= minutes_closed < 30:
            df['minutes,closed_30'] = 1
        elif 30 <= minutes_closed < 45:
            df['minutes,closed_45'] = 1
        else:
            df['minutes,closed_0'] = 1
            
        logger.debug(f"Minutes closed set: {dict(zip(minute_cols, df[minute_cols].iloc[0].tolist()))}")
        return df

    def generate_features(self, df, signal_type, minutes_closed):
        logger.info(f"Generating features for signal_type: {signal_type}, minutes closed: {minutes_closed}")
        if len(df) < 200:
            logger.warning(f"Insufficient data: {len(df)} rows, need 200")
            return None
        
        df = df.tail(200).copy()
        df = self.calculate_technical_indicators(df)
        df = self.calculate_trade_features(df, signal_type, df.iloc[-1]['open'])
        df = self.calculate_categorical_features(df)
        df = self.calculate_minutes_closed(df, minutes_closed)
        
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
        
        combo_key = f"{df['rsi'].iloc[-1]:.2f}_{df['macd_z'].iloc[-1]:.2f}_{df['atr_z'].iloc[-1]:.2f}"
        logger.debug(f"Combo key calculated: {combo_key}")
        combo_flags = {'combo_flag_dead': 0, 'combo_flag_fair': 0, 'combo_flag_fine': 0}
        combo_flags2 = {'combo_flag2_dead': 0, 'combo_flag2_fair': 0, 'combo_flag2_fine': 0}
        if df['rsi'].iloc[-1] < 30 or df['macd_z'].iloc[-1] < -1:
            combo_flags['combo_flag_dead'] = 1
            combo_flags2['combo_flag2_dead'] = 1
        elif df['rsi'].iloc[-1] > 70 or df['macd_z'].iloc[-1] > 1:
            combo_flags['combo_flag_fine'] = 1
            combo_flags2['combo_flag2_fine'] = 1
        else:
            combo_flags['combo_flag_fair'] = 1
            combo_flags2['combo_flag2_fair'] = 1
        for flag, value in combo_flags.items():
            df[flag] = value
        for flag, value in combo_flags2.items():
            df[flag] = value
        logger.debug(f"Combo flags set: {combo_flags}, {combo_flags2}")
        
        df['is_bad_combo'] = 1 if combo_flags['combo_flag_dead'] == 1 else 0
        logger.debug(f"is_bad_combo set to: {df['is_bad_combo'].iloc[-1]}")
        
        df['crt_BUY'] = int(signal_type == 'BUY')
        df['crt_SELL'] = int(signal_type == 'SELL')
        df['trade_type_BUY'] = int(signal_type == 'BUY')
        df['trade_type_SELL'] = int(signal_type == 'SELL')
        logger.debug("CRT and trade type encoding applied")
        
        features = pd.Series(index=self.features, dtype=float)
        for feat in self.features:
            if feat in df.columns:
                features[feat] = df[feat].iloc[-1]
            else:
                logger.warning(f"Feature {feat} not found, setting to 0")
                features[feat] = 0
        
        if features.isna().any():
            missing = features[features.isna()].index.tolist()
            logger.warning(f"Missing features: {missing}")
            for col in missing:
                features[col] = 0
                logger.warning(f"Filled missing feature {col} with default value 0")
        
        logger.info("Feature generation completed successfully")
        return features

# ========================
# TRADING DETECTOR
# ========================
class TradingDetector:
    def __init__(self):
        logger.info("Initializing TradingDetector")
        self.data = pd.DataFrame()
        self.feature_engineer = FeatureEngineer()
        self.scheduler = CandleScheduler(timeframe=15)
        self.last_signal_time = None
        self.active_trades = []  # (entry, sl, tp, start_time, index)
        self.last_signal_candle = None
        self.last_scaled_features = None  # Store the last scaled features for consistency check
        
        logger.info("Loading initial 201 candles")
        self.data = self.fetch_initial_candles()
        
        if self.data.empty or len(self.data) < 200:
            logger.error("Failed to load sufficient initial candles")
            raise RuntimeError("Initial candle fetch failed or insufficient data")
            
        logger.info(f"Initial data loaded with {len(self.data)} rows")
        self.scheduler.register_callback(self.process_signals)
        logger.info("Starting scheduler thread")
        self.scheduler.start()
        logger.info("TradingDetector initialized")

    def fetch_initial_candles(self):
        logger.info("Fetching initial 201 candles")
        for attempt in range(5):
            df = fetch_candles()
            if not df.empty and len(df) >= 200:
                logger.info(f"Successfully fetched {len(df)} initial candles")
                return df
            logger.warning(f"Attempt {attempt+1} failed, retrying in 10s")
            time.sleep(10)
        logger.error("Failed to fetch initial 201 candles after 5 attempts")
        return pd.DataFrame()

    def calculate_candle_age(self, current_time, candle_time):
        """Calculate age of the latest candle in minutes"""
        elapsed = (current_time - candle_time).total_seconds() / 60
        return min(15, max(0, elapsed))

    def _get_next_candle_time(self, current_time):
        """Calculate the next 15-minute candle open time"""
        minute = current_time.minute
        remainder = minute % 15
        if remainder == 0:
            return current_time.replace(second=0, microsecond=0) + timedelta(minutes=15)
        next_minute = minute - remainder + 15
        if next_minute >= 60:
            return current_time.replace(hour=current_time.hour + 1, minute=0, second=0, microsecond=0)
        return current_time.replace(minute=next_minute, second=0, microsecond=0)

    def update_data(self, df_new):
        """Update the data with new candles"""
        logger.info(f"Updating data with new dataframe of size {len(df_new)}")
        if df_new.empty:
            logger.warning("Received empty dataframe in update_data")
            return
        
        if self.data.empty:
            self.data = df_new.dropna(subset=['time', 'open', 'high', 'low', 'close']).tail(201)
            logger.debug("Initialized data with new dataframe")
        else:
            last_existing_time = self.data['time'].max()
            new_data = df_new[df_new['time'] > last_existing_time]
            if not new_data.empty:
                self.data = pd.concat([self.data, new_data]).drop_duplicates(subset=['time'], keep="last")
                self.data = self.data.sort_values('time').reset_index(drop=True).tail(201)
                logger.debug(f"Combined data shape: {self.data.shape}, latest time: {self.data['time'].max()}")
            else:
                latest_new = df_new.iloc[-1]
                if latest_new['time'] >= self.data['time'].max():
                    self.data = pd.concat([self.data, df_new.tail(1)]).drop_duplicates(subset=['time'], keep="last")
                    self.data = self.data.sort_values('time').reset_index(drop=True).tail(201)
                    logger.debug(f"Forced update with latest candle, new shape: {self.data.shape}, latest time: {self.data['time'].max()}")

    def process_signals(self, minutes_closed, latest_candles):
        logger.info(f"Processing signals, minutes closed: {minutes_closed}, candles: {len(latest_candles)}")
        if not latest_candles.empty:
            logger.info(f"Updating data with {len(latest_candles)} new candles")
            self.update_data(latest_candles)
            logger.debug(f"Updated data shape: {self.data.shape}, latest time: {self.data['time'].max()}, last 3 rows: {self.data.tail(3)}")
        else:
            logger.warning("No new candles, using existing data")

        if self.data.empty or len(self.data) < 3:
            logger.warning(f"Insufficient data: {len(self.data)} rows, need at least 3")
            return

        latest_candle_time = self.data.iloc[-1]['time']
        current_time = datetime.now(NY_TZ)
        candle_age = self.calculate_candle_age(current_time, latest_candle_time)
        logger.info(f"Candle age: {candle_age:.2f} minutes")

        signal_type, signal_data = self.feature_engineer.calculate_crt_signal(self.data.tail(3))
        if signal_type and signal_data:
            current_candle = self.data.iloc[-1]
            if (self.last_signal_candle is None or 
                current_candle['time'] > self.last_signal_candle['time'] or 
                (current_candle['time'] == self.last_signal_candle['time'] and 
                 abs(current_candle['close'] - self.last_signal_candle['close']) > 0.5)):
                self.last_signal_candle = current_candle
                self.last_signal_time = current_time
                
                logger.info(f"Signal validated: {signal_type}")
                alert_time = signal_data['time'].astimezone(NY_TZ)
                setup_msg = (
                    f"🔔 *SETUP* {INSTRUMENT.replace('_','/')} {signal_type}\n"
                    f"Timeframe: {TIMEFRAME}\n"
                    f"Time: {alert_time.strftime('%Y-%m-%d %H:%M')} NY\n"
                    f"Entry: {signal_data['entry']:.2f}\n"
                    f"TP: {signal_data['tp']:.2f}\n"
                    f"SL: {signal_data['sl']:.2f}\n"
                    f"Candle Age: {candle_age:.2f} minutes"
                )
                if send_telegram(setup_msg):
                    features = self.feature_engineer.generate_features(self.data, signal_type, 0)
                    if features is not None:
                        self.last_scaled_features = scaler.transform(pd.DataFrame([features], columns=self.feature_engineer.features)).flatten() if scaler else None
                        feature_msg = f"📊 *FEATURES* {INSTRUMENT.replace('_','/')} {signal_type}\n"
                        formatted_features = []
                        for feat, val in features.items():
                            escaped_feat = feat.replace('_', '\\_')
                            formatted_features.append(f"{escaped_feat}: {val:.4f}")
                        feature_msg += "\n".join(formatted_features)
                        if not send_telegram(feature_msg):
                            logger.error("Failed to send features after retries")

                    if scaler is not None and self.last_scaled_features is not None and models:
                        pred_msg = f"🤖 *MODEL PREDICTIONS* {INSTRUMENT.replace('_','/')} {signal_type}\n"
                        scaled_features = self.last_scaled_features.reshape(1, 1, -1)
                        if np.any(np.isnan(scaled_features)):
                            logger.warning("NaN values detected in scaled_features, replacing with 0")
                            scaled_features = np.nan_to_num(scaled_features, nan=0.0)
                        
                        predictions = []
                        for i in range(10):  # Generate 10 predictions
                            probs = np.array([model.predict(scaled_features, verbose=0)[0, 0] for model in models])
                            avg_prob = np.mean(probs)
                            final_pred = 1 if avg_prob >= 0.55 else 0
                            outcome = "Worth Taking" if final_pred == 1 else "Likely Loss"
                            predictions.append(f"Prediction {i+1}: {avg_prob:.4f} (Outcome: {outcome})")
                        
                        pred_msg += "\n".join(predictions)
                        if not send_telegram(pred_msg):
                            logger.error("Failed to send model predictions after retries")
                            
                        if final_pred == 1:
                            trade = (signal_data['entry'], signal_data['sl'], signal_data['tp'], current_time, len(self.data) - 1)
                            self.active_trades.append(trade)
                            logger.info(f"Trade stored: {trade}")
                    else:
                        logger.error("No models or scaler loaded")
                        pred_msg = f"🤖 *MODEL PREDICTIONS* {INSTRUMENT.replace('_','/')} {signal_type}\nError: No models or scaler loaded."
                        send_telegram(pred_msg)
            else:
                logger.debug("Signal skipped due to similar candle conditions")
        else:
            logger.debug("No signal detected, resetting last signal candle")
            self.last_signal_candle = None

        if latest_candles.empty and candle_age > 1:
            next_candle_time = self._get_next_candle_time(current_time)
            sleep_seconds = (next_candle_time - current_time).total_seconds()
            logger.info(f"Waiting for next candle, sleeping {sleep_seconds:.1f} seconds")
            time.sleep(max(1, sleep_seconds))
        else:
            time.sleep(60)  # Check again in 1 minute if new data

        self.check_trade_results(latest_candles)

    def check_trade_results(self, latest_candles):
        """Check trade outcomes after each candle request"""
        if not self.active_trades:
            return

        current_time = datetime.now(NY_TZ)
        latest_candle = latest_candles.iloc[-1] if not latest_candles.empty else self.data.iloc[-1]
        completed_trades = []

        for trade in self.active_trades[:]:
            entry, sl, tp, start_time, trade_index = trade
            logger.info(f"Checking trade: Entry={entry}, SL={sl}, TP={tp}")

            if entry > sl:  # SELL trade
                if latest_candle['high'] >= sl:
                    outcome = 'Hit SL (Loss)'
                    logger.info(f"SELL trade loss: Price {latest_candle['high']} >= SL {sl}")
                elif latest_candle['low'] <= tp:
                    outcome = 'Hit TP (Win)'
                    logger.info(f"SELL trade win: Price {latest_candle['low']} <= TP {tp}")
                else:
                    continue
            else:  # BUY trade (for future consistency, though not checked here per request)
                continue

            completed_trades.append((entry, sl, tp, start_time, trade_index, outcome, current_time))
            self.active_trades.remove(trade)
            outcome_msg = (
                f"📈 *Trade Outcome*\n"
                f"Entry: {entry:.2f}\n"
                f"SL: {sl:.2f}\n"
                f"TP: {tp:.2f}\n"
                f"Outcome: {outcome}\n"
                f"Time: {current_time.strftime('%Y-%m-%d %H:%M')} NY"
            )
            send_telegram(outcome_msg)

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
        return min(44, max(0, int(elapsed)))
    
    def run(self):
        logger.info("Starting CandleScheduler thread")
        while self.active:
            try:
                logger.info("Fetching latest candle data")
                df_candles = fetch_candles()
                
                if df_candles.empty:
                    logger.warning("No candles fetched, forcing callback with existing data")
                    if hasattr(self, 'data') and self.data is not None and len(self.data) >= 3:
                        latest_time = self.data['time'].max()
                        minutes_closed = self.calculate_minutes_closed(latest_time)
                        if self.callback:
                            logger.info(f"Forcing callback with minutes closed: {minutes_closed}")
                            self.callback(minutes_closed, self.data.tail(3))
                else:
                    latest_candle = df_candles.iloc[-1]
                    latest_time = latest_candle['time']
                    minutes_closed = self.calculate_minutes_closed(latest_time)
                    if self.callback:
                        logger.info(f"Calling callback with minutes closed: {minutes_closed}")
                        self.callback(minutes_closed, df_candles.tail(1))
                
                now = datetime.now(NY_TZ)
                next_run = self.calculate_next_candle()
                sleep_seconds = (next_run - now).total_seconds()
                logger.info(f"Sleeping {sleep_seconds:.1f} seconds until next candle")
                time.sleep(max(1, sleep_seconds))
                
            except Exception as e:
                logger.error(f"Scheduler error: {str(e)}")
                logger.error(traceback.format_exc())
                time.sleep(60)

# ========================
# MAIN BOT OPERATION
# ========================
def run_bot():
    logger.info("Starting trading bot")
    send_telegram(f"🚀 *Bot Started*\nInstrument: {INSTRUMENT}\nTimeframe: {TIMEFRAME}\nTime: {datetime.now(NY_TZ)}")
    
    try:
        detector = TradingDetector()
    except Exception as e:
        logger.error(f"Detector initialization failed: {str(e)}")
        send_telegram(f"❌ *Bot Failed to Start*:\n{str(e)}")
        return
        
    logger.info("Bot started successfully")
    
    while True:
        try:
            logger.info("Running bot cycle")
            last_time = detector.data['time'].max() if not detector.data.empty else None
            df = fetch_candles(last_time)
            if not df.empty:
                logger.info(f"Fetched {len(df)} new candles, updating data")
                detector.update_data(df)
            else:
                logger.warning("No new candles fetched in this cycle")
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
        while True:
            time.sleep(1)  # Keep main thread alive
    except KeyboardInterrupt:
        logger.info("Shutting down bot")
        send_telegram("🔔 *Bot Stopped*")
