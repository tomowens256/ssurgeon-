#!/usr/bin/env python3

import csv
import asyncio
import traceback
import logging
import json
from typing import Dict, List, Optional, Tuple
import os
import sys
import time
import re
import requests
import pandas as pd
from datetime import datetime, timedelta
from collections import OrderedDict
from numba import jit
import numpy as np
import pytz
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set
from oandapyV20 import API
from oandapyV20.exceptions import V20Error
from oandapyV20.endpoints import instruments
from pytz import timezone
from zoneinfo import ZoneInfo
NY_TZ = ZoneInfo("America/New_York")
import signal

def ignore_sigint():
    """Ignore Ctrl+C signals"""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

# Call this when you start your bot
ignore_sigint()


# ================================
# CONFIGURATION - BACKWARD COMPATIBLE
# ================================

TRADING_PAIRS = {
    'GOLD_TRIAD': {
        'pair1': 'XAU_USD',  # OLD structure (keep for compatibility)
        'pair2': 'XAU_JPY',  # OLD structure (keep for compatibility)
        'instruments': ['XAU_USD', 'XAU_JPY'],  # NEW structure
        'timeframe_mapping': {
            'monthly': 'H4',
            'weekly': 'H1', 
            'daily': 'M15',
            '90min': 'M5'
        }
    },
    'precious_metals': {
        'pair1': 'XAU_USD',  # OLD structure (keep for compatibility)ssss
        'pair2': 'XAG_USD',  # OLD structure (keep for compatibility)
        'instruments': ['XAU_USD', 'XAG_USD'],  # NEW structure
        'timeframe_mapping': {
            'monthly': 'H4',
            'weekly': 'H1', 
            'daily': 'M15',
            '90min': 'M5'
        }
    },
    'us_indices_triad': {
        'pair1': 'NAS100_USD',  
        'pair2': 'SPX500_USD',  
        'instruments': ['NAS100_USD','SPX500_USD'],  # NEW structuress
        'timeframe_mapping': {
            'monthly': 'H4',
            'weekly': 'H1', 
            'daily': 'M15',
            '90min': 'M5'
        }
    },
    'fx_triad': {
        'pair1': 'GBP_USD',  # OLD structure
        'pair2': 'EUR_USD',  # OLD structure  
        'instruments': ['GBP_USD', 'EUR_USD'],  # NEW structure
        'timeframe_mapping': {
            'monthly': 'H4',
            'weekly': 'H1',
            'daily': 'M15',
            '90min': 'M5'
        }
    # },
    # 'european_indices': {
    #     'pair1': 'DE30_EUR',
    #     'pair2': 'EU50_EUR',
    #     'instruments': ['DE30_EUR', 'EU50_EUR'],
    #     'timeframe_mapping': {
    #         'monthly': 'H4',
    #         'weekly': 'H1',
    #         'daily': 'M15',
    #         '90min': 'M5'
    #     }
    }
}

CRT_TIMEFRAMES = [ 'H1','H2','H4']
CRT_SMT_MAPPING = {
    'H4': ['weekly', 'daily'],    # 4hr CRT can use weekly OR daily S
    'H1': ['daily', '90min'],
    'H6' :['weekly', 'daily'],
    'H12' :['weekly', 'daily'],# 1hr CRT can use daily SMT
    'H2': ['daily', '90min']     # 15min CRT can use daily OR 90min SMT
}
FVG_TIMEFRAMES = ['M15', 'H1', 'H4', 'D']

CYCLE_SLEEP_TIMEFRAMES = {
    'monthly': 'H4',
    'weekly': 'H1',  
    'daily': 'M15',
    '90min': 'M5'
}

# CYCLE HIERARCHY - Two smaller cycles can override one higher cycle
CYCLE_HIERARCHY = {
    'monthly': 4,
    'weekly': 3, 
    'daily': 2,
    '90min': 1
}

# ================================
# ZEBRA SCANNING INSTRUMENTS
# ================================

ZEBRA_INSTRUMENTS = {
    # Format: 'INSTRUMENT': [LIST_OF_TIMEFRAMES]
    'XAU_USD': ['M1', 'M3', 'M5', 'M15'],
    'EUR_USD': ['M1', 'M3', 'M5', 'M15'],
    'GBP_USD': ['M1', 'M3', 'M5', 'M15'],
    'USD_JPY': ['M1', 'M3', 'M5', 'M15'],
    'NAS100_USD': ['M1', 'M3', 'M5', 'M15'],
    'DE30_EUR': ['M1', 'M3', 'M5', 'M15'],
    'XAG_USD': ['M1', 'M3', 'M5', 'M15']
    # Add more instruments as needed
}

NY_TZ = pytz.timezone('America/New_York')  # UTC-4 (or UTC-5 during standard time)
BASE_INTERVAL = 60
MIN_INTERVAL = 10
MAX_RETRIES = 3
CANDLE_BUFFER_SECONDS = 5

# ================================
# LOGGING SETUP  
# ================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('robust_trading_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ================================
# UTILITY FUNCTIONS
# ================================

def parse_oanda_time(time_str):
    """Parse Oanda's timestamp with variable fractional seconds - ENFORCE UTC-4"""
    try:
        if '.' in time_str and len(time_str.split('.')[1]) > 7:
            time_str = re.sub(r'\.(\d{6})\d+', r'.\1', time_str)
        # Parse as UTC first, then convert to UTC-4 (NY time)
        utc_time = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=pytz.utc)
        return utc_time.astimezone(NY_TZ)  # Convert to UTC-4
    except Exception as e:
        logger.error(f"Error parsing time {time_str}: {str(e)}")
        return datetime.now(NY_TZ)

# ========================
# FIBONACCI UTILITY FUNCTIONS
# ========================

def calculate_fibonacci_levels(high_price, low_price, levels=None, direction='bearish'):
    """Calculate Fibonacci retracement and extension levels with direction awareness"""
    if levels is None:
        # Standard Fibonacci levels plus your custom ones
        levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.414, 1.618, 2.0, 2.618]
    
    price_range = high_price - low_price
    fib_levels = {}
    
    if direction == 'bearish':
        # For bearish: 0 = highest_high, 1 = lowest_low
        for level in levels:
            if level <= 1.0:  # Retracement
                price = high_price - (price_range * level)
            else:  # Extension
                price = low_price + (price_range * (level - 1.0))
            fib_levels[level] = price
        
        # Add your specific levels
        fib_levels[0.67] = high_price - (price_range * 0.67)
        fib_levels[1.25] = low_price + (price_range * 0.25)  # For SL in 0.67-1 zone
        fib_levels[1.5] = low_price + (price_range * 0.5)   # For SL in 0.5-0.67 zone
        
    else:  # bullish
        # For bullish: 0 = lowest_low, 1 = highest_high
        for level in levels:
            if level <= 1.0:  # Retracement
                price = low_price + (price_range * level)
            else:  # Extension
                price = high_price + (price_range * (level - 1.0))
            fib_levels[level] = price
        
        # Add your specific levels
        fib_levels[0.67] = low_price + (price_range * 0.67)
        fib_levels[1.25] = high_price - (price_range * 0.25)  # For SL in 0.67-1 zone
        fib_levels[1.5] = high_price - (price_range * 0.5)   # For SL in 0.5-0.67 zone
    
    return fib_levels

def get_price_at_fib_level(high_price, low_price, fib_level):
    """Get price at specific Fibonacci level"""
    price_range = high_price - low_price
    if fib_level <= 1.0:
        return high_price - (price_range * fib_level)
    else:
        return low_price + (price_range * (fib_level - 1.0))

def find_fibonacci_zone(price, fib_levels):
    """Determine which Fibonacci zone a price is in"""
    zones = [
        (0.67, 1.0, "0.67-1"),
        (0.5, 0.67, "0.5-0.67"),
        (0, 0.5, "0-0.5")
    ]
    
    for low_level, high_level, zone_name in zones:
        if low_level in fib_levels and high_level in fib_levels:
            zone_low = fib_levels[high_level]  # Note: reversed because Fibonacci goes from high to low
            zone_high = fib_levels[low_level]
            if zone_low <= price <= zone_high:
                return zone_name, fib_levels[low_level], fib_levels[high_level]
    
    return None, None, None

def get_extreme_prices(df, since_time=None):
    """Get highest high and lowest low from data since specific time"""
    if since_time:
        filtered_df = df[df['time'] >= since_time]
    else:
        filtered_df = df
    
    if filtered_df.empty:
        return None, None
    
    highest_high = filtered_df['high'].max()
    lowest_low = filtered_df['low'].min()
    
    return highest_high, lowest_low

def send_telegram(message, token=None, chat_id=None):
    """Send formatted message to Telegram"""
    if not token or not chat_id:
        logger.error("Telegram credentials missing")
        return False
        
    if len(message) > 4000:
        message = message[:4000] + "... [TRUNCATED]"
    
    escape_chars = '_*[]()~`>#+-=|{}.!'
    for char in escape_chars:
        message = message.replace(char, '\\' + char)
    
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(url, json={
                'chat_id': chat_id,
                'text': message,
                'parse_mode': 'MarkdownV2'
            }, timeout=10)
            
            if response.status_code == 200 and response.json().get('ok'):
                logger.info("Telegram message sent successfully")
                return True
            else:
                error_msg = f"Telegram error {response.status_code}"
                if response.text:
                    error_msg += f": {response.text[:200]}"
                logger.error(error_msg)
                time.sleep(2 ** attempt)
        except Exception as e:
            logger.error(f"Telegram connection failed: {str(e)}")
            time.sleep(2 ** attempt)
    
    logger.error(f"Failed to send Telegram message after {MAX_RETRIES} attempts")
    return False

def fetch_candles(instrument, timeframe, count=100, api_key=None, since=None, use_cache=True):
    """Fetch candles from OANDA API - with global caching support"""
    
    if not api_key:
        return pd.DataFrame()
    
    # Check cache first (if enabled)
    if use_cache:
        cached_data = GLOBAL_CACHE.get(instrument, timeframe, count)
        if cached_data is not None:
            # Filter by 'since' if provided
            if since is not None and not cached_data.empty:
                filtered = cached_data[cached_data['time'] > since]
                if not filtered.empty:
                    return filtered
            else:
                return cached_data
    
    try:
        api = API(access_token=api_key, environment="practice")
        logging.getLogger('oandapyV20.oandapyV20').setLevel(logging.ERROR)
    except Exception:
        return pd.DataFrame()
    
    params = {
        "granularity": timeframe,
        "count": min(count, 500),
        "price": "M",
        "alignmentTimezone": "America/New_York",
        "includeCurrent": True
    }
    if since:
        params["from"] = since.strftime('%Y-%m-%dT%H:%M:%S')
    
    for attempt in range(MAX_RETRIES):
        try:
            request = instruments.InstrumentsCandles(instrument=instrument, params=params)
            response = api.request(request)
            
            candles = response.get('candles', [])
            if not candles:
                continue
            
            data = []
            for candle in candles:
                price_data = candle.get('mid', {})
                if not price_data:
                    continue
                
                try:
                    raw_time = candle['time']
                    parsed_time = pd.to_datetime(raw_time, utc=True).tz_convert(NY_TZ)
                    is_complete = candle.get('complete', False)
                    
                    data.append({
                        'time': parsed_time,
                        'open': float(price_data['o']),
                        'high': float(price_data['h']),
                        'low': float(price_data['l']),
                        'close': float(price_data['c']),
                        'volume': int(candle.get('volume', 0)),
                        'complete': is_complete,
                        'is_current': not is_complete
                    })
                except Exception:
                    continue
            
            if not data:
                continue
            
            df = pd.DataFrame(data).drop_duplicates(subset=['time'], keep='last')
            df = df.sort_values('time').reset_index(drop=True)
            
            if since:
                df = df[df['time'] > since]
            
            # Store in global cache
            if use_cache:
                GLOBAL_CACHE.set(instrument, timeframe, count, df)
            
            return df
            
        except Exception as e:
            error_str = str(e)
            
            if '502' in error_str or 'Bad Gateway' in error_str:
                time.sleep(2)
                continue
                
            elif '429' in error_str or 'rate' in error_str.lower():
                wait_time = min(30, 5 * (2 ** attempt))
                logger.debug(f"[{instrument}] Rate limit, waiting {wait_time}s")
                time.sleep(wait_time)
                continue
                
            else:
                code = getattr(e, 'code', 'No code')
                logger.debug(f"[{instrument}] Error {code}")
                break
    
    return pd.DataFrame()


# ========================
# HUMOR GENERATOR FOR SIGNALS
# ========================

def get_humorous_phrase(direction, pattern_type):
    """Generate humorous phrases for signals"""
    import random
    
    bullish_phrases = [
        "Yoo bro, bulls are charging! 🐂",
        "Ain't no stopping this train! 🚂",
        "To the moon we go! 🚀",
        "Green candles incoming! 💚",
        "Bulls be like: 'Hold my beer' 🍻",
        "Wake up sleeping bulls, dinner is served! 🍽️",
        "This setup is smoother than butter on a bald monkey! 🐒🧈",
        "Bulls are back in town! 🏙️🐂",
        "Time to ride the lightning! ⚡",
        "If this was a movie, we'd be buying popcorn right now! 🍿"
    ]
    
    bearish_phrases = [
        "Yoo bro, bears are waking up! 🐻",
        "Gravity is about to kick in! ⬇️",
        "Time to short the hopium! 📉",
        "Red alert! Bears are hungry! 🚨",
        "Bears be like: 'Not on my watch' ⌚🐻",
        "Pack your bags bears, it's hunting season! 🎒",
        "This drop gonna be steeper than my ex's standards! 📉",
        "Bears are throwing a party and everyone's invited! 🎉",
        "Get ready for the slide! 🛝",
        "If bears had a DJ, they'd be dropping the bass! 🎵"
    ]
    
    if direction == 'bullish':
        phrases = bullish_phrases
    else:
        phrases = bearish_phrases
    
    # Add pattern-specific humor
    if pattern_type == 'FVG+SMT':
        phrases.append("FVG + SMT = Profit Party! 🎊")
    elif pattern_type == 'SD+SMT':
        phrases.append("Supply/Demand zones never looked so good! 🗺️")
    elif pattern_type == 'CRT+SMT':
        phrases.append("CRT confluence? More like CRT confetti! 🎉")
    
    return random.choice(phrases)

def get_hammer_humor(direction, timeframe):
    """Humorous phrases for hammer signals"""
    hammer_jokes = [
        "Hammer time! Can't touch this! 🎵🔨",
        "When life gives you hammers, make profits! 🔨💰",
        "This hammer's hitting harder than my morning coffee! ☕🔨",
        "Not just any hammer - this is the Mjolnir of trading! ⚡🔨",
        "If this hammer was any better, it would have its own reality show! 📺🔨",
        "Warning: This hammer may cause profit explosions! 💥🔨",
        "Hammer so fresh, it came with a receipt! 🧾🔨",
        "This setup is hammerific! 🎯🔨",
        "When you see this hammer, you know what time it is... Profit o'clock! ⏰🔨",
        "Hammer detected: Proceed to profit extraction! 🏗️🔨"
    ]
    
    if timeframe == 'M1':
        hammer_jokes.append("1-minute hammer? That's quicker than my coffee break! ⚡🔨")
    elif timeframe == 'M3':
        hammer_jokes.append("3-minute hammer - perfect for microwave traders! 🍿🔨")
    elif timeframe == 'M5':
        hammer_jokes.append("5-minute hammer: Faster than fast food profits! 🍔🔨")
    elif timeframe == 'M15':
        hammer_jokes.append("15-minute hammer: Take a breather, but not too long! 😮‍💨🔨")
    
    import random
    return random.choice(hammer_jokes)


# ================================
# GLOBAL CACHE (Shared by all components)
# ================================

class GlobalCandleCache:
    """Singleton cache shared by all scanners to prevent duplicate API calls"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, ttl_seconds=45, max_size=5000):
        if not self._initialized:
            self.ttl_seconds = ttl_seconds
            self.max_size = max_size
            self.cache = {}  # key -> (data, timestamp, ttl)  # ✅ CHANGED: Store ttl with entry
            self._lock = threading.Lock()
            self._initialized = True
            logger.info(f"📦 Global Candle Cache initialized (TTL: {ttl_seconds}s, Max: {max_size} entries)")
    
    def _get_cache_key(self, instrument, timeframe, count):
        return f"{instrument}_{timeframe}_{count}"
    
    def get(self, instrument, timeframe, count):
        """Get cached data if available and not expired"""
        key = self._get_cache_key(instrument, timeframe, count)
        
        with self._lock:
            if key in self.cache:
                data, timestamp, entry_ttl = self.cache[key]  # ✅ CHANGED: Unpack 3 values
                if time.time() - timestamp < entry_ttl:  # ✅ CHANGED: Use entry-specific TTL
                    return data
        
        return None
    
    def set(self, instrument, timeframe, count, data, ttl_override=None):
        """Cache data with TTL"""
        key = self._get_cache_key(instrument, timeframe, count)
        ttl = ttl_override if ttl_override is not None else self.ttl_seconds
        
        with self._lock:
            # Remove oldest entries if cache is full
            if len(self.cache) >= self.max_size:
                # Find oldest by timestamp (index 1)
                oldest_key = min(self.cache.keys(), 
                               key=lambda k: self.cache[k][1])
                del self.cache[oldest_key]
            
            # Store data, timestamp, and entry-specific TTL
            self.cache[key] = (data, time.time(), ttl)  # ✅ CHANGED: Store 3 values
    
    def clear(self):
        """Clear entire cache"""
        with self._lock:
            self.cache.clear()
    
    def get_stats(self):
        """Get cache statistics"""
        with self._lock:
            # Calculate hit rate if we track hits/misses
            hits = getattr(self, 'hits', 0)
            misses = getattr(self, 'misses', 0)
            total = hits + misses
            hit_rate = (hits / total * 100) if total > 0 else 0
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'ttl_seconds': self.ttl_seconds,
                'hits': hits,
                'misses': misses,
                'hit_rate_percent': round(hit_rate, 2)
            }
# Create global instance
GLOBAL_CACHE = GlobalCandleCache()


import pandas as pd
import pickle
import warnings
from datetime import datetime
from typing import Dict
import os
from collections import defaultdict

class SignalProcessor:
    def __init__(self, rf_model_path: str, category_maps_path: str, predictions_log_path: str = None):
        """
        Args:
            rf_model_path: Path to Random Forest model pickle file (rf_t2version_v3.pkl)
            category_maps_path: Path to category maps pickle file (category_maps.pkl)
            predictions_log_path: Path to CSV where predictions will be logged
        """
        # Suppress warnings
        warnings.filterwarnings("ignore")
        
        print(f"🎯 Initializing SignalProcessor with Random Forest model...")
        
        # Store paths
        self.rf_model_path = rf_model_path
        self.category_maps_path = category_maps_path
        
        # Define feature columns (EXACT order as model expects)
        self.categorical_cols = [
            'instrument', 'hammer_timeframe', 'direction', 'criteria',
            'trigger_timeframe', 'smt_cycle', 'smt_quarters', 'H4_open_rel',
            'H6_open_rel', 'D_open_rel', 'W_open_rel', '1m_zebra', '3m_zebra',
            '5m_zebra', '15m_zebra', 'h1_zebra', 'h4_zebra', 'h6_zebra', 'd_zebra'
        ]
        
        self.continuous_cols = [
            'sl_distance_pips', 'rsi', 'vwap', 'news_high_count',
            'news_medium_count', 'news_low_count', 'entry_count'
        ]
        
        self.all_columns = self.categorical_cols + self.continuous_cols
        
        # Load everything ONCE at initialization
        self.model = None
        self.cat_maps = None
        
        # Load Random Forest model
        if os.path.exists(rf_model_path):
            try:
                print(f"📊 Loading Random Forest model from: {rf_model_path}")
                with open(rf_model_path, 'rb') as f:
                    self.model = pickle.load(f)
                print(f"✅ Model loaded: {type(self.model).__name__}")
            except Exception as e:
                print(f"❌ Failed to load model: {e}")
        else:
            print(f"❌ Model file not found: {rf_model_path}")
        
        # Load category maps
        if os.path.exists(category_maps_path):
            try:
                print(f"📊 Loading category maps from: {category_maps_path}")
                with open(category_maps_path, 'rb') as f:
                    self.cat_maps = pickle.load(f)
                print(f"✅ Category maps loaded: {len(self.cat_maps)} columns")
            except Exception as e:
                print(f"❌ Failed to load category maps: {e}")
        else:
            print(f"❌ Category maps file not found: {category_maps_path}")
        
        # Setup predictions logging
        if predictions_log_path and os.path.exists(os.path.dirname(predictions_log_path)):
            self.csv_log_path = predictions_log_path
        else:
            # Default to model directory if not specified
            model_dir = os.path.dirname(rf_model_path)
            self.csv_log_path = os.path.join(model_dir, 'ml_predictions.csv')
        
        # Initialize CSV log
        self._init_csv_log()
        
        # For backward compatibility
        self.logger = None
        
        # Track ML status
        self.ml_enabled = self.model is not None and self.cat_maps is not None
        print(f"🎯 SignalProcessor ready. ML Filtering: {'✅ ENABLED' if self.ml_enabled else '❌ DISABLED'}")
    
    def set_logger(self, logger):
        """Allow external logger injection"""
        self.logger = logger
        if self.logger:
            self.logger.info("📊 Logger attached to SignalProcessor")
    
    def _init_csv_log(self):
        """Initialize CSV log file with headers if it doesn't exist"""
        try:
            if not os.path.exists(self.csv_log_path):
                os.makedirs(os.path.dirname(self.csv_log_path), exist_ok=True)
                with open(self.csv_log_path, 'w') as f:
                    f.write("timestamp,trade_id,prediction\n")
                print(f"✅ Created prediction log: {self.csv_log_path}")
            else:
                print(f"📝 Using existing prediction log: {self.csv_log_path}")
        except Exception as e:
            print(f"⚠️ Could not initialize CSV log: {e}")
    
    def _log_prediction(self, signal_id: str, prediction: int):
        """Log prediction to CSV file"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.csv_log_path, 'a') as f:
                f.write(f"{timestamp},{signal_id},{prediction}\n")
        except Exception as e:
            if self.logger:
                self.logger.error(f"⚠️ Failed to log prediction: {e}")
    
    def _clean_value(self, value):
        """Clean a single value - remove tabs, newlines, strip whitespace"""
        if isinstance(value, str):
            return value.replace('\t', '').replace('\n', '').strip()
        return value
    
    def _prepare_single_prediction(self, features: Dict) -> pd.DataFrame:
        """
        Prepare single set of features for prediction
        Returns: DataFrame ready for model prediction
        """
        # Clean all feature values
        cleaned_features = {}
        for key, value in features.items():
            cleaned_key = self._clean_value(key)
            cleaned_value = self._clean_value(value)
            cleaned_features[cleaned_key] = [cleaned_value]  # Wrap in list for DataFrame
        
        # Create DataFrame
        df = pd.DataFrame(cleaned_features)
        
        # Ensure all required columns exist (fill missing with defaults)
        for col in self.all_columns:
            if col not in df.columns:
                if col in self.continuous_cols:
                    df[col] = 0  # Default for continuous
                else:
                    df[col] = ''  # Default for categorical
        
        # Order columns exactly as model expects
        df = df[self.all_columns]
        
        # Encode categorical features
        for cat_col in self.categorical_cols:
            if cat_col in self.cat_maps and cat_col in df.columns:
                try:
                    # Convert to categorical codes
                    df[cat_col] = pd.Categorical(
                        df[cat_col].astype(str).str.strip(), 
                        categories=self.cat_maps[cat_col]
                    ).codes
                except:
                    df[cat_col] = -1  # Unknown category
            else:
                df[cat_col] = -1
        
        # Ensure continuous columns are numeric
        for cont_col in self.continuous_cols:
            if cont_col in df.columns:
                df[cont_col] = pd.to_numeric(df[cont_col], errors='coerce').fillna(0)
        
        return df
    
    def check_with_features(self, signal_id: str, features: Dict) -> int:
        """
        Main method: Check if we should trade based on ML model
        
        Args:
            signal_id: Unique trade identifier
            features: Dictionary with ALL 26 features as shown in your code
            
        Returns:
            1 = Trade approved (model says yes)
            0 = Trade rejected (model says no or error)
        """
        # If ML not enabled, default to approve (for backward compatibility)
        if not self.ml_enabled:
            if self.logger:
                self.logger.warning(f"⚠️ ML disabled, auto-approving trade: {signal_id}")
            self._log_prediction(signal_id, 1)
            return 1
        
        try:
            # Prepare features
            df = self._prepare_single_prediction(features)
            
            # Make prediction
            raw_prediction = self.model.predict(df)[0]
            
            # Map: -1 -> 0 (reject), 2 -> 1 (approve)
            prediction = 1 if raw_prediction == 2 else 0
            
            # Log to CSV
            self._log_prediction(signal_id, prediction)
            
            # Log result
            if self.logger:
                if prediction == 1:
                    self.logger.info(f"✅ ML APPROVED {signal_id} (raw: {raw_prediction})")
                else:
                    self.logger.info(f"⏸️ ML REJECTED {signal_id} (raw: {raw_prediction})")
            
            return prediction
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ ML error for {signal_id}: {str(e)}")
            
            # Log as rejection on error (safer)
            self._log_prediction(signal_id, 0)
            return 0
# ================================
# ENHANCED TIMING MANAGER
# ================================

class RobustTimingManager:
    """Enhanced timing manager with STRONG duplicate prevention"""
    
    def __init__(self):
        self.ny_tz = pytz.timezone('America/New_York')  # UTC-4
        self.sent_signals = {}  # Track sent signals to prevent duplicates
        
    def is_duplicate_signal(self, signal_key, pair_group, cooldown_minutes=30):
        """Check if signal is a duplicate with cooldown period"""
        current_time = datetime.now(NY_TZ)
        
        # Initialize if not exists
        if pair_group not in self.sent_signals:
            self.sent_signals[pair_group] = {}
            return False
        
        # ✅ FIX THIS LINE: Change .keys() to .items()
        for existing_key, last_sent in list(self.sent_signals[pair_group].items()):  # ← FIXED
            # Check if same signal key (exact duplicate)
            if existing_key == signal_key:
                time_diff = (current_time - last_sent).total_seconds() / 60
                if time_diff < cooldown_minutes:
                    logger.info(f"⏳ STRONG DUPLICATE PREVENTION: {signal_key} (sent {time_diff:.1f} min ago)")
                    return True
        
        # Clean up old signals (older than cooldown)
        self.sent_signals[pair_group] = {
            key: time for key, time in self.sent_signals[pair_group].items() 
            if (current_time - time).total_seconds() / 60 < cooldown_minutes
        }
        
        return False
    
    def _signals_are_very_similar(self, signal1, signal2):
        """Check if two signals are VERY similar (same direction and cycle pattern)"""
        # Extract direction
        dir1 = "BULLISH" if "BULLISH" in signal1.upper() else "BEARISH" if "BEARISH" in signal1.upper() else None
        dir2 = "BULLISH" if "BULLISH" in signal2.upper() else "BEARISH" if "BEARISH" in signal2.upper() else None
        
        # Must have same direction
        if dir1 != dir2:
            return False
            
        # Extract cycles
        cycles1 = [cycle for cycle in ['MONTHLY', 'WEEKLY', 'DAILY', '90MIN'] if cycle in signal1.upper()]
        cycles2 = [cycle for cycle in ['MONTHLY', 'WEEKLY', 'DAILY', '90MIN'] if cycle in signal2.upper()]
        
        # If both have same cycle composition, consider very similar
        if cycles1 and cycles2 and set(cycles1) == set(cycles2):
            return True
            
        return False
    
    def _clean_old_entries(self):
        """Clean entries older than 48 hours"""
        current_time = datetime.now(self.ny_tz)
        for pair_group in list(self.sent_signals.keys()):
            for signal_key in list(self.sent_signals[pair_group].keys()):
                if (current_time - self.sent_signals[pair_group][signal_key]).total_seconds() > 172800:  # 48 hours
                    del self.sent_signals[pair_group][signal_key]
    
    def validate_chronological_order(self, prev_time, curr_time):
        """Validate that swing times are in correct chronological order"""
        if prev_time and curr_time:
            time_diff = (curr_time - prev_time).total_seconds()
            is_valid = time_diff > 0
            if not is_valid:
                logger.warning(f"⚠️ NON-CHRONOLOGICAL SWINGS: {prev_time.strftime('%H:%M')} → {curr_time.strftime('%H:%M')} (diff: {time_diff/60:.1f} min)")
            return is_valid
        return True
    
    def is_psp_within_bounds(self, smt_formation_time, psp_formation_time, cycle_type):
        """Check if PSP is within reasonable time of SMT formation"""
        if not smt_formation_time or not psp_formation_time:
            return False
            
        time_diff = abs((psp_formation_time - smt_formation_time).total_seconds() / 3600)  # hours
        
        # Maximum allowed time difference based on cycle
        max_hours = {
            'monthly': 24 * 3,  # 3 days
            'weekly': 24 * 1,   # 1 day  
            'daily': 1,         # 1hour
            '90min': 0.3          # 30 minutes
        }
        
        max_allowed = max_hours.get(cycle_type, 3)
        is_within = time_diff <= max_allowed
        
        if not is_within:
            logger.warning(f"⚠️ PSP TOO FAR FROM SMT: {time_diff:.1f}h > {max_allowed}h for {cycle_type}")
            
        return is_within
    
    def calculate_next_candle_time(self, timeframe):
        """Calculate when the next candle will open for any timeframe"""
        now = datetime.now(self.ny_tz)
        
        if timeframe.startswith('H'):
            hours = int(timeframe[1:])
            return self._calculate_next_htf_candle_time(hours)
        elif timeframe.startswith('M'):
            minutes = int(timeframe[1:])
            return self._calculate_next_ltf_candle_time(minutes)
        else:
            return self._calculate_next_htf_candle_time(1)
    
    def _calculate_next_htf_candle_time(self, hours):
        """Calculate next candle time for hourly timeframes"""
        now = datetime.now(self.ny_tz)
        
        if hours == 1:
            next_hour = now.hour + 1
            if next_hour >= 24:
                next_time = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            else:
                next_time = now.replace(hour=next_hour, minute=0, second=0, microsecond=0)
                
        elif hours == 4:
            current_hour = now.hour
            next_hour = ((current_hour // 4) * 4 + 4) % 24
            if next_hour < current_hour:
                next_time = now.replace(hour=next_hour, minute=0, second=0, microsecond=0) + timedelta(days=1)
            else:
                next_time = now.replace(hour=next_hour, minute=0, second=0, microsecond=0)
                
        else:
            current_hour = now.hour
            next_hour = ((current_hour // hours) * hours + hours) % 24
            if next_hour < current_hour:
                next_time = now.replace(hour=next_hour, minute=0, second=0, microsecond=0) + timedelta(days=1)
            else:
                next_time = now.replace(hour=next_hour, minute=0, second=0, microsecond=0)
        
        return next_time
    
    def _calculate_next_ltf_candle_time(self, minutes):
        """Calculate next candle time for minute timeframes"""
        now = datetime.now(self.ny_tz)
        current_timestamp = now.timestamp()
        next_candle_timestamp = (current_timestamp // (minutes * 60) + 1) * (minutes * 60)
        return datetime.fromtimestamp(next_candle_timestamp, self.ny_tz)
    
    def get_sleep_time_for_cycle(self, cycle_type):
        """Calculate sleep time until next candle for a specific cycle"""
        timeframe = CYCLE_SLEEP_TIMEFRAMES.get(cycle_type)
        if not timeframe:
            return BASE_INTERVAL
            
        next_candle_time = self.calculate_next_candle_time(timeframe)
        current_time = datetime.now(self.ny_tz)
        
        sleep_seconds = (next_candle_time - current_time).total_seconds() + CANDLE_BUFFER_SECONDS
        
        max_sleep_times = {
            '90min': 300,
            'daily': 900, 
            'weekly': 1800,
            'monthly': 3600
        }
        
        max_sleep = max_sleep_times.get(cycle_type, 300)
        calculated_sleep = max(MIN_INTERVAL, sleep_seconds)
        
        final_sleep = min(calculated_sleep, max_sleep)
        
        logger.debug(f"Sleep calculation for {cycle_type}: calculated={calculated_sleep:.1f}s, max={max_sleep}s, final={final_sleep:.1f}s")
        return final_sleep
    
    def get_sleep_time_for_crt(self, crt_timeframe):
        """Calculate sleep time until next CRT candle"""
        next_candle_time = self.calculate_next_candle_time(crt_timeframe)
        current_time = datetime.now(self.ny_tz)
        
        sleep_seconds = (next_candle_time - current_time).total_seconds() + CANDLE_BUFFER_SECONDS
        
        max_sleep = 900
        
        final_sleep = min(max(MIN_INTERVAL, sleep_seconds), max_sleep)
        logger.debug(f"CRT sleep calculation: calculated={sleep_seconds:.1f}s, max={max_sleep}s, final={final_sleep:.1f}s")
        return final_sleep
    
    def is_crt_fresh(self, crt_timestamp, max_age_minutes=50000):  # Increased increased to infinity since we seed more confluences
        """Check if CRT signal is fresh - MORE LENIENT"""
        if not crt_timestamp:
            return False
            
        current_time = datetime.now(self.ny_tz)
        age_seconds = (current_time - crt_timestamp).total_seconds()
        
        return age_seconds <= (max_age_minutes * 60)

# ================================
# FIXED QUARTER MANAGER - ALL POSSIBLE PAIRS
# ================================

# ================================
# FIXED QUARTER MANAGER - USING PROVEN APPROACH
# ================================
class RobustQuarterManager:
    """FIXED Quarter Manager using the proven working approach"""
    
    def __init__(self):
        self.ny_tz = pytz.timezone('America/New_York')
        
        # PROVEN QUARTER DEFINITIONS from working script
        self.DAILY_QUARTERS = {
            "q1": ("Asian", 18, 0),  # 18:00-00:00
            "q2": ("London", 0, 0),  # 00:00-06:00
            "q3": ("NY", 6, 0),      # 06:00-12:00
            "q4": ("PM", 12, 0)      # 12:00-18:00
        }
        
        self.SESSION_QUARTERS = [(0, 0), (1, 30), (3, 0), (4, 30)]  # 90-min splits
        self.WEEKLY_QUARTERS = ["q1", "q2", "q3", "q4", "q_less"]  # Mon, Tue, Wed, Thu, Fri
    
    def get_current_quarters(self, timestamp=None):
        """Get current quarters for all cycles - PROVEN WORKING METHOD"""
        if timestamp is None:
            timestamp = datetime.now(self.ny_tz)
        else:
            timestamp = timestamp.astimezone(self.ny_tz)
            
        return {
            'monthly': self._get_monthly_quarter_fixed(timestamp),
            'weekly': self._get_weekly_quarter_fixed(timestamp),
            'daily': self._get_daily_quarter_fixed(timestamp),
            '90min': self._get_90min_quarter_fixed(timestamp)
        }
    
    def _get_monthly_quarter_fixed(self, timestamp):
        """Monthly quarters based on week of month"""
        week_of_month = (timestamp.day - 1) // 7 + 1
        if week_of_month == 1: return 'q1'
        elif week_of_month == 2: return 'q2'
        elif week_of_month == 3: return 'q3'
        elif week_of_month == 4: return 'q4'
        else: return 'q_less'
    
    def _get_weekly_quarter_fixed(self, timestamp):
        """Weekly quarters - Monday to Friday"""
        weekday = timestamp.weekday()
        if weekday == 0: return 'q1'      # Monday
        elif weekday == 1: return 'q2'    # Tuesday
        elif weekday == 2: return 'q3'    # Wednesday
        elif weekday == 3: return 'q4'    # Thursday
        else: return 'q_less'             # Friday (and weekend)
    
    def _get_daily_quarter_fixed(self, timestamp):
        """PROVEN DAILY QUARTERS - 18:00 NY start"""
        hour = timestamp.hour
        minute = timestamp.minute
        
        # Daily quarters starting at 18:00 NY time
        if 18 <= hour < 24: return 'q1'      # 18:00-00:00 (Asian)
        elif 0 <= hour < 6: return 'q2'      # 00:00-06:00 (London)  
        elif 6 <= hour < 12: return 'q3'     # 06:00-12:00 (NY)
        elif 12 <= hour < 18: return 'q4'    # 12:00-18:00 (PM)
        else: return 'q_less'
    
    def _get_90min_quarter_fixed(self, timestamp):
        """PROVEN 90MIN QUARTERS within daily quarters"""
        daily_quarter = self._get_daily_quarter_fixed(timestamp)
        
        # Get start of current daily quarter
        quarter_start = self._get_daily_quarter_start_time(timestamp, daily_quarter)
        
        # Calculate minutes into the daily quarter
        minutes_into_quarter = (timestamp - quarter_start).total_seconds() / 60
        
        # 90-minute quarters within each 6-hour daily quarter
        if 0 <= minutes_into_quarter < 90: return 'q1'
        elif 90 <= minutes_into_quarter < 180: return 'q2'
        elif 180 <= minutes_into_quarter < 270: return 'q3'
        elif 270 <= minutes_into_quarter < 360: return 'q4'
        else: return 'q_less'
    
    def _get_daily_quarter_start_time(self, timestamp, daily_quarter):
        """Get the start time of a daily quarter"""
        if daily_quarter == 'q1':  # 18:00-00:00
            start_time = timestamp.replace(hour=18, minute=0, second=0, microsecond=0)
        elif daily_quarter == 'q2':  # 00:00-06:00
            start_time = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        elif daily_quarter == 'q3':  # 06:00-12:00
            start_time = timestamp.replace(hour=6, minute=0, second=0, microsecond=0)
        elif daily_quarter == 'q4':  # 12:00-18:00
            start_time = timestamp.replace(hour=12, minute=0, second=0, microsecond=0)
        else:
            start_time = timestamp
            
        return start_time
    
    def get_adjacent_quarter_pairs(self, cycle_type):
        """Get chronologically valid quarter pairs"""
        if cycle_type == 'weekly':
            quarter_sequence = ['q1', 'q2', 'q3', 'q4', 'q_less']
        else:
            quarter_sequence = ['q1', 'q2', 'q3', 'q4']
    
        all_pairs = []
        for i in range(len(quarter_sequence) - 1):
            all_pairs.append((quarter_sequence[i], quarter_sequence[i+1]))
    
        return all_pairs
    
    def group_candles_by_quarters(self, df, cycle_type, num_quarters=4):
        """Group candles into quarters using PROVEN timing"""
        if df is None or df.empty:
            return {}
            
        df = df.sort_values('time').reset_index(drop=True)
        quarters_data = {}
        
        for _, candle in df.iterrows():
            quarter = self._get_candle_quarter_fixed(candle['time'], cycle_type)
            
            if quarter not in quarters_data:
                quarters_data[quarter] = []
            quarters_data[quarter].append(candle)
        
        # Convert to DataFrames
        for quarter in list(quarters_data.keys()):
            if len(quarters_data[quarter]) < 3:  # Minimum candles
                del quarters_data[quarter]
                continue
                
            quarters_data[quarter] = pd.DataFrame(quarters_data[quarter])
            quarters_data[quarter] = quarters_data[quarter].sort_values('time')
        
        return quarters_data
    
    def _get_candle_quarter_fixed(self, candle_time, cycle_type):
        """Get quarter for candle using PROVEN method"""
        if cycle_type == 'monthly':
            return self._get_monthly_quarter_fixed(candle_time)
        elif cycle_type == 'weekly':
            return self._get_weekly_quarter_fixed(candle_time)
        elif cycle_type == 'daily':
            return self._get_daily_quarter_fixed(candle_time)
        elif cycle_type == '90min':
            return self._get_90min_quarter_fixed(candle_time)
        else:
            return 'unknown'
    
    def get_current_quarter(self, cycle_type, timestamp=None):
        """Get current quarter - simple wrapper"""
        quarters = self.get_current_quarters(timestamp)
        return quarters.get(cycle_type)
    
    def get_last_three_quarters(self, cycle_type):
        """Get last 3 quarters for analysis"""
        current_q = self.get_current_quarter(cycle_type)
        
        if cycle_type == 'weekly':
            order = ['q1', 'q2', 'q3', 'q4', 'q_less']
        else:
            order = ['q1', 'q2', 'q3', 'q4']
        
        if current_q not in order:
            return ['q2', 'q3', 'q4']
            
        idx = order.index(current_q)
        last_three = [
            order[idx],
            order[(idx - 1) % len(order)],
            order[(idx - 2) % len(order)]
        ]
        
        return last_three

    def test_quarter_system(self):
        """Test the quarter system"""
        print("\n🧪 TESTING PROVEN QUARTER SYSTEM:")
        
        test_times = [
            datetime(2025, 11, 20, 18, 0),   # Thursday 18:00 - should be q1 daily
            datetime(2025, 11, 21, 0, 0),    # Friday 00:00 - should be q2 daily  
            datetime(2025, 11, 21, 6, 0),    # Friday 06:00 - should be q3 daily
            datetime(2025, 11, 21, 12, 0),   # Friday 12:00 - should be q4 daily
        ]
        
        for test_time in test_times:
            daily_quarter = self._get_daily_quarter_fixed(test_time)
            weekly_quarter = self._get_weekly_quarter_fixed(test_time)
            print(f"   {test_time.strftime('%m-%d %H:%M')} → Daily: {daily_quarter}, Weekly: {weekly_quarter}")


# ================================
# FIXED QUARTER MANAGER - USING PROVEN APPROACH  
# ================================

def test_proven_quarter_patch():
    """Test that the proven quarter system works"""
    quarter_manager = RobustQuarterManager()
    
    print("\n🎯 TESTING PROVEN QUARTER PATCH:")
    
    # Test current quarters
    current = quarter_manager.get_current_quarters()
    print(f"   Current quarters: {current}")
    
    # Test quarter pairs
    for cycle in ['monthly', 'weekly', 'daily', '90min']:
        pairs = quarter_manager.get_adjacent_quarter_pairs(cycle)
        print(f"   {cycle} pairs: {pairs}")
    
    # Test the proven daily quarter system
    quarter_manager.test_quarter_system()


# ================================
# ULTIMATE SWING DETECTOR WITH 3-CANDLE TOLERANCE
# ================================

class UltimateSwingDetector:
    """Ultimate swing detection with 5-CANDLE TOLERANCE and interim price validation"""
    
    @staticmethod
    def find_swing_highs_lows(df):
        """Fixed swing detection with proper time handling"""
        if df is None or len(df) < 3:
            return [], []
    
        swing_highs = []
        swing_lows = []
    
        MIN_SWING_DISTANCE = 3
    
        for i in range(1, len(df) - 1):
            prev = df.iloc[i - 1]
            curr = df.iloc[i]
            nxt = df.iloc[i + 1]
    
            # Check if we have proper time data
            if pd.isna(curr['time']) or curr['time'] is None:
                continue
    
            # swing high
            if curr['high'] > prev['high'] and curr['high'] > nxt['high']:
                if not swing_highs or (i - swing_highs[-1]['index']) >= MIN_SWING_DISTANCE:
                    swing_highs.append({
                        'time': curr['time'],
                        'price': float(curr['high']),
                        'index': i
                    })
    
            # swing low
            if curr['low'] < prev['low'] and curr['low'] < nxt['low']:
                if not swing_lows or (i - swing_lows[-1]['index']) >= MIN_SWING_DISTANCE:
                    swing_lows.append({
                        'time': curr['time'],
                        'price': float(curr['low']),
                        'index': i
                    })
    
        return swing_highs, swing_lows
    
    @staticmethod
    def find_aligned_swings(asset1_swings, asset2_swings, max_candle_diff=3, timeframe_minutes=5):
        """Find swings that occur within 3 CANDLES of each other"""
        asset1_swings = sorted(asset1_swings, key=lambda x: x['time'])
        asset2_swings = sorted(asset2_swings, key=lambda x: x['time'])
        aligned_pairs = []
        
        max_time_diff_minutes = max_candle_diff * timeframe_minutes
        
        for swing1 in asset1_swings:
            for swing2 in asset2_swings:
                time_diff = abs((swing1['time'] - swing2['time']).total_seconds() / 60)
                if time_diff <= max_time_diff_minutes:
                    aligned_pairs.append((swing1, swing2, time_diff))
        
        aligned_pairs.sort(key=lambda x: x[2])
        return aligned_pairs
    
    @staticmethod
    def format_swing_time_description(prev_swing, curr_swing, swing_type="low", timing_manager=None):
        """Create time-based description for swing formation with chronological validation"""
        if not prev_swing or not curr_swing:
            return "insufficient swing data"
        
        prev_time = prev_swing['time']
        curr_time = curr_swing['time']
        
        is_chronological = True
        if timing_manager:
            is_chronological = timing_manager.validate_chronological_order(prev_time, curr_time)
        
        prev_time_str = prev_time.strftime('%H:%M')
        curr_time_str = curr_time.strftime('%H:%M')
        
        if swing_type == "high":
            if curr_swing['price'] > prev_swing['price']:
                if is_chronological:
                    return f"made first high at {prev_time_str} and higher high at {curr_time_str}"
                else:
                    return f"⚠️ NON-CHRONOLOGICAL: first high at {prev_time_str} and higher high at {curr_time_str}"
            else:
                if is_chronological:
                    return f"made first high at {prev_time_str} and lower high at {curr_time_str}"
                else:
                    return f"⚠️ NON-CHRONOLOGICAL: first high at {prev_time_str} and lower high at {curr_time_str}"
        else:  # low
            if curr_swing['price'] < prev_swing['price']:
                if is_chronological:
                    return f"made first low at {prev_time_str} and lower low at {curr_time_str}"
                else:
                    return f"⚠️ NON-CHRONOLOGICAL: first low at {prev_time_str} and lower low at {curr_time_str}"
            else:
                if is_chronological:
                    return f"made first low at {prev_time_str} and higher low at {curr_time_str}"
                else:
                    return f"⚠️ NON-CHRONOLOGICAL: first low at {prev_time_str} and higher low at {curr_time_str}"

    @staticmethod
    def validate_interim_price_action(df, first_swing, second_swing, direction="bearish", swing_type="high"):
        """
        Enhanced validation for BOTH swing highs and swing lows
        Checks from first swing time until most current candle to ensure no price violated the protected level
        """
        if df is None or first_swing is None or second_swing is None:
            return False
    
        first_time = first_swing['time']
        second_time = second_swing['time']
    
        # Ensure correct chronological order
        if first_time >= second_time:
            first_swing, second_swing = second_swing, first_swing
            first_time, second_time = second_time, first_time
    
        # Get the MOST RECENT candle time in the dataframe
        most_recent_time = df['time'].max()
        
        # Check ALL candles from first swing time until the most current candle
        validation_candles = df[df['time'] >= first_time]
        
        if validation_candles.empty:
            return True
    
        if direction == "bearish":
            if swing_type == "high":
                protected_level = max(
                    float(first_swing['price']),
                    float(second_swing['price'])
                )
                
                max_validation_level = float(validation_candles['high'].max())
    
                if max_validation_level > protected_level:
                    return False
    
                return True
                
            else:  # swing_type == "low" for bearish (though less common)
                protected_level = max(
                    float(first_swing['price']),
                    float(second_swing['price'])
                )
                
                max_validation_level = float(validation_candles['low'].max())
    
                if max_validation_level > protected_level:
                    return False
    
                return True
    
        else:  # bullish
            if swing_type == "low":
                protected_level = min(
                    float(first_swing['price']),
                    float(second_swing['price'])
                )
    
                min_validation_level = float(validation_candles['low'].min())
    
                if min_validation_level < protected_level:
                    return False
    
                return True
                
            else:  # swing_type == "high" for bullish (though less common)
                protected_level = min(
                    float(first_swing['price']),
                    float(second_swing['price'])
                )
    
                min_validation_level = float(validation_candles['high'].min())
    
                if min_validation_level < protected_level:
                    return False
    
                return True
# ================================
# ENHANCED CRT DETECTOR WITH PSP TRACKING
# ================================

class RobustCRTDetector:
    """Enhanced CRT detector with PSP tracking for triple confluence"""
    
    def __init__(self, timing_manager):
        self.timing_manager = timing_manager
        self.psp_cache = {}  # Cache PSP signals by timeframe
        self.feature_box = None
    
    def calculate_crt_current_candle(self, df, asset1_data, asset2_data, timeframe):
        """Calculate CRT on current candle with PSP check if available"""
        if df is None or not isinstance(df, pd.DataFrame) or df.empty or len(df) < 3:
            return None
        
        current_candle = df[df['is_current'] == True]
        if current_candle.empty:
            return None
            
        current_candle = current_candle.iloc[0]
        
        if not self.timing_manager.is_crt_fresh(current_candle['time']):
            logger.debug("CRT candle too old, skipping")
            return None
        
        complete_candles = df[df['complete'] == True].tail(2)
        if len(complete_candles) < 2:
            return None
            
        c1 = complete_candles.iloc[0]
        c2 = complete_candles.iloc[1]
        c3 = current_candle
        
        try:
            # Calculate ranges and midpoints
            c1_range = float(c1['high']) - float(c1['low'])
            c1_mid = float(c1['low']) + 0.5 * c1_range
            
            c2_range = float(c2['high']) - float(c2['low'])
            c2_low_30 = float(c2['low']) + 0.3 * c2_range  # Low 30% (for bullish)
            c2_top_30 = float(c2['low']) + 0.7 * c2_range  # Top 30% (for bearish)
            
            current_open = float(c3['open'])
            c2_close = float(c2['close'])
            
            # Bullish CRT conditions
            buy_crt = (float(c2['low']) < float(c1['low']) and 
                      float(c2['close']) > float(c1['low']))
            
            # ADDED: c2 cannot close above c1's 50% level
            if buy_crt:
                buy_crt = buy_crt and (c2_close <= c1_mid)
                
                # ADDED: Current open must be above low 30% of c2
                buy_crt = buy_crt and (current_open > c2_low_30)
            
            # Bearish CRT conditions
            sell_crt = (float(c2['high']) > float(c1['high']) and 
                       float(c2['close']) < float(c1['high']))
            
            # ADDED: c2 cannot close below c1's 50% level
            if sell_crt:
                sell_crt = sell_crt and (c2_close >= c1_mid)
                
                # ADDED: Current open must be below top 30% of c2
                sell_crt = sell_crt and (current_open < c2_top_30)
            
            if buy_crt or sell_crt:
                direction = 'bullish' if buy_crt else 'bearish'
                threshold_used = 'low_30%' if direction == 'bullish' else 'top_30%'
                
                # TRY to check for PSP, but if method doesn't exist, skip it
                psp_signal = False
                is_tpd = False
                try:
                    # Try to call the PSP detection method
                    psp_signal = self._detect_psp_for_crt(asset1_data, asset2_data, timeframe, current_candle['time'])
                    
                    # If we got PSP, then check for TPD
                    if psp_signal:
                        # Get corresponding candles for both assets to check TPD
                        c1_asset1 = c1
                        c1_asset2 = None
                        c3_asset1 = c3
                        c3_asset2 = None
                        
                        # Find the corresponding c1 for asset2 (same time as c1_asset1)
                        if not asset2_data.empty:
                            c1_asset2 = asset2_data[asset2_data['time'] == c1_asset1['time']]
                            if not c1_asset2.empty:
                                c1_asset2 = c1_asset2.iloc[0]
                            
                            # Get current candle for asset2
                            c3_asset2_candidate = asset2_data[asset2_data['is_current'] == True]
                            if not c3_asset2_candidate.empty:
                                c3_asset2 = c3_asset2_candidate.iloc[0]
                        
                        # Check TPD conditions if we have PSP and both c1 candles
                        if psp_signal and c1_asset2 is not None and c3_asset2 is not None:
                            # Check if _check_tpd_conditions method exists
                            if hasattr(self, '_check_tpd_conditions'):
                                is_tpd = self._check_tpd_conditions(
                                    asset1_data, asset2_data, 
                                    c1_asset1, c1_asset2, 
                                    c3_asset1, c3_asset2, 
                                    direction
                                )
                except AttributeError:
                    # Method doesn't exist, just continue without PSP/TPD
                    logger.debug("PSP/TPD detection not available, skipping")
                
                logger.info(f"🔷 {direction.upper()} CRT DETECTED: {timeframe} candle at {c3['time'].strftime('%H:%M')}")
                logger.info(f"   PSP: {'✅' if psp_signal else '❌'}, TPD: {'✅' if is_tpd else '❌'}, Threshold: {threshold_used}")
                
                # Create and return CRT signal
                signal_data = {
                    'direction': direction, 
                    'timestamp': c3['time'],
                    'timeframe': timeframe,
                    'signal_key': f"CRT_{timeframe}_{c3['time'].strftime('%m%d_%H%M')}_{direction}",
                    'psp_signal': psp_signal,
                    'is_tpd': is_tpd,
                    'threshold_used': threshold_used,
                    'c2_close_vs_c1_mid': 'above' if c2_close > c1_mid else 'below' if c2_close < c1_mid else 'at',
                }
                
                # Add TPD details only if TPD is detected
                if is_tpd and 'c1_asset2' in locals() and 'c3_asset2' in locals():
                    signal_data['tpd_details'] = {
                        'asset1_c1_close': float(c1_asset1['close']) if 'close' in c1_asset1 else None,
                        'asset2_c1_close': float(c1_asset2['close']) if c1_asset2 and 'close' in c1_asset2 else None,
                        'asset1_c3_open': float(c3_asset1['open']),
                        'asset2_c3_open': float(c3_asset2['open']) if c3_asset2 else None,
                        'c1_mid': c1_mid,
                        'c2_close': c2_close,
                        'c2_low_30': c2_low_30 if direction == 'bullish' else None,
                        'c2_top_30': c2_top_30 if direction == 'bearish' else None,
                    }
                
                return signal_data
                
        except (ValueError, TypeError) as e:
            logger.error(f"Error in CRT calculation: {e}")
            return None
        
        return None
    def _check_smt_confluence_for_crt(self, crt_signal, timeframe):
        """Check for SMT confluence with CRT"""
        if not self.feature_box:
            return None
        
        CRT_SMT_MAPPING = {
            'H4': ['weekly', 'daily'],  # 4hr CRT → Weekly OR Daily SMT
            'H2': ['daily'],           # 1hr CRT → Daily SMT
            'H1': ['daily', '90min']  # 15min CRT → Daily OR 90min SMT
        }
        
        allowed_cycles = CRT_SMT_MAPPING.get(timeframe, [])
        crt_direction = crt_signal['direction']
        
        # Check active SMTs
        for smt_key, smt_feature in self.feature_box.active_features['smt'].items():
            if self.feature_box._is_feature_expired(smt_feature):
                continue
                
            smt_data = smt_feature['smt_data']
            
            # Check if SMT is allowed cycle and same direction
            if (smt_data['cycle'] in allowed_cycles and 
                smt_data['direction'] == crt_direction):
                
                logger.info(f"✅ CRT-SMT CONFLUENCE IN DETECTOR: {timeframe} CRT + {smt_data['cycle']} SMT")
                return smt_data
        
        return None
    
    def _detect_psp_for_crt(self, asset1_data, asset2_data, timeframe, crt_time):
        """Detect PSP on the same timeframe as CRT (look at recent completed candles)"""
        if (asset1_data is None or not isinstance(asset1_data, pd.DataFrame) or asset1_data.empty or
            asset2_data is None or not isinstance(asset2_data, pd.DataFrame) or asset2_data.empty):
            return None
        
        # Look at last 3 completed candles before CRT
        asset1_complete = asset1_data[asset1_data['complete'] == True]
        asset2_complete = asset2_data[asset2_data['complete'] == True]
        
        if asset1_complete.empty or asset2_complete.empty:
            return None
        
        # Get candles that completed BEFORE the CRT candle
        asset1_before_crt = asset1_complete[asset1_complete['time'] < crt_time].tail(3)
        asset2_before_crt = asset2_complete[asset2_complete['time'] < crt_time].tail(3)
        
        # Look for PSP in these candles (most recent first)
        for i in range(len(asset1_before_crt)-1, -1, -1):
            if i >= len(asset2_before_crt):
                continue
                
            asset1_candle = asset1_before_crt.iloc[i]
            asset2_candle = asset2_before_crt.iloc[i]
            
            if asset1_candle['time'] != asset2_candle['time']:
                continue
            
            try:
                asset1_color = 'green' if float(asset1_candle['close']) > float(asset1_candle['open']) else 'red'
                asset2_color = 'green' if float(asset2_candle['close']) > float(asset2_candle['open']) else 'red'
                
                if asset1_color != asset2_color:
                    formation_time = asset1_candle['time']
                    logger.info(f"🎯 PSP DETECTED for CRT: {timeframe} candle at {formation_time.strftime('%H:%M')} - Asset1: {asset1_color}, Asset2: {asset2_color}")
                    return {
                        'timeframe': timeframe,
                        'asset1_color': asset1_color,
                        'asset2_color': asset2_color,
                        'timestamp': datetime.now(NY_TZ),
                        'formation_time': formation_time,
                        'candle_time': formation_time,
                        'candles_ago': len(asset1_before_crt) - i - 1,
                        'signal_key': f"PSP_CRT_{timeframe}_{asset1_color}_{asset2_color}_{formation_time.strftime('%m%d_%H%M')}"
                    }
            except (ValueError, TypeError) as e:
                logger.error(f"Error in PSP calculation: {e}")
                continue
        
        return None
        
    def _check_tpd_conditions(self, asset1_data, asset2_data, c1_asset1, c1_asset2, c3_asset1, c3_asset2, crt_direction):
        """Check TPD (Two-Pair Divergence) conditions"""
        try:
            # Get open and close prices
            c3_open_asset1 = float(c3_asset1['open'])
            c3_open_asset2 = float(c3_asset2['open'])
            c1_close_asset1 = float(c1_asset1['close'])
            c1_close_asset2 = float(c1_asset2['close'])
            
            # Check if both assets are on the same side (which would NOT be TPD)
            both_below = (c3_open_asset1 < c1_close_asset1) and (c3_open_asset2 < c1_close_asset2)
            both_above = (c3_open_asset1 > c1_close_asset1) and (c3_open_asset2 > c1_close_asset2)
            
            # If both are on same side, it's NOT TPD
            if both_below or both_above:
                logger.info(f"❌ NOT TPD: Both assets on same side of C1 close")
                return False
            
            if crt_direction == 'bullish':
                # Bullish TPD: Asset1 open below close, Asset2 open above close
                tpd_conditions = (
                    c3_open_asset1 < c1_close_asset1 and  # Asset1: open below close
                    c3_open_asset2 > c1_close_asset2      # Asset2: open above close
                )
            elif crt_direction == 'bearish':
                # Bearish TPD: Asset1 open above close, Asset2 open below close
                tpd_conditions = (
                    c3_open_asset1 > c1_close_asset1 and  # Asset1: open above close
                    c3_open_asset2 < c1_close_asset2      # Asset2: open below close
                )
            else:
                return False
            
            if tpd_conditions:
                logger.info(f"✅ TPD DETECTED: {crt_direction.upper()} TPD pattern")
                logger.info(f"   Asset1: C3 open {c3_open_asset1} {'<' if crt_direction == 'bullish' else '>'} C1 close {c1_close_asset1}")
                logger.info(f"   Asset2: C3 open {c3_open_asset2} {'>' if crt_direction == 'bullish' else '<'} C1 close {c1_close_asset2}")
                
            return tpd_conditions
            
        except (ValueError, TypeError) as e:
            logger.error(f"Error checking TPD conditions: {e}")
            return False
# ================================
# ULTIMATE SMT DETECTOR WITH ALL QUARTER PAIRS
# ================================

# In UltimateSMTDetector.__init__ method, replace the quarter manager:
class UltimateSMTDetector:
    def __init__(self, pair_config, timing_manager):
        self.smt_history = []
        self.quarter_manager = RobustQuarterManager()
        self.swing_detector = UltimateSwingDetector()
        self.timing_manager = timing_manager
        self.signal_counts = {}
        self.invalidated_smts = set()
        self.pair_config = pair_config
        self.last_smt_candle = None
        
        self.timeframe_minutes = {
            'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30,
            'H1': 60, 'H2': 120, 'H3': 180, 'H4': 240,
            'H6': 360, 'H8': 480, 'H12': 720
        }
        
        self.smt_psp_tracking = {}
        
    def detect_smt_all_cycles(self, asset1_data, asset2_data, cycle_type):
        """Detect SMT using ONLY valid chronological quarter pairs"""
        try:
            if not self.check_data_quality(asset1_data, asset2_data, cycle_type):
                return None
    
            if (asset1_data is None or not isinstance(asset1_data, pd.DataFrame) or asset1_data.empty or
                asset2_data is None or not isinstance(asset2_data, pd.DataFrame) or asset2_data.empty):
                return None
    
            adjacent_pairs = self.quarter_manager.get_adjacent_quarter_pairs(cycle_type)
            last_3_quarters = self.quarter_manager.get_last_three_quarters(cycle_type)
    
            asset1_quarters = self.quarter_manager.group_candles_by_quarters(asset1_data, cycle_type)
            asset2_quarters = self.quarter_manager.group_candles_by_quarters(asset2_data, cycle_type)
    
            if not asset1_quarters or not asset2_quarters:
                return None
    
            valid_pairs = self.filter_valid_quarter_pairs(cycle_type, asset1_quarters, asset2_quarters, adjacent_pairs)
            
            if not valid_pairs:
                return None
    
            results = []
    
            for prev_q, curr_q in valid_pairs:
                smt_result = self._compare_quarters_with_3_candle_tolerance(
                    asset1_quarters[prev_q], asset1_quarters[curr_q],
                    asset2_quarters[prev_q], asset2_quarters[curr_q],
                    cycle_type, prev_q, curr_q
                )
    
                if smt_result:
                    results.append(smt_result)
    
            if not results:
                return None
    
            for smt_result in results:
                if self._is_duplicate_signal(smt_result):
                    continue
                
                signal_key = smt_result['signal_key']
                candle_time = smt_result['candle_time']
    
                self.last_smt_candle = candle_time
                self.signal_counts[signal_key] = self.signal_counts.get(signal_key, 0) + 1
    
                if signal_key not in self.smt_psp_tracking:
                    self.smt_psp_tracking[signal_key] = {
                        'psp_found': False,
                        'check_count': 0,
                        'max_checks': 15,
                        'last_check': datetime.now(NY_TZ),
                        'formation_time': smt_result['formation_time']
                    }
    
                return smt_result
    
            return None
    
        except Exception as e:
            return None

    def debug_quarter_contents_from_dfs(self, cycle_type, asset_name, quarter_dfs):
        """Debug helper for quarter dicts where each value is a pandas DataFrame."""
        for qname, qdf in quarter_dfs.items():
            if qdf is None or (hasattr(qdf, "empty") and qdf.empty):
                continue
    
            try:
                times = pd.to_datetime(qdf['time'])
            except Exception:
                times = pd.to_datetime(qdf.index)
    
            start_t = times.min()
            end_t = times.max()
            count = len(qdf)
    
            highs, lows = self.swing_detector.find_swing_highs_lows(qdf)
    
            # Minimal debug info kept for troubleshooting
            if count > 0:
                pass

    def check_data_quality(self, pair1_data, pair2_data, cycle_type):
        """Check if we have good quality data for analysis"""
        if pair1_data is None or pair2_data is None:
            return False
        
        if len(pair1_data) < 20 or len(pair2_data) < 20:
            return False
        
        p1_start = pair1_data['time'].min()
        p1_end = pair1_data['time'].max()  
        p2_start = pair2_data['time'].min()
        p2_end = pair2_data['time'].max()
        
        time_coverage = (min(p1_end, p2_end) - max(p1_start, p2_start)).total_seconds() / 3600
        
        if time_coverage < 1:
            return False
        
        return True

    def debug_quarter_validation(self, prev_q, curr_q, asset1_prev, asset1_curr, asset2_prev, asset2_curr):
        """Validate quarter chronology for 18:00-start system"""
        pass

    def filter_valid_quarter_pairs(self, cycle_type, asset1_quarters, asset2_quarters, adjacent_pairs):
        """Filter out quarter pairs that have overlapping time ranges"""
        valid_pairs = []
        
        for prev_q, curr_q in adjacent_pairs:
            if (prev_q not in asset1_quarters or curr_q not in asset1_quarters or
                prev_q not in asset2_quarters or curr_q not in asset2_quarters):
                continue
                
            asset1_prev = asset1_quarters[prev_q]
            asset1_curr = asset1_quarters[curr_q]
            asset2_prev = asset2_quarters[prev_q]
            asset2_curr = asset2_quarters[curr_q]
            
            if asset1_prev.empty or asset1_curr.empty or asset2_prev.empty or asset2_curr.empty:
                continue
            
            prev_end = max(asset1_prev['time'].max(), asset2_prev['time'].max())
            curr_start = min(asset1_curr['time'].min(), asset2_curr['time'].min())
            
            time_gap = (curr_start - prev_end).total_seconds() / 3600
            
            if time_gap > 0:
                valid_pairs.append((prev_q, curr_q))
        
        return valid_pairs

    def debug_swing_data_quality(self, swings, label):
        """Debug the quality of swing data"""
        pass

    def run_comprehensive_debug(self, cycle_type, market_data_pair1, market_data_pair2):
        """Run complete debug for a cycle - FIXED VERSION"""
        pass

    def debug_quarter_time_ranges(self, cycle_type, asset1_quarters, asset2_quarters):
        """Debug the actual time ranges of quarters with sequence validation"""
        pass

    def validate_quarter_sequence(self, cycle_type, asset_quarters):
        """Validate that quarters are in proper sequence"""
        pass
    
    def _compare_quarters_with_3_candle_tolerance(self, asset1_prev, asset1_curr, asset2_prev, asset2_curr, cycle_type, prev_q, curr_q):
        """Compare quarters with debug info"""
        try:
            # Strict chronology checkss
            if not asset1_prev.empty and not asset1_curr.empty:
                prev_end = asset1_prev['time'].max()
                curr_start = asset1_curr['time'].min()
                
                if curr_start <= prev_end:
                    return None
            
            if (asset1_prev.empty or asset1_curr.empty or 
                asset2_prev.empty or asset2_curr.empty):
                return None
            
            timeframe = self.pair_config['timeframe_mapping'][cycle_type]
            timeframe_minutes = self.timeframe_minutes.get(timeframe, 5)
            
            # Get asset names from pair_config - using instruments list
            # Handle both new (instruments) and old (pair1/pair2) structure
            if 'instruments' in self.pair_config and len(self.pair_config['instruments']) >= 2:
                asset1_name = self.pair_config['instruments'][0]
                asset2_name = self.pair_config['instruments'][1]
            else:
                # Fallback to old structure
                asset1_name = self.pair_config.get('pair1', 'Asset1')
                asset2_name = self.pair_config.get('pair2', 'Asset2')
        
            asset1_combined = pd.concat([asset1_prev, asset1_curr]).sort_values('time').reset_index(drop=True)
            asset2_combined = pd.concat([asset2_prev, asset2_curr]).sort_values('time').reset_index(drop=True)
        
            a1_prev_H, a1_prev_L = self.swing_detector.find_swing_highs_lows(asset1_prev)
            a1_curr_H, a1_curr_L = self.swing_detector.find_swing_highs_lows(asset1_curr)
            a2_prev_H, a2_prev_L = self.swing_detector.find_swing_highs_lows(asset2_prev)
            a2_curr_H, a2_curr_L = self.swing_detector.find_swing_highs_lows(asset2_curr)
        
            def normalize_time(swings, tz=NY_TZ):
                for s in swings:
                    if not isinstance(s['time'], pd.Timestamp):
                        s['time'] = pd.to_datetime(s['time'])
                    if s['time'].tzinfo is None:
                        s['time'] = s['time'].tz_localize(tz)
                    else:
                        s['time'] = s['time'].tz_convert(tz)
                return swings
        
            a1_prev_H = normalize_time(a1_prev_H, NY_TZ)
            a1_prev_L = normalize_time(a1_prev_L, NY_TZ)
            a1_curr_H = normalize_time(a1_curr_H, NY_TZ)
            a1_curr_L = normalize_time(a1_curr_L, NY_TZ)
            
            a2_prev_H = normalize_time(a2_prev_H, NY_TZ)
            a2_prev_L = normalize_time(a2_prev_L, NY_TZ)
            a2_curr_H = normalize_time(a2_curr_H, NY_TZ)
            a2_curr_L = normalize_time(a2_curr_L, NY_TZ)
        
            def sort_swings(swings):
                return sorted(swings, key=lambda x: x['time'])
        
            a1_prev_H = sort_swings(a1_prev_H)
            a1_prev_L = sort_swings(a1_prev_L)
            a1_curr_H = sort_swings(a1_curr_H)
            a1_curr_L = sort_swings(a1_curr_L)
        
            a2_prev_H = sort_swings(a2_prev_H)
            a2_prev_L = sort_swings(a2_prev_L)
            a2_curr_H = sort_swings(a2_curr_H)
            a2_curr_L = sort_swings(a2_curr_L)
    
            def filter_by_quarter(swings, quarter_df):
                if not swings:
                    return []
                q_start = quarter_df['time'].min()
                q_end = quarter_df['time'].max()
                filtered = [s for s in swings if (s['time'] >= q_start and s['time'] <= q_end)]
                if not filtered:
                    return sorted(swings, key=lambda s: min(abs((s['time'] - q_start).total_seconds()), abs((s['time'] - q_end).total_seconds())))
                return filtered
    
            a1_prev_H = filter_by_quarter(a1_prev_H, asset1_prev)
            a1_prev_L = filter_by_quarter(a1_prev_L, asset1_prev)
            a1_curr_H = filter_by_quarter(a1_curr_H, asset1_curr)
            a1_curr_L = filter_by_quarter(a1_curr_L, asset1_curr)
    
            a2_prev_H = filter_by_quarter(a2_prev_H, asset2_prev)
            a2_prev_L = filter_by_quarter(a2_prev_L, asset2_prev)
            a2_curr_H = filter_by_quarter(a2_curr_H, asset2_curr)
            a2_curr_L = filter_by_quarter(a2_curr_L, asset2_curr)
    
            bearish_smt = self._find_bearish_smt_with_tolerance(
                a1_prev_H, a1_curr_H,
                a2_prev_H, a2_curr_H,
                asset1_combined, asset2_combined, timeframe_minutes  
            )
            
            bullish_smt = self._find_bullish_smt_with_tolerance(
                a1_prev_L, a1_curr_L,
                a2_prev_L, a2_curr_L,
                asset1_combined, asset2_combined, timeframe_minutes
            )
    
            if not bearish_smt and not bullish_smt:
                return None
    
            if bearish_smt:
                direction = 'bearish'
                smt_type = 'Higher Swing High'
                asset1_prev_high, asset1_curr_high, asset2_prev_high, asset2_curr_high = bearish_smt
            
                if asset1_curr_high['time'] <= asset1_prev_high['time']:
                    asset1_prev_high, asset1_curr_high = asset1_curr_high, asset1_prev_high
                if asset2_curr_high['time'] <= asset2_prev_high['time']:
                    asset2_prev_high, asset2_curr_high = asset2_curr_high, asset2_prev_high
            
                formation_time = asset1_curr_high['time']
                asset1_action = self.swing_detector.format_swing_time_description(asset1_prev_high, asset1_curr_high, "high", self.timing_manager)
                asset2_action = self.swing_detector.format_swing_time_description(asset2_prev_high, asset2_curr_high, "high", self.timing_manager)
                critical_level = asset1_curr_high['price']
            
                if not (asset1_prev_high['time'] < asset1_curr_high['time'] and asset2_prev_high['time'] < asset2_curr_high['time']):
                    return None
    
                # Create swings array for bearish case WITH ASSET NAMES
                swings = {
                    'asset1_prev': {
                        'time': asset1_prev_high['time'],
                        'price': asset1_prev_high['price'],
                        'type': 'high',
                        'asset': asset1_name,  # ADDED
                        'quarter': prev_q,
                        'swing_type': 'prev'
                    },
                    'asset1_curr': {
                        'time': asset1_curr_high['time'],
                        'price': asset1_curr_high['price'],
                        'type': 'high',
                        'asset': asset1_name,  # ADDED
                        'quarter': curr_q,
                        'swing_type': 'curr'
                    },
                    'asset2_prev': {
                        'time': asset2_prev_high['time'],
                        'price': asset2_prev_high['price'],
                        'type': 'high',
                        'asset': asset2_name,  # ADDED
                        'quarter': prev_q,
                        'swing_type': 'prev'
                    },
                    'asset2_curr': {
                        'time': asset2_curr_high['time'],
                        'price': asset2_curr_high['price'],
                        'type': 'high',
                        'asset': asset2_name,  # ADDED
                        'quarter': curr_q,
                        'swing_type': 'curr'
                    }
                }
    
                swing_time_key = f"{asset1_prev_high['time'].strftime('%H%M')}_{asset1_curr_high['time'].strftime('%H%M')}"
                swing_times = {
                    'asset1_prev': asset1_prev_high['time'],
                    'asset1_curr': asset1_curr_high['time'],
                    'asset2_prev': asset2_prev_high['time'],
                    'asset2_curr': asset2_curr_high['time']
                }
            else:
                direction = 'bullish'
                smt_type = 'Lower Swing Low'
                asset1_prev_low, asset1_curr_low, asset2_prev_low, asset2_curr_low = bullish_smt
    
                if asset1_curr_low['time'] <= asset1_prev_low['time']:
                    asset1_prev_low, asset1_curr_low = asset1_curr_low, asset1_prev_low
                if asset2_curr_low['time'] <= asset2_prev_low['time']:
                    asset2_prev_low, asset2_curr_low = asset2_curr_low, asset2_prev_low
    
                formation_time = asset1_curr_low['time']
                asset1_action = self.swing_detector.format_swing_time_description(asset1_prev_low, asset1_curr_low, "low", self.timing_manager)
                asset2_action = self.swing_detector.format_swing_time_description(asset2_prev_low, asset2_curr_low, "low", self.timing_manager)
                critical_level = asset1_curr_low['price']
    
                if not (asset1_prev_low['time'] < asset1_curr_low['time'] and asset2_prev_low['time'] < asset2_curr_low['time']):
                    return None
    
                # Create swings array for bullish case WITH ASSET NAMES
                swings = {
                    'asset1_prev': {
                        'time': asset1_prev_low['time'],
                        'price': asset1_prev_low['price'],
                        'type': 'low',
                        'asset': asset1_name,  # ADDED
                        'quarter': prev_q,
                        'swing_type': 'prev'
                    },
                    'asset1_curr': {
                        'time': asset1_curr_low['time'],
                        'price': asset1_curr_low['price'],
                        'type': 'low',
                        'asset': asset1_name,  # ADDED
                        'quarter': curr_q,
                        'swing_type': 'curr'
                    },
                    'asset2_prev': {
                        'time': asset2_prev_low['time'],
                        'price': asset2_prev_low['price'],
                        'type': 'low',
                        'asset': asset2_name,  # ADDED
                        'quarter': prev_q,
                        'swing_type': 'prev'
                    },
                    'asset2_curr': {
                        'time': asset2_curr_low['time'],
                        'price': asset2_curr_low['price'],
                        'type': 'low',
                        'asset': asset2_name,  # ADDED
                        'quarter': curr_q,
                        'swing_type': 'curr'
                    }
                }
    
                swing_time_key = f"{asset1_prev_low['time'].strftime('%H%M')}_{asset1_curr_low['time'].strftime('%H%M')}"
                swing_times = {
                    'asset1_prev': asset1_prev_low['time'],
                    'asset1_curr': asset1_curr_low['time'],
                    'asset2_prev': asset2_prev_low['time'],
                    'asset2_curr': asset2_curr_low['time']
                }
    
            curr_start = asset1_curr['time'].min()
            curr_end = asset1_curr['time'].max()
            if not (curr_start <= formation_time <= curr_end):
                return None
    
            current_time = datetime.now(NY_TZ)
    
            smt_data = {
                'direction': direction,
                'type': smt_type,
                'cycle': cycle_type,
                'quarters': f"{prev_q}-{curr_q}",
                'prev_q': prev_q,
                'curr_q': curr_q,
                'timestamp': current_time,
                'formation_time': formation_time,
                'asset1_action': asset1_action,
                'asset2_action': asset2_action,
                'details': f"Asset1 {asset1_action}, Asset2 {asset2_action}",
                'signal_key': f"SMT_{cycle_type}_{prev_q}_{curr_q}_{direction}_{swing_time_key}",
                'critical_level': critical_level,
                'timeframe': self.pair_config['timeframe_mapping'][cycle_type],
                'swing_times': swing_times,
                'candle_time': formation_time,
                'swings': swings,  # Now with asset names
                'asset_names': {  # Also add as a separate field for easy access
                    'asset1': asset1_name,
                    'asset2': asset2_name
                }
            }
    
            self.smt_history.append(smt_data)
            self._update_signal_count(smt_data['signal_key'])
    
            return smt_data
    
        except Exception as e:
            return None
    
    def _find_bearish_smt_with_tolerance(self, asset1_prev_highs, asset1_curr_highs, asset2_prev_highs, asset2_curr_highs, asset1_combined_data, asset2_combined_data, timeframe_minutes):
        """Find bearish SMT with 3-CANDLE TOLERANCE - VALIDATES BOTH ASSETS"""
        aligned_prev_highs = self.swing_detector.find_aligned_swings(
            asset1_prev_highs, asset2_prev_highs,
            max_candle_diff=3, timeframe_minutes=timeframe_minutes
        )
        
        aligned_curr_highs = self.swing_detector.find_aligned_swings(
            asset1_curr_highs, asset2_curr_highs,
            max_candle_diff=3, timeframe_minutes=timeframe_minutes
        )
        
        for prev_pair in aligned_prev_highs:
            asset1_prev, asset2_prev, prev_time_diff = prev_pair
                    
            for curr_pair in aligned_curr_highs:
                asset1_curr, asset2_curr, curr_time_diff = curr_pair
                
                asset1_hh = asset1_curr['price'] > asset1_prev['price']
                asset2_lh = asset2_curr['price'] <= asset2_prev['price']
                
                asset1_interim_valid = self.swing_detector.validate_interim_price_action(
                    asset1_combined_data, asset1_prev, asset1_curr, "bearish", "high"
                )
                
                asset2_interim_valid = self.swing_detector.validate_interim_price_action(
                    asset2_combined_data, asset2_prev, asset2_curr, "bearish", "high"
                )
                
                if asset1_hh and asset2_lh and asset1_interim_valid and asset2_interim_valid:
                    return (asset1_prev, asset1_curr, asset2_prev, asset2_curr)
        
        return None
    
    def _find_bullish_smt_with_tolerance(self, asset1_prev_lows, asset1_curr_lows, asset2_prev_lows, asset2_curr_lows, asset1_combined_data, asset2_combined_data, timeframe_minutes):
        """Find bullish SMT with 3-CANDLE TOLERANCE - VALIDATES BOTH ASSETS - UPDATED SIGNATURE"""
        aligned_prev_lows = self.swing_detector.find_aligned_swings(
            asset1_prev_lows, asset2_prev_lows,
            max_candle_diff=3, timeframe_minutes=timeframe_minutes
        )
        
        aligned_curr_lows = self.swing_detector.find_aligned_swings(
            asset1_curr_lows, asset2_curr_lows,
            max_candle_diff=3, timeframe_minutes=timeframe_minutes
        )
        
        for prev_pair in aligned_prev_lows:
            asset1_prev, asset2_prev, prev_time_diff = prev_pair
                    
            for curr_pair in aligned_curr_lows:
                asset1_curr, asset2_curr, curr_time_diff = curr_pair
                
                asset1_ll = asset1_curr['price'] < asset1_prev['price']
                asset2_hl = asset2_curr['price'] >= asset2_prev['price']
                
                asset1_interim_valid = self.swing_detector.validate_interim_price_action(
                    asset1_combined_data, asset1_prev, asset1_curr, "bullish", "low"
                )
                
                asset2_interim_valid = self.swing_detector.validate_interim_price_action(
                    asset2_combined_data, asset2_prev, asset2_curr, "bullish", "low"
                )
                
                if asset1_ll and asset2_hl and asset1_interim_valid and asset2_interim_valid:
                    return (asset1_prev, asset1_curr, asset2_prev, asset2_curr)
        
        return None
    
    def check_psp_for_smt(self, smt_data, asset1_data, asset2_data):
        """Check for PSP in past 5 candles for a specific SMT - WITH TIMING VALIDATION"""
        if not smt_data:
            return None
            
        signal_key = smt_data['signal_key']
        
        if signal_key in self.smt_psp_tracking:
            tracking = self.smt_psp_tracking[signal_key]
            tracking['check_count'] += 1
            tracking['last_check'] = datetime.now(NY_TZ)
        
        psp_signal = self._detect_psp_last_n_candles(asset1_data, asset2_data, smt_data['timeframe'], n=5)
        
        if psp_signal:
            if signal_key in self.smt_psp_tracking:
                self.smt_psp_tracking[signal_key]['psp_found'] = True
            
            return psp_signal
        
        return None
    
    def _detect_psp_last_n_candles(self, asset1_data, asset2_data, timeframe, n=5):
        """Look back at last N closed candles for PSP with time-based naming"""
        if (asset1_data is None or not isinstance(asset1_data, pd.DataFrame) or asset1_data.empty or
            asset2_data is None or not isinstance(asset2_data, pd.DataFrame) or asset2_data.empty):
            return None
        
        asset1_complete = asset1_data[asset1_data['complete'] == True]
        asset2_complete = asset2_data[asset2_data['complete'] == True]
        
        if asset1_complete.empty or asset2_complete.empty:
            return None
        
        asset1_recent = asset1_complete.tail(n)
        asset2_recent = asset2_complete.tail(n)
        
        for i in range(len(asset1_recent)-1, -1, -1):
            if i >= len(asset2_recent):
                continue
                
            asset1_candle = asset1_recent.iloc[i]
            asset2_candle = asset2_recent.iloc[i]
            
            if asset1_candle['time'] != asset2_candle['time']:
                continue
            
            try:
                asset1_color = 'green' if float(asset1_candle['close']) > float(asset1_candle['open']) else 'red'
                asset2_color = 'green' if float(asset2_candle['close']) > float(asset2_candle['open']) else 'red'
                
                if asset1_color != asset2_color:
                    formation_time = asset1_candle['time']
                    return {
                        'timeframe': timeframe,
                        'asset1_color': asset1_color,
                        'asset2_color': asset2_color,
                        'timestamp': datetime.now(NY_TZ),
                        'formation_time': formation_time,
                        'candle_time': formation_time,
                        'candles_ago': len(asset1_recent) - i - 1,
                        'signal_key': f"PSP_{timeframe}_{asset1_color}_{asset2_color}_{formation_time.strftime('%m%d_%H%M')}"
                    }
            except (ValueError, TypeError):
                continue
        
        return None
    
    def should_keep_checking_smt(self, smt_data):
        """Determine if we should keep checking for PSP for this SMT"""
        if not smt_data:
            return False
            
        signal_key = smt_data['signal_key']
        
        if signal_key not in self.smt_psp_tracking:
            return False
        
        tracking = self.smt_psp_tracking[signal_key]
        
        if tracking['psp_found'] or tracking['check_count'] >= tracking['max_checks']:
            return False
        
        if signal_key in self.invalidated_smts:
            return False
        
        return True
    
    def check_smt_invalidation(self, smt_data, asset1_data, asset2_data):
        if not smt_data or smt_data['signal_key'] in self.invalidated_smts:
            return True
            
        direction = smt_data['direction']
        critical_level = smt_data['critical_level']
        
        if direction == 'bearish':
            asset1_current_high = asset1_data['high'].max() if not asset1_data.empty else None
            asset2_current_high = asset2_data['high'].max() if not asset2_data.empty else None
            
            if (asset1_current_high and asset1_current_high > critical_level) or \
               (asset2_current_high and asset2_current_high > critical_level):
                self.invalidated_smts.add(smt_data['signal_key'])
                return True
                
        elif direction == 'bullish':
            asset1_current_low = asset1_data['low'].min() if not asset1_data.empty else None
            asset2_current_low = asset2_data['low'].min() if not asset2_data.empty else None
            
            if (asset1_current_low and asset1_current_low < critical_level) or \
               (asset2_current_low and asset2_current_low < critical_level):
                self.invalidated_smts.add(smt_data['signal_key'])
                return True
        
        return False
    
    def _is_duplicate_signal(self, smt_data):
        signal_key = smt_data.get('signal_key')
        if not signal_key:
            return False
           
        if signal_key in self.invalidated_smts:
            return True
           
        count = self.signal_counts.get(signal_key, 0)
        candle_time = smt_data['candle_time']
        is_same_candle = candle_time == self.last_smt_candle
       
        if count >= 1 and is_same_candle:
            return True
           
        return False
    
    def _update_signal_count(self, signal_key):
        self.signal_counts[signal_key] = self.signal_counts.get(signal_key, 0) + 1
        
        if len(self.signal_counts) > 100:
            keys_to_remove = list(self.signal_counts.keys())[:50]
            for key in keys_to_remove:
                del self.signal_counts[key]

    def debug_quarter_contents(self, cycle_type, asset_name, quarter_data):
        """Print detailed debug for a single asset and cycle"""
        pass


# ================================
# ULTIMATE SIGNAL BUILDER WITH CRT+PSP+SMT TRIPLE CONFLUENCE
# ================================

class UltimateSignalBuilder:
    """Ultimate signal builder with CRT+PSP+SMT triple confluence"""
    
    def __init__(self, pair_group, timing_manager):
        self.pair_group = pair_group
        self.timing_manager = timing_manager
        self.active_crt = None
        self.active_smts = {}
        self.psp_for_smts = {}
        self.psp_for_crt = None  # NEW: PSP specifically for CRT
        self.signal_strength = 0
        self.criteria = []
        self.creation_time = datetime.now(NY_TZ)
        self.crt_timeframe = None
        self.status = "SCANNING_ALL"
        
        # Strength tracking with conflict detection
        self.bullish_strength = 0
        self.bearish_strength = 0
        self.dominant_direction = None
        self.has_conflict = False
        
    def add_smt_signal(self, smt_data, psp_signal=None):
        """Add SMT signal with FIXED cycle hierarchy"""
        if not smt_data:
            return False
            
        cycle = smt_data['cycle']
        direction = smt_data['direction']
        
        # STRONG DUPLICATE PREVENTION - Check if we already have this exact SMT
        smt_key = f"{cycle}_{direction}_{smt_data['quarters']}"
        
        # Check if this exact SMT already exists
        if cycle in self.active_smts:
            existing_smt = self.active_smts[cycle]
            existing_key = f"{cycle}_{existing_smt['direction']}_{existing_smt['quarters']}"
            if smt_key == existing_key:
                logger.warning(f"🔄 EXACT DUPLICATE SMT BLOCKED: {smt_key}")
                # Only update PSP if provided and we don't have one
                if psp_signal and cycle not in self.psp_for_smts:
                    self.set_psp_for_smt(cycle, psp_signal)
                return False
        
        # FIXED CYCLE HIERARCHY: Two smaller cycles can override one higher cycle
        hierarchy_issue = self._check_fixed_cycle_hierarchy(cycle, direction)
        if hierarchy_issue:
            logger.warning(f"🔄 CYCLE HIERARCHY: {cycle} {direction} SMT {hierarchy_issue}")
            # Don't return False - allow it to be added for counting
        
        # PREVENT DUPLICATE CRITERIA
        existing_criteria = f"SMT {cycle}: {direction} {smt_data['quarters']}"
        if psp_signal:
            psp_time = psp_signal['formation_time'].strftime('%H:%M')
            existing_criteria = f"SMT {cycle} with PSP {psp_signal['timeframe']} at {psp_time}: {direction} {smt_data['quarters']}"
        
        # Remove any existing duplicate criteria for this cycle
        self.criteria = [c for c in self.criteria if not c.startswith(f"SMT {cycle}:")]
        
        # Check for conflicts with existing SMTs
        if self.active_smts:
            existing_directions = set(smt['direction'] for smt in self.active_smts.values())
            if direction not in existing_directions:
                self.has_conflict = True
                logger.warning(f"⚠️ CONFLICT DETECTED: {direction} {cycle} SMT conflicts with existing {existing_directions}")
        
        # Check direction match with CRT if exists
        if self.active_crt and direction != self.active_crt['direction']:
            logger.info(f"⚠️ SMT direction mismatch: CRT {self.active_crt['direction']} vs SMT {direction}")
            # Don't return False - allow counting for cycle hierarchy
        
        # Store SMT
        self.active_smts[cycle] = smt_data
        self.signal_strength += 2
        
        # Update strength counters
        if direction == 'bullish':
            self.bullish_strength += 1
        else:
            self.bearish_strength += 1
            
        # Update dominant direction and conflict status
        self._update_direction_strength()
        
        # Store PSP if provided
        if psp_signal:
            self.psp_for_smts[cycle] = psp_signal
            self.signal_strength += 1
            psp_time = psp_signal['formation_time'].strftime('%H:%M')
            self.criteria.append(f"SMT {cycle} with PSP {psp_signal['timeframe']} at {psp_time}: {direction} {smt_data['quarters']}")
            logger.info(f"🔷 {self.pair_group}: {cycle} {direction} SMT + PSP CONFIRMED on {psp_signal['timeframe']}!")
            logger.info(f"   📍 {smt_data['asset1_action']}")
            logger.info(f"   📍 {smt_data['asset2_action']}")
        else:
            self.criteria.append(f"SMT {cycle}: {direction} {smt_data['quarters']}")
            logger.info(f"🔷 {self.pair_group}: {cycle} {direction} SMT detected - looking for PSP")
            logger.info(f"   📍 {smt_data['asset1_action']}")
            logger.info(f"   📍 {smt_data['asset2_action']}")
        
        # Check signal readiness with FIXED hierarchy
        self._check_signal_readiness_fixed()
        
        return True
    
    def _check_fixed_cycle_hierarchy(self, new_cycle, new_direction):
        """FIXED CYCLE HIERARCHY: Two smaller cycles can override one higher cycle"""
        new_cycle_level = CYCLE_HIERARCHY[new_cycle]
        
        # Count SMTs in each direction by cycle level
        bullish_by_level = {1: 0, 2: 0, 3: 0, 4: 0}
        bearish_by_level = {1: 0, 2: 0, 3: 0, 4: 0}
        
        for existing_cycle, existing_smt in self.active_smts.items():
            level = CYCLE_HIERARCHY[existing_cycle]
            direction = existing_smt['direction']
            
            if direction == 'bullish':
                bullish_by_level[level] += 1
            else:
                bearish_by_level[level] += 1
        
        # FIXED RULE: Two smaller cycles (level 1+2 or 2+2) can override one higher cycle
        if new_direction == 'bullish':
            # Check if we have enough smaller cycles to override higher bearish cycles
            smaller_bullish = bullish_by_level[1] + bullish_by_level[2]
            higher_bearish = bearish_by_level[3] + bearish_by_level[4]
            
            if smaller_bullish >= 2 and higher_bearish == 1:
                logger.info(f"🔄 CYCLE OVERRIDE: 2 smaller bullish cycles override 1 higher bearish cycle")
                return "OVERRIDES_HIGHER_BEARISH"
                
        elif new_direction == 'bearish':
            # Check if we have enough smaller cycles to override higher bullish cycles
            smaller_bearish = bearish_by_level[1] + bearish_by_level[2]
            higher_bullish = bullish_by_level[3] + bullish_by_level[4]
            
            if smaller_bearish >= 2 and higher_bullish == 1:
                logger.info(f"🔄 CYCLE OVERRIDE: 2 smaller bearish cycles override 1 higher bullish cycle")
                return "OVERRIDES_HIGHER_BULLISH"
        
        return None
    
    def _update_direction_strength(self):
        """Update direction strength and detect conflicts"""
        if self.bullish_strength > self.bearish_strength:
            self.dominant_direction = 'bullish'
            self.has_conflict = False
        elif self.bearish_strength > self.bullish_strength:
            self.dominant_direction = 'bearish'
            self.has_conflict = False
        else:
            self.dominant_direction = None
            self.has_conflict = self.bullish_strength > 0 and self.bearish_strength > 0
            
        if self.has_conflict:
            logger.warning(f"💥 DIRECTION CONFLICT: Bullish {self.bullish_strength} vs Bearish {self.bearish_strength}")
    
    def set_psp_for_smt(self, cycle, psp_signal):
        """Set PSP confirmation for a specific SMT"""
        if cycle in self.active_smts and psp_signal:
            self.psp_for_smts[cycle] = psp_signal
            self.signal_strength += 1
            
            smt = self.active_smts[cycle]
            formation_time = psp_signal['formation_time'].strftime('%H:%M')
            logger.info(f"🎯 {self.pair_group}: PSP CONFIRMED for {cycle} {smt['direction']} SMT on {psp_signal['timeframe']} at {formation_time}!")
            
            # Update criteria - remove old and add new
            self.criteria = [c for c in self.criteria if not c.startswith(f"SMT {cycle}:")]
            self.criteria.append(f"SMT {cycle} with PSP {psp_signal['timeframe']} at {formation_time}: {smt['direction']} {smt['quarters']}")
            
            self._check_signal_readiness_fixed()
            return True
        return False
    
    def _check_signal_readiness_fixed(self):
        """Check signal readiness with FIXED cycle hierarchy - NOW INCLUDES CRT+PSP+SMT"""
        # Count SMTs with PSP
        smts_with_psp = len(self.psp_for_smts)
        total_smts = len(self.active_smts)
        
        logger.info(f"📊 {self.pair_group}: SMTs: {total_smts}, With PSP: {smts_with_psp}, CRT: {self.active_crt is not None}, CRT-PSP: {self.psp_for_crt is not None}")
        
        # NEW RULE 1: CRT + PSP + SMT TRIPLE CONFLUENCE (HIGHEST PRIORITY)
        if (self.active_crt and self.psp_for_crt and total_smts >= 1 and 
            not self.has_conflict and self.dominant_direction == self.active_crt['direction']):
            self.status = f"CRT_PSP_SMT_{self.active_crt['direction'].upper()}_TRIPLE"
            logger.info(f"🎯 {self.pair_group}: ULTIMATE TRIPLE CONFLUENCE! CRT + PSP + SMT {self.active_crt['direction']}")
            return
        
        # FIXED RULE 2: Multiple SMTs in same direction (2+ SMTs) - ALLOW CONFLICTS if we have cycle override
        if total_smts >= 2 and self.dominant_direction:
            smts_in_direction = self.bullish_strength if self.dominant_direction == 'bullish' else self.bearish_strength
            if smts_in_direction >= 2:
                # Check if we have cycle override situation
                if self._has_cycle_override(self.dominant_direction):
                    self.status = f"MULTIPLE_{self.dominant_direction.upper()}_OVERRIDE"
                    logger.info(f"🎯 {self.pair_group}: Multiple {self.dominant_direction} SMTs with cycle override!")
                    return
                elif not self.has_conflict:
                    self.status = f"MULTIPLE_{self.dominant_direction.upper()}_SMTS"
                    logger.info(f"🎯 {self.pair_group}: Multiple {self.dominant_direction} SMTs confirmed!")
                    return
        
        # FIXED RULE 3: SMT + PSP with dominant direction - ALLOW CONFLICTS if cycle override
        if smts_with_psp >= 1 and self.dominant_direction:
            # Check if PSP SMTs are in dominant direction
            psp_in_dominant = any(
                self.active_smts[cycle]['direction'] == self.dominant_direction 
                for cycle in self.psp_for_smts.keys()
            )
            
            if psp_in_dominant:
                if self._has_cycle_override(self.dominant_direction):
                    self.status = f"SMT_PSP_{self.dominant_direction.upper()}_OVERRIDE"
                    logger.info(f"🎯 {self.pair_group}: SMT + PSP {self.dominant_direction} with cycle override!")
                    return
                elif self.bullish_strength >= self.bearish_strength:
                    self.status = f"SMT_PSP_{self.dominant_direction.upper()}"
                    logger.info(f"🎯 {self.pair_group}: SMT + PSP {self.dominant_direction} signal ready!")
                    return
        
        # FIXED RULE 4: CRT + SMT (direction already matched)
        if self.active_crt and total_smts >= 1:
            direction = self.active_crt['direction']
            if self._has_cycle_override(direction) or not self.has_conflict:
                self.status = f"CRT_SMT_{direction.upper()}"
                logger.info(f"🎯 {self.pair_group}: CRT + SMT {direction} signal ready!")
                return
        
        # FIXED RULE 5: CRT + PSP (without SMT) - lower priority but still valid
        if self.active_crt and self.psp_for_crt and not self.has_conflict:
            direction = self.active_crt['direction']
            self.status = f"CRT_PSP_{direction.upper()}"
            logger.info(f"🎯 {self.pair_group}: CRT + PSP {direction} signal ready!")
            return
    
    def _has_cycle_override(self, direction):
        """Check if we have cycle override situation"""
        # Count SMTs by cycle level
        bullish_by_level = {1: 0, 2: 0, 3: 0, 4: 0}
        bearish_by_level = {1: 0, 2: 0, 3: 0, 4: 0}
        
        for cycle, smt in self.active_smts.items():
            level = CYCLE_HIERARCHY[cycle]
            if smt['direction'] == 'bullish':
                bullish_by_level[level] += 1
            else:
                bearish_by_level[level] += 1
        
        if direction == 'bullish':
            # Check if we have 2+ smaller bullish cycles and only 1 higher bearish cycle
            smaller_bullish = bullish_by_level[1] + bullish_by_level[2]
            higher_bearish = bearish_by_level[3] + bearish_by_level[4]
            return smaller_bullish >= 2 and higher_bearish <= 1
            
        else:  # bearish
            # Check if we have 2+ smaller bearish cycles and only 1 higher bullish cycle
            smaller_bearish = bearish_by_level[1] + bearish_by_level[2]
            higher_bullish = bullish_by_level[3] + bullish_by_level[4]
            return smaller_bearish >= 2 and higher_bullish <= 1
    
    def set_crt_signal(self, crt_data, timeframe, psp_signal=None):
        """Set CRT signal from specific timeframe - NOW INCLUDES PSP"""
        if crt_data and not self.active_crt:
            self.active_crt = crt_data
            self.crt_timeframe = timeframe
            self.signal_strength += 3
            
            # Store PSP for CRT if provided
            if psp_signal:
                self.psp_for_crt = psp_signal
                self.signal_strength += 1
                logger.info(f"🎯 {self.pair_group}: CRT with PSP detected on {timeframe}")
            
            # Remove any existing CRT criteria
            self.criteria = [c for c in self.criteria if not c.startswith("CRT ")]
            
            formation_time = crt_data['timestamp'].strftime('%H:%M')
            if self.psp_for_crt:
                psp_time = self.psp_for_crt['formation_time'].strftime('%H:%M')
                self.criteria.append(f"CRT {timeframe} with PSP at {psp_time}: {crt_data['direction']} at {formation_time}")
            else:
                self.criteria.append(f"CRT {timeframe}: {crt_data['direction']} at {formation_time}")
            
            direction = crt_data['direction']
            self.status = f"CRT_{direction.upper()}_WAITING_SMT"
            logger.info(f"🔷 {self.pair_group}: {timeframe} {direction} CRT detected at {formation_time} → Waiting for SMT confirmation")
            
            self._check_signal_readiness_fixed()
            return True
        return False
    
    def is_signal_ready(self):
        """Check if we have complete signal"""
        return self.status.startswith(('CRT_PSP_SMT_', 'MULTIPLE_', 'SMT_PSP_', 'CRT_SMT_', 'CRT_PSP_'))
    
    def has_serious_conflict(self):
        """Check if there's a serious conflict that should block signals"""
        # With cycle override, fewer conflicts are considered serious
        if self._has_cycle_override(self.dominant_direction if self.dominant_direction else 'bullish'):
            return False
            
        return (self.has_conflict and 
                abs(self.bullish_strength - self.bearish_strength) < 2 and
                self.bullish_strength > 0 and self.bearish_strength > 0)
    
    def get_required_cycles(self):
        """Get which cycles to scan based on current signals"""
        return ['monthly', 'weekly', 'daily', '90min']
    
    def get_sleep_cycle(self):
        """Get which cycle to use for sleep timing"""
        if self.active_crt and self.crt_timeframe:
            return None
            
        return '90min'
    
    def get_progress_status(self):
        return self.status
    
    def get_signal_details(self):
        """Get complete signal details - NOW INCLUDES CRT+PSP+SMT"""
        if not self.is_signal_ready() or self.has_serious_conflict():
            return None
            
        # Determine primary direction
        direction = None
        if self.dominant_direction:
            direction = self.dominant_direction
        elif self.active_crt:
            direction = self.active_crt['direction']
        elif self.active_smts:
            first_smt = next(iter(self.active_smts.values()))
            direction = first_smt['direction']
        
        if not direction:
            return None
        
        # Generate unique signal key
        signal_key = f"{self.pair_group}_{direction}_{self.status}_{datetime.now().strftime('%H%M')}"
        
        # Only include SMTs that match the signal direction
        contributing_smts = {cycle: smt for cycle, smt in self.active_smts.items() if smt['direction'] == direction}
        contributing_psps = {cycle: psp for cycle, psp in self.psp_for_smts.items() if cycle in contributing_smts}
        
        # Build signal details
        signal_data = {
            'pair_group': self.pair_group,
            'direction': direction,
            'strength': self.signal_strength,
            'path': self.status,
            'bullish_strength': self.bullish_strength,
            'bearish_strength': self.bearish_strength,
            'dominant_direction': self.dominant_direction,
            'has_conflict': self.has_conflict,
            'criteria': self.criteria.copy(),
            'crt': self.active_crt,
            'psp_for_crt': self.psp_for_crt,  # NEW: Include PSP for CRT
            'psp_smts': contributing_psps,
            'all_smts': contributing_smts,
            'timestamp': datetime.now(NY_TZ),
            'signal_key': signal_key
        }
        
        # Add description
        if 'TRIPLE' in self.status:
            signal_data['description'] = f"ULTIMATE TRIPLE CONFLUENCE: CRT + PSP + SMT {direction}"
        elif 'OVERRIDE' in self.status:
            signal_data['description'] = f"Multiple {direction} SMTs with cycle hierarchy override"
        elif self.status.startswith('MULTIPLE_'):
            signal_data['description'] = f"Multiple {direction} SMTs across different cycles"
        elif self.status.startswith('SMT_PSP_'):
            signal_data['description'] = f"SMT confirmed by PSP on same timeframe"
        elif self.status.startswith('CRT_SMT_'):
            signal_data['description'] = f"CRT momentum with SMT confirmation"
        elif self.status.startswith('CRT_PSP_'):
            signal_data['description'] = f"CRT + PSP confluence"
        
        logger.info(f"🎯 ULTIMATE SIGNAL: {self.pair_group} {direction} via {self.status}")
        
        return signal_data
    
    def is_expired(self):
        expiry_time = timedelta(minutes=30)
        expired = datetime.now(NY_TZ) - self.creation_time > expiry_time
        if expired:
            logger.info(f"⏰ {self.pair_group}: Signal builder expired")
        return expired
    
    def reset(self):
        self.active_crt = None
        self.active_smts = {}
        self.psp_for_smts = {}
        self.psp_for_crt = None  # NEW: Reset PSP for CRT
        self.signal_strength = 0
        self.criteria = []
        self.crt_timeframe = None
        self.creation_time = datetime.now(NY_TZ)
        self.status = "SCANNING_ALL"
        self.bullish_strength = 0
        self.bearish_strength = 0
        self.dominant_direction = None
        self.has_conflict = False
        logger.info(f"🔄 {self.pair_group}: Signal builder reset")


# ================================
# REAL-TIME FEATURE TRACKING SYSTEM
# ================================

class RealTimeFeatureBox:
    def __init__(self, pair_group, timing_manager, telegram_token=None, telegram_chat_id=None, logger=None):
        self.pair_group = pair_group
        self.timing_manager = timing_manager
        self.telegram_token = telegram_token
        self.telegram_chat_id = telegram_chat_id
        
        # Set up logger
        if logger:
            self.logger = logger
        else:
            # Create a default logger
            import logging
            self.logger = logging.getLogger(f'FeatureBox_{pair_group}')
        
        self.active_features = {
            'smt': {},      
            'crt': {},      
            'psp': {},      
            'sd_zone': {},  
            'tpd': {},      
        }
        
        self.sent_signals = {}
        self.signals_sent_count = 0
        self.sent_signal_signatures = {}  
        self.signature_expiry_hours = 24
        
        # SMT CYCLE EXPIRATION TIMES (in minutes)
        self.smt_cycle_expiration = {
            '90min': 120,    # 2 hours = 120 minutes
            'daily': 540,    # 9 hours = 540 minutes
            'weekly': 1500,  # 25 hours = 1500 minutes
            'monthly': 11520 # 8 days = 11520 minutes
        }
        
        # Original expiration times for other features
        self.expiration_times = {
            'smt': 1200,    
            'crt': 120,     
            'psp': 60,      
            'sd_zone': 2880,
            'tpd': 120,     
        }
        
        self.logger.info(f"📦 Feature Box initialized for {pair_group}")
    
    def add_smt(self, smt_data, psp_data):
        """Add SMT feature - assumes it's already validated as fresh"""
        signal_key = smt_data.get('signal_key')
        
        # Check if already exists
        if signal_key in self.active_features['smt']:
            self.logger.debug(f"SMT {signal_key} already exists")
            return False
        
        # Get cycle and expiration
        cycle = smt_data.get('cycle', 'daily')
        expiration_minutes = self.smt_cycle_expiration.get(cycle, self.expiration_times['smt'])
        formation_time = smt_data.get('formation_time')
        
        # Calculate age (should be fresh since we validated)
        current_time = datetime.now(NY_TZ)
        age_minutes = (current_time - formation_time).total_seconds() / 60
        remaining_minutes = expiration_minutes - age_minutes
        
        # Create feature
        feature = {
            'type': 'smt',
            'smt_data': smt_data,
            'psp_data': psp_data,
            'timestamp': current_time,
            'formation_time': formation_time,
            'age_at_addition': age_minutes,
            'expiration': formation_time + timedelta(minutes=expiration_minutes),
            'cycle': cycle,
            'timeframe': smt_data.get('timeframe', 'unknown'),
            'expiration_minutes': expiration_minutes,
            'remaining_minutes': remaining_minutes
        }
        
        # Add to active features
        self.active_features['smt'][signal_key] = feature
        
        # Log addition
        self.logger.info(f"✅ ADDED FRESH SMT: {signal_key}")
        self.logger.info(f"   Cycle: {cycle}, Age: {age_minutes:.0f}min, Remaining: {remaining_minutes:.0f}min")
        self.logger.info(f"   Formation: {formation_time.strftime('%Y-%m-%d %H:%M')}")
        
        return True

    def is_smt_fresh_enough(self, smt_data):
        """Check if SMT is fresh enough to be added (not too old)"""
        formation_time = smt_data.get('formation_time')
        if not formation_time:
            self.logger.warning(f"❌ SMT {smt_data.get('signal_key', 'unknown')} has no formation_time")
            return False
        
        cycle = smt_data.get('cycle', 'daily')
        expiration_minutes = self.smt_cycle_expiration.get(cycle, self.expiration_times['smt'])
        
        current_time = datetime.now(NY_TZ)
        age_minutes = (current_time - formation_time).total_seconds() / 60
        
        if age_minutes > expiration_minutes:
            self.logger.warning(f"⏰ SMT {smt_data.get('signal_key', 'unknown')} is TOO OLD to add")
            self.logger.warning(f"   Cycle: {cycle}, Age: {age_minutes:.0f}min, Max: {expiration_minutes}min")
            self.logger.warning(f"   Formation: {formation_time.strftime('%Y-%m-%d %H:%M')}")
            return False
        
        return True

    
    
    def _is_feature_expired(self, feature, current_price=None):
        """Check if feature is expired based on expiration time OR SMT swing violation"""
        current_time = datetime.now(NY_TZ)
        
        # Check if feature has SMT data
        smt_data = feature.get('smt_data', {}) if isinstance(feature, dict) else {}
        
        # If it's an SMT feature, check for swing violation
        if smt_data or feature.get('type') == 'smt':
            direction = smt_data.get('direction') or feature.get('direction')
            swings = smt_data.get('swings', feature.get('swings', {}))
            
            # Check for swing violation if we have current price
            if current_price is not None and direction and swings:
                if direction == 'bearish':
                    # Find the HIGHEST HIGH of all bearish swing highs
                    highest_swing_high = None
                    highest_swing_time = None
                    
                    for swing_key, swing_info in swings.items():
                        if isinstance(swing_info, dict):
                            # For bearish SMT, we look at swing highs
                            swing_high = swing_info.get('high')
                            if swing_high is not None:
                                if highest_swing_high is None or swing_high > highest_swing_high:
                                    highest_swing_high = swing_high
                                    highest_swing_time = swing_info.get('time')
                    
                    # NEW LOGIC: Track extreme price since highest swing formation
                    if highest_swing_time:
                        # Get or initialize the tracked extreme high
                        tracked_extreme_high = feature.get('tracked_extreme_high', highest_swing_high)
                        
                        # Update if current price is higher than tracked extreme
                        if current_price > tracked_extreme_high:
                            feature['tracked_extreme_high'] = current_price
                            tracked_extreme_high = current_price
                        
                        # If current price goes ABOVE the highest swing high, SMT is violated
                        if current_price > highest_swing_high:
                            self.logger.info(f"❌ SMT Bearish swing violation: Price {current_price:.5f} > Swing High {highest_swing_high:.5f}")
                            return True
                        
                        # NEW: If tracked extreme has gone above highest swing high, SMT is violated
                        if tracked_extreme_high > highest_swing_high:
                            self.logger.info(f"❌ SMT Bearish violation: Extreme price {tracked_extreme_high:.5f} > Swing High {highest_swing_high:.5f}")
                            return True
                        
                    elif direction == 'bullish':
                        # Find the LOWEST LOW of all bullish swing lows
                        lowest_swing_low = None
                        lowest_swing_time = None
                        
                        for swing_key, swing_info in swings.items():
                            if isinstance(swing_info, dict):
                                # For bullish SMT, we look at swing lows
                                swing_low = swing_info.get('low')
                                if swing_low is not None:
                                    if lowest_swing_low is None or swing_low < lowest_swing_low:
                                        lowest_swing_low = swing_low
                                        lowest_swing_time = swing_info.get('time')
                        
                        # NEW LOGIC: Track extreme price since lowest swing formation
                        if lowest_swing_time:
                            # Get or initialize the tracked extreme low
                            tracked_extreme_low = feature.get('tracked_extreme_low', lowest_swing_low)
                            
                            # Update if current price is lower than tracked extreme
                            if current_price < tracked_extreme_low:
                                feature['tracked_extreme_low'] = current_price
                                tracked_extreme_low = current_price
                            
                            # If current price goes BELOW the lowest swing low, SMT is violated
                            if current_price < lowest_swing_low:
                                self.logger.info(f"❌ SMT Bullish swing violation: Price {current_price:.5f} < Swing Low {lowest_swing_low:.5f}")
                                return True
                            
                            # NEW: If tracked extreme has gone below lowest swing low, SMT is violated
                            if tracked_extreme_low < lowest_swing_low:
                                self.logger.info(f"❌ SMT Bullish violation: Extreme price {tracked_extreme_low:.5f} < Swing Low {lowest_swing_low:.5f}")
                                return True
        
        # For SMT features, also check if formation time is too old based on cycles
        if feature.get('type') == 'smt':
            cycle = feature.get('cycle', 'daily')
            formation_time = feature.get('formation_time')
            
            if formation_time:
                # Calculate age based on cycle
                age_minutes = (current_time - formation_time).total_seconds() / 60
                max_age = self.smt_cycle_expiration.get(cycle, 1200)  # Default 20 hours
                
                if age_minutes > max_age:
                    self.logger.info(f"⏰ SMT expired due to age: {age_minutes:.0f} min > {max_age} min max age")
                    return True
        
        # Check general expiration time
        expiration = feature.get('expiration')
        if expiration and current_time > expiration:
            self.logger.info(f"⏰ Feature expired: {current_time} > {expiration}")
            return True
        
        return False
    
    def add_sd_zone(self, zone_data):
        zone_name = zone_data['zone_name']
        
        if zone_name in self.active_features['sd_zone']:
            return False
        
        feature = {
            'type': 'sd_zone',
            'zone_data': zone_data,
            'timestamp': datetime.now(NY_TZ),
            'expiration': datetime.now(NY_TZ) + timedelta(minutes=self.expiration_times['sd_zone'])
        }
        
        self.active_features['sd_zone'][zone_name] = feature
        return True
    
    def get_active_sd_zones(self, direction=None, timeframe=None, asset=None):
        active_zones = []
        
        for zone_name, zone_feature in self.active_features['sd_zone'].items():
            if self._is_feature_expired(zone_feature):
                continue
                
            zone_data = zone_feature['zone_data']
            
            if direction and zone_data.get('direction') != direction:
                continue
            if timeframe and zone_data.get('timeframe') != timeframe:
                continue
            if asset and zone_data.get('asset') != asset:
                continue
            
            active_zones.append(zone_data)
        
        return active_zones
    
    def cleanup_sd_zones(self, market_data):
        zones_to_remove = []
        
        for zone_name, zone_feature in self.active_features['sd_zone'].items():
            if self._is_feature_expired(zone_feature):
                zones_to_remove.append(zone_name)
                continue
                
            zone_data = zone_feature['zone_data']
            asset = zone_data['asset']
            timeframe = zone_data['timeframe']
            
            if asset in market_data and timeframe in market_data[asset]:
                current_data = market_data[asset][timeframe]
                pass
        
        for zone_name in zones_to_remove:
            del self.active_features['sd_zone'][zone_name]
    
    # def add_smt(self, smt_data, psp_data):
    #     signal_key = smt_data.get('signal_key')
    #     if signal_key in self.active_features['smt']:
    #         return False
        
    #     feature = {
    #         'type': 'smt',
    #         'smt_data': smt_data,
    #         'psp_data': psp_data,
    #         'timestamp': datetime.now(NY_TZ),
    #         'expiration': datetime.now(NY_TZ) + timedelta(minutes=self.expiration_times['smt'])
    #     }
        
    #     self.active_features['smt'][signal_key] = feature
    #     return True

    # def _is_feature_expired(self, feature):
    #     feature_type = feature.get('type')
    #     creation_time = feature.get('timestamp')
        
    #     if not creation_time:
    #         return True
        
    #     expiry_hours = {
    #         'smt': 10,    
    #         'crt': 2,
    #         'psp': 1
    #     }
        
    #     current_time = datetime.now(NY_TZ)
    #     hours_passed = (current_time - creation_time).total_seconds() / 3600
        
    #     expired = hours_passed > expiry_hours.get(feature_type, 1)
        
    #     return expired
    def cleanup_expired_features(self, market_data=None):
        """Remove all expired features with detailed logging"""
        current_time = datetime.now(NY_TZ)
        removed_counts = {}
        
        for feature_type, features in self.active_features.items():
            removed_counts[feature_type] = 0
            features_to_remove = []
            
            for feature_key, feature in features.items():
                # Check if feature is expired based on time
                if current_time > feature['expiration']:
                    features_to_remove.append(feature_key)
                    continue
                
                # SPECIAL CASE: SD Zone price invalidation (only if market_data provided)
                if feature_type == 'sd_zone' and market_data:
                    zone_data = feature.get('zone_data', {})
                    asset = zone_data.get('asset')
                    timeframe = zone_data.get('timeframe')
                    direction = zone_data.get('direction', '').lower()
                    
                    if asset and timeframe and direction:
                        # Get current price using the same structure as your system
                        current_price = self._get_current_price_for_zone(
                            asset, timeframe, market_data
                        )
                        
                        if current_price is not None:
                            zone_low = zone_data.get('zone_low')
                            zone_high = zone_data.get('zone_high')
                            
                            # Check price invalidation
                            if direction == 'bearish' and zone_high is not None:
                                # Bearish (supply) zone invalidated if price goes ABOVE zone high
                                if current_price > zone_high:
                                    features_to_remove.append(feature_key)
                                    self.logger.info(
                                        f"🧹 PRICE-INVALIDATED SD Zone (bearish): {feature_key}"
                                    )
                                    self.logger.info(
                                        f"   {asset} {timeframe}: Price {current_price:.5f} > Zone High {zone_high:.5f}"
                                    )
                                    continue
                                    
                            elif direction == 'bullish' and zone_low is not None:
                                # Bullish (demand) zone invalidated if price goes BELOW zone low
                                if current_price < zone_low:
                                    features_to_remove.append(feature_key)
                                    self.logger.info(
                                        f"🧹 PRICE-INVALIDATED SD Zone (bullish): {feature_key}"
                                    )
                                    self.logger.info(
                                        f"   {asset} {timeframe}: Price {current_price:.5f} < Zone Low {zone_low:.5f}"
                                    )
                                    continue
            
            # Remove identified features
            for feature_key in features_to_remove:
                feature = features[feature_key]
                del features[feature_key]
                removed_counts[feature_type] += 1
                
                # Log removal for non-price invalidation cases
                if feature_type == 'smt':
                    smt_data = feature.get('smt_data', {})
                    signal_key = smt_data.get('signal_key', feature_key)
                    formation_time = smt_data.get('formation_time')
                    cycle = smt_data.get('cycle', 'unknown')
                    
                    if formation_time:
                        age_hours = (current_time - formation_time).total_seconds() / 3600
                        self.logger.info(f"🧹 REMOVED expired SMT: {signal_key}")
                        self.logger.info(f"   Cycle: {cycle}, Age: {age_hours:.1f}h")
                        
                elif feature_type == 'sd_zone' and feature_key not in features_to_remove:
                    # This only logs if removed by time (not price invalidation)
                    zone_data = feature.get('zone_data', {})
                    zone_name = zone_data.get('zone_name', feature_key)
                    direction = zone_data.get('direction', 'unknown')
                    timeframe = zone_data.get('timeframe', 'unknown')
                    
                    self.logger.info(f"🧹 REMOVED expired SD Zone: {zone_name}")
                    self.logger.info(f"   Direction: {direction}, Timeframe: {timeframe}")
                    
                else:
                    self.logger.debug(f"🧹 Removed expired {feature_type}: {feature_key}")
        
        # Log summary
        total_removed = sum(removed_counts.values())
        if total_removed > 0:
            summary_parts = [f"{count} {ftype}" for ftype, count in removed_counts.items() if count > 0]
            self.logger.info(f"🧹 Cleanup removed {total_removed} features: {', '.join(summary_parts)}")
        
        return removed_counts
    
    def _get_current_price_for_zone(self, asset, timeframe, market_data):
        """
        Extract current price from market_data for a specific asset/timeframe
        Uses the same structure as your _check_smt_tap_in_sd_zone function
        """
        if not market_data:
            return None
        
        try:
            # Check if asset exists in market_data
            if asset not in market_data:
                self.logger.debug(f"No market data for asset: {asset}")
                return None
            
            # Check if timeframe exists for this asset
            asset_data = market_data.get(asset, {})
            if not asset_data or timeframe not in asset_data:
                self.logger.debug(f"No {timeframe} data for {asset}")
                return None
            
            # Get the price data (should be a DataFrame)
            price_data = asset_data[timeframe]
            
            # Check if DataFrame is empty
            if price_data is None or price_data.empty:
                self.logger.debug(f"Empty price data for {asset} {timeframe}")
                return None
            
            # Get the latest close price (most recent candle)
            # Using iloc[-1] to get last row, same as in your _check_smt_tap_in_sd_zone
            last_row = price_data.iloc[-1]
            
            # Try to get close price (lowercase 'close' as in your code)
            if 'close' in price_data.columns:
                current_price = last_row['close']
            elif 'Close' in price_data.columns:
                current_price = last_row['Close']
            elif 'last' in price_data.columns:
                current_price = last_row['last']
            else:
                # Try to find any price column
                price_columns = [col for col in price_data.columns if col.lower() in ['close', 'last', 'price']]
                if price_columns:
                    current_price = last_row[price_columns[0]]
                else:
                    self.logger.warning(f"No price column found for {asset} {timeframe}")
                    return None
            
            return float(current_price)
            
        except Exception as e:
            self.logger.error(f"Error getting price for {asset} {timeframe}: {str(e)}")
            return None
    
    def add_crt(self, crt_data, psp_data=None):
        if not crt_data:
            return False
            
        signal_key = crt_data['signal_key']
        
        if signal_key in self.active_features['crt']:
            if not self._is_feature_expired(self.active_features['crt'][signal_key]):
                return False
        
        self.active_features['crt'][signal_key] = {
            'crt_data': crt_data,
            'psp_data': psp_data, 
            'timestamp': datetime.now(NY_TZ),
            'expiration': datetime.now(NY_TZ) + timedelta(minutes=self.expiration_times['crt'])
        }
        
        return self._check_immediate_confluence(signal_key, 'crt')
    
    def add_psp(self, psp_data, associated_smt_key=None):
        if not psp_data:
            return False
            
        signal_key = psp_data['signal_key']
        
        self.active_features['psp'][signal_key] = {
            'psp_data': psp_data,
            'associated_smt': associated_smt_key,
            'timestamp': datetime.now(NY_TZ),
            'expiration': datetime.now(NY_TZ) + timedelta(minutes=self.expiration_times['psp'])
        }
        
        if associated_smt_key:
            return self._check_smt_psp_confluence(associated_smt_key, signal_key)
        else:
            return self._check_immediate_confluence(signal_key, 'psp')

    def add_tpd(self, tpd_data):
        if not tpd_data:
            return False
            
        signal_key = tpd_data['signal_key']
        
        if signal_key in self.active_features['tpd']:
            if not self._is_feature_expired(self.active_features['tpd'][signal_key]):
                return False
        
        self.active_features['tpd'][signal_key] = {
            'tpd_data': tpd_data,
            'timestamp': datetime.now(NY_TZ),
            'expiration': datetime.now(NY_TZ) + timedelta(minutes=self.expiration_times['tpd'])
        }
        
        return self._check_tpd_signal(signal_key)
    
    def _check_immediate_confluence(self, new_feature_key, feature_type):
        signals_sent = []
        
        signals_sent.append(self._check_smt_psp_confluence_global())
        signals_sent.append(self._check_crt_smt_confluence())
        signals_sent.append(self._check_crt_psp_confluence())
        signals_sent.append(self._check_multiple_smts_confluence())
        signals_sent.append(self._check_triple_confluence())
        
        if feature_type == 'tpd':
            signals_sent.append(self._check_tpd_signal(new_feature_key))
        
        return any(signals_sent)
    
    def _check_smt_psp_confluence(self, smt_key, psp_key):
        if smt_key not in self.active_features['smt']:
            return False
        if psp_key not in self.active_features['psp']:
            return False
            
        smt_feature = self.active_features['smt'][smt_key]
        psp_feature = self.active_features['psp'][psp_key]
        
        if self._is_feature_expired(smt_feature) or self._is_feature_expired(psp_feature):
            return False
        
        smt_data = smt_feature['smt_data']
        psp_data = psp_feature['psp_data']
        
        if smt_data.get('timeframe') != psp_data.get('timeframe'):
            return False
        
        signal_data = {
            'pair_group': self.pair_group,
            'direction': smt_data['direction'],
            'confluence_type': 'SMT_PSP_IMMEDIATE',
            'smt': smt_data,
            'psp': psp_data,
            'timestamp': datetime.now(NY_TZ),
            'signal_key': f"SMT_PSP_{smt_key}_{psp_key}",
            'description': f"IMMEDIATE: {smt_data['cycle']} {smt_data['direction']} SMT + PSP confirmed"
        }
        
        if self._send_immediate_signal(signal_data):
            self._remove_feature('smt', smt_key)
            self._remove_feature('psp', psp_key)
            return True
            
        return False
    
    def _check_smt_psp_confluence_global(self):
        signals_sent = 0
        fvg_detector = FVGDetector()
        if not hasattr(self, 'instruments') or not isinstance(self.instruments, (list, tuple)):
            return None
    
        fvgs_per_asset = {inst: [] for inst in self.instruments}
        for tf in ['M15', 'H1', 'H4', 'D']:
            for inst in self.instruments:
                data = self.market_data[inst].get(tf)
                if data is not None and not data.empty:
                    new_fvgs = fvg_detector.scan_tf(data, tf, inst)
                    fvgs_per_asset[inst].extend(new_fvgs)
        
        for smt_key, smt_feature in list(self.active_features['smt'].items()):
            if self._is_feature_expired(smt_feature):
                continue
            smt_data = smt_feature['smt_data']
            if smt_feature['psp_data']:
                tapped = False
                matched_fvg = None
                for inst in self.instruments:
                    for fvg in [f for f in fvgs_per_asset[inst] if f['tf'] == smt_data['timeframe'] and f['direction'] == smt_data['direction']]:
                        is_tapped = self._check_smt_second_swing_in_fvg(smt_data, inst, fvg['fvg_low'], fvg['fvg_high'], smt_data['direction'])
                        if is_tapped:
                            tapped = True
                            matched_fvg = fvg
                            break
                    if tapped:
                        break
                
                if tapped:
                    return True
                
                if smt_data['cycle'] == '90min':
                    if not self._has_daily_smt_confirmation(smt_data['direction']):
                        continue
                
                signal_key = f"SMT_PSP_PRE_{smt_key}"
                if self.timing_manager.is_duplicate_signal(signal_key, self.pair_group, cooldown_minutes=240):
                    continue
                
                signal_data = {
                    'pair_group': self.pair_group,
                    'direction': smt_data['direction'],
                    'confluence_type': 'SMT_PSP_PRE_CONFIRMED',
                    'smt': smt_data,
                    'psp': smt_feature['psp_data'],
                    'timestamp': datetime.now(NY_TZ),
                    'signal_key': signal_key,
                    'description': f"PRE-CONFIRMED: {smt_data['cycle']} {smt_data['direction']} SMT + PSP"
                }
                
                if self._send_immediate_signal(signal_data):
                    signals_sent += 1
        
        return signals_sent > 0
    
    def _has_daily_smt_confirmation(self, direction):
        for smt_key, smt_feature in self.active_features['smt'].items():
            if self._is_feature_expired(smt_feature):
                continue
                
            smt_data = smt_feature['smt_data']
            if smt_data['cycle'] == 'daily' and smt_data['direction'] == direction:
                return True
        
        return False
    
    def _check_crt_smt_confluence(self):
        signals_sent = 0
        
        for crt_key, crt_feature in list(self.active_features['crt'].items()):
            if self._is_feature_expired(crt_feature):
                continue
                
            crt_data = crt_feature['crt_data']
            
            for smt_key, smt_feature in list(self.active_features['smt'].items()):
                if self._is_feature_expired(smt_feature):
                    continue
                    
                smt_data = smt_feature['smt_data']
                
                if crt_data['direction'] == smt_data['direction']:
                    signal_data = {
                        'pair_group': self.pair_group,
                        'direction': crt_data['direction'],
                        'confluence_type': 'CRT_SMT_IMMEDIATE',
                        'crt': crt_data,
                        'smt': smt_data,
                        'timestamp': datetime.now(NY_TZ),
                        'signal_key': f"CRT_SMT_{crt_key}_{smt_key}",
                        'description': f"IMMEDIATE: {crt_data['timeframe']} CRT + {smt_data['cycle']} SMT"
                    }
                    
                    if self._send_immediate_signal(signal_data):
                        self._remove_feature('crt', crt_key)
                        self._remove_feature('smt', smt_key)
                        signals_sent += 1
                        break
        
        return signals_sent > 0

    def _check_multiple_smts_confluence(self):
        signals_sent = 0
        current_time = datetime.now(NY_TZ)
        
        bullish_smts = []
        bearish_smts = []
        
        for smt_key, smt_feature in list(self.active_features['smt'].items()):
            if self._is_feature_expired(smt_feature):
                continue
                
            smt_data = smt_feature['smt_data']
            
            if smt_data['direction'] == 'bullish':
                bullish_smts.append((smt_key, smt_data))
            elif smt_data['direction'] == 'bearish':
                bearish_smts.append((smt_key, smt_data))
        
        def _get_unique_cycle_smts(smt_list):
            cycle_smts = {}
            
            for smt_key, smt_data in smt_list:
                cycle = smt_data.get('cycle', 'Unknown')
                if cycle not in cycle_smts:
                    cycle_smts[cycle] = (smt_key, smt_data)
            
            return list(cycle_smts.values())
        
        unique_bullish = _get_unique_cycle_smts(bullish_smts)
        
        if len(unique_bullish) >= 2:
            smt_details = []
            for smt_key, smt_data in unique_bullish:
                smt_details.append({
                    'cycle': smt_data.get('cycle', 'Unknown'),
                    'quarters': smt_data.get('quarters', ''),
                    'timeframe': smt_data.get('timeframe', 'Unknown'),
                    'asset1_action': smt_data.get('asset1_action', ''),
                    'asset2_action': smt_data.get('asset2_action', ''),
                    'signal_key': smt_data.get('signal_key', '')
                })
            
            signal_data = {
                'pair_group': self.pair_group,
                'direction': 'bullish',
                'confluence_type': 'MULTIPLE_SMTS_BULLISH_DIFFERENT_CYCLES',
                'multiple_smts': smt_details,
                'smt_count': len(unique_bullish),
                'cycle_count': len(unique_bullish),
                'timestamp': current_time,
                'signal_key': f"MULTI_SMT_BULLISH_DIFF_CYCLES_{current_time.strftime('%H%M%S')}",
                'description': f"MULTIPLE BULLISH SMTs: {len(unique_bullish)} SMTs from {len(unique_bullish)} different cycles"
            }
            
            if self._send_immediate_signal(signal_data):
                signals_sent += 1
        
        unique_bearish = _get_unique_cycle_smts(bearish_smts)
        
        if len(unique_bearish) >= 2:
            smt_details = []
            for smt_key, smt_data in unique_bearish:
                smt_details.append({
                    'cycle': smt_data.get('cycle', 'Unknown'),
                    'quarters': smt_data.get('quarters', ''),
                    'timeframe': smt_data.get('timeframe', 'Unknown'),
                    'asset1_action': smt_data.get('asset1_action', ''),
                    'asset2_action': smt_data.get('asset2_action', ''),
                    'signal_key': smt_data.get('signal_key', '')
                })
            
            signal_data = {
                'pair_group': self.pair_group,
                'direction': 'bearish',
                'confluence_type': 'MULTIPLE_SMTS_BEARISH_DIFFERENT_CYCLES',
                'multiple_smts': smt_details,
                'smt_count': len(unique_bearish),
                'cycle_count': len(unique_bearish),
                'timestamp': current_time,
                'signal_key': f"MULTI_SMT_BEARISH_DIFF_CYCLES_{current_time.strftime('%H%M%S')}",
                'description': f"MULTIPLE BEARISH SMTs: {len(unique_bearish)} SMTs from {len(unique_bearish)} different cycles"
            }
            
            if self._send_immediate_signal(signal_data):
                signals_sent += 1
        
        return signals_sent > 0
    
    def _check_triple_confluence(self):
        for crt_key, crt_feature in list(self.active_features['crt'].items()):
            if self._is_feature_expired(crt_feature):
                continue
                
            crt_data = crt_feature['crt_data']
            
            if not crt_feature['psp_data']:
                continue
                
            for smt_key, smt_feature in list(self.active_features['smt'].items()):
                if self._is_feature_expired(smt_feature):
                    continue
                    
                smt_data = smt_feature['smt_data']
                
                if crt_data['direction'] == smt_data['direction']:
                    signal_data = {
                        'pair_group': self.pair_group,
                        'direction': crt_data['direction'],
                        'confluence_type': 'TRIPLE_CONFLUENCE_IMMEDIATE',
                        'crt': crt_data,
                        'crt_psp': crt_feature['psp_data'],
                        'smt': smt_data,
                        'timestamp': datetime.now(NY_TZ),
                        'signal_key': f"TRIPLE_{crt_key}_{smt_key}",
                        'description': f"ULTIMATE TRIPLE: CRT + PSP + {smt_data['cycle']} SMT"
                    }
                    
                    if self._send_immediate_signal(signal_data):
                        self._remove_feature('crt', crt_key)
                        self._remove_feature('smt', smt_key)
                        return True
        
        return False

    def _check_tpd_signal(self, tpd_key):
        if tpd_key not in self.active_features['tpd']:
            return False
            
        tpd_feature = self.active_features['tpd'][tpd_key]
        
        if self._is_feature_expired(tpd_feature):
            return False
        
        tpd_data = tpd_feature['tpd_data']
        
        signal_data = {
            'pair_group': self.pair_group,
            'direction': tpd_data['direction'],
            'confluence_type': 'TPD_IMMEDIATE',
            'tpd': tpd_data,
            'timestamp': datetime.now(NY_TZ),
            'signal_key': tpd_data['signal_key'],
            'description': f"TPD: {tpd_data['timeframe']} {tpd_data['direction']} - Two-Pair Divergence with PSP"
        }
        
        if self._send_immediate_signal(signal_data):
            return True
        
        return False
    
    def _send_immediate_signal(self, signal_data):
        signal_key = signal_data['signal_key']
        
        if not self._validate_signal_before_sending(signal_data):
            return False
        
        if self.timing_manager.is_duplicate_signal(signal_key, self.pair_group, cooldown_minutes=30):
            return False
        
        message = self._format_immediate_signal_message(signal_data)
        success = send_telegram(message, self.telegram_token, self.telegram_chat_id)
        
        if success:
            self.sent_signals[signal_key] = datetime.now(NY_TZ)
            return True
        else:
            return False

    def _validate_signal_before_sending(self, signal_data):
        confluence_type = signal_data.get('confluence_type', '')
        
        if confluence_type.startswith('MULTIPLE_SMTS'):
            multiple_smts = signal_data.get('multiple_smts', [])
            if not multiple_smts:
                return False
            
            valid_smts = [smt for smt in multiple_smts 
                         if smt.get('cycle') and smt.get('quarters')]
            
            if len(valid_smts) < 2:
                return False
                
            return True
        
        elif confluence_type == 'SMT_PSP_PRE_CONFIRMED':
            if not signal_data.get('smt') or not signal_data.get('psp'):
                return False
            return True
        
        elif confluence_type == 'CRT_SMT_IMMEDIATE':
            if not signal_data.get('crt') or not signal_data.get('smt'):
                return False
            return True
        
        return True
    
    def _validate_signal_content(self, signal_data):
        if 'multiple_smts' in signal_data:
            smts = signal_data['multiple_smts']
            if len(smts) < 2:
                return False
            
            for smt in smts:
                if not smt.get('cycle') or not smt.get('quarters'):
                    return False
            return True
        
        elif 'smt' in signal_data:
            smt = signal_data['smt']
            return bool(smt.get('cycle') and smt.get('quarters'))
        
        elif 'crt' in signal_data:
            crt = signal_data['crt']
            return bool(crt.get('timeframe') and crt.get('direction'))
        
        return False
    
    def _check_crt_psp_confluence(self):
        signals_sent = 0
        
        for crt_key, crt_feature in list(self.active_features['crt'].items()):
            if self._is_feature_expired(crt_feature):
                continue
                
            crt_data = crt_feature['crt_data']
            
            if not crt_feature['psp_data']:
                continue
                
            signal_data = {
                'pair_group': self.pair_group,
                'direction': crt_data['direction'],
                'confluence_type': 'CRT_PSP_IMMEDIATE',
                'crt': crt_data,
                'crt_psp': crt_feature['psp_data'],
                'timestamp': datetime.now(NY_TZ),
                'signal_key': f"CRT_PSP_{crt_key}",
                'description': f"IMMEDIATE: {crt_data['timeframe']} CRT + PSP"
            }
            
            if self._send_immediate_signal(signal_data):
                self._remove_feature('crt', crt_key)
                signals_sent += 1
        
        return signals_sent > 0
    
    def _format_immediate_signal_message(self, signal_data):
        direction = signal_data['direction'].upper()
        confluence_type = signal_data['confluence_type']
        description = signal_data['description']
        
        message = f"⚡ *IMMEDIATE SIGNAL* ⚡\n\n"
        message += f"*Pair Group:* {self.pair_group.replace('_', ' ').title()}\n"
        message += f"*Direction:* {direction}\n"
        message += f"*Confluence:* {confluence_type}\n"
        message += f"*Description:* {description}\n\n"
        
        if 'smt' in signal_data and signal_data['smt']:
            smt = signal_data['smt']
            message += f"*SMT Details:*\n"
            message += f"• Cycle: {smt.get('cycle', 'Unknown')} {smt.get('quarters', '')}\n"
            if smt.get('asset1_action'):
                message += f"• {smt['asset1_action']}\n"
            if smt.get('asset2_action'):
                message += f"• {smt['asset2_action']}\n"
            message += f"\n"
        
        if 'multiple_smts' in signal_data and signal_data['multiple_smts']:
            valid_smts = [smt for smt in signal_data['multiple_smts'] 
                         if smt.get('cycle') and smt.get('quarters')]
            
            if valid_smts:
                message += f"*Multiple SMTs Details ({len(valid_smts)}):*\n"
                for i, smt in enumerate(valid_smts, 1):
                    message += f"• {smt['cycle']} {smt.get('quarters', '')}\n"
                    if smt.get('asset1_action'):
                        message += f"  - {smt['asset1_action']}\n"
                    if smt.get('asset2_action'):
                        message += f"  - {smt['asset2_action']}\n"
                message += f"\n"
        
        if 'crt' in signal_data and signal_data['crt']:
            crt = signal_data['crt']
            message += f"*CRT Details:*\n"
            message += f"• Timeframe: {crt.get('timeframe', 'Unknown')}\n"
            if crt.get('timestamp'):
                message += f"• Time: {crt['timestamp'].strftime('%H:%M')}\n"
            message += f"\n"
        
        if 'psp' in signal_data and signal_data['psp']:
            psp = signal_data['psp']
            message += f"*PSP Details:*\n"
            message += f"• Timeframe: {psp.get('timeframe', 'Unknown')}\n"
            message += f"• Colors: {psp.get('asset1_color', 'Unknown')}/{psp.get('asset2_color', 'Unknown')}\n"
            if psp.get('formation_time'):
                message += f"• Time: {psp['formation_time'].strftime('%H:%M')}\n"
            message += f"\n"
        
        if 'crt_psp' in signal_data and signal_data['crt_psp']:
            crt_psp = signal_data['crt_psp']
            message += f"*CRT PSP Details:*\n"
            message += f"• Timeframe: {crt_psp.get('timeframe', 'Unknown')}\n"
            message += f"• Colors: {crt_psp.get('asset1_color', 'Unknown')}/{crt_psp.get('asset2_color', 'Unknown')}\n"
            if crt_psp.get('formation_time'):
                message += f"• Time: {crt_psp['formation_time'].strftime('%H:%M')}\n"
            message += f"\n"
        
        if 'tpd' in signal_data and signal_data['tpd']:
            tpd = signal_data['tpd']
            message += f"*TPD Details:*\n"
            message += f"• Timeframe: {tpd.get('timeframe', 'Unknown')}\n"
            message += f"• Direction: {tpd.get('direction', 'Unknown')}\n"
            if tpd.get('tpd_details'):
                details = tpd['tpd_details']
                message += f"• Asset1: C3 open {details.get('asset1_c3_open', 'N/A'):.4f} vs C1 close {details.get('asset1_c1_close', 'N/A'):.4f}\n"
                message += f"• Asset2: C3 open {details.get('asset2_c3_open', 'N/A'):.4f} vs C1 close {details.get('asset2_c1_close', 'N/A'):.4f}\n"
            if tpd.get('timestamp'):
                message += f"• Time: {tpd['timestamp'].strftime('%H:%M')}\n"
            message += f"• Setup: TPD (No SMT Required)\n"
            message += f"\n"
        
        message += f"*Detection Time:* {signal_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}\n"
        message += f"*Latency:* IMMEDIATE\n\n"
        message += f"#ImmediateSignal #{self.pair_group} #{direction}"
        
        return message

    # def _is_feature_expired(self, feature):
    #     return datetime.now(NY_TZ) > feature['expiration']
    
    def _remove_feature(self, feature_type, feature_key):
        if feature_key in self.active_features[feature_type]:
            del self.active_features[feature_type][feature_key]
    
    # def cleanup_expired_features(self):
    #     current_time = datetime.now(NY_TZ)
        
    #     for feature_type in self.active_features:
    #         for feature_key, feature in list(self.active_features[feature_type].items()):
    #             if current_time > feature['expiration']:
    #                 del self.active_features[feature_type][feature_key]
    
    def get_active_features_summary(self):
        summary = {
            'smt_count': len(self.active_features['smt']),
            'crt_count': len(self.active_features['crt']),
            'psp_count': len(self.active_features['psp']),
            'active_smts': [],
            'active_crts': []
        }
        
        for smt_key, smt_feature in self.active_features['smt'].items():
            if not self._is_feature_expired(smt_feature):
                smt_data = smt_feature['smt_data']
                summary['active_smts'].append({
                    'cycle': smt_data['cycle'],
                    'direction': smt_data['direction'],
                    'quarters': smt_data['quarters'],
                    'has_psp': smt_feature['psp_data'] is not None
                })
        
        for crt_key, crt_feature in self.active_features['crt'].items():
            if not self._is_feature_expired(crt_feature):
                crt_data = crt_feature['crt_data']
                summary['active_crts'].append({
                    'timeframe': crt_data['timeframe'],
                    'direction': crt_data['direction'],
                    'has_psp': crt_feature['psp_data'] is not None
                })

        for tpd_key, tpd_feature in self.active_features['tpd'].items():
            if not self._is_feature_expired(tpd_feature):
                tpd_data = tpd_feature['tpd_data']
                summary['active_tpds'].append({
                    'timeframe': tpd_data['timeframe'],
                    'direction': tpd_data['direction'],
                    'has_psp': tpd_data.get('psp_signal') is not None
                })
        
        return summary

    def _create_signal_signature(self, signal_data):
        if signal_data['confluence_type'] == 'SMT_PSP_PRE_CONFIRMED':
            smt_key = signal_data['smt']['signal_key']
            psp_timeframe = signal_data['psp']['timeframe']
            direction = signal_data['direction']
            signature = f"SMT_PSP_{smt_key}_{psp_timeframe}_{direction}"
        
        elif signal_data['confluence_type'].startswith('MULTIPLE_SMTS'):
            smt_keys = []
            for smt in signal_data['multiple_smts']:
                smt_id = f"{smt['cycle']}_{smt['quarters']}"
                smt_keys.append(smt_id)
            
            smt_keys.sort()
            direction = signal_data['direction']
            signature = f"MULTI_SMT_{'_'.join(smt_keys)}_{direction}"
        
        elif signal_data['confluence_type'] == 'CRT_SMT_IMMEDIATE':
            crt_key = signal_data['crt']['signal_key']
            smt_key = signal_data['smt']['signal_key']
            signature = f"CRT_SMT_{crt_key}_{smt_key}"
        
        else:
            signature = signal_data['signal_key']
        
        return signature
    
    def _is_duplicate_signal_signature(self, signal_data):
        signature = self._create_signal_signature(signal_data)
        current_time = datetime.now(NY_TZ)
        
        self._cleanup_old_signatures()
        
        if signature in self.sent_signal_signatures:
            last_sent = self.sent_signal_signatures[signature]
            hours_since_sent = (current_time - last_sent).total_seconds() / 3600
            
            if hours_since_sent < self.signature_expiry_hours:
                return True
        
        self.sent_signal_signatures[signature] = current_time
        return False
    
    def _cleanup_old_signatures(self):
        current_time = datetime.now(NY_TZ)
        expired_signatures = []
        
        for signature, sent_time in self.sent_signal_signatures.items():
            hours_since_sent = (current_time - sent_time).total_seconds() / 3600
            if hours_since_sent >= self.signature_expiry_hours:
                expired_signatures.append(signature)
        
        for signature in expired_signatures:
            del self.sent_signal_signatures[signature]





class FVGDetector:
    def __init__(self, min_gap_pct: float = 0.05):
        self.min_gap_pct = min_gap_pct
        self.active_fvgs = {}  # tf -> [fvgs]
        self.invalidate_std_mult = 4.0
        self.fvg_expiry_hours = 48884  

    def scan_tf(self, df, tf, asset):
        if tf not in ['M15', 'H1','H2', 'H3', 'H4', 'D'] or len(df) < 39:
            return []

        # ✅ NEW: Filter for CLOSED candles only
        if 'complete' in df.columns:
            # Get only closed candles
            df = df[df['complete'] == True].copy()
        recent = df.tail(70).reset_index(drop=True)
        fvgs = []
        for i in range(2, len(recent)):
            a, b, c = recent.iloc[i-2], recent.iloc[i-1], recent.iloc[i]
            body_size = c['high'] - c['low']

            # Bullish
            if b['low'] < a['high'] and c['low'] > a['high']:
                gap_low, gap_high = a['high'], c['low']
                gap_size = gap_high - gap_low
                if gap_size >= body_size * self.min_gap_pct:
                    fvg = self._create_fvg('bullish', gap_low, gap_high, c['time'], asset, tf, b)
                    post_df = recent.iloc[i+1:] if i+1 < len(recent) else pd.DataFrame()
                    if not self._is_invalidated(fvg, post_df):
                        fvgs.append(fvg)

            # Bearish
            if b['high'] > a['low'] and c['high'] < a['low']:
                gap_low, gap_high = c['high'], a['low']
                gap_size = gap_high - gap_low
                if gap_size >= body_size * self.min_gap_pct:
                    fvg = self._create_fvg('bearish', gap_low, gap_high, c['time'], asset, tf, b)
                    post_df = recent.iloc[i+1:] if i+1 < len(recent) else pd.DataFrame()
                    if not self._is_invalidated(fvg, post_df):
                        fvgs.append(fvg)

        # Merge
        if tf not in self.active_fvgs:
            self.active_fvgs[tf] = fvgs
        else:
            self.active_fvgs[tf] = self._merge_fvgs(self.active_fvgs[tf], fvgs)
        
        # Prune invalidated/over-mitigated (post-formation only, no expiry)
        active = []
        for f in self.active_fvgs.get(tf, []):
            post_df = df[df['time'] > f['formation_time']]
            if not self._is_invalidated(f, post_df) and not self._is_over_mitigated(f, post_df):
                active.append(f)
        self.active_fvgs[tf] = active
        return active

    def _create_fvg(self, direction, low, high, time, asset, tf, candle_b):
        b_ohlc = np.array([candle_b['open'], candle_b['high'], candle_b['low'], candle_b['close']])
        std_b = np.std(b_ohlc)
        return {
            'direction': direction, 'fvg_low': low, 'fvg_high': high, 'formation_time': time,
            'asset': asset, 'tf': tf, 
            'candle_b_low': candle_b['low'], 'candle_b_high': candle_b['high'], 'candle_b_std': std_b,
            'taps': 0, 'in_zone_candles': 0, 'is_hp': False
        }

    def _is_invalidated(self, fvg, post_df):
        """Your rule: Bull: close < B low OR close > B high + 4*std. Flip bear."""
        threshold = self.invalidate_std_mult * fvg['candle_b_std']
        for _, candle in post_df.iterrows():
            close = candle['close']
            if fvg['direction'] == 'bullish':
                if close < fvg['fvg_low']:  # Breach
                    return True
                if close > (fvg['candle_b_high'] + threshold):  # Over-extend up
                    return True
            else:  # Bearish
                if close > fvg['fvg_high']:  # Breach
                    return True
                if close < (fvg['candle_b_low'] - threshold):  # Over-extend down
                    return True
        return False

    def _is_over_mitigated(self, fvg, recent_df):
        """6+ candles in zone *post-formation* only."""
        if recent_df is None or recent_df.empty:
            return False
        
        # Filter post-formation if not already
        post_df = recent_df[recent_df['time'] > fvg['formation_time']] if 'time' in recent_df.columns else recent_df
        in_count = 0
        for _, candle in post_df.iterrows():
            if fvg['direction'] == 'bullish' and candle['low'] <= fvg['fvg_high']:
                in_count += 1
            if fvg['direction'] == 'bearish' and candle['high'] >= fvg['fvg_low']:
                in_count += 1
        fvg['in_zone_candles'] = in_count
        return in_count >= 6

    def _merge_fvgs(self, old, new):
        all_f = old + new
        seen = set((f['formation_time'], round(f['fvg_low'], 4), round(f['fvg_high'], 4)) for f in all_f)
        return [f for f in all_f if (f['formation_time'], round(f['fvg_low'], 4), round(f['fvg_high'], 4)) in seen]

    def check_cross_asset_hp(self, fvgs1, fvgs2, tf):
        for f1 in fvgs1:
            f1['is_hp'] = True
            for f2 in fvgs2:
                if abs((f1['formation_time'] - f2['formation_time']).total_seconds()) < 300:
                    f1['is_hp'] = False
                    break
        for f2 in fvgs2:
            f2['is_hp'] = True
            for f1 in fvgs1:
                if abs((f2['formation_time'] - f1['formation_time']).total_seconds()) < 300:
                    f2['is_hp'] = False
                    break



class HybridTimingSystem:
    """
    Hybrid timing system that combines intelligent timeframe scheduling
    with duplicate prevention and time validation
    """
    def __init__(self, pair_group):
        self.pair_group = pair_group
        self.ny_tz = pytz.timezone('America/New_York')
        
        # Timeframe scanning configurations
        self.timeframe_configs = {
            # For quick SMT/PSP/TPD detection
            'M5': {
                'interval': 5,  # minutes
                'scan_type': 'quick',
                'next_scan': None,
                'last_scan': None,
                'needs_data': True
            },
            'M15': {
                'interval': 15,
                'scan_type': 'quick', 
                'next_scan': None,
                'last_scan': None,
                'needs_data': True
            },
            'H1': {
                'interval': 60,
                'scan_type': 'normal',
                'next_scan': None,
                'last_scan': None,
                'needs_data': True
            },
            # For higher timeframe analysis (less frequent)
            'H4': {
                'interval': 240,
                'scan_type': 'deep',
                'next_scan': None,
                'last_scan': None,
                'needs_data': True,
                'cooldown_minutes': 15  # Only scan every 15 min even if data available
            },
            'D': {
                'interval': 1440,
                'scan_type': 'deep',
                'next_scan': None,
                'last_scan': None,
                'needs_data': True,
                'cooldown_minutes': 30  # Only scan every 30 min
            },
            'W': {
                'interval': 10080,
                'scan_type': 'deep',
                'next_scan': None,
                'last_scan': None,
                'needs_data': True,
                'cooldown_minutes': 120  # Only scan every 2 hours
            }
        }
        
        # Initialize next scan times
        self._calculate_initial_scan_times()
        
        # For duplicate prevention (compatible with RobustTimingManager)
        self.sent_signals = {}
    
    def _calculate_initial_scan_times(self):
        """Calculate when to first scan each timeframe"""
        now = datetime.now(self.ny_tz)
        
        for tf, config in self.timeframe_configs.items():
            # For first run, scan immediately for quick timeframes
            if config['scan_type'] == 'quick':
                config['next_scan'] = now
            else:
                # For deep scans, schedule in the near future
                config['next_scan'] = now + timedelta(minutes=config.get('cooldown_minutes', 5))
    
    def should_scan_timeframe(self, timeframe):
        """Check if we should scan this timeframe now"""
        if timeframe not in self.timeframe_configs:
            return False
        
        config = self.timeframe_configs[timeframe]
        now = datetime.now(self.ny_tz)
        
        # Check if it's time to scan
        if now >= config['next_scan']:
            # Update last scan time
            config['last_scan'] = now
            
            # Calculate next scan time
            if config['scan_type'] == 'quick':
                # Quick scans: scan at next candle + buffer
                next_candle = self._get_next_candle_close_time(timeframe)
                buffer = timedelta(seconds=10 if timeframe in ['M5', 'M15'] else 30)
                config['next_scan'] = next_candle + buffer
            else:
                # Deep scans: use cooldown
                cooldown = timedelta(minutes=config.get('cooldown_minutes', 15))
                config['next_scan'] = now + cooldown
            
            return True
        
        return False
    
    def _get_next_candle_close_time(self, timeframe):
        """Calculate when the next candle will close"""
        now = datetime.now(self.ny_tz)
        
        if timeframe == 'M5':
            minutes_to_add = 5 - (now.minute % 5)
            if minutes_to_add == 0:
                minutes_to_add = 5
            return now.replace(second=0, microsecond=0) + timedelta(minutes=minutes_to_add)
        
        elif timeframe == 'M15':
            minutes_to_add = 15 - (now.minute % 15)
            if minutes_to_add == 0:
                minutes_to_add = 15
            return now.replace(second=0, microsecond=0) + timedelta(minutes=minutes_to_add)
        
        elif timeframe == 'H1':
            next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            return next_hour
        
        elif timeframe == 'H4':
            current_hour = now.hour
            next_h4_hour = ((current_hour // 4) + 1) * 4
            if next_h4_hour >= 24:
                next_day = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
                return next_day
            else:
                return now.replace(hour=next_h4_hour, minute=0, second=0, microsecond=0)
        
        elif timeframe == 'D':
            next_day = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            return next_day
        
        elif timeframe == 'W':
            # Next Sunday at 00:00 (assuming week starts Sunday)
            days_ahead = 6 - now.weekday()  # 0=Monday, 6=Sunday
            if days_ahead <= 0:
                days_ahead += 7
            next_sunday = now + timedelta(days=days_ahead)
            return next_sunday.replace(hour=0, minute=0, second=0, microsecond=0)
        
        return now + timedelta(minutes=5)  # Default
    
    def get_timeframes_to_scan(self):
        """Get list of timeframes that should be scanned now"""
        to_scan = []
        
        for tf in self.timeframe_configs.keys():
            if self.should_scan_timeframe(tf):
                to_scan.append(tf)
        
        return to_scan
    
    def get_sleep_time(self):
        """Calculate sleep time until next M5 candle close + buffer"""
        now = datetime.now(self.ny_tz)
        
        # Calculate next M5 candle close
        current_minute = now.minute
        minutes_past_5 = current_minute % 5
        seconds_past_minute = now.second + (now.microsecond / 1000000)
        
        if minutes_past_5 == 0:
            # At exact 5-minute mark
            if seconds_past_minute < 2:
                # Still within 2-second buffer after candle close
                sleep_seconds = 2 - seconds_past_minute + 2  # Wait for buffer, then add 2 seconds
            else:
                # Candle closed more than 2 seconds ago, wait for next one
                minutes_to_next_close = 5
                seconds_to_next_close = (minutes_to_next_close * 60) - seconds_past_minute
                sleep_seconds = seconds_to_next_close + 2  # Add 2-second buffer
        else:
            # Not at 5-minute mark
            minutes_to_next_close = 5 - minutes_past_5
            seconds_to_next_close = (minutes_to_next_close * 60) - seconds_past_minute
            sleep_seconds = seconds_to_next_close + 2  # Add 2-second buffer
        
        # Ensure minimum sleep of 5 seconds and maximum of 300 seconds (5 minutes)
        sleep_seconds = max(5, min(sleep_seconds, 300))
        
        return sleep_seconds
    
    def mark_scanned(self, timeframe):
        """Mark a timeframe as scanned (for manual updates)"""
        if timeframe in self.timeframe_configs:
            self.timeframe_configs[timeframe]['last_scan'] = datetime.now(self.ny_tz)

class SmartTimingSystem:
    def __init__(self):
        self.candle_watch_times = {
            'M5': None,   # 5-minute candles
            'M15': None,  # 15-minute candles  
            'H1': None,   # 1-hour candles
            'H4': None    # 4-hour candles
        }
        
    def get_smart_sleep_time(self):
        """Calculate sleep time until next relevant candle with buffer"""
        now = datetime.now(NY_TZ)
        
        # Calculate next candle times with 5-second buffer for API latency
        next_m5 = self._get_next_candle_time('M5') + timedelta(seconds=5)
        next_m15 = self._get_next_candle_time('M15') + timedelta(seconds=5)
        next_h1 = self._get_next_candle_time('H1') + timedelta(seconds=5)
        next_h4 = self._get_next_candle_time('H4') + timedelta(seconds=5)
        
        # Find the earliest upcoming candle
        next_candle_time = min(next_m5, next_m15, next_h1, next_h4)
        sleep_seconds = (next_candle_time - now).total_seconds()
        
        logger.info(f"⏰ Next scan in {sleep_seconds:.0f}s (at {next_candle_time.strftime('%H:%M:%S')})")
        return max(sleep_seconds, 5)  # Minimum 5 seconds
    
    def _get_next_candle_time(self, timeframe):
        """Calculate exact time when next candle will be available"""
        now = datetime.now(NY_TZ)
        
        if timeframe == 'M5':
            # Next 5-minute boundary
            next_minute = (now.minute // 5 + 1) * 5
            if next_minute >= 60:
                next_minute = 0
                next_time = now.replace(hour=now.hour + 1, minute=0, second=0, microsecond=0)
            else:
                next_time = now.replace(minute=next_minute, second=0, microsecond=0)
                
        elif timeframe == 'M15':
            # Next 15-minute boundary
            next_minute = (now.minute // 15 + 1) * 15
            if next_minute >= 60:
                next_minute = 0
                next_time = now.replace(hour=now.hour + 1, minute=0, second=0, microsecond=0)
            else:
                next_time = now.replace(minute=next_minute, second=0, microsecond=0)
                
        elif timeframe == 'H1':
            # Next hour boundary
            next_time = now.replace(hour=now.hour + 1, minute=0, second=0, microsecond=0)
            
        elif timeframe == 'H4':
            # Next 4-hour boundary
            next_hour = (now.hour // 4 + 1) * 4
            if next_hour >= 24:
                next_hour = 0
                next_time = now.replace(day=now.day + 1, hour=0, minute=0, second=0, microsecond=0)
            else:
                next_time = now.replace(hour=next_hour, minute=0, second=0, microsecond=0)
        
        return next_time



class SupplyDemandDetector:
    def __init__(self, min_zone_pct=0):
        self.min_zone_pct = min_zone_pct
    
    def detect_swing_highs_lows(self, data, lookback=5):
        if len(data) < lookback * 2:
            return [], []
        
        highs = data['high'].values
        lows = data['low'].values
        times = data['time'].values
        
        swing_highs = []
        swing_lows = []
        
        for i in range(lookback, len(data) - lookback):
            if all(highs[i] > highs[i-j] for j in range(1, lookback+1)):
                if all(highs[i] > highs[i+j] for j in range(1, lookback+1)):
                    swing_highs.append({
                        'index': i,
                        'price': highs[i],
                        'time': times[i],
                        'type': 'high'
                    })
        
        for i in range(lookback, len(data) - lookback):
            if all(lows[i] < lows[i-j] for j in range(1, lookback+1)):
                if all(lows[i] < lows[i+j] for j in range(1, lookback+1)):
                    swing_lows.append({
                        'index': i,
                        'price': lows[i],
                        'time': times[i],
                        'type': 'low'
                    })
        
        return swing_highs, swing_lows
    
    def find_structure_breaks(self, data, swing_highs, swing_lows):
        closes = data['close'].values
        times = data['time'].values
        
        all_swings = swing_highs + swing_lows
        all_swings.sort(key=lambda x: x['index'])
        
        breaks = []
        current_trend = 0
        
        for swing in all_swings:
            swing_idx = swing['index']
            swing_price = swing['price']
            swing_type = swing['type']
            
            for i in range(swing_idx + 1, min(swing_idx + 20, len(data))):
                current_close = closes[i]
                
                if swing_type == 'high':
                    if current_close < swing_price:
                        if current_trend == 1:
                            break_type = 'CHoCH'
                        else:
                            break_type = 'BOS'
                        
                        current_trend = -1
                        
                        breaks.append({
                            'break_type': break_type,
                            'broken_swing': swing,
                            'break_index': i,
                            'break_price': current_close,
                            'break_time': times[i],
                            'trend': 'bearish',
                            'swing_type': 'high'
                        })
                        break
                else:
                    if current_close > swing_price:
                        if current_trend == -1:
                            break_type = 'CHoCH'
                        else:
                            break_type = 'BOS'
                        
                        current_trend = 1
                        
                        breaks.append({
                            'break_type': break_type,
                            'broken_swing': swing,
                            'break_index': i,
                            'break_price': current_close,
                            'break_time': times[i],
                            'trend': 'bullish',
                            'swing_type': 'low'
                        })
                        break
        
        return breaks
    
    def find_order_block_in_broken_leg(self, data, broken_swing, break_index, trend):
        start_idx = broken_swing['index']
        end_idx = break_index
        
        if start_idx >= end_idx:
            return None
        
        leg_data = data.iloc[start_idx:end_idx]
        
        if trend == 'bearish':
            extreme_idx = leg_data['high'].idxmax()
            zone_type = 'supply'
        else:
            extreme_idx = leg_data['low'].idxmin()
            zone_type = 'demand'
        
        extreme_candle = data.iloc[extreme_idx]
        
        return {
            'index': extreme_idx,
            'time': extreme_candle['time'],
            'high': extreme_candle['high'],
            'low': extreme_candle['low'],
            'close': extreme_candle['close'],
            'open': extreme_candle['open'],
            'zone_type': zone_type,
            'trend': trend
        }
    
    def check_zone_still_valid(self, zone, current_data, other_asset_data=None):
        try:
            if current_data is None or current_data.empty:
                return True
            
            zone_type = zone['type']
            zone_low = zone['zone_low']
            zone_high = zone['zone_high']
            formation_time = zone['formation_time']
            creation_index = zone.get('creation_index', 0)
            
            from pytz import timezone
            NY_TZ = timezone('America/New_York')
            
            try:
                if isinstance(formation_time, pd.Timestamp):
                    formation_ts = formation_time
                elif isinstance(formation_time, np.datetime64):
                    formation_ts = pd.Timestamp(formation_time)
                elif isinstance(formation_time, str):
                    formation_ts = pd.to_datetime(formation_time, errors='coerce')
                    if pd.isna(formation_ts):
                        return False
                elif isinstance(formation_time, datetime):
                    formation_ts = pd.Timestamp(formation_time)
                else:
                    return False
                
                if formation_ts.tz is None:
                    formation_ts = formation_ts.tz_localize('UTC').tz_convert(NY_TZ)
                elif str(formation_ts.tz) != str(NY_TZ):
                    formation_ts = formation_ts.tz_convert(NY_TZ)
                    
            except Exception as e:
                return False
            
            current_data_copy = current_data.copy()
            if current_data_copy['time'].dt.tz is None:
                current_data_copy['time'] = current_data_copy['time'].dt.tz_localize('UTC').dt.tz_convert(NY_TZ)
            elif str(current_data_copy['time'].dt.tz) != str(NY_TZ):
                current_data_copy['time'] = current_data_copy['time'].dt.tz_convert(NY_TZ)
            
            subsequent_candles = current_data_copy[current_data_copy['time'] > formation_ts]
            
            candles_since_creation = len(subsequent_candles)
            
            if zone_type == 'demand':
                if (subsequent_candles['low'] < zone_low).any():
                    return False
            else:
                if (subsequent_candles['high'] > zone_high).any():
                    return False
            
            return True
            
        except Exception as e:
            import traceback
            return False
    
    def calculate_wick_percentage(self, candle):
        body_size = abs(candle['close'] - candle['open'])
        total_range = candle['high'] - candle['low']
        
        if total_range == 0:
            return 0
        
        if candle['close'] > candle['open']:
            upper_wick = candle['high'] - candle['close']
            wick_percentage = (upper_wick / total_range) * 100
        else:
            lower_wick = candle['open'] - candle['low']
            wick_percentage = (lower_wick / total_range) * 100
        
        return wick_percentage
    
    def check_candles_opposite_direction(self, candle_a, candle_b):
        a_bullish = candle_a['close'] > candle_a['open']
        b_bullish = candle_b['close'] > candle_b['open']
        return a_bullish != b_bullish
    
    def is_demand_candles(self, candle_a, candle_b):
        if candle_a['close'] >= candle_a['open']:
            return False
        if candle_b['close'] <= candle_b['open']:
            return False
        return True
    
    def is_supply_candles(self, candle_a, candle_b):
        if candle_a['close'] <= candle_a['open']:
            return False
        if candle_b['close'] >= candle_b['open']:
            return False
        return True
    
    def check_zone_activation(self, candle_a, subsequent_candles, zone_type):
        if zone_type == 'demand':
            for _, candle in subsequent_candles.iterrows():
                if candle['close'] > candle_a['high']:
                    return True
            return False
        else:
            for _, candle in subsequent_candles.iterrows():
                if candle['close'] < candle_a['low']:
                    return True
            return False
    
    def check_zone_invalidation(self, zone, subsequent_candles, zone_type, asset, other_asset_data=None):
        if zone_type == 'demand':
            return (subsequent_candles['low'] < zone['zone_low']).any()
        else:
            return (subsequent_candles['high'] > zone['zone_high']).any()
    
    def calculate_invalidation_point(self, candle_a, candle_b, zone_type):
        if zone_type == 'demand':
            return min(candle_a['low'], candle_b['low'])
        else:
            return max(candle_a['high'], candle_b['high'])
    
    def scan_timeframe(self, data, timeframe, asset):
        if data is None or len(data) < 10:
            return []
        
        from pytz import timezone
        NY_TZ = timezone('America/New_York')
        
        if 'complete' in data.columns:
            closed_data = data[data['complete'] == True].copy()
        else:
            closed_data = data.copy()
        
        if len(closed_data) < 10:
            return []

        # Add this right after your length check
        closed_data = closed_data.iloc[:-4] if len(closed_data) > 4 else closed_data
        
        zones = []
        
        if closed_data['time'].dt.tz is None:
            closed_data['time'] = closed_data['time'].dt.tz_localize('UTC').dt.tz_convert(NY_TZ)
        elif str(closed_data['time'].dt.tz) != str(NY_TZ):
            closed_data['time'] = closed_data['time'].dt.tz_convert(NY_TZ)
        
        for i in range(len(closed_data) - 1):
            candle_a = closed_data.iloc[i]
            candle_b = closed_data.iloc[i + 1]
            
            if not self.check_candles_opposite_direction(candle_a, candle_b):
                continue
            
            if self.is_demand_candles(candle_a, candle_b):
                zone_high = min(candle_a['close'], candle_b['open'])
                zone_low = min(candle_a['low'], candle_b['low'])

                current_index = len(closed_data) - 1
                
                subsequent_start = i + 2
                if subsequent_start < len(closed_data):
                    subsequent_candles = closed_data.iloc[subsequent_start:]
                    
                    if not self.check_zone_activation(candle_a, subsequent_candles, 'demand'):
                        continue
                
                subsequent_start = i + 2
                if subsequent_start < len(closed_data):
                    subsequent_candles = closed_data.iloc[subsequent_start:min(subsequent_start + 35, len(closed_data))]
                    
                    if self.check_zone_invalidation(
                        {'zone_low': zone_low, 'zone_high': zone_high, 'type': 'demand'},
                        subsequent_candles,
                        'demand',
                        asset
                    ):
                        continue
                
                zone = {
                    'type': 'demand',
                    'zone_low': zone_low,
                    'zone_high': zone_high,
                    'formation_candle': {
                        'high': max(candle_a['high'], candle_b['high']),
                        'low': min(candle_a['low'], candle_b['low']),
                        'close': candle_a['close'],
                        'open': candle_a['open'],
                        'time': candle_a['time']
                    },
                    'formation_time': candle_a['time'],
                    'timeframe': timeframe,
                    'asset': asset,
                    'wick_percentage': self.calculate_wick_percentage(candle_a),
                    'zone_name': f"{asset}_{timeframe}_DEMAND_{candle_a['time'].strftime('%Y%m%d%H%M')}",
                    'direction': 'bullish',
                    'wick_adjusted': False,
                    'wick_category': 'large' if self.calculate_wick_percentage(candle_a) > 40 else 'normal',
                    'candle_a_low': candle_a['low'],
                    'candle_a_high': candle_a['high'],
                    'candle_a_close': candle_a['close'],
                    'candle_a_open': candle_a['open'],
                    'candle_b_low': candle_b['low'],
                    'candle_b_high': candle_b['high'],
                    'candle_b_close': candle_b['close'],
                    'candle_b_open': candle_b['open'],
                    'highest_high': max(candle_a['high'], candle_b['high']),
                    'lowest_low': min(candle_a['low'], candle_b['low']),
                    'creation_index': i
                }
                zones.append(zone)
            
            elif self.is_supply_candles(candle_a, candle_b):
                zone_high = max(candle_a['high'], candle_b['high'])
                zone_low = min(candle_a['close'], candle_b['open'])

                current_index = len(closed_data) - 1
                if i >= current_index - 3:
                    continue
                    
                subsequent_start = i + 2
                if subsequent_start < len(closed_data):
                    subsequent_candles = closed_data.iloc[subsequent_start:]
                    
                    if not self.check_zone_activation(candle_a, subsequent_candles, 'supply'):
                        continue
                
                subsequent_start = i + 2
                if subsequent_start < len(closed_data):
                    subsequent_candles = closed_data.iloc[subsequent_start:min(subsequent_start + 35, len(closed_data))]
                    
                    if self.check_zone_invalidation(
                        {'zone_low': zone_low, 'zone_high': zone_high, 'type': 'supply'},
                        subsequent_candles,
                        'supply',
                        asset
                    ):
                        continue
                
                zone = {
                    'type': 'supply',
                    'zone_low': zone_low,
                    'zone_high': zone_high,
                    'formation_candle': {
                        'high': max(candle_a['high'], candle_b['high']),
                        'low': min(candle_a['low'], candle_b['low']),
                        'close': candle_a['close'],
                        'open': candle_a['open'],
                        'time': candle_a['time']
                    },
                    'formation_time': candle_a['time'],
                    'timeframe': timeframe,
                    'asset': asset,
                    'wick_percentage': self.calculate_wick_percentage(candle_a),
                    'zone_name': f"{asset}_{timeframe}_SUPPLY_{candle_a['time'].strftime('%Y%m%d%H%M')}",
                    'direction': 'bearish',
                    'wick_adjusted': False,
                    'wick_category': 'large' if self.calculate_wick_percentage(candle_a) > 40 else 'normal',
                    'candle_a_low': candle_a['low'],
                    'candle_a_high': candle_a['high'],
                    'candle_a_close': candle_a['close'],
                    'candle_a_open': candle_a['open'],
                    'candle_b_low': candle_b['low'],
                    'candle_b_high': candle_b['high'],
                    'candle_b_close': candle_b['close'],
                    'candle_b_open': candle_b['open'],
                    'highest_high': max(candle_a['high'], candle_b['high']),
                    'lowest_low': min(candle_a['low'], candle_b['low']),
                    'creation_index': i
                }
                zones.append(zone)
        
        filtered_zones = []
        if zones:
            zones.sort(key=lambda x: x['formation_time'])
            
            for i, zone in enumerate(zones):
                if i == 0:
                    filtered_zones.append(zone)
                    continue
                
                prev_zone = filtered_zones[-1]
                
                overlap_low = max(zone['zone_low'], prev_zone['zone_low'])
                overlap_high = min(zone['zone_high'], prev_zone['zone_high'])
                
                if overlap_low < overlap_high:
                    zone_range = zone['zone_high'] - zone['zone_low']
                    if zone_range > 0:
                        overlap_range = overlap_high - overlap_low
                        overlap_pct = (overlap_range / zone_range) * 100
                        
                        if overlap_pct > 50:
                            if zone['wick_percentage'] > prev_zone['wick_percentage']:
                                filtered_zones[-1] = zone
                            continue
                
                filtered_zones.append(zone)
        
        return filtered_zones
    
    def scan_all_timeframes(self, market_data, instruments, timeframes=['M15', 'H1', 'H4','D' ,'W']):
        all_zones = []
        
        for instrument in instruments:
            for timeframe in timeframes:
                data = market_data.get(instrument, {}).get(timeframe)
                if data is not None and not data.empty:
                    zones = self.scan_timeframe(data, timeframe, instrument)
                    all_zones.extend(zones)
        
        timeframe_order = {'W' :6 ,'D' : 5,'H4': 4, 'H1': 3, 'M15': 2, 'M5': 1}
        all_zones.sort(key=lambda x: timeframe_order.get(x['timeframe'], 0), reverse=True)
        
        return all_zones




class NewsCalendar:
    def __init__(self, rapidapi_key: str, base_path: str = '/content/drive/MyDrive', 
                 logger=None, cache_duration: int = 43200):  # 43200 seconds = 12 hours
        """
        Initialize news calendar - WITH CACHE DIRECTORY SETUP
        
        Args:
            rapidapi_key: Your RapidAPI key
            base_path: Base directory path
            logger: Optional logger instance
            cache_duration: Cache duration in seconds (default: 43200 = 12 hours)
        """
        self.rapidapi_key = rapidapi_key
        self.base_path = base_path.rstrip('/')
        self.news_data_path = f"{self.base_path}/news_data"
        
        # Setup logger
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger('NewsCalendar')
        
        # === CREATE ALL NECESSARY DIRECTORIES ===
        os.makedirs(f"{self.news_data_path}/raw", exist_ok=True)
        os.makedirs(f"{self.news_data_path}/processed", exist_ok=True)
        os.makedirs(f"{self.news_data_path}/cache", exist_ok=True)  # This line is critical
        
        # === DEFINE CACHE DIRECTORY ATTRIBUTE ===
        self.cache_dir = f"{self.news_data_path}/cache"  # This fixes the error
        
        # === SET CACHE DURATION TO 12 HOURS ===
        self.cache_duration = cache_duration  # 43200 seconds = 12 hours
        
        # Timezone setup
        self.utc_tz = pytz.UTC
        self.ny_tz = pytz.timezone('America/New_York')
        
        # Currencies we track
        self.tracked_currencies = ['USD', 'GBP', 'EUR', 'JPY']
        
        # Instrument to currency mapping
        self.instrument_currency_map = self._create_currency_map()
        
        # Cache file for _get_from_cache/_save_to_cache methods
        self.cache_file = f"{self.cache_dir}/rapidapi_cache.json"
        
        self.logger.info(f"📰 News Calendar initialized. Cache duration: {self.cache_duration//3600} hours")

    def get_daily_news(self, force_fetch: bool = False) -> Dict:
        """
        Get today's news - with proper error handling for cached data
        """
        today_str = datetime.now(self.ny_tz).strftime('%Y-%m-%d')
        cache_file = f"{self.cache_dir}/news_cache_{today_str}.json"
        
        # 1. Try to use cache first (unless force_fetch is True)
        if not force_fetch and os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                
                # VALIDATE CACHED DATA STRUCTURE
                if isinstance(cached_data, dict) and 'events' in cached_data:
                    self.logger.info(f"📰 Loaded {len(cached_data['events'])} events from cache")
                    return cached_data
                else:
                    self.logger.warning("⚠️ Cached data has invalid structure, re-fetching")
                    
            except (json.JSONDecodeError, IOError) as e:
                self.logger.warning(f"⚠️ Cache corrupted, re-fetching: {e}")
        
        # 2. Fetch fresh data from API
        self.logger.info(f"📰 Fetching fresh news for {today_str}")
        api_data = self.fetch_news_data()
        
        # 3. Process and cache the new data
        if api_data and 'error' not in api_data:
            processed_data = self._process_raw_news(api_data, today_str)
            
            # Only cache if we have valid events
            if processed_data and 'events' in processed_data:
                try:
                    with open(cache_file, 'w') as f:
                        json.dump(processed_data, f, indent=2)
                    self.logger.info(f"💾 News cached to {cache_file} (valid for {self.cache_duration//3600} hours)")
                except Exception as e:
                    self.logger.error(f"❌ Failed to write cache: {e}")
            
            return processed_data
        
        return {"error": "Failed to fetch news", "events": []}
    
    def _create_currency_map(self) -> Dict[str, List[str]]:
        """Create mapping from instruments to relevant currencies"""
        mapping = {}
        
        # Precious metals
        mapping['XAU_USD'] = ['USD']
        mapping['XAU_JPY'] = ['JPY', 'USD']  # Gold in JPY, but USD news matters
        
        # US Indices
        mapping['NAS100_USD'] = ['USD']
        mapping['SPX500_USD'] = ['USD']
        
        # Forex pairs
        mapping['GBP_USD'] = ['GBP', 'USD']
        mapping['EUR_USD'] = ['EUR', 'USD']
        
        # European indices
        mapping['DE30_EUR'] = ['EUR']
        mapping['EU50_EUR'] = ['EUR']
        
        # Add XAG_USD if you have it
        mapping['XAG_USD'] = ['USD']
        
        return mapping
    
    def fetch_news_data(self, date_str: str = None) -> Dict:
        """
        Fetch and process news data from RapidAPI.
        Returns processed data ready for analysis.
        """
        if date_str is None:
            date_str = datetime.now(self.ny_tz).strftime('%Y-%m-%d')
        
        cache_key = f"news_{date_str}"
        
        # 1. Check cache first for PROCESSED data (valid for 12 hours)
        cached = self._get_from_cache(cache_key)
        if cached and 'events' in cached:
            self.logger.info(f"📰 Using cached processed news for {date_str} (cache valid for {self.cache_duration//3600}h)")
            return cached  # Return already-processed data
        
        try:
            # 2. Fetch from API
            url = "https://economic-calendar-api.p.rapidapi.com/calendar"
            
            headers = {
                "X-RapidAPI-Key": self.rapidapi_key,
                "X-RapidAPI-Host": "economic-calendar-api.p.rapidapi.com"
            }
            
            params = {
                "limit": "50"
            }
            
            self.logger.info(f"📰 Fetching news from RapidAPI for {date_str}...")
            response = requests.get(url, headers=headers, params=params, timeout=30)
            
            if response.status_code == 200:
                # 3. Get raw API response
                raw_api_response = response.json()
                
                # 4. Process the raw response into our format
                processed_data = self._process_raw_news(raw_api_response, date_str)
                
                # 5. Only cache if we have valid processed data
                if processed_data.get("events"):
                    self._save_to_cache(cache_key, processed_data)
                    
                    # Save to processed CSV
                    self._save_to_processed_csv(processed_data, date_str)
                    
                    self.logger.info(f"✅ Fetched {len(processed_data.get('events', []))} news events for {date_str}")
                    return processed_data
                else:
                    error_msg = processed_data.get('error', 'Processing failed')
                    self.logger.error(f"❌ News processing failed: {error_msg}")
                    return processed_data
                    
            else:
                error_msg = f"API Error: {response.status_code} - {response.text}"
                self.logger.error(f"❌ {error_msg}")
                return self._create_error_response(error_msg)
                
        except requests.exceptions.Timeout:
            error_msg = "API timeout"
            self.logger.error(f"❌ {error_msg}")
            return self._create_error_response(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            return self._create_error_response(error_msg)
    
    def _process_raw_news(self, raw_data: Dict, date_str: str) -> Dict:
        """Process raw API response into structured format - UPDATED FOR NEW API"""
        try:
            events = []
    
            # 1. VALIDATE THE NEW API RESPONSE STRUCTURE
            if not isinstance(raw_data, dict):
                self.logger.warning(f"❌ Raw data is not a dict: {type(raw_data)}")
                return {"error": "Invalid data format", "events": []}
    
            if not raw_data.get('success'):
                error_msg = raw_data.get('message', 'API call unsuccessful')
                self.logger.warning(f"❌ API returned error: {error_msg}")
                return {"error": error_msg, "events": []}
    
            if 'data' not in raw_data or not isinstance(raw_data['data'], list):
                self.logger.warning(f"❌ 'data' key missing or not a list")
                return {"error": "No event data in response", "events": []}
    
            # 2. PROCESS EACH EVENT WITH CORRECT FIELD NAMES
            for event in raw_data['data']:
                try:
                    # EXTRACT FIELDS - USING NAMES FROM YOUR EXAMPLE
                    event_time_utc = event.get('dateUtc')  # Key changed
                    event_name = event.get('name')        # Key changed
                    currency = event.get('currencyCode')  # Key changed
                    
                    # IMPACT FIELD: Use 'volatility' (present in example) not 'potency'
                    impact = event.get('volatility', 'NONE').upper()
                    
                    # Skip if no currency or not in tracked currencies
                    if not currency or currency not in self.tracked_currencies:
                        continue
    
                    # CONVERT UTC TO NY TIME
                    try:
                        # Parse UTC time - format: '2025-12-03T13:30:00.000Z'
                        if event_time_utc:
                            utc_dt = datetime.strptime(event_time_utc, '%Y-%m-%dT%H:%M:%S.%fZ')
                        else:
                            continue
                        
                        utc_dt = self.utc_tz.localize(utc_dt)
                        ny_dt = utc_dt.astimezone(self.ny_tz)
                        ny_time_str = ny_dt.strftime('%H:%M')
                        
                        # Check if event is today in NY time
                        ny_date_str = ny_dt.strftime('%Y-%m-%d')
                        if ny_date_str != date_str:
                            continue  # Skip if not today in NY time
                            
                    except Exception as time_error:
                        self.logger.debug(f"⚠️ Time parsing error for event: {time_error}")
                        continue
                    
                    # Get additional values
                    actual = event.get('actual')
                    forecast = event.get('consensus')  # Note: called 'consensus' in API
                    previous = event.get('previous')
                    
                    # IMPACT LEVEL MAPPING
                    impact_level_map = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1, 'NONE': 0}
                    impact_level = impact_level_map.get(impact, 0)
                    
                    # CREATE PROCESSED EVENT
                    processed_event = {
                        'utc_time': event_time_utc,
                        'ny_time': ny_time_str,
                        'ny_datetime': ny_dt.isoformat(),
                        'event': event_name,
                        'currency': currency,
                        'impact': impact,
                        'impact_level': impact_level,
                        'actual': actual if actual is not None else '',
                        'forecast': forecast if forecast is not None else '',
                        'previous': previous if previous is not None else '',
                        'timestamp': datetime.now(self.ny_tz).isoformat()
                    }
                    
                    events.append(processed_event)
                    
                except Exception as e:
                    self.logger.debug(f"⚠️ Skipping event due to error: {str(e)}")
                    continue
    
            # Sort by time
            events.sort(key=lambda x: x['ny_datetime'])
            
            # LOG SUMMARY
            self.logger.info(f"📰 Processed {len(events)} events for {date_str}")
            if events:
                self.logger.info(f"   Time range: {events[0]['ny_time']} to {events[-1]['ny_time']}")
            
            return {
                'fetch_time': datetime.now(self.ny_tz).isoformat(),
                'date': date_str,
                'events': events,
                'summary': self._create_summary(events),
                'raw_response_metadata': {
                    'totalEvents': raw_data.get('totalEvents', 0),
                    'lastUpdated': raw_data.get('lastUpdated'),
                    'timezone': raw_data.get('timezone', 'UTC')
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error in _process_raw_news: {str(e)}", exc_info=True)
            return {"error": str(e), "events": []}
    
    def _impact_to_level(self, impact: str) -> int:
        """Convert impact string to numeric level"""
        impact_upper = impact.upper() if impact else ''
        if 'HIGH' in impact_upper:
            return 3
        elif 'MEDIUM' in impact_upper:
            return 2
        elif 'LOW' in impact_upper:
            return 1
        else:
            return 0
    
    def _create_summary(self, events: List[Dict]) -> Dict:
        """Create summary statistics for events"""
        summary = {
            'total': len(events),
            'high_impact': 0,
            'medium_impact': 0,
            'low_impact': 0,
            'by_currency': {},
            'earliest_time': None,
            'latest_time': None
        }
        
        for currency in self.tracked_currencies:
            summary['by_currency'][currency] = {
                'total': 0,
                'high': 0,
                'medium': 0,
                'low': 0
            }
        
        for event in events:
            impact = event.get('impact', '').upper()
            currency = event.get('currency', '')
            
            # Count by impact
            if 'HIGH' in impact:
                summary['high_impact'] += 1
            elif 'MEDIUM' in impact:
                summary['medium_impact'] += 1
            elif 'LOW' in impact:
                summary['low_impact'] += 1
            
            # Count by currency
            if currency in summary['by_currency']:
                summary['by_currency'][currency]['total'] += 1
                if 'HIGH' in impact:
                    summary['by_currency'][currency]['high'] += 1
                elif 'MEDIUM' in impact:
                    summary['by_currency'][currency]['medium'] += 1
                elif 'LOW' in impact:
                    summary['by_currency'][currency]['low'] += 1
        
        # Get time ranges
        if events:
            times = [datetime.fromisoformat(e['ny_datetime']) for e in events]
            summary['earliest_time'] = min(times).isoformat()
            summary['latest_time'] = max(times).isoformat()
        
        return summary
    
    def _save_to_processed_csv(self, news_data: Dict, date_str: str):
        """Save processed news to CSV file"""
        try:
            csv_file = f"{self.news_data_path}/processed/{date_str}_news.csv"
            
            headers = [
                'utc_time', 'ny_time', 'event', 'currency', 
                'impact', 'impact_level', 'actual', 'forecast', 'previous'
            ]
            
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                
                for event in news_data.get('events', []):
                    row = {key: event.get(key, '') for key in headers}
                    writer.writerow(row)
            
            self.logger.info(f"💾 Saved {len(news_data.get('events', []))} events to {csv_file}")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving to CSV: {str(e)}")
    
    def _get_from_cache(self, key: str) -> Optional[Dict]:
        """Get data from cache if valid (12 hours)"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                if key in cache_data:
                    cached_item = cache_data[key]
                    cache_time = datetime.fromisoformat(cached_item['cache_time'])
                    
                    # Check if cache is still valid (12 hours)
                    age_seconds = (datetime.now(self.ny_tz) - cache_time).total_seconds()
                    if age_seconds < self.cache_duration:  # 12 hours
                        self.logger.debug(f"📦 Cache hit for {key}, age: {age_seconds//3600}h {(age_seconds%3600)//60}m")
                        return cached_item['data']
                    
                    # Cache expired (older than 12 hours)
                    self.logger.info(f"🕒 Cache expired for {key}, age: {age_seconds//3600}h")
                    del cache_data[key]
                    with open(self.cache_file, 'w') as f:
                        json.dump(cache_data, f)
                        
            return None
            
        except Exception as e:
            self.logger.warning(f"⚠️ Cache read error: {str(e)}")
            return None
    
    def _save_to_cache(self, key: str, data: Dict):
        """Save data to cache - ONLY IF SUCCESSFUL"""
        try:
            # === Only cache if we have valid data with events ===
            if 'error' in data or not data.get('events'):
                self.logger.warning(f"⚠️ Not caching error/empty data for {key}")
                return
            
            cache_data = {}
            if os.path.exists(self.cache_file):
                try:
                    with open(self.cache_file, 'r') as f:
                        cache_data = json.load(f)
                except json.JSONDecodeError:
                    self.logger.warning("⚠️ Cache file corrupted, starting fresh")
                    cache_data = {}
            
            cache_data[key] = {
                'cache_time': datetime.now(self.ny_tz).isoformat(),
                'data': data,
                'expires_in_hours': self.cache_duration // 3600
            }
            
            # Limit cache size to 7 days worth of data
            if len(cache_data) > 7:  # Keep last 7 days
                # Sort by cache time and remove oldest
                sorted_keys = sorted(cache_data.keys(), 
                                   key=lambda k: cache_data[k]['cache_time'])
                for old_key in sorted_keys[:-7]:
                    del cache_data[old_key]
            
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
                
            self.logger.debug(f"💾 Cached {key} (valid for {self.cache_duration//3600}h)")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Cache write error: {str(e)}")
    
    def _create_error_response(self, error_msg: str) -> Dict:
        """Create error response structure"""
        return {
            'error': error_msg,
            'fetch_time': datetime.now(self.ny_tz).isoformat(),
            'events': [],
            'summary': {
                'total': 0,
                'high_impact': 0,
                'medium_impact': 0,
                'low_impact': 0,
                'by_currency': {c: {'total': 0, 'high': 0, 'medium': 0, 'low': 0} 
                               for c in self.tracked_currencies}
            }
        }
    
    def get_news_for_instrument(self, instrument: str, signal_time: datetime) -> Dict:
        """
        Get relevant news context for a specific instrument at signal time
        
        Args:
            instrument: Trading instrument (e.g., 'GBP_USD')
            signal_time: Signal timestamp (NY timezone)
            
        Returns:
            Dictionary with news context
        """
        try:
            # Get today's date in NY time
            date_str = signal_time.strftime('%Y-%m-%d')
            
            # Fetch or get cached news (uses 12-hour cache)
            news_data = self.fetch_news_data(date_str)
            
            if 'error' in news_data and news_data['error']:
                return self._create_empty_news_context(instrument, error=news_data['error'])
            
            # Get relevant currencies for this instrument
            relevant_currencies = self.instrument_currency_map.get(instrument, [])
            if not relevant_currencies:
                return self._create_empty_news_context(instrument, error="No currency mapping")
            
            # Filter events for relevant currencies
            relevant_events = []
            for event in news_data.get('events', []):
                if event.get('currency') in relevant_currencies:
                    relevant_events.append(event)
            
            # Calculate timing metrics
            timing_metrics = self._calculate_timing_metrics(relevant_events, signal_time)
            
            # Create news context
            context = {
                'instrument': instrument,
                'signal_time': signal_time.isoformat(),
                'relevant_currencies': relevant_currencies,
                'event_count': len(relevant_events),
                'high_impact_count': sum(1 for e in relevant_events if e.get('impact_level') == 3),
                'medium_impact_count': sum(1 for e in relevant_events if e.get('impact_level') == 2),
                'low_impact_count': sum(1 for e in relevant_events if e.get('impact_level') == 1),
                'timing': timing_metrics,
                'events': relevant_events[:10],  # Limit to first 10 for JSON storage
                'all_events_count': len(relevant_events),
                'fetch_status': 'success',
                'fetch_time': news_data.get('fetch_time', ''),
                'cache_info': {
                    'cache_duration_hours': self.cache_duration // 3600,
                    'cache_used': 'yes' if news_data.get('from_cache') else 'no'
                }
            }
            
            return context
            
        except Exception as e:
            error_msg = f"Error getting news context: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            return self._create_empty_news_context(instrument, error=error_msg)
    
    def _calculate_timing_metrics(self, events: List[Dict], signal_time: datetime) -> Dict:
        """Calculate timing metrics relative to signal"""
        metrics = {
            'closest_future_event': None,
            'closest_past_event': None,
            'seconds_to_next': None,
            'seconds_since_last': None,
            'events_next_1h': 0,
            'events_next_2h': 0,
            'events_prev_1h': 0,
            'events_prev_2h': 0,
            'timing_category': 'no_news'
        }
        
        if not events:
            return metrics
        
        # Convert events to datetime objects
        event_datetimes = []
        for event in events:
            try:
                event_dt = datetime.fromisoformat(event['ny_datetime'])
                event_datetimes.append((event_dt, event))
            except:
                continue
        
        if not event_datetimes:
            return metrics
        
        # Sort by time
        event_datetimes.sort(key=lambda x: x[0])
        
        # Find closest future and past events
        future_events = [(dt, event) for dt, event in event_datetimes if dt > signal_time]
        past_events = [(dt, event) for dt, event in event_datetimes if dt <= signal_time]
        
        # Closest future event
        if future_events:
            closest_future_dt, closest_future_event = min(future_events, key=lambda x: x[0])
            metrics['closest_future_event'] = closest_future_event
            metrics['seconds_to_next'] = (closest_future_dt - signal_time).total_seconds()
            
            # Count events in next 1h and 2h
            one_hour_later = signal_time + timedelta(hours=1)
            two_hours_later = signal_time + timedelta(hours=2)
            
            metrics['events_next_1h'] = sum(1 for dt, _ in future_events if dt <= one_hour_later)
            metrics['events_next_2h'] = sum(1 for dt, _ in future_events if dt <= two_hours_later)
        
        # Closest past event
        if past_events:
            closest_past_dt, closest_past_event = max(past_events, key=lambda x: x[0])
            metrics['closest_past_event'] = closest_past_event
            metrics['seconds_since_last'] = (signal_time - closest_past_dt).total_seconds()
            
            # Count events in previous 1h and 2h
            one_hour_ago = signal_time - timedelta(hours=1)
            two_hours_ago = signal_time - timedelta(hours=2)
            
            metrics['events_prev_1h'] = sum(1 for dt, _ in past_events if dt >= one_hour_ago)
            metrics['events_prev_2h'] = sum(1 for dt, _ in past_events if dt >= two_hours_ago)
        
        # Determine timing category
        if metrics['seconds_to_next'] is not None:
            if metrics['seconds_to_next'] <= 1800:  # 30 minutes
                metrics['timing_category'] = 'within_30min_before'
            elif metrics['seconds_to_next'] <= 3600:  # 1 hour
                metrics['timing_category'] = 'within_1h_before'
            elif metrics['seconds_to_next'] <= 7200:  # 2 hours
                metrics['timing_category'] = 'within_2h_before'
            else:
                metrics['timing_category'] = 'more_than_2h_before'
        elif metrics['seconds_since_last'] is not None:
            if metrics['seconds_since_last'] <= 1800:  # 30 minutes
                metrics['timing_category'] = 'within_30min_after'
            elif metrics['seconds_since_last'] <= 3600:  # 1 hour
                metrics['timing_category'] = 'within_1h_after'
            elif metrics['seconds_since_last'] <= 7200:  # 2 hours
                metrics['timing_category'] = 'within_2h_after'
            else:
                metrics['timing_category'] = 'more_than_2h_after'
        
        return metrics
    
    def _create_empty_news_context(self, instrument: str, error: str = None) -> Dict:
        """Create empty news context for error cases"""
        relevant_currencies = self.instrument_currency_map.get(instrument, [])
        
        return {
            'instrument': instrument,
            'signal_time': datetime.now(self.ny_tz).isoformat(),
            'relevant_currencies': relevant_currencies,
            'event_count': 0,
            'high_impact_count': 0,
            'medium_impact_count': 0,
            'low_impact_count': 0,
            'timing': {
                'closest_future_event': None,
                'closest_past_event': None,
                'seconds_to_next': None,
                'seconds_since_last': None,
                'events_next_1h': 0,
                'events_next_2h': 0,
                'events_prev_1h': 0,
                'events_prev_2h': 0,
                'timing_category': 'no_news_or_error'
            },
            'events': [],
            'all_events_count': 0,
            'fetch_status': 'error' if error else 'no_news',
            'error_message': error if error else '',
            'fetch_time': datetime.now(self.ny_tz).isoformat(),
            'cache_info': {
                'cache_duration_hours': self.cache_duration // 3600,
                'cache_used': 'no'
            }
        }
    
    def get_daily_summary(self, date_str: str = None) -> Dict:
        """Get daily news summary for dashboard/reporting"""
        if date_str is None:
            date_str = datetime.now(self.ny_tz).strftime('%Y-%m-%d')
        
        news_data = self.fetch_news_data(date_str)
        
        if 'error' in news_data:
            return {
                'date': date_str,
                'status': 'error',
                'error': news_data['error'],
                'summary': {
                    'total_events': 0,
                    'high_impact': 0,
                    'medium_impact': 0,
                    'low_impact': 0
                }
            }
        
        return {
            'date': date_str,
            'status': 'success',
            'summary': news_data.get('summary', {}),
            'events_count': len(news_data.get('events', [])),
            'fetch_time': news_data.get('fetch_time', ''),
            'cache_info': f"Cache valid for {self.cache_duration//3600} hours"
        }
    
    def clear_cache(self, older_than_hours: int = None):
        """Clear the cache (all or older than specified hours)"""
        try:
            if os.path.exists(self.cache_file):
                if older_than_hours:
                    # Clear only entries older than specified hours
                    with open(self.cache_file, 'r') as f:
                        cache_data = json.load(f)
                    
                    cutoff_time = datetime.now(self.ny_tz) - timedelta(hours=older_than_hours)
                    keys_to_delete = []
                    
                    for key, value in cache_data.items():
                        cache_time = datetime.fromisoformat(value['cache_time'])
                        if cache_time < cutoff_time:
                            keys_to_delete.append(key)
                    
                    for key in keys_to_delete:
                        del cache_data[key]
                    
                    with open(self.cache_file, 'w') as f:
                        json.dump(cache_data, f, indent=2)
                    
                    self.logger.info(f"🧹 Cleared {len(keys_to_delete)} cache entries older than {older_than_hours} hours")
                else:
                    # Clear entire cache
                    os.remove(self.cache_file)
                    self.logger.info("🧹 Cleared entire cache")
            else:
                self.logger.info("🧹 Cache file does not exist, nothing to clear")
                
        except Exception as e:
            self.logger.error(f"❌ Error clearing cache: {str(e)}")


import threading
import time
from collections import OrderedDict
from datetime import datetime, timedelta

class CandleDataCache:
    """
    Thread-safe cache for candle data with TTL expiration
    Prevents duplicate API calls from multiple threads
    """
    
    def __init__(self, ttl_seconds=30, max_size=100):
        """
        Args:
            ttl_seconds: Time-to-live for cache entries (default: 30 seconds)
            max_size: Maximum number of cache entries (default: 100)
        """
        self.cache = OrderedDict()
        self.ttl = ttl_seconds
        self.max_size = max_size
        self.lock = threading.RLock()  # Reentrant lock for thread safety
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_requests': 0
        }
        
    def _make_key(self, instrument, timeframe, count):
        """Create a unique cache key"""
        return f"{instrument}_{timeframe}_{count}"
    
    def _is_expired(self, entry):
        """Check if cache entry has expired"""
        if 'timestamp' not in entry:
            return True
        
        # Use entry-specific TTL if available, otherwise use instance TTL
        entry_ttl = entry.get('ttl', self.ttl)
        age = time.time() - entry['timestamp']
        return age > entry_ttl
    
    def _cleanup(self):
        """Remove expired entries and maintain max size"""
        current_time = time.time()
        
        # Remove expired entries
        expired_keys = []
        for key, entry in self.cache.items():
            if self._is_expired(entry):
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
            self.stats['evictions'] += 1
        
        # Trim to max size (LRU eviction)
        while len(self.cache) > self.max_size:
            # Remove oldest entry
            self.cache.popitem(last=False)
            self.stats['evictions'] += 1
    
    def get(self, instrument, timeframe, count):
        """
        Get cached candle data if available and not expired
        
        Returns:
            DataFrame if found and valid, None otherwise
        """
        with self.lock:
            self.stats['total_requests'] += 1
            key = self._make_key(instrument, timeframe, count)
            
            if key in self.cache:
                entry = self.cache[key]
                
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                
                if not self._is_expired(entry):
                    self.stats['hits'] += 1
                    # Log cache hit
                    self._log_cache_hit(key, entry)
                    return entry['data']
                else:
                    # Remove expired entry
                    del self.cache[key]
            
            self.stats['misses'] += 1
            return None
    
    def set(self, instrument, timeframe, count, data, ttl_override=None):  # ✅ ADD ttl_override parameter
        """
        Store candle data in cache
        
        Args:
            instrument: Trading instrument
            timeframe: Timeframe
            count: Number of candles
            data: DataFrame with candle data
            ttl_override: Optional TTL override in seconds
        """
        with self.lock:
            key = self._make_key(instrument, timeframe, count)
            
            # Create cache entry with optional TTL override
            entry = {
                'timestamp': time.time(),
                'data': data,
                'instrument': instrument,
                'timeframe': timeframe,
                'count': count,
                'ttl': ttl_override if ttl_override is not None else self.ttl  # ✅ Store entry-specific TTL
            }
            
            # Store in cache
            self.cache[key] = entry
            
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            
            # Cleanup old entries
            self._cleanup()
            
            # Log cache set
            self._log_cache_set(key, entry)
    
    def _log_cache_hit(self, key, entry):
        """Log cache hit (you can customize this)"""
        age = time.time() - entry['timestamp']
        entry_ttl = entry.get('ttl', self.ttl)
        # You can enable this for debugging:
        # print(f"📦 Cache HIT: {key} (age: {age:.1f}s, ttl: {entry_ttl}s)")
    
    def _log_cache_set(self, key, entry):
        """Log cache set (you can customize this)"""
        entry_ttl = entry.get('ttl', self.ttl)
        # You can enable this for debugging:
        # print(f"📦 Cache SET: {key} (ttl: {entry_ttl}s)")
    
    def get_stats(self):
        """Get cache statistics"""
        with self.lock:
            hit_rate = (self.stats['hits'] / self.stats['total_requests'] * 100) if self.stats['total_requests'] > 0 else 0
            return {
                **self.stats,
                'hit_rate_percent': round(hit_rate, 2),
                'current_size': len(self.cache),
                'ttl_seconds': self.ttl
            }
    
    def clear(self):
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            self.stats['hits'] = 0
            self.stats['misses'] = 0
            self.stats['evictions'] = 0
            self.stats['total_requests'] = 0
    
    def __str__(self):
        stats = self.get_stats()
        return (f"CandleCache(Size: {stats['current_size']}/{self.max_size}, "
                f"Hits: {stats['hits']}, Misses: {stats['misses']}, "
                f"Hit Rate: {stats['hit_rate_percent']}%)")

class TimeframeScanner:
    """Manages independent scanning for a specific timeframe"""
    
    def __init__(self, parent_scanner, instrument, tf, direction, zones, scan_duration, signal_data, criteria, signal_id, trigger_data):
        self.parent = parent_scanner
        self.instrument = instrument
        self.timeframe = tf
        self.direction = direction
        self.zones = zones
        self.scan_end = datetime.now(NY_TZ) + scan_duration
        self.signal_data = signal_data
        self.criteria = criteria
        self.signal_id = signal_id
        self.trigger_data = trigger_data
        self.scanned_candles = set()
        self.logger = parent_scanner.logger
        self.hammer_count = 0
        
    def run(self):
        """Run independent scan for this timeframe"""
        try:
            self.logger.info(f"🔍 Starting independent scan for {self.instrument} {self.timeframe}")
            
            while datetime.now(NY_TZ) < self.scan_end:
                # Check for CRT invalidation - USE CACHED VERSION
                if self.criteria == 'CRT+SMT':
                    crt_zone = self.zones[0] if self.zones else None
                    if crt_zone and 'invalidation_level' in crt_zone:
                        # Use force_fetch=True for real-time price check
                        df_current = self.parent.cached_fetch_candles(
                            self.instrument, 
                            'M1', 
                            count=2,
                            force_fetch=True  # Real-time data
                        )
                        
                        if not df_current.empty:
                            current_price = df_current.iloc[-1]['close']
                            
                            if self.direction == 'bearish' and current_price > crt_zone['invalidation_level']:
                                self.logger.info(f"❌ CRT invalidated in {self.timeframe}")
                                break
                            elif self.direction == 'bullish' and current_price < crt_zone['invalidation_level']:
                                self.logger.info(f"❌ CRT invalidated in {self.timeframe}")
                                break
                
                # Wait for candle close
                if not self.parent.wait_for_candle_open(self.timeframe):
                    time.sleep(1)
                    continue
                
                # Small buffer
                time.sleep(1)
                
                # 🔥 USE CACHED DATA FOR CANDLE FETCHING 🔥
                df = self.parent.cached_fetch_candles(
                    self.instrument, 
                    self.timeframe, 
                    count=10,
                    force_fetch=False  # Can use cache for historical data
                )
                
                if df.empty or len(df) < 2:
                    time.sleep(1)
                    continue
                
                # Get last closed candle
                closed_candle = df.iloc[-2]
                candle_key = f"{self.timeframe}_{closed_candle['time']}"
                
                if candle_key in self.scanned_candles:
                    time.sleep(1)
                    continue
                
                self.scanned_candles.add(candle_key)
                
                # DEBUG logging
                self.logger.info(f"📊 {self.timeframe}: Candle {closed_candle['time']}")
                self.logger.info(f"   O:{closed_candle['open']:.5f} H:{closed_candle['high']:.5f} L:{closed_candle['low']:.5f} C:{closed_candle['close']:.5f}")
                
                # Check if in zone
                candle_price = closed_candle['close']
                in_zone = False
                target_zone = None
                
                for zone in self.zones:
                    if self.direction == 'bearish':
                        if zone['low'] <= candle_price <= zone['high']:
                            in_zone = True
                            target_zone = zone
                            break
                    else:
                        if zone['low'] <= candle_price <= zone['high']:
                            in_zone = True
                            target_zone = zone
                            break
                
                if not in_zone:
                    self.logger.debug(f"❌ {self.timeframe}: Not in zone")
                    time.sleep(1)
                    continue
                
                # Check hammer
                is_hammer, upper_ratio, lower_ratio = self.parent.is_hammer_candle(closed_candle, self.direction)
                
                if is_hammer:
                    self.logger.info(f"✅ {self.timeframe}: HAMMER FOUND in zone!")
                    self.hammer_count += 1
                    
                    # Process hammer
                    success = self.parent._process_and_record_hammer(
                        self.instrument, self.timeframe, closed_candle, self.direction,
                        self.criteria, self.signal_data, self.signal_id, self.trigger_data
                    )
                    
                    if success:
                        self.logger.info(f"✅ {self.timeframe}: Hammer #{self.hammer_count} processed")
                    
                    # Continue scanning for more hammers
                
                time.sleep(1)
                
            self.logger.info(f"⏰ {self.timeframe} scan completed")
            return self.hammer_count
            
        except Exception as e:
            self.logger.error(f"❌ Error in {self.timeframe} scanner: {str(e)}")
            return 0


import pandas as pd
import numpy as np
import threading
import time
from datetime import datetime, timedelta
import os
from typing import Dict, List, Tuple, Optional, Set
import traceback
from collections import defaultdict
import csv
import shutil
import json

class SafeTPMonitoringManager:
    """
    SAFE TP Monitoring with historical backfill and live monitoring for open trades
    - Uses csv module (NOT pandas) to preserve data integrity
    - Backfills completed trades from historical data
    - Starts live threads for open trades
    - Creates backups before any modification
    """
    
    def __init__(self, csv_path: str, api_key: str, fetch_candles_func, logger=None, max_workers: int = 10):
        """
        Args:
            csv_path: Path to CSV file with trades
            api_key: OANDA API key for candle fetching
            fetch_candles_func: Your existing fetch_candles function
            logger: Logger instance
            max_workers: Maximum concurrent monitoring threads (DEFAULT: 10)
        """
        self.csv_path = csv_path
        self.api_key = api_key
        self.fetch_candles = fetch_candles_func
        self.logger = logger or self._create_default_logger()
        self.max_workers = max_workers  # Limit concurrent threads
        
        # Thread management
        self.active_threads = {}
        self.thread_states = {}
        self.thread_locks = defaultdict(threading.Lock)
        
        # Control flags
        self.shutdown_flag = threading.Event()
        self.csv_lock = threading.RLock()
        
        # Rate limiting and batching
        self.rate_limit_semaphore = threading.Semaphore(max_workers)  # Limit concurrent API calls
        self.api_call_times = []  # Track API call times for rate limiting
        self.api_call_lock = threading.Lock()
        
        # Performance tracking
        self.heartbeat_times = {}
        self.monitoring_start_times = {}
        
        # Runtime configuration
        self.monitoring_window_hours = 999999
        self.check_interval_live = 5  # Increased from 2 to 5 seconds
        self.heartbeat_interval = 60
        
        # Candle fetching configuration
        self.primary_timeframe = 'M1'
        self.fallback_timeframes = ['M5', 'M15', 'H1']
        
        # BE tracking tolerance (pips)
        self.be_tolerance = 0.0001
        
        # Backfill configuration
        self.max_backfill_threads = 3  # Limit concurrent backfill threads
        
        # Ensure CSV has required columns
        self.safe_check_and_fix_csv()
        
        # Add this check - but make it SAFE
        fieldnames, rows = self._read_csv_safe()
        row_count = len(rows)
        if row_count < 50 and row_count > 0:  # Only warn if suspiciously small but not empty
            self._log(f"⚠️ Warning: CSV has only {row_count} rows", 'warning')
        elif row_count == 0:
            self._log(f"📁 CSV is empty or could not be read")
        else:
            self._log(f"✅ CSV integrity check passed: {row_count} rows")
        
        # Start safe reconciliation with rate limiting - COMMENTED OUT FOR NOW
        # startup_thread = threading.Thread(
        #     target=self.debug_single_trade, #_safe_startup_reconciliation_rate_limited /// _safe_startup_reconciliation_focused
        #     daemon=True,
        #     name="SafeTPMonitor_StartupRecon"
        # )
        # startup_thread.start()
        
        # Start periodic checks
        periodic_thread = threading.Thread(
            target=self._periodic_health_check,
            daemon=True,
            name="SafeTPMonitor_PeriodicCheck"
        )
        periodic_thread.start()
        
        self.logger.info(f"✅ Safe TP Monitoring Manager initialized (max_workers: {max_workers})")
    
    def _create_default_logger(self):
        """Create a default logger if none provided"""
        import logging
        logger = logging.getLogger('SafeTPMonitor')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def start_monitoring(self, trade_data):
        """Alias for start_live_monitoring (for compatibility)"""
        return self.start_live_monitoring(trade_data)

    def _parse_datetime(self, dt_str):
        """Parse datetime string and ensure it's NY_TZ timezone-aware - FIXED VERSION"""
        if not dt_str or pd.isna(dt_str):
            return None
        
        try:
            # If it's already a datetime object
            if isinstance(dt_str, datetime):
                dt = dt_str
            else:
                # Parse string
                dt = pd.to_datetime(dt_str)
            
            # CRITICAL FIX: Ensure it's timezone-aware
            if dt.tzinfo is None:
                # If naive, assume it's NY time
                from pytz import timezone
                NY_TZ = timezone('America/New_York')
                return NY_TZ.localize(dt)
            else:
                # If has timezone, convert to NY_TZ
                from pytz import timezone
                NY_TZ = timezone('America/New_York')
                return dt.astimezone(NY_TZ)
        except Exception as e:
            self._log(f"⚠️ Error parsing datetime {dt_str}: {e}", 'warning')
            return None
    
    def _ensure_timezone_aware(self, dt):
        """Ensure datetime is timezone-aware in NY_TZ - FIXED VERSION"""
        if dt is None:
            return None
        
        from pytz import timezone
        NY_TZ = timezone('America/New_York')
        
        if dt.tzinfo is None:
            # If naive, localize to NY_TZ
            return NY_TZ.localize(dt)
        else:
            # If has timezone, convert to NY_TZ
            return dt.astimezone(NY_TZ)
    
    def _now_ny(self):
        """Get current time in NY_TZ"""
        from pytz import timezone
        NY_TZ = timezone('America/New_York')
        return datetime.now(NY_TZ)
    
    def _log(self, message: str, level: str = 'info'):
        """Log message with appropriate level"""
        if level == 'error':
            self.logger.error(message)
        elif level == 'warning':
            self.logger.warning(message)
        else:
            self.logger.info(message)

    def get_incomplete_trades_waitlist(self, limit=5):
        """Safely creates a list of the N most recent incomplete trades for processing."""
        waitlist = []
        try:
            fieldnames, rows = self._read_csv_safe()
            if not rows:
                self._log("CSV is empty or could not be read.")
                return waitlist
    
            for row in rows:
                trade_id = row.get('trade_id', '').strip()
                # Check if trade needs processing: missing exit time or TP results
                exit_time = row.get('exit_time', '').strip()
                tp_hit = row.get('tp_level_hit', '').strip()
                
                if not trade_id:
                    continue
                if self._is_empty_value(exit_time) or self._is_empty_value(tp_hit) or tp_hit == '0':
                    # Get entry time for sorting
                    entry_time_str = row.get('entry_time')
                    if entry_time_str:
                        try:
                            entry_time = pd.to_datetime(entry_time_str)
                            waitlist.append((entry_time, trade_id, row))
                        except:
                            waitlist.append((datetime.min, trade_id, row))
    
            # Sort by entry_time descending (newest first) and take the limit
            waitlist.sort(key=lambda x: x[0], reverse=True)
            recent_trades = waitlist[:limit]
    
            self._log(f"🧾 Created waitlist with {len(recent_trades)} most recent incomplete trades.")
            for _, trade_id, _ in recent_trades:
                self._log(f"   - {trade_id}")
            
            # Return just the trade_data dictionaries
            return [trade_data for _, _, trade_data in recent_trades]
    
        except Exception as e:
            self._log(f"❌ Failed to create waitlist: {e}", 'error')
            return []
    
    def safe_check_and_fix_csv(self):
        """Check CSV integrity and add missing columns SAFELY"""
        try:
            if not os.path.exists(self.csv_path):
                self._log(f"❌ CSV file not found: {self.csv_path}", 'error')
                # Create empty CSV with headers
                self._create_empty_csv_with_headers()
                return True
            
            # Read CSV safely with csv module
            fieldnames, rows = self._read_csv_safe()
            if not fieldnames:
                self._log(f"❌ CSV has no headers: {self.csv_path}", 'error')
                return False
            
            # Define required columns
            required_columns = [
                'timestamp', 'signal_id', 'trade_id', 'instrument', 'hammer_timeframe',
                'direction', 'tp_level_hit', 'entry_time', 'entry_price', 'sl_price',
                'sl_distance_pips', 'exit_time', 'time_to_exit_seconds',
                'monitoring_status', 'last_heartbeat', 'monitoring_notes',
                'reconciliation_attempts'
            ]
            
            # Add TP result columns (1-10)
            for i in range(1, 11):
                required_columns.extend([
                    f'tp_1_{i}_distance',
                    f'tp_1_{i}_result',
                    f'tp_1_{i}_time_seconds'
                ])
            
            # Add BE tracking columns
            for i in range(1, 11):
                required_columns.append(f'if_BE_TP{i}')
            
            # Check for missing columns
            missing_columns = [col for col in required_columns if col not in fieldnames]
            
            if missing_columns:
                self._log(f"📝 Adding {len(missing_columns)} missing columns", 'info')
                
                # Add missing columns to each row
                for row in rows:
                    for col in missing_columns:
                        if col not in row:
                            # Set default values based on column type
                            if 'result' in col:
                                row[col] = ''
                            elif 'time' in col or 'seconds' in col:
                                row[col] = ''
                            elif 'distance' in col or 'price' in col:
                                row[col] = '0'
                            elif 'status' in col:
                                row[col] = 'not_started'
                            elif 'attempts' in col:
                                row[col] = '0'
                            else:
                                row[col] = ''
                
                # Update fieldnames
                new_fieldnames = list(fieldnames) + missing_columns
                
                # Write back safely - ONLY if we have rows!
                if rows:
                    self._write_csv_safe(new_fieldnames, rows)
                    self._log(f"✅ Added missing columns: {missing_columns}")
                else:
                    self._log(f"⚠️ No rows to write, skipping column addition")
            
            else:
                self._log(f"✅ CSV has all required columns ({len(fieldnames)} total)")
            
            return True
            
        except Exception as e:
            self._log(f"❌ Error checking/fixing CSV: {e}", 'error')
            return False

    def _safe_startup_reconciliation_focused(self):
        """New reconciliation: Only processes a few recent, incomplete trades."""
        self._log("🔍 Starting FOCUSED reconciliation on recent incomplete trades...")
        time.sleep(5)
    
        # Step 1: Get the short waitlist
        trades_to_process = self.get_incomplete_trades_waitlist(limit=5)
        
        if not trades_to_process:
            self._log("✅ No incomplete trades found in the recent waitlist.")
            return
    
        self._log(f"🔄 Beginning sequential processing of {len(trades_to_process)} trades...")
        
        # Step 2: Process each trade slowly and carefully
        for idx, trade_data in enumerate(trades_to_process, 1):
            trade_id = trade_data.get('trade_id', 'Unknown')
            self._log(f"--- Processing Trade {idx}/{len(trades_to_process)}: {trade_id} ---")
            
            try:
                # Re-fetch the LATEST data from CSV right before processing
                current_row = self._get_trade_row_from_csv(trade_id)
                if not current_row:
                    self._log(f"⚠️  Could not reload {trade_id} from CSV, skipping.")
                    continue
                    
                fresh_trade_data = self._row_to_trade_data(current_row)
                entry_time = self._parse_datetime(fresh_trade_data.get('entry_time'))
                
                if not entry_time:
                    self._log(f"⚠️  Invalid entry time for {trade_id}, skipping.")
                    continue
                    
                # Decide: Backfill old trade or start live monitoring for recent one
                hours_since_entry = (self._now_ny() - entry_time).total_seconds() / 3600
                
                if hours_since_entry > 2:  # Old trade, try to backfill
                    self._log(f"🕰️  Backfilling old trade ({hours_since_entry:.1f}h)...")
                    completed = self._backfill_trade_from_history(fresh_trade_data)
                    if completed:
                        self._log(f"✅ Backfill completed for {trade_id}.")
                    else:
                        self._log(f"⏸️  Trade {trade_id} still open after backfill.")
                else:  # Recent trade
                    self._log(f"📡 Starting live monitor for recent trade ({hours_since_entry:.1f}h)...")
                    self.start_live_monitoring(fresh_trade_data)
                    
            except Exception as e:
                self._log(f"❌ Unexpected error processing {trade_id}: {e}", 'error')
            
            # CRITICAL: Wait between trades to respect API limits
            wait_seconds = 10
            self._log(f"⏳ Waiting {wait_seconds}s before next trade...")
            time.sleep(wait_seconds)
    
        self._log("✅ Focused reconciliation cycle complete.")

    def start_monitoring(self, trade_data):
        """Start monitoring for a new trade (called from HammerPatternScanner)"""
        trade_id = trade_data.get('trade_id')
        if not trade_id:
            self._log(f"❌ Cannot start monitoring: No trade_id", 'error')
            return
        
        # Check if trade is already completed (has exit time)
        if trade_data.get('exit_time'):
            self._log(f"⏭️ Trade {trade_id} already has exit time, skipping monitoring")
            return
        
        # Check if already being monitored
        if trade_id in self.active_threads:
            thread = self.active_threads[trade_id]
            if thread.is_alive():
                self._log(f"⏭️ Trade {trade_id} already being monitored")
                return
        
        # Start live monitoring thread
        try:
            thread = threading.Thread(
                target=self._monitor_trade_live,
                args=(trade_data,),
                name=f"TPMonitor_{trade_id}",
                daemon=True
            )
            
            self.active_threads[trade_id] = thread
            self.monitoring_start_times[trade_id] = datetime.now()
            self.heartbeat_times[trade_id] = datetime.now()
            
            # Initialize thread state
            self.thread_states[trade_id] = {
                'status': 'running',
                'hit_tps': set(),
                'last_checked': datetime.now(),
                'be_tracking': {i: {'state': 'waiting', 'be_triggered': False, 'outcome': 'none'} for i in range(1, 11)}
            }
            
            thread.start()
            
            # Update CSV status
            self._update_trade_in_csv_safe(trade_id, {
                'monitoring_status': 'running',
                'last_heartbeat': datetime.now().isoformat()
            })
            
            self._log(f"📡 Started LIVE monitoring for {trade_id}")
            
        except Exception as e:
            self._log(f"❌ Failed to start monitoring for {trade_id}: {e}", 'error')
    
    def _create_empty_csv_with_headers(self):
        """Create empty CSV with required headers"""
        # This is just a fallback - you should have your real CSV
        headers = [
            'timestamp', 'signal_id', 'trade_id', 'instrument', 'hammer_timeframe',
            'direction', 'tp_level_hit', 'entry_time', 'entry_price', 'sl_price',
            'sl_distance_pips', 'exit_time', 'time_to_exit_seconds',
            'monitoring_status', 'last_heartbeat', 'monitoring_notes',
            'reconciliation_attempts'
        ]
        
        for i in range(1, 11):
            headers.extend([
                f'tp_1_{i}_distance',
                f'tp_1_{i}_result',
                f'tp_1_{i}_time_seconds',
                f'if_BE_TP{i}'
            ])
        
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
        
        self._log(f"📁 Created empty CSV with {len(headers)} headers", 'warning')
    
    def _read_csv_safe(self):
        """ULTRA-SAFE CSV reading - NEVER loses data - FIXED VERSION"""
        with self.csv_lock:  # CRITICAL: Lock the entire read operation
            try:
                if not os.path.exists(self.csv_path):
                    self._log(f"CSV file not found at {self.csv_path}")
                    return [], []
    
                # Check file size first
                file_size = os.path.getsize(self.csv_path)
                if file_size == 0:
                    self._log("CSV file is empty")
                    return [], []
                
                # Read raw content first
                with open(self.csv_path, 'r', encoding='utf-8') as f:
                    raw_content = f.read()
                
                if not raw_content.strip():
                    self._log("CSV content is empty after reading")
                    return [], []
                
                # Try multiple parsing strategies
                strategies = [
                    self._parse_csv_with_pandas,  # Most robust
                    self._parse_csv_with_csv_module,  # Standard
                    self._parse_csv_manually  # Last resort
                ]
                
                for strategy in strategies:
                    try:
                        fieldnames, rows = strategy(raw_content)
                        if fieldnames and rows:
                            self._log(f"✅ Successfully read {len(rows)} rows using {strategy.__name__}")
                            return fieldnames, rows
                    except Exception as e:
                        continue
                
                # If all strategies fail, at least try to get headers
                lines = raw_content.strip().split('\n')
                if lines:
                    first_line = lines[0].strip()
                    if ',' in first_line:
                        fieldnames = [col.strip() for col in first_line.split(',')]
                    else:
                        fieldnames = []
                    self._log(f"⚠️ Could only extract headers: {fieldnames}")
                    return fieldnames, []
                
                return [], []
                
            except Exception as e:
                self._log(f"❌ CRITICAL ERROR reading CSV: {e}", 'error')
                # Try to preserve whatever we have
                try:
                    if os.path.exists(self.csv_path):
                        # Create emergency backup
                        backup_path = f"{self.csv_path}.emergency_backup_{int(time.time())}"
                        shutil.copy2(self.csv_path, backup_path)
                        self._log(f"🚨 Created emergency backup: {backup_path}")
                except:
                    pass
                return [], []  # Return empty to prevent writing garbage
    
    def _parse_csv_with_pandas(self, raw_content):
        """Parse CSV using pandas (most robust)"""
        from io import StringIO
        import pandas as pd
        
        df = pd.read_csv(StringIO(raw_content))
        fieldnames = df.columns.tolist()
        rows = df.to_dict('records')
        
        # Convert all values to strings for consistency
        rows = [{k: str(v) if not pd.isna(v) else '' for k, v in row.items()} for row in rows]
        
        return fieldnames, rows

    def _monitor_trade_live_with_resume(self, trade_data, resume_info):
        """Live monitoring with resume capability - FIXED INDENTATION"""
        trade_id = trade_data['trade_id']
        instrument = trade_data['instrument']
        direction = trade_data['direction'].lower()
        
        try:
            entry_price = float(trade_data['entry_price'])
            sl_price = float(trade_data['sl_price'])
            
            # Calculate TP prices
            pip_multiplier = 100 if 'JPY' in instrument else 10000
            tp_prices = {}
            
            for i in range(1, 11):
                distance_key = f'tp_1_{i}_distance'
                if distance_key in trade_data:
                    try:
                        distance_pips = float(trade_data[distance_key])
                        if direction == 'bearish':
                            tp_prices[i] = entry_price - (distance_pips / pip_multiplier)
                        else:  # bullish
                            tp_prices[i] = entry_price + (distance_pips / pip_multiplier)
                    except:
                        tp_prices[i] = None
                else:
                    tp_prices[i] = None
            
            # Initialize tracking with resume info
            hit_tps = set(resume_info.get('hit_tps', []))
            
            # If resuming from a specific TP, mark all lower TPs as already checked
            resume_from_tp = resume_info.get('tp_level', 1) if resume_info.get('type') == 'resume_from_tp' else 1
            
            be_tracking = {i: {'state': 'waiting', 'be_triggered': False, 'outcome': 'none'} for i in range(1, 11)}
            
            # Initialize BE tracking for already hit TPs
            for tp_level in hit_tps:
                be_tracking[tp_level]['state'] = 'tracking'
            
            entry_time = self._parse_datetime(trade_data['entry_time'])
            if entry_time is None:
                self._log(f"❌ Invalid entry_time for {trade_id}", 'error')
                return
            
            self._log(f"📊 RESUMED monitoring for {trade_id} from TP{resume_from_tp}")
            
            # Main monitoring loop
            while not self.shutdown_flag.is_set():
                try:
                    # Check if trade is already completed
                    if not self._is_orphan_row(trade_data):
                        self._log(f"✅ Trade {trade_id} completed during monitoring")
                        break
                    
                    # Fetch current candles
                    candles = self._fetch_current_candles(instrument, 'M1', count=2)
                    
                    if not candles.empty:
                        latest_candle = candles.iloc[-1]
                        current_time = latest_candle['time']
                        current_time = self._ensure_timezone_aware(current_time)
                        
                        current_high = float(latest_candle['high'])
                        current_low = float(latest_candle['low'])
                        current_close = float(latest_candle['close'])
                        
                        # Check SL
                        sl_hit = False
                        if direction == 'bearish' and current_low <= sl_price:
                            sl_hit = True
                        elif direction == 'bullish' and current_high >= sl_price:
                            sl_hit = True
                        
                        if sl_hit:
                            updates = self._record_sl_hit_updates(trade_data, current_time, hit_tps, be_tracking)
                            self._update_trade_in_csv_safe(trade_id, updates)
                            self._log(f"🛑 SL hit for {trade_id}")
                            break
                        
                        # Check TPs starting from resume point
                        for i in range(resume_from_tp, 11):
                            if i not in hit_tps and tp_prices[i] is not None:
                                tp_hit = False
                                if direction == 'bearish' and current_low <= tp_prices[i]:
                                    tp_hit = True
                                elif direction == 'bullish' and current_high >= tp_prices[i]:
                                    tp_hit = True
                                
                                if tp_hit:
                                    hit_tps.add(i)
                                    time_seconds = (current_time - entry_time).total_seconds()
                                    
                                    # Update CSV
                                    updates = {
                                        f'tp_1_{i}_result': f'+{i}',
                                        f'tp_1_{i}_time_seconds': str(int(time_seconds))
                                    }
                                    
                                    # Update highest TP hit
                                    current_highest = int(trade_data.get('tp_level_hit', 0) or 0)
                                    if i > current_highest:
                                        updates['tp_level_hit'] = str(i)
                                        updates['exit_time'] = current_time.strftime('%Y-%m-%d %H:%M:%S')
                                        updates['time_to_exit_seconds'] = str(int(time_seconds))
                                    
                                    self._update_trade_in_csv_safe(trade_id, updates)
                                    trade_data.update(updates)
                                    
                                    # Start BE tracking for this TP
                                    be_tracking[i]['state'] = 'tracking'
                                    self._log(f"✅ TP{i} hit for {trade_id}")
                        
                        # Update BE tracking
                        self._update_be_tracking(trade_data, hit_tps, be_tracking, current_close, tp_prices)
                        
                        # Update heartbeat
                        self.heartbeat_times[trade_id] = datetime.now()
                    
                    # Sleep between checks
                    time.sleep(self.check_interval_live)
                    
                except Exception as e:
                    self._log(f"⚠️ Error in resumed monitoring loop for {trade_id}: {e}", 'warning')
                    time.sleep(5)
            
            # Cleanup
            self._cleanup_trade_monitoring(trade_id, 'completed')
            
        except Exception as e:
            self._log(f"❌ Resumed monitoring crashed for {trade_id}: {e}", 'error')
            self._update_trade_in_csv_safe(trade_id, {'monitoring_status': 'failed'})
            self._cleanup_trade_monitoring(trade_id, 'failed')
    
    def _parse_csv_with_csv_module(self, raw_content):
        """Parse CSV using csv module"""
        from io import StringIO
        import csv
        
        rows = []
        reader = csv.DictReader(StringIO(raw_content))
        fieldnames = reader.fieldnames or []
        
        for row in reader:
            rows.append(dict(row))
        
        return fieldnames, rows
    
    def _parse_csv_manually(self, raw_content):
        """Parse CSV manually as fallback"""
        lines = raw_content.strip().split('\n')
        if not lines:
            return [], []
        
        # Parse headers
        headers = [col.strip() for col in lines[0].split(',')]
        
        # Parse rows
        rows = []
        for line in lines[1:]:
            if not line.strip():
                continue
            values = line.split(',')
            row = {}
            for i, header in enumerate(headers):
                if i < len(values):
                    row[header] = values[i].strip()
                else:
                    row[header] = ''
            rows.append(row)
        
        return headers, rows
    
    def _write_csv_safe(self, fieldnames, rows, backup_reason="update"):
        """ULTRA-SAFE CSV writing - NEVER loses data - FIXED VERSION"""
        with self.csv_lock:
            backup_path = None
            temp_path = None
            
            try:
                # ALWAYS create a backup before ANY write
                if os.path.exists(self.csv_path):
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                    backup_path = f"{self.csv_path}.backup_{timestamp}"
                    shutil.copy2(self.csv_path, backup_path)
                    self._log(f"📂 Created backup: {os.path.basename(backup_path)}")
                
                # CRITICAL: Never write empty if file had data
                original_has_data = False
                if os.path.exists(self.csv_path):
                    file_size = os.path.getsize(self.csv_path)
                    if file_size > 100:  # More than just headers
                        original_has_data = True
                
                if not rows and original_has_data:
                    self._log(f"🚨 EMERGENCY: Attempting to write empty rows to non-empty CSV!", 'error')
                    self._log(f"🚨 Backup created at: {backup_path}")
                    self._log(f"🚨 Restoring from backup...")
                    
                    if backup_path and os.path.exists(backup_path):
                        shutil.copy2(backup_path, self.csv_path)
                        self._log(f"✅ Restored from backup")
                    
                    return False
                
                # Write to temp file
                temp_path = f"{self.csv_path}.temp_{int(time.time())}"
                
                with open(temp_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    # Write ALL rows
                    for row in rows:
                        writer.writerow(row)
                
                # Verify temp file
                temp_size = os.path.getsize(temp_path)
                if temp_size < 50 and len(rows) > 0:
                    self._log(f"🚨 Temp file too small ({temp_size} bytes), aborting!", 'error')
                    if backup_path and os.path.exists(backup_path):
                        shutil.copy2(backup_path, self.csv_path)
                    return False
                
                # Atomic replace
                os.replace(temp_path, self.csv_path)
                
                # Verify final file
                final_size = os.path.getsize(self.csv_path)
                self._log(f"💾 SAFELY wrote {len(rows)} rows ({final_size} bytes)")
                
                # Clean up backup after successful write (optional)
                # if backup_path and os.path.exists(backup_path):
                #     os.remove(backup_path)
                
                return True
                
            except Exception as e:
                self._log(f"❌ CRITICAL ERROR writing CSV: {e}", 'error')
                
                # Try to restore from backup
                if backup_path and os.path.exists(backup_path):
                    try:
                        shutil.copy2(backup_path, self.csv_path)
                        self._log(f"🔄 Restored from backup: {backup_path}")
                    except:
                        self._log(f"💥 Failed to restore from backup!", 'error')
                
                # Clean up temp file
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                
                return False
    
    def _is_empty_value(self, value):
        """Check if a value is empty/undefined (preserves '0' and '-1' as valid)"""
        if value is None:
            return True
        
        if isinstance(value, str):
            str_val = str(value).strip()
            # '0' and '-1' are VALID values (no TP hit, SL hit)
            if str_val in ['', 'nan', 'NaN', 'None', 'null', 'NULL', 'undefined']:
                return True
            return False
        
        # For numeric values, 0 and -1 are valid
        if isinstance(value, (int, float)):
            return False
        
        return True
    
    def _safe_startup_reconciliation_rate_limited(self):
        """SAFE reconciliation with rate limiting - process in batches"""
        self._log("🔍 Starting RATE-LIMITED orphaned trade reconciliation...")
        time.sleep(10)  # Wait for system initialization
        
        try:
            fieldnames, rows = self._read_csv_safe()
            if not rows:
                self._log("✅ CSV is empty - nothing to reconcile")
                return
            
            # Find trades that need processing
            trades_needing_processing = []
            
            for row in rows:
                trade_id = row.get('trade_id', '')
                if not trade_id:
                    continue
                
                # Check if trade needs processing
                needs_processing = self._trade_needs_processing(row)
                if needs_processing:
                    trades_needing_processing.append((trade_id, row))
            
            self._log(f"📊 Found {len(trades_needing_processing)} trades needing processing")
            
            if not trades_needing_processing:
                self._log("✅ No trades need processing")
                return
            
            # Process in batches of max_workers
            batch_size = min(self.max_workers, 10)  # Process max 10 at a time
            batches = [trades_needing_processing[i:i + batch_size] 
                      for i in range(0, len(trades_needing_processing), batch_size)]
            
            self._log(f"📦 Processing in {len(batches)} batches of {batch_size}")
            
            processed_count = 0
            for batch_idx, batch in enumerate(batches):
                self._log(f"🔄 Processing batch {batch_idx + 1}/{len(batches)} ({len(batch)} trades)")
                
                batch_threads = []
                for trade_id, row in batch:
                    try:
                        # Convert row to trade_data
                        trade_data = self._row_to_trade_data(row)
                        
                        # Get entry time
                        entry_time_str = trade_data.get('entry_time')
                        if not entry_time_str:
                            self._log(f"⚠️ No entry_time for {trade_id}, skipping")
                            continue
                        
                        entry_time = self._parse_datetime(entry_time_str)
                        if entry_time is None:
                            continue
                        
                        hours_since_entry = (self._now_ny() - entry_time).total_seconds() / 3600
                        
                        # Create thread for this trade
                        thread = threading.Thread(
                            target=self._process_single_trade_reconciliation,
                            args=(trade_id, trade_data, hours_since_entry),
                            name=f"Recon_{trade_id}",
                            daemon=True
                        )
                        batch_threads.append(thread)
                        
                    except Exception as e:
                        self._log(f"❌ Error preparing {trade_id}: {e}", 'error')
                
                # Start batch threads
                for thread in batch_threads:
                    thread.start()
                
                # Wait for batch to complete
                for thread in batch_threads:
                    thread.join(timeout=30)  # 30 second timeout
                
                # Rate limiting: Wait between batches
                if batch_idx < len(batches) - 1:
                    wait_time = 5  # Wait 5 seconds between batches
                    self._log(f"⏳ Waiting {wait_time}s before next batch...")
                    time.sleep(wait_time)
                
                processed_count += len(batch)
            
            self._log(f"✅ Reconciliation completed for {processed_count} trades")
            
        except Exception as e:
            self._log(f"❌ Reconciliation failed: {e}", 'error')

    def debug_single_trade(self, trade_id):
        """Debug a single trade's backfill process"""
        self._log(f"🔍 DEBUGGING {trade_id}")
        
        # 1. Find trade in CSV
        fieldnames, rows = self._read_csv_safe()
        trade_row = None
        for row in rows:
            if row.get('trade_id') == trade_id:
                trade_row = row
                break
        
        if not trade_row:
            self._log(f"❌ Trade {trade_id} not in CSV")
            return
        
        # 2. Show current state
        self._log(f"📊 Current state:")
        self._log(f"   Entry time: {trade_row.get('entry_time')}")
        self._log(f"   Exit time: {trade_row.get('exit_time')}")
        self._log(f"   TP level hit: {trade_row.get('tp_level_hit')}")
        self._log(f"   Monitoring status: {trade_row.get('monitoring_status')}")
        
        # 3. Run backfill manually
        trade_data = self._row_to_trade_data(trade_row)
        completed = self._backfill_trade_from_history(trade_data)
        
        self._log(f"📊 Backfill result: {completed}")
        
        # 4. Check CSV again
        fieldnames2, rows2 = self._read_csv_safe()
        for row in rows2:
            if row.get('trade_id') == trade_id:
                self._log(f"📊 After backfill:")
                self._log(f"   Exit time: {row.get('exit_time')}")
                self._log(f"   TP level hit: {row.get('tp_level_hit')}")
                break
    
    def _process_single_trade_reconciliation(self, trade_id, trade_data, hours_since_entry):
        """Process single trade reconciliation with rate limiting"""
        try:
            # Acquire rate limit semaphore
            if not self.rate_limit_semaphore.acquire(timeout=30):
                self._log(f"❌ Rate limit timeout for {trade_id}", 'warning')
                return
            
            try:
                # Strategy based on trade age
                if hours_since_entry > 2:
                    self._log(f"🕰️ Processing OLD trade {trade_id} ({hours_since_entry:.1f}h old)")
                    
                    # First try to backfill from historical data
                    completed = self._backfill_trade_from_history(trade_data)
                    
                    if completed:
                        self._log(f"✅ Trade {trade_id} completed via backfill")
                    else:
                        # Still open - start live monitoring thread
                        self._log(f"📡 Trade {trade_id} still open after backfill, starting live thread")
                        self.start_live_monitoring(trade_data)
                
                else:
                    # Recent trade - start live monitoring
                    self._log(f"📡 Starting live thread for RECENT trade {trade_id} ({hours_since_entry:.1f}h old)")
                    self.start_live_monitoring(trade_data)
                    
            finally:
                # Release semaphore
                self.rate_limit_semaphore.release()
                time.sleep(0.5)  # Small delay between API calls
                
        except Exception as e:
            self._log(f"❌ Error processing trade {trade_id}: {e}", 'error')
    
    def _trade_needs_processing(self, row):
        """Check if a trade needs TP/SL results or monitoring - IMPROVED with orphan detection"""
        # Check if trade already has exit results
        exit_time = row.get('exit_time', '')
        if not self._is_empty_value(exit_time):
            return False  # Already has exit time
        
        # Check monitoring status
        monitoring_status = row.get('monitoring_status', '')
        if monitoring_status == 'completed':
            return False
        
        # Check if trade is an orphan (missing TP results)
        return self._is_orphan_row(row)
    
    def _is_orphan_row(self, row):
        """Check if a trade row is orphaned (missing TP/SL results)"""
        # Skip if we don't have essential info
        if self._is_empty_value(row.get('entry_price')) or self._is_empty_value(row.get('sl_price')):
            return False
        
        # List of fields that should be filled for a completed trade
        orphan_fields = [
            'tp_1_1_result', 'tp_1_1_time_seconds',
            'tp_1_2_result', 'tp_1_2_time_seconds',
            'tp_1_3_result', 'tp_1_3_time_seconds',
            'tp_1_4_result', 'tp_1_4_time_seconds',
            'tp_1_5_result', 'tp_1_5_time_seconds',
            'tp_1_6_result', 'tp_1_6_time_seconds',
            'tp_1_7_result', 'tp_1_7_time_seconds',
            'tp_1_8_result', 'tp_1_8_time_seconds',
            'tp_1_9_result', 'tp_1_9_time_seconds',
            'tp_1_10_result', 'tp_1_10_time_seconds',
            'open_tp_result', 'open_tp_time_seconds'
        ]
        
        # Check if any of these fields are empty
        for field in orphan_fields:
            val = row.get(field, '')
            if self._is_empty_value(val) or val == '0':
                return True
        
        return False
    
    def get_orphan_trades(self, limit=10):
        """Get list of orphan trades that need processing"""
        fieldnames, rows = self._read_csv_safe()
        orphans = []
        
        for row in rows:
            if self._is_orphan_row(row):
                trade_id = row.get('trade_id', '')
                if trade_id:
                    orphans.append((row.get('entry_time', ''), trade_id, row))
        
        # Sort by entry_time descending (newest first)
        orphans.sort(key=lambda x: x[0], reverse=True)
        
        # Return limited number
        limited_orphans = orphans[:limit]
        
        self._log(f"🧩 Found {len(orphans)} orphan trades, processing {len(limited_orphans)}")
        
        # Return just the trade_data dictionaries
        return [trade_data for _, _, trade_data in limited_orphans]
    
    def get_resume_point_for_trade(self, trade_data):
        """Determine where to resume monitoring for an orphan trade"""
        # Check which TPs have been hit
        hit_tps = []
        
        for i in range(1, 11):
            result = trade_data.get(f'tp_1_{i}_result', '')
            if result and result.startswith('+'):  # +1, +2, etc.
                hit_tps.append(i)
        
        if hit_tps:
            # Find the highest TP hit
            highest_tp = max(hit_tps)
            self._log(f"📊 Trade {trade_data.get('trade_id')}: TP{highest_tp} was hit")
            
            # Resume from the TP after the highest one hit
            resume_from_tp = highest_tp + 1
            
            if resume_from_tp <= 10:
                self._log(f"🔄 Will resume monitoring from TP{resume_from_tp}")
                return {
                    'type': 'resume_from_tp',
                    'tp_level': resume_from_tp,
                    'hit_tps': hit_tps
                }
            else:
                self._log(f"✅ All TPs hit, checking open TP")
                # Check open TP
                open_tp_result = trade_data.get('open_tp_result', '')
                if self._is_empty_value(open_tp_result):
                    return {
                        'type': 'resume_open_tp',
                        'hit_tps': hit_tps
                    }
                else:
                    self._log(f"✅ Trade already completed (all TPs and open TP hit)")
                    return None
        else:
            # No TP hit yet, resume from TP1
            self._log(f"🔄 No TP hit yet, will resume from TP1")
            return {
                'type': 'resume_from_tp',
                'tp_level': 1,
                'hit_tps': []
            }
    
    def _safe_orphan_reconciliation(self):
        """New reconciliation: Process orphan trades with resume logic"""
        self._log("🧩 Starting ORPHAN trade reconciliation...")
        time.sleep(5)
        
        # Step 1: Get orphan trades
        orphans = self.get_orphan_trades(limit=5)
        
        if not orphans:
            self._log("✅ No orphan trades found")
            return
        
        self._log(f"🔄 Processing {len(orphans)} orphan trades...")
        
        for idx, trade_data in enumerate(orphans, 1):
            trade_id = trade_data.get('trade_id', 'Unknown')
            instrument = trade_data.get('instrument', '')
            
            self._log(f"--- Orphan Trade {idx}/{len(orphans)}: {trade_id} ---")
            
            try:
                # Get the latest data from CSV
                current_row = self._get_trade_row_from_csv(trade_id)
                if not current_row:
                    self._log(f"⚠️ Could not reload {trade_id} from CSV, skipping")
                    continue
                
                fresh_trade_data = self._row_to_trade_data(current_row)
                entry_time = self._parse_datetime(fresh_trade_data.get('entry_time'))
                
                if not entry_time:
                    self._log(f"⚠️ Invalid entry time for {trade_id}, skipping")
                    continue
                
                # Determine if we should backfill or resume live monitoring
                hours_since_entry = (self._now_ny() - entry_time).total_seconds() / 3600
                
                if hours_since_entry > 2:
                    # Old orphan - try to backfill from history
                    self._log(f"🕰️ Backfilling old orphan trade ({hours_since_entry:.1f}h)...")
                    
                    # Determine resume point
                    resume_info = self.get_resume_point_for_trade(fresh_trade_data)
                    
                    if resume_info:
                        self._log(f"📊 Resume info: {resume_info}")
                        
                        # Special backfill that starts from the resume point
                        completed = self._backfill_trade_from_history_resume(fresh_trade_data, resume_info)
                        
                        if completed:
                            self._log(f"✅ Orphan backfill completed for {trade_id}")
                        else:
                            self._log(f"⏸️ Orphan {trade_id} still open after backfill")
                    else:
                        # Already completed, just update status
                        self._update_trade_in_csv_safe(trade_id, {
                            'monitoring_status': 'completed',
                            'last_heartbeat': self._now_ny().isoformat()
                        })
                        self._log(f"✅ Orphan {trade_id} already completed, updating status")
                    
                else:
                    # Recent orphan - resume live monitoring
                    self._log(f"📡 Resuming live monitoring for recent orphan ({hours_since_entry:.1f}h)...")
                    
                    # Determine resume point
                    resume_info = self.get_resume_point_for_trade(fresh_trade_data)
                    
                    if resume_info:
                        self._log(f"📊 Resume info: {resume_info}")
                        
                        # Start monitoring with resume info
                        self.start_live_monitoring_with_resume(fresh_trade_data, resume_info)
                    else:
                        # Already completed
                        self._update_trade_in_csv_safe(trade_id, {
                            'monitoring_status': 'completed',
                            'last_heartbeat': self._now_ny().isoformat()
                        })
                        self._log(f"✅ Orphan {trade_id} already completed")
                
            except Exception as e:
                self._log(f"❌ Error processing orphan {trade_id}: {e}", 'error')
            
            # Wait between orphans
            wait_seconds = 10
            self._log(f"⏳ Waiting {wait_seconds}s before next orphan...")
            time.sleep(wait_seconds)
        
        self._log("✅ Orphan reconciliation complete")
    
    def start_live_monitoring_with_resume(self, trade_data, resume_info):
        """Start live monitoring from a specific resume point"""
        trade_id = trade_data['trade_id']
        
        # Don't start if already completed
        if not self._is_orphan_row(trade_data):
            self._log(f"⏭️ Trade {trade_id} not an orphan, skipping")
            return
        
        # Check if already being monitored
        if trade_id in self.active_threads:
            thread = self.active_threads[trade_id]
            if thread.is_alive():
                self._log(f"⏭️ Trade {trade_id} already being monitored")
                return
        
        try:
            thread = threading.Thread(
                target=self._monitor_trade_live_with_resume,
                args=(trade_data, resume_info),
                name=f"TPMonitor_Resume_{trade_id}",
                daemon=True
            )
            
            self.active_threads[trade_id] = thread
            self.monitoring_start_times[trade_id] = self._now_ny()
            self.heartbeat_times[trade_id] = self._now_ny()
            
            # Initialize thread state with resume info
            self.thread_states[trade_id] = {
                'status': 'resumed',
                'hit_tps': set(resume_info.get('hit_tps', [])),
                'last_checked': self._now_ny(),
                'be_tracking': {i: {'state': 'waiting', 'be_triggered': False, 'outcome': 'none'} for i in range(1, 11)},
                'resume_from_tp': resume_info.get('tp_level', 1) if resume_info.get('type') == 'resume_from_tp' else None
            }
            
            thread.start()
            
            # Update CSV status
            self._update_trade_in_csv_safe(trade_id, {
                'monitoring_status': 'resumed',
                'last_heartbeat': self._now_ny().isoformat(),
                'reconciliation_attempts': str(int(trade_data.get('reconciliation_attempts', 0)) + 1)
            })
            
            self._log(f"📡 Resumed monitoring for {trade_id} from {resume_info}")
            
        except Exception as e:
            self._log(f"❌ Failed to resume monitoring for {trade_id}: {e}", 'error')
    
    def _row_to_trade_data(self, row):
        """Convert CSV row to trade_data dictionary with proper types"""
        trade_data = {}
        
        for key, value in row.items():
            if value is None:
                trade_data[key] = ''
            elif isinstance(value, str):
                # Try to convert numeric values
                if any(x in key.lower() for x in ['price', 'distance', 'pips', 'rsi', 'vwap', 'rr']):
                    try:
                        trade_data[key] = float(value)
                    except:
                        trade_data[key] = value
                else:
                    trade_data[key] = value
            else:
                trade_data[key] = value
        
        return trade_data
    
    def _backfill_trade_from_history(self, trade_data):
        """
        Backfill trade results from historical candles
        Returns: True if trade completed, False if still open
        """
        trade_id = trade_data['trade_id']
        instrument = trade_data['instrument']
        direction = trade_data['direction'].lower()
        
        try:
            # Fix entry_time parsing
            entry_time = self._parse_datetime(trade_data['entry_time'])
            if entry_time is None:
                self._log(f"❌ Invalid entry_time for backfill {trade_id}", 'error')
                return False
            
            now = self._now_ny()
            
            # Choose timeframe based on duration
            time_diff_hours = (now - entry_time).total_seconds() / 3600
            
            if time_diff_hours <= 83:  # Up to ~3.5 days
                timeframe = 'M1'
            elif time_diff_hours <= 416:  # Up to ~17 days
                timeframe = 'M5'
            elif time_diff_hours <= 1250:  # Up to ~52 days
                timeframe = 'M15'
            else:
                timeframe = 'H1'
            
            self._log(f"🕰️ Backfilling {trade_id} with {timeframe} candles ({time_diff_hours:.1f}h)")
            
            # Fetch historical candles
            candles = self._fetch_historical_candles(
                instrument=instrument,
                timeframe=timeframe,
                start_time=entry_time,
                end_time=now
            )
            
            if candles.empty:
                self._log(f"⚠️ No historical candles for {trade_id}", 'warning')
                return False
            
            self._log(f"📊 Processing {len(candles)} candles for {trade_id}")
            
            # Get prices
            entry_price = float(trade_data['entry_price'])
            sl_price = float(trade_data['sl_price'])
            
            # Calculate TP prices
            pip_multiplier = 100 if 'JPY' in instrument else 10000
            tp_prices = {}
            
            for i in range(1, 11):
                distance_key = f'tp_1_{i}_distance'
                if distance_key in trade_data:
                    try:
                        distance_pips = float(trade_data[distance_key])
                        if direction == 'bearish':
                            tp_prices[i] = entry_price - (distance_pips / pip_multiplier)
                        else:  # bullish
                            tp_prices[i] = entry_price + (distance_pips / pip_multiplier)
                    except:
                        tp_prices[i] = None
                else:
                    tp_prices[i] = None
            
            # Process candles chronologically
            hit_tps = set()
            sl_hit = False
            highest_tp_hit = 0
            exit_time = None
            
            for idx, candle in candles.iterrows():
                candle_time = candle['time']
                # FIX: Ensure candle_time is timezone-aware
                candle_time = self._ensure_timezone_aware(candle_time)
                candle_high = float(candle['high'])
                candle_low = float(candle['low'])
                
                # Check SL
                if direction == 'bearish' and candle_low <= sl_price:
                    sl_hit = True
                    exit_time = candle_time
                    self._log(f"🛑 Backfill: SL hit for {trade_id} at {candle_time}")
                    break
                
                elif direction == 'bullish' and candle_high >= sl_price:
                    sl_hit = True
                    exit_time = candle_time
                    self._log(f"🛑 Backfill: SL hit for {trade_id} at {candle_time}")
                    break
                
                # Check TPs
                for i in range(1, 11):
                    if i not in hit_tps and tp_prices[i] is not None:
                        if direction == 'bearish' and candle_low <= tp_prices[i]:
                            hit_tps.add(i)
                            highest_tp_hit = max(highest_tp_hit, i)
                            exit_time = candle_time
                            self._log(f"✅ Backfill: TP{i} hit for {trade_id} at {candle_time}")
                        
                        elif direction == 'bullish' and candle_high >= tp_prices[i]:
                            hit_tps.add(i)
                            highest_tp_hit = max(highest_tp_hit, i)
                            exit_time = candle_time
                            self._log(f"✅ Backfill: TP{i} hit for {trade_id} at {candle_time}")
            
            # Prepare updates for CSV
            updates = {}
            
            if sl_hit:
                # SL was hit - CRITICAL: Check if any TP was hit already
                tp_was_hit = False
                for i in range(1, 11):
                    result = trade_data.get(f'tp_1_{i}_result', '')
                    if result and result.startswith('+'):  # +1, +2, etc.
                        tp_was_hit = True
                        break
                
                # Only set tp_level_hit to -1 if NO TP was hit
                if not tp_was_hit:
                    updates['tp_level_hit'] = '-1'
                
                if exit_time:
                    updates['exit_time'] = exit_time.strftime('%Y-%m-%d %H:%M:%S')
                
                # Set all TPs to -1 (that aren't already filled)
                for i in range(1, 11):
                    if trade_data.get(f'tp_1_{i}_result') == '':
                        updates[f'tp_1_{i}_result'] = '-1'
                        updates[f'tp_1_{i}_time_seconds'] = '0'
                
                # Calculate time to exit
                if exit_time:
                    time_to_exit = (exit_time - entry_time).total_seconds()
                    updates['time_to_exit_seconds'] = int(time_to_exit)
                
                self._update_trade_in_csv_safe(trade_id, updates)
                time.sleep(0.1)
                self.verify_backfill_update(trade_id, updates)
                return True
                
            elif hit_tps:
                # Some TPs were hit
                updates['tp_level_hit'] = str(highest_tp_hit)
                if exit_time:
                    updates['exit_time'] = exit_time.strftime('%Y-%m-%d %H:%M:%S')
                
                # Record TP hits
                for i in range(1, 11):
                    if i in hit_tps:
                        updates[f'tp_1_{i}_result'] = f'+{i}'
                    else:
                        updates[f'tp_1_{i}_result'] = '-1'
                
                # Calculate time to exit
                if exit_time:
                    time_to_exit = (exit_time - entry_time).total_seconds()
                    updates['time_to_exit_seconds'] = int(time_to_exit)
                
                self._update_trade_in_csv_safe(trade_id, updates)
                time.sleep(0.1)  # Small delay for write to complete
                self.verify_backfill_update(trade_id, updates)
                return True
                
            else:
                # No SL or TP hit - trade is still open
                self._log(f"⏳ Backfill: Trade {trade_id} still open (no SL/TP hit)")
                return False
                
        except Exception as e:
            self._log(f"❌ Backfill error for {trade_id}: {e}", 'error')
            return False

    def find_trade_by_pattern(self, pattern):
        """Find trades by pattern (partial trade_id match)"""
        fieldnames, rows = self._read_csv_safe()
        
        matches = []
        for row in rows:
            trade_id = row.get('trade_id', '')
            if pattern in trade_id:
                matches.append(row)
        
        self._log(f"🔍 Found {len(matches)} trades matching '{pattern}'")
        for match in matches[:5]:  # Show first 5
            self._log(f"   - {match.get('trade_id')}: entry={match.get('entry_time')}, exit={match.get('exit_time')}")
        
        return matches
    
    def _rate_limited_fetch_candles(self, instrument, timeframe, count, since=None):
        """Fetch candles with rate limiting"""
        # Track API call times
        with self.api_call_lock:
            now = time.time()
            # Remove calls older than 60 seconds
            self.api_call_times = [t for t in self.api_call_times if now - t < 60]
            
            # Check if we're hitting rate limits (max 60 calls per minute)
            if len(self.api_call_times) >= 50:
                wait_time = 60 - (now - min(self.api_call_times)) + 1
                self._log(f"⏳ Rate limit approaching, waiting {wait_time:.1f}s")
                time.sleep(wait_time)
            
            # Record this call
            self.api_call_times.append(now)
        
        try:
            return self.fetch_candles(
                instrument=instrument,
                timeframe=timeframe,
                count=count,
                api_key=self.api_key,
                since=since,
                use_cache=True
            )
        except Exception as e:
            if '502' in str(e) or '429' in str(e) or 'rate' in str(e).lower():
                # Exponential backoff for rate limits
                wait_time = min(30, 5 * (len(self.api_call_times) // 10))
                self._log(f"⚠️ API rate limit hit for {instrument}, waiting {wait_time}s", 'warning')
                time.sleep(wait_time)
                # Retry once
                return self.fetch_candles(
                    instrument=instrument,
                    timeframe=timeframe,
                    count=count,
                    api_key=self.api_key,
                    since=since,
                    use_cache=True
                )
            else:
                raise
    
    def _fetch_historical_candles(self, instrument, timeframe, start_time, end_time):
        """Fetch historical candles with rate limiting"""
        try:
            # Calculate time difference to estimate candle count
            time_diff_hours = (end_time - start_time).total_seconds() / 3600
            
            # Estimate candle count needed
            if timeframe == 'M1':
                count = int(time_diff_hours * 60 * 1.2) + 50  # 20% buffer
            elif timeframe == 'M5':
                count = int(time_diff_hours * 12 * 1.2) + 50
            elif timeframe == 'M15':
                count = int(time_diff_hours * 4 * 1.2) + 50
            elif timeframe == 'H1':
                count = int(time_diff_hours * 1.2) + 50
            else:
                count = 100
            
            # Limit to max 3000 candles to avoid API limits
            count = min(count, 3000)
            
            self._log(f"📊 Fetching {count} {timeframe} candles for {instrument} ({time_diff_hours:.1f}h)")
            
            # Use rate-limited fetch
            df = self._rate_limited_fetch_candles(
                instrument=instrument,
                timeframe=timeframe,
                count=count,
                since=start_time
            )
            
            # Filter to our date range
            if not df.empty:
                df = df[(df['time'] >= start_time) & (df['time'] <= end_time)]
                self._log(f"✅ Got {len(df)} candles for backfill")
            else:
                self._log(f"⚠️ No candles returned for {instrument}")
            
            return df
            
        except Exception as e:
            self._log(f"❌ Error fetching historical candles for {instrument}: {e}", 'error')
            import pandas as pd
            return pd.DataFrame()
    
    def _fetch_current_candles(self, instrument, timeframe, count):
        """Fetch current candles for live monitoring with rate limiting"""
        try:
            return self._rate_limited_fetch_candles(
                instrument=instrument,
                timeframe=timeframe,
                count=count,
                since=None
            )
        except Exception as e:
            self._log(f"❌ Error fetching current candles for {instrument}: {e}", 'error')
            import pandas as pd
            return pd.DataFrame()
    
    def start_live_monitoring(self, trade_data):
        """Start live monitoring thread for a trade with thread pool limits"""
        trade_id = trade_data['trade_id']
        
        # Check if already being monitored
        if trade_id in self.active_threads:
            thread = self.active_threads[trade_id]
            if thread.is_alive():
                self._log(f"⏭️ Trade {trade_id} already being monitored")
                return
        
        # Check if we've reached max workers
        active_count = len([t for t in self.active_threads.values() if t.is_alive()])
        if active_count >= self.max_workers:
            self._log(f"⏳ Max workers reached ({self.max_workers}), queuing {trade_id}")
            # Queue it for later - maybe store in a pending queue
            return
        
        # Check if trade is already completed
        if self._is_trade_completed(trade_data):
            self._log(f"⏭️ Trade {trade_id} already completed")
            return
        
        try:
            # Acquire rate limit semaphore
            if not self.rate_limit_semaphore.acquire(timeout=10):
                self._log(f"❌ Rate limit timeout for starting {trade_id}", 'warning')
                return
            
            try:
                # Start monitoring thread
                thread = threading.Thread(
                    target=self._monitor_trade_live,
                    args=(trade_data,),
                    name=f"TPMonitor_{trade_id}",
                    daemon=True
                )
                
                self.active_threads[trade_id] = thread
                self.monitoring_start_times[trade_id] = self._now_ny()
                self.heartbeat_times[trade_id] = self._now_ny()
                
                # Initialize thread state
                self.thread_states[trade_id] = {
                    'status': 'running',
                    'hit_tps': set(),
                    'last_checked': self._now_ny(),
                    'be_tracking': {i: {'state': 'waiting', 'be_triggered': False, 'outcome': 'none'} for i in range(1, 11)}
                }
                
                thread.start()
                
                # Update CSV status
                self._update_trade_in_csv_safe(trade_id, {
                    'monitoring_status': 'running',
                    'last_heartbeat': self._now_ny().isoformat(),
                    'reconciliation_attempts': str(int(trade_data.get('reconciliation_attempts', 0)) + 1)
                })
                
                self._log(f"📡 Started LIVE monitoring for {trade_id} (active: {active_count + 1}/{self.max_workers})")
                
            finally:
                # Release semaphore after starting thread (not after it completes)
                self.rate_limit_semaphore.release()
                
        except Exception as e:
            self._log(f"❌ Failed to start live monitoring for {trade_id}: {e}", 'error')
    
    def _monitor_trade_live(self, trade_data):
        """Live monitoring thread - FIXED TIMEZONE ISSUES"""
        trade_id = trade_data['trade_id']
        instrument = trade_data['instrument']
        direction = trade_data['direction'].lower()
        
        try:
            entry_price = float(trade_data['entry_price'])
            sl_price = float(trade_data['sl_price'])
            
            # Calculate TP prices
            pip_multiplier = 100 if 'JPY' in instrument else 10000
            tp_prices = {}
            
            for i in range(1, 11):
                distance_key = f'tp_1_{i}_distance'
                if distance_key in trade_data:
                    try:
                        distance_pips = float(trade_data[distance_key])
                        if direction == 'bearish':
                            tp_prices[i] = entry_price - (distance_pips / pip_multiplier)
                        else:  # bullish
                            tp_prices[i] = entry_price + (distance_pips / pip_multiplier)
                    except:
                        tp_prices[i] = None
                else:
                    tp_prices[i] = None
            
            # Initialize tracking
            hit_tps = set()
            be_tracking = {i: {'state': 'waiting', 'be_triggered': False, 'outcome': 'none'} for i in range(1, 11)}
            
            # Get entry time for timeout calculation - FIXED TIMEZONE
            entry_time = self._parse_datetime(trade_data['entry_time'])
            if entry_time is None:
                self._log(f"❌ Invalid entry_time for {trade_id}", 'error')
                return
            
            self._log(f"📊 LIVE monitoring started for {trade_id}")
            
            # Main monitoring loop
            while not self.shutdown_flag.is_set():
                try:
                    # Check if trade is already completed
                    if self._is_trade_completed(trade_data):
                        self._log(f"✅ Trade {trade_id} completed during monitoring")
                        break
                    
                    # Fetch current candles
                    candles = self._fetch_current_candles(instrument, 'M1', count=2)
                    
                    if not candles.empty:
                        latest_candle = candles.iloc[-1]
                        
                        # FIX TIMEZONE: Ensure current_time is timezone-aware
                        current_time = latest_candle['time']
                        current_time = self._ensure_timezone_aware(current_time)
                        
                        current_high = float(latest_candle['high'])
                        current_low = float(latest_candle['low'])
                        current_close = float(latest_candle['close'])
                        
                        # Check SL
                        sl_hit = False
                        if direction == 'bearish' and current_low <= sl_price:
                            sl_hit = True
                        elif direction == 'bullish' and current_high >= sl_price:
                            sl_hit = True
                        
                        if sl_hit:
                            # Record SL hit
                            updates = self._record_sl_hit_updates(trade_data, current_time, hit_tps, be_tracking)
                            self._update_trade_in_csv_safe(trade_id, updates)
                            self._log(f"🛑 SL hit for {trade_id}")
                            break
                        
                        # Check TPs
                        for i in range(1, 11):
                            if i not in hit_tps and tp_prices[i] is not None:
                                tp_hit = False
                                if direction == 'bearish' and current_low <= tp_prices[i]:
                                    tp_hit = True
                                elif direction == 'bullish' and current_high >= tp_prices[i]:
                                    tp_hit = True
                                
                                if tp_hit:
                                    hit_tps.add(i)
                                    # FIX TIMEZONE: Both datetimes are now timezone-aware
                                    time_seconds = (current_time - entry_time).total_seconds()
                                    
                                    # Update CSV
                                    updates = {
                                        f'tp_1_{i}_result': f'+{i}',
                                        f'tp_1_{i}_time_seconds': str(int(time_seconds))
                                    }
                                    
                                    # Update highest TP hit
                                    current_highest = int(trade_data.get('tp_level_hit', 0) or 0)
                                    if i > current_highest:
                                        updates['tp_level_hit'] = str(i)
                                        updates['exit_time'] = current_time.strftime('%Y-%m-%d %H:%M:%S')
                                        updates['time_to_exit_seconds'] = str(int(time_seconds))
                                    
                                    self._update_trade_in_csv_safe(trade_id, updates)
                                    trade_data.update(updates)  # Update local copy
                                    
                                    # Start BE tracking for this TP
                                    be_tracking[i]['state'] = 'tracking'
                                    self._log(f"✅ TP{i} hit for {trade_id}")
                        
                        # Update BE tracking
                        self._update_be_tracking(trade_data, hit_tps, be_tracking, current_close, tp_prices)
                        
                        # Update heartbeat
                        self.heartbeat_times[trade_id] = datetime.now()
                    
                    # Sleep between checks
                    time.sleep(self.check_interval_live)
                    
                except Exception as e:
                    self._log(f"⚠️ Error in monitoring loop for {trade_id}: {e}", 'warning')
                    time.sleep(5)
            
            # Cleanup
            self._cleanup_trade_monitoring(trade_id, 'completed')
            
        except Exception as e:
            self._log(f"❌ Monitoring crashed for {trade_id}: {e}", 'error')
            self._update_trade_in_csv_safe(trade_id, {'monitoring_status': 'failed'})
            self._cleanup_trade_monitoring(trade_id, 'failed')
    
    def _record_sl_hit_updates(self, trade_data, hit_time, hit_tps, be_tracking):
        """Create updates dictionary for SL hit - FIXED: Only set tp_level_hit=-1 if NO TP was hit"""
        updates = {}
        
        # CRITICAL FIX: First check if ANY TP was hit
        tp_was_hit = False
        for i in range(1, 11):
            result = trade_data.get(f'tp_1_{i}_result', '')
            if result and result.startswith('+'):  # +1, +2, etc.
                tp_was_hit = True
                break
        
        # Record -1 for all TPs not hit
        for i in range(1, 11):
            if trade_data.get(f'tp_1_{i}_result') == '':
                updates[f'tp_1_{i}_result'] = '-1'
                # Also set time_seconds to 0 for consistency
                if trade_data.get(f'tp_1_{i}_time_seconds') == '':
                    updates[f'tp_1_{i}_time_seconds'] = '0'
        
        # Only set highest TP to -1 if NO TP was hit at all
        if not tp_was_hit:
            updates['tp_level_hit'] = '-1'
        else:
            # If TP was hit, don't change tp_level_hit - keep the highest TP that was hit
            # Find the current highest TP hit
            current_highest = 0
            for i in range(1, 11):
                result = trade_data.get(f'tp_1_{i}_result', '')
                if result and result.startswith('+'):
                    try:
                        tp_num = int(result.replace('+', ''))
                        if tp_num > current_highest:
                            current_highest = tp_num
                    except:
                        pass
            
            # Update exit time if not already set
            if trade_data.get('exit_time') == '':
                updates['exit_time'] = hit_time.strftime('%Y-%m-%d %H:%M:%S')
                
                # Calculate time to exit if not set
                if trade_data.get('time_to_exit_seconds') == '':
                    entry_time = self._parse_datetime(trade_data['entry_time'])
                    if entry_time:
                        time_to_exit = (hit_time - entry_time).total_seconds()
                        updates['time_to_exit_seconds'] = str(int(time_to_exit))
        
        # Update BE outcomes
        for tp_level in hit_tps:
            if be_tracking[tp_level]['state'] == 'tracking':
                if be_tracking[tp_level]['be_triggered']:
                    updates[f'if_BE_TP{tp_level}'] = 'hit'
                else:
                    updates[f'if_BE_TP{tp_level}'] = 'miss'
        
        return updates
    
    def _update_be_tracking(self, trade_data, hit_tps, be_tracking, current_price, tp_prices):
        """Update BE tracking logic"""
        entry_price = float(trade_data['entry_price'])
        
        for tp_level in hit_tps:
            if be_tracking[tp_level]['state'] == 'tracking':
                # Check if price returned to entry (BE)
                if abs(current_price - entry_price) <= self.be_tolerance:
                    be_tracking[tp_level]['be_triggered'] = True
                
                # Check if next TP is hit
                if tp_level < 10 and tp_prices.get(tp_level + 1):
                    # We would check if next TP is hit here
                    # This is simplified - you need to implement the logic
                    pass
    
    def _is_trade_completed(self, trade_data):
        """Check if trade is completed (all TPs have results)"""
        for i in range(1, 11):
            result = trade_data.get(f'tp_1_{i}_result', '')
            if self._is_empty_value(result):
                return False
        return True
    
    def _update_trade_in_csv_safe(self, trade_id, updates):
        """Update specific fields for ONE trade - PRESERVES ALL OTHER DATA"""
        # MAX RETRIES to handle race conditions
        for attempt in range(3):
            with self.csv_lock:
                try:
                    # STEP 1: Read current state ATOMICALLY
                    fieldnames, rows = self._read_csv_safe()
                    if not fieldnames:
                        self._log(f"❌ Attempt {attempt+1}: No fieldnames for {trade_id}")
                        time.sleep(0.5)
                        continue
                    
                    if not rows:
                        self._log(f"❌ Attempt {attempt+1}: No rows in CSV for {trade_id}")
                        return False
                    
                    # STEP 2: Find the trade (handle multiple matches)
                    found_indices = []
                    for i, row in enumerate(rows):
                        if row.get('trade_id') == trade_id:
                            found_indices.append(i)
                    
                    if not found_indices:
                        self._log(f"❌ Trade {trade_id} not found in {len(rows)} rows")
                        # Log first few trade IDs for debugging
                        if rows:
                            sample_ids = [r.get('trade_id', 'NO_ID') for r in rows[:5]]
                            self._log(f"   Sample IDs: {sample_ids}")
                        return False
                    
                    # STEP 3: Apply updates to ALL matching rows (should be only one)
                    updated_count = 0
                    for idx in found_indices:
                        row_updated = False
                        for key, new_value in updates.items():
                            if key in fieldnames:
                                old_value = rows[idx].get(key, '')
                                # Only update if changing to a non-empty value
                                if str(new_value).strip() and str(old_value) != str(new_value):
                                    rows[idx][key] = str(new_value)
                                    row_updated = True
                        
                        if row_updated:
                            updated_count += 1
                            # Update heartbeat
                            rows[idx]['last_heartbeat'] = datetime.now().isoformat()
                    
                    # STEP 4: Write back ONLY if changes were made
                    if updated_count > 0:
                        self._log(f"📝 Updating {trade_id}: {len(updates)} fields on {updated_count} rows")
                        success = self._write_csv_safe(fieldnames, rows, f"update_{trade_id}")
                        if success:
                            self._log(f"✅ Successfully updated {trade_id}")
                            return True
                        else:
                            self._log(f"⚠️  Write failed for {trade_id}, retrying...")
                            time.sleep(1)
                            continue
                    else:
                        self._log(f"⏭️  No updates needed for {trade_id} (already current)")
                        return True
                        
                except Exception as e:
                    self._log(f"❌ Update error for {trade_id} (attempt {attempt+1}): {e}")
                    time.sleep(1)
        
        self._log(f"💥 FAILED to update {trade_id} after 3 attempts")
        return False

    def verify_backfill_update(self, trade_id, expected_updates):
        """Verify that backfill updates were applied correctly"""
        fieldnames, rows = self._read_csv_safe()
        
        for row in rows:
            if row.get('trade_id') == trade_id:
                self._log(f"🔍 VERIFICATION for {trade_id}:")
                
                all_correct = True
                for key, expected_value in expected_updates.items():
                    actual_value = row.get(key, '')
                    if str(actual_value) != str(expected_value):
                        self._log(f"❌ Mismatch: {key} = '{actual_value}' (expected '{expected_value}')")
                        all_correct = False
                    else:
                        self._log(f"✅ Match: {key} = '{actual_value}'")
                
                if all_correct:
                    self._log(f"✅ All updates verified for {trade_id}")
                else:
                    self._log(f"❌ Some updates missing for {trade_id}")
                
                return all_correct
        
        self._log(f"❌ Trade {trade_id} not found for verification")
        return False
    
    def _periodic_health_check(self):
        """Periodic health check for monitoring threads"""
        while not self.shutdown_flag.is_set():
            try:
                # Check for dead threads and restart them
                dead_threads = []
                for trade_id, thread in list(self.active_threads.items()):
                    if not thread.is_alive():
                        dead_threads.append(trade_id)
                
                for trade_id in dead_threads:
                    self._log(f"⚠️ Thread for {trade_id} died, attempting to restart", 'warning')
                    
                    # Get trade data from CSV
                    fieldnames, rows = self._read_csv_safe()
                    for row in rows:
                        if row.get('trade_id') == trade_id:
                            trade_data = self._row_to_trade_data(row)
                            
                            # Check if still needs monitoring
                            if not self._is_trade_completed(trade_data):
                                self.start_live_monitoring(trade_data)
                            else:
                                self._log(f"⏭️ Trade {trade_id} already completed, not restarting")
                            break
                
                # Wait before next check
                time.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                self._log(f"⚠️ Error in health check: {e}", 'warning')
                time.sleep(60)
    
    def _cleanup_trade_monitoring(self, trade_id, status):
        """Clean up monitoring resources for a trade"""
        if trade_id in self.active_threads:
            del self.active_threads[trade_id]
        
        if trade_id in self.thread_states:
            del self.thread_states[trade_id]
        
        if trade_id in self.heartbeat_times:
            del self.heartbeat_times[trade_id]
        
        if trade_id in self.monitoring_start_times:
            del self.monitoring_start_times[trade_id]
        
        # Update CSV status
        self._update_trade_in_csv_safe(trade_id, {
            'monitoring_status': status,
            'last_heartbeat': datetime.now().isoformat()
        })
        
        self._log(f"🧹 Cleaned up monitoring for {trade_id} ({status})")
    
    def stop_all_monitoring(self):
        """Gracefully stop all monitoring"""
        self._log("🛑 Stopping all TP monitoring...")
        self.shutdown_flag.set()
        
        # Wait for threads to finish
        for trade_id, thread in list(self.active_threads.items()):
            if thread.is_alive():
                thread.join(timeout=5)
        
        # Clear all data structures
        self.active_threads.clear()
        self.thread_states.clear()
        self.heartbeat_times.clear()
        self.monitoring_start_times.clear()
        
        self._log("✅ All TP monitoring stopped")
    
    def _get_trade_row_from_csv(self, trade_id):
        """Helper to get a specific trade row from CSV"""
        fieldnames, rows = self._read_csv_safe()
        for row in rows:
            if row.get('trade_id') == trade_id:
                return row
        return None


class HammerPatternScanner:
    def __init__(self, credentials, csv_base_path='/content/drive/MyDrive/hammer_trades', 
                 logger=None, news_calendar=None ,signal_processor=None, quarter_manager=None ):  # ADD news_calendar parameter
        """Concurrent hammer pattern scanner with shared news calendar support"""
        
        self.credentials = credentials

        # Use passed logger or create new ones
        if logger:
            self.logger = logger
        else:
            # Create logger if not provided
            import logging
            self.logger = logging.getLogger('HammerScanner')
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)
        
        self.logger.info("🔨 Initializing HammerPatternScanner...")
                     
                     
        # Webhook configuration - set your values here
        self.webhook_url = "https://d4a270af4ba7.ngrok-free.app/webhook"
        self.webhook_token = "uVDdSdTrQCDiAQwU9YR-LIeHMKJ8Ewgz"  
        self.running = False
        self.scanner_thread = None
        self.active_scans = {}
        self.quarter_manager = RobustQuarterManager()
        self.true_open_cache = {}  # Format: {instrument: {cycle: {quarter: price}}}
        self.true_open_lock = threading.Lock()  # For thread safety
        
        # Add caching for API calls
        self.data_cache = {}
        self.cache_expiry = {}  # Track when cache expires
        self.cache_duration = 45  # Cache for 60 seconds

        self.entry_counter = {}
        self.entry_counter_lock = threading.Lock() 

        # Add this instead:
        self.candle_cache = CandleDataCache(
            ttl_seconds=45,      # Cache for 45 seconds
            max_size=2000         # Store up to 200 cache entries
        )
        self.logger.info(f"📦 Candle data cache initialized (TTL: 45s, Max: 2000 entries)")
        self.signal_processor = signal_processor
        if signal_processor and logger:
            logger.info("🔗 SignalProcessor connected to HammerPatternScanner")

        
                     
                     
        
        # Use passed logger or create new one
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger('HammerScanner')
        
        # CSV configuration
        self.csv_base_path = csv_base_path.rstrip('_')
        self.csv_file_path = f"{self.csv_base_path}.csv"
        self.init_csv_storage()
        self.tp_monitor = SafeTPMonitoringManager(
            csv_path=self.csv_file_path,
            api_key=self.credentials['oanda_api_key'],
            fetch_candles_func=fetch_candles,  # Pass the function
            logger=self.logger
        )
        
        # Use shared news calendar if provided, otherwise create one
        self.news_calendar = news_calendar
        if not self.news_calendar:
            rapidapi_key = os.getenv('rapidapi_key')
            if rapidapi_key:
                try:
                    self.news_calendar = NewsCalendar(
                        rapidapi_key=rapidapi_key,
                        base_path='/content/drive/MyDrive',
                        logger=self.logger
                    )
                    self.logger.info(f"📰 News Calendar initialized (local instance)")
                except Exception as e:
                    self.logger.error(f"❌ Failed to initialize News Calendar: {str(e)}")
            else:
                self.logger.warning(f"⚠️ No RapidAPI key provided - News Calendar disabled")
        else:
            self.logger.info(f"📰 Using shared News Calendar instance")
            # Don't start background fetch for shared calendar - it's already managed elsewhere
        
        # Timeframe alignment
        self.timeframe_alignment = {
            'XAU_USD': {
                'FVG+SMT': {'M15': ['M1', 'M3', 'M5'], 'H1': ['M3','M5','M10', 'M15'], 'H4': ['M5','M10', 'M15']},
                'SD+SMT': {'M15': ['M1', 'M3', 'M5'], 'H1': ['M5','M10', 'M15'], 'H4': ['M5','M10', 'M15'], 'D': ['M10','M15'], 'W': ['M15']},
                'CRT+SMT': {'H1': ['M3','M5', 'M10'], 'H4': ['M5','M10', 'M15']}
            },
            'default': {
                'FVG+SMT': {'M15': ['M3', 'M5'], 'H1': ['M5','M10', 'M15'], 'H4': ['M5','M10', 'M15']},
                'SD+SMT': {'M15': ['M3', 'M5','M10'], 'H1': ['M5','M10', 'M15'], 'H4': ['M5','M10', 'M15'], 'D': ['M10','M15'], 'W': ['M10','M15']},
                'CRT+SMT': {'H1': ['M5', 'M10'], 'H4': ['M5','M10', 'M15']}
            }
        }
        
        self.logger.info(f"🔨 Streamlined Hammer Scanner initialized")
        self.logger.info(f"📁 CSV storage: {self.csv_file_path}")
    
    def start(self):
        """Start the scanner - only start background fetch if using local calendar"""
        self.running = True
        
        # Only start background news fetching if we have a local calendar
        # (not a shared one)
        if self.news_calendar and not hasattr(self.news_calendar, '_is_shared'):
            self.start_news_background_fetch(interval_hours=24)  # Fetch once per day
        
        self.logger.info("🔨 Hammer Pattern Scanner started")
        return True
    
    def init_csv_storage(self):
        """Initialize CSV file with NEW columns - FIXED VERSION (no header duplication)"""
        try:
            # Ensure directory exists
            directory = os.path.dirname(self.csv_base_path)
            if directory:  # Only create if path has directory
                os.makedirs(directory, exist_ok=True)
            
            # Define UPDATED headers with ALL new columns
            headers = [
                # Core Identification & Timing
                'timestamp', 'signal_id', 'trade_id', 'instrument', 'hammer_timeframe',
                'direction','tp_level_hit', 'entry_time', 'entry_price',
                
                # Price Levels
                'sl_price', 'tp_1_4_price', 'open_tp_price',
                
                # Trade Levels (distances in pips)
                'sl_distance_pips', 'tp_1_1_distance', 'tp_1_2_distance', 'tp_1_3_distance',
                'tp_1_4_distance', 'tp_1_5_distance', 'tp_1_6_distance', 'tp_1_7_distance',
                'tp_1_8_distance', 'tp_1_9_distance', 'tp_1_10_distance',
                
                # TP Results (1-10) and Time Tracking
                'tp_1_1_result', 'tp_1_1_time_seconds',
                'tp_1_2_result', 'tp_1_2_time_seconds',
                'tp_1_3_result', 'tp_1_3_time_seconds',
                'tp_1_4_result', 'tp_1_4_time_seconds',
                'tp_1_5_result', 'tp_1_5_time_seconds',
                'tp_1_6_result', 'tp_1_6_time_seconds',
                'tp_1_7_result', 'tp_1_7_time_seconds',
                'tp_1_8_result', 'tp_1_8_time_seconds',
                'tp_1_9_result', 'tp_1_9_time_seconds',
                'tp_1_10_result', 'tp_1_10_time_seconds',
                
                # Open TP Tracking
                'open_tp_rr', 'open_tp_result', 'open_tp_time_seconds',
                
                # Trigger Criteria & Context
                'criteria', 'trigger_timeframe',
                'fvg_formation_time', 'sd_formation_time', 'crt_formation_time',
                'smt_cycle', 'smt_quarters', 'has_psp', 'is_hp_fvg', 'is_hp_zone',
                
                # Market Context
                'rsi', 'vwap',

                
                # Higher Timeframe Fibonacci Zones (Pd-tf)
                'H4_fib_zone', 'H6_fib_zone', 'D_fib_zone', 'W_fib_zone',
                'H4_fib_percent', 'H6_fib_percent', 'D_fib_percent', 'W_fib_percent',  # For more granular analysis
                
                # Price Relative to Candle Open
                'H4_open_rel', 'H6_open_rel', 'D_open_rel', 'W_open_rel',
                
                # Candle Quarter Position
                'H4_quarter', 'H6_quarter', 'D_quarter', 'W_quarter',
                
                # Additional: Price position percentage within candle
                'H4_candle_percent', 'H6_candle_percent', 'D_candle_percent', 'W_candle_percent',
                
                # zebra
                '1m_zebra', '3m_zebra', '5m_zebra', '15m_zebra', 'h1_zebra', 'h4_zebra', 'h6_zebra', 'd_zebra',
                # New Features
                'signal_latency_seconds',  # Time from candle close to Telegram send
                'hammer_volume',
                'inducement_count',  # Number of swing highs/lows
                'ma_10', 'ma_20', 'ma_30', 'ma_40', 'ma_60', 'ma_100',  # Moving averages
                'vwap_value', 'vwap_std',
                'upper_band_1', 'lower_band_1',
                'upper_band_2', 'lower_band_2',
                'upper_band_3', 'lower_band_3',
                'touches_upper_band_1', 'touches_lower_band_1',
                'touches_upper_band_2', 'touches_lower_band_2',
                'touches_upper_band_3', 'touches_lower_band_3',
                'touches_vwap',
                'far_ratio_upper_band_1', 'far_ratio_lower_band_1',
                'far_ratio_upper_band_2', 'far_ratio_lower_band_2',
                'far_ratio_upper_band_3', 'far_ratio_lower_band_3',
                'far_ratio_vwap',
                'bearish_stack', 'trend_strength_up', 'trend_strength_down',
                'prev_volume',
                # News Calendar Columns
                'news_context_json',  # JSON string with all news context
                'news_high_count', 'news_medium_count', 'news_low_count',
                'next_news_time', 'next_news_event', 'next_news_currency',
                'prev_news_time', 'prev_news_event', 'prev_news_currency',
                'seconds_to_next_news', 'seconds_since_last_news',
                'news_timing_category', 'news_fetch_status',

                # Be tracking 
                'if_BE_TP1', 'if_BE_TP2', 'if_BE_TP3', 'if_BE_TP4', 'if_BE_TP5',
                'if_BE_TP6', 'if_BE_TP7', 'if_BE_TP8', 'if_BE_TP9', 'if_BE_TP10',
                'if_BE_open_TP',
                # Quater information 
                'current_quarter_monthly', 'true_open_monthly', 'true_open_relation_monthly',
                'current_quarter_weekly', 'true_open_weekly', 'true_open_relation_weekly',
                'current_quarter_daily', 'true_open_daily', 'true_open_relation_daily',
                'current_quarter_90min', 'true_open_90min', 'true_open_relation_90min',
                
                # Result Tracking
                'exit_time', 'time_to_exit_seconds',
                'trigger_type','entry_count'
                
            ]
            
            # Store headers as instance variable for later use
            self.headers = headers
            
            # Check if file exists
            if not os.path.exists(self.csv_file_path):
                # Create new file with headers
                with open(self.csv_file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(headers)
                self.logger.info(f"📁 Created NEW CSV with {len(headers)} columns")
                return
            
            # File exists - check and update headers WITHOUT rewriting entire file
            with open(self.csv_file_path, 'r', newline='', encoding='utf-8') as f:
                try:
                    reader = csv.reader(f)
                    existing_headers = next(reader, None)
                    
                    # If file is empty, write headers
                    if existing_headers is None or len(existing_headers) == 0:
                        with open(self.csv_file_path, 'w', newline='', encoding='utf-8') as f2:
                            writer = csv.writer(f2)
                            writer.writerow(headers)
                        self.logger.info(f"📁 File was empty, wrote {len(headers)} headers")
                        return
                    
                    # Check if headers match (ignoring order)
                    existing_set = set(existing_headers)
                    headers_set = set(headers)
                    
                    if existing_set == headers_set:
                        # Same headers, just check order
                        if existing_headers == headers:
                            self.logger.info(f"📁 CSV file exists with correct headers ({len(headers)} columns)")
                        else:
                            # Same headers but different order - we should reorder
                            self.logger.info(f"📁 Headers have different order, reordering...")
                            self._reorder_csv_headers(existing_headers, headers)
                        return
                    
                    # Headers don't match - add missing columns
                    missing_headers = [h for h in headers if h not in existing_set]
                    if missing_headers:
                        self.logger.info(f"📁 Adding {len(missing_headers)} missing headers: {missing_headers}")
                        self._add_missing_headers(existing_headers, missing_headers, headers)
                    else:
                        self.logger.info(f"📁 CSV file exists with correct headers ({len(headers)} columns)")
                        
                except (csv.Error, StopIteration, UnicodeDecodeError) as e:
                    # Corrupted CSV, recreate it
                    self.logger.warning(f"📁 CSV file corrupted, recreating: {str(e)}")
                    with open(self.csv_file_path, 'w', newline='') as f2:
                        writer = csv.writer(f2)
                        writer.writerow(headers)
                    self.logger.info(f"📁 Recreated CSV with {len(headers)} headers")
                    
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize CSV: {str(e)}")

    def cached_fetch_candles(self, instrument, timeframe, count, force_fetch=False):
        """
        Fetch candles with GLOBAL caching support - used by ALL scanners
        
        Args:
            force_fetch: If True, bypass cache and fetch fresh data (for real-time)
        """
        try:
            # Generate cache key
            cache_key = f"{instrument}_{timeframe}_{count}"
            
            # Skip cache if force_fetch is True (for real-time data like M1)
            if not force_fetch:
                cached_data = self.candle_cache.get(instrument, timeframe, count)
                if cached_data is not None:
                    # Log only occasionally to avoid spam
                    if random.random() < 0.05:  # Log only 5% of cache hits
                        self.logger.debug(f"📦 Global Cache HIT: {instrument} {timeframe} count={count}")
                    return cached_data
            
            # Cache miss or force_fetch - fetch from API
            self.logger.debug(f"📡 Fetching FRESH: {instrument} {timeframe} count={count}")
            
            # Use the enhanced fetch_candles with caching
            fresh_data = fetch_candles(instrument, timeframe, count, 
                                      api_key=self.credentials['oanda_api_key'],
                                      use_cache=True)  # Enable cache
            
            # Store in cache (even if empty, to prevent repeated failed calls)
            if fresh_data is not None and not fresh_data.empty:
                self.candle_cache.set(instrument, timeframe, count, fresh_data)
                if force_fetch:
                    self.logger.debug(f"📦 Global Cache SET (force): {instrument} {timeframe}")
            else:
                self.logger.warning(f"⚠️ No data returned for {instrument} {timeframe}")
                # Still cache empty DataFrame for 10 seconds to prevent repeated failed calls
                self.candle_cache.set(instrument, timeframe, count, pd.DataFrame(), 
                                     ttl_override=10)
            
            return fresh_data
            
        except Exception as e:
            self.logger.error(f"❌ Error in cached_fetch_candles: {str(e)}")
            return pd.DataFrame()
    
    def _generate_signal_id(self, trigger_data):
        """Create a unique signal ID for grouping multiple hammers"""
        import hashlib
        unique_string = f"{trigger_data['type']}_{trigger_data['instrument']}_{trigger_data.get('formation_time')}"
        return hashlib.md5(unique_string.encode()).hexdigest()[:8]
    
    def _generate_trade_id(self, instrument, timeframe):
        """Create unique trade ID with date for 10-year uniqueness"""
        from datetime import datetime
        # Include date to ensure 10-year uniqueness: YYYYMMDD_HHMMSSmmm
        timestamp = datetime.now(NY_TZ).strftime('%Y%m%d_%H%M%S%f')[:-3]
        return f"{instrument}_{timeframe}_{timestamp}"
    
    def wait_for_candle_open(self, timeframe):
        """Wait precisely for next candle open (3 seconds delay) - IMPROVED"""
        try:
            now = datetime.now(NY_TZ)
            
            if timeframe.startswith('M'):
                minutes = int(timeframe[1:])
                current_minute = now.minute
                minutes_past = current_minute % minutes
                seconds_past = now.second + (now.microsecond / 1000000)
                
                # Calculate seconds until next candle CLOSE (which is when next candle opens)
                seconds_to_next_close = (minutes - minutes_past - 1) * 60 + (60 - seconds_past)
                
                if seconds_to_next_close > 0:
                    # Wait for candle to close + 2 seconds for data availability
                    total_wait_time = seconds_to_next_close + 4
                    next_candle_time = now + timedelta(seconds=total_wait_time)
                    self.logger.info(f"⏰ {timeframe}: Candle closes in {seconds_to_next_close:.0f}s")
                    self.logger.info(f"   Waiting {total_wait_time:.0f}s total (close + 4s buffer)")
                    self.logger.info(f"   Next data at: {next_candle_time.strftime('%H:%M:%S')}")
                    time.sleep(total_wait_time)
                    self.logger.info(f"✅ {timeframe} data should now be available")
                    return True
                else:
                    # Candle already closed, check if we need to wait for next candle
                    seconds_since_close = -seconds_to_next_close
                    if seconds_since_close < 4:
                        # Still within 3-second buffer, wait remaining time
                        remaining_buffer = 4 - seconds_since_close
                        self.logger.info(f"⏰ {timeframe}: Candle closed {seconds_since_close:.0f}s ago")
                        self.logger.info(f"   Waiting {remaining_buffer:.0f}s for data buffer")
                        time.sleep(remaining_buffer)
                    self.logger.info(f"✅ {timeframe} data should be available now")
                    return True
            else:
                # For non-minute timeframes, just wait 3 seconds
                self.logger.info(f"⏰ {timeframe}: Waiting 4s for data availability")
                time.sleep(3)
                return True
                
        except Exception as e:
            self.logger.error(f"Error in wait_for_candle_open: {str(e)}")
            # If there's an error, wait a safe amount and continue
            time.sleep(5)
            return True

    def fetch_all_timeframe_data(self, instrument):
        """Fetch data for ALL timeframes needed for features in one go"""
        # Define ALL timeframes needed by BOTH Zebra and Higher TF features
        timeframes = {
            # Zebra features (with required candle count)
            'M1': 200, 'M3': 200, 'M5': 200,'M10': 200, 'M15': 200,
            'H1': 200, 'H4': 200, 'H6': 200, 'D': 200,
            # Higher TF features (with different count requirements)
            'W': 100  # Weekly for higher TF features (not used in Zebra)
        }
        
        data_cache = {}
        
        for tf, required_count in timeframes.items():
            try:
                df = fetch_candles(
                    instrument, 
                    tf, 
                    count=required_count,
                    api_key=self.credentials['oanda_api_key']
                )
                data_cache[tf] = df
                self.logger.info(f"📊 Fetched {tf} data: {len(df)} candles")
            except Exception as e:
                self.logger.error(f"❌ Failed to fetch {tf} data: {str(e)}")
                data_cache[tf] = pd.DataFrame()  # Empty dataframe as fallback
        
        return data_cache

    def _fetch_with_cache(self, instrument, timeframe, count):
        """Fetch candles with caching to reduce API calls"""
        cache_key = f"{instrument}_{timeframe}_{count}"
        current_time = time.time()
        
        # Check if valid cache exists
        if (cache_key in self.data_cache and 
            cache_key in self.cache_expiry and
            current_time < self.cache_expiry[cache_key]):
            self.logger.debug(f"📦 Using cached data for {instrument} {timeframe}")
            return self.data_cache[cache_key]
        
        # Fetch fresh data
        df = fetch_candles(instrument, timeframe, count=count,
                          api_key=self.credentials['oanda_api_key'])
        
        # Store in cache
        self.data_cache[cache_key] = df
        self.cache_expiry[cache_key] = current_time + self.cache_duration
        
        return df

    def start_news_background_fetch(self, interval_hours=6):
        """Start background news fetching thread"""
        if not self.news_calendar:
            self.logger.warning("⚠️ News Calendar not initialized - cannot start background fetch")
            return
        
        def news_fetcher():
            while self.running:
                try:
                    # Fetch news for today
                    date_str = datetime.now(NY_TZ).strftime('%Y-%m-%d')
                    self.logger.info(f"📰 Background news fetch for {date_str}")
                    
                    news_data = self.news_calendar.fetch_news_data(date_str)
                    
                    if 'error' in news_data:
                        self.logger.error(f"❌ Background news fetch failed: {news_data['error']}")
                    else:
                        event_count = len(news_data.get('events', []))
                        self.logger.info(f"📰 Background fetch successful: {event_count} events")
                    
                    # Sleep for interval
                    time.sleep(interval_hours * 3600)
                    
                except Exception as e:
                    self.logger.error(f"❌ Error in background news fetch: {str(e)}")
                    time.sleep(3600)  # Wait 1 hour on error
        
        # Start thread
        self.news_thread = threading.Thread(
            target=news_fetcher,
            name="NewsBackgroundFetch",
            daemon=True
        )
        self.news_thread.start()
        self.logger.info(f"📰 Started background news fetching every {interval_hours} hours")
    
    def is_hammer_candle(self, candle, direction):
        """Simplified hammer detection - only 50% wick rule"""
        try:
            total_range = candle['high'] - candle['low']
            if total_range == 0:
                return False, 0, 0
            
            upper_wick = candle['high'] - max(candle['close'], candle['open'])
            lower_wick = min(candle['close'], candle['open']) - candle['low']
            
            # Calculate ratios
            upper_ratio = upper_wick / total_range if total_range > 0 else 0
            lower_ratio = lower_wick / total_range if total_range > 0 else 0
            
            # Strict 50% wick rule (no body size or pip requirements)
            if direction == 'bearish':
                # Bearish hammer: upper wick > 50%
                if upper_ratio > 0.5:
                    return True, upper_ratio, lower_ratio
            else:  # bullish
                # Bullish hammer: lower wick > 50%
                if lower_ratio > 0.5:
                    return True, upper_ratio, lower_ratio
            
            return False, upper_ratio, lower_ratio
            
        except Exception as e:
            self.logger.error(f"Error in hammer detection: {str(e)}")
            return False, 0, 0

    def _find_swing_lows(self, df, lookback=3, lookforward=3):
        """Find significant swing lows in price data"""
        swing_lows = []
        
        for i in range(lookback, len(df) - lookforward):
            # Current low
            current_low = df.iloc[i]['low']
            
            # Check if current low is lower than previous 'lookback' candles
            is_lowest_backward = True
            for j in range(1, lookback + 1):
                if df.iloc[i - j]['low'] < current_low:
                    is_lowest_backward = False
                    break
            
            # Check if current low is lower than next 'lookforward' candles
            is_lowest_forward = True
            for j in range(1, lookforward + 1):
                if df.iloc[i + j]['low'] < current_low:
                    is_lowest_forward = False
                    break
            
            if is_lowest_backward and is_lowest_forward:
                swing_lows.append(current_low)
        
        return swing_lows
    
    def _find_swing_highs(self, df, lookback=3, lookforward=3):
        """Find significant swing highs in price data"""
        swing_highs = []
        
        for i in range(lookback, len(df) - lookforward):
            # Current high
            current_high = df.iloc[i]['high']
            
            # Check if current high is higher than previous 'lookback' candles
            is_highest_backward = True
            for j in range(1, lookback + 1):
                if df.iloc[i - j]['high'] > current_high:
                    is_highest_backward = False
                    break
            
            # Check if current high is higher than next 'lookforward' candles
            is_highest_forward = True
            for j in range(1, lookforward + 1):
                if df.iloc[i + j]['high'] > current_high:
                    is_highest_forward = False
                    break
            
            if is_highest_backward and is_highest_forward:
                swing_highs.append(current_high)
        
        return swing_highs

    def configure_webhook(self, url=None, token=None):
        """Configure webhook settings"""
        if url:
            self.webhook_url = url
            self.logger.info(f"📡 Webhook URL set to: {url}")
        
        if token:
            self.webhook_token = token
            self.logger.info(f"🔑 Webhook token set (length: {len(token)})")
        
        # Also set as environment variables for backward compatibility
        if url:
            os.environ['WEBHOOK_URL'] = url
        if token:
            os.environ['WEBHOOK_TOKEN'] = token
        
        return True
    
    

    def _get_fib_zones(self, trigger_data):
        
        """Calculate Fibonacci zones with proper validation for FVG+SMT and SD+SMT"""
        try:
            instrument = trigger_data.get('instrument')
            direction = trigger_data.get('direction')
            criteria = trigger_data.get('type')
            signal_data = trigger_data.get('signal_data', {})
                
            # ======== CRT+SMT uses different logic ========
            if criteria == 'CRT+SMT':
                # Use the new range projection method for CRT
                return self._get_crt_zones_with_proper_tp(trigger_data)
                
                # ======== FVG+SMT and SD+SMT logic ========
                # DEBUG: Log what we actually received
            self.logger.info(f"📊 Getting zones for {criteria}, direction: {direction}")
                
                # Get formation time
            if criteria == 'FVG+SMT':
                zone_data = signal_data.get('fvg_idea', {})
                formation_time = zone_data.get('formation_time')
            elif criteria == 'SD+SMT':
                zone_data = signal_data.get('zone', {})
                formation_time = zone_data.get('formation_time')
                
            if not formation_time:
                self.logger.error(f"❌ No formation time for {criteria}")
                return {
                    'zones': [],
                    'sl_price': None,
                    'tp_price': None,
                    'direction': direction,
                    'criteria': criteria,
                    'is_valid': False,
                    'error': 'No formation time'
                    }
                
            # Get SMT data
            smt_data = signal_data.get('smt_data', {})
            smt_cycle = smt_data.get('cycle', 'daily')
                
            # Map SMT cycle to timeframe for data analysis
            tf_map = {
                'daily': 'M15',
                '90min': 'M5', 
                'weekly': 'H1',
                'monthly': 'H4'
            }
            analysis_tf = tf_map.get(smt_cycle, 'M15')
                
            # Fetch enough data from formation time
            self.logger.info(f"📊 Fetching {analysis_tf} data from {formation_time} to now...")
            df = fetch_candles(instrument, analysis_tf, count=200, api_key=self.credentials['oanda_api_key'])
                
            if df.empty:
                self.logger.error(f"❌ No data fetched for {instrument} {analysis_tf}")
                return {
                    'zones': [],
                    'sl_price': None,
                    'tp_price': None,
                    'direction': direction,
                    'criteria': criteria,
                    'is_valid': False,
                    'error': 'No data fetched'
                    }
                
                # Filter data from formation time onward
            df_from_formation = df[df['time'] >= formation_time]
                
            if df_from_formation.empty:
                self.logger.error(f"❌ No data after formation time {formation_time}")
                return {
                    'zones': [],
                    'sl_price': None,
                    'tp_price': None,
                    'direction': direction,
                    'criteria': criteria,
                    'is_valid': False,
                    'error': 'No data after formation'
                }
                
                
                # Get SMT swings and convert to list WITH ASSET FILTERING
            smt_swings_dict = smt_data.get('swings', {})
            swings_list = []
                
            self.logger.info(f"🔍 Filtering swings for {instrument}...")
                
            for key, swing_info in smt_swings_dict.items():
                if isinstance(swing_info, dict) and 'price' in swing_info and 'time' in swing_info:
                    swing_asset = swing_info.get('asset', '')
                        
                    # Check if this swing belongs to our instrument
                    if swing_asset == instrument:
                        swings_list.append({
                            'time': swing_info['time'],
                            'price': swing_info.get('price', 0),
                            'type': swing_info.get('type', 'unknown'),
                            'asset': swing_asset,
                            'swing_type': swing_info.get('swing_type', 'unknown')  # 'prev' or 'curr'
                        })
                        self.logger.info(f"✅ INCLUDING swing for {instrument}: {key} at {swing_info.get('price', 0)} (type: {swing_info.get('type')}, swing_type: {swing_info.get('swing_type')})")
                    else:
                        self.logger.info(f"❌ EXCLUDING swing (wrong asset): {key} at {swing_info.get('price', 0)} (asset: {swing_asset})")
                
            if len(swings_list) < 2:
                self.logger.error(f"❌ Not enough valid SMT swings for {instrument}: {len(swings_list)}")
                # Log what assets we actually found
                found_assets = set()
                for key, swing_info in smt_swings_dict.items():
                    if isinstance(swing_info, dict):
                        found_assets.add(swing_info.get('asset', 'unknown'))
                self.logger.error(f"❌ Found swings for assets: {found_assets}")
                    
                return {
                    'zones': [],
                    'sl_price': None,
                    'tp_price': None,
                    'direction': direction,
                    'criteria': criteria,
                    'is_valid': False,
                    'error': f'Not enough swings for {instrument}: {len(swings_list)}'
                }
                
            # Sort swings by time for chronological order
            swings_list.sort(key=lambda x: x['time'])
                
            # Identify which swing is prev and which is curr based on swing_type
            prev_swing = None
            curr_swing = None
                
            for swing in swings_list:
                if swing.get('swing_type') == 'prev':
                    prev_swing = swing
                elif swing.get('swing_type') == 'curr':
                    curr_swing = swing
                
                # If we couldn't find by swing_type, use chronological order (last one is curr)
            if not curr_swing and swings_list:
                curr_swing = swings_list[-1]
                self.logger.info(f"📊 Using chronological order for curr swing: {curr_swing['time']}")
                
            if not prev_swing and len(swings_list) >= 2:
                prev_swing = swings_list[-2]
                self.logger.info(f"📊 Using chronological order for prev swing: {prev_swing['time']}")
                
            if not curr_swing:
                self.logger.error(f"❌ Could not identify current swing for {instrument}")
                return {
                    'zones': [],
                    'sl_price': None,
                    'tp_price': None,
                    'direction': direction,
                    'criteria': criteria,
                    'is_valid': False,
                    'error': 'Could not identify current swing'
                }
                
                # ============================================
                # Calculate SL and TP based on direction
                # ============================================
                # SL is the CURRENT (most recent) swing
            default_sl = curr_swing['price']
            self.logger.info(f"📊 Using current swing as SL: {default_sl:.5f} (time: {curr_swing['time']}, type: {curr_swing['type']})")
                
            if direction == 'bearish':
                # BEARISH: SL is current swing (should be a high), TP is lowest point between formation and current swing time
                    
                # Validate that current swing is indeed a high for bearish setup
                if curr_swing['type'] != 'high':
                    self.logger.warning(f"⚠️  Bearish setup but current swing is type '{curr_swing['type']}', expected 'high'")
                    
                    # Find TP: lowest point between formation and current swing time
                if curr_swing['time']:
                    mask = (df_from_formation['time'] >= formation_time) & \
                            (df_from_formation['time'] <= curr_swing['time'])
                    df_for_tp = df_from_formation[mask]
                else:
                    df_for_tp = df_from_formation[df_from_formation['time'] >= formation_time]
                    
                if df_for_tp.empty:
                    default_tp = df_from_formation['low'].min()
                else:
                        # Use proper swing detection for TP
                    swing_lows = self._find_swing_lows(df_for_tp, lookback=3, lookforward=3)
                        
                    if not swing_lows:
                        default_tp = df_for_tp['low'].min()
                        self.logger.info(f"📊 Using simple low as TP: {default_tp:.5f}")
                    else:
                        default_tp = min(swing_lows)
                        self.logger.info(f"📊 Found {len(swing_lows)} swing lows, TP: {default_tp:.5f}")
                    
                    # 3. Check if TP has been broken (invalidates setup)
                swing_time_mask = df_from_formation['low'] == default_tp
                if swing_time_mask.any():
                    swing_time = df_from_formation[swing_time_mask].iloc[0]['time']
                    df_after_swing = df_from_formation[df_from_formation['time'] > swing_time]
                    candles_below_tp = df_after_swing[df_after_swing['close'] < default_tp]
                        
                    if not candles_below_tp.empty:
                        self.logger.error(f"❌ BEARISH SETUP INVALID: TP swing low at {default_tp:.5f} was broken!")
                        return {
                            'zones': [],
                            'sl_price': default_sl,
                            'tp_price': default_tp,
                            'direction': direction,
                            'criteria': criteria,
                            'is_valid': False,
                            'error': f'TP broken at {candles_below_tp.iloc[0]["time"]}'
                        }
                    
                self.logger.info(f"✅ Bearish setup VALID: SL={default_sl:.5f} (curr swing), TP={default_tp:.5f}")
                    
            else:  # BULLISH
                    # BULLISH: SL is current swing (should be a low), TP is highest point between formation and current swing time
                    
                    # Validate that current swing is indeed a low for bullish setup
                if curr_swing['type'] != 'low':
                    self.logger.warning(f"⚠️  Bullish setup but current swing is type '{curr_swing['type']}', expected 'low'")
                    
                    # Find TP: highest point between formation and current swing time
                if curr_swing['time']:
                    mask = (df_from_formation['time'] >= formation_time) & \
                            (df_from_formation['time'] <= curr_swing['time'])
                    df_for_tp = df_from_formation[mask]
                else:
                    df_for_tp = df_from_formation[df_from_formation['time'] >= formation_time]
                    
                if df_for_tp.empty:
                    default_tp = df_from_formation['high'].max()
                else:
                        # Use proper swing detection for TP
                    swing_highs = self._find_swing_highs(df_for_tp, lookback=3, lookforward=3)
                        
                    if not swing_highs:
                        default_tp = df_for_tp['high'].max()
                        self.logger.info(f"📊 Using simple high as TP: {default_tp:.5f}")
                    else:
                        default_tp = max(swing_highs)
                        self.logger.info(f"📊 Found {len(swing_highs)} swing highs, TP: {default_tp:.5f}")
                    
                    # 3. Check if TP has been broken (invalidates setup)
                swing_time_mask = df_from_formation['high'] == default_tp
                if swing_time_mask.any():
                    swing_time = df_from_formation[swing_time_mask].iloc[0]['time']
                    df_after_swing = df_from_formation[df_from_formation['time'] > swing_time]
                    candles_above_tp = df_after_swing[df_after_swing['close'] > default_tp]
                        
                    if not candles_above_tp.empty:
                        self.logger.error(f"❌ BULLISH SETUP INVALID: TP swing high at {default_tp:.5f} was broken!")
                        return {
                            'zones': [],
                            'sl_price': default_sl,
                            'tp_price': default_tp,
                            'direction': direction,
                            'criteria': criteria,
                            'is_valid': False,
                            'error': f'TP broken at {candles_above_tp.iloc[0]["time"]}'
                        }
                    
                self.logger.info(f"✅ Bullish setup VALID: SL={default_sl:.5f} (curr swing), TP={default_tp:.5f}")
                
                # ============================================
                # Calculate Fibonacci zones
                # ============================================
            fib_zones = self._calculate_fibonacci_levels(default_sl, default_tp, direction)
                
                
            # In _get_fib_zones, replace the zone calculation and filtering with:
            fib_zones = self._calculate_fibonacci_levels(default_sl, default_tp, direction)
            
            if not fib_zones:
                self.logger.error(f"❌ Failed to calculate 50% zone")
                return {
                    'zones': [],
                    'sl_price': default_sl,
                    'tp_price': default_tp,
                    'direction': direction,
                    'criteria': criteria,
                    'is_valid': False,
                    'error': 'Failed to calculate 50% zone'
                }
            
            # No need to filter - we only have the 50% zone
            valid_zones = fib_zones  # All zones are valid (just the 50% zone)
           
                
            if not valid_zones:
                self.logger.error(f"❌ No valid Fibonacci zones after 50% filter")
                return {
                    'zones': [],
                    'sl_price': default_sl,
                    'tp_price': default_tp,
                    'direction': direction,
                    'criteria': criteria,
                    'is_valid': False,
                    'error': 'No valid zones after 50% filter'
                }
                
                # Return complete setup data
            return {
                'zones': valid_zones,
                'sl_price': default_sl,
                'tp_price': default_tp,
                'direction': direction,
                'criteria': criteria,
                'is_valid': True,
                'error': None
            }
                
        except Exception as e:
            self.logger.error(f"❌ Error calculating Fibonacci zones: {str(e)}", exc_info=True)
            return {
                'zones': [],
                'sl_price': None,
                'tp_price': None,
                'direction': trigger_data.get('direction'),
                'criteria': trigger_data.get('type'),
                'is_valid': False,
                'error': str(e)
            }
    
    def _calculate_fibonacci_levels(self, sl_price, tp_price, direction):
        """Calculate ONLY 50% zone - kill all other Fibonacci levels"""
        try:
            # Calculate 50% line between SL and TP
            if direction == 'bearish':
                fifty_percent_line = sl_price - ((sl_price - tp_price) * 0.5)
            else:  # bullish
                fifty_percent_line = sl_price + ((tp_price - sl_price) * 0.5)
            
            # Create a small buffer zone around the 50% line (0.5% of range)
            price_range = abs(sl_price - tp_price)
            buffer = price_range * 0.005  # 0.5% buffer (adjust as needed)
            
            if direction == 'bearish':
                # For bearish: we sell when price is ABOVE 50% line
                # Create zone that starts at 50% line and goes up a bit
                zone_low = fifty_percent_line
                zone_high = fifty_percent_line + buffer
            else:  # bullish
                # For bullish: we buy when price is BELOW 50% line
                # Create zone that ends at 50% line and goes down a bit
                zone_low = fifty_percent_line - buffer
                zone_high = fifty_percent_line
            
            # Create single 50% zone
            zones = [{
                'ratio': 0.5,
                'low': zone_low,
                'high': zone_high,
                'mid': fifty_percent_line,
                'zone_name': '50%_Zone',
                'is_50_percent_zone': True
            }]
            
            self.logger.info(f"📊 Created 50% zone for {direction.upper()}:")
            self.logger.info(f"   Zone: {zone_low:.5f} to {zone_high:.5f}")
            self.logger.info(f"   SL: {sl_price:.5f}, TP: {tp_price:.5f}")
            
            return zones
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating 50% zone: {str(e)}")
            return []

    def calculate_pips(self, instrument, price1, price2):
        """
        Correct pip calculation for OANDA instruments (Forex, Indices, Metals)
        Based on actual market conventions and contract specifications
        """
        try:
            diff = abs(price1 - price2)
            symbol = instrument.upper().replace("_", "")
            
            # ---- FOREX PAIRS ----
            # Forex pairs: 1 pip = 0.0001 for most pairs, 0.01 for JPY pairs
            if len(symbol) == 6 and symbol.isalpha():
                if symbol.endswith("JPY") or symbol[3:] == "JPY":
                    # JPY pairs: 1 pip = 0.01
                    return round(diff / 0.01, 1)
                else:
                    # Non-JPY pairs: 1 pip = 0.0001
                    return round(diff / 0.0001, 1)
            
            # ---- METALS ----
            # Gold (XAU): Typically quoted to 2 decimal places, 1 pip = 0.01
            if "XAU" in symbol:
                # XAU/USD, XAU/EUR, XAU/JPY: 1 pip = 0.01
                return round(diff / 0.01, 1)
            
            # Silver (XAG): Typically quoted to 3 decimal places, 1 pip = 0.001
            if "XAG" in symbol:
                return round(diff / 0.001, 1)
            
            # ---- INDICES ----
            indices = ['DE30', 'EU50', 'NAS100', 'SPX500', 'US500', 'USTEC', 'US30', 
                       'UK100', 'AUS200', 'FRA40', 'ESP35', 'JPN225']
            
            # Check if it's an index
            for idx in indices:
                if idx in symbol:
                    # For indices: 1 point = 1 pip (minimum price movement)
                    # Most indices move in 1.0 increments (points)
                    return round(diff / 1.0, 1)
            
            # ---- ENERGY/COMMODITIES ----
            # USOIL, UKOIL: Typically quoted to 2 decimal places, 1 pip = 0.01
            if "OIL" in symbol:
                return round(diff / 0.01, 1)
            
            # NATGAS: Typically quoted to 3 decimal places
            if "GAS" in symbol:
                return round(diff / 0.001, 1)
            
            # ---- CRYPTO ----
            # Crypto: Typically quoted to 2 decimal places
            if any(crypto in symbol for crypto in ['BTC', 'ETH', 'XRP', 'LTC', 'BCH']):
                return round(diff / 1.00, 1)  # Crypto often moves in 1.00 increments
            
            # Default fallback for unknown instruments
            self.logger.warning(f"⚠️ Unknown instrument type for pip calculation: {instrument}, using default 0.0001")
            return round(diff / 0.0001, 1)
            
        except Exception as e:
            self.logger.error(f"❌ Pip calculation error for {instrument}: {e}")
            return None

    def _clean_string_for_csv(self, text):
        """Clean string to avoid encoding issues in CSV"""
        if not isinstance(text, str):
            return str(text) if text is not None else ''
        
        # Replace problematic characters
        text = text.replace('–', '-')  # en dash
        text = text.replace('—', '-')  # em dash
        text = text.replace('−', '-')  # minus sign
        text = text.replace('―', '-')  # horizontal bar
        
        # Remove any non-ASCII characters
        text = text.encode('ascii', 'ignore').decode('ascii')
        
        return text.strip()

    def get_pip_value_per_micro_lot(self, instrument, current_price=None):
        """
        Get accurate pip value in USD for 1 micro lot (1000 units) of instrument
        Based on actual contract specifications for OANDA
        """
        try:
            symbol = instrument.upper().replace("_", "")
            
            # ---- FOREX ----
            # Forex standard: 1 standard lot = 100,000 units, 1 micro lot = 1,000 units
            # For Forex, pip value = (0.0001 / exchange rate) * units
            
            # Forex with USD as quote currency (EUR/USD, GBP/USD, AUD/USD, NZD/USD)
            # 1 pip = $0.10 per micro lot
            usd_quote_pairs = ['EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD']
            if symbol in usd_quote_pairs:
                return 0.10  # $0.10 per pip per micro lot
            
            # Forex with USD as base currency (USD/JPY, USD/CAD, USD/CHF)
            if symbol.startswith('USD'):
                # For USD/JPY: 1 pip = 0.01, pip value = 0.01 * 1000 / current_price
                if symbol == 'USDJPY':
                    if current_price:
                        return round(0.01 * 1000 / current_price, 2)
                    return 0.09  # Approx at 110.00 JPY/USD
                
                # For USD/CAD: 1 pip = 0.0001, pip value = 0.0001 * 1000 / USDCAD rate
                if symbol == 'USDCAD':
                    if current_price:
                        return round(0.0001 * 1000 / current_price, 2)
                    return 0.08  # Approx at 1.25 CAD/USD
                
                # For USD/CHF
                if symbol == 'USDCHF':
                    if current_price:
                        return round(0.0001 * 1000 / current_price, 2)
                    return 0.10  # Approx at 1.00 CHF/USD
            
            # Cross pairs (EUR/JPY, GBP/JPY, etc.)
            # Need to convert through USD
            # Formula: pip value = (0.01 * 1000) / USDJPY_rate (for JPY pairs)
            jpy_crosses = ['EURJPY', 'GBPJPY', 'AUDJPY', 'NZDJPY', 'CADJPY', 'CHFJPY']
            if symbol in jpy_crosses:
                # For JPY crosses, we need USD/JPY rate
                # This is simplified - in production you'd fetch the actual rate
                return 0.09  # Approximate
            
            # EUR/GBP, EUR/AUD, etc.
            if len(symbol) == 6 and symbol.isalpha() and symbol not in jpy_crosses:
                # For other crosses, approximate as $0.10
                return 0.10
            
            # ---- METALS ----
            # Gold (XAU/USD): 1 standard lot = 100 ounces, 1 micro lot = 1 ounce
            # 1 pip = 0.01, value = 0.01 * 1 = $0.01 per pip per micro lot
            if symbol == 'XAUUSD':
                return 0.01
            
            # Silver (XAG/USD): 1 standard lot = 5000 ounces, 1 micro lot = 50 ounces
            # 1 pip = 0.001, value = 0.001 * 50 = $0.05 per pip per micro lot
            if symbol == 'XAGUSD':
                return 0.05
            
            # Gold in other currencies (XAU/EUR, XAU/JPY)
            if 'XAU' in symbol and symbol != 'XAUUSD':
                # Need conversion to USD
                # Simplified approximation
                return 0.01
            
            # ---- INDICES ----
            # Indices: 1 point = $1.00 per standard contract
            # For OANDA, indices CFD: 1 unit = 1 contract
            indices_usd = ['NAS100', 'SPX500', 'US500', 'USTEC', 'US30']
            indices_eur = ['DE30', 'EU50', 'FRA40']
            indices_gbp = ['UK100']
            indices_aud = ['AUS200']
            
            if any(idx in symbol for idx in indices_usd):
                # US indices: 1 point = $1.00 per standard lot
                # 1 micro lot = 0.01 standard lot = $0.01 per point
                return 0.01
            
            elif any(idx in symbol for idx in indices_eur):
                # EUR indices: 1 point = €1.00 per standard lot
                # Need EUR/USD conversion
                eur_usd_rate = 1.10  # Approximate
                return round(0.01 * eur_usd_rate, 2)  # ~$0.011
            
            elif any(idx in symbol for idx in indices_gbp):
                # UK indices: 1 point = £1.00 per standard lot
                gbp_usd_rate = 1.30  # Approximate
                return round(0.01 * gbp_usd_rate, 2)  # ~$0.013
            
            elif any(idx in symbol for idx in indices_aud):
                # AUS indices: 1 point = AUD 1.00 per standard lot
                aud_usd_rate = 0.70  # Approximate
                return round(0.01 * aud_usd_rate, 2)  # ~$0.007
            
            # ---- COMMODITIES ----
            if 'OIL' in symbol:
                # Oil: 1 pip = 0.01, 1 standard lot = 1000 barrels
                # 1 micro lot = 10 barrels, value = 0.01 * 10 = $0.10 per pip
                return 0.10
            
            if 'GAS' in symbol:
                # Natural gas: 1 pip = 0.001, 1 standard lot = 10,000 MMBtu
                # 1 micro lot = 100 MMBtu, value = 0.001 * 100 = $0.10 per pip
                return 0.10
            
            # Default for unknown instruments
            self.logger.warning(f"⚠️ Unknown instrument for pip value: {instrument}, using default $0.10")
            return 0.10
            
        except Exception as e:
            self.logger.error(f"❌ Error getting pip value for {instrument}: {str(e)}")
            return 0.10  # Safe default
    
    def calculate_position_sizes(self, instrument, entry_price, sl_price, risk_amount=50.0):
        """
        Calculate position sizes for given risk amount (in USD)
        Based on accurate pip value calculations
        
        Args:
            instrument: Trading symbol (e.g., 'EUR_USD')
            entry_price: Entry price
            sl_price: Stop loss price
            risk_amount: Risk amount in USD (default: $50)
        
        Returns:
            position_size: Position size in units (NOT lots)
            risk_per_pip: Risk per pip in USD
        """
        try:
            # Safety checks
            if entry_price <= 0 or sl_price <= 0:
                self.logger.warning(f"Invalid prices for {instrument}: entry={entry_price}, sl={sl_price}")
                return 0, 0
            
            if risk_amount <= 0:
                risk_amount = 50.0  # Default $50 risk
            
            # Calculate pip distance to stop loss
            pip_distance = self.calculate_pips(instrument, entry_price, sl_price)
            if pip_distance is None or pip_distance <= 0:
                self.logger.warning(f"Invalid pip distance for {instrument}: {pip_distance}")
                return 0, 0
            
            # Get pip value per micro lot
            pip_value_per_micro = self.get_pip_value_per_micro_lot(instrument, entry_price)
            
            # Calculate risk per pip for position sizing
            # Formula: Risk per pip = Risk amount / Pip distance
            risk_per_pip = risk_amount / pip_distance
            
            # Calculate position size in units
            # Formula: Position size (units) = Risk per pip / Pip value per unit
            # For Forex: Standard lot = 100,000 units, Micro lot = 1,000 units
            
            # First, get pip value per unit (1 unit)
            pip_value_per_unit = pip_value_per_micro / 1000.0  # Convert from micro lot (1000 units) to per unit
            
            # Calculate position size in units
            if pip_value_per_unit > 0:
                position_units = risk_per_pip / pip_value_per_unit
            else:
                self.logger.error(f"Zero pip value per unit for {instrument}")
                return 0, 0
            
            # Convert to lots for logging (1 standard lot = 100,000 units)
            standard_lots = position_units / 100000.0
            micro_lots = position_units / 1000.0
            
            # Apply reasonable limits
            # Maximum position size: 100 standard lots (10,000,000 units)
            max_units = 10000000
            if position_units > max_units:
                self.logger.warning(f"Position size {position_units:.0f} units exceeds max, capping at {max_units}")
                position_units = max_units
            
            # Minimum position size: 1000 units (1 micro lot)
            min_units = 1000
            if position_units < min_units:
                self.logger.warning(f"Position size {position_units:.0f} units below min, using {min_units}")
                position_units = min_units
            
            # Round to nearest 1000 units (micro lot increments)
            position_units = round(position_units / 1000) * 1000
            
            # Log the calculation
            self.logger.info(f"📊 Position sizing for {instrument}:")
            self.logger.info(f"   Entry: {entry_price:.5f}, SL: {sl_price:.5f}")
            self.logger.info(f"   Pip distance: {pip_distance:.1f} pips")
            self.logger.info(f"   Pip value per micro lot: ${pip_value_per_micro:.3f}")
            self.logger.info(f"   Risk amount: ${risk_amount:.2f}")
            self.logger.info(f"   Risk per pip: ${risk_per_pip:.3f}")
            self.logger.info(f"   Position size: {position_units:.0f} units ({standard_lots:.3f} standard lots)")
            
            return position_units, risk_per_pip
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating position size for {instrument}: {str(e)}", exc_info=True)
            return 0, 0
    
    def _get_crt_zones(self, trigger_data):
        """Calculate zones for CRT setups (different logic)"""
        try:
            instrument = trigger_data.get('instrument')
            direction = trigger_data.get('direction')
            trigger_timeframe = trigger_data.get('trigger_timeframe')
            criteria = trigger_data.get('type')
            signal_data = trigger_data.get('signal_data', {})
            
            if criteria != 'CRT+SMT':
                return []
            
            # For CRT, we need to check the previous candle's high/low
            crt_signal = signal_data.get('crt_signal', {})
            
            # Fetch the previous candle of the CRT timeframe
            df = fetch_candles(instrument, trigger_timeframe, count=3, api_key=self.credentials['oanda_api_key'])
            
            if df.empty or len(df) < 2:
                self.logger.error(f"❌ No data for CRT {trigger_timeframe}")
                return []
            
            previous_candle = df.iloc[-2]  # The completed candle
            
            if direction == 'bearish':
                # For bearish CRT: invalidate if price goes above previous candle's high
                invalidation_level = previous_candle['high']
                current_price = df.iloc[-1]['close']
                
                # We'll scan within a price range below the invalidation level
                # Use a fixed range for scanning (e.g., 50% retracement of previous candle)
                candle_range = previous_candle['high'] - previous_candle['low']
                scan_low = current_price - (candle_range * 2)  # Extend below for scanning
                
                # Create one big zone for scanning
                zones = [{
                    'ratio': 0.5,
                    'high': invalidation_level,
                    'low': scan_low,
                    'mid': (invalidation_level + scan_low) / 2,
                    'is_crt': True,
                    'invalidation_level': invalidation_level
                }]
                
                self.logger.info(f"📊 CRT Bearish setup:")
                self.logger.info(f"   Invalidation (SL): {invalidation_level:.5f}")
                self.logger.info(f"   Scan range: {scan_low:.5f} - {invalidation_level:.5f}")
                
            else:  # bullish
                # For bullish CRT: invalidate if price goes below previous candle's low
                invalidation_level = previous_candle['low']
                current_price = df.iloc[-1]['close']
                
                candle_range = previous_candle['high'] - previous_candle['low']
                scan_high = current_price + (candle_range * 1)  # Extend above for scanning
                
                zones = [{
                    'ratio': 0.5,
                    'low': invalidation_level,
                    'high': scan_high,
                    'mid': (invalidation_level + scan_high) / 2,
                    'is_crt': True,
                    'invalidation_level': invalidation_level
                }]
                
                self.logger.info(f"📊 CRT Bullish setup:")
                self.logger.info(f"   Invalidation (SL): {invalidation_level:.5f}")
                self.logger.info(f"   Scan range: {invalidation_level:.5f} - {scan_high:.5f}")
            
            return zones
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating CRT zones: {str(e)}")
            return []

    def _get_crt_zones_with_proper_tp(self, trigger_data):
        """Calculate zones for CRT setups using previous candle range projection"""
        try:
            instrument = trigger_data.get('instrument')
            direction = trigger_data.get('direction')
            trigger_timeframe = trigger_data.get('trigger_timeframe')
            criteria = trigger_data.get('type')
            
            self.logger.info(f"🔷 CRT Setup: {instrument} {direction} on {trigger_timeframe}")
            
            # For CRT, we use the previous candle of the CRT timeframe
            # Use cached version
            df = self.cached_fetch_candles(instrument, trigger_timeframe, count=5, force_fetch=True)
            
            if df.empty or len(df) < 2:
                self.logger.error(f"❌ No data for CRT {trigger_timeframe}")
                return {
                    'zones': [],
                    'sl_price': None,
                    'tp_price': None,
                    'direction': direction,
                    'criteria': criteria,
                    'is_valid': False,
                    'error': f'No data for {trigger_timeframe}'
                }
            
            # Get the CRT candle (previous completed candle - index -2)
            crt_candle = df.iloc[-2]
            
            # Get current/latest candle for validation (index -1)
            latest_candle = df.iloc[-1]
            
            self.logger.info(f"📊 CRT Candle (index -2):")
            self.logger.info(f"   Time: {crt_candle['time']}")
            self.logger.info(f"   Open: {crt_candle['open']:.5f}, High: {crt_candle['high']:.5f}")
            self.logger.info(f"   Low: {crt_candle['low']:.5f}, Close: {crt_candle['close']:.5f}")
            
            # 1. Set SL based on CRT candle
            if direction == 'bearish':
                # Bearish CRT: SL is above CRT candle high
                sl_price = crt_candle['high']
                
                # 2. Check if price has already traded above SL (using ONLY current candle)
                # We only check the current/latest candle, NOT past candles
                if latest_candle['high'] > sl_price:
                    self.logger.warning(f"⚠️  Bearish CRT Invalidated: Current candle high exceeded SL")
                    self.logger.info(f"   Current candle high: {latest_candle['high']:.5f}, SL: {sl_price:.5f}")
                    self.logger.info(f"   Time of current candle: {latest_candle['time']}")
                    return {
                        'zones': [],
                        'sl_price': None,
                        'tp_price': None,
                        'direction': direction,
                        'criteria': criteria,
                        'is_valid': False,
                        'error': 'SL breached by current candle'
                    }
                
                # 3. Calculate TP: Subtract range from CRT candle low
                candle_range = crt_candle['high'] - crt_candle['low']
                tp_price = crt_candle['low'] - candle_range
                
                self.logger.info(f"📊 CRT Bearish Setup:")
                self.logger.info(f"   SL (CRT candle high): {sl_price:.5f}")
                self.logger.info(f"   CRT Candle Low: {crt_candle['low']:.5f}")
                self.logger.info(f"   Candle Range: {candle_range:.5f}")
                self.logger.info(f"   TP (CRT Low - Range): {tp_price:.5f}")
                
            else:  # bullish
                # Bullish CRT: SL is below CRT candle low
                sl_price = crt_candle['low']
                
                # 2. Check if price has already traded below SL (using ONLY current candle)
                # We only check the current/latest candle, NOT past candles
                if latest_candle['low'] < sl_price:
                    self.logger.warning(f"⚠️  Bullish CRT Invalidated: Current candle low below SL")
                    self.logger.info(f"   Current candle low: {latest_candle['low']:.5f}, SL: {sl_price:.5f}")
                    self.logger.info(f"   Time of current candle: {latest_candle['time']}")
                    return {
                        'zones': [],
                        'sl_price': None,
                        'tp_price': None,
                        'direction': direction,
                        'criteria': criteria,
                        'is_valid': False,
                        'error': 'SL breached by current candle'
                    }
                
                # 3. Calculate TP: Add range to CRT candle high
                candle_range = crt_candle['high'] - crt_candle['low']
                tp_price = crt_candle['high'] + candle_range
                
                self.logger.info(f"📊 CRT Bullish Setup:")
                self.logger.info(f"   SL (CRT candle low): {sl_price:.5f}")
                self.logger.info(f"   CRT Candle High: {crt_candle['high']:.5f}")
                self.logger.info(f"   Candle Range: {candle_range:.5f}")
                self.logger.info(f"   TP (CRT High + Range): {tp_price:.5f}")
            
            # ✅ REMOVED: The 5-candle history check - we only check the current candle
            
            # Calculate Fibonacci zones from SL to TP
            fib_zones = self._calculate_fibonacci_levels(sl_price, tp_price, direction)
            
            # For CRT, we accept ALL zones (no 50% filter)
            self.logger.info(f"📊 CRT: Using ALL {len(fib_zones)} Fibonacci zones")
            
            return {
                'zones': fib_zones,
                'sl_price': sl_price,
                'tp_price': tp_price,
                'direction': direction,
                'criteria': criteria,
                'is_valid': True,
                'error': None
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error in CRT zone calculation: {str(e)}", exc_info=True)
            return {
                'zones': [],
                'sl_price': None,
                'tp_price': None,
                'direction': trigger_data.get('direction'),
                'criteria': trigger_data.get('type'),
                'is_valid': False,
                'error': str(e)
            }
    
    def get_aligned_timeframes(self, instrument, criteria, trigger_tf):
        """Get aligned timeframes for scanning"""
        config = self.timeframe_alignment.get(instrument, self.timeframe_alignment['default'])
        trigger_map = config.get(criteria, {})
        return trigger_map.get(trigger_tf, ['M5', 'M15'])

    def _reorder_csv_headers(self, existing_headers, target_headers):
        """Re-order CSV headers to match target order"""
        try:
            # Read all data
            with open(self.csv_file_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                all_rows = list(reader)
            
            # Write with new header order
            with open(self.csv_file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=target_headers)
                writer.writeheader()
                
                for row in all_rows:
                    # Reorder row to match target headers
                    ordered_row = {header: row.get(header, '') for header in target_headers}
                    writer.writerow(ordered_row)
            
            self.logger.info(f"📁 Successfully reordered CSV headers")
            
        except Exception as e:
            self.logger.error(f"❌ Error reordering CSV headers: {str(e)}")
    
    def _add_missing_headers(self, existing_headers, missing_headers, target_headers):
        """Add missing headers to CSV file"""
        try:
            # Read all data
            with open(self.csv_file_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                all_rows = list(reader)
            
            # Create new headers (existing + missing in target order)
            new_headers = []
            for header in target_headers:
                if header in existing_headers or header in missing_headers:
                    new_headers.append(header)
            
            # Write with new headers
            with open(self.csv_file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=new_headers)
                writer.writeheader()
                
                for row in all_rows:
                    # Add missing columns with empty values
                    for header in missing_headers:
                        row[header] = ''
                    writer.writerow(row)
            
            self.logger.info(f"📁 Successfully added {len(missing_headers)} missing headers")
            
        except Exception as e:
            self.logger.error(f"❌ Error adding missing headers: {str(e)}")

    def get_true_open_for_cycle(self, instrument, cycle_type, current_time=None):
        """
        Get the True Open price for a specific cycle.
        True Open = price at the open of Q2 for the current cycle
        """
        if current_time is None:
            current_time = datetime.now(NY_TZ)
        
        # Get current quarter
        quarters = self.quarter_manager.get_current_quarters(current_time)
        current_quarter = quarters.get(cycle_type)
        
        # Get the start time of Q2 for this cycle
        q2_start_time = self._get_q2_start_time(cycle_type, current_time)
        
        if not q2_start_time:
            self.logger.warning(f"⚠️ Could not get Q2 start time for {cycle_type} at {current_time}")
            return None, None
        
        # Cache key
        cache_key = f"{instrument}_{cycle_type}_{q2_start_time.strftime('%Y%m%d')}"
        
        with self.true_open_lock:
            if cache_key in self.true_open_cache:
                return self.true_open_cache[cache_key], current_quarter
        
        # Fetch the candle at Q2 start time
        # Determine appropriate timeframe for fetching based on cycle
        cycle_to_tf = {
            'monthly': 'H4',      # H4 candles for monthly cycles
            'weekly': 'H1',       # H1 candles for weekly cycles  
            'daily': 'M15',       # 15-hour candles for daily cycles
            '90min': 'M5'       # 5-minute candles for 90min cycles
        }
        
        tf = cycle_to_tf.get(cycle_type, 'H1')
        
        # Fetch candles around Q2 start time
        try:
            df = fetch_candles(
                instrument,
                tf,
                count=50,
                api_key=self.credentials['oanda_api_key']
            )
            
            if df.empty:
                self.logger.warning(f"⚠️ No data for {instrument} {tf} to find True Open")
                return None, current_quarter
            
            # Find the candle closest to Q2 start time
            # We'll look for the candle that started at or just before Q2 start
            df['time_diff'] = abs((df['time'] - q2_start_time).dt.total_seconds())
            closest_candle = df.loc[df['time_diff'].idxmin()]
            
            true_open = closest_candle['open']
            
            # Cache the result
            with self.true_open_lock:
                self.true_open_cache[cache_key] = true_open
            
            self.logger.info(f"✅ {instrument} {cycle_type} True Open: {true_open:.5f} (Q2 start: {q2_start_time})")
            
            return true_open, current_quarter
            
        except Exception as e:
            self.logger.error(f"❌ Error getting True Open for {instrument} {cycle_type}: {str(e)}")
            return None, current_quarter
    
    def _get_q2_start_time(self, cycle_type, current_time):
        """
        Calculate the start time of Q2 for a given cycle.
        """
        if cycle_type == 'monthly':
            # Q2 starts on the 8th day of the month at 00:00 NY time
            # Monthly quarters: week 1 (days 1-7), week 2 (days 8-14), etc.
            # So Q2 starts on day 8
            if current_time.day >= 8:
                # Q2 started earlier this month
                q2_start = current_time.replace(day=8, hour=0, minute=0, second=0, microsecond=0)
            else:
                # Q2 hasn't started yet this month, so use previous month
                prev_month = current_time - timedelta(days=current_time.day)
                q2_start = prev_month.replace(day=8, hour=0, minute=0, second=0, microsecond=0)
            
        elif cycle_type == 'weekly':
            # Q2 is Tuesday (Monday=Q1, Tuesday=Q2, etc.)
            # Start of Tuesday 00:00 NY time
            days_since_monday = current_time.weekday()  # Monday=0
            if days_since_monday >= 1:  # Tuesday or later
                # This week's Tuesday
                days_to_tuesday = 1 - days_since_monday
                q2_start = current_time + timedelta(days=days_to_tuesday)
            else:
                # It's Monday, Q2 hasn't started yet, so use last week's Tuesday
                days_to_last_tuesday = -6  # Go back 6 days to last Tuesday
                q2_start = current_time + timedelta(days=days_to_last_tuesday)
            
            q2_start = q2_start.replace(hour=0, minute=0, second=0, microsecond=0)
            
        elif cycle_type == 'daily':
            # Daily Q2 is 00:00-06:00, so Q2 starts at 00:00
            q2_start = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
            
        elif cycle_type == '90min':
            # Get current daily quarter first
            daily_quarter = self.quarter_manager._get_daily_quarter_fixed(current_time)
            daily_quarter_start = self.quarter_manager._get_daily_quarter_start_time(current_time, daily_quarter)
            
            # Within the daily quarter, Q2 starts 90 minutes after the start
            q2_start = daily_quarter_start + timedelta(minutes=90)
            
        else:
            return None
        
        # Ensure timezone
        if q2_start.tzinfo is None:
            q2_start = NY_TZ.localize(q2_start)
        
        return q2_start

    def get_zebra_timeframes(self, instrument, criteria, trigger_timeframe):
        """
        Get timeframes for Zebra scanning.
        Always includes M1, M3, M5 for all signals.
        Removes M10 and M15.
        """
        # First get the original timeframes
        original_timeframes = self.get_aligned_timeframes(instrument, criteria, trigger_timeframe)
        
        # Define mandatory Zebra timeframes
        mandatory_tfs = ['M1', 'M3', 'M5']
        
        # Start with mandatory timeframes
        zebra_timeframes = mandatory_tfs.copy()
        
        # Add original timeframes that are not already included and not M10/M15
        for tf in original_timeframes:
            if tf not in zebra_timeframes and tf not in ['M10', 'M15']:
                zebra_timeframes.append(tf)
        
        # Sort by timeframe priority (smaller first for Zebra)
        timeframe_order = {
            'M1': 1, 'M3': 2, 'M5': 3, 'M15': 4, 'M30': 5,
            'H1': 6, 'H4': 7, 'D': 8, 'W': 9
        }
        
        zebra_timeframes.sort(key=lambda x: timeframe_order.get(x, 99))
        
        self.logger.info(f"🦓 Zebra timeframes for {instrument} ({criteria}): {zebra_timeframes}")
        return zebra_timeframes

    def run_zebra_scan_with_signal(self, tf, shared_state):
        """Run Zebra scanner for signal-triggered setups with Fibonacci validation - USING WORKING FRESHNESS CHECK"""
        try:
            import numpy as np
            import time
            import pytz
            from datetime import datetime, timedelta
            
            instrument = shared_state['instrument']
            direction = shared_state['direction']
            criteria = shared_state['criteria']
            fib_zones = shared_state['fib_zones']
            sl_price = shared_state['sl_price']
            tp_price = shared_state['tp_price']
            signal_data = shared_state['signal_data']
            signal_id = shared_state['signal_id']
            trigger_data = shared_state['trigger_data']
            
            self.logger.info(f"🦓 Signal-Triggered Zebra Thread Started: {instrument} {tf}")
            
            scanned_candles = set()
            
            while datetime.now(NY_TZ) < shared_state['scan_end']:
                try:
                    # 1. Wait for candle open/tick (using working freshness check from run_zebra_scan)
                    if not self.wait_for_candle_open(tf):
                        time.sleep(1)
                        continue
                    
                    # 2. Fetch 150 candles (same as working version)
                    df = self.cached_fetch_candles(instrument, tf, count=150, force_fetch=True)
                    if df.empty or len(df) < 100:
                        time.sleep(2)
                        continue
                    
                    # 3. GET THE LATEST CANDLE'S START TIME
                    latest_candle = df.iloc[-1]
                    latest_candle_start = latest_candle['time']
                    
                    # 4. CALCULATE: What *should* be the current candle's start time?
                    current_utc_time = datetime.now(pytz.utc)
                    expected_candle_start = self._calc_expected_candle_start(current_utc_time, tf)
                    
                    # 5. THE CRITICAL FRESHNESS CHECK (using same working logic)
                    # 5a. Make expected_candle_start NY-timezone-aware
                    if expected_candle_start.tzinfo is None:
                        # If it has no timezone, assume it's meant to be NY time
                        expected_candle_start_ny = NY_TZ.localize(expected_candle_start)
                    else:
                        # If it has a timezone (like UTC), convert it to NY time
                        expected_candle_start_ny = expected_candle_start.astimezone(NY_TZ)
                    
                    # 5b. Ensure API timestamp is NY-timezone-aware
                    latest_candle_start_ny = latest_candle['time']
                    if latest_candle_start_ny.tzinfo is None:
                        # If pandas didn't parse timezone, add NY timezone
                        latest_candle_start_ny = NY_TZ.localize(latest_candle_start_ny)
                    
                    # 5c. THE CRITICAL FRESHNESS CHECK (comparing NY to NY)
                    if latest_candle_start_ny != expected_candle_start_ny:
                        # DATA IS STALE: API hasn't sent the new candle yet
                        self.logger.warning(
                            f"⏳ {instrument} {tf}: Data STALE for Zebra. "
                            f"Latest NY candle: {latest_candle_start_ny}, "
                            f"Expected NY candle: {expected_candle_start_ny}. "
                            f"Skipping cycle."
                        )
                        
                        # Efficient sleep until next candle open
                        seconds_to_next = self._calc_seconds_to_next_candle(current_utc_time, tf) + 2
                        self.logger.info(f"💤 Zebra Sleeping {seconds_to_next:.0f}s until next candle.")
                        time.sleep(seconds_to_next)
                        scanned_candles.add(expected_candle_start_ny)
                        continue
                    
                    # 6. ✅ DATA IS FRESH: Proceed with existing logic
                    current_time = latest_candle_start
                    
                    # Prevent multiple entries on the same live candle
                    if current_time in scanned_candles:
                        time.sleep(1)
                        continue
                    
                    scanned_candles.add(current_time)
                    
                    # 7. Get the closed candle for analysis (index -2)
                    if len(df) >= 2:
                        closed_candle = df.iloc[-2]
                    else:
                        closed_candle = df.iloc[-1]
                    
                    # 8. CHECK SL/TP HIT
                    candle_high = closed_candle['high']
                    candle_low = closed_candle['low']
                    
                    if direction == 'bearish':
                        if candle_high > sl_price:
                            self.logger.info(f"❌ {instrument} {tf}: SL HIT in Zebra scanner! Candle high {candle_high:.5f} >= SL {sl_price:.5f}")
                            break
                        if candle_low < tp_price:
                            self.logger.info(f"✅ {instrument} {tf}: TP HIT in Zebra scanner! Candle low {candle_low:.5f} <= TP {tp_price:.5f}")
                            break
                    else:  # bullish
                        if candle_low < sl_price:
                            self.logger.info(f"❌ {instrument} {tf}: SL HIT in Zebra scanner! Candle low {candle_low:.5f} <= SL {sl_price:.5f}")
                            break
                        if candle_high > tp_price:
                            self.logger.info(f"✅ {instrument} {tf}: TP HIT in Zebra scanner! Candle high {candle_high:.5f} >= TP {tp_price:.5f}")
                            break
                    
                    # 9. CHECK 50% ZONE VALIDATION
                    candle_price = closed_candle['close']
                    in_valid_zone = False
                    
                    # Calculate 50% line
                    if direction == 'bearish':
                        fifty_percent_line = sl_price - ((sl_price - tp_price) * 0.5)
                    else:  # bullish
                        fifty_percent_line = sl_price + ((tp_price - sl_price) * 0.5)
                    
                    # Check if price is on correct side of 50% line
                    if direction == 'bearish':
                        if candle_price >= fifty_percent_line:
                            in_valid_zone = True
                            self.logger.info(f"✅ {instrument} {tf}: Price {candle_price:.5f} ABOVE 50% line {fifty_percent_line:.5f}")
                        else:
                            self.logger.info(f"❌ {instrument} {tf}: Price {candle_price:.5f} BELOW 50% line {fifty_percent_line:.5f}")
                            time.sleep(1)
                            continue
                    else:  # bullish
                        if candle_price <= fifty_percent_line:
                            in_valid_zone = True
                            self.logger.info(f"✅ {instrument} {tf}: Price {candle_price:.5f} BELOW 50% line {fifty_percent_line:.5f}")
                        else:
                            self.logger.info(f"❌ {instrument} {tf}: Price {candle_price:.5f} ABOVE 50% line {fifty_percent_line:.5f}")
                            time.sleep(1)
                            continue
                    
                    if not in_valid_zone:
                        time.sleep(1)
                        continue
                    
                    # 10. Calculate HalfTrend logic
                    candles_for_half_trend = df.iloc[-100:]
                    arrup, arrdwn = self._calculate_half_trend(candles_for_half_trend)
                    
                    if len(arrup) == 0 or len(arrdwn) == 0:
                        time.sleep(1)
                        continue
    
                    # 11. Check if arrow is present on index [-1]
                    detected_dir = None
                    arrow_found = False
                    
                    if direction == 'bearish' and not np.isnan(arrdwn[-1]):
                        detected_dir = 'bearish'
                        arrow_found = True
                    elif direction == 'bullish' and not np.isnan(arrup[-1]):
                        detected_dir = 'bullish'
                        arrow_found = True
                    
                    # 12. If arrow is found in correct direction, process it
                    if arrow_found:
                        self.logger.info(f"🎯 ZEBRA {detected_dir.upper()} ARROW found on {tf} {instrument}")
                        
                        entry_price = closed_candle['open']
                        sl_price_zebra = self.find_3_candle_pivot(df, detected_dir)
                        
                        if sl_price_zebra:
                            # HUGE SL FILTER
                            recent_ranges = (df['high'].iloc[-6:-1] - df['low'].iloc[-6:-1]).mean()
                            sl_dist = abs(entry_price - sl_price_zebra)
                            
                            if sl_dist > (recent_ranges * 2.5):
                                self.logger.warning(f"⏭️ Zebra Skip: SL is too wide ({round(sl_dist, 5)} vs Avg {round(recent_ranges, 5)})")
                                time.sleep(1)
                                continue
    
                            # 13. PROCESS AND JOURNAL
                            success = self._process_and_record_hammer(
                                instrument=instrument,
                                tf=tf,
                                candle=closed_candle,
                                direction=detected_dir,
                                criteria='zebra',
                                signal_data=signal_data,
                                signal_id=f"ZEB_{signal_id}_{tf}_{int(time.time())}",
                                trigger_data=trigger_data,
                                zebra_entry=entry_price,
                                zebra_sl=sl_price_zebra,
                                trigger_type='zebra'
                            )
                            
                            if success:
                                with shared_state['zebra_lock']:
                                    shared_state['zebra_count'] += 1
                                self.logger.info(f"✅ Zebra {detected_dir} logged to CSV and Journaled.")
                    
                    time.sleep(1)
                    
                except Exception as e:
                    self.logger.error(f"❌ Error in Signal-Triggered Zebra {tf} loop for {instrument}: {str(e)}", exc_info=True)
                    time.sleep(5)
            
            self.logger.info(f"⏰ {instrument} {tf} Signal-Triggered Zebra scanner thread completed")
            
        except Exception as e:
            self.logger.error(f"❌ Error in Signal-Triggered Zebra scanner thread: {str(e)}", exc_info=True)
    
    def scan_fibonacci_hammer(self, trigger_data):
        """Main hammer scanning function with CONCURRENT timeframe scanning AND Zebra threads"""
        try:
            instrument = trigger_data.get('instrument')
            direction = trigger_data.get('direction')
            trigger_timeframe = trigger_data.get('trigger_timeframe')
            criteria = trigger_data.get('type')
            signal_data = trigger_data.get('signal_data', {})
            
            if not signal_data:
                self.logger.error("❌ No signal_data in trigger_data")
                return False
            
            # Get Fibonacci zones WITH SL and TP
            fib_data = self._get_fib_zones(trigger_data)
            
            if not fib_data or not fib_data.get('is_valid', False):
                error_msg = fib_data.get('error', 'Unknown error') if fib_data else 'No data returned'
                self.logger.error(f"❌ Invalid Fibonacci setup: {error_msg}")
                return False
            
            fib_zones = fib_data['zones']
            sl_price = fib_data['sl_price']
            tp_price = fib_data['tp_price']
            
            if not fib_zones:
                self.logger.error(f"❌ No Fibonacci zones calculated")
                return False
            
            # Get hammer timeframes (original logic)
            hammer_timeframes = self.get_aligned_timeframes(instrument, criteria, trigger_timeframe)
            self.logger.info(f"🔨 Hammer timeframes: {hammer_timeframes}")
            
            # NEW: Get Zebra timeframes (M1, M3, M5 only)
            zebra_timeframes = self.get_zebra_timeframes(instrument, criteria, trigger_timeframe)
            self.logger.info(f"🦓 Zebra timeframes: {zebra_timeframes}")
            
            signal_id = self._generate_signal_id(trigger_data)
            
            self.logger.info(f"🎯 Starting CONCURRENT scan for {instrument} {criteria}")
            self.logger.info(f"   Signal ID: {signal_id}")
            self.logger.info(f"   Direction: {direction}")
            self.logger.info(f"   Trigger TF: {trigger_timeframe}")
            self.logger.info(f"   SL: {sl_price:.5f}, TP: {tp_price:.5f}")
            
            # Set scan duration based on criteria
            if criteria == 'CRT+SMT':
                if trigger_timeframe == 'H1':
                    max_scan_duration = timedelta(minutes=30)
                elif trigger_timeframe == 'H4':
                    max_scan_duration = timedelta(hours=2)
                else:
                    max_scan_duration = timedelta(minutes=30)
            else:
                max_scan_duration = timedelta(hours=2)  # FVG/SD scan for 2 hours
            
            scan_start = datetime.now(NY_TZ)
            scan_end = scan_start + max_scan_duration
            
            self.logger.info(f"⏰ Scan duration: {max_scan_duration}")
            
            # Create a shared state for all scanners
            shared_state = {
                'hammer_count': 0,
                'zebra_count': 0,
                'scan_end': scan_end,
                'fib_zones': fib_zones,
                'sl_price': sl_price,
                'tp_price': tp_price,
                'instrument': instrument,
                'direction': direction,
                'criteria': criteria,
                'signal_data': signal_data,  # This contains the signal data for Zebra
                'signal_id': signal_id,
                'trigger_data': trigger_data,
                'scanned_candles': {},       # For hammer
                'zebra_scanned_candles': {}, # For Zebra
                'lock': threading.Lock(),
                'zebra_lock': threading.Lock(),
                'trigger_types_found': set(),
            }
            
            # Initialize scanned_candles for each timeframe
            for tf in hammer_timeframes:
                shared_state['scanned_candles'][tf] = set()
            for tf in zebra_timeframes:
                shared_state['zebra_scanned_candles'][tf] = set()
            
            def scan_timeframe(tf):
                """Scan a single timeframe for hammer patterns (runs in separate thread)"""
                try:
                    self.logger.info(f"🔍 Starting {instrument} {tf} hammer scanner thread")
                    
                    tf_scanned_candles = shared_state['scanned_candles'][tf]
                    
                    # Calculate 50% line ONCE (outside the loop)
                    sl_price = shared_state['sl_price']
                    tp_price = shared_state['tp_price']
                    direction = shared_state['direction']
                    
                    if not sl_price or not tp_price:
                        self.logger.error(f"❌ {instrument} {tf}: Missing SL or TP!")
                        return
                    
                    if direction == 'bearish':
                        fifty_percent_line = sl_price - ((sl_price - tp_price) * 0.5)
                    else:  # bullish
                        fifty_percent_line = sl_price + ((tp_price - sl_price) * 0.5)
                    
                    self.logger.info(f"📊 {instrument} {tf}: 50% line at {fifty_percent_line:.5f}")
                    
                    # Log all zones for debugging
                    for i, zone in enumerate(shared_state['fib_zones']):
                        self.logger.info(f"📊 {instrument} {tf} Zone {i+1}: {zone['zone_name']} - {zone['low']:.5f} to {zone['high']:.5f}")
                    
                    while datetime.now(NY_TZ) < shared_state['scan_end']:
                        # 1. Check for CRT invalidation (if applicable)
                        if shared_state['criteria'] == 'CRT+SMT':
                            crt_zone = shared_state['fib_zones'][0] if shared_state['fib_zones'] else None
                            if crt_zone and 'invalidation_level' in crt_zone:
                                df_current = fetch_candles(
                                    shared_state['instrument'], 
                                    'M1', 
                                    count=2, 
                                    api_key=self.credentials['oanda_api_key']
                                )
                                if not df_current.empty:
                                    current_price = df_current.iloc[-1]['close']
                                    
                                    if (shared_state['direction'] == 'bearish' and 
                                        current_price > crt_zone['invalidation_level']):
                                        self.logger.info(
                                            f"❌ CRT invalidated in {tf}: "
                                            f"Price {current_price:.5f} > invalidation {crt_zone['invalidation_level']:.5f}"
                                        )
                                        break
                                    elif (shared_state['direction'] == 'bullish' and 
                                          current_price < crt_zone['invalidation_level']):
                                        self.logger.info(
                                            f"❌ CRT invalidated in {tf}: "
                                            f"Price {current_price:.5f} < invalidation {crt_zone['invalidation_level']:.5f}"
                                        )
                                        break
                        
                        # 2. Wait for this specific timeframe's candle
                        self.logger.info(f"⏰ {tf}: Waiting for candle close...")
                        if not self.wait_for_candle_open(tf):
                            self.logger.warning(f"⚠️ {tf}: Could not wait for candle open, continuing...")
                            time.sleep(1)
                            continue
                        
                        # 3. Add small buffer for API data
                        time.sleep(1)
                        self.logger.info(f"✅ {tf}: Candle should be available, fetching data...")
                        
                        # 4. Fetch data after candle close
                        df = fetch_candles(
                            shared_state['instrument'], 
                            tf, 
                            count=10, 
                            api_key=self.credentials['oanda_api_key']
                        )
                        
                        if df.empty or len(df) < 2:
                            time.sleep(1)
                            continue
                        
                        # 5. Get the last CLOSED candle (index -2)
                        closed_candle = df.iloc[-2]
                        candle_key = f"{tf}_{closed_candle['time']}"
                        
                        if candle_key in tf_scanned_candles:
                            time.sleep(1)
                            continue
                        
                        tf_scanned_candles.add(candle_key)
                        
                        # 6. CHECK IF SL OR TP HAS BEEN HIT
                        candle_high = closed_candle['high']
                        candle_low = closed_candle['low']
                        
                        if direction == 'bearish':
                            # For bearish: if candle high goes above SL, setup is invalid
                            if candle_high > sl_price:
                                self.logger.info(f"❌ {instrument} {tf}: SL HIT! Candle high {candle_high:.5f} >= SL {sl_price:.5f}")
                                break
                            
                            # For bearish: if candle low goes below TP, target reached
                            if candle_low < tp_price:
                                self.logger.info(f"✅ {instrument} {tf}: TP HIT! Candle low {candle_low:.5f} <= TP {tp_price:.5f}")
                                break
                        else:  # bullish
                            # For bullish: if candle low goes below SL, setup is invalid
                            if candle_low < sl_price:
                                self.logger.info(f"❌ {instrument} {tf}: SL HIT! Candle low {candle_low:.5f} <= SL {sl_price:.5f}")
                                break
                            
                            # For bullish: if candle high goes above TP, target reached
                            if candle_high > tp_price:
                                self.logger.info(f"✅ {instrument} {tf}: TP HIT! Candle high {candle_high:.5f} >= TP {tp_price:.5f}")
                                break
                        # END OF SL/TP CHECK
                        
                        # 7. DEBUG: Log candle details
                        self.logger.info(f"📊 {instrument} {tf}: Scanning at {datetime.now(NY_TZ).strftime('%H:%M:%S')}")
                        self.logger.info(f"   Candle time: {closed_candle['time']}")
                        self.logger.info(f"   Prices: O:{closed_candle['open']:.5f} H:{closed_candle['high']:.5f} "
                                       f"L:{closed_candle['low']:.5f} C:{closed_candle['close']:.5f}")
                        self.logger.info(f"   Direction: {shared_state['direction']}")
                        
                        # 8. Check hammer pattern
                        is_hammer, upper_ratio, lower_ratio = self.is_hammer_candle(
                            closed_candle, 
                            shared_state['direction']
                        )
                        self.logger.info(f"   {instrument} {tf}: Hammer check: {is_hammer}")
                        self.logger.info(f"   Wick ratios: upper={upper_ratio:.2f}, lower={lower_ratio:.2f}")
                        
                        if is_hammer:
                            self.logger.info(f"✅ {instrument} {tf}: HAMMER DETECTED! Checking if in zone...")
                            self.log_detailed_candle_analysis(closed_candle, tf, shared_state['direction'])
                        
                        # 9. Check if price is on correct side of 50% line
                        candle_price = closed_candle['close']
                        in_valid_zone = False
                        
                        # Get 50% line from zone (or calculate it)
                        for zone in shared_state['fib_zones']:
                            fifty_percent_line = zone.get('mid', zone['high'])  # Use mid or high as 50% line
                            
                            if direction == 'bearish':
                                # For bearish: price must be ABOVE 50% line
                                if candle_price >= fifty_percent_line:
                                    in_valid_zone = True
                                    target_zone = zone
                                    self.logger.info(f"✅ {instrument} {tf}: Price {candle_price:.5f} ABOVE 50% line {fifty_percent_line:.5f}")
                                    break
                                else:
                                    self.logger.info(f"❌ {instrument} {tf}: Price {candle_price:.5f} BELOW 50% line {fifty_percent_line:.5f}")
                            else:  # bullish
                                # For bullish: price must be BELOW 50% line
                                if candle_price <= fifty_percent_line:
                                    in_valid_zone = True
                                    target_zone = zone
                                    self.logger.info(f"✅ {instrument} {tf}: Price {candle_price:.5f} BELOW 50% line {fifty_percent_line:.5f}")
                                    break
                                else:
                                    self.logger.info(f"❌ {instrument} {tf}: Price {candle_price:.5f} ABOVE 50% line {fifty_percent_line:.5f}")
                            
                            break  # Only check first zone (should be the 50% zone)
                        
                        if not in_valid_zone:
                            self.logger.info(f"❌ {instrument} {tf}: Candle not on correct side of 50% line")
                            self.logger.info(f"   Price: {candle_price:.5f}, 50% line: {fifty_percent_line:.5f}")
                            time.sleep(1)
                            continue
                        
                        # 10. If hammer AND in zone, process it
                        if is_hammer:
                            self.logger.info(f"✅ {instrument} {tf}: HAMMER FOUND in Fibonacci zone!")
                            self.logger.info(f"   Time: {closed_candle['time']}")
                            self.logger.info(f"   Price: {candle_price:.5f}, Zone: {target_zone['zone_name'] if target_zone else 'N/A'}")
                            self.logger.info(f"   Wick ratios: upper={upper_ratio:.2f}, lower={lower_ratio:.2f}")
                            
                            # 11. Process and record hammer with thread safety
                            with shared_state['lock']:
                                shared_state['hammer_count'] += 1
                                current_hammer_count = shared_state['hammer_count']
                            
                            success = self._process_and_record_hammer(
                                shared_state['instrument'], 
                                tf, 
                                closed_candle, 
                                shared_state['direction'],
                                shared_state['criteria'], 
                                shared_state['signal_data'], 
                                shared_state['signal_id'], 
                                shared_state['trigger_data']
                            )
                            
                            if success:
                                self.logger.info(f"✅ {instrument} {tf}: Trade #{current_hammer_count} processed successfully")
                            
                            # Continue scanning for more hammers in this timeframe
                            # (Don't break, keep looking for more)
                        
                        # Small pause to avoid API rate limits and excessive CPU
                        time.sleep(1)
                        
                    self.logger.info(f"⏰ {instrument} {tf} hammer scanner thread completed")
                    
                except Exception as e:
                    self.logger.error(f"❌ Error in {instrument} {tf} hammer scanner thread: {str(e)}", exc_info=True)
            
            # Start threads for each timeframe
            threads = []
            for tf in hammer_timeframes:
                hammer_thread = threading.Thread(
                    target=scan_timeframe,  # Your existing hammer scanner function
                    args=(tf,),
                    name=f"HammerScan_{instrument}_{tf}",
                    daemon=True
                )
                hammer_thread.start()
                threads.append(hammer_thread)
                self.logger.info(f"🚀 Started Hammer {tf} scanner thread")
            
            # NEW: Start Zebra Scanner Threads (with different timeframes)
            for tf in zebra_timeframes:
                zebra_thread = threading.Thread(
                    target=self.run_zebra_scan_with_signal,
                    args=(tf, shared_state),
                    name=f"ZebraScan_{instrument}_{tf}",
                    daemon=True
                )
                zebra_thread.start()
                threads.append(zebra_thread)
                self.logger.info(f"🦓 Started Zebra {tf} scanner thread")
            
            # Main thread waits for scan duration or until interrupted
            try:
                # Calculate total seconds to wait
                total_seconds = (shared_state['scan_end'] - datetime.now(NY_TZ)).total_seconds()
                if total_seconds > 0:
                    self.logger.info(f"⏰ Main thread waiting {total_seconds:.0f}s for scan completion...")
                    time.sleep(total_seconds)
                else:
                    self.logger.warning(f"⚠️ Scan duration has already passed")
            except KeyboardInterrupt:
                self.logger.info(f"🛑 Scan interrupted for {instrument}")
            except Exception as e:
                self.logger.error(f"❌ Error in main thread wait: {str(e)}")
            
            # Wait for all threads to finish (they should finish when scan_end is reached)
            for thread in threads:
                thread.join(timeout=5)  # Wait up to 5 seconds for each thread
                self.logger.info(f"✓ {thread.name} joined")
            
            total_hammers = shared_state['hammer_count']
            total_zebras = shared_state['zebra_count']
            self.logger.info(f"✅ CONCURRENT scan completed. Found {total_hammers} hammers and {total_zebras} Zebra signals.")
            
            # Log summary of scanned candles
            for tf, candles in shared_state['scanned_candles'].items():
                self.logger.info(f"📊 {tf}: Scanned {len(candles)} candles for hammers")
            for tf, candles in shared_state['zebra_scanned_candles'].items():
                self.logger.info(f"🦓 {tf}: Scanned {len(candles)} candles for Zebra")
            
            if total_hammers > 0 or total_zebras > 0:
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Error in hammer scan: {str(e)}", exc_info=True)
            return False

    def log_detailed_candle_analysis(self, candle, timeframe, direction):
        """Log detailed analysis of a candle for debugging"""
        try:
            total_range = candle['high'] - candle['low']
            if total_range == 0:
                self.logger.warning("   Zero range candle")
                return
            
            upper_wick = candle['high'] - max(candle['close'], candle['open'])
            lower_wick = min(candle['close'], candle['open']) - candle['low']
            
            # Calculate ratios
            upper_ratio = upper_wick / total_range if total_range > 0 else 0
            lower_ratio = lower_wick / total_range if total_range > 0 else 0
            
            self.logger.info(f"📊 DETAILED CANDLE ANALYSIS ({timeframe}, {direction}):")
            self.logger.info(f"   Total range: {total_range:.5f}")
            self.logger.info(f"   Upper wick: {upper_wick:.5f} ({upper_ratio:.1%})")
            self.logger.info(f"   Lower wick: {lower_wick:.5f} ({lower_ratio:.1%})")
            
            if direction == 'bearish':
                hammer_criteria = upper_ratio > 0.5
                self.logger.info(f"   Hammer criteria: Upper wick > 50% → {hammer_criteria}")
                if hammer_criteria:
                    self.logger.info(f"   ✅ BEARISH HAMMER: Upper wick is {upper_ratio:.1%} (>50%)")
                else:
                    self.logger.info(f"   ❌ NOT A BEARISH HAMMER: Upper wick is {upper_ratio:.1%} (need >50%)")
            else:  # bullish
                hammer_criteria = lower_ratio > 0.5
                self.logger.info(f"   Hammer criteria: Lower wick > 50% → {hammer_criteria}")
                if hammer_criteria:
                    self.logger.info(f"   ✅ BULLISH HAMMER: Lower wick is {lower_ratio:.1%} (>50%)")
                else:
                    self.logger.info(f"   ❌ NOT A BULLISH HAMMER: Lower wick is {lower_ratio:.1%} (need >50%)")
                    
        except Exception as e:
            self.logger.error(f"   Error in detailed analysis: {str(e)}")
    
    # def _get_next_candle_close_time(self, timeframe, current_time):
    #     """Calculate next candle close time"""
    #     if timeframe.startswith('M'):
    #         minutes = int(timeframe[1:])
    #         current_minute = current_time.minute
    #         minutes_past = current_minute % minutes
    #         minutes_to_close = minutes - minutes_past
            
    #         # If exactly at close time, wait for next candle
    #         if minutes_to_close == 0:
    #             minutes_to_close = minutes
            
    #         next_close = current_time + timedelta(minutes=minutes_to_close)
    #         next_close = next_close.replace(second=0, microsecond=0)
    #     else:
    #         # For hourly timeframes (shouldn't be used for hammer scanning)
    #         next_close = current_time + timedelta(hours=1)
    #         next_close = next_close.replace(minute=0, second=0, microsecond=0)
        
    #     return next_close
    
    def _should_stop_scanning(self, instrument, trigger_data, scan_start_time, 
                             max_duration, hammer_count):
        """Check if we should stop scanning"""
        current_time = datetime.now(NY_TZ)
        
        # 1. Check max duration
        if current_time - scan_start_time > max_duration:
            self.logger.info(f"⏰ Max scan duration reached (2 hours)")
            return True
        
        # 2. Check if setup is invalidated
        if not self._is_setup_still_valid_overall(instrument, trigger_data):
            self.logger.info(f"❌ Setup completely invalidated")
            return True
        
        # 3. Optional: stop after finding N hammers
        if hammer_count >= 5:  # Optional limit
            self.logger.info(f"✅ Found {hammer_count} hammers - stopping scan")
            return True
        
        return False
    
    

    
    def _calculate_sleep_time(self, timeframes):
        """Calculate sleep time based on the shortest timeframe"""
        # Find the minimum timeframe in minutes
        min_minutes = float('inf')
        
        for tf in timeframes:
            if tf.startswith('M'):
                minutes = int(tf[1:])
            elif tf == 'H1':
                minutes = 60
            elif tf == 'H4':
                minutes = 240
            else:
                minutes = 60
            
            min_minutes = min(min_minutes, minutes)
        
        # Sleep for 70% of the shortest timeframe (in seconds)
        sleep_minutes = min_minutes * 0.7
        return int(sleep_minutes * 60)

    def _get_safe_news_data(self, news_context, instrument):
        """Safely extract news data with proper error handling"""
        if not news_context or not isinstance(news_context, dict):
            return {
                'news_context_json': '',
                'news_high_count': 0,
                'news_medium_count': 0,
                'news_low_count': 0,
                'next_news_time': '',
                'next_news_event': '',
                'next_news_currency': '',
                'prev_news_time': '',
                'prev_news_event': '',
                'prev_news_currency': '',
                'seconds_to_next_news': '',
                'seconds_since_last_news': '',
                'news_timing_category': '',
                'news_fetch_status': 'error' if news_context is None else 'disabled'
            }
        
        try:
            # Get timing data safely
            timing_data = news_context.get('timing', {})
            if not isinstance(timing_data, dict):
                timing_data = {}
            
            # Get future and past events
            closest_future = timing_data.get('closest_future_event')
            closest_past = timing_data.get('closest_past_event')
            
            if closest_future and not isinstance(closest_future, dict):
                closest_future = {}
            if closest_past and not isinstance(closest_past, dict):
                closest_past = {}
            
            return {
                'news_context_json': json.dumps(news_context),
                'news_high_count': news_context.get('high_impact_count', 0),
                'news_medium_count': news_context.get('medium_impact_count', 0),
                'news_low_count': news_context.get('low_impact_count', 0),
                'next_news_time': closest_future.get('ny_time', '') if closest_future else '',
                'next_news_event': closest_future.get('event', '') if closest_future else '',
                'next_news_currency': closest_future.get('currency', '') if closest_future else '',
                'prev_news_time': closest_past.get('ny_time', '') if closest_past else '',
                'prev_news_event': closest_past.get('event', '') if closest_past else '',
                'prev_news_currency': closest_past.get('currency', '') if closest_past else '',
                'seconds_to_next_news': timing_data.get('seconds_to_next', ''),
                'seconds_since_last_news': timing_data.get('seconds_since_last', ''),
                'news_timing_category': timing_data.get('timing_category', ''),
                'news_fetch_status': news_context.get('fetch_status', 'success')
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error extracting news data: {str(e)}")
            return {
                'news_context_json': '',
                'news_high_count': 0,
                'news_medium_count': 0,
                'news_low_count': 0,
                'next_news_time': '',
                'next_news_event': '',
                'next_news_currency': '',
                'prev_news_time': '',
                'prev_news_event': '',
                'prev_news_currency': '',
                'seconds_to_next_news': '',
                'seconds_since_last_news': '',
                'news_timing_category': '',
                'news_fetch_status': f'error: {str(e)[:50]}'
            }

    def calculate_zebra_features(self, instrument, hammer_time, timeframe_data=None):
        """Calculate HalfTrend Zebra indicator using pre-fetched data"""
        try:
            features = {}
            timeframe_map = {
                'M1': '1m', 'M3': '3m', 'M5': '5m', 'M15': '15m',
                'H1': 'h1', 'H4': 'h4', 'H6': 'h6', 'D': 'd'
            }
            
            for tf, csv_key in timeframe_map.items():
                try:
                    # Use pre-fetched data if available
                    if timeframe_data and tf in timeframe_data:
                        df = timeframe_data[tf]
                    else:
                        df = fetch_candles(instrument, tf, count=200, 
                                          api_key=self.credentials['oanda_api_key'])
                    
                    if df.empty or len(df) < 100:
                        features[f'{csv_key}_zebra'] = 'NaN'
                        continue
                    
                    # Calculate HalfTrend
                    arrup, arrdwn = self._calculate_half_trend(df)
                    
                    # DEBUG: Check if calculation returned valid data
                    if len(arrup) == 0 or len(arrdwn) == 0:
                        self.logger.warning(f"⚠️ HalfTrend calculation failed for {tf}")
                        features[f'{csv_key}_zebra'] = 'ERROR'
                        continue
                    
                    # Get last signal
                    last_signal = self._get_last_half_trend_signal(arrup, arrdwn, hammer_time, df)
                    features[f'{csv_key}_zebra'] = last_signal
                    
                    # DEBUG: Log the result
                    self.logger.info(f"📊 {tf} Zebra signal: {last_signal}")
                    
                except Exception as e:
                    self.logger.error(f"❌ Error in Zebra for {tf}: {str(e)}")
                    features[f'{csv_key}_zebra'] = 'ERROR'
            
            return features
        except Exception as e:
            self.logger.error(f"❌ Error in calculate_zebra_features: {str(e)}")
            return {f'{csv_key}_zebra': 'ERROR' for csv_key in timeframe_map.values()}

    def _calculate_half_trend(self, df, amplitude=2, atr_period=100):
        try:
            import numpy as np
            import pandas as pd  # ADD THIS LINE
            
            n = len(df)
            trend = np.zeros(n)
            up = np.zeros(n)
            down = np.zeros(n)
            atrlo = np.zeros(n)
            atrhi = np.zeros(n)
            arrup = np.full(n, np.nan)
            arrdwn = np.full(n, np.nan)
            
            # Initialize as in MQL4
            nexttrend = 0
            minhighprice = df['high'].iloc[0] if n > 0 else 0
            maxlowprice = df['low'].iloc[0] if n > 0 else 0
            
            # Calculate ATR/2 for each bar
            # First calculate True Range
            high_low = df['high'] - df['low']
            high_close_prev = abs(df['high'] - df['close'].shift(1))
            low_close_prev = abs(df['low'] - df['close'].shift(1))
            tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
            
            # Calculate ATR (SMA of TR)
            atr = tr.rolling(window=atr_period, min_periods=1).mean() / 2
            
            # Main loop following MQL4 logic
            for i in range(1, n):
                # Get lowest low and highest high of Amplitude period
                start_idx = max(0, i - amplitude + 1)
                lowprice_i = df['low'].iloc[start_idx:i+1].min()
                highprice_i = df['high'].iloc[start_idx:i+1].max()
                
                # Calculate SMA of low and high
                lowma = df['low'].iloc[start_idx:i+1].mean()
                highma = df['high'].iloc[start_idx:i+1].mean()
                
                # Copy previous trend
                trend[i] = trend[i-1]
                
                if nexttrend == 1:
                    maxlowprice = max(lowprice_i, maxlowprice)
                    if highma < maxlowprice and df['close'].iloc[i] < df['low'].iloc[i-1]:
                        trend[i] = 1.0
                        nexttrend = 0
                        minhighprice = highprice_i
                        
                if nexttrend == 0:
                    minhighprice = min(highprice_i, minhighprice)
                    if lowma > minhighprice and df['close'].iloc[i] > df['high'].iloc[i-1]:
                        trend[i] = 0.0
                        nexttrend = 1
                        maxlowprice = lowprice_i
                
                # Set up and down lines
                if trend[i] == 0.0:  # Uptrend
                    if trend[i-1] != 0.0:  # Trend just changed
                        up[i] = down[i-1]
                        if i-1 >= 0:
                            up[i-1] = up[i]
                        arrup[i] = up[i] - 2 * atr.iloc[i]
                    else:
                        up[i] = max(maxlowprice, up[i-1])
                    
                    atrhi[i] = up[i] - atr.iloc[i]
                    atrlo[i] = up[i]
                    down[i] = 0.0
                    
                else:  # Downtrend
                    if trend[i-1] != 1.0:  # Trend just changed
                        down[i] = up[i-1]
                        if i-1 >= 0:
                            down[i-1] = down[i]
                        arrdwn[i] = down[i] + 2 * atr.iloc[i]
                    else:
                        down[i] = min(minhighprice, down[i-1])
                    
                    atrhi[i] = down[i] + atr.iloc[i]
                    atrlo[i] = down[i]
                    up[i] = 0.0
            
            return arrup, arrdwn
            
        except Exception as e:
            self.logger.error(f"❌ Error in _calculate_half_trend: {str(e)}")
            return np.array([]), np.array([])
    
    def _get_last_half_trend_signal(self, arrup, arrdwn, hammer_time, df):
        """Get the most recent HalfTrend arrow signal"""
        try:
            # Find indices where we have arrows
            arrup_indices = np.where(~np.isnan(arrup))[0]
            arrdwn_indices = np.where(~np.isnan(arrdwn))[0]
            
            if len(arrup_indices) == 0 and len(arrdwn_indices) == 0:
                return 'NaN'
            
            # Get the most recent arrow before or at hammer_time
            # Since we're looking for most recent signal overall (not just at hammer_time),
            # we take the last index from either array
            
            last_arrup_idx = arrup_indices[-1] if len(arrup_indices) > 0 else -1
            last_arrdwn_idx = arrdwn_indices[-1] if len(arrdwn_indices) > 0 else -1
            
            if last_arrup_idx > last_arrdwn_idx:
                return 'green'
            elif last_arrdwn_idx > last_arrup_idx:
                return 'red'
            else:
                # If both have same index (shouldn't happen), check which one is not NaN at that index
                idx = last_arrup_idx
                if idx >= 0:
                    if not np.isnan(arrup[idx]):
                        return 'green'
                    elif not np.isnan(arrdwn[idx]):
                        return 'red'
            
            return 'NaN'
            
        except Exception as e:
            self.logger.error(f"❌ Error in _get_last_half_trend_signal: {str(e)}")
            return 'NaN'

    def send_webhook_signal(self, instrument, direction, entry_price, sl_price, tp_price, 
                        signal_id, trade_id, timeframe, criteria, risk_usd=50.0):
        """
        UPDATED: Uses environment variables for webhook URL and secret.
        Send low-latency webhook signal for immediate execution.
        """
        try:
            # ============================================
            # 🚀 CONFIGURATION FROM ENVIRONMENT
            # ============================================
            webhook_url = os.getenv("WEBHOOK_URL")           # e.g., "https://abc123.ngrok-free.app/webhook"
            webhook_token = os.getenv("WEBHOOK_SECRET")      # Must match local server's WEBHOOK_SECRET
            
            if not webhook_url or not webhook_token:
                self.logger.error("❌ WEBHOOK_URL or WEBHOOK_SECRET not set in environment.")
                return False
            
            # ============================================
            # SYMBOL MAPPING (Colab → Local Server)
            # ============================================
            symbol_map = {
                'EUR_USD': 'EUR_USD',
                'GBP_USD': 'GBP_USD', 
                'XAU_USD': 'XAU_USD',
                'XAG_USD': 'XAG_USD',
                'NAS100_USD': 'NAS100_USD',
                'SPX500_USD': 'SPX500_USD',
                'SPX500_USD': 'SPX500_USD',
                'EU50_EUR': 'EU50_EUR'
            }
            
            instrument_upper = instrument.upper()
            local_symbol = symbol_map.get(instrument_upper, instrument_upper)
            
            # ============================================
            # HEADERS (Authentication)
            # ============================================
            headers = {
                "Authorization": f"Bearer {webhook_token}",
                "Content-Type": "application/json",
                "User-Agent": "HammerScanner/1.0",
                "X-Signal-ID": signal_id
            }
            
            # ============================================
            # PAYLOAD (EXACT format local server expects)
            # ============================================
            from datetime import datetime  # Ensure datetime is imported
            body = {
                "signal_id": signal_id,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "symbol": local_symbol,
                "direction": "BUY" if direction.lower() == 'bullish' else "SELL",
                "stop_loss": round(float(sl_price), 5),
                "take_profit": round(float(tp_price), 5),
                "risk_usd": float(risk_usd),
                "strategy_tag": f"Hammer_{criteria}"
            }
            
            self.logger.info("=" * 60)
            self.logger.info(f"🚀 WEBHOOK SIGNAL (via ENV Vars)")
            self.logger.info(f"   📡 URL: {webhook_url}")
            self.logger.info(f"   🎯 Symbol: {instrument} → {local_symbol}")
            self.logger.info(f"   📊 Direction: {direction} → {body['direction']}")
            self.logger.info(f"   💰 Entry: {entry_price:.5f}, SL: {sl_price:.5f}, TP: {tp_price:.5f}")
            self.logger.info(f"   🔑 Signal ID: {signal_id}")
            self.logger.info("=" * 60)
            
            # ============================================
            # SEND REQUEST
            # ============================================
            import requests
            response = requests.post(webhook_url, json=body, headers=headers, timeout=5)
            
            # Handle response
            if 200 <= response.status_code < 300:
                self.logger.info(f"✅ WEBHOOK SUCCESS: Order #{response.json().get('order_ticket', 'N/A')}")
                return True
            else:
                self.logger.error(f"❌ WEBHOOK FAILED: HTTP {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Webhook error: {str(e)}")
            return False
    
    def _save_failed_webhook(self, webhook_data, instrument, trade_id):
        """Save failed webhook data for later retry"""
        try:
            failed_dir = os.path.join(self.csv_base_path, 'failed_webhooks')
            os.makedirs(failed_dir, exist_ok=True)
            
            filename = f"failed_{instrument}_{trade_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(failed_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(webhook_data, f, indent=2)
            
            self.logger.info(f"💾 Saved failed webhook to {filepath}")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving failed webhook: {str(e)}")

    def find_3_candle_pivot(self, df, direction, lookback=25):
        """Finds the 3-candle pivot SL logic"""
        for i in range(1, lookback):
            prev, mid, next_c = df.iloc[-(i+2)], df.iloc[-(i+1)], df.iloc[-i]
            if direction.lower() == 'bullish' or direction.lower() == 'buy':
                if mid['low'] < prev['low'] and mid['low'] < next_c['low']:
                    return mid['low']
            else: # bearish
                if mid['high'] > prev['high'] and mid['high'] > next_c['high']:
                    return mid['high']
        return None

    def _calc_expected_candle_start(self, current_dt, timeframe):
        """
        Returns the START time (UTC datetime) of the candle that SHOULD be 
        currently forming at the given 'current_dt' for the specified 'timeframe'.
        
        OANDA's 'time' field is the START time of the candle period.
        """
        # First convert current_dt to NY time if it isn't already
        if current_dt.tzinfo is None:
            current_dt = NY_TZ.localize(current_dt)
        else:
            current_dt = current_dt.astimezone(NY_TZ)
        # For minute-based timeframes (M1, M5, M15, M30)
        if timeframe.startswith('M'):
            minutes = int(timeframe[1:])
            # Calculate total minutes since midnight
            total_minutes = current_dt.hour * 60 + current_dt.minute
            # Find the start minute of the current X-minute period
            start_minute = (total_minutes // minutes) * minutes
            # Return datetime with that start minute
            return current_dt.replace(hour=start_minute//60, 
                                      minute=start_minute%60, 
                                      second=0, 
                                      microsecond=0)
        
        # For hour-based timeframes (H1, H4)
        elif timeframe.startswith('H'):
            hours = int(timeframe[1:])
            start_hour = (current_dt.hour // hours) * hours
            return current_dt.replace(hour=start_hour, 
                                      minute=0, 
                                      second=0, 
                                      microsecond=0)
        
        # For daily timeframe (D)
        elif timeframe == 'D':
            return current_dt.replace(hour=0, 
                                      minute=0, 
                                      second=0, 
                                      microsecond=0)
        
        # For weekly timeframe (W)
        elif timeframe == 'W':
            # Find most recent Monday at 00:00
            days_since_monday = current_dt.weekday()  # Monday=0, Sunday=6
            return current_dt - timedelta(days=days_since_monday)
        
        else:
            self.logger.error(f"Unhandled timeframe in time calc: {timeframe}")
            # Fallback: assume current minute start
            return current_dt.replace(second=0, microsecond=0)
    
    
    def _calc_seconds_to_next_candle(self, current_dt, timeframe):
        """
        Calculate seconds until the start of the NEXT candle.
        Used for efficient sleeping when data is stale.
        """
        # Get start of current period
        current_start = self._calc_expected_candle_start(current_dt, timeframe)
        
        # Calculate start of next period based on timeframe
        if timeframe.startswith('M'):
            minutes = int(timeframe[1:])
            next_start = current_start + timedelta(minutes=minutes)
        elif timeframe.startswith('H'):
            hours = int(timeframe[1:])
            next_start = current_start + timedelta(hours=hours)
        elif timeframe == 'D':
            next_start = current_start + timedelta(days=1)
        elif timeframe == 'W':
            next_start = current_start + timedelta(weeks=1)
        else:
            # Default fallback
            return 60
        
        # Return seconds until next period starts
        return max(0, (next_start - current_dt).total_seconds())

    def analyze_quarter_structure(self, instrument, price, timestamp=None):
        """
        Analyze the quarter structure for debugging and logging.
        Simplified to show up/down relation.
        """
        if timestamp is None:
            timestamp = datetime.now(NY_TZ)
        
        analysis = {
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'instrument': instrument,
            'price': price,
            'quarters': {},
            'summary': ''
        }
        
        # Get all current quarters
        quarters = self.quarter_manager.get_current_quarters(timestamp)
        
        summary_lines = []
        
        for cycle_type in ['monthly', 'weekly', 'daily', '90min']:
            current_quarter = quarters.get(cycle_type)
            true_open, _ = self.get_true_open_for_cycle(instrument, cycle_type, timestamp)
            
            if true_open:
                if price > true_open:
                    relation = "UP"
                    emoji = "🔼"
                elif price < true_open:
                    relation = "DOWN"
                    emoji = "🔽"
                else:
                    relation = "EQUAL"
                    emoji = "⚖️"
            else:
                relation = "UNKNOWN"
                emoji = "❓"
            
            analysis['quarters'][cycle_type] = {
                'current_quarter': current_quarter,
                'true_open': true_open,
                'relation': relation
            }
            
            if true_open:
                summary_lines.append(f"{cycle_type.upper():8} | Q{current_quarter} | "
                                   f"True Open: {true_open:.5f} | {emoji} {relation}")
        
        # Create summary string
        analysis['summary'] = "\n".join(summary_lines)
        
        # Log the analysis
        self.logger.info(f"📊 QUARTER ANALYSIS for {instrument}:")
        for line in summary_lines:
            self.logger.info(f"   {line}")
        
        return analysis

    def check_quarter_filters(self, instrument, direction, price, timestamp=None):
        """
        Apply quarter-based filters to determine trade validity.
        Returns True if passes all filters.
        """
        if timestamp is None:
            timestamp = datetime.now(NY_TZ)
        
        filters_passed = {
            'monthly': True,
            'weekly': True,
            'daily': True,
            '90min': True
        }
        
        # Get all current quarters
        quarters = self.quarter_manager.get_current_quarters(timestamp)
        
        for cycle_type in ['daily', '90min']:  # Focus on shorter cycles for filtering
            current_quarter = quarters.get(cycle_type)
            true_open, _ = self.get_true_open_for_cycle(instrument, cycle_type, timestamp)
            
            if not true_open:
                continue
            
            # Filter logic based on direction and quarter
            if direction == 'bearish':
                # For bearish trades, prefer being below true open in Q3/Q4
                if current_quarter in ['q3', 'q4'] and price > true_open:
                    filters_passed[cycle_type] = False
                    self.logger.info(f"⏸️ {cycle_type} filter failed: Bearish in {current_quarter} but price above true open")
            
            elif direction == 'bullish':
                # For bullish trades, prefer being above true open in Q1/Q2
                if current_quarter in ['q1', 'q2'] and price < true_open:
                    filters_passed[cycle_type] = False
                    self.logger.info(f"⏸️ {cycle_type} filter failed: Bullish in {current_quarter} but price below true open")
        
        # Check if all required filters passed
        required_cycles = ['daily']  # Can adjust based on strategy
        for cycle in required_cycles:
            if not filters_passed[cycle]:
                return False
        
        return True

    def run_zebra_scan(self, tf, instrument, signal_id_prefix):
        """
        Threaded Zebra scanner with CORRECTED time validation.
        Verifies the latest candle's START TIME matches the expected current candle.
        """
        self.logger.info(f"🦓 Zebra Scanner Thread Started: {instrument} {tf}")
        import numpy as np
        import time
        from datetime import datetime, timedelta
        import pytz
        
        scanned_candles = set()
        
        while self.running:
            try:
                # 1. Wait for candle open/tick (your existing logic)
                self.wait_for_candle_open(tf)
                
                # 2. Fetch 150 candles
                df = self.cached_fetch_candles(instrument, tf, count=150, force_fetch=True)
                if df.empty or len(df) < 100:
                    time.sleep(2)
                    continue
                
                # 3. GET THE LATEST CANDLE'S START TIME
                latest_candle = df.iloc[-1]
                latest_candle_start = latest_candle['time']
                
                # 4. CALCULATE: What *should* be the current candle's start time?
                current_utc_time = datetime.now(pytz.utc)
                expected_candle_start = self._calc_expected_candle_start(current_utc_time, tf)
                
                # 5. THE CRITICAL FRESHNESS CHECK
                # 5a. Make expected_candle_start NY-timezone-aware
                if expected_candle_start.tzinfo is None:
                    # If it has no timezone, assume it's meant to be NY time
                    expected_candle_start_ny = NY_TZ.localize(expected_candle_start)
                else:
                    # If it has a timezone (like UTC), convert it to NY time
                    expected_candle_start_ny = expected_candle_start.astimezone(NY_TZ)
                
                # 5b. Ensure API timestamp is NY-timezone-aware
                latest_candle_start_ny = latest_candle['time']
                if latest_candle_start_ny.tzinfo is None:
                    # If pandas didn't parse timezone, add NY timezone
                    latest_candle_start_ny = NY_TZ.localize(latest_candle_start_ny)
                
                # 5c. THE CRITICAL FRESHNESS CHECK (comparing NY to NY)
                if latest_candle_start_ny != expected_candle_start_ny:
                    # DATA IS STALE: API hasn't sent the new candle yet
                    self.logger.warning(
                        f"⏳ {instrument} {tf}: Data STALE. "
                        f"Latest NY candle: {latest_candle_start_ny}, "
                        f"Expected NY candle: {expected_candle_start_ny}. "
                        f"Skipping cycle."
                    )
                    
                    # Efficient sleep until next candle open
                    seconds_to_next = self._calc_seconds_to_next_candle(current_utc_time, tf) + 2
                    self.logger.info(f"💤 Sleeping {seconds_to_next:.0f}s until next candle.")
                    time.sleep(seconds_to_next)
                    scanned_candles.add(expected_candle_start_ny)  # Use NY time for tracking
                    continue
                
                # 6. ✅ DATA IS FRESH: Proceed with existing logic
                current_time = latest_candle_start  # This is the start time of the fresh candle
                
                # Prevent multiple entries on the same live candle
                if current_time in scanned_candles:
                    time.sleep(1)
                    continue
    
                # 7. Calculate HalfTrend logic
                # 7. Calculate HalfTrend logic (using last 100 candles only)
                candles_for_half_trend = df.iloc[-100:]  # Takes the last 100 rows
                arrup, arrdwn = self._calculate_half_trend(candles_for_half_trend)
                
                if len(arrup) == 0 or len(arrdwn) == 0:
                    continue
    
                # 8. Check if arrow is present on index [-1] (Current Candle)
                detected_dir = None
                if not np.isnan(arrup[-1]):
                    detected_dir = 'bullish'
                elif not np.isnan(arrdwn[-1]):
                    detected_dir = 'bearish'
    
                # 9. If arrow is live, proceed with checks
                if detected_dir:
                    self.logger.info(f"🎯 ZEBRA {detected_dir.upper()} ARROW on Fresh Candle: {tf} {instrument}")
                    
                    # Entry is the current open (from the fresh candle)
                    entry_price = latest_candle['open']
                    
                    # SL is the 3-candle pivot
                    sl_price = self.find_3_candle_pivot(df, detected_dir)
                    
                    if sl_price:
                        # --- HUGE SL FILTER ---
                        recent_ranges = (df['high'].iloc[-6:-1] - df['low'].iloc[-6:-1]).mean()
                        sl_dist = abs(entry_price - sl_price)
                        
                        if sl_dist > (recent_ranges * 2.5):
                            self.logger.warning(f"⏭️ Zebra Skip: SL is too wide ({round(sl_dist, 5)} vs Avg {round(recent_ranges, 5)})")
                            scanned_candles.add(current_time)
                            continue
    
                        # 10. PROCESS AND JOURNAL
                        # In run_zebra_scan method:
                        success = self._process_and_record_hammer(
                            instrument=instrument,
                            tf=tf,
                            candle=latest_candle,
                            direction=detected_dir,
                            criteria='zebra',
                            signal_data={},  # Empty but still passed
                            signal_id=f"ZEB_{tf}_{int(time.time())}",
                            trigger_data={'type': 'zebra_independent'},  # Add minimal trigger data
                            zebra_entry=entry_price,
                            zebra_sl=sl_price,
                            trigger_type='zebra'
                        )
                        
                        if success:
                            self.logger.info(f"✅ Zebra {detected_dir} logged to CSV and Journaled.")
                            scanned_candles.add(current_time)
                
                # Brief sleep to be kind to CPU
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"❌ Error in Zebra {tf} loop for {instrument}: {str(e)}", exc_info=True)
                time.sleep(5)
    
    def _process_and_record_hammer(self, instrument, tf, candle, direction, criteria, 
                               signal_data, signal_id, trigger_data, 
                               zebra_entry=None, zebra_sl=None, trigger_type='hammer'):
        """Process a single signal (Hammer or Zebra) and record it - FIXED VERSION"""
        try:
            # === NEW: Add trigger_type parameter ===
            trigger_criteria = trigger_data.get('type', '')  # Get original signal type
            trigger_type = trigger_type  # 'hammer' or 'zebra'
            
            # === FIXED: Properly extract signal data based on trigger type ===
            if trigger_type == 'zebra' and trigger_criteria and trigger_criteria != 'zebra_independent':
                # Zebra triggered by a signal (SD/FVG/CRT)
                fvg_idea = signal_data.get('fvg_idea', {})
                smt_data = signal_data.get('smt_data', {})
                zone = signal_data.get('zone', {})
                crt_signal = signal_data.get('crt_signal', {})
                has_psp = signal_data.get('has_psp', False)
                is_hp_fvg = signal_data.get('is_hp_fvg', False)
                is_hp_zone = signal_data.get('is_hp_zone', False)
            elif trigger_type == 'zebra':
                # Independent Zebra (no signal trigger)
                fvg_idea, smt_data, zone, crt_signal, has_psp = {}, {}, {}, {}, False
                is_hp_fvg, is_hp_zone = False, False
            else:
                # Hammer logic
                fvg_idea = signal_data.get('fvg_idea', {})
                smt_data = signal_data.get('smt_data', {})
                zone = signal_data.get('zone', {})
                crt_signal = signal_data.get('crt_signal', {})
                has_psp = signal_data.get('has_psp', False)
                is_hp_fvg = signal_data.get('is_hp_fvg', False)
                is_hp_zone = signal_data.get('is_hp_zone', False)
            
            # === FIXED: ML FILTER CHECK - ONLY FOR HAMMER TRADES ===
            webhook_approved = False
            
            # Zebra trades never use ML filtering
            # if trigger_type == 'zebra':
            #     self.logger.info(f"🦓 Zebra trade {signal_id} - auto-approved")
            #     webhook_approved = True  # Zebra trades are auto-approved
            # elif trigger_type == 'hammer':
            #     # Only hammer trades use ML filtering (if available)
            #     if hasattr(self, 'signal_processor') and self.signal_processor:
            #         # Extract required features for ML filter
            #         smt_cycle = signal_data.get('smt_data', {}).get('cycle', '')
            #         smt_quarters = signal_data.get('smt_data', {}).get('quarters', '')
            #         trigger_tf = trigger_data.get('trigger_timeframe', '')
                    
            #         # Check with ML filter if we should trade
            #         should_trade, hammer_count, prediction = self.signal_processor.check_and_predict(
            #             signal_id=signal_id,
            #             hammer_timeframe=tf,
            #             criteria=criteria,
            #             smt_cycle=smt_cycle,
            #             smt_quarters=smt_quarters,
            #             trigger_timeframe=trigger_tf
            #         )
                    
            #         if hammer_count == 1:
            #             self.logger.info(f"🔍 Signal {signal_id} | TF: {tf} | 1st hammer | ML: {prediction}")
            #         else:
            #             self.logger.info(f"🔍 Signal {signal_id} | TF: {tf} | Hammer #{hammer_count}")
                    
            #         if not should_trade:
            #             if hammer_count > 1:
            #                 self.logger.info(f"⏸️ Skipping - Not 1st hammer on {tf} for signal {signal_id}")
            #             else:
            #                 self.logger.info(f"⏸️ ML rejected 1st hammer on {tf} (prediction: {prediction})")
            #             return False
            #         else:
            #             webhook_approved = True  # ✅ Store the approval
            #             self.logger.info(f"✅ ML approved 1st hammer on {tf} (prediction: {prediction})")
            #     else:
            #         self.logger.warning(f"⚠️ No signal_processor for hammer {signal_id}")
            #         webhook_approved = True  # Approve without ML
            # else:
            #     self.logger.error(f"❌ Unknown trigger_type: {trigger_type}")
            #     return False
            
            # 1. SETUP PRICES BASED ON CRITERIA
            if trigger_type == 'zebra':
                current_price = zebra_entry
                sl_price = zebra_sl
            else:
                # Get current price for entry
                current_df = fetch_candles(instrument, tf, count=1, api_key=self.credentials['oanda_api_key'])
                if current_df.empty:
                    self.logger.error(f"❌ Cannot get current price for {instrument}")
                    return False
                current_price = current_df.iloc[-1]['open']
    
            current_time = datetime.now(NY_TZ)
            pip_multiplier = 100 if 'JPY' in instrument else 10000
    
            # 2. CALCULATE SIGNAL LATENCY
            candle_close_time = candle['time']
            if isinstance(candle_close_time, str):
                candle_close_time = datetime.strptime(candle_close_time, '%Y-%m-%d %H:%M:%S')
            signal_latency_seconds = (current_time - candle_close_time).total_seconds()
    
            # === ADD QUARTER ANALYSIS HERE ===
            signal_time = candle['time']
            
            quarter_features = {}
            for cycle_type in ['monthly', 'weekly', 'daily', '90min']:
                try:
                    true_open, current_quarter = self.get_true_open_for_cycle(instrument, cycle_type, signal_time)
                    
                    # Store features
                    quarter_features[f'current_quarter_{cycle_type}'] = current_quarter or ''
                    quarter_features[f'true_open_{cycle_type}'] = round(true_open, 5) if true_open else ''
                    
                    # Calculate relation to true open
                    if true_open and current_price:
                        if current_price > true_open:
                            quarter_features[f'true_open_relation_{cycle_type}'] = 'up'
                        elif current_price < true_open:
                            quarter_features[f'true_open_relation_{cycle_type}'] = 'down'
                        else:
                            quarter_features[f'true_open_relation_{cycle_type}'] = 'equal'
                    else:
                        quarter_features[f'true_open_relation_{cycle_type}'] = ''
                        
                except Exception as cycle_error:
                    self.logger.error(f"❌ Error calculating {cycle_type} quarter: {str(cycle_error)}")
                    quarter_features[f'current_quarter_{cycle_type}'] = ''
                    quarter_features[f'true_open_{cycle_type}'] = ''
                    quarter_features[f'true_open_relation_{cycle_type}'] = ''
            
            # Log quarter analysis
            self.logger.info(f"📊 QUARTER ANALYSIS for {instrument} ({trigger_type.upper()}):")
            for cycle_type in ['monthly', 'weekly', 'daily', '90min']:
                true_open = quarter_features.get(f'true_open_{cycle_type}')
                if true_open:
                    relation = quarter_features.get(f'true_open_relation_{cycle_type}', '')
                    current_quarter = quarter_features.get(f'current_quarter_{cycle_type}', '')
                    if relation == 'up':
                        emoji = "🔼"
                        relation_text = "UP"
                    elif relation == 'down':
                        emoji = "🔽"
                        relation_text = "DOWN"
                    elif relation == 'equal':
                        emoji = "⚖️"
                        relation_text = "EQUAL"
                    else:
                        emoji = "❓"
                        relation_text = "UNKNOWN"
                    self.logger.info(f"   {cycle_type.upper():8} | Q{current_quarter} | "
                                   f"True Open: {true_open:.5f} | {emoji} {relation_text}")
    
            # 3. GET NEWS CONTEXT
            news_context = {}
            if hasattr(self, 'news_calendar') and self.news_calendar:
                try:
                    news_context = self.news_calendar.get_news_for_instrument(instrument, current_time)
                except Exception as e:
                    self.logger.error(f"❌ Error getting news from calendar: {e}")
                    news_context = {'fetch_status': 'error'}
            else:
                news_context = {'fetch_status': 'disabled'}
            
            # Extract news data safely
            safe_news_data = self._get_safe_news_data(news_context, instrument)
    
            # 4. CALCULATE SL FOR HAMMER (IF NOT ZEBRA)
            tp_1_4_price = None
            tp_1_2_price = None
            
            if trigger_type == 'hammer':
                hammer_high, hammer_low = candle['high'], candle['low']
                hammer_range = hammer_high - hammer_low
                if direction == 'bearish':
                    sl_price = hammer_high + (hammer_range * 0.25)
                    tp_1_4_price = current_price - (4 * (sl_price - current_price))
                    tp_1_2_price = current_price - (2 * (sl_price - current_price))
                else:
                    sl_price = hammer_low - (hammer_range * 0.25)
                    tp_1_4_price = current_price + (4 * (current_price - sl_price))
                    tp_1_2_price = current_price + (2 * (current_price - sl_price))
            else:
                # For ZEBRA: Calculate tp_1_4_price and tp_1_2_price from zebra SL
                if direction == 'bearish':
                    tp_1_4_price = current_price - (4 * (sl_price - current_price))
                    tp_1_2_price = current_price - (2 * (sl_price - current_price))
                else:
                    tp_1_4_price = current_price + (4 * (current_price - sl_price))
                    tp_1_2_price = current_price + (2 * (current_price - sl_price))
    
            # 5. CALCULATE TP LEVELS
            tp_prices = {}
            for i in range(1, 11):
                if direction.lower() in ['bearish', 'sell']:
                    tp_prices[i] = current_price - (i * abs(sl_price - current_price))
                else:  # bullish / buy
                    tp_prices[i] = current_price + (i * abs(current_price - sl_price))
            
            sl_distance_pips = abs(current_price - sl_price) * pip_multiplier
            tp_distances = {f'tp_1_{i}_distance': round(sl_distance_pips * i, 1) for i in range(1, 11)}
            
            # Calculate position sizes
            risk_10_lots, risk_100_lots = self.calculate_position_sizes(
                instrument, 
                sl_distance_pips, 
                current_price
            )
            
            # Calculate open TP          
            open_tp_data = self.calculate_open_tp(
                instrument, direction, current_price, sl_price
            )
            
            if open_tp_data and open_tp_data[0] is not None:
                open_tp_price, open_tp_rr, open_tp_type = open_tp_data
            else:
                open_tp_price, open_tp_rr, open_tp_type = None, 0, None
    
            # 6. CALCULATE INDUCEMENT COUNT (for hammer only)
            inducement_count = 0
            if trigger_type == 'hammer' and criteria in ['FVG+SMT', 'SD+SMT']:
                # Get formation time and second swing
                formation_time = fvg_idea.get('formation_time') if criteria == 'FVG+SMT' else zone.get('formation_time')
                smt_swings_dict = smt_data.get('swings', {})
                
                # Convert dictionary to list and sort by time
                swings_list = []
                for key, swing_info in smt_swings_dict.items():
                    if isinstance(swing_info, dict) and 'time' in swing_info:
                        swings_list.append({
                            'time': swing_info['time'],
                            'price': swing_info.get('price', 0),
                            'type': swing_info.get('type', 'unknown')
                        })
                
                # Sort by time
                swings_list.sort(key=lambda x: x['time'])
                
                # Get second swing time if available
                second_swing_time = swings_list[1]['time'] if len(swings_list) > 1 else None
                
                if formation_time and second_swing_time:
                    zone_timeframe = tf
                    inducement_count = self.calculate_inducement(
                        instrument, direction, trigger_data.get('fib_zones', []),
                        formation_time, second_swing_time, zone_timeframe
                    )
    
            # 7. FETCH INDICATORS & FEATURES
            df_ind = fetch_candles(instrument, tf, count=150, api_key=self.credentials['oanda_api_key'])
            if not df_ind.empty:
                # Find index for the signal candle
                candle_index = next((i for i, t in enumerate(df_ind['time']) if t == candle['time']), -2)
                indicators = self.calculate_simple_indicators(df_ind, candle_index)
                advanced_features = self.calculate_advanced_features(df_ind, candle_index)
            else:
                indicators = {'rsi': 50, 'vwap': current_price}
                advanced_features = {}
    
            timeframe_data = self.fetch_all_timeframe_data(instrument)
            higher_tf_features = self.calculate_higher_tf_features(instrument, current_price, candle['time'], timeframe_data)
            zebra_features = self.calculate_zebra_features(instrument, candle['time'], timeframe_data)
    
            # === ENTRY COUNT FEATURE ===
            with self.entry_counter_lock:
                if signal_id not in self.entry_counter:
                    self.entry_counter[signal_id] = 0
                self.entry_counter[signal_id] += 1
                entry_count = self.entry_counter[signal_id]
            
            self.logger.info(f"📊 Entry #{entry_count} for signal {signal_id}")
    
            # 8. BUILD TRADE DATA
            trade_id = self._generate_trade_id(instrument, tf) if hasattr(self, '_generate_trade_id') else f"T_{int(current_time.timestamp())}"
            
            # Initialize webhook_sent early
            webhook_sent = 0
            
            # NOW create trade_data dictionary
            trade_data = {
                'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                'signal_id': signal_id,
                'trade_id': trade_id,
                'entry_count': entry_count,
                'instrument': instrument,
                'hammer_timeframe': tf,
                'direction': direction.upper(),
                'entry_time': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                'entry_price': round(current_price, 5),
                'sl_price': round(sl_price, 5),
                'tp_1_4_price': round(tp_1_4_price, 5) if tp_1_4_price is not None else '',
                'open_tp_price': round(open_tp_price, 5) if open_tp_price is not None else '',
                'open_tp_rr': round(open_tp_rr, 2) if open_tp_rr else 0,
                'sl_distance_pips': round(sl_distance_pips, 1),
                'signal_latency_seconds': round(signal_latency_seconds, 2),
                'hammer_volume': int(candle.get('volume', 0)),
                'inducement_count': inducement_count,
                'trigger_type': trigger_type,
                
                # NEW QUARTER FEATURES
                'current_quarter_monthly': quarter_features.get('current_quarter_monthly', ''),
                'true_open_monthly': quarter_features.get('true_open_monthly', ''),
                'true_open_relation_monthly': quarter_features.get('true_open_relation_monthly', ''),
                
                'current_quarter_weekly': quarter_features.get('current_quarter_weekly', ''),
                'true_open_weekly': quarter_features.get('true_open_weekly', ''),
                'true_open_relation_weekly': quarter_features.get('true_open_relation_weekly', ''),
                
                'current_quarter_daily': quarter_features.get('current_quarter_daily', ''),
                'true_open_daily': quarter_features.get('true_open_daily', ''),
                'true_open_relation_daily': quarter_features.get('true_open_relation_daily', ''),
                
                'current_quarter_90min': quarter_features.get('current_quarter_90min', ''),
                'true_open_90min': quarter_features.get('true_open_90min', ''),
                'true_open_relation_90min': quarter_features.get('true_open_relation_90min', ''),
                
                # Continue with rest of fields...
                'trigger_criteria': trigger_criteria,
                'trigger_timeframe': trigger_data.get('trigger_timeframe', ''),
                'fvg_formation_time': fvg_idea.get('formation_time', '').strftime('%Y-%m-%d %H:%M:%S') if fvg_idea.get('formation_time') else '',
                'sd_formation_time': zone.get('formation_time', '').strftime('%Y-%m-%d %H:%M:%S') if zone.get('formation_time') else '',
                'crt_formation_time': crt_signal.get('timestamp', '').strftime('%Y-%m-%d %H:%M:%S') if crt_signal and crt_signal.get('timestamp') else '',
                'smt_cycle': smt_data.get('cycle', ''),
                'smt_quarters': smt_data.get('quarters', ''),
                'has_psp': 1 if has_psp else 0,
                'is_hp_fvg': 1 if is_hp_fvg else 0,
                'is_hp_zone': 1 if is_hp_zone else 0,
                'rsi': indicators.get('rsi', 50),
                'vwap': indicators.get('vwap', current_price),
                'exit_time': '',
                'time_to_exit_seconds': 0,
                'tp_level_hit': 0,
                'webhook_sent': 0,
                'news_context_json': safe_news_data.get('news_context_json', ''),
                'news_high_count': safe_news_data.get('news_high_count', 0),
                'news_medium_count': safe_news_data.get('news_medium_count', 0),
                'news_low_count': safe_news_data.get('news_low_count', 0),
                'next_news_time': safe_news_data.get('next_news_time', ''),
                'next_news_event': safe_news_data.get('next_news_event', ''),
                'next_news_currency': safe_news_data.get('next_news_currency', ''),
                'prev_news_time': safe_news_data.get('prev_news_time', ''),
                'prev_news_event': safe_news_data.get('prev_news_event', ''),
                'prev_news_currency': safe_news_data.get('prev_news_currency', ''),
                'seconds_to_next_news': safe_news_data.get('seconds_to_next_news', 0),
                'seconds_since_last_news': safe_news_data.get('seconds_since_last_news', 0),
                'news_timing_category': safe_news_data.get('news_timing_category', ''),
                'news_fetch_status': safe_news_data.get('news_fetch_status', ''),
            }
            
            # Add TP prices and distances
            for i in range(1, 11):
                trade_data[f'tp_1_{i}_price'] = round(tp_prices[i], 5)
            trade_data.update(tp_distances)
            
            # Add all features
            trade_data.update(advanced_features)
            trade_data.update(higher_tf_features)
            trade_data.update(zebra_features)
            self.save_trade_to_csv(trade_data)
            self.tp_monitor.start_monitoring(trade_data)
    
            # 9. MODIFIED WEBHOOK SECTION
            # ===== EXTRACT ML FEATURES DIRECTLY FROM TRADE_DATA =====
            webhook_approved = False
            
            # Zebra trades never use ML filtering
            if trigger_type == 'zebra':
                self.logger.info(f"🦓 Zebra trade {signal_id} - auto-approved")
                webhook_approved = True  # Zebra trades are auto-approved
            elif trigger_type == 'hammer':
                # Only hammer trades use ML filtering (if available)
                if hasattr(self, 'signal_processor') and self.signal_processor:
                    try:
                        # ===== EXTRACT EXACT FEATURES FROM TRADE_DATA =====
                        ml_features = {
                            'instrument': trade_data.get('instrument', ''),
                            'hammer_timeframe': trade_data.get('hammer_timeframe', ''),
                            'direction': trade_data.get('direction', '').lower(),
                            'criteria': trade_data.get('criteria', ''),
                            'trigger_timeframe': trade_data.get('trigger_timeframe', ''),
                            'smt_cycle': trade_data.get('smt_cycle', ''),
                            'smt_quarters': trade_data.get('smt_quarters', ''),
                            'H4_open_rel': trade_data.get('H4_open_rel', ''),  # Direct from trade_data
                            'H6_open_rel': trade_data.get('H6_open_rel', ''),  # Direct from trade_data
                            'D_open_rel': trade_data.get('D_open_rel', ''),    # Direct from trade_data
                            'W_open_rel': trade_data.get('W_open_rel', ''),    # Direct from trade_data
                            '1m_zebra': trade_data.get('1m_zebra', ''),
                            '3m_zebra': trade_data.get('3m_zebra', ''),
                            '5m_zebra': trade_data.get('5m_zebra', ''),
                            '15m_zebra': trade_data.get('15m_zebra', ''),
                            'h1_zebra': trade_data.get('h1_zebra', ''),
                            'h4_zebra': trade_data.get('h4_zebra', ''),
                            'h6_zebra': trade_data.get('h6_zebra', ''),
                            'd_zebra': trade_data.get('d_zebra', ''),
                            'sl_distance_pips': trade_data.get('sl_distance_pips', 0),
                            'rsi': trade_data.get('rsi', 50),
                            'vwap': trade_data.get('vwap', current_price),
                            'news_high_count': trade_data.get('news_high_count', 0),
                            'news_medium_count': trade_data.get('news_medium_count', 0),
                            'news_low_count': trade_data.get('news_low_count', 0),
                            'entry_count': trade_data.get('entry_count', 1)
                        }
                        
                        # Log what we're sending to ML
                        self.logger.info(f"📊 Sending ML features for {signal_id}:")
                        for key, value in ml_features.items():
                            if value:  # Only log non-empty/non-zero values
                                self.logger.info(f"   {key}: {value}")
                        
                        # Check with ML filter if we should trade
                        # The signal_processor should return 1 for trade, 0 for skip
                        should_trade = self.signal_processor.check_with_features(
                            signal_id=signal_id,
                            features=ml_features
                        )
                        
                        if not should_trade:
                            self.logger.info(f"⏸️ ML rejected hammer on {tf} for signal {signal_id}")
                            return False
                        else:
                            webhook_approved = True
                            self.logger.info(f"✅ ML approved hammer on {tf}")
                            
                    except Exception as ml_error:
                        self.logger.error(f"❌ ML filter error: {str(ml_error)}")
                        webhook_approved = True  # Default to approve if ML fails
                        
                else:
                    self.logger.warning(f"⚠️ No signal_processor for hammer {signal_id}")
                    webhook_approved = True  # Approve without ML
            else:
                self.logger.error(f"❌ Unknown trigger_type: {trigger_type}")
                return False
            
            # ===== WEBHOOK SECTION =====
            webhook_sent = 0
    
            if trigger_type == 'hammer' and webhook_approved:
                try:
                    target_tp_level = 2  # Always use TP2
                    
                    webhook_sent = self.send_webhook_signal(
                        instrument=instrument, 
                        direction=direction, 
                        entry_price=current_price,
                        sl_price=sl_price, 
                        tp_price=tp_prices[target_tp_level],
                        signal_id=signal_id, 
                        trade_id=trade_id,
                        timeframe=tf,
                        criteria=criteria, 
                        risk_usd=50.0
                    )
                    webhook_sent = 1 if webhook_sent else 0
                    
                except Exception as webhook_err:
                    self.logger.error(f"⚠️ Webhook error: {webhook_err}")
                    webhook_sent = 0
            elif trigger_type == 'hammer' and not webhook_approved:
                self.logger.info(f"⏸️ Skipping webhook - ML prediction was not approved")
            elif trigger_type == 'zebra':
                self.logger.info(f"🦓 Zebra trade - no webhook sent (manual execution)")
                webhook_sent = 0
    
            # ===== FINALIZE =====
            trade_data['webhook_sent'] = webhook_sent
            trade_data['ai_node'] = 0  # Always 0 as requested
    
            self.logger.info(f"🔄 {trigger_type.upper()} trade_data created: {signal_id}")
            
            self.send_hammer_signal(trade_data, trigger_data)
            
            
            return True
    
        except Exception as e:
            self.logger.error(f"❌ Error processing {trigger_type} signal: {str(e)}", exc_info=True)
            return False
    
    def calculate_simple_indicators(self, df, candle_index):
        """Calculate only essential indicators - OPTIMIZED VERSION"""
        try:
            if len(df) < 14 or candle_index < 13:
                return {'rsi': 50, 'macd_line': 0, 'vwap': df.iloc[candle_index]['close'] if not df.empty else 0}
            
            # Convert to numpy for speed
            close_np = df['close'].values[:candle_index+1]
            high_np = df['high'].values
            low_np = df['low'].values
            volume_np = df['volume'].values
            current_close = close_np[-1]
            
            # RSI (14) - Fixed calculation
            if len(close_np) >= 14:
                # Calculate price changes
                deltas = np.diff(close_np)
                
                # Separate gains and losses
                gains = np.where(deltas > 0, deltas, 0)
                losses = np.where(deltas < 0, -deltas, 0)
                
                # Initial SMA (first 14 periods)
                avg_gain = np.mean(gains[:14])
                avg_loss = np.mean(losses[:14])
                
                # Wilder's Smoothing (RSI uses SMMA)
                for i in range(14, len(deltas)):
                    avg_gain = (avg_gain * 13 + gains[i]) / 14
                    avg_loss = (avg_loss * 13 + losses[i]) / 14
                
                # Calculate RS and RSI
                if avg_loss != 0:
                    rs = avg_gain / avg_loss
                    rsi_value = 100 - (100 / (1 + rs))
                else:
                    rsi_value = 100  # No losses means RSI = 100
            else:
                rsi_value = 50
            
            # MACD Line only (12, 26) - Fixed calculation
            if len(close_np) >= 26:
                # EMA calculation function with proper initialization
                def calculate_ema(data, period):
                    ema = np.zeros_like(data)
                    # Start with SMA for first period
                    ema[period-1] = np.mean(data[:period])
                    
                    # Calculate multiplier
                    multiplier = 2.0 / (period + 1)
                    
                    # Calculate EMA for remaining periods
                    for i in range(period, len(data)):
                        ema[i] = (data[i] - ema[i-1]) * multiplier + ema[i-1]
                    return ema
                
                # Calculate both EMAs
                ema_12 = calculate_ema(close_np, 12)
                ema_26 = calculate_ema(close_np, 26)
                
                # MACD line is difference between them
                macd_line = ema_12 - ema_26
                macd_value = macd_line[-1]
            else:
                macd_value = 0
            
            # Simple VWAP (20-period) - Keep original but fix slice
            if len(df) >= 20:
                # Ensure we don't go out of bounds
                start_idx = max(0, candle_index - 19)
                end_idx = candle_index + 1
                
                # Use numpy slicing for speed
                high_slice = high_np[start_idx:end_idx]
                low_slice = low_np[start_idx:end_idx]
                close_slice = close_np[start_idx:end_idx]
                volume_slice = volume_np[start_idx:end_idx]
                
                typical_price = (high_slice + low_slice + close_slice) / 3
                vwap = np.sum(typical_price * volume_slice) / np.sum(volume_slice)
            else:
                vwap = current_close
            
            return {
                'rsi': round(float(rsi_value), 2),
                'macd_line': round(float(macd_value), 6),
                'vwap': round(float(vwap), 5)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {str(e)}")
            return {'rsi': 50, 'macd_line': 0, 'vwap': 0}

    # def calculate_higher_tf_features(self, instrument, hammer_close_price, hammer_time, timeframe_data=None):
    #     """Calculate higher timeframe features using vectorization - OPTIMIZED VERSION"""
    #     try:
    #         features = {}
    #         higher_tfs = ['H4', 'H6', 'D', 'W']
            
    #         for tf in higher_tfs:
    #             # Use pre-fetched data
    #             if timeframe_data and tf in timeframe_data:
    #                 df = timeframe_data[tf]
    #             else:
    #                 # Fallback (keep original)
    #                 df = fetch_candles(instrument, tf, count=100, 
    #                                   api_key=self.credentials['oanda_api_key'])
                
    #             if df.empty:
    #                 self.logger.warning(f"⚠️ No {tf} data for {instrument}")
    #                 # Set default values for all features
    #                 features[f'{tf}_fib_zone'] = 0
    #                 features[f'{tf}_fib_percent'] = 0.0
    #                 features[f'{tf}_open_rel'] = 'unknown'
    #                 features[f'{tf}_quarter'] = 0
    #                 features[f'{tf}_candle_percent'] = 50.0
    #                 continue
                
    #             # Timezone handling (same as original)
    #             if isinstance(hammer_time, str):
    #                 from datetime import datetime
    #                 hammer_time = datetime.strptime(hammer_time, '%Y-%m-%d %H:%M:%S')
                
    #             if df['time'].dt.tz is None:
    #                 df['time'] = df['time'].dt.tz_localize('UTC').dt.tz_convert(NY_TZ)
                
    #             if hammer_time.tzinfo is None:
    #                 hammer_time = hammer_time.replace(tzinfo=NY_TZ)
                
    #             # Find the higher TF candle (optimized)
    #             htf_candle = None
    #             times = df['time'].values
    #             for idx in range(len(df)):
    #                 candle_open = times[idx]
                    
    #                 # Calculate candle close time
    #                 from datetime import timedelta
    #                 if tf == 'H4':
    #                     candle_close = candle_open + timedelta(hours=4)
    #                 elif tf == 'H6':
    #                     candle_close = candle_open + timedelta(hours=6)
    #                 elif tf == 'D':
    #                     candle_close = candle_open + timedelta(days=1)
    #                 elif tf == 'W':
    #                     candle_close = candle_open + timedelta(weeks=1)
    #                 else:
    #                     candle_close = candle_open + timedelta(hours=4)
                    
    #                 if candle_open <= hammer_time < candle_close:
    #                     htf_candle = df.iloc[idx]
    #                     break
                
    #             if htf_candle is None:
    #                 htf_candle = df.iloc[-1]
                
    #             # Vectorized calculations
    #             fib_zone, fib_percent = _calculate_fib_zone_vectorized(
    #                 df['high'].values, 
    #                 df['low'].values, 
    #                 hammer_close_price
    #             )
    #             features[f'{tf}_fib_zone'] = fib_zone
    #             features[f'{tf}_fib_percent'] = round(fib_percent, 2)
                
    #             # Open relation (same logic)
    #             if hammer_close_price > htf_candle['open']:
    #                 features[f'{tf}_open_rel'] = 'up'
    #             elif hammer_close_price < htf_candle['open']:
    #                 features[f'{tf}_open_rel'] = 'down'
    #             else:
    #                 features[f'{tf}_open_rel'] = 'equal'
                
    #             # Vectorized quarter calculation
    #             quarter = _calculate_candle_quarter_vectorized(
    #                 htf_candle['high'], 
    #                 htf_candle['low'], 
    #                 htf_candle['open'], 
    #                 hammer_close_price
    #             )
    #             features[f'{tf}_quarter'] = quarter
                
    #             # Vectorized position percentage
    #             candle_percent = _calculate_candle_position_percent_vectorized(
    #                 htf_candle['high'], 
    #                 htf_candle['low'], 
    #                 hammer_close_price
    #             )
    #             features[f'{tf}_candle_percent'] = round(candle_percent, 1)
                
    #             # Log for debugging (same as original)
    #             self.logger.info(f"📊 {tf} Features for {instrument}:")
    #             self.logger.info(f"   Fib Zone: {fib_zone} ({fib_percent:.1f}%)")
    #             self.logger.info(f"   Open Rel: {features[f'{tf}_open_rel']}")
    #             self.logger.info(f"   Quarter: {quarter}")
    #             self.logger.info(f"   Candle %: {candle_percent:.1f}%")
            
    #         return features
            
    #     except Exception as e:
    #         self.logger.error(f"❌ Error calculating higher TF features: {str(e)}", exc_info=True)
    #         # Return empty features with all required keys (same as original)
    #         features = {}
    #         for tf in ['H4', 'H6', 'D', 'W']:
    #             features[f'{tf}_fib_zone'] = 0
    #             features[f'{tf}_fib_percent'] = 0.0
    #             features[f'{tf}_open_rel'] = 'error'
    #             features[f'{tf}_quarter'] = 0
    #             features[f'{tf}_candle_percent'] = 50.0
    #         return features
            
    
    def _calculate_fib_zone(self, df, current_price, timeframe):
            """Original method - keep for compatibility"""
            try:
                if df.empty or len(df) < 10:
                    return 0, 0.0
                
                highest = df['high'].max()
                lowest = df['low'].min()
                total_range = highest - lowest
                
                if total_range <= 0:
                    return 0, 0.0
                
                zone_size = total_range / 10
                distance_from_top = highest - current_price
                zone = int(distance_from_top // zone_size) + 1
                
                zone = max(1, min(10, zone))
                
                percent_from_top = (distance_from_top / total_range) * 100 if total_range > 0 else 0.0
                
                return zone, round(percent_from_top, 2)
                
            except Exception as e:
                self.logger.error(f"Error calculating Fib zone for {timeframe}: {str(e)}")
                return 0, 0.0
        
    def _calculate_candle_quarter(self, candle, current_price):
        """Original method - keep for compatibility"""
        try:
            candle_high = candle['high']
            candle_low = candle['low']
            candle_range = candle_high - candle_low
            
            if candle_range <= 0:
                return 0
            
            quarter_size = candle_range / 4
            
            distance_from_bottom = current_price - candle_low
            
            if distance_from_bottom <= 0:
                return 1
            elif distance_from_bottom >= candle_range:
                return 4
            
            quarter = int(distance_from_bottom // quarter_size) + 1
            
            quarter = max(1, min(4, quarter))
            
            return quarter
            
        except Exception as e:
            self.logger.error(f"Error calculating candle quarter: {str(e)}")
            return 0
    
    def _calculate_candle_position_percent(self, candle, current_price):
        """Original method - keep for compatibility"""
        try:
            candle_high = candle['high']
            candle_low = candle['low']
            candle_range = candle_high - candle_low
            
            if candle_range <= 0:
                return 50.0
            
            position = ((current_price - candle_low) / candle_range) * 100
            position = max(0.0, min(100.0, position))
            
            return round(position, 1)
            
        except Exception as e:
            self.logger.error(f"Error calculating candle position %: {str(e)}")
            return 50.0

    def calculate_higher_tf_features(self, instrument, hammer_close_price, hammer_time, timeframe_data=None):
        """Calculate higher timeframe features using pre-fetched data - FIXED VERSION"""
        try:
            features = {}
            higher_tfs = ['H4', 'H6', 'D', 'W']
            
            for tf in higher_tfs:
                # Use pre-fetched data
                if timeframe_data and tf in timeframe_data:
                    df = timeframe_data[tf]
                else:
                    # Fallback
                    df = fetch_candles(instrument, tf, count=100, 
                                      api_key=self.credentials['oanda_api_key'])
                
                if df.empty:
                    self.logger.warning(f"⚠️ No {tf} data for {instrument}")
                    # Set default values for all features
                    features[f'{tf}_fib_zone'] = 0
                    features[f'{tf}_fib_percent'] = 0.0
                    features[f'{tf}_open_rel'] = 'unknown'
                    features[f'{tf}_quarter'] = 0
                    features[f'{tf}_candle_percent'] = 50.0
                    continue
                
                # Convert hammer_time to match dataframe timezone
                if isinstance(hammer_time, str):
                    hammer_time = datetime.strptime(hammer_time, '%Y-%m-%d %H:%M:%S')
                
                # Ensure both are timezone aware
                if df['time'].dt.tz is None:
                    df['time'] = df['time'].dt.tz_localize('UTC').dt.tz_convert(NY_TZ)
                
                if hammer_time.tzinfo is None:
                    hammer_time = hammer_time.replace(tzinfo=NY_TZ)
                
                # Find the higher TF candle that contains the hammer time
                htf_candle = None
                
                for idx in range(len(df)):
                    candle = df.iloc[idx]
                    candle_open = candle['time']
                    
                    # Calculate candle close time based on timeframe
                    if tf == 'H4':
                        candle_close = candle_open + timedelta(hours=4)
                    elif tf == 'H6':
                        candle_close = candle_open + timedelta(hours=6)
                    elif tf == 'D':
                        candle_close = candle_open + timedelta(days=1)
                    elif tf == 'W':
                        candle_close = candle_open + timedelta(weeks=1)
                    else:
                        candle_close = candle_open + timedelta(hours=4)  # default
                    
                    if candle_open <= hammer_time < candle_close:
                        htf_candle = candle
                        break
                
                # If no candle found, use the last candle
                if htf_candle is None:
                    htf_candle = df.iloc[-1]
                
                # ============================================
                # 1. Pd-(tf): Fibonacci Zone (10 zones)
                # ============================================
                fib_zone, fib_percent = self._calculate_fib_zone(df, hammer_close_price, tf)
                features[f'{tf}_fib_zone'] = fib_zone
                features[f'{tf}_fib_percent'] = fib_percent
                
                # ============================================
                # 2. Price Relative to Candle Open
                # ============================================
                if hammer_close_price > htf_candle['open']:
                    features[f'{tf}_open_rel'] = 'up'
                elif hammer_close_price < htf_candle['open']:
                    features[f'{tf}_open_rel'] = 'down'
                else:
                    features[f'{tf}_open_rel'] = 'equal'
                
                # ============================================
                # 3. Candle Quarter Position
                # ============================================
                quarter = self._calculate_candle_quarter(htf_candle, hammer_close_price)
                features[f'{tf}_quarter'] = quarter
                
                # ============================================
                # 4. Price Position Percentage within Candle
                # ============================================
                candle_percent = self._calculate_candle_position_percent(htf_candle, hammer_close_price)
                features[f'{tf}_candle_percent'] = candle_percent
                
                # Log for debugging
                self.logger.info(f"📊 {tf} Features for {instrument}:")
                self.logger.info(f"   Fib Zone: {fib_zone} ({fib_percent:.1f}%)")
                self.logger.info(f"   Open Rel: {features[f'{tf}_open_rel']}")
                self.logger.info(f"   Quarter: {quarter}")
                self.logger.info(f"   Candle %: {candle_percent:.1f}%")
            
            return features
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating higher TF features: {str(e)}", exc_info=True)
            # Return empty features with all required keys
            features = {}
            for tf in ['H4', 'H6', 'D', 'W']:
                features[f'{tf}_fib_zone'] = 0
                features[f'{tf}_fib_percent'] = 0.0
                features[f'{tf}_open_rel'] = 'error'
                features[f'{tf}_quarter'] = 0
                features[f'{tf}_candle_percent'] = 50.0
            return features
    
    # ============================================
    # OPTIMIZED ADVANCED FEATURES (SAME INTERFACE)
    # ============================================
    
    @jit(nopython=True, parallel=True, cache=True)
    def _calculate_moving_averages_vectorized(prices, windows):
        """Vectorized moving averages for multiple windows"""
        n = len(prices)
        m = len(windows)
        results = np.zeros((n, m))
        
        for j in prange(m):
            window = windows[j]
            # Cumulative sum for fast moving average
            cumsum = np.zeros(n + 1)
            cumsum[1:] = np.cumsum(prices)
            
            for i in range(n):
                if i >= window - 1:
                    results[i, j] = (cumsum[i+1] - cumsum[i+1-window]) / window
                else:
                    results[i, j] = np.mean(prices[:i+1])
        
        return results
    
    @jit(nopython=True, cache=True)
    def _calculate_vwap_vectorized(high, low, close, volume):
        """Vectorized VWAP calculation"""
        n = len(close)
        typical_price = (high + low + close) / 3
        
        # Cumulative sums
        cum_tp_volume = np.cumsum(typical_price * volume)
        cum_volume = np.cumsum(volume)
        
        vwap = np.zeros(n)
        for i in range(n):
            if cum_volume[i] > 0:
                vwap[i] = cum_tp_volume[i] / cum_volume[i]
            else:
                vwap[i] = typical_price[i]
        
        return vwap
    
    def calculate_advanced_features(self, df, candle_index):
        """Calculate advanced features for the hammer candle - OPTIMIZED & FIXED"""
        try:
            if df.empty or len(df) < 100:
                return {}
            
            # Create cache key
            cache_key = f"advanced_{df.iloc[candle_index]['time'] if candle_index < len(df) else 'unknown'}"
            
            # Check cache
            if hasattr(self, '_advanced_features_cache'):
                cached = self._advanced_features_cache.get(cache_key)
                if cached:
                    cache_age = (datetime.now() - cached['timestamp']).seconds
                    if cache_age < 30:  # Cache for 30 seconds
                        return cached['features']
            
            # Make a copy to avoid modifying original
            df_features = df.copy()
            
            # Ensure we have necessary columns
            if 'adj close' not in df_features.columns:
                df_features['adj close'] = df_features['close']
            
            # OPTIMIZATION: Calculate all moving averages at once
            ma_lengths = [10, 20, 30, 40, 60, 100]
            for length in ma_lengths:
                df_features[f'ma_{length}'] = df_features['adj close'].rolling(window=length, min_periods=1).mean()
            
            # VWAP calculation
            try:
                typical_price = (df_features['high'] + df_features['low'] + df_features['close']) / 3
                vwap_num = (typical_price * df_features['volume']).cumsum()
                vwap_den = df_features['volume'].cumsum()
                df_features['vwap_value'] = vwap_num / vwap_den
                df_features['vwap_std'] = df_features['vwap_value'].rolling(window=20, min_periods=1).std()
            except:
                df_features['vwap_value'] = df_features['adj close']
                df_features['vwap_std'] = 0
            
            # VWAP bands
            try:
                for i in range(1, 4):
                    df_features[f'upper_band_{i}'] = df_features['vwap_value'] + i * df_features['vwap_std']
                    df_features[f'lower_band_{i}'] = df_features['vwap_value'] - i * df_features['vwap_std']
            except:
                for i in range(1, 4):
                    df_features[f'upper_band_{i}'] = df_features['vwap_value']
                    df_features[f'lower_band_{i}'] = df_features['vwap_value']
            
            # Initialize features dictionary
            features = {}
            
            # Get hammer candle values
            try:
                hammer_low = df_features['low'].iloc[candle_index]
                hammer_high = df_features['high'].iloc[candle_index]
                hammer_close = df_features['close'].iloc[candle_index]
                
                # Touch indicators (CORRECTED - get single values)
                for i in range(1, 4):
                    upper_band_val = df_features[f'upper_band_{i}'].iloc[candle_index]
                    lower_band_val = df_features[f'lower_band_{i}'].iloc[candle_index]
                    
                    # CORRECT: Store single value, not entire series
                    features[f'touches_upper_band_{i}'] = int(hammer_low <= upper_band_val <= hammer_high)
                    features[f'touches_lower_band_{i}'] = int(hammer_low <= lower_band_val <= hammer_high)
                
                vwap_val = df_features['vwap_value'].iloc[candle_index]
                features['touches_vwap'] = int(hammer_low <= vwap_val <= hammer_high)
                
                # Distance ratios (CORRECTED - get single values)
                vwap_std_val = df_features['vwap_std'].iloc[candle_index] or 1e-6
                
                for i in range(1, 4):
                    upper_band_val = df_features[f'upper_band_{i}'].iloc[candle_index]
                    lower_band_val = df_features[f'lower_band_{i}'].iloc[candle_index]
                    
                    upper_dist = abs(hammer_close - upper_band_val)
                    lower_dist = abs(hammer_close - lower_band_val)
                    
                    # CORRECT: Store single value, not entire series
                    features[f'far_ratio_upper_band_{i}'] = float(upper_dist / vwap_std_val)
                    features[f'far_ratio_lower_band_{i}'] = float(lower_dist / vwap_std_val)
                
                vwap_dist = abs(hammer_close - vwap_val)
                features['far_ratio_vwap'] = float(vwap_dist / vwap_std_val)
                
            except Exception as e:
                # Set defaults if calculation fails
                for i in range(1, 4):
                    features[f'touches_upper_band_{i}'] = 0
                    features[f'touches_lower_band_{i}'] = 0
                    features[f'far_ratio_upper_band_{i}'] = 0.0
                    features[f'far_ratio_lower_band_{i}'] = 0.0
                features['touches_vwap'] = 0
                features['far_ratio_vwap'] = 0.0
            
            # Bearish stack
            try:
                ma_20 = df_features['ma_20'].iloc[candle_index]
                ma_30 = df_features['ma_30'].iloc[candle_index]
                ma_40 = df_features['ma_40'].iloc[candle_index]
                ma_60 = df_features['ma_60'].iloc[candle_index]
                
                bearish = (ma_20 < ma_30) and (ma_30 < ma_40) and (ma_40 < ma_60)
                features['bearish_stack'] = int(bearish)
            except:
                features['bearish_stack'] = 0
            
            # Trend strength
            try:
                trend_up = (ma_20 > ma_30) and (ma_30 > ma_40) and (ma_40 > ma_60)
                trend_down = (ma_20 < ma_30) and (ma_30 < ma_40) and (ma_40 < ma_60)
                
                features['trend_strength_up'] = int(trend_up)
                features['trend_strength_down'] = int(trend_down)
            except:
                features['trend_strength_up'] = 0
                features['trend_strength_down'] = 0
            
            # Previous volume
            try:
                if candle_index > 0:
                    features['prev_volume'] = float(df_features['volume'].iloc[candle_index - 1])
                else:
                    features['prev_volume'] = float(df_features['volume'].iloc[candle_index])
            except:
                features['prev_volume'] = 0.0
            
            # Extract other values (CORRECTED - use .iloc[candle_index] for single value)
            try:
                features['vwap_value'] = float(df_features['vwap_value'].iloc[candle_index])
                features['vwap_std'] = float(df_features['vwap_std'].iloc[candle_index])
                
                for length in ma_lengths:
                    features[f'ma_{length}'] = float(df_features[f'ma_{length}'].iloc[candle_index])
                
                for i in range(1, 4):
                    features[f'upper_band_{i}'] = float(df_features[f'upper_band_{i}'].iloc[candle_index])
                    features[f'lower_band_{i}'] = float(df_features[f'lower_band_{i}'].iloc[candle_index])
                    
            except Exception as e:
                self.logger.error(f"❌ Error extracting values: {str(e)}")
                # Set defaults
                features['vwap_value'] = 0.0
                features['vwap_std'] = 0.0
                for length in ma_lengths:
                    features[f'ma_{length}'] = 0.0
                for i in range(1, 4):
                    features[f'upper_band_{i}'] = 0.0
                    features[f'lower_band_{i}'] = 0.0
            
            # Store in cache
            if not hasattr(self, '_advanced_features_cache'):
                self._advanced_features_cache = {}
            self._advanced_features_cache[cache_key] = {
                'features': features,
                'timestamp': datetime.now()
            }
            
            return features
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating advanced features: {str(e)}")
            return {}
    
    @jit(nopython=True, cache=True)
    def _count_swing_extremes_vectorized(prices, direction, threshold):
        """Vectorized swing extremes counting"""
        n = len(prices)
        count = 0
        
        if direction == 'bearish':
            for i in range(1, n-1):
                if (prices[i] > prices[i-1] and 
                    prices[i] > prices[i+1] and 
                    prices[i] > threshold):
                    count += 1
        else:
            for i in range(1, n-1):
                if (prices[i] < prices[i-1] and 
                    prices[i] < prices[i+1] and 
                    prices[i] < threshold):
                    count += 1
        
        return count
    
    def calculate_inducement(self, instrument, direction, fib_zones, 
                             formation_time, second_swing_time, zone_timeframe):
        """Calculate inducement - OPTIMIZED VERSION"""
        try:
            if not fib_zones or len(fib_zones) < 3:
                return 0
            
            # Find the 0.5 Fibonacci zone
            fib_50_zone = None
            for zone in fib_zones:
                if abs(zone.get('ratio', 0) - 0.5) < 0.01:
                    fib_50_zone = zone
                    break
            
            if not fib_50_zone:
                return 0
            
            # Fetch data
            df = fetch_candles(instrument, zone_timeframe, count=500, 
                              api_key=self.credentials['oanda_api_key'])
            
            if df.empty:
                return 0
            
            # Filter to time range
            df_period = df[(df['time'] >= formation_time) & 
                          (df['time'] <= second_swing_time)]
            
            if df_period.empty:
                return 0
            
            # Vectorized counting
            if direction == 'bearish':
                threshold = fib_50_zone.get('high', 0)
                prices = df_period['high'].values
            else:
                threshold = fib_50_zone.get('low', float('inf'))
                prices = df_period['low'].values
            
            inducement_count = _count_swing_extremes_vectorized(prices, direction, threshold)
            
            self.logger.info(f"📊 Inducement count: {inducement_count} swing {'highs' if direction == 'bearish' else 'lows'}")
            return inducement_count
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating inducement: {str(e)}")
            return 0
    
    
    def calculate_open_tp(self, instrument, direction, entry_price, sl_price):
        """Calculate open TP - OPTIMIZED VERSION"""
        try:
            from datetime import datetime, timedelta
            
            # Fetch data
            df_daily = fetch_candles(instrument, 'D', count=5, 
                                    api_key=self.credentials['oanda_api_key'])
            df_weekly = fetch_candles(instrument, 'W', count=4,
                                     api_key=self.credentials['oanda_api_key'])
            df_monthly = fetch_candles(instrument, 'M', count=3,
                                      api_key=self.credentials['oanda_api_key'])
            
            open_tp_candidates = []
            
            # Vectorized processing for each timeframe
            for df, tf, lookback in [(df_daily, 'daily', 10), 
                                      (df_weekly, 'weekly', 5), 
                                      (df_monthly, 'monthly', 3)]:
                
                if df.empty or len(df) < 2:
                    continue
                
                if direction == 'bearish':
                    # Previous day low
                    prev_val = df.iloc[-2]['low']
                    recent_min = df['low'].tail(lookback).min()
                    
                    if prev_val < recent_min:
                        distance = abs(entry_price - prev_val)
                        open_tp_candidates.append({
                            'price': prev_val,
                            'type': f'{tf}_low',
                            'distance': distance
                        })
                else:
                    # Previous day high
                    prev_val = df.iloc[-2]['high']
                    recent_max = df['high'].tail(lookback).max()
                    
                    if prev_val > recent_max:
                        distance = abs(entry_price - prev_val)
                        open_tp_candidates.append({
                            'price': prev_val,
                            'type': f'{tf}_high',
                            'distance': distance
                        })
            
            if not open_tp_candidates:
                return None, None, None
            
            # Find closest using numpy
            import numpy as np
            distances = np.array([c['distance'] for c in open_tp_candidates])
            closest_idx = np.argmin(distances)
            closest = open_tp_candidates[closest_idx]
            
            # Calculate risk:reward
            risk = abs(entry_price - sl_price)
            reward = abs(entry_price - closest['price'])
            rr_ratio = round(reward / risk, 2) if risk > 0 else 0
            
            self.logger.info(f"📊 Open TP: {closest['type']} at {closest['price']:.5f}, RR: {rr_ratio}")
            
            return closest['price'], rr_ratio, closest['type']
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating open TP: {str(e)}")
            return None, None, None
    
    def save_trade_to_csv(self, trade_data):
        """Save trade data safely - APPEND only (no overwrite) - SUPER DEBUG VERSION"""
        try:
            # ============ CRITICAL DEBUG: CATCH THE ZEBRA BUG ============
            trade_id = trade_data.get('trade_id', 'UNKNOWN')
            criteria = trade_data.get('criteria', 'UNKNOWN')
            instrument = trade_data.get('instrument', 'UNKNOWN')
            
            self.logger.info(f"🔍 CSV SAVE START | Trade: {trade_id} | Criteria: {criteria} | Instrument: {instrument}")
            
            # ============ PHASE 1: CHECK FOR NONE KEYS ============
            self.logger.info(f"📊 Phase 1: Checking for None keys in trade_data...")
            
            none_keys = [k for k in trade_data.keys() if k is None]
            if none_keys:
                self.logger.error(f"🚨 BUG CONFIRMED: Found {len(none_keys)} None keys in trade_data!")
                self.logger.error(f"🚨 None keys list: {none_keys}")
                
                # Check what values these None keys have
                for k in none_keys:
                    value = trade_data[k]
                    self.logger.error(f"   - Key None has value: {value} (type: {type(value)})")
                
                # Where did these None keys come from? Let's trace...
                if criteria == 'zebra':
                    self.logger.error("🚨 THIS IS A ZEBRA TRADE - the bug only happens with zebra!")
                    
                    # Check if zebra features might be the culprit
                    self.logger.info("🔍 Checking zebra-specific keys in trade_data:")
                    zebra_keys = [k for k in trade_data.keys() if 'zebra' in str(k).lower()]
                    self.logger.info(f"   Zebra-related keys: {zebra_keys}")
                
                # Remove None keys to fix the CSV error
                self.logger.info("🛠️ Removing None keys from trade_data...")
                for k in none_keys:
                    removed_value = trade_data.pop(k)
                    self.logger.info(f"   ✅ Removed None key, value was: {removed_value}")
            
            # ============ PHASE 2: CHECK FOR NONE IN HEADERS ============
            self.logger.info(f"📊 Phase 2: Checking self.headers...")
            
            if not self.headers:
                self.logger.critical("🚨 CRITICAL: self.headers is None or empty!")
                return False
            
            none_in_headers = [i for i, h in enumerate(self.headers) if h is None]
            if none_in_headers:
                self.logger.error(f"🚨 Found None in self.headers at indices: {none_in_headers}")
                self.logger.error(f"🚨 self.headers: {self.headers}")
                
                # Show context around each None
                for idx in none_in_headers:
                    start = max(0, idx - 3)
                    end = min(len(self.headers), idx + 4)
                    self.logger.error(f"   Context around index {idx}: {self.headers[start:end]}")
                
                # Filter out None
                original_count = len(self.headers)
                self.headers = [h for h in self.headers if h is not None]
                self.logger.info(f"🛠️ Filtered headers: {original_count} → {len(self.headers)}")
            
            # ============ PHASE 3: CHECK FOR MISMATCHED KEYS ============
            self.logger.info(f"📊 Phase 3: Checking key mismatches...")
            
            headers_set = set(self.headers)
            trade_keys_set = set(trade_data.keys())
            
            # Keys in trade_data but not in headers
            extra_keys = trade_keys_set - headers_set
            if extra_keys:
                self.logger.warning(f"⚠️ Trade data has {len(extra_keys)} keys not in CSV headers:")
                for key in sorted(extra_keys):
                    value = trade_data.get(key)
                    self.logger.warning(f"   - '{key}' = {value} (type: {type(value)})")
                    
                    # Special check for zebra
                    if criteria == 'zebra' and 'zebra' in str(key).lower():
                        self.logger.error(f"🚨 ZEBRA BUG CLUE: Extra key '{key}' might be from zebra_features!")
            
            # Keys in headers but not in trade_data (will get empty values)
            missing_keys = headers_set - trade_keys_set
            if missing_keys:
                self.logger.info(f"📝 {len(missing_keys)} headers will get empty values")
                if len(missing_keys) < 10:  # Don't spam if there are many
                    for key in sorted(missing_keys):
                        self.logger.info(f"   - '{key}'")
            
            # ============ PHASE 4: LOG TRADE DATA STRUCTURE ============
            if criteria == 'zebra':
                self.logger.info(f"📊 Phase 4: Dumping zebra trade_data structure...")
                
                # Log all keys for zebra trades
                all_keys = list(trade_data.keys())
                self.logger.info(f"📋 Zebra trade_data has {len(all_keys)} keys total")
                
                # Group keys by category for easier debugging
                fvg_keys = [k for k in all_keys if 'fvg' in k.lower()]
                smt_keys = [k for k in all_keys if 'smt' in k.lower()]
                crt_keys = [k for k in all_keys if 'crt' in k.lower()]
                news_keys = [k for k in all_keys if 'news' in k.lower()]
                tp_keys = [k for k in all_keys if 'tp_' in k]
                feature_keys = [k for k in all_keys if k not in ['timestamp', 'signal_id', 'trade_id', 'instrument', 'criteria']]
                
                self.logger.info(f"   FVG keys ({len(fvg_keys)}): {fvg_keys}")
                self.logger.info(f"   SMT keys ({len(smt_keys)}): {smt_keys}")
                self.logger.info(f"   CRT keys ({len(crt_keys)}): {crt_keys}")
                self.logger.info(f"   News keys ({len(news_keys)}): {news_keys}")
                self.logger.info(f"   TP keys ({len(tp_keys)}): {tp_keys[:5]}...")  # Just first 5
            
            # ============ ORIGINAL SAVE LOGIC (with extra safety) ============
            self.logger.info(f"💾 Saving trade {trade_id} to CSV...")
            
            # Create new row with all headers
            new_row = {}
            for header in self.headers:
                try:
                    new_row[header] = trade_data.get(header, '')
                except Exception as e:
                    self.logger.error(f"❌ Error getting value for header '{header}': {e}")
                    new_row[header] = ''
            
            # Final safety check: ensure new_row has no None keys
            final_none_keys = [k for k in new_row.keys() if k is None]
            if final_none_keys:
                self.logger.critical(f"🚨 EMERGENCY: new_row STILL has None keys: {final_none_keys}")
                self.logger.critical("🛠️ Creating emergency clean row...")
                new_row = {k: v for k, v in new_row.items() if k is not None}
            
            # Write to CSV
            file_exists = os.path.exists(self.csv_file_path)
            
            with open(self.csv_file_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.headers)
                
                if not file_exists:
                    writer.writeheader()
                    self.logger.info(f"📝 Created new CSV file with headers")
                
                # Log sample of what we're writing
                sample_keys = ['instrument', 'entry_price', 'sl_price', 'criteria', 'signal_latency_seconds']
                sample_data = {k: new_row.get(k, 'N/A') for k in sample_keys}
                self.logger.info(f"📝 Writing row sample: {sample_data}")
                
                writer.writerow(new_row)
            
            self.logger.info(f"✅ SUCCESS: Saved trade {trade_id} to CSV")
            
            # ============ POST-SAVE VERIFICATION ============
            try:
                with open(self.csv_file_path, 'r', encoding='utf-8') as f:
                    line_count = sum(1 for _ in f)
                self.logger.info(f"📊 CSV now has {line_count} total lines")
                
                # Read last line to verify
                if line_count > 1:
                    with open(self.csv_file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        last_line = lines[-1].strip()
                        self.logger.info(f"📝 Last CSV line: {last_line[:100]}..." if len(last_line) > 100 else last_line)
            except Exception as e:
                self.logger.warning(f"⚠️ Couldn't verify CSV write: {e}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ CRITICAL ERROR saving to CSV: {str(e)}", exc_info=True)
            
            # ============ EMERGENCY ERROR LOGGING ============
            try:
                timestamp = int(time.time())
                error_log_path = f"{self.csv_file_path}.zebra_error_{timestamp}.log"
                
                self.logger.info(f"🆘 Creating emergency error log: {error_log_path}")
                
                with open(error_log_path, 'w', encoding='utf-8') as f:
                    f.write(f"=== ZEBRA CSV SAVE ERROR LOG ===\n")
                    f.write(f"Time: {datetime.now()}\n")
                    f.write(f"Trade ID: {trade_id}\n")
                    f.write(f"Criteria: {criteria}\n")
                    f.write(f"Instrument: {instrument}\n")
                    f.write(f"Error: {str(e)}\n\n")
                    
                    f.write("=== TRADE DATA KEYS ===\n")
                    for key in sorted(trade_data.keys()):
                        value = trade_data.get(key)
                        f.write(f"{key}: {value} (type: {type(value)})\n")
                    f.write("\n")
                    
                    f.write("=== HEADERS ===\n")
                    f.write(f"Count: {len(self.headers) if self.headers else 0}\n")
                    f.write(f"Headers: {self.headers}\n\n")
                    
                    f.write("=== TRACE FOR NONE KEYS ===\n")
                    if 'criteria' in trade_data and trade_data['criteria'] == 'zebra':
                        f.write("THIS IS A ZEBRA TRADE - check zebra_features() function!\n")
                    
                    # Check feature function returns
                    f.write("\n=== FEATURE FUNCTION CHECK ===\n")
                    f.write("Check these functions in _process_and_record_hammer:\n")
                    f.write("1. calculate_advanced_features()\n")
                    f.write("2. calculate_higher_tf_features()\n")
                    f.write("3. calculate_zebra_features()  <-- MOST LIKELY CULPRIT FOR ZEBRA!\n")
                    f.write("4. _get_safe_news_data()\n")
                
                self.logger.info(f"📦 Emergency log saved to: {error_log_path}")
                
            except Exception as log_err:
                self.logger.error(f"❌ Failed to save error log: {log_err}")
            
            # Try to save anyway with stripped data
            try:
                backup_path = f"{self.csv_file_path}.backup_{timestamp}.csv"
                self.logger.info(f"🆘 Attempting emergency backup to {backup_path}")
                
                # Create a clean version of trade_data
                clean_data = {k: v for k, v in trade_data.items() if k is not None}
                
                with open(backup_path, 'w', newline='', encoding='utf-8') as f:
                    if self.headers:
                        clean_headers = [h for h in self.headers if h is not None]
                        writer = csv.DictWriter(f, fieldnames=clean_headers)
                        writer.writeheader()
                        
                        row = {}
                        for header in clean_headers:
                            row[header] = clean_data.get(header, '')
                        
                        writer.writerow(row)
                
                self.logger.info(f"📦 Emergency backup saved to: {backup_path}")
                
            except Exception as backup_err:
                self.logger.error(f"❌ Emergency backup failed: {backup_err}")
            
            return False
        
    def send_hammer_signal(self, trade_data, trigger_data):
        """Send hammer signal to Telegram with price levels and risk management"""
        try:
            direction = trade_data['direction']
            instrument = trade_data['instrument']
            tf = trade_data['hammer_timeframe']
            criteria = trade_data['criteria']
            
            # Get humorous phrases
            trigger_humor = get_humorous_phrase(direction.lower(), criteria)
            hammer_humor = get_hammer_humor(direction.lower(), tf)
            
            # Build message with PRICE LEVELS and LOT SIZES
            message = f"🔨 *HAMMER ENTRY SIGNAL* 🔨\n\n"
            message += f"*Yoo bro! {trigger_humor}*\n"
            message += f"*{hammer_humor}*\n\n"
            
            message += f"*📊 CRITERIA:* {criteria}\n"
            message += f"*🎯 ENTRY:* {direction} {instrument} on {tf}\n"
            message += f"*💰 ENTRY PRICE:* {trade_data['entry_price']:.5f}\n\n"
            
            # PRICE LEVELS
            message += f"*🛑 STOP LOSS:*\n"
            message += f"  • Price: {trade_data['sl_price']:.5f}\n"
            message += f"  • Distance: {trade_data['sl_distance_pips']:.1f} pips\n\n"
            
            message += f"*🎯 TAKE PROFIT 1:4:*\n"
            message += f"  • Price: {trade_data['tp_1_4_price']:.5f}\n"
            message += f"  • Distance: {trade_data.get('tp_1_4_distance', 0):.1f} pips\n\n"
            
            # RISK MANAGEMENT - LOT SIZES
            message += f"*💰 RISK MANAGEMENT (MICRO LOTS):*\n"
            message += f"  • Risk $10: {trade_data['risk_10_lots']:.2f} lots\n"
            # message += f"  • Risk $100: {trade_data['risk_100_lots']:.2f} lots\n\n"
            
            message += f"*⏰ TIME:* {trade_data['entry_time']}\n"
            message += f"*⚡ LATENCY:* {trade_data.get('signal_latency_seconds', 0):.1f}s (candle close → signal)\n\n"
            
            # Add signal context
            if criteria == 'FVG+SMT':
                message += f"*📈 FVG+SMT Setup:*\n"
                message += f"• SMT Cycle: {trade_data['smt_cycle']}\n"
                message += f"• Has PSP: {'✅' if trade_data['has_psp'] else '❌'}\n"
                message += f"• HP FVG: {'✅' if trade_data['is_hp_fvg'] else '❌'}\n"
            elif criteria == 'SD+SMT':
                message += f"*📈 SD+SMT Setup:*\n"
                message += f"• SMT Cycle: {trade_data['smt_cycle']}\n"
                message += f"• Has PSP: {'✅' if trade_data['has_psp'] else '❌'}\n"
                message += f"• HP Zone: {'✅' if trade_data['is_hp_zone'] else '❌'}\n"
            elif criteria == 'CRT+SMT':
                message += f"*📈 CRT+SMT Setup:*\n"
                message += f"• SMT Cycle: {trade_data['smt_cycle']}\n"
                message += f"• Has PSP: {'✅' if trade_data['has_psp'] else '❌'}\n"
            
            message += f"\n*💡 TRADER NOTE:* Signal ID: {trade_data['signal_id']} (groups all hammers)\n"
            message += f"*🤙 REMEMBER:* Trade safe, bro! Manage your risk!\n\n"
            
            message += f"#{instrument.replace('_', '')} #{tf}Hammer #{direction}Signal #{criteria.replace('+', '')}"
            
            # Send to Telegram
            success = send_telegram(
                message,
                self.credentials['telegram_token'],
                self.credentials['telegram_chat_id']
            )
            
            if success:
                self.logger.info(f"📤 Signal sent: {instrument} {tf} {direction}")
                self.logger.info(f"📊 SL Price: {trade_data['sl_price']:.5f}, TP 1:4 Price: {trade_data['tp_1_4_price']:.5f}")
                self.logger.info(f"💰 Lot Sizes: ${trade_data['risk_10_lots']:.2f} (risk $10), ${trade_data['risk_100_lots']:.2f} (risk $100)")
            else:
                self.logger.error(f"❌ Failed to send signal")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error sending signal: {str(e)}")
            return False
    
    
    
    
    def on_signal_detected(self, signal_data):
        """Trigger hammer scanner when main signal detected"""
        try:
            thread = threading.Thread(
                target=self.scan_fibonacci_hammer,
                args=(signal_data,),
                name=f"HammerScan_{signal_data.get('instrument')}",
                daemon=True
            )
            thread.start()
            return True
        except Exception as e:
            self.logger.error(f"Error starting scanner: {str(e)}")
            return False
    
    # def start(self):
    #     """Start the scanner"""
    #     self.running = True
        
    #     # Start background news fetching
    #     if self.news_calendar:
    #         self.start_news_background_fetch(interval_hours=6)
        
    #     self.logger.info("🔨 Hammer Pattern Scanner started")
    #     return True
    # def is_orphan_row(self, row):
    #     # Skip if we don’t even know entry or SL
    #     if pd.isna(row.get("entry_price")) or pd.isna(row.get("sl_price")):
    #         return False
    
    #     orphan_fields = [
    #         'tp_1_1_result', 'tp_1_1_time_seconds',
    #         'tp_1_2_result', 'tp_1_2_time_seconds',
    #         'tp_1_3_result', 'tp_1_3_time_seconds',
    #         'tp_1_4_result', 'tp_1_4_time_seconds',
    #         'tp_1_5_result', 'tp_1_5_time_seconds',
    #         'tp_1_6_result', 'tp_1_6_time_seconds',
    #         'tp_1_7_result', 'tp_1_7_time_seconds',
    #         'tp_1_8_result', 'tp_1_8_time_seconds',
    #         'tp_1_9_result', 'tp_1_9_time_seconds',
    #         'tp_1_10_result', 'tp_1_10_time_seconds',
    #         'open_tp_rr',
    #         'open_tp_result',
    #         'open_tp_time_seconds',
    #     ]
    
    #     for field in orphan_fields:
    #         val = row.get(field)
    #         if pd.isna(val) or val == 0 or val == "":
    #             return True
    
    #     return False
    # def reconcile_and_resume_trades(self, csv_paths=None):
    #     """
    #     SAFE reconciliation - doesn't interfere with main bot operations
    #     Only processes trades that are COMPLETELY orphaned (no monitoring)
    #     """
    #     if csv_paths is None:
    #         csv_paths = [
    #             '/content/drive/My Drive/hammer_trades.csv',
    #             '/content/drive/My Drive/hammer_trades/zebra.csv'
    #         ]
        
    #     for csv_path in csv_paths:
    #         self.logger.info(f"🔧 SAFE reconciling from: {os.path.basename(csv_path)}")
    #         try:
    #             self._safe_csv_reconciliation(csv_path)
    #         except Exception as e:
    #             self.logger.error(f"❌ Reconciliation error for {csv_path}: {e}")
    
    # def _safe_csv_reconciliation(self, csv_path):
    #     """Safe reconciliation that avoids conflicts with live monitoring"""
        
    #     # Check if file exists
    #     if not os.path.exists(csv_path):
    #         self.logger.warning(f"📁 CSV not found: {csv_path}")
    #         return
        
    #     try:
    #         # Read CSV
    #         df = pd.read_csv(csv_path).fillna('')
            
    #         # SAFE: Only process trades that are COMPLETELY missing results
    #         # This avoids interfering with partially completed trades
    #         def is_completely_orphaned(row):
    #             """Check if trade has NO TP results and NO exit time"""
    #             # Has exit_time? Not an orphan
    #             if row.get('exit_time') and str(row['exit_time']).strip():
    #                 return False
                
    #             # Check if ANY TP has been recorded
    #             for i in range(1, 11):
    #                 result = row.get(f'tp_1_{i}_result', '')
    #                 if str(result).strip() and result != '':
    #                     return False
                
    #             # Check open TP
    #             open_tp_result = row.get('open_tp_result', '')
    #             if str(open_tp_result).strip() and open_tp_result != '':
    #                 return False
                
    #             # Has entry and SL? Then it's an orphan
    #             if pd.isna(row.get('entry_price')) or pd.isna(row.get('sl_price')):
    #                 return False
                
    #             return True
            
    #         # Find completely orphaned trades
    #         orphan_mask = df.apply(is_completely_orphaned, axis=1)
    #         orphaned_trades = df[orphan_mask]
            
    #         if orphaned_trades.empty:
    #             self.logger.info(f"✅ No completely orphaned trades in {os.path.basename(csv_path)}")
    #             return
            
    #         self.logger.info(f"🔍 Found {len(orphaned_trades)} orphaned trades in {os.path.basename(csv_path)}")
            
    #         # Track which trades we're reconciling to avoid duplicates
    #         reconciling_ids = []
            
    #         for _, row in orphaned_trades.iterrows():
    #             trade_data = row.to_dict()
    #             trade_id = trade_data['trade_id']
                
    #             # SAFE CHECK: Skip if this trade is already being monitored
    #             # Check if there's already a thread for this trade
    #             if self._is_trade_being_monitored(trade_id):
    #                 self.logger.info(f"⏭️ Trade {trade_id} already being monitored, skipping")
    #                 continue
                
    #             reconciling_ids.append(trade_id)
                
    #             # Process the trade
    #             self._process_single_orphaned_trade(trade_data, csv_path)
            
    #         self.logger.info(f"✅ SAFE reconciliation complete for {os.path.basename(csv_path)}")
            
    #     except Exception as e:
    #         self.logger.error(f"❌ Error in safe reconciliation: {e}")
    #         import traceback
    #         self.logger.error(f"❌ Traceback: {traceback.format_exc()}")
    
    # def _is_trade_being_monitored(self, trade_id):
    #     """Check if a trade is already being monitored by main bot"""
    #     # Check active threads
    #     for thread in threading.enumerate():
    #         if f"TPMonitor_{trade_id}" in thread.name:
    #             return True
    #     return False
    
    # def _process_single_orphaned_trade(self, trade_data, csv_path):
    #     """Process a single orphaned trade without interfering with main bot"""
    #     try:
    #         trade_id = trade_data['trade_id']
            
    #         # Parse entry_time safely
    #         entry_time_str = trade_data['entry_time']
    #         if pd.isna(entry_time_str) or not entry_time_str:
    #             self.logger.warning(f"⚠️ Missing entry_time for {trade_id}")
    #             return
            
    #         entry_time = pd.to_datetime(entry_time_str)
    #         if entry_time.tz is None:
    #             entry_time = entry_time.tz_localize(NY_TZ)
    #         else:
    #             entry_time = entry_time.tz_convert(NY_TZ)
            
    #         # Check if replayable
    #         if not self.is_trade_replayable(trade_data):
    #             self.logger.warning(f"⏭️ Unreplayable trade {trade_id}")
    #             return
            
    #         instrument = trade_data['instrument']
    #         self.logger.info(f"🔁 Replaying {trade_id} from {entry_time}")
            
    #         # Fetch candles
    #         candles = self.cached_fetch_candles(
    #             instrument,
    #             'M5',
    #             count=5000,
    #             force_fetch=True
    #         )
            
    #         if candles.empty:
    #             self.logger.warning(f"⚠️ No candles for {instrument}")
    #             return
            
    #         # Filter candles
    #         if 'time' in candles.columns:
    #             # Convert candle times
    #             if candles['time'].dt.tz is None:
    #                 candles['time'] = pd.to_datetime(candles['time']).dt.tz_localize('UTC').dt.tz_convert(NY_TZ)
    #             candles = candles[candles['time'] >= entry_time].copy()
            
    #         if candles.empty:
    #             self.logger.info(f"ℹ️ No candles after entry for {trade_id}")
    #             return
            
    #         # Use a SEPARATE monitoring function that doesn't conflict
    #         self._replay_orphaned_trade(trade_data, candles, csv_path)
            
    #     except Exception as e:
    #         self.logger.error(f"❌ Error processing trade {trade_data.get('trade_id', 'UNKNOWN')}: {e}")
    
    # def _replay_orphaned_trade(self, trade_data, replay_candles, csv_path):
    #     """
    #     Special replay function for orphaned trades only
    #     Doesn't start new threads, just replays historical data
    #     """
    #     try:
    #         trade_id = trade_data['trade_id']
            
    #         # Extract trade info
    #         instrument = trade_data['instrument']
    #         direction = trade_data['direction'].lower()
            
    #         try:
    #             entry_price = float(trade_data['entry_price'])
    #             sl_price = float(trade_data['sl_price'])
    #         except (ValueError, TypeError) as e:
    #             self.logger.error(f"❌ Invalid prices for {trade_id}: {e}")
    #             return
            
    #         # Calculate TP prices
    #         tp_prices = self._calculate_tp_prices_for_recovery(trade_data)
            
    #         # Track hits
    #         hit_tps = set()
    #         exit_time = None
    #         exit_reason = None
    #         highest_tp_hit = 0
            
    #         # Process each candle in replay data
    #         for _, candle in replay_candles.iterrows():
    #             candle_time = candle['time']
    #             candle_high = candle['high']
    #             candle_low = candle['low']
                
    #             # Check SL
    #             if direction == 'bearish' and candle_high >= sl_price:
    #                 exit_time = candle_time
    #                 exit_reason = 'SL'
    #                 break
    #             elif direction == 'bullish' and candle_low <= sl_price:
    #                 exit_time = candle_time
    #                 exit_reason = 'SL'
    #                 break
                
    #             # Check TPs
    #             for i in range(1, 11):
    #                 if i in hit_tps:
    #                     continue
                        
    #                 tp_hit = False
    #                 if direction == 'bearish' and candle_low <= tp_prices[i]:
    #                     tp_hit = True
    #                 elif direction == 'bullish' and candle_high >= tp_prices[i]:
    #                     tp_hit = True
                    
    #                 if tp_hit:
    #                     hit_tps.add(i)
    #                     highest_tp_hit = max(highest_tp_hit, i)
    #                     trade_data[f'tp_1_{i}_result'] = f"+{i}"
    #                     # Calculate time in seconds
    #                     entry_time = pd.to_datetime(trade_data['entry_time'])
    #                     if entry_time.tz is None:
    #                         entry_time = entry_time.tz_localize(NY_TZ)
    #                     time_diff = (candle_time - entry_time).total_seconds()
    #                     trade_data[f'tp_1_{i}_time_seconds'] = int(time_diff)
                        
    #                     # If this is the highest TP hit, update exit info
    #                     if i > trade_data.get('tp_level_hit', 0):
    #                         trade_data['tp_level_hit'] = i
    #                         trade_data['time_to_exit_seconds'] = int(time_diff)
    #                         exit_time = candle_time
    #                         exit_reason = f'TP_{i}'
            
    #         # Update exit info if trade was closed
    #         if exit_time:
    #             trade_data['exit_time'] = exit_time.strftime('%Y-%m-%d %H:%M:%S')
    #             if exit_reason == 'SL':
    #                 # Mark all unhit TPs as -1
    #                 for i in range(1, 11):
    #                     if i not in hit_tps:
    #                         trade_data[f'tp_1_{i}_result'] = "-1"
    #                 trade_data['open_tp_result'] = "-1"
    #                 trade_data['tp_level_hit'] = -1
            
    #         # Update CSV
    #         self._safe_update_csv(trade_data, csv_path)
            
    #         # If still open, let main bot handle live monitoring
    #         if not exit_time:
    #             self.logger.info(f"📝 Trade {trade_id} still open after replay - main bot will monitor")
            
    #         self.logger.info(f"✅ Replay complete for {trade_id}")
            
    #     except Exception as e:
    #         self.logger.error(f"❌ Replay error for {trade_data.get('trade_id', 'UNKNOWN')}: {e}")
    
    # def _safe_update_csv(self, trade_data, csv_path):
    #     """Update CSV without interfering with main bot's file operations"""
    #     try:
    #         # Add a small delay to avoid file lock conflicts
    #         time.sleep(0.1)
            
    #         # Read current file
    #         if not os.path.exists(csv_path):
    #             return False
            
    #         df = pd.read_csv(csv_path)
            
    #         # Find and update the row
    #         trade_id = trade_data['trade_id']
    #         mask = df['trade_id'] == trade_id
            
    #         if mask.any():
    #             # Update only the columns that exist in both
    #             for col in trade_data.keys():
    #                 if col in df.columns:
    #                     df.loc[mask, col] = trade_data[col]
                
    #             # Save back
    #             df.to_csv(csv_path, index=False)
    #             self.logger.info(f"💾 Updated {trade_id} in {os.path.basename(csv_path)}")
    #             return True
            
    #         return False
            
    #     except Exception as e:
    #         self.logger.error(f"❌ Safe CSV update failed: {e}")
    #         return False


    # def _calculate_tp_prices_for_recovery(self, trade_data):
    #     """Helper to get TP prices without redundant code"""
    #     tp_prices = {}
    #     instrument = trade_data['instrument']
    #     entry_price = float(trade_data['entry_price'])  # CONVERT TO FLOAT
    #     direction = trade_data['direction'].lower()
    #     pip_multiplier = 100 if 'JPY' in instrument else 10000
        
    #     for i in range(1, 11):
    #         # Get TP distance in pips from trade data AND CONVERT TO FLOAT
    #         try:
    #             dist = float(trade_data.get(f'tp_1_{i}_distance', 0))
    #         except (ValueError, TypeError):
    #             dist = 0.0
            
    #         if direction == 'bearish':
    #             tp_prices[i] = entry_price - (dist / pip_multiplier)
    #         else:
    #             tp_prices[i] = entry_price + (dist / pip_multiplier)
    #     return tp_prices

    # def _start_tp_monitoring(self, trade_data):
    #     """Start monitoring TPs in background thread"""
    #     try:
    #         thread = threading.Thread(
    #             target=self._monitor_tp_levels,
    #             args=(trade_data,),
    #             name=f"TPMonitor_{trade_data['trade_id']}",
    #             daemon=True
    #         )
    #         thread.start()
    #         self.logger.info(f"📊 Started TP monitoring for {trade_data['trade_id']}")
    #     except Exception as e:
    #         self.logger.error(f"❌ Error starting TP monitoring: {str(e)}")
    
    # def _monitor_tp_levels(self, trade_data, replay_candles=None):
    #     """Monitor and record TP hits in background with PROPER BE tracking"""
    #     try:
    #         # Extract basic trade information
    #         instrument = trade_data['instrument']
    #         direction = trade_data['direction'].lower()
            
    #         # Convert entry_price and sl_price to float
    #         try:
    #             entry_price = float(trade_data['entry_price'])
    #         except (ValueError, TypeError):
    #             self.logger.error(f"❌ Invalid entry_price: {trade_data.get('entry_price')}")
    #             return
            
    #         try:
    #             sl_price = float(trade_data['sl_price'])
    #         except (ValueError, TypeError):
    #             self.logger.error(f"❌ Invalid sl_price: {trade_data.get('sl_price')}")
    #             return
            
    #         # Calculate TP prices for levels 1-10 based on pip distance
    #         tp_prices = {}
    #         pip_multiplier = 100 if 'JPY' in instrument else 10000
            
    #         for i in range(1, 11):
    #             # Get TP distance in pips from trade data AND CONVERT TO FLOAT
    #             try:
    #                 distance_pips = float(trade_data.get(f'tp_1_{i}_distance', 0))
    #             except (ValueError, TypeError):
    #                 distance_pips = 0.0  # Default to 0 if conversion fails
                
    #             # Calculate TP price based on direction
    #             if direction == 'bearish':
    #                 tp_price = entry_price - (distance_pips / pip_multiplier)
    #             else:
    #                 tp_price = entry_price + (distance_pips / pip_multiplier)
                
    #             tp_prices[i] = tp_price
            
    #         # Get optional open TP price (for trailing or flexible TP)
    #         open_tp_price = trade_data.get('open_tp_price')
    
    #         # Initialize break-even tracking for each TP level (1-10)
    #         be_tracking = {
    #             i: {
    #                 'state': 'waiting',           # waiting, tracking, completed
    #                 'price_returned_to_entry': False,  # Did price return to entry after TP hit?
    #                 'next_tp_hit': False,         # Was the next TP level hit?
    #                 'sl_hit': False,              # Was SL hit after this TP?
    #                 'outcome': 'pending'          # pending, hit, miss, incomplete
    #             }
    #             for i in range(1, 11)
    #         }
            
    #         # Track which TP levels have been hit
    #         hit_tps = set()
            
    #         # Set monitoring parameters
    #         start_time = datetime.now(NY_TZ)
    #         monitor_duration = timedelta(hours=24)  # Monitor for 24 hours
    #         check_interval = 1  # Check every 1 second
            
    #         # Variable to avoid processing the same candle multiple times
    #         last_candle_time = None
            
    #         # ============================================================================
    #         # MAIN MONITORING LOOP: Monitor trade for 24 hours
    #         # ============================================================================
            
    #         while datetime.now(NY_TZ) - start_time < monitor_duration:
    #             # Get current M1 candle data
    #             if replay_candles is not None:
    #                 # For replay mode (reconciliation)
    #                 if replay_candles.empty:
    #                     break
    #                 df = replay_candles.iloc[:2]
    #                 replay_candles = replay_candles.iloc[1:]
    #             else:
    #                 # Live monitoring
    #                 df = self.cached_fetch_candles(instrument, 'M1', count=2, force_fetch=True)
                    
    #             if df.empty:
    #                 time.sleep(check_interval)
    #                 continue
                
    #             # Get the latest candle
    #             current_candle = df.iloc[-1]
    #             current_time = current_candle['time']
                
    #             # Skip if we've already processed this candle (avoid duplicate work)
    #             if last_candle_time and current_time == last_candle_time:
    #                 time.sleep(check_interval)
    #                 continue
                
    #             # Update last processed candle time
    #             last_candle_time = current_time
                
    #             # Extract OHLC data from current candle
    #             candle_open = current_candle['open']
    #             candle_high = current_candle['high']
    #             candle_low = current_candle['low']
    #             candle_close = current_candle['close']
                
    #             # ========================================================================
    #             # CHECK 1: STOP LOSS HIT
    #             # Using candle extremes (high/low) for accurate SL detection
    #             # ========================================================================
                
    #             sl_hit = False
    #             if direction == 'bearish':
    #                 # For bearish trade, check if price went up to SL
    #                 if candle_high >= sl_price:
    #                     sl_hit = True
    #                     self._record_tp_result(trade_data, 'SL', -1, current_time)
                        
    #             elif direction == 'bullish':
    #                 # For bullish trade, check if price went down to SL
    #                 if candle_low <= sl_price:
    #                     sl_hit = True
    #                     self._record_tp_result(trade_data, 'SL', -1, current_time)
                
    #             # If SL hit, process BE tracking and exit monitoring
    #             if sl_hit:
    #                 # Check all TPs that were in tracking mode
    #                 for tp_level in hit_tps:
    #                     if be_tracking[tp_level]['state'] == 'tracking':
    #                         # RULE: If SL hit after TP, it's ALWAYS a HIT for BE
    #                         be_tracking[tp_level]['outcome'] = 'hit'
    #                         be_tracking[tp_level]['sl_hit'] = True
    #                         trade_data[f'if_BE_TP{tp_level}'] = 'hit'
                    
    #                 # Update CSV with results
    #                 self._update_trade_in_csv(trade_data)
    #                 self.logger.info(f"🛑 SL HIT for trade {trade_data['trade_id']} at {current_time}")
    #                 break  # Exit monitoring loop
                
    #             # ========================================================================
    #             # CHECK 2: REGULAR TP LEVELS (1-10)
    #             # Using candle extremes for accurate TP detection
    #             # ========================================================================
                
    #             for i in range(1, 11):
    #                 tp_result_key = f'tp_1_{i}_result'
                    
    #                 # Only check if this TP hasn't been recorded yet
    #                 if trade_data.get(tp_result_key) == '':
    #                     tp_hit = False
                        
    #                     if direction == 'bearish':
    #                         # For bearish, TP is below entry - check if candle low touched TP
    #                         if candle_low <= tp_prices[i]:
    #                             tp_hit = True
    #                     elif direction == 'bullish':
    #                         # For bullish, TP is above entry - check if candle high touched TP
    #                         if candle_high >= tp_prices[i]:
    #                             tp_hit = True
                        
    #                     # If TP hit, record it and start BE tracking
    #                     if tp_hit:
    #                         time_seconds = (current_time - start_time).total_seconds()
    #                         self._record_tp_result(trade_data, f'TP_{i}', i, current_time, time_seconds)
                            
    #                         # Start BE tracking for this TP level
    #                         if i not in hit_tps:
    #                             hit_tps.add(i)
    #                             be_tracking[i]['state'] = 'tracking'
    #                             be_tracking[i]['price_returned_to_entry'] = False
    #                             be_tracking[i]['next_tp_hit'] = False
                
    #             # ========================================================================
    #             # CHECK 3: OPEN TP (if specified)
    #             # Optional flexible TP level
    #             # ========================================================================
                
    #             if open_tp_price and trade_data.get('open_tp_result') == '':
    #                 open_tp_hit = False
                    
    #                 if direction == 'bearish' and candle_low <= open_tp_price:
    #                     open_tp_hit = True
    #                 elif direction == 'bullish' and candle_high >= open_tp_price:
    #                     open_tp_hit = True
                    
    #                 if open_tp_hit:
    #                     time_seconds = (current_time - start_time).total_seconds()
    #                     self._record_tp_result(trade_data, 'OPEN_TP', 
    #                                           trade_data.get('open_tp_rr', 0), 
    #                                           current_time, time_seconds)
                
    #             # ========================================================================
    #             # UPDATE BREAK-EVEN TRACKING FOR EACH HIT TP
    #             # ========================================================================
                
    #             for tp_level in list(hit_tps):
    #                 # Skip TPs not in tracking mode or already completed
    #                 if be_tracking[tp_level]['state'] != 'tracking':
    #                     continue
    #                 if be_tracking[tp_level]['outcome'] != 'pending':
    #                     continue
                    
    #                 # RULE 1: Check if price returned to entry (within tolerance)
    #                 entry_tolerance = 0.0001  # Adjust based on instrument precision
                    
    #                 # Check if entry price is within current candle range
    #                 # OR if closing price is very close to entry
    #                 if (candle_low <= entry_price <= candle_high) or \
    #                    abs(candle_close - entry_price) <= entry_tolerance:
    #                     be_tracking[tp_level]['price_returned_to_entry'] = True
                    
    #                 # RULE 2: Check if next TP is hit
    #                 next_tp_hit = False
                    
    #                 if tp_level < 10:  # There's a next TP (1-9)
    #                     next_tp_result_key = f'tp_1_{tp_level + 1}_result'
    #                     if trade_data.get(next_tp_result_key) != '':
    #                         next_tp_hit = True
    #                 else:  # For TP10, check if open TP is hit
    #                     if open_tp_price and trade_data.get('open_tp_result') != '':
    #                         next_tp_hit = True
                    
    #                 # If next TP hit, determine BE outcome
    #                 if next_tp_hit:
    #                     be_tracking[tp_level]['next_tp_hit'] = True
                        
    #                     # DETERMINE OUTCOME:
    #                     if be_tracking[tp_level]['price_returned_to_entry']:
    #                         # Price returned to entry before next TP → MISS
    #                         be_tracking[tp_level]['outcome'] = 'miss'
    #                         trade_data[f'if_BE_TP{tp_level}'] = 'miss'
    #                     else:
    #                         # Price went directly to next TP → HIT
    #                         be_tracking[tp_level]['outcome'] = 'hit'
    #                         trade_data[f'if_BE_TP{tp_level}'] = 'hit'
                        
    #                     # Update CSV and mark as completed
    #                     self._update_trade_in_csv(trade_data)
    #                     be_tracking[tp_level]['state'] = 'completed'
                
    #             # Wait before next check
    #             if replay_candles is None:
    #                 time.sleep(check_interval)
            
    #         # ============================================================================
    #         # POST-MONITORING CLEANUP
    #         # Handle TPs still in tracking mode when monitoring ends
    #         # ============================================================================
            
    #         for tp_level in hit_tps:
    #             if be_tracking[tp_level]['state'] == 'tracking' and be_tracking[tp_level]['outcome'] == 'pending':
    #                 # Monitoring ended without next TP or SL → mark as incomplete
    #                 trade_data[f'if_BE_TP{tp_level}'] = 'incomplete'
            
    #         # Log final results
    #         self.logger.info(f"📊 TP monitoring completed for {trade_data['trade_id']}")
    #         be_results = [f'TP{i}:{be_tracking[i].get("outcome", "N/A")}' for i in hit_tps]
    #         self.logger.info(f"   BE Results: {be_results}")
            
    #     except Exception as e:
    #         self.logger.error(f"❌ Error in TP monitoring: {str(e)}")
    #         import traceback
    #         self.logger.error(f"❌ Traceback: {traceback.format_exc()}")
    
    # def _record_tp_result(self, trade_data, tp_type, result_value, hit_time, time_seconds=None):
    #     """Record TP result to CSV with RR values"""
    #     try:
    #         # Ensure hit_time is timezone-aware
    #         if hasattr(hit_time, 'tz') and hit_time.tz is None:
    #             hit_time = hit_time.tz_localize(NY_TZ)
            
    #         # Update trade_data
    #         if tp_type.startswith('TP_'):
    #             tp_num = int(tp_type.split('_')[1])
    #             trade_data[f'tp_1_{tp_num}_result'] = f"+{result_value}"
    #             if time_seconds:
    #                 trade_data[f'tp_1_{tp_num}_time_seconds'] = int(time_seconds)
                
    #             # Update highest RR achieved (TP number = RR multiple)
    #             current_highest_rr = trade_data.get('tp_level_hit', 0)
    #             if tp_num > current_highest_rr:
    #                 trade_data['tp_level_hit'] = tp_num
    #                 trade_data['time_to_exit_seconds'] = int(time_seconds) if time_seconds else 0
    #                 trade_data['exit_time'] = hit_time.strftime('%Y-%m-%d %H:%M:%S')
                    
    #         elif tp_type == 'OPEN_TP':
    #             trade_data['open_tp_result'] = f"+{result_value}"
    #             if time_seconds:
    #                 trade_data['open_tp_time_seconds'] = int(time_seconds)
                
    #             # For open TP, we store the RR value from open_tp_rr
    #             if trade_data.get('tp_level_hit', 0) == 0:
    #                 open_tp_rr = trade_data.get('open_tp_rr', 0)
    #                 trade_data['tp_level_hit'] = open_tp_rr
    #                 trade_data['time_to_exit_seconds'] = int(time_seconds) if time_seconds else 0
    #                 trade_data['exit_time'] = hit_time.strftime('%Y-%m-%d %H:%M:%S')
                    
    #         elif tp_type == 'SL':
    #             # Record -1 for all TPs that weren't hit
    #             for i in range(1, 11):
    #                 if trade_data.get(f'tp_1_{i}_result') == '':
    #                     trade_data[f'tp_1_{i}_result'] = "-1"
    #             if trade_data.get('open_tp_result') == '':
    #                 trade_data['open_tp_result'] = "-1"
                
    #             # ✅ CRITICAL FIX: Only set -1 if NO TP was hit yet
    #             tp_was_hit = False
    #             for i in range(1, 11):
    #                 if trade_data.get(f'tp_1_{i}_result', '').startswith('+'):
    #                     tp_was_hit = True
    #                     break
                
    #             if not tp_was_hit and trade_data.get('open_tp_result', '').startswith('+'):
    #                 tp_was_hit = True
                
    #             # Only set tp_level_hit to -1 if NO TP was hit
    #             if not tp_was_hit:
    #                 trade_data['tp_level_hit'] = -1
                
    #             trade_data['time_to_exit_seconds'] = int(time_seconds) if time_seconds else 0
    #             trade_data['exit_time'] = hit_time.strftime('%Y-%m-%d %H:%M:%S')
            
    #     except Exception as e:
    #         self.logger.error(f"❌ Error recording TP result: {str(e)}")
    
    # def _update_trade_in_csv(self, trade_data, csv_path=None):
    #     if csv_path is None:
    #         csv_path = self.csv_file_path
    #     try:
    #         if not os.path.exists(self.csv_file_path):
    #             return False
            
    #         # Read all data
    #         rows = []
    #         with open(self.csv_file_path, 'r', newline='') as f:
    #             reader = csv.DictReader(f)
    #             fieldnames = reader.fieldnames
    #             rows = list(reader)
            
    #         # Find and update the trade
    #         updated = False
    #         for i, row in enumerate(rows):
    #             if row.get('trade_id') == trade_data['trade_id']:
    #                 # Update the row with new data
    #                 for key, value in trade_data.items():
    #                     if key in fieldnames:
    #                         rows[i][key] = value
    #                 updated = True
    #                 break
            
    #         if updated:
    #             # Write back to CSV
    #             with open(self.csv_file_path, 'w', newline='') as f:
    #                 writer = csv.DictWriter(f, fieldnames=fieldnames)
    #                 writer.writeheader()
    #                 writer.writerows(rows)
                
    #             self.logger.info(f"💾 Updated trade {trade_data['trade_id']} in CSV")
    #             return True
            
    #         return False
            
    #     except Exception as e:
    #         self.logger.error(f"❌ Error updating CSV: {str(e)}")
    #         return False




# ===============================
# ZEBRA SCANNER CLASS
# ================================

class ZebraScanner:
    """Independent HalfTrend Zebra scanner - uses HammerScanner functions"""
    
    def __init__(self, credentials, instrument, timeframe, news_calendar=None, logger=None):
        """
        Initialize Zebra scanner for a specific instrument and timeframe
        
        Args:
            credentials: Dictionary with API keys and Telegram credentials
            instrument: Trading instrument (e.g., 'EUR_USD')
            timeframe: Timeframe to scan (e.g., 'M5')
            news_calendar: Shared NewsCalendar instance (optional)
            logger: Logger instance
        """
        self.instrument = instrument
        self.timeframe = timeframe
        self.credentials = credentials
        self.news_calendar = news_calendar
        self.running = True
        
        # Set up logger
        if logger:
            self.logger = logger
        else:
            import logging
            self.logger = logging.getLogger(f'ZebraScanner.{instrument}.{timeframe}')
        
        # Create own HammerScanner instance for Zebra (different CSV directory)
        zebra_csv_path = '/content/drive/MyDrive/hammer_trades/zebra'
        
        self.hammer_scanner = HammerPatternScanner(
            credentials=credentials,
            csv_base_path=zebra_csv_path,
            logger=self.logger,
            news_calendar=news_calendar
            # use_global_cache=True  # Use the shared global cache
        )
        
        # Start the hammer scanner (but not background news fetch since we use shared)
        self.hammer_scanner.running = True
        # self.hammer_scanner.reconcile_and_resume_trades()
        
        self.logger.info(f"🦓 Initialized Zebra Scanner for {instrument} {timeframe}")
    
    def run(self):
        """Main Zebra scanning loop - uses HammerScanner's run_zebra_scan method"""
        try:
            # Generate a unique signal ID prefix for this scanner
            signal_id_prefix = f"ZEBRA_{self.instrument}_{self.timeframe}"
            
            # Use HammerScanner's existing run_zebra_scan method
            self.logger.info(f"🦓 Starting Zebra scanning for {self.instrument} {self.timeframe}")
            self.hammer_scanner.run_zebra_scan(self.timeframe, self.instrument, signal_id_prefix)
            
        except Exception as e:
            self.logger.error(f"❌ Zebra scanner error for {self.instrument} {self.timeframe}: {str(e)}", exc_info=True)
    
    def stop(self):
        """Stop the Zebra scanner"""
        self.running = False
        self.hammer_scanner.running = False
        self.logger.info(f"🛑 Stopped Zebra Scanner for {self.instrument} {self.timeframe}")



# ================================
# ZEBRA MANAGER CLASS
# ================================

class ZebraManager:
    """Manager for all ZebraScanner instances"""
    
    def __init__(self, credentials, news_calendar=None, instruments_dict=None, logger=None):
        """
        Initialize Zebra Manager
        
        Args:
            credentials: Dictionary with API keys and Telegram credentials
            news_calendar: Shared NewsCalendar instance (optional)
            instruments_dict: Dictionary of instruments and timeframes (uses ZEBRA_INSTRUMENTS if None)
            logger: Logger instance
        """
        self.credentials = credentials
        self.news_calendar = news_calendar
        self.instruments_dict = instruments_dict or ZEBRA_INSTRUMENTS
        self.running = False
        self.quarter_manager = RobustQuarterManager()
        # self.signal_processor = SignalProcessor()
        
        # Set up logger
        if logger:
            self.logger = logger
        else:
            import logging
            self.logger = logging.getLogger('ZebraManager')
        
        # Store scanners and threads
        self.scanners = []  # List of ZebraScanner instances
        self.threads = []   # List of Thread objects
        
        self.logger.info(f"🎯 Initialized Zebra Manager for {len(self.instruments_dict)} instruments")
    
    def start(self):
        """Start all Zebra scanners in separate threads"""
        if self.running:
            self.logger.warning("⚠️ Zebra Manager is already running")
            return
        
        self.running = True
        self.logger.info("🚀 Starting all Zebra scanners...")
        
        # Count total scanners to start
        total_scanners = 0
        for instrument, timeframes in self.instruments_dict.items():
            total_scanners += len(timeframes)
        
        self.logger.info(f"📊 Will start {total_scanners} Zebra scanners")
        
        # Create and start scanners for each instrument/timeframe combination
        for instrument, timeframes in self.instruments_dict.items():
            for timeframe in timeframes:
                try:
                    # Create ZebraScanner instance
                    scanner = ZebraScanner(
                        credentials=self.credentials,
                        instrument=instrument,
                        timeframe=timeframe,
                        news_calendar=self.news_calendar,
                        logger=self.logger
                    )
                    
                    # Create and start thread
                    thread = threading.Thread(
                        target=scanner.run,
                        name=f"Zebra_{instrument}_{timeframe}",
                        daemon=True
                    )
                    thread.start()
                    
                    # Store for management
                    self.scanners.append(scanner)
                    self.threads.append(thread)
                    
                    self.logger.info(f"✅ Started Zebra scanner: {instrument} {timeframe}")
                    
                    # Small delay to avoid overwhelming system
                    time.sleep(0.1)
                    
                except Exception as e:
                    self.logger.error(f"❌ Failed to start Zebra scanner for {instrument} {timeframe}: {str(e)}")
        
        self.logger.info(f"🎯 All Zebra scanners started. Total: {len(self.scanners)}")
    
    def stop(self):
        """Stop all Zebra scanners"""
        if not self.running:
            return
        
        self.running = False
        self.logger.info("🛑 Stopping all Zebra scanners...")
        
        # Stop all scanners
        for scanner in self.scanners:
            try:
                scanner.stop()
            except Exception as e:
                self.logger.error(f"Error stopping scanner: {str(e)}")
        
        # Wait for threads to finish (with timeout)
        for thread in self.threads:
            try:
                thread.join(timeout=5)
            except Exception as e:
                self.logger.error(f"Error joining thread: {str(e)}")
        
        self.logger.info("✅ All Zebra scanners stopped")
    
    def get_status(self):
        """Get status of all Zebra scanners"""
        status = {
            'running': self.running,
            'total_scanners': len(self.scanners),
            'total_threads': len(self.threads),
            'instruments': len(self.instruments_dict),
            'cache_stats': GLOBAL_CACHE.get_stats() if 'GLOBAL_CACHE' in globals() else None
        }
        
        # Count active threads
        active_threads = 0
        for thread in self.threads:
            if thread.is_alive():
                active_threads += 1
        
        status['active_threads'] = active_threads
        
        return status

# ================================
# ULTIMATE TRADING SYSTEM WITH TRIPLE CONFLUENCE
# ================================

class UltimateTradingSystem:
    def __init__(self, pair_group, pair_config, news_calendar=None, telegram_token=None, telegram_chat_id=None, signal_processor=None ):  
        # Store the parameters as instance variables
        self.pair_group = pair_group
        self.pair_config = pair_config
        self.sd_detector = SupplyDemandDetector(min_zone_pct=0)
        self.volatile_pairs = ['XAU_USD']
        
        # Store the shared news calendar
        self.news_calendar = news_calendar
        
        # Handle Telegram credentials
        self.telegram_token = telegram_token
        self.telegram_chat_id = telegram_chat_id
        
        # Set up instance logger
        import logging
        self.logger = logging.getLogger(f'UltimateTradingSystem.{pair_group}')
        
        # BACKWARD COMPATIBLE: Handle both old and new structures
        if 'instruments' in pair_config:
            self.instruments = pair_config['instruments']
        else:
            self.instruments = [pair_config['pair1'], pair_config['pair2']]
            self.logger.info(f"🔄 Converted old structure for {pair_group} to instruments: {self.instruments}")
        
        # Initialize components
        self.timing_manager = RobustTimingManager()
        self.quarter_manager = RobustQuarterManager()
        
        # FIRST create FeatureBox - NOW WITH THE LOGGER
        self.feature_box = RealTimeFeatureBox(
            self.pair_group, 
            self.timing_manager, 
            self.telegram_token, 
            self.telegram_chat_id,
            logger=self.logger  # Now self.logger exists!
        )
        
        # THEN create detectors and connect FeatureBox
        self.smt_detector = UltimateSMTDetector(pair_config, self.timing_manager)
        self.crt_detector = RobustCRTDetector(self.timing_manager)
        self.crt_detector.feature_box = self.feature_box
        self.feature_box.sd_detector = self.sd_detector
        
        # Data storage for all instruments
        self.market_data = {inst: {} for inst in self.instruments}
        
        self.logger.info(f"🎯 Initialized ULTIMATE trading system for {self.pair_group}: {', '.join(self.instruments)}")
        self.logger.info(f"🎯 FVG Analyzer initialized for {pair_group}")
        self.fvg_detector = FVGDetector(min_gap_pct=0.20)
        self.fvg_smt_tap_sent = {}
        self.crt_smt_ideas_sent = {}
        self.sd_zone_sent = {}
        self.sd_hp_sent = {}
        self.fvg_ideas_sent = {}
        self.double_smt_sent = {}
        self.hybrid_timing = HybridTimingSystem(pair_group)
        self.last_candle_scan = {}
        self.COOLDOWN_HOURS = 24 * 3600
        self.CLEANUP_DAYS = 7 * 24 * 3600
        self.timeframe_cycle_map = {
            'H4': ['weekly', 'daily'],
            'H1': ['daily'], 
            'M15': ['daily', '90min']
        }
        
        # Create hammer credentials
        hammer_credentials = {
            'telegram_token': telegram_token,
            'telegram_chat_id': telegram_chat_id,
            'oanda_api_key': os.getenv('OANDA_API_KEY')
        }
        self.signal_processor = signal_processor
        
        # Pass the shared news calendar to hammer scanner
        self.hammer_scanner = HammerPatternScanner(
            hammer_credentials,
            csv_base_path='/content/drive/MyDrive/hammer_trades',
            logger=logger,
            news_calendar=self.news_calendar,  # PASS SHARED CALENDAR
            signal_processor=signal_processor,  # ⬅️ PASS TO SCANNER
            quarter_manager = self.quarter_manager
        )
        
        logger.info(f"🔨 Hammer Pattern Scanner initialized for {pair_group}")
        
        
      
        # Start the hammer scanner if we have a news calendar
        if self.news_calendar:
            self.hammer_scanner.start()
            
            
        
    def get_sleep_time(self):
        """Use smart timing instead of fixed intervals"""
        return self.hybrid_timing.get_sleep_time()
    
    async def run_ultimate_analysis(self, api_key):
        """Run analysis with prioritized scan pipeline"""
        try:
            # Cleanup expired features first
            self.feature_box.cleanup_expired_features()
            self.cleanup_old_signals()
            
            # Fetch data
            await self._fetch_all_data_parallel(api_key)
        
            self.reset_smt_detector_state()
                
                # Check if we have new candles that warrant immediate scanning
            new_candles_detected = self._check_new_candles()
                
            if not new_candles_detected:
                logger.info(f"⏸️ No new candles - skipping analysis")
                return None
                
            logger.info(f"🎯 NEW CANDLES DETECTED - Running analysis")
                
                # Scan for new features and add to Feature Box
            await self._scan_and_add_features_immediate()
                
                # Scan for Supply/Demand zones
            self._scan_and_add_sd_zones()
                            
            self.debug_feature_box()
            self.debug_smt_detection()
                
                # Define scan pipeline in priority order
            scan_pipeline = [
                ("FVG+SMT", self._scan_fvg_with_smt_tap),
                ("SD+SMT", self._scan_sd_with_smt_tap),
                ("CRT/TPD", self._scan_crt_smt_confluence),
                ("Double SMT", self._scan_double_smts_temporal)
            ]
                
                # Run scans in priority order with short-circuit
            signals_found = 0
            signal_type = None
                
            for scan_name, scan_method in scan_pipeline:
                logger.info(f"🔍 Running {scan_name} scan...")
                    
                signal_detected = scan_method()
                if signal_detected:
                    signals_found = 1
                    signal_type = scan_name
                    logger.info(f"✅ {scan_name} signal detected - stopping scan pipeline")
                    break
                
            # Log results
            if signal_type:
                logger.info(f"🎯 Signal found: {signal_type}")
            else:
                logger.info(f"🔍 No signals detected in any scan")
                
            # Get feature summary
            summary = self.feature_box.get_active_features_summary()
            sd_count = len(self.feature_box.active_features['sd_zone'])
            logger.info(f"📊 {self.pair_group} Feature Summary: {summary['smt_count']} SMTs, {sd_count} SD zones, {summary['crt_count']} CRTs, {summary['psp_count']} PSPs, {summary.get('tpd_count', 0)} TPDs")
                
            return None
                
        except Exception as e:
            logger.error(f"❌ Error in analysis for {self.pair_group}: {str(e)}", exc_info=True)
            return None
            

        async def run_optimized_analysis(self, api_key):
            """Run analysis only for timeframes that need scanning"""
            try:
                # Cleanup
                self.feature_box.cleanup_expired_features()
                self.cleanup_old_signals()
                
                # Get timeframes that need scanning
                timeframes_to_scan = self.hybrid_timing.get_timeframes_to_scan()
                
                if not timeframes_to_scan:
                    sleep_time = self.hybrid_timing.get_sleep_time()
                    logger.info(f"⏰ {self.pair_group}: No timeframes need scanning, sleeping {sleep_time:.0f}s")
                    return sleep_time
                
                logger.info(f"🔍 {self.pair_group}: Scanning timeframes: {timeframes_to_scan}")
                
                # Optimized data fetching - only fetch needed timeframes
                await self._fetch_data_selective(timeframes_to_scan, api_key)
                
                # Run specific scans based on timeframe
                results = {
                    'fvg': False,
                    'sd': False,
                    'crt': False,
                    'double_smt': False,
                    'tpd': False
                }
                
                # Check for quick scans (M5, M15) - mostly for TPD/CRT
                quick_tfs = [tf for tf in timeframes_to_scan if tf in ['M5', 'M15']]
                if quick_tfs:
                    results['tpd'] = self._scan_quick_timeframes(quick_tfs)
                
                # Check for normal scans (H1) - SMT/CRT
                normal_tfs = [tf for tf in timeframes_to_scan if tf in ['H1']]
                if normal_tfs:
                    results['crt'] = self._scan_crt_smt_confluence()
                    results['double_smt'] = self._scan_double_smts_temporal()
                
                # Check for deep scans (H4, D, W) - SD zones, SMT
                deep_tfs = [tf for tf in timeframes_to_scan if tf in ['H4', 'D', 'W']]
                if deep_tfs:
                    results['sd'] = self._scan_sd_with_smt_tap()
                    results['fvg'] = self._scan_fvg_with_smt_tap()
                
                # Log results
                signals_found = sum(results.values())
                logger.info(f"🎯 Signals found: {signals_found} {results}")
                
                # Update feature summary
                summary = self.feature_box.get_active_features_summary()
                logger.info(f"📊 {self.pair_group} Feature Summary: {summary['smt_count']} SMTs, {summary['sd_zone_count']} SD zones, {summary['crt_count']} CRTs, {summary['psp_count']} PSPs, {summary['tpd_count']} TPDs")
                
                return self.hybrid_timing.get_sleep_time()
                
            except Exception as e:
                logger.error(f"❌ Error in optimized analysis for {self.pair_group}: {str(e)}", exc_info=True)
                return 60

        async def _fetch_data_selective(self, timeframes_to_scan, api_key):
            """Fetch data only for timeframes that need scanning"""
            tasks = []
            
            # Determine which instruments need which timeframes
            for instrument in self.instruments:
                for tf in timeframes_to_scan:
                    # Different candle counts for different purposes
                    if tf in ['H4', 'D', 'W']:
                        count = 50  # More for SD zones
                    else:
                        count = 40   # Less for quick analysis
                    
                    task = asyncio.create_task(
                        self._fetch_single_instrument_data(instrument, tf, count, api_key)
                    )
                    tasks.append(task)
            
            if tasks:
                await asyncio.gather(*tasks)
                logger.info(f"✅ Fetched data for {len(timeframes_to_scan)} timeframes")

    def debug_sd_zones(self):
        """Debug Supply/Demand zones in FeatureBox"""
        logger.info(f"🔧 DEBUG: SD Zones in FeatureBox for {self.pair_group}")
        
        active_zones = self.feature_box.get_active_sd_zones()
        logger.info(f"🔧 Active SD zones: {len(active_zones)}")
        
        for zone in active_zones:
            status = "✅ VALID" if zone.get('is_valid', True) else "❌ INVALID"
            logger.info(f"🔧   {zone['zone_name']}: {zone['type']} at {zone['zone_low']:.4f}-{zone['zone_high']:.4f} - {status}")

    def _cleanup_old_double_smt_signals(self):
        """Remove old Double SMT signals from tracking (7-day cleanup)"""
        if not hasattr(self, 'double_smt_sent') or not self.double_smt_sent:
            return
        
        current_time = datetime.now(NY_TZ)
        signals_to_remove = []
        
        for signal_id, sent_time in self.double_smt_sent.items():
            if (current_time - sent_time).total_seconds() > self.CLEANUP_DAYS:  # 7 days
                signals_to_remove.append(signal_id)
        
        for signal_id in signals_to_remove:
            del self.double_smt_sent[signal_id]
        
        if signals_to_remove:
            logger.debug(f"🧹 Cleaned up {len(signals_to_remove)} old Double SMT signals (7+ days)")

    def _scan_and_add_sd_zones(self):
        """Scan for Supply/Demand zones with timezone debug"""
        from pytz import timezone
        NY_TZ = timezone('America/New_York')
        
        for instrument in self.instruments:
            for timeframe in ['M15', 'H1', 'H4','D' , 'W']:
                data = self.market_data[instrument].get(timeframe)
                if data is not None and not data.empty:
                    if data['time'].dt.tz is None:
                        data['time'] = data['time'].dt.tz_localize('UTC').dt.tz_convert(NY_TZ)
        
        timeframes_to_scan = ['M15', 'H1', 'H4','D' , 'W']
        if 'XAU_USD' in self.instruments:
            timeframes_to_scan.append('M5')
        
        zones_added = 0
        zones_invalidated = 0
        
        for instrument in self.instruments:
            for timeframe in timeframes_to_scan:
                data = self.market_data[instrument].get(timeframe)
                if data is not None and not data.empty:
                    other_instrument = [inst for inst in self.instruments if inst != instrument][0]
                    other_data = self.market_data[other_instrument].get(timeframe)
                    
                    zones = self.sd_detector.scan_timeframe(data, timeframe, instrument)
                    
                    for zone in zones:
                        is_valid = self.sd_detector.check_zone_still_valid(zone, data, other_data)
                        
                        if is_valid:
                            if self.feature_box.add_sd_zone(zone):
                                zones_added += 1
                        else:
                            zones_invalidated += 1
        
        active_zones = self.feature_box.get_active_sd_zones()
        
        return zones_added

    def _cleanup_old_sd_zone_signals(self):
        """Remove old SD zone signals from tracking (7-day cleanup)"""
        if not hasattr(self, 'sd_zone_sent') or not self.sd_zone_sent:
            return
        
        current_time = datetime.now(NY_TZ)
        signals_to_remove = []
        
        for signal_id, sent_time in self.sd_zone_sent.items():
            if (current_time - sent_time).total_seconds() > self.CLEANUP_DAYS:
                signals_to_remove.append(signal_id)
        
        for signal_id in signals_to_remove:
            del self.sd_zone_sent[signal_id]
        
        if signals_to_remove:
            logger.debug(f"🧹 Cleaned up {len(signals_to_remove)} old SD zone signals (7+ days)")
    
    def _cleanup_old_sd_hp_signals(self):
        """Remove old SD HP zone signals from tracking (7-day cleanup)"""
        if not hasattr(self, 'sd_hp_sent') or not self.sd_hp_sent:
            return
        
        current_time = datetime.now(NY_TZ)
        signals_to_remove = []
        
        for signal_id, sent_time in self.sd_hp_sent.items():
            if (current_time - sent_time).total_seconds() > self.CLEANUP_DAYS:
                signals_to_remove.append(signal_id)
        
        for signal_id in signals_to_remove:
            del self.sd_hp_sent[signal_id]
        
        if signals_to_remove:
            logger.debug(f"🧹 Cleaned up {len(signals_to_remove)} old SD HP zone signals (7+ days)")
    
    def _cleanup_old_fvg_ideas_signals(self):
        """Remove old FVG ideas signals from tracking (7-day cleanup)"""
        if not hasattr(self, 'fvg_ideas_sent') or not self.fvg_ideas_sent:
            return
        
        current_time = datetime.now(NY_TZ)
        signals_to_remove = []
        
        for signal_id, sent_time in self.fvg_ideas_sent.items():
            if (current_time - sent_time).total_seconds() > self.CLEANUP_DAYS:
                signals_to_remove.append(signal_id)
        
        for signal_id in signals_to_remove:
            del self.fvg_ideas_sent[signal_id]
        
        if signals_to_remove:
            logger.debug(f"🧹 Cleaned up {len(signals_to_remove)} old FVG idea signals (7+ days)")

    def cleanup_old_signals(self):
        """Cleanup old signals from all tracking dictionaries (7-day cleanup)"""
        # Call all individual cleanup methods
        self._cleanup_old_fvg_smt_signals()
        self._cleanup_old_crt_smt_signals()
        self._cleanup_old_double_smt_signals()
        self._cleanup_old_sd_zone_signals()      # NEW
        self._cleanup_old_sd_hp_signals()        # NEW
        self._cleanup_old_fvg_ideas_signals()    # NEW
        
        logger.debug("✅ All old signal cleanups completed (7-day threshold)")
    
    def _check_new_candles(self):
        """Check if we have new candles that warrant immediate scanning"""
        new_candles = False
        
        for instrument in self.instruments:
            for timeframe in ['M5', 'M15', 'H1', 'H4','D' , 'W']:
                data = self.market_data[instrument].get(timeframe)
                if data is None or data.empty:
                    continue
                
                # Get the latest candle time
                latest_candle_time = data.iloc[-1]['time']
                
                # Check if this is a new candle we haven't processed
                key = f"{instrument}_{timeframe}"
                if key not in self.last_candle_scan:
                    self.last_candle_scan[key] = latest_candle_time
                    new_candles = True
                    # logger.info(f"🕯️ First scan for {instrument} {timeframe}")
                elif latest_candle_time > self.last_candle_scan[key]:
                    self.last_candle_scan[key] = latest_candle_time
                    new_candles = True
                    # logger.info(f"🕯️ NEW CANDLE: {instrument} {timeframe} at {latest_candle_time.strftime('%H:%M')}")
        
        return new_candles
    
    async def _fetch_all_data_parallel(self, api_key):
        """Fetch data with correct counts for each purpose"""
        tasks = []
        
        # Clear market data for this cycle
        for instrument in self.instruments:
            self.market_data[instrument] = {}
        
        # Define counts for different purposes
        counts_by_purpose = {
            'SMT': {
                'H4': 40,   # Monthly
                'H1': 40,   # Weekly
                'M15': 40,  # Daily
                'M5': 40,   # 90min
            },
            'SD': {
                'M15': 101,
                'H1': 101,
                'H4': 101,
                'D': 101,
                'W': 101,
                'M5': 101 if 'XAU_USD' in self.instruments else 0,
            },
            'CRT': {  # ADD THIS SECTION FOR CRT
                'H1': 10,   # CRT needs recent candles
                'H4': 10,
            }
        }
        
        # Create fetch tasks
        for instrument in self.instruments:
            # Get ALL unique timeframes needed across purposes
            all_timeframes = set()
            
            for purpose in counts_by_purpose:
                for tf in counts_by_purpose[purpose]:
                    all_timeframes.add(tf)
            
            # For each timeframe, fetch the MAXIMUM count needed
            for tf in all_timeframes:
                max_count = 0
                # Find maximum count needed for this timeframe
                for purpose in counts_by_purpose:
                    if tf in counts_by_purpose[purpose]:
                        max_count = max(max_count, counts_by_purpose[purpose][tf])
                
                if max_count > 0:
                    task = asyncio.create_task(
                        self._fetch_single_instrument_data(instrument, tf, max_count, api_key)
                    )
                    tasks.append(task)
                    # logger.info(f"📤 FETCH: {instrument} {tf} - {max_count} candles (max of all purposes)")
        
        # Wait for ALL data
        try:
            results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=45.0)
            
            # Count successful
            successful = sum(1 for r in results if r)
            # logger.info(f"✅ Parallel fetch: {successful}/{len(tasks)} successful for {self.pair_group}")
            
            # Debug what we have
            for instrument in self.instruments:
                # logger.info(f"📦 Data for {instrument}:")
                for tf in ['H4', 'H1', 'M15', 'M5', 'D', 'W']:
                    if tf in self.market_data[instrument]:
                        df = self.market_data[instrument][tf]
                        # logger.info(f"   {tf}: {len(df)} total candles fetched")
            
        except asyncio.TimeoutError:
            logger.warning(f"⚠️ Fetch timeout for {self.pair_group}")
    
    async def _fetch_single_instrument_data(self, instrument, timeframe, count, api_key):
        """Fetch data for single instrument"""
        try:
            # logger.info(f"🔄 Fetching {instrument} {timeframe} with {count} candles...")
            
            df = await asyncio.get_event_loop().run_in_executor(
                None, fetch_candles, instrument, timeframe, count, api_key
            )
            
            if df is None:
                logger.error(f"❌ fetch_candles returned None for {instrument} {timeframe}")
                return False
                
            if not isinstance(df, pd.DataFrame):
                logger.error(f"❌ fetch_candles returned {type(df)}, not DataFrame for {instrument} {timeframe}")
                return False
                
            if df.empty:
                logger.error(f"❌ fetch_candles returned empty DataFrame for {instrument} {timeframe}")
                return False
            
            # Convert timestamps to NY_TZ
            if df['time'].dt.tz is None:
                df['time'] = df['time'].dt.tz_localize('UTC').dt.tz_convert(NY_TZ)
            
            self.market_data[instrument][timeframe] = df
            
            # logger.info(f"✅ Fetched {instrument} {timeframe}: {len(df)} candles")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error fetching {instrument} {timeframe}: {str(e)}")
            return False
    async def _fetch_single(self, inst: str, tf: str, count: int, since: datetime) -> None:
        try:
            df_new = fetch_candles(inst, tf, count, self.api_key, since)
            if not df_new.empty:
                group = next(g for g, c in TRADING_PAIRS.items() if inst in c['instruments'])
                if tf not in self.market_data[group][inst]:
                    self.market_data[group][inst][tf] = df_new
                else:
                    old = self.market_data[group][inst][tf]
                    new_concat = pd.concat([old, df_new[df_new['time'] > old['time'].max() if not old.empty else pd.Series([since])]]).drop_duplicates('time').sort_values('time').reset_index(drop=True)
                    self.market_data[group][inst][tf] = new_concat
                self.last_timestamps[group][inst] = self.market_data[group][inst][tf]['time'].max()
        except Exception as e:
            logger.error(f"❌ {inst} {tf}: {e}")

    def _has_new_candle_data(self, timeframe):
        """Check if any instrument has new candle data for this timeframe (last 2min)."""
        for instrument in self.instruments:
            key = f"{instrument}_{timeframe}"
            if key in self.last_candle_scan:
                last_scan = self.last_candle_scan[key]
                time_since_scan = (datetime.now(NY_TZ) - last_scan).total_seconds()
                if time_since_scan < 120:  # 2 minutes
                    return True
        return False
    
    async def _scan_and_add_features_immediate(self):
        """Scan for features immediately when new candles are detected"""
        cycles = ['monthly', 'weekly', 'daily', '90min']
        logger.info(f"🔍 SMT SCAN: Starting scan for {len(cycles)} cycles")
        smt_detected_count = 0
        
        for cycle in cycles:
            timeframe = self.pair_config['timeframe_mapping'][cycle]
            
            # Get data for this timeframe
            asset1_data = self.market_data[self.instruments[0]].get(timeframe)
            asset2_data = self.market_data[self.instruments[1]].get(timeframe)
            
            # Check if we have data
            if asset1_data is None or asset2_data is None:
                logger.warning(f"⚠️ Missing data for SMT: {self.instruments[0]}={asset1_data is not None}, {self.instruments[1]}={asset2_data is not None}")
                continue
            
            # For SMT we need exactly 40 candles
            # If we have more than 40, take the most recent 40
            if len(asset1_data) > 40:
                asset1_data = asset1_data.tail(40)
                # logger.info(f"📏 Using last 40 candles for {self.instruments[0]} {timeframe}")
            
            if len(asset2_data) > 40:
                asset2_data = asset2_data.tail(40)
                # logger.info(f"📏 Using last 40 candles for {self.instruments[1]} {timeframe}")
            
            # Log what we have
            # logger.info(f"🔍 SMT Data Check - {self.instruments[0]} {timeframe}: Has {len(asset1_data)} candles")
            # logger.info(f"🔍 SMT Data Check - {self.instruments[1]} {timeframe}: Has {len(asset2_data)} candles")
            
            # Check if we have enough data
            if len(asset1_data) >= 40 and len(asset2_data) >= 40:
                # logger.info(f"🔍 Scanning {cycle} cycle ({timeframe}) for SMT...")
                smt_signal = self.smt_detector.detect_smt_all_cycles(asset1_data, asset2_data, cycle)
                
                if smt_signal:
                    # Check if SMT is fresh enough
                    if not self.feature_box.is_smt_fresh_enough(smt_signal):
                        # logger.info(f"🕒 SMT {smt_signal.get('signal_key', 'unknown')} is TOO OLD, skipping addition")
                        continue
                    
                    # Check for PSP immediately
                    psp_signal = self.smt_detector.check_psp_for_smt(smt_signal, asset1_data, asset2_data)
                    
                    # Add it (it's fresh enough)
                    added = self.feature_box.add_smt(smt_signal, psp_signal)
                    
                    if added:
                        smt_detected_count += 1
                        logger.info(f"✅ FRESH SMT ADDED: {cycle} {smt_signal['direction']} - PSP: {'Yes' if psp_signal else 'No'}")
                    else:
                        logger.info(f"❌ Failed to add SMT {smt_signal.get('signal_key', 'unknown')}")
                else:
                    logger.info(f"🔍 SMT detector returned None for {cycle}")
            else:
                logger.warning(f"⚠️ Not enough candles for {cycle} SMT scan: {len(asset1_data)}/{len(asset2_data)}")
        
        # logger.info(f"📊 SMT Scan Complete: Detected {smt_detected_count} SMTs")

    def reset_smt_detector_state(self):
        """Reset SMT detector state to avoid duplicate issues"""
        logger.info("🔄 Resetting SMT detector state")
        self.smt_detector.last_smt_candle = None
        self.smt_detector.signal_counts = {}
        self.smt_detector.invalidated_smts = set()

    def debug_quarter_comparison(self):
        """Compare quarter data between debug and main scan"""
        logger.info("🔬 QUARTER COMPARISON DEBUG")
        
        for cycle in ['weekly', 'daily']:
            timeframe = self.pair_config['timeframe_mapping'][cycle]
            
            asset1_data = self.market_data[self.instruments[0]].get(timeframe)
            asset2_data = self.market_data[self.instruments[1]].get(timeframe)
            
            if asset1_data is None or asset2_data is None:
                continue
            
            # Get quarter grouping
            quarters1 = self.smt_detector.quarter_manager.group_candles_by_quarters(asset1_data, cycle)
            quarters2 = self.smt_detector.quarter_manager.group_candles_by_quarters(asset2_data, cycle)
            
            logger.info(f"🔬 {cycle} ({timeframe}) quarters:")
            for q in ['q1', 'q2', 'q3', 'q4']:
                if q in quarters1:
                    logger.info(f"🔬   Asset1 {q}: {len(quarters1[q])} candles")
                if q in quarters2:
                    logger.info(f"🔬   Asset2 {q}: {len(quarters2[q])} candles")

    def test_smt_workflow(self):
        """Test the complete SMT workflow"""
        logger.info("🧪 TESTING COMPLETE SMT WORKFLOW")
        
        # 1. Reset everything
        self.reset_smt_detector_state()
        self.feature_box.active_features['smt'] = {}  # Clear FeatureBox
        
        # 2. Test weekly cycle
        timeframe = self.pair_config['timeframe_mapping']['weekly']
        asset1_data = self.market_data[self.instruments[0]].get(timeframe)
        asset2_data = self.market_data[self.instruments[1]].get(timeframe)
        
        if asset1_data is None or asset2_data is None:
            logger.info("❌ No data for test")
            return
        
        # logger.info(f"🧪 Testing with {len(asset1_data)} candles")
        
        # 3. Detect SMT
        smt_signal = self.smt_detector.detect_smt_all_cycles(asset1_data, asset2_data, 'weekly')
        
        if smt_signal:
        #     logger.info(f"🧪 SMT Detected: {smt_signal['signal_key']}")
            
            # 4. Check PSP
            psp_signal = self.smt_detector.check_psp_for_smt(smt_signal, asset1_data, asset2_data)
            # logger.info(f"🧪 PSP: {'Found' if psp_signal else 'Not found'}")
            
            # 5. Add to FeatureBox
            success = self.feature_box.add_smt(smt_signal, psp_signal)
            # logger.info(f"🧪 Added to FeatureBox: {success}")
            
            # 6. Check FeatureBox
            # logger.info(f"🧪 FeatureBox now has {len(self.feature_box.active_features['smt'])} SMTs")
        else:
            logger.info("🧪 No SMT detected")

    def _scan_crt_smt_confluence(self):
        """Check CRT with SMT confluence (CRT on higher TF, SMT on lower TF)"""
        logger.info(f"🔷 SCANNING: CRT + SMT Confluence")
        
        # CRT timeframes to check
        crt_timeframes = ['H4', 'H1']
        
        # Check each instrument for CRT
        for instrument in self.instruments:
            for crt_tf in crt_timeframes:
                data = self.market_data[instrument].get(crt_tf)
                if data is None or data.empty:
                    continue
                
                # For CRT we only need recent candles - limit to 10
                if len(data) > 10:
                    data = data.tail(10)
                    # logger.info(f"📏 CRT: Using last 10 candles for {instrument} {crt_tf}")
                
                # Get the other instrument's data
                other_instrument = [inst for inst in self.instruments if inst != instrument][0]
                other_data = self.market_data[other_instrument].get(crt_tf)
                
                if other_data is None or other_data.empty:
                    continue
                
                # Limit other data to 10 candles too
                if len(other_data) > 10:
                    other_data = other_data.tail(10)
                
                # Detect CRT
                crt_signal = self.crt_detector.calculate_crt_current_candle(
                    data, data, other_data, crt_tf
                )
                
              
                
                if crt_signal:
                    logger.info(f"🔷 CRT DETECTED: {crt_tf} {crt_signal['direction']} on {instrument}")
                    
                    # Check if it's a TPD setup
                    if crt_signal.get('is_tpd', False):
                        logger.info(f"🔄 TPD SETUP DETECTED - Adding to FeatureBox")
                        # Add to FeatureBox - it will handle the signal
                        self.feature_box.add_tpd(crt_signal)
                        # Skip SMT confluence for TPD
                        continue
                    
                    # Otherwise, check for SMT confluence (original logic)
                    allowed_cycles = CRT_SMT_MAPPING.get(crt_tf, [])
                    crt_direction = crt_signal['direction']
                    
                    # Look for active SMTs in allowed cycles
                    for smt_key, smt_feature in self.feature_box.active_features['smt'].items():
                    
                        if self.feature_box._is_feature_expired(smt_feature):
                            continue
                            
                        smt_data = smt_feature['smt_data']
                        smt_cycle = smt_data['cycle']
                        
                        # Check if SMT is allowed cycle and same direction
                        if (smt_cycle in allowed_cycles and 
                            smt_data['direction'] == crt_direction):
                            
                            # Check PSP
                            has_psp = smt_feature['psp_data'] is not None
                            
                            logger.info(f"✅ CRT-SMT CONFLUENCE: {crt_tf} CRT + {smt_cycle} SMT "
                                       f"({crt_direction}, PSP: {has_psp})")
                            
                            # ✅ TRIGGER HAMMER SCANNER (CORRECTED VERSION)
                            try:
                                # Send the original signal first
                                signal_result = self._send_crt_smt_signal(crt_signal, smt_data, has_psp, instrument)
                                
                                if signal_result:
                                    # Prepare trigger data for hammer scanner
                                    trigger_data = {
                                        'type': 'CRT+SMT',
                                        'direction': crt_direction,
                                        'instrument': instrument,
                                        'trigger_timeframe': crt_tf,
                                        'formation_time': datetime.now(NY_TZ),  # CRT is current candle
                                        'signal_data': {
                                            'crt_signal': crt_signal,
                                            'smt_data': smt_data,
                                            'has_psp': has_psp
                                        }
                                    }
                                    
                                    # Trigger hammer scanner
                                    if hasattr(self, 'hammer_scanner') and self.hammer_scanner:
                                        logger.info(f"🔨 Triggering hammer scanner for {instrument}")
                                        self.hammer_scanner.on_signal_detected(trigger_data)
                                    else:
                                        logger.warning(f"⚠️ Hammer scanner not available for {self.pair_group}")
                                        
                            except Exception as e:
                                logger.error(f"Error triggering hammer scanner: {str(e)}")
                                signal_result = False
                            
                            return signal_result
        
        logger.info(f"🔷 No CRT+SMT confluence found")
        return False

        def _cleanup_old_tpd_signals(self):
            """Remove old TPD signals from tracking (7-day cleanup)"""
            if not hasattr(self, 'tpd_signals_sent') or not self.tpd_signals_sent:
                return
            
            current_time = datetime.now(NY_TZ)
            signals_to_remove = []
            
            for signal_id, sent_time in self.tpd_signals_sent.items():
                if (current_time - sent_time).total_seconds() > self.CLEANUP_DAYS:
                    signals_to_remove.append(signal_id)
            
            for signal_id in signals_to_remove:
                del self.tpd_signals_sent[signal_id]
            
            if signals_to_remove:
                logger.debug(f"🧹 Cleaned up {len(signals_to_remove)} old TPD signals (7+ days)")

    def debug_feature_box(self):
        """Debug what's in FeatureBox"""
        logger.info("📦 DEBUGGING FEATURE BOX")
        
        # Check SMTs
        smt_count = len(self.feature_box.active_features['smt'])
        logger.info(f"📦 Active SMTs: {smt_count}")
        
        for smt_key, smt_feature in self.feature_box.active_features['smt'].items():
            smt_data = smt_feature['smt_data']
            expired = self.feature_box._is_feature_expired(smt_feature)
            logger.info(f"📦 SMT: {smt_data['cycle']} {smt_data['direction']} "
                       f"(expired: {expired}, PSP: {smt_feature['psp_data'] is not None})")
    
    def _send_crt_smt_signal(self, crt_signal, smt_data, has_psp, instrument):
        """Send CRT+SMT confluence signal with cooldown"""
        crt_direction = crt_signal['direction']
        crt_tf = crt_signal['timeframe']
        smt_cycle = smt_data['cycle']
        
        # Create unique signal ID
        signal_id = f"CRT_SMT_{self.pair_group}_{crt_tf}_{smt_cycle}_{instrument}_{crt_signal['timestamp'].strftime('%H%M')}"
        
        # Check cooldown (24 hours)
        if hasattr(self, 'crt_smt_ideas_sent') and signal_id in self.crt_smt_ideas_sent:
            last_sent = self.crt_smt_ideas_sent[signal_id]
            if (datetime.now(NY_TZ) - last_sent).total_seconds() < self.COOLDOWN_HOURS:
                logger.info(f"⏳ CRT+SMT 24H COOLDOWN ACTIVE: {signal_id}")
                return False
        
        idea = {
            'type': 'CRT_SMT_CONFLUENCE',
            'pair_group': self.pair_group,
            'direction': crt_direction,
            'crt_timeframe': crt_tf,
            'smt_cycle': smt_cycle,
            'smt_data': smt_data,  # ADD THIS LINE
            'asset': instrument,
            'crt_time': crt_signal['timestamp'],
            'smt_time': smt_data['formation_time'],
            'has_psp': has_psp,
            'strength': 'VERY STRONG' if has_psp else 'STRONG',
            'reasoning': f"{crt_tf} {crt_direction} CRT + {smt_cycle} {crt_direction} SMT confluence",
            'timestamp': datetime.now(NY_TZ),
            'idea_key': signal_id
        }
        
        # Format and send
        message = self._format_crt_smt_message(idea)
        if self._send_telegram_message(message):
            # Initialize if not exists
            if not hasattr(self, 'crt_smt_ideas_sent'):
                self.crt_smt_ideas_sent = {}
            self.crt_smt_ideas_sent[signal_id] = datetime.now(NY_TZ)
            logger.info(f"🚀 CRT-SMT SIGNAL SENT: {crt_tf} CRT + {smt_cycle} SMT {crt_direction}")
            return True
        return False
    
    def _format_crt_smt_message(self, idea):
        """Format CRT+SMT confluence message with SMT details"""
        dir_emoji = "🟢" if idea['direction'] == 'bullish' else "🔴"
        crt_time = idea['crt_time'].strftime('%H:%M')
        smt_time = idea['smt_time'].strftime('%H:%M')
        
        # Get SMT details
        smt_data = idea.get('smt_data', {})
        
        # Extract SMT information
        smt_cycle = idea['smt_cycle']
        quarters = smt_data.get('quarters', '')
        
        # Format quarters
        if quarters:
            quarters_display = quarters.replace('_', '→')
        else:
            quarters_display = ''
        
        # Get asset actions
        asset1_action = smt_data.get('asset1_action', '')
        asset2_action = smt_data.get('asset2_action', '')
        
        # Build SMT details
        smt_details = ""
        if quarters_display or asset1_action or asset2_action:
            smt_details = f"\n*SMT Quarter Details:*\n"
            if quarters_display:
                smt_details += f"• {smt_cycle} {quarters_display}\n"
            if asset1_action:
                smt_details += f"  - {asset1_action}\n"
            if asset2_action:
                smt_details += f"  - {asset2_action}\n"
        
        # Get PSP details if available
        psp_details = ""
        if idea['has_psp'] and 'psp_data' in smt_data:
            psp_data = smt_data['psp_data']
            if psp_data:
                psp_timeframe = psp_data.get('timeframe', '')
                psp_time = psp_data.get('formation_time', '')
                if psp_time and isinstance(psp_time, datetime):
                    psp_time_str = psp_time.strftime('%H:%M')
                    psp_details = f"\n*PSP Details:*\n• Timeframe: {psp_timeframe}\n• Time: {psp_time_str}\n"
        
        return f"""
            🔷 *CRT + SMT CONFLUENCE* 🔷
                
            *Group:* {idea['pair_group'].replace('_', ' ').title()}
            *Direction:* {idea['direction'].upper()} {dir_emoji}
            *Asset:* {idea['asset']}
            *Strength:* {idea['strength']}
                
            *Details:*
            • CRT: {idea['crt_timeframe']} at {crt_time}
            • SMT: {idea['smt_cycle']} cycle at {smt_time}
            • PSP: {'✅ Confirmed' if idea['has_psp'] else '❌ Not Confirmed'}
            {smt_details}{psp_details}
            *Reasoning:* {idea['reasoning']}
                
            *Detection:* {idea['timestamp'].strftime('%H:%M:%S')}
                
            #{idea['pair_group']} #CRT_SMT #{idea['direction']}
        """

        

    def _cleanup_old_crt_smt_signals(self):
        """Remove old CRT+SMT signals from tracking (7-day cleanup)"""
        if not hasattr(self, 'crt_smt_ideas_sent') or not self.crt_smt_ideas_sent:
            return
        
        current_time = datetime.now(NY_TZ)
        signals_to_remove = []
        
        for signal_id, sent_time in self.crt_smt_ideas_sent.items():
            if (current_time - sent_time).total_seconds() > self.CLEANUP_DAYS:  # 7 days
                signals_to_remove.append(signal_id)
        
        for signal_id in signals_to_remove:
            del self.crt_smt_ideas_sent[signal_id]
        
        if signals_to_remove:
            logger.debug(f"🧹 Cleaned up {len(signals_to_remove)} old CRT+SMT signals (7+ days)")
    
    
    def debug_smt_detection(self):
        """Debug why SMTs aren't being detected"""
        logger.info("🔧 DEBUGGING SMT DETECTION")
        
        for cycle in ['weekly', 'daily', '90min']:
            timeframe = self.pair_config['timeframe_mapping'][cycle]
            
            asset1_data = self.market_data[self.instruments[0]].get(timeframe)
            asset2_data = self.market_data[self.instruments[1]].get(timeframe)
            
            if asset1_data is None or asset2_data is None:
                logger.info(f"🔧 {cycle} ({timeframe}): No data")
                continue
            
            logger.info(f"🔧 {cycle} ({timeframe}): {len(asset1_data)} & {len(asset2_data)} candles")
            
            # Check if we have the 'complete' column
            if 'complete' in asset1_data.columns:
                complete_candles = asset1_data[asset1_data['complete'] == True]
                logger.info(f"🔧 Complete candles in {cycle}: {len(complete_candles)}")
            
            # Try to detect SMT
            smt_signal = self.smt_detector.detect_smt_all_cycles(asset1_data, asset2_data, cycle)
            
            if smt_signal:
                logger.info(f"🔧 SMT DETECTED: {cycle} {smt_signal['direction']}")
            else:
                logger.info(f"🔧 NO SMT for {cycle}")
    
    def _generate_good_reasoning(self, fvg_idea, smt_confluence):
        """Generate reasoning for good confluence"""
        reasons = []
        
        # FVG context
        zone = "premium" if fvg_idea['fib_zone'] == 'premium_zone' else "discount"
        reasons.append(f"{zone.upper()} zone {fvg_idea['direction']} FVG for reversal")
        
        # SMT details
        best_smt = smt_confluence['smts'][0]
        reasons.append(f"{best_smt['smt_data']['cycle']} cycle SMT confirming direction")
        
        # Tap confirmation
        reasons.append("SMT tapped the FVG zone")
        
        # Missing PSP
        reasons.append("Note: PSP confirmation missing")
        
        return ". ".join(reasons)
    

    def _create_ultimate_fvg_smt_idea(self, fvg_idea, smt_confluence):
        """Create ULTIMATE FVG + Multiple SMTs with PSP idea"""
        # Get the multiple SMTs
        multiple_smts = smt_confluence['smts'][:2]  # Take first 2 SMTs
        
        idea = {
            'type': 'ULTIMATE_FVG_MULTIPLE_SMTS_PSP',
            'pair_group': self.pair_group,
            'direction': fvg_idea['direction'],
            'asset': fvg_idea['asset'],
            'timeframe': fvg_idea['timeframe'],
            'fvg_name': fvg_idea['fvg_name'],
            'fvg_type': fvg_idea['fvg_type'],
            'fvg_levels': fvg_idea['fvg_levels'],
            'formation_time': fvg_idea['formation_time'],
            'fib_zone': fvg_idea['fib_zone'],
            'smt_count': len(multiple_smts),
            'smt_cycles': [smt['smt_data']['cycle'] for smt in multiple_smts],
            'all_smts_have_psp': all(smt['has_psp'] for smt in multiple_smts),
            'confluence_strength': 'ULTIMATE',
            'reasoning': self._generate_ultimate_reasoning(fvg_idea, smt_confluence),
            'timestamp': datetime.now(NY_TZ),
            'idea_key': f"ULTIMATE_{self.pair_group}_{fvg_idea['asset']}_{fvg_idea['timeframe']}_{datetime.now(NY_TZ).strftime('%H%M')}"
        }
        
        return idea

    
    def _generate_strong_reasoning(self, fvg_idea, smt_confluence):
        """Generate reasoning for strong confluence"""
        reasons = []
        
        # FVG context
        zone = "premium" if fvg_idea['fib_zone'] == 'premium_zone' else "discount"
        reasons.append(f"{zone.upper()} zone {fvg_idea['direction']} FVG for reversal")
        
        # SMT details
        best_smt = smt_confluence['smts'][0]
        reasons.append(f"{best_smt['smt_data']['cycle']} cycle SMT confirming direction")
        
        # PSP confirmation
        if best_smt['has_psp']:
            reasons.append("SMT has PSP confirmation")
        
        return ". ".join(reasons)

    def _is_valid_data(self, df):
        """Check if dataframe is valid for analysis"""
        return (df is not None and 
                isinstance(df, pd.DataFrame) and 
                not df.empty and 
                len(df) >= 10)

    def _find_smt_confluence_for_fvg(self, fvg_idea):
        """Find SMTs that match the FVG's direction and timeframe - WITH the  TAP DEBUG"""
        confluence = {
            'has_confluence': False,
            'smts': [],
            'with_psp': False,
            'tapped_fvg': False
        }
        
        fvg_direction = fvg_idea['direction']
        fvg_timeframe = fvg_idea['timeframe']
        fvg_levels = fvg_idea['fvg_levels']  # "low - high" string
        fvg_low = float(fvg_levels.split(' - ')[0])
        fvg_high = float(fvg_levels.split(' - ')[1])
        
        logger.info(f"🔍 SMT-FVG TAP DEBUG: FVG {fvg_idea['fvg_name']} - Levels: {fvg_low:.4f} to {fvg_high:.4f}")
        
        # Map timeframe to relevant SMT cycles
        timeframe_cycle_map = {
            'H4': ['monthly', 'weekly','daily'],
            'H1': ['weekly', 'daily'],  
            'M15': ['daily', '90min']
        }
        relevant_cycles = timeframe_cycle_map.get(fvg_timeframe, [])
        
        # Check all active SMTs
        for smt_key, smt_feature in self.feature_box.active_features['smt'].items():
            if self.feature_box._is_feature_expired(smt_feature):
                continue
                
            smt_data = smt_feature['smt_data']
            
            # Check if SMT matches direction and cycle
            if (smt_data['direction'] == fvg_direction and 
                smt_data['cycle'] in relevant_cycles):
                
                # DEBUG: Check if SMT actually tapped the FVG
                tapped = self._check_smt_tapped_fvg(smt_data, fvg_idea, fvg_low, fvg_high)
                
                confluence['smts'].append({
                    'smt_data': smt_data,
                    'has_psp': smt_feature['psp_data'] is not None,
                    'tapped_fvg': tapped
                })
                
                logger.info(f"🔍 SMT-FVG TAP: {smt_data['cycle']} {smt_data['direction']} - Tapped FVG: {tapped}")
        
        confluence['has_confluence'] = len(confluence['smts']) > 0
        confluence['with_psp'] = any(smt['has_psp'] for smt in confluence['smts'])
        confluence['tapped_fvg'] = any(smt['tapped_fvg'] for smt in confluence['smts'])
        
        logger.info(f"🔍 SMT Confluence for {fvg_idea['fvg_name']}: {len(confluence['smts'])} SMTs, PSP: {confluence['with_psp']}, Tapped: {confluence['tapped_fvg']}")
        return confluence
    
    def _check_smt_tapped_fvg(self, smt_data, fvg_idea, fvg_low, fvg_high):
        """Check if SMT actually tapped the FVG zone"""
        try:
            # Get the instrument and timeframe for this SMT
            instrument = fvg_idea['asset']  # Same instrument as FVG
            timeframe = smt_data.get('timeframe')
            
            if not timeframe:
                logger.warning(f"⚠️ SMT missing timeframe: {smt_data}")
                return False
            
            # Get FVG formation time from fvg_idea
            fvg_time = fvg_idea.get('formation_time')
            if not fvg_time:
                logger.warning(f"⚠️ FVG missing formation_time: {fvg_idea}")
                return False
            
            # Get the second swing time from SMT data
            swing_times = smt_data.get('swing_times', [])
            if not swing_times or len(swing_times) < 2:
                logger.warning(f"⚠️ SMT missing swing_times: {smt_data}")
                return False
            
            second_swing_time = swing_times[1]  # Second element is the second swing
            
            # CRITICAL: If SMT second swing was BEFORE FVG formation, return False immediately
            if second_swing_time <= fvg_time:
                logger.info(f"❌ SMT REJECTED: Second swing at {second_swing_time} was BEFORE FVG formation at {fvg_time}")
                return False
            
            # Get market data for this instrument and timeframe
            data = self.market_data[instrument].get(timeframe)
            if not self._is_valid_data(data):
                return False
            
            # Get SMT formation time (approximate)
            smt_time = smt_data.get('timestamp')
            if not smt_time:
                return False
            
            # Look for candles around SMT formation time that entered FVG zone
            lookback_candles = 10  # Check 10 candles around SMT formation
            
            # Find the candle index closest to SMT formation time
            time_diffs = abs(data['time'] - smt_time)
            closest_idx = time_diffs.idxmin()
            
            # Check candles around this time for FVG tap
            start_idx = max(0, closest_idx - lookback_candles)
            end_idx = min(len(data) - 1, closest_idx + lookback_candles)
            
            for idx in range(start_idx, end_idx + 1):
                candle = data.iloc[idx]
                
                # Check if price entered FVG zone
                if fvg_idea['direction'] == 'bullish':
                    # For bullish FVG, tap occurs when price (low) <= fvg_high
                    if candle['low'] <= fvg_high:
                        logger.info(f"✅ SMT TAP CONFIRMED: {instrument} {timeframe} - Low {candle['low']:.4f} entered FVG up to {fvg_high:.4f}")
                        return True
                else:  # bearish
                    # For bearish FVG, tap occurs when price (high) >= fvg_low
                    if candle['high'] >= fvg_low:
                        logger.info(f"✅ SMT TAP CONFIRMED: {instrument} {timeframe} - High {candle['high']:.4f} entered FVG from {fvg_low:.4f}")
                        return True
            
            logger.info(f"❌ SMT DID NOT TAP: {instrument} {timeframe} - No candle entered FVG zone")
            return False
            
        except Exception as e:
            logger.error(f"❌ Error checking SMT-FVG tap: {e}")
            return False

    def _scan_sd_with_smt_tap(self):
        """Find Supply/Demand zones where SMT's SECOND SWING traded in the zone - USING FEATUREBOX ZONES"""
        logger.info(f"🔍 SCANNING: Supply/Demand + SMT Tap - USING FEATUREBOX ZONES")
    
        # First, cleanup expired features WITH market data for price invalidation
        if hasattr(self, 'feature_box') and self.feature_box:
            # Pass the same market_data that this function uses for scanning
            self.feature_box.cleanup_expired_features(market_data=self.market_data)
        
        # Rest of your existing code...
        
        # Timeframe mapping: SD Zone -> allowed SMT cycles
        sd_to_smt_cycles = {
            'H4': ['weekly', 'daily','monthly'],      # H4 Zone → Weekly (H1) or Daily (M15) SMTs
            'H1': ['weekly', 'daily','90min'],      # H1 Zone → Weekly (H1) or Daily (M15) SMT  
            'M15': ['daily','90min'],               # M15 Zone → Daily (M15) SMT
            'M5': ['daily','90min'],                 # M5 Zone → 90min (M5) SMT
            'D' :['weekly', 'daily','monthly'],
            'W' :['weekly', 'daily','monthly']
        }
        
        # Get all active SD zones from FeatureBox
        active_zones = self.feature_box.get_active_sd_zones()
        logger.info(f"🔍 Found {len(active_zones)} active SD zones in FeatureBox")
        
        if not active_zones:
            logger.info(f"❌ No active SD zones in FeatureBox")
            return False
        
        # Sort zones by timeframe importance (H4 > H1 > M15 > M5)
        timeframe_order = {'W': 6,'D': 5,'H4': 4, 'H1': 3, 'M15': 2, 'M5': 1}
        active_zones.sort(key=lambda x: timeframe_order.get(x['timeframe'], 0), reverse=True)
        
        for zone in active_zones:
            zone_type = zone['type']  # 'supply' or 'demand'
            zone_direction = 'bearish' if zone_type == 'supply' else 'bullish'  # Convert to direction
            zone_timeframe = zone['timeframe']
            zone_asset = zone['asset']
            zone_low = zone['zone_low']
            zone_high = zone['zone_high']
            zone_formation_time = zone['formation_time']
            
            # logger.info(f"🔍 Checking {zone_type.upper()} zone: {zone['zone_name']} "
                       # f"({zone_low:.4f} - {zone_high:.4f})")
            
            # Get which SMT cycles can tap this zone timeframe
            relevant_cycles = sd_to_smt_cycles.get(zone_timeframe, [])
            
            if not relevant_cycles:
                continue
            
            # Check all active SMTs
            for smt_key, smt_feature in self.feature_box.active_features['smt'].items():
                if self.feature_box._is_feature_expired(smt_feature):
                    continue
                    
                smt_data = smt_feature['smt_data']
                smt_cycle = smt_data['cycle']
                
                # Check if this SMT cycle is allowed
                if smt_cycle not in relevant_cycles:
                    continue
                    
                # ✅ CRITICAL: Check direction match
                # Supply zones need BEARISH SMTs, Demand zones need BULLISH SMTs
                if zone_type == 'supply' and smt_data['direction'] != 'bearish':
                    continue
                if zone_type == 'demand' and smt_data['direction'] != 'bullish':
                    continue
                
                # Check PSP requirement
                has_psp = smt_feature['psp_data'] is not None
                if not has_psp:
                    continue
                
                # ✅ USE THE SAME FUNCTION AS FVG!
                tapped = self._check_cross_tf_smt_second_swing_in_fvg(
                    smt_data, zone_asset, zone_low, zone_high, zone_direction,
                    zone_timeframe, smt_cycle, zone_formation_time
                )
                
                if tapped:
                    # Check for High Probability: Zone within higher TF zone of same direction
                    is_hp_zone = self._check_hp_sd_zone(zone, zone_direction)
                    
                    logger.info(f"✅ SD+SMT TAP CONFIRMED: {smt_cycle} {smt_data['direction']} "
                               f"tapped {zone_timeframe} {zone_type} on {zone_asset}")
                    
                    # ✅ TRIGGER HAMMER SCANNER (CORRECTED VERSION)
                    try:
                        # Send the original signal first
                        signal_result = self._send_sd_smt_tap_signal(
                            zone, smt_data, has_psp, is_hp_zone
                        )
                        
                        if signal_result:
                            # Prepare trigger data for hammer scanner
                            trigger_data = {
                                'type': 'SD+SMT',
                                'direction': zone_direction,
                                'instrument': zone_asset,
                                'trigger_timeframe': zone_timeframe,
                                'formation_time': zone_formation_time,
                                'signal_data': {
                                    'zone': zone,
                                    'smt_data': smt_data,
                                    'is_hp_zone': is_hp_zone,
                                    'has_psp': has_psp
                                }
                            }
                            
                            # Trigger hammer scanner
                            if hasattr(self, 'hammer_scanner') and self.hammer_scanner:
                                logger.info(f"🔨 Triggering hammer scanner for {zone_asset}")
                                self.hammer_scanner.on_signal_detected(trigger_data)
                            else:
                                logger.warning(f"⚠️ Hammer scanner not available for {self.pair_group}")
                                
                    except Exception as e:
                        logger.error(f"Error triggering hammer scanner: {str(e)}")
                        signal_result = False
                    
                    return signal_result
        
        logger.info(f"🔍 No SD+SMT setups found")
        return False

    def _check_smt_tap_in_sd_zone(self, smt_data, asset, zone_low, zone_high, zone_direction, zone_tf, smt_cycle, zone_formation_time):
        """Check if SMT's second swing traded in Supply/Demand zone"""
        try:
            # Get SMT's timeframe from config
            smt_tf = self.pair_config['timeframe_mapping'][smt_cycle]
            
            # Get price data for SMT timeframe
            smt_price_data = self.market_data[asset].get(smt_tf)
            if smt_price_data is None or smt_price_data.empty:
                return False
            
            # Get SMT formation time
            smt_formation_time = smt_data.get('formation_time')
            if not smt_formation_time:
                return False
            
            # Get second swing time
            second_swing_time = smt_data.get('second_swing_time', smt_formation_time)
            
            # Look for candles around second swing time
            time_diffs = abs(smt_price_data['time'] - second_swing_time)
            closest_idx = time_diffs.idxmin()
            
            # Check 5 candles before and after
            start_idx = max(0, closest_idx - 5)
            end_idx = min(len(smt_price_data) - 1, closest_idx + 5)
            
            logger.info(f"🔍 SD Zone Tap: {smt_cycle}({smt_tf}) → {zone_tf} {zone_direction.upper()} zone at {zone_low:.4f}-{zone_high:.4f}")
            
            for idx in range(start_idx, end_idx + 1):
                candle = smt_price_data.iloc[idx]
                
                # Check if candle entered the zone
                if zone_direction == 'bearish':  # Supply zone
                    # Price enters from above, zone tapped if candle's high >= zone_low AND candle's low <= zone_high
                    if candle['high'] >= zone_low and candle['low'] <= zone_high:
                        logger.info(f"✅ SD ZONE TAP: {smt_tf} candle entered supply zone (high: {candle['high']:.4f} >= {zone_low:.4f})")
                        return True
                else:  # Demand zone (bullish)
                    # Price enters from below, zone tapped if candle's low <= zone_high AND candle's high >= zone_low
                    if candle['low'] <= zone_high and candle['high'] >= zone_low:
                        logger.info(f"✅ SD ZONE TAP: {smt_tf} candle entered demand zone (low: {candle['low']:.4f} <= {zone_high:.4f})")
                        return True
            
            logger.info(f"❌ No {smt_tf} candle entered SD zone around second swing")
            return False
            
        except Exception as e:
            logger.error(f"❌ SD zone tap check error: {e}")
            return False
    
    def _check_hp_sd_zone(self, zone, zone_direction):
        """Check if SD zone is High Probability (zone within higher TF zone)"""
        try:
            zone_tf = zone['timeframe']
            zone_asset = zone['asset']
            zone_low = zone['zone_low']
            zone_high = zone['zone_high']
            
            # Map to higher timeframe
            higher_tf_map = {
                'M15': 'H1',
                'H1': 'H4',
                'H4': 'D',
                'D': 'W'
            }
            
            higher_tf = higher_tf_map.get(zone_tf)
            if not higher_tf:
                return False
            
            # Get higher timeframe data
            higher_data = self.market_data[zone_asset].get(higher_tf)
            if higher_data is None or higher_data.empty:
                return False
            
            # Find higher TF zones that contain this zone
            higher_zones = self.sd_detector.scan_timeframe(higher_data, higher_tf, zone_asset)
            
            for higher_zone in higher_zones:
                higher_direction = 'bearish' if higher_zone['type'] == 'supply' else 'bullish'
                
                # Check if same direction and contains our zone
                if higher_direction == zone_direction:
                    if (zone_low >= higher_zone['zone_low'] and 
                        zone_high <= higher_zone['zone_high']):
                        logger.info(f"🎯 HP SD ZONE: {zone_tf} zone is within {higher_tf} zone of same direction")
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ HP SD zone check error: {e}")
            return False

    def _send_sd_smt_tap_signal(self, zone, smt_data, has_psp, is_hp_zone):
        """Send Supply/Demand + SMT tap signal with 24-hour cooldown"""
        zone_type = zone['type']
        zone_direction = 'bearish' if zone_type == 'supply' else 'bullish'
        zone_tf = zone['timeframe']
        smt_cycle = smt_data['cycle']
        
        # Create unique signal ID
        signal_id = f"SD_SMT_TAP_{self.pair_group}_{zone['asset']}_{zone_tf}_{smt_data.get('signal_key', '')}"
        
        # CRITICAL FIX: Check both dictionaries for 24-hour cooldown
        current_time = datetime.now(NY_TZ)
        
        # Check sd_zone_sent
        if signal_id in self.sd_zone_sent:
            last_sent = self.sd_zone_sent[signal_id]
            if (current_time - last_sent).total_seconds() < self.COOLDOWN_HOURS:
                logger.info(f"⏳ SD+SMT 24H COOLDOWN ACTIVE: {signal_id}")
                return False
        
        # Check sd_hp_sent (if it's an HP zone)
        if is_hp_zone and signal_id in self.sd_hp_sent:
            last_sent = self.sd_hp_sent[signal_id]
            if (current_time - last_sent).total_seconds() < self.COOLDOWN_HOURS:
                logger.info(f"⏳ SD HP ZONE 24H COOLDOWN ACTIVE: {signal_id}")
                return False
        
        idea = {
            'type': 'SD_SMT_TAP',
            'pair_group': self.pair_group,
            'direction': zone_direction,
            'zone_type': zone_type,
            'asset': zone['asset'],
            'zone_timeframe': zone_tf,
            'zone_levels': f"{zone['zone_low']:.4f} - {zone['zone_high']:.4f}",
            'zone_formation_time': zone['formation_time'],
            'zone_name': zone['zone_name'],
            'wick_percentage': zone.get('wick_percentage', 0),
            'smt_cycle': smt_cycle,
            'smt_direction': smt_data['direction'],
            'smt_data': smt_data,
            'has_psp': has_psp,
            'is_hp_zone': is_hp_zone,
            'timestamp': datetime.now(NY_TZ),
            'idea_key': signal_id
        }
        
        # Format and send
        message = self._format_sd_smt_tap_message(idea)
        
        if self._send_telegram_message(message):
            # Store in BOTH dictionaries based on type
            self.sd_zone_sent[signal_id] = datetime.now(NY_TZ)
            if is_hp_zone:
                self.sd_hp_sent[signal_id] = datetime.now(NY_TZ)
            logger.info(f"🚀 SD+SMT SIGNAL SENT: {zone['asset']} {zone_tf} {zone_type} + {smt_cycle} SMT")
            return True
        return False
    
    def _debug_market_data(self):
        """Debug market data for FVG analysis"""
        logger.info(f"🔧 FVG MARKET DATA DEBUG for {self.pair_group}")
        
        for instrument in self.instruments:
            instrument_data = self.market_data.get(instrument, {})
            logger.info(f"🔧 {instrument}: {len(instrument_data)} timeframes")
            
            for tf, data in instrument_data.items():
                if data is not None and isinstance(data, pd.DataFrame):
                    status = f"DataFrame({len(data)} rows, {len(data.columns)} cols)"
                    logger.info(f"🔧   {tf}: {status}")
                else:
                    logger.warning(f"🔧   {tf}: INVALID DATA - {type(data)}")

    def _format_sd_smt_tap_message(self, idea):
        """Format Supply/Demand + SMT tap message"""
        direction_emoji = "🔴" if idea['direction'] == 'bearish' else "🟢"
        zone_type_emoji = "📈" if idea['zone_type'] == 'demand' else "📉"
        zone_formation_time = idea['zone_formation_time'].strftime('%m/%d %H:%M')
        
        # SMT details
        smt_data = idea.get('smt_data', {})
        quarters = smt_data.get('quarters', '')
        if quarters:
            quarters_display = quarters.replace('_', '→')
        else:
            quarters_display = ''
        
        asset1_action = smt_data.get('asset1_action', '')
        asset2_action = smt_data.get('asset2_action', '')
        
        hp_emoji = "🎯" if idea['is_hp_zone'] else ""
        psp_emoji = "✅" if idea['has_psp'] else "❌"
        
        # Zone details based on wick percentage
        wick_pct = idea.get('wick_percentage', 0)
        zone_note = ""
        if wick_pct > 60:
            zone_note = f"\n*Note:* Zone adjusted due to large wick ({wick_pct:.1f}%)"
        
        message = f"""
            {zone_type_emoji} *{idea['zone_type'].upper()} ZONE + SMT TAP* {zone_type_emoji}
            
            *Pair Group:* {idea['pair_group'].replace('_', ' ').title()}
            *Direction:* {idea['direction'].upper()} {direction_emoji}
            *Asset:* {idea['asset']}
            *Strength:* {'ULTRA STRONG' if idea['is_hp_zone'] and idea['has_psp'] else 'STRONG'}
            
            *Zone Details:*
            • Type: {idea['zone_type'].upper()}
            • Timeframe: {idea['zone_timeframe']} at {zone_formation_time}
            • Levels: {idea['zone_levels']}
            • HP Zone: {hp_emoji} {'YES' if idea['is_hp_zone'] else 'NO'}
            {zone_note}
            
            *SMT Confluence:*
            • Cycle: {idea['smt_cycle']} {quarters_display}
              - {asset1_action}
              - {asset2_action}
            • PSP: {psp_emoji} {'Confirmed' if idea['has_psp'] else 'Not Confirmed'}
            
            *Detection Time:* {idea['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
            
            #{idea['pair_group']} #{idea['zone_type'].upper()}Zone #{idea['direction']} #{idea['zone_timeframe']}
            """
        return message
    
    def _send_fvg_trade_idea(self, trade_idea):
        """Send formatted FVG-SMT confluence trade idea"""
        idea_key = trade_idea['idea_key']
        
        # Check cooldown (24 hours)
        if idea_key in self.fvg_ideas_sent:
            last_sent = self.fvg_ideas_sent[idea_key]
            if (datetime.now(NY_TZ) - last_sent).total_seconds() < self.COOLDOWN_HOURS:
                logger.debug(f"⏳ FVG idea 24H COOLDOWN ACTIVE: {idea_key}")
                return False
        
        # Format and send message
        message = self._format_fvg_smt_idea_message(trade_idea)
        
        if self._send_telegram_message(message):
            self.fvg_ideas_sent[idea_key] = datetime.now(NY_TZ)
            logger.info(f"🎯 FVG-SMT CONFLUENCE: {trade_idea['fvg_name']} "
                       f"(Score: {trade_idea['confluence_score']}, Confidence: {trade_idea['confidence']:.1%})")
            return True
        
        return False


    

    
    def _format_fvg_smt_idea_message(self, idea):
        """Format FVG-SMT confluence trade idea - REMOVED CONFIDENCE"""
        direction_emoji = "🔴" if idea['direction'] == 'bearish' else "🟢"
        formation_time = idea['formation_time'].strftime('%m/%d %H:%M')
        
        message = f"""
        ⚡ *{idea['type']}* ⚡
        
        *Pair Group:* {idea['pair_group'].replace('_', ' ').title()}
        *Direction:* {idea['direction'].upper()} {direction_emoji}
        *Timeframe:* {idea['timeframe']}
        *Asset:* {idea['asset']}
        *Confluence Strength:* {idea['confluence_strength']}
        
        *FVG Details:*
        • Name: {idea['fvg_name']}
        • Type: {idea['fvg_type'].replace('_', ' ').title()}
        • Levels: {idea['fvg_levels']}
        • Formation: {formation_time}
        • Fibonacci: {idea['fib_zone'].replace('_', ' ').title()}
        
        *SMT Confluence:*
        • Cycle: {idea['smt_cycle']}
        • PSP: {'✅ Confirmed' if idea['smt_has_psp'] else '❌ Not Confirmed'}
        
        *Reasoning:*
        {idea['reasoning']}
        
        *Detection Time:* {idea['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
        
        #FVGSMTConfluence #{idea['pair_group']} #{idea['direction']} #{idea['timeframe']}
        """
        return message
    
    def _send_telegram_message(self, message):
        """Send message via Telegram (using your existing method)"""
        try:
            # Use your existing Telegram sending logic
            if hasattr(self, 'telegram_token') and hasattr(self, 'telegram_chat_id'):
                return send_telegram(message, self.telegram_token, self.telegram_chat_id)
            else:
                logger.warning("⚠️ Telegram credentials not available")
                return False
        except Exception as e:
            logger.error(f"❌ Error sending Telegram message: {str(e)}")
            return False

    
    def _scan_fvg_trade_ideas(self):
        """
        Scan for FVG-based trade ideas with SMT confluence
        """
        try:
            # Scan for FVG trade ideas across all timeframes
            trade_ideas = self.fvg_analyzer.scan_all_timeframes(
                self.market_data, self.pair_group, self.instruments
            )
            
            # Send the best idea
            if trade_ideas:
                best_idea = trade_ideas[0]  # Already sorted by confidence
                
                # Only send if confidence is high enough and not recently sent
                if best_idea['confidence'] >= 0.7:
                    return self._send_fvg_trade_idea(best_idea)
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Error scanning FVG trade ideas: {str(e)}")
            return False
    
    def _send_fvg_trade_idea(self, trade_idea):
        """Send formatted FVG trade idea"""
        idea_key = trade_idea['idea_key']
        
        # Check if we recently sent similar idea (1 hour cooldown)
        if idea_key in self.fvg_ideas_sent:
            last_sent = self.fvg_ideas_sent[idea_key]
            if (datetime.now(NY_TZ) - last_sent).total_seconds() < 3600:
                logger.debug(f"⏳ FVG idea recently sent: {idea_key}")
                return False
        
        # Format and send the message
        message = self._format_fvg_idea_message(trade_idea)
        
        if self._send_telegram_message(message):
            self.fvg_ideas_sent[idea_key] = datetime.now(NY_TZ)
            # logger.info(f"🎯 FVG TRADE IDEA SENT: {trade_idea['fvg_name']} "
            #            f"(Confidence: {trade_idea['confidence']:.1%})")
            return True
        
        return False
    
    def _format_fvg_idea_message(self, idea):
        """Format FVG trade idea for Telegram - REMOVED CONFIDENCE"""
        direction_emoji = "🔴" if idea['direction'] == 'bearish' else "🟢"
        formation_time = idea['formation_time'].strftime('%m/%d %H:%M')
        
        # Determine FVG type emoji and description
        if idea.get('is_inversion', False):
            fvg_emoji = "🔄"
            fvg_type_desc = "INVERSION FVG"
        elif idea.get('is_hp_fvg', False):
            fvg_emoji = "🎯" 
            fvg_type_desc = "HIGH PROBABILITY FVG"
        else:
            fvg_emoji = "⚡"
            fvg_type_desc = "REGULAR FVG"
        
        message = f"""
        {fvg_emoji} *{fvg_type_desc}* {fvg_emoji}
        
        *Pair Group:* {idea['pair_group'].replace('_', ' ').title()}
        *Direction:* {idea['direction'].upper()} {direction_emoji}
        *Timeframe:* {idea['timeframe']}
        *Asset:* {idea['asset']}
        *Confluence:* {idea['confluence_strength']}
        
        *FVG Details:*
        • Name: {idea['fvg_name']}
        • Type: {idea['fvg_type'].replace('_', ' ').title()}
        • Class: {idea.get('fvg_class', 'regular').upper()}
        • Levels: {idea['fvg_levels']}
        • Formation: {formation_time}
        • Fibonacci Zone: {idea['fib_zone'].replace('_', ' ').title()}
        • HP FVG: {'✅ YES' if idea.get('is_hp_fvg', False) else '❌ NO'}
        • Inversion: {'✅ YES' if idea.get('is_inversion', False) else '❌ NO'}
        
        *Reasoning:*
        {idea['reasoning']}
        
        *Detection Time:* {idea['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
        
        #FVG #{idea['pair_group']} #{idea['direction']} #{idea['timeframe']}
        """
        return message

    def _scan_fvg_smt_confluence(self):
        """Cross-timeframe: Lower TF SMT taps into Higher TF FVG"""
        try:
            logger.info(f"🔍 CROSS-TF FVG+SMT SCAN for {self.pair_group}")
            
            # DEFINE CROSS-TF MAPPING
            smt_to_fvg_tfs = {
                'weekly': ['H4'],      # Weekly SMT → H4 FVGs
                'daily': ['H4', 'H1'], # Daily SMT → H4 & H1 FVGs
                '90min': ['M15']       # 90min SMT → M15 FVGs
            }
            
            # Get FVGs for ALL instruments and timeframes first
            all_fvgs = {}
            for inst in self.instruments:
                all_fvgs[inst] = {}
                for fvg_tf in ['H4', 'H1', 'M15']:
                    data = self.market_data[inst].get(fvg_tf)
                    if data is not None and not data.empty:
                        fvgs = self.fvg_detector.scan_tf(data, fvg_tf, inst)
                        all_fvgs[inst][fvg_tf] = fvgs
                        logger.info(f"🔍 {inst} {fvg_tf}: Found {len(fvgs)} FVGs")
            
            # Check all active SMTs
            for smt_key, smt_feature in list(self.feature_box.active_features['smt'].items()):
                if self.feature_box._is_feature_expired(smt_feature):
                    continue
                    
                smt_data = smt_feature['smt_data']
                smt_cycle = smt_data['cycle']
                smt_direction = smt_data['direction']
                has_psp = smt_feature['psp_data'] is not None
                
                # Get which FVG timeframes to check for this SMT cycle
                fvg_timeframes_to_check = smt_to_fvg_tfs.get(smt_cycle, [])
                
                logger.info(f"🔍 Checking {smt_cycle} SMT against FVGs: {fvg_timeframes_to_check}")
                
                # Check each instrument for matching FVGs
                for instrument in self.instruments:
                    for fvg_tf in fvg_timeframes_to_check:
                        fvgs = all_fvgs.get(instrument, {}).get(fvg_tf, [])
                        
                        # Filter for same direction FVGs
                        same_dir_fvgs = [f for f in fvgs if f['direction'] == smt_direction]
                        
                        for fvg in same_dir_fvgs:
                            # Check if SMT tapped this FVG (cross-timeframe check)
                            tapped = self._check_cross_tf_smt_tap(
                                fvg, smt_data, instrument, fvg_tf, smt_cycle
                            )
                            
                            if tapped and has_psp:
                                # Check if HP FVG (only one asset has FVG)
                                is_hp = fvg.get('is_hp', False)
                                
                                logger.info(f"✅ CROSS-TF CONFLUENCE: {smt_cycle} SMT tapped {fvg_tf} FVG on {instrument}")
                                return self._send_fvg_smt_tap_signal(fvg, smt_data, has_psp, is_hp)
            
            # Fallback to double SMTs
            logger.info(f"🔍 No Cross-TF FVG+SMT - checking doubles")
            return self._scan_double_smts_temporal()
            
        except Exception as e:
            logger.error(f"❌ Cross-TF FVG+SMT scan: {str(e)}", exc_info=True)
            return False

    
    
    def _scan_fvg_with_smt_tap(self):
        """Find FVGs where SMT's SECOND SWING traded in FVG zone - USING CLOSED CANDLES ONLY"""
        
        # First, cleanup expired features
        if hasattr(self, 'feature_box') and self.feature_box:
            self.feature_box.cleanup_expired_features()
        
        # CORRECT CROSS-TIMEFRAME MAPPING
        fvg_to_smt_cycles = {
            'H4': ['weekly', 'daily'],    # H4 FVG → Weekly (H1) or Daily (M15) SMT
            'H1': ['daily'],              # H1 FVG → Daily (M15) SMT
            'M15': ['daily', '90min']     # M15 FVG → Daily (M15) or 90min (M5) SMT
        }
        
        # Let's scan for FVGs using FVGDetector with ONLY CLOSED CANDLES
        all_fvgs = []
        
        for instrument in self.instruments:
            for fvg_tf in ['H4', 'H1', 'M15']:  # Only these FVG timeframes
                data = self.market_data[instrument].get(fvg_tf)
                if data is not None and not data.empty:
                    # ✅ NEW: Use only CLOSED candles for FVG detection
                    if 'complete' in data.columns:
                        closed_data = data[data['complete'] == True].copy()
                        
                        if len(closed_data) < 5:  # Need at least 5 closed candles
                            continue
                        
                        # Use FVGDetector to get FVGs from CLOSED CANDLES ONLY
                        fvgs = self.fvg_detector.scan_tf(closed_data, fvg_tf, instrument)
                    else:
                        # Fallback if no 'complete' column
                        fvgs = self.fvg_detector.scan_tf(data, fvg_tf, instrument)
                    
                    for fvg in fvgs:
                        # Convert FVGDetector format to the format expected by this method
                        fvg_idea = {
                            'fvg_name': f"{instrument}_{fvg_tf}_{fvg['formation_time'].strftime('%m%d%H%M')}",
                            'fvg_levels': f"{fvg['fvg_low']:.4f} - {fvg['fvg_high']:.4f}",
                            'direction': fvg['direction'],
                            'timeframe': fvg_tf,
                            'asset': instrument,
                            'formation_time': fvg['formation_time'],
                            'fvg_low': fvg['fvg_low'],  # Keep for easy access
                            'fvg_high': fvg['fvg_high'],
                            'is_hp': fvg.get('is_hp', False),
                            'is_closed_candle': True  # Mark as from closed candle
                        }
                        all_fvgs.append(fvg_idea)
        
        for fvg_idea in all_fvgs:
            fvg_direction = fvg_idea['direction']
            fvg_timeframe = fvg_idea['timeframe']
            fvg_asset = fvg_idea['asset']
            fvg_low = fvg_idea['fvg_low']
            fvg_high = fvg_idea['fvg_high']
            fvg_formation_time = fvg_idea['formation_time']  # Get FVG formation time
            
            # Get which SMT cycles can tap this FVG timeframe
            relevant_cycles = fvg_to_smt_cycles.get(fvg_timeframe, [])
            
            # Check all active SMTs
            for smt_key, smt_feature in self.feature_box.active_features['smt'].items():
                if self.feature_box._is_feature_expired(smt_feature):
                    continue
                    
                smt_data = smt_feature['smt_data']
                smt_cycle = smt_data['cycle']
                
                # Check if this SMT cycle is allowed to tap this FVG timeframe
                if smt_cycle not in relevant_cycles:
                    continue
                    
                # Check direction match
                if smt_data['direction'] != fvg_direction:
                    continue
                
                # ✅ CHECK PSP REQUIREMENT FIRST
                has_psp = smt_feature['psp_data'] is not None
                
                # CRITICAL: Check temporal relationship BEFORE checking tap
                # Get swing_times - it's a dictionary, not a list!
                swing_times = smt_data.get('swing_times', {})
                
                # Determine which asset key to use
                if fvg_asset == self.instruments[0]:
                    asset_key = 'asset1_curr'
                else:
                    asset_key = 'asset2_curr'
                
                # Get the current swing for this asset
                asset_curr = swing_times.get(asset_key, {})
                
                if not asset_curr:
                    continue
                
                # Extract second swing time (could be dict or Timestamp)
                if isinstance(asset_curr, dict):
                    second_swing_time = asset_curr.get('time')
                else:
                    second_swing_time = asset_curr  # Assuming it's already a Timestamp
                
                if not second_swing_time:
                    continue
                
                # Check if SMT's second swing traded in FVG zone (CROSS-TIMEFRAME)
                tapped = self._check_cross_tf_smt_second_swing_in_fvg(
                    smt_data, fvg_asset, fvg_low, fvg_high, fvg_direction, 
                    fvg_timeframe, smt_cycle, fvg_formation_time  # PASS FVG FORMATION TIME!
                )
                
                if tapped:
                    # Check if only ONE asset tapped (HP FVG)
                    is_hp_fvg = self._check_hp_fvg_fix(fvg_idea, fvg_asset)
                    
                    # ✅ TRIGGER HAMMER SCANNER (CORRECTED VERSION)
                    try:
                        # Send the original signal first
                        signal_result = self._send_fvg_smt_tap_signal(
                            fvg_idea, smt_data, has_psp, is_hp_fvg
                        )
                        
                        if signal_result:
                            # Prepare trigger data for hammer scanner
                            trigger_data = {
                                'type': 'FVG+SMT',
                                'direction': fvg_direction,
                                'instrument': fvg_asset,
                                'trigger_timeframe': fvg_timeframe,
                                'formation_time': fvg_formation_time,
                                'signal_data': {
                                    'fvg_idea': fvg_idea,
                                    'smt_data': smt_data,
                                    'is_hp_fvg': is_hp_fvg,
                                    'has_psp': has_psp
                                }
                            }
                            
                            # Trigger hammer scanner
                            if hasattr(self, 'hammer_scanner') and self.hammer_scanner:
                                self.hammer_scanner.on_signal_detected(trigger_data)
                            else:
                                # Keep this warning as it indicates a configuration issue
                                logger.warning(f"Hammer scanner not available for {self.pair_group}")
                                
                    except Exception as e:
                        # Keep error logging for exceptions
                        logger.error(f"Error triggering hammer scanner: {str(e)}")
                        signal_result = False
                    
                    return signal_result
        
        return False

    def _check_cross_tf_smt_second_swing_in_fvg(self, smt_data, asset, fvg_low, fvg_high, fvg_direction, fvg_tf, smt_cycle, fvg_formation_time):
        """Check if SMT's second swing (on lower TF) entered FVG zone (on higher TF)"""
        try:
            # Get SMT's timeframe from config
            smt_tf = self.pair_config['timeframe_mapping'][smt_cycle]
            
            # Get price data for SMT timeframe (LOWER timeframe)
            smt_price_data = self.market_data[asset].get(smt_tf)
            if smt_price_data is None or smt_price_data.empty:
                logger.info(f"❌ No {smt_tf} data for {asset}")
                return False
            
            # Get the swing_times dictionary
            swing_times = smt_data.get('swing_times', {})
            
            # Determine which asset key to use
            if asset == self.instruments[0]:
                asset_key = 'asset1_curr'
            else:
                asset_key = 'asset2_curr'
            
            # Get the current swing for this asset
            asset_curr = swing_times.get(asset_key, {})
            
            if not asset_curr:
                logger.info(f"❌ No swing data for {asset} in SMT {smt_cycle}")
                return False
            
            # Extract second swing time
            if isinstance(asset_curr, dict):
                second_swing_time = asset_curr.get('time')
            else:
                second_swing_time = asset_curr  # Assuming it's already a Timestamp
            
            if not second_swing_time:
                logger.info(f"❌ No second swing time for {asset} in SMT {smt_cycle}")
                return False
            
            # CRITICAL: Check if second swing happens AFTER FVG formation
            if second_swing_time <= fvg_formation_time:
                # logger.info(f"❌ CROSS-TF REJECTED: SMT second swing at {second_swing_time} is BEFORE FVG formation at {fvg_formation_time}")
                return False
            
            # Look for candles around second swing time in SMT timeframe
            time_diffs = abs(smt_price_data['time'] - second_swing_time)
            closest_idx = time_diffs.idxmin()
            
            # Check 2 candles before  (adjust as needed)
            start_idx = max(0, closest_idx - 2)
            end_idx = min(len(smt_price_data) - 1, closest_idx + 0)
            
            # logger.info(f"🔍 Cross-TF Tap: {smt_cycle}({smt_tf}) → {fvg_tf} FVG at {fvg_low:.4f}-{fvg_high:.4f}")
            # logger.info(f"   Checking {smt_tf} candles around {second_swing_time.strftime('%H:%M')}")
            
            for idx in range(start_idx, end_idx + 1):
                candle = smt_price_data.iloc[idx]
                
                # Check if this candle entered the FVG zone
                if fvg_direction == 'bullish':
                    # Bullish FVG: zone is above, price enters from below
                    # Zone tapped if candle's low <= fvg_high AND candle's high >= fvg_low
                    if candle['low'] <= fvg_high and candle['high'] >= fvg_low:
                        entry_price = candle['low']  # Price entered from bottom
                        # logger.info(f"✅ CROSS-TF TAP: {smt_tf} candle at {candle['time'].strftime('%H:%M')} "
                                   # f"entered bullish FVG zone (low: {candle['low']:.4f} <= {fvg_high:.4f})")
                        return True
                else:  # bearish
                    # Bearish FVG: zone is below, price enters from above  
                    # Zone tapped if candle's high >= fvg_low AND candle's low <= fvg_high
                    if candle['high'] >= fvg_low and candle['low'] <= fvg_high:
                        entry_price = candle['high']  # Price entered from top
                        # logger.info(f"✅ CROSS-TF TAP: {smt_tf} candle at {candle['time'].strftime('%H:%M')} "
                                   # f"entered bearish FVG zone (high: {candle['high']:.4f} >= {fvg_low:.4f})")
                        return True
            
            # logger.info(f"❌ No {smt_tf} candle entered FVG zone around second swing")
            return False
            
        except Exception as e:
            logger.error(f"❌ Cross-TF tap check error: {e}")
            return False
    
    def _check_smt_second_swing_in_fvg(self, smt_data, asset, fvg_low, fvg_high, direction, fvg_formation=None):
        """Check SMT 2nd swing taps FVG zone (post-formation, safe extract)."""
        try:
            cycle = smt_data['cycle']
            timeframe = self.pair_config['timeframe_mapping'][cycle]
            data = self.market_data[asset].get(timeframe)
            if not self._is_valid_data(data):
                logger.info(f"TRACE TAP SKIP: Invalid data {asset} {timeframe}")
                return False
            
            # Safe extract 2nd swing (curr swing, handle dict or Timestamp)
            swing_times = smt_data.get('swing_times', {})
            asset_curr = swing_times.get('asset1_curr' if asset == self.instruments[0] else 'asset2_curr', {})
            if isinstance(asset_curr, dict):
                second_swing_time = asset_curr.get('time', smt_data.get('formation_time'))
                second_swing_price = asset_curr.get('price', data['close'].iloc[-1] if not data.empty else 0)
            else:
                # If Timestamp direct
                second_swing_time = asset_curr if isinstance(asset_curr, (pd.Timestamp, datetime)) else smt_data.get('formation_time')
                second_swing_price = data['close'].iloc[-1] if not data.empty else 0
            
            if not second_swing_time:
                logger.warning(f"TRACE TAP SKIP: No second swing time in {smt_data['signal_key']}")
                return False
            
            # CRITICAL FIX: Check if second swing happens AFTER FVG formation
            if fvg_formation and second_swing_time <= fvg_formation:
                # logger.info(f"❌ TRACE TAP REJECTED: SMT second swing at {second_swing_time} is BEFORE FVG formation at {fvg_formation}")
                return False
            
            # logger.info(f"TRACE TAP {smt_data['cycle']} on {asset} FVG: 2nd swing {second_swing_time.strftime('%H:%M')} price {second_swing_price:.4f}")
            
            # Post-formation filter
            if fvg_formation:
                data_post = data[data['time'] >= fvg_formation]
                logger.info(f"TRACE TAP: {len(data_post)} post-FVG candles from {fvg_formation}")
                if data_post.empty:
                    logger.info(f"TRACE TAP SKIP: No post-FVG data")
                    return False
            else:
                data_post = data
            
            # Lookback window around second swing
            time_diffs = abs(data_post['time'] - second_swing_time)
            closest_idx = time_diffs.idxmin()
            start_idx = max(0, closest_idx - 3)
            end_idx = min(len(data_post) - 1, closest_idx + 3)
            
            logger.info(f"TRACE TAP window: {start_idx}-{end_idx} around {second_swing_time.strftime('%H:%M')}")
            for idx in range(start_idx, end_idx + 1):
                candle = data_post.iloc[idx]
                logger.info(f"TRACE TAP candle {candle['time'].strftime('%H:%M')}: low/high {candle['low']:.4f}/{candle['high']:.4f} vs zone {fvg_low}-{fvg_high} ({direction})")
                if direction == 'bullish' and candle['low'] <= fvg_high:
                    logger.info(f"✅ TRACE TAP HIT: Low {candle['low']:.4f} <= high {fvg_high:.4f}")
                    return True
                if direction == 'bearish' and candle['high'] >= fvg_low:
                    logger.info(f"✅ TRACE TAP HIT: High {candle['high']:.4f} >= low {fvg_low:.4f}")
                    return True
            
            logger.info(f"TRACE TAP MISS: No entry in window")
            return False
            
        except Exception as e:
            logger.error(f"TRACE TAP ERROR: {e}")
            return False

    
    def _extract_smt_second_swing_time(self, smt_data):
        """Extract the second swing time from SMT data"""
        # Try different possible field names
        if 'second_swing_time' in smt_data:
            return smt_data['second_swing_time']
        elif 'recent_swing_time' in smt_data:
            return smt_data['recent_swing_time']
        elif 'timestamp' in smt_data:
            # This is usually when the SMT was detected
            return smt_data['timestamp']
        
        # If we can't find it, try to parse from signal_key
        signal_key = smt_data.get('signal_key', '')
        if 'q_' in signal_key and 'q_' in signal_key:
            # Try to extract time from something like "1600_0300"
            import re
            times = re.findall(r'(\d{4})', signal_key)
            if times and len(times) >= 2:
                # The second time is the recent swing
                hour = int(times[1][:2])
                minute = int(times[1][2:])
                # Create approximate time (today)
                now = datetime.now(NY_TZ)
                return now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        
        return None
    
    def _send_telegram_message(self, message):
        """Send message via Telegram (using your existing method)"""
        try:
            # Use your existing Telegram sending logic
            if hasattr(self, 'telegram_token') and hasattr(self, 'telegram_chat_id'):
                return send_telegram(message, self.telegram_token, self.telegram_chat_id)
            else:
                logger.warning("⚠️ Telegram credentials not available")
                return False
        except Exception as e:
            logger.error(f"❌ Error sending Telegram message: {str(e)}")
            return False
    
    async def _scan_and_add_features(self):
        """Scan for all features and add to Feature Box immediately"""
        # Scan for SMTs
        await self._scan_all_smt_for_feature_box()
        
        # ✅ UNCOMMENT THESE LINES!
        # Scan for CRTs  
        await self._scan_crt_signals_for_feature_box()
        
        # Scan for PSPs for existing SMTs
        await self._scan_psp_for_existing_smts_feature_box()
    
    async def _scan_all_smt_for_feature_box(self):
        """Scan for SMTs and add to Feature Box immediately - FIXED"""
        cycles = ['monthly', 'weekly', 'daily', '90min']
        
        for cycle in cycles:
            timeframe = self.pair_config['timeframe_mapping'][cycle]
            asset1_data = self.market_data[self.instruments[0]].get(timeframe)
            asset2_data = self.market_data[self.instruments[1]].get(timeframe)
            
            # ✅ FIXED: Explicit DataFrame checking
            if (asset1_data is None or not isinstance(asset1_data, pd.DataFrame) or asset1_data.empty or
                asset2_data is None or not isinstance(asset2_data, pd.DataFrame) or asset2_data.empty):
                logger.debug(f"⚠️ No data for {cycle} ({timeframe}) - skipping SMT scan")
                continue
            
            logger.info(f"🔍 Scanning {cycle} cycle ({timeframe}) for SMT...")
            smt_signal = self.smt_detector.detect_smt_all_cycles(asset1_data, asset2_data, cycle)
            
            if smt_signal:
                # Check for PSP immediately
                psp_signal = self.smt_detector.check_psp_for_smt(smt_signal, asset1_data, asset2_data)
                
                # Add to Feature Box (triggers immediate confluence check)
                self.feature_box.add_smt(smt_signal, psp_signal)

    def _check_hp_fvg_fix(self, fvg_idea, tapped_asset):
        """Check if only ONE asset tapped the FVG (using FVGDetector data)"""
        other_asset = [inst for inst in self.instruments if inst != tapped_asset][0]
        fvg_time = fvg_idea['formation_time']
        timeframe = fvg_idea['timeframe']
        
        # Get FVGs for other asset
        other_data = self.market_data[other_asset].get(timeframe)
        if other_data is None or other_data.empty:
            # If no data for other asset, can't determine - assume NOT HP to be safe
            return False
        
        # Scan for FVGs in other asset
        other_fvgs = self.fvg_detector.scan_tf(other_data, timeframe, other_asset)
        
        # Check if other asset has FVG around same time (±2 hours)
        for other_fvg in other_fvgs:
            time_diff = abs((other_fvg['formation_time'] - fvg_time).total_seconds() / 3600)
            if time_diff < 2:  # Within 2 hours
                logger.info(f"❌ NOT HP FVG: Both assets have FVGs within 2 hours")
                return False  # Both have FVGs, not HP
        
        logger.info(f"✅ HP FVG: Only {tapped_asset} has FVG")
        return True  # Only this asset has FVG

    def _scan_double_smts_temporal(self):
        """Check for double SMT signals with cooldown"""
        logger.info(f"🔍 SCANNING: Double SMTs (Temporal)")
        
        # Group SMTs by direction and sort by cycle importance
        cycle_order = {'weekly': 3, 'daily': 2, '90min': 1, 'monthly': 4}
        
        # Check bullish SMTs
        bullish_smts = []
        for smt_key, smt_feature in self.feature_box.active_features['smt'].items():
            if self.feature_box._is_feature_expired(smt_feature):
                continue
                
            smt_data = smt_feature['smt_data']
            if smt_data['direction'] == 'bullish':
                # Create a copy with both SMT and PSP data
                smt_data_copy = smt_data.copy()
                smt_data_copy['has_psp'] = smt_feature['psp_data'] is not None
                smt_data_copy['signal_key'] = smt_key
                
                # If there's PSP data, include it
                if smt_feature['psp_data']:
                    smt_data_copy['psp_data'] = smt_feature['psp_data']
                bullish_smts.append(smt_data_copy)
        
        # Sort by cycle importance
        bullish_smts.sort(key=lambda x: cycle_order.get(x['cycle'], 0), reverse=True)
        
        # Check for double SMTs
        if len(bullish_smts) >= 2:
            primary = bullish_smts[0]
            secondary = bullish_smts[1]
            
            # Check cycles are different
            if primary['cycle'] != secondary['cycle']:
                # Create unique signal ID
                signal_id = f"DOUBLE_SMT_{self.pair_group}_{primary['cycle']}_{secondary['cycle']}_bullish"
                
                # Check cooldown (24 hours)
                if hasattr(self, 'double_smt_sent') and signal_id in self.double_smt_sent:
                    last_sent = self.double_smt_sent[signal_id]
                    if (datetime.now(NY_TZ) - last_sent).total_seconds() < self.COOLDOWN_HOURS:
                        logger.info(f"⏳ Double SMT 24H COOLDOWN ACTIVE: {signal_id}")
                        return False
                
                # Calculate span
                span_minutes = 0
                if 'formation_time' in primary and 'formation_time' in secondary:
                    span_minutes = abs((primary['formation_time'] - secondary['formation_time']).total_seconds() / 60)
                
                idea = {
                    'pair_group': self.pair_group,
                    'direction': 'bullish',
                    'primary_smt': primary,
                    'secondary_smt': secondary,
                    'span_minutes': span_minutes,
                    'reasoning': f"{primary['cycle']} bullish SMT + {secondary['cycle']} confirm (span: {span_minutes:.1f}min from 2nd swings)",
                    'detection_time': datetime.now(NY_TZ),
                    'idea_key': signal_id
                }
                
                message = self._format_double_smt_message(idea)
                if self._send_telegram_message(message):
                    # Initialize if not exists
                    if not hasattr(self, 'double_smt_sent'):
                        self.double_smt_sent = {}
                    self.double_smt_sent[signal_id] = datetime.now(NY_TZ)
                    logger.info(f"🚀 DOUBLE SMT SIGNAL SENT: {primary['cycle']} + {secondary['cycle']} bullish")
                    return True
        
        # Similar logic for bearish SMTs
        bearish_smts = []
        for smt_key, smt_feature in self.feature_box.active_features['smt'].items():
            if self.feature_box._is_feature_expired(smt_feature):
                continue
                
            smt_data = smt_feature['smt_data']
            if smt_data['direction'] == 'bearish':
                smt_data_copy = smt_data.copy()
                smt_data_copy['has_psp'] = smt_feature['psp_data'] is not None
                smt_data_copy['signal_key'] = smt_key
                bearish_smts.append(smt_data_copy)
        
        bearish_smts.sort(key=lambda x: cycle_order.get(x['cycle'], 0), reverse=True)
        
        if len(bearish_smts) >= 2:
            primary = bearish_smts[0]
            secondary = bearish_smts[1]
            
            if primary['cycle'] != secondary['cycle']:
                signal_id = f"DOUBLE_SMT_{self.pair_group}_{primary['cycle']}_{secondary['cycle']}_bearish"
                
                if hasattr(self, 'double_smt_sent') and signal_id in self.double_smt_sent:
                    last_sent = self.double_smt_sent[signal_id]
                    if (datetime.now(NY_TZ) - last_sent).total_seconds() < 3600:
                        logger.info(f"⏳ Double SMT recently sent: {signal_id}")
                        return False
                
                span_minutes = 0
                if 'formation_time' in primary and 'formation_time' in secondary:
                    span_minutes = abs((primary['formation_time'] - secondary['formation_time']).total_seconds() / 60)
                
                idea = {
                    'pair_group': self.pair_group,
                    'direction': 'bearish',
                    'primary_smt': primary,
                    'secondary_smt': secondary,
                    'span_minutes': span_minutes,
                    'reasoning': f"{primary['cycle']} bearish SMT + {secondary['cycle']} confirm (span: {span_minutes:.1f}min from 2nd swings)",
                    'detection_time': datetime.now(NY_TZ),
                    'idea_key': signal_id
                }
                
                message = self._format_double_smt_message(idea)
                if self._send_telegram_message(message):
                    if not hasattr(self, 'double_smt_sent'):
                        self.double_smt_sent = {}
                    self.double_smt_sent[signal_id] = datetime.now(NY_TZ)
                    logger.info(f"🚀 DOUBLE SMT SIGNAL SENT: {primary['cycle']} + {secondary['cycle']} bearish")
                    return True
        
        return False

    

    async def _fetch_all_data(self, api_key):
        """Fetch data with PROVEN candle counts - FIXED DataFrame checks"""
        
        # PROVEN CANDLE COUNTS from the working script
        proven_counts = {
            'H4': 40,   # Monthly timeframe
            'H1': 40,   # Weekly timeframe  
            'M15': 40,   # Daily timeframe
            'M5': 40,    # 90min timeframe
            'H2': 10, 'H3': 10, 'H6': 10, 'H8': 10, 'H12': 10
        }
        
        required_timeframes = list(self.pair_config['timeframe_mapping'].values())
        
        # ALWAYS include CRT timeframes for better detection
        for tf in CRT_TIMEFRAMES:
            if tf not in required_timeframes:
                required_timeframes.append(tf)
        
        # FIXED: Use self.instruments instead of [self.pair1, self.pair2]
        for instrument in self.instruments:
            for tf in required_timeframes:
                # Use proven count if available, otherwise default to 100
                count = proven_counts.get(tf, 100)
                
                try:
                    df = await asyncio.get_event_loop().run_in_executor(
                        None, fetch_candles, instrument, tf, count, api_key
                    )
                    
                    # ✅ FIXED: Explicit DataFrame checking
                    if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
                        self.market_data[instrument][tf] = df
                        logger.debug(f"📥 Fetched {len(df)} {tf} candles for {instrument} (requested: {count})")
                    else:
                        # ✅ FIXED: Check what exactly is returned
                        if df is None:
                            logger.warning(f"⚠️ NULL data received for {instrument} {tf}")
                        elif not isinstance(df, pd.DataFrame):
                            logger.warning(f"⚠️ Non-DataFrame returned for {instrument} {tf}: {type(df)}")
                        else:
                            logger.warning(f"⚠️ Empty DataFrame for {instrument} {tf}")
                            
                except Exception as e:
                    logger.error(f"❌ Error fetching {instrument} {tf}: {str(e)}")

    def debug_data_structure(self):
        """Debug method to check data structure and identify problems"""
        logger.info(f"🔧 DEBUG: Checking data structure for {self.pair_group}")
        
        for instrument in self.instruments:
            logger.info(f"🔧 DEBUG: Instrument {instrument} has {len(self.market_data.get(instrument, {}))} timeframes")
            
            for tf, data in self.market_data.get(instrument, {}).items():
                status = "❌ NULL" if data is None else f"✅ DataFrame({data.shape})" if isinstance(data, pd.DataFrame) else f"⚠️ {type(data)}"
                logger.info(f"🔧 DEBUG:   {tf}: {status}")
                
                if isinstance(data, pd.DataFrame):
                    logger.info(f"🔧 DEBUG:     Columns: {list(data.columns)}, Empty: {data.empty}")

    def _get_proven_count(self, timeframe):
        """Get proven candle counts"""
        proven_counts = {
            'H4': 40,  # Monthly
            'H1': 40,  # Weekly
            'M15': 40,  # Daily
            'M5': 45,  # 90min
            'H2': 10, 'H3': 10, 'H6': 10, 'H8': 10, 'H12': 10
        }
        return proven_counts.get(timeframe, 100)



    # KEEPING ONLY ONE VERSION OF TRIAD METHODS (the better one)
    async def _analyze_triad(self, api_key):
        """Analyze triad of 3 instruments - check all pair combinations with better error handling"""
        if len(self.instruments) != 3:
            logger.error(f"❌ _analyze_triad called but only {len(self.instruments)} instruments")
            return None
            
        instrument_a, instrument_b, instrument_c = self.instruments
        
        # Analyze all pairs: AB, AC, BC
        signals = []
        
        try:
            # Pair AB
            signal_ab = await self._analyze_pair_combo(instrument_a, instrument_b, "AB")
            if signal_ab and isinstance(signal_ab, dict):
                signals.append(('AB', signal_ab))
            else:
                logger.debug(f"🔍 No valid signal for AB pair")
        
            # Pair AC  
            signal_ac = await self._analyze_pair_combo(instrument_a, instrument_c, "AC")
            if signal_ac and isinstance(signal_ac, dict):
                signals.append(('AC', signal_ac))
            else:
                logger.debug(f"🔍 No valid signal for AC pair")
        
            # Pair BC
            signal_bc = await self._analyze_pair_combo(instrument_b, instrument_c, "BC")
            if signal_bc and isinstance(signal_bc, dict):
                signals.append(('BC', signal_bc))
            else:
                logger.debug(f"🔍 No valid signal for BC pair")
        
        except Exception as e:
            logger.error(f"❌ Error in triad analysis for {self.pair_group}: {str(e)}")
            return None
        
        # Find confluence - at least 2 pairs agreeing on direction
        return self._find_triad_confluence(signals)
    
    async def _analyze_pair_combo(self, inst1, inst2, combo_name):
        """Analyze a specific pair combination"""
        logger.info(f"🔍 Analyzing {combo_name} ({inst1}/{inst2})")
        
        # TODO: This method still uses signal_builder - need to update to use feature_box
        # For now, just return None since we're using feature_box approach
        return None


    def _send_fvg_smt_tap_signal(self, fvg_idea, smt_data, has_psp, is_hp_fvg):
        """Send FVG+SMT tap signal with cooldown tracking"""
        fvg_direction = fvg_idea['direction']
        fvg_tf = fvg_idea['timeframe']
        smt_cycle = smt_data['cycle']
        
        # Create a unique identifier for this signal
        signal_id = f"FVG_SMT_TAP_{self.pair_group}_{fvg_idea['asset']}_{fvg_tf}_{smt_data.get('signal_key', '')}"
        
        # Check cooldown (24 hours cooldown)
        if hasattr(self, 'fvg_smt_tap_sent') and signal_id in self.fvg_smt_tap_sent:
            last_sent = self.fvg_smt_tap_sent[signal_id]
            if (datetime.now(NY_TZ) - last_sent).total_seconds() < self.COOLDOWN_HOURS:  # 24 hours
                logger.info(f"⏳ FVG+SMT 24H COOLDOWN ACTIVE: {signal_id}")
                return False
                
        idea = {
            'type': 'FVG_SMT_TAP',
            'pair_group': self.pair_group,
            'direction': fvg_direction,
            'asset': fvg_idea['asset'],
            'fvg_timeframe': fvg_tf,
            'fvg_levels': fvg_idea['fvg_levels'],
            'fvg_formation_time': fvg_idea['formation_time'],
            'smt_cycle': smt_cycle,
            'smt_direction': smt_data['direction'],
            'smt_data': smt_data,  # ADD THIS LINE - pass the full SMT data
            'has_psp': has_psp,
            'is_hp_fvg': is_hp_fvg,
            'timestamp': datetime.now(NY_TZ),
            'idea_key': signal_id
        }
        
        # Format and send
        message = self._format_fvg_smt_tap_message(idea)
        
        if self._send_telegram_message(message):
            # Record when we sent this signal
            if not hasattr(self, 'fvg_smt_tap_sent'):
                self.fvg_smt_tap_sent = {}
            self.fvg_smt_tap_sent[signal_id] = datetime.now(NY_TZ)
            logger.info(f"🚀 FVG+SMT TAP SIGNAL SENT: {fvg_idea['asset']} {fvg_tf} FVG + {smt_cycle} SMT")
            return True
        return False
    def _cleanup_old_fvg_smt_signals(self):
        """Remove old FVG+SMT signals from tracking (7-day cleanup)"""
        if not hasattr(self, 'fvg_smt_tap_sent') or not self.fvg_smt_tap_sent:
            return
        
        current_time = datetime.now(NY_TZ)
        signals_to_remove = []
        
        for signal_id, sent_time in self.fvg_smt_tap_sent.items():
            if (current_time - sent_time).total_seconds() > self.CLEANUP_DAYS:  # 7 days
                signals_to_remove.append(signal_id)
        
        for signal_id in signals_to_remove:
            del self.fvg_smt_tap_sent[signal_id]
        
        if signals_to_remove:
            logger.debug(f"🧹 Cleaned up {len(signals_to_remove)} old FVG+SMT signals (7+ days)")
    
    def _format_fvg_smt_tap_message(self, idea):
        """Format FVG+SMT tap message with SMT quarter details"""
        direction_emoji = "🟢" if idea['direction'] == 'bullish' else "🔴"
        fvg_time = idea['fvg_formation_time'].strftime('%m/%d %H:%M')
        
        # Get SMT details - now it should be available
        smt_data = idea.get('smt_data', {})
        
        # Extract SMT information
        smt_cycle = idea['smt_cycle']
        quarters = smt_data.get('quarters', '')
        
        # Format quarters for display
        if quarters:
            quarters_display = quarters.replace('_', '→')
        else:
            quarters_display = ''
        
        # Get asset actions
        asset1_action = smt_data.get('asset1_action', '')
        asset2_action = smt_data.get('asset2_action', '')
        
        hp_emoji = "🎯" if idea['is_hp_fvg'] else ""
        psp_emoji = "✅" if idea['has_psp'] else "❌"
        
        # Build SMT details section
        smt_details = ""
        if quarters_display or asset1_action or asset2_action:
            smt_details = f"• {smt_cycle} {quarters_display}\n"
            if asset1_action:
                smt_details += f"  - {asset1_action}\n"
            if asset2_action:
                smt_details += f"  - {asset2_action}\n"
        else:
            smt_details = f"• {smt_cycle} cycle\n"
        
        # Get PSP details if available
        psp_details = ""
        if idea['has_psp'] and 'psp_data' in smt_data:
            psp_data = smt_data['psp_data']
            if psp_data:
                psp_timeframe = psp_data.get('timeframe', '')
                psp_time = psp_data.get('formation_time', '')
                if psp_time and isinstance(psp_time, datetime):
                    psp_time_str = psp_time.strftime('%H:%M')
                    psp_details = f"*PSP Details:*\n• Timeframe: {psp_timeframe}\n• Time: {psp_time_str}\n\n"
        
        message = f"""
        🎯 *FVG + SMT TAP CONFIRMED* 🎯
        
        *Pair Group:* {idea['pair_group'].replace('_', ' ').title()}
        *Direction:* {idea['direction'].upper()} {direction_emoji}
        *Asset:* {idea['asset']}
        *Strength:* {'ULTRA STRONG' if idea['is_hp_fvg'] and idea['has_psp'] else 'STRONG'}
        
        *Cross-Timeframe Confluence:*
        • FVG: {idea['fvg_timeframe']} at {fvg_time}
        • SMT: {smt_cycle} cycle
        • HP FVG: {hp_emoji} {'YES' if idea['is_hp_fvg'] else 'NO'}
        • PSP: {psp_emoji} {'Confirmed' if idea['has_psp'] else 'Not Confirmed'}
        
        {psp_details}
        *FVG Details:*
        • Levels: {idea['fvg_levels']}
        • Formation: {fvg_time}
        
        *SMT Quarter Details:*
        {smt_details}
        
        *Detection Time:* {idea['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
        
        #FVG_SMT_Tap #{idea['pair_group']} #{idea['direction']} #{idea['fvg_timeframe']}
        """
        return message
    def _send_double_smt_only_signal(self, primary_smt, secondary_smt, span_min):
        """Send double SMT signal w/criteria deets."""
        idea = {
            'type': 'DOUBLE_SMT',
            'pair_group': self.pair_group,
            'direction': primary_smt['direction'],
            'primary_cycle': primary_smt['cycle'],
            'secondary_cycle': secondary_smt['cycle'],
            'primary_time': primary_smt.get('second_swing_time', primary_smt['formation_time']),
            'secondary_time': secondary_smt.get('second_swing_time', secondary_smt['formation_time']),
            'span_minutes': span_min,
            'psp_confirmed': True,  # Both have
            'strength': "STRONG",
            'reasoning': f"{primary_smt['cycle']} {primary_smt['direction']} SMT + {secondary_smt['cycle']} confirm (span: {span_min}min from 2nd swings)",
            'timestamp': datetime.now(NY_TZ),
            'idea_key': f"DOUBLE_SMT_{self.pair_group}_{primary_smt['cycle']}_{secondary_smt['cycle']}_{datetime.now(NY_TZ).strftime('%H%M%S')}"
        }
        
        # Format & send (your hook)
        message = self._format_double_smt_message(idea)
        if self._send_telegram_message(message):
            logger.info(f"🚀 DOUBLE SMT SENT: {idea['primary_cycle']}-{idea['secondary_cycle']} {idea['direction']} ({span_min}min)")
            return True
        return False

    def _format_double_smt_message(self, idea):
        """Format Double SMT message with detailed SMT and PSP information"""
        direction = idea['direction'].upper()
        emoji = "🟢" if direction == "BULLISH" else "🔴"
        
        # Get SMT details
        primary_smt = idea['primary_smt']
        secondary_smt = idea['secondary_smt']
        
        def format_smt_section(smt, label):
            """Format a single SMT section"""
            section = f"*{label} SMT ({smt['cycle']}):*\n"
            
            # Quarters
            quarters = smt.get('quarters', '')
            if quarters:
                quarters_display = quarters.replace('_', '→')
                section += f"• Quarter Transition: {quarters_display}\n"
            
            # PSP status
            has_psp = smt.get('has_psp', False)
            section += f"• PSP: {'✅ Confirmed' if has_psp else '❌ Not confirmed'}\n"
            
            # PSP details if available
            if has_psp and 'psp_data' in smt:
                psp_data = smt['psp_data']
                psp_timeframe = psp_data.get('timeframe', '')
                psp_time = psp_data.get('formation_time', '')
                if psp_time and isinstance(psp_time, datetime):
                    psp_time_str = psp_time.strftime('%H:%M')
                    section += f"• PSP Time: {psp_timeframe} at {psp_time_str}\n"
            
            # Asset actions
            asset1_action = smt.get('asset1_action', '')
            asset2_action = smt.get('asset2_action', '')
            if asset1_action or asset2_action:
                section += "• Actions:\n"
                if asset1_action:
                    section += f"  - {asset1_action}\n"
                if asset2_action:
                    section += f"  - {asset2_action}\n"
            
            # Time
            formation_time = smt.get('formation_time', datetime.now(NY_TZ))
            if isinstance(formation_time, str):
                try:
                    formation_time = datetime.strptime(formation_time, '%Y-%m-%d %H:%M:%S')
                except:
                    formation_time = datetime.now(NY_TZ)
            section += f"• Time: {formation_time.strftime('%H:%M')}\n"
            
            return section
        
        # Build sections
        primary_section = format_smt_section(primary_smt, "Primary")
        secondary_section = format_smt_section(secondary_smt, "Secondary")
        
        # Determine overall strength
        if primary_smt.get('has_psp', False) and secondary_smt.get('has_psp', False):
            strength = "ULTRA STRONG"
            psp_status = "✅ Both confirmed"
        elif primary_smt.get('has_psp', False) or secondary_smt.get('has_psp', False):
            strength = "VERY STRONG"
            psp_status = "⚠️ One confirmed"
        else:
            strength = "STRONG"
            psp_status = "❌ None"
        
        message = f"""
        {emoji} *DOUBLE SMT CONFIRM* {emoji}
        
        *Group:* {idea['pair_group'].replace('_', ' ').title()}
        *Direction:* {direction}
        *Strength:* {strength}
        
        {primary_section}
        {secondary_section}
        
        *Confluence Details:*
        • Span: {idea['span_minutes']:.1f}min from 2nd swings
        • PSP Status: {psp_status}
        
        *Reasoning:* {idea['reasoning']}
        
        *Detect:* {idea['detection_time'].strftime('%H:%M:%S')}
        
        #{idea['pair_group']} #DoubleSMT #{idea['direction']}
        """
        return message

    def _check_alternative_confluences_with_fvgs(self, fvgs):
        """Only double SMTs as alternative"""
        logger.info(f"🔍 Checking double SMTs as alternative")
        return self._scan_double_smts_temporal()
    
    def _check_alternative_confluences(self):
        """Only double SMTs as alternative"""
        logger.info(f"🔍 Checking double SMTs as alternative")
        return self._scan_double_smts_temporal()


        
        
        # OLD CODE (commented out for now):
        """
        # Reset signal builder for this pair
        self.signal_builder.reset()
        
        # Get data for this pair
        asset1_data = self.market_data[inst1]
        asset2_data = self.market_data[inst2]
        
        if not asset1_data or not asset2_data:
            return None
        
        # Step 1: Check SMT invalidations and PSP tracking
        await self._check_smt_tracking()
        
        # Step 2: Scan for NEW SMT signals
        await self._scan_all_smt()
        
        # Step 3: Check for PSP for existing SMTs
        await self._check_psp_for_existing_smts()
        
        # Step 4: Scan for CRT signals
        await self._scan_crt_signals()
        
        # Check if signal is complete
        if self.signal_builder.is_signal_ready() and not self.signal_builder.has_serious_conflict():
            signal = self.signal_builder.get_signal_details()
            if signal and not self.timing_manager.is_duplicate_signal(signal['signal_key'], self.pair_group):
                logger.info(f"🎯 {combo_name}: SIGNAL COMPLETE via {signal['path']}")
                return signal
        
        return None
        """

    # def get_sleep_time(self):
    #     """Calculate sleep time until next relevant candle - SIMPLIFIED FOR NOW"""
    #     # Since we're using Feature Box now, we'll use a simpler approach
    #     # TODO: Implement proper sleep timing based on active features
        
    #     # For now, use base interval or check if we have any active features
    #     summary = self.feature_box.get_active_features_summary()
        
    #     if summary['smt_count'] > 0 or summary['crt_count'] > 0:
    #         # We have active features, check more frequently
    #         sleep_time = 30  # 30 seconds
    #         logger.info(f"⏰ {self.pair_group}: Active features detected - sleeping {sleep_time}s")
    #     else:
    #         # No active features, use normal interval
    #         sleep_time = 60  # 60 seconds
    #         logger.info(f"⏰ {self.pair_group}: No active features - sleeping {sleep_time}s")
        
    #     return sleep_time
    
    def _find_triad_confluence(self, signals):
        """Find confluence across triad pairs - FIXED UNPACKING"""
        if not signals:
            logger.info(f"🔍 {self.pair_group}: No signals for triad confluence")
            return None
            
        if len(signals) < 2:
            logger.info(f"🔍 {self.pair_group}: No triad confluence (only {len(signals)} signals)")
            return None
        
        # Count directions - FIXED: Properly unpack the signals
        bullish_count = 0
        bearish_count = 0
        signal_details = []
        
        for signal_tuple in signals:
            # Each signal_tuple should be (combo_name, signal_dict)
            if len(signal_tuple) != 2:
                logger.error(f"❌ Invalid signal tuple format: {signal_tuple}")
                continue
                
            combo_name, signal_dict = signal_tuple  # ← PROPER UNPACKING
            
            if not isinstance(signal_dict, dict):
                logger.error(f"❌ Signal is not a dictionary: {signal_dict}")
                continue
                
            direction = signal_dict.get('direction')
            if not direction:
                logger.error(f"❌ Signal missing direction: {signal_dict}")
                continue
                
            if direction == 'bullish':
                bullish_count += 1
            elif direction == 'bearish':
                bearish_count += 1
            else:
                logger.warning(f"⚠️ Unknown direction: {direction}")
                continue
                
            signal_details.append(f"{combo_name}: {direction}")
        
        # Check if we have enough valid signals
        if bullish_count + bearish_count < 2:
            logger.info(f"🔍 {self.pair_group}: Insufficient valid signals for confluence")
            return None
        
        # Check for confluence (at least 2 pairs agreeing)
        if bullish_count >= 2:
            confluence_direction = 'bullish'
        elif bearish_count >= 2:
            confluence_direction = 'bearish'
        else:
            logger.info(f"🔍 {self.pair_group}: No clear confluence - Bullish: {bullish_count}, Bearish: {bearish_count}")
            return None
        
        # Create triad signal
        triad_signal = {
            'pair_group': self.pair_group,
            'direction': confluence_direction,
            'confluence_strength': max(bullish_count, bearish_count),
            'total_pairs': len(signals),
            'signal_details': signal_details,
            'instruments': self.instruments,
            'timestamp': datetime.now(NY_TZ),
            'signal_key': f"TRIAD_{self.pair_group}_{confluence_direction}_{datetime.now().strftime('%H%M')}",
            'description': f"TRIAD CONFLUENCE: {confluence_direction.upper()} ({max(bullish_count, bearish_count)}/3 pairs)"
        }
        
        logger.info(f"🎯 TRIAD CONFLUENCE DETECTED: {self.pair_group} {confluence_direction.upper()} "
                   f"({max(bullish_count, bearish_count)}/3 pairs agreeing)")
        
        return triad_signal

    async def _scan_crt_signals_for_feature_box(self):
        """Scan for CRT signals and add to Feature Box"""
        logger.info(f"🔷 Scanning CRT signals for {self.pair_group}")
        
        crt_detected = False
        
        for timeframe in CRT_TIMEFRAMES:
            asset1_data = self.market_data[self.instruments[0]].get(timeframe)
            asset2_data = self.market_data[self.instruments[1]].get(timeframe)
            
            if (asset1_data is None or asset1_data.empty or 
                asset2_data is None or asset2_data.empty):
                continue
            
            # Detect CRT
            crt_signal = self.crt_detector.calculate_crt_current_candle(
                asset1_data, asset1_data, asset2_data, timeframe
            )
            
            if crt_signal:
                # Check for PSP for this CRT
                psp_signal = self._check_psp_for_crt(asset1_data, asset2_data, timeframe)
                
                # Add CRT to Feature Box
                self.feature_box.add_crt(crt_signal, psp_signal)
                crt_detected = True
                logger.info(f"🔷 CRT detected on {timeframe} for {self.pair_group}")
        
        if not crt_detected:
            logger.debug(f"🔷 No CRT signals for {self.pair_group}")
    
    def _check_psp_for_crt(self, asset1_data, asset2_data, timeframe):
        """Check for PSP confirmation for CRT"""
        try:
            # Use the same PSP detection logic but for CRT timeframe
            psp_signal = self.smt_detector.detect_price_swing_points(
                asset1_data, asset2_data, timeframe, lookback=5
            )
            return psp_signal
        except Exception as e:
            logger.error(f"❌ Error checking PSP for CRT: {e}")
            return None
    
    async def _scan_psp_for_existing_smts_feature_box(self):
        """Scan for PSP confirmation for existing SMTs in Feature Box"""
        active_smts = self.feature_box.get_active_features_summary()['smt_count']
        
        if active_smts == 0:
            return
        
        logger.info(f"🔄 Scanning PSP for {active_smts} existing SMTs in {self.pair_group}")
        
        psp_updates = 0
        for smt_key, smt_feature in list(self.feature_box.active_features['smt'].items()):
            # Skip if already has PSP
            if smt_feature['psp_data']:
                continue
                
            smt_data = smt_feature['smt_data']
            timeframe = smt_data.get('timeframe')
            
            if not timeframe:
                continue
                
            asset1_data = self.market_data[self.instruments[0]].get(timeframe)
            asset2_data = self.market_data[self.instruments[1]].get(timeframe)
            
            if (asset1_data is None or asset1_data.empty or 
                asset2_data is None or asset2_data.empty):
                continue
            
            # Check for PSP in recent candles
            psp_signal = self.smt_detector.check_psp_for_smt(smt_data, asset1_data, asset2_data)
            
            if psp_signal:
                # Update the SMT with PSP in Feature Box
                smt_feature['psp_data'] = psp_signal
                psp_updates += 1
                logger.info(f"✅ PSP confirmed for {smt_data['cycle']} {smt_data['direction']}")
        
        if psp_updates > 0:
            logger.info(f"🔄 Updated {psp_updates} SMTs with PSP confirmation")

# ================================
# ULTIMATE MAIN MANAGER
# ================================

class UltimateTradingManager:
    def __init__(self, api_key, telegram_token, chat_id, news_calendar=None, signal_processor=None):
        self.api_key = api_key
        self.telegram_token = telegram_token
        self.chat_id = chat_id
        self.news_calendar = news_calendar
        
        self.trading_systems = {}
        
        self.signal_processor = signal_processor
        
        for pair_group, pair_config in TRADING_PAIRS.items():
            self.trading_systems[pair_group] = UltimateTradingSystem(
                pair_group, 
                pair_config,
                news_calendar=self.news_calendar,
                telegram_token=telegram_token,
                telegram_chat_id=chat_id,
                signal_processor=self.signal_processor  # Pass it
            )
        
        logger.info(f"🎯 Initialized ULTIMATE trading manager with {len(self.trading_systems)} pair groups")
        if self.news_calendar:
            logger.info(f"📰 Using shared News Calendar instance")
        if self.signal_processor:
            logger.info(f"🤖 SignalProcessor loaded for ML filtering")
        

    def _format_ultimate_signal_message(self, signal):
        """Format ultimate signal for Telegram - NOW WITH TRIAD SUPPORT"""
        
        # Check if this is a TRIAD signal (has confluence_strength)
        if 'confluence_strength' in signal:
            return self._format_triad_signal_message(signal)
        else:
            # This is a regular pair signal - use your existing formatting
            pair_group = signal.get('pair_group', 'Unknown')
            direction = signal.get('direction', 'UNKNOWN').upper()
            strength = signal.get('strength', 0)
            path = signal.get('path', 'UNKNOWN')
            description = signal.get('description', '')
            bull_strength = signal.get('bullish_strength', 0)
            bear_strength = signal.get('bearish_strength', 0)
            has_conflict = signal.get('has_conflict', False)
            
            message = f"🛡️ *ULTIMATE TRADING SIGNAL* 🛡️\n\n"
            message += f"*Pair Group:* {pair_group.replace('_', ' ').title()}\n"
            message += f"*Direction:* {direction}\n"
            message += f"*Strength:* {strength}/9\n"
            message += f"*Path:* {path}\n"
            message += f"*Bullish SMTs:* {bull_strength}\n"
            message += f"*Bearish SMTs:* {bear_strength}\n"
            message += f"*Conflict Detected:* {'YES ⚠️' if has_conflict else 'NO ✅'}\n"
            message += f"*Description:* {description}\n\n"
            
            if 'criteria' in signal:
                message += "*Signal Criteria:*\n"
                # Remove duplicate criteria before displaying
                unique_criteria = []
                for criterion in signal['criteria']:
                    if criterion not in unique_criteria:
                        unique_criteria.append(criterion)
                        message += f"• {criterion}\n"
            
            # Show CRT+PSP confluence if present
            if signal.get('psp_for_crt'):
                psp = signal['psp_for_crt']
                psp_time = psp['formation_time'].strftime('%H:%M')
                message += f"• CRT with PSP on {psp['timeframe']} at {psp_time}\n"
            
            if 'all_smts' in signal and signal['all_smts']:
                message += f"\n*SMT Swing Details:*\n"
                for cycle, smt in signal['all_smts'].items():
                    psp_status = "✅ WITH PSP" if cycle in signal.get('psp_smts', {}) else "⏳ Waiting PSP"
                    message += f"• {cycle}: {smt['direction']} {smt['quarters']} {psp_status}\n"
                    message += f"  📍 {smt['asset1_action']}\n"
                    message += f"  📍 {smt['asset2_action']}\n"
                    
                    # Add PSP timeframe and time if available
                    if cycle in signal.get('psp_smts', {}):
                        psp = signal['psp_smts'][cycle]
                        psp_time = psp['formation_time'].strftime('%H:%M')
                        message += f"  🕒 PSP on {psp['timeframe']} at {psp_time}\n"
            
            message += f"\n*Detection Time:* {signal['timestamp'].strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
            
            if 'TRIPLE' in path:
                message += f"\n*🎯 ULTIMATE TRIPLE CONFLUENCE: CRT + PSP + SMT - HIGHEST PROBABILITY*\n"
            elif 'OVERRIDE' in path:
                message += f"\n*🎯 CYCLE OVERRIDE: Multiple smaller cycles overriding higher timeframes*\n"
            elif has_conflict:
                message += f"\n*⚠️ NOTE: Trading with caution due to conflicting signals*\n"
            
            message += f"\n#UltimateSignal #{pair_group} #{path}"
            
            return message
    
    def _format_triad_signal_message(self, signal):
        """Format triad confluence signal for Telegram"""
        pair_group = signal.get('pair_group', 'Unknown')
        direction = signal.get('direction', 'UNKNOWN').upper()
        confluence = signal.get('confluence_strength', 0)
        total_pairs = signal.get('total_pairs', 0)
        instruments = signal.get('instruments', [])
        details = signal.get('signal_details', [])
        
        message = f"🔄 *TRIAD CONFLUENCE SIGNAL* 🔄\n\n"
        message += f"*Group:* {pair_group.replace('_', ' ').title()}\n"
        message += f"*Direction:* {direction}\n"
        message += f"*Confluence:* {confluence}/{total_pairs} pairs\n"
        message += f"*Instruments:* {', '.join(instruments)}\n\n"
        
        message += "*Pair Signals:*\n"
        for detail in details:
            message += f"• {detail}\n"
        
        message += f"\n*Detection Time:* {signal['timestamp'].strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
        
        if confluence == 3:
            message += f"\n*🎯 PERFECT TRIAD CONFLUENCE: All 3 pairs agreeing!*\n"
        elif confluence == 2:
            message += f"\n*✅ STRONG TRIAD CONFLUENCE: 2/3 pairs agreeing*\n"
        
        message += f"\n#TriadConfluence #{pair_group} #{direction}"
        
        return message
    
    async def run_ultimate_systems(self):
        """Run all trading systems with ultimate decision making - CONCURRENT VERSION"""
        logger.info("🎯 Starting CONCURRENT Multi-Pair Trading System...")
        
        # Create and start all trading system tasks
        tasks = {}
        for pair_group, system in self.trading_systems.items():
            task = asyncio.create_task(
                self._run_system_continuously(system, pair_group),
                name=f"system_{pair_group}"
            )
            tasks[pair_group] = task
            logger.info(f"🚀 Started concurrent task for {pair_group}")
        
        # Wait for all tasks (they run forever until cancelled)
        await asyncio.gather(*tasks.values())
        
        logger.info("🛑 All trading system tasks completed")
    
    async def _run_system_continuously(self, system, pair_group):
        """Run a single trading system continuously with its own timing"""
        try:
            while True:
                try:
                    # Run analysis for this specific systemss
                    await system.run_ultimate_analysis(self.api_key)
                    
                    # Get sleep time specific to THIS system
                    sleep_time = system.get_sleep_time()  # FIXED METHOD NAME
                    
                    logger.info(f"⏰ {pair_group}: Sleeping for {sleep_time:.1f}s")
                    await asyncio.sleep(sleep_time)
                    
                except Exception as e:
                    logger.error(f"❌ Error in {pair_group} system: {str(e)}")
                    await asyncio.sleep(60)  # Error cooldown
                    
        except asyncio.CancelledError:
            logger.info(f"🛑 {pair_group} task cancelled")
            raise
        except Exception as e:
            logger.error(f"💥 Fatal error in {pair_group} task: {str(e)}")
    
    async def _process_ultimate_signals(self, signals):
        """Process and send ultimate signals to Telegram"""
        for signal in signals:
            try:
                message = self._format_ultimate_signal_message(signal)
                success = send_telegram(message, self.telegram_token, self.chat_id)
                
                if success:
                    logger.info(f"📤 Ultimate signal sent to Telegram for {signal['pair_group']}")
                else:
                    logger.error(f"❌ Failed to send ultimate signal for {signal['pair_group']}")
                    
            except Exception as e:
                logger.error(f"❌ Error processing ultimate signal: {str(e)}")
    
    def _format_ultimate_signal_message(self, signal):
        """Format ultimate signal for Telegram - NOW INCLUDES TRIPLE CONFLUENCE"""
        pair_group = signal.get('pair_group', 'Unknown')
        direction = signal.get('direction', 'UNKNOWN').upper()
        strength = signal.get('strength', 0)
        path = signal.get('path', 'UNKNOWN')
        description = signal.get('description', '')
        bull_strength = signal.get('bullish_strength', 0)
        bear_strength = signal.get('bearish_strength', 0)
        has_conflict = signal.get('has_conflict', False)
        
        message = f"🛡️ *ULTIMATE TRADING SIGNAL* 🛡️\n\n"
        message += f"*Pair Group:* {pair_group.replace('_', ' ').title()}\n"
        message += f"*Direction:* {direction}\n"
        message += f"*Strength:* {strength}/9\n"
        message += f"*Path:* {path}\n"
        message += f"*Bullish SMTs:* {bull_strength}\n"
        message += f"*Bearish SMTs:* {bear_strength}\n"
        message += f"*Conflict Detected:* {'YES ⚠️' if has_conflict else 'NO ✅'}\n"
        message += f"*Description:* {description}\n\n"
        
        if 'criteria' in signal:
            message += "*Signal Criteria:*\n"
            # Remove duplicate criteria before displaying
            unique_criteria = []
            for criterion in signal['criteria']:
                if criterion not in unique_criteria:
                    unique_criteria.append(criterion)
                    message += f"• {criterion}\n"
        
        # NEW: Show CRT+PSP confluence if present
        if signal.get('psp_for_crt'):
            psp = signal['psp_for_crt']
            psp_time = psp['formation_time'].strftime('%H:%M')
            message += f"• CRT with PSP on {psp['timeframe']} at {psp_time}\n"
        
        if 'all_smts' in signal and signal['all_smts']:
            message += f"\n*SMT Swing Details:*\n"
            for cycle, smt in signal['all_smts'].items():
                psp_status = "✅ WITH PSP" if cycle in signal.get('psp_smts', {}) else "⏳ Waiting PSP"
                message += f"• {cycle}: {smt['direction']} {smt['quarters']} {psp_status}\n"
                message += f"  📍 {smt['asset1_action']}\n"
                message += f"  📍 {smt['asset2_action']}\n"
                
                # Add PSP timeframe and time if available
                if cycle in signal.get('psp_smts', {}):
                    psp = signal['psp_smts'][cycle]
                    psp_time = psp['formation_time'].strftime('%H:%M')
                    message += f"  🕒 PSP on {psp['timeframe']} at {psp_time}\n"
        
        message += f"\n*Detection Time:* {signal['timestamp'].strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
        
        if 'TRIPLE' in path:
            message += f"\n*🎯 ULTIMATE TRIPLE CONFLUENCE: CRT + PSP + SMT - HIGHEST PROBABILITY*\n"
        elif 'OVERRIDE' in path:
            message += f"\n*🎯 CYCLE OVERRIDE: Multiple smaller cycles overriding higher timeframes*\n"
        elif has_conflict:
            message += f"\n*⚠️ NOTE: Trading with caution due to conflicting signals*\n"
        
        message += f"\n#UltimateSignal #{pair_group} #{path}"
        
        return message


# ================================
# UTILITY FUNCTIONS
# ================================

def print_system_status():
    """Print current system status including cache statistics"""
    print("\n" + "="*60)
    print("SYSTEM STATUS")
    print("="*60)
    
    # Global cache stats
    if 'GLOBAL_CACHE' in globals():
        cache_stats = GLOBAL_CACHE.get_stats()
        print(f"📦 Global Cache: {cache_stats['size']}/{cache_stats['max_size']} entries")
        print(f"   TTL: {cache_stats['ttl_seconds']} seconds")
    
    # Log current time
    current_time = datetime.now(NY_TZ).strftime('%Y-%m-%d %H:%M:%S')
    print(f"⏰ Current Time (NY): {current_time}")
    print("="*60 + "\n")

# Optional: Add this at the beginning of main() for status
# print_system_status()
# ================================
# MAIN EXECUTION
# ================================

async def main():
    """Main entry point"""
    quick_hammer_test()
    logger.info("HEY TOM'S SNIPER JUST WOKE UP")
    
    api_key = os.getenv('OANDA_API_KEY')
    telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
    telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
    rapidapi_key = os.getenv('rapidapi_key')
    logger.info(f"RapidAPI Key found: {'Yes' if rapidapi_key else 'No'}")
    
    if not all([api_key, telegram_token, telegram_chat_id]):
        logger.error("❌ Missing required environment variables")
        logger.info("💡 Please set OANDA_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID")
        return

    print_system_status()
    
    # Create ONE NewsCalendar instance with 24-hour cache
    news_calendar = None
    if rapidapi_key:
        news_calendar = NewsCalendar(
            rapidapi_key=rapidapi_key,
            base_path='/content/drive/MyDrive',
            logger=logger,
            cache_duration=86400  # 24 hours = 86400 seconds
        )
        
        # Mark it as shared to prevent background fetches
        news_calendar._is_shared = True
        
        # Fetch news ONCE at startup
        logger.info("📰 Fetching news ONCE at startup...")
        news_calendar.get_daily_news(force_fetch=True)  # Force fetch to get fresh data
        
        # Also fetch for today to ensure cache is created
        today_str = datetime.now(pytz.timezone('America/New_York')).strftime('%Y-%m-%d')
        news_calendar.fetch_news_data(today_str)
        
        logger.info(f"📰 News fetched and cached for today ({today_str})")
    else:
        logger.warning("⚠️ RapidAPI key missing. News features disabled.")
    
    # ============================================
    # CREATE CREDENTIALS DICTIONARY
    # ============================================
    credentials = {
        'telegram_token': telegram_token,
        'telegram_chat_id': telegram_chat_id,
        'oanda_api_key': api_key
    }
    
    # ============================================
    # START ZEBRA MANAGER (INDEPENDENT)
    # ============================================
    logger.info("🦓 Initializing Zebra Manager...")
    zebra_manager = None
    
    try:
        zebra_manager = ZebraManager(
            credentials=credentials,
            news_calendar=news_calendar,
            instruments_dict=ZEBRA_INSTRUMENTS,
            logger=logger
        )
        
        # Start all Zebra scanners
        zebra_manager.start()
        logger.info("✅ Zebra Manager started successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to start Zebra Manager: {str(e)}")
    
    # ============================================
    # START ULTIMATE TRADING MANAGER (HAMMER SCANNER)
    # ============================================
    try:
        # Initialize the manager with the shared news calendar
        manager = UltimateTradingManager(
            api_key, 
            telegram_token, 
            telegram_chat_id, 
            news_calendar=news_calendar  # PASS THE SHARED CALENDAR
        )
        
        # Make sure all hammer scanners are started
        logger.info("🔨 Starting all hammer scanners...")
        for pair_group, system in manager.trading_systems.items():
            if hasattr(system, 'hammer_scanner'):
                try:
                    system.hammer_scanner.start()
                    logger.info(f"✅ Hammer scanner started for {pair_group}")
                except Exception as e:
                    logger.error(f"❌ Failed to start hammer scanner for {pair_group}: {str(e)}")
        
        # Run the main systems
        await manager.run_ultimate_systems()
        
    except KeyboardInterrupt:
        logger.info("🛑 System stopped by user")
        
        # Stop Zebra Manager if it exists
        if zebra_manager:
            zebra_manager.stop()
            logger.info("🛑 Zebra Manager stopped")
        
        # Stop all hammer scanners
        if 'manager' in locals():
            for pair_group, system in manager.trading_systems.items():
                if hasattr(system, 'hammer_scanner'):
                    system.hammer_scanner.stop()
                    logger.info(f"🛑 Hammer scanner stopped for {pair_group}")
                    
    except Exception as e:
        logger.error(f"💥 Fatal error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Try to stop Zebra Manager on error
        if zebra_manager:
            try:
                zebra_manager.stop()
            except:
                pass
        
        # Try to stop hammer scanners on error
        if 'manager' in locals():
            for pair_group, system in manager.trading_systems.items():
                if hasattr(system, 'hammer_scanner'):
                    try:
                        system.hammer_scanner.stop()
                    except:
                        pass
        
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
