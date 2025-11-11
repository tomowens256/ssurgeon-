#!/usr/bin/env python3
"""
ROBUST MULTI-PAIR SMT TRADING SYSTEM
With SMT-CRT direction matching and SMT invalidation logic - FIXED VERSION
"""

import asyncio
import logging
import os
import sys
import time
import re
import requests
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set
from oandapyV20 import API
from oandapyV20.exceptions import V20Error
from oandapyV20.endpoints import instruments

# ================================
# CONFIGURATION
# ================================

# Trading pairs configuration
TRADING_PAIRS = {
    'precious_metals': {
        'pair1': 'XAU_USD',
        'pair2': 'XAG_USD',
        'timeframe_mapping': {
            'monthly': 'H4',
            'weekly': 'H1', 
            'daily': 'M15',
            '90min': 'M5'
        }
    },
    'us_indices': {
        'pair1': 'NAS100_USD',
        'pair2': 'SPX500_USD',
        'timeframe_mapping': {
            'monthly': 'H4',
            'weekly': 'H1',
            'daily': 'M15', 
            '90min': 'M5'
        }
    },
    'european_indices': {
        'pair1': 'DE30_EUR',
        'pair2': 'EU50_EUR',
        'timeframe_mapping': {
            'monthly': 'H4',
            'weekly': 'H1',
            'daily': 'M15',
            '90min': 'M5'
        }
    }
}

# CRT Timeframes (1H and above only)
CRT_TIMEFRAMES = ['H1', 'H2', 'H3', 'H4', 'H6', 'H8', 'H12']

# Cycle to Timeframe mapping for sleep timing
CYCLE_SLEEP_TIMEFRAMES = {
    'monthly': 'H4',    # Sleep until next H4 candle
    'weekly': 'H1',     # Sleep until next H1 candle  
    'daily': 'M15',     # Sleep until next M15 candle
    '90min': 'M5'       # Sleep until next M5 candle
}

# System Configuration
NY_TZ = pytz.timezone('America/New_York')
BASE_INTERVAL = 300
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
        logging.FileHandler('trading_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ================================
# UTILITY FUNCTIONS
# ================================

def parse_oanda_time(time_str):
    """Parse Oanda's timestamp with variable fractional seconds"""
    try:
        if '.' in time_str and len(time_str.split('.')[1]) > 7:
            time_str = re.sub(r'\.(\d{6})\d+', r'.\1', time_str)
        return datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=pytz.utc).astimezone(NY_TZ)
    except Exception as e:
        logger.error(f"Error parsing time {time_str}: {str(e)}")
        return datetime.now(NY_TZ)

def send_telegram(message, token=None, chat_id=None):
    """Send formatted message to Telegram"""
    if not token or not chat_id:
        logger.error("Telegram credentials missing")
        return False
        
    if len(message) > 4000:
        message = message[:4000] + "... [TRUNCATED]"
    
    # Escape special Markdown characters
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

def fetch_candles(instrument, timeframe, count=100, api_key=None):
    """Fetch candles from OANDA API"""
    logger.debug(f"Fetching {count} candles for {instrument} {timeframe}")
    
    if not api_key:
        logger.error("Oanda API key missing")
        return pd.DataFrame()
        
    try:
        api = API(access_token=api_key, environment="practice")
    except Exception as e:
        logger.error(f"Oanda API initialization failed: {str(e)}")
        return pd.DataFrame()
        
    params = {
        "granularity": timeframe,
        "count": count,
        "price": "M",
        "alignmentTimezone": "America/New_York",
        "includeCurrent": True
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            request = instruments.InstrumentsCandles(instrument=instrument, params=params)
            response = api.request(request)
            candles = response.get('candles', [])
            
            logger.debug(f"Received {len(candles)} candles for {instrument}")
            
            if not candles:
                logger.warning(f"No candles received for {instrument} on attempt {attempt+1}")
                continue
            
            data = []
            for candle in candles:
                price_data = candle.get('mid', {})
                if not price_data:
                    continue
                
                try:
                    parsed_time = parse_oanda_time(candle['time'])
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
                except Exception as e:
                    logger.error(f"Error parsing candle for {instrument}: {str(e)}")
                    continue
            
            if not data:
                logger.warning(f"Empty data after parsing for {instrument} on attempt {attempt+1}")
                continue
                
            df = pd.DataFrame(data).drop_duplicates(subset=['time'], keep='last')
            df = df.sort_values('time').reset_index(drop=True)
            
            logger.info(f"Successfully fetched {len(df)} candles for {instrument} {timeframe}")
            return df
            
        except V20Error as e:
            if "rate" in str(e).lower() or getattr(e, 'code', 0) in [429, 502]:
                wait_time = 10 * (2 ** attempt)
                logger.warning(f"Rate limit hit for {instrument}, waiting {wait_time}s: {str(e)}")
                time.sleep(wait_time)
            else:
                error_details = f"Status: {getattr(e, 'code', 'N/A')} | Message: {getattr(e, 'msg', str(e))}"
                logger.error(f"Oanda API error for {instrument}: {error_details}")
                break
        except Exception as e:
            logger.error(f"General error fetching candles for {instrument}: {str(e)}")
            time.sleep(10)
    
    logger.error(f"Failed to fetch candles for {instrument} after {MAX_RETRIES} attempts")
    return pd.DataFrame()

# ================================
# ROBUST TIMING MANAGER
# ================================

class RobustTimingManager:
    """Robust timing manager with exact sleep calculations"""
    
    def __init__(self):
        self.ny_tz = pytz.timezone('America/New_York')
        
    def calculate_next_candle_time(self, timeframe):
        """Calculate when the next candle will open for any timeframe"""
        now = datetime.now(self.ny_tz)
        
        if timeframe.startswith('H'):
            # Handle hourly timeframes
            hours = int(timeframe[1:])
            return self._calculate_next_htf_candle_time(hours)
        elif timeframe.startswith('M'):
            # Handle minute timeframes
            minutes = int(timeframe[1:])
            return self._calculate_next_ltf_candle_time(minutes)
        else:
            return self._calculate_next_htf_candle_time(1)  # Default to H1
    
    def _calculate_next_htf_candle_time(self, hours):
        """Calculate next candle time for hourly timeframes (H1, H2, H4, etc.)"""
        now = datetime.now(self.ny_tz)
        
        if hours == 1:
            # H1 candles: every hour
            next_hour = now.hour + 1
            if next_hour >= 24:
                next_time = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            else:
                next_time = now.replace(hour=next_hour, minute=0, second=0, microsecond=0)
                
        elif hours == 4:
            # H4 candles: 00:00, 04:00, 08:00, 12:00, 16:00, 20:00 (NY time)
            current_hour = now.hour
            next_hour = ((current_hour // 4) * 4 + 4) % 24
            if next_hour < current_hour:  # Next day
                next_time = now.replace(hour=next_hour, minute=0, second=0, microsecond=0) + timedelta(days=1)
            else:
                next_time = now.replace(hour=next_hour, minute=0, second=0, microsecond=0)
                
        else:
            # Other hourly timeframes (H2, H3, H6, H8, H12)
            current_hour = now.hour
            next_hour = ((current_hour // hours) * hours + hours) % 24
            if next_hour < current_hour:  # Next day
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
        
        # Add buffer seconds for API data availability
        sleep_seconds = (next_candle_time - current_time).total_seconds() + CANDLE_BUFFER_SECONDS
        
        return max(MIN_INTERVAL, sleep_seconds)
    
    def get_sleep_time_for_crt(self, crt_timeframe):
        """Calculate sleep time until next CRT candle"""
        next_candle_time = self.calculate_next_candle_time(crt_timeframe)
        current_time = datetime.now(self.ny_tz)
        
        sleep_seconds = (next_candle_time - current_time).total_seconds() + CANDLE_BUFFER_SECONDS
        return max(MIN_INTERVAL, sleep_seconds)
    
    def is_crt_fresh(self, crt_timestamp, max_age_minutes=1):
        """Check if CRT signal is fresh (not older than max_age_minutes)"""
        if not crt_timestamp:
            return False
            
        current_time = datetime.now(self.ny_tz)
        age_seconds = (current_time - crt_timestamp).total_seconds()
        
        return age_seconds <= (max_age_minutes * 60)

# ================================
# QUARTER MANAGER
# ================================

class QuarterManager:
    """Manage quarter detection with corrected 90min cycles"""
    
    def __init__(self):
        self.ny_tz = pytz.timezone('America/New_York')
        
    def detect_current_quarters(self, timestamp=None):
        """Detect current quarter for all cycles"""
        if timestamp is None:
            timestamp = datetime.now(self.ny_tz)
        else:
            if timestamp.tzinfo is None:
                timestamp = self.ny_tz.localize(timestamp)
            else:
                timestamp = timestamp.astimezone(self.ny_tz)
                
        return {
            'monthly': self._get_monthly_quarter(timestamp),
            'weekly': self._get_weekly_quarter(timestamp),
            'daily': self._get_daily_quarter(timestamp),
            '90min': self._get_90min_quarter_fixed(timestamp)
        }
    
    def _get_monthly_quarter(self, timestamp):
        """Monthly: Q1=Days 1-7, Q2=8-14, Q3=15-21, Q4=22-28, Q_Less=29+"""
        day = timestamp.day
        if 1 <= day <= 7: return 'q1'
        elif 8 <= day <= 14: return 'q2'
        elif 15 <= day <= 21: return 'q3'
        elif 22 <= day <= 28: return 'q4'
        else: return 'q_less'
    
    def _get_weekly_quarter(self, timestamp):
        """Weekly: Mon=Q1, Tue=Q2, Wed=Q3, Thu=Q4, Fri=Q_Less"""
        weekday = timestamp.weekday()
        if weekday == 0: return 'q1'
        elif weekday == 1: return 'q2'
        elif weekday == 2: return 'q3'
        elif weekday == 3: return 'q4'
        else: return 'q_less'
    
    def _get_daily_quarter(self, timestamp):
        """Daily quarters in UTC-4"""
        hour = timestamp.hour
        if 0 <= hour < 6: return 'q2'
        elif 6 <= hour < 12: return 'q3'
        elif 12 <= hour < 18: return 'q4'
        else: return 'q1'
    
    def _get_90min_quarter_fixed(self, timestamp):
        """Fixed 90min quarters within daily quarters as specified"""
        daily_quarter = self._get_daily_quarter(timestamp)
        hour = timestamp.hour
        minute = timestamp.minute
        total_minutes = hour * 60 + minute
        
        # Define 90min quarter boundaries for each daily quarter
        boundaries = {
            'q1': [  # 18:00 - 00:00
                (18*60, 19*60+30, 'q1'),   # 18:00 - 19:30
                (19*60+30, 21*60, 'q2'),   # 19:30 - 21:00  
                (21*60, 22*60+30, 'q3'),   # 21:00 - 22:30
                (22*60+30, 24*60, 'q4')    # 22:30 - 00:00
            ],
            'q2': [  # 00:00 - 06:00
                (0, 1*60+30, 'q1'),        # 00:00 - 01:30
                (1*60+30, 3*60, 'q2'),     # 01:30 - 03:00
                (3*60, 4*60+30, 'q3'),     # 03:00 - 04:30
                (4*60+30, 6*60, 'q4')      # 04:30 - 06:00
            ],
            'q3': [  # 06:00 - 12:00
                (6*60, 7*60+30, 'q1'),     # 06:00 - 07:30
                (7*60+30, 9*60, 'q2'),     # 07:30 - 09:00
                (9*60, 10*60+30, 'q3'),    # 09:00 - 10:30
                (10*60+30, 12*60, 'q4')    # 10:30 - 12:00
            ],
            'q4': [  # 12:00 - 18:00
                (12*60, 13*60+30, 'q1'),   # 12:00 - 13:30
                (13*60+30, 15*60, 'q2'),   # 13:30 - 15:00
                (15*60, 16*60+30, 'q3'),   # 15:00 - 16:30
                (16*60+30, 18*60, 'q4')    # 16:30 - 18:00
            ]
        }
        
        # Check which 90min quarter we're in
        for start_min, end_min, quarter in boundaries[daily_quarter]:
            if start_min <= total_minutes < end_min:
                return quarter
        
        return 'q_less'
    
    def get_valid_quarter_pairs(self, current_quarter, cycle_type):
        """Get valid consecutive quarter pairs to check based on current quarter"""
        # Define quarter sequences
        quarter_sequence = ['q1', 'q2', 'q3', 'q4']
        
        # Find current quarter index
        try:
            current_idx = quarter_sequence.index(current_quarter)
        except ValueError:
            return []
        
        # Only check the 2 most recent consecutive pairs (don't skip quarters)
        valid_pairs = []
        
        # Pair 1: current-2 → current-1 (if available)
        if current_idx >= 2:
            valid_pairs.append((quarter_sequence[current_idx-2], quarter_sequence[current_idx-1]))
        
        # Pair 2: current-1 → current (if available)  
        if current_idx >= 1:
            valid_pairs.append((quarter_sequence[current_idx-1], quarter_sequence[current_idx]))
        
        logger.debug(f"Current {cycle_type} quarter: {current_quarter}, Valid pairs: {valid_pairs}")
        return valid_pairs
    
    def group_candles_by_quarters(self, df, cycle_type, num_quarters=4):
        """Group candles into exact quarters based on their timestamps"""
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return {}
        
        quarters_data = {}
        
        for _, candle in df.iterrows():
            candle_time = candle['time']
            quarter = self._get_candle_quarter(candle_time, cycle_type)
            
            if quarter not in quarters_data:
                quarters_data[quarter] = []
            quarters_data[quarter].append(candle)
        
        # Convert to DataFrames and sort by time
        for quarter in quarters_data:
            quarters_data[quarter] = pd.DataFrame(quarters_data[quarter])
            quarters_data[quarter] = quarters_data[quarter].sort_values('time')
        
        return quarters_data
    
    def _get_candle_quarter(self, candle_time, cycle_type):
        """Get the quarter for a specific candle based on cycle type"""
        if cycle_type == 'monthly':
            return self._get_monthly_quarter(candle_time)
        elif cycle_type == 'weekly':
            return self._get_weekly_quarter(candle_time)
        elif cycle_type == 'daily':
            return self._get_daily_quarter(candle_time)
        elif cycle_type == '90min':
            return self._get_90min_quarter_fixed(candle_time)
        else:
            return 'unknown'

# ================================
# SWING DETECTOR
# ================================

class SwingDetector:
    """Detect swing highs and swing lows within quarters"""
    
    @staticmethod
    def find_swing_highs_lows(df, lookback=3):
        """Find swing highs and swing lows in a DataFrame"""
        if df is None or not isinstance(df, pd.DataFrame) or len(df) < lookback + 1:
            return [], []
        
        swing_highs = []
        swing_lows = []
        
        for i in range(lookback, len(df) - lookback):
            # Check for swing high
            is_swing_high = True
            current_high = float(df.iloc[i]['high'])
            
            for j in range(1, lookback + 1):
                if (float(df.iloc[i - j]['high']) >= current_high or 
                    float(df.iloc[i + j]['high']) >= current_high):
                    is_swing_high = False
                    break
            
            if is_swing_high:
                swing_highs.append({
                    'time': df.iloc[i]['time'],
                    'price': current_high,
                    'index': i
                })
            
            # Check for swing low
            is_swing_low = True
            current_low = float(df.iloc[i]['low'])
            
            for j in range(1, lookback + 1):
                if (float(df.iloc[i - j]['low']) <= current_low or 
                    float(df.iloc[i + j]['low']) <= current_low):
                    is_swing_low = False
                    break
            
            if is_swing_low:
                swing_lows.append({
                    'time': df.iloc[i]['time'],
                    'price': current_low,
                    'index': i
                })
        
        return swing_highs, swing_lows
    
    @staticmethod
    def get_highest_swing_high(swing_highs):
        """Get the highest swing high from a list"""
        if not swing_highs:
            return None
        return max(swing_highs, key=lambda x: x['price'])
    
    @staticmethod
    def get_lowest_swing_low(swing_lows):
        """Get the lowest swing low from a list"""
        if not swing_lows:
            return None
        return min(swing_lows, key=lambda x: x['price'])

# ================================
# PATTERN DETECTORS
# ================================

class RobustCRTDetector:
    """CRT detector for current candle only with freshness check"""
    
    def __init__(self, timing_manager):
        self.timing_manager = timing_manager
    
    def calculate_crt_current_candle(self, df):
        """Calculate CRT only on the current (incomplete) candle"""
        if df is None or not isinstance(df, pd.DataFrame) or df.empty or len(df) < 3:
            return None
        
        # Get only the current (incomplete) candle
        current_candle = df[df['is_current'] == True]
        if current_candle.empty:
            return None
            
        current_candle = current_candle.iloc[0]
        
        # Check if current candle is fresh (not older than 1 minute)
        if not self.timing_manager.is_crt_fresh(current_candle['time']):
            logger.debug("CRT candle too old, skipping")
            return None
        
        # We need the previous 2 complete candles for CRT calculation
        complete_candles = df[df['complete'] == True].tail(2)
        if len(complete_candles) < 2:
            return None
            
        c1 = complete_candles.iloc[0]  # candle 1 (two candles back)
        c2 = complete_candles.iloc[1]  # candle 2 (one candle back)
        c3 = current_candle            # candle 3 (current)
        
        try:
            # CRT calculations with proper type conversion
            c2_range = float(c2['high']) - float(c2['low'])
            c2_mid = float(c2['low']) + 0.5 * c2_range
            
            # Buy CRT: c2 low < c1 low AND c2 close > c1 low AND c3 open > c2 mid
            buy_crt = (float(c2['low']) < float(c1['low']) and 
                      float(c2['close']) > float(c1['low']) and 
                      float(c3['open']) > c2_mid)
            
            # Sell CRT: c2 high > c1 high AND c2 close < c1 high AND c3 open < c2 mid
            sell_crt = (float(c2['high']) > float(c1['high']) and 
                       float(c2['close']) < float(c1['high']) and 
                       float(c3['open']) < c2_mid)
            
            if buy_crt:
                return {'direction': 'bullish', 'timestamp': c3['time']}
            elif sell_crt:
                return {'direction': 'bearish', 'timestamp': c3['time']}
        except (ValueError, TypeError) as e:
            logger.error(f"Error in CRT calculation: {e}")
            return None
        
        return None

class RobustPSPDetector:
    """PSP detector for candle 2 (previous candle) of CRT"""
    
    @staticmethod
    def detect_psp_previous_candle(asset1_data, asset2_data, timeframe):
        """Detect PSP on the previous candle (candle 2 of CRT)"""
        if (asset1_data is None or not isinstance(asset1_data, pd.DataFrame) or asset1_data.empty or
            asset2_data is None or not isinstance(asset2_data, pd.DataFrame) or asset2_data.empty):
            return None
        
        # Get the previous complete candle (candle 2 of CRT)
        asset1_prev = asset1_data[asset1_data['complete'] == True]
        asset2_prev = asset2_data[asset2_data['complete'] == True]
        
        if asset1_prev.empty or asset2_prev.empty:
            return None
            
        # Get the most recent complete candle (previous candle)
        asset1_candle = asset1_prev.iloc[-1]
        asset2_candle = asset2_prev.iloc[-1]
        
        # Check if both previous candles have the same timestamp (same candle)
        if asset1_candle['time'] != asset2_candle['time']:
            logger.debug(f"PSP timestamps don't match: {asset1_candle['time']} vs {asset2_candle['time']}")
            return None
        
        try:
            asset1_color = 'green' if float(asset1_candle['close']) > float(asset1_candle['open']) else 'red'
            asset2_color = 'green' if float(asset2_candle['close']) > float(asset2_candle['open']) else 'red'
            
            if asset1_color != asset2_color:
                return {
                    'timeframe': timeframe,
                    'asset1_color': asset1_color,
                    'asset2_color': asset2_color,
                    'timestamp': asset1_candle['time']
                }
        except (ValueError, TypeError) as e:
            logger.error(f"Error in PSP calculation: {e}")
            return None
        
        return None

class RobustSMTDetector:
    """ROBUST SMT detector with invalidation logic - FIXED VERSION"""
    
    def __init__(self, pair_config):
        self.smt_history = []
        self.quarter_manager = QuarterManager()
        self.swing_detector = SwingDetector()
        self.signal_counts = {}
        self.invalidated_smts = set()  # Track invalidated SMTs
        self.pair_config = pair_config  # Store pair config for timeframe mapping
        
    def detect_smt_all_cycles(self, asset1_data, asset2_data, cycle_type):
        """Detect SMT for a specific cycle - always scan"""
        try:
            if (asset1_data is None or not isinstance(asset1_data, pd.DataFrame) or asset1_data.empty or
                asset2_data is None or not isinstance(asset2_data, pd.DataFrame) or asset2_data.empty):
                return None
            
            # Get current quarter
            current_quarters = self.quarter_manager.detect_current_quarters()
            current_quarter = current_quarters.get(cycle_type)
            
            if not current_quarter:
                return None
            
            # Get valid quarter pairs to check (only consecutive quarters, no skipping)
            valid_pairs = self.quarter_manager.get_valid_quarter_pairs(current_quarter, cycle_type)
            
            if not valid_pairs:
                return None
            
            # Group candles into quarters for both assets
            asset1_quarters = self.quarter_manager.group_candles_by_quarters(asset1_data, cycle_type)
            asset2_quarters = self.quarter_manager.group_candles_by_quarters(asset2_data, cycle_type)
            
            if not asset1_quarters or not asset2_quarters:
                return None
            
            # Check each valid quarter pair for SMT
            for prev_q, curr_q in valid_pairs:
                if prev_q not in asset1_quarters or curr_q not in asset1_quarters:
                    continue
                if prev_q not in asset2_quarters or curr_q not in asset2_quarters:
                    continue
                
                smt_result = self._compare_quarters_swing_based(
                    asset1_quarters[prev_q], asset1_quarters[curr_q],
                    asset2_quarters[prev_q], asset2_quarters[curr_q],
                    cycle_type, prev_q, curr_q
                )
                
                if smt_result and not self._is_duplicate_signal(smt_result):
                    return smt_result
            
            return None
            
        except Exception as e:
            logger.error(f"Error in SMT detection for {cycle_type}: {str(e)}")
            return None
    
    def _compare_quarters_swing_based(self, asset1_prev, asset1_curr, asset2_prev, asset2_curr, cycle_type, prev_q, curr_q):
        """Compare two consecutive quarters using swing highs/lows"""
        try:
            if (asset1_prev.empty or asset1_curr.empty or 
                asset2_prev.empty or asset2_curr.empty):
                return None
            
            # Find swing highs and lows for each quarter
            asset1_prev_swing_highs, asset1_prev_swing_lows = self.swing_detector.find_swing_highs_lows(asset1_prev)
            asset1_curr_swing_highs, asset1_curr_swing_lows = self.swing_detector.find_swing_highs_lows(asset1_curr)
            
            asset2_prev_swing_highs, asset2_prev_swing_lows = self.swing_detector.find_swing_highs_lows(asset2_prev)
            asset2_curr_swing_highs, asset2_curr_swing_lows = self.swing_detector.find_swing_highs_lows(asset2_curr)
            
            # Get highest swing highs and lowest swing lows
            asset1_prev_highest = self.swing_detector.get_highest_swing_high(asset1_prev_swing_highs)
            asset1_curr_highest = self.swing_detector.get_highest_swing_high(asset1_curr_swing_highs)
            asset1_prev_lowest = self.swing_detector.get_lowest_swing_low(asset1_prev_swing_lows)
            asset1_curr_lowest = self.swing_detector.get_lowest_swing_low(asset1_curr_swing_lows)
            
            asset2_prev_highest = self.swing_detector.get_highest_swing_high(asset2_prev_swing_highs)
            asset2_curr_highest = self.swing_detector.get_highest_swing_high(asset2_curr_swing_highs)
            asset2_prev_lowest = self.swing_detector.get_lowest_swing_low(asset2_prev_swing_lows)
            asset2_curr_lowest = self.swing_detector.get_lowest_swing_low(asset2_curr_swing_lows)
            
            # Check for valid SMT patterns
            bearish_smt = self._check_bearish_smt(
                asset1_prev_highest, asset1_curr_highest,
                asset2_prev_highest, asset2_curr_highest
            )
            
            bullish_smt = self._check_bullish_smt(
                asset1_prev_lowest, asset1_curr_lowest,
                asset2_prev_lowest, asset2_curr_lowest
            )
            
            if bearish_smt:
                direction = 'bearish'
                smt_type = 'Higher Swing High'
                asset1_action = f"made higher swing high ({asset1_prev_highest['price']:.4f} → {asset1_curr_highest['price']:.4f})"
                asset2_action = f"no higher swing high ({asset2_prev_highest['price']:.4f} → {asset2_curr_highest['price']:.4f})"
                critical_level = asset1_curr_highest['price']  # Highest high for bearish SMT
                
            elif bullish_smt:
                direction = 'bullish'
                smt_type = 'Lower Swing Low'
                asset1_action = f"made lower swing low ({asset1_prev_lowest['price']:.4f} → {asset1_curr_lowest['price']:.4f})"
                asset2_action = f"no lower swing low ({asset2_prev_lowest['price']:.4f} → {asset2_curr_lowest['price']:.4f})"
                critical_level = asset1_curr_lowest['price']  # Lowest low for bullish SMT
                
            else:
                return None  # No SMT pattern
            
            # Create detailed SMT data
            smt_data = {
                'direction': direction,
                'type': smt_type,
                'cycle': cycle_type,
                'quarters': f"{prev_q}→{curr_q}",
                'timestamp': datetime.now(NY_TZ),
                'asset1_action': asset1_action,
                'asset2_action': asset2_action,
                'details': f"Asset1 {asset1_action}, Asset2 {asset2_action}",
                'signal_key': f"{cycle_type}_{prev_q}_{curr_q}_{direction}",
                'critical_level': critical_level,  # Store level for invalidation checking
                'timeframe': self.pair_config['timeframe_mapping'][cycle_type]  # Store timeframe for data access
            }
            
            self.smt_history.append(smt_data)
            self._update_signal_count(smt_data['signal_key'])
            
            logger.info(f"🎯 SMT: {direction} {cycle_type} {prev_q}→{curr_q}")
            logger.info(f"   Asset1: {asset1_action}")
            logger.info(f"   Asset2: {asset2_action}")
            logger.info(f"   Critical Level: {critical_level:.4f}")
            
            return smt_data
            
        except Exception as e:
            logger.error(f"Error comparing quarters {prev_q}→{curr_q}: {str(e)}")
            return None
    
    def _check_bearish_smt(self, asset1_prev_high, asset1_curr_high, asset2_prev_high, asset2_curr_high):
        """Check for bearish SMT: Asset1 makes HH, Asset2 doesn't"""
        if not all([asset1_prev_high, asset1_curr_high, asset2_prev_high, asset2_curr_high]):
            return False
        
        # Asset1: current swing high > previous swing high (makes HH)
        asset1_hh = asset1_curr_high['price'] > asset1_prev_high['price']
        
        # Asset2: current swing high <= previous swing high (doesn't make HH)
        asset2_no_hh = asset2_curr_high['price'] <= asset2_prev_high['price']
        
        return asset1_hh and asset2_no_hh
    
    def _check_bullish_smt(self, asset1_prev_low, asset1_curr_low, asset2_prev_low, asset2_curr_low):
        """Check for bullish SMT: Asset1 makes LL, Asset2 doesn't"""
        if not all([asset1_prev_low, asset1_curr_low, asset2_prev_low, asset2_curr_low]):
            return False
        
        # Asset1: current swing low < previous swing low (makes LL)
        asset1_ll = asset1_curr_low['price'] < asset1_prev_low['price']
        
        # Asset2: current swing low >= previous swing low (doesn't make LL)
        asset2_no_ll = asset2_curr_low['price'] >= asset2_prev_low['price']
        
        return asset1_ll and asset2_no_ll
    
    def check_smt_invalidation(self, smt_data, asset1_data, asset2_data):
        """Check if SMT has been invalidated by price action"""
        if not smt_data or smt_data['signal_key'] in self.invalidated_smts:
            return True  # Already invalidated or no data
            
        direction = smt_data['direction']
        critical_level = smt_data['critical_level']
        
        # Check for invalidation based on direction
        if direction == 'bearish':
            # Bearish SMT invalidated if price trades ABOVE critical level (highest high)
            asset1_current_high = asset1_data['high'].max() if not asset1_data.empty else None
            asset2_current_high = asset2_data['high'].max() if not asset2_data.empty else None
            
            if (asset1_current_high and asset1_current_high > critical_level) or \
               (asset2_current_high and asset2_current_high > critical_level):
                logger.info(f"❌ BEARISH SMT INVALIDATED: Price above critical level {critical_level:.4f}")
                self.invalidated_smts.add(smt_data['signal_key'])
                return True
                
        elif direction == 'bullish':
            # Bullish SMT invalidated if price trades BELOW critical level (lowest low)
            asset1_current_low = asset1_data['low'].min() if not asset1_data.empty else None
            asset2_current_low = asset2_data['low'].min() if not asset2_data.empty else None
            
            if (asset1_current_low and asset1_current_low < critical_level) or \
               (asset2_current_low and asset2_current_low < critical_level):
                logger.info(f"❌ BULLISH SMT INVALIDATED: Price below critical level {critical_level:.4f}")
                self.invalidated_smts.add(smt_data['signal_key'])
                return True
        
        return False
    
    def _is_duplicate_signal(self, smt_data):
        """Check if this signal has been sent too many times"""
        signal_key = smt_data.get('signal_key')
        if not signal_key:
            return False
            
        # Check if invalidated
        if signal_key in self.invalidated_smts:
            return True
            
        count = self.signal_counts.get(signal_key, 0)
        if count >= 2:  # Max 2 signals per unique SMT
            logger.info(f"⚠️ Skipping duplicate SMT signal: {signal_key} (count: {count})")
            return True
            
        return False
    
    def _update_signal_count(self, signal_key):
        """Update count for a signal key"""
        self.signal_counts[signal_key] = self.signal_counts.get(signal_key, 0) + 1
        
        # Clean up old signals (keep only last 100 keys to prevent memory issues)
        if len(self.signal_counts) > 100:
            # Remove oldest keys
            keys_to_remove = list(self.signal_counts.keys())[:50]
            for key in keys_to_remove:
                del self.signal_counts[key]

# ================================
# ROBUST SIGNAL BUILDER
# ================================

class RobustSignalBuilder:
    def __init__(self, pair_group):
        self.pair_group = pair_group
        self.active_crt = None
        self.active_psp = None
        self.active_smts = {}  # Track SMTs by cycle
        self.signal_strength = 0
        self.criteria = []
        self.creation_time = datetime.now(NY_TZ)
        self.crt_timeframe = None
        self.status = "SCANNING_ALL"
        
    def add_smt_signal(self, smt_data, crt_direction=None):
        """Add SMT signal from any cycle - ONLY if direction matches CRT"""
        if not smt_data:
            return False
            
        cycle = smt_data['cycle']
        direction = smt_data['direction']
        
        # CHECK: SMT direction must match CRT direction if CRT exists
        if self.active_crt and direction != self.active_crt['direction']:
            logger.info(f"⚠️ SMT direction mismatch: CRT {self.active_crt['direction']} vs SMT {direction} - skipping")
            return False
            
        # Store SMT by cycle
        self.active_smts[cycle] = smt_data
        self.signal_strength += 2
        self.criteria.append(f"SMT {cycle}: {direction} {smt_data['quarters']}")
        
        logger.info(f"🔷 {self.pair_group}: {cycle} {direction} SMT detected")
        
        # Check if we have multiple SMTs in same direction
        self._check_multiple_smts()
        
        return True
    
    def _check_multiple_smts(self):
        """Check if we have multiple SMTs in same direction"""
        if len(self.active_smts) < 2:
            return
            
        # Group SMTs by direction
        bullish_smts = []
        bearish_smts = []
        
        for cycle, smt in self.active_smts.items():
            if smt['direction'] == 'bullish':
                bullish_smts.append(smt)
            else:
                bearish_smts.append(smt)
        
        # If we have 2+ SMTs in same direction, signal is stronger
        if len(bullish_smts) >= 2:
            self.status = f"MULTIPLE_BULLISH_SMTS_READY"
            logger.info(f"🎯 {self.pair_group}: Multiple bullish SMTs confirmed!")
            
        elif len(bearish_smts) >= 2:
            self.status = f"MULTIPLE_BEARISH_SMTS_READY" 
            logger.info(f"🎯 {self.pair_group}: Multiple bearish SMTs confirmed!")
    
    def set_crt_signal(self, crt_data, timeframe):
        """Set CRT signal from specific timeframe"""
        if crt_data and not self.active_crt:
            self.active_crt = crt_data
            self.crt_timeframe = timeframe
            self.signal_strength += 3
            self.criteria.append(f"CRT {timeframe}: {crt_data['direction']}")
            self.status = f"CRT_{crt_data['direction'].upper()}_WAITING_SMT"
            logger.info(f"🔷 {self.pair_group}: {timeframe} {crt_data['direction']} CRT detected → Waiting for SMT confirmation")
            
            # Remove any SMTs that don't match CRT direction
            self._remove_mismatched_smts()
            
            return True
        return False
    
    def _remove_mismatched_smts(self):
        """Remove SMTs that don't match the current CRT direction"""
        if not self.active_crt:
            return
            
        crt_direction = self.active_crt['direction']
        smts_to_remove = []
        
        for cycle, smt in self.active_smts.items():
            if smt['direction'] != crt_direction:
                smts_to_remove.append(cycle)
                logger.info(f"🔄 Removing mismatched SMT: {cycle} {smt['direction']} (CRT is {crt_direction})")
        
        for cycle in smts_to_remove:
            self._remove_smt(cycle)
    
    def _remove_smt(self, cycle):
        """Remove an SMT and adjust signal strength"""
        if cycle in self.active_smts:
            del self.active_smts[cycle]
            self.signal_strength = max(0, self.signal_strength - 2)
            # Remove from criteria
            self.criteria = [c for c in self.criteria if not c.startswith(f"SMT {cycle}:")]
    
    def set_psp_signal(self, psp_data):
        """Set PSP signal (must be on previous candle of CRT)"""
        if psp_data and self.active_crt:
            # Check if PSP is on same timeframe and approximate time as CRT's previous candle
            time_diff = abs((psp_data['timestamp'] - self.active_crt['timestamp']).total_seconds())
            if time_diff < 3600:  # Within 1 hour (same general period)
                self.active_psp = psp_data
                self.signal_strength += 2
                self.criteria.append(f"PSP {psp_data['timeframe']}: {psp_data['asset1_color']}/{psp_data['asset2_color']}")
                self.status = f"CRT_PSP_{self.active_crt['direction'].upper()}_WAITING_SMT"
                logger.info(f"🔷 {self.pair_group}: PSP confirmed on previous candle → Waiting for SMT")
                return True
        return False
    
    def is_signal_ready(self):
        """Check if we have complete signal"""
        # Path 1: Multiple SMTs in same direction
        multiple_smts = len(self.active_smts) >= 2
        
        # Path 2: CRT + any SMT (direction already matched)
        crt_smt = self.active_crt and len(self.active_smts) >= 1
        
        # Path 3: CRT + PSP + any SMT (direction already matched)
        crt_psp_smt = self.active_crt and self.active_psp and len(self.active_smts) >= 1
        
        return (multiple_smts or crt_smt or crt_psp_smt) and self.signal_strength >= 5
    
    def get_required_cycles(self):
        """Get which cycles to scan based on current signals"""
        # Always scan all cycles for SMT
        return ['monthly', 'weekly', 'daily', '90min']
    
    def get_sleep_cycle(self):
        """Get which cycle to use for sleep timing"""
        # If we have active signals, use the smallest timeframe cycle
        if self.active_crt:
            # For CRT, sleep until next candle of CRT timeframe
            return None  # Special handling for CRT
            
        # For SMT, use the smallest cycle we're monitoring
        if '90min' in self.active_smts:
            return '90min'
        elif 'daily' in self.active_smts:
            return 'daily'
        elif 'weekly' in self.active_smts:
            return 'weekly'
        else:
            return 'monthly'
    
    def get_progress_status(self):
        """Get current progress status for logging"""
        return self.status
    
    def get_signal_details(self):
        """Get complete signal details"""
        if not self.is_signal_ready():
            return None
            
        # Determine primary direction from SMTs or CRT
        direction = None
        if self.active_smts:
            # Use direction from first SMT
            first_smt = next(iter(self.active_smts.values()))
            direction = first_smt['direction']
        elif self.active_crt:
            direction = self.active_crt['direction']
        
        if not direction:
            return None
        
        # Determine signal path
        if len(self.active_smts) >= 2:
            path = "MULTIPLE_SMTS"
        elif self.active_crt and self.active_psp:
            path = "CRT_PSP_SMT"
        elif self.active_crt:
            path = "CRT_SMT"
        else:
            path = "UNKNOWN"
        
        return {
            'pair_group': self.pair_group,
            'direction': direction,
            'strength': self.signal_strength,
            'path': path,
            'criteria': self.criteria.copy(),
            'crt': self.active_crt,
            'psp': self.active_psp,
            'smts': self.active_smts.copy(),
            'timestamp': datetime.now(NY_TZ)
        }
    
    def is_expired(self):
        """Check if signal builder has expired (too old)"""
        expiry_time = timedelta(minutes=30)
        expired = datetime.now(NY_TZ) - self.creation_time > expiry_time
        if expired:
            logger.info(f"⏰ {self.pair_group}: Signal builder expired (30min timeout)")
        return expired
    
    def reset(self):
        """Reset builder"""
        self.active_crt = None
        self.active_psp = None
        self.active_smts = {}
        self.signal_strength = 0
        self.criteria = []
        self.crt_timeframe = None
        self.creation_time = datetime.now(NY_TZ)
        self.status = "SCANNING_ALL"
        logger.info(f"🔄 {self.pair_group}: Signal builder reset")

# ================================
# ROBUST TRADING SYSTEM
# ================================

class RobustTradingSystem:
    def __init__(self, pair_group, pair_config):
        self.pair_group = pair_group
        self.pair_config = pair_config
        self.pair1 = pair_config['pair1']
        self.pair2 = pair_config['pair2']
        
        # Initialize components
        self.timing_manager = RobustTimingManager()
        self.quarter_manager = QuarterManager()
        self.crt_detector = RobustCRTDetector(self.timing_manager)
        self.psp_detector = RobustPSPDetector()
        self.smt_detector = RobustSMTDetector(pair_config)  # FIXED: Pass pair_config
        self.signal_builder = RobustSignalBuilder(pair_group)
        
        # Data storage
        self.market_data = {self.pair1: {}, self.pair2: {}}
        
        logger.info(f"🎯 Initialized ROBUST trading system for {self.pair1}/{self.pair2}")
    
    async def run_robust_analysis(self, api_key):
        """Run robust analysis - always scan all timeframes"""
        try:
            # Log current status
            current_status = self.signal_builder.get_progress_status()
            logger.info(f"📊 {self.pair_group}: Current status - {current_status}")
            
            # Fetch ALL data needed for analysis
            await self._fetch_all_data(api_key)
            
            # Check SMT invalidations FIRST
            await self._check_smt_invalidations()
            
            # ALWAYS SCAN ALL PATTERNS
            logger.info(f"🔍 {self.pair_group}: Scanning ALL patterns")
            
            # Step 1: Scan for SMT on ALL cycles
            await self._scan_all_smt()
            
            # Step 2: Scan for CRT signals (if no multiple SMTs yet)
            if not self.signal_builder.is_signal_ready():
                await self._scan_crt_signals()
            
            # Step 3: If we have CRT, scan for PSP on previous candle
            if self.signal_builder.active_crt and not self.signal_builder.active_psp:
                await self._scan_psp_previous_candle()
            
            # Check if signal is complete
            if self.signal_builder.is_signal_ready():
                signal = self.signal_builder.get_signal_details()
                if signal:
                    logger.info(f"🎯 {self.pair_group}: ROBUST SIGNAL COMPLETE via {signal['path']}")
                    self.signal_builder.reset()
                    return signal
            
            # Check if expired
            if self.signal_builder.is_expired():
                self.signal_builder.reset()
            
            logger.info(f"✅ {self.pair_group}: Robust analysis complete - no signal")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error in robust analysis for {self.pair_group}: {str(e)}")
            return None
    
    async def _check_smt_invalidations(self):
        """Check if any active SMTs have been invalidated by price action"""
        if not self.signal_builder.active_smts:
            return
            
        for cycle, smt in list(self.signal_builder.active_smts.items()):
            timeframe = smt['timeframe']
            asset1_data = self.market_data[self.pair1].get(timeframe)
            asset2_data = self.market_data[self.pair2].get(timeframe)
            
            if self.smt_detector.check_smt_invalidation(smt, asset1_data, asset2_data):
                # Remove invalidated SMT
                self.signal_builder._remove_smt(cycle)
                logger.info(f"🔄 Removed invalidated SMT: {cycle}")
    
    def get_sleep_time(self):
        """Calculate sleep time until next relevant candle"""
        sleep_cycle = self.signal_builder.get_sleep_cycle()
        
        if sleep_cycle:
            # Sleep until next candle of the current cycle's timeframe
            return self.timing_manager.get_sleep_time_for_cycle(sleep_cycle)
        elif self.signal_builder.active_crt and self.signal_builder.crt_timeframe:
            # Sleep until next CRT candle
            return self.timing_manager.get_sleep_time_for_crt(self.signal_builder.crt_timeframe)
        else:
            # Default sleep time
            return BASE_INTERVAL
    
    async def _fetch_all_data(self, api_key):
        """Fetch ALL data needed for complete analysis"""
        # Always fetch all SMT timeframes
        required_timeframes = list(self.pair_config['timeframe_mapping'].values())
        
        # Also fetch CRT timeframes if we might need them
        if not self.signal_builder.is_signal_ready():
            for tf in CRT_TIMEFRAMES:
                if tf not in required_timeframes:
                    required_timeframes.append(tf)
        
        # Fetch data for all required timeframes
        for pair in [self.pair1, self.pair2]:
            for tf in required_timeframes:
                try:
                    df = await asyncio.get_event_loop().run_in_executor(
                        None, fetch_candles, pair, tf, 100, api_key
                    )
                    if df is not None and not df.empty:
                        self.market_data[pair][tf] = df
                        logger.debug(f"📥 Fetched {len(df)} {tf} candles for {pair}")
                    else:
                        logger.warning(f"⚠️ No data received for {pair} {tf}")
                except Exception as e:
                    logger.error(f"❌ Error fetching {pair} {tf}: {str(e)}")
    
    async def _scan_all_smt(self):
        """Scan for SMT on ALL cycles"""
        cycles = self.signal_builder.get_required_cycles()
        
        for cycle in cycles:
            timeframe = self.pair_config['timeframe_mapping'][cycle]
            pair1_data = self.market_data[self.pair1].get(timeframe)
            pair2_data = self.market_data[self.pair2].get(timeframe)
            
            if (pair1_data is None or not isinstance(pair1_data, pd.DataFrame) or pair1_data.empty or
                pair2_data is None or not isinstance(pair2_data, pd.DataFrame) or pair2_data.empty):
                continue
            
            smt_signal = self.smt_detector.detect_smt_all_cycles(pair1_data, pair2_data, cycle)
            
            if smt_signal:
                # Pass CRT direction to ensure SMT matches
                crt_direction = self.signal_builder.active_crt['direction'] if self.signal_builder.active_crt else None
                self.signal_builder.add_smt_signal(smt_signal, crt_direction)
    
    async def _scan_crt_signals(self):
        """Scan for CRT signals on all timeframes"""
        for timeframe in CRT_TIMEFRAMES:
            pair1_data = self.market_data[self.pair1].get(timeframe)
            pair2_data = self.market_data[self.pair2].get(timeframe)
            
            if (pair1_data is None or not isinstance(pair1_data, pd.DataFrame) or pair1_data.empty or
                pair2_data is None or not isinstance(pair2_data, pd.DataFrame) or pair2_data.empty):
                continue
            
            # Check CRT for both assets
            crt_asset1 = self.crt_detector.calculate_crt_current_candle(pair1_data)
            crt_asset2 = self.crt_detector.calculate_crt_current_candle(pair2_data)
            
            # We need at least one asset to have CRT
            crt_signal = crt_asset1 if crt_asset1 else crt_asset2
            
            if crt_signal and self.signal_builder.set_crt_signal(crt_signal, timeframe):
                logger.info(f"🔷 {self.pair_group}: Fresh CRT detected on {timeframe}")
                break
    
    async def _scan_psp_previous_candle(self):
        """Scan for PSP on the previous candle (candle 2 of CRT)"""
        if not self.signal_builder.active_crt or not self.signal_builder.crt_timeframe:
            return
        
        timeframe = self.signal_builder.crt_timeframe
        pair1_data = self.market_data[self.pair1].get(timeframe)
        pair2_data = self.market_data[self.pair2].get(timeframe)
        
        if (pair1_data is None or not isinstance(pair1_data, pd.DataFrame) or pair1_data.empty or
            pair2_data is None or not isinstance(pair2_data, pd.DataFrame) or pair2_data.empty):
            return
        
        psp_signal = self.psp_detector.detect_psp_previous_candle(pair1_data, pair2_data, timeframe)
        if psp_signal:
            self.signal_builder.set_psp_signal(psp_signal)
            logger.info(f"🔷 {self.pair_group}: PSP confirmed on previous candle")

# ================================
# ROBUST MAIN MANAGER
# ================================

class RobustTradingManager:
    def __init__(self, api_key, telegram_token, chat_id):
        self.api_key = api_key
        self.telegram_token = telegram_token
        self.chat_id = chat_id
        self.timing_manager = RobustTimingManager()
        self.trading_systems = {}
        
        # Initialize trading systems for all pairs
        for pair_group, pair_config in TRADING_PAIRS.items():
            self.trading_systems[pair_group] = RobustTradingSystem(pair_group, pair_config)
        
        logger.info(f"🎯 Initialized ROBUST trading manager with {len(self.trading_systems)} pair groups")
    
    async def run_robust_systems(self):
        """Run all trading systems with robust timing"""
        logger.info("🎯 Starting ROBUST Multi-Pair Trading System...")
        
        while True:
            try:
                # Run analysis for all pairs
                tasks = []
                sleep_times = []
                
                for pair_group, system in self.trading_systems.items():
                    task = asyncio.create_task(
                        system.run_robust_analysis(self.api_key),
                        name=f"analysis_{pair_group}"
                    )
                    tasks.append(task)
                    # Get individual sleep time for each system
                    sleep_times.append(system.get_sleep_time())
                
                # Wait for all analyses to complete
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                signals = []
                for i, result in enumerate(results):
                    pair_group = list(self.trading_systems.keys())[i]
                    if isinstance(result, Exception):
                        logger.error(f"❌ Analysis task failed for {pair_group}: {str(result)}")
                    elif result is not None:
                        signals.append(result)
                        logger.info(f"🎯 ROBUST SIGNAL FOUND for {pair_group}")
                
                # Send signals to Telegram
                if signals:
                    await self._process_signals(signals)
                
                # Use the minimum sleep time from all systems
                sleep_time = min(sleep_times) if sleep_times else BASE_INTERVAL
                logger.info(f"⏰ Robust cycle complete. Sleeping for {sleep_time:.1f} seconds")
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"❌ Error in robust main loop: {str(e)}")
                await asyncio.sleep(60)
    
    async def _process_signals(self, signals):
        """Process and send signals to Telegram"""
        for signal in signals:
            try:
                message = self._format_robust_signal_message(signal)
                success = send_telegram(message, self.telegram_token, self.chat_id)
                
                if success:
                    logger.info(f"📤 Robust signal sent to Telegram for {signal['pair_group']}")
                else:
                    logger.error(f"❌ Failed to send robust signal for {signal['pair_group']}")
                    
            except Exception as e:
                logger.error(f"❌ Error processing robust signal: {str(e)}")
    
    def _format_robust_signal_message(self, signal):
        """Format robust signal for Telegram"""
        pair_group = signal.get('pair_group', 'Unknown')
        direction = signal.get('direction', 'UNKNOWN').upper()
        strength = signal.get('strength', 0)
        path = signal.get('path', 'UNKNOWN')
        
        message = f"🎯 *ROBUST TRADING SIGNAL* 🎯\n\n"
        message += f"*Pair Group:* {pair_group.replace('_', ' ').title()}\n"
        message += f"*Direction:* {direction}\n"
        message += f"*Strength:* {strength}/9\n"
        message += f"*Path:* {path}\n\n"
        
        # Add criteria
        if 'criteria' in signal:
            message += "*Signal Criteria:*\n"
            for criterion in signal['criteria']:
                message += f"• {criterion}\n"
        
        # Add SMT details
        if 'smts' in signal and signal['smts']:
            message += f"\n*SMT Details:*\n"
            for cycle, smt in signal['smts'].items():
                message += f"• {cycle}: {smt['details']}\n"
        
        message += f"\n*Detection Time:* {signal['timestamp'].strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
        message += f"\n#RobustSignal #{pair_group} #{path}"
        
        return message

# ================================
# MAIN EXECUTION
# ================================

async def main():
    """Main entry point"""
    logger.info("🎯 Starting ROBUST Multi-Pair SMT Trading System")
    
    # Get credentials from environment
    api_key = os.getenv('OANDA_API_KEY')
    telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
    telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if not all([api_key, telegram_token, telegram_chat_id]):
        logger.error("❌ Missing required environment variables:")
        logger.error("OANDA_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID")
        logger.info("💡 Please set these environment variables and try again")
        return
    
    try:
        # Initialize manager
        manager = RobustTradingManager(api_key, telegram_token, telegram_chat_id)
        
        # Run all systems with robust timing
        await manager.run_robust_systems()
        
    except KeyboardInterrupt:
        logger.info("🛑 System stopped by user")
    except Exception as e:
        logger.error(f"💥 Fatal error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    # Run the system
    asyncio.run(main())
