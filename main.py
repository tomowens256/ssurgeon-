#!/usr/bin/env python3
"""
ADVANCED SMT TRADING SYSTEM WITH INTELLIGENT MINDSET
- Looks back for PSP when SMT forms
- Tracks multiple SMTs in same direction
- Strength-based decision making
- Continuous PSP monitoring
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
    },
    'fx_triad': {
        'pair1': 'GBP_USD',
        'pair2': 'EUR_USD',
        'timeframe_mapping': {
            'monthly': 'H4',
            'weekly': 'H1',
            'daily': 'M15',
            '90min': 'M5'
        }
    }
}

CRT_TIMEFRAMES = ['H1', 'H2', 'H3', 'H4', 'H6', 'H8', 'H12']

CYCLE_SLEEP_TIMEFRAMES = {
    'monthly': 'H4',
    'weekly': 'H1',  
    'daily': 'M15',
    '90min': 'M5'
}

NY_TZ = pytz.timezone('America/New_York')
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
        logging.FileHandler('advanced_trading_system.log'),
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
# ADVANCED TIMING MANAGER
# ================================

class AdvancedTimingManager:
    """Advanced timing manager with SMART sleep calculations"""
    
    def __init__(self):
        self.ny_tz = pytz.timezone('America/New_York')
        
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
    
    def is_crt_fresh(self, crt_timestamp, max_age_minutes=1):
        """Check if CRT signal is fresh"""
        if not crt_timestamp:
            return False
            
        current_time = datetime.now(self.ny_tz)
        age_seconds = (current_time - crt_timestamp).total_seconds()
        
        return age_seconds <= (max_age_minutes * 60)

# ================================
# ADVANCED QUARTER MANAGER
# ================================

class AdvancedQuarterManager:
    """Advanced quarter management with enhanced tracking"""
    
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
        day = timestamp.day
        if 1 <= day <= 7: return 'q1'
        elif 8 <= day <= 14: return 'q2'
        elif 15 <= day <= 21: return 'q3'
        elif 22 <= day <= 28: return 'q4'
        else: return 'q_less'
    
    def _get_weekly_quarter(self, timestamp):
        weekday = timestamp.weekday()
        if weekday == 0: return 'q1'
        elif weekday == 1: return 'q2'
        elif weekday == 2: return 'q3'
        elif weekday == 3: return 'q4'
        else: return 'q_less'
    
    def _get_daily_quarter(self, timestamp):
        hour = timestamp.hour
        if 0 <= hour < 6: return 'q2'
        elif 6 <= hour < 12: return 'q3'
        elif 12 <= hour < 18: return 'q4'
        else: return 'q1'
    
    def _get_90min_quarter_fixed(self, timestamp):
        daily_quarter = self._get_daily_quarter(timestamp)
        hour = timestamp.hour
        minute = timestamp.minute
        total_minutes = hour * 60 + minute
        
        boundaries = {
            'q1': [
                (18*60, 19*60+30, 'q1'),
                (19*60+30, 21*60, 'q2'),  
                (21*60, 22*60+30, 'q3'),
                (22*60+30, 24*60, 'q4')
            ],
            'q2': [
                (0, 1*60+30, 'q1'),
                (1*60+30, 3*60, 'q2'),
                (3*60, 4*60+30, 'q3'),
                (4*60+30, 6*60, 'q4')
            ],
            'q3': [
                (6*60, 7*60+30, 'q1'),
                (7*60+30, 9*60, 'q2'),
                (9*60, 10*60+30, 'q3'),
                (10*60+30, 12*60, 'q4')
            ],
            'q4': [
                (12*60, 13*60+30, 'q1'),
                (13*60+30, 15*60, 'q2'),
                (15*60, 16*60+30, 'q3'),
                (16*60+30, 18*60, 'q4')
            ]
        }
        
        for start_min, end_min, quarter in boundaries[daily_quarter]:
            if start_min <= total_minutes < end_min:
                return quarter
        
        return 'q_less'
    
    def get_valid_quarter_pairs(self, current_quarter, cycle_type):
        """Get valid consecutive quarter pairs to check"""
        quarter_sequence = ['q1', 'q2', 'q3', 'q4']
        
        try:
            current_idx = quarter_sequence.index(current_quarter)
        except ValueError:
            return []
        
        valid_pairs = []
        
        if current_idx >= 2:
            valid_pairs.append((quarter_sequence[current_idx-2], quarter_sequence[current_idx-1]))
        
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
# ADVANCED SWING DETECTOR
# ================================

class AdvancedSwingDetector:
    """Enhanced swing detection with better pattern recognition"""
    
    @staticmethod
    def find_swing_highs_lows(df, lookback=3):
        """Find swing highs and swing lows in a DataFrame"""
        if df is None or not isinstance(df, pd.DataFrame) or len(df) < lookback + 1:
            return [], []
        
        swing_highs = []
        swing_lows = []
        
        for i in range(lookback, len(df) - lookback):
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
        if not swing_highs:
            return None
        return max(swing_highs, key=lambda x: x['price'])
    
    @staticmethod
    def get_lowest_swing_low(swing_lows):
        if not swing_lows:
            return None
        return min(swing_lows, key=lambda x: x['price'])

# ================================
# ADVANCED PATTERN DETECTORS
# ================================

class AdvancedCRTDetector:
    """Enhanced CRT detector with better validation"""
    
    def __init__(self, timing_manager):
        self.timing_manager = timing_manager
    
    def calculate_crt_current_candle(self, df):
        """Calculate CRT only on the current (incomplete) candle"""
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
            c2_range = float(c2['high']) - float(c2['low'])
            c2_mid = float(c2['low']) + 0.5 * c2_range
            
            buy_crt = (float(c2['low']) < float(c1['low']) and 
                      float(c2['close']) > float(c1['low']) and 
                      float(c3['open']) > c2_mid)
            
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

class AdvancedPSPDetector:
    """Advanced PSP detector with lookback capability"""
    
    @staticmethod
    def detect_psp_closed_candle(asset1_data, asset2_data, timeframe):
        """Detect PSP on the most recent CLOSED candle"""
        if (asset1_data is None or not isinstance(asset1_data, pd.DataFrame) or asset1_data.empty or
            asset2_data is None or not isinstance(asset2_data, pd.DataFrame) or asset2_data.empty):
            return None
        
        asset1_complete = asset1_data[asset1_data['complete'] == True]
        asset2_complete = asset2_data[asset2_data['complete'] == True]
        
        if asset1_complete.empty or asset2_complete.empty:
            return None
            
        asset1_candle = asset1_complete.iloc[-1]
        asset2_candle = asset2_complete.iloc[-1]
        
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
                    'timestamp': asset1_candle['time'],
                    'candle_time': asset1_candle['time']
                }
        except (ValueError, TypeError) as e:
            logger.error(f"Error in PSP calculation: {e}")
            return None
        
        return None
    
    @staticmethod
    def detect_psp_last_n_candles(asset1_data, asset2_data, timeframe, n=5):
        """Look back at last N closed candles for PSP"""
        if (asset1_data is None or not isinstance(asset1_data, pd.DataFrame) or asset1_data.empty or
            asset2_data is None or not isinstance(asset2_data, pd.DataFrame) or asset2_data.empty):
            return None
        
        asset1_complete = asset1_data[asset1_data['complete'] == True]
        asset2_complete = asset2_data[asset2_data['complete'] == True]
        
        if asset1_complete.empty or asset2_complete.empty:
            return None
        
        # Get last N complete candles
        asset1_recent = asset1_complete.tail(n)
        asset2_recent = asset2_complete.tail(n)
        
        # Look for PSP in recent candles (most recent first)
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
                    return {
                        'timeframe': timeframe,
                        'asset1_color': asset1_color,
                        'asset2_color': asset2_color,
                        'timestamp': asset1_candle['time'],
                        'candle_time': asset1_candle['time'],
                        'candles_ago': len(asset1_recent) - i - 1
                    }
            except (ValueError, TypeError) as e:
                logger.error(f"Error in PSP calculation: {e}")
                continue
        
        return None

# ================================
# INTELLIGENT SMT DETECTOR WITH MINDSET
# ================================

class IntelligentSMTDetector:
    """INTELLIGENT SMT detector with advanced decision making"""
    
    def __init__(self, pair_config):
        self.smt_history = []
        self.quarter_manager = AdvancedQuarterManager()
        self.swing_detector = AdvancedSwingDetector()
        self.signal_counts = {}
        self.invalidated_smts = set()
        self.pair_config = pair_config
        self.active_smts = {}  # Track active SMTs by cycle and quarters
        
        # PSP tracking for each SMT
        self.smt_psp_tracking = {}  # signal_key -> {'psp_found': bool, 'check_count': int, 'max_checks': 20}
        
    def detect_smt_all_cycles(self, asset1_data, asset2_data, cycle_type):
        """Detect SMT for a specific cycle with enhanced logic"""
        try:
            if (asset1_data is None or not isinstance(asset1_data, pd.DataFrame) or asset1_data.empty or
                asset2_data is None or not isinstance(asset2_data, pd.DataFrame) or asset2_data.empty):
                return None
            
            current_quarters = self.quarter_manager.detect_current_quarters()
            current_quarter = current_quarters.get(cycle_type)
            
            if not current_quarter:
                return None
            
            valid_pairs = self.quarter_manager.get_valid_quarter_pairs(current_quarter, cycle_type)
            
            if not valid_pairs:
                return None
            
            asset1_quarters = self.quarter_manager.group_candles_by_quarters(asset1_data, cycle_type)
            asset2_quarters = self.quarter_manager.group_candles_by_quarters(asset2_data, cycle_type)
            
            if not asset1_quarters or not asset2_quarters:
                return None
            
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
                    # Initialize PSP tracking for this SMT
                    signal_key = smt_result['signal_key']
                    if signal_key not in self.smt_psp_tracking:
                        self.smt_psp_tracking[signal_key] = {
                            'psp_found': False,
                            'check_count': 0,
                            'max_checks': 20,  # Check up to 20 new candles
                            'last_check': datetime.now(NY_TZ)
                        }
                    
                    return smt_result
            
            return None
            
        except Exception as e:
            logger.error(f"Error in SMT detection for {cycle_type}: {str(e)}")
            return None
    
    def check_psp_for_smt(self, smt_data, asset1_data, asset2_data):
        """Check for PSP in past 5 candles for a specific SMT"""
        if not smt_data:
            return None
            
        signal_key = smt_data['signal_key']
        timeframe = smt_data['timeframe']
        
        # Update tracking
        if signal_key in self.smt_psp_tracking:
            tracking = self.smt_psp_tracking[signal_key]
            tracking['check_count'] += 1
            tracking['last_check'] = datetime.now(NY_TZ)
        
        # Look for PSP in last 5 candles
        psp_detector = AdvancedPSPDetector()
        psp_signal = psp_detector.detect_psp_last_n_candles(asset1_data, asset2_data, timeframe, n=5)
        
        if psp_signal:
            logger.info(f"🎯 PSP FOUND for SMT {smt_data['cycle']} {smt_data['quarters']} - {psp_signal['candles_ago']} candles ago")
            
            # Mark PSP as found for this SMT
            if signal_key in self.smt_psp_tracking:
                self.smt_psp_tracking[signal_key]['psp_found'] = True
            
            return psp_signal
        
        return None
    
    def should_keep_checking_smt(self, smt_data):
        """Determine if we should keep checking for PSP for this SMT"""
        if not smt_data:
            return False
            
        signal_key = smt_data['signal_key']
        
        if signal_key not in self.smt_psp_tracking:
            return False
        
        tracking = self.smt_psp_tracking[signal_key]
        
        # Stop checking if we found PSP or reached max checks
        if tracking['psp_found'] or tracking['check_count'] >= tracking['max_checks']:
            return False
        
        # Stop checking if SMT is invalidated
        if signal_key in self.invalidated_smts:
            return False
        
        return True
    
    def get_smts_by_direction(self):
        """Group active SMTs by direction for strength analysis"""
        bullish_smts = []
        bearish_smts = []
        
        for smt in self.smt_history[-20:]:  # Check recent SMTs
            if smt['signal_key'] in self.invalidated_smts:
                continue
                
            if smt['direction'] == 'bullish':
                bullish_smts.append(smt)
            else:
                bearish_smts.append(smt)
        
        return bullish_smts, bearish_smts
    
    def get_stronger_direction(self):
        """Determine which direction has stronger SMT evidence"""
        bullish_smts, bearish_smts = self.get_smts_by_direction()
        
        bull_strength = len(bullish_smts)
        bear_strength = len(bearish_smts)
        
        logger.info(f"💪 Strength Analysis - Bullish: {bull_strength} SMTs, Bearish: {bear_strength} SMTs")
        
        if bull_strength > bear_strength and bull_strength >= 2:
            return 'bullish', bull_strength
        elif bear_strength > bull_strength and bear_strength >= 2:
            return 'bearish', bear_strength
        else:
            return None, 0
    
    def _compare_quarters_swing_based(self, asset1_prev, asset1_curr, asset2_prev, asset2_curr, cycle_type, prev_q, curr_q):
        """Compare two consecutive quarters using swing highs/lows"""
        try:
            if (asset1_prev.empty or asset1_curr.empty or 
                asset2_prev.empty or asset2_curr.empty):
                return None
            
            asset1_prev_swing_highs, asset1_prev_swing_lows = self.swing_detector.find_swing_highs_lows(asset1_prev)
            asset1_curr_swing_highs, asset1_curr_swing_lows = self.swing_detector.find_swing_highs_lows(asset1_curr)
            
            asset2_prev_swing_highs, asset2_prev_swing_lows = self.swing_detector.find_swing_highs_lows(asset2_prev)
            asset2_curr_swing_highs, asset2_curr_swing_lows = self.swing_detector.find_swing_highs_lows(asset2_curr)
            
            asset1_prev_highest = self.swing_detector.get_highest_swing_high(asset1_prev_swing_highs)
            asset1_curr_highest = self.swing_detector.get_highest_swing_high(asset1_curr_swing_highs)
            asset1_prev_lowest = self.swing_detector.get_lowest_swing_low(asset1_prev_swing_lows)
            asset1_curr_lowest = self.swing_detector.get_lowest_swing_low(asset1_curr_swing_lows)
            
            asset2_prev_highest = self.swing_detector.get_highest_swing_high(asset2_prev_swing_highs)
            asset2_curr_highest = self.swing_detector.get_highest_swing_high(asset2_curr_swing_highs)
            asset2_prev_lowest = self.swing_detector.get_lowest_swing_low(asset2_prev_swing_lows)
            asset2_curr_lowest = self.swing_detector.get_lowest_swing_low(asset2_curr_swing_lows)
            
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
                critical_level = asset1_curr_highest['price']
                
            elif bullish_smt:
                direction = 'bullish'
                smt_type = 'Lower Swing Low'
                asset1_action = f"made lower swing low ({asset1_prev_lowest['price']:.4f} → {asset1_curr_lowest['price']:.4f})"
                asset2_action = f"no lower swing low ({asset2_prev_lowest['price']:.4f} → {asset2_curr_lowest['price']:.4f})"
                critical_level = asset1_curr_lowest['price']
                
            else:
                return None
            
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
                'critical_level': critical_level,
                'timeframe': self.pair_config['timeframe_mapping'][cycle_type]
            }
            
            self.smt_history.append(smt_data)
            self._update_signal_count(smt_data['signal_key'])
            
            logger.info(f"🎯 INTELLIGENT SMT: {direction} {cycle_type} {prev_q}→{curr_q}")
            logger.info(f"   Asset1: {asset1_action}")
            logger.info(f"   Asset2: {asset2_action}")
            
            return smt_data
            
        except Exception as e:
            logger.error(f"Error comparing quarters {prev_q}→{curr_q}: {str(e)}")
            return None
    
    def _check_bearish_smt(self, asset1_prev_high, asset1_curr_high, asset2_prev_high, asset2_curr_high):
        if not all([asset1_prev_high, asset1_curr_high, asset2_prev_high, asset2_curr_high]):
            return False
        
        asset1_hh = asset1_curr_high['price'] > asset1_prev_high['price']
        asset2_no_hh = asset2_curr_high['price'] <= asset2_prev_high['price']
        
        return asset1_hh and asset2_no_hh
    
    def _check_bullish_smt(self, asset1_prev_low, asset1_curr_low, asset2_prev_low, asset2_curr_low):
        if not all([asset1_prev_low, asset1_curr_low, asset2_prev_low, asset2_curr_low]):
            return False
        
        asset1_ll = asset1_curr_low['price'] < asset1_prev_low['price']
        asset2_no_ll = asset2_curr_low['price'] >= asset2_prev_low['price']
        
        return asset1_ll and asset2_no_ll
    
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
                logger.info(f"❌ BEARISH SMT INVALIDATED: Price above critical level {critical_level:.4f}")
                self.invalidated_smts.add(smt_data['signal_key'])
                return True
                
        elif direction == 'bullish':
            asset1_current_low = asset1_data['low'].min() if not asset1_data.empty else None
            asset2_current_low = asset2_data['low'].min() if not asset2_data.empty else None
            
            if (asset1_current_low and asset1_current_low < critical_level) or \
               (asset2_current_low and asset2_current_low < critical_level):
                logger.info(f"❌ BULLISH SMT INVALIDATED: Price below critical level {critical_level:.4f}")
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
        if count >= 2:
            logger.info(f"⚠️ Skipping duplicate SMT signal: {signal_key} (count: {count})")
            return True
            
        return False
    
    def _update_signal_count(self, signal_key):
        self.signal_counts[signal_key] = self.signal_counts.get(signal_key, 0) + 1
        
        if len(self.signal_counts) > 100:
            keys_to_remove = list(self.signal_counts.keys())[:50]
            for key in keys_to_remove:
                del self.signal_counts[key]

# ================================
# INTELLIGENT SIGNAL BUILDER WITH MINDSET
# ================================

class IntelligentSignalBuilder:
    """INTELLIGENT signal builder with advanced decision making"""
    
    def __init__(self, pair_group):
        self.pair_group = pair_group
        self.active_crt = None
        self.active_smts = {}
        self.psp_for_smts = {}  # Track PSP for each SMT
        self.signal_strength = 0
        self.criteria = []
        self.creation_time = datetime.now(NY_TZ)
        self.crt_timeframe = None
        self.status = "SCANNING_ALL"
        
        # Strength tracking
        self.bullish_strength = 0
        self.bearish_strength = 0
        self.dominant_direction = None
        
    def add_smt_signal(self, smt_data, psp_signal=None):
        """Add SMT signal with optional PSP confirmation"""
        if not smt_data:
            return False
            
        cycle = smt_data['cycle']
        direction = smt_data['direction']
        
        # Check direction match with CRT if exists
        if self.active_crt and direction != self.active_crt['direction']:
            logger.info(f"⚠️ SMT direction mismatch: CRT {self.active_crt['direction']} vs SMT {direction}")
            return False
            
        # Store SMT
        self.active_smts[cycle] = smt_data
        self.signal_strength += 2
        
        # Update strength counters
        if direction == 'bullish':
            self.bullish_strength += 1
        else:
            self.bearish_strength += 1
            
        # Update dominant direction
        self._update_dominant_direction()
        
        # Store PSP if provided
        if psp_signal:
            self.psp_for_smts[cycle] = psp_signal
            self.signal_strength += 1
            self.criteria.append(f"SMT {cycle} with PSP: {direction} {smt_data['quarters']}")
            logger.info(f"🔷 {self.pair_group}: {cycle} {direction} SMT + PSP CONFIRMED!")
        else:
            self.criteria.append(f"SMT {cycle}: {direction} {smt_data['quarters']}")
            logger.info(f"🔷 {self.pair_group}: {cycle} {direction} SMT detected - looking for PSP")
        
        # Check signal readiness
        self._check_signal_readiness()
        
        return True
    
    def set_psp_for_smt(self, cycle, psp_signal):
        """Set PSP confirmation for a specific SMT"""
        if cycle in self.active_smts and psp_signal:
            self.psp_for_smts[cycle] = psp_signal
            self.signal_strength += 1
            
            smt = self.active_smts[cycle]
            logger.info(f"🎯 {self.pair_group}: PSP CONFIRMED for {cycle} {smt['direction']} SMT!")
            
            self._check_signal_readiness()
            return True
        return False
    
    def _update_dominant_direction(self):
        """Update which direction has stronger evidence"""
        if self.bullish_strength > self.bearish_strength:
            self.dominant_direction = 'bullish'
        elif self.bearish_strength > self.bullish_strength:
            self.dominant_direction = 'bearish'
        else:
            self.dominant_direction = None
            
        logger.info(f"💪 {self.pair_group}: Strength - Bullish: {self.bullish_strength}, Bearish: {self.bearish_strength}, Dominant: {self.dominant_direction}")
    
    def _check_signal_readiness(self):
        """Check if we have enough evidence for a signal"""
        # Count SMTs with PSP
        smts_with_psp = len(self.psp_for_smts)
        total_smts = len(self.active_smts)
        
        logger.info(f"📊 {self.pair_group}: SMTs: {total_smts}, With PSP: {smts_with_psp}, Dominant: {self.dominant_direction}")
        
        # SIGNAL LOGIC 1: Multiple SMTs in same direction (2+ SMTs)
        if total_smts >= 2 and self.dominant_direction:
            smts_in_direction = self.bullish_strength if self.dominant_direction == 'bullish' else self.bearish_strength
            if smts_in_direction >= 2:
                self.status = f"MULTIPLE_{self.dominant_direction.upper()}_SMTS"
                logger.info(f"🎯 {self.pair_group}: Multiple {self.dominant_direction} SMTs confirmed!")
                return
        
        # SIGNAL LOGIC 2: SMT + PSP (any SMT with PSP confirmation)
        if smts_with_psp >= 1:
            # Get direction from first SMT with PSP
            for cycle, psp in self.psp_for_smts.items():
                if cycle in self.active_smts:
                    direction = self.active_smts[cycle]['direction']
                    self.status = f"SMT_PSP_{direction.upper()}"
                    logger.info(f"🎯 {self.pair_group}: SMT + PSP {direction} signal ready!")
                    return
        
        # SIGNAL LOGIC 3: CRT + SMT (direction already matched)
        if self.active_crt and total_smts >= 1:
            direction = self.active_crt['direction']
            self.status = f"CRT_SMT_{direction.upper()}"
            logger.info(f"🎯 {self.pair_group}: CRT + SMT {direction} signal ready!")
            return
        
        # SIGNAL LOGIC 4: CRT + PSP + SMT
        if self.active_crt and smts_with_psp >= 1:
            direction = self.active_crt['direction']
            self.status = f"CRT_PSP_SMT_{direction.upper()}"
            logger.info(f"🎯 {self.pair_group}: CRT + PSP + SMT {direction} signal ready!")
            return
    
    def set_crt_signal(self, crt_data, timeframe):
        """Set CRT signal from specific timeframe"""
        if crt_data and not self.active_crt:
            self.active_crt = crt_data
            self.crt_timeframe = timeframe
            self.signal_strength += 3
            self.criteria.append(f"CRT {timeframe}: {crt_data['direction']}")
            
            direction = crt_data['direction']
            self.status = f"CRT_{direction.upper()}_WAITING_SMT"
            logger.info(f"🔷 {self.pair_group}: {timeframe} {direction} CRT detected → Waiting for SMT confirmation")
            
            # Remove any SMTs that don't match CRT direction
            self._remove_mismatched_smts()
            
            self._check_signal_readiness()
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
            smt = self.active_smts[cycle]
            direction = smt['direction']
            
            # Adjust strength counters
            if direction == 'bullish':
                self.bullish_strength = max(0, self.bullish_strength - 1)
            else:
                self.bearish_strength = max(0, self.bearish_strength - 1)
            
            del self.active_smts[cycle]
            self.signal_strength = max(0, self.signal_strength - 2)
            
            # Remove PSP if exists
            if cycle in self.psp_for_smts:
                del self.psp_for_smts[cycle]
                self.signal_strength = max(0, self.signal_strength - 1)
            
            # Remove from criteria
            self.criteria = [c for c in self.criteria if not c.startswith(f"SMT {cycle}:")]
            
            self._update_dominant_direction()
    
    def is_signal_ready(self):
        """Check if we have complete signal with intelligent decision making"""
        return self.status.startswith(('MULTIPLE_', 'SMT_PSP_', 'CRT_SMT_', 'CRT_PSP_SMT_'))
    
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
        """Get complete signal details with intelligent analysis"""
        if not self.is_signal_ready():
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
        
        # Build signal details
        signal_data = {
            'pair_group': self.pair_group,
            'direction': direction,
            'strength': self.signal_strength,
            'path': self.status,
            'bullish_strength': self.bullish_strength,
            'bearish_strength': self.bearish_strength,
            'dominant_direction': self.dominant_direction,
            'criteria': self.criteria.copy(),
            'crt': self.active_crt,
            'psp_smts': self.psp_for_smts.copy(),
            'all_smts': self.active_smts.copy(),
            'timestamp': datetime.now(NY_TZ)
        }
        
        # Add description based on signal path
        if self.status.startswith('MULTIPLE_'):
            signal_data['description'] = f"Multiple {direction} SMTs across different cycles - Strong directional bias"
        elif self.status.startswith('SMT_PSP_'):
            signal_data['description'] = f"SMT confirmed by PSP on same timeframe - High probability setup"
        elif self.status.startswith('CRT_SMT_'):
            signal_data['description'] = f"CRT momentum with SMT confirmation - Good entry timing"
        elif self.status.startswith('CRT_PSP_SMT_'):
            signal_data['description'] = f"CRT + PSP + SMT confluence - Highest probability setup"
        
        logger.info(f"🎯 INTELLIGENT SIGNAL: {self.pair_group} {direction} via {self.status}")
        logger.info(f"📋 Description: {signal_data['description']}")
        
        return signal_data
    
    def is_expired(self):
        expiry_time = timedelta(minutes=30)
        expired = datetime.now(NY_TZ) - self.creation_time > expiry_time
        if expired:
            logger.info(f"⏰ {self.pair_group}: Signal builder expired (30min timeout)")
        return expired
    
    def reset(self):
        self.active_crt = None
        self.active_smts = {}
        self.psp_for_smts = {}
        self.signal_strength = 0
        self.criteria = []
        self.crt_timeframe = None
        self.creation_time = datetime.now(NY_TZ)
        self.status = "SCANNING_ALL"
        self.bullish_strength = 0
        self.bearish_strength = 0
        self.dominant_direction = None
        logger.info(f"🔄 {self.pair_group}: Intelligent signal builder reset")

# ================================
# INTELLIGENT TRADING SYSTEM
# ================================

class IntelligentTradingSystem:
    def __init__(self, pair_group, pair_config):
        self.pair_group = pair_group
        self.pair_config = pair_config
        self.pair1 = pair_config['pair1']
        self.pair2 = pair_config['pair2']
        
        # Initialize advanced components
        self.timing_manager = AdvancedTimingManager()
        self.quarter_manager = AdvancedQuarterManager()
        self.crt_detector = AdvancedCRTDetector(self.timing_manager)
        self.psp_detector = AdvancedPSPDetector()
        self.smt_detector = IntelligentSMTDetector(pair_config)
        self.signal_builder = IntelligentSignalBuilder(pair_group)
        
        # Data storage
        self.market_data = {self.pair1: {}, self.pair2: {}}
        
        logger.info(f"🎯 Initialized INTELLIGENT trading system for {self.pair1}/{self.pair2}")
    
    async def run_intelligent_analysis(self, api_key):
        """Run intelligent analysis with advanced decision making"""
        try:
            current_status = self.signal_builder.get_progress_status()
            logger.info(f"📊 {self.pair_group}: Current status - {current_status}")
            
            # Fetch ALL data needed for analysis
            await self._fetch_all_data(api_key)
            
            # Step 1: Check SMT invalidations and PSP tracking
            await self._check_smt_tracking()
            
            # Step 2: Scan for NEW SMT signals
            await self._scan_all_smt()
            
            # Step 3: Check for PSP for existing SMTs
            await self._check_psp_for_existing_smts()
            
            # Step 4: Scan for CRT signals (if no strong signals yet)
            if not self.signal_builder.is_signal_ready():
                await self._scan_crt_signals()
            
            # Check if signal is complete
            if self.signal_builder.is_signal_ready():
                signal = self.signal_builder.get_signal_details()
                if signal:
                    logger.info(f"🎯 {self.pair_group}: INTELLIGENT SIGNAL COMPLETE via {signal['path']}")
                    self.signal_builder.reset()
                    return signal
            
            # Check if expired
            if self.signal_builder.is_expired():
                self.signal_builder.reset()
            
            logger.info(f"✅ {self.pair_group}: Intelligent analysis complete - monitoring continues")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error in intelligent analysis for {self.pair_group}: {str(e)}")
            return None
    
    async def _check_smt_tracking(self):
        """Check SMT invalidations and update PSP tracking"""
        if not self.signal_builder.active_smts:
            return
            
        for cycle, smt in list(self.signal_builder.active_smts.items()):
            timeframe = smt['timeframe']
            asset1_data = self.market_data[self.pair1].get(timeframe)
            asset2_data = self.market_data[self.pair2].get(timeframe)
            
            # Check invalidation
            if self.smt_detector.check_smt_invalidation(smt, asset1_data, asset2_data):
                self.signal_builder._remove_smt(cycle)
                logger.info(f"🔄 Removed invalidated SMT: {cycle}")
                continue
            
            # Check if we should keep tracking this SMT for PSP
            if not self.smt_detector.should_keep_checking_smt(smt):
                logger.info(f"⏹️ Stopping PSP tracking for SMT: {cycle}")
                continue
    
    async def _check_psp_for_existing_smts(self):
        """Check for PSP confirmation for existing SMTs"""
        if not self.signal_builder.active_smts:
            return
            
        for cycle, smt in self.signal_builder.active_smts.items():
            # Skip if already has PSP
            if cycle in self.signal_builder.psp_for_smts:
                continue
                
            timeframe = smt['timeframe']
            asset1_data = self.market_data[self.pair1].get(timeframe)
            asset2_data = self.market_data[self.pair2].get(timeframe)
            
            if (asset1_data is None or not isinstance(asset1_data, pd.DataFrame) or asset1_data.empty or
                asset2_data is None or not isinstance(asset2_data, pd.DataFrame) or asset2_data.empty):
                continue
            
            # Check for PSP in last 5 candles
            psp_signal = self.smt_detector.check_psp_for_smt(smt, asset1_data, asset2_data)
            
            if psp_signal:
                self.signal_builder.set_psp_for_smt(cycle, psp_signal)
    
    async def _fetch_all_data(self, api_key):
        """Fetch ALL data needed for complete analysis"""
        required_timeframes = list(self.pair_config['timeframe_mapping'].values())
        
        if not self.signal_builder.is_signal_ready():
            for tf in CRT_TIMEFRAMES:
                if tf not in required_timeframes:
                    required_timeframes.append(tf)
        
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
                # Check for PSP immediately for this new SMT
                psp_signal = self.smt_detector.check_psp_for_smt(smt_signal, pair1_data, pair2_data)
                
                # Add SMT with PSP if found
                self.signal_builder.add_smt_signal(smt_signal, psp_signal)
    
    async def _scan_crt_signals(self):
        """Scan for CRT signals on all timeframes"""
        for timeframe in CRT_TIMEFRAMES:
            pair1_data = self.market_data[self.pair1].get(timeframe)
            pair2_data = self.market_data[self.pair2].get(timeframe)
            
            if (pair1_data is None or not isinstance(pair1_data, pd.DataFrame) or pair1_data.empty or
                pair2_data is None or not isinstance(pair2_data, pd.DataFrame) or pair2_data.empty):
                continue
            
            crt_asset1 = self.crt_detector.calculate_crt_current_candle(pair1_data)
            crt_asset2 = self.crt_detector.calculate_crt_current_candle(pair2_data)
            
            crt_signal = crt_asset1 if crt_asset1 else crt_asset2
            
            if crt_signal and self.signal_builder.set_crt_signal(crt_signal, timeframe):
                logger.info(f"🔷 {self.pair_group}: Fresh CRT detected on {timeframe}")
                break
    
    def get_sleep_time(self):
        """Calculate sleep time until next relevant candle"""
        sleep_cycle = self.signal_builder.get_sleep_cycle()
        
        if sleep_cycle:
            sleep_time = self.timing_manager.get_sleep_time_for_cycle(sleep_cycle)
            timeframe = CYCLE_SLEEP_TIMEFRAMES.get(sleep_cycle, 'M5')
            next_candle = datetime.now(NY_TZ) + timedelta(seconds=sleep_time)
            logger.info(f"⏰ {self.pair_group}: Sleeping {sleep_time:.1f}s until next {timeframe} candle at {next_candle.strftime('%H:%M:%S')}")
            return sleep_time
        elif self.signal_builder.active_crt and self.signal_builder.crt_timeframe:
            sleep_time = self.timing_manager.get_sleep_time_for_crt(self.signal_builder.crt_timeframe)
            next_candle = datetime.now(NY_TZ) + timedelta(seconds=sleep_time)
            logger.info(f"⏰ {self.pair_group}: Sleeping {sleep_time:.1f}s until next {self.signal_builder.crt_timeframe} CRT candle at {next_candle.strftime('%H:%M:%S')}")
            return sleep_time
        else:
            sleep_time = BASE_INTERVAL
            logger.info(f"⏰ {self.pair_group}: Default sleep {sleep_time}s")
            return sleep_time

# ================================
# INTELLIGENT MAIN MANAGER
# ================================

class IntelligentTradingManager:
    def __init__(self, api_key, telegram_token, chat_id):
        self.api_key = api_key
        self.telegram_token = telegram_token
        self.chat_id = chat_id
        self.timing_manager = AdvancedTimingManager()
        self.trading_systems = {}
        
        for pair_group, pair_config in TRADING_PAIRS.items():
            self.trading_systems[pair_group] = IntelligentTradingSystem(pair_group, pair_config)
        
        logger.info(f"🎯 Initialized INTELLIGENT trading manager with {len(self.trading_systems)} pair groups")
    
    async def run_intelligent_systems(self):
        """Run all trading systems with intelligent decision making"""
        logger.info("🎯 Starting INTELLIGENT Multi-Pair Trading System...")
        
        while True:
            try:
                tasks = []
                sleep_times = []
                
                for pair_group, system in self.trading_systems.items():
                    task = asyncio.create_task(
                        system.run_intelligent_analysis(self.api_key),
                        name=f"intelligent_analysis_{pair_group}"
                    )
                    tasks.append(task)
                    sleep_times.append(system.get_sleep_time())
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                signals = []
                for i, result in enumerate(results):
                    pair_group = list(self.trading_systems.keys())[i]
                    if isinstance(result, Exception):
                        logger.error(f"❌ Intelligent analysis task failed for {pair_group}: {str(result)}")
                    elif result is not None:
                        signals.append(result)
                        logger.info(f"🎯 INTELLIGENT SIGNAL FOUND for {pair_group}")
                
                if signals:
                    await self._process_intelligent_signals(signals)
                
                sleep_time = min(sleep_times) if sleep_times else BASE_INTERVAL
                logger.info(f"⏰ Intelligent cycle complete. Sleeping for {sleep_time:.1f} seconds")
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"❌ Error in intelligent main loop: {str(e)}")
                await asyncio.sleep(60)
    
    async def _process_intelligent_signals(self, signals):
        """Process and send intelligent signals to Telegram"""
        for signal in signals:
            try:
                message = self._format_intelligent_signal_message(signal)
                success = send_telegram(message, self.telegram_token, self.chat_id)
                
                if success:
                    logger.info(f"📤 Intelligent signal sent to Telegram for {signal['pair_group']}")
                else:
                    logger.error(f"❌ Failed to send intelligent signal for {signal['pair_group']}")
                    
            except Exception as e:
                logger.error(f"❌ Error processing intelligent signal: {str(e)}")
    
    def _format_intelligent_signal_message(self, signal):
        """Format intelligent signal for Telegram"""
        pair_group = signal.get('pair_group', 'Unknown')
        direction = signal.get('direction', 'UNKNOWN').upper()
        strength = signal.get('strength', 0)
        path = signal.get('path', 'UNKNOWN')
        description = signal.get('description', '')
        bull_strength = signal.get('bullish_strength', 0)
        bear_strength = signal.get('bearish_strength', 0)
        
        message = f"🧠 *INTELLIGENT TRADING SIGNAL* 🧠\n\n"
        message += f"*Pair Group:* {pair_group.replace('_', ' ').title()}\n"
        message += f"*Direction:* {direction}\n"
        message += f"*Strength:* {strength}/9\n"
        message += f"*Path:* {path}\n"
        message += f"*Bullish SMTs:* {bull_strength}\n"
        message += f"*Bearish SMTs:* {bear_strength}\n"
        message += f"*Description:* {description}\n\n"
        
        if 'criteria' in signal:
            message += "*Signal Criteria:*\n"
            for criterion in signal['criteria']:
                message += f"• {criterion}\n"
        
        if 'all_smts' in signal and signal['all_smts']:
            message += f"\n*SMT Details:*\n"
            for cycle, smt in signal['all_smts'].items():
                psp_status = "✅ WITH PSP" if cycle in signal.get('psp_smts', {}) else "⏳ Waiting PSP"
                message += f"• {cycle}: {smt['direction']} {smt['quarters']} {psp_status}\n"
        
        message += f"\n*Detection Time:* {signal['timestamp'].strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
        message += f"\n#IntelligentSignal #{pair_group} #{path}"
        
        return message

# ================================
# MAIN EXECUTION
# ================================

async def main():
    """Main entry point"""
    logger.info("🧠 Starting INTELLIGENT Multi-Pair SMT Trading System")
    
    api_key = os.getenv('OANDA_API_KEY')
    telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
    telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if not all([api_key, telegram_token, telegram_chat_id]):
        logger.error("❌ Missing required environment variables")
        logger.info("💡 Please set OANDA_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID")
        return
    
    try:
        manager = IntelligentTradingManager(api_key, telegram_token, telegram_chat_id)
        await manager.run_intelligent_systems()
        
    except KeyboardInterrupt:
        logger.info("🛑 System stopped by user")
    except Exception as e:
        logger.error(f"💥 Fatal error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
