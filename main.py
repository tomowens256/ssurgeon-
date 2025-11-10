#!/usr/bin/env python3
"""
PRECISE MULTI-PAIR SMT TRADING SYSTEM - CORRECTED QUARTER-BASED SMT
Fixed SMT detection comparing HH/LL across consecutive quarters
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

# CRT to SMT Cycle Mapping
CRT_SMT_MAPPING = {
    'H1': ['daily', '90min'],      # 1H CRT -> daily/90min SMT confirmation
    'H2': ['daily', '90min'],      # 2H CRT -> daily/90min SMT confirmation  
    'H3': ['daily', '90min'],      # 3H CRT -> daily/90min SMT confirmation
    'H4': ['daily', '90min'],      # 4H CRT -> daily/90min SMT confirmation
    'H6': ['daily'],               # 6H CRT -> daily SMT confirmation
    'H8': ['daily'],               # 8H CRT -> daily SMT confirmation
    'H12': ['daily']               # 12H CRT -> daily SMT confirmation
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
    """Parse Oanda's timestamp with variable fractional seconds and convert to UTC-4"""
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
    """Fetch candles from OANDA API - Increased count for quarter analysis"""
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
# CANDLE TIMING MANAGER
# ================================

class CandleTimingManager:
    """Manage precise candle timing and data availability"""
    
    def __init__(self):
        self.ny_tz = pytz.timezone('America/New_York')
        
    def calculate_next_candle_time(self, timeframe, current_time=None):
        """Calculate when the next candle will open for a given timeframe"""
        if current_time is None:
            current_time = datetime.now(self.ny_tz)
        
        # Convert timeframe to minutes
        tf_minutes = self._timeframe_to_minutes(timeframe)
        if tf_minutes is None:
            return None
            
        # Calculate next candle open time
        current_timestamp = current_time.timestamp()
        next_candle_timestamp = (current_timestamp // (tf_minutes * 60) + 1) * (tf_minutes * 60)
        next_candle_time = datetime.fromtimestamp(next_candle_timestamp, self.ny_tz)
        
        return next_candle_time
    
    def _timeframe_to_minutes(self, timeframe):
        """Convert timeframe string to minutes"""
        tf_map = {
            'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30,
            'H1': 60, 'H2': 120, 'H3': 180, 'H4': 240,
            'H6': 360, 'H8': 480, 'H12': 720
        }
        return tf_map.get(timeframe)
    
    def should_wait_for_candle(self, timeframe, data_df):
        """Check if we should wait for new candle data"""
        if data_df is None or not isinstance(data_df, pd.DataFrame) or data_df.empty:
            return False
            
        # Get the most recent complete candle time
        complete_candles = data_df[data_df['complete'] == True]
        if complete_candles.empty:
            return False
            
        latest_complete_time = complete_candles['time'].max()
        
        # Calculate when next candle should be available
        next_candle_time = self.calculate_next_candle_time(timeframe, latest_complete_time)
        buffer_time = next_candle_time + timedelta(seconds=CANDLE_BUFFER_SECONDS)
        
        current_time = datetime.now(self.ny_tz)
        
        # If we're within the buffer period after next candle, wait
        if current_time < buffer_time:
            wait_seconds = (buffer_time - current_time).total_seconds()
            logger.info(f"Waiting {wait_seconds:.1f}s for {timeframe} candle data")
            return True
            
        return False
    
    def get_sleep_time_to_next_candle(self, timeframes):
        """Calculate sleep time until next important candle"""
        next_times = []
        
        for tf in timeframes:
            next_time = self.calculate_next_candle_time(tf)
            if next_time:
                next_times.append(next_time)
        
        if not next_times:
            return BASE_INTERVAL
            
        next_candle = min(next_times)
        current_time = datetime.now(self.ny_tz)
        
        sleep_seconds = (next_candle - current_time).total_seconds() + CANDLE_BUFFER_SECONDS
        
        return max(MIN_INTERVAL, sleep_seconds)

# ================================
# QUARTER MANAGER - FIXED 90MIN CYCLES
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
        
        # Get only the most recent quarters
        all_quarters = list(quarters_data.keys())
        recent_quarters = all_quarters[-num_quarters:] if len(all_quarters) >= num_quarters else all_quarters
        
        return {q: quarters_data[q] for q in recent_quarters if q in quarters_data}
    
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
# PATTERN DETECTORS - CORRECTED SMT
# ================================

class PreciseCRTDetector:
    """CRT detector for current candle only"""
    
    @staticmethod
    def calculate_crt_current_candle(df):
        """Calculate CRT only on the current (incomplete) candle"""
        if df is None or not isinstance(df, pd.DataFrame) or df.empty or len(df) < 3:
            return None
        
        # Get only the current (incomplete) candle
        current_candle = df[df['is_current'] == True]
        if current_candle.empty:
            return None
            
        current_candle = current_candle.iloc[0]
        
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

class PrecisePSPDetector:
    """PSP detector for current candle only"""
    
    @staticmethod
    def detect_psp_current_candle(asset1_data, asset2_data, timeframe):
        """Detect PSP only on the current candle"""
        if (asset1_data is None or not isinstance(asset1_data, pd.DataFrame) or asset1_data.empty or
            asset2_data is None or not isinstance(asset2_data, pd.DataFrame) or asset2_data.empty):
            return None
        
        # Get current candles for both assets
        asset1_current = asset1_data[asset1_data['is_current'] == True]
        asset2_current = asset2_data[asset2_data['is_current'] == True]
        
        if asset1_current.empty or asset2_current.empty:
            return None
            
        asset1_candle = asset1_current.iloc[0]
        asset2_candle = asset2_current.iloc[0]
        
        # Check if both current candles have the same timestamp (same candle)
        if asset1_candle['time'] != asset2_candle['time']:
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

class CorrectedSMTDetector:
    """CORRECTED SMT detector comparing HH/LL across consecutive quarters"""
    
    def __init__(self):
        self.smt_history = []
        self.quarter_manager = QuarterManager()
        self.signal_counts = {}  # Track signal counts to prevent duplicates
        
    def detect_smt_corrected(self, asset1_data, asset2_data, cycle_type):
        """Detect SMT by comparing HH/LL across consecutive quarters"""
        try:
            if (asset1_data is None or not isinstance(asset1_data, pd.DataFrame) or asset1_data.empty or
                asset2_data is None or not isinstance(asset2_data, pd.DataFrame) or asset2_data.empty):
                return None
            
            # Group candles into quarters for both assets
            asset1_quarters = self.quarter_manager.group_candles_by_quarters(asset1_data, cycle_type, num_quarters=4)
            asset2_quarters = self.quarter_manager.group_candles_by_quarters(asset2_data, cycle_type, num_quarters=4)
            
            if not asset1_quarters or not asset2_quarters:
                return None
            
            # Get quarter names in chronological order
            quarter_names = sorted(asset1_quarters.keys())
            if len(quarter_names) < 2:
                return None
            
            # Check consecutive quarter pairs (limit to 3 quarters back)
            max_pairs = min(3, len(quarter_names) - 1)
            
            for i in range(len(quarter_names) - max_pairs, len(quarter_names) - 1):
                prev_quarter = quarter_names[i]
                curr_quarter = quarter_names[i + 1]
                
                smt_result = self._compare_consecutive_quarters(
                    asset1_quarters[prev_quarter], asset1_quarters[curr_quarter],
                    asset2_quarters[prev_quarter], asset2_quarters[curr_quarter],
                    cycle_type, prev_quarter, curr_quarter
                )
                
                if smt_result and not self._is_duplicate_signal(smt_result):
                    return smt_result
            
            return None
            
        except Exception as e:
            logger.error(f"Error in corrected SMT detection for {cycle_type}: {str(e)}")
            return None
    
    def _compare_consecutive_quarters(self, asset1_prev, asset1_curr, asset2_prev, asset2_curr, cycle_type, prev_q, curr_q):
        """Compare two consecutive quarters for HH/LL mismatches"""
        try:
            if (asset1_prev.empty or asset1_curr.empty or 
                asset2_prev.empty or asset2_curr.empty):
                return None
            
            # Get extreme highs and lows for each quarter
            asset1_prev_high = float(asset1_prev['high'].max())
            asset1_curr_high = float(asset1_curr['high'].max())
            asset1_prev_low = float(asset1_prev['low'].min())
            asset1_curr_low = float(asset1_curr['low'].min())
            
            asset2_prev_high = float(asset2_prev['high'].max())
            asset2_curr_high = float(asset2_curr['high'].max())
            asset2_prev_low = float(asset2_prev['low'].min())
            asset2_curr_low = float(asset2_curr['low'].min())
            
            # Check for Bearish SMT: Asset1 makes HH, Asset2 doesn't
            asset1_hh = asset1_curr_high > asset1_prev_high  # TRUE: makes higher high
            asset2_hh = asset2_curr_high > asset2_prev_high  # FALSE: doesn't make higher high
            
            # Check for Bullish SMT: Asset1 makes LL, Asset2 doesn't  
            asset1_ll = asset1_curr_low < asset1_prev_low    # TRUE: makes lower low
            asset2_ll = asset2_curr_low < asset2_prev_low    # FALSE: doesn't make lower low
            
            if asset1_hh and not asset2_hh:
                # Bearish SMT detected
                direction = 'bearish'
                smt_type = 'Higher High'
                asset1_action = f"made HH ({asset1_prev_high:.4f} → {asset1_curr_high:.4f})"
                asset2_action = f"no HH ({asset2_prev_high:.4f} → {asset2_curr_high:.4f})"
                
            elif asset1_ll and not asset2_ll:
                # Bullish SMT detected
                direction = 'bullish' 
                smt_type = 'Lower Low'
                asset1_action = f"made LL ({asset1_prev_low:.4f} → {asset1_curr_low:.4f})"
                asset2_action = f"no LL ({asset2_prev_low:.4f} → {asset2_curr_low:.4f})"
                
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
                'signal_key': f"{cycle_type}_{prev_q}_{curr_q}_{direction}"  # Unique key for duplicate tracking
            }
            
            self.smt_history.append(smt_data)
            self._update_signal_count(smt_data['signal_key'])
            
            logger.info(f"🎯 CORRECTED SMT: {direction} {cycle_type} {prev_q}→{curr_q}")
            logger.info(f"   Asset1: {asset1_action}")
            logger.info(f"   Asset2: {asset2_action}")
            
            return smt_data
            
        except Exception as e:
            logger.error(f"Error comparing quarters {prev_q}→{curr_q}: {str(e)}")
            return None
    
    def _is_duplicate_signal(self, smt_data):
        """Check if this signal has been sent too many times"""
        signal_key = smt_data.get('signal_key')
        if not signal_key:
            return False
            
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
# PROGRESS SIGNAL BUILDER
# ================================

class ProgressSignalBuilder:
    def __init__(self, pair_group):
        self.pair_group = pair_group
        self.active_crt = None
        self.active_psp = None
        self.htf_smt = None
        self.ltf_smt = None
        self.signal_strength = 0
        self.criteria = []
        self.creation_time = datetime.now(NY_TZ)
        self.crt_timeframe = None
        self.status = "IDLE"
        
    def set_crt_signal(self, crt_data, timeframe):
        """Set CRT signal from specific timeframe"""
        if crt_data and not self.active_crt:
            self.active_crt = crt_data
            self.crt_timeframe = timeframe
            self.signal_strength += 3
            self.criteria.append(f"CRT {timeframe}: {crt_data['direction']}")
            self.status = f"CRT_{crt_data['direction'].upper()}_WAITING_LTF_SMT"
            logger.info(f"🔷 {self.pair_group}: {timeframe} {crt_data['direction']} CRT detected → Waiting for LTF SMT confirmation")
            return True
        return False
    
    def set_psp_signal(self, psp_data):
        """Set PSP signal (must be same candle as CRT)"""
        if psp_data and self.active_crt:
            # Check if PSP is on same timeframe and approximate time as CRT
            time_diff = abs((psp_data['timestamp'] - self.active_crt['timestamp']).total_seconds())
            if time_diff < 60:  # Within 1 minute (same candle)
                self.active_psp = psp_data
                self.signal_strength += 2
                self.criteria.append(f"PSP {psp_data['timeframe']}: {psp_data['asset1_color']}/{psp_data['asset2_color']}")
                self.status = f"CRT_PSP_{self.active_crt['direction'].upper()}_WAITING_LTF_SMT"
                logger.info(f"🔷 {self.pair_group}: PSP confirmed on same candle → Waiting for LTF SMT")
                return True
        return False
    
    def set_htf_smt(self, smt_data):
        """Set higher timeframe SMT"""
        if smt_data and not self.htf_smt:
            self.htf_smt = smt_data
            self.signal_strength += 2
            self.criteria.append(f"HTF SMT {smt_data['cycle']}: {smt_data['direction']} {smt_data['quarters']}")
            self.status = f"HTF_SMT_{smt_data['direction'].upper()}_WAITING_LTF_SMT"
            logger.info(f"🔷 {self.pair_group}: HTF {smt_data['cycle']} {smt_data['direction']} SMT detected → Waiting for LTF SMT")
            return True
        return False
    
    def set_ltf_smt(self, smt_data):
        """Set lower timeframe confirmation SMT"""
        if smt_data and (self.active_crt or self.htf_smt):
            # Check direction consistency
            required_direction = self.active_crt['direction'] if self.active_crt else self.htf_smt['direction']
            
            if smt_data['direction'] == required_direction:
                self.ltf_smt = smt_data
                self.signal_strength += 2
                self.criteria.append(f"LTF SMT {smt_data['cycle']}: {smt_data['direction']} {smt_data['quarters']}")
                
                if self.active_crt and self.active_psp:
                    self.status = f"CRT_PSP_LTFSMT_{smt_data['direction'].upper()}_READY"
                    logger.info(f"🎯 {self.pair_group}: LTF SMT CONFIRMED! CRT+PSP+LTF_SMT signal complete!")
                elif self.active_crt:
                    self.status = f"CRT_LTFSMT_{smt_data['direction'].upper()}_READY" 
                    logger.info(f"🎯 {self.pair_group}: LTF SMT CONFIRMED! CRT+LTF_SMT signal complete!")
                else:
                    self.status = f"HTFSMT_LTFSMT_{smt_data['direction'].upper()}_READY"
                    logger.info(f"🎯 {self.pair_group}: LTF SMT CONFIRMED! HTF_SMT+LTF_SMT signal complete!")
                return True
            else:
                logger.warning(f"⚠️ {self.pair_group}: LTF SMT direction mismatch. Expected {required_direction}, got {smt_data['direction']}")
        return False
    
    def is_signal_ready(self):
        """Check if we have complete signal"""
        # Path 1: CRT + PSP -> LTF SMT
        path1 = (self.active_crt and self.active_psp and self.ltf_smt)
        
        # Path 2: HTF SMT -> LTF SMT -> (PSP optional)
        path2 = (self.htf_smt and self.ltf_smt)
        
        # Path 3: CRT -> LTF SMT (PSP not required but adds strength)
        path3 = (self.active_crt and self.ltf_smt)
        
        return (path1 or path2 or path3) and self.signal_strength >= 5
    
    def get_required_confirmation_cycles(self):
        """Get which cycles to scan for confirmation SMT based on current signals"""
        if self.active_crt and self.crt_timeframe:
            # Use CRT to SMT mapping
            return CRT_SMT_MAPPING.get(self.crt_timeframe, ['daily', '90min'])
        elif self.htf_smt:
            # For HTF SMT, go one level lower
            cycle_hierarchy = ['monthly', 'weekly', 'daily', '90min']
            if self.htf_smt['cycle'] in cycle_hierarchy:
                idx = cycle_hierarchy.index(self.htf_smt['cycle'])
                return cycle_hierarchy[idx+1:] if idx < len(cycle_hierarchy)-1 else []
        return ['daily', '90min']  # Default
    
    def get_progress_status(self):
        """Get current progress status for logging"""
        return self.status
    
    def get_signal_details(self):
        """Get complete signal details"""
        if not self.is_signal_ready():
            return None
            
        # Determine primary direction
        if self.active_crt:
            direction = self.active_crt['direction']
        elif self.htf_smt:
            direction = self.htf_smt['direction']
        else:
            direction = self.ltf_smt['direction']
        
        # Determine signal path
        if self.active_crt and self.active_psp:
            path = "CRT_PSP_LTFSMT"
        elif self.active_crt:
            path = "CRT_LTFSMT"  
        else:
            path = "HTFSMT_LTFSMT"
        
        return {
            'pair_group': self.pair_group,
            'direction': direction,
            'strength': self.signal_strength,
            'path': path,
            'criteria': self.criteria.copy(),
            'crt': self.active_crt,
            'psp': self.active_psp,
            'htf_smt': self.htf_smt,
            'ltf_smt': self.ltf_smt,
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
        self.htf_smt = None
        self.ltf_smt = None
        self.signal_strength = 0
        self.criteria = []
        self.crt_timeframe = None
        self.creation_time = datetime.now(NY_TZ)
        self.status = "IDLE"
        logger.info(f"🔄 {self.pair_group}: Signal builder reset")

# ================================
# CORRECTED TRADING SYSTEM
# ================================

class CorrectedTradingSystem:
    def __init__(self, pair_group, pair_config):
        self.pair_group = pair_group
        self.pair_config = pair_config
        self.pair1 = pair_config['pair1']
        self.pair2 = pair_config['pair2']
        
        # Initialize components
        self.timing_manager = CandleTimingManager()
        self.quarter_manager = QuarterManager()
        self.crt_detector = PreciseCRTDetector()
        self.psp_detector = PrecisePSPDetector()
        self.smt_detector = CorrectedSMTDetector()  # Use corrected SMT detector
        self.signal_builder = ProgressSignalBuilder(pair_group)
        
        # Data storage
        self.market_data = {self.pair1: {}, self.pair2: {}}
        
        logger.info(f"🎯 Initialized CORRECTED trading system for {self.pair1}/{self.pair2}")
    
    async def run_corrected_analysis(self, api_key):
        """Run corrected analysis with proper quarter-based SMT"""
        try:
            # Log current status
            current_status = self.signal_builder.get_progress_status()
            if current_status != "IDLE":
                logger.info(f"📊 {self.pair_group}: Current status - {current_status}")
            
            # Fetch market data for required timeframes
            await self._fetch_required_data(api_key)
            
            # Check if we should wait for candle data
            if await self._should_wait_for_data():
                return None
            
            # Step 1: Scan for CRT signals (1H and above only)
            if not self.signal_builder.active_crt and not self.signal_builder.htf_smt:
                await self._scan_crt_signals()
            
            # Step 2: If we have CRT, scan for PSP on same candle
            if self.signal_builder.active_crt and not self.signal_builder.active_psp:
                await self._scan_psp_same_candle()
            
            # Step 3: Scan for HTF SMT (if no CRT)
            if not self.signal_builder.active_crt and not self.signal_builder.htf_smt:
                await self._scan_htf_smt()
            
            # Step 4: Scan for LTF SMT confirmation
            if self.signal_builder.active_crt or self.signal_builder.htf_smt:
                await self._scan_ltf_confirmation()
            
            # Check if signal is complete
            if self.signal_builder.is_signal_ready():
                signal = self.signal_builder.get_signal_details()
                if signal:
                    logger.info(f"🎯 {self.pair_group}: CORRECTED SIGNAL COMPLETE via {signal['path']}")
                    self.signal_builder.reset()
                    return signal
            
            # Check if expired
            if self.signal_builder.is_expired():
                self.signal_builder.reset()
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error in corrected analysis for {self.pair_group}: {str(e)}")
            return None
    
    async def _fetch_required_data(self, api_key):
        """Fetch only required market data based on current signal state"""
        required_timeframes = set()
        
        # Always fetch cycle timeframes
        for tf in self.pair_config['timeframe_mapping'].values():
            required_timeframes.add(tf)
        
        # Add CRT timeframes if we're scanning for CRT
        if not self.signal_builder.active_crt:
            for tf in CRT_TIMEFRAMES:
                required_timeframes.add(tf)
        
        # Fetch data for required timeframes
        for pair in [self.pair1, self.pair2]:
            for tf in required_timeframes:
                if tf not in self.market_data[pair] or self._is_data_stale(self.market_data[pair].get(tf), tf):
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
    
    def _is_data_stale(self, df, timeframe):
        """Check if data is stale and needs refresh"""
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return True
        
        # Check if we should refresh based on timeframe
        tf_minutes = self.timing_manager._timeframe_to_minutes(timeframe)
        if tf_minutes and tf_minutes <= 60:  # Refresh hourly or more frequently
            return True
            
        return False
    
    async def _should_wait_for_data(self):
        """Check if we should wait for new candle data"""
        important_timeframes = list(self.pair_config['timeframe_mapping'].values()) + CRT_TIMEFRAMES
        
        for tf in important_timeframes:
            pair1_data = self.market_data[self.pair1].get(tf)
            if pair1_data is not None and isinstance(pair1_data, pd.DataFrame) and not pair1_data.empty:
                if self.timing_manager.should_wait_for_candle(tf, pair1_data):
                    return True
                
        return False
    
    async def _scan_crt_signals(self):
        """Scan for CRT signals on 1H and above timeframes"""
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
                break
    
    async def _scan_psp_same_candle(self):
        """Scan for PSP on the same candle as CRT"""
        if not self.signal_builder.active_crt or not self.signal_builder.crt_timeframe:
            return
        
        timeframe = self.signal_builder.crt_timeframe
        pair1_data = self.market_data[self.pair1].get(timeframe)
        pair2_data = self.market_data[self.pair2].get(timeframe)
        
        if (pair1_data is None or not isinstance(pair1_data, pd.DataFrame) or pair1_data.empty or
            pair2_data is None or not isinstance(pair2_data, pd.DataFrame) or pair2_data.empty):
            return
        
        psp_signal = self.psp_detector.detect_psp_current_candle(pair1_data, pair2_data, timeframe)
        if psp_signal:
            self.signal_builder.set_psp_signal(psp_signal)
    
    async def _scan_htf_smt(self):
        """Scan for higher timeframe SMT with corrected quarter comparison"""
        htf_cycles = ['monthly', 'weekly']  # Higher timeframes
        
        for cycle in htf_cycles:
            timeframe = self.pair_config['timeframe_mapping'][cycle]
            pair1_data = self.market_data[self.pair1].get(timeframe)
            pair2_data = self.market_data[self.pair2].get(timeframe)
            
            if (pair1_data is None or not isinstance(pair1_data, pd.DataFrame) or pair1_data.empty or
                pair2_data is None or not isinstance(pair2_data, pd.DataFrame) or pair2_data.empty):
                continue
            
            smt_signal = self.smt_detector.detect_smt_corrected(pair1_data, pair2_data, cycle)
            
            if smt_signal and self.signal_builder.set_htf_smt(smt_signal):
                break
    
    async def _scan_ltf_confirmation(self):
        """Scan for lower timeframe SMT confirmation with corrected quarter comparison"""
        confirmation_cycles = self.signal_builder.get_required_confirmation_cycles()
        
        for cycle in confirmation_cycles:
            timeframe = self.pair_config['timeframe_mapping'][cycle]
            pair1_data = self.market_data[self.pair1].get(timeframe)
            pair2_data = self.market_data[self.pair2].get(timeframe)
            
            if (pair1_data is None or not isinstance(pair1_data, pd.DataFrame) or pair1_data.empty or
                pair2_data is None or not isinstance(pair2_data, pd.DataFrame) or pair2_data.empty):
                continue
            
            smt_signal = self.smt_detector.detect_smt_corrected(pair1_data, pair2_data, cycle)
            
            if smt_signal:
                if self.signal_builder.set_ltf_smt(smt_signal):
                    logger.info(f"🎯 {self.pair_group}: LTF SMT confirmed with corrected quarter comparison")
                    break
            else:
                logger.info(f"🔍 {self.pair_group}: No LTF SMT found in {cycle} with corrected quarter comparison")

# ================================
# CORRECTED MAIN MANAGER
# ================================

class CorrectedTradingManager:
    def __init__(self, api_key, telegram_token, chat_id):
        self.api_key = api_key
        self.telegram_token = telegram_token
        self.chat_id = chat_id
        self.timing_manager = CandleTimingManager()
        self.trading_systems = {}
        
        # Initialize trading systems for all pairs
        for pair_group, pair_config in TRADING_PAIRS.items():
            self.trading_systems[pair_group] = CorrectedTradingSystem(pair_group, pair_config)
        
        logger.info(f"🎯 Initialized CORRECTED trading manager with {len(self.trading_systems)} pair groups")
    
    async def run_corrected_systems(self):
        """Run all trading systems with corrected SMT detection"""
        logger.info("🎯 Starting CORRECTED Multi-Pair Trading System...")
        
        while True:
            try:
                # Calculate optimal sleep time based on next important candles
                important_timeframes = []
                for pair_config in TRADING_PAIRS.values():
                    important_timeframes.extend(pair_config['timeframe_mapping'].values())
                important_timeframes.extend(CRT_TIMEFRAMES)
                
                sleep_time = self.timing_manager.get_sleep_time_to_next_candle(important_timeframes)
                
                # Run analysis for all pairs
                tasks = []
                for pair_group, system in self.trading_systems.items():
                    task = asyncio.create_task(
                        system.run_corrected_analysis(self.api_key),
                        name=f"analysis_{pair_group}"
                    )
                    tasks.append(task)
                
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
                        logger.info(f"🎯 CORRECTED SIGNAL FOUND for {pair_group}")
                
                # Send signals to Telegram
                if signals:
                    await self._process_signals(signals)
                
                logger.info(f"⏰ Corrected cycle complete. Sleeping for {sleep_time:.1f} seconds")
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"❌ Error in corrected main loop: {str(e)}")
                await asyncio.sleep(60)
    
    async def _process_signals(self, signals):
        """Process and send signals to Telegram"""
        for signal in signals:
            try:
                message = self._format_corrected_signal_message(signal)
                success = send_telegram(message, self.telegram_token, self.chat_id)
                
                if success:
                    logger.info(f"📤 Corrected signal sent to Telegram for {signal['pair_group']}")
                else:
                    logger.error(f"❌ Failed to send corrected signal for {signal['pair_group']}")
                    
            except Exception as e:
                logger.error(f"❌ Error processing corrected signal: {str(e)}")
    
    def _format_corrected_signal_message(self, signal):
        """Format corrected signal for Telegram"""
        pair_group = signal.get('pair_group', 'Unknown')
        direction = signal.get('direction', 'UNKNOWN').upper()
        strength = signal.get('strength', 0)
        path = signal.get('path', 'UNKNOWN')
        
        message = f"🎯 *CORRECTED TRADING SIGNAL* 🎯\n\n"
        message += f"*Pair Group:* {pair_group.replace('_', ' ').title()}\n"
        message += f"*Direction:* {direction}\n"
        message += f"*Strength:* {strength}/9\n"
        message += f"*Path:* {path}\n\n"
        
        # Add criteria with quarter comparison details
        if 'criteria' in signal:
            message += "*Signal Criteria:*\n"
            for criterion in signal['criteria']:
                message += f"• {criterion}\n"
        
        # Add quarter comparison details from SMT
        if signal.get('htf_smt') and 'details' in signal['htf_smt']:
            message += f"\n*HTF SMT Details:* {signal['htf_smt']['details']}\n"
        if signal.get('ltf_smt') and 'details' in signal['ltf_smt']:
            message += f"*LTF SMT Details:* {signal['ltf_smt']['details']}\n"
        
        message += f"\n*Detection Time:* {signal['timestamp'].strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
        message += f"\n#CorrectedSignal #{pair_group} #{path}"
        
        return message

# ================================
# MAIN EXECUTION
# ================================

async def main():
    """Main entry point"""
    logger.info("🎯 Starting CORRECTED Multi-Pair SMT Trading System")
    
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
        manager = CorrectedTradingManager(api_key, telegram_token, telegram_chat_id)
        
        # Run all systems with corrected SMT detection
        await manager.run_corrected_systems()
        
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
