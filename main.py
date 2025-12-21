#!/usr/bin/env python3


import asyncio
import traceback
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
from pytz import timezone
NY_TZ = timezone('America/New_York')  # automatically handles EST/EDT



# ================================
# CONFIGURATION - BACKWARD COMPATIBLE
# ================================

TRADING_PAIRS = {
    'precious_metals': {
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
    'us_indices_triad': {
        'instruments': ['NAS100_USD', 'SPX500_USD'],  
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
    },
    'jpy_triad': {
        'pair1': 'EUR_JPY',  # OLD structure
        'pair2': 'GBP_JPY',  # OLD structure
        'instruments': ['EUR_JPY', 'GBP_JPY'],  # NEW structure
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
        'instruments': ['DE30_EUR', 'EU50_EUR'],
        'timeframe_mapping': {
            'monthly': 'H4',
            'weekly': 'H1',
            'daily': 'M15',
            '90min': 'M5'
        }
    }
}

CRT_TIMEFRAMES = [ 'H1','H2','H4']
CRT_SMT_MAPPING = {
    'H4': ['weekly', 'daily'],    # 4hr CRT can use weekly OR daily SMT
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

def fetch_candles(instrument, timeframe, count=100, api_key=None, since=None):
    """Fetch candles from OANDA API - ENFORCE UTC-4, incremental since."""
   
    if not api_key:
        logger.error("Oanda API key missing")
        return pd.DataFrame()
       
    try:
        from oandapyV20 import API
        from oandapyV20.endpoints import instruments as instruments
        api = API(access_token=api_key, environment="practice")
    except Exception as e:
        logger.error(f"Oanda API initialization failed: {str(e)}")
        return pd.DataFrame()
       
    params = {
        "granularity": timeframe,
        "count": count,
        "price": "M",
        "alignmentTimezone": "America/New_York",  # UTC-4
        "includeCurrent": True
    }
    if since:
        params["from"] = since.strftime('%Y-%m-%dT%H:%M:%S')  # Delta

    for attempt in range(MAX_RETRIES):
        try:
            request = instruments.InstrumentsCandles(instrument=instrument, params=params)
            response = api.request(request)
            candles = response.get('candles', [])
           
            if not candles:
                logger.warning(f"No candles received for {instrument} on attempt {attempt+1}")
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
                except Exception as e:
                    logger.error(f"Error parsing candle for {instrument}: {str(e)} (raw time: {candle.get('time', 'N/A')})")
                    continue
           
            if not data:
                logger.warning(f"Empty data after parsing for {instrument} on attempt {attempt+1}")
                continue
               
            df = pd.DataFrame(data).drop_duplicates(subset=['time'], keep='last')
            df = df.sort_values('time').reset_index(drop=True)
           
            if since:
                df = df[df['time'] > since]
           
            logger.info(f"Successfully fetched {len(df)} candles for {instrument} {timeframe}")
            return df
           
        except Exception as e:
            import traceback
            if "rate" in str(e).lower() or (hasattr(e, 'code') and e.code in [429, 502]):
                wait_time = 10 * (2 ** attempt)
                logger.warning(f"Rate limit hit for {instrument}, waiting {wait_time}s: {str(e)}")
                import time
                time.sleep(wait_time)
            else:
                logger.error(f"Oanda API error for {instrument}: Status: {getattr(e, 'code', 'N/A')} | Message: {str(e)}")
                break
   
    logger.error(f"Failed to fetch candles for {instrument} after {MAX_RETRIES} attempts")
    return pd.DataFrame()

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
        """Calculate CRT on current candle AND check for PSP on same timeframe"""
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
            
            if buy_crt or sell_crt:
                direction = 'bullish' if buy_crt else 'bearish'
                
                # CHECK FOR PSP ON SAME TIMEFRAME
                psp_signal = self._detect_psp_for_crt(asset1_data, asset2_data, timeframe, current_candle['time'])
                
                logger.info(f"🔷 {direction.upper()} CRT DETECTED: {timeframe} candle at {c3['time'].strftime('%H:%M')}")
                if psp_signal:
                    logger.info(f"🎯 PSP FOUND for CRT: {psp_signal['asset1_color']}/{psp_signal['asset2_color']} at {psp_signal['formation_time'].strftime('%H:%M')}")
                
                # Create and return CRT signal
                return {
                    'direction': direction, 
                    'timestamp': c3['time'],
                    'timeframe': timeframe,
                    'signal_key': f"CRT_{timeframe}_{c3['time'].strftime('%m%d_%H%M')}_{direction}",
                    'psp_signal': psp_signal  # Include PSP if found
                }
                
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
            # Strict chronology check
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
    
            if direction == 'bearish':
                swing_time_key = f"{asset1_prev_high['time'].strftime('%H%M')}_{asset1_curr_high['time'].strftime('%H%M')}"
                swing_times = {
                    'asset1_prev': asset1_prev_high['time'],
                    'asset1_curr': asset1_curr_high['time'],
                    'asset2_prev': asset2_prev_high['time'],
                    'asset2_curr': asset2_curr_high['time']
                }
            else:
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
                'quarters': f"{prev_q}→{curr_q}",
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
                'candle_time': formation_time
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
    def __init__(self, pair_group, timing_manager, telegram_token=None, telegram_chat_id=None):
        # Store all parameters as instance variables
        self.pair_group = pair_group
        self.timing_manager = timing_manager
        self.telegram_token = telegram_token
        self.telegram_chat_id = telegram_chat_id
        self.instruments = instruments or []
        
        # Active features storage with expiration tracking
        self.active_features = {
            'smt': {},      # key: signal_key, value: {smt_data, psp_data, timestamp, expiration}
            'crt': {},      # key: signal_key, value: {crt_data, psp_data, timestamp, expiration}  
            'psp': {},       # key: signal_key, value: {psp_data, timestamp, expiration}
            'sd_zone': {}
        }
        
        # Signal cooldown to prevent duplicates
        self.sent_signals = {}
        self.signals_sent_count = 0
        self.sent_signal_signatures = {}  # key: signature_hash, value: timestamp
        self.signature_expiry_hours = 24  # Keep signatures for 24 hours
        
        # Feature expiration times (minutes)
        self.expiration_times = {
            'smt': 1200,    # 4 hours for SMT
            'crt': 120,    # 2 hours for CRT  
            'psp': 60 ,     # 1 hour for PSP
            'sd_zone': 288000000
        }
        
        logger.info(f"🎯 RealTimeFeatureBox initialized for {pair_group}")
    
        

    def add_sd_zone(self, zone_data):
        """Add Supply/Demand zone to active features"""
        zone_name = zone_data['zone_name']
        
        # Check if already exists
        if zone_name in self.active_features['sd_zone']:
            logger.info(f"📦 SD Zone already exists: {zone_name}")
            return False
        
        # Create feature with expiration
        feature = {
            'type': 'sd_zone',
            'zone_data': zone_data,
            'timestamp': datetime.now(NY_TZ),
            'expiration': datetime.now(NY_TZ) + timedelta(minutes=self.expiration_times['sd_zone'])
        }
        
        self.active_features['sd_zone'][zone_name] = feature
        logger.info(f"📦 SD Zone ADDED: {zone_name}")
        return True
    
    def get_active_sd_zones(self, direction=None, timeframe=None, asset=None):
        """Get active SD zones with optional filters"""
        active_zones = []
        
        for zone_name, zone_feature in self.active_features['sd_zone'].items():
            if self._is_feature_expired(zone_feature):
                continue
                
            zone_data = zone_feature['zone_data']
            
            # Apply filters
            if direction and zone_data.get('direction') != direction:
                continue
            if timeframe and zone_data.get('timeframe') != timeframe:
                continue
            if asset and zone_data.get('asset') != asset:
                continue
            
            active_zones.append(zone_data)
        
        return active_zones
    
    def cleanup_sd_zones(self, market_data):
        """Clean up invalidated SD zones"""
        zones_to_remove = []
        
        for zone_name, zone_feature in self.active_features['sd_zone'].items():
            if self._is_feature_expired(zone_feature):
                zones_to_remove.append(zone_name)
                continue
                
            zone_data = zone_feature['zone_data']
            asset = zone_data['asset']
            timeframe = zone_data['timeframe']
            
            # Get current market data for this asset/timeframe
            if asset in market_data and timeframe in market_data[asset]:
                current_data = market_data[asset][timeframe]
                
                # Check if zone is still valid
                # We need the SupplyDemandDetector for this - we'll pass it as parameter
                # This should be called from UltimateTradingSystem which has the detector
                pass
        
        for zone_name in zones_to_remove:
            del self.active_features['sd_zone'][zone_name]
            logger.info(f"🧹 Removed expired SD zone: {zone_name}")
    # In RealTimeFeatureBox.add_smt():
    def add_smt(self, smt_data, psp_data):
        """Add SMT to active features"""
        logger.info(f"📦 ADD_SMT CALLED: {smt_data['cycle']} {smt_data['direction']}")
        
        # Check if already exists
        signal_key = smt_data.get('signal_key')
        if signal_key in self.active_features['smt']:
            logger.info(f"📦 SMT already exists: {signal_key}")
            return False
        
        # Create feature WITH EXPIRATION
        feature = {
            'type': 'smt',
            'smt_data': smt_data,
            'psp_data': psp_data,
            'timestamp': datetime.now(NY_TZ),
            'expiration': datetime.now(NY_TZ) + timedelta(minutes=self.expiration_times['smt'])
        }
        
        self.active_features['smt'][signal_key] = feature
        logger.info(f"📦 SMT ADDED SUCCESS: {signal_key}")
        logger.info(f"📦 Total active SMTs: {len(self.active_features['smt'])}")
        
        return True

    def _is_feature_expired(self, feature):
        """Check if feature is expired - EXTENDED FOR TESTING"""
        feature_type = feature.get('type')
        creation_time = feature.get('timestamp')
        
        if not creation_time:
            return True
        
        # EXTENDED EXPIRY FOR TESTING
        expiry_hours = {
            'smt': 10,    # 24 hours instead of 1-2
            'crt': 2,
            'psp': 1
        }
        
        current_time = datetime.now(NY_TZ)
        hours_passed = (current_time - creation_time).total_seconds() / 3600
        
        expired = hours_passed > expiry_hours.get(feature_type, 1)
        
        if expired:
            logger.debug(f"🕒 Feature expired: {feature.get('signal_key', 'Unknown')} - {hours_passed:.1f}h old")
        
        return expired
    
    def add_crt(self, crt_data, psp_data=None):
        """Add CRT feature to tracking - triggers immediate confluence check"""
        if not crt_data:
            return False
            
        signal_key = crt_data['signal_key']
        
        # Check if already exists and is still valid
        if signal_key in self.active_features['crt']:
            if not self._is_feature_expired(self.active_features['crt'][signal_key]):
                logger.debug(f"🔄 CRT already active: {signal_key}")
                return False
        
        # Store CRT feature
        self.active_features['crt'][signal_key] = {
            'crt_data': crt_data,
            'psp_data': psp_data, 
            'timestamp': datetime.now(NY_TZ),
            'expiration': datetime.now(NY_TZ) + timedelta(minutes=self.expiration_times['crt'])
        }
        
        logger.info(f"📥 CRT ADDED to FeatureBox: {crt_data['timeframe']} {crt_data['direction']}")
        
        # Immediate confluence check
        return self._check_immediate_confluence(signal_key, 'crt')
    
    def add_psp(self, psp_data, associated_smt_key=None):
        """Add PSP feature to tracking - triggers immediate confluence check"""
        if not psp_data:
            return False
            
        signal_key = psp_data['signal_key']
        
        # Store PSP feature
        self.active_features['psp'][signal_key] = {
            'psp_data': psp_data,
            'associated_smt': associated_smt_key,
            'timestamp': datetime.now(NY_TZ),
            'expiration': datetime.now(NY_TZ) + timedelta(minutes=self.expiration_times['psp'])
        }
        
        logger.info(f"📥 PSP ADDED to FeatureBox: {psp_data['timeframe']} {psp_data['asset1_color']}/{psp_data['asset2_color']}")
        
        # If PSP is associated with specific SMT, check that SMT immediately
        if associated_smt_key:
            return self._check_smt_psp_confluence(associated_smt_key, signal_key)
        else:
            # Check all possible confluences
            return self._check_immediate_confluence(signal_key, 'psp')
    
    def _check_immediate_confluence(self, new_feature_key, feature_type):
        """
        IMMEDIATE confluence check when new feature is added
        Returns True if signal was sent
        """
        logger.info(f"🔍 IMMEDIATE CONFLUENCE CHECK for {feature_type}: {new_feature_key}")
        
        # Check all possible confluence combinations
        signals_sent = []
        
        # 1. Check SMT + PSP confluence
        signals_sent.append(self._check_smt_psp_confluence_global())
        
        # 2. Check CRT + SMT confluence  
        signals_sent.append(self._check_crt_smt_confluence())
        
        # 3. Check CRT + PSP confluence
        signals_sent.append(self._check_crt_psp_confluence())
        
        # 4. Check Multiple SMTs confluence
        signals_sent.append(self._check_multiple_smts_confluence())
        
        # 5. Check Triple Confluence (CRT + PSP + SMT)
        signals_sent.append(self._check_triple_confluence())
        
        return any(signals_sent)
    
    def _check_smt_psp_confluence(self, smt_key, psp_key):
        """Check specific SMT + PSP confluence"""
        if smt_key not in self.active_features['smt']:
            return False
        if psp_key not in self.active_features['psp']:
            return False
            
        smt_feature = self.active_features['smt'][smt_key]
        psp_feature = self.active_features['psp'][psp_key]
        
        # Validate they're still active
        if self._is_feature_expired(smt_feature) or self._is_feature_expired(psp_feature):
            return False
        
        smt_data = smt_feature['smt_data']
        psp_data = psp_feature['psp_data']
        
        # Check if PSP timeframe matches SMT timeframe
        if smt_data.get('timeframe') != psp_data.get('timeframe'):
            logger.debug(f"⚠️ SMT/PSP timeframe mismatch: {smt_data.get('timeframe')} vs {psp_data.get('timeframe')}")
            return False
        
        # Create signal
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
            # Remove used features
            self._remove_feature('smt', smt_key)
            self._remove_feature('psp', psp_key)
            return True
            
        return False
    
    def _check_smt_psp_confluence_global(self):
        """SMT+PSP w/FVG tap priority (abort old if tap, send enhanced)."""
        signals_sent = 0
        fvg_detector = FVGDetector()  # Quick scan
        if not hasattr(self, 'instruments') or not isinstance(self.instruments, (list, tuple)):
            logger.error(f"❌ Invalid instruments: {self.instruments} (expected list, got {type(self.instruments)})")
            return None
    
        fvgs_per_asset = {inst: [] for inst in self.instruments}
        for tf in ['M15', 'H1', 'H4', 'D']:
            for inst in self.instruments:
                data = self.market_data[inst].get(tf)  # Assume access (or pass from system)
                if data is not None and not data.empty:
                    new_fvgs = fvg_detector.scan_tf(data, tf, inst)
                    fvgs_per_asset[inst].extend(new_fvgs)
        
        for smt_key, smt_feature in list(self.active_features['smt'].items()):
            if self._is_feature_expired(smt_feature):
                continue
            smt_data = smt_feature['smt_data']
            if smt_feature['psp_data']:
                # FVG tap check first
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
                    # Enhanced FVG + SMT + PSP ping
                    logger.info(f"🚀 PRIORITY HIT: SMT+PSP tapped FVG {smt_data['cycle']} {smt_data['direction']}")
                    # Send enhanced (your _send_fvg_smt_tap_signal with fvg)
                    return True  # Or call it
                
                # Fall to old SMT+PSP
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
        """Check if we have a daily SMT in the same direction"""
        for smt_key, smt_feature in self.active_features['smt'].items():
            if self._is_feature_expired(smt_feature):
                continue
                
            smt_data = smt_feature['smt_data']
            if smt_data['cycle'] == 'daily' and smt_data['direction'] == direction:
                return True
        
        return False
    
    def _check_crt_smt_confluence(self):
        """Check CRT + SMT confluence"""
        signals_sent = 0
        
        for crt_key, crt_feature in list(self.active_features['crt'].items()):
            if self._is_feature_expired(crt_feature):
                continue
                
            crt_data = crt_feature['crt_data']
            
            for smt_key, smt_feature in list(self.active_features['smt'].items()):
                if self._is_feature_expired(smt_feature):
                    continue
                    
                smt_data = smt_feature['smt_data']
                
                # Check direction match
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
                        break  # One signal per CRT
        
        return signals_sent > 0

    def _check_multiple_smts_confluence(self):
        """Check for multiple SMTs confluence - ONLY ONE SMT PER CYCLE"""
        signals_sent = 0
        current_time = datetime.now(NY_TZ)
        
        # Group SMTs by direction
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
        
        logger.info(f"🔍 Multiple SMTs check: {len(bullish_smts)} bullish, {len(bearish_smts)} bearish")
        
        def _get_unique_cycle_smts(smt_list):
            """Get only one SMT per cycle, preferring the strongest or first found"""
            cycle_smts = {}
            
            for smt_key, smt_data in smt_list:
                cycle = smt_data.get('cycle', 'Unknown')
                
                # If we haven't seen this cycle yet, or if we want to apply some selection logic
                if cycle not in cycle_smts:
                    cycle_smts[cycle] = (smt_key, smt_data)
                # Optional: Add logic here to select "better" SMT if multiple from same cycle
                # For now, we just take the first one we encounter
            
            return list(cycle_smts.values())
        
        # Process bullish SMTs - get only one per cycle
        unique_bullish = _get_unique_cycle_smts(bullish_smts)
        
        # Check for multiple bullish SMTs from DIFFERENT cycles
        if len(unique_bullish) >= 2:
            logger.info(f"🎯 MULTIPLE BULLISH SMTs from different cycles: {len(unique_bullish)} unique cycles")
            
            # Get ALL SMT details - only one per cycle
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
                'cycle_count': len(unique_bullish),  # Same as smt_count since one per cycle
                'timestamp': current_time,
                'signal_key': f"MULTI_SMT_BULLISH_DIFF_CYCLES_{current_time.strftime('%H%M%S')}",
                'description': f"MULTIPLE BULLISH SMTs: {len(unique_bullish)} SMTs from {len(unique_bullish)} different cycles"
            }
            
            logger.info(f"🔍 Unique Bullish SMTs by cycle: {[s['cycle'] for s in smt_details]}")
            
            if self._send_immediate_signal(signal_data):
                signals_sent += 1
        
        # Process bearish SMTs - get only one per cycle  
        unique_bearish = _get_unique_cycle_smts(bearish_smts)
        
        # Check for multiple bearish SMTs from DIFFERENT cycles
        if len(unique_bearish) >= 2:
            logger.info(f"🎯 MULTIPLE BEARISH SMTs from different cycles: {len(unique_bearish)} unique cycles")
            
            # Get ALL SMT details - only one per cycle
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
                'cycle_count': len(unique_bearish),  # Same as smt_count since one per cycle
                'timestamp': current_time,
                'signal_key': f"MULTI_SMT_BEARISH_DIFF_CYCLES_{current_time.strftime('%H%M%S')}",
                'description': f"MULTIPLE BEARISH SMTs: {len(unique_bearish)} SMTs from {len(unique_bearish)} different cycles"
            }
            
            logger.info(f"🔍 Unique Bearish SMTs by cycle: {[s['cycle'] for s in smt_details]}")
            
            if self._send_immediate_signal(signal_data):
                signals_sent += 1
        
        return signals_sent > 0
    
    def _check_triple_confluence(self):
        """Check CRT + PSP + SMT triple confluence"""
        for crt_key, crt_feature in list(self.active_features['crt'].items()):
            if self._is_feature_expired(crt_feature):
                continue
                
            crt_data = crt_feature['crt_data']
            
            # Check if CRT has PSP
            if not crt_feature['psp_data']:
                continue
                
            for smt_key, smt_feature in list(self.active_features['smt'].items()):
                if self._is_feature_expired(smt_feature):
                    continue
                    
                smt_data = smt_feature['smt_data']
                
                # Check direction match
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
    
    def _send_immediate_signal(self, signal_data):
        """Send signal with content validation"""
        signal_key = signal_data['signal_key']
        
        # 1. Validate signal content
        if not self._validate_signal_before_sending(signal_data):
            logger.warning(f"⏳ SIGNAL BLOCKED (invalid content): {signal_key}")
            return False
        
        # 2. Check duplicate prevention
        if self.timing_manager.is_duplicate_signal(signal_key, self.pair_group, cooldown_minutes=30):
            logger.info(f"⏳ SIGNAL BLOCKED (duplicate): {signal_key}")
            return False
        
        # Format and send message
        message = self._format_immediate_signal_message(signal_data)
        success = send_telegram(message, self.telegram_token, self.telegram_chat_id)
        
        if success:
            logger.info(f"🚀 SIGNAL SENT: {signal_data['description']}")
            self.sent_signals[signal_key] = datetime.now(NY_TZ)
            return True
        else:
            logger.error(f"❌ FAILED to send signal: {signal_key}")
            return False

    def _validate_signal_before_sending(self, signal_data):
        """Validate signal has meaningful content before sending"""
        confluence_type = signal_data.get('confluence_type', '')
        
        if confluence_type.startswith('MULTIPLE_SMTS'):
            # For multiple SMTs, check we have valid SMT details
            multiple_smts = signal_data.get('multiple_smts', [])
            if not multiple_smts:
                logger.warning(f"⚠️ BLOCKED EMPTY MULTIPLE SMTs signal")
                return False
            
            valid_smts = [smt for smt in multiple_smts 
                         if smt.get('cycle') and smt.get('quarters')]
            
            if len(valid_smts) < 2:
                logger.warning(f"⚠️ BLOCKED: Only {len(valid_smts)} valid SMTs in multiple SMTs signal")
                return False
                
            return True
        
        elif confluence_type == 'SMT_PSP_PRE_CONFIRMED':
            # For SMT+PSP, check both exist
            if not signal_data.get('smt') or not signal_data.get('psp'):
                logger.warning(f"⚠️ BLOCKED INCOMPLETE SMT+PSP signal")
                return False
            return True
        
        elif confluence_type == 'CRT_SMT_IMMEDIATE':
            # For CRT+SMT, check both exist
            if not signal_data.get('crt') or not signal_data.get('smt'):
                logger.warning(f"⚠️ BLOCKED INCOMPLETE CRT+SMT signal")
                return False
            return True
        
        # Default: allow other signal types
        return True
    
    def _validate_signal_content(self, signal_data):
        """Validate that signal has meaningful content"""
        if 'multiple_smts' in signal_data:
            # For multiple SMTs, check we have at least 2 valid SMTs
            smts = signal_data['multiple_smts']
            if len(smts) < 2:
                return False
            
            # Check each SMT has basic data
            for smt in smts:
                if not smt.get('cycle') or not smt.get('quarters'):
                    return False
            return True
        
        elif 'smt' in signal_data:
            # For single SMT, check basic data
            smt = signal_data['smt']
            return bool(smt.get('cycle') and smt.get('quarters'))
        
        elif 'crt' in signal_data:
            # For CRT, check basic data
            crt = signal_data['crt']
            return bool(crt.get('timeframe') and crt.get('direction'))
        
        return False
    def _check_crt_psp_confluence(self):
        """Check CRT + PSP confluence"""
        signals_sent = 0
        
        for crt_key, crt_feature in list(self.active_features['crt'].items()):
            if self._is_feature_expired(crt_feature):
                continue
                
            crt_data = crt_feature['crt_data']
            
            # Check if CRT has PSP
            if not crt_feature['psp_data']:
                continue
                
            # We have a CRT with PSP, send signal
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
        """Format immediate signal for Telegram - FIXED EMPTY MULTIPLE SMTs"""
        direction = signal_data['direction'].upper()
        confluence_type = signal_data['confluence_type']
        description = signal_data['description']
        
        message = f"⚡ *IMMEDIATE SIGNAL* ⚡\n\n"
        message += f"*Pair Group:* {self.pair_group.replace('_', ' ').title()}\n"
        message += f"*Direction:* {direction}\n"
        message += f"*Confluence:* {confluence_type}\n"
        message += f"*Description:* {description}\n\n"
        
        # Single SMT details
        if 'smt' in signal_data and signal_data['smt']:
            smt = signal_data['smt']
            message += f"*SMT Details:*\n"
            message += f"• Cycle: {smt.get('cycle', 'Unknown')} {smt.get('quarters', '')}\n"
            if smt.get('asset1_action'):
                message += f"• {smt['asset1_action']}\n"
            if smt.get('asset2_action'):
                message += f"• {smt['asset2_action']}\n"
            message += f"\n"
        
        # Multiple SMTs details - FIXED: Check if data exists and is valid
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
            else:
                logger.warning(f"⚠️ Multiple SMTs signal has no valid SMT details")
        
        # CRT details
        if 'crt' in signal_data and signal_data['crt']:
            crt = signal_data['crt']
            message += f"*CRT Details:*\n"
            message += f"• Timeframe: {crt.get('timeframe', 'Unknown')}\n"
            if crt.get('timestamp'):
                message += f"• Time: {crt['timestamp'].strftime('%H:%M')}\n"
            message += f"\n"
        
        # PSP details
        if 'psp' in signal_data and signal_data['psp']:
            psp = signal_data['psp']
            message += f"*PSP Details:*\n"
            message += f"• Timeframe: {psp.get('timeframe', 'Unknown')}\n"
            message += f"• Colors: {psp.get('asset1_color', 'Unknown')}/{psp.get('asset2_color', 'Unknown')}\n"
            if psp.get('formation_time'):
                message += f"• Time: {psp['formation_time'].strftime('%H:%M')}\n"
            message += f"\n"
        
        # CRT+PSP details
        if 'crt_psp' in signal_data and signal_data['crt_psp']:
            crt_psp = signal_data['crt_psp']
            message += f"*CRT PSP Details:*\n"
            message += f"• Timeframe: {crt_psp.get('timeframe', 'Unknown')}\n"
            message += f"• Colors: {crt_psp.get('asset1_color', 'Unknown')}/{crt_psp.get('asset2_color', 'Unknown')}\n"
            if crt_psp.get('formation_time'):
                message += f"• Time: {crt_psp['formation_time'].strftime('%H:%M')}\n"
            message += f"\n"
        
        message += f"*Detection Time:* {signal_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}\n"
        message += f"*Latency:* IMMEDIATE\n\n"
        message += f"#ImmediateSignal #{self.pair_group} #{direction}"
        
        return message

    def log_detailed_status(self):
        """Log detailed status of all active features and recent signals"""
        logger.info(f"📋 FEATURE BOX DETAILED STATUS for {self.pair_group}")
        
        # Log active SMTs - FIXED SYNTAX
        smt_count = len(self.active_features['smt'])
        logger.info(f"  Active SMTs ({smt_count}):")
        for key, feature in self.active_features['smt'].items():
            smt_data = feature['smt_data']
            has_psp = "✅ WITH PSP" if feature['psp_data'] else "❌ NO PSP"
            expires_in = (feature['expiration'] - datetime.now(NY_TZ)).total_seconds() / 60
            logger.info(f"    - {smt_data['cycle']} {smt_data['direction']} {smt_data['quarters']} {has_psp} (expires in {expires_in:.1f}m)")
        
        # Log active CRTs - FIXED SYNTAX
        crt_count = len(self.active_features['crt'])
        logger.info(f"  Active CRTs ({crt_count}):")
        for key, feature in self.active_features['crt'].items():
            crt_data = feature['crt_data']
            has_psp = "✅ WITH PSP" if feature['psp_data'] else "❌ NO PSP"
            expires_in = (feature['expiration'] - datetime.now(NY_TZ)).total_seconds() / 60
            logger.info(f"    - {crt_data['timeframe']} {crt_data['direction']} {has_psp} (expires in {expires_in:.1f}m)")
        
        # Log active PSPs - FIXED SYNTAX
        psp_count = len(self.active_features['psp'])
        logger.info(f"  Active PSPs ({psp_count}):")
        for key, feature in self.active_features['psp'].items():
            psp_data = feature['psp_data']
            associated = f"→ {feature['associated_smt']}" if feature['associated_smt'] else "→ STANDALONE"
            expires_in = (feature['expiration'] - datetime.now(NY_TZ)).total_seconds() / 60
            logger.info(f"    - {psp_data['timeframe']} {psp_data['asset1_color']}/{psp_data['asset2_color']} {associated} (expires in {expires_in:.1f}m)")
        
        # Log recent signals - FIXED SYNTAX
        signal_count = len(self.sent_signals)
        logger.info(f"  Recent Signals ({signal_count}):")
        # Get last 5 signals (or all if less than 5)
        recent_signals = list(self.sent_signals.items())[-5:]
        for signal_key, sent_time in recent_signals:
            time_ago = (datetime.now(NY_TZ) - sent_time).total_seconds() / 60
            logger.info(f"    - {signal_key} ({time_ago:.1f}m ago)")
    
    def _is_feature_expired(self, feature):
        """Check if feature has expired"""
        return datetime.now(NY_TZ) > feature['expiration']
    
    def _remove_feature(self, feature_type, feature_key):
        """Remove feature from tracking"""
        if feature_key in self.active_features[feature_type]:
            del self.active_features[feature_type][feature_key]
            logger.debug(f"🗑️ Removed {feature_type}: {feature_key}")
    
    def cleanup_expired_features(self):
        """Remove expired features periodically"""
        current_time = datetime.now(NY_TZ)
        
        for feature_type in self.active_features:
            for feature_key, feature in list(self.active_features[feature_type].items()):
                if current_time > feature['expiration']:
                    logger.info(f"🧹 Expired {feature_type}: {feature_key}")
                    del self.active_features[feature_type][feature_key]
    
    def get_active_features_summary(self):
        """Get summary of currently active features"""
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
        
        return summary

    def debug_confluence_checks(self):
        """Temporary debug method to see why signals aren't firing"""
        logger.info(f"🔧 DEBUG: Checking why no signals for {self.pair_group}")
        
        # Check multiple SMTs
        bullish = []
        bearish = []
        
        for key, feature in self.active_features['smt'].items():
            if feature['smt_data']['direction'] == 'bullish':
                bullish.append(key)
            else:
                bearish.append(key)
        
        logger.info(f"🔧 DEBUG: Bullish SMTs: {bullish}")
        logger.info(f"🔧 DEBUG: Bearish SMTs: {bearish}")
        
        if len(bullish) >= 2:
            logger.info(f"🔧 DEBUG: SHOULD TRIGGER MULTIPLE BULLISH SMTs!")
        if len(bearish) >= 2:
            logger.info(f"🔧 DEBUG: SHOULD TRIGGER MULTIPLE BEARISH SMTs!")

    def debug_confluence_checks_detailed(self):
        """Detailed debug to see exactly why signals aren't firing"""
        logger.info(f"🔍 DETAILED DEBUG: Checking confluence for {self.pair_group}")
        
        # Check multiple SMTs confluence in detail
        bullish_smts = []
        bearish_smts = []
        
        for key, feature in self.active_features['smt'].items():
            if self._is_feature_expired(feature):
                logger.info(f"🔍 SMT {key} is EXPIRED - skipping")
                continue
                
            smt_data = feature['smt_data']
            if smt_data['direction'] == 'bullish':
                bullish_smts.append((key, smt_data))
            else:
                bearish_smts.append((key, smt_data))
        
        logger.info(f"🔍 Valid bullish SMTs: {len(bullish_smts)}")
        logger.info(f"🔍 Valid bearish SMTs: {len(bearish_smts)}")
        
        # Check if multiple SMTs confluence should trigger
        if len(bullish_smts) >= 2:
            logger.info(f"🎯 MULTIPLE BULLISH SMTs CONFLUENCE SHOULD TRIGGER!")
            logger.info(f"🔍 Calling _check_multiple_smts_confluence...")
            
            # Manually call the method to see what happens
            result = self._check_multiple_smts_confluence()
            logger.info(f"🔍 _check_multiple_smts_confluence returned: {result}")
            
        if len(bearish_smts) >= 2:
            logger.info(f"🎯 MULTIPLE BEARISH SMTs CONFLUENCE SHOULD TRIGGER!")
            logger.info(f"🔍 Calling _check_multiple_smts_confluence...")
            
            # Manually call the method to see what happens
            result = self._check_multiple_smts_confluence()
            logger.info(f"🔍 _check_multiple_smts_confluence returned: {result}")

    def debug_telegram_credentials(self):
        """Debug Telegram credential flow"""
        logger.info(f"🔧 TELEGRAM CREDENTIALS DEBUG for {self.pair_group}")
        logger.info(f"🔧 Telegram token type: {type(self.telegram_token)}")
        logger.info(f"🔧 Telegram token value: {'SET' if self.telegram_token else 'MISSING'}")
        if self.telegram_token:
            logger.info(f"🔧 Token preview: {str(self.telegram_token)[:10]}...")
        
        logger.info(f"🔧 Chat ID type: {type(self.telegram_chat_id)}")
        logger.info(f"🔧 Chat ID value: {'SET' if self.telegram_chat_id else 'MISSING'}")
        if self.telegram_chat_id:
            logger.info(f"🔧 Chat ID: {self.telegram_chat_id}")

    def _create_signal_signature(self, signal_data):
        """Create a unique signature for a signal based on its content"""
        # Extract the essence of the signal (what matters for duplicates)
        if signal_data['confluence_type'] == 'SMT_PSP_PRE_CONFIRMED':
            # For SMT+PSP: use SMT key + PSP timeframe + direction
            smt_key = signal_data['smt']['signal_key']
            psp_timeframe = signal_data['psp']['timeframe']
            direction = signal_data['direction']
            signature = f"SMT_PSP_{smt_key}_{psp_timeframe}_{direction}"
        
        elif signal_data['confluence_type'].startswith('MULTIPLE_SMTS'):
            # For multiple SMTs: use sorted SMT keys + direction
            smt_keys = []
            for smt in signal_data['multiple_smts']:
                # Create a unique identifier for each SMT
                smt_id = f"{smt['cycle']}_{smt['quarters']}"
                smt_keys.append(smt_id)
            
            # Sort to make order consistent
            smt_keys.sort()
            direction = signal_data['direction']
            signature = f"MULTI_SMT_{'_'.join(smt_keys)}_{direction}"
        
        elif signal_data['confluence_type'] == 'CRT_SMT_IMMEDIATE':
            # For CRT+SMT: use CRT key + SMT key
            crt_key = signal_data['crt']['signal_key']
            smt_key = signal_data['smt']['signal_key']
            signature = f"CRT_SMT_{crt_key}_{smt_key}"
        
        else:
            # Fallback: use the signal key
            signature = signal_data['signal_key']
        
        return signature
    
    def _is_duplicate_signal_signature(self, signal_data):
        """Check if we've already sent a signal with the same signature"""
        signature = self._create_signal_signature(signal_data)
        current_time = datetime.now(NY_TZ)
        
        # Clean up old signatures first
        self._cleanup_old_signatures()
        
        if signature in self.sent_signal_signatures:
            last_sent = self.sent_signal_signatures[signature]
            hours_since_sent = (current_time - last_sent).total_seconds() / 3600
            
            if hours_since_sent < self.signature_expiry_hours:
                logger.info(f"⏳ SIGNATURE DUPLICATE BLOCKED: {signature} (sent {hours_since_sent:.1f}h ago)")
                return True
        
        # Not a duplicate - store the signature
        self.sent_signal_signatures[signature] = current_time
        return False
    
    def _cleanup_old_signatures(self):
        """Remove signatures older than expiry time"""
        current_time = datetime.now(NY_TZ)
        expired_signatures = []
        
        for signature, sent_time in self.sent_signal_signatures.items():
            hours_since_sent = (current_time - sent_time).total_seconds() / 3600
            if hours_since_sent >= self.signature_expiry_hours:
                expired_signatures.append(signature)
        
        for signature in expired_signatures:
            del self.sent_signal_signatures[signature]
        
        if expired_signatures:
            logger.debug(f"🧹 Cleaned up {len(expired_signatures)} old signal signatures")







class FVGDetector:
    def __init__(self, min_gap_pct: float = 0.05):
        self.min_gap_pct = min_gap_pct
        self.active_fvgs = {}  # tf -> [fvgs]
        self.invalidate_std_mult = 4.0
        self.fvg_expiry_hours = 4888  

    def scan_tf(self, df, tf, asset):
        if tf not in ['M15', 'H1','H2', 'H3', 'H4', 'D'] or len(df) < 39:
            return []

        # ✅ NEW: Filter for CLOSED candles only
        if 'complete' in df.columns:
            # Get only closed candles
            df = df[df['complete'] == True].copy()
            logger.info(f"🔍 FVG Scan {asset} {tf}: Using {len(df)} closed candles")
        else:
            logger.warning(f"⚠️ FVG Scan {asset} {tf}: No 'complete' column - using all candles")
        recent = df.tail(38).reset_index(drop=True)
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
        # Fixed log (no nested min glitch)
        if self.active_fvgs[tf]:
            min_form = min(f['formation_time'] for f in self.active_fvgs[tf])
            post_count = len(df[df['time'] > min_form])
        else:
            post_count = 0
        logger.info(f"🔍 Active FVGs {tf}: {len(active)} (scanned {post_count} post-formation candles)")
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

    def test_fvg_detection(self):
        """Test FVG detection"""
        for instrument in self.instruments:
            for tf in ['M15', 'H1', 'H4']:
                data = self.market_data[instrument].get(tf)
                if data is not None and not data.empty:
                    fvgs = self.fvg_detector.scan_tf(data, tf, instrument)
                    logger.info(f"Test {instrument} {tf}: Found {len(fvgs)} FVGs")
                    for fvg in fvgs:
                        logger.info(f"  - {fvg['direction']} FVG at {fvg['fvg_low']:.4f}-{fvg['fvg_high']:.4f}")

    def _is_invalidated(self, fvg, post_df):
        """Your rule: Bull: close < B low OR close > B high + 4*std. Flip bear."""
        threshold = self.invalidate_std_mult * fvg['candle_b_std']
        for _, candle in post_df.iterrows():
            close = candle['close']
            if fvg['direction'] == 'bullish':
                if close < fvg['fvg_low']:  # Breach
                    logger.info(f"❌ Bull FVG invalidated: Close {close:.4f} < B low {fvg['candle_b_low']:.4f}")
                    return True
                if close > (fvg['candle_b_high'] + threshold):  # Over-extend up
                    logger.info(f"❌ Bull FVG invalidated: Over-extend {close:.4f} > B high +4std {fvg['candle_b_high'] + threshold:.4f}")
                    return True
            else:  # Bearish
                if close > fvg['fvg_high']:  # Breach
                    logger.info(f"❌ Bear FVG invalidated: Close {close:.4f} > B high {fvg['candle_b_high']:.4f}")
                    return True
                if close < (fvg['candle_b_low'] - threshold):  # Over-extend down
                    logger.info(f"❌ Bear FVG invalidated: Over-extend {close:.4f} < B low -4std {fvg['candle_b_low'] - threshold:.4f}")
                    return True
        return False

    def _is_over_mitigated(self, fvg, recent_df):
        """6+ candles in zone *post-formation* only."""
        if recent_df is None or recent_df.empty:
            logger.warning(f"⚠️ Over-mit check: Empty DF for {fvg['asset']} {fvg['tf']}")
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
        logger.info(f"🔍 Over-mit {fvg['asset']} {fvg['tf']}: {in_count} post-formation in-zone (threshold 6)")
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
    def __init__(self, min_zone_pct=0):  # Default to 0 to disable filtering
        self.min_zone_pct = min_zone_pct
        if min_zone_pct != 0:
            logger.warning(f"⚠️ min_zone_pct={min_zone_pct} is set but zone size filtering is disabled")
        logger.info(f"✅ SupplyDemandDetector initialized with min_zone_pct: {min_zone_pct}")
    
    def detect_swing_highs_lows(self, data, lookback=5):
        """
        Detect swing highs and lows - more sensitive for 100 bars
        """
        if len(data) < lookback * 2:
            return [], []
        
        highs = data['high'].values
        lows = data['low'].values
        times = data['time'].values
        
        swing_highs = []
        swing_lows = []
        
        # Find swing highs (peak with lower highs on both sides)
        for i in range(lookback, len(data) - lookback):
            # Check if current high is higher than previous 'lookback' highs
            if all(highs[i] > highs[i-j] for j in range(1, lookback+1)):
                # Check if current high is higher than next 'lookback' highs
                if all(highs[i] > highs[i+j] for j in range(1, lookback+1)):
                    swing_highs.append({
                        'index': i,
                        'price': highs[i],
                        'time': times[i],
                        'type': 'high'
                    })
        
        # Find swing lows (trough with higher lows on both sides)
        for i in range(lookback, len(data) - lookback):
            # Check if current low is lower than previous 'lookback' lows
            if all(lows[i] < lows[i-j] for j in range(1, lookback+1)):
                # Check if current low is lower than next 'lookback' lows
                if all(lows[i] < lows[i+j] for j in range(1, lookback+1)):
                    swing_lows.append({
                        'index': i,
                        'price': lows[i],
                        'time': times[i],
                        'type': 'low'
                    })
        
        return swing_highs, swing_lows
    
    def find_structure_breaks(self, data, swing_highs, swing_lows):
        """
        Find BOS/CHoCH by waiting for CLOSE beyond swing points
        """
        closes = data['close'].values
        times = data['time'].values
        
        # Combine and sort all swings
        all_swings = swing_highs + swing_lows
        all_swings.sort(key=lambda x: x['index'])
        
        breaks = []
        
        # Track current trend (1 = bullish, -1 = bearish, 0 = neutral)
        current_trend = 0
        
        # Check each swing point for breaks
        for swing in all_swings:
            swing_idx = swing['index']
            swing_price = swing['price']
            swing_type = swing['type']
            
            # Look forward from swing point for a break
            for i in range(swing_idx + 1, min(swing_idx + 20, len(data))):
                current_close = closes[i]
                
                if swing_type == 'high':
                    # Bearish break: CLOSE below swing high
                    if current_close < swing_price:
                        # Determine break type based on current trend
                        if current_trend == 1:  # Was bullish, now breaking down = CHoCH
                            break_type = 'CHoCH'
                        else:  # Was bearish or neutral = BOS
                            break_type = 'BOS'
                        
                        current_trend = -1  # Now bearish
                        
                        breaks.append({
                            'break_type': break_type,
                            'broken_swing': swing,
                            'break_index': i,
                            'break_price': current_close,
                            'break_time': times[i],
                            'trend': 'bearish',  # Price broke below = bearish trend
                            'swing_type': 'high'
                        })
                        break
                
                else:  # swing_type == 'low'
                    # Bullish break: CLOSE above swing low
                    if current_close > swing_price:
                        # Determine break type based on current trend
                        if current_trend == -1:  # Was bearish, now breaking up = CHoCH
                            break_type = 'CHoCH'
                        else:  # Was bullish or neutral = BOS
                            break_type = 'BOS'
                        
                        current_trend = 1  # Now bullish
                        
                        breaks.append({
                            'break_type': break_type,
                            'broken_swing': swing,
                            'break_index': i,
                            'break_price': current_close,
                            'break_time': times[i],
                            'trend': 'bullish',  # Price broke above = bullish trend
                            'swing_type': 'low'
                        })
                        break
        
        return breaks
    
    def find_order_block_in_broken_leg(self, data, broken_swing, break_index, trend):
        """
        Find the Order Block (extreme candle) in the broken leg
        For bearish break (price below swing high): Find highest high in leg
        For bullish break (price above swing low): Find lowest low in leg
        """
        start_idx = broken_swing['index']
        end_idx = break_index
        
        if start_idx >= end_idx:
            return None
        
        leg_data = data.iloc[start_idx:end_idx]
        
        if trend == 'bearish':
            # For bearish break: Find candle with highest high (Supply Zone)
            extreme_idx = leg_data['high'].idxmax()
            zone_type = 'supply'
        else:  # bullish
            # For bullish break: Find candle with lowest low (Demand Zone)
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
        """Check if a zone is still valid - NEW INVALIDATION RULES"""
        try:
            if current_data is None or current_data.empty:
                logger.debug(f"📭 No current data for zone validation: {zone.get('zone_name', 'Unknown')}")
                return True  # Assume valid if no data
            
            zone_type = zone['type']
            zone_low = zone['zone_low']
            zone_high = zone['zone_high']
            formation_time = zone['formation_time']
            creation_index = zone.get('creation_index', 0)
            
            # Import NY_TZ
            from pytz import timezone
            NY_TZ = timezone('America/New_York')
            
            # Convert formation_time to pandas Timestamp in NY_TZ
            try:
                # If formation_time is already pandas Timestamp
                if isinstance(formation_time, pd.Timestamp):
                    formation_ts = formation_time
                # If formation_time is numpy.datetime64
                elif isinstance(formation_time, np.datetime64):
                    formation_ts = pd.Timestamp(formation_time)
                # If formation_time is string
                elif isinstance(formation_time, str):
                    formation_ts = pd.to_datetime(formation_time, errors='coerce')
                    if pd.isna(formation_ts):
                        logger.error(f"❌ Could not parse formation_time: {formation_time}")
                        return False
                # If formation_time is datetime
                elif isinstance(formation_time, datetime):
                    formation_ts = pd.Timestamp(formation_time)
                else:
                    logger.error(f"❌ Unknown formation_time type: {type(formation_time)}")
                    return False
                
                # Ensure formation_time is in NY_TZ
                if formation_ts.tz is None:
                    # Assume UTC and convert to NY_TZ
                    formation_ts = formation_ts.tz_localize('UTC').tz_convert(NY_TZ)
                elif str(formation_ts.tz) != str(NY_TZ):
                    # Convert to NY_TZ
                    formation_ts = formation_ts.tz_convert(NY_TZ)
                    
            except Exception as e:
                logger.error(f"❌ Error converting formation_time: {e}")
                return False
            
            # Ensure current_data time is in NY_TZ
            current_data_copy = current_data.copy()
            if current_data_copy['time'].dt.tz is None:
                # If timezone-naive, assume UTC and convert to NY_TZ
                current_data_copy['time'] = current_data_copy['time'].dt.tz_localize('UTC').dt.tz_convert(NY_TZ)
            elif str(current_data_copy['time'].dt.tz) != str(NY_TZ):
                # Convert from current timezone to NY_TZ
                current_data_copy['time'] = current_data_copy['time'].dt.tz_convert(NY_TZ)
            
            # Get candles AFTER formation (using NY_TZ)
            subsequent_candles = current_data_copy[current_data_copy['time'] > formation_ts]
            
            if len(subsequent_candles) == 0:
                logger.debug(f"📭 No candles after formation for: {zone.get('zone_name', 'Unknown')}")
                return True
            
            # ===== NEW INVALIDATION RULES =====
            # Check expiration: 35 candles from creation
            candles_since_creation = len(subsequent_candles)
            if candles_since_creation >= 35:
                logger.info(f"⏰ ZONE EXPIRED (35 candles): {zone.get('zone_name', 'Unknown')}")
                return False
            
            # Get the highest high of candles A and B for DEMAND invalidation
            # Get the lowest low of candles A and B for SUPPLY invalidation
            highest_high = zone.get('highest_high', zone_high)
            lowest_low = zone.get('lowest_low', zone_low)
            
            if zone_type == 'demand':
                # DEMAND zone invalidated if ANY candle's HIGH > highest high of A & B
                if (subsequent_candles['high'] > highest_high).any():
                    breach_idx = subsequent_candles['high'].idxmax()
                    breach_candle = subsequent_candles.loc[breach_idx]
                    logger.info(f"❌ DEMAND ZONE INVALIDATED: {zone.get('zone_name', 'Unknown')}")
                    logger.info(f"   High {breach_candle['high']:.4f} > Highest high {highest_high:.4f}")
                    return False
            else:  # supply zone
                # SUPPLY zone invalidated if ANY candle's LOW < lowest low of A & B
                if (subsequent_candles['low'] < lowest_low).any():
                    breach_idx = subsequent_candles['low'].idxmin()
                    breach_candle = subsequent_candles.loc[breach_idx]
                    logger.info(f"❌ SUPPLY ZONE INVALIDATED: {zone.get('zone_name', 'Unknown')}")
                    logger.info(f"   Low {breach_candle['low']:.4f} < Lowest low {lowest_low:.4f}")
                    return False
            
            # Zone is valid
            return True
            
        except Exception as e:
            logger.error(f"❌ Error checking zone validity for {zone.get('zone_name', 'Unknown')}: {str(e)}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            return False
    
    def calculate_wick_percentage(self, candle):
        """Calculate wick percentage for a candle"""
        body_size = abs(candle['close'] - candle['open'])
        total_range = candle['high'] - candle['low']
        
        if total_range == 0:
            return 0
        
        if candle['close'] > candle['open']:  # Bullish
            upper_wick = candle['high'] - candle['close']
            wick_percentage = (upper_wick / total_range) * 100
        else:  # Bearish
            lower_wick = candle['open'] - candle['low']
            wick_percentage = (lower_wick / total_range) * 100
        
        return wick_percentage
    
    def is_demand_zone(self, candle_a, candle_b):
        """Check if two consecutive candles form a valid demand zone"""
        # Demand: Candle A bearish, Candle B bullish
        if candle_a['close'] >= candle_a['open']:  # Candle A not bearish
            return False
        if candle_b['close'] <= candle_b['open']:  # Candle B not bullish
            return False
        
        # The zone is between close of A and open of B
        # They should be at the same price level (or very close)
        price_diff_pct = abs(candle_a['close'] - candle_b['open']) / candle_a['close'] * 100
        if price_diff_pct > 0.05:  # More than 0.05% gap (tight tolerance)
            return False
        
        return True
    
    def is_supply_zone(self, candle_a, candle_b):
        """Check if two consecutive candles form a valid supply zone"""
        # Supply: Candle A bullish, Candle B bearish
        if candle_a['close'] <= candle_a['open']:  # Candle A not bullish
            return False
        if candle_b['close'] >= candle_b['open']:  # Candle B not bearish
            return False
        
        # The zone is between close of A and open of B
        # They should be at the same price level (or very close)
        price_diff_pct = abs(candle_a['close'] - candle_b['open']) / candle_a['close'] * 100
        if price_diff_pct > 0.05:  # More than 0.05% gap (tight tolerance)
            return False
        
        return True
    
    def check_zone_invalidation(self, zone, subsequent_candles, zone_type, asset, other_asset_data=None):
        """Check if zone is invalidated - NEW RULES"""
        highest_high = zone.get('highest_high', zone['zone_high'])
        lowest_low = zone.get('lowest_low', zone['zone_low'])
        
        if zone_type == 'demand':
            # Demand zone invalidated if ANY candle's HIGH > highest high of A & B
            if (subsequent_candles['high'] > highest_high).any():
                logger.debug(f"❌ Demand zone invalidated: High {subsequent_candles['high'].max():.4f} > Highest high {highest_high:.4f}")
                return True
        else:  # supply zone
            # Supply zone invalidated if ANY candle's LOW < lowest low of A & B
            if (subsequent_candles['low'] < lowest_low).any():
                logger.debug(f"❌ Supply zone invalidated: Low {subsequent_candles['low'].min():.4f} < Lowest low {lowest_low:.4f}")
                return True
        
        return False
    
    def calculate_invalidation_point(self, candle_a, candle_b, zone_type):
        """Calculate the invalidation point based on candles A and B"""
        if zone_type == 'demand':
            # For demand: highest high of A and B
            return max(candle_a['high'], candle_b['high'])
        else:  # supply
            # For supply: lowest low of A and B
            return min(candle_a['low'], candle_b['low'])
    
    def scan_timeframe(self, data, timeframe, asset):
        """Scan for supply and demand zones using A-B candle logic"""
        if data is None or len(data) < 10:
            logger.warning(f"⚠️ Not enough data for zone scan: {asset} {timeframe}")
            return []
        
        # Import NY_TZ
        from pytz import timezone
        NY_TZ = timezone('America/New_York')
        
        # Use only closed candles
        if 'complete' in data.columns:
            closed_data = data[data['complete'] == True].copy()
            logger.info(f"📊 {asset} {timeframe}: Using {len(closed_data)} closed candles for zones")
        else:
            closed_data = data.copy()
            logger.info(f"📊 {asset} {timeframe}: Using {len(closed_data)} candles (no 'complete' column)")
        
        if len(closed_data) < 10:
            logger.warning(f"⚠️ Not enough closed candles for {asset} {timeframe}: {len(closed_data)}")
            return []
        
        zones = []
        
        # Ensure all timestamps are in NY_TZ
        if closed_data['time'].dt.tz is None:
            # Assume UTC and convert to NY_TZ
            closed_data['time'] = closed_data['time'].dt.tz_localize('UTC').dt.tz_convert(NY_TZ)
        elif str(closed_data['time'].dt.tz) != str(NY_TZ):
            # Convert from current timezone to NY_TZ
            closed_data['time'] = closed_data['time'].dt.tz_convert(NY_TZ)
        
        # ==================== A-B CANDLE LOGIC ====================
        logger.info(f"🔄 Scanning for A-B candle patterns for {asset} {timeframe}")
        
        # Iterate through candles to find A-B patterns
        for i in range(len(closed_data) - 1):
            candle_a = closed_data.iloc[i]
            candle_b = closed_data.iloc[i + 1]
            
            # Check for DEMAND zone (A bearish, B bullish)
            if self.is_demand_zone(candle_a, candle_b):
                # The zone is defined by close of A and open of B (same price level)
                zone_price = (candle_a['close'] + candle_b['open']) / 2
                zone_low = zone_price * 0.9999  # Tiny buffer below
                zone_high = zone_price * 1.0001  # Tiny buffer above
                
                # Get the highest high of A and B for invalidation
                highest_high = max(candle_a['high'], candle_b['high'])
                lowest_low = min(candle_a['low'], candle_b['low'])
                
                # Check subsequent candles for immediate invalidation
                subsequent_start = i + 2
                if subsequent_start < len(closed_data):
                    subsequent_candles = closed_data.iloc[subsequent_start:min(subsequent_start + 5, len(closed_data))]
                    
                    # Skip if immediately invalidated (price trades above highest high)
                    invalidated = False
                    for _, candle in subsequent_candles.iterrows():
                        if candle['high'] > highest_high:
                            invalidated = True
                            break
                    
                    if invalidated:
                        continue
                
                # Create demand zone
                zone = {
                    'type': 'demand',
                    'zone_low': zone_low,
                    'zone_high': zone_high,
                    'formation_candle': {
                        'high': highest_high,
                        'low': lowest_low,
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
                    'wick_category': 'normal',
                    'candle_a_low': candle_a['low'],
                    'candle_a_high': candle_a['high'],
                    'candle_a_close': candle_a['close'],
                    'candle_a_open': candle_a['open'],
                    'candle_b_low': candle_b['low'],
                    'candle_b_high': candle_b['high'],
                    'candle_b_close': candle_b['close'],
                    'candle_b_open': candle_b['open'],
                    'highest_high': highest_high,
                    'lowest_low': lowest_low,
                    'creation_index': i,
                    'zone_price': zone_price  # The actual level (close of A = open of B)
                }
                zones.append(zone)
                logger.debug(f"✅ Found DEMAND zone: {zone['zone_name']} at {zone_price:.4f}")
                logger.debug(f"   Candle A: {candle_a['open']:.4f} -> {candle_a['close']:.4f} (bearish)")
                logger.debug(f"   Candle B: {candle_b['open']:.4f} -> {candle_b['close']:.4f} (bullish)")
                logger.debug(f"   Zone level: {zone_price:.4f}, Invalidation: {highest_high:.4f}")
            
            # Check for SUPPLY zone (A bullish, B bearish)
            elif self.is_supply_zone(candle_a, candle_b):
                # The zone is defined by close of A and open of B (same price level)
                zone_price = (candle_a['close'] + candle_b['open']) / 2
                zone_low = zone_price * 0.9999  # Tiny buffer below
                zone_high = zone_price * 1.0001  # Tiny buffer above
                
                # Get the lowest low of A and B for invalidation
                highest_high = max(candle_a['high'], candle_b['high'])
                lowest_low = min(candle_a['low'], candle_b['low'])
                
                # Check subsequent candles for immediate invalidation
                subsequent_start = i + 2
                if subsequent_start < len(closed_data):
                    subsequent_candles = closed_data.iloc[subsequent_start:min(subsequent_start + 5, len(closed_data))]
                    
                    # Skip if immediately invalidated (price trades below lowest low)
                    invalidated = False
                    for _, candle in subsequent_candles.iterrows():
                        if candle['low'] < lowest_low:
                            invalidated = True
                            break
                    
                    if invalidated:
                        continue
                
                # Create supply zone
                zone = {
                    'type': 'supply',
                    'zone_low': zone_low,
                    'zone_high': zone_high,
                    'formation_candle': {
                        'high': highest_high,
                        'low': lowest_low,
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
                    'wick_category': 'normal',
                    'candle_a_low': candle_a['low'],
                    'candle_a_high': candle_a['high'],
                    'candle_a_close': candle_a['close'],
                    'candle_a_open': candle_a['open'],
                    'candle_b_low': candle_b['low'],
                    'candle_b_high': candle_b['high'],
                    'candle_b_close': candle_b['close'],
                    'candle_b_open': candle_b['open'],
                    'highest_high': highest_high,
                    'lowest_low': lowest_low,
                    'creation_index': i,
                    'zone_price': zone_price  # The actual level (close of A = open of B)
                }
                zones.append(zone)
                logger.debug(f"✅ Found SUPPLY zone: {zone['zone_name']} at {zone_price:.4f}")
                logger.debug(f"   Candle A: {candle_a['open']:.4f} -> {candle_a['close']:.4f} (bullish)")
                logger.debug(f"   Candle B: {candle_b['open']:.4f} -> {candle_b['close']:.4f} (bearish)")
                logger.debug(f"   Zone level: {zone_price:.4f}, Invalidation: {lowest_low:.4f}")
        
        # ==================== FILTER OVERLAPPING ZONES ====================
        # Remove zones that overlap significantly
        filtered_zones = []
        zones.sort(key=lambda x: x['formation_time'])
        
        for i, zone in enumerate(zones):
            if i == 0:
                filtered_zones.append(zone)
                continue
            
            # Check if this zone overlaps with previous zone
            prev_zone = filtered_zones[-1]
            
            # Calculate overlap percentage
            overlap_low = max(zone['zone_low'], prev_zone['zone_low'])
            overlap_high = min(zone['zone_high'], prev_zone['zone_high'])
            
            if overlap_low < overlap_high:  # There is overlap
                zone_range = zone['zone_high'] - zone['zone_low']
                overlap_range = overlap_high - overlap_low
                overlap_pct = (overlap_range / zone_range) * 100
                
                if overlap_pct > 50:  # More than 50% overlap
                    # Keep the zone with stronger pattern (closer price match)
                    zone_price_diff = abs(zone['candle_a_close'] - zone['candle_b_open'])
                    prev_price_diff = abs(prev_zone['candle_a_close'] - prev_zone['candle_b_open'])
                    
                    if zone_price_diff < prev_price_diff:  # Better price match
                        filtered_zones[-1] = zone
                    continue
            
            filtered_zones.append(zone)
        
        logger.info(f"🔍 A-B Pattern Scan {asset} {timeframe}: Found {len(filtered_zones)} zones (from {len(zones)} raw)")
        
        # Debug: Log first few zones found
        if filtered_zones and len(filtered_zones) > 0:
            for i, zone in enumerate(filtered_zones[:3]):
                invalidation_point = zone['highest_high'] if zone['type'] == 'demand' else zone['lowest_low']
                logger.info(f"   Zone {i+1}: {zone['zone_name']}")
                logger.info(f"      {zone['type']}: {zone['zone_price']:.4f} (close of A = open of B)")
                logger.info(f"      Invalidation: {invalidation_point:.4f}")
                logger.info(f"      Formed: {zone['formation_time']}")
        else:
            logger.info(f"   No zones found for {asset} {timeframe}")
        
        return filtered_zones
    
    def scan_all_timeframes(self, market_data, instruments, timeframes=['M15', 'H1', 'H4']):
        """Scan all instruments and timeframes for zones"""
        all_zones = []
        
        for instrument in instruments:
            for timeframe in timeframes:
                data = market_data.get(instrument, {}).get(timeframe)
                if data is not None and not data.empty:
                    zones = self.scan_timeframe(data, timeframe, instrument)
                    all_zones.extend(zones)
        
        # Sort by timeframe importance (H4 > H1 > M15 > M5)
        timeframe_order = {'H4': 4, 'H1': 3, 'M15': 2, 'M5': 1}
        all_zones.sort(key=lambda x: timeframe_order.get(x['timeframe'], 0), reverse=True)
        
        return all_zones


# ================================
# ULTIMATE TRADING SYSTEM WITH TRIPLE CONFLUENCE
# ================================
# ULTIMATE TRADING SYSTEM WITH TRIPLE CONFLUENCE
# ================================

class UltimateTradingSystem:
    def __init__(self, pair_group, pair_config, telegram_token=None, telegram_chat_id=None):
        # Store the parameters as instance variables
        self.pair_group = pair_group
        self.pair_config = pair_config
        self.sd_detector = SupplyDemandDetector(min_zone_pct=0)  # 0.5% minimum zone
        self.volatile_pairs = ['XAU_USD']
        
        
        # Handle Telegram credentials
        self.telegram_token = telegram_token
        self.telegram_chat_id = telegram_chat_id
        
        # BACKWARD COMPATIBLE: Handle both old and new structures
        if 'instruments' in pair_config:
            self.instruments = pair_config['instruments']  # NEW structure
        else:
            # OLD structure: convert pair1/pair2 to instruments list
            self.instruments = [pair_config['pair1'], pair_config['pair2']]
            logger.info(f"🔄 Converted old structure for {pair_group} to instruments: {self.instruments}")
        
        # Initialize components - REORDERED!
        self.timing_manager = RobustTimingManager()
        self.quarter_manager = RobustQuarterManager()
        
        # FIRST create FeatureBox
        self.feature_box = RealTimeFeatureBox(
            self.pair_group, 
            self.timing_manager, 
            self.telegram_token, 
            self.telegram_chat_id
        )
        
        # THEN create detectors and connect FeatureBox
        self.smt_detector = UltimateSMTDetector(pair_config, self.timing_manager)
        self.crt_detector = RobustCRTDetector(self.timing_manager)
        self.crt_detector.feature_box = self.feature_box  # ← NOW THIS WORKS!
        self.feature_box.sd_detector = self.sd_detector
        
        # Data storage for all instruments
        self.market_data = {inst: {} for inst in self.instruments}
        
        logger.info(f"🎯 Initialized ULTIMATE trading system for {self.pair_group}: {', '.join(self.instruments)}")
        logger.info(f"🎯 FVG Analyzer initialized for {pair_group}")
        self.fvg_detector = FVGDetector(min_gap_pct=0.20)
        self.fvg_smt_tap_sent = {}  # Track FVG+SMT tap signals sent
        self.crt_smt_ideas_sent = {}
        self.sd_zone_sent = {}  # For Supply/Demand zone signals
        self.sd_hp_sent = {}    # For High Probability SD zone signals    
        self.fvg_ideas_sent = {}
        self.double_smt_sent = {}
        self.smart_timing = SmartTimingSystem()
        self.last_candle_scan = {}
        # Cooldown periods (in seconds)
        self.COOLDOWN_HOURS = 24 * 3600  # 24 hours
        self.CLEANUP_DAYS = 7 * 24 * 3600  # Clean up after 7 days
        self.timeframe_cycle_map = {
                'H4': ['weekly', 'daily'],   # Monthly removed
                'H1': ['daily'], 
                'M15': ['daily', '90min']
            }
        
    def get_sleep_time(self):
        """Use smart timing instead of fixed intervals"""
        return self.smart_timing.get_smart_sleep_time()
    
    async def run_ultimate_analysis(self, api_key):
        """Run analysis triggered by new candle formation"""
        try:
            # Cleanup expired features first
            self.feature_box.cleanup_expired_features()
            self.cleanup_old_signals()
            
            # Fetch data (this will get the new candle)
            await self._fetch_all_data_parallel(api_key)
    
            self.reset_smt_detector_state()
            
            # Check if we have new candles that warrant immediate scanning
            new_candles_detected = self._check_new_candles()
            
            if new_candles_detected:
                logger.info(f"🎯 NEW CANDLES DETECTED - Running analysis")
                
                # Scan for new features and add to Feature Box
                await self._scan_and_add_features_immediate()
                
                # ✅ Scan for Supply/Demand zones and add to FeatureBox
                self._scan_and_add_sd_zones()
                
                self.debug_feature_box()
                self.debug_smt_detection()
                
                # ✅ Run ALL scans - they each check for signals independently
                
                # 1. FVG+SMT Scan
                fvg_signal = self._scan_fvg_with_smt_tap()
                
                # 2. Supply/Demand+SMT Scan (regardless of FVG result)
                sd_signal = self._scan_sd_with_smt_tap()
                
                # 3. CRT+SMT Scan (regardless of previous results)
                crt_signal = self._scan_crt_smt_confluence()
                
                # 4. Double SMT Scan (regardless of previous results)
                double_signal = self._scan_double_smts_temporal()
                
                # Log what we found
                signals_found = sum([fvg_signal, sd_signal, crt_signal, double_signal])
                logger.info(f"🎯 Signals found: {signals_found} (FVG:{fvg_signal}, SD:{sd_signal}, CRT:{crt_signal}, Double:{double_signal})")
                
                # Get current feature summary
                summary = self.feature_box.get_active_features_summary()
                sd_count = len(self.feature_box.active_features['sd_zone'])
                logger.info(f"📊 {self.pair_group} Feature Summary: {summary['smt_count']} SMTs, {sd_count} SD zones, {summary['crt_count']} CRTs, {summary['psp_count']} PSPs")
            else:
                logger.info(f"⏸️ No new candles - skipping analysis")
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error in candle-triggered analysis for {self.pair_group}: {str(e)}", exc_info=True)
            return None

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
        logger.info(f"🔍 SCANNING: Supply/Demand Zones")
        
        # Import NY_TZ
        from pytz import timezone
        NY_TZ = timezone('America/New_York')
        
        # Debug: Check timezone of data
        for instrument in self.instruments:
            for timeframe in ['M15', 'H1', 'H4']:
                data = self.market_data[instrument].get(timeframe)
                if data is not None and not data.empty:
                    sample_time = data['time'].iloc[0]
                    if hasattr(sample_time, 'tz'):
                        tz_info = str(sample_time.tz)
                    else:
                        tz_info = 'NO TIMEZONE'
                    logger.info(f"📊 {instrument} {timeframe}: First candle at {sample_time}, TZ: {tz_info}")
                    
                    # Convert to NY_TZ if needed
                    if data['time'].dt.tz is None:
                        data['time'] = data['time'].dt.tz_localize('UTC').dt.tz_convert(NY_TZ)
                        logger.info(f"   ↳ Converted to NY_TZ")
        
        timeframes_to_scan = ['M15', 'H1', 'H4']
        if 'XAU_USD' in self.instruments:
            timeframes_to_scan.append('M5')
        
        zones_added = 0
        zones_invalidated = 0
        
        for instrument in self.instruments:
            for timeframe in timeframes_to_scan:
                data = self.market_data[instrument].get(timeframe)
                if data is not None and not data.empty:
                    # Get OTHER instrument's data for dual validation
                    other_instrument = [inst for inst in self.instruments if inst != instrument][0]
                    other_data = self.market_data[other_instrument].get(timeframe)
                    
                    # Scan for zones
                    zones = self.sd_detector.scan_timeframe(data, timeframe, instrument)
                    logger.info(f"📊 {instrument} {timeframe}: Found {len(zones)} zones")
                    
                    for zone in zones:
                        # Check if zone is still valid
                        is_valid = self.sd_detector.check_zone_still_valid(zone, data, other_data)
                        
                        if is_valid:
                            # Add to FeatureBox
                            if self.feature_box.add_sd_zone(zone):
                                zones_added += 1
                                logger.info(f"📦 Added {zone['type']} zone: {zone['zone_name']}")
                                logger.info(f"   Range: {zone['zone_low']:.4f}-{zone['zone_high']:.4f}")
                                logger.info(f"   Formed: {zone['formation_time']}")
                        else:
                            zones_invalidated += 1
                            logger.info(f"❌ Zone invalidated: {zone['zone_name']}")
        
        logger.info(f"📊 SD Zones Summary: {zones_added} added, {zones_invalidated} invalidated")
        
        # DEBUG: Show what's in FeatureBox
        active_zones = self.feature_box.get_active_sd_zones()
        logger.info(f"📦 FeatureBox now has {len(active_zones)} active SD zones")
        
        for zone in active_zones:
            wick_note = f"(wick-adjusted)" if zone.get('wick_adjusted', False) else ""
            logger.info(f"📦   {zone['zone_name']} {wick_note}: {zone['type']} at {zone['zone_low']:.4f}-{zone['zone_high']:.4f}")
            logger.info(f"     Formed: {zone['formation_time']}")
        
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
            for timeframe in ['M5', 'M15', 'H1', 'H4']:
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
                    logger.info(f"🕯️ First scan for {instrument} {timeframe}")
                elif latest_candle_time > self.last_candle_scan[key]:
                    self.last_candle_scan[key] = latest_candle_time
                    new_candles = True
                    logger.info(f"🕯️ NEW CANDLE: {instrument} {timeframe} at {latest_candle_time.strftime('%H:%M')}")
        
        return new_candles
    
    async def _fetch_all_data_parallel(self, api_key):
        """Fetch data in parallel with PROVEN candle counts that WORKED"""
        tasks = []
        
        # PROVEN CANDLE COUNTS FROM WORKING VERSION
        proven_counts = {
            # For SMT/CRT detection (keep these SHORT)
            'H4': 40,   # Monthly timeframe - 40 candles (was working)
            'H1': 40,   # Weekly timeframe - 40 candles (was working)  
            'M15': 40,   # Daily timeframe - 40 candles (was working)
            'M5': 40,    # 90min timeframe - 40 candles (was working)
            
            # For CRT only
            'H2': 10, 'H3': 10, 'H6': 10, 'H8': 10, 'H12': 10,
            
            # For SD Zones ONLY - use MORE candles
            'SD_H4': 100,  # For SD zones on H4
            'SD_H1': 100,  # For SD zones on H1  
            'SD_M15': 100, # For SD zones on M15
            'SD_M5': 100,  # For SD zones on M5
        }
        
        # Combine all required timeframes
        required_timeframes = []
        
        # 1. Add SMT/CRT timeframes (SHORT lookback)
        for cycle in self.pair_config['timeframe_mapping'].values():
            if cycle not in required_timeframes:
                required_timeframes.append(cycle)
        
        # 2. Add CRT timeframes
        for tf in CRT_TIMEFRAMES:
            if tf not in required_timeframes:
                required_timeframes.append(tf)
        
        # 3. ADD SD Zone timeframes (LONG lookback - separate calls)
        sd_timeframes = ['M15', 'H1', 'H4']
        if 'XAU_USD' in self.instruments:
            sd_timeframes.append('M5')
        
        # Create fetch tasks
        for instrument in self.instruments:
            # FIRST: Fetch SHORT data for SMT/CRT
            for tf in required_timeframes:
                count = proven_counts.get(tf, 40)  # Default to 40
                task = asyncio.create_task(
                    self._fetch_single_instrument_data(instrument, tf, count, api_key)
                )
                tasks.append(task)
            
            # THEN: Fetch LONG data for SD Zones (separate calls)
            for tf in sd_timeframes:
                count = 100  # 100 candles for SD zones
                task = asyncio.create_task(
                    self._fetch_single_instrument_data(instrument, tf, count, api_key)
                )
                tasks.append(task)
        
        # Wait for ALL data
        try:
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=45.0)
            logger.info(f"✅ Parallel fetch: SHORT data for SMT/CRT, LONG data for SD zones")
        except asyncio.TimeoutError:
            logger.warning(f"⚠️ Parallel data fetch timeout for {self.pair_group}")
    
    async def _fetch_single_instrument_data(self, instrument, timeframe, count, api_key):
        """Fetch data for single instrument and convert to NY_TZ"""
        try:
            from pytz import timezone
            NY_TZ = timezone('America/New_York')
            
            df = await asyncio.get_event_loop().run_in_executor(
                None, fetch_candles, instrument, timeframe, count, api_key
            )
            
            if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
                # Convert timestamps to NY_TZ
                if df['time'].dt.tz is None:
                    # Assume UTC and convert to NY_TZ
                    df['time'] = df['time'].dt.tz_localize('UTC').dt.tz_convert(NY_TZ)
                    logger.debug(f"📅 Converted {instrument} {timeframe} to NY_TZ")
                
                self.market_data[instrument][timeframe] = df
                return True
            return False
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
            
            # Check if we have new data for this cycle's timeframe
            #if self._has_new_candle_data(timeframe):
            logger.info(f"🔍 Immediate scan: {cycle} cycle ({timeframe})")
                
            asset1_data = self.market_data[self.instruments[0]].get(timeframe)
            asset2_data = self.market_data[self.instruments[1]].get(timeframe)

            logger.info(f"🔍 SMT Data Check - {self.instruments[0]} {timeframe}: "
                    f"{'Has data' if asset1_data is not None else 'NO DATA'}, "
                    f"{len(asset1_data) if asset1_data is not None else 0} candles")
            logger.info(f"🔍 SMT Data Check - {self.instruments[1]} {timeframe}: "
                        f"{'Has data' if asset2_data is not None else 'NO DATA'}, "
                        f"{len(asset2_data) if asset2_data is not None else 0} candles")
            
                
            if (asset1_data is not None and isinstance(asset1_data, pd.DataFrame) and not asset1_data.empty and
                asset2_data is not None and isinstance(asset2_data, pd.DataFrame) and not asset2_data.empty):
                    
                logger.info(f"🔍 Scanning {cycle} cycle ({timeframe}) for SMT...")
                smt_signal = self.smt_detector.detect_smt_all_cycles(asset1_data, asset2_data, cycle)
                if smt_signal:
                    logger.info(f"🔍 SMT RETURNED: {smt_signal.get('signal_key', 'No key')}")
                else:
                    logger.info(f"🔍 SMT detector returned None for {cycle}")

                # logger.info(f"🔍 Scanning {cycle} cycle ({timeframe}) for SMT...")
                # smt_signal = self.smt_detector.detect_smt_all_cycles(asset1_data, asset2_data, cycle)
                    
                if smt_signal:
                        # Check for PSP immediately
                    psp_signal = self.smt_detector.check_psp_for_smt(smt_signal, asset1_data, asset2_data)
                        
                    self.feature_box.add_smt(smt_signal, psp_signal)
                    smt_detected_count += 1
                    logger.info(f"✅ SMT DETECTED: {cycle} {smt_signal['direction']} - PSP: {'Yes' if psp_signal else 'No'}")
                else:
                    logger.info(f"🔍 No SMT found for {cycle} cycle")
            else:
                logger.warning(f"⚠️ No data for {cycle} SMT scan")
        
        logger.info(f"📊 SMT Scan Complete: Detected {smt_detected_count} SMTs")

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
        
        logger.info(f"🧪 Testing with {len(asset1_data)} candles")
        
        # 3. Detect SMT
        smt_signal = self.smt_detector.detect_smt_all_cycles(asset1_data, asset2_data, 'weekly')
        
        if smt_signal:
            logger.info(f"🧪 SMT Detected: {smt_signal['signal_key']}")
            
            # 4. Check PSP
            psp_signal = self.smt_detector.check_psp_for_smt(smt_signal, asset1_data, asset2_data)
            logger.info(f"🧪 PSP: {'Found' if psp_signal else 'Not found'}")
            
            # 5. Add to FeatureBox
            success = self.feature_box.add_smt(smt_signal, psp_signal)
            logger.info(f"🧪 Added to FeatureBox: {success}")
            
            # 6. Check FeatureBox
            logger.info(f"🧪 FeatureBox now has {len(self.feature_box.active_features['smt'])} SMTs")
        else:
            logger.info("🧪 No SMT detected")

    def _scan_crt_smt_confluence(self):
        """Check CRT with SMT confluence (CRT on higher TF, SMT on lower TF)"""
        logger.info(f"🔷 SCANNING: CRT + SMT Confluence")
        
        # CRT timeframes to check
        crt_timeframes = ['H4', 'H1', 'H6','H12']
        
        # Mapping: CRT timeframe -> allowed SMT cycles
        CRT_SMT_MAPPING = {
            'H4': ['weekly', 'daily'],  # 4hr CRT → Weekly OR Daily SMT
            'H1': ['daily'],           # 1hr CRT → Daily SMT
            'H6': ['daily','weekly'],  # 15min CRT → Daily OR 90min SMT
            'H12': ['daily','weekly']
        }
        
        # Check each instrument for CRT
        for instrument in self.instruments:
            for crt_tf in crt_timeframes:
                data = self.market_data[instrument].get(crt_tf)
                if data is None or data.empty:
                    continue
                
                # Get the other instrument's data for CRT detection
                other_instrument = [inst for inst in self.instruments if inst != instrument][0]
                other_data = self.market_data[other_instrument].get(crt_tf)
                
                if other_data is None or other_data.empty:
                    continue
                
                # Detect CRT
                crt_signal = self.crt_detector.calculate_crt_current_candle(
                    data, data, other_data, crt_tf
                )
                
                if crt_signal:
                    logger.info(f"🔷 CRT DETECTED: {crt_tf} {crt_signal['direction']} on {instrument}")
                    
                    # Check for SMT confluence
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
                            
                            # Send the signal
                            return self._send_crt_smt_signal(crt_signal, smt_data, has_psp, instrument)
        
        logger.info(f"🔷 No CRT+SMT confluence found")
        return False

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
        """Find SMTs that match the FVG's direction and timeframe - WITH TAP DEBUG"""
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
        
        # Timeframe mapping: SD Zone -> allowed SMT cycles
        sd_to_smt_cycles = {
            'H4': ['weekly', 'daily'],      # H4 Zone → Weekly (H1) or Daily (M15) SMT
            'H1': ['weekly', 'daily'],      # H1 Zone → Weekly (H1) or Daily (M15) SMT  
            'M15': ['daily','90min'],               # M15 Zone → Daily (M15) SMT
            'M5': ['daily','90min']                 # M5 Zone → 90min (M5) SMT
        }
        
        # Get all active SD zones from FeatureBox
        active_zones = self.feature_box.get_active_sd_zones()
        logger.info(f"🔍 Found {len(active_zones)} active SD zones in FeatureBox")
        
        if not active_zones:
            logger.info(f"❌ No active SD zones in FeatureBox")
            return False
        
        # Sort zones by timeframe importance (H4 > H1 > M15 > M5)
        timeframe_order = {'H4': 4, 'H1': 3, 'M15': 2, 'M5': 1}
        active_zones.sort(key=lambda x: timeframe_order.get(x['timeframe'], 0), reverse=True)
        
        for zone in active_zones:
            zone_type = zone['type']  # 'supply' or 'demand'
            zone_direction = 'bearish' if zone_type == 'supply' else 'bullish'  # Convert to direction
            zone_timeframe = zone['timeframe']
            zone_asset = zone['asset']
            zone_low = zone['zone_low']
            zone_high = zone['zone_high']
            zone_formation_time = zone['formation_time']
            
            logger.info(f"🔍 Checking {zone_type.upper()} zone: {zone['zone_name']} "
                       f"({zone_low:.4f} - {zone_high:.4f})")
            
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
                    
                    # Send the signal
                    return self._send_sd_smt_tap_signal(
                        zone, smt_data, has_psp, is_hp_zone
                    )
        
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
                'H4': 'D'
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
        logger.info(f"🔍 SCANNING: FVG + SMT Second Swing Tap (Cross-TF) - PSP REQUIRED - CLOSED CANDLES ONLY")
        
        # CORRECT CROSS-TIMEFRAME MAPPING
        fvg_to_smt_cycles = {
            'H4': ['weekly', 'daily'],    # H4 FVG → Weekly (H1) or Daily (M15) SMT
            'H1': ['daily'],              # H1 FVG → Daily (M15) SMT
            'M15': ['daily', '90min']     # M15 FVG → Daily (M15) or 90min (M5) SMT
        }
        
        # We need to get FVGs from FVGDetector, not from _find_zone_fvgs (which uses Fibonacci)
        # Let's scan for FVGs using FVGDetector with ONLY CLOSED CANDLES
        all_fvgs = []
        
        for instrument in self.instruments:
            for fvg_tf in ['H4', 'H1', 'M15']:  # Only these FVG timeframes
                data = self.market_data[instrument].get(fvg_tf)
                if data is not None and not data.empty:
                    # ✅ NEW: Use only CLOSED candles for FVG detection
                    if 'complete' in data.columns:
                        closed_data = data[data['complete'] == True].copy()
                        logger.info(f"📊 {instrument} {fvg_tf}: {len(data)} total candles, {len(closed_data)} closed candles")
                        
                        if len(closed_data) < 5:  # Need at least 5 closed candles
                            logger.warning(f"⚠️ Not enough closed candles for {instrument} {fvg_tf}: {len(closed_data)}")
                            continue
                        
                        # Use FVGDetector to get FVGs from CLOSED CANDLES ONLY
                        fvgs = self.fvg_detector.scan_tf(closed_data, fvg_tf, instrument)
                    else:
                        # Fallback if no 'complete' column
                        logger.warning(f"⚠️ No 'complete' column in {instrument} {fvg_tf} data")
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
        
        logger.info(f"🔍 Found {len(all_fvgs)} FVGs from CLOSED CANDLES")
        
        for fvg_idea in all_fvgs:
            fvg_direction = fvg_idea['direction']
            fvg_timeframe = fvg_idea['timeframe']
            fvg_asset = fvg_idea['asset']
            fvg_low = fvg_idea['fvg_low']
            fvg_high = fvg_idea['fvg_high']
            fvg_formation_time = fvg_idea['formation_time']  # Get FVG formation time
            
            # Get which SMT cycles can tap this FVG timeframe
            relevant_cycles = fvg_to_smt_cycles.get(fvg_timeframe, [])
            
            logger.info(f"🔍 Checking FVG {fvg_idea['fvg_name']} formed at {fvg_formation_time} - Can be tapped by: {relevant_cycles}")
            
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
                if not has_psp:
                    logger.info(f"⏳ Skipping FVG+SMT: {smt_cycle} SMT has no PSP confirmation")
                    continue  # Skip SMTs without PSP
                
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
                    logger.info(f"⚠️ No swing data for {fvg_asset} in SMT {smt_cycle}")
                    continue
                
                # Extract second swing time (could be dict or Timestamp)
                if isinstance(asset_curr, dict):
                    second_swing_time = asset_curr.get('time')
                else:
                    second_swing_time = asset_curr  # Assuming it's already a Timestamp
                
                if not second_swing_time:
                    logger.info(f"⚠️ No second swing time found for {fvg_asset} in SMT {smt_cycle}")
                    continue
                
                # REJECT if SMT second swing is BEFORE FVG formation
                if second_swing_time <= fvg_formation_time:
                    logger.info(f"❌ FVG+SMT REJECTED: SMT {smt_cycle} second swing at {second_swing_time} is BEFORE FVG formation at {fvg_formation_time}")
                    continue  # Skip this SMT
                
                # Check if SMT's second swing traded in FVG zone (CROSS-TIMEFRAME)
                tapped = self._check_cross_tf_smt_second_swing_in_fvg(
                    smt_data, fvg_asset, fvg_low, fvg_high, fvg_direction, 
                    fvg_timeframe, smt_cycle, fvg_formation_time  # PASS FVG FORMATION TIME!
                )
                
                if tapped:
                    # Check if only ONE asset tapped (HP FVG)
                    is_hp_fvg = self._check_hp_fvg_fix(fvg_idea, fvg_asset)
                    
                    logger.info(f"✅ FVG+SMT TAP CONFIRMED WITH PSP: {smt_cycle} {smt_data['direction']} "
                               f"tapped {fvg_timeframe} FVG on {fvg_asset}, HP: {is_hp_fvg}")
                    
                    # Send the signal
                    return self._send_fvg_smt_tap_signal(
                        fvg_idea, smt_data, has_psp, is_hp_fvg
                    )
        
        logger.info(f"🔍 No FVG+SMT setups with PSP found")
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
                logger.info(f"❌ CROSS-TF REJECTED: SMT second swing at {second_swing_time} is BEFORE FVG formation at {fvg_formation_time}")
                return False
            
            # Look for candles around second swing time in SMT timeframe
            time_diffs = abs(smt_price_data['time'] - second_swing_time)
            closest_idx = time_diffs.idxmin()
            
            # Check 5 candles before and after (adjust as needed)
            start_idx = max(0, closest_idx - 5)
            end_idx = min(len(smt_price_data) - 1, closest_idx + 5)
            
            logger.info(f"🔍 Cross-TF Tap: {smt_cycle}({smt_tf}) → {fvg_tf} FVG at {fvg_low:.4f}-{fvg_high:.4f}")
            logger.info(f"   Checking {smt_tf} candles around {second_swing_time.strftime('%H:%M')}")
            
            for idx in range(start_idx, end_idx + 1):
                candle = smt_price_data.iloc[idx]
                
                # Check if this candle entered the FVG zone
                if fvg_direction == 'bullish':
                    # Bullish FVG: zone is above, price enters from below
                    # Zone tapped if candle's low <= fvg_high AND candle's high >= fvg_low
                    if candle['low'] <= fvg_high and candle['high'] >= fvg_low:
                        entry_price = candle['low']  # Price entered from bottom
                        logger.info(f"✅ CROSS-TF TAP: {smt_tf} candle at {candle['time'].strftime('%H:%M')} "
                                   f"entered bullish FVG zone (low: {candle['low']:.4f} <= {fvg_high:.4f})")
                        return True
                else:  # bearish
                    # Bearish FVG: zone is below, price enters from above  
                    # Zone tapped if candle's high >= fvg_low AND candle's low <= fvg_high
                    if candle['high'] >= fvg_low and candle['low'] <= fvg_high:
                        entry_price = candle['high']  # Price entered from top
                        logger.info(f"✅ CROSS-TF TAP: {smt_tf} candle at {candle['time'].strftime('%H:%M')} "
                                   f"entered bearish FVG zone (high: {candle['high']:.4f} >= {fvg_low:.4f})")
                        return True
            
            logger.info(f"❌ No {smt_tf} candle entered FVG zone around second swing")
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
                logger.info(f"❌ TRACE TAP REJECTED: SMT second swing at {second_swing_time} is BEFORE FVG formation at {fvg_formation}")
                return False
            
            logger.info(f"TRACE TAP {smt_data['cycle']} on {asset} FVG: 2nd swing {second_swing_time.strftime('%H:%M')} price {second_swing_price:.4f}")
            
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

    def get_sleep_time(self):
        """Calculate sleep time until next relevant candle - SIMPLIFIED FOR NOW"""
        # Since we're using Feature Box now, we'll use a simpler approach
        # TODO: Implement proper sleep timing based on active features
        
        # For now, use base interval or check if we have any active features
        summary = self.feature_box.get_active_features_summary()
        
        if summary['smt_count'] > 0 or summary['crt_count'] > 0:
            # We have active features, check more frequently
            sleep_time = 30  # 30 seconds
            logger.info(f"⏰ {self.pair_group}: Active features detected - sleeping {sleep_time}s")
        else:
            # No active features, use normal interval
            sleep_time = 60  # 60 seconds
            logger.info(f"⏰ {self.pair_group}: No active features - sleeping {sleep_time}s")
        
        return sleep_time
    
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
    def __init__(self, api_key, telegram_token, chat_id):
        self.api_key = api_key
        self.telegram_token = telegram_token
        self.chat_id = chat_id
        self.trading_systems = {}
        
        for pair_group, pair_config in TRADING_PAIRS.items():
            self.trading_systems[pair_group] = UltimateTradingSystem(pair_group, pair_config)
        
        logger.info(f"🎯 Initialized ULTIMATE trading manager with {len(self.trading_systems)} pair groups")
        

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
        """Run all trading systems with ultimate decision making"""
        logger.info("🎯 Starting ULTIMATE Multi-Pair Trading System...")
        test_proven_quarter_patch()
        api_key = os.getenv('OANDA_API_KEY')
        telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        
        while True:
            try:
                tasks = []
                sleep_times = []
                
                for pair_group, system in self.trading_systems.items():
                    task = asyncio.create_task(
                        system.run_ultimate_analysis(self.api_key),
                        name=f"ultimate_analysis_{pair_group}"
                    )
                    tasks.append(task)
                    
                    # Get sleep time for the fastest cycle (usually '90min')
                    sleep_time = system.get_sleep_time_for_cycle('90min')  # or whichever cycle type you want
                    sleep_times.append(sleep_time)
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                signals = []
                for i, result in enumerate(results):
                    pair_group = list(self.trading_systems.keys())[i]
                    if isinstance(result, Exception):
                        logger.error(f"❌ Ultimate analysis task failed for {pair_group}: {str(result)}")
                    elif result is not None:
                        signals.append(result)
                        logger.info(f"🎯 ULTIMATE SIGNAL FOUND for {pair_group}")
                
                if signals:
                    await self._process_ultimate_signals(signals)
                
                sleep_time = min(sleep_times) if sleep_times else BASE_INTERVAL
                logger.info(f"⏰ Ultimate cycle complete. Sleeping for {sleep_time:.1f} seconds")
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"❌ Error in ultimate main loop: {str(e)}")
                await asyncio.sleep(60)
    
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
# MAIN EXECUTION
# ================================

async def main():
    """Main entry point"""
    logger.info("🛡️ Starting ULTIMATE Multi-Pair SMT Trading System")
    
    api_key = os.getenv('OANDA_API_KEY')
    telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
    telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if not all([api_key, telegram_token, telegram_chat_id]):
        logger.error("❌ Missing required environment variables")
        logger.info("💡 Please set OANDA_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID")
        return
    
    try:
        manager = UltimateTradingManager(api_key, telegram_token, telegram_chat_id)
        await manager.run_ultimate_systems()
        
    except KeyboardInterrupt:
        logger.info("🛑 System stopped by user")
    except Exception as e:
        logger.error(f"💥 Fatal error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
