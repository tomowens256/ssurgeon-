#!/usr/bin/env python3
"""
ROBUST SMT TRADING SYSTEM - CRT+PSP+SMT CONFLUENCE VERSION
- Detects CRT + PSP + SMT triple confluence
- ALL possible quarter pairs
- 3-candle tolerance for swing alignment
- Enhanced timing validation
"""

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
    },
    'jpy_pairs': {
        'pair1': 'EUR_JPY',
        'pair2': 'GBP_JPY',
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

def fetch_candles(instrument, timeframe, count=100, api_key=None):
    """Fetch candles from OANDA API - ENFORCE UTC-4"""
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
        "alignmentTimezone": "America/New_York",  # UTC-4
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
                    parsed_time = parse_oanda_time(candle['time'])  # Already UTC-4
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
# ENHANCED TIMING MANAGER
# ================================

class RobustTimingManager:
    """Enhanced timing manager with STRONG duplicate prevention"""
    
    def __init__(self):
        self.ny_tz = pytz.timezone('America/New_York')  # UTC-4
        self.sent_signals = {}  # Track sent signals to prevent duplicates
        
    def is_duplicate_signal(self, signal_key, pair_group, cooldown_minutes=60):
        """STRONG duplicate prevention - 60 minute cooldown"""
        current_time = datetime.now(self.ny_tz)
        
        if pair_group not in self.sent_signals:
            self.sent_signals[pair_group] = {}
            
        # Check for exact matches first
        if signal_key in self.sent_signals[pair_group]:
            last_sent = self.sent_signals[pair_group][signal_key]
            time_diff = (current_time - last_sent).total_seconds() / 60
            if time_diff < cooldown_minutes:
                logger.info(f"⏳ STRONG DUPLICATE PREVENTION: {signal_key} (sent {time_diff:.1f} min ago)")
                return True
        
        # Check for similar signals (same direction and cycles)
        for existing_key, last_sent in list(self.sent_signals[pair_group].keys()):
            time_diff = (current_time - last_sent).total_seconds() / 60
            
            if self._signals_are_very_similar(signal_key, existing_key) and time_diff < cooldown_minutes:
                logger.info(f"⏳ SIMILAR SIGNAL BLOCKED: {signal_key} similar to {existing_key} (sent {time_diff:.1f} min ago)")
                return True
                
        self.sent_signals[pair_group][signal_key] = current_time
        
        # Clean old entries
        self._clean_old_entries()
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

class RobustQuarterManager:
    """Enhanced quarter manager with ALL possible quarter pairs"""
    
    def __init__(self):
        self.ny_tz = pytz.timezone('America/New_York')  # UTC-4
        
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

    def get_adjacent_quarter_pairs(self, cycle_type):
        """Get ONLY chronologically valid quarter pairs"""
        if cycle_type == 'weekly':
            # For weekly: q1→q2→q3→q4→q_less (NO circular q_less→q1)
            quarter_sequence = ['q1', 'q2', 'q3', 'q4', 'q_less']
        else:
            # For other cycles: q1→q2→q3→q4 (NO circular q4→q1)
            quarter_sequence = ['q1', 'q2', 'q3', 'q4']
        
        # ONLY consecutive pairs, NO circular transitions
        all_pairs = []
        for i in range(len(quarter_sequence) - 1):
            current = quarter_sequence[i]
            next_q = quarter_sequence[i + 1]
            all_pairs.append((current, next_q))
        
        logger.debug(f"🔍 {cycle_type}: Valid quarter pairs: {all_pairs}")
        return all_pairs


    def get_current_quarter(self, cycle_type):
        """
        Determine the current quarter based on cycle type and current time.
        """
    
        now = datetime.now(self.timezone)
    
        # Daily cycle → quarters of the day
        if cycle_type == "daily":
            hour = now.hour
            if 0 <= hour < 6:
                return "Q1"
            elif 6 <= hour < 12:
                return "Q2"
            elif 12 <= hour < 18:
                return "Q3"
            else:
                return "Q4"
    
        # 90min cycle → use mod-6 logic (6 periods in 9 hours)
        if cycle_type == "90min":
            minute_block = (now.hour * 60 + now.minute) // 90
            quarter_index = minute_block % 4
            return ["Q1", "Q2", "Q3", "Q4"][quarter_index]
    
        # Weekly cycle → divide the week
        if cycle_type == "weekly":
            weekday = now.weekday()  # 0 Monday → 6 Sunday
            if weekday < 2:
                return "Q1"
            elif weekday < 4:
                return "Q2"
            elif weekday == 4:
                return "Q3"
            else:
                return "Q4"
    
        # Monthly cycle → divide month into 4 parts
        if cycle_type == "monthly":
            day = now.day
            days_in_month = monthrange(now.year, now.month)[1]
            quarter_size = days_in_month // 4
    
            if day <= quarter_size:
                return "Q1"
            elif day <= quarter_size * 2:
                return "Q2"
            elif day <= quarter_size * 3:
                return "Q3"
            else:
                return "Q4"
    
        # fallback
        return "Q1"


    def get_last_three_quarters(self, cycle_type):
        """Get last three quarters - PROPERLY HANDLES q_less"""
        # Use the appropriate quarter sequence based on cycle type
        if cycle_type == 'weekly':
            # For weekly, include q_less in the sequence
            order = ['q1', 'q2', 'q3', 'q4', 'q_less']
        else:
            order = ['q1', 'q2', 'q3', 'q4']
        
        current_q = self.get_current_quarter(cycle_type)
        
        logger.debug(f"🔍 {cycle_type}: Current quarter = '{current_q}'")
        
        # Ensure current_q is in order list
        if current_q not in order:
            logger.error(f"❌ {cycle_type}: Invalid quarter '{current_q}'")
            # Fallback to a safe default
            return ['q2', 'q3', 'q4']
        
        try:
            idx = order.index(current_q)
            last_three = [
                order[idx],
                order[(idx - 1) % len(order)],
                order[(idx - 2) % len(order)]
            ]
            logger.debug(f"🔍 {cycle_type}: Last three quarters = {last_three}")
            return last_three
            
        except ValueError as e:
            logger.error(f"❌ {cycle_type}: Error in get_last_three_quarters: {e}")
            return ['q2', 'q3', 'q4']  # Emergency fallback


    
    def _get_weekly_quarter(self, timestamp):
        weekday = timestamp.weekday()
        if weekday == 0: return 'q1'
        elif weekday == 1: return 'q2'
        elif weekday == 2: return 'q3'
        elif weekday == 3: return 'q4'
        else: return 'q_less'
    
    def _get_daily_quarter(self, timestamp):
        """FIXED: Daily quarter that handles multi-day data properly"""
        hour = timestamp.hour
        
        # Since it's Saturday morning, adjust for the trading week
        # For daily quarters, we need to handle the fact that data spans multiple days
        
        # Simple fixed quarters based on hour only
        if 0 <= hour < 6: 
            return 'q1'
        elif 6 <= hour < 12: 
            return 'q2'
        elif 12 <= hour < 18: 
            return 'q3'
        else: 
            return 'q4'
    
    def _get_90min_quarter_fixed(self, timestamp):
        """FIXED: 90min quarters within the 18:00-start day"""
        daily_quarter = self._get_daily_quarter(timestamp)
        hour = timestamp.hour
        minute = timestamp.minute
        total_minutes = hour * 60 + minute
        
        # Adjust for 18:00 start - map hours to the correct 6-hour block
        adjusted_hour = (hour + 6) % 24  # Shift so 18:00 becomes 00:00 for calculation
        adjusted_total_minutes = adjusted_hour * 60 + minute
        
        # 90min quarters within each 6-hour daily quarter
        boundaries = {
            'q1': [  # 18:00-00:00 (adjusted: 00:00-06:00)
                (0*60, 1*60+30, 'q1'),
                (1*60+30, 3*60, 'q2'),
                (3*60, 4*60+30, 'q3'), 
                (4*60+30, 6*60, 'q4')
            ],
            'q2': [  # 00:00-06:00 (adjusted: 06:00-12:00)
                (6*60, 7*60+30, 'q1'),
                (7*60+30, 9*60, 'q2'),
                (9*60, 10*60+30, 'q3'),
                (10*60+30, 12*60, 'q4')
            ],
            'q3': [  # 06:00-12:00 (adjusted: 12:00-18:00)
                (12*60, 13*60+30, 'q1'),
                (13*60+30, 15*60, 'q2'),
                (15*60, 16*60+30, 'q3'),
                (16*60+30, 18*60, 'q4')
            ],
            'q4': [  # 12:00-18:00 (adjusted: 18:00-24:00)
                (18*60, 19*60+30, 'q1'),
                (19*60+30, 21*60, 'q2'),
                (21*60, 22*60+30, 'q3'),
                (22*60+30, 24*60, 'q4')
            ]
        }
        
        for start_min, end_min, quarter in boundaries[daily_quarter]:
            if start_min <= adjusted_total_minutes < end_min:
                return quarter
        
        return 'q_less'

    def test_18hr_quarter_system(self):
        """Test that the 18:00-start quarter system works correctly"""
        print("\n🧪 TESTING 18:00-START QUARTER SYSTEM:")
        
        test_times = [
            # q1: 18:00-23:59
            datetime(2025, 11, 20, 18, 0),   # Should be q1
            datetime(2025, 11, 20, 22, 0),   # Should be q1
            # q2: 00:00-05:59  
            datetime(2025, 11, 21, 0, 0),    # Should be q2
            datetime(2025, 11, 21, 4, 0),    # Should be q2
            # q3: 06:00-11:59
            datetime(2025, 11, 21, 6, 0),    # Should be q3
            datetime(2025, 11, 21, 10, 0),   # Should be q3
            # q4: 12:00-17:59
            datetime(2025, 11, 21, 12, 0),   # Should be q4
            datetime(2025, 11, 21, 16, 0),   # Should be q4
        ]
        
        for test_time in test_times:
            daily_quarter = self._get_daily_quarter(test_time)
            print(f"   {test_time.strftime('%m-%d %H:%M')} → Daily: {daily_quarter}")
    
    def get_all_possible_quarter_pairs(self, cycle_type):
        """Get ALL possible consecutive quarter pairs regardless of current quarter"""
        if cycle_type == 'weekly':
            # For weekly, include transitions to/from q_less
            quarter_sequence = ['q1', 'q2', 'q3', 'q4', 'q_less']
        else:
            quarter_sequence = ['q1', 'q2', 'q3', 'q4']
        
        # ALL possible consecutive pairs
        all_pairs = []
        for i in range(len(quarter_sequence)):
            current = quarter_sequence[i]
            next_q = quarter_sequence[(i + 1) % len(quarter_sequence)]
            all_pairs.append((current, next_q))
        
        logger.debug(f"🔍 {cycle_type}: Checking ALL quarter pairs: {all_pairs}")
        return all_pairs
    
    def group_candles_by_quarters(self, df, cycle_type, num_quarters=4):
        """Group candles into exact quarters with STRICT chronological validation"""
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            logger.warning(f"⚠️ No data to group for {cycle_type}")
            return {}
        
        # Sort by time first - CRITICAL
        df = df.sort_values('time').reset_index(drop=True)
        
        quarters_data = {}
        
        for _, candle in df.iterrows():
            candle_time = candle['time']
            quarter = self._get_candle_quarter(candle_time, cycle_type)
            
            if quarter not in quarters_data:
                quarters_data[quarter] = []
            quarters_data[quarter].append(candle)
        
        # Convert to DataFrames and ensure chronological order
        for quarter in quarters_data:
            quarters_data[quarter] = pd.DataFrame(quarters_data[quarter])
            quarters_data[quarter] = quarters_data[quarter].sort_values('time')
            
            # Validate quarter data chronology
            if not quarters_data[quarter].empty:
                times = quarters_data[quarter]['time']
                time_range = f"{times.min().strftime('%m-%d %H:%M')} to {times.max().strftime('%m-%d %H:%M')}"
                logger.debug(f"📊 {cycle_type} {quarter}: {len(quarters_data[quarter])} candles, {time_range}")
                
                # Check if quarter data is chronological
                if not times.is_monotonic_increasing:
                    logger.warning(f"⚠️ {cycle_type} {quarter}: Data is not chronological!")
                    # Force chronological order
                    quarters_data[quarter] = quarters_data[quarter].sort_values('time').reset_index(drop=True)
        
        # Remove quarters with insufficient data
        quarters_to_remove = []
        for quarter in quarters_data:
            if len(quarters_data[quarter]) < 5:  # Minimum 5 candles for swing detection
                quarters_to_remove.append(quarter)
                logger.debug(f"📊 {cycle_type} {quarter}: Removing - only {len(quarters_data[quarter])} candles")
        
        for quarter in quarters_to_remove:
            del quarters_data[quarter]
        
        valid_quarters = [q for q in quarters_data if not quarters_data[q].empty]
        logger.debug(f"📊 {cycle_type}: Valid quarters with sufficient data: {valid_quarters}")
        
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
    def get_current_quarter(self, cycle_type, timestamp=None):
        """Return current quarter using the existing quarter detection logic."""
        quarters = self.detect_current_quarters(timestamp)
        return quarters.get(cycle_type, None)

    def validate_quarter_sequence(self, cycle_type, asset_quarters):
        """Validate that quarters are in proper sequence"""
        print(f"\n🔍 VALIDATING {cycle_type} QUARTER SEQUENCE:")
        
        # Define expected sequence
        if cycle_type == 'weekly':
            expected_sequence = ['q1', 'q2', 'q3', 'q4', 'q_less']
        else:
            expected_sequence = ['q1', 'q2', 'q3', 'q4']
        
        # Get quarters that actually have data
        available_quarters = [q for q in expected_sequence if q in asset_quarters and not asset_quarters[q].empty]
        
        print(f"   Expected: {expected_sequence}")
        print(f"   Available: {available_quarters}")
        
        # Check if available quarters are in expected order
        for i in range(len(available_quarters) - 1):
            current_q = available_quarters[i]
            next_q = available_quarters[i + 1]
            
            current_idx = expected_sequence.index(current_q)
            next_idx = expected_sequence.index(next_q)
            
            if next_idx != current_idx + 1:
                print(f"   ❌ SEQUENCE BREAK: {current_q}→{next_q} (expected {expected_sequence[current_idx]}→{expected_sequence[current_idx+1]})")
            else:
                print(f"   ✅ Sequence OK: {current_q}→{next_q}")
        
        return available_quarters

    def adjust_for_weekend_data(self, cycle_type, asset_quarters):
        """Adjust quarter logic for weekend data"""
        print(f"\n🔍 ADJUSTING FOR WEEKEND DATA ({cycle_type}):")
        
        # On weekends, the data might span across the week boundary
        # We need to be more careful about quarter transitions
        
        adjusted_quarters = {}
        
        for quarter, data in asset_quarters.items():
            if not data.empty:
                times = data['time']
                day_range = f"{times.min().strftime('%a %H:%M')} to {times.max().strftime('%a %H:%M')}"
                print(f"   {quarter}: {day_range} ({len(data)} candles)")
                
                # On weekends, we might want to exclude certain quarter transitions
                if cycle_type == 'daily':
                    # For daily cycles on weekend, be more restrictive
                    min_date = times.min().date()
                    max_date = times.max().date()
                    if max_date > min_date:
                        print(f"   ⚠️ {quarter} spans multiple days: {min_date} to {max_date}")
                
                adjusted_quarters[quarter] = data
        
        return adjusted_quarters

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
    
            # DEBUG: Check if we have proper time data
            if pd.isna(curr['time']) or curr['time'] is None:
                continue
    
            # swing high
            if curr['high'] > prev['high'] and curr['high'] > nxt['high']:
                if not swing_highs or (i - swing_highs[-1]['index']) >= MIN_SWING_DISTANCE:
                    swing_highs.append({
                        'time': curr['time'],  # Use actual candle time
                        'price': float(curr['high']),
                        'index': i
                    })
    
            # swing low
            if curr['low'] < prev['low'] and curr['low'] < nxt['low']:
                if not swing_lows or (i - swing_lows[-1]['index']) >= MIN_SWING_DISTANCE:
                    swing_lows.append({
                        'time': curr['time'],  # Use actual candle time
                        'price': float(curr['low']),
                        'index': i
                    })
    
        return swing_highs, swing_lows


    
    @staticmethod
    def find_aligned_swings(asset1_swings, asset2_swings, max_candle_diff=3, timeframe_minutes=5):
        
        """Find swings that occur within 3 CANDLES of each other"""
        # 🔥 FIX 1: enforce chronological order
        asset1_swings = sorted(asset1_swings, key=lambda x: x['time'])
        asset2_swings = sorted(asset2_swings, key=lambda x: x['time'])
        aligned_pairs = []
        
        # Calculate time tolerance based on timeframe and max_candle_diff
        max_time_diff_minutes = max_candle_diff * timeframe_minutes
        
        for swing1 in asset1_swings:
            for swing2 in asset2_swings:
                time_diff = abs((swing1['time'] - swing2['time']).total_seconds() / 60)
                if time_diff <= max_time_diff_minutes:
                    aligned_pairs.append((swing1, swing2, time_diff))
        
        # Sort by time difference (closest first)
        aligned_pairs.sort(key=lambda x: x[2])
        #logger.debug(f"🕒 Found {len(aligned_pairs)} aligned swings within {max_time_diff_minutes} minutes")
        return aligned_pairs
    
    @staticmethod
    def format_swing_time_description(prev_swing, curr_swing, swing_type="low", timing_manager=None):
        """Create time-based description for swing formation with chronological validation"""
        if not prev_swing or not curr_swing:
            return "insufficient swing data"
        
        prev_time = prev_swing['time']
        curr_time = curr_swing['time']
        
        # Validate chronological order
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
    def validate_interim_price_action(df, first_swing, second_swing, direction="bearish"):
        if df is None or first_swing is None or second_swing is None:
            return False
    
        first_time = first_swing['time']
        second_time = second_swing['time']
    
        # FIX 1: ensure correct ordering (first_swing must be before second_swing)
        if first_time >= second_time:
            logger.warning("⚠️ Swing times out of order — swapping them.")
            first_swing, second_swing = second_swing, first_swing
            first_time, second_time = second_time, first_time
    
        # Interim candles
        interim_candles = df[(df['time'] > first_time) & (df['time'] < second_time)]
    
        if interim_candles.empty:
            #logger.debug("✅ No interim candles to validate")
            return True
    
        if direction == "bearish":
    
            protected_high = max(
                float(first_swing['price']),
                float(second_swing['price'])
            )
    
            max_interim_high = float(interim_candles['high'].max())
    
            if max_interim_high > protected_high:
                #logger.warning(
                    #f"❌ INTERIM PRICE INVALIDATION (Bearish): Interim high {max_interim_high:.4f} > protected high {protected_high:.4f}"
                #)
                return False
    
            #logger.info(
                #f"✅ Valid interim price action (Bearish): Max interim high {max_interim_high:.4f} <= protected high {protected_high:.4f}"
            #)
            return True
    
        else:  # bullish
    
            protected_low = min(
                float(first_swing['price']),
                float(second_swing['price'])
            )
    
            min_interim_low = float(interim_candles['low'].min())
    
            if min_interim_low < protected_low:
                #logger.warning(
                    #f"❌ INTERIM PRICE INVALIDATION (Bullish): Interim low {min_interim_low:.4f} < protected low {protected_low:.4f}"
                #)
                return False
    
            #logger.info(
                #f"✅ Valid interim price action (Bullish): Min interim low {min_interim_low:.4f} >= protected low {protected_low:.4f}"
            #)
            return True



# ================================
# ENHANCED CRT DETECTOR WITH PSP TRACKING
# ================================

class RobustCRTDetector:
    """Enhanced CRT detector with PSP tracking for triple confluence"""
    
    def __init__(self, timing_manager):
        self.timing_manager = timing_manager
        self.psp_cache = {}  # Cache PSP signals by timeframe
    
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

class UltimateSMTDetector:
    """Ultimate SMT detector checking ALL quarter pairs"""
    
    def __init__(self, pair_config, timing_manager):
        self.smt_history = []
        self.quarter_manager = RobustQuarterManager()
        self.swing_detector = UltimateSwingDetector()
        self.timing_manager = timing_manager
        self.signal_counts = {}
        self.invalidated_smts = set()
        self.pair_config = pair_config
        self.last_smt_candle = None

        
        # Timeframe to minutes mapping for tolerance calculation
        self.timeframe_minutes = {
            'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30,
            'H1': 60, 'H2': 120, 'H3': 180, 'H4': 240,
            'H6': 360, 'H8': 480, 'H12': 720
        }
        
        # PSP tracking for each SMT
        self.smt_psp_tracking = {}
        
    def detect_smt_all_cycles(self, asset1_data, asset2_data, cycle_type):
        """Detect SMT using ONLY valid chronological quarter pairs"""
        try:
            logger.info(f"🔍 Scanning {cycle_type} for SMT signals...")
    
            if not self.check_data_quality(asset1_data, asset2_data, cycle_type):
                return None
    
            if (asset1_data is None or not isinstance(asset1_data, pd.DataFrame) or asset1_data.empty or
                asset2_data is None or not isinstance(asset2_data, pd.DataFrame) or asset2_data.empty):
                logger.warning(f"⚠️ No data for {cycle_type} SMT detection")
                return None
    
            # Get adjacent quarter pairs
            adjacent_pairs = self.quarter_manager.get_adjacent_quarter_pairs(cycle_type)
    
            # Get last 3 quarters 
            last_3_quarters = self.quarter_manager.get_last_three_quarters(cycle_type)
    
            logger.debug(f"🔍 {cycle_type}: Last 3 quarters: {last_3_quarters}")
            logger.debug(f"🔍 {cycle_type}: Adjacent pairs: {adjacent_pairs}")
    
            asset1_quarters = self.quarter_manager.group_candles_by_quarters(asset1_data, cycle_type)
            asset2_quarters = self.quarter_manager.group_candles_by_quarters(asset2_data, cycle_type)
    
            if not asset1_quarters or not asset2_quarters:
                logger.warning(f"⚠️ No quarter grouped data for {cycle_type}")
                return None
    
            # === FILTER OUT INVALID/OVERLAPPING PAIRS ===
            valid_pairs = self.filter_valid_quarter_pairs(cycle_type, asset1_quarters, asset2_quarters, adjacent_pairs)
            
            if not valid_pairs:
                logger.warning(f"⚠️ No valid quarter pairs for {cycle_type}")
                return None
    
            results = []
    
            # Scan ONLY valid chronological pairs
            for prev_q, curr_q in valid_pairs:
                smt_result = self._compare_quarters_with_3_candle_tolerance(
                    asset1_quarters[prev_q], asset1_quarters[curr_q],
                    asset2_quarters[prev_q], asset2_quarters[curr_q],
                    cycle_type, prev_q, curr_q
                )
    
                if smt_result:
                    results.append(smt_result)
    
            # ... rest of your existing method ...
    
            if not results:
                logger.debug(f"🔍 No SMT found for {cycle_type}")
                return None
    
            # --- PROCESS RESULTS: Find first valid non-duplicate SMT ---
            for smt_result in results:
    
                # DUPLICATE PROTECTION
                if self._is_duplicate_signal(smt_result):
                    continue
    
                signal_key = smt_result['signal_key']
                candle_time = smt_result['candle_time']
    
                # --- UPDATE SMT STATE (VERY IMPORTANT) ---
                self.last_smt_candle = candle_time
                self.signal_counts[signal_key] = self.signal_counts.get(signal_key, 0) + 1
    
                # --- INITIALIZE PSP TRACKING ---
                if signal_key not in self.smt_psp_tracking:
                    self.smt_psp_tracking[signal_key] = {
                        'psp_found': False,
                        'check_count': 0,
                        'max_checks': 15,
                        'last_check': datetime.now(NY_TZ),
                        'formation_time': smt_result['formation_time']
                    }

                
    
                logger.info(
                    f"🎯 SMT DETECTED: {cycle_type} "
                    f"{smt_result['prev_q']}→{smt_result['curr_q']} "
                    f"{smt_result['direction']}"
                )
    
                return smt_result
    
            # If all SMTs were duplicates
            return None
    
        except Exception as e:
            logger.error(f"❌ Error in SMT detection for {cycle_type}: {str(e)}")
            return None

    def check_data_quality(self, pair1_data, pair2_data, cycle_type):
        """Check if we have good quality data for analysis"""
        print(f"\n🔍 DATA QUALITY CHECK for {cycle_type}:")
        
        if pair1_data is None or pair2_data is None:
            print(f"   ❌ Missing data")
            return False
        
        # Check data length
        if len(pair1_data) < 20 or len(pair2_data) < 20:
            print(f"   ❌ Insufficient data: {len(pair1_data)} vs {len(pair2_data)} candles")
            return False
        
        # Check time range coverage
        p1_start = pair1_data['time'].min()
        p1_end = pair1_data['time'].max()  
        p2_start = pair2_data['time'].min()
        p2_end = pair2_data['time'].max()
        
        time_coverage = (min(p1_end, p2_end) - max(p1_start, p2_start)).total_seconds() / 3600
        print(f"   Time overlap: {time_coverage:.1f}h")
        
        if time_coverage < 1:
            print(f"   ❌ Insufficient time overlap")
            return False
        
        print(f"   ✅ Good data quality")
        return True

    def debug_quarter_validation(self, prev_q, curr_q, asset1_prev, asset1_curr, asset2_prev, asset2_curr):
        """Validate quarter chronology for 18:00-start system"""
        print(f"\n🔍 VALIDATING {prev_q}→{curr_q} (18:00-start system):")
        
        if not asset1_prev.empty and not asset1_curr.empty:
            prev_end = asset1_prev['time'].max()
            curr_start = asset1_curr['time'].min()
            time_gap = (curr_start - prev_end).total_seconds() / 3600
            
            # Special handling for q4→q1 transition (crosses day boundary)
            if prev_q == 'q4' and curr_q == 'q1':
                # q4 ends at 17:59, q1 starts at 18:00 (could be same day or next day)
                expected_gap = 0.02  # ~1 minute gap is acceptable
                if -1 <= time_gap <= 1:  # Allow small gaps around the boundary
                    print(f"   ✅ Q4→Q1 TRANSITION: {prev_end.strftime('%m-%d %H:%M')} → {curr_start.strftime('%m-%d %H:%M')} ({time_gap:+.1f}h)")
                else:
                    print(f"   ⚠️ UNUSUAL Q4→Q1 GAP: {prev_end.strftime('%m-%d %H:%M')} → {curr_start.strftime('%m-%d %H:%M')} ({time_gap:+.1f}h)")
            elif time_gap < -1:
                print(f"   ❌ REVERSED TIME: Current quarter starts BEFORE previous quarter! ({time_gap:+.1f}h)")
            elif time_gap > 6:
                print(f"   ⚠️ LARGE GAP: {time_gap:.1f}h between quarters")
            else:
                print(f"   ✅ Reasonable gap: {time_gap:.1f}h")

    def filter_valid_quarter_pairs(self, cycle_type, asset1_quarters, asset2_quarters, adjacent_pairs):
        """Filter out quarter pairs that have overlapping time ranges"""
        print(f"\n🔍 FILTERING VALID QUARTER PAIRS for {cycle_type}:")
        
        valid_pairs = []
        
        for prev_q, curr_q in adjacent_pairs:
            # Check if both quarters exist and have data
            if (prev_q not in asset1_quarters or curr_q not in asset1_quarters or
                prev_q not in asset2_quarters or curr_q not in asset2_quarters):
                continue
                
            asset1_prev = asset1_quarters[prev_q]
            asset1_curr = asset1_quarters[curr_q]
            asset2_prev = asset2_quarters[prev_q]
            asset2_curr = asset2_quarters[curr_q]
            
            if asset1_prev.empty or asset1_curr.empty or asset2_prev.empty or asset2_curr.empty:
                continue
            
            # STRICT: Check that current quarter starts AFTER previous quarter ends
            prev_end = max(asset1_prev['time'].max(), asset2_prev['time'].max())
            curr_start = min(asset1_curr['time'].min(), asset2_curr['time'].min())
            
            time_gap = (curr_start - prev_end).total_seconds() / 3600
            
            if time_gap > 0:
                print(f"   ✅ {prev_q}→{curr_q}: Valid gap {time_gap:.1f}h")
                valid_pairs.append((prev_q, curr_q))
            else:
                print(f"   ❌ {prev_q}→{curr_q}: INVALID (overlap {-time_gap:.1f}h)")
        
        print(f"   Final valid pairs: {valid_pairs}")
        return valid_pairs

    def debug_swing_data_quality(self, swings, label):
        """Debug the quality of swing data"""
        print(f"\n🔍 SWING DATA QUALITY - {label}:")
        
        if not swings:
            print(f"   No swings found")
            return
        
        for i, swing in enumerate(swings):
            print(f"   Swing {i}: time={swing['time'].strftime('%m-%d %H:%M')}, price={swing['price']:.4f}")
            
            # Check for invalid timestamps
            if swing['time'].year < 2020:
                print(f"   ❌ INVALID TIMESTAMP: {swing['time']}")

    def run_comprehensive_debug(self, cycle_type, market_data_pair1, market_data_pair2):
        """Run complete debug for a cycle - FIXED VERSION"""
        print(f"\n🎯 COMPREHENSIVE DEBUG FOR {cycle_type.upper()}")
        
        timeframe = self.pair_config['timeframe_mapping'][cycle_type]
        
        # Get data from passed market_data dictionaries
        pair1_data = market_data_pair1.get(timeframe)
        pair2_data = market_data_pair2.get(timeframe)
        
        if not self.check_data_quality(pair1_data, pair2_data, cycle_type):
            return
        
        # Test quarter grouping
        asset1_quarters = self.quarter_manager.group_candles_by_quarters(pair1_data, cycle_type)
        asset2_quarters = self.quarter_manager.group_candles_by_quarters(pair2_data, cycle_type)
        
        self.debug_quarter_time_ranges(cycle_type, asset1_quarters, asset2_quarters)
        
        # Test swing detection on sample quarter
        test_quarter = 'q1'
        if test_quarter in asset1_quarters:
            test_data = asset1_quarters[test_quarter]
            swings_high, swings_low = self.swing_detector.find_swing_highs_lows(test_data)
            self.debug_swing_data_quality(swings_high, f"{cycle_type} {test_quarter} Highs")
            self.debug_swing_data_quality(swings_low, f"{cycle_type} {test_quarter} Lows")

    def debug_quarter_time_ranges(self, cycle_type, asset1_quarters, asset2_quarters):
        """Debug the actual time ranges of quarters with sequence validation"""
        print(f"\n🔍 DEBUG QUARTER TIME RANGES for {cycle_type}:")
        
        for quarter in ['q1', 'q2', 'q3', 'q4', 'q_less']:
            if quarter in asset1_quarters and not asset1_quarters[quarter].empty:
                a1_times = asset1_quarters[quarter]['time']
                print(f"   {quarter}: Asset1 → {a1_times.min().strftime('%m-%d %H:%M')} to {a1_times.max().strftime('%m-%d %H:%M')} ({len(a1_times)} candles)")
            
            if quarter in asset2_quarters and not asset2_quarters[quarter].empty:
                a2_times = asset2_quarters[quarter]['time']
                print(f"   {quarter}: Asset2 → {a2_times.min().strftime('%m-%d %H:%M')} to {a2_times.max().strftime('%m-%d %H:%M')} ({len(a2_times)} candles)")
        
        # Validate sequence - THIS MUST BE INSIDE THE METHOD
        print(f"\n🔍 ASSET1 QUARTER SEQUENCE:")
        asset1_sequence = self.validate_quarter_sequence(cycle_type, asset1_quarters)
        
        if asset2_quarters:
            print(f"\n🔍 ASSET2 QUARTER SEQUENCE:")
            asset2_sequence = self.validate_quarter_sequence(cycle_type, asset2_quarters)

    def validate_quarter_sequence(self, cycle_type, asset_quarters):
        """Validate that quarters are in proper sequence"""
        print(f"   VALIDATING {cycle_type} QUARTER SEQUENCE:")
        
        # Define expected sequence
        if cycle_type == 'weekly':
            expected_sequence = ['q1', 'q2', 'q3', 'q4', 'q_less']
        else:
            expected_sequence = ['q1', 'q2', 'q3', 'q4']
        
        # Get quarters that actually have data
        available_quarters = [q for q in expected_sequence if q in asset_quarters and not asset_quarters[q].empty]
        
        print(f"      Expected: {expected_sequence}")
        print(f"      Available: {available_quarters}")
        
        # Check if available quarters are in expected order
        for i in range(len(available_quarters) - 1):
            current_q = available_quarters[i]
            next_q = available_quarters[i + 1]
            
            current_idx = expected_sequence.index(current_q)
            next_idx = expected_sequence.index(next_q)
            
            if next_idx != current_idx + 1:
                print(f"      ❌ SEQUENCE BREAK: {current_q}→{next_q} (expected {expected_sequence[current_idx]}→{expected_sequence[current_idx+1]})")
            else:
                print(f"      ✅ Sequence OK: {current_q}→{next_q}")
        
        return available_quarters

    
    def _compare_quarters_with_3_candle_tolerance(self, asset1_prev, asset1_curr, asset2_prev, asset2_curr, cycle_type, prev_q, curr_q):
        """Compare quarters with debug info"""
        try:
            print(f"\n🔍 COMPARING QUARTERS: {cycle_type} {prev_q}→{curr_q}")
            
            # Debug the input data
            print(f"   Asset1 prev: {len(asset1_prev)} candles, time range: {asset1_prev['time'].min() if not asset1_prev.empty else 'empty'} to {asset1_prev['time'].max() if not asset1_prev.empty else 'empty'}")
            print(f"   Asset1 curr: {len(asset1_curr)} candles, time range: {asset1_curr['time'].min() if not asset1_curr.empty else 'empty'} to {asset1_curr['time'].max() if not asset1_curr.empty else 'empty'}")

            # === ADD STRICT CHRONOLOGY CHECK ===
            if not asset1_prev.empty and not asset1_curr.empty:
                prev_end = asset1_prev['time'].max()
                curr_start = asset1_curr['time'].min()
                
                # STRICT: Current quarter MUST start AFTER previous quarter ends
                if curr_start <= prev_end:
                    print(f"   ❌ CHRONOLOGY ERROR: Skipping {prev_q}→{curr_q} - current starts at {curr_start.strftime('%m-%d %H:%M')} (BEFORE previous ends at {prev_end.strftime('%m-%d %H:%M')})")
                    return None
                
                time_gap = (curr_start - prev_end).total_seconds() / 3600
                if time_gap > 24:
                    print(f"   ⚠️ LARGE GAP: {time_gap:.1f}h between quarters")
                else:
                    print(f"   ✅ Chronology OK: {time_gap:.1f}h gap")
    
            if (asset1_prev.empty or asset1_curr.empty or 
                asset2_prev.empty or asset2_curr.empty):
                print(f"   ⚠️ SKIPPING: Missing quarter data")
                return None
            # === ADD THIS VALIDATION CALL ===
            self.debug_quarter_validation(prev_q, curr_q, asset1_prev, asset1_curr, asset2_prev, asset2_curr)
            
            if (asset1_prev.empty or asset1_curr.empty or 
                asset2_prev.empty or asset2_curr.empty):
                return None
        
            # timeframe / tolerance
            timeframe = self.pair_config['timeframe_mapping'][cycle_type]
            timeframe_minutes = self.timeframe_minutes.get(timeframe, 5)
        
            # combined (sorted) frames for interim validations
            asset1_combined = pd.concat([asset1_prev, asset1_curr]).sort_values('time').reset_index(drop=True)
            asset2_combined = pd.concat([asset2_prev, asset2_curr]).sort_values('time').reset_index(drop=True)
        
            # --- find swings (original functions) ---
            a1_prev_H, a1_prev_L = self.swing_detector.find_swing_highs_lows(asset1_prev)
            a1_curr_H, a1_curr_L = self.swing_detector.find_swing_highs_lows(asset1_curr)
            a2_prev_H, a2_prev_L = self.swing_detector.find_swing_highs_lows(asset2_prev)
            a2_curr_H, a2_curr_L = self.swing_detector.find_swing_highs_lows(asset2_curr)
        
            # --- FIX: ensure all swing times are real pandas Timestamps ---
            def normalize_time(swings, tz=NY_TZ):
                for s in swings:
                    # ensure timestamp
                    if not isinstance(s['time'], pd.Timestamp):
                        s['time'] = pd.to_datetime(s['time'])
                    # make timezone aware if naive
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

            # --------------------------------------------------------------
        
            # helper: sort swings by time
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

    
            # --- FILTER: keep only swings that fall INSIDE their quarter timeframe ---
            def filter_by_quarter(swings, quarter_df):
                if not swings:
                    return []
                q_start = quarter_df['time'].min()
                q_end = quarter_df['time'].max()
                filtered = [s for s in swings if (s['time'] >= q_start and s['time'] <= q_end)]
                if not filtered:
                    # fallback: keep nearest by time if none fall inside (rare)
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
    
            logger.debug(f"🔍 {cycle_type} {prev_q}→{curr_q}: After filtering -> A1 prev H:{len(a1_prev_H)} L:{len(a1_prev_L)} | A1 curr H:{len(a1_curr_H)} L:{len(a1_curr_L)}")
            logger.debug(f"🔍 {cycle_type} {prev_q}→{curr_q}: After filtering -> A2 prev H:{len(a2_prev_H)} L:{len(a2_prev_L)} | A2 curr H:{len(a2_curr_H)} L:{len(a2_curr_L)}")
    
            # --- find bearish & bullish using your tolerance functions ---
            bearish_smt = self._find_bearish_smt_with_tolerance(
                a1_prev_H, a1_curr_H,
                a2_prev_H, a2_curr_H,
                asset1_combined, timeframe_minutes
            )
    
            bullish_smt = self._find_bullish_smt_with_tolerance(
                a1_prev_L, a1_curr_L,
                a2_prev_L, a2_curr_L,
                asset1_combined, timeframe_minutes
            )
    
            # No candidate
            if not bearish_smt and not bullish_smt:
                return None
    
            # Choose found result and unpack safely
            if bearish_smt:
                direction = 'bearish'
                smt_type = 'Higher Swing High'
                asset1_prev_high, asset1_curr_high, asset2_prev_high, asset2_curr_high = bearish_smt
    
                # Ensure chronological order for both assets; if reversed, swap
                if asset1_curr_high['time'] <= asset1_prev_high['time']:
                    logger.warning(f"⚠️ Fixing chronology A1: {asset1_prev_high['time']} -> {asset1_curr_high['time']}")
                    asset1_prev_high, asset1_curr_high = asset1_curr_high, asset1_prev_high
                if asset2_curr_high['time'] <= asset2_prev_high['time']:
                    logger.warning(f"⚠️ Fixing chronology A2: {asset2_prev_high['time']} -> {asset2_curr_high['time']}")
                    asset2_prev_high, asset2_curr_high = asset2_curr_high, asset2_prev_high
    
                formation_time = asset1_curr_high['time']
                asset1_action = self.swing_detector.format_swing_time_description(asset1_prev_high, asset1_curr_high, "high", self.timing_manager)
                asset2_action = self.swing_detector.format_swing_time_description(asset2_prev_high, asset2_curr_high, "high", self.timing_manager)
                critical_level = asset1_curr_high['price']
    
                # Extra sanity: ensure prev < curr across both assets (otherwise reject)
                if not (asset1_prev_high['time'] < asset1_curr_high['time'] and asset2_prev_high['time'] < asset2_curr_high['time']):
                    logger.warning("⚠️ Rejected bearish SMT because swings are not chronological across both assets")
                    return None
    
            else:  # bullish_smt
                direction = 'bullish'
                smt_type = 'Lower Swing Low'
                asset1_prev_low, asset1_curr_low, asset2_prev_low, asset2_curr_low = bullish_smt
    
                if asset1_curr_low['time'] <= asset1_prev_low['time']:
                    logger.warning(f"⚠️ Fixing chronology A1 low: {asset1_prev_low['time']} -> {asset1_curr_low['time']}")
                    asset1_prev_low, asset1_curr_low = asset1_curr_low, asset1_prev_low
                if asset2_curr_low['time'] <= asset2_prev_low['time']:
                    logger.warning(f"⚠️ Fixing chronology A2 low: {asset2_prev_low['time']} -> {asset2_curr_low['time']}")
                    asset2_prev_low, asset2_curr_low = asset2_curr_low, asset2_prev_low
    
                formation_time = asset1_curr_low['time']
                asset1_action = self.swing_detector.format_swing_time_description(asset1_prev_low, asset1_curr_low, "low", self.timing_manager)
                asset2_action = self.swing_detector.format_swing_time_description(asset2_prev_low, asset2_curr_low, "low", self.timing_manager)
                critical_level = asset1_curr_low['price']
    
                if not (asset1_prev_low['time'] < asset1_curr_low['time'] and asset2_prev_low['time'] < asset2_curr_low['time']):
                    logger.warning("⚠️ Rejected bullish SMT because swings are not chronological across both assets")
                    return None
    
            # Build signal_key time pieces from the actual chosen swings
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
    
            # Final sanity: ensure formation_time is inside curr quarter bounds
            curr_start = asset1_curr['time'].min()
            curr_end = asset1_curr['time'].max()
            if not (curr_start <= formation_time <= curr_end):
                logger.warning(f"⚠️ Formation time {formation_time} outside curr quarter bounds ({curr_start} → {curr_end}). Rejecting.")
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
    
            logger.info(f"🎯 SMT DETECTED with 3-candle tolerance: {direction} {cycle_type} {prev_q}→{curr_q}")
            logger.info(f"   Signal ID: {smt_data['signal_key']}")
            logger.info(f"   Asset1: {asset1_action}")
            logger.info(f"   Asset2: {asset2_action}")
    
            return smt_data
    
        except Exception as e:
            logger.error(f"Error comparing quarters {prev_q}→{curr_q}: {str(e)}\n{traceback.format_exc()}")
            return None

    
    def _find_bearish_smt_with_tolerance(self, asset1_prev_highs, asset1_curr_highs, asset2_prev_highs, asset2_curr_highs, asset1_combined_data, timeframe_minutes):
        """Find bearish SMT with 3-CANDLE TOLERANCE"""
        # Find aligned previous swings with tolerance
        aligned_prev_highs = self.swing_detector.find_aligned_swings(
            asset1_prev_highs, asset2_prev_highs, 
            max_candle_diff=3, timeframe_minutes=timeframe_minutes
        )
        
        # Find aligned current swings with tolerance
        aligned_curr_highs = self.swing_detector.find_aligned_swings(
            asset1_curr_highs, asset2_curr_highs,
            max_candle_diff=3, timeframe_minutes=timeframe_minutes
        )
        
        logger.debug(f"🔍 Bearish SMT: {len(aligned_prev_highs)} aligned prev highs, {len(aligned_curr_highs)} aligned curr highs")
        
        for prev_pair in aligned_prev_highs:
            asset1_prev, asset2_prev, prev_time_diff = prev_pair
            
            for curr_pair in aligned_curr_highs:
                asset1_curr, asset2_curr, curr_time_diff = curr_pair
                
                # Check SMT conditions - FIXED LOGIC FOR YOUR SCENARIO
                # Asset1 makes HIGHER high (price goes above and closes above)
                asset1_hh = asset1_curr['price'] > asset1_prev['price']
                
                # Asset2 makes LOWER high (price may go above but fails to close above, OR doesn't go above at all)
                # This matches your scenario: GBPJPY goes above but closes below
                asset2_lh = asset2_curr['price'] <= asset2_prev['price']  # Lower high
                
                # CRITICAL: Check interim price validation for bearish SMT
                interim_valid = self.swing_detector.validate_interim_price_action(
                    asset1_combined_data, asset1_prev, asset1_curr, "bearish"
                )
                
                if asset1_hh and asset2_lh and interim_valid:
                    logger.info(f"✅ BEARISH SMT FOUND with 3-candle tolerance:")
                    logger.info(f"   Prev swings: {asset1_prev['time'].strftime('%H:%M')} & {asset2_prev['time'].strftime('%H:%M')} (diff: {prev_time_diff:.1f}min)")
                    logger.info(f"   Curr swings: {asset1_curr['time'].strftime('%H:%M')} & {asset2_curr['time'].strftime('%H:%M')} (diff: {curr_time_diff:.1f}min)")
                    logger.info(f"   Asset1: Higher High ({asset1_prev['price']:.4f} → {asset1_curr['price']:.4f})")
                    logger.info(f"   Asset2: Lower High ({asset2_prev['price']:.4f} → {asset2_curr['price']:.4f})")
                    logger.info(f"   Interim validation: ✅ PASSED")
                    return (asset1_prev, asset1_curr, asset2_prev, asset2_curr)
                elif asset1_hh and asset2_lh and not interim_valid:
                    logger.warning(f"❌ BEARISH SMT REJECTED - Interim price invalid")
        
        return None
    
    def _find_bullish_smt_with_tolerance(self, asset1_prev_lows, asset1_curr_lows, asset2_prev_lows, asset2_curr_lows, asset1_combined_data, timeframe_minutes):
        """Find bullish SMT with 3-CANDLE TOLERANCE"""
        # Find aligned previous swings with tolerance
        aligned_prev_lows = self.swing_detector.find_aligned_swings(
            asset1_prev_lows, asset2_prev_lows,
            max_candle_diff=3, timeframe_minutes=timeframe_minutes
        )
        
        # Find aligned current swings with tolerance
        aligned_curr_lows = self.swing_detector.find_aligned_swings(
            asset1_curr_lows, asset2_curr_lows,
            max_candle_diff=3, timeframe_minutes=timeframe_minutes
        )
        
        logger.debug(f"🔍 Bullish SMT: {len(aligned_prev_lows)} aligned prev lows, {len(aligned_curr_lows)} aligned curr lows")
        
        for prev_pair in aligned_prev_lows:
            asset1_prev, asset2_prev, prev_time_diff = prev_pair
            
            for curr_pair in aligned_curr_lows:
                asset1_curr, asset2_curr, curr_time_diff = curr_pair
                
                # Check SMT conditions
                asset1_ll = asset1_curr['price'] < asset1_prev['price']  # Lower low
                asset2_hl = asset2_curr['price'] >= asset2_prev['price']  # Higher low
                
                # CRITICAL: Check interim price validation for bullish SMT
                interim_valid = self.swing_detector.validate_interim_price_action(
                    asset1_combined_data, asset1_prev, asset1_curr, "bullish"
                )
                
                if asset1_ll and asset2_hl and interim_valid:
                    logger.info(f"✅ BULLISH SMT FOUND with 3-candle tolerance:")
                    logger.info(f"   Prev swings: {asset1_prev['time'].strftime('%H:%M')} & {asset2_prev['time'].strftime('%H:%M')} (diff: {prev_time_diff:.1f}min)")
                    logger.info(f"   Curr swings: {asset1_curr['time'].strftime('%H:%M')} & {asset2_curr['time'].strftime('%H:%M')} (diff: {curr_time_diff:.1f}min)")
                    logger.info(f"   Asset1: Lower Low ({asset1_prev['price']:.4f} → {asset1_curr['price']:.4f})")
                    logger.info(f"   Asset2: Higher Low ({asset2_prev['price']:.4f} → {asset2_curr['price']:.4f})")
                    logger.info(f"   Interim validation: ✅ PASSED")
                    return (asset1_prev, asset1_curr, asset2_prev, asset2_curr)
                elif asset1_ll and asset2_hl and not interim_valid:
                    logger.warning(f"❌ BULLISH SMT REJECTED - Interim price invalid")
        
        return None
    
    def check_psp_for_smt(self, smt_data, asset1_data, asset2_data):
        """Check for PSP in past 5 candles for a specific SMT - WITH TIMING VALIDATION"""
        if not smt_data:
            return None
            
        signal_key = smt_data['signal_key']
        timeframe = smt_data['timeframe']
        cycle_type = smt_data['cycle']
        
        # Update tracking
        if signal_key in self.smt_psp_tracking:
            tracking = self.smt_psp_tracking[signal_key]
            tracking['check_count'] += 1
            tracking['last_check'] = datetime.now(NY_TZ)
        
        # Look for PSP in last 5 candles
        psp_signal = self._detect_psp_last_n_candles(asset1_data, asset2_data, timeframe, n=5)
        
        if psp_signal:
            # VALIDATE PSP TIMING - Must be within reasonable time of SMT formation
            smt_formation_time = tracking.get('formation_time', smt_data['formation_time'])
            psp_formation_time = psp_signal['formation_time']
            
            if not self.timing_manager.is_psp_within_bounds(smt_formation_time, psp_formation_time, cycle_type):
                logger.warning(f"⚠️ PSP TOO FAR FROM SMT: {cycle_type} SMT at {smt_formation_time.strftime('%H:%M')}, PSP at {psp_formation_time.strftime('%H:%M')}")
                return None
            
            logger.info(f"🎯 PSP FOUND for SMT {smt_data['cycle']} {smt_data['quarters']} - {psp_signal['candles_ago']} candles ago")
            
            # Mark PSP as found for this SMT
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
                    formation_time = asset1_candle['time']
                    logger.info(f"🎯 PSP DETECTED: {timeframe} candle at {formation_time.strftime('%H:%M')} - Asset1: {asset1_color}, Asset2: {asset2_color}")
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
            except (ValueError, TypeError) as e:
                logger.error(f"Error in PSP calculation: {e}")
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
        
        # Stop checking if we found PSP or reached max checks
        if tracking['psp_found'] or tracking['check_count'] >= tracking['max_checks']:
            return False
        
        # Stop checking if SMT is invalidated
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
    
        # Only block if SAME candle + already sent
        if count >= 1 and smt_data['candle_time'] == self.last_smt_candle:
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
# ULTIMATE TRADING SYSTEM WITH TRIPLE CONFLUENCE
# ================================

class UltimateTradingSystem:
    def __init__(self, pair_group, pair_config):
        self.pair_group = pair_group
        self.pair_config = pair_config
        self.pair1 = pair_config['pair1']
        self.pair2 = pair_config['pair2']
        
        # Initialize ultimate components
        self.timing_manager = RobustTimingManager()
        self.quarter_manager = RobustQuarterManager()
        self.crt_detector = RobustCRTDetector(self.timing_manager)  # UPDATED: Enhanced CRT detector
        self.smt_detector = UltimateSMTDetector(pair_config, self.timing_manager)
        self.signal_builder = UltimateSignalBuilder(pair_group, self.timing_manager)
        
        # Data storage
        self.market_data = {self.pair1: {}, self.pair2: {}}
        
        logger.info(f"🎯 Initialized ULTIMATE trading system for {self.pair1}/{self.pair2}")
    
    async def run_ultimate_analysis(self, api_key):
        """Run ultimate analysis with all fixes"""
        try:
            current_status = self.signal_builder.get_progress_status()
            logger.info(f"📊 {self.pair_group}: Current status - {current_status}")
            
            # Fetch ALL data needed for analysis
            await self._fetch_all_data(api_key)

            # === FIXED DEBUG CALL - Pass market data ===
            # === FIXED DEBUG CALL ===
            for cycle in ['monthly', 'weekly', 'daily', '90min']:
                self.smt_detector.run_comprehensive_debug(
                    cycle, 
                    self.market_data[self.pair1], 
                    self.market_data[self.pair2]
                )
            
            # Step 1: Check SMT invalidations and PSP tracking
            await self._check_smt_tracking()
            
            # Step 2: Scan for NEW SMT signals
            await self._scan_all_smt()
            
            # Step 3: Check for PSP for existing SMTs
            await self._check_psp_for_existing_smts()
            
            # Step 4: Scan for CRT signals (ALWAYS scan for CRT) - NOW WITH PSP DETECTION
            await self._scan_crt_signals()
            
            # Check if signal is complete and not conflicted
            if self.signal_builder.is_signal_ready() and not self.signal_builder.has_serious_conflict():
                signal = self.signal_builder.get_signal_details()
                if signal and not self.timing_manager.is_duplicate_signal(signal['signal_key'], self.pair_group):
                    logger.info(f"🎯 {self.pair_group}: ULTIMATE SIGNAL COMPLETE via {signal['path']}")
                    self.signal_builder.reset()
                    return signal
                elif self.timing_manager.is_duplicate_signal(signal['signal_key'], self.pair_group):
                    logger.info(f"⏳ {self.pair_group}: STRONG DUPLICATE PREVENTION - skipping signal")
            
            # Check if expired
            if self.signal_builder.is_expired():
                self.signal_builder.reset()
            
            logger.info(f"✅ {self.pair_group}: Ultimate analysis complete")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error in ultimate analysis for {self.pair_group}: {str(e)}")
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
                # Remove from signal builder
                if cycle in self.signal_builder.active_smts:
                    del self.signal_builder.active_smts[cycle]
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
        
        # ALWAYS include CRT timeframes for better detection
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
                logger.warning(f"⚠️ No data for {cycle} ({timeframe})")
                continue
            
            logger.info(f"🔍 Scanning {cycle} cycle ({timeframe}) for SMT...")
            smt_signal = self.smt_detector.detect_smt_all_cycles(pair1_data, pair2_data, cycle)
            
            if smt_signal:
                # Check for PSP immediately for this new SMT
                psp_signal = self.smt_detector.check_psp_for_smt(smt_signal, pair1_data, pair2_data)
                
                # Add SMT with PSP if found
                self.signal_builder.add_smt_signal(smt_signal, psp_signal)
    
    async def _scan_crt_signals(self):
        """Scan for CRT signals on all timeframes - ENHANCED WITH PSP DETECTION"""
        crt_detected = False
        
        for timeframe in CRT_TIMEFRAMES:
            pair1_data = self.market_data[self.pair1].get(timeframe)
            pair2_data = self.market_data[self.pair2].get(timeframe)
            
            if (pair1_data is None or not isinstance(pair1_data, pd.DataFrame) or pair1_data.empty or
                pair2_data is None or not isinstance(pair2_data, pd.DataFrame) or pair2_data.empty):
                continue
            
            # UPDATED: CRT detection now includes PSP checking
            crt_signal = self.crt_detector.calculate_crt_current_candle(
                pair1_data, pair1_data, pair2_data, timeframe
            )
            
            if crt_signal and self.signal_builder.set_crt_signal(crt_signal, timeframe, crt_signal.get('psp_signal')):
                logger.info(f"🔷 {self.pair_group}: Fresh CRT detected on {timeframe}")
                if crt_signal.get('psp_signal'):
                    logger.info(f"🎯 {self.pair_group}: CRT with PSP confluence detected!")
                crt_detected = True
                break
        
        if not crt_detected:
            logger.debug(f"🔍 {self.pair_group}: No CRT signals detected this cycle")
    
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
    
    async def run_ultimate_systems(self):
        """Run all trading systems with ultimate decision making"""
        logger.info("🎯 Starting ULTIMATE Multi-Pair Trading System...")
        
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
                    sleep_times.append(system.get_sleep_time())
                
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
