#!/usr/bin/env python3
"""
MULTI-PAIR SMT TRADING SYSTEM - ADVANCED HIERARCHICAL VERSION WITH CRT
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
from typing import Dict, List, Tuple, Optional
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
        'pair2': 'US500_USD',
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

# System Configuration
NY_TZ = pytz.timezone('America/New_York')
BASE_INTERVAL = 300
MIN_INTERVAL = 30
MAX_RETRIES = 3

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

def fetch_candles(instrument, timeframe, count=300, api_key=None):
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
# CYCLE MANAGER - UNCHANGED
# ================================

class UTC4CycleManager:
    def __init__(self):
        self.ny_tz = pytz.timezone('America/New_York')
        
    def get_current_ny_time(self):
        return datetime.now(self.ny_tz)
    
    def detect_current_quarters(self, timestamp=None):
        if timestamp is None:
            timestamp = self.get_current_ny_time()
        else:
            if timestamp.tzinfo is None:
                timestamp = self.ny_tz.localize(timestamp)
            else:
                timestamp = timestamp.astimezone(self.ny_tz)
                
        return {
            'monthly': self._get_monthly_quarter(timestamp),
            'weekly': self._get_weekly_quarter(timestamp),
            'daily': self._get_daily_quarter(timestamp),
            '90min': self._get_90min_quarter(timestamp)
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
    
    def _get_90min_quarter(self, timestamp):
        daily_quarter = self._get_daily_quarter(timestamp)
        minute_of_day = timestamp.hour * 60 + timestamp.minute
        
        daily_quarter_start = {
            'q1': 18 * 60,
            'q2': 0,  
            'q3': 6 * 60,
            'q4': 12 * 60
        }[daily_quarter]
        
        segment = (minute_of_day - daily_quarter_start) // 90
        return f'q{segment + 1}' if segment < 4 else 'q_less'

# ================================
# PATTERN DETECTORS - WITH CRT
# ================================

class CRTDetector:
    """Candlestick Rejection Pattern Detector"""
    
    @staticmethod
    def calculate_crt_vectorized(df):
        """Vectorized implementation of CRT signal calculation"""
        df = df.copy()
        df['crt'] = None

        # Shifted columns for previous candles
        df['c1_low'] = df['low'].shift(2)
        df['c1_high'] = df['high'].shift(2)
        df['c2_low'] = df['low'].shift(1)
        df['c2_high'] = df['high'].shift(1)
        df['c2_close'] = df['close'].shift(1)

        # Candle metrics
        df['c2_range'] = df['c2_high'] - df['c2_low']
        df['c2_mid'] = df['c2_low'] + 0.5 * df['c2_range']

        # Vectorized conditions
        buy_mask = (df['c2_low'] < df['c1_low']) & (df['c2_close'] > df['c1_low']) & (df['open'] > df['c2_mid'])
        sell_mask = (df['c2_high'] > df['c1_high']) & (df['c2_close'] < df['c1_high']) & (df['open'] < df['c2_mid'])

        df.loc[buy_mask, 'crt'] = 'BUY'
        df.loc[sell_mask, 'crt'] = 'SELL'

        # Cleanup
        df.drop(columns=['c1_low', 'c1_high', 'c2_low', 'c2_high', 'c2_close', 'c2_range', 'c2_mid'], inplace=True)

        return df
    
    def detect_crt(self, df, lookback=10):
        """Detect CRT patterns in recent data"""
        try:
            if df.empty or len(df) < 3:
                return None
            
            # Calculate CRT signals
            df_with_crt = self.calculate_crt_vectorized(df)
            
            # Get the most recent CRT signal
            recent_crt = df_with_crt[df_with_crt['crt'].notnull()].tail(lookback)
            
            if recent_crt.empty:
                return None
            
            # Return the most recent CRT signal
            latest_crt = recent_crt.iloc[-1]
            crt_data = {
                'direction': latest_crt['crt'].lower(),  # 'buy' or 'sell'
                'timestamp': latest_crt['time'],
                'cycle_type': 'crt_pattern'
            }
            
            logger.info(f"CRT detected: {crt_data['direction']}")
            return crt_data
            
        except Exception as e:
            logger.error(f"Error in CRT detection: {str(e)}")
            return None

class SMTDetector:
    def __init__(self):
        self.smt_history = []
    
    def detect_smt_between_quarters(self, asset1_q1, asset1_q2, asset2_q1, asset2_q2, cycle_type, quarters):
        """Detect SMT between two consecutive quarters"""
        try:
            if not all([asset1_q1, asset1_q2, asset2_q1, asset2_q2]):
                return None
            
            # Bearish SMT: Asset1 makes HH, Asset2 doesn't
            bearish_condition1 = (asset1_q2['high'] > asset1_q1['high'] and 
                                 asset2_q2['high'] <= asset2_q1['high'])
            
            bearish_condition2 = (asset1_q2['close'] > asset1_q1['high'] and 
                                 asset2_q2['close'] <= asset2_q1['high'])
            
            # Bullish SMT: Asset1 makes LL, Asset2 doesn't
            bullish_condition1 = (asset1_q2['low'] < asset1_q1['low'] and 
                                 asset2_q2['low'] >= asset2_q1['low'])
            
            bullish_condition2 = (asset1_q2['close'] < asset1_q1['low'] and 
                                 asset2_q2['close'] >= asset2_q1['low'])
            
            if bearish_condition1 or bearish_condition2:
                direction = 'bearish'
            elif bullish_condition1 or bullish_condition2:
                direction = 'bullish'
            else:
                return None
            
            smt_data = {
                'direction': direction,
                'type': 'regular',
                'cycle': cycle_type,
                'quarters': quarters,
                'timestamp': datetime.now(NY_TZ)
            }
            
            self.smt_history.append(smt_data)
            logger.info(f"SMT detected: {direction} {cycle_type} {quarters}")
            return smt_data
            
        except Exception as e:
            logger.error(f"Error in SMT detection: {str(e)}")
            return None

class PSPDetector:
    @staticmethod
    def detect_psp_in_history(asset1_data, asset2_data, timeframe, lookback=5):
        """Detect PSP in recent history (lookback candles)"""
        try:
            if asset1_data.empty or asset2_data.empty:
                return None
            
            # Get recent complete candles
            recent_asset1 = asset1_data[asset1_data['complete'] == True].tail(lookback)
            recent_asset2 = asset2_data[asset2_data['complete'] == True].tail(lookback)
            
            if recent_asset1.empty or recent_asset2.empty:
                return None
            
            # Check for PSP in any of the recent candles
            for i in range(min(len(recent_asset1), len(recent_asset2))):
                asset1_candle = recent_asset1.iloc[i]
                asset2_candle = recent_asset2.iloc[i]
                
                asset1_color = 'green' if float(asset1_candle['close']) > float(asset1_candle['open']) else 'red'
                asset2_color = 'green' if float(asset2_candle['close']) > float(asset2_candle['open']) else 'red'
                
                if asset1_color != asset2_color:
                    psp_data = {
                        'timeframe': timeframe,
                        'asset1_color': asset1_color,
                        'asset2_color': asset2_color,
                        'timestamp': asset1_candle['time'],
                        'candle_index': i
                    }
                    logger.info(f"PSP detected on {timeframe}: {asset1_color}/{asset2_color}")
                    return psp_data
            
            return None
        except Exception as e:
            logger.error(f"Error in PSP detection: {str(e)}")
            return None

# ================================
# ADVANCED SIGNAL BUILDER - MULTI-PATH
# ================================

class AdvancedSignalBuilder:
    def __init__(self, pair_group):
        self.pair_group = pair_group
        
        # Signal components
        self.primary_smt = None
        self.primary_psp = None
        self.primary_crt = None
        self.confirmation_smt = None
        self.confirmation_psp = None
        
        # Signal state
        self.signal_strength = 0
        self.signal_criteria = []
        self.current_state = "IDLE"
        
        # Signal paths
        self.valid_paths = {
            "PATH_1": ["HTF_SMT", "LTF_SMT", "PSP"],  # Original path
            "PATH_2": ["HTF_PSP", "HTF_CRT", "LTF_SMT"],  # New path 1
            "PATH_3": ["HTF_CRT", "LTF_SMT"],  # New path 2
            "PATH_4": ["HTF_PSP", "LTF_SMT"]   # New path 3
        }
        
        self.active_path = None
    
    def set_primary_smt(self, smt_data):
        """Set primary SMT from higher timeframe"""
        if smt_data and not self.primary_smt:
            self.primary_smt = smt_data
            self.signal_strength += 3
            self.signal_criteria.append(f"Primary SMT: {smt_data['direction']} {smt_data['cycle']} {smt_data['quarters']}")
            self.current_state = "HTF_SMT_FOUND"
            logger.info(f"{self.pair_group}: Primary SMT set - {smt_data['direction']} {smt_data['cycle']} {smt_data['quarters']}")
            return True
        return False
    
    def set_primary_psp(self, psp_data):
        """Set primary PSP from higher timeframe"""
        if psp_data and not self.primary_psp:
            self.primary_psp = psp_data
            self.signal_strength += 2
            self.signal_criteria.append(f"Primary PSP: {psp_data['timeframe']}")
            
            if self.current_state == "IDLE":
                self.current_state = "HTF_PSP_FOUND"
            elif self.current_state == "HTF_CRT_FOUND":
                self.current_state = "HTF_PSP_CRT_FOUND"
                
            logger.info(f"{self.pair_group}: Primary PSP set - {psp_data['timeframe']}")
            return True
        return False
    
    def set_primary_crt(self, crt_data):
        """Set primary CRT from higher timeframe"""
        if crt_data and not self.primary_crt:
            self.primary_crt = crt_data
            self.signal_strength += 2
            self.signal_criteria.append(f"Primary CRT: {crt_data['direction']}")
            
            if self.current_state == "IDLE":
                self.current_state = "HTF_CRT_FOUND"
            elif self.current_state == "HTF_PSP_FOUND":
                self.current_state = "HTF_PSP_CRT_FOUND"
                
            logger.info(f"{self.pair_group}: Primary CRT set - {crt_data['direction']}")
            return True
        return False
    
    def set_confirmation_smt(self, smt_data, required_direction=None):
        """Set confirmation SMT from lower timeframe"""
        if smt_data and not self.confirmation_smt:
            # Check direction consistency if required
            if required_direction and smt_data['direction'] != required_direction:
                return False
                
            self.confirmation_smt = smt_data
            self.signal_strength += 2
            self.signal_criteria.append(f"Confirmation SMT: {smt_data['direction']} {smt_data['cycle']} {smt_data['quarters']}")
            self.current_state = "LTF_SMT_FOUND"
            logger.info(f"{self.pair_group}: Confirmation SMT set - {smt_data['direction']} {smt_data['cycle']} {smt_data['quarters']}")
            return True
        return False
    
    def set_confirmation_psp(self, psp_data):
        """Set confirmation PSP (optional)"""
        if psp_data and not self.confirmation_psp:
            self.confirmation_psp = psp_data
            self.signal_strength += 1
            self.signal_criteria.append(f"Confirmation PSP: {psp_data['timeframe']}")
            logger.info(f"{self.pair_group}: Confirmation PSP set - {psp_data['timeframe']}")
            return True
        return False
    
    def is_signal_ready(self):
        """Check if we have complete signal based on active path"""
        # Path 1: HTF SMT -> LTF SMT -> PSP
        if (self.primary_smt and self.confirmation_smt and 
            (self.primary_psp or self.confirmation_psp)):
            self.active_path = "PATH_1"
            return True
        
        # Path 2: HTF PSP + HTF CRT -> LTF SMT
        if (self.primary_psp and self.primary_crt and self.confirmation_smt):
            self.active_path = "PATH_2"
            return True
        
        # Path 3: HTF CRT -> LTF SMT
        if (self.primary_crt and self.confirmation_smt):
            self.active_path = "PATH_3"
            return True
        
        # Path 4: HTF PSP -> LTF SMT
        if (self.primary_psp and self.confirmation_smt):
            self.active_path = "PATH_4"
            return True
        
        return False
    
    def get_signal_details(self):
        """Get complete signal details"""
        if not self.is_signal_ready():
            return None
            
        # Determine direction from available components
        direction = None
        if self.primary_smt:
            direction = self.primary_smt['direction']
        elif self.primary_crt:
            direction = self.primary_crt['direction']
        elif self.confirmation_smt:
            direction = self.confirmation_smt['direction']
        else:
            direction = 'unknown'
            
        return {
            'pair_group': self.pair_group,
            'direction': direction,
            'strength': self.signal_strength,
            'path': self.active_path,
            'criteria': self.signal_criteria.copy(),
            'primary_smt': self.primary_smt,
            'primary_psp': self.primary_psp,
            'primary_crt': self.primary_crt,
            'confirmation_smt': self.confirmation_smt,
            'confirmation_psp': self.confirmation_psp,
            'timestamp': datetime.now(NY_TZ)
        }
    
    def get_next_required_action(self):
        """Get what we need to look for next based on current state"""
        state_actions = {
            "IDLE": "SCAN_HTF_PATTERNS",
            "HTF_SMT_FOUND": "SCAN_LTF_SMT",
            "HTF_PSP_FOUND": "SCAN_HTF_CRT_OR_LTF_SMT",
            "HTF_CRT_FOUND": "SCAN_HTF_PSP_OR_LTF_SMT", 
            "HTF_PSP_CRT_FOUND": "SCAN_LTF_SMT",
            "LTF_SMT_FOUND": "SCAN_PSP_CONFIRMATION"
        }
        return state_actions.get(self.current_state, "SCAN_HTF_PATTERNS")
    
    def reset(self):
        """Reset for new signal"""
        self.primary_smt = None
        self.primary_psp = None
        self.primary_crt = None
        self.confirmation_smt = None
        self.confirmation_psp = None
        self.signal_strength = 0
        self.signal_criteria = []
        self.current_state = "IDLE"
        self.active_path = None
        logger.info(f"{self.pair_group}: Signal builder reset")

# ================================
# ADVANCED TRADING SYSTEM
# ================================

class AdvancedTradingSystem:
    def __init__(self, pair_group, pair_config):
        self.pair_group = pair_group
        self.pair_config = pair_config
        self.pair1 = pair_config['pair1']
        self.pair2 = pair_config['pair2']
        
        # Initialize components
        self.cycle_manager = UTC4CycleManager()
        self.smt_detector = SMTDetector()
        self.psp_detector = PSPDetector()
        self.crt_detector = CRTDetector()
        self.signal_builder = AdvancedSignalBuilder(pair_group)
        
        # Data storage
        self.market_data = {self.pair1: {}, self.pair2: {}}
        
        # Define hierarchy (higher to lower)
        self.cycle_hierarchy = ['monthly', 'weekly', 'daily', '90min']
        
        logger.info(f"Initialized advanced trading system for {self.pair1}/{self.pair2}")
    
    async def run_analysis(self, api_key):
        """Run advanced hierarchical analysis"""
        try:
            # Fetch market data
            await self._fetch_market_data(api_key)
            
            # Get current state and determine what to do next
            next_action = self.signal_builder.get_next_required_action()
            
            if next_action == "SCAN_HTF_PATTERNS":
                await self._scan_htf_patterns()
            elif next_action == "SCAN_LTF_SMT":
                await self._scan_ltf_smt()
            elif next_action == "SCAN_HTF_CRT_OR_LTF_SMT":
                await self._scan_htf_crt_or_ltf_smt()
            elif next_action == "SCAN_HTF_PSP_OR_LTF_SMT":
                await self._scan_htf_psp_or_ltf_smt()
            elif next_action == "SCAN_PSP_CONFIRMATION":
                await self._scan_psp_confirmation()
            
            # Check if signal is complete
            if self.signal_builder.is_signal_ready():
                signal = self.signal_builder.get_signal_details()
                if signal:
                    logger.info(f"🚨 SIGNAL GENERATED for {self.pair_group} via {signal['path']}")
                    self.signal_builder.reset()
                    return signal
            
            return None
            
        except Exception as e:
            logger.error(f"Error in analysis for {self.pair_group}: {str(e)}")
            return None
    
    async def _fetch_market_data(self, api_key):
        """Fetch market data for both pairs"""
        timeframes = self.pair_config['timeframe_mapping']
        
        for pair in [self.pair1, self.pair2]:
            for cycle, tf in timeframes.items():
                try:
                    df = await asyncio.get_event_loop().run_in_executor(
                        None, fetch_candles, pair, tf, 200, api_key
                    )
                    if not df.empty:
                        self.market_data[pair][cycle] = df
                        logger.debug(f"Fetched {len(df)} candles for {pair} {tf}")
                    else:
                        logger.warning(f"No data for {pair} {tf}")
                except Exception as e:
                    logger.error(f"Error fetching {pair} {tf}: {str(e)}")
    
    async def _scan_htf_patterns(self):
        """Scan for primary patterns in higher timeframes"""
        logger.info(f"{self.pair_group}: Scanning for HTF patterns...")
        
        # Try from highest to lowest timeframe for primary patterns
        for cycle_type in self.cycle_hierarchy[:-1]:
            # Check for SMT
            smt_found = await self._check_smt_in_cycle(cycle_type)
            if smt_found:
                return True
            
            # Check for PSP
            psp_found = await self._check_psp_in_cycle(cycle_type)
            if psp_found:
                # Continue to check for CRT or move to next action
                pass
            
            # Check for CRT
            crt_found = await self._check_crt_in_cycle(cycle_type)
            if crt_found:
                # Continue to check for PSP or move to next action
                pass
        
        logger.info(f"{self.pair_group}: No HTF patterns found")
        return False
    
    async def _scan_ltf_smt(self):
        """Scan for confirmation SMT in lower timeframes"""
        if not self.signal_builder.primary_smt and not self.signal_builder.primary_psp and not self.signal_builder.primary_crt:
            return False
        
        # Determine required direction and starting cycle
        required_direction = None
        start_cycle = None
        
        if self.signal_builder.primary_smt:
            required_direction = self.signal_builder.primary_smt['direction']
            start_cycle = self.signal_builder.primary_smt['cycle']
        elif self.signal_builder.primary_crt:
            required_direction = self.signal_builder.primary_crt['direction']
            start_cycle = 'weekly'  # Default start for CRT
        elif self.signal_builder.primary_psp:
            # For PSP, we need to determine direction from market context
            required_direction = await self._determine_direction_from_context()
            start_cycle = 'weekly'  # Default start for PSP
        
        if not required_direction or not start_cycle:
            return False
        
        # Find lower timeframes than the start cycle
        try:
            start_index = self.cycle_hierarchy.index(start_cycle)
            lower_cycles = self.cycle_hierarchy[start_index + 1:]
        except ValueError:
            lower_cycles = self.cycle_hierarchy[1:]  # Fallback
        
        logger.info(f"{self.pair_group}: Scanning for LTF SMT in {lower_cycles} with direction {required_direction}...")
        
        for cycle_type in lower_cycles:
            smt_found = await self._check_smt_in_cycle(cycle_type, required_direction)
            if smt_found:
                return True
        
        logger.info(f"{self.pair_group}: No LTF SMT found yet")
        return False
    
    async def _scan_htf_crt_or_ltf_smt(self):
        """Scan for HTF CRT or LTF SMT when we have HTF PSP"""
        if not self.signal_builder.primary_psp:
            return False
        
        # First try to find HTF CRT
        for cycle_type in self.cycle_hierarchy[:-1]:
            crt_found = await self._check_crt_in_cycle(cycle_type)
            if crt_found:
                return True
        
        # If no CRT found, try LTF SMT
        return await self._scan_ltf_smt()
    
    async def _scan_htf_psp_or_ltf_smt(self):
        """Scan for HTF PSP or LTF SMT when we have HTF CRT"""
        if not self.signal_builder.primary_crt:
            return False
        
        # First try to find HTF PSP
        for cycle_type in self.cycle_hierarchy[:-1]:
            psp_found = await self._check_psp_in_cycle(cycle_type)
            if psp_found:
                return True
        
        # If no PSP found, try LTF SMT
        return await self._scan_ltf_smt()
    
    async def _scan_psp_confirmation(self):
        """Scan for PSP confirmation (optional)"""
        if not self.signal_builder.confirmation_smt:
            return False
        
        # Use the confirmation SMT's timeframe for PSP scanning
        confirmation_cycle = self.signal_builder.confirmation_smt['cycle']
        timeframe = self.pair_config['timeframe_mapping'][confirmation_cycle]
        
        logger.info(f"{self.pair_group}: Scanning for optional PSP confirmation on {timeframe}...")
        
        pair1_data = self.market_data[self.pair1].get(confirmation_cycle)
        pair2_data = self.market_data[self.pair2].get(confirmation_cycle)
        
        if pair1_data is None or pair2_data is None:
            return False
        
        psp_signal = self.psp_detector.detect_psp_in_history(pair1_data, pair2_data, timeframe, lookback=5)
        if psp_signal:
            return self.signal_builder.set_confirmation_psp(psp_signal)
        
        logger.info(f"{self.pair_group}: No PSP confirmation found (optional)")
        return False
    
    async def _check_smt_in_cycle(self, cycle_type, required_direction=None):
        """Check for SMT in a specific cycle"""
        pair1_data = self.market_data[self.pair1].get(cycle_type)
        pair2_data = self.market_data[self.pair2].get(cycle_type)
        
        if pair1_data is None or pair2_data is None or pair1_data.empty or pair2_data.empty:
            return False
        
        try:
            # Simplified SMT detection using recent data
            if len(pair1_data) < 10 or len(pair2_data) < 10:
                return False
            
            # Get recent highs and lows
            asset1_recent_high = float(pair1_data['high'].tail(5).max())
            asset1_prev_high = float(pair1_data['high'].head(5).max())
            asset2_recent_high = float(pair2_data['high'].tail(5).max())
            asset2_prev_high = float(pair2_data['high'].head(5).max())
            
            asset1_recent_low = float(pair1_data['low'].tail(5).min())
            asset1_prev_low = float(pair1_data['low'].head(5).min())
            asset2_recent_low = float(pair2_data['low'].tail(5).min())
            asset2_prev_low = float(pair2_data['low'].head(5).min())
            
            # Check directions
            bearish = (asset1_recent_high > asset1_prev_high and asset2_recent_high <= asset2_prev_high)
            bullish = (asset1_recent_low < asset1_prev_low and asset2_recent_low >= asset2_prev_low)
            
            if required_direction:
                if required_direction == 'bearish' and bearish:
                    direction = 'bearish'
                elif required_direction == 'bullish' and bullish:
                    direction = 'bullish'
                else:
                    return False
            else:
                if bearish:
                    direction = 'bearish'
                elif bullish:
                    direction = 'bullish'
                else:
                    return False
            
            # Create SMT data
            smt_data = {
                'direction': direction,
                'type': 'regular',
                'cycle': cycle_type,
                'quarters': "q1→q2",  # Simplified quarters for now
                'timestamp': datetime.now(NY_TZ)
            }
            
            # Add to signal builder based on current state
            if self.signal_builder.current_state in ["IDLE", "HTF_PSP_FOUND", "HTF_CRT_FOUND", "HTF_PSP_CRT_FOUND"]:
                return self.signal_builder.set_primary_smt(smt_data)
            else:
                return self.signal_builder.set_confirmation_smt(smt_data, required_direction)
                
        except Exception as e:
            logger.error(f"Error checking SMT in {cycle_type} for {self.pair_group}: {str(e)}")
            return False
    
    async def _check_psp_in_cycle(self, cycle_type):
        """Check for PSP in a specific cycle"""
        pair1_data = self.market_data[self.pair1].get(cycle_type)
        pair2_data = self.market_data[self.pair2].get(cycle_type)
        
        if pair1_data is None or pair2_data is None:
            return False
        
        timeframe = self.pair_config['timeframe_mapping'][cycle_type]
        psp_signal = self.psp_detector.detect_psp_in_history(pair1_data, pair2_data, timeframe, lookback=5)
        
        if psp_signal:
            return self.signal_builder.set_primary_psp(psp_signal)
        
        return False
    
    async def _check_crt_in_cycle(self, cycle_type):
        """Check for CRT in a specific cycle"""
        # Check CRT for both assets
        pair1_data = self.market_data[self.pair1].get(cycle_type)
        pair2_data = self.market_data[self.pair2].get(cycle_type)
        
        if pair1_data is None or pair2_data is None:
            return False
        
        # Check CRT for pair1
        crt_signal_1 = self.crt_detector.detect_crt(pair1_data)
        crt_signal_2 = self.crt_detector.detect_crt(pair2_data)
        
        # We need at least one asset to have CRT
        if crt_signal_1 or crt_signal_2:
            # Prefer the stronger signal or combine them
            crt_signal = crt_signal_1 if crt_signal_1 else crt_signal_2
            return self.signal_builder.set_primary_crt(crt_signal)
        
        return False
    
    async def _determine_direction_from_context(self):
        """Determine market direction from price action context"""
        # Simplified direction detection - you can enhance this
        try:
            # Use daily data for context
            pair1_daily = self.market_data[self.pair1].get('daily')
            if pair1_daily is not None and not pair1_daily.empty:
                recent_trend = 'bullish' if float(pair1_daily['close'].iloc[-1]) > float(pair1_daily['open'].iloc[-1]) else 'bearish'
                return recent_trend
        except:
            pass
        
        return 'bullish'  # Default fallback

# ================================
# MAIN MANAGER - UPDATED
# ================================

class MultiPairTradingManager:
    def __init__(self, api_key, telegram_token, chat_id):
        self.api_key = api_key
        self.telegram_token = telegram_token
        self.chat_id = chat_id
        self.trading_systems = {}
        
        # Initialize trading systems for all pairs
        for pair_group, pair_config in TRADING_PAIRS.items():
            self.trading_systems[pair_group] = AdvancedTradingSystem(pair_group, pair_config)
        
        logger.info(f"Initialized multi-pair manager with {len(self.trading_systems)} pair groups")
    
    async def run_all_systems(self):
        """Run all trading systems in parallel"""
        logger.info("Starting Advanced Multi-Pair SMT Trading System...")
        
        while True:
            try:
                tasks = []
                
                # Create tasks for all pair groups
                for pair_group, system in self.trading_systems.items():
                    task = asyncio.create_task(
                        system.run_analysis(self.api_key),
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
                        logger.error(f"Analysis task failed for {pair_group}: {str(result)}")
                    elif result is not None:
                        signals.append(result)
                        logger.info(f"Signal found for {pair_group}")
                
                # Send signals to Telegram
                if signals:
                    await self._process_signals(signals)
                
                # Calculate adaptive sleep interval based on system states
                sleep_time = self._calculate_sleep_interval()
                logger.info(f"Sleeping for {sleep_time} seconds")
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                await asyncio.sleep(60)
    
    async def _process_signals(self, signals):
        """Process and send signals to Telegram"""
        for signal in signals:
            try:
                message = self._format_signal_message(signal)
                success = send_telegram(message, self.telegram_token, self.chat_id)
                
                if success:
                    logger.info(f"Signal sent to Telegram for {signal['pair_group']}")
                else:
                    logger.error(f"Failed to send signal for {signal['pair_group']}")
                    
            except Exception as e:
                logger.error(f"Error processing signal: {str(e)}")
    
    def _format_signal_message(self, signal):
        """Format signal for Telegram with detailed criteria"""
        pair_group = signal.get('pair_group', 'Unknown')
        direction = signal.get('direction', 'UNKNOWN').upper()
        strength = signal.get('strength', 0)
        path = signal.get('path', 'UNKNOWN')
        
        message = f"🚨 *ADVANCED TRADING SIGNAL* 🚨\n\n"
        message += f"*Pair Group:* {pair_group.replace('_', ' ').title()}\n"
        message += f"*Direction:* {direction}\n"
        message += f"*Strength:* {strength}/10\n"
        message += f"*Path:* {path}\n\n"
        
        # Add all criteria
        if 'criteria' in signal:
            message += "*Signal Criteria:*\n"
            for criterion in signal['criteria']:
                message += f"• {criterion}\n"
        
        message += f"\n*Time:* {datetime.now(NY_TZ).strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
        message += f"\n#AdvancedSignal #{pair_group} #{path}"
        
        return message
    
    def _calculate_sleep_interval(self):
        """Calculate adaptive sleep interval based on system states"""
        base_interval = BASE_INTERVAL
        min_interval = MIN_INTERVAL
        
        # Check active signal builders and their states
        active_weight = 0
        for system in self.trading_systems.values():
            state = system.signal_builder.current_state
            if state != "IDLE":
                # Weight states by how close they are to signal
                state_weights = {
                    "IDLE": 0,
                    "HTF_SMT_FOUND": 1,
                    "HTF_PSP_FOUND": 1,
                    "HTF_CRT_FOUND": 1,
                    "HTF_PSP_CRT_FOUND": 2,
                    "LTF_SMT_FOUND": 3
                }
                active_weight += state_weights.get(state, 1)
        
        if active_weight > 0:
            return max(min_interval, base_interval // (active_weight + 1))
        
        return base_interval

# ================================
# MAIN EXECUTION
# ================================

async def main():
    """Main entry point"""
    logger.info("Starting Advanced Multi-Pair SMT Trading System with CRT")
    
    # Get credentials from environment
    api_key = os.getenv('OANDA_API_KEY')
    telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
    telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if not all([api_key, telegram_token, telegram_chat_id]):
        logger.error("Missing required environment variables:")
        logger.error("OANDA_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID")
        logger.info("Please set these environment variables and try again")
        return
    
    try:
        # Initialize manager
        manager = MultiPairTradingManager(api_key, telegram_token, telegram_chat_id)
        
        # Run all systems
        await manager.run_all_systems()
        
    except KeyboardInterrupt:
        logger.info("System stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    # Run the system
    asyncio.run(main())
