#!/usr/bin/env python3
"""
MULTI-PAIR SMT TRADING SYSTEM - HIERARCHICAL VERSION
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
        'pair1': 'US30_USD',
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
# PATTERN DETECTORS - IMPROVED
# ================================

class QuarterData:
    """Track quarter data properly"""
    def __init__(self, cycle_type, quarter_name, start_time, end_time):
        self.cycle_type = cycle_type
        self.quarter_name = quarter_name
        self.start_time = start_time
        self.end_time = end_time
        self.assets_data = {}
        self.quarter_ohlc = {}
    
    def add_candle(self, asset, candle):
        if asset not in self.assets_data:
            self.assets_data[asset] = []
        self.assets_data[asset].append(candle)
        self._update_quarter_ohlc(asset)
    
    def _update_quarter_ohlc(self, asset):
        candles = self.assets_data[asset]
        if not candles:
            return
        df = pd.DataFrame(candles)
        self.quarter_ohlc[asset] = {
            'open': float(df['open'].iloc[0]),
            'high': float(df['high'].max()),
            'low': float(df['low'].min()),
            'close': float(df['close'].iloc[-1]),
            'candle_count': len(candles)
        }
    
    def get_quarter_ohlc(self, asset):
        return self.quarter_ohlc.get(asset)

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
                'timestamp': datetime.now(NY_TZ),
                'asset1_data': {'q1': asset1_q1, 'q2': asset1_q2},
                'asset2_data': {'q1': asset2_q1, 'q2': asset2_q2}
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
# SIGNAL BUILDER - HIERARCHICAL
# ================================

class HierarchicalSignalBuilder:
    def __init__(self, pair_group):
        self.pair_group = pair_group
        self.primary_smt = None  # HTF SMT
        self.confirmation_smt = None  # LTF SMT  
        self.psp_signal = None
        self.signal_strength = 0
        self.current_state = "IDLE"  # IDLE -> HTF_SCAN -> LTF_SCAN -> PSP_WAIT -> READY
        
    def set_primary_smt(self, smt_data):
        """Set the primary (HTF) SMT"""
        if smt_data and not self.primary_smt:
            self.primary_smt = smt_data
            self.current_state = "HTF_FOUND"
            self.signal_strength += 3
            logger.info(f"{self.pair_group}: Primary SMT set - {smt_data['direction']} {smt_data['cycle']} {smt_data['quarters']}")
            return True
        return False
    
    def set_confirmation_smt(self, smt_data):
        """Set the confirmation (LTF) SMT"""
        if (smt_data and self.primary_smt and 
            smt_data['direction'] == self.primary_smt['direction'] and
            smt_data['cycle'] != self.primary_smt['cycle']):
            
            self.confirmation_smt = smt_data
            self.current_state = "LTF_FOUND"
            self.signal_strength += 2
            logger.info(f"{self.pair_group}: Confirmation SMT set - {smt_data['direction']} {smt_data['cycle']} {smt_data['quarters']}")
            return True
        return False
    
    def set_psp(self, psp_data):
        """Set PSP confirmation"""
        if psp_data and self.confirmation_smt:
            self.psp_signal = psp_data
            self.current_state = "PSP_FOUND"
            self.signal_strength += 1
            logger.info(f"{self.pair_group}: PSP confirmed - {psp_data['timeframe']}")
            return True
        return False
    
    def is_signal_ready(self):
        """Check if we have complete signal"""
        return (self.primary_smt is not None and 
                self.confirmation_smt is not None and 
                self.psp_signal is not None and
                self.signal_strength >= 5)
    
    def get_signal_details(self):
        """Get complete signal details"""
        if not self.is_signal_ready():
            return None
            
        return {
            'pair_group': self.pair_group,
            'direction': self.primary_smt['direction'],
            'strength': self.signal_strength,
            'primary_smt': self.primary_smt,
            'confirmation_smt': self.confirmation_smt,
            'psp': self.psp_signal,
            'timestamp': datetime.now(NY_TZ)
        }
    
    def get_next_required_action(self):
        """Get what we need to look for next"""
        if self.current_state == "IDLE":
            return "SCAN_HTF_SMT"
        elif self.current_state == "HTF_FOUND":
            return "SCAN_LTF_SMT"
        elif self.current_state == "LTF_FOUND":
            return "SCAN_PSP"
        elif self.current_state == "PSP_FOUND":
            return "SIGNAL_READY"
        return "IDLE"
    
    def reset(self):
        """Reset for new signal"""
        self.primary_smt = None
        self.confirmation_smt = None
        self.psp_signal = None
        self.signal_strength = 0
        self.current_state = "IDLE"
        logger.info(f"{self.pair_group}: Signal builder reset")

# ================================
# TRADING SYSTEM - HIERARCHICAL
# ================================

class HierarchicalTradingSystem:
    def __init__(self, pair_group, pair_config):
        self.pair_group = pair_group
        self.pair_config = pair_config
        self.pair1 = pair_config['pair1']
        self.pair2 = pair_config['pair2']
        
        # Initialize components
        self.cycle_manager = UTC4CycleManager()
        self.smt_detector = SMTDetector()
        self.psp_detector = PSPDetector()
        self.signal_builder = HierarchicalSignalBuilder(pair_group)
        
        # Data storage
        self.market_data = {self.pair1: {}, self.pair2: {}}
        self.quarter_history = {}
        
        # Define hierarchy (higher to lower)
        self.cycle_hierarchy = ['monthly', 'weekly', 'daily', '90min']
        
        logger.info(f"Initialized hierarchical trading system for {self.pair1}/{self.pair2}")
    
    async def run_analysis(self, api_key):
        """Run hierarchical analysis"""
        try:
            # Fetch market data
            await self._fetch_market_data(api_key)
            
            # Get current state and determine what to do next
            next_action = self.signal_builder.get_next_required_action()
            
            if next_action == "SCAN_HTF_SMT":
                await self._scan_htf_smt()
            elif next_action == "SCAN_LTF_SMT":
                await self._scan_ltf_smt()
            elif next_action == "SCAN_PSP":
                await self._scan_psp()
            elif next_action == "SIGNAL_READY":
                signal = self.signal_builder.get_signal_details()
                if signal:
                    logger.info(f"🚨 SIGNAL GENERATED for {self.pair_group}")
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
    
    async def _scan_htf_smt(self):
        """Scan for primary SMT in higher timeframes"""
        logger.info(f"{self.pair_group}: Scanning for HTF SMT...")
        
        # Try from highest to lowest timeframe for primary SMT
        for i, cycle_type in enumerate(self.cycle_hierarchy[:-1]):
            smt_found = await self._check_smt_in_cycle(cycle_type)
            if smt_found:
                logger.info(f"{self.pair_group}: Found HTF SMT in {cycle_type}")
                return True
        
        logger.info(f"{self.pair_group}: No HTF SMT found")
        return False
    
    async def _scan_ltf_smt(self):
        """Scan for confirmation SMT in lower timeframes"""
        if not self.signal_builder.primary_smt:
            return False
            
        primary_cycle = self.signal_builder.primary_smt['cycle']
        primary_direction = self.signal_builder.primary_smt['direction']
        
        # Find lower timeframes than the primary
        primary_index = self.cycle_hierarchy.index(primary_cycle)
        lower_cycles = self.cycle_hierarchy[primary_index + 1:]
        
        logger.info(f"{self.pair_group}: Scanning for LTF SMT in {lower_cycles}...")
        
        for cycle_type in lower_cycles:
            smt_found = await self._check_smt_in_cycle(cycle_type, required_direction=primary_direction)
            if smt_found:
                logger.info(f"{self.pair_group}: Found LTF SMT in {cycle_type}")
                return True
        
        logger.info(f"{self.pair_group}: No LTF SMT found yet")
        return False
    
    async def _scan_psp(self):
        """Scan for PSP confirmation"""
        if not self.signal_builder.confirmation_smt:
            return False
            
        # Use the confirmation SMT's timeframe for PSP scanning
        confirmation_cycle = self.signal_builder.confirmation_smt['cycle']
        timeframe = self.pair_config['timeframe_mapping'][confirmation_cycle]
        
        logger.info(f"{self.pair_group}: Scanning for PSP on {timeframe}...")
        
        pair1_data = self.market_data[self.pair1].get(confirmation_cycle)
        pair2_data = self.market_data[self.pair2].get(confirmation_cycle)
        
        if pair1_data is None or pair2_data is None:
            return False
        
        psp_signal = self.psp_detector.detect_psp_in_history(pair1_data, pair2_data, timeframe, lookback=5)
        if psp_signal:
            return self.signal_builder.set_psp(psp_signal)
        
        logger.info(f"{self.pair_group}: No PSP found yet")
        return False
    
    async def _check_smt_in_cycle(self, cycle_type, required_direction=None):
        """Check for SMT in a specific cycle (simplified for now)"""
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
            if self.signal_builder.current_state == "IDLE":
                return self.signal_builder.set_primary_smt(smt_data)
            else:
                return self.signal_builder.set_confirmation_smt(smt_data)
                
        except Exception as e:
            logger.error(f"Error checking SMT in {cycle_type} for {self.pair_group}: {str(e)}")
            return False

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
            self.trading_systems[pair_group] = HierarchicalTradingSystem(pair_group, pair_config)
        
        logger.info(f"Initialized multi-pair manager with {len(self.trading_systems)} pair groups")
    
    async def run_all_systems(self):
        """Run all trading systems in parallel"""
        logger.info("Starting hierarchical multi-pair trading analysis...")
        
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
        """Format signal for Telegram with proper hierarchy details"""
        pair_group = signal.get('pair_group', 'Unknown')
        direction = signal.get('direction', 'UNKNOWN').upper()
        strength = signal.get('strength', 0)
        
        message = f"🚨 *TRADING SIGNAL* 🚨\n\n"
        message += f"*Pair Group:* {pair_group.replace('_', ' ').title()}\n"
        message += f"*Direction:* {direction}\n"
        message += f"*Strength:* {strength}/6\n\n"
        
        # Primary SMT
        if 'primary_smt' in signal:
            psmt = signal['primary_smt']
            message += f"*Primary SMT:* {psmt['direction']} {pair_group} {psmt['cycle']} cycle {psmt['quarters']}\n"
        
        # Confirmation SMT
        if 'confirmation_smt' in signal:
            csmt = signal['confirmation_smt']
            message += f"*Confirmation SMT:* {csmt['direction']} {csmt['cycle']} cycle {csmt['quarters']}\n"
        
        # PSP
        if 'psp' in signal:
            psp = signal['psp']
            message += f"*PSP Confirmation:* {psp['timeframe']} {pair_group}\n"
        
        message += f"\n*Time:* {datetime.now(NY_TZ).strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
        message += f"\n#TradingSignal #{pair_group}"
        
        return message
    
    def _calculate_sleep_interval(self):
        """Calculate adaptive sleep interval based on system states"""
        base_interval = BASE_INTERVAL
        min_interval = MIN_INTERVAL
        
        # Check active signal builders
        active_builders = 0
        for system in self.trading_systems.values():
            if system.signal_builder.current_state != "IDLE":
                active_builders += 1
                # If we're close to signal, reduce interval more
                if system.signal_builder.current_state == "PSP_FOUND":
                    active_builders += 2  # Extra weight for PSP waiting
        
        if active_builders > 0:
            return max(min_interval, base_interval // (active_builders + 1))
        
        return base_interval

# ================================
# MAIN EXECUTION
# ================================

async def main():
    """Main entry point"""
    logger.info("Starting Hierarchical Multi-Pair SMT Trading System")
    
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
