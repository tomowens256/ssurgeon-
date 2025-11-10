#!/usr/bin/env python3
"""
MULTI-PAIR SMT TRADING SYSTEM - DEBUGGED VERSION
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

# Trading pairs configuration - USING VALID OANDA INSTRUMENTS
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
        'pair1': 'US30_USD',  # Dow Jones
        'pair2': 'US500_USD', # S&P 500 (FIXED from SPX500_USD)
        'timeframe_mapping': {
            'monthly': 'H4',
            'weekly': 'H1',
            'daily': 'M15', 
            '90min': 'M5'
        }
    },
    'european_indices': {
        'pair1': 'DE30_EUR',  # DAX (FIXED from GER40_EUR)
        'pair2': 'EU50_EUR',  # Euro Stoxx 50
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
# UTILITY FUNCTIONS - FIXED
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
    """Fetch candles from OANDA API - FIXED INSTRUMENT NAMES"""
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
# PATTERN DETECTORS - FIXED
# ================================

class EnhancedFVGDetector:
    @staticmethod
    def detect_all_fvgs(df):
        """Detect ALL Fair Value Gaps in the entire DataFrame - FIXED"""
        fvgs = []
        
        if len(df) < 3:
            return fvgs
            
        for i in range(len(df) - 2):
            c1, c2, c3 = df.iloc[i], df.iloc[i+1], df.iloc[i+2]
            
            # Extract scalar values to avoid Series comparison
            c1_high = float(c1['high'])
            c1_low = float(c1['low'])
            c2_close = float(c2['close'])
            c2_open = float(c2['open'])
            c3_high = float(c3['high'])
            c3_low = float(c3['low'])
            
            # Bullish FVG: c2 is up candle AND c1 high < c3 low
            if (c2_close > c2_open and c1_high < c3_low):
                fvgs.append({
                    'type': 'bullish',
                    'gap_low': c1_high,
                    'gap_high': c3_low,
                    'timestamp': c2['time'],
                    'candle_index': i+1
                })
            
            # Bearish FVG: c2 is down candle AND c1 low > c3 high  
            elif (c2_close < c2_open and c1_low > c3_high):
                fvgs.append({
                    'type': 'bearish', 
                    'gap_high': c1_low,
                    'gap_low': c3_high,
                    'timestamp': c2['time'],
                    'candle_index': i+1
                })
        
        return fvgs

class SMTDetector:
    def __init__(self):
        self.smt_history = []
    
    def detect_smt(self, asset1_data, asset2_data, cycle_type):
        """Simplified SMT detection for debugging - FIXED SERIES COMPARISON"""
        try:
            if asset1_data.empty or asset2_data.empty:
                return None
            
            # Use last few candles for demonstration
            if len(asset1_data) < 10 or len(asset2_data) < 10:
                return None
            
            # Get recent highs and lows as SCALAR values
            asset1_recent_high = float(asset1_data['high'].tail(5).max())
            asset1_prev_high = float(asset1_data['high'].head(5).max())
            asset2_recent_high = float(asset2_data['high'].tail(5).max())
            asset2_prev_high = float(asset2_data['high'].head(5).max())
            
            asset1_recent_low = float(asset1_data['low'].tail(5).min())
            asset1_prev_low = float(asset1_data['low'].head(5).min())
            asset2_recent_low = float(asset2_data['low'].tail(5).min())
            asset2_prev_low = float(asset2_data['low'].head(5).min())
            
            # Bearish SMT: Asset1 makes HH, Asset2 doesn't
            bearish_condition1 = (asset1_recent_high > asset1_prev_high and 
                                 asset2_recent_high <= asset2_prev_high)
            
            bearish_condition2 = (asset1_recent_low > asset1_prev_low and 
                                 asset2_recent_low <= asset2_prev_low)
            
            # Bullish SMT: Asset1 makes LL, Asset2 doesn't
            bullish_condition1 = (asset1_recent_low < asset1_prev_low and 
                                 asset2_recent_low >= asset2_prev_low)
            
            bullish_condition2 = (asset1_recent_high < asset1_prev_high and 
                                 asset2_recent_high >= asset2_prev_high)
            
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
                'quarters': "q1→q2",  # Simplified for now
                'timestamp': datetime.now(NY_TZ),
                'strength': 1
            }
            
            self.smt_history.append(smt_data)
            logger.info(f"SMT detected: {direction} {cycle_type}")
            return smt_data
            
        except Exception as e:
            logger.error(f"Error in SMT detection for {cycle_type}: {str(e)}")
            return None

class PSPDetector:
    @staticmethod
    def detect_psp(asset1_data, asset2_data, timeframe):
        """PSP detection - FIXED SERIES COMPARISON"""
        try:
            if asset1_data.empty or asset2_data.empty:
                return None
            
            # Get the most recent complete candle
            recent_asset1 = asset1_data[asset1_data['complete'] == True].tail(1)
            recent_asset2 = asset2_data[asset2_data['complete'] == True].tail(1)
            
            if recent_asset1.empty or recent_asset2.empty:
                return None
            
            # Extract scalar values
            asset1_candle = recent_asset1.iloc[0]
            asset2_candle = recent_asset2.iloc[0]
            
            asset1_color = 'green' if float(asset1_candle['close']) > float(asset1_candle['open']) else 'red'
            asset2_color = 'green' if float(asset2_candle['close']) > float(asset2_candle['open']) else 'red'
            
            if asset1_color != asset2_color:
                psp_data = {
                    'timeframe': timeframe,
                    'asset1_color': asset1_color,
                    'asset2_color': asset2_color,
                    'timestamp': datetime.now(NY_TZ)
                }
                logger.info(f"PSP detected on {timeframe}: {asset1_color}/{asset2_color}")
                return psp_data
            
            return None
        except Exception as e:
            logger.error(f"Error in PSP detection: {str(e)}")
            return None

# ================================
# SIGNAL BUILDER - FIXED
# ================================

class AdvancedSignalBuilder:
    def __init__(self):
        self.accumulated_smts = []
        self.psp_detected = False
        self.psp_data = None
        self.signal_strength = 0
        
    def add_smt(self, smt_data):
        if smt_data and smt_data not in self.accumulated_smts:
            self.accumulated_smts.append(smt_data)
            self._calculate_strength()
            logger.info(f"Added SMT: {smt_data['direction']} {smt_data['cycle']}")
    
    def add_psp(self, psp_data):
        if psp_data:
            self.psp_detected = True
            self.psp_data = psp_data
            self._calculate_strength()
            logger.info("Added PSP confirmation")
    
    def _calculate_strength(self):
        strength = 0
        cycle_weights = {'monthly': 3, 'weekly': 2, 'daily': 1, '90min': 1}
        
        for smt in self.accumulated_smts:
            strength += cycle_weights.get(smt['cycle'], 1)
            strength += smt.get('strength', 0)
        
        if self.psp_detected:
            strength += 2
            
        self.signal_strength = min(strength, 10)  # Cap at 10
    
    def is_signal_ready(self):
        if len(self.accumulated_smts) < 2:
            return False
            
        unique_cycles = len(set(smt['cycle'] for smt in self.accumulated_smts))
        same_direction = len(set(smt['direction'] for smt in self.accumulated_smts)) == 1
        
        return (unique_cycles >= 2 and 
                same_direction and
                self.psp_detected and 
                self.signal_strength >= 3)  # Lowered threshold for testing
    
    def get_signal_details(self):
        direction = self.accumulated_smts[0]['direction'] if self.accumulated_smts else 'unknown'
        
        return {
            'direction': direction,
            'strength': self.signal_strength,
            'smts': self.accumulated_smts.copy(),
            'psp': self.psp_data,
            'timestamp': datetime.now(NY_TZ)
        }
    
    def reset(self):
        self.accumulated_smts = []
        self.psp_detected = False
        self.psp_data = None
        self.signal_strength = 0
        logger.info("Signal builder reset")

# ================================
# TRADING SYSTEM - FIXED
# ================================

class PairTradingSystem:
    def __init__(self, pair_group, pair_config):
        self.pair_group = pair_group
        self.pair_config = pair_config
        self.pair1 = pair_config['pair1']
        self.pair2 = pair_config['pair2']
        
        # Initialize components
        self.cycle_manager = UTC4CycleManager()
        self.fvg_detector = EnhancedFVGDetector()
        self.smt_detector = SMTDetector()
        self.psp_detector = PSPDetector()
        self.signal_builder = AdvancedSignalBuilder()
        
        # Data storage
        self.market_data = {self.pair1: {}, self.pair2: {}}
        
        logger.info(f"Initialized trading system for {self.pair1}/{self.pair2}")
    
    async def run_analysis(self, api_key):
        """Run single analysis cycle for this pair group - FIXED ERROR HANDLING"""
        try:
            # Fetch market data
            await self._fetch_market_data(api_key)
            
            # Update current quarters
            current_quarters = self.cycle_manager.detect_current_quarters()
            
            # Process each cycle
            for cycle_type in current_quarters.keys():
                await self._analyze_cycle(cycle_type)
            
            # Check if signal is complete
            if self.signal_builder.is_signal_ready():
                final_signal = self.signal_builder.get_signal_details()
                final_signal['pair_group'] = self.pair_group
                logger.info(f"🚨 SIGNAL GENERATED for {self.pair_group}: {final_signal['direction']} strength {final_signal['strength']}")
                
                # Reset for next signal
                self.signal_builder.reset()
                return final_signal
            
            return None
            
        except Exception as e:
            logger.error(f"Error in analysis for {self.pair_group}: {str(e)}")
            return None
    
    async def _fetch_market_data(self, api_key):
        """Fetch market data for both pairs - FIXED"""
        timeframes = self.pair_config['timeframe_mapping']
        
        for pair in [self.pair1, self.pair2]:
            for cycle, tf in timeframes.items():
                try:
                    # Use synchronous fetch in thread
                    df = await asyncio.get_event_loop().run_in_executor(
                        None, fetch_candles, pair, tf, 100, api_key  # Reduced count for testing
                    )
                    if not df.empty:
                        self.market_data[pair][cycle] = df
                        logger.debug(f"Fetched {len(df)} candles for {pair} {tf}")
                    else:
                        logger.warning(f"No data for {pair} {tf}")
                except Exception as e:
                    logger.error(f"Error fetching {pair} {tf}: {str(e)}")
    
    async def _analyze_cycle(self, cycle_type):
        """Analyze specific cycle for patterns - FIXED ERROR HANDLING"""
        timeframe = self.pair_config['timeframe_mapping'][cycle_type]
        
        # Get data for current cycle
        pair1_data = self.market_data[self.pair1].get(cycle_type)
        pair2_data = self.market_data[self.pair2].get(cycle_type)
        
        if pair1_data is None or pair2_data is None:
            logger.debug(f"No data for {cycle_type} in {self.pair_group}")
            return
        
        try:
            # SMT detection
            smt_signal = self.smt_detector.detect_smt(pair1_data, pair2_data, cycle_type)
            if smt_signal:
                self.signal_builder.add_smt(smt_signal)
            
            # PSP detection
            psp_signal = self.psp_detector.detect_psp(pair1_data, pair2_data, timeframe)
            if psp_signal:
                self.signal_builder.add_psp(psp_signal)
                        
        except Exception as e:
            logger.error(f"Error analyzing {cycle_type} for {self.pair_group}: {str(e)}")

# ================================
# MAIN MANAGER - FIXED
# ================================

class MultiPairTradingManager:
    def __init__(self, api_key, telegram_token, chat_id):
        self.api_key = api_key
        self.telegram_token = telegram_token
        self.chat_id = chat_id
        self.trading_systems = {}
        
        # Initialize trading systems for all pairs
        for pair_group, pair_config in TRADING_PAIRS.items():
            self.trading_systems[pair_group] = PairTradingSystem(pair_group, pair_config)
        
        logger.info(f"Initialized multi-pair manager with {len(self.trading_systems)} pair groups")
    
    async def run_all_systems(self):
        """Run all trading systems in parallel - FIXED"""
        logger.info("Starting multi-pair trading analysis...")
        
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
                else:
                    logger.info("No signals found this cycle")
                
                # Calculate adaptive sleep interval
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
        """Format signal for Telegram"""
        pair_group = signal.get('pair_group', 'Unknown')
        direction = signal.get('direction', 'UNKNOWN').upper()
        strength = signal.get('strength', 0)
        
        message = f"🚨 *TRADING SIGNAL* 🚨\n\n"
        message += f"*Pair Group:* {pair_group.replace('_', ' ').title()}\n"
        message += f"*Direction:* {direction}\n"
        message += f"*Strength:* {strength}/10\n"
        message += f"*Time:* {datetime.now(NY_TZ).strftime('%Y-%m-%d %H:%M:%S %Z')}\n\n"
        
        # Add SMT details
        if 'smts' in signal:
            message += "*SMT Patterns:*\n"
            for smt in signal['smts']:
                message += f"• {smt['cycle']} {smt['type']} ({smt['quarters']})\n"
        
        # Add PSP confirmation
        if signal.get('psp'):
            message += f"\n*PSP Confirmation:* ✅\n"
        
        message += f"\n#TradingSignal #{pair_group}"
        
        return message
    
    def _calculate_sleep_interval(self):
        """Calculate adaptive sleep interval based on signal activity"""
        base_interval = BASE_INTERVAL
        
        # Check if any system has active signal building
        active_builders = 0
        for system in self.trading_systems.values():
            if system.signal_builder.signal_strength > 0:
                active_builders += 1
        
        if active_builders > 0:
            return max(MIN_INTERVAL, base_interval // (active_builders + 1))
        
        return base_interval

# ================================
# MAIN EXECUTION
# ================================

async def main():
    """Main entry point"""
    logger.info("Starting Multi-Pair SMT Trading System - DEBUGGED VERSION")
    
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
