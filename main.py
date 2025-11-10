#!/usr/bin/env python3
"""
MULTI-PAIR SMT TRADING SYSTEM
Advanced algorithmic trading system that detects SMT patterns across multiple correlated pairs
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
# CONFIGURATION - UPDATED WITH CORRECT OANDA INSTRUMENTS
# ================================

# Trading pairs configuration - UPDATED WITH CORRECT NAMES
TRADING_PAIRS = {
    'precious_metals': {
        'pair1': 'XAU_USD',  # Gold
        'pair2': 'XAG_USD',  # Silver
        'timeframe_mapping': {
            'monthly': 'H4',
            'weekly': 'H1', 
            'daily': 'M15',
            '90min': 'M5'
        }
    },
    'us_indices': {
        'pair1': 'US30_USD',  # Dow Jones - CORRECT
        'pair2': 'SPX500_USD', # S&P 500 - CORRECT NAME
        'timeframe_mapping': {
            'monthly': 'H4',
            'weekly': 'H1',
            'daily': 'M15', 
            '90min': 'M5'
        }
    },
    'european_indices': {
        'pair1': 'DE30_EUR',  # DAX - CORRECT
        'pair2': 'EU50_EUR',  # Euro Stoxx 50 - CORRECT
        'timeframe_mapping': {
            'monthly': 'H4',
            'weekly': 'H1',
            'daily': 'M15',
            '90min': 'M5'
        }
    }
}

# System Configuration
NY_TZ = pytz.timezone('America/New_York')  # UTC-4
BASE_INTERVAL = 300  # 5 minutes
MIN_INTERVAL = 30    # 30 seconds
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
            logger.debug(f"Telegram attempt {attempt+1}/{MAX_RETRIES}")
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
            logger.debug(f"Fetch attempt {attempt+1}/{MAX_RETRIES} for {instrument}")
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
            
            logger.debug(f"Returning {len(df)} candles for {instrument}")
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
# CYCLE MANAGER
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
        if weekday == 0: return 'q1'      # Monday
        elif weekday == 1: return 'q2'    # Tuesday
        elif weekday == 2: return 'q3'    # Wednesday
        elif weekday == 3: return 'q4'    # Thursday
        else: return 'q_less'             # Friday
    
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
            'q1': 18 * 60,  # 18:00
            'q2': 0,        # 00:00  
            'q3': 6 * 60,   # 06:00
            'q4': 12 * 60   # 12:00
        }[daily_quarter]
        
        segment = (minute_of_day - daily_quarter_start) // 90
        return f'q{segment + 1}' if segment < 4 else 'q_less'

# ================================
# PATTERN DETECTORS
# ================================

class EnhancedFVGDetector:
    @staticmethod
    def detect_all_fvgs(df):
        """Detect ALL Fair Value Gaps in the entire DataFrame"""
        fvgs = []
        
        if len(df) < 3:
            return fvgs
            
        for i in range(len(df) - 2):
            c1, c2, c3 = df.iloc[i], df.iloc[i+1], df.iloc[i+2]
            
            # Bullish FVG: c2 is up candle AND c1 high < c3 low
            if (c2['close'] > c2['open'] and 
                c1['high'] < c3['low']):
                fvgs.append({
                    'type': 'bullish',
                    'gap_low': c1['high'],
                    'gap_high': c3['low'],
                    'timestamp': c2['time'],
                    'candle_index': i+1
                })
            
            # Bearish FVG: c2 is down candle AND c1 low > c3 high  
            elif (c2['close'] < c2['open'] and 
                  c1['low'] > c3['high']):
                fvgs.append({
                    'type': 'bearish', 
                    'gap_high': c1['low'],
                    'gap_low': c3['high'],
                    'timestamp': c2['time'],
                    'candle_index': i+1
                })
        
        return fvgs
    
    @staticmethod
    def check_fvg_fill(current_candle, fvg):
        if fvg['type'] == 'bullish':
            return (current_candle['low'] <= fvg['gap_high'] and 
                    current_candle['high'] >= fvg['gap_low'])
        else:  # bearish
            return (current_candle['low'] <= fvg['gap_high'] and 
                    current_candle['high'] >= fvg['gap_low'])

class SMTDetector:
    def __init__(self):
        self.smt_history = []
    
    def detect_smt(self, asset1_data, asset2_data, cycle_type, current_quarter, prev_quarter):
        """Detect SMT between two consecutive quarters"""
        try:
            # Get quarter data (simplified - you'd implement proper quarter tracking)
            q1_a1 = self._get_quarter_stats(asset1_data, prev_quarter)
            q2_a1 = self._get_quarter_stats(asset1_data, current_quarter)
            q1_a2 = self._get_quarter_stats(asset2_data, prev_quarter)
            q2_a2 = self._get_quarter_stats(asset2_data, current_quarter)
            
            if not all([q1_a1, q2_a1, q1_a2, q2_a2]):
                return None
            
            # Bearish SMT detection
            bearish = self._check_bearish_smt(q1_a1, q2_a1, q1_a2, q2_a2)
            bullish = self._check_bullish_smt(q1_a1, q2_a1, q1_a2, q2_a2)
            
            if bearish:
                direction = 'bearish'
                smt_type = 'regular'
            elif bullish:
                direction = 'bullish'
                smt_type = 'regular'
            else:
                return None
            
            smt_data = {
                'direction': direction,
                'type': smt_type,
                'cycle': cycle_type,
                'quarters': f"{prev_quarter}→{current_quarter}",
                'timestamp': datetime.now(NY_TZ)
            }
            
            self.smt_history.append(smt_data)
            return smt_data
            
        except Exception as e:
            logger.error(f"Error in SMT detection: {str(e)}")
            return None
    
    def _get_quarter_stats(self, df, quarter):
        """Simplified quarter stats - implement proper quarter tracking"""
        if df.empty:
            return None
        return {
            'high': df['high'].max(),
            'low': df['low'].min(),
            'close': df['close'].iloc[-1]
        }
    
    def _check_bearish_smt(self, q1_a1, q2_a1, q1_a2, q2_a2):
        condition1 = (q2_a1['high'] > q1_a1['high'] and 
                     q2_a2['high'] <= q1_a2['high'])
        condition2 = (q2_a1['close'] > q1_a1['high'] and 
                     q2_a2['close'] <= q1_a2['high'])
        return condition1 or condition2
    
    def _check_bullish_smt(self, q1_a1, q2_a1, q1_a2, q2_a2):
        condition1 = (q2_a1['low'] < q1_a1['low'] and 
                     q2_a2['low'] >= q1_a2['low'])
        condition2 = (q2_a1['close'] < q1_a1['low'] and 
                     q2_a2['close'] >= q1_a2['low'])
        return condition1 or condition2

class PSPDetector:
    @staticmethod
    def detect_psp(asset1_candle, asset2_candle, timeframe):
        if not asset1_candle or not asset2_candle:
            return False
        asset1_color = 'green' if asset1_candle['close'] > asset1_candle['open'] else 'red'
        asset2_color = 'green' if asset2_candle['close'] > asset2_candle['open'] else 'red'
        return asset1_color != asset2_color

# ================================
# SIGNAL BUILDER
# ================================

class AdvancedSignalBuilder:
    def __init__(self):
        self.accumulated_smts = []
        self.psp_detected = False
        self.signal_strength = 0
        self.required_conditions = {
            'smt_weekly': False,
            'smt_daily': False, 
            'smt_monthly': False,
            'smt_90min': False,
            'psp': False
        }
    
    def add_smt(self, smt_data):
        self.accumulated_smts.append(smt_data)
        
        # Update required conditions
        for smt in self.accumulated_smts:
            self.required_conditions[f"smt_{smt['cycle']}"] = True
        
        self._calculate_strength()
    
    def add_psp(self, psp_data):
        self.psp_detected = True
        self.required_conditions['psp'] = True
        self._calculate_strength()
    
    def _calculate_strength(self):
        strength = 0
        cycle_weights = {'monthly': 3, 'weekly': 2, 'daily': 1, '90min': 1}
        for smt in self.accumulated_smts:
            strength += cycle_weights.get(smt['cycle'], 1)
        if self.psp_detected:
            strength += 2
        self.signal_strength = strength
    
    def is_signal_ready(self):
        unique_cycles = len(set(smt['cycle'] for smt in self.accumulated_smts))
        return (unique_cycles >= 2 and 
                self.psp_detected and 
                self.signal_strength >= 5)
    
    def get_signal_details(self):
        return {
            'direction': self.accumulated_smts[0]['direction'] if self.accumulated_smts else None,
            'strength': self.signal_strength,
            'smts': self.accumulated_smts.copy(),
            'psp': self.psp_detected,
            'timestamp': datetime.now(NY_TZ)
        }
    
    def reset(self):
        self.accumulated_smts = []
        self.psp_detected = False
        self.signal_strength = 0
        for key in self.required_conditions:
            self.required_conditions[key] = False

# ================================
# TRADING SYSTEM
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
        """Run single analysis cycle for this pair group"""
        try:
            # Fetch market data
            await self._fetch_market_data(api_key)
            
            # Update current quarters
            current_quarters = self.cycle_manager.detect_current_quarters()
            
            # Process each cycle
            for cycle_type, quarter_name in current_quarters.items():
                await self._analyze_cycle(cycle_type, quarter_name)
            
            # Check if signal is complete
            if self.signal_builder.is_signal_ready():
                final_signal = self.signal_builder.get_signal_details()
                final_signal['pair_group'] = self.pair_group
                logger.info(f"Signal generated for {self.pair_group}: {final_signal}")
                
                # Reset for next signal
                self.signal_builder.reset()
                return final_signal
            
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
                    # Run in thread to avoid blocking
                    df = await asyncio.get_event_loop().run_in_executor(
                        None, fetch_candles, pair, tf, 300, api_key
                    )
                    if not df.empty:
                        self.market_data[pair][cycle] = df
                    else:
                        logger.warning(f"No data for {pair} {tf}")
                except Exception as e:
                    logger.error(f"Error fetching {pair} {tf}: {str(e)}")
    
    async def _analyze_cycle(self, cycle_type, quarter_name):
        """Analyze specific cycle for patterns"""
        timeframe = self.pair_config['timeframe_mapping'][cycle_type]
        
        # Get data for current cycle
        pair1_data = self.market_data[self.pair1].get(cycle_type)
        pair2_data = self.market_data[self.pair2].get(cycle_type)
        
        if pair1_data is None or pair2_data is None or pair1_data.empty or pair2_data.empty:
            return
        
        try:
            # Simplified SMT detection (you'd implement proper quarter tracking)
            if len(pair1_data) > 10 and len(pair2_data) > 10:
                # Use last 2 "quarters" as demonstration
                smt_signal = self.smt_detector.detect_smt(
                    pair1_data, pair2_data, cycle_type, 'q2', 'q1'
                )
                if smt_signal:
                    self.signal_builder.add_smt(smt_signal)
            
            # PSP detection on current candle
            if not pair1_data.empty and not pair2_data.empty:
                current_candle1 = pair1_data.iloc[-1] if not pair1_data.empty else None
                current_candle2 = pair2_data.iloc[-1] if not pair2_data.empty else None
                
                if current_candle1 is not None and current_candle2 is not None:
                    psp_detected = self.psp_detector.detect_psp(
                        current_candle1, current_candle2, timeframe
                    )
                    if psp_detected:
                        self.signal_builder.add_psp({'timeframe': timeframe})
                        
        except Exception as e:
            logger.error(f"Error analyzing {cycle_type} for {self.pair_group}: {str(e)}")

# ================================
# MAIN MANAGER
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
        """Run all trading systems in parallel"""
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
                for result in results:
                    if isinstance(result, Exception):
                        logger.error(f"Analysis task failed: {str(result)}")
                    elif result is not None:
                        signals.append(result)
                
                # Send signals to Telegram
                if signals:
                    await self._process_signals(signals)
                
                # Calculate adaptive sleep interval
                sleep_time = self._calculate_sleep_interval()
                logger.debug(f"Sleeping for {sleep_time} seconds")
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
        if 'psp' in signal:
            message += f"\n*PSP Confirmation:* {'✅' if signal['psp'] else '❌'}\n"
        
        message += f"\n#TradingSignal #{pair_group}"
        
        return message
    
    def _calculate_sleep_interval(self):
        """Calculate adaptive sleep interval based on signal activity"""
        base_interval = BASE_INTERVAL
        min_interval = MIN_INTERVAL
        
        # Check if any system has active signal building
        active_builders = 0
        for system in self.trading_systems.values():
            if system.signal_builder.signal_strength > 0:
                active_builders += 1
        
        if active_builders > 0:
            return max(min_interval, base_interval // (active_builders + 1))
        
        return base_interval

# ================================
# MAIN EXECUTION
# ================================

async def main():
    """Main entry point"""
    logger.info("Starting Multi-Pair SMT Trading System")
    
    # Get credentials from environment
    api_key = os.getenv('OANDA_API_KEY')
    telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
    telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if not all([api_key, telegram_token, telegram_chat_id]):
        logger.error("Missing required environment variables:")
        logger.error("OANDA_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID")
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
        sys.exit(1)

if __name__ == "__main__":
    # Run the system
    asyncio.run(main())
