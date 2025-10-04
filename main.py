# ========================
# SIMPLIFIED TRADING BOT FOR BUCKET-BASED SIGNALS
# REMOVED ALL MODEL, SCALING, AND UNNECESSARY CODE
# ========================

import os
import time
import threading
import logging
import pytz
import numpy as np
import pandas as pd
import pandas_ta as ta
import requests
import re
import traceback
import sys
from datetime import datetime, timedelta
from oandapyV20 import API
from oandapyV20.exceptions import V20Error
import oandapyV20.endpoints.instruments as instruments
from google.colab import drive
from IPython.display import clear_output

# ========================
# CONSTANTS & CONFIG
# ========================
NY_TZ = pytz.timezone("America/New_York")
DEBUG_MODE = True

# Initialize logging
log_format = '%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s'
logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format=log_format,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('trading_bot_simple.log')
    ]
)
logger = logging.getLogger(__name__)

# Pre-defined bucket combinations for trading signals
M5_TRADE_BUCKETS = {
    "SELL_downtrend_(30,40]",
    "SELL_uptrend_(40,50]", 
    "BUY_downtrend_(50,60]",
    "SELL_sideways_(30,40]",
    "BUY_sideways_(60,70]",
    "BUY_uptrend_(70,80]",
    "SELL_downtrend_(20,30]",
    "BUY_downtrend_(60,70]",
    "BUY_sideways_(70,80]",
    "SELL_uptrend_(30,40]",
    "BUY_uptrend_(80,100]",
    "SELL_sideways_(20,30]"
}

M15_TRADE_BUCKETS = {
    "SELL_downtrend_(30,40]",
    "BUY_uptrend_(60,70]",
    "BUY_downtrend_(50,60]",
    "SELL_uptrend_(40,50]",
    "BUY_sideways_(60,70]",
    "SELL_sideways_(30,40]",
    "BUY_uptrend_(70,80]"
}

# ========================
# UTILITY FUNCTIONS
# ========================
def parse_oanda_time(time_str):
    """Parse Oanda's timestamp with variable fractional seconds"""
    try:
        if '.' in time_str and len(time_str.split('.')[1]) > 7:
            time_str = re.sub(r'\.(\d{6})\d+', r'.\1', time_str)
        return datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=pytz.utc).astimezone(NY_TZ)
    except Exception as e:
        logger.error(f"Error parsing time {time_str}: {str(e)}")
        return datetime.now(NY_TZ)

def send_telegram(message, token, chat_id):
    """Send formatted message to Telegram with detailed error handling and retries"""
    logger.debug(f"Attempting to send Telegram message: {message[:50]}...")
    
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
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            logger.debug(f"Telegram attempt {attempt+1}/{max_retries}")
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
    logger.error(f"Failed to send Telegram message after {max_retries} attempts")
    return False

def fetch_candles(timeframe, last_time=None, count=201, api_key=None):
    """Fetch candles for XAU_USD with full precision and robust error handling"""
    logger.debug(f"Fetching {count} candles for {timeframe}, last_time: {last_time}")
    
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
    if last_time:
        params["from"] = last_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    sleep_time = 10
    max_attempts = 5
    
    for attempt in range(max_attempts):
        try:
            logger.debug(f"Fetch attempt {attempt+1}/{max_attempts}")
            request = instruments.InstrumentsCandles(instrument="XAU_USD", params=params)
            response = api.request(request)
            candles = response.get('candles', [])
            
            logger.debug(f"Received {len(candles)} candles")
            
            if not candles:
                logger.warning(f"No candles received on attempt {attempt+1}")
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
                    logger.error(f"Error parsing candle: {str(e)}")
                    continue
            
            if not data:
                logger.warning(f"Empty data after parsing on attempt {attempt+1}")
                continue
                
            df = pd.DataFrame(data).drop_duplicates(subset=['time'], keep='last')
            df = df.reset_index(drop=True)
            if last_time:
                df = df[df['time'] > last_time].sort_values('time')
                
            logger.debug(f"Returning {len(df)} candles")
            return df
            
        except V20Error as e:
            if "rate" in str(e).lower() or getattr(e, 'code', 0) in [429, 502]:
                wait_time = sleep_time * (2 ** attempt)
                logger.warning(f"Rate limit hit, waiting {wait_time}s: {str(e)}")
                time.sleep(wait_time)
            else:
                error_details = f"Status: {getattr(e, 'code', 'N/A')} | Message: {getattr(e, 'msg', str(e))}"
                logger.error(f"❌ Oanda API error: {error_details}")
                break
                
        except Exception as e:
            logger.error(f"❌ General error fetching candles: {str(e)}")
            logger.error(traceback.format_exc())
            time.sleep(sleep_time)
    
    logger.error(f"Failed to fetch candles after {max_attempts} attempts")
    return pd.DataFrame()

def validate_candle_data(df):
    """Validate candle data quality"""
    if df.empty:
        logger.error("Empty DataFrame received")
        return False
    
    # Check for NaN values
    if df[['open', 'high', 'low', 'close']].isna().any().any():
        logger.error("NaN values detected in price data")
        return False
    
    # Check for zero or negative prices
    if (df[['open', 'high', 'low', 'close']] <= 0).any().any():
        logger.error("Invalid price values detected")
        return False
    
    # Check timestamp monotonicity
    if not df['time'].is_monotonic_increasing:
        logger.warning("Timestamps not monotonically increasing - sorting")
        df = df.sort_values('time').reset_index(drop=True)
    
    logger.debug("Candle data validation passed")
    return True

# ========================
# SIMPLIFIED FEATURE ENGINEER
# ========================
class SimpleFeatureEngineer:
    def __init__(self, timeframe):
        self.timeframe = timeframe
        self.rsi_bins = [0, 20, 30, 40, 50, 60, 70, 80, 100]
        logger.debug(f"Initialized SimpleFeatureEngineer for {timeframe}")

    def calculate_crt_signal(self, df):
        """Calculate CRT signal at OPEN of current candle (c0) with minimal latency"""
        try:
            if len(df) < 3:
                logger.warning("Insufficient data for CRT signal (need at least 3 candles)")
                return None, None

            # Use the last COMPLETED candle as c2 (index -2)
            c1 = df.iloc[-3]
            c2 = df.iloc[-2]
            current_open = df.iloc[-1]['open']

            # Calculate c2 metrics
            c2_range = c2['high'] - c2['low']
            c2_mid = c2['low'] + (0.5 * c2_range)

            # CRT conditions - using ONLY completed candles and current open
            if (c2['low'] < c1['low'] and 
                c2['close'] > c1['low'] and 
                current_open > c2_mid):
                signal_type = 'BUY'
                entry = current_open
                sl = c2['low']
                risk = abs(entry - sl)
                tp = entry + 4 * risk
                logger.info(f"BUY CRT signal detected: entry={entry:.5f}, sl={sl:.5f}, tp={tp:.5f}")
                return signal_type, {'entry': entry, 'sl': sl, 'tp': tp, 'time': df.iloc[-1]['time']}

            elif (c2['high'] > c1['high'] and 
                  c2['close'] < c1['high'] and 
                  current_open < c2_mid):
                signal_type = 'SELL'
                entry = current_open
                sl = c2['high']
                risk = abs(sl - entry)
                tp = entry - 4 * risk
                logger.info(f"SELL CRT signal detected: entry={entry:.5f}, sl={sl:.5f}, tp={tp:.5f}")
                return signal_type, {'entry': entry, 'sl': sl, 'tp': tp, 'time': df.iloc[-1]['time']}

            return None, None
        except Exception as e:
            logger.error(f"Error in calculate_crt_signal: {str(e)}")
            logger.error(traceback.format_exc())
            return None, None

    def calculate_trend_direction(self, df):
        """Calculate trend direction using moving averages"""
        try:
            # Calculate moving averages
            df['ma_20'] = df['close'].rolling(window=20).mean()
            df['ma_30'] = df['close'].rolling(window=30).mean()
            df['ma_40'] = df['close'].rolling(window=40).mean()
            df['ma_60'] = df['close'].rolling(window=60).mean()
            
            # Get current values
            current = df.iloc[-1]
            
            # Determine trend direction
            if (current['ma_20'] > current['ma_30'] > current['ma_40'] > current['ma_60']):
                trend = 'uptrend'
            elif (current['ma_20'] < current['ma_30'] < current['ma_40'] < current['ma_60']):
                trend = 'downtrend'
            else:
                trend = 'sideways'
            
            logger.debug(f"Trend calculated: {trend}")
            return trend
                
        except Exception as e:
            logger.error(f"Error calculating trend direction: {str(e)}")
            return 'sideways'

    def calculate_rsi_bucket(self, df):
        """Calculate RSI and return the appropriate bucket - EXACT format matching historical analysis"""
        try:
            rsi = ta.rsi(df['close'], length=14).iloc[-1]
            
            if pd.isna(rsi):
                logger.warning("RSI calculation returned NaN")
                return 'nan'
            
            # CRITICAL: Exact bin format matching historical analysis
            for i in range(len(self.rsi_bins)-1):
                if self.rsi_bins[i] <= rsi < self.rsi_bins[i+1]:
                    # ✅ EXACT format: "(30,40]" - no spaces, exact brackets
                    bucket = f"({self.rsi_bins[i]},{self.rsi_bins[i+1]}]"
                    logger.debug(f"RSI {rsi:.2f} -> bucket {bucket}")
                    return bucket
            
            logger.warning(f"RSI {rsi:.2f} outside expected bins")
            return 'nan'
        except Exception as e:
            logger.error(f"Error calculating RSI bucket: {str(e)}")
            return 'nan'

    def generate_bucket_key(self, df, signal_type):
        """Generate bucket key for signal validation - EXACT format matching historical analysis"""
        try:
            trend = self.calculate_trend_direction(df)
            rsi_bucket = self.calculate_rsi_bucket(df)
            
            # ✅ CRITICAL: Exact format matching your historical analysis
            # Format: "BUY_uptrend_(70,80]" 
            bucket_key = f"{signal_type}_{trend}_{rsi_bucket}"
            
            logger.debug(f"Generated bucket key: {bucket_key}")
            
            return bucket_key
            
        except Exception as e:
            logger.error(f"Error generating bucket key: {str(e)}")
            return None

    def debug_bucket_generation(self, df, signal_type):
        """Temporary debug function to see bucket components"""
        trend = self.calculate_trend_direction(df)
        rsi_bucket = self.calculate_rsi_bucket(df)
        current_rsi = ta.rsi(df['close'], length=14).iloc[-1]
        
        logger.info("=== BUCKET GENERATION DEBUG ===")
        logger.info(f"Signal type: {signal_type}")
        logger.info(f"Current RSI: {current_rsi:.2f}")
        logger.info(f"Trend: {trend}")
        logger.info(f"RSI bucket: {rsi_bucket}")
        
        # Test the exact combination
        test_bucket = f"{signal_type}_{trend}_{rsi_bucket}"
        logger.info(f"Generated bucket: {test_bucket}")
        
        return test_bucket

# ========================
# SIMPLIFIED TRADING BOT
# ========================
class SimpleTradingBot:
    def __init__(self, timeframe, credentials, trade_buckets):
        self.timeframe = timeframe
        self.credentials = credentials
        self.trade_buckets = trade_buckets
        self.logger = logging.getLogger(f"{timeframe}_bot")
        self.start_time = time.time()
        self.max_duration = 11.5 * 3600
        
        self.feature_engineer = SimpleFeatureEngineer(timeframe)
        self.data = pd.DataFrame()
        
        # Performance monitoring
        self.performance_stats = {
            'candle_fetches': 0,
            'signals_generated': 0,
            'signals_acted_upon': 0,
            'errors': 0,
            'last_signal_time': None
        }
        
        logger.info(f"Initialized {timeframe} bot with {len(trade_buckets)} trade buckets")

    def calculate_next_candle_time(self):
        now = datetime.now(NY_TZ)
        
        if self.timeframe == "M5":
            minutes_past = now.minute % 5
            next_minute = now.minute - minutes_past + 5
            if next_minute >= 60:
                next_time = now.replace(hour=now.hour+1, minute=0, second=0, microsecond=0)
            else:
                next_time = now.replace(minute=next_minute, second=0, microsecond=0)
        else:  # M15
            minutes_past = now.minute % 15
            next_minute = now.minute - minutes_past + 15
            if next_minute >= 60:
                next_time = now.replace(hour=now.hour+1, minute=0, second=0, microsecond=0)
            else:
                next_time = now.replace(minute=next_minute, second=0, microsecond=0)
        
        # Add network latency compensation
        next_time += timedelta(seconds=0.3)
        
        # If we're at the exact candle start, move to next candle
        if now >= next_time:
            next_time += timedelta(minutes=5 if self.timeframe == "M5" else 15)
        
        logger.debug(f"Current time: {now}, Next candle: {next_time}")
        return next_time

    def send_trade_signal(self, signal_type, signal_data, bucket_key):
        """Send formatted trade signal to Telegram"""
        latency_ms = (datetime.now(NY_TZ) - signal_data['time']).total_seconds() * 1000
        
        message = (
            f"🚨 XAU/USD Trade Signal ({self.timeframe})\n"
            f"Type: {signal_type}\n"
            f"Entry: {signal_data['entry']:.5f}\n"
            f"SL: {signal_data['sl']:.5f}\n"
            f"TP: {signal_data['tp']:.5f}\n"
            f"Bucket: {bucket_key}\n"
            f"Risk/Reward: 1:4\n"
            f"Latency: {latency_ms:.1f}ms\n"
            f"Time: {signal_data['time'].strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        success = send_telegram(message, self.credentials['telegram_token'], self.credentials['telegram_chat_id'])
        if success:
            self.performance_stats['signals_acted_upon'] += 1
            self.performance_stats['last_signal_time'] = datetime.now(NY_TZ)
            logger.info(f"Trade signal sent for bucket: {bucket_key}")
        else:
            logger.error("Failed to send Telegram signal")

    def test_credentials(self):
        """Test both Telegram and Oanda credentials"""
        logger.info("Testing credentials...")
        
        # Test Telegram
        test_msg = f"🔧 {self.timeframe} bot credentials test at {datetime.now(NY_TZ).strftime('%Y-%m-%d %H:%M:%S')}"
        telegram_ok = send_telegram(test_msg, self.credentials['telegram_token'], self.credentials['telegram_chat_id'])
        
        # Test Oanda
        oanda_ok = False
        try:
            test_data = fetch_candles("M5", count=1, api_key=self.credentials['oanda_api_key'])
            oanda_ok = not test_data.empty
            if oanda_ok:
                logger.info(f"Oanda test successful - received {len(test_data)} candles")
            else:
                logger.error("Oanda test failed - no data received")
        except Exception as e:
            logger.error(f"Oanda test failed: {str(e)}")
            
        logger.info(f"Credentials test result: {'PASS' if telegram_ok and oanda_ok else 'FAIL'}")
        return telegram_ok and oanda_ok

    def test_bucket_matching(self):
        """Test that our bucket generation matches expected format"""
        logger.info("=== BUCKET MATCHING TEST ===")
        test_cases = [
            ("BUY", "uptrend", "(70,80]"),
            ("SELL", "downtrend", "(30,40]"),
            ("BUY", "sideways", "(60,70]"),
            ("SELL", "uptrend", "(40,50]"),
        ]
        
        for signal, trend, rsi_bin in test_cases:
            test_key = f"{signal}_{trend}_{rsi_bin}"
            exists = test_key in self.trade_buckets
            status = "✅ EXISTS" if exists else "❌ MISSING"
            logger.info(f"Testing: {test_key} -> {status}")

    def robust_fetch_candles(self, max_retries=5, initial_delay=1):
        """Enhanced fetch with exponential backoff"""
        for attempt in range(max_retries):
            try:
                df = fetch_candles(self.timeframe, count=201, api_key=self.credentials['oanda_api_key'])
                
                if not df.empty and validate_candle_data(df):
                    self.performance_stats['candle_fetches'] += 1
                    return df
                else:
                    delay = initial_delay * (2 ** attempt)
                    logger.warning(f"Empty data, retry {attempt+1}/{max_retries} in {delay}s")
                    time.sleep(delay)
                    
            except Exception as e:
                delay = initial_delay * (2 ** attempt)
                logger.error(f"Fetch attempt {attempt+1} failed: {str(e)}, retrying in {delay}s")
                time.sleep(delay)
        
        logger.error(f"All {max_retries} fetch attempts failed")
        return pd.DataFrame()

    def log_performance_stats(self):
        """Regular performance logging"""
        stats = self.performance_stats
        if stats['candle_fetches'] > 0:
            signal_rate = (stats['signals_generated'] / stats['candle_fetches']) * 100
            action_rate = (stats['signals_acted_upon'] / max(1, stats['signals_generated'])) * 100
            
            logger.info(f"Performance: {stats['candle_fetches']} fetches, "
                       f"{stats['signals_generated']} signals ({signal_rate:.1f}%), "
                       f"{stats['signals_acted_upon']} acted ({action_rate:.1f}%), "
                       f"{stats['errors']} errors")
            
            # Reset stats periodically
            if stats['candle_fetches'] > 1000:
                self.performance_stats = {k: 0 for k in stats.keys()}
                self.performance_stats['last_signal_time'] = stats['last_signal_time']

    def run(self):
        """Main bot execution loop"""
        thread_name = threading.current_thread().name
        logger.info(f"Starting bot thread: {thread_name}")
        
        session_start = time.time()
        start_msg = f"🚀 {self.timeframe} bot started at {datetime.now(NY_TZ).strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Test credentials before starting
        creds_valid = self.test_credentials()
        if not creds_valid:
            logger.error("Credentials test failed. Exiting bot.")
            return
            
        # Test bucket matching
        self.test_bucket_matching()
            
        send_telegram(start_msg, self.credentials['telegram_token'], self.credentials['telegram_chat_id'])
        
        while True:
            try:
                # Check session time remaining
                elapsed = time.time() - session_start
                if elapsed > self.max_duration:
                    logger.warning("Session timeout reached, exiting")
                    end_msg = f"🔴 {self.timeframe} bot session ended after 12 hours"
                    send_telegram(end_msg, self.credentials['telegram_token'], self.credentials['telegram_chat_id'])
                    return
                
                # Calculate precise wakeup time
                now = datetime.now(NY_TZ)
                next_candle = self.calculate_next_candle_time()
                sleep_seconds = max(0, (next_candle - now).total_seconds() - 0.1)
                
                if sleep_seconds > 0:
                    logger.debug(f"Sleeping for {sleep_seconds:.2f} seconds until next candle")
                    time.sleep(sleep_seconds)
                
                # Busy-wait for precise candle open
                while datetime.now(NY_TZ) < next_candle:
                    time.sleep(0.001)
                
                logger.debug("Candle open detected - waiting 5s for candle availability")
                time.sleep(5)
                
                # Fetch candles for analysis
                logger.debug("Fetching candles for analysis")
                new_data = self.robust_fetch_candles()
                
                if new_data.empty:
                    logger.error("Failed to fetch candle data after retries")
                    self.performance_stats['errors'] += 1
                    continue
                    
                self.data = new_data
                logger.debug(f"Total records: {len(self.data)}")
                
                # Detect CRT pattern
                signal_type, signal_data = self.feature_engineer.calculate_crt_signal(self.data)
                
                if not signal_type:
                    logger.debug("No CRT pattern detected")
                    continue
                    
                self.performance_stats['signals_generated'] += 1
                logger.info(f"CRT pattern detected: {signal_type} at {signal_data['time']}")
                
                # ✅ FIXED: Use debug bucket generation temporarily to see exactly what's happening
                bucket_key = self.feature_engineer.debug_bucket_generation(self.data, signal_type)
                
                if bucket_key and bucket_key in self.trade_buckets:
                    logger.info(f"✅ BUCKET MATCH FOUND: {bucket_key} - Sending trade signal")
                    self.send_trade_signal(signal_type, signal_data, bucket_key)
                else:
                    logger.warning(f"❌ Bucket {bucket_key} not in trade list - ignoring signal")
                    # Log available buckets for debugging
                    matching_buckets = [b for b in self.trade_buckets if signal_type in b]
                    logger.info(f"Available {signal_type} buckets: {matching_buckets}")
                
                # Log performance every 10 candles
                if self.performance_stats['candle_fetches'] % 10 == 0:
                    self.log_performance_stats()
                    
            except Exception as e:
                error_msg = f"❌ {self.timeframe} bot error: {str(e)}"
                logger.error(error_msg, exc_info=True)
                self.performance_stats['errors'] += 1
                send_telegram(error_msg[:1000], self.credentials['telegram_token'], self.credentials['telegram_chat_id'])
                time.sleep(60)

# ========================
# MAIN EXECUTION
# ========================
if __name__ == "__main__":
    print("===== SIMPLIFIED BUCKET-BASED TRADING BOT STARTING =====")
    print(f"Start time: {datetime.now(NY_TZ)}")
    
    # Configure logging
    debug_handler = logging.StreamHandler()
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(debug_handler)
    
    logger.info("Starting simplified bucket-based trading bot")
    
    # Load credentials from environment variables
    credentials = {
        'telegram_token': os.getenv("TELEGRAM_BOT_TOKEN"),
        'telegram_chat_id': os.getenv("TELEGRAM_CHAT_ID"),
        'oanda_account_id': os.getenv("OANDA_ACCOUNT_ID"),
        'oanda_api_key': os.getenv("OANDA_API_KEY")
    }
    
    # Log credentials status
    logger.info("Checking credentials...")
    credentials_status = {k: "SET" if v else "MISSING" for k, v in credentials.items()}
    for k, status in credentials_status.items():
        logger.info(f"{k}: {status}")
    
    if not all(credentials.values()):
        logger.error("Missing one or more credentials in environment variables")
        if credentials.get('telegram_token') and credentials.get('telegram_chat_id'):
            send_telegram("❌ Bot failed to start: Missing credentials", 
                         credentials['telegram_token'], credentials['telegram_chat_id'])
        sys.exit(1)
    
    logger.info("All credentials present")
    
    try:
        # Start bots with their respective trade buckets
        logger.info("Creating M5 bot with 12 trade buckets")
        bot_5m = SimpleTradingBot("M5", credentials, M5_TRADE_BUCKETS)
        
        logger.info("Creating M15 bot with 7 trade buckets")
        bot_15m = SimpleTradingBot("M15", credentials, M15_TRADE_BUCKETS)
        
        # Run in separate threads
        logger.info("Starting bot threads")
        t1 = threading.Thread(target=bot_5m.run, name="M5_Bot")
        t2 = threading.Thread(target=bot_15m.run, name="M15_Bot")
        
        t1.daemon = True
        t2.daemon = True
        
        t1.start()
        logger.info("M5 bot thread started")
        t2.start()
        logger.info("M15 bot thread started")
        
        # Keep main thread alive with status updates
        logger.info("Main thread entering monitoring loop")
        last_status_time = time.time()
        while True:
            current_time = time.time()
            if current_time - last_status_time > 300:  # Every 5 minutes
                logger.info(f"Bot status: M5 {'alive' if t1.is_alive() else 'dead'}, "
                           f"M15 {'alive' if t2.is_alive() else 'dead'}")
                last_status_time = current_time
                
                # Log performance stats
                bot_5m.log_performance_stats()
                bot_15m.log_performance_stats()
                
            time.sleep(30)
            
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}", exc_info=True)
        if credentials.get('telegram_token') and credentials.get('telegram_chat_id'):
            send_telegram(f"❌ Bot crashed: {str(e)[:500]}", 
                         credentials['telegram_token'], credentials['telegram_chat_id'])
        sys.exit(1)
