import sys
import os
import time
import threading
import logging
import pytz
import numpy as np
import pandas as pd
import pandas_ta as ta
import joblib
import requests
import re
import traceback
from datetime import datetime, timedelta
from oandapyV20 import API
from oandapyV20.exceptions import V20Error
import oandapyV20.endpoints.instruments as instruments

# ========================
# CONSTANTS & CONFIG
# ========================
NY_TZ = pytz.timezone("America/New_York")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Oanda configuration
ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
API_KEY = os.getenv("OANDA_API_KEY")

# Only needed instrument and timeframe
INSTRUMENT = "XAU_USD"
TIMEFRAME = "M15"

# Global variables
GLOBAL_LOCK = threading.Lock()
CRT_SIGNAL_COUNT = 0
LAST_SIGNAL_TIME = 0
SIGNALS = []

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Oanda API
oanda_api = API(access_token=API_KEY, environment="practice")

# ========================
# UTILITY FUNCTIONS
# ========================
def parse_oanda_time(time_str):
    """Parse Oanda's timestamp with variable fractional seconds"""
    logger.debug(f"Parsing Oanda timestamp: {time_str}")
    try:
        if '.' in time_str and len(time_str.split('.')[1]) > 7:
            time_str = re.sub(r'\.(\d{6})\d+', r'.\1', time_str)
        result = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=pytz.utc).astimezone(NY_TZ)
        logger.debug(f"Parsed time: {result}")
        return result
    except Exception as e:
        logger.error(f"Error parsing time {time_str}: {str(e)}")
        return datetime.now(NY_TZ)

def send_telegram(message):
    """Send formatted message to Telegram with detailed error handling"""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram credentials not set, skipping message")
        return False
        
    logger.info(f"Attempting to send Telegram message: {message}")
    if len(message) > 4000:
        message = message[:4000] + "... [TRUNCATED]"
    
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        response = requests.post(url, json={
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': 'Markdown'
        }, timeout=10)
        
        logger.info(f"Telegram response: {response.status_code} - {response.text}")
        
        if response.status_code != 200:
            logger.error(f"Telegram error: {response.status_code} - {response.text}")
            return False
            
        if not response.json().get('ok'):
            logger.error(f"Telegram API error: {response.json()}")
            return False
            
        logger.info("Telegram message sent successfully")
        return True
    except Exception as e:
        logger.error(f"Telegram connection failed: {str(e)}")
        return False

def fetch_candles():
    """Fetch 201 candles for XAU_USD M15 with robust error handling"""
    logger.info(f"Fetching 201 candles for {INSTRUMENT} with timeframe {TIMEFRAME}")
    params = {
        "granularity": TIMEFRAME,
        "count": 201,
        "price": "M"
    }
    
    sleep_time = 10
    max_attempts = 3
    
    for attempt in range(max_attempts):
        try:
            request = instruments.InstrumentsCandles(instrument=INSTRUMENT, params=params)
            logger.debug(f"API request: {request}")
            response = oanda_api.request(request)
            logger.debug(f"API response: {response}")
            candles = response.get('candles', [])
            
            if not candles:
                logger.warning("No candles received from Oanda")
                continue
            
            data = []
            for candle in candles:
                price_data = candle.get('mid', {})
                if not price_data:
                    logger.warning(f"Skipping candle with missing mid price data: {candle}")
                    continue
                
                try:
                    parsed_time = parse_oanda_time(candle['time'])
                    data.append({
                        'time': parsed_time,
                        'open': float(price_data['o']),
                        'high': float(price_data['h']),
                        'low': float(price_data['l']),
                        'close': float(price_data['c']),
                        'volume': int(candle.get('volume', 0)),
                        'complete': candle.get('complete', False)
                    })
                except Exception as e:
                    logger.error(f"Error processing candle: {str(e)}")
            
            if not data:
                logger.warning("No valid candles found in response")
                continue
                
            df = pd.DataFrame(data)
            logger.info(f"Successfully fetched {len(df)} candles")
            return df
            
        except V20Error as e:
            if "rate" in str(e).lower():
                logger.warning(f"Rate limit hit, sleeping {sleep_time}s (attempt {attempt+1}/{max_attempts})")
                time.sleep(sleep_time)
                sleep_time *= 2
            else:
                logger.error(f"Oanda API error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error fetching candles: {str(e)}")
            logger.error(traceback.format_exc())
    
    logger.error(f"Failed to fetch candles after {max_attempts} attempts")
    return pd.DataFrame()

# ========================
# FEATURE ENGINEER (SIMPLIFIED FOR SIGNAL DETECTION)
# ========================
class FeatureEngineer:
    def calculate_crt_signal(self, df):
        """Calculate CRT signal on the current candle using the last 3 candles"""
        logger.info("Calculating CRT signals on last 3 candles")
        if len(df) < 3:
            logger.warning(f"Insufficient data: {len(df)} rows, need at least 3")
            return None, None

        c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
        
        signal_type = None
        entry = c3['open']
        sl = None
        tp = None
        
        # SELL Signal
        if (c2['high'] > c1['high']) and (c2['close'] < c1['high']):
            signal_type = 'SELL'
            sl = c2['high']
            risk = abs(entry - sl)
            tp = entry - 4 * risk
        
        # BUY Signal
        elif (c2['low'] < c1['low']) and (c2['close'] > c1['low']):
            signal_type = 'BUY'
            sl = c2['low']
            risk = abs(entry - sl)
            tp = entry + 4 * risk
        
        if signal_type:
            logger.info(f"Detected signal: {signal_type} on current candle")
            return signal_type, {'entry': entry, 'sl': sl, 'tp': tp, 'time': c3['time']}
        return None, None

# ========================
# TRADING DETECTOR
# ========================
class TradingDetector:
    def __init__(self):
        logger.info("Initializing TradingDetector")
        self.data = pd.DataFrame()
        self.feature_engineer = FeatureEngineer()
        self.scheduler = CandleScheduler(timeframe=15)
        
        logger.info("Loading initial 201 candles")
        self.data = self.fetch_initial_candles()
        
        if self.data.empty or len(self.data) < 200:
            logger.error("Failed to load sufficient initial candles")
            raise RuntimeError("Initial candle fetch failed or insufficient data")
            
        logger.info(f"Initial data loaded with {len(self.data)} rows")
        self.scheduler.register_callback(self.process_signals)
        logger.info("Starting scheduler thread")
        self.scheduler.start()
        logger.info("TradingDetector initialized")

    def fetch_initial_candles(self):
        logger.info("Fetching initial 201 candles")
        for attempt in range(5):
            df = fetch_candles()
            if not df.empty and len(df) >= 200:
                logger.info(f"Successfully fetched {len(df)} initial candles")
                return df
            logger.warning(f"Attempt {attempt+1} failed, retrying in 10s")
            time.sleep(10)
        logger.error("Failed to fetch initial 201 candles after 5 attempts")
        return pd.DataFrame()

    def process_signals(self, minutes_closed, latest_candles):
        logger.info(f"Processing signals, minutes closed: {minutes_closed}")
        if not latest_candles.empty:
            logger.info(f"Updating data with {len(latest_candles)} new candles")
            self.data = pd.concat([self.data, latest_candles]).drop_duplicates(subset=["time"], keep="last").sort_values("time").tail(201)
            logger.debug(f"Updated data shape: {self.data.shape}, latest time: {self.data['time'].max()}")
        else:
            logger.warning("No new candles to update")

        if self.data.empty or len(self.data) < 3:
            logger.warning(f"Insufficient data: {len(self.data)} rows, need at least 3")
            return
        
        signal_type, signal_data = self.feature_engineer.calculate_crt_signal(self.data.tail(3))
        if signal_type and signal_data:
            logger.info(f"Signal validated: {signal_type}")
            alert_time = signal_data['time'].astimezone(NY_TZ)
            send_telegram(
                f"🔔 *SETUP* {INSTRUMENT.replace('_','/')} {signal_type}\n"
                f"Timeframe: {TIMEFRAME}\n"
                f"Time: {alert_time.strftime('%Y-%m-%d %H:%M')} NY\n"
                f"Entry: {signal_data['entry']:.2f}\n"
                f"TP: {signal_data['tp']:.2f}\n"
                f"SL: {signal_data['sl']:.2f}"
            )

    def update_data(self, df_new):
        logger.info(f"Updating data with new dataframe of size {len(df_new)}")
        if df_new.empty:
            logger.warning("Received empty dataframe in update_data")
            return
        
        if self.data.empty:
            self.data = df_new.dropna(subset=['time', 'open', 'high', 'low', 'close']).tail(201)
            logger.debug("Initialized data with new dataframe")
        else:
            last_existing_time = self.data['time'].max()
            new_data = df_new[df_new['time'] > last_existing_time]
            if not new_data.empty:
                self.data = pd.concat([self.data, new_data]).drop_duplicates(subset=['time'], keep='last')
                self.data = self.data.sort_values('time').reset_index(drop=True).tail(201)
                logger.debug(f"Combined data shape: {self.data.shape}, latest time: {self.data['time'].max()}")
            else:
                latest_new = df_new.iloc[-1]
                if latest_new['time'] >= self.data['time'].max():
                    self.data = pd.concat([self.data, df_new.tail(1)]).drop_duplicates(subset=['time'], keep='last')
                    self.data = self.data.sort_values('time').reset_index(drop=True).tail(201)
                    logger.debug(f"Forced update with latest candle, new shape: {self.data.shape}, latest time: {self.data['time'].max()}")

# ========================
# CANDLE SCHEDULER
# ========================
class CandleScheduler(threading.Thread):
    def __init__(self, timeframe=15):
        super().__init__(daemon=True)
        self.timeframe = timeframe
        self.callback = None
        self.active = True
        self.event = threading.Event()
    
    def register_callback(self, callback):
        self.callback = callback
        
    def calculate_next_candle(self):
        now = datetime.now(NY_TZ)
        current_minute = now.minute
        remainder = current_minute % self.timeframe
        if remainder == 0:
            return now.replace(second=0, microsecond=0) + timedelta(minutes=self.timeframe)
        next_minute = current_minute - remainder + self.timeframe
        if next_minute >= 60:
            return now.replace(hour=now.hour + 1, minute=0, second=0, microsecond=0)
        return now.replace(minute=next_minute, second=0, microsecond=0)
    
    def calculate_minutes_closed(self, latest_time):
        if latest_time is None:
            return 0
            
        logger.info(f"Calculating minutes closed for latest time: {latest_time}")
        now = datetime.now(NY_TZ)
        elapsed = (now - latest_time).total_seconds() / 60
        logger.debug(f"Elapsed time since last candle: {elapsed:.2f} minutes")
        return min(44, max(0, int(elapsed)))
    
    def run(self):
        logger.info("Starting CandleScheduler thread")
        while self.active:
            try:
                logger.info("Fetching latest candle data")
                df_candles = fetch_candles()
                
                if df_candles.empty:
                    logger.warning("No candles fetched")
                    time.sleep(60)
                    continue
                
                latest_candle = df_candles.iloc[-1]
                latest_time = latest_candle['time']
                minutes_closed = self.calculate_minutes_closed(latest_time)
                
                if self.callback:
                    logger.info(f"Calling callback with minutes closed: {minutes_closed}")
                    self.callback(minutes_closed, df_candles.tail(1))
                
                now = datetime.now(NY_TZ)
                next_run = now + timedelta(minutes=15 - (now.minute % 15), seconds=0, microseconds=0)
                sleep_seconds = (next_run - now).total_seconds()
                logger.info(f"Sleeping {sleep_seconds:.1f} seconds until next candle")
                time.sleep(max(1, sleep_seconds))
                
            except Exception as e:
                logger.error(f"Scheduler error: {str(e)}")
                logger.error(traceback.format_exc())
                time.sleep(60)

# ========================
# MAIN BOT OPERATION
# ========================
def run_bot():
    logger.info("Starting trading bot")
    send_telegram(f"🚀 *Bot Started*\nInstrument: {INSTRUMENT}\nTimeframe: {TIMEFRAME}\nTime: {datetime.now(NY_TZ)}")
    
    try:
        detector = TradingDetector()
    except Exception as e:
        logger.error(f"Detector initialization failed: {str(e)}")
        send_telegram(f"❌ *Bot Failed to Start*:\n{str(e)}")
        return
        
    logger.info("Bot started successfully")
    
    while True:
        try:
            logger.info("Running bot cycle")
            df = fetch_candles()
            if not df.empty:
                logger.info(f"Fetched {len(df)} candles, updating data")
                detector.update_data(df)
            else:
                logger.warning("No candles fetched in this cycle")
            time.sleep(60)  # Check every minute
        except Exception as e:
            logger.error(f"Main loop error: {e}")
            logger.error(traceback.format_exc())
            time.sleep(60)

if __name__ == "__main__":
    logger.info("Launching main application")
    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()
    logger.info("Bot thread started")
    
    try:
        while True:
            time.sleep(1)  # Keep main thread alive
    except KeyboardInterrupt:
        logger.info("Shutting down bot")
        send_telegram("🛑 *Bot Stopped*")
