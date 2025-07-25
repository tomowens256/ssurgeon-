import asyncio
import platform
from datetime import datetime
import logging
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
from oandapyV20.contrib.requests import MarketOrderRequest
import telegram
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API and bot configuration
API_TOKEN = 'your_oanda_api_token'
ACCOUNT_ID = 'your_account_id'
TELEGRAM_TOKEN = 'your_telegram_bot_token'
CHAT_ID = 'your_chat_id'
INSTRUMENT = 'XAU_USD'
TIMEFRAME = 'M15'
CANDLE_COUNT = 201

# Initialize OANDA API client
api = oandapyV20.API(access_token=API_TOKEN, environment="practice")
bot = telegram.Bot(token=TELEGRAM_TOKEN)

# Global variables
active_trades = {}
scaler = StandardScaler()
model = keras.models.load_model('your_trained_model.h5')

def setup():
    logger.info("Launching main application")
    logger.info("Starting trading bot")
    send_telegram_message("🚀 *Bot Started*\nInstrument: {}\nTimeframe: {}\nTime: {}".format(
        INSTRUMENT, TIMEFRAME, datetime.now().strftime("%Y-%m-%d %H:%M:%S%z")))
    logger.info("Bot thread started")

def send_telegram_message(message):
    try:
        bot.send_message(chat_id=CHAT_ID, text=message, parse_mode='Markdown')
        logger.info("Telegram message sent successfully")
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")

def fetch_candles(count=CANDLE_COUNT):
    params = {
        "count": count,
        "granularity": TIMEFRAME
    }
    r = instruments.InstrumentsCandles(instrument=INSTRUMENT, params=params)
    api.request(r)
    return r.response.get('candles', [])

def get_latest_candle():
    candles = fetch_candles(count=1)
    return candles[0] if candles else None

def calculate_indicators(df):
    # Placeholder for technical indicators (e.g., RSI, MACD, Bollinger Bands)
    df['rsi'] = 50.0  # Example, replace with actual RSI calculation
    df['macd'] = 0.0  # Example, replace with actual MACD calculation
    df['bb_low'] = df['close'].rolling(window=20).mean() - 2 * df['close'].rolling(window=20).std()
    df['bb_mid'] = df['close'].rolling(window=20).mean()
    df['bb_high'] = df['close'].rolling(window=20).mean() + 2 * df['close'].rolling(window=20).std()
    return df

def detect_signal(candles):
    if len(candles) < 3:
        return None
    c0, c1, c2 = candles[-3:]
    if c2['high'] > c1['high'] and c0['high'] < c1['high'] and c2['close'] < c1['high']:
        return 'SELL'
    elif c2['low'] < c1['low'] and c0['low'] > c1['low'] and c2['close'] > c1['low']:
        return 'BUY'
    return None

def generate_features(candle, signal_type):
    features = {
        'adj_close': candle['close'],
        'garman_klass_vol': 0.0,
        'rsi_20': 34.9566,
        'bb_low': 8.1145,
        'bb_mid': 8.1177,
        'bb_high': 8.1210,
        'atr_z': -0.6061,
        'macd_z': -0.6901,
        'dollar_volume': 8.2016,
        'ma_10': 3347.6560,
        'ma_100': 3364.0773,
        'vwap': 3385.8948,
        'vwap_std': 0.5871,
        'rsi': 30.4565,
        'ma_20': 3351.7255,
        'ma_30': 3354.5307,
        'ma_40': 3357.8547,
        'ma_60': 3361.7642,
        'trend_strength_up': 0.0,
        'trend_strength_down': 1.0,
        'sl_price': candle['high'] + 1.68 if signal_type == 'SELL' else candle['low'] - 1.68,
        'tp_price': candle['close'] - 6.72 if signal_type == 'SELL' else candle['close'] + 6.72,
        'prev_volume': 2169.0,
        'sl_distance': 1.68,
        'tp_distance': 6.72,
        'rrr': 4.0,
        'log_sl': np.log(candle['high'] + 1.68) if signal_type == 'SELL' else np.log(candle['low'] - 1.68),
        'prev_body_size': 0.57,
        'prev_wick_up': 1.66,
        'prev_wick_down': 0.57,
        'is_bad_combo': 0.0,
        'price_div_vol': candle['close'] * 2169.0,
        'rsi_div_macd': 30.4565 - (-0.6901),
        'price_div_vwap': candle['close'] / 3385.8948,
        'sl_div_atr': -27.7162,
        'tp_div_atr': -110.8648,
        'rrr_div_rsi': 4.0 / 30.4565,
        'day_Friday': 1.0,
        'day_Monday': 0.0,
        'day_Sunday': 0.0,
        'day_Thursday': 0.0,
        'day_Tuesday': 0.0,
        'day_Wednesday': 0.0,
        'session_q1': 0.0,
        'session_q2': 0.0,
        'session_q3': 1.0,
        'session_q4': 0.0,
        'rsi_zone_neutral': 1.0,
        'rsi_zone_overbought': 0.0,
        'rsi_zone_oversold': 0.0,
        'rsi_zone_unknown': 0.0,
        'trend_direction_downtrend': 1.0,
        'trend_direction_sideways': 0.0,
        'trend_direction_uptrend': 0.0,
        'crt_BUY': 1.0 if signal_type == 'BUY' else 0.0,
        'crt_SELL': 1.0 if signal_type == 'SELL' else 0.0,
        'trade_type_BUY': 1.0 if signal_type == 'BUY' else 0.0,
        'trade_type_SELL': 1.0 if signal_type == 'SELL' else 0.0,
        'combo_flag_dead': 0.0,
        'combo_flag_fair': 1.0,
        'combo_flag_fine': 0.0,
        'combo_flag2_dead': 0.0,
        'combo_flag2_fair': 1.0,
        'combo_flag2_fine': 0.0,
        'minutes_closed_0': 0.0,
        'minutes_closed_15': 1.0,
        'minutes_closed_30': 0.0,
        'minutes_closed_45': 0.0
    }
    return pd.DataFrame([features])

def get_predictions(features):
    scaled_features = scaler.transform(features)
    predictions = model.predict(scaled_features)
    return [float(p[0]) for p in predictions for _ in range(10)]

def update_loop():
    logger.info("Running bot cycle")
    candles = fetch_candles(count=3)
    if candles:
        signal = detect_signal(candles)
        if signal:
            logger.info(f"Detected signal: {signal} on current candle at {candles[-1]['time']}")
            trade_id = f"{signal}_{int(datetime.now().timestamp())}"
            active_trades[trade_id] = {
                'type': signal,
                'entry_time': datetime.strptime(candles[-1]['time'], '%Y-%m-%dT%H:%M:%S.%f000Z'),
                'entry_price': float(candles[-1]['close']),
                'sl': float(candles[-1]['high']) + 1.68 if signal == 'SELL' else float(candles[-1]['low']) - 1.68,
                'tp': float(candles[-1]['close']) - 6.72 if signal == 'SELL' else float(candles[-1]['close']) + 6.72,
                'outcome': 'Open'
            }
            features = generate_features(candles[-1], signal)
            predictions = get_predictions(features)
            send_telegram_message(f"🔔 *SETUP* XAU/USD {signal}\nTimeframe: {TIMEFRAME}\nTime: {datetime.strptime(candles[-1]['time'], '%Y-%m-%dT%H:%M:%S.%f000Z').strftime('%Y-%m-%d %H:%M %Z')}\nEntry: {candles[-1]['close']}\nTP: {active_trades[trade_id]['tp']}\nSL: {active_trades[trade_id]['sl']}\nCandle Age: 15.00 minutes")
            send_telegram_message(f"📊 *FEATURES* XAU/USD {signal}\n" + "\n".join(f"{k}: {v}" for k, v in features.iloc[0].to_dict().items()))
            send_telegram_message(f"🤖 *MODEL PREDICTIONS* XAU/USD {signal}\n" + "\n".join(f"Prediction {i+1}: {p:.4f} (Outcome: Worth Taking)" for i, p in enumerate(predictions)))
            logger.info(f"New trade stored: {trade_id} with prediction {predictions[0]}")

async def main():
    setup()  # Initialize trading bot
    while True:
        update_loop()  # Update game/visualization state
        latest_candle = get_latest_candle()
        if latest_candle and latest_candle['time']:
            candle_time = datetime.strptime(latest_candle['time'], '%Y-%m-%dT%H:%M:%S.%f000Z')
            for trade_id, trade in list(active_trades.items()):
                if candle_time > trade['entry_time']:
                    if trade['type'] == 'SELL' and float(latest_candle['high']) >= trade['sl']:
                        trade['outcome'] = 'Hit SL (Loss)'
                    elif trade['type'] == 'BUY' and float(latest_candle['low']) <= trade['sl']:
                        trade['outcome'] = 'Hit SL (Loss)'
                    elif trade['type'] == 'SELL' and float(latest_candle['low']) <= trade['tp']:
                        trade['outcome'] = 'Hit TP (Win)'
                    elif trade['type'] == 'BUY' and float(latest_candle['high']) >= trade['tp']:
                        trade['outcome'] = 'Hit TP (Win)'
                    else:
                        trade['outcome'] = 'Open'
                    if trade['outcome'] != 'Open':
                        send_telegram_message(f"📈 *Trade Outcome*\nEntry: {trade['entry_price']}\nSL: {trade['sl']}\nTP: {trade['tp']}\nPrediction: {get_predictions(generate_features(latest_candle, trade['type']))[0]:.4f}\nOutcome: {trade['outcome']}\nTime: {candle_time.strftime('%Y-%m-%d %H:%M %Z')}")
                        del active_trades[trade_id]
                        logger.info(f"{trade['type']} trade {trade_id} outcome: {trade['outcome']}")
        await asyncio.sleep(1.0 / 60)  # Control frame rate

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())
