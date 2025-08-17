# config.py

import pytz
from datetime import datetime, timedelta, time as dt_time

class Config:
    # ——— Alpaca Credentials ———
    APCA_API_KEY    = "PKL73VSRAJ6ITX0LV4G9"
    APCA_API_SECRET = "hrtKUxxUvrLMSQUoZfV6gfGSfpvglVcCyfncreJC"
    PAPER           = True  # set to False for live trading

    # ——— Trading Universe & Market Hours ———
    symbols = [
        "NVDA","TSLA","AMZN","AAPL","MSTR","AMD","UNH","HOOD","PLTR","SOFI",
        "COIN","INTC","META","OPEN","GOOGL","MSFT","SMCI","RKT","RIOT","MARA",
        "BAC","AAL","CRWV","C","RDDT","PCG","SNAP","PYPL","MU","BABA",
        "GME","MRNA","NIO","AVGO","NVO","CRCL","TSM","GOOG","BULL","SMR",
        "HIMS","APLD","LCID","MRVL","GLXY","SBET","BA","RGTI","OSCR","PANW"
    ]

    eastern       = pytz.timezone("US/Eastern")
    market_open   = dt_time(9, 30)
    market_close  = dt_time(16, 0)

    # ——— Backtest & Retraining Settings ———
    # How many past minutes to fetch per call
    lookback_min     = 120

    # How often to retrain (in seconds)
    retrain_interval = 60 * 60    # every hour

    # How often the live loop wakes up (in seconds)
    refresh_seconds  = 60         # once per minute

    # Minimum samples per class and test split
    test_size    = 0.20
    min_samples  = 5

    # Feature & model settings
    features     = ["ret", "sma5", "sma10", "rsi14"]
    model_params = {
        "n_estimators": 100,
        "max_depth":    5,
        "random_state": 42
    }

# single global config object
cfg = Config()