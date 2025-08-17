import os
import requests
import pandas as pd

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

# -----------------------------------------------------------------------------
# Credentials & Endpoints
# -----------------------------------------------------------------------------
API_KEY_ID     = os.getenv("APCA_API_KEY_ID")     or os.getenv("APCA_API_KEY")
API_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY") or os.getenv("APCA_API_SECRET")
if not (API_KEY_ID and API_SECRET_KEY):
    raise RuntimeError(
        "Missing Alpaca credentials. "
        "Set APCA_API_KEY_ID & APCA_API_SECRET_KEY (or APCA_API_KEY & APCA_API_SECRET)."
    )

HEADERS = {
    "APCA-API-KEY-ID":     API_KEY_ID,
    "APCA-API-SECRET-KEY": API_SECRET_KEY,
}

TRADING_BASE_URL = os.getenv(
    "APCA_API_BASE_URL",
    "https://paper-api.alpaca.markets"
)
DATA_BASE_URL = os.getenv(
    "APCA_API_DATA_URL",
    "https://data.alpaca.markets"
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _to_utc_iso(dt: datetime) -> str:
    """Convert a tz‐aware datetime to a UTC ISO string ending in Z."""
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

# -----------------------------------------------------------------------------
# Bars & Quotes (IEX feed)
# -----------------------------------------------------------------------------
def fetch_bars(symbol: str, lookback_min: int) -> pd.DataFrame:
    """
    Fetch the last `lookback_min` minutes of 1-minute bars for `symbol`
    from the IEX feed (no SIP subscription required).
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=lookback_min)

    params = {
        "start":     _to_utc_iso(start),
        "end":       _to_utc_iso(end),
        "timeframe": "1Min",
        "limit":     lookback_min,
        "feed":      "iex",
    }
    url = f"{DATA_BASE_URL}/v2/stocks/{symbol}/bars"
    resp = requests.get(url, headers=HEADERS, params=params, timeout=10)
    resp.raise_for_status()

    bars = resp.json().get("bars", [])
    if not bars:
        return pd.DataFrame()

    df = pd.DataFrame(bars)
    # ISO strings can be parsed without unit="s"
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df.set_index("t", inplace=True)
    df.rename(columns={
        "o": "open", "h": "high",
        "l": "low",  "c": "close",
        "v": "volume"
    }, inplace=True)
    return df

def get_last_quote(symbol: str) -> SimpleNamespace:
    """
    Fetch the latest NBBO quote for `symbol` from IEX.
    Returns .askprice, .bidprice, .asksize, .bidsize
    """
    url = f"{DATA_BASE_URL}/v2/stocks/{symbol}/quotes/latest"
    resp = requests.get(
        url, headers=HEADERS, params={"feed": "iex"}, timeout=5
    )
    resp.raise_for_status()

    q = resp.json().get("quote", {})
    return SimpleNamespace(
        askprice=q.get("a"), bidprice=q.get("b"),
        asksize=q.get("as"),  bidsize=q.get("bs")
    )

# -----------------------------------------------------------------------------
# Option Chains (Trading API)
# -----------------------------------------------------------------------------
def fetch_option_chain(
    symbol: str,
    expiry_days: int,
    limit: int,
    strike: float = None,
    option_type: str = None
) -> SimpleNamespace:
    """
    Fetch option chains via Alpaca’s trading API.
    Returns .strikes (sorted list) and .chains (list of SimpleNamespace).
    """
    url = f"{TRADING_BASE_URL}/v2/options/chains"
    params = {
        "symbol":      symbol,
        "expiry_days": expiry_days,
        "limit":       limit,
    }
    if strike is not None:
        params["strike"] = strike
    if option_type is not None:
        params["type"] = option_type

    resp = requests.get(url, headers=HEADERS, params=params, timeout=10)
    resp.raise_for_status()

    chains = resp.json().get("chains", [])
    strikes = sorted({opt["strike"] for opt in chains})
    objs = [SimpleNamespace(**opt) for opt in chains]
    return SimpleNamespace(strikes=strikes, chains=objs)