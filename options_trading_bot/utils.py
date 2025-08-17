import pytz
from datetime import datetime
from typing import Any

def to_iso_utc(dt: datetime) -> str:
    return dt.astimezone(pytz.utc).isoformat()

def in_live_hours(now: datetime, cfg: Any) -> bool:
    t = now.astimezone(cfg.eastern).time()
    return cfg.market_open <= t <= cfg.market_close