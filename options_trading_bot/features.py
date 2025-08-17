import numpy as np
import pandas as pd
from scipy.stats import norm

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ret"]   = df["close"].pct_change()
    df["sma5"]  = df["close"].rolling(5).mean()
    df["sma10"] = df["close"].rolling(10).mean()

    d = df["close"].diff()
    up, down = d.clip(lower=0), -d.clip(upper=0)
    rs = up.rolling(14).mean() / down.rolling(14).mean()
    df["rsi14"] = 100 - 100 / (1 + rs)

    df.dropna(inplace=True)
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    return df.dropna()

def bs_delta_array(S, K, r, sigma, call_flags, dt=1/252):
    from scipy.stats import norm
    import numpy as np

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * dt) / (sigma * np.sqrt(dt))
    pdf = norm.cdf(d1)
    return np.where(call_flags, pdf, pdf - 1)
