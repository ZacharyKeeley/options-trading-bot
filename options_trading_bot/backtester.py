#!/usr/bin/env python3 -u
"""
backtester.py

Run multithreaded backtests:
  1. Fetch historical bars
  2. Generate features and targets
  3. Train/test RandomForest models
  4. Compute P&L series
  5. Aggregate into portfolio equity curve
  6. Compute and log performance metrics
  7. Save PnL CSV and equity chart to disk
"""

import os
import logging

from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from data_fetcher import fetch_bars
from features import add_features, bs_delta_array

# Constants
TRADING_MINUTES_PER_DAY = 390
TRADING_DAYS_PER_YEAR = 252

logger = logging.getLogger(__name__)


class Backtester:
    def __init__(
        self,
        api_client,
        cfg,
        lookback_days: int = 120,
        test_size: float = 0.2
    ) -> None:
        """
        api_client: data/API client
        cfg: configuration object with attributes:
             - symbols (list[str])
             - features (list[str])
             - model_params (dict)
             - r (float): risk-free rate
             - sigma (float): volatility
        lookback_days: days to fetch (default=120)
        test_size: test split fraction (0 < test_size < 1)
        """
        self.api = api_client
        self.cfg = cfg
        self.lookback_minutes = lookback_days * TRADING_MINUTES_PER_DAY
        self.test_size = test_size

    def simulate(self, symbol: str) -> Optional[pd.Series]:
        """
        Simulate a single symbol:
          - Fetch data, add features
          - Train/test RandomForest
          - Compute per‐period P&L series
        Returns PnL Series (index=timestamp) or None on error.
        """
        try:
            bars_df = fetch_bars(self.api, symbol,
                                 self.lookback_minutes, self.cfg)
        except Exception as exc:
            logger.warning(f"Skipping {symbol}: {exc}")
            return None

        feature_df = add_features(bars_df)
        if len(feature_df) < 20:
            logger.warning(f"{symbol}: insufficient data ({len(feature_df)} rows)")
            return None

        # Train/test split
        X = feature_df[self.cfg.features]
        y = feature_df["target"]
        X_train, X_test, y_train, _ = train_test_split(
            X, y, test_size=self.test_size, shuffle=False
        )

        # Model fitting
        model = RandomForestClassifier(**self.cfg.model_params)
        model.fit(X_train, y_train)

        # P&L calculation
        test_idx = X_test.index
        prices_t0 = feature_df.loc[test_idx, "close"].to_numpy()
        prices_t1 = feature_df["close"].shift(-1).loc[test_idx].to_numpy()
        signals = model.predict(X_test).astype(bool)

        deltas = bs_delta_array(
        prices_t0, prices_t0,
        r=self.cfg.r,
        sigma=self.cfg.sigma,
        call_flags=signals,
        dt=1 / TRADING_DAYS_PER_YEAR
        )

        pnl_array = deltas * ((prices_t1 - prices_t0) / prices_t0)

        pnl_series = pd.Series(pnl_array, index=test_idx).dropna().sort_index()
        return pnl_series

    def run_all(
        self,
        workers: int,
        out_dir: str,
        start_capital: float
    ) -> Dict[str, float]:
        """
        Execute backtests in parallel, aggregate results, compute metrics,
        and save outputs.
        Returns a dict of metrics.
        """
        os.makedirs(out_dir, exist_ok=True)
        pnl_series_map: Dict[str, pd.Series] = {}

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self.simulate, sym): sym
                for sym in self.cfg.symbols
            }
            for fut in as_completed(futures):
                symbol = futures[fut]
                result = fut.result()
                if result is not None:
                    pnl_series_map[symbol] = result

        if not pnl_series_map:
            logger.error("No backtest results generated.")
            return {}

        # Aggregate to DataFrame
        pnl_df = pd.DataFrame(pnl_series_map).fillna(0.0)
        portfolio_ret = pnl_df.sum(axis=1)
        equity_curve = start_capital * (1.0 + portfolio_ret.cumsum())

        # Compute metrics
        periods_per_year = TRADING_DAYS_PER_YEAR * TRADING_MINUTES_PER_DAY
        mu = portfolio_ret.mean()
        sigma = portfolio_ret.std()

        annual_return = mu * periods_per_year
        annual_vol = sigma * np.sqrt(periods_per_year)
        sharpe_ratio = annual_return / annual_vol if annual_vol else np.nan
        max_drawdown = (equity_curve / equity_curve.cummax() - 1.0).min()

        metrics = {
            "start_capital":       start_capital,
            "end_capital":         float(equity_curve.iloc[-1]),
            "total_return_pct":    (equity_curve.iloc[-1] / start_capital - 1.0) * 100.0,
            "annual_return_pct":   annual_return * 100.0,
            "annual_vol_pct":      annual_vol * 100.0,
            "sharpe_ratio":        sharpe_ratio,
            "max_drawdown_pct":    max_drawdown * 100.0,
        }

        logger.info("Backtest metrics: %s", metrics)

        # Save PnL and equity chart
        pnl_df.to_csv(os.path.join(out_dir, "pnl.csv"), index_label="timestamp")

        fig, ax = plt.subplots(figsize=(10, 5))
        equity_curve.plot(ax=ax, title="Portfolio Equity Curve")
        ax.set_ylabel("Equity ($)")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "equity_curve.png"))
        plt.close(fig)

        return metrics