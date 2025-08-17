#!/usr/bin/env python3

import time
import logging
import pandas as pd
import numpy as np

from datetime import datetime, timedelta, time as dt_time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

logger = logging.getLogger("OptionsBot.LiveTrader")


class LiveTrader:
    def __init__(self, trading_api, options_api, cfg):
        """
        trading_api   – TradingClient (new) or REST (legacy)
        options_api   – OptionsClient (new) or REST (legacy)
        cfg           – your config.Config instance
        """
        self.trading_api      = trading_api
        self.options_api      = options_api
        self.cfg              = cfg

        # config fields
        self.symbols          = cfg.symbols
        self.lookback_min     = cfg.lookback_min
        self.retrain_interval = cfg.retrain_interval
        self.trade_cycle_sec  = cfg.refresh_seconds

        self.market_open      = cfg.market_open
        self.market_close     = cfg.market_close
        self.tz               = cfg.eastern

        self.features         = cfg.features
        self.model_params     = cfg.model_params
        self.test_size        = cfg.test_size
        self.min_samples      = cfg.min_samples

        # model and scheduling
        self.model            = None
        self.next_retrain     = time.time()

    def run(self):
        mode = "PAPER" if self.cfg.PAPER else "LIVE"
        logger.info(f"Starting LiveTrader in {mode} mode for {self.symbols!r}")

        while True:
            now = datetime.now(self.tz)

            # sleep until market open
            if not self._is_market_open(now):
                secs = self._seconds_until_open(now)
                logger.info(f"Market closed → sleeping {secs/60:.1f} minutes")
                time.sleep(secs)
                continue

            # retrain on schedule
            if time.time() >= self.next_retrain:
                self._retrain_models()
                self.next_retrain = time.time() + self.retrain_interval

            # run one trade cycle
            try:
                self._trade_cycle()
            except Exception:
                logger.exception("Error in trade cycle")

            time.sleep(self.trade_cycle_sec)

    def _retrain_models(self):
        logger.info("Retraining models")

        # 1) fetch historical minute bars
        df = self._fetch_history(self.symbols, self.lookback_min)

        # 2) build features & targets
        for sym in self.symbols:
            sub = df[df.symbol == sym].copy()
            sub["ret"]   = sub["close"].pct_change().fillna(0)
            sub["sma5"]  = sub["close"].rolling(5).mean()
            sub["sma10"] = sub["close"].rolling(10).mean()
            sub["rsi14"] = self._rsi(sub["close"], 14)
            sub["target"] = (sub["ret"].shift(-1) > 0).astype(int)

            # assign back to master df
            df.loc[sub.index, self.features] = sub[self.features]
            df.loc[sub.index, "target"]     = sub["target"]

        df.dropna(inplace=True)
        X = df[self.features]
        y = df["target"].astype(int)

        # 3) skip if too few up/down samples
        if y.sum() < self.min_samples or (len(y) - y.sum()) < self.min_samples:
            logger.warning("Not enough samples to train; skipping retrain")
            return

        # 4) train/test split + fit
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=self.test_size, shuffle=False
        )
        model = RandomForestClassifier(**self.model_params)
        model.fit(X_train, y_train)
        self.model = model

        logger.info("Retrain complete")

    def _trade_cycle(self):
        if self.model is None:
            logger.debug("Model uninitialized; skipping trade cycle")
            return

        # 1) build latest-feature DataFrame
        X_new = self._build_latest_features(self.symbols)
        probs = self.model.predict_proba(X_new)[:, 1]

        # 2) fetch all positions
        positions = self._fetch_positions()
        pos_map   = {p.symbol: p for p in positions}

        signals = []
        for sym, prob in zip(self.symbols, probs):
            # if bullish & no open call → buy ATM call
            if prob > 0.6 and not self._has_open_call(sym, pos_map):
                call = self._get_atm_call(sym)
                if call:
                    signals.append({
                        "action":      "buy",
                        "option_sym":  call.symbol,
                        "quantity":    1
                    })

            # if bearish & we hold a call → sell all
            elif prob < 0.4:
                for ct, pos in pos_map.items():
                    if ct.startswith(sym) and pos.qty > 0:
                        signals.append({
                            "action":      "sell",
                            "option_sym":  ct,
                            "quantity":    int(pos.qty)
                        })

        # 3) execute all option signals
        for sig in signals:
            self._execute_option_signal(sig)

    def _fetch_history(self, syms, lookback):
        # new alpaca_trade_api get_barset
        if hasattr(self.trading_api, "get_barset"):
            barset = self.trading_api.get_barset(syms, "1Min", limit=lookback).df
            df = barset.stack().reset_index()
            df.columns = ["symbol","t","open","high","low","close","volume"]
            return df

        # fallback get_bars()
        rows = []
        for s in syms:
            try:
                bars = self.trading_api.get_bars(s, "1Min", limit=lookback)
                for b in bars:
                    rows.append({
                        "symbol": s,
                        "t":       getattr(b, "t", getattr(b, "timestamp", None)),
                        "open":    getattr(b, "o", getattr(b, "open", None)),
                        "high":    getattr(b, "h", getattr(b, "high", None)),
                        "low":     getattr(b, "l", getattr(b, "low", None)),
                        "close":   getattr(b, "c", getattr(b, "close", None)),
                        "volume":  getattr(b, "v", getattr(b, "volume", None))
                    })
            except Exception:
                logger.exception(f"Failed to fetch history for {s}")
        return pd.DataFrame(rows)

    def _build_latest_features(self, syms):
        rows = []
        for s in syms:
            # fetch last n bars to compute indicators
            bars = self.trading_api.get_bars(s, "1Min", limit=20)
            df   = pd.DataFrame([{
                "open":   getattr(bars[-1], "o", None),
                "high":   getattr(bars[-1], "h", None),
                "low":    getattr(bars[-1], "l", None),
                "close":  getattr(bars[-1], "c", None),
                "volume": getattr(bars[-1], "v", None)
            }])
            df["ret"]   = df["close"].pct_change().fillna(0)
            df["sma5"]  = df["close"].rolling(5).mean()
            df["sma10"] = df["close"].rolling(10).mean()
            df["rsi14"] = self._rsi(df["close"], 14)
            rows.append(df[self.features].iloc[-1])

        return pd.DataFrame(rows, index=syms)

    def _fetch_positions(self):
        if hasattr(self.trading_api, "get_all_positions"):
            return self.trading_api.get_all_positions()
        return self.trading_api.list_positions()

    def _get_latest_price(self, symbol):
        # new TradingClient
        if hasattr(self.trading_api, "get_latest_trade"):
            return self.trading_api.get_latest_trade(symbol).price
        # legacy REST
        return self.trading_api.get_last_trade(symbol).price

    def _get_atm_call(self, symbol):
        # 1) nearest expiry
        dates = self.options_api.get_expiration_dates(
            symbol=symbol, limit=5, include_all_tradable=True
        )
        if not dates:
            logger.warning(f"No expiries for {symbol}")
            return None
        expiry = dates[0]

        # 2) call chain
        chain = self.options_api.get_option_chain(
            symbol=symbol,
            expiration_date=expiry,
            option_type="call"
        )
        if not chain:
            logger.warning(f"No call chain for {symbol} @ {expiry}")
            return None

        # 3) ATM strike
        spot       = self._get_latest_price(symbol)
        strikes    = [c.strike for c in chain]
        atm_strike = min(strikes, key=lambda x: abs(x - spot))

        # 4) return matching contract
        for c in chain:
            if c.strike == atm_strike:
                return c
        return None

    def _has_open_call(self, symbol, pos_map):
        return any(
            ct.startswith(symbol) and pos_map[ct].qty > 0
            for ct in pos_map
        )

    def _execute_option_signal(self, sig):
        action = sig["action"]
        opt_sym= sig["option_sym"]
        qty    = sig["quantity"]
        logger.info(f"{action.upper()} {qty}× {opt_sym}")
        try:
            self.trading_api.submit_order(
                symbol       = opt_sym,
                qty          = qty,
                side         = action,
                type         = "market",
                time_in_force= "day"
            )
        except Exception:
            logger.exception(f"Failed to {action} {opt_sym}")

    def _rsi(self, series, length):
        delta = series.diff().fillna(0)
        up    = delta.clip(lower=0)
        down  = -delta.clip(upper=0)
        ma_up = up.rolling(length).mean()
        ma_dn = down.rolling(length).mean()
        rs    = ma_up / ma_dn
        return 100 - (100 / (1 + rs))

    def _is_market_open(self, now: datetime) -> bool:
        return self.market_open <= now.time() < self.market_close

    def _seconds_until_open(self, now: datetime) -> float:
        open_dt = datetime.combine(now.date(), self.market_open, tzinfo=self.tz)
        if now.time() < self.market_open:
            return (open_dt - now).total_seconds()
        nxt = now.date() + timedelta(days=1)
        next_open = datetime.combine(nxt, self.market_open, tzinfo=self.tz)
        return (next_open - now).total_seconds()