#!/usr/bin/env python3

import os
import sys
import logging
import argparse

from dotenv import load_dotenv
from config import cfg
from live_trader import LiveTrader

LOG_FORMAT = "%(asctime)s %(name)s %(levelname)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("OptionsBot.Main")


def parse_args():
    parser = argparse.ArgumentParser(description="Start the OptionsBot LiveTrader")
    parser.add_argument(
        "-p", "--paper",
        action="store_true",
        default=getattr(cfg, "PAPER", True),
        help="Run in paper-trading mode (overrides config)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG logging"
    )
    return parser.parse_args()


def load_credentials():
    api_key = (
        os.getenv("APCA_API_KEY_ID")
        or os.getenv("APCA_API_KEY")
        or getattr(cfg, "APCA_API_KEY", None)
    )
    api_secret = (
        os.getenv("APCA_API_SECRET_KEY")
        or os.getenv("APCA_API_SECRET")
        or getattr(cfg, "APCA_API_SECRET", None)
    )
    if not api_key or not api_secret:
        logger.error(
            "Alpaca credentials not found. "
            "Please set APCA_API_KEY_ID & APCA_API_SECRET_KEY (or legacy names)."
        )
        sys.exit(1)
    return api_key, api_secret


def instantiate_clients(api_key, api_secret, paper_mode):
    try:
        from alpaca.trading.client import TradingClient as NewTradingClient
        from alpaca.data.options   import OptionsClient as NewOptionsClient

        trading_api = NewTradingClient(
            api_key=api_key, secret_key=api_secret, paper=paper_mode
        )
        options_api = NewOptionsClient(
            api_key=api_key, secret_key=api_secret, paper=paper_mode
        )
        logger.debug("Using alpaca-py TradingClient & OptionsClient")
        return trading_api, options_api

    except (ImportError, ModuleNotFoundError):
        logger.warning(
            "Could not import alpaca-py (alpaca.data.options). "
            "Falling back to legacy alpaca_trade_api REST client.\n"
            "Install alpaca-py for best experience: pip install alpaca-py"
        )
        from alpaca_trade_api.rest import REST as LegacyREST

        base = "https://paper-api.alpaca.markets" if paper_mode else "https://api.alpaca.markets"
        data_base = getattr(cfg, "DATA_BASE_URL", "https://data.alpaca.markets/v1beta1")

        trading_api = LegacyREST(
            key_id=api_key, secret_key=api_secret, base_url=base, api_version="v2"
        )
        options_api = LegacyREST(
            key_id=api_key, secret_key=api_secret, base_url=data_base, api_version="v2"
        )
        return trading_api, options_api


def main():
    load_dotenv()
    args = parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    api_key, api_secret = load_credentials()
    trading_api, options_api = instantiate_clients(api_key, api_secret, args.paper)

    trader = LiveTrader(trading_api, options_api, cfg)
    try:
        trader.run()
    except Exception:
        logger.exception("Unhandled error in LiveTrader")
        sys.exit(1)


if __name__ == "__main__":
    main()