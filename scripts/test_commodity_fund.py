import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.engines.mufap import mufap_engine

def test():
    ticker = "UBL Retirement Saving Fund (VPS-Commodities  Gold)"
    print("Testing ticker:", ticker)
    fc = mufap_engine.forecast(ticker)
    print("SUCCESS!")
    print("Ticker:", fc.ticker)
    print("Current Price:", fc.current_price)
    print("Headline Return:", fc.headline_predicted_return_pct)
    print("Horizons count:", len(fc.horizons))

if __name__ == "__main__":
    test()
