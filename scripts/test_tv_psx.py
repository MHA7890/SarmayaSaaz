import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.tradingview_fetch import fetch_bars

def test():
    for symbol in ["PSX:HBL", "PSX:SYS", "PSX:OGDC", "PSX:LUCK"]:
        print(f"Testing {symbol}...")
        df = fetch_bars(symbol, n_bars=10)
        print(f"{symbol} result:")
        print(df)
        print("-" * 50)

if __name__ == "__main__":
    test()
