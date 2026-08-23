import sys
import urllib.parse
import requests
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.engines.mufap import MUFAPEngine

def test():
    engine = MUFAPEngine()
    ticker = "UBL Retirement Saving Fund (VPS-Commodities  Gold)"
    print("Testing engine directly for ticker:", ticker)
    fc = engine.forecast(ticker)
    print("ENGINE SUCCESS!")
    print("Ticker:", fc.ticker)
    print("Current Price:", fc.current_price)

    print("\nTesting HTTP API endpoint...")
    encoded = urllib.parse.quote(ticker)
    url = f"http://127.0.0.1:8000/api/forecast/{encoded}"
    r = requests.get(url)
    print("HTTP STATUS CODE:", r.status_code)
    if r.status_code == 200:
        data = r.json()
        print("HTTP SUCCESS!")
        print("API Ticker:", data.get("ticker"))
        print("API Current Price:", data.get("current_price"))
    else:
        print("HTTP ERROR RESPONSE:", r.text)

if __name__ == "__main__":
    test()
