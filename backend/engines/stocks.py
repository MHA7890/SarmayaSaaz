"""
PSX engine - Pakistan Stock Exchange equities, grouped into 7 super-sectors.

Feature width varies by sector: every sector carries the PKR currency proxy,
while Energy_Power additionally carries oil and Tech_Telecom carries NASDAQ.
Each sector therefore has its own scaler, and the clustered datasets are the
authority on column order.
"""
from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
import pandas as pd

from backend.config import settings
from backend.engines.base import Engine, EngineError
from backend.engines.routed import RoutedEngine
from backend.schemas import (
    Asset,
    AssetClass,
    Forecast,
    PricePoint,
    Quote,
)
from backend.services import live_prices, news_live

logger = logging.getLogger(__name__)

SECTOR_MAP = {
    "Financials": ["ABL", "AHCL", "AICL", "AKBL", "BAFL", "BAHL", "BOP", "FABL",
                   "HBL", "HMB", "MCB", "MEBL", "NBP", "PSX", "SCBPL", "UBL"],
    "Energy_Power": ["APL", "ATRL", "CNERGY", "HUBC", "KAPCO", "KEL", "MARI",
                     "OGDC", "POL", "PPL", "PSO", "SNGP", "SSGC"],
    "Cement_Construction": ["BWCL", "CHCC", "DGKC", "FCCL", "KOHC", "LUCK", "MLCF",
                            "PIOC", "POWER", "THALL", "ISL", "INIL"],
    "Tech_Telecom": ["AIRLINK", "NETSOL", "PTC", "SYS", "TRG"],
    "Pharmaceuticals": ["ABOT", "AGP", "CPHL", "GLAXO", "HALEON", "HINOON", "IBFL",
                        "SEARL", "SHFA"],
    "Fertilizers_Chemicals": ["COLG", "EFERT", "ENGROH", "FATIMA", "FFC", "GHGL",
                              "LCI", "LOTCHEM", "PABC", "EPCL"],
    "Consumer_Autos": ["ATLH", "BNWM", "FFL", "GADT", "GAL", "GHNI", "HCAR", "HUMNL",
                       "ILP", "INDU", "JDWS", "JVDC", "KTML", "MEHT", "MTL", "MUREB",
                       "NATF", "NESTLE", "NML", "PAEL", "PAKT", "PGLC", "PIBTL",
                       "PKGS", "PSEL", "RMPL", "SAZEW", "SRVI", "SSOM", "TGL",
                       "UNITY", "UPFL", "YOUW"],
}

# Company names for every ticker in SECTOR_MAP - anything still unmapped
# displays its symbol rather than an invented name. Cross-referenced against
# stockanalysis.com's PSX listing (2026-08) rather than typed from memory,
# since a wrong company name is worse than the bare ticker it replaces.
# MARI was Mari Petroleum Company; the exchange listing now carries it as
# Mari Energies following its 2024 rebrand.
NAMES = {
    "OGDC": "Oil & Gas Development Company", "PPL": "Pakistan Petroleum",
    "MARI": "Mari Energies", "PSO": "Pakistan State Oil", "HUBC": "Hub Power Company",
    "POL": "Pakistan Oilfields", "LUCK": "Lucky Cement", "DGKC": "D.G. Khan Cement",
    "MLCF": "Maple Leaf Cement", "FCCL": "Fauji Cement", "CHCC": "Cherat Cement",
    "HBL": "Habib Bank", "UBL": "United Bank", "MCB": "MCB Bank",
    "MEBL": "Meezan Bank", "BAFL": "Bank Alfalah", "BAHL": "Bank AL Habib",
    "NBP": "National Bank of Pakistan", "AKBL": "Askari Bank", "BOP": "Bank of Punjab",
    "SYS": "Systems Limited", "TRG": "TRG Pakistan", "NETSOL": "NetSol Technologies",
    "PTC": "Pakistan Telecommunication", "AIRLINK": "Air Link Communication",
    "FFC": "Fauji Fertilizer", "EFERT": "Engro Fertilizers", "ENGROH": "Engro Holdings",
    "FATIMA": "Fatima Fertilizer", "LOTCHEM": "Lotte Chemical", "COLG": "Colgate-Palmolive",
    "NESTLE": "Nestle Pakistan", "INDU": "Indus Motor Company", "HCAR": "Honda Atlas Cars",
    "MTL": "Millat Tractors", "PKGS": "Packages Limited", "NML": "Nishat Mills",
    "SEARL": "The Searle Company", "GLAXO": "GlaxoSmithKline Pakistan",
    "ABOT": "Abbott Laboratories Pakistan", "HINOON": "Highnoon Laboratories",
    # Financials
    "ABL": "Allied Bank", "AHCL": "Arif Habib Corporation", "AICL": "Adamjee Insurance",
    "FABL": "Faysal Bank", "HMB": "Habib Metropolitan Bank",
    "SCBPL": "Standard Chartered Bank (Pakistan)",
    # Energy_Power
    "APL": "Attock Petroleum", "ATRL": "Attock Refinery", "CNERGY": "Cnergyico PK",
    "KAPCO": "Kot Addu Power Company", "KEL": "K-Electric", "SNGP": "Sui Northern Gas Pipelines",
    "SSGC": "Sui Southern Gas Company",
    # Cement_Construction
    "BWCL": "Bestway Cement", "KOHC": "Kohat Cement", "PIOC": "Pioneer Cement",
    "POWER": "Power Cement", "THALL": "Thal Limited", "ISL": "International Steels",
    "INIL": "International Industries",
    # Pharmaceuticals
    "AGP": "AGP Limited", "CPHL": "Citi Pharma", "HALEON": "Haleon Pakistan",
    "IBFL": "Ibrahim Fibres", "SHFA": "Shifa International Hospitals",
    # Fertilizers_Chemicals
    "GHGL": "Ghani Glass", "LCI": "Lucky Core Industries",
    "PABC": "Pakistan Aluminium Beverage Cans", "EPCL": "Engro Polymer and Chemicals",
    # Consumer_Autos
    "ATLH": "Atlas Honda", "BNWM": "Bannu Woollen Mills", "FFL": "Fauji Foods",
    "GADT": "Gadoon Textile Mills", "GAL": "Ghandhara Automobiles",
    "GHNI": "Ghandhara Industries", "HUMNL": "Hum Network", "ILP": "Interloop",
    "JDWS": "JDW Sugar Mills", "JVDC": "Javedan Corporation", "KTML": "Kohinoor Textile Mills",
    "MEHT": "Mahmood Textile Mills", "MUREB": "Murree Brewery", "NATF": "National Foods",
    "PAEL": "Pak Elektron", "PAKT": "Pakistan Tobacco Company",
    "PGLC": "Pak-Gulf Leasing Company", "PIBTL": "Pakistan International Bulk Terminal",
    "PSEL": "Pakistan Services", "RMPL": "Rafhan Maize Products", "SAZEW": "Sazgar Engineering Works",
    "SRVI": "Service Industries", "SSOM": "S.S. Oil Mills", "TGL": "Tariq Glass Industries",
    "UNITY": "Unity Foods", "UPFL": "Unilever Pakistan Foods", "YOUW": "Yousaf Weaving Mills",
    # Financials (PSX is itself a listed company)
    "PSX": "Pakistan Stock Exchange",
}

TICKER_TO_SECTOR = {t: s for s, ts in SECTOR_MAP.items() for t in ts}


class StocksEngine(RoutedEngine):
    asset_class = AssetClass.STOCK
    name = "PSX"
    currency = "PKR"
    unit = "PKR/share"

    def __init__(self):
        Engine.__init__(self)
        self.models_dir = settings.stocks_models
        # PSX keeps its routing table beside the artifacts rather than in results/.
        self.routing = self.load_json(settings.stocks_models / "ensemble_weights.json")
        self._file_for: dict[str, tuple[str, str]] = {}
        # Model-input frames: the engineered features the serving artifacts were
        # trained on, produced by training-scripts-new/psx/01_feature_engineering.py
        # from the re-collected data-new/psx-data OHLCV. The older
        # data/stocks/clustered frames expose the same 20 columns in the same
        # order, so serving from them would raise nothing and silently score
        # these models against a different price source - see config.data_ready_dir.
        root = settings.data_ready_dir / "psx"
        if root.exists():
            for sector_dir in root.iterdir():
                if sector_dir.is_dir():
                    for path in sector_dir.glob("*.csv"):
                        self._file_for[path.stem] = (sector_dir.name, path.name)
        logger.info("PSX engine ready: %d tickers across %d sectors",
                    len(self._file_for), len({s for s, _ in self._file_for.values()}))

    def _build_catalog(self) -> dict[str, Asset]:
        return {
            ticker: Asset(
                ticker=ticker,
                name=NAMES.get(ticker, ticker),
                asset_class=self.asset_class,
                group=sector,
                currency="PKR",
                unit="PKR/share",
            )
            for ticker, (sector, _) in self._file_for.items()
        }

    @lru_cache(maxsize=128)
    def _frame(self, ticker: str) -> pd.DataFrame:
        sector, filename = self._file_for[ticker]
        path = settings.data_ready_dir / "psx" / sector / filename
        return pd.read_csv(path, index_col="Date", parse_dates=True)

    @lru_cache(maxsize=128)
    def _price_frame(self, ticker: str) -> pd.DataFrame:
        path = settings.data_dir / "stocks" / "raw" / f"{ticker}.csv"
        if not path.exists():
            raise EngineError(f"Price history missing for {ticker}")
        return pd.read_csv(path, index_col="Date", parse_dates=True)

    @lru_cache(maxsize=128)
    def _chart_frame(self, ticker: str) -> pd.DataFrame:
        """Display price series - the freshly re-collected PSX DPS OHLCV under
        data-new/, which is what the chart and quoted price should reflect.
        Falls back to the (feature-pipeline) stored price file if a ticker
        somehow has no data-new export yet."""
        path = settings.data_new_dir / "psx-data" / f"{ticker}.csv"
        if path.exists():
            return pd.read_csv(path, index_col="Date", parse_dates=True)
        return self._price_frame(ticker)

    def _close(self, ticker: str) -> pd.Series:
        df = self._chart_frame(ticker)
        col = "Adj Close" if "Adj Close" in df.columns else "Close"
        return df[col].dropna()

    def quote(self, ticker: str) -> Quote:
        asset = self.require(ticker)
        close = self._close(asset.ticker)
        change = None
        if len(close) >= 2 and close.iloc[-2]:
            change = round(float((close.iloc[-1] / close.iloc[-2] - 1) * 100), 2)
        return Quote(
            ticker=asset.ticker,
            name=asset.name,
            asset_class=self.asset_class,
            group=asset.group,
            currency="PKR",
            unit="PKR/share",
            current_price=round(float(close.iloc[-1]), 4),
            change_24h_pct=change,
        )

    def forecast(self, ticker: str, *, live: bool = True) -> Forecast:
        asset = self.require(ticker)
        sector = asset.group
        df = self._frame(asset.ticker)
        close = self._close(asset.ticker)

        live_quote = live_prices.fetch_latest(asset.ticker, self.asset_class) if live else None
        if live_quote is not None:
            current_price = live_quote.price
            as_of = live_quote.as_of
            price_source = "live"
            change_24h_pct = round(live_quote.change_pct, 2) if live_quote.change_pct is not None else None
        else:
            current_price = float(close.iloc[-1])
            as_of = close.index[-1].strftime("%Y-%m-%d")
            price_source = "stored"
            change_24h_pct = self.quote(asset.ticker).change_24h_pct

        feature_cols = [
            c for c in df.columns
            if not c.startswith("Target_") and pd.api.types.is_numeric_dtype(df[c])
        ]
        X_raw = df.iloc[-1:][feature_cols].to_numpy(dtype=np.float64)
        if not np.isfinite(X_raw).all():
            X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)

        scaler = self._scaler(sector)
        X_scaled = self._scale(scaler, X_raw)

        warnings_out: list[str] = []
        horizons = self._forecast_horizons(
            sector, X_scaled, current_price, settings.horizons, warnings_out
        )
        if not horizons:
            raise EngineError(f"No horizon could be forecast for {asset.ticker}")

        history = [
            PricePoint(date=idx.strftime("%Y-%m-%d"), price=round(float(v), 4))
            for idx, v in close.tail(400).items()
        ]
        # `name` drives the Google News query that fills PSX's gaps in
        # TradingView coverage - see news_live._psx_google_catalysts.
        catalysts = (
            news_live.fetch_latest(asset.ticker, self.asset_class, name=asset.name)
            if live
            else []
        )
        headline = horizons[len(horizons) // 2].predicted_return_pct

        return Forecast(
            ticker=asset.ticker,
            name=asset.name,
            asset_class=self.asset_class,
            group=sector,
            currency="PKR",
            unit="PKR/share",
            current_price=round(current_price, 4),
            change_24h_pct=change_24h_pct,
            as_of=as_of,
            price_source=price_source,
            action=self.action_of(headline),
            horizons=horizons,
            history=history,
            catalysts=catalysts,
            warnings=warnings_out,
        )
