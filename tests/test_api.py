"""
End-to-end API tests.

Exercises the real engines against the real artifacts - no mocks. The client
fixture is session-scoped because engine startup deserializes registries and
reads the 94MB MUFAP export.
"""
from __future__ import annotations

import warnings

import pytest
from fastapi.testclient import TestClient

warnings.filterwarnings("ignore")


@pytest.fixture(scope="session")
def client():
    from backend.main import app

    with TestClient(app) as c:
        yield c


# --- System -----------------------------------------------------------------


def test_health_reports_engines(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in {"ok", "degraded"}
    assert len(body["engines"]) == 4
    online = [e for e in body["engines"] if e["online"]]
    assert len(online) == 4, f"engines offline: {[e for e in body['engines'] if not e['online']]}"


def test_stats(client):
    body = client.get("/api/stats").json()
    assert body["engines_online"] == 4
    assert body["total_assets"] > 200
    assert body["horizons"] == [7, 14, 28, 42, 60, 90, 120]


# --- Assets -----------------------------------------------------------------


def test_asset_catalog_spans_all_classes(client):
    body = client.get("/api/assets", params={"limit": 2000}).json()
    classes = {a["asset_class"] for a in body["assets"]}
    assert classes == {"crypto", "commodity", "stock", "mutual_fund"}


@pytest.mark.parametrize(
    "asset_class,minimum",
    [
        ("commodity", 6),
        # 20 of 26: GRT, IMX, SUI, RNDR, APT and UNI have stored price history
        # ending 490-1596 days behind the rest of the universe and are
        # withheld. See CryptoEngine._quarantine.
        ("crypto", 20),
        ("stock", 90),
        # 81 of 82 data-ready clusters: Meezan Pakistan ETF has a feature frame
        # but no NAV in data-new or the raw export, and a forecast is a return
        # applied to a NAV - so it is excluded at catalog build rather than
        # 503-ing on first click. See MUFAPEngine._drop_navless.
        ("mutual_fund", 81),
    ],
)
def test_per_class_counts(client, asset_class, minimum):
    body = client.get("/api/assets", params={"asset_class": asset_class}).json()
    assert body["count"] >= minimum


def test_search_filters(client):
    body = client.get("/api/assets", params={"search": "gold"}).json()
    assert body["count"] >= 1
    assert any("gold" in a["ticker"].lower() or "Gold" in a["name"] for a in body["assets"])


def test_unknown_asset_is_404(client):
    assert client.get("/api/assets/NOT_A_REAL_TICKER").status_code == 404


# --- Forecasts --------------------------------------------------------------


@pytest.mark.parametrize(
    "ticker,asset_class",
    [
        ("gold", "commodity"),
        ("copper", "commodity"),   # reverted champions, width 25 vs scaler 34
        ("wheat", "commodity"),    # width 28 vs scaler 37
        ("BTC", "crypto"),         # regression guard: cluster-map pinning
        ("ETH", "crypto"),
        ("SYS", "stock"),
        ("OGDC", "stock"),
    ],
)
def test_forecast_shape_and_sanity(client, ticker, asset_class):
    r = client.get(f"/api/forecast/{ticker}", params={"asset_class": asset_class})
    assert r.status_code == 200, r.text
    f = r.json()

    assert f["current_price"] > 0
    assert f["horizons"], "no horizon produced a forecast"
    assert f["action"] in {"STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"}

    for h in f["horizons"]:
        assert h["horizon_days"] in (7, 14, 28, 42, 60, 90, 120)
        assert h["direction"] in {"up", "down", "neutral"}
        assert h["projected_price"] > 0, f"{ticker} {h['horizon_days']}d went non-positive"
        # A multi-horizon forecast beyond this is a broken model, not a market call.
        assert abs(h["predicted_return_pct"]) < 200, (
            f"{ticker} {h['horizon_days']}d returned "
            f"{h['predicted_return_pct']}% - implausible"
        )
        if h["lower_bound"] is not None and h["upper_bound"] is not None:
            assert h["lower_bound"] <= h["upper_bound"]


def test_commodity_forecast_has_shap_drivers(client):
    f = client.get("/api/forecast/gold", params={"asset_class": "commodity"}).json()
    assert f["drivers"], "SHAP attribution missing"
    assert all(d["direction"] in {"up", "down", "neutral"} for d in f["drivers"])


def test_history_present_for_charting(client):
    f = client.get("/api/forecast/gold", params={"asset_class": "commodity"}).json()
    assert len(f["history"]) > 100
    assert f["history"][0]["date"] < f["history"][-1]["date"]


def test_mufap_forecast_in_rupees(client):
    body = client.get("/api/assets", params={"asset_class": "mutual_fund"}).json()
    ticker = body["assets"][0]["ticker"]
    f = client.get(f"/api/forecast/{ticker}", params={"asset_class": "mutual_fund"}).json()
    assert f["currency"] == "PKR"
    assert f["current_price"] > 0


def test_withheld_asset_is_not_served(client):
    """UNI's stored history is a mislabeled download; it must not be forecastable."""
    body = client.get("/api/assets", params={"asset_class": "crypto"}).json()
    assert "UNI" not in {a["ticker"] for a in body["assets"]}


# --- Models -----------------------------------------------------------------


def test_leaderboard_returns_real_metrics(client):
    lb = client.get(
        "/api/models/leaderboard",
        params={"horizon_days": 60, "asset_class": "commodity"},
    ).json()
    assert lb["entries"]
    scores = lb["entries"][0]["scores"]
    assert any(s["directional_accuracy"] is not None for s in scores)
    assert any(s["mae"] is not None for s in scores)


def test_leaderboard_rejects_bad_horizon(client):
    r = client.get("/api/models/leaderboard", params={"horizon_days": 999})
    assert r.status_code == 422


def test_available_models_documents_coverage(client):
    body = client.get("/api/models/available").json()
    assert set(body) == {"commodity", "crypto", "stock", "mutual_fund"}
    assert "PatchTST" not in str(body), "PatchTST was never trained in this system"
