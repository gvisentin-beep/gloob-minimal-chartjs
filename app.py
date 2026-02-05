from flask import Flask, jsonify, render_template, request
import pandas as pd
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

FILES = {
    "ls80": DATA_DIR / "ls80.csv",
    "gold": DATA_DIR / "gold.csv",
    "btc": DATA_DIR / "btc.csv",
}

# Pesi portafoglio "B" (modifica qui se vuoi)
WEIGHTS = {"ls80": 0.80, "gold": 0.10, "btc": 0.10}


# --------- Helpers ---------
def _pick_date_col(df: pd.DataFrame) -> str:
    # prova nomi tipici
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in ("date", "data", "datetime", "time"):
            return c
    # altrimenti prima colonna
    return df.columns[0]


def _pick_value_col(df: pd.DataFrame, date_col: str) -> str:
    # prova nomi tipici
    candidates = [
        "value", "valore", "close", "adj close", "adj_close", "price", "prezzo", "nav"
    ]
    cols = [c for c in df.columns if c != date_col]
    for want in candidates:
        for c in cols:
            if str(c).strip().lower() == want:
                return c
    # altrimenti la prima colonna non-data
    if len(cols) >= 1:
        return cols[0]
    raise ValueError("CSV deve avere almeno 2 colonne (data, valore)")


def _read_asset(asset: str) -> pd.Series:
    if asset not in FILES:
        raise ValueError(f"Asset non valido: {asset}")
    path = FILES[asset]
    if not path.exists():
        raise FileNotFoundError(f"File mancante: {path}")

    # lettura robusta (pandas spesso indovina; se separatore “;” va comunque)
    df = pd.read_csv(path)

    if df.shape[1] < 2:
        raise ValueError(f"CSV {asset} deve avere almeno 2 colonne (data, valore)")

    date_col = _pick_date_col(df)
    val_col = _pick_value_col(df, date_col)

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")

    df = df.dropna(subset=[date_col, val_col]).sort_values(date_col)

    s = df.set_index(date_col)[val_col]
    s.index = pd.to_datetime(s.index)
    s = s[~s.index.duplicated(keep="last")]
    return s.dropna()


def _freq_to_rule(freq: str) -> str:
    f = (freq or "monthly").strip().lower()

    # accetta anche errori di digitazione tipo "monthl"
    if f in ("d", "day", "daily", "giorno", "giornaliero"):
        return "D"
    if f in ("w", "week", "weekly", "sett", "settimanale"):
        return "W-FRI"
    if f in ("m", "mon", "month", "monthly", "mese", "mensile", "monthl"):
        # Pandas nuovi: usare ME (month end) invece di M
        return "ME"
    if f in ("q", "quarter", "quarterly", "trimestre", "trimestrale"):
        return "QE"
    if f in ("y", "year", "yearly", "annuale", "anno"):
        return "YE"

    # fallback
    return "ME"


def _resample(s: pd.Series, freq: str) -> pd.Series:
    rule = _freq_to_rule(freq)
    if rule == "D":
        out = s.asfreq("D").ffill()
    else:
        out = s.resample(rule).last()
    return out.dropna()


def _normalize_100(s: pd.Series) -> pd.Series:
    if s.empty:
        return s
    base = float(s.iloc[0])
    if base == 0:
        return s * 0.0
    return (s / base) * 100.0


def _to_payload(series: pd.Series, asset: str, freq: str) -> dict:
    series = series.dropna()
    labels = [d.strftime("%Y-%m") for d in series.index]  # per mensile, leggibile
    values = [round(float(v), 2) for v in series.values]
    base_date = series.index[0].strftime("%Y-%m-%d") if len(series) else None
    return {
        "asset": asset,
        "freq": freq,
        "base_date": base_date,
        "points": len(values),
        "labels": labels,
        "values": values,
    }


# --------- Routes ---------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/data")
def api_data():
    asset = request.args.get("asset", "ls80").strip().lower()
    freq = request.args.get("freq", "monthly").strip().lower()

    s = _read_asset(asset)
    s = _resample(s, freq)
    s = _normalize_100(s)

    return jsonify(_to_payload(s, asset, freq))


@app.route("/api/combined")
def api_combined():
    """
    Ritorna due serie NORMALIZZATE a 100, allineate sulle stesse date:
    - B: Portafoglio (LS80 80% + Gold 10% + BTC 10%)
    - C: LS80 (benchmark semplice)
    """
    freq = request.args.get("freq", "monthly").strip().lower()

    s_ls80 = _normalize_100(_resample(_read_asset("ls80"), freq))
    s_gold = _normalize_100(_resample(_read_asset("gold"), freq))
    s_btc = _normalize_100(_resample(_read_asset("btc"), freq))

    # Allinea sulle date comuni
    df = pd.concat(
        {"ls80": s_ls80, "gold": s_gold, "btc": s_btc},
        axis=1
    ).dropna()

    # se non ci sono date comuni, evita crash
    if df.empty:
        return jsonify({
            "freq": freq,
            "base_date": None,
            "points": 0,
            "labels": [],
            "portfolio": [],
            "ls80": [],
            "weights": WEIGHTS,
        })

    # Portafoglio "B" (somma pesata delle serie normalizzate)
    b = (
        df["ls80"] * WEIGHTS["ls80"]
        + df["gold"] * WEIGHTS["gold"]
        + df["btc"] * WEIGHTS["btc"]
    )

    # Benchmark "C" = LS80
    c = df["ls80"]

    labels = [d.strftime("%Y-%m") for d in df.index]
    base_date = df.index[0].strftime("%Y-%m-%d")

    return jsonify({
        "freq": freq,
        "base_date": base_date,
        "points": int(len(labels)),
        "labels": labels,
        "portfolio": [round(float(v), 2) for v in b.values],  # B
        "ls80": [round(float(v), 2) for v in c.values],       # C
        "weights": WEIGHTS,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
