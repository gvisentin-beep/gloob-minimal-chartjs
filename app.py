from flask import Flask, jsonify, render_template, request
import pandas as pd
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

FILES = {
    "ls80": DATA_DIR / "ls80.csv",
    "gold": DATA_DIR / "gold.csv",
    "btc":  DATA_DIR / "btc.csv",
}

# Pesi del portafoglio "Pigro"
WEIGHTS = {"ls80": 0.80, "gold": 0.10, "btc": 0.10}


def _pick_two_columns(df: pd.DataFrame, asset: str) -> pd.DataFrame:
    """
    Ritorna un DataFrame con colonne: ['date','value'].
    - prende la prima colonna come data (se non trova una colonna esplicita)
    - prende la prima colonna numerica come value (se non trova una colonna esplicita)
    """
    if df is None or df.shape[1] < 2:
        raise ValueError(f"CSV {asset} deve avere almeno 2 colonne (data, valore)")

    # normalizza nomi colonne
    cols = [str(c).strip().lower() for c in df.columns]
    df.columns = cols

    # prova a trovare la colonna data
    date_candidates = [c for c in df.columns if c in ("date", "data", "datetime", "time")]
    if date_candidates:
        date_col = date_candidates[0]
    else:
        date_col = df.columns[0]

    # prova a trovare la colonna valore
    value_candidates = [c for c in df.columns if c in ("value", "valore", "close", "price", "prezzo", "adj close", "adj_close")]
    if value_candidates:
        value_col = value_candidates[0]
    else:
        # prima colonna numerica (escludendo la data)
        tmp = df.copy()
        for c in tmp.columns:
            if c == date_col:
                continue
            tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
        numeric_cols = [c for c in tmp.columns if c != date_col and tmp[c].notna().any()]
        if not numeric_cols:
            raise ValueError(f"CSV {asset}: non trovo una colonna numerica per il valore.")
        value_col = numeric_cols[0]

    out = df[[date_col, value_col]].copy()
    out.columns = ["date", "value"]

    out["date"] = pd.to_datetime(out["date"], errors="coerce", utc=False)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")

    out = out.dropna(subset=["date", "value"]).sort_values("date")

    if out.shape[0] < 2:
        raise ValueError(f"CSV {asset}: dati insufficienti dopo la pulizia.")
    return out


def _read_asset(asset: str) -> pd.Series:
    if asset not in FILES:
        raise ValueError(f"Asset sconosciuto: {asset}. Usa: {list(FILES.keys())}")

    path = FILES[asset]
    if not path.exists():
        raise FileNotFoundError(f"File mancante: {path}")

    # Auto-detect separatore (funziona per ',' o ';' ecc.)
    df = pd.read_csv(path, sep=None, engine="python")

    df = _pick_two_columns(df, asset)
    s = pd.Series(df["value"].values, index=df["date"].values)
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    return s


def _normalize_to_100(s: pd.Series) -> pd.Series:
    s = s.dropna()
    if s.empty:
        return s
    base = float(s.iloc[0])
    if base == 0:
        return s * 0.0
    return (s / base) * 100.0


def _map_freq(freq: str) -> str:
    """
    Supporta: daily, weekly, monthly, quarterly, yearly
    Nota: per pandas moderni, fine mese = 'ME' (NON 'M').
    """
    f = (freq or "monthly").lower().strip()

    # accetta typo comuni
    if f in ("monthl", "mensile", "mese"):
        f = "monthly"

    mapping = {
        "daily": "D",
        "weekly": "W-FRI",
        "monthly": "ME",      # month-end
        "quarterly": "QE",    # quarter-end
        "yearly": "YE",       # year-end
    }
    return mapping.get(f, "ME")


def _resample(s: pd.Series, freq: str) -> pd.Series:
    rule = _map_freq(freq)
    # last del periodo (fine periodo)
    return s.resample(rule).last().dropna()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/data")
def api_data():
    asset = request.args.get("asset", "ls80").lower().strip()
    freq = request.args.get("freq", "monthly").lower().strip()

    s = _read_asset(asset)
    s = _resample(s, freq)
    idx = _normalize_to_100(s)

    labels = [pd.Timestamp(d).strftime("%Y-%m") for d in idx.index]
    values = [round(float(v), 2) for v in idx.values]

    return jsonify({
        "asset": asset,
        "freq": freq,
        "base_date": pd.Timestamp(idx.index[0]).strftime("%Y-%m-%d"),
        "points": len(values),
        "labels": labels,
        "values": values,
    })


@app.route("/api/combined")
def api_combined():
    """
    Ritorna 2 serie NORMALIZZATE a 100:
      - portfolio: LS80 80% + Gold 10% + BTC 10%
      - benchmark: LS80 (solo)
    """
    freq = request.args.get("freq", "monthly").lower().strip()

    s_ls80 = _resample(_read_asset("ls80"), freq)
    s_gold = _resample(_read_asset("gold"), freq)
    s_btc  = _resample(_read_asset("btc"),  freq)

    df = pd.concat({"ls80": s_ls80, "gold": s_gold, "btc": s_btc}, axis=1).dropna()

    # Portfolio (somma pesata)
    port = (df["ls80"] * WEIGHTS["ls80"] +
            df["gold"] * WEIGHTS["gold"] +
            df["btc"]  * WEIGHTS["btc"])

    port_idx = _normalize_to_100(port)
    bench_idx = _normalize_to_100(df["ls80"])

    labels = [pd.Timestamp(d).strftime("%Y-%m") for d in port_idx.index]

    return jsonify({
        "freq": freq,
        "base_date": pd.Timestamp(port_idx.index[0]).strftime("%Y-%m-%d"),
        "points": len(labels),
        "labels": labels,
        "series": {
            "portfolio": [round(float(v), 2) for v in port_idx.values],
            "benchmark": [round(float(v), 2) for v in bench_idx.reindex(port_idx.index).values],
        }
    })


# Render/Gunicorn: usa "gunicorn app:app"
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
