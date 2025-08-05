import pandas as pd
from pathlib import Path
from datetime import datetime

from src.fetch_data import download_data
from src.detect import detect_anomalies, apply_pretrained_anomalies
from src.visualize import plot_anomalies
from src.config import TRAIN_TICKERS, TEST_TICKERS, DATA_DIR

DATE_FORMAT = "%Y-%m-%d"

def load_all_data(tickers):
    frames = []
    base = Path(DATA_DIR)

    for t in tickers:
        path = base / f"{t}.csv"
        # 1) Read raw CSV, index_col=0
        df = pd.read_csv(str(path), index_col=0)

        # 2) Drop any index labels that aren’t in YYYY-MM-DD form.
        #    This will remove stray rows such as a header-leftover "Ticker"
        mask = df.index.astype(str).str.match(r"\d{4}-\d{2}-\d{2}")
        if not mask.all():
            df = df.loc[mask]

        # 3) Now parse the cleaned index to actual datetimes
        df.index = pd.to_datetime(
            df.index.astype(str),
            format=DATE_FORMAT
        )

        # 4) If you still want a “date” column alongside the index
        df["date"] = df.index

        # 5) Cast all data cols to numeric, add ticker
        df = df.apply(pd.to_numeric, errors="coerce")
        df["ticker"] = t

        frames.append(df)

    return pd.concat(frames, axis=0)


def run_pipeline():
    download_data()

    df_train = load_all_data(TRAIN_TICKERS)
    df_train, models = detect_anomalies(df_train)
    for t in TRAIN_TICKERS:
        df_t = df_train[df_train["ticker"] == t]
        plot_anomalies(df_t, title=f"{t} Anomalies (Train)")

    df_test_raw = load_all_data(TEST_TICKERS)
    df_test, _ = apply_pretrained_anomalies(df_test_raw, models)
    for t in TEST_TICKERS:
        df_t = df_test[df_test["ticker"] == t]
        plot_anomalies(df_t, title=f"{t} Anomalies (Test)")


if __name__ == "__main__":
    run_pipeline()