import os
import yfinance as yf
from src.config import ALL_TICKERS, START_DATE, END_DATE, DATA_DIR

def download_data(force_download: bool = False):
   """
   Download historical OHLC data for all tickers in ALL_TICKERS.
   Skips existing CSVs unless force_download is True.
   """
   os.makedirs(DATA_DIR, exist_ok=True)

   for ticker in ALL_TICKERS:
      outfile = os.path.join(DATA_DIR, f"{ticker}.csv")

      # skip if file already exists
      if not force_download and os.path.exists(outfile):
         print(f"Skipping {ticker} (already downloaded)")
         continue

      # fetch from Yahoo Finance
      df = yf.download(
         ticker,
         start=START_DATE,
         end=END_DATE,
         auto_adjust=False,
         progress=False
      )

      # warn if no data returned
      if df is None or df.empty:
         print(f"Warning: No data for {ticker}")
         continue

      # write to CSV
      df.to_csv(outfile)
      print(f"Downloaded {ticker}")

if __name__ == "__main__":
   download_data()