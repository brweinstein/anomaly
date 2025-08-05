import pandas as pd
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.trend import MACD
from ta.volume import OnBalanceVolumeIndicator
from scipy.stats import skew, kurtosis

def compute_features(df):
   df["return"]     = df["Close"].pct_change(fill_method=None)
   df["volatility"] = df["return"].rolling(window=20).std()

   rsi = RSIIndicator(close=df["Close"])
   df["rsi"] = rsi.rsi()

   macd = MACD(close=df["Close"])
   df["macd_diff"] = macd.macd_diff()

   bb = BollingerBands(close=df["Close"])
   df["bollinger_h"] = bb.bollinger_hband()
   df["bollinger_l"] = bb.bollinger_lband()

   obv = OnBalanceVolumeIndicator(close=df["Close"], volume=df["Volume"])
   df["obv"] = obv.on_balance_volume()

   # Statistical features
   df["skew_10"] = df["return"].rolling(window=10).apply(skew, raw=True)
   df["kurt_10"] = df["return"].rolling(window=10).apply(kurtosis, raw=True)

   df["rolling_mean"] = df["Close"].rolling(window=20).mean()

   df = df.dropna()
   return df
