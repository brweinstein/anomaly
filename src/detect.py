import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.features import compute_features

def detect_anomalies(
   df: pd.DataFrame,
   feature_cols: list[str] | None = None,
   iso_cont: float = 0.02,
   svm_nu: float = 0.05,
   vote_weight: tuple[float, float] = (0.7, 0.3),
   vote_thr: float = 0.3,
   z_thr: float = 2.0
) -> tuple[pd.DataFrame, dict]:
   # 1) Feature engineering
   feats = compute_features(df).copy()

   # 2) Fit a linear trend
   X_trend = np.arange(len(feats)).reshape(-1, 1)             # numpy array
   y_trend = feats["Close"].to_numpy()                       # numpy array, shape (n,)
   ridge   = Ridge(alpha=1.0).fit(X_trend, y_trend)

   feats["trend"]    = ridge.predict(X_trend)                # shape (n,)
   feats["residual"] = feats["Close"] - feats["trend"]

   # 3) Rolling z-score of residual
   roll = feats["residual"].rolling(20)
   feats["z_residual"] = (feats["residual"] - roll.mean()) / roll.std()

   # 4) Assemble features for anomaly detectors
   if feature_cols is None:
      feature_cols = [
         "return", "volatility", "rsi", "macd_diff",
         "bollinger_h", "bollinger_l", "obv",
         "skew_10", "kurt_10", "z_residual"
      ]

   X = feats[feature_cols].dropna()
   idx = X.index

   # 5) Standardize
   scaler = StandardScaler().fit(X)
   Xs     = scaler.transform(X)

   # 6) Fit detectors
   iso   = IsolationForest(contamination=iso_cont, random_state=42)
   svm   = OneClassSVM(nu=svm_nu, kernel="rbf", gamma="auto")

   iso_pred = iso.fit_predict(Xs)    #  1 = normal, -1 = anomaly
   svm_pred = svm.fit_predict(Xs)    #  1 = normal, -1 = anomaly

   # 7) Build flag Series indexed by the scored rows
   iso_flags = pd.Series((iso_pred == -1).astype(int), index=idx, name="iso_flag")
   svm_flags = pd.Series((svm_pred == -1).astype(int), index=idx, name="svm_flag")

   # 8) Join flags back, filling unscored rows with 0
   feats = feats.join(iso_flags, how="left").fillna({"iso_flag": 0})
   feats = feats.join(svm_flags, how="left").fillna({"svm_flag": 0})
   feats["iso_flag"] = feats["iso_flag"].astype(int)
   feats["svm_flag"] = feats["svm_flag"].astype(int)

   # 9) Ensemble vote + z-score threshold
   w_iso, w_svm = vote_weight
   feats["ensemble_score"] = w_iso * feats["iso_flag"] + w_svm * feats["svm_flag"]
   feats["anomaly"] = (
      (feats["ensemble_score"] >= vote_thr) |
      (feats["z_residual"].abs() > z_thr)
   ).astype(int)

   # 10) Direction tagging
   feats["rolling_mean"] = feats["Close"].rolling(20).mean()
   feats["direction"] = np.where(
      (feats["anomaly"] == 1) & (feats["Close"] > feats["rolling_mean"]),
      "high",
      np.where(
         (feats["anomaly"] == 1) & (feats["Close"] < feats["rolling_mean"]),
         "low",
         "normal"
      )
   )

   return feats.dropna(), {
      "IsolationForest": iso,
      "OneClassSVM":     svm,
      "RidgeTrend":      ridge,
      "Scaler":          scaler
   }


def apply_pretrained_anomalies(
   df: pd.DataFrame,
   models: dict,
   feature_cols: list[str] | None = None,
   vote_weight: tuple[float, float] = (0.7, 0.3),
   vote_thr: float = 0.3,
   z_thr: float = 2.0
) -> tuple[pd.DataFrame, dict]:
   # recompute features + trend/residual
   feats = compute_features(df).copy()
   X_trend = np.arange(len(feats)).reshape(-1, 1)
   feats["trend"]    = models["RidgeTrend"].predict(X_trend)
   feats["residual"] = feats["Close"] - feats["trend"]

   roll = feats["residual"].rolling(20)
   feats["z_residual"] = (feats["residual"] - roll.mean()) / roll.std()

   if feature_cols is None:
      feature_cols = [
         "return", "volatility", "rsi", "macd_diff",
         "bollinger_h", "bollinger_l", "obv",
         "skew_10", "kurt_10", "z_residual"
      ]

   X   = feats[feature_cols].dropna()
   idx = X.index

   scaler = models["Scaler"]
   Xs     = scaler.transform(X)

   iso_flags = pd.Series(
      (models["IsolationForest"].predict(Xs) == -1).astype(int),
      index=idx, name="iso_flag"
   )
   svm_flags = pd.Series(
      (models["OneClassSVM"].predict(Xs) == -1).astype(int),
      index=idx, name="svm_flag"
   )

   feats = feats.join(iso_flags, how="left").fillna({"iso_flag": 0})
   feats = feats.join(svm_flags, how="left").fillna({"svm_flag": 0})
   feats["iso_flag"] = feats["iso_flag"].astype(int)
   feats["svm_flag"] = feats["svm_flag"].astype(int)

   w_iso, w_svm = vote_weight
   feats["ensemble_score"] = w_iso * feats["iso_flag"] + w_svm * feats["svm_flag"]
   feats["anomaly"] = (
      (feats["ensemble_score"] >= vote_thr) |
      (feats["z_residual"].abs() > z_thr)
   ).astype(int)

   feats["rolling_mean"] = feats["Close"].rolling(20).mean()
   feats["direction"] = np.where(
      (feats["anomaly"] == 1) & (feats["Close"] > feats["rolling_mean"]),
      "high",
      np.where(
         (feats["anomaly"] == 1) & (feats["Close"] < feats["rolling_mean"]),
         "low",
         "normal"
      )
   )

   return feats.dropna(), models