# Market Anomaly Detector
Detects unusual activity in stock prices using unsupervised learning models like Isolation Forest and technical indicators.

### Features:
- Real stock data via Yahoo Finance
- Technical indicator calculation (returns, moving averages, etc.)
- ML-based anomaly detection (Isolation Forest)
- Time series visualization with anomalies marked
- (Optional) Streamlit dashboard or news analysis

### Tech Stack:
Python, yfinance, pandas, scikit-learn, matplotlib/plotly

## Quickstart
```bash
pip install -r requirements.txt
python main.py
```
Will save graphs to data/graphs

## Output Example
![Apple](data/graphs/AAPL_Anomalies_(Train).png)
![SPY](data/graphs/GOOG_Anomalies_(Test).png)
Overfitting?