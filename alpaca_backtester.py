# pip install alpaca-py ta scikit-learn
import pandas as pd
import numpy as np
import ta
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# Configuración de las claves de Alpaca (reemplaza con tus propias claves)
api_key = "PKZVCAWZ9F9A0M4ISSTD"
secret_key = "19bUAxLayh5K1Vbps4b3NyaWT5R0I7tmKdNdQJYg"

# Inicializar el cliente de Alpaca
stock_historical_data_client = StockHistoricalDataClient(api_key, secret_key)

def print_log(message):
    current_time = datetime.now(ZoneInfo("UTC")).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] {message}")

... (truncated here, full code continues below in actual output)
