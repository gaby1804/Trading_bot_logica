# Trading_bot_logica
# 📈 AI-Driven Backtesting with Alpaca and Net Change SDEVs

Este proyecto realiza un análisis automatizado y backtesting de estrategias de trading basadas en aprendizaje automático y estadísticas del cambio porcentual neto (`Net Change SDEVs`). Utiliza datos históricos de la API de Alpaca Markets, indicadores técnicos avanzados y un clasificador Random Forest para generar señales de compra y venta.

## 🚀 Funcionalidades Principales

- 🔄 Descarga de datos históricos minuto a minuto desde Alpaca para múltiples años.
- 📊 Cálculo de indicadores técnicos avanzados como MACD, RSI, ATR, Bollinger Bands, etc.
- 🧠 Clasificación de velas con `Net Change SDEVs` y entrenamiento de un modelo Random Forest.
- 💹 Backtesting optimizado con control de riesgo y tamaño de posición ajustado al ATR.
- 📆 Análisis detallado de rentabilidades por año, mes y hora del día.
- 📈 Cálculo de métricas de rendimiento como TWR anual, drawdown y ratio de Sharpe.

## 🛠️ Requisitos

Instala las dependencias ejecutando:

```bash
pip install alpaca-py ta scikit-learn pandas numpy
