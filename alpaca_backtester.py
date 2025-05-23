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

def descargar_datos(ticker, timeframe):
    print_log(f"Iniciando descarga de datos por años calendario completos para {ticker}")

    # Configurar fechas por años completos
    years = [2022, 2023, 2024]
    all_data = []

    for year in years:
        start_date = datetime(year, 1, 1).replace(tzinfo=ZoneInfo("America/New_York"))
        end_date = datetime(year, 12, 31, 23, 59, 59).replace(tzinfo=ZoneInfo("America/New_York"))

        print_log(f"Descargando datos del año {year}")

        # Descargar por meses para evitar límite de datos
        current_start = start_date
        while current_start < end_date:
            # Calcular fin del mes actual
            if current_start.month == 12:
                current_end = datetime(current_start.year + 1, 1, 1).replace(tzinfo=ZoneInfo("America/New_York"))
            else:
                current_end = datetime(current_start.year, current_start.month + 1, 1).replace(tzinfo=ZoneInfo("America/New_York"))

            current_end = min(current_end, end_date)

            print_log(f"Descargando {year}-{current_start.month:02d}")

            req = StockBarsRequest(
                symbol_or_symbols=[ticker],
                timeframe=TimeFrame(5, timeframe),
                start=current_start,
                end=current_end,
                limit=10000
            )

            try:
                chunk_data = stock_historical_data_client.get_stock_bars(req).df
                if not chunk_data.empty:
                    all_data.append(chunk_data)
                    print_log(f"Descargadas {len(chunk_data)} barras para {year}-{current_start.month:02d}")
                else:
                    print_log(f"No hay datos para {year}-{current_start.month:02d}")
            except Exception as e:
                print_log(f"Error descargando datos para {year}-{current_start.month:02d}: {str(e)}")

            current_start = current_end

    if not all_data:
        print_log("No se pudieron obtener datos históricos")
        return pd.DataFrame()

    final_data = pd.concat(all_data)
    print_log(f"Descarga completada - Total de barras: {len(final_data)}")
    print_log(f"Rango de fechas: {final_data.index.min()} a {final_data.index.max()}")

    return final_data

def add_enhanced_indicators(df):
    print_log("Añadiendo indicadores técnicos mejorados...")

    # --- Aquí se mantienen todos los indicadores originales (MACD, RSI, etc.) por si los quieres conservar. ---
    # --- Pero no se usarán en la clasificación. ---
    
    # Indicadores MACD existentes
    macd = ta.trend.MACD(
        df['close'],
        window_slow=26,
        window_fast=12,
        window_sign=9
    )
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()

    # Indicadores de volumen existentes
    df['Volume_EMA'] = ta.trend.EMAIndicator(df['volume'], window=20).ema_indicator()
    df['Volume_Change'] = df['volume'].pct_change()

    # ATR existente
    df['ATR'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()

    # SMA de 20 periodos
    df['SMA_20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()

    # Bandas de Bollinger de 20 periodos
    bollinger = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['BB_upper'] = bollinger.bollinger_hband()
    df['BB_lower'] = bollinger.bollinger_lband()

    # Distancia del precio a la SMA (z-score de 20 periodos)
    df['Price_to_SMA'] = (df['close'] - df['SMA_20']) / df['close'].rolling(window=20).std()

    # RSI
    df['RSI'] = ta.momentum.RSIIndicator(df['close']).rsi()

    # Fuerza de tendencia (absoluto del Price_to_SMA)
    df['Trend_Strength'] = abs(df['Price_to_SMA'])

    # EMA 200
    df['EMA_200'] = ta.trend.EMAIndicator(df['close'], window=200).ema_indicator()
    df['Trend_Filter'] = df['close'] > df['EMA_200']

    # --- Nuevo cálculo de Net Change SDEVs ---
    df['net_change'] = df['close'].diff()  # cambio neto entre esta vela y la anterior
    df['net_change_std'] = df['net_change'].rolling(window=20).std()  # std rolling (20 velas)
    # Para evitar problemas de división por 0, usamos np.where:
    df['net_change_sdevs'] = np.where(
        df['net_change_std'] != 0,
        df['net_change'] / df['net_change_std'],
        0
    )

    print_log("Indicadores técnicos mejorados añadidos")
    return df

def enhanced_classify_candle(row):
    """
    Clasificamos la vela en función del número de desviaciones estándar (Z-score)
    que representa el movimiento del precio respecto a la vela anterior.
    """
    threshold = 1.0  # Puedes ajustar este umbral a tu gusto

    if row['net_change_sdevs'] > threshold:
        return 'Bullish'
    elif row['net_change_sdevs'] < -threshold:
        return 'Bearish'
    else:
        return 'Neutral'

def analyze_trading_hours(trades):
    print_log("Analizando distribución horaria de operaciones...")

    # Convertir todos los tiempos a EST/EDT (hora de NY)
    hourly_stats = {hour: {'total': 0, 'wins': 0, 'returns': []} for hour in range(24)}

    for trade in trades:
        hour = trade['entry_time'].hour
        hourly_stats[hour]['total'] += 1
        if trade['type'] == 'win':
            hourly_stats[hour]['wins'] += 1
        hourly_stats[hour]['returns'].append(trade['return'])

    print("\nAnálisis por Hora de Operación (EST/EDT):")
    print("Hora  | Operaciones | Win Rate | Retorno Promedio")
    print("-" * 50)

    trading_hours = []
    for hour in range(24):
        stats = hourly_stats[hour]
        if stats['total'] > 0:
            win_rate = (stats['wins'] / stats['total']) * 100
            avg_return = np.mean(stats['returns']) * 100
            trading_hours.append({
                'hour': hour,
                'trades': stats['total'],
                'win_rate': win_rate,
                'avg_return': avg_return
            })
            print(f"{hour:02d}:00 | {stats['total']:10d} | {win_rate:7.2f}% | {avg_return:7.2f}%")

    # Encontrar las mejores horas
    best_hours = sorted(trading_hours, key=lambda x: x['avg_return'], reverse=True)[:3]
    print("\nMejores Horas de Trading:")
    for hour in best_hours:
        print(f"- {hour['hour']:02d}:00 EST: {hour['trades']} operaciones, {hour['win_rate']:.2f}% win rate, {hour['avg_return']:.2f}% retorno promedio")

    return hourly_stats

def is_market_hours(timestamp):
    """
    Verifica si el timestamp está dentro del horario de mercado (9:30 AM - 4:00 PM EST/EDT)
    """
    # Convertir a hora de Nueva York
    ny_time = timestamp.astimezone(ZoneInfo("America/New_York"))
    
    # Verificar si es día de semana (0 = lunes, 4 = viernes)
    if ny_time.weekday() > 4:  # Es fin de semana
        return False
        
    # Crear objetos de tiempo para apertura y cierre del mercado
    market_open = ny_time.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = ny_time.replace(hour=16, minute=0, second=0, microsecond=0)
    
    # Verificar si está dentro del horario de mercado
    return market_open <= ny_time <= market_close

def calculate_position_size(capital, risk_per_trade, entry_price, stop_loss):
    """
    Calcula cuántas acciones comprar/vender dependiendo del riesgo por operación.
    """
    # Evitar división por 0 en caso de que stop_loss == entry_price
    risk_amount = abs(entry_price - stop_loss)
    if risk_amount == 0:
        return 0
    # Capital total que arriesgaremos
    total_risk = capital * risk_per_trade
    # Número de acciones
    shares = total_risk // risk_amount
    return int(shares)

def optimized_backtest(data, initial_capital=100000, risk_per_trade=0.01, debug=False, sample_size=10):
    print_log("Ejecutando backtesting optimizado...")
    
    # Filtrar datos fuera del horario de mercado
    data = data[data.index.map(is_market_hours)]
    
    back_df = data[['open', 'high', 'low', 'close', 'signal', 'ATR']].copy()
    back_df.columns = ["open", "high", "low", "price", "signal", "ATR"]

    ret_i = []
    trades = []
    last_trade_idx = -np.inf
    current_capital = initial_capital
    max_capital = initial_capital

    for j in range(len(back_df)):
        if back_df["signal"].iloc[j] != 0 and j > last_trade_idx + 1:
            entry_time = back_df.index[j]
            
            if not is_market_hours(entry_time):
                continue
                
            entry_price = back_df["price"].iloc[j]
            direction = back_df["signal"].iloc[j]
            atr = back_df["ATR"].iloc[j]

            if direction == 1:
                stop_loss = entry_price - 0.60 * atr
                take_profit = entry_price + 1.25 * atr
            else:
                stop_loss = entry_price + 0.60 * atr
                take_profit = entry_price - 1.25 * atr

            # Calcular tamaño de la posición
            position_size = calculate_position_size(
                capital=current_capital,
                risk_per_trade=risk_per_trade,
                entry_price=entry_price,
                stop_loss=stop_loss
            )
            
            # Si el tamaño de la posición es 0, saltamos esta operación
            if position_size == 0:
                continue

            trade_candles = []
            trade_candles.append({
                'time': entry_time,
                'open': back_df["open"].iloc[j],
                'high': back_df["high"].iloc[j],
                'low': back_df["low"].iloc[j],
                'close': back_df["price"].iloc[j]
            })

            for k in range(j + 1, len(back_df)):
                current_high = back_df["high"].iloc[k]
                current_low = back_df["low"].iloc[k]
                exit_time = back_df.index[k]
                
                trade_candles.append({
                    'time': exit_time,
                    'open': back_df["open"].iloc[k],
                    'high': back_df["high"].iloc[k],
                    'low': back_df["low"].iloc[k],
                    'close': back_df["price"].iloc[k]
                })

                if direction == 1:  # LONG
                    if current_high >= take_profit:
                        ret = (take_profit - entry_price) / entry_price
                        pnl = position_size * (take_profit - entry_price)
                        current_capital += pnl
                        max_capital = max(max_capital, current_capital)
                        
                        ret_i.append(ret)
                        trades.append({
                            'trade_index': len(trades),
                            'type': 'win',
                            'return': ret,
                            'pnl': pnl,
                            'position_size': position_size,
                            'entry_time': entry_time,
                            'exit_time': exit_time,
                            'entry_price': entry_price,
                            'exit_price': take_profit,
                            'direction': 'LONG',
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'candles': trade_candles,
                            'capital': current_capital
                        })
                        last_trade_idx = k
                        break
                    elif current_low <= stop_loss:
                        ret = (stop_loss - entry_price) / entry_price
                        pnl = position_size * (stop_loss - entry_price)
                        current_capital += pnl
                        max_capital = max(max_capital, current_capital)
                        
                        ret_i.append(ret)
                        trades.append({
                            'trade_index': len(trades),
                            'type': 'loss',
                            'return': ret,
                            'pnl': pnl,
                            'position_size': position_size,
                            'entry_time': entry_time,
                            'exit_time': exit_time,
                            'entry_price': entry_price,
                            'exit_price': stop_loss,
                            'direction': 'LONG',
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'candles': trade_candles,
                            'capital': current_capital
                        })
                        last_trade_idx = k
                        break
                else:  # SHORT
                    if current_low <= take_profit:
                        ret = (entry_price - take_profit) / entry_price
                        pnl = position_size * (entry_price - take_profit)
                        current_capital += pnl
                        max_capital = max(max_capital, current_capital)
                        
                        ret_i.append(ret)
                        trades.append({
                            'trade_index': len(trades),
                            'type': 'win',
                            'return': ret,
                            'pnl': pnl,
                            'position_size': position_size,
                            'entry_time': entry_time,
                            'exit_time': exit_time,
                            'entry_price': entry_price,
                            'exit_price': take_profit,
                            'direction': 'SHORT',
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'candles': trade_candles,
                            'capital': current_capital
                        })
                        last_trade_idx = k
                        break
                    elif current_high >= stop_loss:
                        ret = (entry_price - stop_loss) / entry_price
                        pnl = position_size * (entry_price - stop_loss)
                        current_capital += pnl
                        max_capital = max(max_capital, current_capital)
                        
                        ret_i.append(ret)
                        trades.append({
                            'trade_index': len(trades),
                            'type': 'loss',
                            'return': ret,
                            'pnl': pnl,
                            'position_size': position_size,
                            'entry_time': entry_time,
                            'exit_time': exit_time,
                            'entry_price': entry_price,
                            'exit_price': stop_loss,
                            'direction': 'SHORT',
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'candles': trade_candles,
                            'capital': current_capital
                        })
                        last_trade_idx = k
                        break

    if debug and trades:
        print("\n=== MUESTRA ALEATORIA DE OPERACIONES ===")
        sample_size = min(sample_size, len(trades))
        selected_trades = random.sample(trades, sample_size)
        
        for trade in sorted(selected_trades, key=lambda x: x['entry_time']):
            entry_time_ny = trade['entry_time'].astimezone(ZoneInfo("America/New_York"))
            exit_time_ny = trade['exit_time'].astimezone(ZoneInfo("America/New_York"))
            
            print(f"\n--- Operación #{trade['trade_index']} ---")
            print(f"Dirección: {trade['direction']}")
            print(f"Tamaño: {trade['position_size']:.0f} acciones")
            print(f"Capital: ${trade['capital']:,.2f}")
            print(f"Entrada: {entry_time_ny} @ ${trade['entry_price']:.2f}")
            print(f"Stop Loss: ${trade['stop_loss']:.2f}")
            print(f"Take Profit: ${trade['take_profit']:.2f}")
            print(f"Salida: {exit_time_ny} @ ${trade['exit_price']:.2f}")
            print(f"P&L: ${trade['pnl']:,.2f}")
            print(f"Duración: {trade['exit_time'] - trade['entry_time']}")
            print(f"Resultado: {trade['type'].upper()} ({trade['return']*100:.2f}%)")
            
            print("\nVelas durante la operación:")
            print("Tiempo (EST/EDT)                | Open    | High    | Low     | Close")
            print("-" * 70)
            for candle in trade['candles']:
                ny_time = candle['time'].astimezone(ZoneInfo("America/New_York"))
                print(f"{ny_time} | {candle['open']:7.2f} | {candle['high']:7.2f} | {candle['low']:7.2f} | {candle['close']:7.2f}")

        print("\n=== RESUMEN GENERAL ===")
        print(f"Capital Inicial: ${initial_capital:,.2f}")
        print(f"Capital Final: ${current_capital:,.2f}")
        print(f"Retorno Total: {((current_capital/initial_capital)-1)*100:.2f}%")
        print(f"Máximo Drawdown: {((max_capital-current_capital)/max_capital)*100:.2f}%")
        print(f"Total de operaciones: {len(trades)}")
        wins = sum(1 for t in trades if t['type'] == 'win')
        print(f"Operaciones ganadoras: {wins} ({wins/len(trades)*100:.2f}%)")
        print(f"Retorno promedio: {np.mean([t['return'] for t in trades])*100:.2f}%")
        print(f"P&L Promedio: ${np.mean([t['pnl'] for t in trades]):,.2f}")

    return ret_i, trades

def calculate_returns(ret_i, trades, data_length, data):
    total_trades = len(trades)
    winning_trades = sum(1 for trade in trades if trade['type'] == 'win')

    if total_trades == 0:
        return 0, 0, 0, 0, {}, {}

    # Análisis por año
    yearly_stats = {}
    max_drawdown = 0

    for year in [2022, 2023, 2024]:
        year_trades = [t for t in trades if t['entry_time'].year == year]

        if year_trades:
            # Calcular retorno del año
            year_equity = 1.0
            year_peak = 1.0
            year_max_dd = 0

            for t in year_trades:
                year_equity *= (1 + t['return'])
                year_peak = max(year_peak, year_equity)
                year_dd = (year_peak - year_equity) / year_peak
                year_max_dd = max(year_max_dd, year_dd)

            year_return = (year_equity - 1) * 100  # Convertir a porcentaje
            year_wins = sum(1 for t in year_trades if t['type'] == 'win')

            yearly_stats[year] = {
                'trades': len(year_trades),
                'wins': year_wins,
                'win_rate': year_wins / len(year_trades) * 100,
                'return': year_return,
                'max_drawdown': year_max_dd * 100
            }

    # Calcular TWR
    yearly_returns = []
    for year in sorted(yearly_stats.keys()):
        yearly_returns.append(1 + (yearly_stats[year]['return'] / 100))

    # TWR Anualizado
    n_years = len(yearly_returns)
    cumulative_return = np.prod(yearly_returns)
    twr_annual = (cumulative_return ** (1/n_years)) - 1 if n_years > 0 else 0

    # Métricas de riesgo
    all_returns = [t['return'] for t in trades]
    if len(all_returns) < 2:
        std_returns = 1e-6
    else:
        std_returns = np.std(all_returns) * np.sqrt(252)  # Volatilidad anualizada aprox.
    
    risk_metrics = {
        'max_drawdown': max([stats['max_drawdown'] for stats in yearly_stats.values()]) if yearly_stats else 0,
        'sharpe_ratio': (twr_annual - 0.02) / std_returns if std_returns != 0 else 0
    }

    avg_return_per_trade = sum(all_returns) / total_trades

    return total_trades, winning_trades, avg_return_per_trade, twr_annual, yearly_stats, risk_metrics

def calculate_monthly_returns(trades):
    monthly_returns = {}
    monthly_equity = {}
    current_equity = 1.0

    # Ordenar trades por fecha
    sorted_trades = sorted(trades, key=lambda x: x['entry_time'])

    for trade in sorted_trades:
        year_month = trade['entry_time'].strftime('%Y-%m')
        if year_month not in monthly_returns:
            monthly_returns[year_month] = []
        monthly_returns[year_month].append(trade['return'])

    # Calcular equity acumulado por mes
    for year_month in sorted(monthly_returns.keys()):
        month_return = np.prod([1 + r for r in monthly_returns[year_month]]) - 1
        current_equity *= (1 + month_return)
        monthly_equity[year_month] = {
            'return': month_return * 100,
            'equity': (current_equity - 1) * 100,
            'trades': len(monthly_returns[year_month])
        }

    return monthly_equity

def print_monthly_analysis(monthly_equity):
    print("\nAnálisis Mensual de Rendimientos:")
    print("Fecha     | Retorno Mes | Retorno Acum | Operaciones")
    print("-" * 55)

    for month, stats in monthly_equity.items():
        print(f"{month}  | {stats['return']:10.2f}% | {stats['equity']:11.2f}% | {stats['trades']:11d}")

def main():
    print_log(f"Iniciando análisis por años calendario completos para el usuario JulioVerdeja")

    ticker = "AAPL"

    print_log(f"Configurando descarga de datos para {ticker}")
    data = descargar_datos(ticker, TimeFrameUnit.Minute)

    if data.empty:
        print_log("No se pudo proceder con el análisis debido a falta de datos")
        return

    print_log(f"Datos descargados exitosamente. Procesando {len(data)} barras")

    # Si el índice es MultiIndex, lo reseteamos (suele pasar cuando Alpaca devuelve varios símbolos)
    if isinstance(data.index, pd.MultiIndex):
        data = data.reset_index(level=0)
        print_log("Índice multinivel detectado y reseteado")

    print_log("Añadiendo indicadores técnicos...")
    data = add_enhanced_indicators(data)

    print_log("Preparando datos para el modelo...")
    data['Candle_Type'] = data.apply(enhanced_classify_candle, axis=1)
    data['Next_Candle_Type'] = data['Candle_Type'].shift(-1)
    data['Target'] = data['Next_Candle_Type'].map({'Bullish': 1, 'Bearish': -1, 'Neutral': 0})

    data = data.dropna()
    print_log(f"Datos preparados. Tamaño final del dataset: {len(data)} barras")

    # Usamos algunas de las columnas como features (puedes ajustar a tu gusto)
    feature_columns = ['net_change_sdevs']  # El foco es net_change_sdevs
    # Si prefieres usar más features, inclúyelas aquí: 
    # feature_columns = ['net_change_sdevs', 'MACD', 'MACD_signal', 'Volume_Change', 'RSI', 'Price_to_SMA']

    X = data[feature_columns].copy()
    y = data['Target'].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print_log("Entrenando modelo...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_split=3,
        min_samples_leaf=2,
        random_state=42,
        class_weight={0: 1, 1: 4, -1: 4}  # más peso a señales alcistas/bajistas
    )
    model.fit(X_train, y_train)

    # Generamos las señales con el modelo entrenado
    predictions = model.predict(X)
    data['signal'] = predictions

    print_log("Ejecutando backtesting...")
    ret_i, trades = optimized_backtest(data, debug=True)
    results = calculate_returns(ret_i, trades, len(data), data)
    total_trades, winning_trades, avg_return_per_trade, twr_annual, yearly_stats, risk_metrics = results

    monthly_equity = calculate_monthly_returns(trades)
    print_monthly_analysis(monthly_equity)

    hourly_analysis = analyze_trading_hours(trades)

    print("\nResultados del modelo basados en Net Change SDEVs:")
    print(f"Precisión del modelo (set de prueba): {accuracy_score(y_test, model.predict(X_test)):.4f}")

    print(f"\nEstadísticas Globales:")
    print(f"Número total de operaciones: {total_trades}")
    print(f"Operaciones ganadoras: {winning_trades}")
    print(f"Ratio de éxito global: {(winning_trades/total_trades*100 if total_trades > 0 else 0):.2f}%")
    print(f"Rendimiento promedio por operación: {avg_return_per_trade:.4f}")
    print(f"TWR Anual: {twr_annual*100:.2f}%")
    print(f"Máximo drawdown: {risk_metrics['max_drawdown']:.2f}%")
    print(f"Ratio de Sharpe: {risk_metrics['sharpe_ratio']:.2f}")

    print("\nEstadísticas por Año:")
    for year, stats in yearly_stats.items():
        print(f"\nAño {year}:")
        print(f"  Operaciones: {stats['trades']}")
        print(f"  Ratio de éxito: {stats['win_rate']:.2f}%")
        print(f"  Rendimiento: {stats['return']:.2f}%")
        print(f"  Máximo drawdown: {stats['max_drawdown']:.2f}%")

    # Mostrar retornos acumulados
    cumulative_return = 1.0
    print("\nRetornos Acumulados:")
    for year in sorted(yearly_stats.keys()):
        cumulative_return *= (1 + yearly_stats[year]['return']/100)
        print(f"  {year}: {(cumulative_return-1)*100:.2f}%")

    # Importancia de características
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nImportancia de características:")
    print(feature_importance)

    print_log("Análisis completado")

if _name_ == "_main_":
    main()
