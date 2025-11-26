import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from chronos import ChronosPipeline
from sklearn.preprocessing import StandardScaler

class ChronosForecaster:
    def __init__(self, context_len: int = 512):
        """
        Inizializza Amazon Chronos (T5-Small).
        """
        self.model_name = "amazon/chronos-t5-small"

        # FIX AMD/Windows: CPU è la scelta più stabile.
        self.device = "cpu"
        self.context_len = context_len

        # Parametri indicatori (devono coincidere con MarketDataProvider per coerenza)
        self.ema_fast_span = 12
        self.ema_slow_span = 26

        print(f"[Chronos] Loading {self.model_name} on {self.device}...")

        try:
            self.pipeline = ChronosPipeline.from_pretrained(
                self.model_name,
                device_map=self.device,
                torch_dtype=torch.float32,
            )
            print("[Chronos] Model loaded successfully.")
        except Exception as e:
            print(f"[Chronos] CRITICAL ERROR: {e}")
            raise e

    def predict_candles(self, candles: List[Dict[str, Any]], timeframe: str, horizon: int) -> List[Dict[str, Any]]:
        if not candles or len(candles) < 30:
            print("[Chronos] Error: Not enough candles.")
            return []

        # 1. Preparazione Dati
        df = pd.DataFrame(candles)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.sort_values('timestamp', inplace=True)

        # Pulizia dati
        cols_to_fix = ['close', 'high', 'low', 'volume', 'rsi', 'atr']
        for col in cols_to_fix:
            if col in df.columns:
                df[col] = df[col].astype(float).ffill().fillna(0)

        # Salviamo l'ultimo stato noto per ricostruire bid/ask/spread
        last_row = df.iloc[-1]
        last_ts = last_row['timestamp']
        last_close = last_row['close']
        last_high = last_row['high']
        last_low = last_row['low']

        # Calcoliamo lo spread medio o usiamo l'ultimo
        last_spread = last_row.get('spread', 0)
        if last_spread == 0 or pd.isna(last_spread):
            # Fallback se lo spread non c'è: stima 0.01%
            last_spread = last_close * 0.0001

        # --- FEATURE ENGINEERING (Log Returns) ---
        close_vals = df['close'].values
        close_log_ret = np.log(close_vals[1:] / close_vals[:-1])

        high_vals = df['high'].values
        high_log_ret = np.log(high_vals[1:] / high_vals[:-1])

        low_vals = df['low'].values
        low_log_ret = np.log(low_vals[1:] / low_vals[:-1])

        vol_vals = df['volume'].values[1:]
        vol_log = np.log1p(vol_vals)

        rsi_vals = df['rsi'].values[1:]
        atr_vals = df['atr'].values[1:]

        # Preparazione Context
        def get_context(arr):
            tensor = torch.tensor(arr, dtype=torch.float32)
            if len(tensor) > self.context_len:
                return tensor[-self.context_len:]
            return tensor

        list_of_tensors = [
            get_context(close_log_ret),
            get_context(high_log_ret),
            get_context(low_log_ret),
            get_context(vol_log),
            get_context(rsi_vals),
            get_context(atr_vals)
        ]

        # --- SCALING ---
        scalers = [StandardScaler() for _ in range(6)]
        scaled_tensors_list = []

        for i, t in enumerate(list_of_tensors):
            data_np = t.numpy().reshape(-1, 1)
            norm_data = scalers[i].fit_transform(data_np).flatten()
            scaled_tensors_list.append(torch.tensor(norm_data, dtype=torch.float32))

        batch_context = torch.stack(scaled_tensors_list)

        # --- INFERENZA ---
        forecast = self.pipeline.predict(
            batch_context,
            prediction_length=horizon,
            num_samples=20,
            limit_prediction_length=False
        )

        # Estraiamo la Mediana (P50)
        p50_forecast = torch.quantile(forecast, 0.5, dim=1).numpy()

        # --- RICOSTRUZIONE ---
        future_candles = []
        time_delta = self._parse_timeframe_pandas(timeframe)
        current_ts = last_ts

        curr_close = last_close
        curr_high = last_high
        curr_low = last_low

        # Variabili temporanee per il loop
        pair_name = candles[0].get('pair', 'N/A')

        for i in range(horizon):
            current_ts += time_delta

            def denorm(idx, val):
                return scalers[idx].inverse_transform(val.reshape(1, -1))[0,0]

            pred_close_ret = denorm(0, p50_forecast[0][i])
            pred_high_ret  = denorm(1, p50_forecast[1][i])
            pred_low_ret   = denorm(2, p50_forecast[2][i])
            pred_vol_log   = denorm(3, p50_forecast[3][i])
            pred_rsi       = denorm(4, p50_forecast[4][i])
            pred_atr       = denorm(5, p50_forecast[5][i])

            next_open = curr_close
            next_close = curr_close * np.exp(pred_close_ret)
            next_high = curr_high * np.exp(pred_high_ret)
            next_low = curr_low * np.exp(pred_low_ret)
            next_vol = max(0, np.expm1(pred_vol_log))

            # Geometria Candela
            final_high = max(next_high, next_open, next_close)
            final_low = min(next_low, next_open, next_close)

            # Ricostruzione Bid/Ask/Mid basata sullo spread storico
            # Assumiamo spread costante per la previsione
            final_bid = next_close - (last_spread / 2)
            final_ask = next_close + (last_spread / 2)
            final_mid = next_close

            curr_close = next_close
            curr_high = final_high
            curr_low = final_low

            # Struttura IDENTICA a MarketDataProvider.getCandles
            candle = {
                'timestamp': current_ts.isoformat(), # Stringa ISO
                'pair': pair_name,
                'open': next_open,
                'high': final_high,
                'low': final_low,
                'close': next_close,
                'volume': next_vol,
                'spread': last_spread,
                'bid': final_bid,
                'ask': final_ask,
                'last': next_close, # Alias per close
                'mid': final_mid,
                'rsi': pred_rsi,
                'atr': pred_atr,
                'ema_fast': 0, # Placeholder, calcoliamo dopo
                'ema_slow': 0, # Placeholder, calcoliamo dopo
                'type': 'forecast' # Tag per distinguerle
            }
            future_candles.append(candle)

        # --- CALCOLO EMA SUI DATI FUTURI ---
        # Per avere EMA corrette, dobbiamo unirle allo storico e ricalcolarle
        # Creiamo un DF temporaneo Storico + Futuro
        df_future = pd.DataFrame(future_candles)

        # Convertiamo timestamp stringa back to datetime per il merge
        df_future['timestamp'] = pd.to_datetime(df_future['timestamp'])

        # Concatenazione
        full_df = pd.concat([df[['timestamp', 'close']], df_future[['timestamp', 'close']]], ignore_index=True)

        # Calcolo EMA Continuo
        full_df['EMA_Fast'] = full_df['close'].ewm(span=self.ema_fast_span, adjust=False).mean()
        full_df['EMA_Slow'] = full_df['close'].ewm(span=self.ema_slow_span, adjust=False).mean()

        # Ri-assegnazione valori alle candele future
        # Prendiamo solo le ultime N righe corrispondenti al forecast
        forecast_emas = full_df.iloc[-horizon:].reset_index(drop=True)

        for i, candle in enumerate(future_candles):
            candle['ema_fast'] = float(forecast_emas.iloc[i]['EMA_Fast'])
            candle['ema_slow'] = float(forecast_emas.iloc[i]['EMA_Slow'])

        return future_candles

    def _parse_timeframe_pandas(self, tf: str) -> pd.Timedelta:
        map_tf = {'m': 'min', 'h': 'H', 'd': 'D', 'w': 'W', 'mo': 'M'}
        if not tf: return pd.Timedelta(minutes=5)
        unit_char = tf[-1].lower()
        try:
            val = int(tf[:-1])
            pd_unit = map_tf.get(unit_char, 'min')
            return pd.Timedelta(value=val, unit=pd_unit)
        except:
            return pd.Timedelta(minutes=5)

if __name__ == "__main__":
    print("--- TEST CHRONOS FULL STRUCTURE ---")
    fake_candles = []
    price = 60000.0
    ts = pd.Timestamp.now()

    # Generiamo storico
    for i in range(50):
        price = price * (1 + np.random.normal(0, 0.002))
        fake_candles.append({
            'timestamp': (ts + pd.Timedelta(minutes=5*i)).isoformat(),
            'open': price, 'high': price*1.001, 'low': price*0.999, 'close': price,
            'volume': 1000.0, 'rsi': 50.0, 'atr': 100.0, 'spread': 10.0,
            'bid': price-5, 'ask': price+5, 'mid': price, 'last': price,
            'ema_fast': price, 'ema_slow': price,
            'pair': 'BTC/EUR'
        })

    forecaster = ChronosForecaster()
    res = forecaster.predict_candles(fake_candles, "5m", 20)

    print(f"Output Structure Keys: {res[0].keys()}")
    for r in res:
        print(f"Pred: {r['timestamp']} | Close: {r['close']:.2f} | EMA_F: {r['ema_fast']:.2f} | Bid: {r['bid']:.2f}")
