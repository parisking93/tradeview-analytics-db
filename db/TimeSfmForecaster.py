import os
import time
import torch
import numpy as np
import pandas as pd
import timesfm
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Any, Tuple, Optional

class TimeSfmForecaster:
    def __init__(self, context_len: int = 512):
        """
        Inizializza il modello TimeSfm 2.5 (500M Parameters).
        """
        self.repo_id = "google/timesfm-2.5-500m-pytorch"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.context_len = context_len
        
        print(f"[TimeSfm] Loading model {self.repo_id} on {self.device}...")
        
        try:
            self.model = timesfm.TimesFm(
                hparams=timesfm.TimesFmHparams(
                    backend=self.device,
                    per_core_batch_size=32,
                    context_len=self.context_len,
                ),
                checkpoint=timesfm.TimesFmCheckpoint(
                    huggingface_repo_id=self.repo_id
                )
            )
        except AttributeError:
            self.model = timesfm.TimesFM_2p5_500M_torch.from_pretrained(self.repo_id)
            if self.device == "cuda":
                self.model.cuda()

        print("[TimeSfm] Model loaded.")

    def predict_candles(self, candles: List[Dict[str, Any]], timeframe: str, horizon: int) -> List[Dict[str, Any]]:
        """
        Advanced Forecasting con Log-Returns e Covariates (RSI, ATR).
        """
        if not candles or len(candles) < 20:
            print("[TimeSfm] Error: Not enough candles for context.")
            return []

        # 1. Preparazione DataFrame
        df = pd.DataFrame(candles)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.sort_values('timestamp', inplace=True)
        
        # Riempimento valori nulli (FFill per gestire i 'None' del realtime)
        for col in ['close', 'high', 'low', 'volume', 'rsi', 'atr']:
            if col in df.columns:
                df[col] = df[col].astype(float).fillna(method='ffill').fillna(0)

        # Dati Reali Finali (per ricostruzione)
        last_real_close = df['close'].iloc[-1]
        last_real_high = df['high'].iloc[-1]
        last_real_low = df['low'].iloc[-1]
        last_real_ts = df['timestamp'].iloc[-1]

        # --- FEATURE ENGINEERING (Log Returns & Scaling) ---
        # Usiamo Log Returns per stazionarietà: ln(P_t / P_{t-1})
        
        close_series = df['close'].values
        close_log_ret = np.log(close_series[1:] / close_series[:-1])
        
        high_series = df['high'].values
        high_log_ret = np.log(high_series[1:] / high_series[:-1])

        low_series = df['low'].values
        low_log_ret = np.log(low_series[1:] / low_series[:-1])

        vol_series = df['volume'].values[1:] 
        vol_log = np.log1p(vol_series)

        rsi_series = df['rsi'].values[1:]
        atr_series = df['atr'].values[1:]

        # Taglio al context length
        def tail(arr): return arr[-self.context_len:] if len(arr) > self.context_len else arr
        
        input_close = tail(close_log_ret)
        input_high = tail(high_log_ret)
        input_low = tail(low_log_ret)
        input_vol = tail(vol_log)
        input_rsi = tail(rsi_series)
        input_atr = tail(atr_series)

        # Scaling
        scalers = [StandardScaler() for _ in range(6)]
        
        norm_close = scalers[0].fit_transform(input_close.reshape(-1, 1)).flatten()
        norm_high  = scalers[1].fit_transform(input_high.reshape(-1, 1)).flatten()
        norm_low   = scalers[2].fit_transform(input_low.reshape(-1, 1)).flatten()
        norm_vol   = scalers[3].fit_transform(input_vol.reshape(-1, 1)).flatten()
        norm_rsi   = scalers[4].fit_transform(input_rsi.reshape(-1, 1)).flatten()
        norm_atr   = scalers[5].fit_transform(input_atr.reshape(-1, 1)).flatten()

        # Batch Input: [Close, High, Low, Vol, RSI, ATR]
        inputs = [norm_close, norm_high, norm_low, norm_vol, norm_rsi, norm_atr]

        # 2. INFERENCE
        p50_raw, _, _ = self._run_inference(inputs, horizon)

        # 3. DENORMALIZZAZIONE & RICOSTRUZIONE
        future_candles = []
        time_delta = self._parse_timeframe_pandas(timeframe)
        current_ts = last_real_ts
        
        curr_close = last_real_close
        curr_high = last_real_high
        curr_low = last_real_low

        for i in range(horizon):
            current_ts += time_delta

            # Denormalizza
            pred_close_ret = scalers[0].inverse_transform(p50_raw[0][i].reshape(1, -1))[0,0]
            pred_high_ret  = scalers[1].inverse_transform(p50_raw[1][i].reshape(1, -1))[0,0]
            pred_low_ret   = scalers[2].inverse_transform(p50_raw[2][i].reshape(1, -1))[0,0]
            pred_vol_log   = scalers[3].inverse_transform(p50_raw[3][i].reshape(1, -1))[0,0]
            pred_rsi       = scalers[4].inverse_transform(p50_raw[4][i].reshape(1, -1))[0,0]
            pred_atr       = scalers[5].inverse_transform(p50_raw[5][i].reshape(1, -1))[0,0]

            # Ricostruzione
            next_open = curr_close
            next_close = curr_close * np.exp(pred_close_ret)
            
            # High & Low indipendenti (trend follower)
            next_high = curr_high * np.exp(pred_high_ret)
            next_low = curr_low * np.exp(pred_low_ret)

            next_vol = max(0, np.expm1(pred_vol_log))

            # Geometria Candela
            final_high = max(next_high, next_open, next_close)
            final_low = min(next_low, next_open, next_close)

            curr_close = next_close
            curr_high = final_high
            curr_low = final_low

            future_candles.append({
                'timestamp': current_ts.isoformat(),
                'pair': candles[0].get('pair', 'N/A'),
                'open': next_open,
                'high': final_high,
                'low': final_low,
                'close': next_close,
                'volume': next_vol,
                'rsi': pred_rsi,
                'atr': pred_atr,
                'type': 'forecast'
            })

        return future_candles

    def _run_inference(self, inputs: List[np.ndarray], horizon: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Conversione sicura in Batch Numpy Array
        # Shape diventa: (6, context_len) -> 6 canali trattati come batch paralleli
        inputs_np = np.stack([x.astype(np.float32) for x in inputs])
        
        # freq=0 -> integer index
        point, quants = self.model.forecast(inputs=inputs_np, horizon=horizon, freq=0)

        if hasattr(point, 'cpu'): point = point.cpu().numpy()
        if hasattr(quants, 'cpu'): quants = quants.cpu().numpy()

        p50 = point
        # Placeholder quantili
        p10 = p50 * 0.99
        p90 = p50 * 1.01
        
        return p50, p10, p90

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
    # Test Standalone
    print("--- TEST STANDALONE ---")
    fake_candles = []
    price = 50000.0
    vol = 1000.0
    ts = pd.Timestamp.now()
    
    for i in range(100):
        price = price * (1 + np.random.normal(0, 0.001)) 
        high = price * 1.002
        low = price * 0.998
        vol = abs(vol + np.random.normal(0, 100))
        
        fake_candles.append({
            'timestamp': (ts + pd.Timedelta(minutes=15*i)).isoformat(),
            'open': price, 'high': high, 'low': low, 'close': price,
            'volume': vol, 'rsi': 50 + np.random.normal(0, 5), 'atr': 50.0,
            'pair': 'BTC/EUR'
        })

    forecaster = TimeSfmForecaster()
    forecasts = forecaster.predict_candles(fake_candles, "15m", horizon=5)

    print("\n--- PREVISIONI ---")
    for f in forecasts:
        print(f"TS: {f['timestamp']} | O: {f['open']:.2f} C: {f['close']:.2f} H: {f['high']:.2f} L: {f['low']:.2f}| RSI: {f['rsi']:.1f}")