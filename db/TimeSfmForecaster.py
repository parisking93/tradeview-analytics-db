import os
import torch
import numpy as np
import pandas as pd
import timesfm
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Any

class TimeSfmForecaster:
    def __init__(self, context_len: int = 512):
        """
        Inizializza TimeSfm 2.5 (Versione 200M).
        Versione Legacy con supporto Frequenza temporale.
        """
        self.repo_id = "google/timesfm-2.5-200m-pytorch"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.context_len = context_len

        self.ema_fast_span = 12
        self.ema_slow_span = 26

        print(f"[TimeSfm] Loading {self.repo_id} on {self.device}...")

        try:
            # --- GESTIONE CLASSE LEGACY ---
            if hasattr(timesfm, "TimesFM_2p5_200M_torch"):
                print("[TimeSfm] Found legacy class 'TimesFM_2p5_200M_torch'. Using it.")

                self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.repo_id)
                if hasattr(self.model, "to"):
                    self.model.to(self.device)

                # Configurazione esplicita (richiesta dalla legacy)
                print("[TimeSfm] Creating ForecastConfig...")
                config = timesfm.ForecastConfig(
                    max_context=self.context_len,
                    max_horizon=128,
                    normalize_inputs=True,
                    use_continuous_quantile_head=True,
                    force_flip_invariance=True,
                    infer_is_positive=True,
                    fix_quantile_crossing=True,
                )

                print("[TimeSfm] Compiling model with config...")
                self.model.compile(config)
                self.use_legacy = True

            # --- GESTIONE NUOVA API ---
            else:
                print("[TimeSfm] Legacy class not found. Using new 'TimesFm' class.")
                self.model = timesfm.TimesFm(
                    context_len=self.context_len,
                    backend=self.device,
                    per_core_batch_size=32,
                    horizon_len=128
                )
                self.model.load_from_checkpoint(repo_id=self.repo_id)
                self.use_legacy = False

            print(f"[TimeSfm] Model loaded and COMPILED successfully on {self.device}.")

        except Exception as e:
            print(f"\n[TimeSfm] CRITICAL ERROR during init: {e}")
            raise e

    def predict_candles(self, candles: List[Dict[str, Any]], timeframe: str, horizon: int, pair) -> List[Dict[str, Any]]:
        if not candles or len(candles) < 30:
            return []

        eps = 1e-9

        # 1. Preparazione Dati
        df = pd.DataFrame(candles)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.sort_values('timestamp', inplace=True)

        # Pulizia
        cols = ['close', 'high', 'low', 'volume', 'rsi', 'atr']
        for c in cols:
            if c in df.columns:
                df[c] = df[c].astype(float).ffill().fillna(0)

        def _safe_array(series):
            arr = pd.to_numeric(series, errors='coerce').to_numpy(dtype=float)
            # replace inf/-inf with nan then ffill/bfill and fallback to 0
            arr[~np.isfinite(arr)] = np.nan
            arr = pd.Series(arr).ffill().bfill().fillna(0).to_numpy()
            return arr

        # Ultimo stato
        last_row = df.iloc[-1]
        last_ts = last_row['timestamp']
        last_spread = last_row.get('spread', last_row['close'] * 0.0001)
        if pd.isna(last_spread) or last_spread == 0: last_spread = last_row['close'] * 0.0001

        # --- FEATURE ENGINEERING ---
        def get_log_ret(series):
            vals = _safe_array(series)
            if len(vals) < 2:
                return np.zeros(1, dtype=float)
            # Clamp to avoid divide-by-zero / negatives
            vals = np.clip(vals, eps, None)
            return np.log(vals[1:] / vals[:-1])

        close_log = get_log_ret(df['close'])
        high_log = get_log_ret(df['high'])
        low_log = get_log_ret(df['low'])
        vol_arr = _safe_array(df['volume'])
        vol_arr = np.clip(vol_arr, eps, None)
        vol_log = np.log1p(vol_arr[1:]) if len(vol_arr) > 1 else np.zeros(1, dtype=float)
        rsi_vals = _safe_array(df['rsi'])[1:]
        atr_vals = _safe_array(df['atr'])[1:]

        # --- SCALING ---
        scalers = [StandardScaler() for _ in range(6)]
        inputs_list = [close_log, high_log, low_log, vol_log, rsi_vals, atr_vals]

        prepared_inputs = []
        for i, arr in enumerate(inputs_list):
            arr = np.asarray(arr, dtype=float)
            arr[~np.isfinite(arr)] = 0.0
            # Se tutti zero, StandardScaler darebbe nan; gestiamo il caso
            if np.all(arr == arr[0]):
                scaled = np.zeros_like(arr, dtype=float)
            else:
                scaled = scalers[i].fit_transform(arr.reshape(-1, 1)).flatten()
            if len(scaled) > self.context_len:
                scaled = scaled[-self.context_len:]
            prepared_inputs.append(scaled)

        min_len = min(len(x) for x in prepared_inputs)
        # Shape: (Batch=6, Time)
        inputs_np = np.array([x[-min_len:] for x in prepared_inputs], dtype=np.float32)

        # --- GESTIONE FREQUENZA (TEMPO) ---
        # Mappiamo il timeframe stringa (es "15m") nel codice intero di TimeSfm
        # 0: High freq, 1: Minutes, 2: Hours, 3: Days, 4: Weeks, 5: Months
        freq_map = {
            'm': 1, # 1m, 5m, 15m, 30m
            'h': 2, # 1h, 4h
            'd': 3, # 1d
            'w': 4, # 1wk
            'M': 5  # 1mo
        }
        # Estraiamo l'unità (l'ultimo carattere, es 'm' da '15m')
        unit_char = timeframe[-1] if timeframe and timeframe[-1].isalpha() else 'm'
        freq_val = freq_map.get(unit_char, 1) # Default a minuti (1) se non trovato

        # Creiamo un array di frequenze grande quanto il batch (6)
        # Ogni serie temporale (Close, High, Low...) ha la stessa frequenza
        freq_input = np.full((inputs_np.shape[0],), freq_val, dtype=np.int32)

        # --- INFERENZA ---
        try:
            if self.use_legacy:
                with torch.no_grad():
                    # Passiamo inputs E freq
                    try:
                        p_forecast, _ = self.model.forecast(
                            inputs=inputs_np,
                            freq=freq_input,  # <--- QUI passiamo il tempo
                            horizon=horizon
                        )
                    except TypeError:
                        # Fallback se la versione legacy specifica installata non supporta freq
                        print("[TimeSfm] Warning: This legacy version ignores 'freq'. Running without it.")
                        p_forecast, _ = self.model.forecast(inputs=inputs_np, horizon=horizon)
            else:
                # Nuova API
                p_forecast, _ = self.model.forecast(
                    inputs=inputs_np,
                    freq=freq_input,
                    horizon=horizon
                )

            if hasattr(p_forecast, 'cpu'):
                p_forecast = p_forecast.cpu().numpy()

            if len(p_forecast.shape) == 3:
                p_forecast = p_forecast[0]

        except Exception as e:
            print(f"[TimeSfm] Error during inference: {e}")
            return []

        # --- RICOSTRUZIONE ---
        future_candles = []
        time_delta = self._parse_timeframe_pandas(timeframe)
        current_ts = last_ts

        curr = {
            'close': last_row['close'],
            'high': last_row['high'],
            'low': last_row['low']
        }

        for i in range(horizon):
            current_ts += time_delta

            def denorm(idx, val):
                return scalers[idx].inverse_transform(val.reshape(1, -1))[0,0]

            # Denormalizzazione
            ret_close = denorm(0, p_forecast[0][i])
            ret_high = denorm(1, p_forecast[1][i])
            ret_low = denorm(2, p_forecast[2][i])
            val_vol = denorm(3, p_forecast[3][i])
            val_rsi = denorm(4, p_forecast[4][i])
            val_atr = denorm(5, p_forecast[5][i])

            # Applicazione log-returns
            next_close = curr['close'] * np.exp(ret_close)
            next_high = curr['high'] * np.exp(ret_high)
            next_low = curr['low'] * np.exp(ret_low)
            next_vol = max(0, np.expm1(val_vol))

            # Geometria candela coerente
            final_high = max(next_high, last_row['close'], next_close)
            final_low = min(next_low, last_row['close'], next_close)

            curr['close'] = next_close
            curr['high'] = final_high
            curr['low'] = final_low

            bid = next_close - (last_spread/2)
            ask = next_close + (last_spread/2)

            ts_str = current_ts.strftime('%Y-%m-%d %H:%M:%S') if hasattr(current_ts, 'strftime') else str(current_ts)

            candle = {
                'timestamp': ts_str,
                'pair': pair['pair'],
                'open': last_row['close'],
                'high': final_high,
                'low': final_low,
                'close': next_close,
                'volume': next_vol,
                'spread': last_spread,
                'bid': bid,
                'ask': ask,
                'mid': next_close,
                'last': next_close,
                'rsi': val_rsi,
                'atr': val_atr,
                'ema_fast': 0, 'ema_slow': 0,
                'type': 'forecast',
                'timeframe': timeframe + '+' + str(i+1)  # manteniamo stesso formato di data_1d
            }
            future_candles.append(candle)
            last_row = pd.Series(candle)

        # Ricalcolo EMA
        df_fut = pd.DataFrame(future_candles)
        df_fut['timestamp'] = pd.to_datetime(df_fut['timestamp'])
        full = pd.concat([df[['timestamp','close']], df_fut[['timestamp','close']]], ignore_index=True)
        full['f'] = full['close'].ewm(span=self.ema_fast_span).mean()
        full['s'] = full['close'].ewm(span=self.ema_slow_span).mean()

        res_emas = full.iloc[-horizon:].reset_index(drop=True)
        for i, c in enumerate(future_candles):
            c['ema_fast'] = float(res_emas.iloc[i]['f'])
            c['ema_slow'] = float(res_emas.iloc[i]['s'])

        # Normalizza tipi per evitare np.float* nei salvataggi/DB
        def _to_float(val):
            try:
                return float(val)
            except Exception:
                return None

        for c in future_candles:
            c['timestamp'] = str(c.get('timestamp'))
            for key in ['open', 'high', 'low', 'close', 'volume', 'bid', 'ask', 'mid', 'spread',
                        'rsi', 'atr', 'ema_fast', 'ema_slow', 'last']:
                if key in c:
                    c[key] = _to_float(c.get(key))

        return future_candles

    def _parse_timeframe_pandas(self, tf: str) -> pd.Timedelta:
        try:
            unit = tf[-1]
            val = int(tf[:-1])
            map_tf = {'m':'min', 'h':'H', 'd':'D', 'w':'W'}
            return pd.Timedelta(value=val, unit=map_tf.get(unit, 'min'))
        except:
            return pd.Timedelta(minutes=5)

if __name__ == "__main__":
    print("--- TEST TIMESFM WITH FREQ ---")
    fake = []
    p = 50000.0
    ts = pd.Timestamp.now()
    for i in range(100):
        p *= (1 + np.random.normal(0,0.001))
        fake.append({
            'timestamp': (ts + pd.Timedelta(minutes=5*i)).isoformat(),
            'close': p, 'high': p*1.001, 'low': p*0.999, 'volume': 100,
            'rsi': 50, 'atr': 50, 'pair': 'BTC/EUR'
        })

    forecaster = TimeSfmForecaster()
    # Testiamo con 15 minuti (freq=1)
    res = forecaster.predict_candles(fake, "5m", 20)
    print("\n--- PREVISIONI ---")
    for f in res:
        print(f"TS: {f['timestamp']} | O: {f['open']:.2f} C: {f['close']:.2f} H: {f['high']:.2f} L: {f['low']:.2f}| RSI: {f['rsi']:.1f}")
