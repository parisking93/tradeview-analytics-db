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
        
        # Caricamento del modello 500m
        # Nota: Usiamo from_pretrained generico o la classe specifica se disponibile.
        # Qui usiamo il caricamento standard suggerito dalla libreria per compatibilità.
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
             # Fallback per versioni diverse della lib, prova caricamento diretto torch se la classe wrapper varia
            self.model = timesfm.TimesFM_2p5_500M_torch.from_pretrained(self.repo_id)
            if self.device == "cuda":
                self.model.cuda()

        print("[TimeSfm] Model loaded.")

    def predict_candles(self, candles: List[Dict[str, Any]], timeframe: str, horizon: int) -> List[Dict[str, Any]]:
        """
        Riceve l'output di MarketDataProvider (lista di candele), il timeframe e l'orizzonte.
        Restituisce una lista di candele future (forecast).
        """
        if not candles:
            return []

        # 1. Preparazione Dati
        df = pd.DataFrame(candles)
        # Assicuriamoci che i dati siano ordinati e puliti
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.sort_values('timestamp', inplace=True)
        
        # Estrazione serie Close e Volume
        # Gestiamo il caso in cui i dati siano stringhe o None
        close_series = df['close'].astype(float).fillna(method='ffill').values
        volume_series = df['volume'].astype(float).fillna(0).values

        # Tagliamo al context length massimo per efficienza, se necessario
        if len(close_series) > self.context_len:
            close_series = close_series[-self.context_len:]
            volume_series = volume_series[-self.context_len:]

        last_real_close = close_series[-1]
        last_real_ts = df['timestamp'].iloc[-1]

        # 2. Forecasting
        # Eseguiamo la predizione su Close e Volume insieme
        # Close: ci servono i quantili per High/Low
        # Volume: ci basta la mediana (P50)
        
        # Scaling (Importante per TimeSfm su dati finanziari grezzi)
        scaler_close = StandardScaler()
        close_norm = scaler_close.fit_transform(close_series.reshape(-1, 1)).flatten()
        
        scaler_vol = StandardScaler()
        vol_norm = scaler_vol.fit_transform(volume_series.reshape(-1, 1)).flatten()

        # Input array per il modello: shape (batch_size, time_steps)
        # Batch 0: Close, Batch 1: Volume
        inputs = [close_norm, vol_norm]

        # Esecuzione Inferenza (Low-level method forecast)
        # Richiede compilazione interna o gestione automatica da parte della libreria
        p50_raw, p10_raw, p90_raw = self._run_inference(inputs, horizon)

        # 3. Denormalizzazione
        # Close
        pred_close_p50 = scaler_close.inverse_transform(p50_raw[0].reshape(-1, 1)).flatten()
        pred_close_p10 = scaler_close.inverse_transform(p10_raw[0].reshape(-1, 1)).flatten()
        pred_close_p90 = scaler_close.inverse_transform(p90_raw[0].reshape(-1, 1)).flatten()

        # Volume (Usiamo solo P50) -> Clip a 0 perché il volume negativo non esiste
        pred_vol_p50 = scaler_vol.inverse_transform(p50_raw[1].reshape(-1, 1)).flatten()
        pred_vol_p50 = np.maximum(pred_vol_p50, 0)

        # 4. Costruzione Candele Future
        future_candles = []
        
        # Calcolo delta tempo
        time_delta = self._parse_timeframe_pandas(timeframe)
        current_ts = last_real_ts

        # Per la prima candela futura, l'Open è l'ultimo Close reale.
        # Per le successive, l'Open è il Close della candela precedente predetta.
        prev_close = last_real_close

        for i in range(horizon):
            current_ts += time_delta
            
            # Valori predetti
            c_p50 = float(pred_close_p50[i])
            c_p10 = float(pred_close_p10[i])
            c_p90 = float(pred_close_p90[i])
            v_p50 = float(pred_vol_p50[i])

            # Logica OHLC derivata
            # Open = Close precedente
            open_price = prev_close
            # Close = P50 (Previsione mediana)
            close_price = c_p50
            # Low = Minimo tra (Open, Close, P10) - usiamo P10 come shadow bassa
            low_price = min(open_price, close_price, c_p10)
            # High = Massimo tra (Open, Close, P90) - usiamo P90 come shadow alta
            high_price = max(open_price, close_price, c_p90)

            # Aggiorniamo prev_close per il prossimo giro
            prev_close = close_price

            # Costruzione oggetto
            candle_obj = {
                'timestamp': current_ts.isoformat(),
                'pair': candles[0].get('pair', 'N/A'), # Mantiene il pair originale
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': v_p50,
                'bid': close_price, # Simulato
                'ask': close_price, # Simulato
                'mid': close_price,
                # Info extra debugging
                'p10': c_p10,
                'p90': c_p90,
                'type': 'forecast'
            }
            future_candles.append(candle_obj)

        return future_candles

    def _run_inference(self, inputs: List[np.ndarray], horizon: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Wrapper robusto per il metodo forecast.
        Gestisce i quantile heads per estrarre p10, p50, p90.
        """
        # TimesFM forecast restituisce (point_forecast, quantile_forecast)
        # point_forecast: (batch, horizon)
        # quantile_forecast: (batch, horizon, quantiles) se qhead è attivo
        
        # Assicuriamo che gli input siano float32
        inputs_np = [x.astype(np.float32) for x in inputs]

        # Richiamiamo il metodo forecast (NON predict, NON forecast_on_df)
        # freq=0 indica integer index (non time-aware nel senso di calendario, che gestiamo noi fuori)
        point, quants = self.model.forecast(inputs=inputs_np, horizon=horizon, freq=0)

        # Convertiamo in numpy se sono tensor
        if hasattr(point, 'cpu'): point = point.cpu().numpy()
        if hasattr(quants, 'cpu'): quants = quants.cpu().numpy()

        p50 = point # Mediana

        # Estrazione quantili (assumendo che il modello restituisca 10 quantili standard o 3)
        # Di solito TimesFM restituisce quantili 0.1, 0.2 ... 0.9
        # Controlliamo la shape di quants: [batch, horizon, num_quantiles]
        
        if quants is not None and len(quants.shape) == 3:
            num_q = quants.shape[2]
            # Se ci sono circa 10 quantili, prendiamo il primo (~10%) e l'ultimo (~90%)
            idx_10 = 0 
            idx_90 = num_q - 1
            
            p10 = quants[:, :, idx_10]
            p90 = quants[:, :, idx_90]
        else:
            # Fallback se i quantili non sono disponibili: usiamo p50 +/- un margine euristico (es. 1%)
            # Ma il modello 500m dovrebbe averli.
            p10 = p50 * 0.99
            p90 = p50 * 1.01

        return p50, p10, p90

    def _parse_timeframe_pandas(self, tf: str) -> pd.Timedelta:
        """Converte timeframe stringa (es. 5m, 1h) in Pandas Timedelta."""
        map_tf = {
            'm': 'min',
            'h': 'H',
            'd': 'D',
            'w': 'W',
            'mo': 'M' # Approssimativo
        }
        if not tf: return pd.Timedelta(minutes=5)
        
        unit_char = tf[-1].lower()
        try:
            val = int(tf[:-1])
            pd_unit = map_tf.get(unit_char, 'min')
            return pd.Timedelta(value=val, unit=pd_unit)
        except:
            return pd.Timedelta(minutes=5)

if __name__ == "__main__":
    # Test rapido standalone
    # Simula dei dati di input
    fake_candles = []
    base_price = 50000.0
    ts = pd.Timestamp.now()
    for i in range(100):
        base_price += np.random.normal(0, 100)
        fake_candles.append({
            'timestamp': (ts + pd.Timedelta(minutes=5*i)).isoformat(),
            'open': base_price,
            'high': base_price + 50,
            'low': base_price - 50,
            'close': base_price + 10,
            'volume': np.random.randint(1, 100),
            'pair': 'BTC/EUR'
        })

    forecaster = TimeSfmForecaster() # Carica il 500m
    
    print("Avvio Forecasting...")
    forecasts = forecaster.predict_candles(fake_candles, "5m", horizon=10)
    
    print("\n--- RISULTATO FORECAST ---")
    for f in forecasts:
        print(f"Time: {f['timestamp']} | Open: {f['open']:.2f} | Close: {f['close']:.2f} | High: {f['high']:.2f} | Low: {f['low']:.2f} | Vol: {f['volume']:.2f}")