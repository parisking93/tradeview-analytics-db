import torch
import math
import zlib
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

@dataclass
class VectorizerConfig:
    # Definizione dinamica: { "1d": 10, "4h": 15, "1h": 20 }
    candle_history_config: Dict[str, int]

    # --- CONFIGURAZIONE COLONNE ---

    # MODIFICA 1: Rimesso 'id' da questa lista
    numeric_columns_candle: List[str] = field(default_factory=lambda: [
        'open', 'high', 'low', 'close', 'volume',
        'bid', 'ask', 'mid', 'spread',
        'ema_fast', 'ema_slow', 'rsi', 'atr'
    ])

    string_columns_candle: List[str] = field(default_factory=lambda: [
        'pair', 'kr_pair', 'base', 'quote', 'timeframe'
    ])

    # time_columns_candle resta solo timestamp o created_at
    time_columns_candle: List[str] = field(default_factory=lambda: [
        'timestamp', 'created_at'
    ])

    # MODIFICA 2: Rimosso 'id' anche dagli ordini (non serve alla rete)
    numeric_columns_order: List[str] = field(default_factory=lambda: [
        'qty', 'price_entry', 'price_avg',
        'take_profit', 'stop_loss', 'price', 'value_eur', 'pnl'
    ])

    string_columns_order: List[str] = field(default_factory=lambda: [
        'pair', 'kr_pair', 'base', 'quote', 'type', 'subtype', 'status'
    ])

    # MODIFICA 3: Rimosso 'record_date'
    time_columns_order: List[str] = field(default_factory=lambda: [
        'created_at'
    ])

class DataVectorizer:
    def __init__(self, config: VectorizerConfig):
        self.cfg = config

        # Calcolo dimensioni vettori
        self.candle_dim = (
            len(self.cfg.numeric_columns_candle) +
            len(self.cfg.string_columns_candle) +
            (len(self.cfg.time_columns_candle) * 4)
        )

        self.order_dim = (
            len(self.cfg.numeric_columns_order) +
            len(self.cfg.string_columns_order) +
            (len(self.cfg.time_columns_order) * 4)
        )

        # Order Dim + 1 (Flag Order Exists) + 3 (Leverage Features) +1 per bilancio wallet
        self.static_total_dim = self.order_dim + 1 + 3 + 1
    # --- Utilities di Conversione ---

    def _safe_float(self, val: Any, default: float = 0.0) -> float:
        if val is None: return default
        try: return float(val)
        except: return default

    def _hash_string(self, text: Any) -> float:
        if text is None: return 0.0
        s = str(text)
        hash_val = zlib.adler32(s.encode('utf-8'))
        return (hash_val & 0xffffffff) / 0xffffffff * 2.0 - 1.0

    def _encode_time(self, val: Any) -> List[float]:
        try:
            if isinstance(val, str):
                try: dt = datetime.strptime(val, "%Y-%m-%d %H:%M:%S")
                except: dt = datetime.strptime(val, "%Y-%m-%d")
            elif isinstance(val, datetime):
                dt = val
            else:
                return [0.0, 0.0, 0.0, 0.0]

            ts = (dt.timestamp() - 1704067200) / 31536000.0
            hour_angle = (dt.hour + dt.minute/60.0) * 2 * math.pi / 24.0
            sin_h = math.sin(hour_angle)
            cos_h = math.cos(hour_angle)
            weekday = dt.weekday() / 6.0

            return [ts, sin_h, cos_h, weekday]
        except Exception:
            return [0.0, 0.0, 0.0, 0.0]

    def _vectorize_row(self, row: Dict[str, Any], col_numeric: List[str], col_string: List[str], col_time: List[str], ref_price: float) -> List[float]:
        feats = []

        # 1. Numerici
        for col in col_numeric:
            raw_val = self._safe_float(row.get(col))

            # MODIFICA 4: Rimosso check per 'id', lasciato solo volume
            if col == 'volume':
                feats.append(math.log1p(max(0.0, raw_val)))
            elif col == 'rsi':
                feats.append(raw_val / 100.0)
            elif col in ['spread', 'atr']:
                feats.append(raw_val / (ref_price + 1e-9))
            elif col in ['pnl', 'qty']:
                feats.append(math.tanh(raw_val))
            elif col in ['open', 'high', 'low', 'close', 'bid', 'ask', 'mid', 'ema_fast', 'ema_slow', 'price', 'price_entry', 'price_avg', 'take_profit', 'stop_loss']:
                if raw_val == 0: feats.append(0.0)
                else: feats.append((raw_val - ref_price) / (ref_price + 1e-9))
            else:
                feats.append(raw_val)

        # 2. Stringhe
        for col in col_string:
            feats.append(self._hash_string(row.get(col)))

        # 3. Tempo
        for col in col_time:
            feats.extend(self._encode_time(row.get(col)))

        return feats

    def vectorize(self,
                  candles_db_data: Dict[str, List[Dict[str, Any]]],
                  open_order: Optional[Dict[str, Any]],
                  forecast_db_data: List[Dict[str, Any]],
                  pair_limits: Optional[Dict[str, Any]] = None,
                  wallet_balance=0.0) -> Dict[str, torch.Tensor]: # <--- NUOVO PARAMETRO

        output = {}

        # --- Prezzo di Riferimento ---
        ref_price = 1.0
        shortest_tf = list(self.cfg.candle_history_config.keys())[-1]
        if candles_db_data.get(shortest_tf):
            ref_price = self._safe_float(candles_db_data[shortest_tf][0].get('close'))
            if ref_price <= 0: ref_price = 1.0

        # --- 1. Candele Storiche (Sequence) ---
        for tf, count in self.cfg.candle_history_config.items():
            raw_rows = candles_db_data.get(tf, [])
            needed_rows = raw_rows[:count]
            ordered_rows = needed_rows[::-1]

            seq_data = []
            padding_needed = count - len(ordered_rows)
            zero_vec = [0.0] * self.candle_dim
            if padding_needed > 0:
                seq_data.extend([zero_vec] * padding_needed)

            for row in ordered_rows:
                vec = self._vectorize_row(
                    row,
                    self.cfg.numeric_columns_candle,
                    self.cfg.string_columns_candle,
                    self.cfg.time_columns_candle,
                    ref_price
                )
                seq_data.append(vec)

            output[f"seq_{tf}"] = torch.tensor(seq_data, dtype=torch.float32)

        # --- 2. Forecast (Sequence) ---
        fc_data = []
        if forecast_db_data:
            sorted_fc = sorted(forecast_db_data, key=lambda x: str(x.get('timestamp')))
            for row in sorted_fc:
                vec = self._vectorize_row(
                    row,
                    self.cfg.numeric_columns_candle,
                    self.cfg.string_columns_candle,
                    self.cfg.time_columns_candle,
                    ref_price
                )
                fc_data.append(vec)

        if len(fc_data) == 0:
            fc_data.append([0.0] * self.candle_dim)

        output["seq_forecast"] = torch.tensor(fc_data, dtype=torch.float32)

        # --- 3. Ordine Aperto (Static) ---
        if open_order:
            order_vec = self._vectorize_row(
                open_order,
                self.cfg.numeric_columns_order,
                self.cfg.string_columns_order,
                self.cfg.time_columns_order,
                ref_price
            )
            order_vec.append(1.0) # Flag: Order Exists
        else:
            order_vec = [0.0] * self.order_dim
            order_vec.append(0.0)


        # [MaxLevBuy, MaxLevSell, CanLev(1/0)]
        lev_feats = [1.0, 1.0, 0.0]

        if pair_limits:
            lev_buy_max = self._safe_float(pair_limits.get('leverage_buy_max'), 1.0)
            lev_sell_max = self._safe_float(pair_limits.get('leverage_sell_max'), 1.0)
            # Normalizziamo dividendo per un fattore (es. 50) o log, o grezzo se piccolo
            # Qui usiamo raw / 10.0 per scalarlo
            can_lev = 1.0 if (pair_limits.get('can_leverage_buy') and pair_limits.get('can_leverage_sell')) else 0.0

            lev_feats = [lev_buy_max / 10.0, lev_sell_max / 10.0, can_lev]

        # Normalizziamo il bilancio del wallet (log10)
        balance_norm = math.log10(wallet_balance + 1.0)

        final_static = order_vec + lev_feats + [balance_norm]
        output["static"] = torch.tensor(final_static, dtype=torch.float32)


        return output, ref_price
