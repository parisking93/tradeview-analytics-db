import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union

class RunningNorm(nn.Module):
    """Mantenuta dal vecchio file per normalizzazione on-the-fly"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.register_buffer("count", torch.tensor(eps))
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("M2", torch.zeros(dim))
        self.eps = eps

    def update(self, x: torch.Tensor) -> None:
        if self.training: # Aggiorna solo in training o se forzato
            if x.ndim == 1: x = x.unsqueeze(0)
            batch_n = torch.tensor(x.shape[0], device=x.device, dtype=self.count.dtype)
            batch_mean = x.mean(0)
            delta = batch_mean - self.mean
            total = self.count + batch_n
            # Algoritmo di Welford semplificato per batch
            self.mean = self.mean + delta * (batch_n / total)
            self.count = total

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (torch.sqrt(self.M2 / self.count + self.eps) + 1.0) # Semplificato dev std a 1 se non calcoliamo var completa

class NewTinyRecursiveModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 384):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Normalizzatore
        self.norm = RunningNorm(input_dim)

        # Encoder: GRU layer. Nota: Input è (Batch, FeatureDim).
        # Lo trattiamo come una sequenza di lunghezza 1 o lo proiettiamo.
        # Per mantenere la struttura "Recursive", usiamo una proiezione + GRU Cell o GRU su seq=1
        self.encoder_proj = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        # Residual Block (il "pensiero" ricorrente)
        self.residual = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2), # Concatena input corrente + stato memoria
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # --- HEADS (Identiche al vecchio file) ---
        self.y_dim = hidden_dim
        self.head_side = nn.Linear(self.y_dim, 3)   # buy/sell/hold logits
        self.head_qty  = nn.Linear(self.y_dim, 1)   # 0..1 fraction
        self.head_px   = nn.Linear(self.y_dim, 1)   # price offset tanh -> ±5%
        self.head_tp   = nn.Linear(self.y_dim, 1)   # softplus -> tp mult (0..)
        self.head_sl   = nn.Linear(self.y_dim, 1)   # softplus -> sl mult (0..)
        self.head_tif  = nn.Linear(self.y_dim, 3)   # IOC/FOK/GTC logits
        self.head_lev  = nn.Linear(self.y_dim, 1)   # softplus + 1
        self.head_score= nn.Linear(self.y_dim, 1)   # ranking
        self.head_ordertype = nn.Linear(self.y_dim, 2)  # logits: 0=limit, 1=market
        self.head_reduce    = nn.Linear(self.y_dim, 1)  # sigmoid -> prob reduce_only

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None):
        """
        x: (Batch, InputDim) - Vettore creato dal Vectorizer
        h: (Batch, HiddenDim) - Stato nascosto precedente (memoria)
        """
        x_norm = self.norm(x)

        # Encoding
        enc = F.relu(self.encoder_proj(x_norm))

        # Inizializza h se None
        if h is None:
            h = torch.zeros_like(enc)

        # GRU step: aggiorna la memoria basandosi sul nuovo input
        h_new = self.gru(enc, h)

        # Residual thinking step (Refinement)
        # Concateniamo l'input encodato con la memoria aggiornata
        combined = torch.cat([enc, h_new], dim=-1)
        delta = self.residual(combined)

        # Output feature vector per le heads
        y = h_new + delta

        return y, h_new

    def get_heads(self, y: torch.Tensor):
        return {
            "side": self.head_side(y),
            "qty": torch.sigmoid(self.head_qty(y)),
            "px": torch.tanh(self.head_px(y)),
            "tp": F.softplus(self.head_tp(y)),
            "sl": F.softplus(self.head_sl(y)),
            "tif": self.head_tif(y),
            "lev": F.softplus(self.head_lev(y)) + 1.0,
            "score": torch.tanh(self.head_score(y)),
            "ordertype": self.head_ordertype(y),
            "reduce": torch.sigmoid(self.head_reduce(y))
        }


class MultiTimeframeTRM(nn.Module):
    def __init__(self,
                 tf_configs: Dict[str, int],
                 input_size_per_candle: int,
                 static_size: int,
                 hidden_dim: int = 512):
        super().__init__()

        self.hidden_dim = hidden_dim

        # --- CONFIGURAZIONE ENCODER (GLI OCCHI) ---
        # Aumentiamo a 128 per catturare più dettagli
        self.encoder_hidden = 128

        # Definiamo quante features estraiamo per ogni sequenza:
        # 1. Last State (Come finisce)
        # 2. Max Pool (Picchi massimi)
        # 3. Mean Pool (Trend medio)
        # Totale: 128 * 3 = 384 features per ogni timeframe
        self.per_seq_dim = self.encoder_hidden * 3

        self.timeframe_encoders = nn.ModuleDict()
        self.tf_names = list(tf_configs.keys())

        # 1. Encoders per lo Storico
        for tf in self.tf_names:
            self.timeframe_encoders[tf] = nn.GRU(
                input_size=input_size_per_candle,
                hidden_size=self.encoder_hidden,
                num_layers=2,        # Un po' di profondità extra
                dropout=0.1,         # Evita overfitting sui layer interni
                batch_first=True
            )

        # 2. Encoder per il Forecast
        self.forecast_encoder = nn.GRU(
            input_size=input_size_per_candle,
            hidden_size=self.encoder_hidden,
            num_layers=2,
            dropout=0.1,
            batch_first=True
        )

        # --- CALCOLO DIMENSIONE CONTESTO ---
        # (Dimensione seq * (numero TF storici + 1 forecast)) + dati statici
        total_context_size = (self.per_seq_dim * (len(self.tf_names) + 1)) + static_size

        # Proiezione verso il cervello centrale
        self.context_proj = nn.Linear(total_context_size, hidden_dim)

        # --- IL CERVELLO CENTRALE (RIMANE UGUALE) ---
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)

        self.residual = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # --- HEADS (OUTPUT) ---
        self.head_side = nn.Linear(hidden_dim, 3)
        self.head_qty = nn.Linear(hidden_dim, 1)
        self.head_px = nn.Linear(hidden_dim, 1)
        self.head_tp = nn.Linear(hidden_dim, 1)
        self.head_sl = nn.Linear(hidden_dim, 1)
        self.head_lev  = nn.Linear(hidden_dim, 1)
        self.head_ordertype = nn.Linear(hidden_dim, 2)
        self.head_halt      = nn.Linear(hidden_dim, 1)

    def _summarize_seq(self, out: torch.Tensor) -> torch.Tensor:
        """
        Helper function per estrarre Last + Max + Mean da un output GRU.
        out shape: (Batch, SeqLen, Hidden)
        """
        # 1. Last step (l'ultimo stato temporale)
        last_step = out[:, -1, :]

        # 2. Max Pooling (il valore massimo raggiunto su ogni feature nella sequenza)
        max_pool = torch.max(out, dim=1)[0]

        # 3. Mean Pooling (la media dei valori nella sequenza)
        mean_pool = torch.mean(out, dim=1)

        # Concatena tutto: (Batch, Hidden*3)
        return torch.cat([last_step, max_pool, mean_pool], dim=-1)

    def forward(self, inputs: Dict[str, torch.Tensor], h: Optional[torch.Tensor] = None):
        tf_summaries = []

        # 1. Processa Storico
        for tf in self.tf_names:
            seq = inputs[f"seq_{tf}"]
            if seq.ndim == 2: seq = seq.unsqueeze(0)

            # GRU ritorna: output, h_n. A noi serve output per fare pooling.
            gru_out, _ = self.timeframe_encoders[tf](seq)

            # Usiamo l'helper per estrarre il riassunto ricco
            tf_summaries.append(self._summarize_seq(gru_out))

        # 2. Processa Forecast
        seq_fc = inputs["seq_forecast"]
        if seq_fc.ndim == 2: seq_fc = seq_fc.unsqueeze(0)

        fc_out, _ = self.forecast_encoder(seq_fc)
        tf_summaries.append(self._summarize_seq(fc_out))

        # 3. Statico
        static_data = inputs["static"]
        if static_data.ndim == 1: static_data = static_data.unsqueeze(0)

        # 4. Fusione e Ragionamento
        full_context = torch.cat(tf_summaries + [static_data], dim=-1)

        brain_input = F.relu(self.context_proj(full_context))

        if h is None: h = torch.zeros_like(brain_input)

        h_new = self.gru_cell(brain_input, h)
        combined = torch.cat([brain_input, h_new], dim=-1)
        y = h_new + self.residual(combined)

        return y, h_new

    def get_heads_dict(self, y):
        # Mantenuto uguale all'originale per compatibilità con il Trainer
        return {
            "side": self.head_side(y),
            "qty": torch.sigmoid(self.head_qty(y)),
            "price_offset": torch.tanh(self.head_px(y)),
            "tp_mult": F.softplus(self.head_tp(y)),
            "sl_mult": F.softplus(self.head_sl(y)),
            "halt_prob": torch.sigmoid(self.head_halt(y)),
            "ordertype": self.head_ordertype(y),
            "leverage": F.softplus(self.head_lev(y)) + 1.0
        }
