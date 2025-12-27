import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, Optional


class RunningNorm(nn.Module):
    """Normalizzazione on-the-fly"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.register_buffer("count", torch.tensor(eps))
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("M2", torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (torch.sqrt(self.M2 / self.count + self.eps) + 1.0)


class S4DLayer(nn.Module):
    """
    Simplified S4D (Structured State Space for Sequences) Layer.

    Questo è il precursore di Mamba, ottimizzato per time series.
    - Complessità O(n log n) grazie a FFT
    - Gestisce dipendenze lunghe naturalmente
    - Più veloce di Transformer su sequenze lunghe

    Paper: "On the Parameterization and Initialization of Diagonal State Space Models"
    """
    def __init__(self, d_model: int, d_state: int = 64, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Parametri dello State Space Model
        # A: matrice di stato (diagonale per efficienza)
        # B, C: matrici di input e output
        # D: skip connection

        # Inizializzazione HiPPO-inspired per catturare dipendenze lunghe
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        A = -0.5 * A  # Eigenvalues negativi per stabilità
        self.register_buffer("A", A)

        # Parametri learnable
        self.B = nn.Parameter(torch.randn(d_model, d_state) * 0.02)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.02)
        self.D = nn.Parameter(torch.ones(d_model))

        # Delta (discretization step) - learnable per timestep adaptation
        self.log_delta = nn.Parameter(torch.zeros(d_model))

        # Proiezioni
        self.in_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (Batch, SeqLen, d_model)
        Returns: (Batch, SeqLen, d_model)
        """
        batch, seq_len, d_model = x.shape
        residual = x

        x = self.norm(x)
        x = self.in_proj(x)

        # Discretize: A_bar, B_bar usando Zero-Order Hold
        delta = F.softplus(self.log_delta)  # (d_model,)

        # Per efficienza, usiamo convoluzione nel dominio della frequenza
        # Costruiamo il kernel di convoluzione
        kernel = self._compute_kernel(seq_len, delta)  # (d_model, seq_len)

        # Convoluzione efficiente via FFT
        # x: (batch, seq_len, d_model) -> (batch, d_model, seq_len)
        x = x.transpose(1, 2)

        # FFT convolution
        k_f = torch.fft.rfft(kernel, n=2*seq_len)  # (d_model, seq_len+1)
        x_f = torch.fft.rfft(x, n=2*seq_len)       # (batch, d_model, seq_len+1)
        y_f = x_f * k_f.unsqueeze(0)
        y = torch.fft.irfft(y_f, n=2*seq_len)[..., :seq_len]  # (batch, d_model, seq_len)

        # Skip connection
        y = y + x[..., :seq_len] * self.D.unsqueeze(0).unsqueeze(-1)

        # Back to (batch, seq_len, d_model)
        y = y.transpose(1, 2)
        y = self.out_proj(y)
        y = self.dropout(y)

        return residual + y

    def _compute_kernel(self, seq_len: int, delta: torch.Tensor) -> torch.Tensor:
        """
        Computa il kernel di convoluzione per SSM.
        kernel[k] = C * A^k * B

        Per A diagonale, questo è efficiente.
        """
        # delta: (d_model,)
        # A: (d_state,)
        # B, C: (d_model, d_state)

        # Discretize A
        A_bar = torch.exp(delta.unsqueeze(-1) * self.A.unsqueeze(0))  # (d_model, d_state)
        B_bar = self.B  # (d_model, d_state)

        # Powers of A_bar: A^0, A^1, ..., A^(seq_len-1)
        # (seq_len, d_model, d_state)
        powers = torch.arange(seq_len, device=delta.device, dtype=delta.dtype)
        A_powers = A_bar.unsqueeze(0) ** powers.view(-1, 1, 1)  # (seq_len, d_model, d_state)

        # kernel = C * A^k * B summed over state dimension
        # (d_model, seq_len)
        kernel = torch.einsum('lds,ds,ds->dl', A_powers, B_bar, self.C)

        return kernel


class S4Block(nn.Module):
    """S4D + FFN block (simile a Transformer block)"""
    def __init__(self, d_model: int, d_state: int = 64, dropout: float = 0.1):
        super().__init__()
        self.s4 = S4DLayer(d_model, d_state, dropout)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.s4(x)
        x = x + self.ffn(x)
        return x


class MultiTimeframeTRM(nn.Module):
    """
    Architettura S4D-based (State Space Model) per trading multi-timeframe.

    S4D è il precursore di Mamba:
    - Complessità O(n log n) vs O(n²) del Transformer
    - Cattura dipendenze lunghe naturalmente (HiPPO framework)
    - Più veloce su GPU per sequenze lunghe
    - Non richiede librerie CUDA custom
    """
    def __init__(self,
                 tf_configs: Dict[str, int],
                 input_size_per_candle: int,
                 static_size: int,
                 hidden_dim: int = 512,
                 use_forecast: bool = True): # Flag per ignorare i forecast se rumorosi
        super().__init__()
        self.use_forecast = use_forecast

        self.hidden_dim = hidden_dim
        self.encoder_hidden = 256
        self.d_state = 64  # Dimensione dello stato interno S4

        # Output: ultimo timestep
        self.per_seq_dim = self.encoder_hidden

        self.tf_names = list(tf_configs.keys())

        # --- PROIEZIONI INPUT ---
        self.timeframe_projections = nn.ModuleDict()
        self.timeframe_norms = nn.ModuleDict()

        # --- S4 ENCODERS (sostituiscono Transformer/GRU) ---
        self.timeframe_s4 = nn.ModuleDict()

        for tf in self.tf_names:
            self.timeframe_projections[tf] = nn.Linear(input_size_per_candle, self.encoder_hidden)
            self.timeframe_norms[tf] = nn.LayerNorm(self.encoder_hidden)

            # Stack di 2 S4 blocks
            self.timeframe_s4[tf] = nn.Sequential(
                S4Block(self.encoder_hidden, self.d_state, dropout=0.1),
                S4Block(self.encoder_hidden, self.d_state, dropout=0.1)
            )

        # --- FORECAST ENCODER ---
        self.forecast_proj = nn.Linear(input_size_per_candle, self.encoder_hidden)
        self.forecast_norm = nn.LayerNorm(self.encoder_hidden)
        self.forecast_s4 = nn.Sequential(
            S4Block(self.encoder_hidden, self.d_state, dropout=0.1),
            S4Block(self.encoder_hidden, self.d_state, dropout=0.1)
        )

        # --- STATIC NORM ---
        self.static_norm = nn.LayerNorm(static_size)

        # --- CALCOLO DIMENSIONE CONTESTO ---
        total_context_size = (self.per_seq_dim * (len(self.tf_names) + 1)) + static_size

        # Proiezione verso il cervello centrale
        self.context_proj = nn.Linear(total_context_size, hidden_dim)

        # --- IL CERVELLO CENTRALE (GRU Cell per memoria tra thinking steps) ---
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)

        self.residual = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # Learned initial hidden state
        self.h_init = nn.Parameter(torch.zeros(1, hidden_dim))

        # Dropout nel thinking loop
        self.think_dropout = nn.Dropout(0.1)

        # --- HEADS (OUTPUT) ---
        self.head_side = nn.Linear(hidden_dim, 3)
        self.head_qty = nn.Linear(hidden_dim, 1)
        self.head_px = nn.Linear(hidden_dim, 1)
        self.head_tp = nn.Linear(hidden_dim, 1)
        self.head_sl = nn.Linear(hidden_dim, 1)
        self.head_lev = nn.Linear(hidden_dim, 1)
        self.head_ordertype = nn.Linear(hidden_dim, 2)
        self.head_halt = nn.Linear(hidden_dim, 1)

    def extract_features(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Estrae i context tokens usando S4 (run una sola volta per step)"""
        tf_summaries = []

        # 1. Processa ogni Timeframe con S4
        for tf in self.tf_names:
            seq = inputs[f"seq_{tf}"]
            if seq.ndim == 2:
                seq = seq.unsqueeze(0)

            # Proiezione + Normalizzazione
            x = self.timeframe_projections[tf](seq)
            x = self.timeframe_norms[tf](x)

            # S4 forward
            x = self.timeframe_s4[tf](x)

            # Prendi l'ultimo timestep come summary
            summary = x[:, -1, :]  # (batch, encoder_hidden)
            tf_summaries.append(summary)

        # 2. Processa Forecast con S4
        if self.use_forecast:
            seq_fc = inputs["seq_forecast"]
            if seq_fc.ndim == 2:
                seq_fc = seq_fc.unsqueeze(0)

            x_fc = self.forecast_proj(seq_fc)
            x_fc = self.forecast_norm(x_fc)
            x_fc = self.forecast_s4(x_fc)
            summary_fc = x_fc[:, -1, :]
            tf_summaries.append(summary_fc)
        else:
            # Token zero se forecast disabilitato
            zero_fc = torch.zeros(1, self.encoder_hidden, device=inputs["static"].device)
            tf_summaries.append(zero_fc)

        # 3. Statico
        static_data = inputs["static"]
        if static_data.ndim == 1:
            static_data = static_data.unsqueeze(0)
        static_data = self.static_norm(static_data)

        # 4. Fusione
        full_context = torch.cat(tf_summaries + [static_data], dim=-1)
        brain_input = F.gelu(self.context_proj(full_context))
        return brain_input

    def think(self, brain_input: torch.Tensor, h: Optional[torch.Tensor] = None):
        """Step ricorrente (GRU + Residual) - il 'pensiero'"""
        if h is None:
            h = self.h_init.expand(brain_input.size(0), -1).contiguous()

        h_new = self.gru_cell(brain_input, h)
        h_new = self.think_dropout(h_new)

        combined = torch.cat([brain_input, h_new], dim=-1)
        y = h_new + self.residual(combined)
        return y, h_new

    def forward(self, inputs: Dict[str, torch.Tensor], h: Optional[torch.Tensor] = None):
        """Forward completo: extract_features -> think"""
        brain_input = self.extract_features(inputs)
        return self.think(brain_input, h)

    def get_heads_dict(self, y, temperature: float = 1.0):
        """Restituisce le predizioni dalle heads - interfaccia per Trainer"""
        # Temperature scaling per softmax-based heads
        temp = max(0.1, temperature)  # Evita divisione per zero
        return {
            "side": self.head_side(y) / temp,  # Scaled logits
            "qty": torch.sigmoid(self.head_qty(y)),
            "price_offset": torch.tanh(self.head_px(y)),
            "tp_mult": F.softplus(self.head_tp(y)),
            "sl_mult": F.softplus(self.head_sl(y)),
            "halt_prob": torch.sigmoid(self.head_halt(y) / temp), # Temperature sensitive halt
            "ordertype": self.head_ordertype(y) / temp,  # Scaled logits
            "leverage": (F.softplus(self.head_lev(y)) + 1.0).clamp(1.0, 5.0)
        }
