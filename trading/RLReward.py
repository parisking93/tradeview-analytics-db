import torch

class PnlRewardManager:
    """
    Gestore dei reward per il Reinforcement Learning basato sul PnL.
    Fornisce reward positivi per profitti e negativi per perdite,
    oltre a penalità per hold non ottimali.
    """
    def __init__(self,
                 pnl_scale=1.0,
                 win_bonus=0.2,
                 loss_penalty=0.5,
                 halt_penalty=0.05,
                 overtrading_penalty=0.1):
        self.pnl_scale = pnl_scale
        self.win_bonus = win_bonus
        self.loss_penalty = loss_penalty
        self.halt_penalty = halt_penalty
        self.overtrading_penalty = overtrading_penalty

    def calculate_reward(self, pnl_delta, is_closed=False, total_pnl=0.0, side='HOLD'):
        """
        Calcola il reward istantaneo basato sulla variazione di PnL.

        Args:
            pnl_delta (float): Variazione di PnL dall'ultimo step.
            is_closed (bool): Se la posizione è stata chiusa in questo step.
            total_pnl (float): PnL totale dell'operazione.
            side (str): L'azione intrapresa ('BUY', 'SELL', 'HOLD').

        Returns:
            float: Il reward calcolato.
        """
        reward = pnl_delta * self.pnl_scale

        if is_closed:
            # Bonus/Penalità extra alla chiusura
            if total_pnl > 0:
                reward += self.win_bonus
            else:
                reward -= self.loss_penalty

        # Penalità per HOLD se il mercato si muove drasticamente contro una potenziale posizione
        # (Questo richiede logica esterna per determinare se un'opportunità è stata persa)

        return reward

    def get_halt_penalty(self, trend_strength, side_prediction):
        """
        Penalizza il modello se predice HALT (HOLD) durante un trend forte.
        """
        if trend_strength > 0.02 and side_prediction == 2: # 2 = HOLD
            return -self.halt_penalty * trend_strength
        return 0.0

    def compute_policy_loss(self, rewards, log_probs):
        """
        Calcola la loss per un gradiente di policy (metodo semplice).
        REINFORCE style: loss = -reward * log_prob
        """
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards, device=log_probs.device)

        # Sottraiamo una baseline (opzionale) per ridurre la varianza
        # baseline = rewards.mean()
        # adjusted_rewards = rewards - baseline

        return -(rewards * log_probs).mean()
