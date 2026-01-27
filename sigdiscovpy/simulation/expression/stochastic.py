"""
Stochastic gene expression models for simulation.

Implements biologically realistic expression patterns:
- Lognormal distribution for expression levels
- Bernoulli on/off states for sender expression
- Hill function dose-response for receiver activation
- Zero-inflation for dropout modeling
"""

import numpy as np

from sigdiscovpy.simulation.config.dataclasses import (
    ExpressionConfig,
    StochasticConfig,
)


class ExpressionGenerator:
    """
    Generates stochastic gene expression patterns.

    Models both factor (sender) and response (receiver) expression with
    biologically realistic noise models including:
    - Lognormal noise on expression levels
    - Bernoulli on/off states
    - Hill function dose-response
    - Zero-inflation (dropout)

    Parameters
    ----------
    expr_config : ExpressionConfig
        Expression level parameters.
    stoch_config : StochasticConfig
        Stochastic model parameters.
    seed : int, optional
        Random seed for reproducibility.

    Example
    -------
    >>> expr_config = ExpressionConfig(F_high=10.0, fold_change=5.0)
    >>> stoch_config = StochasticConfig(p_sender_express=0.9)
    >>> generator = ExpressionGenerator(expr_config, stoch_config)
    >>> factor_expr, mask = generator.generate_factor_expression(1000, 20, active_idx)
    """

    def __init__(
        self,
        expr_config: ExpressionConfig,
        stoch_config: StochasticConfig,
        seed: int = None,
    ):
        self.expr = expr_config
        self.stoch = stoch_config
        self._rng = np.random.default_rng(seed)

    def generate_factor_expression(
        self,
        n_total: int,
        n_active: int,
        active_indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate stochastic factor expression.

        Model:
            F_i = S_i * F_high * LogNormal(0, σ_f²) +
                  (1-S_i) * F_basal * LogNormal(0, σ_basal²)
            S_i ~ Bernoulli(p_sender_express)

        Parameters
        ----------
        n_total : int
            Total number of cells.
        n_active : int
            Number of designated sender cells.
        active_indices : np.ndarray
            Indices of sender cells.

        Returns
        -------
        factor_expr : np.ndarray
            Expression values for all cells (n_total,).
        expressing_mask : np.ndarray
            Boolean mask indicating which senders are "on" (n_active,).
        """
        # Initialize with basal expression
        factor_expr = self.expr.F_basal * self._rng.lognormal(0, self.expr.sigma_f_basal, n_total)

        # Zero-inflation for non-senders
        if self.stoch.zero_inflate_factor > 0:
            zero_mask = self._rng.random(n_total) < self.stoch.zero_inflate_factor
            # Never zero-inflate senders
            zero_mask[active_indices] = False
            factor_expr[zero_mask] = 0.0

        # Stochastic sender expression (Bernoulli on/off)
        expressing_mask = self._rng.random(n_active) < self.stoch.p_sender_express
        n_expressing = np.sum(expressing_mask)

        if n_expressing > 0:
            expressing_indices = active_indices[expressing_mask]
            factor_expr[expressing_indices] = self.expr.F_high * self._rng.lognormal(
                0, self.expr.sigma_f, n_expressing
            )

        return factor_expr, expressing_mask

    def generate_response_expression(
        self,
        n_total: int,
        receiver_indices: np.ndarray,
        concentrations: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate stochastic response expression.

        Hybrid Model:
            1. B_j ~ Bernoulli(p_max * C^n / (Kd^n + C^n))  [Hill function]
            2. Act_j = C / (Kd + C)  [Activation level]
            3. If responding: R_j = R_basal * (1 + FC * Act_j) * LogNormal(0, σ_r²)
               If not: R_j = R_basal * LogNormal(0, σ_basal²)

        Parameters
        ----------
        n_total : int
            Total number of cells.
        receiver_indices : np.ndarray
            Indices of receiver cells.
        concentrations : np.ndarray
            Concentration at each cell (n_total,).

        Returns
        -------
        response_expr : np.ndarray
            Expression values for all cells (n_total,).
        responding_mask : np.ndarray
            Boolean mask for responding receivers (n_receivers,).
        response_probs : np.ndarray
            Response probability for each receiver (n_receivers,).
        """
        n_receivers = len(receiver_indices)

        # Initialize with basal expression
        response_expr = self.expr.R_basal * self._rng.lognormal(0, self.expr.sigma_r_basal, n_total)

        # Zero-inflation for non-receivers
        if self.stoch.zero_inflate_response > 0:
            zero_mask = self._rng.random(n_total) < self.stoch.zero_inflate_response
            zero_mask[receiver_indices] = False
            response_expr[zero_mask] = 0.0

        # Get concentrations at receiver positions
        C = concentrations[receiver_indices]

        # Calculate response probability (Hill function)
        n = self.stoch.hill_coefficient
        Kd = 1.0  # Normalized Kd for response probability
        C_n = np.power(np.maximum(C, 0), n)
        Kd_n = np.power(Kd, n)
        response_probs = self.stoch.p_receiver_respond_max * C_n / (Kd_n + C_n)

        # Stochastic binary decision
        responding_mask = self._rng.random(n_receivers) < response_probs

        # Non-responding receivers: basal expression
        non_responding_indices = receiver_indices[~responding_mask]
        n_non_responding = len(non_responding_indices)
        if n_non_responding > 0:
            response_expr[non_responding_indices] = self.expr.R_basal * self._rng.lognormal(
                0, self.expr.sigma_r_basal, n_non_responding
            )

        # Responding receivers: activated expression
        responding_indices = receiver_indices[responding_mask]
        n_responding = len(responding_indices)
        if n_responding > 0:
            C_responding = C[responding_mask]
            # Michaelis-Menten activation
            activation = C_responding / (Kd + C_responding)
            # Mean expression scales with activation
            mean_expr = self.expr.R_basal * (1 + self.expr.fold_change * activation)
            response_expr[responding_indices] = mean_expr * self._rng.lognormal(
                0, self.expr.sigma_r, n_responding
            )

        return response_expr, responding_mask, response_probs

    def generate_noise_only(
        self,
        n_total: int,
        base_expression: float,
        sigma: float,
        zero_inflate: float = 0.0,
    ) -> np.ndarray:
        """
        Generate expression with noise only (no spatial signal).

        Useful for null hypothesis testing and control simulations.

        Parameters
        ----------
        n_total : int
            Number of cells.
        base_expression : float
            Mean expression level.
        sigma : float
            Lognormal sigma parameter.
        zero_inflate : float
            Fraction of cells with zero expression.

        Returns
        -------
        np.ndarray
            Expression values (n_total,).
        """
        expr = base_expression * self._rng.lognormal(0, sigma, n_total)

        if zero_inflate > 0:
            zero_mask = self._rng.random(n_total) < zero_inflate
            expr[zero_mask] = 0.0

        return expr

    def add_technical_noise(
        self,
        expression: np.ndarray,
        noise_level: float = 0.1,
        dropout_rate: float = 0.0,
    ) -> np.ndarray:
        """
        Add technical noise to expression values.

        Models sequencing/measurement noise on top of biological variation.

        Parameters
        ----------
        expression : np.ndarray
            Input expression values.
        noise_level : float
            Multiplicative noise level (CV).
        dropout_rate : float
            Probability of dropout (expression -> 0).

        Returns
        -------
        np.ndarray
            Expression with added technical noise.
        """
        noisy_expr = expression.copy()

        # Multiplicative noise
        if noise_level > 0:
            noise = self._rng.lognormal(0, noise_level, len(expression))
            noisy_expr *= noise

        # Dropout
        if dropout_rate > 0:
            dropout_mask = self._rng.random(len(expression)) < dropout_rate
            noisy_expr[dropout_mask] = 0.0

        return noisy_expr
