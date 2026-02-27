"""
Stochastic gene expression models for simulation.

Implements biologically realistic expression patterns matching reference core.py:

Factor models:
- deterministic: Active get HIGH, others get BASAL (both with lognormal noise)
- stochastic: Bernoulli on/off + gamma or lognormal (CV-based)
- stochastic_ref: Bernoulli on/off with separate sigma_f / sigma_f_basal
- bernoulli_mixture: Full mixture with non-sender basal expression

Response models:
- deterministic: Pure Michaelis-Menten activation, no Bernoulli switching
- stochastic_hill: Hill prob -> Bernoulli -> mixture
- bernoulli_constant: Fixed p_r -> Bernoulli -> mixture
- bernoulli_hill: Hill p_r(C) with K_p param -> Bernoulli -> mixture
"""

import numpy as np

from sigdiscovpy.simulation.config.dataclasses import (
    ExpressionConfig,
    StochasticConfig,
)


class ExpressionGenerator:
    """
    Generates stochastic gene expression patterns.

    Supports multiple factor and response models matching reference core.py.

    Parameters
    ----------
    expr_config : ExpressionConfig
        Expression level parameters.
    stoch_config : StochasticConfig
        Stochastic model parameters.
    seed : int, optional
        Random seed for reproducibility.
    diffusion_Kd : float
        Dissociation constant from diffusion config, used by response models
        for Michaelis-Menten activation.
    """

    def __init__(
        self,
        expr_config: ExpressionConfig,
        stoch_config: StochasticConfig,
        seed: int = None,
        diffusion_Kd: float = 30.0,
    ):
        self.expr = expr_config
        self.stoch = stoch_config
        self._rng = np.random.default_rng(seed)
        self.diffusion_Kd = diffusion_Kd

    # =========================================================================
    # Factor Expression Models
    # =========================================================================

    def generate_factor_expression(
        self,
        n_total: int,
        n_active: int,
        active_indices: np.ndarray,
        sender_indices: np.ndarray = None,
        silent_indices: np.ndarray = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate factor expression using the configured model.

        Parameters
        ----------
        n_total : int
            Total number of cells.
        n_active : int
            Number of designated active sender cells.
        active_indices : np.ndarray
            Indices of active sender cells.
        sender_indices : np.ndarray, optional
            Indices of all sender cells (active + silent). Needed for bernoulli_mixture.
        silent_indices : np.ndarray, optional
            Indices of silent sender cells.

        Returns
        -------
        factor_expr : np.ndarray
            Expression values for all cells (n_total,).
        expressing_mask : np.ndarray or None
            Boolean mask indicating which senders are "on" (n_active,).
        """
        model = self.expr.factor_model

        if model == "deterministic":
            return self._factor_deterministic(n_total, n_active, active_indices, silent_indices)
        elif model == "stochastic":
            return self._factor_stochastic(n_total, n_active, active_indices)
        elif model == "stochastic_ref":
            return self._factor_stochastic_ref(n_total, n_active, active_indices)
        elif model == "bernoulli_mixture":
            if sender_indices is None:
                raise ValueError("sender_indices required for bernoulli_mixture model")
            return self._factor_bernoulli_mixture(n_total, sender_indices)
        else:
            raise ValueError(f"Unknown factor_model: {model}")

    def _factor_deterministic(
        self,
        n_total: int,
        n_active: int,
        active_indices: np.ndarray,
        silent_indices: np.ndarray = None,
    ) -> tuple[np.ndarray, None]:
        """Deterministic factor expression matching reference unified_sim.py.

        All cells: BASAL * LN(0, sigma_f)
        Active: HIGH * LN(0, sigma_f)
        Silent: BASAL * LN(0, sigma_f) or 0 if silent_expr_zero
        """
        factor_expr = self.expr.F_basal * np.random.lognormal(0, self.expr.sigma_f, n_total)
        factor_expr[active_indices] = self.expr.F_high * np.random.lognormal(
            0, self.expr.sigma_f, n_active
        )
        if silent_indices is not None and len(silent_indices) > 0:
            # silent_expr_zero is handled by the runner, here we just assign basal
            factor_expr[silent_indices] = self.expr.F_basal * np.random.lognormal(
                0, self.expr.sigma_f, len(silent_indices)
            )
        return factor_expr, None

    def _factor_stochastic(
        self,
        n_total: int,
        n_active: int,
        active_indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Stochastic factor expression matching reference generate_stochastic_expression().

        Bernoulli on/off + gamma or lognormal (CV-based).
        """
        factor_expr = self.expr.F_basal * np.random.lognormal(0, self.expr.sigma_f, n_total)

        expressing_mask = np.random.rand(n_active) < self.stoch.p_sender_express
        n_expressing = np.sum(expressing_mask)

        if n_expressing > 0:
            expressing_indices = active_indices[expressing_mask]
            expr_cv = self.stoch.expr_cv

            if self.stoch.use_gamma_dist:
                shape = 1.0 / (expr_cv**2)
                scale = self.expr.F_high * (expr_cv**2)
                expr_values = np.random.gamma(shape, scale, n_expressing)
            else:
                sigma_ln = np.sqrt(np.log(1 + expr_cv**2))
                mu_ln = np.log(self.expr.F_high) - sigma_ln**2 / 2
                expr_values = np.random.lognormal(mu_ln, sigma_ln, n_expressing)

            # Ensure minimum expression above basal
            expr_values = np.maximum(expr_values, self.expr.F_basal * 2)
            factor_expr[expressing_indices] = expr_values

        return factor_expr, expressing_mask

    def _factor_stochastic_ref(
        self,
        n_total: int,
        n_active: int,
        active_indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Stochastic ref factor expression matching reference generate_stochastic_expression_ref().

        F_i = S_i * F_high * LN(0, sigma_f^2) + (1-S_i) * F_basal * LN(0, sigma_f_basal^2)
        """
        factor_expr = self.expr.F_basal * np.random.lognormal(0, self.stoch.sigma_f_basal, n_total)

        expressing_mask = np.random.rand(n_active) < self.stoch.p_sender_express
        n_expressing = np.sum(expressing_mask)

        if n_expressing > 0:
            expressing_indices = active_indices[expressing_mask]
            factor_expr[expressing_indices] = self.expr.F_high * np.random.lognormal(
                0, self.expr.sigma_f, n_expressing
            )

        return factor_expr, expressing_mask

    def _factor_bernoulli_mixture(
        self,
        n_total: int,
        sender_indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Bernoulli mixture factor expression matching reference generate_bernoulli_factor_expression().

        Non-senders: F_basal * LN(0, sigma_f_b)
        Senders: S_i * F_high * LN(0, sigma_f) + (1-S_i) * F_basal * LN(0, sigma_f_b)
        """
        factor_expr = np.zeros(n_total)

        # Non-sender cells: always basal expression
        non_sender_mask = np.ones(n_total, dtype=bool)
        non_sender_mask[sender_indices] = False
        n_non_senders = np.sum(non_sender_mask)
        factor_expr[non_sender_mask] = self.expr.F_basal * np.random.lognormal(
            0, self.stoch.sigma_f_b, n_non_senders
        )

        # Sender cells: stochastic S_i ~ Bernoulli(p_s)
        n_senders = len(sender_indices)
        S = np.random.binomial(1, self.stoch.p_s, n_senders)

        high_expr = self.expr.F_high * np.random.lognormal(0, self.expr.sigma_f, n_senders)
        basal_expr = self.expr.F_basal * np.random.lognormal(0, self.stoch.sigma_f_b, n_senders)
        factor_expr[sender_indices] = S * high_expr + (1 - S) * basal_expr

        active_sender_mask = S == 1
        return factor_expr, active_sender_mask

    # =========================================================================
    # Response Expression Models
    # =========================================================================

    def generate_response_expression(
        self,
        n_total: int,
        receiver_indices: np.ndarray,
        concentrations: np.ndarray,
        active_receiver_indices: np.ndarray = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate response expression using the configured model.

        Parameters
        ----------
        n_total : int
            Total number of cells.
        receiver_indices : np.ndarray
            Indices of all receiver cells.
        concentrations : np.ndarray
            Concentration at each cell (n_total,).
        active_receiver_indices : np.ndarray, optional
            Indices of active (non-silent) receivers. Used for deterministic model
            when receiver_silent_fraction > 0.

        Returns
        -------
        response_expr : np.ndarray
            Expression values for all cells (n_total,).
        responding_mask : np.ndarray or None
            Boolean mask for responding receivers.
        response_probs : np.ndarray or None
            Response probability for each receiver.
        """
        model = self.expr.response_model

        if model == "deterministic":
            return self._response_deterministic(
                n_total, receiver_indices, concentrations, active_receiver_indices
            )
        elif model == "stochastic_hill":
            return self._response_stochastic_hill(n_total, receiver_indices, concentrations)
        elif model == "bernoulli_constant":
            return self._response_bernoulli_constant(n_total, receiver_indices, concentrations)
        elif model == "bernoulli_hill":
            return self._response_bernoulli_hill(n_total, receiver_indices, concentrations)
        else:
            raise ValueError(f"Unknown response_model: {model}")

    def _response_deterministic(
        self,
        n_total: int,
        receiver_indices: np.ndarray,
        concentrations: np.ndarray,
        active_receiver_indices: np.ndarray = None,
    ) -> tuple[np.ndarray, None, None]:
        """Deterministic response matching reference unified_sim.py.

        R = BASAL * (1 + FC * Act) * LN(0, sigma_r)
        """
        Kd = self.diffusion_Kd

        responsive_expr = self.expr.R_basal * np.random.lognormal(0, self.expr.sigma_r, n_total)
        ri = active_receiver_indices if active_receiver_indices is not None else receiver_indices
        activation = concentrations[ri] / (Kd + concentrations[ri])
        responsive_expr[ri] = self.expr.R_basal * (1 + self.expr.fold_change * activation)

        return responsive_expr, None, None

    def _response_stochastic_hill(
        self,
        n_total: int,
        receiver_indices: np.ndarray,
        concentrations: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Stochastic Hill response matching reference generate_stochastic_response().

        1. p_j = p_max * C^n / (Kd^n + C^n) using diffusion Kd
        2. B_j ~ Bernoulli(p_j)
        3. If B_j=1: R = R_basal * (1 + FC * Act) * LN(0, sigma_r)
           If B_j=0: R = R_basal * LN(0, sigma_r_basal)
        """
        Kd = self.diffusion_Kd
        n_receivers = len(receiver_indices)

        responsive_expr = self.expr.R_basal * np.random.lognormal(0, self.stoch.sigma_r_b, n_total)

        C = concentrations[receiver_indices]
        hill_coef = self.stoch.response_hill_coef
        C_n = np.power(C, hill_coef)
        Kd_n = np.power(Kd, hill_coef)
        response_probs = self.stoch.p_respond_max * C_n / (Kd_n + C_n)

        responding_mask = np.random.rand(n_receivers) < response_probs
        n_responding = np.sum(responding_mask)

        # Non-responding receivers
        non_responding_mask = ~responding_mask
        n_non_responding = np.sum(non_responding_mask)
        if n_non_responding > 0:
            non_responding_indices = receiver_indices[non_responding_mask]
            responsive_expr[non_responding_indices] = self.expr.R_basal * np.random.lognormal(
                0, self.stoch.sigma_r_b, n_non_responding
            )

        # Responding receivers
        if n_responding > 0:
            responding_indices = receiver_indices[responding_mask]
            C_responding = C[responding_mask]
            activation = C_responding / (Kd + C_responding)
            mean_expr = self.expr.R_basal * (1 + self.expr.fold_change * activation)
            responsive_expr[responding_indices] = mean_expr * np.random.lognormal(
                0, self.expr.sigma_r, n_responding
            )

        return responsive_expr, responding_mask, response_probs

    def _response_bernoulli_constant(
        self,
        n_total: int,
        receiver_indices: np.ndarray,
        concentrations: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, None]:
        """Bernoulli constant response matching reference generate_bernoulli_response_constant().

        B_j ~ Bernoulli(p_r) for receiver cells.
        R_j = B_j * R_basal * (1+FC*Act) * LN(0,sigma_r) + (1-B_j) * R_basal * LN(0,sigma_r_b)
        """
        Kd = self.diffusion_Kd
        n_receivers = len(receiver_indices)

        responsive_expr = np.zeros(n_total)

        # Non-receiver cells: always basal
        non_receiver_mask = np.ones(n_total, dtype=bool)
        non_receiver_mask[receiver_indices] = False
        n_non_receivers = np.sum(non_receiver_mask)
        responsive_expr[non_receiver_mask] = self.expr.R_basal * np.random.lognormal(
            0, self.stoch.sigma_r_b, n_non_receivers
        )

        # Receiver cells: stochastic B_j ~ Bernoulli(p_r)
        B = np.random.binomial(1, self.stoch.p_r, n_receivers)

        C_receivers = concentrations[receiver_indices]
        activation = C_receivers / (Kd + C_receivers)

        activated_expr = (
            self.expr.R_basal
            * (1 + self.expr.fold_change * activation)
            * np.random.lognormal(0, self.expr.sigma_r, n_receivers)
        )
        basal_expr = self.expr.R_basal * np.random.lognormal(0, self.stoch.sigma_r_b, n_receivers)
        responsive_expr[receiver_indices] = B * activated_expr + (1 - B) * basal_expr

        responding_mask = B == 1
        return responsive_expr, responding_mask, None

    def _response_bernoulli_hill(
        self,
        n_total: int,
        receiver_indices: np.ndarray,
        concentrations: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bernoulli Hill response matching reference generate_bernoulli_response_hill().

        p_r(C_j) = p_r_max * C_j^n / (K_p^n + C_j^n)
        B_j ~ Bernoulli(p_r(C_j))
        R_j = B_j * R_basal * (1+FC*Act) * LN(0,sigma_r) + (1-B_j) * R_basal * LN(0,sigma_r_b)
        """
        Kd = self.diffusion_Kd
        n_receivers = len(receiver_indices)

        responsive_expr = np.zeros(n_total)

        # Non-receiver cells: always basal
        non_receiver_mask = np.ones(n_total, dtype=bool)
        non_receiver_mask[receiver_indices] = False
        n_non_receivers = np.sum(non_receiver_mask)
        responsive_expr[non_receiver_mask] = self.expr.R_basal * np.random.lognormal(
            0, self.stoch.sigma_r_b, n_non_receivers
        )

        # Receiver cells: concentration-dependent p_r via Hill function
        C_receivers = concentrations[receiver_indices]
        C_n = np.power(np.maximum(C_receivers, 0), self.stoch.hill_n)
        K_n = np.power(self.stoch.K_p, self.stoch.hill_n)
        p_r_values = self.stoch.p_r_max * C_n / (K_n + C_n)

        B = np.random.binomial(1, p_r_values)

        activation = C_receivers / (Kd + C_receivers)

        activated_expr = (
            self.expr.R_basal
            * (1 + self.expr.fold_change * activation)
            * np.random.lognormal(0, self.expr.sigma_r, n_receivers)
        )
        basal_expr = self.expr.R_basal * np.random.lognormal(0, self.stoch.sigma_r_b, n_receivers)
        responsive_expr[receiver_indices] = B * activated_expr + (1 - B) * basal_expr

        responding_mask = B == 1
        return responsive_expr, responding_mask, p_r_values

    # =========================================================================
    # VST Factor/Response (special handling for VST mode)
    # =========================================================================

    def generate_vst_factor(
        self,
        n_total: int,
        n_active: int,
        active_indices: np.ndarray,
        sender_indices: np.ndarray,
        silent_indices: np.ndarray = None,
    ) -> tuple[np.ndarray, None]:
        """Generate raw factor expression for VST mode (before transform).

        Matches reference unified_sim.py VST block.
        """
        factor_raw = self.expr.F_basal * np.random.lognormal(0, self.expr.sigma_f, n_total)

        # Zero-inflate
        if self.stoch.zero_inflate_factor > 0:
            zero_mask = np.random.rand(n_total) < self.stoch.zero_inflate_factor
            zero_mask[sender_indices] = False
            factor_raw[zero_mask] = 0.0

        factor_raw[active_indices] = self.expr.F_high * np.random.lognormal(
            0, self.expr.sigma_f, n_active
        )
        if silent_indices is not None and len(silent_indices) > 0:
            factor_raw[silent_indices] = self.expr.F_basal * np.random.lognormal(
                0, self.expr.sigma_f, len(silent_indices)
            )

        return factor_raw, None

    def generate_vst_response(
        self,
        n_total: int,
        receiver_indices: np.ndarray,
        concentrations: np.ndarray,
    ) -> tuple[np.ndarray, None, None]:
        """Generate raw response expression for VST mode (before transform).

        Matches reference unified_sim.py VST block.
        """
        Kd = self.diffusion_Kd

        responsive_raw = self.expr.R_basal * np.random.lognormal(0, self.expr.sigma_r, n_total)

        if self.stoch.zero_inflate_response > 0:
            zero_mask_r = np.random.rand(n_total) < self.stoch.zero_inflate_response
            zero_mask_r[receiver_indices] = False
            responsive_raw[zero_mask_r] = 0.0

        activation = concentrations[receiver_indices] / (Kd + concentrations[receiver_indices])
        responsive_raw[receiver_indices] = self.expr.R_basal * (
            1 + self.expr.fold_change * activation
        )

        return responsive_raw, None, None

    # =========================================================================
    # Utility methods
    # =========================================================================

    def generate_noise_only(
        self,
        n_total: int,
        base_expression: float,
        sigma: float,
        zero_inflate: float = 0.0,
    ) -> np.ndarray:
        """Generate expression with noise only (no spatial signal)."""
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
        """Add technical noise to expression values."""
        noisy_expr = expression.copy()

        if noise_level > 0:
            noise = self._rng.lognormal(0, noise_level, len(expression))
            noisy_expr *= noise

        if dropout_rate > 0:
            dropout_mask = self._rng.random(len(expression)) < dropout_rate
            noisy_expr[dropout_mask] = 0.0

        return noisy_expr
