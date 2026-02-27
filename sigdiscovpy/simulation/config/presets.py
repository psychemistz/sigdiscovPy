"""
Preset configurations for common simulation scenarios.

Includes:
- 13 demo presets matching reference unified_sim.py exactly
- 14 generic presets for common use cases
"""

import numpy as np

from sigdiscovpy.simulation.config.dataclasses import (
    AnalysisConfig,
    CellTypeConfig,
    DiffusionConfig,
    DomainConfig,
    ExpressionConfig,
    PositionConfig,
    SenderPositionMode,
    SimulationConfig,
    StochasticConfig,
    WeightType,
)


class SimulationPresets:
    """
    Factory class for preset simulation configurations.

    Example
    -------
    >>> config = SimulationPresets.demo()
    >>> config = SimulationPresets.demo3b()
    >>> config = SimulationPresets.get_preset("demo_vst")
    """

    # =========================================================================
    # 13 Demo Presets (matching unified_sim.py _register() calls exactly)
    # =========================================================================

    @staticmethod
    def demo() -> SimulationConfig:
        """demo.py preset - Basic deterministic simulation."""
        return SimulationConfig(
            domain=DomainConfig(center=(0.0, 0.0)),
            cell_types=CellTypeConfig(
                n_active_senders=20,
                n_silent_senders=0,
                receiver_fractions=[0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
            ),
            position=PositionConfig(mode=SenderPositionMode.CENTER),
            diffusion=DiffusionConfig(D=100.0, k_max=10.0, Kd=30.0, active_threshold=1.0),
            expression=ExpressionConfig(
                F_basal=0.1,
                F_high=10.0,
                R_basal=0.1,
                fold_change=2.0,
                sigma_f=0.1,
                sigma_r=0.1,
                factor_model="deterministic",
                response_model="deterministic",
            ),
            analysis=AnalysisConfig(
                bandwidth=100.0,
                ind_methods=["ring"],
                radii=list(np.arange(50, 5001, 25)),
            ),
        )

    @staticmethod
    def demo1() -> SimulationConfig:
        """demo1.py preset - Same as demo (basic deterministic)."""
        return SimulationPresets.demo()

    @staticmethod
    def demo1b() -> SimulationConfig:
        """demo1b.py preset - Ring + Gaussian annular comparison."""
        return SimulationConfig(
            domain=DomainConfig(center=(0.0, 0.0)),
            cell_types=CellTypeConfig(
                n_active_senders=20,
                n_silent_senders=0,
                receiver_fractions=[0.02, 0.05, 0.10, 0.20, 0.30, 0.50],
            ),
            position=PositionConfig(mode=SenderPositionMode.CENTER),
            diffusion=DiffusionConfig(D=100.0, k_max=10.0, Kd=30.0, active_threshold=1.0),
            expression=ExpressionConfig(
                F_basal=0.1,
                F_high=10.0,
                R_basal=0.1,
                fold_change=2.0,
                sigma_f=0.1,
                sigma_r=0.1,
                factor_model="deterministic",
                response_model="deterministic",
            ),
            analysis=AnalysisConfig(
                bandwidth=100.0,
                sigma_fraction=3.0,
                ind_methods=["ring", "gaussian_annular"],
                radii=list(np.arange(110, 3001, 25)),
            ),
        )

    @staticmethod
    def demo1_stc() -> SimulationConfig:
        """demo1_stc.py preset - Bernoulli mixture factor + constant response."""
        return SimulationConfig(
            domain=DomainConfig(center=(0.0, 0.0)),
            cell_types=CellTypeConfig(
                n_active_senders=20,
                n_silent_senders=0,
                receiver_fractions=[0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
            ),
            position=PositionConfig(mode=SenderPositionMode.CENTER),
            diffusion=DiffusionConfig(D=100.0, k_max=10.0, Kd=30.0, active_threshold=1.0),
            expression=ExpressionConfig(
                F_basal=0.1,
                F_high=10.0,
                R_basal=0.1,
                fold_change=2.0,
                sigma_f=0.1,
                sigma_r=0.01,
                factor_model="bernoulli_mixture",
                response_model="bernoulli_constant",
            ),
            stochastic=StochasticConfig(
                p_s=1.0,
                sigma_f_b=0.1,
                p_r=1.0,
                sigma_r_b=0.1,
            ),
            analysis=AnalysisConfig(
                bandwidth=100.0, ind_methods=["ring"], radii=list(np.arange(50, 5001, 25))
            ),
        )

    @staticmethod
    def demo1_stc2() -> SimulationConfig:
        """demo1_stc2.py preset - Bernoulli mixture factor + Hill response."""
        return SimulationConfig(
            domain=DomainConfig(center=(0.0, 0.0)),
            cell_types=CellTypeConfig(
                n_active_senders=20,
                n_silent_senders=0,
                receiver_fractions=[0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
            ),
            position=PositionConfig(mode=SenderPositionMode.CENTER),
            diffusion=DiffusionConfig(D=100.0, k_max=10.0, Kd=30.0, active_threshold=1.0),
            expression=ExpressionConfig(
                F_basal=0.1,
                F_high=10.0,
                R_basal=0.1,
                fold_change=2.0,
                sigma_f=0.1,
                sigma_r=0.01,
                factor_model="bernoulli_mixture",
                response_model="bernoulli_hill",
            ),
            stochastic=StochasticConfig(
                p_s=1.0,
                sigma_f_b=0.1,
                p_r_max=1.0,
                K_p=1.0,
                hill_n=1.0,
                sigma_r_b=0.1,
            ),
            analysis=AnalysisConfig(
                bandwidth=100.0, ind_methods=["ring"], radii=list(np.arange(50, 5001, 25))
            ),
        )

    @staticmethod
    def demo2a() -> SimulationConfig:
        """demo2a.py preset - Center active, distributed silent senders."""
        return SimulationConfig(
            domain=DomainConfig(center=(0.0, 0.0)),
            cell_types=CellTypeConfig(
                n_active_senders=20,
                n_silent_senders=100,
                receiver_fractions=[0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
            ),
            position=PositionConfig(mode=SenderPositionMode.CENTER_SILENT_DISTRIBUTED),
            diffusion=DiffusionConfig(D=100.0, k_max=10.0, Kd=30.0, active_threshold=1.0),
            expression=ExpressionConfig(
                F_basal=0.1,
                F_high=100.0,
                R_basal=0.1,
                fold_change=50.0,
                sigma_f=0.1,
                sigma_r=0.1,
                factor_model="deterministic",
                response_model="deterministic",
            ),
            analysis=AnalysisConfig(
                bandwidth=100.0, ind_methods=["ring"], radii=list(np.arange(50, 5001, 25))
            ),
        )

    @staticmethod
    def demo2b() -> SimulationConfig:
        """demo2b.py preset - Center active + distributed silent + silent receivers."""
        return SimulationConfig(
            domain=DomainConfig(center=(0.0, 0.0)),
            cell_types=CellTypeConfig(
                n_active_senders=20,
                n_silent_senders=20,
                receiver_fractions=[0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
                receiver_silent_fraction=0.8,
            ),
            position=PositionConfig(mode=SenderPositionMode.CENTER_SILENT_DISTRIBUTED),
            diffusion=DiffusionConfig(D=100.0, k_max=10.0, Kd=30.0, active_threshold=1.0),
            expression=ExpressionConfig(
                F_basal=0.1,
                F_high=10.0,
                R_basal=0.1,
                fold_change=2.0,
                sigma_f=0.1,
                sigma_r=0.1,
                factor_model="deterministic",
                response_model="deterministic",
            ),
            analysis=AnalysisConfig(
                bandwidth=100.0,
                ind_methods=["ring"],
                ind_uses_active_receivers_only=True,
                radii=list(np.arange(50, 5001, 25)),
            ),
        )

    @staticmethod
    def demo2c() -> SimulationConfig:
        """demo2c.py preset - Multi random positions, fixed senders."""
        return SimulationConfig(
            domain=DomainConfig(center=(0.0, 0.0)),
            cell_types=CellTypeConfig(
                n_active_senders=20,
                n_silent_senders=40,
                receiver_fractions=[0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
                fix_senders_across_fractions=True,
            ),
            position=PositionConfig(
                mode=SenderPositionMode.RANDOM,
                n_positions=20,
                min_separation=800.0,
                offset_distance=3000.0,
            ),
            diffusion=DiffusionConfig(
                D=100.0,
                k_max=10.0,
                Kd=30.0,
                secretion_rate=100.0,
                active_threshold=1.0,
            ),
            expression=ExpressionConfig(
                F_basal=0.1,
                F_high=1.5,
                R_basal=0.1,
                fold_change=2.0,
                sigma_f=0.1,
                sigma_r=0.1,
                factor_model="deterministic",
                response_model="deterministic",
            ),
            analysis=AnalysisConfig(
                bandwidth=100.0, ind_methods=["ring"], radii=list(np.arange(50, 5001, 25))
            ),
        )

    @staticmethod
    def demo3() -> SimulationConfig:
        """demo3.py preset - Stochastic factor, random position, low threshold."""
        return SimulationConfig(
            domain=DomainConfig(center=(0.0, 0.0)),
            cell_types=CellTypeConfig(
                n_active_senders=200,
                n_silent_senders=100,
                receiver_fractions=[0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
                fix_senders_across_fractions=True,
            ),
            position=PositionConfig(
                mode=SenderPositionMode.RANDOM,
                n_positions=1,
                min_separation=0.0,
                offset_distance=3000.0,
            ),
            diffusion=DiffusionConfig(
                D=100.0,
                k_max=50.0,
                Kd=30.0,
                secretion_rate=0.5,
                active_threshold=0.2,
            ),
            expression=ExpressionConfig(
                F_basal=0.1,
                F_high=1.5,
                R_basal=0.1,
                fold_change=2.0,
                sigma_f=0.1,
                sigma_r=0.1,
                factor_model="stochastic",
                response_model="deterministic",
            ),
            stochastic=StochasticConfig(
                p_sender_express=0.5,
                expr_cv=0.5,
                use_gamma_dist=False,
            ),
            analysis=AnalysisConfig(
                bandwidth=20.0, ind_methods=["ring"], radii=list(np.arange(10, 5001, 25))
            ),
        )

    @staticmethod
    def demo3b() -> SimulationConfig:
        """demo3b.py preset - Stochastic ref factor + stochastic Hill response."""
        return SimulationConfig(
            domain=DomainConfig(center=(0.0, 0.0)),
            cell_types=CellTypeConfig(
                n_active_senders=200,
                n_silent_senders=100,
                receiver_fractions=[0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
                fix_senders_across_fractions=True,
            ),
            position=PositionConfig(
                mode=SenderPositionMode.RANDOM,
                n_positions=1,
                min_separation=0.0,
                offset_distance=3000.0,
            ),
            diffusion=DiffusionConfig(
                D=100.0,
                k_max=10.0,
                Kd=1.0,
                secretion_rate=1.0,
                active_threshold=0.2,
            ),
            expression=ExpressionConfig(
                F_basal=0.1,
                F_high=10.0,
                R_basal=0.1,
                fold_change=10.0,
                sigma_f=0.1,
                sigma_r=0.1,
                sigma_f_basal=0.1,
                factor_model="stochastic_ref",
                response_model="stochastic_hill",
            ),
            stochastic=StochasticConfig(
                p_sender_express=0.9,
                sigma_f_basal=0.1,
                p_respond_max=1.0,
                response_hill_coef=1.0,
                sigma_r_b=0.1,
            ),
            analysis=AnalysisConfig(
                bandwidth=20.0, ind_methods=["ring"], radii=list(np.arange(10, 5001, 25))
            ),
        )

    @staticmethod
    def demo_det() -> SimulationConfig:
        """demo_det.py preset - Deterministic with silent senders zeroed."""
        return SimulationConfig(
            domain=DomainConfig(center=(2500.0, 2500.0)),
            cell_types=CellTypeConfig(
                n_active_senders=20,
                n_silent_senders=20,
                receiver_fractions=[0.10, 0.20, 0.30, 0.40, 0.50],
                silent_expr_zero=True,
            ),
            position=PositionConfig(mode=SenderPositionMode.CENTER),
            diffusion=DiffusionConfig(D=1060.0, k_max=300.0, Kd=5.0, active_threshold=1.0),
            expression=ExpressionConfig(
                F_basal=0.1,
                F_high=2.0,
                R_basal=2.0,
                fold_change=2.0,
                sigma_f=0.1,
                sigma_r=0.1,
                factor_model="deterministic",
                response_model="deterministic",
            ),
            analysis=AnalysisConfig(
                bandwidth=100.0, ind_methods=["ring"], radii=list(np.arange(50, 5001, 25))
            ),
        )

    @staticmethod
    def demo_det_dec() -> SimulationConfig:
        """demo_det_dec.py preset - Deterministic decoy with distributed silent."""
        return SimulationConfig(
            domain=DomainConfig(center=(2500.0, 2500.0)),
            cell_types=CellTypeConfig(
                n_active_senders=20,
                n_silent_senders=200,
                receiver_fractions=[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
                silent_expr_zero=True,
            ),
            position=PositionConfig(mode=SenderPositionMode.CENTER_SILENT_DISTRIBUTED),
            diffusion=DiffusionConfig(D=1060.0, k_max=300.0, Kd=5.0, active_threshold=1.0),
            expression=ExpressionConfig(
                F_basal=0.1,
                F_high=2.0,
                R_basal=2.0,
                fold_change=2.0,
                sigma_f=0.1,
                sigma_r=0.1,
                factor_model="deterministic",
                response_model="deterministic",
            ),
            analysis=AnalysisConfig(
                bandwidth=100.0,
                ind_methods=["ring"],
                radii=list(np.arange(50, 5001, 25)),
            ),
        )

    @staticmethod
    def demo_vst() -> SimulationConfig:
        """demo_vst.py preset - VST normalized with zero inflation."""
        return SimulationConfig(
            domain=DomainConfig(center=(0.0, 0.0)),
            cell_types=CellTypeConfig(
                n_active_senders=20,
                n_silent_senders=0,
                receiver_fractions=[0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
            ),
            position=PositionConfig(mode=SenderPositionMode.CENTER),
            diffusion=DiffusionConfig(D=100.0, k_max=10.0, Kd=30.0, active_threshold=1.0),
            expression=ExpressionConfig(
                F_basal=0.5,
                F_high=50.0,
                R_basal=0.5,
                fold_change=5.0,
                sigma_f=0.8,
                sigma_r=0.8,
                factor_model="deterministic",
                response_model="deterministic",
                vst_method="log1p",
                vst_active_threshold=5.0,
            ),
            stochastic=StochasticConfig(
                zero_inflate_factor=0.7,
                zero_inflate_response=0.5,
            ),
            analysis=AnalysisConfig(
                bandwidth=100.0, ind_methods=["ring"], radii=list(np.arange(50, 5001, 25))
            ),
        )

    # =========================================================================
    # 14 Generic Presets (existing)
    # =========================================================================

    @staticmethod
    def default() -> SimulationConfig:
        """Default balanced configuration."""
        return SimulationConfig()

    @staticmethod
    def strong_signal() -> SimulationConfig:
        """Configuration with strong, clear spatial signal."""
        config = SimulationConfig()
        config.expression.F_high = 50.0
        config.expression.fold_change = 20.0
        config.stochastic.p_sender_express = 1.0
        config.stochastic.p_receiver_respond_max = 1.0
        return config

    @staticmethod
    def weak_signal() -> SimulationConfig:
        """Configuration with weak spatial signal."""
        config = SimulationConfig()
        config.expression.F_high = 2.0
        config.expression.fold_change = 2.0
        config.stochastic.p_sender_express = 0.5
        config.stochastic.p_receiver_respond_max = 0.5
        return config

    @staticmethod
    def high_noise() -> SimulationConfig:
        """Configuration with high expression noise."""
        config = SimulationConfig()
        config.expression.sigma_f = 0.8
        config.expression.sigma_f_basal = 0.8
        config.expression.sigma_r = 0.8
        config.expression.sigma_r_basal = 0.8
        config.stochastic.zero_inflate_factor = 0.3
        config.stochastic.zero_inflate_response = 0.3
        return config

    @staticmethod
    def low_noise() -> SimulationConfig:
        """Configuration with minimal expression noise."""
        config = SimulationConfig()
        config.expression.sigma_f = 0.01
        config.expression.sigma_f_basal = 0.01
        config.expression.sigma_r = 0.01
        config.expression.sigma_r_basal = 0.01
        return config

    @staticmethod
    def large_scale() -> SimulationConfig:
        """Large-scale simulation (500k cells)."""
        config = SimulationConfig()
        config.domain.n_cells = 500000
        config.domain.max_radius = 10000.0
        config.cell_types.n_active_senders = 100
        return config

    @staticmethod
    def small_scale() -> SimulationConfig:
        """Small-scale simulation for quick testing."""
        config = SimulationConfig()
        config.domain.n_cells = 5000
        config.domain.max_radius = 1000.0
        config.cell_types.n_active_senders = 10
        config.cell_types.receiver_fractions = [0.2, 0.4]
        config.analysis.radii = list(range(50, 1001, 100))
        return config

    @staticmethod
    def long_range() -> SimulationConfig:
        """Configuration for long-range signaling."""
        config = SimulationConfig()
        config.diffusion.D = 500.0
        config.diffusion.k_max = 1.0
        config.domain.max_radius = 10000.0
        return config

    @staticmethod
    def short_range() -> SimulationConfig:
        """Configuration for short-range signaling."""
        config = SimulationConfig()
        config.diffusion.D = 10.0
        config.diffusion.k_max = 50.0
        config.analysis.bandwidth = 50.0
        config.analysis.radii = list(range(10, 501, 10))
        return config

    @staticmethod
    def stochastic_full() -> SimulationConfig:
        """Full stochastic model with all features enabled."""
        config = SimulationConfig()
        config.position.mode = SenderPositionMode.RANDOM
        config.position.n_positions = 1
        config.cell_types.n_active_senders = 200
        config.cell_types.n_silent_senders = 100
        config.stochastic.p_sender_express = 0.9
        config.stochastic.p_receiver_respond_max = 1.0
        config.stochastic.hill_coefficient = 1.0
        config.analysis.bandwidth = 20.0
        return config

    @staticmethod
    def multi_position() -> SimulationConfig:
        """Senders at multiple random positions."""
        config = SimulationConfig()
        config.position.mode = SenderPositionMode.RANDOM
        config.position.n_positions = 5
        config.position.min_separation = 500.0
        config.cell_types.n_active_senders = 100
        return config

    @staticmethod
    def fixed_5_positions() -> SimulationConfig:
        """Senders at 5 fixed positions (Center, N, S, E, W)."""
        config = SimulationConfig()
        config.position.mode = SenderPositionMode.FIXED_5
        config.position.offset_distance = 3000.0
        config.cell_types.n_active_senders = 100
        return config

    @staticmethod
    def annular_weights() -> SimulationConfig:
        """Configuration using annular (donut) weights."""
        config = SimulationConfig()
        config.analysis.weight_type = WeightType.ANNULAR
        config.analysis.annular_width = 50.0
        return config

    @staticmethod
    def gaussian_weights() -> SimulationConfig:
        """Configuration using Gaussian distance decay weights."""
        config = SimulationConfig()
        config.analysis.weight_type = WeightType.GAUSSIAN
        config.analysis.bandwidth = 100.0
        return config

    # =========================================================================
    # Lookup Methods
    # =========================================================================

    @classmethod
    def get_preset(cls, name: str) -> SimulationConfig:
        """Get a preset configuration by name."""
        presets = {
            # 13 demo presets
            "demo": cls.demo,
            "demo1": cls.demo1,
            "demo1b": cls.demo1b,
            "demo1_stc": cls.demo1_stc,
            "demo1_stc2": cls.demo1_stc2,
            "demo2a": cls.demo2a,
            "demo2b": cls.demo2b,
            "demo2c": cls.demo2c,
            "demo3": cls.demo3,
            "demo3b": cls.demo3b,
            "demo_det": cls.demo_det,
            "demo_det_dec": cls.demo_det_dec,
            "demo_vst": cls.demo_vst,
            # 14 generic presets
            "default": cls.default,
            "strong_signal": cls.strong_signal,
            "weak_signal": cls.weak_signal,
            "high_noise": cls.high_noise,
            "low_noise": cls.low_noise,
            "large_scale": cls.large_scale,
            "small_scale": cls.small_scale,
            "long_range": cls.long_range,
            "short_range": cls.short_range,
            "stochastic_full": cls.stochastic_full,
            "multi_position": cls.multi_position,
            "fixed_5_positions": cls.fixed_5_positions,
            "annular_weights": cls.annular_weights,
            "gaussian_weights": cls.gaussian_weights,
        }

        if name not in presets:
            available = ", ".join(sorted(presets.keys()))
            raise ValueError(f"Unknown preset: '{name}'. Available: {available}")

        return presets[name]()

    @classmethod
    def list_presets(cls) -> list:
        """List all available preset names."""
        return [
            "demo",
            "demo1",
            "demo1b",
            "demo1_stc",
            "demo1_stc2",
            "demo2a",
            "demo2b",
            "demo2c",
            "demo3",
            "demo3b",
            "demo_det",
            "demo_det_dec",
            "demo_vst",
            "default",
            "strong_signal",
            "weak_signal",
            "high_noise",
            "low_noise",
            "large_scale",
            "small_scale",
            "long_range",
            "short_range",
            "stochastic_full",
            "multi_position",
            "fixed_5_positions",
            "annular_weights",
            "gaussian_weights",
        ]
