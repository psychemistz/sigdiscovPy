"""
Preset configurations for common simulation scenarios.

Each preset provides a complete SimulationConfig optimized for a specific
use case, from basic validation to realistic biological scenarios.
"""

from sigdiscovpy.simulation.config.dataclasses import (
    SimulationConfig,
    SenderPositionMode,
    WeightType,
)


class SimulationPresets:
    """
    Factory class for preset simulation configurations.

    Example
    -------
    >>> config = SimulationPresets.default()
    >>> config = SimulationPresets.strong_signal()
    >>> config = SimulationPresets.get_preset("large_scale")
    """

    @staticmethod
    def default() -> SimulationConfig:
        """
        Default balanced configuration.

        Good starting point for most analyses:
        - 100k cells
        - Moderate signal strength
        - Center-positioned senders
        """
        return SimulationConfig()

    @staticmethod
    def strong_signal() -> SimulationConfig:
        """
        Configuration with strong, clear spatial signal.

        Use for:
        - Method validation
        - Testing detection sensitivity
        - Demonstrating expected behavior
        """
        config = SimulationConfig()
        config.expression.F_high = 50.0
        config.expression.fold_change = 20.0
        config.stochastic.p_sender_express = 1.0
        config.stochastic.p_receiver_respond_max = 1.0
        return config

    @staticmethod
    def weak_signal() -> SimulationConfig:
        """
        Configuration with weak spatial signal.

        Use for:
        - Testing method sensitivity
        - Determining detection limits
        """
        config = SimulationConfig()
        config.expression.F_high = 2.0
        config.expression.fold_change = 2.0
        config.stochastic.p_sender_express = 0.5
        config.stochastic.p_receiver_respond_max = 0.5
        return config

    @staticmethod
    def high_noise() -> SimulationConfig:
        """
        Configuration with high expression noise.

        Use for:
        - Testing robustness to noise
        - Realistic scRNA-seq conditions
        """
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
        """
        Configuration with minimal expression noise.

        Use for:
        - Algorithm validation
        - Theoretical comparisons
        """
        config = SimulationConfig()
        config.expression.sigma_f = 0.01
        config.expression.sigma_f_basal = 0.01
        config.expression.sigma_r = 0.01
        config.expression.sigma_r_basal = 0.01
        return config

    @staticmethod
    def large_scale() -> SimulationConfig:
        """
        Large-scale simulation (500k cells).

        Use for:
        - Scalability testing
        - Realistic tissue-scale analysis
        """
        config = SimulationConfig()
        config.domain.n_cells = 500000
        config.domain.max_radius = 10000.0
        config.cell_types.n_active_senders = 100
        return config

    @staticmethod
    def small_scale() -> SimulationConfig:
        """
        Small-scale simulation for quick testing.

        Use for:
        - Unit tests
        - Quick prototyping
        - Debugging
        """
        config = SimulationConfig()
        config.domain.n_cells = 5000
        config.domain.max_radius = 1000.0
        config.cell_types.n_active_senders = 10
        config.cell_types.receiver_fractions = [0.2, 0.4]
        config.analysis.radii = list(range(50, 1001, 100))
        return config

    @staticmethod
    def long_range() -> SimulationConfig:
        """
        Configuration for long-range signaling.

        Use for:
        - Testing diffusible factors
        - Morphogen gradient analysis
        """
        config = SimulationConfig()
        config.diffusion.D = 500.0
        config.diffusion.k_max = 1.0
        config.domain.max_radius = 10000.0
        return config

    @staticmethod
    def short_range() -> SimulationConfig:
        """
        Configuration for short-range signaling.

        Use for:
        - Contact-dependent signaling
        - Juxtacrine interactions
        """
        config = SimulationConfig()
        config.diffusion.D = 10.0
        config.diffusion.k_max = 50.0
        config.analysis.bandwidth = 50.0
        config.analysis.radii = list(range(10, 501, 10))
        return config

    @staticmethod
    def stochastic_full() -> SimulationConfig:
        """
        Full stochastic model with all features enabled.

        Use for:
        - Realistic biological simulations
        - Model validation
        """
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
        """
        Senders at multiple random positions.

        Use for:
        - Testing spatial heterogeneity
        - Multiple source analysis
        """
        config = SimulationConfig()
        config.position.mode = SenderPositionMode.RANDOM
        config.position.n_positions = 5
        config.position.min_separation = 500.0
        config.cell_types.n_active_senders = 100
        return config

    @staticmethod
    def fixed_5_positions() -> SimulationConfig:
        """
        Senders at 5 fixed positions (Center, N, S, E, W).

        Use for:
        - Reproducible spatial patterns
        - Comparing across conditions
        """
        config = SimulationConfig()
        config.position.mode = SenderPositionMode.FIXED_5
        config.position.offset_distance = 3000.0
        config.cell_types.n_active_senders = 100
        return config

    @staticmethod
    def annular_weights() -> SimulationConfig:
        """
        Configuration using annular (donut) weights.

        Use for:
        - Distance-specific analysis
        - Multi-radius profiling
        """
        config = SimulationConfig()
        config.analysis.weight_type = WeightType.ANNULAR
        config.analysis.annular_width = 50.0
        return config

    @staticmethod
    def gaussian_weights() -> SimulationConfig:
        """
        Configuration using Gaussian distance decay weights.

        Use for:
        - Smooth spatial weighting
        - Standard spatial autocorrelation
        """
        config = SimulationConfig()
        config.analysis.weight_type = WeightType.GAUSSIAN
        config.analysis.bandwidth = 100.0
        return config

    @classmethod
    def get_preset(cls, name: str) -> SimulationConfig:
        """
        Get a preset configuration by name.

        Parameters
        ----------
        name : str
            Name of the preset.

        Returns
        -------
        SimulationConfig
            The preset configuration.

        Raises
        ------
        ValueError
            If the preset name is not recognized.
        """
        presets = {
            'default': cls.default,
            'strong_signal': cls.strong_signal,
            'weak_signal': cls.weak_signal,
            'high_noise': cls.high_noise,
            'low_noise': cls.low_noise,
            'large_scale': cls.large_scale,
            'small_scale': cls.small_scale,
            'long_range': cls.long_range,
            'short_range': cls.short_range,
            'stochastic_full': cls.stochastic_full,
            'multi_position': cls.multi_position,
            'fixed_5_positions': cls.fixed_5_positions,
            'annular_weights': cls.annular_weights,
            'gaussian_weights': cls.gaussian_weights,
        }

        if name not in presets:
            available = ', '.join(sorted(presets.keys()))
            raise ValueError(f"Unknown preset: '{name}'. Available: {available}")

        return presets[name]()

    @classmethod
    def list_presets(cls) -> list:
        """List all available preset names."""
        return [
            'default',
            'strong_signal',
            'weak_signal',
            'high_noise',
            'low_noise',
            'large_scale',
            'small_scale',
            'long_range',
            'short_range',
            'stochastic_full',
            'multi_position',
            'fixed_5_positions',
            'annular_weights',
            'gaussian_weights',
        ]
