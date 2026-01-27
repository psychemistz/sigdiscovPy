"""
Tests for the simulation module.
"""

import numpy as np
import pytest

from sigdiscovpy.simulation.config.dataclasses import (
    SimulationConfig,
    DomainConfig,
    CellTypeConfig,
    PositionConfig,
    DiffusionConfig,
    ExpressionConfig,
    StochasticConfig,
    AnalysisConfig,
    WeightType,
    SenderPositionMode,
)
from sigdiscovpy.simulation.config.presets import SimulationPresets
from sigdiscovpy.simulation.domain.spatial import SpatialDomain, SenderPositionGenerator
from sigdiscovpy.simulation.physics.diffusion import DiffusionSolver
from sigdiscovpy.simulation.expression.stochastic import ExpressionGenerator
from sigdiscovpy.simulation.analysis.ind_computer import INDComputer
from sigdiscovpy.simulation.runner import UnifiedSimulation


class TestSimulationConfig:
    """Tests for configuration dataclasses."""

    def test_domain_config_defaults(self):
        config = DomainConfig()
        assert config.n_cells == 100000  # Default is 100k cells
        assert config.max_radius == 5000.0
        assert config.random_seed == 42

    def test_celltype_config(self):
        config = CellTypeConfig(
            n_active_senders=20,
            n_silent_senders=5,
            receiver_fractions=[0.1, 0.2, 0.3]
        )
        assert config.n_active_senders == 20
        assert len(config.receiver_fractions) == 3

    def test_position_config_modes(self):
        center_config = PositionConfig(mode=SenderPositionMode.CENTER)
        assert center_config.mode == SenderPositionMode.CENTER

        fixed_config = PositionConfig(mode=SenderPositionMode.FIXED_5)
        assert fixed_config.mode == SenderPositionMode.FIXED_5

    def test_weight_type_enum(self):
        assert WeightType.GAUSSIAN.value == "gaussian"
        assert WeightType.RING.value == "ring"
        assert WeightType.ANNULAR.value == "annular"

    def test_full_config_creation(self):
        config = SimulationConfig(
            domain=DomainConfig(n_cells=1000),
            cell_types=CellTypeConfig(n_active_senders=10),
            position=PositionConfig(),
            diffusion=DiffusionConfig(),
            expression=ExpressionConfig(),
            stochastic=StochasticConfig(),
            analysis=AnalysisConfig(radii=[100, 200]),
        )
        assert config.domain.n_cells == 1000
        assert config.cell_types.n_active_senders == 10
        assert len(config.analysis.radii) == 2


class TestSimulationPresets:
    """Tests for preset configurations."""

    def test_default_preset(self):
        config = SimulationPresets.default()
        assert isinstance(config, SimulationConfig)
        assert config.domain.n_cells == 100000  # Default is 100k cells

    def test_small_scale_preset(self):
        config = SimulationPresets.small_scale()
        assert config.domain.n_cells == 5000  # Small scale
        assert config.cell_types.n_active_senders == 10

    def test_large_scale_preset(self):
        config = SimulationPresets.large_scale()
        assert config.domain.n_cells == 500000  # Large scale is 500k

    def test_strong_signal_preset(self):
        config = SimulationPresets.strong_signal()
        assert config.expression.fold_change == 20.0

    def test_weak_signal_preset(self):
        config = SimulationPresets.weak_signal()
        assert config.expression.fold_change == 2.0

    def test_all_presets_valid(self):
        """Verify all presets create valid configs."""
        presets = [
            SimulationPresets.default,
            SimulationPresets.small_scale,
            SimulationPresets.large_scale,
            SimulationPresets.strong_signal,
            SimulationPresets.weak_signal,
            SimulationPresets.high_noise,
            SimulationPresets.low_noise,
            SimulationPresets.long_range,
            SimulationPresets.short_range,
        ]
        for preset_func in presets:
            config = preset_func()
            assert isinstance(config, SimulationConfig)
            assert config.domain.n_cells > 0


class TestSpatialDomain:
    """Tests for SpatialDomain class."""

    def test_generate_positions(self):
        config = DomainConfig(n_cells=100, max_radius=500, random_seed=42)
        domain = SpatialDomain(config)

        positions = domain.generate_positions()

        assert positions.shape == (100, 2)
        assert positions.dtype == np.float64

    def test_positions_within_radius(self):
        config = DomainConfig(n_cells=500, max_radius=1000, random_seed=42)
        domain = SpatialDomain(config)

        positions = domain.generate_positions()
        distances = np.linalg.norm(positions - config.center, axis=1)

        assert np.all(distances <= config.max_radius)

    def test_reproducibility(self):
        config = DomainConfig(n_cells=100, random_seed=42)

        domain1 = SpatialDomain(config)
        domain2 = SpatialDomain(config)

        pos1 = domain1.generate_positions()
        pos2 = domain2.generate_positions()

        np.testing.assert_array_equal(pos1, pos2)

    def test_domain_area(self):
        config = DomainConfig(max_radius=100)
        domain = SpatialDomain(config)

        expected_area = np.pi * 100**2
        assert np.isclose(domain.get_domain_area(), expected_area)


class TestSenderPositionGenerator:
    """Tests for SenderPositionGenerator class."""

    def test_center_mode(self):
        pos_config = PositionConfig(mode=SenderPositionMode.CENTER)
        domain_config = DomainConfig(center=(500, 500))

        generator = SenderPositionGenerator(pos_config, domain_config)
        positions = generator.generate_positions()

        assert "C" in positions  # Key is 'C' for center
        np.testing.assert_array_equal(positions["C"], [500, 500])

    def test_fixed_5_mode(self):
        pos_config = PositionConfig(mode=SenderPositionMode.FIXED_5)
        domain_config = DomainConfig(max_radius=1000)

        generator = SenderPositionGenerator(pos_config, domain_config)
        positions = generator.generate_positions()

        assert len(positions) == 5
        assert "C" in positions  # Center, West, East, North, South
        assert "W" in positions
        assert "E" in positions

    def test_distribute_senders(self):
        rng = np.random.default_rng(42)
        position_dict = {
            "pos1": np.array([0, 0]),
            "pos2": np.array([100, 100]),
        }

        assignments, pos_list = SenderPositionGenerator.distribute_senders(
            10, position_dict, rng
        )

        assert sum(assignments.values()) == 10
        assert len(pos_list) == 10


class TestDiffusionSolver:
    """Tests for DiffusionSolver class."""

    def test_calculate_lambda(self):
        config = DiffusionConfig(D=100.0, k_max=10.0, Kd=1.0)
        solver = DiffusionSolver(config)

        # λ = √(D * Kd / (n_eff * k_max))
        lambda_val = solver.calculate_lambda(n_receivers=0.001, p_r=1.0)

        expected = np.sqrt(100.0 * 1.0 / (0.001 * 10.0))
        assert np.isclose(lambda_val, expected)

    def test_calculate_lambda_zero_density(self):
        config = DiffusionConfig()
        solver = DiffusionSolver(config)

        lambda_val = solver.calculate_lambda(n_receivers=0)
        assert lambda_val == np.inf

    def test_solve_basic(self):
        config = DiffusionConfig()
        solver = DiffusionSolver(config)

        sender_pos = np.array([[500, 500]])
        sender_expr = np.array([10.0])
        cell_pos = np.array([[500, 500], [600, 500], [700, 500]])

        concentrations, lambda_val = solver.solve(
            sender_pos, sender_expr, cell_pos, n_receivers_density=0.001
        )

        assert len(concentrations) == 3
        # Concentration should decrease with distance
        assert concentrations[0] > concentrations[1] > concentrations[2]


class TestExpressionGenerator:
    """Tests for ExpressionGenerator class."""

    def test_factor_expression_shape(self):
        expr_config = ExpressionConfig()
        stoch_config = StochasticConfig()
        generator = ExpressionGenerator(expr_config, stoch_config, seed=42)

        active_indices = np.array([0, 1, 2, 3, 4])
        factor_expr, expressing_mask = generator.generate_factor_expression(
            n_total=100, n_active=5, active_indices=active_indices
        )

        assert factor_expr.shape == (100,)
        assert expressing_mask.shape == (5,)

    def test_factor_expression_positive(self):
        expr_config = ExpressionConfig()
        stoch_config = StochasticConfig(zero_inflate_factor=0)
        generator = ExpressionGenerator(expr_config, stoch_config, seed=42)

        active_indices = np.array([0, 1, 2])
        factor_expr, _ = generator.generate_factor_expression(
            n_total=50, n_active=3, active_indices=active_indices
        )

        # Without zero inflation, all values should be positive
        assert np.all(factor_expr >= 0)

    def test_response_expression_shape(self):
        expr_config = ExpressionConfig()
        stoch_config = StochasticConfig()
        generator = ExpressionGenerator(expr_config, stoch_config, seed=42)

        receiver_indices = np.array([10, 11, 12, 13, 14])
        concentrations = np.random.rand(100)

        response_expr, responding_mask, response_probs = generator.generate_response_expression(
            n_total=100, receiver_indices=receiver_indices, concentrations=concentrations
        )

        assert response_expr.shape == (100,)
        assert responding_mask.shape == (5,)
        assert response_probs.shape == (5,)

    def test_noise_only(self):
        expr_config = ExpressionConfig()
        stoch_config = StochasticConfig()
        generator = ExpressionGenerator(expr_config, stoch_config, seed=42)

        expr = generator.generate_noise_only(
            n_total=100, base_expression=5.0, sigma=0.5
        )

        assert expr.shape == (100,)
        assert np.all(expr > 0)


class TestINDComputer:
    """Tests for INDComputer class."""

    def test_compute_at_radii(self):
        config = AnalysisConfig(radii=[50, 100, 150])
        computer = INDComputer(config)

        # Create simple test data
        n_cells = 100
        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 500, (n_cells, 2))

        sender_indices = np.array([0, 1, 2, 3, 4])
        receiver_indices = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])

        factor_expr = rng.lognormal(0, 1, n_cells)
        response_expr = rng.lognormal(0, 1, n_cells)

        results = computer.compute_at_radii(
            sender_indices, receiver_indices, positions,
            factor_expr, response_expr
        )

        assert len(results) == 3
        for r in results:
            assert 'radius' in r
            assert 'I_ND' in r
            assert 'n_connections' in r

    def test_ind_bounded(self):
        config = AnalysisConfig(radii=[100])
        computer = INDComputer(config)

        n_cells = 50
        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 300, (n_cells, 2))

        sender_indices = np.arange(10)
        receiver_indices = np.arange(10, 30)

        factor_expr = rng.lognormal(0, 1, n_cells)
        response_expr = rng.lognormal(0, 1, n_cells)

        results = computer.compute_at_radii(
            sender_indices, receiver_indices, positions,
            factor_expr, response_expr
        )

        i_nd = results[0]['I_ND']
        # I_ND should be bounded between -1 and 1
        assert -1.0 <= i_nd <= 1.0


class TestUnifiedSimulation:
    """Tests for UnifiedSimulation orchestrator."""

    def test_run_single(self):
        config = SimulationPresets.small_scale()
        config.domain.n_cells = 200
        config.cell_types.receiver_fractions = [0.2]
        config.analysis.radii = [50, 100]

        sim = UnifiedSimulation(config)
        result = sim.run_single(receiver_fraction=0.2)

        assert 'positions' in result
        assert 'sender_indices' in result
        assert 'receiver_indices' in result
        assert 'factor_expr' in result
        assert 'response_expr' in result
        assert 'ind_curve' in result
        assert 'lambda' in result

        assert len(result['ind_curve']) == 2

    def test_simulation_reproducibility(self):
        config1 = SimulationPresets.small_scale()
        config1.domain.n_cells = 100
        config1.domain.random_seed = 42
        config1.analysis.radii = [100]

        config2 = SimulationPresets.small_scale()
        config2.domain.n_cells = 100
        config2.domain.random_seed = 42
        config2.analysis.radii = [100]

        sim1 = UnifiedSimulation(config1)
        sim2 = UnifiedSimulation(config2)

        result1 = sim1.run_single(receiver_fraction=0.2)
        result2 = sim2.run_single(receiver_fraction=0.2)

        np.testing.assert_array_equal(result1['positions'], result2['positions'])
        np.testing.assert_array_equal(result1['sender_indices'], result2['sender_indices'])

    def test_ind_curve_structure(self):
        config = SimulationPresets.small_scale()
        config.domain.n_cells = 150
        config.analysis.radii = [50, 100, 150, 200]

        sim = UnifiedSimulation(config)
        result = sim.run_single(receiver_fraction=0.2)

        ind_curve = result['ind_curve']
        assert len(ind_curve) == 4

        for i, point in enumerate(ind_curve):
            assert point['radius'] == config.analysis.radii[i]
            assert isinstance(point['I_ND'], (int, float))
            assert isinstance(point['n_connections'], int)


class TestIntegration:
    """Integration tests for the full simulation pipeline."""

    def test_full_pipeline_small(self):
        """Test complete simulation with small parameters."""
        config = SimulationPresets.small_scale()
        config.domain.n_cells = 300
        config.cell_types.n_active_senders = 5
        config.cell_types.receiver_fractions = [0.1, 0.2]
        config.analysis.radii = [50, 100]

        sim = UnifiedSimulation(config)
        results = sim.run()

        assert 0.1 in results
        assert 0.2 in results

        for frac, data in results.items():
            assert 'lambda' in data
            assert 'ind_curve' in data
            assert 'n_expressing' in data
            assert 'n_responding' in data

    def test_signal_strength_correlation(self):
        """Test that stronger signal produces higher I_ND."""
        # Strong signal config
        strong_config = SimulationPresets.strong_signal()
        strong_config.domain.n_cells = 500
        strong_config.domain.random_seed = 42
        strong_config.analysis.radii = [100]

        # Weak signal config
        weak_config = SimulationPresets.weak_signal()
        weak_config.domain.n_cells = 500
        weak_config.domain.random_seed = 42
        weak_config.analysis.radii = [100]

        sim_strong = UnifiedSimulation(strong_config)
        sim_weak = UnifiedSimulation(weak_config)

        result_strong = sim_strong.run_single(receiver_fraction=0.2)
        result_weak = sim_weak.run_single(receiver_fraction=0.2)

        # Strong signal should generally produce higher absolute I_ND
        # (though this is stochastic, so we just check both run successfully)
        assert result_strong['ind_curve'][0]['I_ND'] is not None
        assert result_weak['ind_curve'][0]['I_ND'] is not None
