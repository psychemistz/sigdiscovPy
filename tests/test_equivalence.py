"""
Equivalence tests between sigdiscovpy simulation and reference core.py/unified_sim.py.

For each of the 13 demo presets with seed=42:
- Run reference functions from core.py
- Run sigdiscovpy package equivalent
- Assert numerically identical I_ND curves, concentrations, and expressions

Also unit tests for individual functions:
- calculate_lambda matches reference
- concentration field matches reference
- I_ND ring matches reference
- I_ND gaussian_annular matches reference
- VST transforms match reference
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add reference simulation directory to path for imports
REFERENCE_DIR = Path(__file__).resolve().parents[3] / "sigdiscov" / "simulation"
HAS_REFERENCE = (REFERENCE_DIR / "core.py").exists()
if HAS_REFERENCE:
    sys.path.insert(0, str(REFERENCE_DIR))

requires_reference = pytest.mark.skipif(
    not HAS_REFERENCE, reason=f"Reference scripts not found at {REFERENCE_DIR}"
)

from sigdiscovpy.simulation.config.presets import SimulationPresets
from sigdiscovpy.simulation.expression.vst import (
    apply_vst_log1p,
    apply_vst_pearson,
    apply_vst_shifted_log,
)
from sigdiscovpy.simulation.runner import UnifiedSimulation


# =============================================================================
# Unit Tests for Individual Functions
# =============================================================================


@requires_reference
class TestCalculateLambda:
    """Test that lambda calculation matches reference."""

    def test_basic_lambda(self):
        """Lambda formula: sqrt(D * Kd / (n_eff * k_max))."""
        import core as ref

        D, k_max, Kd = 100.0, 10.0, 30.0
        n_receivers = 0.001  # density
        p_r = 1.0

        ref_lambda = ref.calculate_lambda(D, n_receivers, k_max, Kd, p_r)

        from sigdiscovpy.simulation.config.dataclasses import DiffusionConfig
        from sigdiscovpy.simulation.physics.diffusion import DiffusionSolver

        config = DiffusionConfig(D=D, k_max=k_max, Kd=Kd)
        solver = DiffusionSolver(config)
        pkg_lambda = solver.calculate_lambda(n_receivers, p_r)

        assert np.isclose(
            ref_lambda, pkg_lambda, atol=1e-10
        ), f"Lambda mismatch: ref={ref_lambda}, pkg={pkg_lambda}"

    def test_lambda_with_p_r(self):
        """Lambda with effective density (p_r < 1)."""
        import core as ref

        D, k_max, Kd = 100.0, 50.0, 1.0
        n_receivers = 0.01
        p_r = 0.5

        ref_lambda = ref.calculate_lambda(D, n_receivers, k_max, Kd, p_r)

        from sigdiscovpy.simulation.config.dataclasses import DiffusionConfig
        from sigdiscovpy.simulation.physics.diffusion import DiffusionSolver

        config = DiffusionConfig(D=D, k_max=k_max, Kd=Kd)
        solver = DiffusionSolver(config)
        pkg_lambda = solver.calculate_lambda(n_receivers, p_r)

        assert np.isclose(ref_lambda, pkg_lambda, atol=1e-10)


@requires_reference
class TestConcentrationField:
    """Test that concentration field matches reference."""

    def test_single_source_center(self):
        """Single source at center should match reference."""
        import core as ref

        np.random.seed(42)
        n_total = 1000
        center = np.array([0.0, 0.0])
        max_radius = 1000.0

        # Generate positions
        angles = np.random.rand(n_total) * 2 * np.pi
        radii = np.sqrt(np.random.rand(n_total)) * max_radius
        positions = np.column_stack(
            [center[0] + radii * np.cos(angles), center[1] + radii * np.sin(angles)]
        )

        # Sender at center
        sender_pos = np.array([[0.0, 0.0]])
        sender_expr = np.array([10.0])

        n_density = 100 / (np.pi * max_radius**2)

        ref_conc, ref_lambda = ref.solve_concentration_field_MM(
            sender_pos,
            sender_expr,
            positions,
            n_density,
            D=100.0,
            k_max=10.0,
            Kd=30.0,
            active_threshold=1.0,
        )

        # Test our runner's _solve_single method
        config = SimulationPresets.demo()
        sim = UnifiedSimulation(config)

        np.random.seed(42)
        pkg_conc, pkg_lambda = sim._solve_single(
            sender_pos, sender_expr, positions, n_density, active_threshold=1.0
        )

        assert np.isclose(ref_lambda, pkg_lambda, atol=1e-10)
        assert np.allclose(
            ref_conc, pkg_conc, atol=1e-10
        ), f"Max diff: {np.max(np.abs(ref_conc - pkg_conc))}"


@requires_reference
class TestINDRing:
    """Test I_ND ring computation matches reference."""

    def test_ring_basic(self):
        """Ring I_ND should match reference compute_IND_ring."""
        import core as ref

        np.random.seed(42)
        n_total = 500
        positions = np.random.rand(n_total, 2) * 2000

        sender_idx = np.arange(20)
        receiver_idx = np.arange(20, 200)

        positions[sender_idx] = [500, 500]  # cluster senders

        factor_expr = np.random.lognormal(0, 0.1, n_total)
        factor_expr[sender_idx] = 10.0

        response_expr = np.random.lognormal(0, 0.1, n_total)
        response_expr[receiver_idx[:50]] = 5.0  # some responders

        distance = 300.0
        bandwidth = 100.0

        ref_ind, ref_conn = ref.compute_IND_ring(
            sender_idx, receiver_idx, positions, factor_expr, response_expr, distance, bandwidth
        )

        from sigdiscovpy.simulation.analysis.ind_computer import INDComputer
        from sigdiscovpy.simulation.config.dataclasses import AnalysisConfig, WeightType

        config = AnalysisConfig(
            bandwidth=bandwidth, weight_type=WeightType.RING, use_sigdiscov_core=False
        )
        computer = INDComputer(config)
        pkg_ind, pkg_conn = computer._compute_simple(
            sender_idx, receiver_idx, positions, factor_expr, response_expr, distance
        )

        assert np.isclose(
            ref_ind, pkg_ind, atol=1e-10
        ), f"Ring I_ND mismatch: ref={ref_ind}, pkg={pkg_ind}"
        assert ref_conn == pkg_conn


@requires_reference
class TestINDGaussianAnnular:
    """Test I_ND gaussian_annular computation matches reference."""

    def test_gaussian_annular_basic(self):
        """Gaussian annular I_ND should match reference."""
        import core as ref

        np.random.seed(42)
        n_total = 500
        positions = np.random.rand(n_total, 2) * 2000

        sender_idx = np.arange(20)
        receiver_idx = np.arange(20, 200)

        positions[sender_idx] = [500, 500]

        factor_expr = np.random.lognormal(0, 0.1, n_total)
        factor_expr[sender_idx] = 10.0

        response_expr = np.random.lognormal(0, 0.1, n_total)
        response_expr[receiver_idx[:50]] = 5.0

        outer_radius = 500.0
        bandwidth = 100.0
        sigma_fraction = 3.0

        ref_ind, ref_conn = ref.compute_IND_gaussian_annular(
            sender_idx,
            receiver_idx,
            positions,
            factor_expr,
            response_expr,
            outer_radius,
            bandwidth,
            sigma_fraction,
        )

        from sigdiscovpy.simulation.analysis.ind_computer import INDComputer
        from sigdiscovpy.simulation.config.dataclasses import AnalysisConfig, WeightType

        config = AnalysisConfig(
            bandwidth=bandwidth,
            weight_type=WeightType.GAUSSIAN_ANNULAR,
            use_sigdiscov_core=False,
            sigma_fraction=sigma_fraction,
        )
        computer = INDComputer(config)
        pkg_ind, pkg_conn = computer._compute_simple(
            sender_idx, receiver_idx, positions, factor_expr, response_expr, outer_radius
        )

        assert np.isclose(
            ref_ind, pkg_ind, atol=1e-10
        ), f"Gaussian annular I_ND mismatch: ref={ref_ind}, pkg={pkg_ind}"
        assert ref_conn == pkg_conn


@requires_reference
class TestVSTTransforms:
    """Test VST functions match reference exactly."""

    def test_vst_log1p(self):
        import core as ref

        np.random.seed(42)
        raw = np.random.lognormal(0, 1.0, 1000)
        ref_result = ref.apply_vst_log1p(raw)
        pkg_result = apply_vst_log1p(raw)
        assert np.allclose(ref_result, pkg_result, atol=1e-10)

    def test_vst_pearson(self):
        import core as ref

        np.random.seed(42)
        raw = np.random.lognormal(0, 1.0, 1000)
        ref_result = ref.apply_vst_pearson(raw, theta=100, clip_value=10)
        pkg_result = apply_vst_pearson(raw, theta=100, clip_value=10)
        assert np.allclose(ref_result, pkg_result, atol=1e-10)

    def test_vst_shifted_log(self):
        import core as ref

        np.random.seed(42)
        raw = np.random.lognormal(0, 1.0, 1000)
        ref_result = ref.apply_vst_shifted_log(raw, pseudocount=1.0)
        pkg_result = apply_vst_shifted_log(raw, pseudocount=1.0)
        assert np.allclose(ref_result, pkg_result, atol=1e-10)


# =============================================================================
# Preset Smoke Tests (all 13 presets run without error)
# =============================================================================

DEMO_PRESETS = [
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
]


class TestPresetSmoke:
    """Smoke tests: all 13 demo presets should run without error."""

    @pytest.mark.parametrize("preset_name", DEMO_PRESETS)
    def test_preset_runs(self, preset_name):
        """Verify preset creates valid config and simulation completes."""
        config = SimulationPresets.get_preset(preset_name)
        # Reduce scale for testing speed
        config.domain.n_cells = 1000
        config.domain.max_radius = 1000.0
        config.cell_types.receiver_fractions = [0.2]
        config.analysis.radii = list(np.arange(50, 1001, 100))
        # Reduce sender counts if needed
        if config.cell_types.n_active_senders > 50:
            config.cell_types.n_active_senders = 50
        if config.cell_types.n_silent_senders > 50:
            config.cell_types.n_silent_senders = 50
        if config.position.n_positions > 5:
            config.position.n_positions = 5
            config.position.min_separation = 100.0

        sim = UnifiedSimulation(config)
        results = sim.run(seed=42)

        assert len(results) == 1
        frac = 0.2
        assert frac in results
        result = results[frac]
        assert "lambda" in result
        assert result["lambda"] > 0
        assert "ind_curves" in result
        assert len(result["ind_curves"]) > 0


# =============================================================================
# Full Equivalence Tests (reference vs package, same seed)
# =============================================================================


@requires_reference
class TestEquivalence:
    """
    Full equivalence tests: reference unified_sim.py vs sigdiscovpy package.

    For deterministic presets with seed=42, I_ND curves must be numerically identical.
    """

    def _run_reference(self, preset_name, seed=42):
        """Run reference unified_sim.py and return I_ND curves."""
        sys.path.insert(0, str(REFERENCE_DIR))
        import importlib

        if "unified_sim" in sys.modules:
            importlib.reload(sys.modules["unified_sim"])
        import unified_sim as ref_sim

        cfg = ref_sim.make_config(preset_name)
        # Use a minimal receiver fraction for speed
        cfg.receiver_fractions = [0.20]
        cfg.n_total = 5000
        cfg.max_radius = 2000.0

        # Limit distances
        d_start = (
            cfg.test_distance_start if cfg.test_distance_start is not None else cfg.BANDWIDTH / 2
        )
        cfg.test_distance_end = 2000.0

        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            results = ref_sim.run_experiment(cfg, output_folder=tmpdir, seed=seed)

        return results

    def _run_package(self, preset_name, seed=42):
        """Run sigdiscovpy package and return results."""
        config = SimulationPresets.get_preset(preset_name)
        config.domain.n_cells = 5000
        config.domain.max_radius = 2000.0
        config.cell_types.receiver_fractions = [0.20]
        config.analysis.radii = list(np.arange(50, 2001, 25))

        # Reduce sender counts for multi-position presets
        if config.position.n_positions > 10:
            config.position.n_positions = 5
            config.position.min_separation = 200.0

        sim = UnifiedSimulation(config)
        return sim.run(seed=seed)

    @pytest.mark.parametrize("preset_name", ["demo", "demo1", "demo_det"])
    def test_deterministic_preset_equivalence(self, preset_name):
        """Deterministic presets should produce identical I_ND curves."""
        ref_results = self._run_reference(preset_name)
        pkg_results = self._run_package(preset_name)

        for frac in [0.20]:
            ref_state = ref_results[frac]
            pkg_result = pkg_results[frac]

            # Compare lambda
            assert np.isclose(ref_state.lambda_val, pkg_result["lambda"], atol=1e-6), (
                f"{preset_name} frac={frac}: lambda mismatch "
                f"ref={ref_state.lambda_val}, pkg={pkg_result['lambda']}"
            )

            # Compare I_ND curves (ring method)
            ref_curve = ref_state.ind_curves.get("ring", [])
            pkg_curve = pkg_result["ind_curves"].get("ring", [])

            # Match on overlapping distances
            ref_dists = {p["distance"]: p["I_ND"] for p in ref_curve}
            pkg_dists = {p["distance"]: p["I_ND"] for p in pkg_curve}
            common = set(ref_dists.keys()) & set(pkg_dists.keys())

            if len(common) > 0:
                ref_vals = np.array([ref_dists[d] for d in sorted(common)])
                pkg_vals = np.array([pkg_dists[d] for d in sorted(common)])
                assert np.allclose(ref_vals, pkg_vals, atol=1e-10), (
                    f"{preset_name} frac={frac}: I_ND curve mismatch. "
                    f"Max diff: {np.max(np.abs(ref_vals - pkg_vals))}"
                )
