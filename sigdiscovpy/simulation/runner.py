"""
UnifiedSimulation - Main orchestrator for spatial simulation.

Coordinates all simulation components to generate synthetic spatial
transcriptomics data with known ground truth for method validation.
Matches reference unified_sim.py run_experiment() logic exactly.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sigdiscovpy.simulation.analysis.ind_computer import INDComputer
from sigdiscovpy.simulation.config.dataclasses import (
    SenderPositionMode,
    SimulationConfig,
    WeightType,
)
from sigdiscovpy.simulation.domain.spatial import (
    SenderPositionGenerator,
    SpatialDomain,
)
from sigdiscovpy.simulation.expression.stochastic import ExpressionGenerator
from sigdiscovpy.simulation.expression.vst import apply_vst
from sigdiscovpy.simulation.physics.diffusion import DiffusionSolver

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class UnifiedSimulation:
    """
    Main orchestrator for unified spatial simulation.

    Matches reference unified_sim.py run_experiment() for numerical equivalence.

    Parameters
    ----------
    config : SimulationConfig
        Complete simulation configuration.

    Example
    -------
    >>> from sigdiscovpy.simulation import SimulationPresets
    >>> config = SimulationPresets.demo()
    >>> sim = UnifiedSimulation(config)
    >>> results = sim.run()
    """

    def __init__(self, config: SimulationConfig):
        self.config = config

        # Initialize components
        self.domain = SpatialDomain(config.domain)
        self.position_gen = SenderPositionGenerator(config.position, config.domain)
        self.diffusion = DiffusionSolver(config.diffusion)
        self.expression = ExpressionGenerator(
            config.expression, config.stochastic, seed=config.domain.random_seed,
            diffusion_Kd=config.diffusion.Kd,
        )

        # Random generator
        self._rng = np.random.default_rng(config.domain.random_seed)

    def run(self, seed: int = None) -> dict[float, dict]:
        """
        Run simulation for all receiver fractions.

        Uses np.random global state (matching reference) when seed is provided.

        Parameters
        ----------
        seed : int, optional
            Random seed. If provided, sets np.random.seed for reference compatibility.

        Returns
        -------
        Dict[float, Dict]
            Results keyed by receiver fraction.
        """
        if seed is not None:
            np.random.seed(seed)

        logger.info("=" * 70)
        logger.info("UNIFIED SIMULATION")
        logger.info("=" * 70)

        cfg = self.config
        n_total = cfg.domain.n_cells
        center = np.array(cfg.domain.center)
        max_radius = cfg.domain.max_radius
        domain_area = np.pi * max_radius ** 2
        n_senders = cfg.cell_types.n_active_senders + cfg.cell_types.n_silent_senders
        n_active = cfg.cell_types.n_active_senders

        # Compute test distances
        radii = cfg.analysis.radii

        # Generate positions (using np.random for reference compatibility)
        all_positions_orig = self._generate_positions(n_total, center, max_radius)

        # Setup multi-position senders if needed
        position_dict = None
        active_assignments = None
        sender_pos_list = None

        pos_mode = cfg.position.mode
        if pos_mode == SenderPositionMode.FIXED_5:
            position_dict = self._get_5_positions(center, cfg.position.offset_distance)
        elif pos_mode == SenderPositionMode.RANDOM:
            position_dict = self._get_random_positions(
                cfg.position.n_positions, center, max_radius, cfg.position.min_separation
            )

        if position_dict is not None:
            logger.info(f"  {len(position_dict)} sender positions:")
            for pn, coord in position_dict.items():
                logger.info(f"    {pn}: ({coord[0]:.0f}, {coord[1]:.0f})")

        # Fixed sender setup
        fixed_sender_indices = None
        fixed_active_indices = None
        fixed_silent_indices = None

        if cfg.cell_types.fix_senders_across_fractions:
            all_positions = all_positions_orig.copy()
            all_indices = np.arange(n_total)
            fixed_sender_indices = np.random.choice(all_indices, n_senders, replace=False)
            fixed_active_indices = fixed_sender_indices[:n_active]
            fixed_silent_indices = fixed_sender_indices[n_active:]

            if position_dict is not None:
                from sigdiscovpy.simulation.domain.spatial import SenderPositionGenerator as SPG
                active_assignments, sender_pos_list = self._distribute_active_senders(
                    n_active, position_dict
                )
                for i, (pn, coord) in enumerate(sender_pos_list):
                    all_positions[fixed_active_indices[i]] = coord

        # Save config if output_dir specified
        if cfg.output_dir:
            output_path = Path(cfg.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            cfg.save(str(output_path / "config.json"))

        results = {}

        for frac in cfg.cell_types.receiver_fractions:
            logger.info(f"\nProcessing {frac*100:.0f}% Receivers...")

            result = self._run_single_fraction(
                frac, n_total, center, max_radius, domain_area,
                n_senders, n_active, all_positions_orig,
                position_dict, active_assignments, sender_pos_list,
                fixed_sender_indices, fixed_active_indices, fixed_silent_indices,
                radii,
            )
            results[frac] = result

        # Save results
        if cfg.output_dir:
            self._save_results(results)

        logger.info("\nSimulation complete.")
        return results

    def _run_single_fraction(
        self, frac, n_total, center, max_radius, domain_area,
        n_senders, n_active, all_positions_orig,
        position_dict, active_assignments, sender_pos_list,
        fixed_sender_indices, fixed_active_indices, fixed_silent_indices,
        radii,
    ) -> dict[str, Any]:
        """Run simulation for a single receiver fraction, matching reference exactly."""
        cfg = self.config
        pos_mode = cfg.position.mode

        # Position setup
        if cfg.cell_types.fix_senders_across_fractions:
            if pos_mode in (SenderPositionMode.FIXED_5, SenderPositionMode.RANDOM):
                all_pos = all_positions_orig  # positions already modified for fixed mode
            else:
                all_pos = all_positions_orig.copy()
            sender_indices = fixed_sender_indices
            active_indices = fixed_active_indices
            silent_indices = fixed_silent_indices
        else:
            all_pos = all_positions_orig.copy()
            all_indices = np.arange(n_total)
            sender_indices = np.random.choice(all_indices, n_senders, replace=False)
            active_indices = sender_indices[:n_active]
            silent_indices = sender_indices[n_active:]

        # Position senders
        if pos_mode == SenderPositionMode.CENTER:
            all_pos[active_indices] = center
            all_pos[silent_indices] = center
        elif pos_mode == SenderPositionMode.CENTER_SILENT_DISTRIBUTED:
            all_pos[active_indices] = center
            # silent senders remain at random positions
        elif pos_mode in (SenderPositionMode.FIXED_5, SenderPositionMode.RANDOM):
            if not cfg.cell_types.fix_senders_across_fractions:
                if position_dict is not None:
                    active_assignments, sender_pos_list = self._distribute_active_senders(
                        n_active, position_dict
                    )
                    for i, (pn, coord) in enumerate(sender_pos_list):
                        all_pos[active_indices[i]] = coord

        # Select receivers
        all_indices = np.arange(n_total)
        non_sender_indices = np.setdiff1d(all_indices, sender_indices)
        n_receivers = int(n_total * frac)
        receiver_indices = np.random.choice(non_sender_indices, n_receivers, replace=False)

        # Split receivers into active/silent if needed
        active_receiver_indices = None
        silent_receiver_indices = None
        if cfg.cell_types.receiver_silent_fraction > 0:
            n_active_recv = int(n_receivers * (1 - cfg.cell_types.receiver_silent_fraction))
            active_receiver_indices = receiver_indices[:n_active_recv]
            silent_receiver_indices = receiver_indices[n_active_recv:]

        # ===== Factor Expression =====
        expressing_mask = None
        factor_raw = None
        factor_vst = None

        if cfg.expression.vst_method:
            # VST mode: generate raw counts then transform
            factor_raw, _ = self.expression.generate_vst_factor(
                n_total, n_active, active_indices, sender_indices, silent_indices
            )
            factor_vst = apply_vst(factor_raw, cfg.expression.vst_method)
            factor_expr_for_diffusion = factor_raw
            factor_expr_for_ind = factor_vst
        elif cfg.expression.factor_model == "deterministic":
            factor_expr, _ = self.expression._factor_deterministic(
                n_total, n_active, active_indices, silent_indices
            )
            # Handle silent_expr_zero
            if cfg.cell_types.silent_expr_zero and len(silent_indices) > 0:
                factor_expr[silent_indices] = 0.0
            factor_expr_for_diffusion = factor_expr
            factor_expr_for_ind = factor_expr
        elif cfg.expression.factor_model == "bernoulli_mixture":
            factor_expr, expressing_mask = self.expression._factor_bernoulli_mixture(
                n_total, sender_indices
            )
            factor_expr_for_diffusion = factor_expr
            factor_expr_for_ind = factor_expr
        elif cfg.expression.factor_model == "stochastic":
            factor_expr, expressing_mask = self.expression._factor_stochastic(
                n_total, n_active, active_indices
            )
            factor_expr_for_diffusion = factor_expr
            factor_expr_for_ind = factor_expr
        elif cfg.expression.factor_model == "stochastic_ref":
            factor_expr, expressing_mask = self.expression._factor_stochastic_ref(
                n_total, n_active, active_indices
            )
            factor_expr_for_diffusion = factor_expr
            factor_expr_for_ind = factor_expr
        else:
            raise ValueError(f"Unknown factor_model: {cfg.expression.factor_model}")

        # ===== Diffusion =====
        n_density = n_receivers / domain_area

        if pos_mode in (SenderPositionMode.FIXED_5, SenderPositionMode.RANDOM) and position_dict is not None:
            # Multi-position: use solve_concentration_field_MM_multipos logic
            p_r_eff = 1.0
            if cfg.expression.response_model == "bernoulli_constant":
                p_r_eff = cfg.stochastic.p_r
            elif cfg.expression.response_model == "stochastic_hill":
                p_r_eff = cfg.stochastic.p_respond_max

            concentrations, lambda_val = self._solve_multipos(
                all_pos[sender_indices], factor_expr_for_diffusion[sender_indices],
                all_pos, n_density, position_dict, active_assignments,
                cfg.diffusion.secretion_rate, p_r_eff,
                cfg.diffusion.active_threshold,
            )
        else:
            # Single source
            if cfg.expression.response_model == "bernoulli_constant":
                n_density_eff = n_density * cfg.stochastic.p_r
            elif cfg.expression.response_model == "bernoulli_hill":
                n_density_eff = n_density * cfg.stochastic.p_r_max
            else:
                n_density_eff = n_density

            at = cfg.expression.vst_active_threshold if cfg.expression.vst_method else cfg.diffusion.active_threshold
            concentrations, lambda_val = self._solve_single(
                all_pos[sender_indices], factor_expr_for_diffusion[sender_indices],
                all_pos, n_density_eff, at,
            )

        # ===== Response Expression =====
        responding_mask = None
        response_probs = None
        responsive_raw = None
        responsive_vst = None

        if cfg.expression.vst_method:
            responsive_raw, _, _ = self.expression.generate_vst_response(
                n_total, receiver_indices, concentrations
            )
            responsive_vst = apply_vst(responsive_raw, cfg.expression.vst_method)
            responsive_expr_for_ind = responsive_vst
        elif cfg.expression.response_model == "deterministic":
            responsive_expr, _, _ = self.expression._response_deterministic(
                n_total, receiver_indices, concentrations, active_receiver_indices
            )
            responsive_expr_for_ind = responsive_expr
        elif cfg.expression.response_model == "bernoulli_constant":
            responsive_expr, responding_mask, _ = self.expression._response_bernoulli_constant(
                n_total, receiver_indices, concentrations
            )
            responsive_expr_for_ind = responsive_expr
        elif cfg.expression.response_model == "bernoulli_hill":
            responsive_expr, responding_mask, response_probs = self.expression._response_bernoulli_hill(
                n_total, receiver_indices, concentrations
            )
            responsive_expr_for_ind = responsive_expr
        elif cfg.expression.response_model == "stochastic_hill":
            responsive_expr, responding_mask, response_probs = self.expression._response_stochastic_hill(
                n_total, receiver_indices, concentrations
            )
            responsive_expr_for_ind = responsive_expr
        else:
            raise ValueError(f"Unknown response_model: {cfg.expression.response_model}")

        # ===== Compute I_ND =====
        ri_for_ind = receiver_indices
        if cfg.analysis.ind_uses_active_receivers_only and active_receiver_indices is not None:
            ri_for_ind = active_receiver_indices

        ind_curves = {}
        method_to_weight = {
            "ring": WeightType.RING,
            "gaussian_annular": WeightType.GAUSSIAN_ANNULAR,
        }
        for method in cfg.analysis.ind_methods:
            if method not in method_to_weight:
                raise ValueError(f"Unknown I_ND method: {method}")

            # Create INDComputer once per method, reuse across all radii
            method_config = type(cfg.analysis)(
                radii=cfg.analysis.radii,
                bandwidth=cfg.analysis.bandwidth,
                weight_type=method_to_weight[method],
                annular_width=cfg.analysis.annular_width,
                use_sigdiscov_core=False,
                ind_uses_active_receivers_only=cfg.analysis.ind_uses_active_receivers_only,
                sigma_fraction=cfg.analysis.sigma_fraction,
                ind_methods=cfg.analysis.ind_methods,
            )
            ind_computer = INDComputer(method_config)

            curve = []
            for d in radii:
                val, n_conn = ind_computer._compute_simple(
                    sender_indices, ri_for_ind, all_pos,
                    factor_expr_for_ind, responsive_expr_for_ind, d
                )
                curve.append({"distance": d, "I_ND": val, "n_connections": n_conn})
            ind_curves[method] = curve

        n_expressing = int(np.sum(expressing_mask)) if expressing_mask is not None else None
        n_responding = int(np.sum(responding_mask)) if responding_mask is not None else None

        logger.info(f"  Lambda: {lambda_val:.0f} um")
        if n_expressing is not None:
            logger.info(f"  Expressing: {n_expressing}/{n_active}")
        if n_responding is not None:
            logger.info(f"  Responding: {n_responding}/{n_receivers}")

        return {
            "lambda": lambda_val,
            "ind_curves": ind_curves,
            "n_expressing": n_expressing,
            "n_responding": n_responding,
            "assignments": dict(active_assignments) if active_assignments else None,
            "n_senders": len(sender_indices),
            "n_receivers": n_receivers,
            "positions": all_pos,
            "sender_indices": sender_indices,
            "active_indices": active_indices,
            "silent_indices": silent_indices,
            "receiver_indices": receiver_indices,
            "factor_expr": factor_expr_for_ind,
            "response_expr": responsive_expr_for_ind,
            "concentrations": concentrations,
            "expressing_mask": expressing_mask,
            "responding_mask": responding_mask,
            "response_probs": response_probs,
            "factor_raw": factor_raw,
            "factor_vst": factor_vst,
            "responsive_raw": responsive_raw,
            "responsive_vst": responsive_vst,
            "active_receiver_indices": active_receiver_indices,
            "position_dict": position_dict,
        }

    # =========================================================================
    # Reference-compatible helper methods (using np.random global state)
    # =========================================================================

    @staticmethod
    def _generate_positions(n_total, center, max_radius):
        """Generate positions using np.random (matching reference)."""
        angles = np.random.rand(n_total) * 2 * np.pi
        radii = np.sqrt(np.random.rand(n_total)) * max_radius
        return np.column_stack([
            center[0] + radii * np.cos(angles),
            center[1] + radii * np.sin(angles)
        ])

    @staticmethod
    def _get_5_positions(center, offset_distance):
        """Get 5 fixed positions matching reference get_5_positions()."""
        return {
            'C': np.array(center, dtype=float),
            'W': np.array([center[0] - offset_distance, center[1]]),
            'E': np.array([center[0] + offset_distance, center[1]]),
            'N': np.array([center[0], center[1] + offset_distance]),
            'S': np.array([center[0], center[1] - offset_distance]),
        }

    @staticmethod
    def _get_random_positions(n_positions, center, max_radius, min_separation=500.0):
        """Get random positions matching reference get_random_positions()."""
        positions = {}
        coords_list = []
        effective_radius = max_radius * 0.8
        attempts = 0
        max_attempts = 1000

        while len(positions) < n_positions and attempts < max_attempts:
            angle = np.random.rand() * 2 * np.pi
            r = np.sqrt(np.random.rand()) * effective_radius
            new_pos = np.array([center[0] + r * np.cos(angle), center[1] + r * np.sin(angle)])

            if len(coords_list) == 0:
                valid = True
            else:
                distances = [np.linalg.norm(new_pos - existing) for existing in coords_list]
                valid = min(distances) >= min_separation

            if valid:
                pos_label = f'P{len(positions) + 1}'
                positions[pos_label] = new_pos
                coords_list.append(new_pos)

            attempts += 1

        return positions

    @staticmethod
    def _distribute_active_senders(n_active, position_dict):
        """Distribute senders matching reference distribute_active_senders()."""
        position_names = list(position_dict.keys())
        P = len(position_names)

        if n_active < P:
            raise ValueError("n_active must be >= number of positions")

        assignments = {name: 1 for name in position_names}
        sender_positions = [(name, position_dict[name]) for name in position_names]

        remaining = n_active - P
        for _ in range(remaining):
            chosen = np.random.choice(position_names)
            assignments[chosen] += 1
            sender_positions.append((chosen, position_dict[chosen]))

        return assignments, sender_positions

    def _solve_single(self, sender_positions, sender_expression, cell_positions,
                       n_density_eff, active_threshold):
        """Solve concentration for single source, matching reference."""
        cfg = self.config.diffusion
        lambda_val = np.sqrt(cfg.D * cfg.Kd / (n_density_eff * cfg.k_max)) if n_density_eff > 0 else np.inf
        n_cells = len(cell_positions)
        concentrations = np.zeros(n_cells)

        active_mask = sender_expression > active_threshold
        active_pos = sender_positions[active_mask]
        active_expr = sender_expression[active_mask]

        if len(active_pos) > 0:
            center_pos = active_pos[0]
            total_factor = np.sum(active_expr)

            # Vectorized distance computation
            r = np.sqrt(np.sum((cell_positions - center_pos) ** 2, axis=1))
            near_mask = r < 1e-3
            far_mask = ~near_mask
            concentrations[near_mask] = total_factor * 100
            concentrations[far_mask] = total_factor * np.exp(-r[far_mask] / lambda_val) / np.sqrt(r[far_mask])

        return concentrations, lambda_val

    def _solve_multipos(self, sender_positions, sender_expression, cell_positions,
                         n_density, position_dict, active_assignments,
                         secretion_rate, p_r_eff, active_threshold):
        """Solve concentration for multi-position, matching reference."""
        cfg = self.config.diffusion
        lambda_val = np.sqrt(cfg.D * cfg.Kd / (n_density * cfg.k_max * p_r_eff)) if n_density * p_r_eff > 0 else np.inf
        n_cells = len(cell_positions)
        concentrations = np.zeros(n_cells)

        active_mask = sender_expression > active_threshold
        active_expr = sender_expression[active_mask]
        active_pos = sender_positions[active_mask]

        if len(active_pos) > 0 and position_dict is not None:
            unique_positions = {}
            for i, (pos, expr) in enumerate(zip(active_pos, active_expr)):
                pos_tuple = tuple(pos)
                if pos_tuple not in unique_positions:
                    unique_positions[pos_tuple] = {'pos': pos, 'total_expr': 0}
                unique_positions[pos_tuple]['total_expr'] += expr

            for pos_data in unique_positions.values():
                source_pos = pos_data['pos']
                total_factor = pos_data['total_expr'] * secretion_rate

                # Vectorized distance computation
                r = np.sqrt(np.sum((cell_positions - source_pos) ** 2, axis=1))
                near_mask = r < 1e-3
                far_mask = ~near_mask
                concentrations[near_mask] += total_factor * 100
                concentrations[far_mask] += total_factor * np.exp(-r[far_mask] / lambda_val) / np.sqrt(r[far_mask])

        return concentrations, lambda_val

    def _save_results(self, results: dict[float, dict]) -> None:
        """Save results to output directory."""
        output_path = Path(self.config.output_dir)

        # I_ND curves as CSV
        all_rows = []
        for frac, data in results.items():
            for method, curve in data["ind_curves"].items():
                for point in curve:
                    all_rows.append(
                        {
                            "receiver_fraction": frac,
                            "method": method,
                            "distance": point["distance"],
                            "I_ND": point["I_ND"],
                            "n_connections": point["n_connections"],
                            "lambda": data["lambda"],
                        }
                    )

        if all_rows:
            df = pd.DataFrame(all_rows)
            df.to_csv(output_path / "ind_results.csv", index=False)

        # Summary
        summary = []
        for frac, data in results.items():
            summary.append(
                {
                    "receiver_fraction": frac,
                    "lambda": data["lambda"],
                    "n_expressing": data["n_expressing"],
                    "n_responding": data["n_responding"],
                    "n_senders": data["n_senders"],
                    "n_receivers": data["n_receivers"],
                }
            )

        if summary:
            pd.DataFrame(summary).to_csv(output_path / "simulation_summary.csv", index=False)

    def run_single(
        self,
        receiver_fraction: float = 0.2,
        seed: int = None,
    ) -> dict[str, Any]:
        """
        Run simulation for a single receiver fraction.

        Parameters
        ----------
        receiver_fraction : float
            Fraction of cells that are receivers.
        seed : int, optional
            Random seed for reference compatibility.

        Returns
        -------
        Dict[str, Any]
            Simulation results.
        """
        cfg = self.config
        orig_fractions = cfg.cell_types.receiver_fractions
        cfg.cell_types.receiver_fractions = [receiver_fraction]
        results = self.run(seed=seed)
        cfg.cell_types.receiver_fractions = orig_fractions
        return results[receiver_fraction]
