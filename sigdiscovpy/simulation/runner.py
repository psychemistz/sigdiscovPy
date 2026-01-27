"""
UnifiedSimulation - Main orchestrator for spatial simulation.

Coordinates all simulation components to generate synthetic spatial
transcriptomics data with known ground truth for method validation.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

import numpy as np
import pandas as pd

from sigdiscovpy.simulation.config.dataclasses import SimulationConfig
from sigdiscovpy.simulation.domain.spatial import (
    SpatialDomain,
    SenderPositionGenerator,
)
from sigdiscovpy.simulation.physics.diffusion import DiffusionSolver
from sigdiscovpy.simulation.expression.stochastic import ExpressionGenerator
from sigdiscovpy.simulation.analysis.ind_computer import INDComputer


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UnifiedSimulation:
    """
    Main orchestrator for unified spatial simulation.

    Coordinates domain generation, expression simulation, diffusion modeling,
    and I_ND computation to produce complete simulation results.

    Parameters
    ----------
    config : SimulationConfig
        Complete simulation configuration.
    use_gpu : bool
        Whether to use GPU acceleration for I_ND computation.

    Example
    -------
    >>> from sigdiscovpy.simulation import SimulationPresets
    >>> config = SimulationPresets.default()
    >>> config.output_dir = "./output"
    >>> sim = UnifiedSimulation(config)
    >>> results = sim.run()
    """

    def __init__(self, config: SimulationConfig, use_gpu: bool = False):
        self.config = config
        self.use_gpu = use_gpu

        # Initialize components
        self.domain = SpatialDomain(config.domain)
        self.position_gen = SenderPositionGenerator(
            config.position, config.domain
        )
        self.diffusion = DiffusionSolver(config.diffusion)
        self.expression = ExpressionGenerator(
            config.expression, config.stochastic,
            seed=config.domain.random_seed
        )
        self.ind_computer = INDComputer(config.analysis, use_gpu=use_gpu)

        # Random generator
        self._rng = np.random.default_rng(config.domain.random_seed)

    def run(self) -> Dict[float, Dict]:
        """
        Run simulation for all receiver fractions.

        Returns
        -------
        Dict[float, Dict]
            Results keyed by receiver fraction, each containing:
            - 'lambda': characteristic diffusion length
            - 'ind_curve': list of {radius, I_ND, n_connections}
            - 'n_expressing': number of expressing senders
            - 'n_responding': number of responding receivers
            - 'assignments': sender position assignments
        """
        logger.info("=" * 70)
        logger.info("UNIFIED SIMULATION")
        logger.info("=" * 70)

        # Generate cell positions
        all_positions = self.domain.generate_positions()
        position_dict = self.position_gen.generate_positions()

        logger.info(f"Generated {len(all_positions)} cell positions")
        logger.info(f"Sender positions: {list(position_dict.keys())}")

        # Assign sender cells
        n_senders = (
            self.config.cell_types.n_active_senders +
            self.config.cell_types.n_silent_senders
        )
        n_active = self.config.cell_types.n_active_senders

        all_indices = np.arange(len(all_positions))
        sender_indices = self._rng.choice(all_indices, n_senders, replace=False)
        active_indices = sender_indices[:n_active]
        silent_indices = sender_indices[n_active:]

        # Distribute active senders across positions
        assignments, sender_pos_list = SenderPositionGenerator.distribute_senders(
            n_active, position_dict, self._rng
        )

        # Move sender cells to their assigned positions
        SenderPositionGenerator.assign_cells_to_positions(
            all_positions, sender_pos_list, active_indices
        )

        logger.info(f"Active sender distribution: {assignments}")

        # Save config if output_dir specified
        if self.config.output_dir:
            output_path = Path(self.config.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            self.config.save(str(output_path / 'config.json'))

        results = {}

        for frac in self.config.cell_types.receiver_fractions:
            logger.info(f"\nProcessing {frac*100:.0f}% Receivers...")

            result = self._run_single_fraction(
                frac, all_positions, all_indices,
                sender_indices, active_indices, silent_indices,
                position_dict, assignments
            )
            results[frac] = result

        # Save results
        if self.config.output_dir:
            self._save_results(results)

        logger.info(f"\nSimulation complete.")
        if self.config.output_dir:
            logger.info(f"Results saved to {self.config.output_dir}")

        return results

    def _run_single_fraction(
        self,
        frac: float,
        all_positions: np.ndarray,
        all_indices: np.ndarray,
        sender_indices: np.ndarray,
        active_indices: np.ndarray,
        silent_indices: np.ndarray,
        position_dict: Dict[str, np.ndarray],
        assignments: Dict[str, int],
    ) -> Dict[str, Any]:
        """Run simulation for a single receiver fraction."""
        n_cells = len(all_positions)

        # Assign receiver cells
        non_sender_indices = np.setdiff1d(all_indices, sender_indices)
        n_receivers = int(n_cells * frac)
        receiver_indices = self._rng.choice(
            non_sender_indices, n_receivers, replace=False
        )

        # Generate factor expression
        n_active = len(active_indices)
        factor_expr, expressing_mask = self.expression.generate_factor_expression(
            n_cells, n_active, active_indices
        )

        n_expressing = np.sum(expressing_mask)
        logger.info(f"  Expressing senders: {n_expressing}/{n_active}")

        # Solve diffusion
        domain_area = self.domain.get_domain_area()
        n_density = n_receivers / domain_area

        concentrations, lambda_val = self.diffusion.solve(
            all_positions[sender_indices],
            factor_expr[sender_indices],
            all_positions,
            n_density,
            position_dict,
            self.config.stochastic.p_receiver_respond_max
        )

        # Generate response expression
        response_expr, responding_mask, response_probs = (
            self.expression.generate_response_expression(
                n_cells, receiver_indices, concentrations
            )
        )

        n_responding = np.sum(responding_mask)
        logger.info(f"  Responding receivers: {n_responding}/{n_receivers}")
        logger.info(f"  Lambda: {lambda_val:.0f} um")

        # Compute I_ND curve
        ind_curve = self.ind_computer.compute_at_radii(
            sender_indices, receiver_indices, all_positions,
            factor_expr, response_expr
        )

        return {
            'lambda': lambda_val,
            'ind_curve': ind_curve,
            'n_expressing': int(n_expressing),
            'n_responding': int(n_responding),
            'assignments': dict(assignments),
            'n_senders': len(sender_indices),
            'n_receivers': n_receivers,
        }

    def _save_results(self, results: Dict[float, Dict]) -> None:
        """Save results to output directory."""
        output_path = Path(self.config.output_dir)

        # I_ND curves as CSV
        all_rows = []
        for frac, data in results.items():
            for point in data['ind_curve']:
                all_rows.append({
                    'receiver_fraction': frac,
                    'radius': point['radius'],
                    'I_ND': point['I_ND'],
                    'n_connections': point['n_connections'],
                    'lambda': data['lambda'],
                })

        if all_rows:
            df = pd.DataFrame(all_rows)
            df.to_csv(output_path / 'ind_results.csv', index=False)

        # Summary
        summary = []
        for frac, data in results.items():
            summary.append({
                'receiver_fraction': frac,
                'lambda': data['lambda'],
                'n_expressing': data['n_expressing'],
                'n_responding': data['n_responding'],
                'n_senders': data['n_senders'],
                'n_receivers': data['n_receivers'],
            })

        if summary:
            pd.DataFrame(summary).to_csv(
                output_path / 'simulation_summary.csv', index=False
            )

    def run_single(
        self,
        receiver_fraction: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Run simulation for a single receiver fraction.

        Convenience method for quick testing.

        Parameters
        ----------
        receiver_fraction : float
            Fraction of cells that are receivers.

        Returns
        -------
        Dict[str, Any]
            Simulation results including positions, expression, and I_ND curve.
        """
        # Generate positions
        all_positions = self.domain.generate_positions()
        position_dict = self.position_gen.generate_positions()

        # Setup cells
        n_cells = len(all_positions)
        n_active = self.config.cell_types.n_active_senders
        n_silent = self.config.cell_types.n_silent_senders
        n_senders = n_active + n_silent

        all_indices = np.arange(n_cells)
        sender_indices = self._rng.choice(all_indices, n_senders, replace=False)
        active_indices = sender_indices[:n_active]
        silent_indices = sender_indices[n_active:]

        assignments, sender_pos_list = SenderPositionGenerator.distribute_senders(
            n_active, position_dict, self._rng
        )
        SenderPositionGenerator.assign_cells_to_positions(
            all_positions, sender_pos_list, active_indices
        )

        # Receivers
        non_sender_indices = np.setdiff1d(all_indices, sender_indices)
        n_receivers = int(n_cells * receiver_fraction)
        receiver_indices = self._rng.choice(
            non_sender_indices, n_receivers, replace=False
        )

        # Expression
        factor_expr, expressing_mask = self.expression.generate_factor_expression(
            n_cells, n_active, active_indices
        )

        # Diffusion
        domain_area = self.domain.get_domain_area()
        n_density = n_receivers / domain_area

        concentrations, lambda_val = self.diffusion.solve(
            all_positions[sender_indices],
            factor_expr[sender_indices],
            all_positions,
            n_density,
            position_dict
        )

        # Response
        response_expr, responding_mask, _ = self.expression.generate_response_expression(
            n_cells, receiver_indices, concentrations
        )

        # I_ND curve
        ind_curve = self.ind_computer.compute_at_radii(
            sender_indices, receiver_indices, all_positions,
            factor_expr, response_expr
        )

        return {
            'positions': all_positions,
            'position_dict': position_dict,
            'sender_indices': sender_indices,
            'receiver_indices': receiver_indices,
            'active_indices': active_indices,
            'silent_indices': silent_indices,
            'factor_expr': factor_expr,
            'response_expr': response_expr,
            'concentrations': concentrations,
            'lambda': lambda_val,
            'ind_curve': ind_curve,
            'expressing_mask': expressing_mask,
            'responding_mask': responding_mask,
        }
