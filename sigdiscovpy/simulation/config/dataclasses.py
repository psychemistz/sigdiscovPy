"""
Configuration dataclasses for spatial simulation.

These dataclasses provide type-safe, IDE-friendly configuration for
all aspects of the simulation: domain, cell types, diffusion, expression, etc.
"""

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np


class WeightType(Enum):
    """Weight matrix types for I_ND computation."""

    RING = "ring"
    GAUSSIAN = "gaussian"
    ANNULAR = "annular"
    GAUSSIAN_ANNULAR = "gaussian_annular"


class SenderPositionMode(Enum):
    """Sender cell position modes."""

    CENTER = "center"
    CENTER_SILENT_DISTRIBUTED = "center_silent_distributed"
    FIXED_5 = "fixed_5"  # Center, West, East, North, South
    RANDOM = "random"


@dataclass
class DomainConfig:
    """
    Spatial domain configuration.

    Parameters
    ----------
    n_cells : int
        Total number of cells in the simulation.
    max_radius : float
        Maximum radius of the circular domain (in microns).
    center : Tuple[float, float]
        Center coordinates of the domain.
    random_seed : int
        Random seed for reproducibility.
    """

    n_cells: int = 100000
    max_radius: float = 5000.0
    center: tuple[float, float] = (0.0, 0.0)
    random_seed: int = 42


@dataclass
class CellTypeConfig:
    """
    Cell type assignment configuration.

    Parameters
    ----------
    n_active_senders : int
        Number of sender cells that actively express the factor.
    n_silent_senders : int
        Number of sender cells that don't express (for negative controls).
    receiver_fractions : List[float]
        List of receiver cell fractions to simulate.
    silent_expr_zero : bool
        If True, silent senders get 0 expression instead of basal.
    fix_senders_across_fractions : bool
        If True, reuse same sender assignments across receiver fractions.
    receiver_silent_fraction : float
        Fraction of receivers that are "silent" (non-responding).
    """

    n_active_senders: int = 20
    n_silent_senders: int = 0
    receiver_fractions: list[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.5])
    silent_expr_zero: bool = False
    fix_senders_across_fractions: bool = False
    receiver_silent_fraction: float = 0.0


@dataclass
class PositionConfig:
    """
    Sender position configuration.

    Parameters
    ----------
    mode : SenderPositionMode
        How to position sender cells.
    n_positions : int
        Number of distinct positions (for RANDOM mode).
    offset_distance : float
        Distance from center for FIXED_5 mode.
    min_separation : float
        Minimum separation between random positions.
    """

    mode: SenderPositionMode = SenderPositionMode.CENTER
    n_positions: int = 1
    offset_distance: float = 3000.0
    min_separation: float = 500.0


@dataclass
class DiffusionConfig:
    """
    Diffusion model parameters.

    Parameters
    ----------
    D : float
        Diffusion coefficient (um^2/s).
    k_max : float
        Maximum uptake rate.
    Kd : float
        Dissociation constant for uptake.
    secretion_rate : float
        Expression to concentration scaling factor.
    active_threshold : float
        Expression threshold to consider a sender "active". Default 1.0 matches reference.
    """

    D: float = 100.0
    k_max: float = 10.0
    Kd: float = 1.0
    secretion_rate: float = 1.0
    active_threshold: float = 1.0


@dataclass
class ExpressionConfig:
    """
    Gene expression parameters.

    Parameters
    ----------
    F_basal : float
        Basal factor expression level.
    F_high : float
        High factor expression for active senders.
    R_basal : float
        Basal response expression level.
    fold_change : float
        Maximum fold change for response.
    sigma_f : float
        Lognormal sigma for expressing senders.
    sigma_f_basal : float
        Lognormal sigma for non-expressing cells.
    sigma_r : float
        Lognormal sigma for responding receivers.
    sigma_r_basal : float
        Lognormal sigma for non-responding cells.
    factor_model : str
        Factor expression model: deterministic | stochastic | stochastic_ref | bernoulli_mixture
    response_model : str
        Response expression model: deterministic | stochastic_hill | bernoulli_constant | bernoulli_hill
    vst_method : str or None
        VST normalization method: None | log1p | pearson | shifted_log
    vst_active_threshold : float
        Active threshold for VST mode diffusion.
    """

    F_basal: float = 0.1
    F_high: float = 10.0
    R_basal: float = 0.1
    fold_change: float = 10.0
    sigma_f: float = 0.1
    sigma_f_basal: float = 0.1
    sigma_r: float = 0.1
    sigma_r_basal: float = 0.1
    factor_model: str = "deterministic"
    response_model: str = "deterministic"
    vst_method: Optional[str] = None
    vst_active_threshold: float = 1.0


@dataclass
class StochasticConfig:
    """
    Stochastic expression model parameters.

    Parameters
    ----------
    p_sender_express : float
        Probability that a sender cell expresses (Bernoulli on/off).
    p_receiver_respond_max : float
        Maximum probability of receiver response.
    hill_coefficient : float
        Hill coefficient for dose-response curve.
    zero_inflate_factor : float
        Fraction of non-sender cells with zero factor expression.
    zero_inflate_response : float
        Fraction of non-receiver cells with zero response expression.
    p_s : float
        Bernoulli probability for sender active state (bernoulli_mixture model).
    sigma_f_b : float
        Lognormal sigma for basal sender expression (bernoulli_mixture model).
    expr_cv : float
        Coefficient of variation for stochastic on expression.
    use_gamma_dist : bool
        If True, use gamma distribution; else lognormal (stochastic model).
    sigma_f_basal : float
        Lognormal sigma for basal cells (stochastic_ref model).
    p_r : float
        Fixed receiver response probability (bernoulli_constant model).
    sigma_r_b : float
        Lognormal sigma for non-responding receivers (bernoulli models).
    p_r_max : float
        Maximum receiver probability (bernoulli_hill model).
    K_p : float
        Concentration at half-maximal probability (bernoulli_hill model).
    hill_n : float
        Hill coefficient for bernoulli_hill model.
    p_respond_max : float
        Maximum response probability (stochastic_hill model).
    response_hill_coef : float
        Hill coefficient for stochastic_hill response model.
    """

    p_sender_express: float = 0.9
    p_receiver_respond_max: float = 1.0
    hill_coefficient: float = 1.0
    zero_inflate_factor: float = 0.0
    zero_inflate_response: float = 0.0
    # bernoulli_mixture params
    p_s: float = 1.0
    sigma_f_b: float = 0.1
    # stochastic params
    expr_cv: float = 0.5
    use_gamma_dist: bool = True
    # stochastic_ref params
    sigma_f_basal: float = 0.1
    # bernoulli_constant params
    p_r: float = 1.0
    sigma_r_b: float = 0.1
    # bernoulli_hill params
    p_r_max: float = 1.0
    K_p: float = 10.0
    hill_n: float = 1.0
    # stochastic_hill params
    p_respond_max: float = 0.9
    response_hill_coef: float = 1.0


@dataclass
class AnalysisConfig:
    """
    I_ND analysis configuration.

    Parameters
    ----------
    radii : List[float]
        List of radii at which to compute I_ND.
    bandwidth : float
        Ring bandwidth for ring weights.
    weight_type : WeightType
        Type of weight matrix to use.
    annular_width : float
        Width of annular ring for annular weights.
    use_sigdiscov_core : bool
        Whether to use sigdiscovpy core functions (recommended).
    ind_uses_active_receivers_only : bool
        If True, only active (non-silent) receivers are used for I_ND.
    sigma_fraction : float
        Sigma = outer_radius / sigma_fraction for gaussian_annular weights.
    ind_methods : List[str]
        List of I_ND methods to compute (ring, gaussian_annular).
    """

    radii: list[float] = field(default_factory=lambda: list(np.arange(50, 5001, 50)))
    bandwidth: float = 100.0
    weight_type: WeightType = WeightType.RING
    annular_width: float = 50.0
    use_sigdiscov_core: bool = True
    ind_uses_active_receivers_only: bool = False
    sigma_fraction: float = 3.0
    ind_methods: list[str] = field(default_factory=lambda: ["ring"])


def _convert_to_native(obj: Any) -> Any:
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_native(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, Enum):
        return obj.value
    else:
        return obj


@dataclass
class SimulationConfig:
    """
    Complete simulation configuration.

    Combines all configuration components into a single object.

    Example
    -------
    >>> config = SimulationConfig()
    >>> config.domain.n_cells = 50000
    >>> config.cell_types.receiver_fractions = [0.1, 0.2]
    >>> config.save("my_config.json")
    """

    domain: DomainConfig = field(default_factory=DomainConfig)
    cell_types: CellTypeConfig = field(default_factory=CellTypeConfig)
    position: PositionConfig = field(default_factory=PositionConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    expression: ExpressionConfig = field(default_factory=ExpressionConfig)
    stochastic: StochasticConfig = field(default_factory=StochasticConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    output_dir: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        d = {
            "domain": asdict(self.domain),
            "cell_types": asdict(self.cell_types),
            "position": {**asdict(self.position), "mode": self.position.mode.value},
            "diffusion": asdict(self.diffusion),
            "expression": asdict(self.expression),
            "stochastic": asdict(self.stochastic),
            "analysis": {**asdict(self.analysis), "weight_type": self.analysis.weight_type.value},
            "output_dir": self.output_dir,
        }
        return _convert_to_native(d)

    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SimulationConfig":
        """Load configuration from JSON file."""
        with open(path) as f:
            d = json.load(f)

        config = cls()

        # Domain
        if "domain" in d:
            for k, v in d["domain"].items():
                if k == "center":
                    v = tuple(v)
                setattr(config.domain, k, v)

        # Cell types
        if "cell_types" in d:
            for k, v in d["cell_types"].items():
                setattr(config.cell_types, k, v)

        # Position
        if "position" in d:
            for k, v in d["position"].items():
                if k == "mode":
                    v = SenderPositionMode(v)
                setattr(config.position, k, v)

        # Diffusion
        if "diffusion" in d:
            for k, v in d["diffusion"].items():
                setattr(config.diffusion, k, v)

        # Expression
        if "expression" in d:
            for k, v in d["expression"].items():
                setattr(config.expression, k, v)

        # Stochastic
        if "stochastic" in d:
            for k, v in d["stochastic"].items():
                setattr(config.stochastic, k, v)

        # Analysis
        if "analysis" in d:
            for k, v in d["analysis"].items():
                if k == "weight_type":
                    v = WeightType(v)
                setattr(config.analysis, k, v)

        # Output dir
        config.output_dir = d.get("output_dir")

        return config
