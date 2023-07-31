from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class RunCfg:
    experiment_root_dir: str
    cache_dir: str
    run_name: str
    save_path: str
    checkpoint_path: Optional[str]
    filtering_model_path: str
    score_model_path: str
    num_samples: int
    save_top_n_samples: int
    prediction_storage: str
    torchhub_path: str
    tensorboard_path: str


@dataclass
class LoggingCfg:
    project: Optional[str]
    logger: str  # Literal["wandb", "tensorboard"]
    run_name: Optional[str] = None
    wandb_run_name: Optional[str] = None
    log_frequency: int = 10
    entity: Optional[str] = None
    group: Optional[str] = None
    visualize_n_val_graphs: int = 5
    visualization_path: str = "./visualization"


# @dataclass
# class DataReadCfg:
#     dataset: Literal["dips", "db5", "toy"]
#     data_path: str
#     data_file: str
#     pose_path: str
#     pose_file: str


# @dataclass
# class PredCfg:
# pdb_id: str
# pdb_path: str
# receptor_chains: list[str]
# ligand_chains: list[str]


@dataclass
class DataCfg:
    dataset: str  # Literal["dips", "db5", "single_pair", "sabdab"]
    structures_dir: str
    data_file: str
    recache: bool
    debug: bool
    resolution: str  # Literal["residue", "backbone", "atom"]
    no_graph_cache: bool
    use_orientation_features: bool
    knn_size: int
    use_unbound: bool
    multiplicity: int


@dataclass
class ProcessingCfg:
    receptor_radius: float = 30
    c_alpha_max_neighbours: int = 10
    atom_radius: int = 5
    atom_max_neighbours: int = 8
    matching_popsize: int = 20
    matching_maxiter: int = 20
    max_lig_size: Optional[int] = None
    remove_hs: bool = False
    multiplicity: int = 1


@dataclass
class DiffusionCfg:
    tr_weight: float = 0.33
    rot_weight: float = 0.33
    tor_weight: float = 0.33
    tr_s_min: float = 0.1
    tr_s_max: float = 0.30
    rot_s_min: float = 0.1
    rot_s_max: float = 1.65
    tor_s_min: float = 0.0314
    tor_s_max: float = 3.14


@dataclass
class BaseModelCfg:
    max_radius: float
    cross_max_dist: float
    dynamic_max_cross: bool
    cross_cutoff_weight: float
    cross_cutoff_bias: float
    ns: int
    nv: int
    scale_by_sigma: bool
    no_torsion: bool
    num_conv_layers: int
    embedding_type: str
    sigma_embed_dim: int
    embedding_scale: int
    dist_embed_dim: int
    cross_dist_embed_dim: int
    lm_embed_dim: int
    no_batch_norm: bool
    use_second_order_repr: bool
    dropout: float


@dataclass
class ModelCfg:
    knn_size: int
    lm_embed_dim: int
    score_model_cfg: BaseModelCfg
    confidence_model_cfg: BaseModelCfg


@dataclass
class TrainingCfg:
    mode: str
    batch_size: int
    num_workers: int
    epochs: int
    num_folds: int
    tr_weight: float
    rot_weight: float
    tor_weight: float
    test_fold: int = 0
    patience: int = 10
    seed: int = 0
    num_gpu: int = 0
    gpu: int = 0
    metric: str = "loss"
    save_pred: bool = False
    no_tqdm: bool = False
    save_model_every: int = 10


@dataclass
class TempCfg:
    temp_sampling: float = 1.0
    temp_psi: float = 0.0
    temp_sigma_data_tr: float = 0.5
    temp_sigma_data_rot: float = 0.5


@dataclass
class InferenceCfg:
    num_steps: int
    actual_steps: int
    num_inference_complexes: Optional[int] = None
    num_inference_complexes_train_data: Optional[int] = None
    val_inference_freq: int = 5
    ode: bool = False
    no_random: bool = False
    no_final_noise: bool = False
    sample_train: bool = False
    mirror_ligand: bool = False
    run_inference_without_confidence_model: bool = False
    wandb_sweep: bool = False
    no_final_noise: bool = False
    temp_cfg: Optional[TempCfg] = None


@dataclass
class EmbeddingCfg:
    hidden_size: int = 64
    dropout: float = 0.1
    knn_size: int = 20


@dataclass
class OptimizerCfg:
    lr: float = 1e-4
    weight_decay: float = 1e-6
    score_loss_weight: float = 0.0
    energy_loss_weight: float = 1.0


@dataclass
class ConfidenceModelCfg:
    rmsd_prediction: bool = False
    rmsd_classification_cutoff: float = 5.0
    generate_n_predictions: int = 7
    samples_directory: str = ""
    use_randomized_confidence_data: bool = False
    rmsd_type: str = "simple"  # Literal["complex", "interface", "simple"] = "simple"


@dataclass
class DiffDockCfg:
    run_cfg: RunCfg
    logging_cfg: LoggingCfg
    data_cfg: DataCfg
    processing_cfg: ProcessingCfg
    model_cfg: ModelCfg
    embedding_cfg: EmbeddingCfg
    training_cfg: TrainingCfg
    optimizer_cgf: OptimizerCfg
    diffusion_cfg: DiffusionCfg
    inference_cfg: InferenceCfg
