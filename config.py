"""
CLIR-Net Configuration File
Central configuration for all hyperparameters, paths, and settings
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path


@dataclass
class DataConfig:
    """Data and preprocessing configuration"""
    # Paths
    data_root: Path = Path("./data")
    raw_data_dir: Path = Path("./data/raw")
    processed_data_dir: Path = Path("./data/processed")
    cache_dir: Path = Path("./data/cache")
    
    # Audio settings
    sample_rate: int = 16000
    n_mels: int = 80
    hop_length_ms: float = 10.0  # 10ms
    win_length_ms: float = 25.0  # 25ms
    n_fft: int = 400  # Will be computed from win_length
    
    # Prosodic features
    f0_min: float = 75.0  # Hz
    f0_max: float = 600.0  # Hz
    
    # Video/AU settings
    au_features: List[str] = field(default_factory=lambda: [
        'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r',
        'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r',
        'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r'
    ])
    au_dim: int = 16
    
    # Datasets to use
    datasets: List[str] = field(default_factory=lambda: [
        'emodb', 'emovo', 'shemo', 'casia', 'cremad'
    ])
    
    # Emotion labels (6 basic emotions)
    emotions: List[str] = field(default_factory=lambda: [
        'anger', 'disgust', 'fear', 'happiness', 'sadness', 'neutral'
    ])
    n_classes: int = 6
    
    # Augmentation
    use_augmentation: bool = True
    noise_snr_range: tuple = (5, 20)  # dB
    pitch_shift_range: tuple = (-1, 1)  # semitones
    time_stretch_range: tuple = (0.95, 1.05)


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # Audio encoder
    audio_pretrained: str = "facebook/wav2vec2-base"  # or "facebook/wav2vec2-xls-r-300m"
    audio_freeze_layers: int = 6  # Freeze first N layers
    audio_out_dim: int = 512
    prosody_dim: int = 128
    
    # AU encoder
    au_encoder_dim: int = 256
    au_encoder_heads: int = 8
    au_encoder_layers: int = 3
    au_out_dim: int = 512
    
    # SCM
    speaker_dim: int = 64
    lang_dim: int = 16
    latent_dim: int = 256
    scm_hidden_dim: int = 512
    
    # Emotion head
    emotion_hidden_dim: int = 128
    
    # Dropout
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Optimization
    batch_size: int = 16
    num_epochs: int = 100
    learning_rate: float = 1e-4
    encoder_lr: float = 1e-5  # Lower LR for pretrained encoders
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    
    # Loss weights
    alpha_inv: float = 1.0  # Invariance loss weight
    beta_cf_pred: float = 1.0  # CF prediction consistency weight
    gamma_acyc: float = 1e-2  # Acyclicity penalty weight
    delta_gen: float = 0.0  # Generator loss (not used initially)
    
    # Scheduler
    warmup_epochs: int = 5
    scheduler_type: str = "cosine"  # "cosine" or "plateau"
    
    # Training strategy
    use_mixed_precision: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    
    # Validation
    val_every_n_epochs: int = 1
    early_stopping_patience: int = 15
    
    # Checkpointing
    checkpoint_dir: Path = Path("./checkpoints")
    save_top_k: int = 3
    
    # Logging
    log_dir: Path = Path("./logs")
    use_wandb: bool = False
    wandb_project: str = "clir-net"
    log_every_n_steps: int = 10


@dataclass
class ExperimentConfig:
    """Experiment configuration"""
    # Cross-validation
    n_folds: int = 5
    use_speaker_independent_split: bool = True
    
    # Cross-lingual evaluation
    leave_one_out_languages: List[str] = field(default_factory=lambda: [
        'german', 'italian', 'persian', 'chinese', 'english'
    ])
    
    # Robustness testing
    test_snr_levels: List[int] = field(default_factory=lambda: [20, 10, 5, 0])
    
    # Statistical testing
    n_bootstrap: int = 1000
    confidence_level: float = 0.95
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True


@dataclass
class Config:
    """Main configuration class"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    # Device
    device: str = "cuda"  # Will be set automatically
    
    def __post_init__(self):
        """Create directories if they don't exist"""
        for path_attr in ['data_root', 'raw_data_dir', 'processed_data_dir', 'cache_dir']:
            path = getattr(self.data, path_attr)
            path.mkdir(parents=True, exist_ok=True)
        
        self.training.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.training.log_dir.mkdir(parents=True, exist_ok=True)


# Default configuration instance
def get_config() -> Config:
    """Get default configuration"""
    return Config()


if __name__ == "__main__":
    # Test configuration
    cfg = get_config()
    print("Data config:", cfg.data)
    print("Model config:", cfg.model)
    print("Training config:", cfg.training)
