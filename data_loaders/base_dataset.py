"""
Base Dataset Class for Emotion Recognition
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json


class EmotionDataset(Dataset):
    """
    Base dataset class for emotion recognition
    
    Expects preprocessed features saved as .pt files
    """
    
    # Standard emotion mapping (6 basic emotions)
    EMOTION_MAP = {
        'anger': 0,
        'disgust': 1,
        'fear': 2,
        'happiness': 3,
        'sadness': 4,
        'neutral': 5
    }
    
    # Language mapping
    LANGUAGE_MAP = {
        'german': 0,
        'italian': 1,
        'persian': 2,
        'chinese': 3,
        'english': 4
    }
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        language: str = 'english',
        max_audio_len: int = 16000 * 10,  # 10 seconds max
        max_frames: int = 1000,  # Max frames for prosody/AU
        augment: bool = False
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.language = language
        self.max_audio_len = max_audio_len
        self.max_frames = max_frames
        self.augment = augment
        
        # Load metadata
        self.samples = self._load_metadata()
        
        # Build speaker vocabulary
        self.speaker_to_id = self._build_speaker_vocab()
        
    def _load_metadata(self) -> List[Dict]:
        """Load dataset metadata"""
        metadata_path = self.data_dir / f"{self.split}_metadata.json"
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            samples = json.load(f)
        
        return samples
    
    def _build_speaker_vocab(self) -> Dict[str, int]:
        """Build speaker ID vocabulary"""
        speakers = set()
        for sample in self.samples:
            speakers.add(sample['speaker_id'])
        
        speaker_to_id = {spk: idx for idx, spk in enumerate(sorted(speakers))}
        return speaker_to_id
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _pad_or_truncate(
        self,
        tensor: torch.Tensor,
        max_len: int,
        dim: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pad or truncate tensor to max_len
        
        Returns:
            padded_tensor: Padded/truncated tensor
            mask: Padding mask (True for padded positions)
        """
        current_len = tensor.size(dim)
        
        if current_len > max_len:
            # Truncate
            if dim == 0:
                tensor = tensor[:max_len]
            elif dim == 1:
                tensor = tensor[:, :max_len]
            mask = torch.zeros(max_len, dtype=torch.bool)
        else:
            # Pad
            pad_len = max_len - current_len
            
            if dim == 0:
                pad_shape = (pad_len,) + tensor.shape[1:]
                padding = torch.zeros(pad_shape, dtype=tensor.dtype)
                tensor = torch.cat([tensor, padding], dim=0)
            elif dim == 1:
                pad_shape = (tensor.shape[0], pad_len) + tensor.shape[2:]
                padding = torch.zeros(pad_shape, dtype=tensor.dtype)
                tensor = torch.cat([tensor, padding], dim=1)
            
            # Create mask (True for padded positions)
            mask = torch.zeros(max_len, dtype=torch.bool)
            mask[current_len:] = True
        
        return tensor, mask
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample
        
        Returns:
            Dictionary containing:
                - waveform: (T_wave,) audio waveform
                - prosody_frames: (T_frames, prosody_dim) prosodic features
                - aus: (T_au, au_dim) AU intensities
                - audio_mask: (T_wave,) audio padding mask
                - prosody_mask: (T_frames,) prosody padding mask
                - au_mask: (T_au,) AU padding mask
                - speaker_id: int
                - lang_id: int
                - label: int
        """
        sample = self.samples[idx]
        
        # Load features
        features_path = self.data_dir / sample['features_path']
        features = torch.load(features_path, weights_only=False)
        
        # Extract components
        waveform = torch.from_numpy(features['waveform'])  # (T,)
        prosody_frames = torch.from_numpy(features['prosody_frames'].T)  # (T, prosody_dim)
        
        # Handle AU features (may not exist for audio-only datasets)
        if 'aus' in features:
            aus = torch.from_numpy(features['aus'])  # (T, au_dim)
        else:
            # Create dummy AUs if not available
            aus = torch.zeros(prosody_frames.size(0), 16)
        
        # Pad/truncate
        waveform, audio_mask = self._pad_or_truncate(waveform, self.max_audio_len, dim=0)
        prosody_frames, prosody_mask = self._pad_or_truncate(prosody_frames, self.max_frames, dim=0)
        aus, au_mask = self._pad_or_truncate(aus, self.max_frames, dim=0)
        
        # Get IDs
        speaker_id = self.speaker_to_id[sample['speaker_id']]
        lang_id = self.LANGUAGE_MAP[self.language]
        label = self.EMOTION_MAP[sample['emotion']]
        
        return {
            'waveform': waveform,
            'prosody_frames': prosody_frames,
            'aus': aus,
            'audio_mask': audio_mask,
            'prosody_mask': prosody_mask,
            'au_mask': au_mask,
            'speaker_id': torch.tensor(speaker_id, dtype=torch.long),
            'lang_id': torch.tensor(lang_id, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }


class MultiCorpusDataset(Dataset):
    """
    Multi-corpus dataset combining multiple languages
    """
    
    def __init__(
        self,
        datasets: List[EmotionDataset]
    ):
        self.datasets = datasets
        
        # Compute cumulative lengths
        self.cumulative_lengths = []
        total = 0
        for dataset in datasets:
            total += len(dataset)
            self.cumulative_lengths.append(total)
    
    def __len__(self) -> int:
        return self.cumulative_lengths[-1] if self.cumulative_lengths else 0
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Find which dataset this index belongs to
        dataset_idx = 0
        for i, cum_len in enumerate(self.cumulative_lengths):
            if idx < cum_len:
                dataset_idx = i
                break
        
        # Compute local index
        if dataset_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumulative_lengths[dataset_idx - 1]
        
        return self.datasets[dataset_idx][local_idx]


def create_dummy_dataset(output_dir: str, n_samples: int = 100):
    """
    Create a dummy dataset for testing
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create metadata
    emotions = list(EmotionDataset.EMOTION_MAP.keys())
    speakers = [f"speaker_{i:03d}" for i in range(10)]
    
    metadata = []
    for i in range(n_samples):
        sample = {
            'id': f'sample_{i:04d}',
            'speaker_id': np.random.choice(speakers),
            'emotion': np.random.choice(emotions),
            'features_path': f'features/sample_{i:04d}.pt'
        }
        metadata.append(sample)
    
    # Save metadata
    with open(output_dir / 'train_metadata.json', 'w') as f:
        json.dump(metadata[:80], f, indent=2)
    
    with open(output_dir / 'val_metadata.json', 'w') as f:
        json.dump(metadata[80:90], f, indent=2)
    
    with open(output_dir / 'test_metadata.json', 'w') as f:
        json.dump(metadata[90:], f, indent=2)
    
    # Create dummy features
    features_dir = output_dir / 'features'
    features_dir.mkdir(exist_ok=True)
    
    for i in range(n_samples):
        # Random lengths
        audio_len = np.random.randint(16000 * 2, 16000 * 5)  # 2-5 seconds
        n_frames = audio_len // 160  # 100Hz
        
        features = {
            'waveform': torch.randn(audio_len),
            'prosody_frames': torch.randn(6, n_frames),  # (prosody_dim, T)
            'aus': torch.randn(n_frames, 16)  # (T, au_dim)
        }
        
        torch.save(features, features_dir / f'sample_{i:04d}.pt')
    
    print(f"Created dummy dataset at {output_dir}")
    print(f"  Train: 80 samples")
    print(f"  Val: 10 samples")
    print(f"  Test: 10 samples")


if __name__ == "__main__":
    # Create dummy dataset for testing
    create_dummy_dataset('./data/processed/dummy', n_samples=100)
    
    # Test dataset
    dataset = EmotionDataset(
        data_dir='./data/processed/dummy',
        split='train',
        language='english'
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Speakers: {len(dataset.speaker_to_id)}")
    
    # Test loading a sample
    sample = dataset[0]
    print("\nSample keys:", sample.keys())
    print(f"Waveform shape: {sample['waveform'].shape}")
    print(f"Prosody shape: {sample['prosody_frames'].shape}")
    print(f"AUs shape: {sample['aus'].shape}")
    print(f"Speaker ID: {sample['speaker_id']}")
    print(f"Language ID: {sample['lang_id']}")
    print(f"Label: {sample['label']}")
