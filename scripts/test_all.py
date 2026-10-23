"""
Quick Test Script for CLIR-Net
Tests all components end-to-end
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.clirnet import CLIRNet
from training.losses import CLIRNetLoss
from data_loaders.base_dataset import create_dummy_dataset, EmotionDataset
from torch.utils.data import DataLoader


def test_preprocessing():
    """Test preprocessing modules"""
    print("\n" + "="*60)
    print("Testing Preprocessing Modules")
    print("="*60)
    
    try:
        from preprocessing.audio_features import AudioFeatureExtractor
        extractor = AudioFeatureExtractor()
        print("[OK] Audio feature extractor initialized")
    except Exception as e:
        print(f"[FAIL] Audio feature extractor failed: {e}")
    
    try:
        from preprocessing.video_features import AUExtractor
        au_extractor = AUExtractor()
        print("[OK] AU extractor initialized")
    except Exception as e:
        print(f"[FAIL] AU extractor failed: {e}")


def test_model():
    """Test model architecture"""
    print("\n" + "="*60)
    print("Testing CLIR-Net Model Architecture")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Create dummy inputs
    batch_size = 2
    T_wave = 16000 * 3
    T_frames = 300
    prosody_dim = 6
    au_dim = 16
    
    waveform = torch.randn(batch_size, T_wave).to(device)
    prosody = torch.randn(batch_size, T_frames, prosody_dim).to(device)
    aus = torch.randn(batch_size, T_frames, au_dim).to(device)
    speaker_ids = torch.randint(0, 10, (batch_size,)).to(device)
    lang_ids = torch.randint(0, 5, (batch_size,)).to(device)
    cf_lang_ids = torch.randint(0, 5, (batch_size,)).to(device)
    
    # Create model
    model = CLIRNet(
        n_speakers=10,
        n_languages=5,
        n_classes=6,
        learn_structure=True
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Forward pass
    try:
        with torch.no_grad():
            outputs = model(
                waveform, prosody, aus, speaker_ids, lang_ids,
                return_counterfactual=True,
                cf_lang_ids=cf_lang_ids
            )
        
        print(f"[OK] Forward pass successful")
        print(f"  Logits shape: {outputs['logits'].shape}")
        print(f"  E shape: {outputs['E'].shape}")
        print(f"  E_cf shape: {outputs['E_cf'].shape}")
        
        # Test causal structure
        adj = model.get_causal_structure()
        print(f"[OK] Causal structure shape: {adj.shape}")
        
        # Test acyclicity
        h = model.compute_acyclicity_penalty()
        print(f"[OK] Acyclicity penalty: {h.item():.4f}")
        
    except Exception as e:
        print(f"[FAIL] Forward pass failed: {e}")
        import traceback
        traceback.print_exc()


def test_losses():
    """Test loss functions"""
    print("\n" + "="*60)
    print("Testing CLIR-Net Loss Functions")
    print("="*60)
    
    batch_size = 4
    n_classes = 6
    latent_dim = 256
    
    # Create dummy outputs
    outputs = {
        'logits': torch.randn(batch_size, n_classes),
        'E': torch.randn(batch_size, latent_dim),
        'E_cf': torch.randn(batch_size, latent_dim),
        'logits_cf': torch.randn(batch_size, n_classes)
    }
    labels = torch.randint(0, n_classes, (batch_size,))
    acyc_penalty = torch.tensor(0.5)
    
    # Create loss
    criterion = CLIRNetLoss()
    
    try:
        losses = criterion(outputs, labels, acyc_penalty)
        print(f"[OK] Loss computation successful")
        print(f"  Total loss: {losses['loss'].item():.4f}")
        print(f"  Classification: {losses['loss_cls'].item():.4f}")
        print(f"  Invariance: {losses['loss_inv'].item():.4f}")
        print(f"  CF prediction: {losses['loss_cf_pred'].item():.4f}")
        print(f"  Acyclicity: {losses['loss_acyc'].item():.4f}")
    except Exception as e:
        print(f"[FAIL] Loss computation failed: {e}")


def test_dataset():
    """Test dataset loading"""
    print("\n" + "="*60)
    print("Testing Dataset")
    print("="*60)
    
    # Create dummy dataset
    data_dir = './data/processed/test_dummy'
    print(f"Creating dummy dataset at {data_dir}...")
    
    try:
        create_dummy_dataset(data_dir, n_samples=20)
        print("[OK] Dummy dataset created")
        
        # Load dataset
        dataset = EmotionDataset(data_dir, split='train', language='english')
        print(f"[OK] Dataset loaded: {len(dataset)} samples")
        
        # Test dataloader
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        batch = next(iter(loader))
        
        print(f"[OK] DataLoader working")
        print(f"  Batch keys: {batch.keys()}")
        print(f"  Waveform shape: {batch['waveform'].shape}")
        print(f"  Prosody shape: {batch['prosody_frames'].shape}")
        print(f"  AUs shape: {batch['aus'].shape}")
        
    except Exception as e:
        print(f"[FAIL] Dataset test failed: {e}")
        import traceback
        traceback.print_exc()


def test_training_step():
    """Test a single training step"""
    print("\n" + "="*60)
    print("Testing CLIR-Net Training Step")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create dummy dataset
    data_dir = './data/processed/test_dummy'
    if not Path(data_dir).exists():
        create_dummy_dataset(data_dir, n_samples=20)
    
    try:
        # Load data
        dataset = EmotionDataset(data_dir, split='train', language='english')
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        batch = next(iter(loader))
        
        # Create model
        model = CLIRNet(
            n_speakers=len(dataset.speaker_to_id),
            n_languages=5,
            n_classes=6
        ).to(device)
        
        # Create optimizer and loss
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        criterion = CLIRNetLoss()
        
        # Move batch to device
        waveform = batch['waveform'].to(device)
        prosody = batch['prosody_frames'].to(device)
        aus = batch['aus'].to(device)
        speaker_ids = batch['speaker_id'].to(device)
        lang_ids = batch['lang_id'].to(device)
        labels = batch['label'].to(device)
        
        # Sample CF language
        cf_lang_ids = torch.randint(0, 5, (waveform.size(0),), device=device)
        
        # Forward
        outputs = model(
            waveform, prosody, aus, speaker_ids, lang_ids,
            return_counterfactual=True,
            cf_lang_ids=cf_lang_ids
        )
        
        # Loss
        acyc_penalty = model.compute_acyclicity_penalty()
        losses = criterion(outputs, labels, acyc_penalty)
        
        # Backward
        optimizer.zero_grad()
        losses['loss'].backward()
        optimizer.step()
        
        print("[OK] Training step successful")
        print(f"  Loss: {losses['loss'].item():.4f}")
        
        # Check predictions
        _, predicted = outputs['logits'].max(1)
        acc = predicted.eq(labels).sum().item() / labels.size(0)
        print(f"  Accuracy: {acc*100:.2f}%")
        
    except Exception as e:
        print(f"[FAIL] Training step failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("CLIR-Net Component Tests")
    print("="*60)
    
    test_preprocessing()
    test_model()
    test_losses()
    test_dataset()
    test_training_step()
    
    print("\n" + "="*60)
    print("All Tests Completed!")
    print("="*60)
    print("\nTo train the model, run:")
    print("  python scripts/train_clirnet.py --create_dummy --num_epochs 10")


if __name__ == "__main__":
    main()
