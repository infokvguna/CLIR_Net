"""
Main Training Script for CLIR-Net
(Cross-Lingual Invariant Representation Network)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import argparse
from typing import Dict, Optional
import random

# Import CLIR-Net components
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.clirnet import CLIRNet
from training.losses import CLIRNetLoss
from data_loaders.base_dataset import EmotionDataset, create_dummy_dataset
from config import get_config


class Trainer:
    """CLIR-Net Trainer"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        device: str = 'cuda',
        checkpoint_dir: str = './checkpoints',
        log_every: int = 10,
        val_every: int = 1,
        patience: int = 15
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_every = log_every
        self.val_every = val_every
        self.patience = patience
        
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        total_cls_loss = 0
        total_inv_loss = 0
        total_cf_pred_loss = 0
        total_acyc_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            waveform = batch['waveform'].to(self.device)
            prosody = batch['prosody_frames'].to(self.device)
            aus = batch['aus'].to(self.device)
            speaker_ids = batch['speaker_id'].to(self.device)
            lang_ids = batch['lang_id'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Sample counterfactual language (random different language)
            batch_size = waveform.size(0)
            cf_lang_ids = torch.randint(0, 5, (batch_size,), device=self.device)
            # Ensure different from original
            same_mask = cf_lang_ids == lang_ids
            if same_mask.any():
                cf_lang_ids[same_mask] = (lang_ids[same_mask] + 1) % 5
            
            # Forward pass with counterfactual intervention
            outputs = self.model(
                waveform=waveform,
                prosody_frames=prosody,
                aus=aus,
                speaker_ids=speaker_ids,
                lang_ids=lang_ids,
                return_counterfactual=True,
                cf_lang_ids=cf_lang_ids
            )
            
            # Compute acyclicity penalty
            acyc_penalty = self.model.compute_acyclicity_penalty()
            
            # Compute losses
            losses = self.criterion(outputs, labels, acyc_penalty)
            
            # Backward
            self.optimizer.zero_grad()
            losses['loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Statistics
            total_loss += losses['loss'].item()
            total_cls_loss += losses['loss_cls'].item()
            total_inv_loss += losses['loss_inv'].item()
            total_cf_pred_loss += losses['loss_cf_pred'].item()
            total_acyc_loss += losses['loss_acyc'].item()
            
            # Accuracy
            _, predicted = outputs['logits'].max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            if batch_idx % self.log_every == 0:
                pbar.set_postfix({
                    'loss': losses['loss'].item(),
                    'acc': 100. * correct / total
                })
        
        # Epoch statistics
        n_batches = len(self.train_loader)
        return {
            'loss': total_loss / n_batches,
            'loss_cls': total_cls_loss / n_batches,
            'loss_inv': total_inv_loss / n_batches,
            'loss_cf_pred': total_cf_pred_loss / n_batches,
            'loss_acyc': total_acyc_loss / n_batches,
            'acc': 100. * correct / total
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set"""
        self.model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in tqdm(self.val_loader, desc='Validation'):
            # Move to device
            waveform = batch['waveform'].to(self.device)
            prosody = batch['prosody_frames'].to(self.device)
            aus = batch['aus'].to(self.device)
            speaker_ids = batch['speaker_id'].to(self.device)
            lang_ids = batch['lang_id'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass (no counterfactual for validation)
            outputs = self.model(
                waveform=waveform,
                prosody_frames=prosody,
                aus=aus,
                speaker_ids=speaker_ids,
                lang_ids=lang_ids,
                return_counterfactual=False
            )
            
            # Compute loss
            losses = self.criterion(outputs, labels)
            total_loss += losses['loss'].item()
            
            # Accuracy
            _, predicted = outputs['logits'].max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        return {
            'loss': total_loss / len(self.val_loader),
            'acc': 100. * correct / total
        }
    
    def train(self, num_epochs: int):
        """Full training loop"""
        print(f"Starting CLIR-Net training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M")
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch)
            self.history['train_loss'].append(train_metrics['loss'])
            
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Train Acc: {train_metrics['acc']:.2f}%")
            print(f"  Loss breakdown:")
            print(f"    Classification: {train_metrics['loss_cls']:.4f}")
            print(f"    Invariance: {train_metrics['loss_inv']:.4f}")
            print(f"    CF Prediction: {train_metrics['loss_cf_pred']:.4f}")
            print(f"    Acyclicity: {train_metrics['loss_acyc']:.4f}")
            
            # Validate
            if epoch % self.val_every == 0:
                val_metrics = self.validate()
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_acc'].append(val_metrics['acc'])
                
                print(f"  Val Loss: {val_metrics['loss']:.4f}")
                print(f"  Val Acc: {val_metrics['acc']:.2f}%")
                
                # Learning rate scheduling
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                elif self.scheduler is not None:
                    self.scheduler.step()
                
                # Save best model
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.epochs_without_improvement = 0
                    self.save_checkpoint(epoch, is_best=True)
                    print(f"  [BEST] New best model saved!")
                else:
                    self.epochs_without_improvement += 1
                
                # Early stopping
                if self.epochs_without_improvement >= self.patience:
                    print(f"\nEarly stopping after {epoch} epochs")
                    break
            
            # Save regular checkpoint
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, is_best=False)
        
        # Save training history
        self.save_history()
        print("\nTraining completed!")
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }
        
        if is_best:
            path = self.checkpoint_dir / 'best_model.pt'
        else:
            path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        
        torch.save(checkpoint, path)
    
    def save_history(self):
        """Save training history"""
        with open(self.checkpoint_dir / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description='Train CLIR-Net')
    parser.add_argument('--data_dir', type=str, default='./data/processed/dummy',
                        help='Data directory')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--create_dummy', action='store_true', help='Create dummy dataset')
    parser.add_argument('--pooled', action='store_true', help='Train on pooled multilingual datasets')
    parser.add_argument('--audio_only', action='store_true', help='Train AUDIO-SCM (ignore video)')
    parser.add_argument('--language', type=str, default='english', help='Language for single dataset')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create dummy dataset if requested
    if args.create_dummy:
        print("Creating dummy dataset...")
        create_dummy_dataset(args.data_dir, n_samples=100)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load datasets
    print("Loading datasets...")
    
    if args.pooled:
        from data_loaders.base_dataset import MultiCorpusDataset
        
        datasets_list = ['emodb', 'cremad', 'emovo', 'shemo', 'casia']
        train_datasets = []
        val_datasets = []
        
        for ds_name in datasets_list:
            ds_path = Path(args.data_dir).parent / ds_name
            if not ds_path.exists():
                print(f"Warning: Dataset {ds_name} not found at {ds_path}, skipping.")
                continue
                
            # Determine language
            lang = 'english'
            if ds_name == 'emodb': lang = 'german'
            elif ds_name == 'emovo': lang = 'italian'
            elif ds_name == 'shemo': lang = 'persian'
            elif ds_name == 'casia': lang = 'chinese'
            
            try:
                train_ds = EmotionDataset(ds_path, split='train', language=lang)
                val_ds = EmotionDataset(ds_path, split='val', language=lang)
                train_datasets.append(train_ds)
                val_datasets.append(val_ds)
                print(f"Loaded {ds_name}: {len(train_ds)} train, {len(val_ds)} val")
            except Exception as e:
                print(f"Failed to load {ds_name}: {e}")
        
        if not train_datasets:
            raise ValueError("No datasets loaded for pooled training!")
            
        train_dataset = MultiCorpusDataset(train_datasets)
        val_dataset = MultiCorpusDataset(val_datasets)
        
        total_speakers = sum(len(ds.speaker_to_id) for ds in train_datasets)
        print(f"Total pooled speakers: {total_speakers}")
        
    else:
        # Single dataset
        train_dataset = EmotionDataset(args.data_dir, split='train', language=args.language)
        val_dataset = EmotionDataset(args.data_dir, split='val', language=args.language)
        total_speakers = len(train_dataset.speaker_to_id)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # 0 for cross-platform stability
        pin_memory=True if device == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device == 'cuda' else False
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create model
    print("Creating CLIR-Net model...")
    model = CLIRNet(
        audio_pretrained="facebook/wav2vec2-base",
        audio_freeze_layers=6,
        n_speakers=total_speakers,
        n_languages=5,
        n_classes=6,
        learn_structure=True,
        audio_only=args.audio_only
    ).to(device)
    
    # Loss
    criterion = CLIRNetLoss(
        alpha_inv=1.0,
        beta_cf_pred=1.0,
        gamma_acyc=1e-2
    )
    
    # Optimizer params
    params = [
        {'params': model.audio_encoder.parameters(), 'lr': args.lr * 0.1},
        {'params': model.scm.parameters(), 'lr': args.lr},
        {'params': model.emotion_head.parameters(), 'lr': args.lr},
        {'params': model.speaker_embedding.parameters(), 'lr': args.lr},
        {'params': model.language_embedding.parameters(), 'lr': args.lr}
    ]
    
    if not args.audio_only:
        params.append({'params': model.au_encoder.parameters(), 'lr': args.lr * 0.1})
    
    optimizer = AdamW(
        params,
        weight_decay=1e-4
    )
    
    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    
    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        log_every=10,
        val_every=1,
        patience=15
    )
    
    # Train
    trainer.train(args.num_epochs)


if __name__ == "__main__":
    main()
