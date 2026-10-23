"""
CLIR-Net Model Evaluation Script
Evaluates trained model on test set and generates comprehensive metrics
(Accuracy, Weighted Precision/Recall/F1, Per-Class F1, Confusion Matrix)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
import argparse
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.clirnet import CLIRNet
from data_loaders.base_dataset import EmotionDataset, MultiCorpusDataset


class ModelEvaluator:
    """Evaluate CLIR-Net model on test set"""
    
    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: str = 'cuda',
        emotion_labels: list = None
    ):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.emotion_labels = emotion_labels or [
            'anger', 'disgust', 'fear', 'happiness', 'sadness', 'neutral'
        ]
        
    @torch.no_grad()
    def evaluate(self):
        """Run evaluation on test set"""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        print("Evaluating CLIR-Net on test set...")
        for batch in tqdm(self.test_loader):
            # Move to device
            waveform = batch['waveform'].to(self.device)
            prosody = batch['prosody_frames'].to(self.device)
            aus = batch['aus'].to(self.device)
            speaker_ids = batch['speaker_id'].to(self.device)
            lang_ids = batch['lang_id'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                waveform=waveform,
                prosody_frames=prosody,
                aus=aus,
                speaker_ids=speaker_ids,
                lang_ids=lang_ids,
                return_counterfactual=False
            )
            
            # Get predictions
            logits = outputs['logits']
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            
            # Store results
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        return all_preds, all_labels, all_probs
    
    def compute_metrics(self, preds, labels):
        """Compute evaluation metrics"""
        # Overall accuracy
        accuracy = accuracy_score(labels, preds)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, preds, average=None, labels=range(len(self.emotion_labels))
        )
        
        # Weighted average
        precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(
            labels, preds, average='weighted'
        )
        
        # Confusion matrix
        cm = confusion_matrix(labels, preds, labels=range(len(self.emotion_labels)))
        
        metrics = {
            'accuracy': float(accuracy),
            'precision_weighted': float(precision_avg),
            'recall_weighted': float(recall_avg),
            'f1_weighted': float(f1_avg),
            'per_class': {
                self.emotion_labels[i]: {
                    'precision': float(precision[i]),
                    'recall': float(recall[i]),
                    'f1': float(f1[i]),
                    'support': int(support[i])
                }
                for i in range(len(self.emotion_labels))
            },
            'confusion_matrix': cm.tolist()
        }
        
        return metrics
    
    def plot_confusion_matrix(self, cm, save_path):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.emotion_labels,
            yticklabels=self.emotion_labels
        )
        plt.title('CLIR-Net Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved to {save_path}")
    
    def plot_per_class_metrics(self, metrics, save_path):
        """Plot per-class F1 scores"""
        emotions = list(metrics['per_class'].keys())
        f1_scores = [metrics['per_class'][e]['f1'] for e in emotions]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(emotions, f1_scores, color='steelblue')
        plt.xlabel('Emotion')
        plt.ylabel('F1 Score')
        plt.title('CLIR-Net Per-Class F1 Scores')
        plt.ylim(0, 1.0)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Per-class metrics saved to {save_path}")
    
    def print_results(self, metrics):
        """Print evaluation results"""
        print("\n" + "="*60)
        print("CLIR-Net EVALUATION RESULTS")
        print("="*60)
        print(f"\nOverall Accuracy: {metrics['accuracy']:.4f}")
        print(f"Weighted Precision: {metrics['precision_weighted']:.4f}")
        print(f"Weighted Recall: {metrics['recall_weighted']:.4f}")
        print(f"Weighted F1 Score: {metrics['f1_weighted']:.4f}")
        
        print("\nPer-Class Metrics:")
        print("-" * 60)
        print(f"{'Emotion':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<12}")
        print("-" * 60)
        for emotion, scores in metrics['per_class'].items():
            print(f"{emotion:<12} {scores['precision']:<12.4f} {scores['recall']:<12.4f} "
                  f"{scores['f1']:<12.4f} {scores['support']:<12}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Evaluate CLIR-Net Model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='./data/processed/dummy',
                        help='Data directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--pooled', action='store_true', help='Use pooled datasets')
    parser.add_argument('--audio_only', action='store_true', help='Audio-only model')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory for results')
    parser.add_argument('--language', type=str, default='english',
                        help='Language for single dataset')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load test dataset
    print("Loading test dataset...")
    
    if args.pooled:
        from data_loaders.base_dataset import MultiCorpusDataset
        
        datasets_list = ['emodb', 'cremad', 'emovo', 'shemo']
        test_datasets = []
        
        for ds_name in datasets_list:
            ds_path = Path(args.data_dir).parent / ds_name
            if not ds_path.exists():
                print(f"Warning: Dataset {ds_name} not found, skipping.")
                continue
                
            # Determine language
            lang = 'english'
            if ds_name == 'emodb': lang = 'german'
            elif ds_name == 'emovo': lang = 'italian'
            elif ds_name == 'shemo': lang = 'persian'
            
            try:
                test_ds = EmotionDataset(ds_path, split='test', language=lang)
                test_datasets.append(test_ds)
                print(f"Loaded {ds_name}: {len(test_ds)} test samples")
            except Exception as e:
                print(f"Failed to load {ds_name}: {e}")
        
        if not test_datasets:
            raise ValueError("No test datasets loaded!")
            
        test_dataset = MultiCorpusDataset(test_datasets)
        total_speakers = sum(len(ds.speaker_to_id) for ds in test_datasets)
        
    else:
        test_dataset = EmotionDataset(args.data_dir, split='test', language=args.language)
        total_speakers = len(test_dataset.speaker_to_id)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Load checkpoint first to get model configuration
    print("Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    
    # Get number of speakers from checkpoint (to match training)
    if 'model_state_dict' in checkpoint:
        # Extract n_speakers from the checkpoint's speaker_embedding weight shape
        speaker_emb_shape = checkpoint['model_state_dict']['speaker_embedding.weight'].shape
        n_speakers_from_checkpoint = speaker_emb_shape[0]
        print(f"Number of speakers in checkpoint: {n_speakers_from_checkpoint}")
    else:
        # Fallback to calculated value
        n_speakers_from_checkpoint = total_speakers
        print(f"Using calculated speakers: {n_speakers_from_checkpoint}")
    
    # Load model with correct speaker count
    print("Creating CLIR-Net model...")
    model = CLIRNet(
        audio_pretrained="facebook/wav2vec2-base",
        audio_freeze_layers=6,
        n_speakers=n_speakers_from_checkpoint,  # Use checkpoint value
        n_languages=5,
        n_classes=6,
        learn_structure=True,
        audio_only=args.audio_only
    ).to(device)
    
    # Load checkpoint weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from {args.checkpoint}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    
    # Create evaluator
    evaluator = ModelEvaluator(model, test_loader, device)
    
    # Run evaluation
    try:
        preds, labels, probs = evaluator.evaluate()
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Compute metrics
    metrics = evaluator.compute_metrics(preds, labels)
    
    # Print results
    evaluator.print_results(metrics)
    
    # Save metrics
    metrics_path = output_dir / 'test_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")
    
    # Plot confusion matrix
    cm = np.array(metrics['confusion_matrix'])
    cm_path = output_dir / 'confusion_matrix.png'
    evaluator.plot_confusion_matrix(cm, cm_path)
    
    # Plot per-class metrics
    metrics_plot_path = output_dir / 'per_class_f1.png'
    evaluator.plot_per_class_metrics(metrics, metrics_plot_path)
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
