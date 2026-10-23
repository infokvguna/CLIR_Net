"""
Dataset Preprocessing Script for CLIR-Net
Extracts audio waveforms, prosodic features, and video Action Units (AUs)
from raw emotion datasets and prepares standardized train/val/test splits.

Supported datasets:
- EMO-DB (German)
- CREMA-D (English, Audio-Visual)
- EMOVO (Italian)
- ShEMO (Persian)
- CASIA (Chinese)
"""

import os
import glob
import json
import torch
import argparse
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from preprocessing.audio_features import AudioFeatureExtractor
from data_loaders.download import DatasetDownloader


def _save_splits(metadata, processed_dir: Path):
    """Split metadata into train (70%), val (15%), test (15%) and save JSONs"""
    if not metadata:
        print("No samples found.")
        return
        
    train_meta, test_meta = train_test_split(
        metadata, test_size=0.2, random_state=42,
        stratify=[m['emotion'] for m in metadata]
    )
    val_meta, test_meta = train_test_split(
        test_meta, test_size=0.5, random_state=42,
        stratify=[m['emotion'] for m in test_meta]
    )
    
    print(f"Split: Train={len(train_meta)}, Val={len(val_meta)}, Test={len(test_meta)}")
    
    with open(processed_dir / "train_metadata.json", "w") as f:
        json.dump(train_meta, f, indent=2)
    with open(processed_dir / "val_metadata.json", "w") as f:
        json.dump(val_meta, f, indent=2)
    with open(processed_dir / "test_metadata.json", "w") as f:
        json.dump(test_meta, f, indent=2)
    print(f"Preprocessing complete! Data saved to {processed_dir}")


def preprocess_emodb(data_root: Path):
    """
    Preprocess EMO-DB dataset
    Structure:
        data_root/EmoDB/wav/*.wav or data_root/raw/emodb/wav/*.wav
    Filename format:
        03a01Fa.wav
        03: Speaker ID
        a01: Text ID
        F: Emotion (W=Anger, L=Boredom, E=Disgust, A=Fear, F=Happiness, T=Sadness, N=Neutral)
        a: Version
    """
    raw_dir = data_root / "EmoDB"
    if not raw_dir.exists():
        raw_dir = data_root / "raw" / "emodb"
    
    processed_dir = data_root / "processed" / "emodb"
    features_dir = processed_dir / "features"
    
    # Check if raw data exists
    wav_dir = raw_dir / "wav" if (raw_dir / "wav").exists() else raw_dir
    wav_files = list(wav_dir.glob("*.wav"))
    
    if not wav_files:
        print(f"Raw data not found at {wav_dir}")
        print("Attempting to download EMO-DB...")
        downloader = DatasetDownloader(str(data_root))
        downloader.download_emodb()
        
        wav_dir = data_root / "raw" / "emodb" / "wav"
        if not wav_dir.exists():
            wav_dir = data_root / "raw" / "emodb"
        wav_files = list(wav_dir.glob("*.wav"))
        if not wav_files:
            print("Could not find audio files for EMO-DB. Please verify raw data.")
            return
    
    features_dir.mkdir(parents=True, exist_ok=True)
    extractor = AudioFeatureExtractor()
    
    emotion_map = {
        'W': 'anger',
        'L': 'neutral',  # Mapping Boredom to Neutral
        'E': 'disgust',
        'A': 'fear',
        'F': 'happiness',
        'T': 'sadness',
        'N': 'neutral'
    }
    
    print(f"Found {len(wav_files)} EMO-DB audio files")
    metadata = []
    
    for wav_path in tqdm(wav_files, desc="EMO-DB"):
        filename = wav_path.stem
        speaker_id = filename[:2]
        emotion_code = filename[5]
        
        if emotion_code not in emotion_map:
            continue
            
        emotion = emotion_map[emotion_code]
        
        try:
            features = extractor.extract_all_features(str(wav_path))
            output_path = features_dir / f"{filename}.pt"
            torch.save(features, output_path)
            
            metadata.append({
                'id': filename,
                'speaker_id': speaker_id,
                'emotion': emotion,
                'features_path': f"features/{filename}.pt",
                'duration': features['duration']
            })
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    _save_splits(metadata, processed_dir)


def preprocess_cremad(data_root: Path):
    """
    Preprocess CREMA-D dataset (Audio-Visual)
    Structure:
        data_root/CREMA-D/AudioWAV/*.wav
        data_root/CREMA-D/VideoFlash/*.flv (optional video)
    """
    raw_dir = data_root / "CREMA-D"
    if not raw_dir.exists():
        raw_dir = data_root / "raw" / "cremad"
    
    processed_dir = data_root / "processed" / "cremad"
    features_dir = processed_dir / "features"
    
    audio_wav_dir = raw_dir / "AudioWAV" if (raw_dir / "AudioWAV").exists() else raw_dir
    wav_files = list(audio_wav_dir.glob("*.wav"))
    
    if not wav_files:
        print(f"CREMA-D AudioWAV not found at {audio_wav_dir}")
        print("Please download CREMA-D from https://github.com/CheyneyComputerScience/CREMA-D")
        return

    features_dir.mkdir(parents=True, exist_ok=True)
    audio_extractor = AudioFeatureExtractor()
    
    # Optional video extraction
    has_video_support = False
    try:
        from preprocessing.video_features import AUExtractor
        video_extractor = AUExtractor()
        has_video_support = True
    except Exception:
        video_extractor = None

    emotion_map = {
        'ANG': 'anger',
        'DIS': 'disgust',
        'FEA': 'fear',
        'HAP': 'happiness',
        'NEU': 'neutral',
        'SAD': 'sadness'
    }
    
    print(f"Found {len(wav_files)} CREMA-D audio files")
    metadata = []
    
    for wav_path in tqdm(wav_files, desc="CREMA-D"):
        filename = wav_path.stem
        parts = filename.split('_')
        
        if len(parts) < 3:
            continue
            
        speaker_id = parts[0]
        emotion_code = parts[2]
        
        if emotion_code not in emotion_map:
            continue
            
        emotion = emotion_map[emotion_code]
        
        try:
            features = audio_extractor.extract_all_features(str(wav_path))
            
            # Check for corresponding video
            video_path = None
            if has_video_support:
                for ext in ['.flv', '.mp4', '.avi']:
                    v_path = raw_dir / "VideoFlash" / f"{filename}{ext}"
                    if v_path.exists():
                        video_path = v_path
                        break
            
            if video_path and has_video_support:
                au_list = ['AU01', 'AU02', 'AU04', 'AU05', 'AU06', 'AU07', 'AU09', 'AU10', 
                           'AU12', 'AU14', 'AU15', 'AU17', 'AU20', 'AU23', 'AU25', 'AU26']
                aus_df = video_extractor.extract_aus_from_video(str(video_path), au_list, target_fps=30)
                aus_tensor = torch.tensor(aus_df[[f"{au}_r" for au in au_list]].values, dtype=torch.float32)
                features['aus'] = aus_tensor
            
            output_path = features_dir / f"{filename}.pt"
            torch.save(features, output_path)
            
            metadata.append({
                'id': filename,
                'speaker_id': speaker_id,
                'emotion': emotion,
                'features_path': f"features/{filename}.pt",
                'duration': features['duration'],
                'has_video': video_path is not None
            })
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
    _save_splits(metadata, processed_dir)


def preprocess_emovo(data_root: Path):
    """Preprocess EMOVO (Italian) dataset"""
    raw_dir = data_root / "EMOVO"
    if not raw_dir.exists():
        raw_dir = data_root / "raw" / "emovo"
    
    processed_dir = data_root / "processed" / "emovo"
    features_dir = processed_dir / "features"
    
    if not raw_dir.exists():
        print(f"EMOVO not found at {raw_dir}")
        print("Attempting to download...")
        downloader = DatasetDownloader(str(data_root))
        downloader.download_emovo()
        if not raw_dir.exists():
            return

    features_dir.mkdir(parents=True, exist_ok=True)
    extractor = AudioFeatureExtractor()
    
    # EMOVO emotions: rab=anger, dis=disgust, pau=fear, gio=joy, tri=sadness, neu=neutral
    emotion_map = {
        'rab': 'anger',
        'dis': 'disgust',
        'pau': 'fear',
        'gio': 'happiness',
        'tri': 'sadness',
        'neu': 'neutral'
    }
    
    wav_files = list(raw_dir.rglob("*.wav"))
    print(f"Found {len(wav_files)} EMOVO audio files")
    
    metadata = []
    for wav_path in tqdm(wav_files, desc="EMOVO"):
        filename = wav_path.stem
        parts = filename.split('-')
        if len(parts) < 2:
            parts = filename.split('_')
        if len(parts) < 2:
            continue
            
        emotion_code = parts[0]
        speaker_id = parts[1]
        
        if emotion_code not in emotion_map:
            continue
            
        emotion = emotion_map[emotion_code]
        
        try:
            features = extractor.extract_all_features(str(wav_path))
            output_path = features_dir / f"{filename}.pt"
            torch.save(features, output_path)
            
            metadata.append({
                'id': filename,
                'speaker_id': speaker_id,
                'emotion': emotion,
                'features_path': f"features/{filename}.pt",
                'duration': features['duration']
            })
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    _save_splits(metadata, processed_dir)


def preprocess_shemo(data_root: Path):
    """Preprocess ShEMO (Persian) dataset"""
    raw_dir = data_root / "ShEMO"
    if not raw_dir.exists():
        raw_dir = data_root / "raw" / "shemo"
        
    processed_dir = data_root / "processed" / "shemo"
    features_dir = processed_dir / "features"
    
    if not raw_dir.exists():
        print(f"ShEMO not found at {raw_dir}")
        print("Attempting to download...")
        downloader = DatasetDownloader(str(data_root))
        downloader.download_shemo()
        if not raw_dir.exists():
            return

    features_dir.mkdir(parents=True, exist_ok=True)
    extractor = AudioFeatureExtractor()
    
    # ShEMO: A=Anger, F=Fear, H=Happiness, N=Neutral, S=Sadness
    emotion_map = {
        'A': 'anger',
        'F': 'fear',
        'H': 'happiness',
        'N': 'neutral',
        'S': 'sadness'
    }
    
    wav_files = list(raw_dir.rglob("*.wav"))
    metadata = []
    
    for wav_path in tqdm(wav_files, desc="ShEMO"):
        filename = wav_path.stem
        if len(filename) < 4:
            continue
        
        speaker_id = filename[:3]
        emotion_code = filename[3]
        
        if emotion_code not in emotion_map:
            continue
        emotion = emotion_map[emotion_code]
        
        try:
            features = extractor.extract_all_features(str(wav_path))
            output_path = features_dir / f"{filename}.pt"
            torch.save(features, output_path)
            metadata.append({
                'id': filename,
                'speaker_id': speaker_id,
                'emotion': emotion,
                'features_path': f"features/{filename}.pt",
                'duration': features['duration']
            })
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    _save_splits(metadata, processed_dir)


def preprocess_casia(data_root: Path):
    """Preprocess CASIA (Chinese) dataset"""
    raw_dir = data_root / "CASIA"
    if not raw_dir.exists():
        raw_dir = data_root / "raw" / "casia"
        
    processed_dir = data_root / "processed" / "casia"
    features_dir = processed_dir / "features"
    
    if not raw_dir.exists():
        print(f"CASIA not found at {raw_dir}")
        return

    features_dir.mkdir(parents=True, exist_ok=True)
    extractor = AudioFeatureExtractor()
    
    emotion_map = {
        'angry': 'anger',
        'fear': 'fear',
        'happy': 'happiness',
        'neutral': 'neutral',
        'sad': 'sadness'
    }
    
    wav_files = list(raw_dir.rglob("*.wav"))
    metadata = []
    
    for wav_path in tqdm(wav_files, desc="CASIA"):
        filename = wav_path.stem
        emotion_folder = wav_path.parent.name.lower()
        
        found_emotion = None
        for k, v in emotion_map.items():
            if k in emotion_folder:
                found_emotion = v
                break
        
        if not found_emotion:
            continue
        
        speaker_id = wav_path.parent.parent.name if wav_path.parent.parent != raw_dir else "speaker_001"
        
        try:
            features = extractor.extract_all_features(str(wav_path))
            output_path = features_dir / f"{filename}.pt"
            torch.save(features, output_path)
            metadata.append({
                'id': filename,
                'speaker_id': speaker_id,
                'emotion': found_emotion,
                'features_path': f"features/{filename}.pt",
                'duration': features['duration']
            })
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    _save_splits(metadata, processed_dir)


def main():
    parser = argparse.ArgumentParser(description="Preprocess emotion datasets for CLIR-Net")
    parser.add_argument("--dataset", type=str, default="emodb",
                        help="Dataset to preprocess (emodb, cremad, emovo, shemo, casia, all)")
    parser.add_argument("--data_root", type=str, default="./data",
                        help="Root directory containing datasets")
    
    args = parser.parse_args()
    data_root = Path(args.data_root)
    
    if args.dataset == 'all':
        datasets = ['emodb', 'cremad', 'emovo', 'shemo', 'casia']
    else:
        datasets = [args.dataset]
        
    for ds in datasets:
        print(f"\nProcessing {ds}...")
        if ds == "emodb":
            preprocess_emodb(data_root)
        elif ds == "cremad":
            preprocess_cremad(data_root)
        elif ds == "emovo":
            preprocess_emovo(data_root)
        elif ds == "shemo":
            preprocess_shemo(data_root)
        elif ds == "casia":
            preprocess_casia(data_root)
        else:
            print(f"Unknown dataset: {ds}")


if __name__ == "__main__":
    main()
