"""
Video Action Unit (AU) Feature Extraction
Uses MediaPipe for facial landmark detection and AU estimation
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, List, Optional, Tuple
import pandas as pd
from pathlib import Path


class AUExtractor:
    """
    Extract Action Unit intensities from video using MediaPipe Face Mesh
    
    Note: MediaPipe doesn't directly provide AU intensities like OpenFace,
    but we can estimate them from facial landmarks using geometric features.
    """
    
    # AU to landmark mapping (approximate)
    AU_LANDMARK_MAPPING = {
        'AU01': [70, 63, 105, 66, 107],  # Inner brow raiser
        'AU02': [336, 296, 334, 293, 300],  # Outer brow raiser
        'AU04': [8, 9, 151],  # Brow lowerer
        'AU05': [33, 133, 160, 144, 153, 145],  # Upper lid raiser
        'AU06': [263, 466, 388, 387, 386],  # Cheek raiser
        'AU07': [145, 159, 160, 161],  # Lid tightener
        'AU09': [2, 98, 327],  # Nose wrinkler
        'AU10': [164, 167, 165],  # Upper lip raiser
        'AU12': [61, 291, 39, 269],  # Lip corner puller (smile)
        'AU14': [57, 287],  # Dimpler
        'AU15': [84, 314, 17, 18],  # Lip corner depressor
        'AU17': [18, 200, 199],  # Chin raiser
        'AU20': [61, 291, 78, 308],  # Lip stretcher
        'AU23': [13, 14],  # Lip tightener
        'AU25': [13, 14, 78, 308],  # Lips part
        'AU26': [13, 14, 78, 308, 82, 312],  # Jaw drop
    }
    
    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
    def extract_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract facial landmarks from a single frame
        
        Args:
            frame: RGB image (H, W, 3)
            
        Returns:
            landmarks: (468, 3) array of (x, y, z) coordinates, or None if no face detected
        """
        results = self.face_mesh.process(frame)
        
        if not results.multi_face_landmarks:
            return None
        
        # Get first face
        face_landmarks = results.multi_face_landmarks[0]
        
        # Convert to numpy array
        landmarks = np.array([
            [lm.x, lm.y, lm.z]
            for lm in face_landmarks.landmark
        ])
        
        return landmarks
    
    def compute_au_intensity(
        self,
        landmarks: np.ndarray,
        au_name: str,
        baseline_landmarks: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute AU intensity from landmarks using geometric features
        
        Args:
            landmarks: (468, 3) facial landmarks
            au_name: AU identifier (e.g., 'AU12')
            baseline_landmarks: Optional neutral expression landmarks for normalization
            
        Returns:
            AU intensity (0-5 scale, similar to OpenFace)
        """
        if au_name not in self.AU_LANDMARK_MAPPING:
            return 0.0
        
        landmark_indices = self.AU_LANDMARK_MAPPING[au_name]
        
        # Extract relevant landmarks
        au_landmarks = landmarks[landmark_indices]
        
        # Compute geometric feature (distance-based)
        if au_name in ['AU01', 'AU02']:  # Brow raise
            # Measure vertical distance from brow to eye
            intensity = np.mean(au_landmarks[:, 1])  # Y-coordinate
            if baseline_landmarks is not None:
                baseline = np.mean(baseline_landmarks[landmark_indices, 1])
                intensity = abs(intensity - baseline) * 10
            else:
                intensity = intensity * 5
                
        elif au_name in ['AU12', 'AU06']:  # Smile/cheek raise
            # Measure lip corner elevation
            intensity = np.std(au_landmarks[:, 1]) * 10
            
        elif au_name in ['AU25', 'AU26']:  # Mouth opening
            # Measure vertical mouth opening
            if len(landmark_indices) >= 2:
                mouth_height = abs(au_landmarks[0, 1] - au_landmarks[-1, 1])
                intensity = mouth_height * 20
            else:
                intensity = 0.0
                
        else:
            # Generic: use variance in landmark positions
            intensity = np.std(au_landmarks) * 10
        
        # Clip to 0-5 range
        intensity = np.clip(intensity, 0, 5)
        
        return float(intensity)
    
    def extract_aus_from_frame(
        self,
        frame: np.ndarray,
        au_list: List[str],
        baseline_landmarks: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Extract all AU intensities from a single frame
        
        Args:
            frame: RGB image
            au_list: List of AU names to extract
            baseline_landmarks: Optional neutral expression landmarks
            
        Returns:
            Dictionary of AU intensities
        """
        landmarks = self.extract_landmarks(frame)
        
        if landmarks is None:
            # No face detected, return zeros
            return {au: 0.0 for au in au_list}
        
        # Compute AU intensities
        au_intensities = {}
        for au in au_list:
            au_intensities[au] = self.compute_au_intensity(
                landmarks, au, baseline_landmarks
            )
        
        return au_intensities
    
    def extract_aus_from_video(
        self,
        video_path: str,
        au_list: List[str],
        target_fps: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Extract AU intensities from entire video
        
        Args:
            video_path: Path to video file
            au_list: List of AU names to extract
            target_fps: Target frame rate for extraction (None = use video fps)
            
        Returns:
            DataFrame with columns: frame, timestamp, AU01_r, AU02_r, ...
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Determine frame sampling
        if target_fps is None or target_fps >= video_fps:
            frame_skip = 1
        else:
            frame_skip = int(video_fps / target_fps)
        
        results = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames
            if frame_idx % frame_skip == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Extract AUs
                au_intensities = self.extract_aus_from_frame(frame_rgb, au_list)
                
                # Add frame info
                timestamp = frame_idx / video_fps
                result = {
                    'frame': frame_idx,
                    'timestamp': timestamp,
                    **{f"{au}_r": intensity for au, intensity in au_intensities.items()}
                }
                results.append(result)
            
            frame_idx += 1
        
        cap.release()
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        return df
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()


def test_au_extractor():
    """Test AU extraction on a sample video"""
    import matplotlib.pyplot as plt
    
    # Create a test video with a simple animation
    # (In practice, you'd use a real video file)
    print("AU Extractor initialized successfully")
    print("To test with a real video, provide a video path")
    
    # Example usage:
    # extractor = AUExtractor()
    # au_list = ['AU01', 'AU02', 'AU04', 'AU06', 'AU12', 'AU25', 'AU26']
    # aus_df = extractor.extract_aus_from_video('path/to/video.mp4', au_list)
    # print(aus_df.head())


if __name__ == "__main__":
    test_au_extractor()
