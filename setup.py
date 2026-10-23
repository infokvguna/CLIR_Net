from setuptools import setup, find_packages

setup(
    name="clirnet",
    version="1.0.0",
    description="CLIR-Net: Cross-Lingual Invariant Representation Network for Speech and Audio-Visual Emotion Recognition",
    author="CLIR-Net Research Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.30.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "praat-parselmouth>=0.4.3",
        "opencv-python>=4.8.0",
        "mediapipe>=0.10.0",
        "numpy>=1.24.0,<2.0.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0"
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
