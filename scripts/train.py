"""
Primary training entrypoint for CLIR-Net.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.train_clirnet import main

if __name__ == "__main__":
    main()
