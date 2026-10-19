"""Raíz de la suite VX-One: path del proyecto + seeds deterministas."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

torch.manual_seed(42)
np.random.seed(42)
