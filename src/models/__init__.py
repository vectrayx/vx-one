"""VX-One engine (spec F2): multi-head network, losses, data, calibration,
training loop and System One inference contract."""

from .calibration import (fit_choice_temperature, fit_corn_temperature,
                          fit_noul_temperature, fit_temperature_scalar)
from .data import EmberBatcher, RunData, build_run
from .engine import VXOneEngine
from .feature_stage import FeatureStage
from .losses import (KendallMultiTask, choice_loss, class_weights, corn_loss,
                     corn_pmf, noul_loss)
from .network import SCALES, HeadSpec, VXOneNet, head_spec_for
from .train import TRAIN_TABLE, train_run
from .vocab import TextVocab

__all__ = [
    "VXOneNet", "HeadSpec", "head_spec_for", "SCALES",
    "FeatureStage", "TextVocab",
    "class_weights", "noul_loss", "choice_loss", "corn_loss", "corn_pmf",
    "KendallMultiTask",
    "fit_temperature_scalar", "fit_noul_temperature", "fit_choice_temperature",
    "fit_corn_temperature",
    "RunData", "build_run", "EmberBatcher",
    "VXOneEngine", "train_run", "TRAIN_TABLE",
]
