from .builder import build_model
from .default import DefaultSegmentor, DefaultClassifier

# Backbones - wrapped in try/except for CPU-only environments missing some dependencies
def _safe_import(module_path):
    try:
        exec(f"from {module_path} import *", globals())
    except (ImportError, ModuleNotFoundError):
        pass

_safe_import(".sparse_unet")
_safe_import(".point_transformer")
_safe_import(".point_transformer_v2")
from .point_transformer_v3 import *
_safe_import(".stratified_transformer")
_safe_import(".spvcnn")
_safe_import(".octformer")
_safe_import(".oacnns")

# Semantic Segmentation
_safe_import(".context_aware_classifier")

# Instance Segmentation
_safe_import(".point_group")

# Pretraining
_safe_import(".masked_scene_contrast")
_safe_import(".point_prompt_training")
