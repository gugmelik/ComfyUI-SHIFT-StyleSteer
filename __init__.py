import os

import folder_paths

# vectors live in ComfyUI/models/shift_vectors (listed by the Load node)
folder_paths.add_model_folder_path("shift_vectors", os.path.join(folder_paths.models_dir, "shift_vectors"), is_default=True)
if "shift_vectors" in folder_paths.folder_names_and_paths:
    paths, _ = folder_paths.folder_names_and_paths["shift_vectors"]
    folder_paths.folder_names_and_paths["shift_vectors"] = (paths, {".safetensors"})

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: E402

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
