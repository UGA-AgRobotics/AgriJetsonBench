from __future__ import annotations

import logging
import os


def _quiet_noisy_optional_runtime_imports() -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    logging.getLogger("torchao").setLevel(logging.ERROR)
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    logging.getLogger("absl").setLevel(logging.ERROR)


def _patch_torch_pytree() -> None:
    try:
        from torch.utils import _pytree
    except Exception:
        return

    if hasattr(_pytree, "register_pytree_node"):
        return

    legacy_register = getattr(_pytree, "_register_pytree_node", None)
    if legacy_register is None:
        return

    def register_pytree_node(
        typ,
        flatten_fn,
        unflatten_fn,
        *,
        to_dumpable_context=None,
        from_dumpable_context=None,
        serialized_type_name=None,
        flatten_with_keys_fn=None,
    ):
        # PyTorch 2.1 lacks the metadata kwargs newer transformers versions pass here.
        del serialized_type_name, flatten_with_keys_fn
        return legacy_register(
            typ,
            flatten_fn,
            unflatten_fn,
            to_dumpable_context=to_dumpable_context,
            from_dumpable_context=from_dumpable_context,
        )

    _pytree.register_pytree_node = register_pytree_node


def _patch_pillow_for_large_training_images() -> None:
    try:
        from PIL import Image
    except Exception:
        return

    # The training dataset contains trusted ultra-high-resolution images.
    # RT-DETR uses PIL-backed crop/resize transforms, so Pillow's default
    # decompression-bomb guard aborts the dataloader before transforms can
    # downscale the sample.
    Image.MAX_IMAGE_PIXELS = None


_quiet_noisy_optional_runtime_imports()
_patch_torch_pytree()
_patch_pillow_for_large_training_images()
