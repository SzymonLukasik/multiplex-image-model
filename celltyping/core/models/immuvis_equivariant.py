"""Cell-typing setup hook for the *equivariant* ConvNeXt multiplex autoencoder.

Mirrors ``core/models/immuvis.py`` but:

* loads ``EquivariantMultiplexAutoencoder`` (model_type dispatch, same logic as
  the training repo's ``load_model_for_evaluation.py``) instead of
  ``MultiplexAutoencoder``;
* builds the model config straight from the *training* YAML (``encoder`` /
  ``decoder`` blocks), not the ImmuVis-schema ``build_model_config_from_yaml``;
* calls ``model.encode_images(x, ch)`` (the equivariant API) rather than
  ``model.encode(x, ch)``;
* is fully self-contained for preprocessing (arcsinh -> butterworth ->
  clip-normalize) so it does **not** import ``core.utils.immuvis`` and therefore
  does not pull in OpenCV (``libGL`` is unavailable in the training venv).

Returned tuple matches the runner contract:
``(model, prepare_fn, get_channels, compute_fn, device)``.
"""

from pathlib import Path

import numpy as np
import torch
from ruamel.yaml import YAML
from skimage.filters import butterworth

from core.utils import load_yaml, add_model_repo


def _build_equivariant_model(repo_path, conf, num_channels):
    """Construct the model exactly like the training repo does, dispatching on
    ``model_type``. ``conf`` is the training YAML used to produce the
    checkpoint."""
    add_model_repo(repo_path)

    model_yaml = load_yaml(conf)
    model_type = model_yaml.get("model_type", "EquivariantConvnextV2")
    model_config = {
        "num_channels": num_channels,
        "encoder_config": model_yaml["encoder"],
        "decoder_config": model_yaml["decoder"],
    }

    if model_type == "FullyEquivariantConvnext":
        from train_masked_model_ddp_ed import FullyEquivariantMultiplexAutoencoder
        model = FullyEquivariantMultiplexAutoencoder(**model_config)
    elif model_type == "EquivariantConvnext":
        from multiplex_model.equivariant_modules import EquivariantMultiplexAutoencoder
        model = EquivariantMultiplexAutoencoder(**model_config)
    elif model_type == "EquivariantConvnextV2":
        from multiplex_model.equivariant_modules_v2 import (
            EquivariantMultiplexAutoencoder,
        )
        model = EquivariantMultiplexAutoencoder(**model_config)
    elif model_type == "Convnext":
        from multiplex_model.modules import MultiplexAutoencoder
        model = MultiplexAutoencoder(**model_config)
    else:
        raise ValueError(f"Unknown model_type in {conf}: {model_type}")

    return model, model_type


def _preprocess(
    img: torch.Tensor,
    dataset_name: str,
    clip_limits: dict | None,
    clip_upper_bound: float = 5.0,
) -> torch.Tensor:
    """arcsinh(x/5) -> butterworth low-pass -> clip[0, ub]/ub.

    Matches the ImmuVis preprocessing chain used at training time
    (``use_arcsinh``, ``use_butterworth_filter``, ``use_clip_normalization``;
    median denoising is OFF, so OpenCV is never needed). ``img`` is (C, H, W).
    """
    arr = img.float().cpu().numpy()
    arr = np.arcsinh(arr / 5.0)
    arr = np.stack(
        [butterworth(arr[c], cutoff_frequency_ratio=0.2, high_pass=False)
         for c in range(arr.shape[0])]
    )
    if clip_limits is not None and dataset_name in clip_limits:
        ub = float(clip_limits[dataset_name])
    else:
        ub = clip_upper_bound
    arr = np.clip(arr, 0, ub) / ub
    return torch.from_numpy(arr).float()


def setup_immuvis_equivariant(
    dataset_name: str,
    repo_path: str,
    checkpoint_path: str,
    conf: str,
    panel_conf_path,
    tokenizer_path,
    scheme: str,
    device: torch.device = None,
    batch_size: int = 128,
    input_size: int | None = None,
    skip_preprocessing: bool = False,
):
    """Set up the equivariant autoencoder for cell-typing embedding inference.

    Args:
        input_size: if set, crops are bilinearly resized to
            ``input_size`` x ``input_size`` before encoding. Cell crops are
            32x32 (``core/data.py`` patch_size); the equivariant encoder
            downsamples ~8x with antialiased pooling, so a larger input can
            help feature-map resolution. Leave ``None`` to feed crops as-is.
        skip_preprocessing: if the processed crops are already
            arcsinh/clip-normalized, set True to feed them unchanged.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    panel_config = YAML().load(open(Path(panel_conf_path)))
    tokenizer = YAML().load(open(Path(tokenizer_path)))
    clip_limits = panel_config.get("clip_limits")

    markers = [
        m for m in panel_config["markers"][dataset_name]
        if m not in ("DNA1", "DNA2")
    ]
    channel_ids = torch.tensor(
        [tokenizer[m] for m in markers], dtype=torch.long
    )
    print(f"[equivariant] {dataset_name}: {len(markers)} markers -> "
          f"channel ids {channel_ids.tolist()}")

    model, model_type = _build_equivariant_model(
        repo_path, conf, num_channels=len(tokenizer)
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    print(f"[equivariant] loaded {model_type} from {checkpoint_path} "
          f"(epoch {ckpt.get('epoch', 'unknown')})")

    def get_channels(tid, dataset):
        return channel_ids

    def prepare_fn(x_raw, mask_raw, tid, dataset):
        if skip_preprocessing:
            return x_raw.float(), mask_raw
        return _preprocess(x_raw, dataset_name, clip_limits), mask_raw

    if scheme != "patch":
        # The 'context' scheme requires the ImmuVis tile/patch-assignment
        # machinery (core.utils.immuvis -> OpenCV). Start with 'patch'.
        raise ValueError(
            f"scheme={scheme!r} not supported by the equivariant hook; "
            f"use --scheme patch"
        )

    def compute_fn(model, x, channels, mask, device, batch_size):
        if isinstance(x, (list, tuple)):
            x_list = [item.to(device, dtype=torch.float32) for item in x]
            channels_list = [ch.to(device) for ch in channels]
        else:
            x_list = [x.to(device, dtype=torch.float32)]
            channels_list = [channels.to(device)]

        crop_b = torch.stack(x_list)          # (B, C, H, W)
        ch_b = torch.stack(channels_list)     # (B, C)

        if input_size is not None and crop_b.shape[-1] != input_size:
            crop_b = torch.nn.functional.interpolate(
                crop_b, size=(input_size, input_size),
                mode="bilinear", align_corners=False,
            )

        with torch.no_grad():
            output = model.encode_images(crop_b, ch_b)["output"]
            if output.dim() != 4:
                print(f"[WARNING] equivariant: expected 4D encode output, "
                      f"got {tuple(output.shape)}")
            cell_tokens = output.mean(dim=(2, 3))  # (B, latent_dim)

        return None, cell_tokens, None, None

    return model, prepare_fn, get_channels, compute_fn, device
