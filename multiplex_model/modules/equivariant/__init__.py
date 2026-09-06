"""Fully-equivariant (escnn) multiplex autoencoder.

Import-isolated on purpose: nothing in ``multiplex_model.modules`` imports this
subpackage, so the vanilla path never pays for escnn and a broken escnn install
cannot break vanilla training. The training entrypoint imports it lazily only
when ``model_type == "fully_equivariant_v3"``.
"""

from .autoencoder_v3 import EquivariantMultiplexAutoencoderV3

__all__ = ["EquivariantMultiplexAutoencoderV3"]
