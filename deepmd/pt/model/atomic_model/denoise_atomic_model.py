# SPDX-License-Identifier: LGPL-3.0-or-later

import torch
import logging

from deepmd.pt.model.task.denoise import (
    DenoiseNet,
)

from .dp_atomic_model import (
    DPAtomicModel,
)

log = logging.getLogger(__name__)

class DPDenoiseAtomicModel(DPAtomicModel):
    def __init__(self, descriptor, fitting, type_map, **kwargs):
        if not isinstance(fitting, DenoiseNet):
            raise TypeError(
                "fitting must be an instance of DenoiseNet for DPDenoiseAtomicModel"
            )
        super().__init__(descriptor, fitting, type_map, **kwargs)

    def apply_out_stat(
        self,
        ret: dict[str, torch.Tensor],
        atype: torch.Tensor,
    ):
        return ret
