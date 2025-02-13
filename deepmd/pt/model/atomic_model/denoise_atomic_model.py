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
        nbz = ret["virial"].shape[0]
        nloc = ret["virial"].shape[1]
        symmetry_virial = torch.zeros(
            nbz,
            nloc,
            3,
            3,
            dtype=ret["virial"].dtype,
            device=ret["virial"].device,
        )
        for ii in range(nbz):
            for jj in range(nloc):
                e = ret["virial"][ii][jj]
                symmetry_virial[ii][jj] = torch.tensor(
                    [
                        [1 + e[0], 0.5 * e[5], 0.5 * e[4]],
                        [0.5 * e[5], 1 + e[1], 0.5 * e[3]],
                        [0.5 * e[4], 0.5 * e[3], 1 + e[2]],
                    ],
                    dtype=ret["virial"].dtype,
                    device=ret["virial"].device
                )
        ret["virial"] = symmetry_virial.reshape(nbz, nloc, 9)
        return ret
