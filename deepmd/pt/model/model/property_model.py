# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Dict,
    Optional,
)

import torch

from .dp_model import (
    DPModel,
)


class PropertyModel(DPModel):
    model_type = "property"

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        coord,
        atype,
        box: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        do_atomic_virial: bool = False,
    ) -> Dict[str, torch.Tensor]:
        model_ret = self.forward_common(
            coord,
            atype,
            box,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
        )
        model_predict = {}
        model_predict["atom_property"] = model_ret["property"]
        if self.get_fitting_net().intensive:
            natoms = model_predict["atom_property"].shape[1]
            model_predict["property"] = model_ret["property_redu"] / natoms
        else:
            model_predict["property"] = model_ret["property_redu"]
        if "mask" in model_ret:
            model_predict["mask"] = model_ret["mask"]
        return model_predict

    @torch.jit.export
    def get_numb_task(self) -> int:
        """Get the number of  tasks for PropertyFittingNet."""
        return self.get_fitting_net().dim_out
    
    @torch.jit.export
    def forward_lower(
        self,
        extended_coord,
        extended_atype,
        nlist,
        mapping: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        do_atomic_virial: bool = False,
    ):
        model_ret = self.forward_common_lower(
            extended_coord,
            extended_atype,
            nlist,
            mapping,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
        )
        model_predict = {}
        model_predict["atom_property"] = model_ret["property"]
        natoms = model_predict["atom_property"].shape[1]
        model_predict["property"] = model_ret["property_redu"] / natoms
        if "mask" in model_ret:
            model_predict["mask"] = model_ret["mask"]
        return model_predict
