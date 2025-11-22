# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
    Callable,
    NoReturn,
    Optional,
    Union,
)

import torch

from deepmd.pt.model.task.property import (
    PropertyFittingNet,
)

from .dp_atomic_model import (
    DPAtomicModel,
)

from deepmd.utils.atomic_mass import MASS_DICT

class DPPropertyAtomicModel(DPAtomicModel):
    def __init__(
        self, descriptor: Any, fitting: Any, type_map: Any, **kwargs: Any
    ) -> None:
        if not isinstance(fitting, PropertyFittingNet):
            raise TypeError(
                "fitting must be an instance of PropertyFittingNet for DPPropertyAtomicModel"
            )
        super().__init__(descriptor, fitting, type_map, **kwargs)
        atomic_mass_dict = {}
        for idx, ele in enumerate(type_map):
            atomic_mass_dict[idx] = MASS_DICT[ele]
        self.atomic_mass_dict = atomic_mass_dict

    def get_compute_stats_distinguish_types(self) -> bool:
        """Get whether the fitting net computes stats which are not distinguished between different types of atoms."""
        return False

    def get_intensive(self) -> bool:
        """Whether the fitting property is intensive."""
        return self.fitting_net.get_intensive()

    def forward_common_atomic(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        comm_dict: Optional[dict[str, torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        """QM9 R2 interface for atomic inference.

        This method accept extended coordinates, extended atom typs, neighbor list,
        and predict the atomic contribution of the fit property.

        Parameters
        ----------
        extended_coord
            extended coordinates, shape: nf x (nall x 3)
        extended_atype
            extended atom typs, shape: nf x nall
            for a type < 0 indicating the atomic is virtual.
        nlist
            neighbor list, shape: nf x nloc x nsel
        mapping
            extended to local index mapping, shape: nf x nall
        fparam
            frame parameters, shape: nf x dim_fparam
        aparam
            atomic parameter, shape: nf x nloc x dim_aparam
        comm_dict
            The data needed for communication for parallel inference.

        Returns
        -------
        ret_dict
            dict of output atomic properties.
            should implement the definition of `fitting_output_def`.
            ret_dict["mask"] of shape nf x nloc will be provided.
            ret_dict["mask"][ff,ii] == 1 indicating the ii-th atom of the ff-th frame is real.
            ret_dict["mask"][ff,ii] == 0 indicating the ii-th atom of the ff-th frame is virtual.

        """
        nbz, nloc, _ = nlist.shape
        assert extended_atype.shape == (nbz, nloc)
        assert extended_coord.shape == (nbz, nloc, 3)
        assert self.pair_excl is None

        atype = extended_atype
        coord = extended_coord

        ext_atom_mask = self.make_atom_mask(extended_atype)
        assert torch.all(ext_atom_mask)
        ret_dict = self.forward_atomic(
            extended_coord,
            torch.where(ext_atom_mask, extended_atype, 0),
            nlist,
            mapping=mapping,
            fparam=fparam,
            aparam=aparam,
            comm_dict=comm_dict,
        )
        assert list(ret_dict.keys()) == ["r2"], f"This branch is only used to predict qm9 r2."
        #ret_dict = self.apply_out_stat(ret_dict, atype) # ignore atom bias

        # atomic mass
        mass = torch.zeros_like(atype, dtype=coord.dtype)
        for atomic_number, mass_value in self.atomic_mass_dict.items():
            mask = (atype == atomic_number)
            mass[mask] = mass_value

        mass = mass.unsqueeze(-1)  # [nbz, nloc, 1]
        x = ret_dict["r2"]  # [nbz, nloc, 1]

        # Mass center
        assert mass.shape == (nbz, nloc, 1)
        assert coord.shape == (nbz, nloc, 3)
        mass_weighted_coord = mass * coord  # [nbz, nloc, 3]

        # Total mass
        total_mass = torch.sum(mass, dim=1, keepdim=True)  # [nbz, 1, 1]
        mass_weighted_sum = torch.sum(mass_weighted_coord, dim=1, keepdim=True)  # [nbz, 1, 3]

        center = mass_weighted_sum / total_mass.clamp(min=1e-8)  # [nbz, 1, 3]

        # Distance to mass center for each atom
        center_expanded = center.expand(-1, nloc, -1)  # [nbz, nloc, 3]
        assert center_expanded.shape == (nbz, nloc, 3)
        dist_to_center = torch.norm(coord - center_expanded, dim=-1, keepdim=True)  # [nbz, nloc, 1]

        # atomic contribution
        assert dist_to_center.shape == (nbz, nloc, 1)
        assert x.shape == (nbz, nloc, 1)
        yi = (dist_to_center ** 2) * x  # [nbz, nloc, 1]

        assert yi.shape == ret_dict["r2"].shape
        ret_dict["r2"] = yi

        # nf x nloc
        atom_mask = ext_atom_mask[:, :nloc].to(torch.int32)
        if self.atom_excl is not None:
            atom_mask *= self.atom_excl(atype)

        for kk in ret_dict.keys():
            out_shape = ret_dict[kk].shape
            out_shape2 = 1
            for ss in out_shape[2:]:
                out_shape2 *= ss
            ret_dict[kk] = (
                ret_dict[kk].reshape([out_shape[0], out_shape[1], out_shape2])
                * atom_mask[:, :, None]
            ).view(out_shape)
        ret_dict["mask"] = atom_mask

        return ret_dict