# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
    Optional,
    Union,
)

import numpy as np

from deepmd.dpmodel.output_def import (
    FittingOutputDef,
    ModelOutputDef,
    OutputVariableDef,
)

from .deep_eval import (
    DeepEval,
)
from deepmd.pt.utils.region import (
    phys2inter,
    inter2phys,
)
import torch
from IPython import embed


class DeepDenoise(DeepEval):
    """Properties of structures.

    Parameters
    ----------
    model_file : Path
        The name of the frozen model file.
    *args : list
        Positional arguments.
    auto_batch_size : bool or int or AutoBatchSize, default: True
        If True, automatic batch size will be used. If int, it will be used
        as the initial batch size.
    neighbor_list : ase.neighborlist.NewPrimitiveNeighborList, optional
        The ASE neighbor list class to produce the neighbor list. If None, the
        neighbor list will be built natively in the model.
    **kwargs : dict
        Keyword arguments.
    """
    
    @property
    def output_def(self) -> ModelOutputDef:
        """
        Get the output definition of this model.
        But in property_fitting, the output definition is not known until the model is loaded.
        So we need to rewrite the output definition after the model is loaded.
        See detail in change_output_def.
        """
        return ModelOutputDef(
            FittingOutputDef(
                [
                    OutputVariableDef(
                        "virial",
                        [9],
                        reducible=True,
                        r_differentiable=False,
                        c_differentiable=False,
                        intensive=True,
                    ),
                    OutputVariableDef(
                        "force",
                        [3],
                        reducible=False,
                        r_differentiable=False,
                        c_differentiable=False,
                    ),
                ]
            )
        )

    def eval(
        self,
        coords: np.ndarray,
        cells: Optional[np.ndarray],
        atom_types: Union[list[int], np.ndarray],
        atomic: bool = False,
        fparam: Optional[np.ndarray] = None,
        aparam: Optional[np.ndarray] = None,
        mixed_type: bool = False,
        **kwargs: dict[str, Any],
    ) -> tuple[np.ndarray, ...]:
        """Evaluate properties. If atomic is True, also return atomic property.

        Parameters
        ----------
        coords : np.ndarray
            The coordinates of the atoms, in shape (nframes, natoms, 3).
        cells : np.ndarray
            The cell vectors of the system, in shape (nframes, 9). If the system
            is not periodic, set it to None.
        atom_types : list[int] or np.ndarray
            The types of the atoms. If mixed_type is False, the shape is (natoms,);
            otherwise, the shape is (nframes, natoms).
        atomic : bool, optional
            Whether to return atomic property, by default False.
        fparam : np.ndarray, optional
            The frame parameters, by default None.
        aparam : np.ndarray, optional
            The atomic parameters, by default None.
        mixed_type : bool, optional
            Whether the atom_types is mixed type, by default False.
        **kwargs : dict[str, Any]
            Keyword arguments.

        Returns
        -------
        property
            The properties of the system, in shape (nframes, num_tasks).
        """
        (
            coords,
            cells,
            atom_types,
            fparam,
            aparam,
            nframes,
            natoms,
        ) = self._standard_input(coords, cells, atom_types, fparam, aparam, mixed_type)
        results = self.deep_eval.eval(
            coords,
            cells,
            atom_types,
            atomic,
            fparam=fparam,
            aparam=aparam,
            **kwargs,
        )
        force = results["force"].reshape(
            nframes, natoms, 3
        )
        atomic_virial = results["virial"].reshape(
            nframes, natoms, 3, 3
        )
        virial = results["virial_redu"].reshape(
            nframes, 3, 3
        )

        # update frac coord
        rec_cells = np.linalg.inv(cells)
        #frac_coords = np.remainder(np.matmul(coords, rec_cells), 1.0)
        frac_coords = np.matmul(coords, rec_cells)
        assert frac_coords.shape == force.shape
        relax_frac_coords = frac_coords + force

        # update box 
        # box_noise = box_relax @ cell_pert_matrix(virial),
        # box_relax = box_noise @ cell_pert_matrix(virial).inv
        assert virial.shape == cells.shape
        relax_box = np.matmul(cells, np.linalg.inv(virial))
        # get final coord
        relax_coords = np.matmul(relax_frac_coords, relax_box)

        return (relax_coords, relax_box, force, virial)

__all__ = ["DeepDenoise"]