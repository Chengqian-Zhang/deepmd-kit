# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
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


class DeepProperty(DeepEval):
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
        """Get the output definition of this model."""
        return ModelOutputDef(
            FittingOutputDef(
                [
                    OutputVariableDef(
                        "property",
                        shape=[-1],
                        reduciable=True,
                        atomic=True,
                    ),
                ]
            )
        )

    @property
    def numb_task(self) -> int:
        """Get the number of task."""
        return self.get_numb_task()

    def eval(
        self,
        coords: np.ndarray,
        cells: Optional[np.ndarray],
        atom_types: Union[List[int], np.ndarray],
        atomic: bool = False,
        fparam: Optional[np.ndarray] = None,
        aparam: Optional[np.ndarray] = None,
        mixed_type: bool = False,
        **kwargs: Dict[str, Any],
    ) -> Tuple[np.ndarray, ...]:
        """Evaluate properties. If atomic is True, also return atomic property.

        Parameters
        ----------
        coords : np.ndarray
            The coordinates of the atoms, in shape (nframes, natoms, 3).
        cells : np.ndarray
            The cell vectors of the system, in shape (nframes, 9). If the system
            is not periodic, set it to None.
        atom_types : List[int] or np.ndarray
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
        **kwargs : Dict[str, Any]
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
        atomic_property = results["property"].reshape(nframes, natoms, -1)
        property = results["property_redu"].reshape(nframes, -1)

        if atomic:
            return (
                property,
                atomic_property,
            )
        else:
            return (property,)

    def get_numb_task(self) -> int:
        return self.deep_eval.get_numb_task()


__all__ = ["DeepProperty"]
