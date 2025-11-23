# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from typing import (
    Any,
    Callable,
    Optional,
    Union,
)

import torch

from deepmd.dpmodel import (
    FittingOutputDef,
    OutputVariableDef,
)
from deepmd.pt.model.task.ener import (
    InvarFitting,
)
from deepmd.pt.model.task.fitting import (
    Fitting,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    DEFAULT_PRECISION,
)
from deepmd.utils.version import (
    check_version_compatibility,
)
from deepmd.utils.path import (
    DPPath,
)

dtype = env.GLOBAL_PT_FLOAT_PRECISION
device = env.DEVICE

log = logging.getLogger(__name__)


@Fitting.register("property")
class PropertyFittingNet(InvarFitting):
    """Fitting the rotationally invariant properties of `task_dim` of the system.

    Parameters
    ----------
    ntypes : int
        Element count.
    dim_descrpt : int
        Embedding width per atom.
    embedding_width : int
        The dimension of rotation matrix, m1.
    task_dim : int
        The dimension of outputs of fitting net.
    property_name:
        The name of fitting property, which should be consistent with the property name in the dataset.
        If the data file is named `humo.npy`, this parameter should be "humo".
    neuron : list[int]
        Number of neurons in each hidden layers of the fitting net.
    bias_atom_p : torch.Tensor, optional
        Average property per atom for each element.
    intensive : bool, optional
        Whether the fitting property is intensive.
    resnet_dt : bool
        Using time-step in the ResNet construction.
    numb_fparam : int
        Number of frame parameters.
    numb_aparam : int
        Number of atomic parameters.
    dim_case_embd : int
        Dimension of case specific embedding.
    activation_function : str
        Activation function.
    precision : str
        Numerical precision.
    mixed_types : bool
        If true, use a uniform fitting net for all atom types, otherwise use
        different fitting nets for different atom types.
    seed : int, optional
        Random seed.
    """

    def __init__(
        self,
        ntypes: int,
        dim_descrpt: int,
        property_name: str,
        embedding_width: int,
        task_dim: int = 1,
        neuron: list[int] = [128, 128, 128],
        bias_atom_p: Optional[torch.Tensor] = None,
        intensive: bool = False,
        resnet_dt: bool = True,
        numb_fparam: int = 0,
        numb_aparam: int = 0,
        dim_case_embd: int = 0,
        activation_function: str = "tanh",
        precision: str = DEFAULT_PRECISION,
        mixed_types: bool = True,
        trainable: Union[bool, list[bool]] = True,
        seed: Optional[int] = None,
        default_fparam: Optional[list] = None,
        **kwargs: Any,
    ) -> None:
        self.embedding_width = embedding_width
        self.task_dim = task_dim
        self.intensive = intensive
        super().__init__(
            var_name=property_name,
            ntypes=ntypes,
            dim_descrpt=dim_descrpt,
            dim_out=task_dim,
            neuron=neuron,
            bias_atom_e=bias_atom_p,
            resnet_dt=resnet_dt,
            numb_fparam=numb_fparam,
            numb_aparam=numb_aparam,
            dim_case_embd=dim_case_embd,
            activation_function=activation_function,
            precision=precision,
            mixed_types=mixed_types,
            trainable=trainable,
            seed=seed,
            default_fparam=default_fparam,
            **kwargs,
        )

    def output_def(self) -> FittingOutputDef:
        return FittingOutputDef(
            [
                OutputVariableDef(
                    self.var_name,
                    [self.dim_out],
                    reducible=True,
                    r_differentiable=False,
                    c_differentiable=False,
                    intensive=self.intensive,
                ),
            ]
        )

    def get_intensive(self) -> bool:
        """Whether the fitting property is intensive."""
        return self.intensive

    @classmethod
    def deserialize(cls, data: dict) -> "PropertyFittingNet":
        data = data.copy()
        check_version_compatibility(data.pop("@version", 1), 5, 1)
        data.pop("dim_out")
        data["property_name"] = data.pop("var_name")
        obj = super().deserialize(data)

        return obj

    def _net_out_dim(self) -> int:
        """Set the FittingNet output dim."""
        return self.embedding_width

    def serialize(self) -> dict:
        """Serialize the fitting to dict."""
        dd = {
            **InvarFitting.serialize(self),
            "type": "property",
            "task_dim": self.task_dim,
            "intensive": self.intensive,
        }
        dd["@version"] = 5
        dd["embedding_width"] = self.embedding_width

        return dd

    def compute_output_stats(
        self,
        merged: Union[Callable[[], list[dict]], list[dict]],
        stat_file_path: Optional[DPPath] = None,
    ) -> None:
        """
        Compute the output statistics (e.g. energy bias) for the fitting net from packed data.

        Parameters
        ----------
        merged : Union[Callable[[], list[dict]], list[dict]]
            - list[dict]: A list of data samples from various data systems.
                Each element, `merged[i]`, is a data dictionary containing `keys`: `torch.Tensor`
                originating from the `i`-th data system.
            - Callable[[], list[dict]]: A lazy function that returns data samples in the above format
                only when needed. Since the sampling process can be slow and memory-intensive,
                the lazy function helps by only sampling once.
        stat_file_path : Optional[DPPath]
            The path to the stat file.

        """
        pass

    def forward(
        self,
        descriptor: torch.Tensor,
        atype: torch.Tensor,
        gr: Optional[torch.Tensor] = None,
        g2: Optional[torch.Tensor] = None,
        h2: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        assert self.var_name == "mu"
        nframes, nloc, _ = descriptor.shape
        assert gr is not None, "Must provide the rotation matrix for dipole fitting."
        # cast the input to internal precsion
        gr = gr.to(self.prec)
        # (nframes, nloc, m1)
        out = self._forward_common(descriptor, atype, gr, g2, h2, fparam, aparam)[
            self.var_name
        ]
        # (nframes * nloc, 1, m1)
        out = out.view(-1, 1, self.embedding_width)
        # (nframes * nloc, m1, 3)
        gr = gr.view(nframes * nloc, self.embedding_width, 3)
        # (nframes, nloc, 3)
        out = torch.bmm(out, gr).squeeze(-2).view(nframes, nloc, 3)

        return {self.var_name: out.to(env.GLOBAL_PT_FLOAT_PRECISION)}

    # make jit happy with torch 2.0.0
    exclude_types: list[int]
