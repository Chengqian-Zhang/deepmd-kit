# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Optional,
)

import numpy as np
import logging
import torch
import torch.nn.functional as F

from deepmd.pt.loss.loss import (
    TaskLoss,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    GLOBAL_PT_FLOAT_PRECISION,
)
from deepmd.utils.data import (
    DataRequirementItem,
)
from deepmd.utils.version import (
    check_version_compatibility,
)
from deepmd.pt.utils.region import (
    phys2inter,
    inter2phys,
)
from IPython import embed
import sys

log = logging.getLogger(__name__)

def get_cell_perturb_matrix(cell_pert_fraction: float):
    if cell_pert_fraction < 0:
        raise RuntimeError("cell_pert_fraction can not be negative")
    e0 = torch.rand(6)
    e = e0 * 2 * cell_pert_fraction - cell_pert_fraction
    cell_pert_matrix = torch.tensor(
        [
            [1 + e[0], 0,        0],
            [e[5],     1 + e[1], 0],
            [e[4],     e[3],     1 + e[2]],
        ],
        dtype=env.GLOBAL_PT_FLOAT_PRECISION,
        device=env.DEVICE
    )
    return cell_pert_matrix, e

def get_cell_perturb_matrix_HEA(cell_pert_fraction: float):
    if cell_pert_fraction < 0:
        raise RuntimeError("cell_pert_fraction can not be negative")
    e0 = torch.rand(3)
    e = e0 * 2 * cell_pert_fraction - cell_pert_fraction
    cell_pert_matrix = torch.tensor(
        [
            [1 + e[0], 0,        0],
            [e[2],     1 + e[1], 0],
            [0,        0,        1],
        ],
        dtype=env.GLOBAL_PT_FLOAT_PRECISION,
        device=env.DEVICE
    )
    return cell_pert_matrix, e

class DenoiseLoss(TaskLoss):
    def __init__(
        self,
        use_l1_all: bool = False,
        inference=False,
        noise_type: str = "uniform",
        noise: float = 0.2,
        noise_mode: str = "prob",
        mask_num: int = 1,
        mask_prob: float = 0.2,
        mask_coord: bool = True,
        mask_cell: bool = False,
        cell_noise: float = 0.0,
        loss_func: str = "rmse",
        pref_f: float = 1.0,
        pref_v: float = 1.0,
        **kwargs,
    ) -> None:
        r"""Construct a layer to compute loss on energy, force and virial.

        Parameters
        ----------
        use_l1_all : bool
            Whether to use L1 loss, if False (default), it will use L2 loss.
        inference : bool
            If true, it will output all losses found in output, ignoring the pre-factors.
        noise_type : str
            The type of noise to add to the coordinate. It can be 'uniform' or 'Gaussian'.
        noise : float
            The magnitude of noise to add to the coordinate.
        noise_mode : str
            "'prob' means the noise is added with a probability.'fix_num' means the noise is added with a fixed number."
        mask_prob : float
            The probability of masking a coordinate.
        mask_coord : bool
            Whether to mask the coordinate.
        mask_cell : bool
            Whether to mask the cell.
        cell_noise: float
            A value determines how much will cell deform.
        **kwargs
            Other keyword arguments.
        """
        super().__init__()
        self.use_l1_all = use_l1_all
        self.inference = inference

        self.noise_type = noise_type
        self.noise = noise
        self.noise_mode = noise_mode
        self.mask_num = mask_num
        self.mask_prob = mask_prob
        self.mask_coord = mask_coord
        self.mask_cell = mask_cell
        self.cell_noise = cell_noise
        self.loss_func = loss_func
        self.pref_f = pref_f
        self.pref_v = pref_v

    def forward(self, input_dict, model, label, natoms, learning_rate, mae=False):
        """Return loss on energy and force.

        Parameters
        ----------
        input_dict : dict[str, torch.Tensor]
            Model inputs.
        model : torch.nn.Module
            Model to be used to output the predictions.
        label : dict[str, torch.Tensor]
            Labels.
        natoms : int
            The local atom number.

        Returns
        -------
        model_pred: dict[str, torch.Tensor]
            Model predictions.
        loss: torch.Tensor
            Loss for model to minimize.
        more_loss: dict[str, torch.Tensor]
            Other losses for display.
        """

        nloc = input_dict["atype"].shape[1]
        nbz = input_dict["atype"].shape[0]
        input_dict["box"] = input_dict["box"].cuda() # box在cpu上，转到gpu上

        # TODO: 把所有的盒子转化成下三角矩阵，HEA数据集均已经是下三角
        # 检查下三角
        for single_box in input_dict["box"]:
            assert single_box[0] > 0
            assert single_box[1] == 0
            assert single_box[2] == 0
            #assert single_box[3] == 0
            assert single_box[4] > 0
            assert single_box[5] == 0
            assert single_box[6] == 0
            assert single_box[7] == 0
            assert single_box[8] > 0

        '''
        def affine_map(self, trans, box, coord):
            assert torch.det(trans) != 0
            new_box = torch.matmul(box, trans)
            new_coord = torch.matmul(coord, trans)
            return new_box, new_coord
        for f_idx in range(nbz):
            embed()
            sys.exit()
            qq, rr = torch.linalg.qr(input_dict["box"][f_idx].reshape(3,3).T)
            if torch.det(qq) < 0:
                qq = -qq
                rr = -rr
            affine_map(qq, input_dict["box"][f_idx].reshape(3,3), input_dict["coord"])
            rot = np.eye(3)
            if self.data["cells"][f_idx][0][0] < 0:
                rot[0][0] = -1
            if self.data["cells"][f_idx][1][1] < 0:
                rot[1][1] = -1
            if self.data["cells"][f_idx][2][2] < 0:
                rot[2][2] = -1
            assert np.linalg.det(rot) == 1
            self.affine_map(rot, f_idx=f_idx)
        '''
        
        nbz = input_dict["atype"].shape[0]
        frac_coord = phys2inter(input_dict["coord"], input_dict["box"].reshape(nbz,3,3)).clone().detach()
        label["clean_frac_coord"] = phys2inter(label["clean_coord"], label["clean_box"].reshape(nbz,3,3)).clone().detach()
        label["force"] = (label["clean_frac_coord"] - frac_coord).clone().detach()

        # input["box"] = cell_pert_matrix @ label["clean_box"]
        cell_pert_matrix = input_dict["box"].reshape(nbz,3,3) @ torch.inverse(label["clean_box"].reshape(nbz,3,3))
        e = torch.zeros((nbz,3), dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE)
        e[:,0]=cell_pert_matrix[:,0,0]-1
        e[:,1]=cell_pert_matrix[:,1,1]-1
        e[:,2]=cell_pert_matrix[:,1,0]
        label["virial"] = e.clone().detach()

        model_pred = model(**input_dict)      

        loss = torch.zeros(1, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE)[0]
        more_loss = {}

        diff_f = (label["force"] - model_pred["force"]).reshape(-1)
        diff_v = (label["virial"] - model_pred["virial"]).reshape(-1)
        if self.loss_func == "rmse":
            l2_force_loss = torch.mean(torch.square(diff_f))
            l2_virial_loss = torch.mean(torch.square(diff_v))
            rmse_f = l2_force_loss.sqrt()
            rmse_v = l2_virial_loss.sqrt()
            more_loss["rmse_force"] = rmse_f.detach()
            more_loss["rmse_virial"] = rmse_v.detach()
            loss += 200 * (self.pref_f * l2_force_loss.to(GLOBAL_PT_FLOAT_PRECISION) + self.pref_v * l2_virial_loss.to(GLOBAL_PT_FLOAT_PRECISION))
        elif self.loss_func == "mae":
            l1_force_loss = F.l1_loss(label["force"], model_pred["force"], reduction="none")
            l1_virial_loss = F.l1_loss(label["virial"], model_pred["virial"], reduction="none")
            more_loss["mae_force"] = l1_force_loss.mean().detach()
            more_loss["mae_virial"] = l1_virial_loss.mean().detach()
            l1_force_loss = l1_force_loss.sum(-1).mean(-1).sum()
            l1_virial_loss = l1_virial_loss.sum()
            loss += 200 * (self.pref_f * l1_force_loss.to(GLOBAL_PT_FLOAT_PRECISION) + self.pref_v * l1_virial_loss.to(GLOBAL_PT_FLOAT_PRECISION))
        else:
            raise RuntimeError(f"Unknown loss function {self.loss_func}!")
        return model_pred, loss, more_loss

    @property
    def label_requirement(self) -> list[DataRequirementItem]:
        """Return data label requirements needed for this loss calculation."""
        label_requirement = []
        label_requirement.append(
            DataRequirementItem(
                "clean_coord",
                ndof=3,
                atomic=True,
                must=True,
                high_prec=False,
            )
        )
        label_requirement.append(
            DataRequirementItem(
                "clean_box",
                ndof=9,
                atomic=False,
                must=True,
                high_prec=False,
            )
        )
        return label_requirement

    def serialize(self) -> dict:
        pass

    @classmethod
    def deserialize(cls, data: dict) -> "TaskLoss":
        pass