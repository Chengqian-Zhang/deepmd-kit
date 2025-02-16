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

log = logging.getLogger(__name__)

def get_cell_perturb_matrix(cell_pert_fraction: float):
    if cell_pert_fraction < 0:
        raise RuntimeError("cell_pert_fraction can not be negative")
    e0 = torch.rand(6)
    e = e0 * 2 * cell_pert_fraction - cell_pert_fraction
    cell_pert_matrix = torch.tensor(
        [
            [1 + e[0], 0.5 * e[5], 0.5 * e[4]],
            [0.5 * e[5], 1 + e[1], 0.5 * e[3]],
            [0.5 * e[4], 0.5 * e[3], 1 + e[2]],
        ],
        dtype=env.GLOBAL_PT_FLOAT_PRECISION,
        device=env.DEVICE
    )
    return cell_pert_matrix

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
        cell_pert_fraction: float = 0.0,
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
        cell_pert_fraction: float
            A fraction determines how much (relatively) will cell deform.
            The cell of each frame is deformed by a symmetric matrix perturbed from identity.
            The perturbation to the diagonal part is subject to a uniform distribution in [-cell_pert_fraction, cell_pert_fraction),
            and the perturbation to the off-diagonal part is subject to a uniform distribution in [-0.5*cell_pert_fraction, 0.5*cell_pert_fraction).
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
        self.cell_pert_fraction = cell_pert_fraction
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
        label["clean_coord"] = input_dict["coord"].clone().detach()
        label["clean_box"] = input_dict["box"].clone().detach()
        label["clean_frac_coord"] = phys2inter(label["clean_coord"], label["clean_box"].reshape(-1,3,3)).clone().detach()
        label["clean_frac_coord"] = torch.remainder(label["clean_frac_coord"], 1.0)
        frac_coord = label["clean_frac_coord"].clone().detach()
        if self.mask_cell:
            cell_perturb_matrix_all = torch.zeros((nbz,9), dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE)
            for ii in range(nbz):
                # 对于每个batch单独处理
                cell_perturb_matrix = get_cell_perturb_matrix(self.cell_pert_fraction)
                input_dict["box"][ii] = torch.matmul(input_dict["box"][ii].reshape(3,3), cell_perturb_matrix).reshape(-1) #盒子乘对称矩阵cell_perturb_matrix得到形变盒子
                input_dict["coord"][ii] = torch.matmul(input_dict["coord"][ii].reshape(nloc,3), cell_perturb_matrix) #原子笛卡尔坐标也要随之变化
                cell_perturb_matrix_all[ii] = cell_perturb_matrix.reshape(-1)
            label["virial"] = cell_perturb_matrix_all.clone().detach()

        if self.mask_coord:
            # 将x加noise，并更新label['force']
            mask_num = 0
            if self.noise_mode == "fix_num":
                mask_num = self.mask_num
                if(nloc < mask_num):
                    mask_num = nloc
            elif self.noise_mode == "prob":
                mask_num = int(self.mask_prob * nloc)
                if mask_num == 0:
                    mask_num = 1
            else:
                NotImplementedError(f"Unknown noise mode {self.noise_mode}!")

            coord_mask_all = torch.zeros(input_dict["atype"].shape, dtype=torch.bool, device=env.DEVICE) 
            for ii in range(nbz):
                # 对于每个batch单独处理
                noise_on_coord = 0.0
                coord_mask_res = np.random.choice(range(nloc), mask_num, replace=False).tolist()
                coord_mask = np.isin(range(nloc), coord_mask_res) # nloc
                if self.noise_type == "uniform":
                    noise_on_coord = np.random.uniform(
                        low=-self.noise, high=self.noise, size=(mask_num, 3)
                    )
                else:
                    NotImplementedError(f"Unknown noise type {self.noise_type}!")
                
                noise_on_coord = torch.tensor(noise_on_coord, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE) # mask_num 3
                frac_coord[ii][coord_mask ,:] += noise_on_coord # nbz mask_num 3 //       
                input_dict["coord"][ii] = inter2phys(frac_coord[ii], input_dict["box"][ii].reshape(3,3))
                coord_mask_all[ii] = torch.tensor(coord_mask, dtype=torch.bool, device=env.DEVICE)
            label['coord_mask'] = coord_mask_all
            label["force"] = (label["clean_frac_coord"] - frac_coord).clone().detach()

        if (not self.mask_coord) and (not self.mask_cell):
            raise RuntimeError("At least one of mask_coord and mask_cell should be True!")

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
            loss += self.pref_f * l2_force_loss.to(GLOBAL_PT_FLOAT_PRECISION) + self.pref_v * l2_virial_loss.to(GLOBAL_PT_FLOAT_PRECISION) 
        elif self.loss_func == "mae":
            l1_force_loss = F.l1_loss(label["force"], model_pred["force"], reduction="none")
            l1_virial_loss = F.l1_loss(label["virial"], model_pred["virial"], reduction="none")
            more_loss["mae_force"] = l1_force_loss.mean().detach()
            more_loss["mae_virial"] = l1_virial_loss.mean().detach()
            l1_force_loss = l1_force_loss.sum(-1).mean(-1).sum()
            l1_virial_loss = l1_virial_loss.sum()
            loss += self.pref_f * l1_force_loss.to(GLOBAL_PT_FLOAT_PRECISION) + self.pref_v * l1_virial_loss.to(GLOBAL_PT_FLOAT_PRECISION)
        else:
            raise RuntimeError(f"Unknown loss function {self.loss_func}!")
        return model_pred, loss, more_loss

    @property
    def label_requirement(self) -> list[DataRequirementItem]:
        """Return data label requirements needed for this loss calculation."""
        return []

    def serialize(self) -> dict:
        pass

    @classmethod
    def deserialize(cls, data: dict) -> "TaskLoss":
        pass