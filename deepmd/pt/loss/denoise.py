# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Optional,
)

import numpy as np
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
        mask_box: bool = False,
        **kwargs,
    ) -> None:
        r"""Construct a layer to compute loss on energy, force and virial.

        Parameters
        ----------
        use_l1_all : bool
            Whether to use L1 loss, if False (default), it will use L2 loss.
        inference : bool
            If true, it will output all losses found in output, ignoring the pre-factors.
        
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
        self.mask_box = mask_box

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
        label["clean_coord"] = input_dict["coord"].clone().detach()
        if self.mask_box:
            label["clean_box"] = input_dict["box"].clone().detach()

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

        if self.mask_coord:
            noise_on_coord_all = torch.zeros(input_dict["coord"].shape, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE) 
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
                input_dict["coord"][ii][coord_mask ,:] += noise_on_coord # nbz mask_num 3 //                 
                noise_on_coord = noise_on_coord.detach()

                noise_on_coord_all[ii] = noise_on_coord
                coord_mask_all[ii] = torch.tensor(coord_mask, dtype=torch.bool, device=env.DEVICE)
            label['coord_mask'] = coord_mask_all
            label["force"] = (label["clean_coord"] - input_dict["coord"]).clone().detach()
            assert label["force"][coord_mask_all].view(nbz,nloc,3).allclose(-1.00 * noise_on_coord_all.view(nbz,nloc,3))
        else:           
            NotImplementedError(f"One must mask coord in denoise mode!")
        
        if self.mask_box:
            raise RuntimeError(f"Mask box is not supported yet.")

        model_pred = model(**input_dict)

        loss = torch.zeros(1, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE)[0]
        more_loss = {}

        force_pred = model_pred["force"]
        force_label = label["force"]
        diff_f = (force_label - force_pred).reshape(-1)

        if not self.use_l1_all:
            l2_force_loss = torch.mean(torch.square(diff_f))
            if not self.inference:
                more_loss["l2_coord_loss"] = l2_force_loss.detach()
            loss += l2_force_loss.to(GLOBAL_PT_FLOAT_PRECISION)
            rmse_f = l2_force_loss.sqrt()
            more_loss["rmse_coord"] = rmse_f.detach()
        else:
            l1_force_loss = F.l1_loss(force_label, force_pred, reduction="none")
            more_loss["mae_coord"] = l1_force_loss.mean().detach()
            l1_force_loss = l1_force_loss.sum(-1).mean(-1).sum()
            loss += l1_force_loss.to(GLOBAL_PT_FLOAT_PRECISION)
        if mae:
            mae_f = torch.mean(torch.abs(diff_f))
            more_loss["mae_coord"] = mae_f.detach()

        if not self.inference:
            more_loss["rmse"] = torch.sqrt(loss.detach())
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
