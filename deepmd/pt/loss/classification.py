# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

import torch
import torch.nn.functional as F

from deepmd.pt.loss.loss import (
    TaskLoss,
)
from deepmd.utils.data import (
    DataRequirementItem,
)
from deepmd.utils.version import (
    check_version_compatibility,
)


class ClassificationLoss(TaskLoss):
    """Cross-entropy loss for system-level single-label classification."""

    def __init__(self, num_classes: int, var_name: str, **kwargs: Any) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.var_name = var_name

    def forward(
        self,
        input_dict: dict[str, torch.Tensor],
        model: torch.nn.Module,
        label: dict[str, torch.Tensor],
        natoms: int,
        learning_rate: float = 0.0,
        mae: bool = False,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
        model_pred = self._inject_atom_mask(model(**input_dict), input_dict)
        logits = model_pred[self.var_name]
        if logits.ndim != 2 or logits.shape[1] != self.num_classes:
            raise ValueError(
                f"Expected logits with shape (batch, {self.num_classes}), "
                f"but got {tuple(logits.shape)}"
            )

        target = label[self.var_name]
        if target.ndim == 2 and target.shape[1] == 1:
            target = target.squeeze(1)
        if target.ndim != 1 or target.shape[0] != logits.shape[0]:
            raise ValueError(
                f"Expected labels with shape ({logits.shape[0]},) or "
                f"({logits.shape[0]}, 1), but got {tuple(target.shape)}"
            )
        target = target.to(device=logits.device, dtype=torch.long)
        if bool(torch.any(target < 0)) or bool(
            torch.any(target >= self.num_classes)
        ):
            raise ValueError(f"Class labels must be in [0, {self.num_classes})")

        loss = F.cross_entropy(logits, target)
        accuracy = (torch.argmax(logits, dim=-1) == target).to(logits.dtype).mean()
        return model_pred, loss, {"accuracy": accuracy.detach()}

    @property
    def label_requirement(self) -> list[DataRequirementItem]:
        return [
            DataRequirementItem(
                self.var_name,
                ndof=1,
                atomic=False,
                must=True,
            )
        ]

    def serialize(self) -> dict:
        return {
            "@class": "ClassificationLoss",
            "@version": 1,
            "num_classes": self.num_classes,
            "var_name": self.var_name,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "ClassificationLoss":
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 1, 1)
        data.pop("@class")
        return cls(**data)
