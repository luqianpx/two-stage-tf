
# Code adapted from https://github.com/facebookresearch/moco
# author:px
# date:2022-01-07
# version:1.5

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Tuple, Optional, Type
from argparse import ArgumentParser
import torchvision.models as models

class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(
        self,
        encoder_q: nn.Module,
        encoder_k: nn.Module,
        dim: int = 128,
        K: int = 65536,
        m: float = 0.999,
        T: float = 0.07,
        mlp: bool = False
    ):
        """
        Args:
            encoder_q: Query encoder
            encoder_k: Key encoder
            dim: Feature dimension (default: 128)
            K: Queue size; number of negative keys (default: 65536)
            m: Moco momentum of updating key encoder (default: 0.999)
            T: Softmax temperature (default: 0.07)
            mlp: Whether to use MLP head (default: False)
        """
        super().__init__()

        self.K = K
        self.m = m
        self.T = T

        # Create encoders
        self.encoder_q = encoder_q
        self.encoder_k = encoder_k

        if mlp:  # Hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp),
                nn.ReLU(inplace=True),
                nn.Linear(dim_mlp, dim)
            )
            self.encoder_k.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp),
                nn.ReLU(inplace=True),
                nn.Linear(dim_mlp, dim)
            )

        # Initialize key encoder
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # Create queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder"""
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """Update queue"""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        # Replace the keys at ptr
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # Move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """Batch shuffle for making use of BN"""
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # Random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # Index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # Shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """Undo batch shuffle"""
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q: torch.Tensor, im_k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        Args:
            im_q: Query images
            im_k: Key images
        Returns:
            logits: Classification logits
            labels: Ground truth labels
        """
        # Compute query features
        q = self.encoder_q(im_q)
        q = F.normalize(q, dim=1)

        # Compute key features
        with torch.no_grad():
            self._momentum_update_key_encoder()

            # Shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            k = self.encoder_k(im_k)
            k = F.normalize(k, dim=1)
            # Undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # Compute logits
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1)

        # Apply temperature
        logits /= self.T

        # Labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # Dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels


class MoCoModule(pl.LightningModule):
    """PyTorch Lightning module for MoCo training"""
    def __init__(
        self,
        arch: str = "resnet3d50",
        feature_dim: int = 128,
        queue_size: int = 65536,
        moco_momentum: float = 0.999,
        temperature: float = 0.07,
        use_mlp: bool = True,
        learning_rate: float = 0.03,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        epochs: int = 100,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Create MoCo model
        self.model = MoCo(
            encoder_q=self._get_encoder(arch, feature_dim),
            encoder_k=self._get_encoder(arch, feature_dim),
            dim=feature_dim,
            K=queue_size,
            m=moco_momentum,
            T=temperature,
            mlp=use_mlp
        )

        self.criterion = nn.CrossEntropyLoss()

    def _get_encoder(self, arch: str, num_classes: int) -> nn.Module:
        """Get encoder architecture"""
        try:
            return models.__dict__[arch](num_classes=num_classes)
        except KeyError:
            raise ValueError(f"Architecture {arch} not found")

    def forward(self, im_q: torch.Tensor, im_k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model(im_q, im_k)

    def training_step(self, batch, batch_idx):
        """Training step"""
        im_q, im_k = batch["image0"], batch["image1"]
        output, target = self(im_q, im_k)
        loss = self.criterion(output, target)

        # Log metrics
        acc1, acc5 = self._accuracy(output, target, topk=(1, 5))
        self.log("train/loss", loss, on_step=True, on_epoch=True)
        self.log("train/acc1", acc1, on_step=True, on_epoch=True)
        self.log("train/acc5", acc5, on_step=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        """Configure optimizer and scheduler"""
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            self.hparams.learning_rate,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            self.hparams.epochs
        )
        
        return [optimizer], [scheduler]

    @staticmethod
    def _accuracy(output, target, topk=(1,)):
        """Compute accuracy for given topk"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    @staticmethod
    def add_model_specific_args(parent_parser):
        """Add model specific arguments"""
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        
        # Model parameters
        parser.add_argument("--arch", default="resnet3d50", type=str)
        parser.add_argument("--feature_dim", default=128, type=int)
        parser.add_argument("--queue_size", default=65536, type=int)
        parser.add_argument("--moco_momentum", default=0.999, type=float)
        parser.add_argument("--temperature", default=0.07, type=float)
        parser.add_argument("--use_mlp", default=True, type=bool)
        
        # Optimization parameters
        parser.add_argument("--learning_rate", default=0.03, type=float)
        parser.add_argument("--momentum", default=0.9, type=float)
        parser.add_argument("--weight_decay", default=1e-4, type=float)
        parser.add_argument("--epochs", default=100, type=int)
        
        return parser


@torch.no_grad()
def concat_all_gather(tensor):
    """Performs all_gather operation on the provided tensors"""
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output