from typing import Any, Dict

import hydra
import numpy as np
import omegaconf
import torch
import pytorch_lightning as pl
import torch.nn as nn
from eval_utils import load_model
from torch.nn import functional as F
from torch_scatter import scatter
from tqdm import tqdm

from cdvae.common.utils import PROJECT_ROOT
from cdvae.common.data_utils import (EPSILON, cart_to_frac_coords, mard,
                                     lengths_angles_to_volume,
                                     frac_to_cart_coords, min_distance_sqr_pbc)
from cdvae.pl_modules.embeddings import MAX_ATOMIC_NUM
from cdvae.pl_modules.embeddings import KHOT_EMBEDDINGS


def build_mlp(in_dim, hidden_dim, fc_num_layers, out_dim):
    mods = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for i in range(fc_num_layers - 1):
        mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    mods += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*mods)


class BaseModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # populate self.hparams with args and kwargs automagically!
        self.save_hyperparameters()

    def configure_optimizers(self):
        opt = hydra.utils.instantiate(self.hparams.optim.optimizer,
                                      params=self.parameters(),
                                      _convert_="partial")
        if not self.hparams.optim.use_lr_scheduler:
            return [opt]
        scheduler = hydra.utils.instantiate(self.hparams.optim.lr_scheduler,
                                            optimizer=opt)
        return {
            "optimizer": opt,
            "lr_scheduler": scheduler,
            "monitor": "val_loss"
        }


class CDVAE_z(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = hydra.utils.instantiate(
            self.hparams.encoder, num_targets=self.hparams.latent_dim)
        self.decoder = hydra.utils.instantiate(self.hparams.decoder)

        self.fc_mu = nn.Linear(self.hparams.latent_dim,
                               self.hparams.latent_dim)
        self.fc_var = nn.Linear(self.hparams.latent_dim,
                                self.hparams.latent_dim)

        self.fc_num_atoms = build_mlp(self.hparams.latent_dim,
                                      self.hparams.hidden_dim,
                                      self.hparams.fc_num_layers,
                                      self.hparams.max_atoms + 1)
        self.fc_lattice = build_mlp(self.hparams.latent_dim,
                                    self.hparams.hidden_dim,
                                    self.hparams.fc_num_layers, 6)
        self.fc_composition = build_mlp(self.hparams.latent_dim,
                                        self.hparams.hidden_dim,
                                        self.hparams.fc_num_layers,
                                        MAX_ATOMIC_NUM)

        sigmas = torch.tensor(np.exp(
            np.linspace(np.log(self.hparams.sigma_begin),
                        np.log(self.hparams.sigma_end),
                        self.hparams.num_noise_level)),
                              dtype=torch.float32)

        self.sigmas = nn.Parameter(sigmas, requires_grad=False)

        type_sigmas = torch.tensor(np.exp(
            np.linspace(np.log(self.hparams.type_sigma_begin),
                        np.log(self.hparams.type_sigma_end),
                        self.hparams.num_noise_level)),
                                   dtype=torch.float32)

        self.type_sigmas = nn.Parameter(type_sigmas, requires_grad=False)

        self.embedding = torch.zeros(100, 92)
        for i in range(100):
            self.embedding[i] = torch.tensor(KHOT_EMBEDDINGS[i + 1])

        self.fc_property = build_mlp(self.hparams.latent_dim,
                                     self.hparams.hidden_dim,
                                     self.hparams.fc_num_layers, 1)

        # obtain from datamodule.
        self.lattice_scaler = None
        self.scaler = None
        # self.pretrained_model = self.load_pretrained_model(
        #     '/home/zlsong/cdvae-main/hydra/singlerun/2022-06-15/gaspy/')
        # self.pretrained_model.freeze()

    def forward(self, batch, training):
        # _, _, z = self.pretrained_model.encode(batch)
        self.encoder.freeze()
        _, _, z = self.encode(batch)
        property_loss = self.property_loss(z, batch)
        return {'property_loss': property_loss, 'z': z}

    def load_pretrained_model(self, model_path):
        model, _, _ = load_model(model_path, load_data=False)
        return model

    def predict_property(self, z):
        self.scaler.match_device(z)
        return self.scaler.inverse_transform(self.fc_property(z))

    def property_loss(self, z, batch):
        return F.mse_loss(self.fc_property(z), batch.y)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        outputs = self(batch, training=True)
        log_dict, loss = self.compute_stats(batch, outputs, prefix='train')
        self.log_dict(
            log_dict,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs = self(batch, teacher_forcing=False, training=False)
        log_dict, loss = self.compute_stats(batch, outputs, prefix='val')
        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs = self(batch, teacher_forcing=False, training=False)
        log_dict, loss = self.compute_stats(batch, outputs, prefix='test')
        self.log_dict(log_dict, )
        return loss

    def compute_stats(self, batch, outputs, prefix):
        property_loss = outputs['property_loss']

        loss = property_loss

        log_dict = {
            f'{prefix}_property_loss': property_loss,
        }

        return log_dict, loss


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    model: pl.LightningModule = hydra.utils.instantiate(
        cfg.model,
        optim=cfg.optim,
        data=cfg.data,
        logging=cfg.logging,
        _recursive_=False,
    )
    return model


if __name__ == "__main__":
    main()
