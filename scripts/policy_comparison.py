import sys
import pickle
from collections import defaultdict
from pathlib import Path
from typing import List

import wandb
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.distributions as D
from tensordict.nn.distributions import NormalParamExtractor
from torchaudio.models import Conformer

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import StochasticWeightAveraging
from torchmetrics.classification import F1Score

sys.path.append('~/dev/EMGHero')

from emg_hero.datasets import load_histories
from emg_hero.defs import MoveConfig
from emg_hero.label_transformer import LabelTransformer
from emg_hero.metrics import get_reachable_notes
from encoders.tcn import TCN

from emg_hero.configs import PolicyConfig

# TODO add transformer encoder?
# sigmoid vs softmax
# layernorm



def binary_to_decimal(actions):
    # convert binary ([1,0,0,1,0]) to decimal ([1, -1])
    dec_actions = actions[:,:-1:2] - actions[:,1::2]
    return dec_actions


def decimal_to_binary(dec_actions):
    bin_actions = torch.zeros(dec_actions.shape[0], dec_actions.shape[1]*2+1).to(dec_actions.device)
    bin_actions[:,:-1:2][dec_actions>0] = dec_actions[dec_actions>0]
    bin_actions[:,1::2][dec_actions<0] = dec_actions[dec_actions<0].abs()
    return bin_actions


class EMGDataset(Dataset):
    def __init__(self, obs_list: List[Tensor], act_list: List[Tensor], n = None):
        if n is None:
            self.obs = torch.cat(obs_list, dim=0)
            self.actions = torch.cat(act_list, dim=0)
        else:
            self.obs = torch.cat([o.unfold(0, n, 1) for o in obs_list], dim=0)#.swapaxes(1,2) # shape (batch_size, n_feats, n)
            self.actions = torch.cat([a[n-1:] for a in act_list], dim=0)

        assert len(self.obs) == len(self.actions), 'Obs action missmatch'

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.actions[idx]


class MLP(nn.Module):
    def __init__(self, output_size, hidden_sizes, use_batch_norm=False, dropout=0.1,
                 activation=nn.ReLU, output_activation=nn.Tanh):
        super(MLP, self).__init__()

        layers = []
        layers.append(nn.LazyLinear(hidden_sizes[0]))
        layers.append(nn.Dropout(p=dropout))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        layers.append(activation(inplace=True))

        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.Dropout(p=dropout))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_sizes[i + 1]))
            layers.append(activation(inplace=True))

        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        if output_activation is not None:
            layers.append(output_activation())

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        return self.encoder(x)


class TCNEncoder(nn.Module):
    def __init__(self, n_stacked_features, out_channels, kernel_size, channels, layers,
                 n_layers=0, hidden_size=256, dropout=0.1, do_pooling=True):
        super().__init__()
        self.encoder = TCN(in_channels=n_stacked_features, out_channels=out_channels, kernel_size=kernel_size,
                channels=channels, layers=layers, bias=True, fwd_time=True)
        
        self.do_pooling = do_pooling
        if do_pooling:
            self.pooling = nn.AvgPool1d(kernel_size=2, stride=1, padding=1)

        layers = []
        for _ in range(n_layers):
            layers.append(nn.LazyLinear(hidden_size))
            layers.append(nn.Dropout(p=dropout))
            layers.append(nn.ReLU(inplace=True))

        if n_layers > 0:
            self.mlp_layers = nn.Sequential(*layers)
        else:
            self.mlp_layers = None

    def forward(self, x):
        x = self.encoder(x)
        if self.do_pooling:
            x = self.pooling(x)
        if self.mlp_layers is not None:
            x = self.mlp_layers(x)
        return x


class ConformerEncoder(nn.Module):
    def __init__(self, input_dim, n_layers=0, hidden_size=256, dropout=0.1, pool_out_size=8):
        super().__init__()

        self.conformer = Conformer(
            input_dim=input_dim,
            num_heads=4,
            ffn_dim=128,
            num_layers=4,
            depthwise_conv_kernel_size=31,
        )

        self.pooling = nn.AdaptiveMaxPool1d(output_size=pool_out_size)

        layers = []
        for _ in range(n_layers):
            layers.append(nn.LazyLinear(hidden_size))
            layers.append(nn.Dropout(p=dropout))
            layers.append(nn.ReLU(inplace=True))

        if n_layers > 0:
            self.mlp_layers = nn.Sequential(*layers)
        else:
            self.mlp_layers = None

    def forward(self, x):
        # input shape (batch_size, n_feats, n_stacked_features)
        x = x.permute(0, 2, 1) # we want (batch_size, n_stacked_features, n_feats)
        B, C, _ = x.shape
        x, _ = self.conformer(x, lengths=torch.tensor([C]*B))
        x = self.pooling(x)
        if self.mlp_layers is not None:
            x = self.mlp_layers(x)
        return x


class CNN1dEncoder(nn.Module):
    def __init__(self, input_size, n_channels, dropout, kernel_size, padding, n_layers=0, hidden_size=256):
        super().__init__()

        conv_layers = [nn.Conv1d(input_size, n_channels, kernel_size=kernel_size, padding=padding),
                       nn.Dropout(dropout),
                       nn.ReLU(),]
        
        n_conv = 2
        # TODO change kernel sizes and channels
        for _ in range(n_conv):
            conv_layers.append(nn.Conv1d(n_channels, n_channels, kernel_size=kernel_size, padding=padding))
            conv_layers.append(nn.Dropout(dropout))
            conv_layers.append(nn.ReLU())

        self.encoder = nn.Sequential(*conv_layers)

        pool_out_size = 8
        # self.pooling = nn.AvgPool1d(kernel_size=5, padding=1)
        # self.pooling = nn.AdaptiveAvgPool1d(output_size=pool_out_size)
        self.pooling = nn.AdaptiveMaxPool1d(output_size=pool_out_size)

        layers = []
        for _ in range(n_layers):
            layers.append(nn.LazyLinear(hidden_size))
            layers.append(nn.Dropout(p=dropout))
            layers.append(nn.ReLU(inplace=True))

        if n_layers > 0:
            self.mlp_layers = nn.Sequential(*layers)
        else:
            self.mlp_layers = None

    def forward(self, x):
        # input shape (batch_size, n_feats, n_stacked_features)
        x = x.permute(0, 2, 1)
        # for cnn we want (batch_size, n_stacked_features, n_feats)
        x = self.encoder(x)
        x = x.permute(0, 2, 1)
        x = self.pooling(x)
        # TODO could use LazyLinear here
        if self.mlp_layers is not None:
            x = self.mlp_layers(x)
        return x


def build_encoder(config):
    if config.encoder_type == 'mlp':
        # input_size = config.n_feats * config.n_stacked_features
        output_size = config.mlp.hidden_size
        hidden_sizes = [config.mlp.hidden_size for _ in range(config.mlp.n_layers)]
        encoder = MLP(output_size,
                      hidden_sizes,
                      use_batch_norm=config.mlp.use_batch_norm,
                      dropout=config.mlp.dropout,
                      output_activation=None)

    elif config.encoder_type == 'cnn':
        input_size = config.n_stacked_features
        # output_size = config.cnn.n_channels * config.n_feats
        encoder = CNN1dEncoder(input_size,
                               config.cnn.n_channels,
                               config.cnn.dropout,
                               kernel_size=config.cnn.kernel_size,
                               padding=config.cnn.padding,
                               n_layers=config.cnn.n_layers,
                               hidden_size=config.cnn.hidden_size)
    
    elif config.encoder_type == 'tcn':
        # output_size = config.tcn.out_channels * config.n_feats
        encoder = TCNEncoder(config.n_stacked_features,
                             out_channels=config.tcn.out_channels,
                             kernel_size=config.tcn.kernel_size,
                             channels=config.tcn.channels,
                             layers=config.tcn.layers,
                             n_layers=config.tcn.n_mlp_layers,
                             hidden_size=config.tcn.hidden_size,
                             dropout=config.tcn.dropout,
                             do_pooling=config.tcn.do_pooling)
    elif config.encoder_type == 'conformer':
        # output_size = config.tcn.out_channels * config.n_feats
        encoder = ConformerEncoder(input_dim=32)
    else:
        raise ValueError(f"Unknown encoder type {config.encoder_type}")
    output_size = None
    return encoder, output_size



class DeterministicDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # encoder, output_size = build_encoder(config)
        if config.out_actiation_key == 'sigmoid':
            output_activation = nn.Sigmoid()
            n_actions = config.n_actions
        elif config.out_actiation_key == 'tanh':
            output_activation = nn.Tanh()
            n_actions = config.n_actions // 2
        else:
            raise ValueError(f"Unknown output activation {config.out_actiation_key}")
        
        self.decoder = nn.Sequential(
                # encoder,
                nn.Flatten(start_dim=1),
                nn.LazyLinear(n_actions),
                output_activation,
            )

    def forward(self, x):
        return self.decoder(x)


class StochasticDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # encoder, output_size = build_encoder(config)
        if not config.out_actiation_key == 'sigmoid':
            raise ValueError(f"Stochastic decoder only supports sigmoid activation, got {config.out_actiation_key}")

        self.decoder = nn.Sequential(
            # encoder,
            nn.Flatten(start_dim=1),
            nn.LazyLinear(2 * config.n_actions),
            NormalParamExtractor(),
        )

    def forward(self, x):
        mean, log_std = self.decoder(x)
        # mean, log_std = x.split(x.size(-1) // 2, -1)
        return mean, log_std
    
    def sample(self, x, deterministic=False):
        mean, std = self.forward(x)
        if deterministic:
            return F.sigmoid(mean)
        # std = torch.exp(log_std)
        dist = D.Normal(mean, std)
        sample = dist.rsample()
        return F.sigmoid(sample)


class EMGHeroModel(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.lr = config.lr
        self.deterministic = config.deterministic

        self.encoder, _ = build_encoder(config)

        if self.deterministic:
            self.decoder = DeterministicDecoder(config)
        else:
            self.decoder = StochasticDecoder(config)

        if config.criterion_key == 'mse':
            self.criterion = nn.MSELoss()
        elif config.criterion_key == 'bce':
            self.criterion = nn.BCELoss()
        else:
            raise ValueError(f"Unknown criterion {config.criterion_key}")

        self.f1_score = F1Score(task='binary', num_classes=config.n_actions, average='macro')

    def forward(self, x):
        x = self.encoder(x)
        if self.deterministic:
            return self.decoder(x)
        else:
            return self.decoder.sample(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        # TODO what to do with f1 when using tanh?
        y_hat_bin = decimal_to_binary(y_hat)
        y_bin = decimal_to_binary(y)
        f1 = self.f1_score(y_hat_bin, y_bin)
        self.log('val_loss', loss)
        self.log('val_f1', f1)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    

def load_history_dataloader(history_filenames, batch_size, n_stacked_features, shuffle=False, num_workers=1):
    history, terminal_idxs = load_histories(history_filenames)

    # TODO split recordings
    observations = torch.tensor(history['features'], dtype=torch.float32)
    move_config = MoveConfig()
    label_transformer = LabelTransformer(move_config=move_config)
    ideal_actions = get_reachable_notes(history, label_transformer, switch_lines=False)
    ideal_actions = torch.tensor(ideal_actions, dtype=torch.float32)

    last_terminal = 0
    obs_list = []
    act_list = []
    for term_idx in terminal_idxs:
        obs_list.append(observations[last_terminal:term_idx])
        act_list.append(ideal_actions[last_terminal:term_idx])
        last_terminal = term_idx

    # TODO change dataset with tanh activation

    dataset = EMGDataset(obs_list, act_list, n=n_stacked_features)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader


def get_dataloaders(data_path, batch_size, n_stacked_features, num_workers=1, tanh_activation=False):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    obs_list = [torch.tensor(ep["observations"], dtype=torch.float32) for ep in data["episodes"]]
    actions_list = [torch.tensor(ep["ideal_actions"], dtype=torch.float32) for ep in data["episodes"]]

    if tanh_activation:
        # convert binary actions to one row per DOF
        # actions_list = [act[:,:-1:2] - act[:,1::2] for act in actions_list]
        actions_list = [binary_to_decimal(act) for act in actions_list]
    
    train_dataset = EMGDataset(obs_list[:-1], actions_list[:-1], n=n_stacked_features)
    val_dataset = EMGDataset(obs_list[-1:], actions_list[-1:], n=n_stacked_features)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=True)
    return train_loader, val_loader


def train_experiment(data_path, config, logger):
    tanh_activation = True if config.out_actiation_key == 'tanh' else False

    train_loader, val_loader = get_dataloaders(data_path,
                                               batch_size=config.batch_size,
                                               n_stacked_features=config.n_stacked_features,
                                               num_workers=config.n_workers,
                                               tanh_activation=tanh_activation)

    model = EMGHeroModel(config)

    # init lazy layers
    x, _ = next(iter(train_loader))
    model(x)
    
    callbacks = [StochasticWeightAveraging(swa_lrs=1e-2)] if config.do_swag else []
    trainer = L.Trainer(max_epochs=config.epochs, logger=logger,
                        enable_checkpointing=False, callbacks=callbacks, accelerator='cpu')
    trainer.fit(model, train_loader, val_loader)

    val_metrics = trainer.validate(model, val_loader)
    return val_metrics


def main():
    entity = "kilian"
    _ = wandb.init(project='emg_hero', entity=entity)
    config = PolicyConfig(**wandb.config)
    L.seed_everything(config.seed)
    logger = WandbLogger(project='emg-hero', name='det_vs_stochastic', config=config)
    emg_hero_path = Path('./') # TODO use file location
    dataset_path = emg_hero_path / 'datasets' / 'online'
    
    person_keys = ['person_2.pkl']
    # person_keys = [f'person_{i}.pkl' for i in range(17)]

    population_metrics = defaultdict(list)
    for person_key in person_keys:
        data_path = dataset_path / person_key
        val_metrics = train_experiment(data_path, config, logger)
        for key, value in val_metrics[-1].items():
            population_metrics[key].append(value)
    
    mean_population_metrics = {f"population/{key}": np.mean(values) for key, values in population_metrics.items()}
    wandb.log(mean_population_metrics)


if __name__ == "__main__":
    main()
