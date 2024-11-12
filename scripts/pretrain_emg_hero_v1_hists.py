from pathlib import Path

import torch
from torch import nn

from emg_hero.datasets import get_sl_data
from emg_hero.trainer import Trainer
from emg_hero.encoders import MLP


def pretrain(experiment_folder):
    supervised_filename = experiment_folder / "supervised_data.mat"
    train_obs, train_actions, val_obs, val_actions = get_sl_data(supervised_filename, use_numpy=False)

    batch_size = 128
    epochs = 101
    lr = 1e-4
    device = "mps"

    trainer = Trainer(train_obs, train_actions, val_obs, val_actions, batch_size, epochs, lr, device)

    pretrained_hidden = [128 for _ in range(6)]
    policy = MLP(32, pretrained_hidden, 7, output_activation=nn.Sigmoid(), dropout=0.1)

    trainer.train(policy)

    torch.save(policy, experiment_folder / "pretrained_policy_v3.pt")


if __name__ == "__main__":
    experiment_folders = [
        Path("logs/2_2023_4_27_14_25_2"),
        Path("logs/3_2023_4_27_16_38_5"),
        Path("logs/4_2023_4_28_10_30_41"),
        Path("logs/5_2023_4_28_13_15_15"),
        Path("logs/6_2023_5_1_11_39_54"),
        Path("logs/7_2023_5_1_14_47_16"),
        Path("logs/8_2023_5_2_16_4_41"),
        Path("logs/9_2023_5_3_10_35_1"),
        Path("logs/10_2023_5_3_14_23_26"),
        Path("logs/11_2023_5_17_16_32_53"),
        Path("logs/11_2023_5_17_testing"),
        Path("logs/12_2023_6_6_22_11_55"),
        Path("logs/13_2023_6_12_12_59_19"),
        Path("logs/14_2023_6_12_22_53_45"),
        Path("logs/15_2023_6_21_11_14_33"),
        Path("logs/16_2023_6_22_15_26_7"),
    ]

    for experiment_folder in experiment_folders:
        pretrain(experiment_folder)