import copy
import json
import logging
import os
import pickle
import random
import re
import socket
from pathlib import Path

# import matplotlib.pyplot as plt
import numpy as np                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    

import torch

from d3rlpy import load_learnable
from d3rlpy.algos import AWACConfig, BCConfig
from d3rlpy.models import VectorEncoderFactory
from d3rlpy.dataset import MDPDataset
from d3rlpy.logging import FileAdapterFactory, CombineAdapterFactory
from d3rlpy.metrics import TDErrorEvaluator
from d3rlpy.constants import ActionSpace
from emg_hero.metrics import F1MacroEvaluator
from emg_hero.model_utility import get_newest_directory

from libemg.utils import get_windows

def pretrain_emg_hero_model_modified(
    pretrain_dataset: MDPDataset,
    pretest_dataset:MDPDataset,
    experiment_folder: Path,
    n_steps: int = 10_000,
):
    """Pretrains RL actor on supervised data

    Args:
        supervised_filename (str): file to supervised data file

    Returns:
        _type_: pretrained RL model
    """
    print('Potentially deprecated')
    supervised_train = pretrain_dataset 
    supervised_test = pretest_dataset

    filepath = Path(__file__).parent.resolve()
    folderpath = Path(*filepath.parts[:-1])
    #bc_config_file = folderpath / "config" / "bc_params.json"
    bc_model = BCConfig().create(device="cpu")#.from_json(bc_config_file)
    bc_model.build_with_dataset(supervised_train)

    #n_samples = np.sum([e.observations.shape[0] for e in supervised_train.episodes])
    #LOGGER.info("Train dataset size: %i", n_samples)

    f1_macro_evaluator = F1MacroEvaluator(supervised_test.episodes)

    bc_hist = bc_model.fit(
        supervised_train,
        n_steps=n_steps,
        n_steps_per_epoch=50,
        evaluators={
            "f1_macro": f1_macro_evaluator,
        },
        show_progress=False,
        experiment_name=("emg_hero_training"),
        logger_adapter=FileAdapterFactory(root_dir=experiment_folder),
    )

    f1s_array = [h[1]["f1_macro"] for h in bc_hist]

    newest_dir_path = get_newest_directory(experiment_folder)

    model_files = []
    for _, _, files in os.walk(newest_dir_path):
        for file in files:
            if file.endswith(".d3"):
                model_files.append(file)

    # sort model_files
    numbers = [int(re.findall("[0-9]+", m_file)[0]) for m_file in model_files]
    sorted_model_names = sorted(zip(numbers, model_files))

    # pick best model
    best_model_idx = np.argmax(f1s_array)
    best_model_filename = sorted_model_names[best_model_idx][1]

    # get path of best model
    best_model_path = newest_dir_path / best_model_filename
    path_best = print('The best models path is',best_model_path)
    #LOGGER.info(
     #   "Best model is %s with index %i and F1 score %f",
     #   best_model_filename,
      #  best_model_idx,
      #  np.max(f1s_array),
    #)
    #LOGGER.info(best_model_path)
    print(f"Took model {best_model_idx}: {best_model_filename}")

    bc_model = load_learnable(best_model_path)

    # plt.plot(f1s_array)
    # plt.title("Validation F1 macro")
    # plt.xlabel("epochs")
    # plt.ylabel("F1 macro")
    # plt.savefig((experiment_folder / "val_f1_macros.png"))
    # plt.show()

    # awac_config_file = folderpath / "config" / "awac_params.json"
    # awac_model = AWACConfig().create(device="cpu").from_json(awac_config_file)
    # awac_model.build_with_dataset(supervised_train)

    # copy BC weights to AWAC
    # awac_model.impl.modules.policy._mu = bc_model.impl.modules.imitator._mu
    # awac_model._impl._targ_policy._encoder = bc_model._impl._imitator._encoder
    # awac_model._impl._targ_policy._mu = bc_model._impl._imitator._fc

    return bc_model, path_best