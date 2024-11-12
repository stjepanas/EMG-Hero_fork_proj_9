'''Defines functions to load datasets
'''
import copy
import pickle
import logging
from typing import Tuple

import numpy as np
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset
from d3rlpy.dataset import MDPDataset
from d3rlpy.constants import ActionSpace


class FeatureDataset(Dataset):
    def __init__(self, features, actions):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.actions = torch.tensor(actions, dtype=torch.float32)

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.features[index]
        y = self.actions[index]
        return x, y


def extend_numbers(date_string):
    return '_'.join([f'{int(x):02d}' for x in date_string.split('_')])


# FIXME duplicate with BionicRLHF
def get_history_filenames(experiment_folder):
    history_filenames = [
        file
        for file in experiment_folder.glob("emg_hero_history_????_*.pkl")
        if not file.as_posix().endswith("backup.pkl")
    ]
    history_filenames.sort(key=lambda x: int(extend_numbers(x.stem.split('history_')[-1])))
    return history_filenames


# FIXME duplicate with BionicRLHF
def get_sl_data(supervised_filename, use_numpy=True):
    mat_data = loadmat(supervised_filename, squeeze_me=True)

    if use_numpy:
        sl_train_obs = np.swapaxes(
            mat_data["person"]["featsTrain"].take(0), 0, 1
        ).astype(np.float64)
        sl_train_actions = np.swapaxes(
            mat_data["person"]["yTrain"].take(0), 0, 1
        ).astype(np.float64)
        sl_val_obs = np.swapaxes(mat_data["person"]["featsTest"].take(0), 0, 1).astype(
            np.float64
        )
        sl_val_actions = np.swapaxes(mat_data["person"]["yTest"].take(0), 0, 1).astype(
            np.float64
        )
    else:
        sl_train_obs = torch.tensor(
            np.swapaxes(mat_data["person"]["featsTrain"].take(0), 0, 1),
            dtype=torch.float32,
        )
        sl_train_actions = torch.tensor(
            np.swapaxes(mat_data["person"]["yTrain"].take(0), 0, 1), dtype=torch.float32
        )
        sl_val_obs = torch.tensor(
            np.swapaxes(mat_data["person"]["featsTest"].take(0), 0, 1),
            dtype=torch.float32,
        )
        sl_val_actions = torch.tensor(
            np.swapaxes(mat_data["person"]["yTest"].take(0), 0, 1), dtype=torch.float32
        )

    return sl_train_obs, sl_train_actions, sl_val_obs, sl_val_actions



def load_supervised_dataset(filename: str,
                            reward: float = 1.,
                            use_state_with_last_action: bool = False,
                            ) -> tuple[MDPDataset, MDPDataset]:
    '''Load supervised dataset from *.mat file

    Args:
        filename (str): Filename of *.mat file
        reward (float, optional): Reward for each sample. Defaults to 1..

    Returns:
        tuple[MDPDataset, MDPDataset]: Train and test dataset.
    '''
    mat_data = loadmat(filename, squeeze_me = True)

    pretrain_train_observations = np.swapaxes(mat_data['person']['featsTrain'].take(0),
                                              0, 1).astype(np.float64)
    pretrain_train_actions = np.swapaxes(mat_data['person']['yTrain'].take(0),
                                         0, 1).astype(np.float64)
    pretrain_test_observations = np.swapaxes(mat_data['person']['featsTest'].take(0),
                                             0, 1).astype(np.float64)
    pretrain_test_actions = np.swapaxes(mat_data['person']['yTest'].take(0),
                                        0, 1).astype(np.float64)

    action_size = pretrain_train_actions.shape[1]

    n_pretrain_train_samples = pretrain_train_observations.shape[0]
    n_pretrain_test_samples = pretrain_test_observations.shape[0]

    # pretend to have perfect rewards
    pretrain_train_rewards = np.zeros((n_pretrain_train_samples)) + reward
    pretrain_test_rewards = np.zeros((n_pretrain_test_samples)) + reward

    pretrain_train_terminals = np.array([1 if ((i+1) % 47 == 0 and i > 0) else 0 \
                                         for i in range(pretrain_train_actions.shape[0])])
    pretrain_test_terminals = np.array([1 if ((i+1) % 47 == 0 and i > 0) else 0 \
                                        for i in range(pretrain_test_actions.shape[0])])

    # create 0 terminals, which would mean that something is "done", we only have timeouts
    train_terminals_placeholder = np.zeros_like(pretrain_train_terminals)
    test_terminals_placeholder = np.zeros_like(pretrain_test_terminals)

    # FIXME
    pretrain_train_actions[pretrain_train_actions == 1.] = 0.9999

    if use_state_with_last_action:
        pretrain_train_observations, _ = append_state_with_last_action(
                                            observations=pretrain_train_observations,
                                            actions=pretrain_train_actions,
                                            terminal_inds=pretrain_train_terminals,
                                            action_size=action_size)
        pretrain_test_observations, _ = append_state_with_last_action(
                                            observations=pretrain_test_observations,
                                            actions=pretrain_test_actions,
                                            terminal_inds=pretrain_test_terminals,
                                            action_size=action_size)
    pretrain_train = MDPDataset(
        pretrain_train_observations,
        pretrain_train_actions,
        pretrain_train_rewards,
        train_terminals_placeholder,
        timeouts = pretrain_train_terminals,
        action_space = ActionSpace.CONTINUOUS
    )

    pretrain_test = MDPDataset(
        pretrain_test_observations,
        pretrain_test_actions,
        pretrain_test_rewards,
        test_terminals_placeholder,
        timeouts = pretrain_test_terminals,
        action_space = ActionSpace.CONTINUOUS
    )

    return pretrain_train, pretrain_test

def load_histories(history_filenames: list[str]) -> tuple[dict, list]:
    '''Load EMG Hero histories from a list of filenames

    Args:
        history_filenames (list): list of history filenames

    Returns:
        dict: concatenated histories
        list: indices of last entry of a given history
    '''
    history = None
    terminal_inds = []
    current_length = 0
    for hist_filename in history_filenames:
        with open(hist_filename, 'rb') as file:
            file_history = pickle.load(file)

        sample_length = len(file_history['actions'])
        terminal_inds.append((current_length+sample_length-1))
        current_length += sample_length

        if history is None:
            history = file_history
        else:
            for key in history.keys():
                if isinstance(history[key], list):
                    history[key] = history[key] + file_history[key]

    return history, terminal_inds

def append_state_with_last_action(observations: np.ndarray,
                                  actions: np.ndarray,
                                  terminal_inds: list,
                                  action_size: int
                                  ) -> tuple[np.ndarray, int]:
    '''Appends observations with last action

    Args:
        observations (np.ndarray): features / observations
        actions (np.ndarray): predicted actions
        terminal_inds (list): episode terminals
        action_size (int): size of actions space

    Returns:
        tuple[np.ndarray, int]: updated observations and size of feature space
    '''
    initial_action = np.zeros(action_size)
    initial_action[-1] = 1.

    shifted_actions = np.vstack((initial_action, actions))[:-1, :]

    # set all initial action test
    for term_ind in terminal_inds[:-1]:
        shifted_actions[(term_ind + 1)] = initial_action

    observations = np.hstack((observations, shifted_actions))
    feature_size = observations.shape[1]

    return observations, feature_size

def load_emg_hero_dataset(history: dict,
                          terminal_inds: list,
                          use_state_with_last_action: bool = False):
    '''Build MDPDataset from EMG Hero history

    Args:
        history (dict): EMG Hero histories, that are saved after finishing a song
        terminal_inds (list): terminal indices, so last indices of each song

    Returns:
        d3rlpy.dataset.MDPDataset: dataset to train / test model
    '''
    feature_size = len(history['features'][0])
    action_size = len(history['actions'][0])

    observations = np.array(history['features'])
    rewards = np.array(history['rewards'])
    actions = np.array(history['actions'])
    actions[actions == 1.] = 0.9999 # make sure that dataset is continuos

    new_terminal_inds = copy.deepcopy(terminal_inds)

    if use_state_with_last_action:
        observations, feature_size = append_state_with_last_action(
                                            observations=observations,
                                            actions=actions,
                                            terminal_inds=new_terminal_inds,
                                            action_size=action_size)

    # delete big elements
    idxs_to_delete = []
    for i in range(observations.shape[0]):
        if np.max(observations[i,:]) > 10000 or (observations[i,:] == 0).all():
            logging.warning('Deleting %i cause too high value', i)
            idxs_to_delete.append(i)

    filtered_history = copy.deepcopy(history)
    list_history_keys = [k for k, v in history.items() if isinstance(v, list) and not k == 'song']

    count = 0
    for idx_ in idxs_to_delete:
        new_idx = idx_ - count
        observations = np.delete(observations, new_idx, 0)
        rewards = np.delete(rewards, new_idx, 0)
        actions = np.delete(actions, new_idx, 0)

        # delete in history
        for h_key in list_history_keys:
            del filtered_history[h_key][new_idx]

        for i, t_idx in enumerate(new_terminal_inds):
            if t_idx >= new_idx:
                new_terminal_inds[i] = t_idx - 1
        count += 1

    terminals = np.zeros_like(rewards)
    episode_terminals = np.zeros_like(rewards)
    for term_idx in new_terminal_inds:
        episode_terminals[term_idx] = 1

    # avoid all 0 actions
    if (actions == 0.).all():
        actions[0,0] = 0.999

    dataset = MDPDataset(
                observations,
                actions,
                rewards,
                terminals,
                timeouts = episode_terminals,
                action_space = ActionSpace.CONTINUOUS
                )

    return dataset, filtered_history, feature_size, action_size

def load_motion_test_data(files: list[str]) -> dict:
    '''Loads data from motion test files that has been transformed with matlab helper script

    Args:
        files (list[str]): list of filenames

    Returns:
        dict: motion test data
    '''

    motion_test_data = {}
    for label, file in files.items():
        data = {
            'mean_accuracy': [],
            'accuracy': [],
            'completion_time': [],
            'predictions': [],
            'individual_movements': [],
            'movements': [],
        }
        test_data = loadmat(('../data/emg_hero/'+file), squeeze_me=True)
        accs = test_data['data'].take(0)[0]
        accs[np.isnan(accs)] = 0.
        completion_time = test_data['data'].take(0)[1]
        completion_time[np.isnan(completion_time)] = 10.

        data['mean_accuracy'].append(accs.mean())
        data['accuracy'].append(accs)
        data['completion_time'].append(completion_time.mean())

        if len(test_data['data'].take(0)) > 2:
            predictions = test_data['data'].take(0)['predictedIndex']
            data['predictions'].append(predictions)
            individual_movements = test_data['data'].take(0)['individualMovements']
            data['individual_movements'].append(individual_movements)
            movements = test_data['data'].take(0)['movements']
            data['movements'].append(movements)

        motion_test_data[label] = data

    return motion_test_data
