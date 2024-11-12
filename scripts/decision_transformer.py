import collections
from pathlib import Path

import torch
import wandb
from scipy.io import loadmat
import d3rlpy
import numpy as np
from d3rlpy.constants import ActionSpace
from d3rlpy.dataset import Episode, ReplayBuffer, FIFOBuffer, MDPDataset

from emg_hero.datasets import load_histories, load_emg_hero_dataset
from emg_hero.label_transformer import LabelTransformer, MultilabelConverter
from emg_hero.metrics import get_reachable_notes
from emg_hero.defs import MoveConfig
from emg_hero.metrics import get_rewards
from emg_hero.utils import WandbAdapterFactory

from emg_hero.configs import BaseConfig


def load_dataset(experiment_folder, history_filenames, switch_lines):
    mat_filename = experiment_folder / 'pretrained_network_params.mat'
    mat_params = loadmat(mat_filename, squeeze_me=True)
    movements = mat_params["movements"]
    move_config = MoveConfig(movements=movements)

    label_transformer = LabelTransformer(move_config=move_config)

    episodes_list = []
    for history_filename in history_filenames:
        history, _ = load_histories(history_filenames=[history_filename])
        ideal_actions = get_reachable_notes(history, label_transformer, switch_lines=switch_lines)
        episodes_list.append({
            'observations': np.array(history['features']),
            'recording_reward': np.sum(history['rewards']),
            'ideal_actions': ideal_actions,
            'recording_actions': np.array(history['actions']),
        })
    return episodes_list


def main():
    entity = "kilian"
    _ = wandb.init(project='emg_hero', entity=entity)
    base_config = BaseConfig(**wandb.config)
    wandb.config = base_config
    torch.manual_seed(base_config.random_seed)

    experiment_folder = Path(".") / "logs" / "test_exp"
    history_filenames = [
                    './logs/test_exp/emg_hero_history_2024_1_11_9_48_34.pkl',
                    './logs/test_exp/emg_hero_history_2024_1_11_9_51_10.pkl',
                    './logs/test_exp/emg_hero_history_2024_1_11_9_53_27.pkl',
                    './logs/test_exp/emg_hero_history_2024_1_11_9_55_53.pkl',
                    './logs/test_exp/emg_hero_history_2024_1_11_9_58_39.pkl',
                    './logs/test_exp/emg_hero_history_2024_1_11_10_1_13.pkl',
                    './logs/test_exp/emg_hero_history_2024_1_11_10_4_4.pkl',
                    './logs/test_exp/emg_hero_history_2024_1_11_10_6_37.pkl',
                    './logs/test_exp/emg_hero_history_2024_1_11_10_9_18.pkl',]
    switch_lines = True

    config = base_config.algo

    episodes_list = load_dataset(experiment_folder, history_filenames, switch_lines)
    converter = MultilabelConverter()

    dt = d3rlpy.algos.DiscreteDecisionTransformerConfig(context_size=20, batch_size=128, max_timestep=2200).create(device="cpu")

    episode_metrics = collections.defaultdict(list)
    train_dataset = None
    buffer = FIFOBuffer(limit=config.buffer_size)

    target_reward = 1.0
    for episode_idx, episode_dict in enumerate(episodes_list):
        observations = episode_dict['observations']
        ideal_actions = episode_dict['ideal_actions']
        ideal_labels = converter.convert_list(ideal_actions)
        recording_actions = episode_dict['recording_actions']

        if train_dataset is None:
            timeouts = np.zeros_like(ideal_labels)
            timeouts[-1] = 1
            build_dataset = MDPDataset(observations=observations,
                            actions=ideal_labels,
                            rewards=np.ones_like(ideal_labels),
                            terminals=np.zeros_like(ideal_labels),
                            timeouts=timeouts,
                            action_space=ActionSpace.DISCRETE,)
            dt.build_with_dataset(build_dataset)

        actor = dt.as_stateful_wrapper(target_return=1)
        pred_labels = []
        for observation in observations:
            val_label = actor.predict(observation, target_reward)
            pred_labels.append(val_label)
        pred_labels = np.stack(pred_labels)
        actor.reset()

        actions = converter.convert_back_list(pred_labels)
        rewards = get_rewards(actions, ideal_actions)

        ep_sim_rec_reward = get_rewards(recording_actions, ideal_actions).sum() 
        ep_reward = rewards.sum()
        ep_emr =  (pred_labels == ideal_labels).astype(float).mean()

        episode_metrics['reward'].append(ep_reward)
        episode_metrics['emr'].append(ep_emr)
        wandb.run.log({'reward': ep_reward,
                       'recording_reward': episode_dict['recording_reward'],
                       'sim_rec_reward': ep_sim_rec_reward,
                       'emr': ep_emr})
        
        id_rewards = np.ones_like(rewards)
        ideal_episode = Episode(observations=observations, actions=ideal_labels, rewards=id_rewards, terminated=True)
        episode = Episode(observations=observations, actions=pred_labels, rewards=rewards, terminated=True)

        if train_dataset is None:
            terminals = np.zeros_like(rewards)
            terminals[-1] = 1
            episodes = [episode, ideal_episode]
            train_dataset = ReplayBuffer(buffer=buffer, episodes=episodes, action_space=ActionSpace.DISCRETE)
        else:
            train_dataset.append_episode(episode)
            train_dataset.append_episode(ideal_episode)

        dt.fit(
            train_dataset,
            n_steps=2000,
            n_steps_per_epoch=200,
            logger_adapter=WandbAdapterFactory(),
        )

if __name__ == "__main__":
    main()