"""Defines functions that are needed to load and train models
"""
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
from d3rlpy.algos import AWACConfig, BCConfig, TD3PlusBCConfig, TD3Config
from d3rlpy.models import VectorEncoderFactory
from d3rlpy.dataset import MDPDataset
from d3rlpy.logging import FileAdapterFactory, CombineAdapterFactory
from d3rlpy.metrics import TDErrorEvaluator
from d3rlpy.constants import ActionSpace
from scipy.io import loadmat, savemat

from libemg.utils import get_windows

from emg_hero.datasets import (
    load_emg_hero_dataset,
    load_histories,
    load_supervised_dataset,
)
from emg_hero.defs import (
    get_current_timestamp,
    MoveConfig,
    PRETRAINED_MODEL_FILENAME,
    PRETRAINED_POLICY_FILENAME,
    SUPERVISED_DATA_FILENAME,
)

from emg_hero.label_transformer import LabelTransformer
from emg_hero.metrics import EMGHeroMetrics, F1MacroEvaluator, ExactMatchEvaluator
from emg_hero.utils import WandbAdapterFactory

# set logger level
LOGGER = logging.getLogger(__name__)
LOGGING_HANDLER = logging.StreamHandler()
LOGGING_HANDLER.setLevel(logging.WARNING)
LOGGING_HANDLER.setFormatter(
    logging.Formatter("%(name)s - %(levelname)s - %(message)s")
)
LOGGER.addHandler(LOGGING_HANDLER)


def get_policy_params_dict(algo) -> dict:
    """Returns policy params of d3rlpy algorithm as dict

    Args:
        algo (_type_): Algorithm to extract policy parameters

    Returns:
        dict: policy parameters
    """
    policy_state_dict = algo._impl.policy.state_dict()
    numpy_dict = {}
    for key, item in policy_state_dict.items():
        new_key = key.replace(".", "").replace("_", "")
        numpy_dict[new_key] = item.numpy()

    return numpy_dict


def copy_params_to_policy(policy_params, all_weights, all_biases, deterministic=False):
    layer_idx = 0
    for layer in policy_params._encoder._layers:
        if isinstance(layer, torch.nn.Linear):
            mat_weights = torch.tensor(all_weights[layer_idx]).to('cpu')
            if len(torch.where(torch.isnan(mat_weights))[0]):
                LOGGER.warning("Found NaN weights, will randomize")
            mat_weights[torch.isnan(mat_weights)] = torch.rand(1)
            layer.weight.data = mat_weights

            mat_bias = torch.tensor(all_biases[layer_idx]).to('cpu')
            if len(torch.where(torch.isnan(mat_bias))[0]):
                LOGGER.warning("Found NaN bias, will randomize")
            mat_bias[torch.isnan(mat_bias)] = torch.rand(1)
            layer.bias.data = mat_bias

            layer_idx += 1

    if deterministic:
        policy_params._fc.weight.data = torch.tensor(all_weights[6]).to('cpu')#.clone().detach().requires_grad_(True)
        policy_params._fc.bias.data = torch.tensor(all_biases[6]).to('cpu')#.clone().detach().requires_grad_(True)
    else:
        policy_params._mu.weight.data = torch.tensor(all_weights[6]).to('cpu')#.clone().detach().requires_grad_(True)
        policy_params._mu.bias.data = torch.tensor(all_biases[6]).to('cpu')#.clone().detach().requires_grad_(True)


def load_torch_policy_params(pt_filename):
    pt_model_params = torch.load(pt_filename, map_location='cpu')
    weights = []
    biases = []
    for layer in pt_model_params.network:
        if isinstance(layer, torch.nn.Linear):
            weights.append(layer.weight.data)
            biases.append(layer.bias.data)

    return weights, biases


def build_algo(pt_weights, pt_biases, base_config):
    config = base_config.algo

    use_pt_params = False
    floornoise = None
    if pt_weights is not None:
        obs_size = pt_weights[0].shape[1]
        action_size = pt_biases[-1].shape[0]
        actor_hidden_size = pt_weights[0].shape[0]
        # -1 because last layer is not counted
        actor_n_layers = len(pt_weights) - 1
        use_pt_params = True
    else:
        print('Using default params')
        obs_size = 32
        action_size = 7
        actor_hidden_size = config.actor_hidden_size
        actor_n_layers = config.actor_n_layers

    print("actor h size", actor_hidden_size)

    timeouts = np.zeros(2)
    timeouts[-1] = 1
    dummy_dataset = MDPDataset(
        observations=np.random.rand(2, obs_size),
        actions=np.random.rand(2, action_size),
        rewards=np.ones(2),
        terminals=np.zeros(2),
        timeouts=timeouts,
        action_space = ActionSpace.CONTINUOUS,
    )

    actor_hidden_units = [actor_hidden_size for _ in range(actor_n_layers)]
    actor_encoder = VectorEncoderFactory(hidden_units=actor_hidden_units, activation='relu',
                                         use_batch_norm=False, dropout_rate=config.actor_dropout)

    critic_hidden_units = [config.critic_hidden_size for _ in range(config.critic_n_layers)]
    critic_encoder = VectorEncoderFactory(hidden_units=critic_hidden_units, activation='relu', use_batch_norm=False,
                                         dropout_rate=config.critic_dropout)

    rl_args = {
        "actor_learning_rate": config.actor_learning_rate,
        "critic_learning_rate": config.critic_learning_rate,
        "actor_encoder_factory": actor_encoder,
        "critic_encoder_factory": critic_encoder,
        "batch_size": config.batch_size,
        "gamma": config.gamma,
        "tau": config.tau,
        "n_critics": config.n_critics,
        "final_activation_function": "sigmoid",
    }

    if config.type == 'awac':
        algo = AWACConfig(
            lam = config.lam,
            n_action_samples = config.n_action_samples,
            **rl_args,
        ).create(device=base_config.device)
        algo.build_with_dataset(dummy_dataset)
        if use_pt_params:
            copy_params_to_policy(algo.impl.modules.policy, pt_weights, pt_biases)

    elif config.type == 'td3bc':
        algo = TD3PlusBCConfig(
            target_smoothing_sigma = config.target_smoothing_sigma,
            target_smoothing_clip = config.target_smoothing_clip,
            alpha = config.alpha,
            update_actor_interval = config.update_actor_interval,
            **rl_args,
        ).create(device=base_config.device)
        algo.build_with_dataset(dummy_dataset)
        if use_pt_params:
            copy_params_to_policy(algo.impl.modules.policy, pt_weights, pt_biases, deterministic=True)

    elif config.type == 'td3':
        algo = TD3Config(
            target_smoothing_sigma = config.target_smoothing_sigma,
            target_smoothing_clip = config.target_smoothing_clip,
            update_actor_interval = config.update_actor_interval,
            **rl_args,
        ).create(device=base_config.device)
        algo.build_with_dataset(dummy_dataset)
        if use_pt_params:
            copy_params_to_policy(algo.impl.modules.policy, pt_weights, pt_biases, deterministic=True)


    elif config.type == 'bc':
        algo = BCConfig(encoder_factory=actor_encoder,
                        batch_size=config.bc_batch_size,
                        gamma=config.gamma,
                        learning_rate=config.bc_learning_rate,
                        policy_type='stochastic',
                        final_activation_function = "sigmoid"
                        ).create(device=base_config.device)
        algo.build_with_dataset(dummy_dataset)
        if use_pt_params:
            copy_params_to_policy(algo.impl.modules.imitator, pt_weights, pt_biases)

    else:
        raise ValueError(f"algo type {config.type} not recognized")

    return algo, floornoise


def load_mat_network_params(mat_filename: str, algo=None, config_file=None):
    """Loads params from mat file generated by EMGHero.m

    Args:
        mat_filename (str): filename of the *.mat
        algo (_type_): d3rlpy algorithm

    Returns:
        _type_: updated algo
    """
    mat_params = loadmat(mat_filename, squeeze_me=True)

    all_weights = mat_params["allWeights"]
    all_biases = mat_params["allBiases"]

    if "floornoise" in mat_params.keys():
        floornoise = mat_params["floornoise"]
    else:
        floornoise = None

    if "movements" in mat_params.keys():
        movements = mat_params["movements"]
        move_config = MoveConfig(movements=movements)
    else:
        LOGGER.warning('No movements found in *.mat, using default')
        move_config = MoveConfig()


    if algo is None:
        assert (
            config_file is not None
        ), "Please provice config when algo is not specified"
        observation_shape = all_weights[0].shape[1]
        action_shape = all_biases[-1].shape[0]
        LOGGER.info('Observation shape: %i', observation_shape)

        tmp_json = config_file.parent / "tmp.json"
        with open(config_file, "rb") as f:
            data = json.load(f)
        data["observation_shape"] = [observation_shape]
        data["action_size"] = action_shape
        with open(tmp_json, "w") as f:
            json.dump(data, f)

        algo = AWACConfig().create(device="cpu").from_json(tmp_json)

    copy_params_to_policy(algo.impl.modules.policy, all_weights, all_biases)

    return algo, floornoise, move_config


def pick_model(experiment_folder, metrics_array, pick_best=True, verbose=False, save_chosen_model=False):
    train_log_dir = get_newest_directory(experiment_folder)

    model_files = []
    for file in os.listdir(train_log_dir):
        if file.endswith(".d3"):
            model_files.append(file)

    # sort model_files
    numbers = [int(re.findall("[0-9]+", m_file)[0]) for m_file in model_files]
    sorted_model_names = sorted(zip(numbers, model_files))

    if not pick_best:
        best_model_filename = sorted_model_names[-1][1]
        best_model_path = train_log_dir / best_model_filename
        return model, best_model_path, -1

    # pick best model
    best_model_idx = np.argmax(metrics_array)
    best_model_filename = sorted_model_names[best_model_idx][1]

    # get path of best model
    best_model_path = train_log_dir / best_model_filename
    if verbose:
        print(
            f"Best model is {best_model_filename} with index {best_model_idx} and metric {np.max(metrics_array)}"
        )
    model = load_learnable(best_model_path)

    # delete other models
    for model_file in model_files:
        if not model_file == best_model_filename or not save_chosen_model:
            os.remove(train_log_dir / model_file)

    if save_chosen_model:
        model.save(train_log_dir / "chosen_model.d3")

    return model, best_model_path, best_model_idx


def randomize_wrong_notes(
    dataset: MDPDataset,
    label_transformer: LabelTransformer,
    emg_hero,
    history: dict,
    movements: list,
    randomize_chance: float = 0.1,
) -> MDPDataset:
    """Randomizes wrong actions, to incorperate exploration in offline training

    Args:
        dataset (MDPDataset): original dataset
        label_transformer (LabelTransformer): _description_
        emg_hero (EMGHero): EMG Hero class to predict new reward
        history (dict): EMG Hero history
        randomize_chance (float, optional): Chance to randomize wrong prediction. Defaults to 0.1.

    Returns:
        MDPDataset: New, changed dataset
    """
    possible_actions = [
        label_transformer.move_name_to_onehot(move_name)
        for move_name in movements
    ]

    # take all notes that don't have correct prediction
    new_dataset = copy.deepcopy(dataset)
    for episode, new_episode in zip(dataset.episodes, new_dataset.episodes):
        neg_reward_inds = np.where(episode.rewards < 0.0)

        for neg_reward_ind in neg_reward_inds[0]:
            if np.random.rand() < randomize_chance:
                # randomely pick other actions
                new_action = random.choice(possible_actions)
                # get new reward
                emg_hero.notes = history["notes"][neg_reward_ind]
                action_line_dir = label_transformer.move_onehot_to_line(new_action)
                _, new_reward = emg_hero.check_note_hit(action_line_dir)
                new_episode.actions[neg_reward_ind, :] = new_action
                new_episode.rewards[neg_reward_ind] = new_reward

    return new_dataset


def get_newest_directory(directory: str) -> Path:
    """returns newest directory in given directory

    Args:
        directory (str): directory to search in

    Returns:
        str: newest directory
    """
    all_dirs = [
        d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))
    ]
    all_dirs.sort(
        key=lambda d: os.path.getmtime(os.path.join(directory, d)), reverse=True
    )

    if all_dirs:
        return Path(os.path.join(directory, all_dirs[0]))
    else:
        return None


def load_evaluators(supervised_test, dataset, emg_hero_metrics):

    td_error_evaluator = TDErrorEvaluator(episodes=dataset.episodes)
    rl_f1_macro_evaluator = F1MacroEvaluator(dataset.episodes)
    sl_f1_macro_evaluator = F1MacroEvaluator(supervised_test.episodes)
    rl_emr_evaluator = ExactMatchEvaluator(dataset.episodes)
    sl_emr_evaluator = ExactMatchEvaluator(supervised_test.episodes)
    # reward_evaluator = RewardEvaluator()

    # evaluators["train_reward"].load_data(train_observations, train_ideal_actions, n_episodes=(episode_idx+1))

    evaluators = {
            "td_error": td_error_evaluator,
            "f1_macro": rl_f1_macro_evaluator,
            "emr": rl_emr_evaluator,
            "reward": emg_hero_metrics.simulate_song,
            # "train_reward": reward_evaluator,
            "supervised_f1": sl_f1_macro_evaluator,
            "supervised_emr": sl_emr_evaluator,
        }
    
    return evaluators


def train_emg_hero_model(
    model,
    history_filenames: list[str],
    supervised_filename: str,
    experiment_folder: Path,
    emg_hero_metrics: EMGHeroMetrics,
    move_config: MoveConfig,
    take_best_reward_model: bool = True,
    wrong_note_randomization: float = 0.1,
    use_state_with_last_action: bool = False,
    n_steps: int = 400,
    use_wandb: bool = False,
):
    """Trains a given model on the histories.

    Args:
        model (_type_): Model to train.
        history_filenames (list[str]): Filenames of the saved EMG Hero histories.
        supervised_filename (str): Filename of the supervised traning data (only for evaluation).
        experiment_folder (str): Folder where all the data of this experiment is saved.
        take_best_reward_model (bool): If model with best reward should be chosen. Defaults to True.
        wrong_note_randomization (float): If model wrong notes are randomized. Defaults to 0.1.
        use_state_with_last_action (bool): If state should be appended with last action

    Returns:
        _type_: trained model
    """
    (supervised_train, supervised_test) = load_supervised_dataset(
        supervised_filename, use_state_with_last_action=use_state_with_last_action
    )

    raw_history, terminal_inds = load_histories(history_filenames)
    (dataset, history, feature_size, action_size) = load_emg_hero_dataset(
        raw_history,
        terminal_inds,
        use_state_with_last_action=use_state_with_last_action,
    )

    assert (
        supervised_train.episodes[0].observations.shape[1] == feature_size
    ), "supervised and RL features size should match"
    assert (
        supervised_train.episodes[0].actions.shape[1] == action_size
    ), "supervised and RL action size should match"

    label_transformer = LabelTransformer(move_config=move_config)

    if wrong_note_randomization > 0.0:
        dataset = randomize_wrong_notes(
            dataset=dataset,
            label_transformer=label_transformer,
            emg_hero=emg_hero_metrics.emg_hero,
            history=history,
            movements=move_config.movements,
            randomize_chance=wrong_note_randomization,
        )

    emg_hero_metrics.load_data(
        dataset=dataset, history=history, supervised_dataset=supervised_test
    )

    # initial reward should always be the same as current game score
    initial_reward = emg_hero_metrics.simulate_song(model, None)
    print(initial_reward)

    n_samples = np.sum([e.observations.shape[0] for e in dataset.episodes])
    LOGGER.info("Dataset size: %i", n_samples)

    evaluators = load_evaluators(supervised_test, dataset, emg_hero_metrics)

    if use_wandb:
        logger_adapter = CombineAdapterFactory(
            [
                FileAdapterFactory(root_dir=experiment_folder),
                WandbAdapterFactory(),
            ]
        )
    else:
        logger_adapter = FileAdapterFactory(root_dir=experiment_folder)

    # train model
    train_hist = model.fit(
        dataset,
        n_steps=n_steps,
        n_steps_per_epoch=10,
        evaluators=evaluators,
        experiment_name=("emg_hero_training"),
        logger_adapter=logger_adapter,
    )

    train_rewards_array = [h[1]["reward"] for h in train_hist]

    model, best_model_path, best_model_idx = pick_model(experiment_folder,
                                                        train_rewards_array,
                                                        pick_best=take_best_reward_model,
                                                        save_chosen_model=True)

    LOGGER.info(
        "Best model has index %i and reward %i",
        best_model_idx,
        np.max(train_rewards_array),
    )
    LOGGER.info(best_model_path)

    return model, best_model_path


def pretrain_emg_hero_model(
    supervised_filename: str,
    model_config_file: str,
    experiment_folder: Path,
    n_steps: int = 10_000,
    use_state_with_last_action: bool = False,
):
    """Pretrains RL actor on supervised data

    Args:
        supervised_filename (str): file to supervised data file

    Returns:
        _type_: pretrained RL model
    """
    print('Potentially deprecated')
    supervised_train, supervised_test = load_supervised_dataset(
        supervised_filename, use_state_with_last_action=use_state_with_last_action
    )

    filepath = Path(__file__).parent.resolve()
    folderpath = Path(*filepath.parts[:-1])
    bc_config_file = folderpath / "config" / "bc_params.json"
    bc_model = BCConfig().create(device="cpu").from_json(bc_config_file)
    # bc_model.build_with_dataset(supervised_train)

    n_samples = np.sum([e.observations.shape[0] for e in supervised_train.episodes])
    LOGGER.info("Train dataset size: %i", n_samples)

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
    LOGGER.info(
        "Best model is %s with index %i and F1 score %f",
        best_model_filename,
        best_model_idx,
        np.max(f1s_array),
    )
    LOGGER.info(best_model_path)
    print(f"Took model {best_model_idx}: {best_model_filename}")

    bc_model = load_learnable(best_model_path)

    # plt.plot(f1s_array)
    # plt.title("Validation F1 macro")
    # plt.xlabel("epochs")
    # plt.ylabel("F1 macro")
    # plt.savefig((experiment_folder / "val_f1_macros.png"))
    # plt.show()

    awac_config_file = folderpath / "config" / "awac_params.json"
    awac_model = AWACConfig().create(device="cpu").from_json(awac_config_file)
    # awac_model.build_with_dataset(supervised_train)

    # copy BC weights to AWAC
    awac_model.impl.modules.policy._encoder = bc_model.impl.modules.imitator._encoder
    awac_model.impl.modules.policy._mu = bc_model.impl.modules.imitator._mu
    # awac_model._impl._targ_policy._encoder = bc_model._impl._imitator._encoder
    # awac_model._impl._targ_policy._mu = bc_model._impl._imitator._fc

    return awac_model


def save_emg_hero_model(
    model,
    # model_config_file: str,
    experiment_folder: str,
    supervised_data_file: str,
) -> None:
    """Saves all relevant files of model

    Args:
        model (_type_): model to save
        model_config_file (str): config file of the model
        experiment_folder (str): main folder of current experiment
        supervised_data_file (str): filename of supervised data used to pretrain
    """
    # os.popen('cp '+model_config_file+' '+experiment_folder+MODEL_CONFIG_FILENAME)
    os.popen(
        "cp "
        + supervised_data_file
        + " "
        + (experiment_folder / SUPERVISED_DATA_FILENAME).as_posix()
    )
    model.save_policy((experiment_folder / PRETRAINED_POLICY_FILENAME).as_posix())
    model.save((experiment_folder / PRETRAINED_MODEL_FILENAME).as_posix())
    LOGGER.info("Saved model files to %s", experiment_folder)


class ModelHandle:
    """Class to deal with obtaining the predicted movement from EMG"""

    def __init__(
        self,
        model,
        online_data_handler,
        feature_extractor,
        model_path: str,
        experiment_folder: str,
        play_with_emg: bool,
        tcp_host: str,
        tcp_port: int,
        n_actions: int,
        emg_hero_metrics: EMGHeroMetrics,
        label_transformer,
        take_best_reward_model: bool = True,
        observation_with_last_action: bool = False,
        floornoise: float = None,
        n_features: int = 4,
    ):
        self.model = model

        self.experiment_folder = experiment_folder
        self.play_with_emg = play_with_emg
        self.n_actions = n_actions
        self.take_best_reward_model = take_best_reward_model
        self.observation_with_last_action = observation_with_last_action
        self.emg_hero_metrics = emg_hero_metrics
        self.floornoise = floornoise
        self.n_features = n_features
        self.label_transformer = label_transformer

        self.online_data_handler = online_data_handler
        self.feature_extractor = feature_extractor

        # try connection to TCP server
        # self.tcp_client = None
        # if self.play_with_emg:
        #     self.tcp_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #     try:
        #         self.tcp_client.connect((tcp_host, tcp_port))
        #     except ConnectionRefusedError:
        #         LOGGER.error("Could not connect to TCP server, is EMGHero.m running?")
        #     data = self.tcp_client.recv(1024)
        #     LOGGER.info("TCP server response: %s", data.decode())

        self.model_filenames = [model_path]

        # initialize last action as 'Rest'
        self.last_action_prediction = np.zeros(n_actions)
        self.last_action_prediction[-1] = 1.0

    # def stop_acqusition(self) -> None:
    #     """Stop EMG acquisition"""
    #     msg = "set_false".encode()
    #     self.tcp_client.sendall(msg)

    # def start_acqusition(self) -> None:
    #     """Start EMG acquisition"""
    #     msg = "set_true".encode()
    #     self.tcp_client.sendall(msg)

    def get_emg_keys(self) -> tuple[dict, np.ndarray, np.ndarray, bool]:
        # """Requests newest features from MatLAB over TCP"""
        # msg = "request".encode()
        # self.tcp_client.sendall(msg)
        # data = self.tcp_client.recv(1024)

        # str_data = "".join([*data.decode("utf-8")])
        # split_str_data = str(str_data).split("&&")

        # if not len(split_str_data) == 2:
        #     LOGGER.warning('Split string data len not correct')

        data, count = self.online_data_handler.get_data(N=300)
        emg = data['emg']
        windows = get_windows(emg,300,75)
        features = self.feature_extractor.extract_features(feature_list = ['MAV', 'SSC', 'ZC', 'WL'],
                                        windows = windows)

        # print("data: ", data)

        # print("Features: ", features)

        mavs = features['MAV']
        wls = features['WL']
        zcs = features['ZC']
        sscs = features['SSC']

        # Stack the features in the correct order and flatten
        feat_data = np.ravel(np.column_stack((mavs, wls, zcs, sscs)))
        # print("feat_data: ", feat_data)

        mean_mav = np.mean(features['MAV'])
        # print("mean_mav: ", mean_mav)

        # feat_str_data = split_str_data[0]
        # mean_mav_str_data = split_str_data[1]

        # split_feat_str_data = feat_str_data.split("$")
        # feat_data = np.array([float(x) for x in split_feat_str_data])

        # mean_mav = float(mean_mav_str_data)

        # don't predict if values too high or below floornoise
        if self.floornoise is not None:
            below_floornoise = self.floornoise > mean_mav
        else:
            below_floornoise = False

        too_high_values = (feat_data > 10**4).any() or mean_mav > 80
        if below_floornoise or too_high_values:
            # predict rest in this case
            emg_one_hot_preds = np.zeros(self.n_actions)
            emg_one_hot_preds[-1] = 1
            LOGGER.warning("Feature values too high")
        else:
            if self.observation_with_last_action:
                model_input = np.hstack((feat_data, self.last_action_prediction))
            else:
                model_input = feat_data
            predictions = self.model.predict(np.expand_dims(model_input, axis=0))[0]
            emg_one_hot_preds = np.zeros_like(predictions)
            emg_one_hot_preds[predictions > 0.5] = 1

        emg_keys = self.label_transformer.move_onehot_to_line(emg_one_hot_preds)

        new_features = True

        return emg_keys, emg_one_hot_preds, feat_data, new_features, too_high_values
        # return -1, -1, -1, -1, -1

    def retrain_model(
        self,
        history_filenames: list[str],
        supervised_filename: str,
        experiment_folder: str,
        move_config: list,
        only_use_last_history: bool = False,
        wrong_note_randomization: float = 0.1,
        n_steps: int = 400,
    ):
        """Retrains the current model on last history.

        Args:
            history_filenames (list[str]): List of histories to train model.
            supervised_filename (str): Supervised file, only for validation.
            experiment_folder (str): Folder where all experiment data is stored.
            only_use_last_history (bool): If only last or all histories should be used.
                                            Defaults to False.
        """
        # if self.play_with_emg:
        #     self.stop_acqusition()
        # else:
        #     LOGGER.error("Cannot train model without EMG")
        #     return
    
        if only_use_last_history:
            train_history_filenames = [history_filenames[-1]]
        else:
            train_history_filenames = history_filenames

        print(train_history_filenames)
        self.model, model_path = train_emg_hero_model(
            self.model,
            train_history_filenames,
            supervised_filename,
            experiment_folder=experiment_folder,
            emg_hero_metrics=self.emg_hero_metrics,
            move_config=move_config,
            take_best_reward_model=self.take_best_reward_model,
            wrong_note_randomization=wrong_note_randomization,
            n_steps=n_steps,
        )

        self.model_filenames.append(model_path)

        # Save last rl policy, only for inference. Overwrite if already existing.
        self.model.save_policy((experiment_folder / "final_rl_policy.pt").as_posix())
        final_model_path = experiment_folder / "final_rl_model.d3"
        self.model.save(final_model_path)
        savemat(
            (experiment_folder / "final_rl_params.mat").as_posix(),
            get_policy_params_dict(algo=self.model),
        )

        # if self.play_with_emg:
        #     self.start_acqusition()


    def __del__(self):
        # end TCP connection
        # if self.play_with_emg:
        #     self.tcp_client.close()

        # save model filenames
        now = get_current_timestamp()
        model_filenames_save_path = self.experiment_folder / (
            "emg_hero_model_filenames_" + now + ".pkl"
        )

        with open(model_filenames_save_path, "wb") as _file:
            pickle.dump(self.model_filenames, _file)
            LOGGER.info(
                "Policy filenames successfully saved to %s", model_filenames_save_path
            )
