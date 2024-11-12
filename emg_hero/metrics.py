'''Defines custom metrics for evaluating a EMG Hero model
'''
import copy
import logging
from typing import Optional, Sequence

import wandb
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from d3rlpy.dataset import MDPDataset
from scipy.stats import entropy

from d3rlpy.dataset import (
    EpisodeBase,
    ReplayBuffer,
)
from d3rlpy.interface import QLearningAlgoProtocol
from d3rlpy.metrics.evaluators import EvaluatorProtocol

from emg_hero.defs import Note
from emg_hero.label_transformer import LabelTransformer

#---------------------------------------------------#
#                   metrics                         #
#---------------------------------------------------#

# ---------------------
#       online


class RewardCallback:
    def __init__(self, reward_evaluator) -> None:
        self.reward_evaluator = reward_evaluator
        self.last_epoch = -1

    def __call__(self, algo, epoch, total_step):
        if not epoch == self.last_epoch: 
            reward = self.reward_evaluator(algo, None)
            wandb.run.log({"ideal_data_reward": reward})
            self.last_epoch = epoch


def discrete_f1_macro(model, test, threshold: float = 0.5) -> float:
    '''Calculates discrete F1 macro for model that outputs floats

    Args:
        model (_type_): Model to test
        test (MDPDataset): Test dataset
        threshold (float, optional): Threshold for correct prediction. Defaults to 0.5.

    Returns:
        float: F1 score
    '''
    predictions = None
    actions = None
    for sample in test:
        prediction = model.predict(sample.observations)
        if predictions is None:
            predictions = prediction
        else:
            predictions = np.vstack((predictions, prediction))

        if actions is None:
            actions = np.ceil(sample.actions)
        else:
            actions = np.vstack((actions, sample.actions))

    int_preds = np.zeros_like(predictions)
    int_preds[predictions > threshold] = 1

    return f1_score(actions, int_preds, average='macro')


def convert_rest(actions):
    action_size = actions.shape[1]
    rest1 = np.zeros(action_size).astype(float) #np.array([0.,0.,0.,0.,0.,0.,1.,])
    rest1[-1] = 1.
    rest2 = np.zeros(action_size).astype(float)# np.array([0.,0.,0.,0.,0.,0.,0.,])
    actions[(actions==rest1).all(axis=1)] = rest2
    return actions


def get_rewards(actions, ideal_actions):
    _actions = convert_rest(actions.copy())
    _ideal_actions = convert_rest(ideal_actions.copy())

    # calculate move rewards
    calculated_rewards = np.zeros((_actions.shape[0], 1)) - 1
    correct_idxs = np.where((_actions == _ideal_actions).all(axis=1))[0]
    calculated_rewards[correct_idxs] = 1.

    # calculate rest rewards
    action_size = actions.shape[1]
    rest2 = np.zeros(action_size).astype(float)
    correct_rest = np.where(np.logical_and((_ideal_actions == rest2).all(axis=1), (calculated_rewards == 1.).squeeze()))
    calculated_rewards[correct_rest] = 0.
    return calculated_rewards


def get_predictions_and_actions(algo, episodes, threshold: float = 0.5):
    predictions = None
    actions = None
    for episode in episodes:
        prediction = algo.predict(episode.observations)
        if predictions is None:
            predictions = prediction
        else:
            predictions = np.vstack((predictions, prediction))

        if actions is None:
            actions = np.ceil(episode.actions)
        else:
            actions = np.vstack((actions, np.ceil(episode.actions)))

    int_preds = np.zeros_like(predictions)
    int_preds[predictions > threshold] = 1

    int_preds = convert_rest(int_preds)
    actions = convert_rest(actions)

    return actions, int_preds


class RewardEvaluator(EvaluatorProtocol):
    _episodes: Optional[Sequence[EpisodeBase]]

    def __init__(self, ideal_actions: Optional[np.ndarray] = None, observations: Optional[np.ndarray] = None, threshold: float = 0.5):
        self._ideal_actions = ideal_actions
        self._observations = observations
        self._threshold = threshold
        self._n_episodes = 1

    def load_data(self, observations, ideal_actions, n_episodes=None):
        assert len(observations) == len(ideal_actions), 'observation and ideal action missmatch'

        self._ideal_actions = ideal_actions
        self._observations = observations
        if n_episodes is not None:
            self._n_episodes = n_episodes
        else:
            self._n_episodes = 1

    def __call__(
        self,
        algo: QLearningAlgoProtocol,
        dataset: ReplayBuffer,
    ) -> float:
        assert self._ideal_actions is not None, 'ideal actions not set'
        assert self._observations is not None, 'observations not set'

        float_predictions = algo.predict(self._observations)
        predictions = (float_predictions > self._threshold).astype(int)

        rewards = get_rewards(predictions, self._ideal_actions).sum()
        rewards /= self._n_episodes
        return rewards


class F1MacroEvaluator(EvaluatorProtocol):
    _episodes: Optional[Sequence[EpisodeBase]]

    def __init__(self, episodes: Optional[Sequence[EpisodeBase]] = None, threshold: float = 0.5):
        self._episodes = episodes
        self._threshold = threshold

    def __call__(
        self,
        algo: QLearningAlgoProtocol,
        dataset: ReplayBuffer,
    ) -> float:
        episodes = self._episodes if self._episodes else dataset.episodes
        actions, int_preds = get_predictions_and_actions(algo, episodes, self._threshold)
        return f1_score(actions, int_preds, average='macro', zero_division=1.)


class ExactMatchEvaluator(EvaluatorProtocol):
    _episodes: Optional[Sequence[EpisodeBase]]

    def __init__(self, episodes: Optional[Sequence[EpisodeBase]] = None, threshold: float = 0.5):
        self._episodes = episodes
        self._threshold = threshold

    def __call__(
        self,
        algo: QLearningAlgoProtocol,
        dataset: ReplayBuffer,
    ) -> float:
        episodes = self._episodes if self._episodes else dataset.episodes
        actions, int_preds = get_predictions_and_actions(algo, episodes, self._threshold)
        return accuracy_score(actions, int_preds)


class ExactMatchEvaluatorPerLabel(EvaluatorProtocol):
    _episodes: Optional[Sequence[EpisodeBase]]

    def __init__(self, episodes: Optional[Sequence[EpisodeBase]] = None, threshold: float = 0.5):
        self._episodes = episodes
        self._threshold = threshold

    def __call__(
        self,
        algo: QLearningAlgoProtocol,
        dataset: ReplayBuffer,
    ) -> float:
        episodes = self._episodes if self._episodes else dataset.episodes
        actions, int_preds = get_predictions_and_actions(algo, episodes, self._threshold)

        per_label_emrs = {}
        unique_actions = np.unique(actions, axis=0)
        for u_action in unique_actions:
            # filter results
            u_action_idxs = np.where((actions == u_action).all(axis=1))

            u_int_preds = int_preds[u_action_idxs]
            u_action_samples = actions[u_action_idxs]
            per_label_emrs[str(u_action)] = accuracy_score(u_action_samples, u_int_preds)

        return per_label_emrs


class EMGHeroMetrics:
    '''Calculates custom EMG Hero metrics to evaluate model performance
    '''
    def __init__(self,
                 emg_hero,
                 label_transformer,
                 song_dataset,
                 history,
                 supervised_dataset,
                 action_size):

        self.song_dataset = song_dataset
        self.history = history
        self.supervised_dataset = supervised_dataset
        self.action_size = action_size

        self.hit_note_score = 10
        self.hit_long_score = 1
        self.note_not_hit_score = -1

        self.emg_hero = emg_hero
        self.label_transformer = label_transformer

    # def get_reachable_notes(self, desired_note_length=None):
    #     '''Finds reachable notes in a song for a given history

    #     Returns:
    #         np.ndarray: notes that are reachable for given timestep,
    #         list: length of each note
    #     '''
    #     reachable_notes_array = []
    #     note_lengths = []
    #     n_notes = 0
    #     for note_array in self.history['notes']:
    #         correct_prediction = np.zeros(self.action_size)
    #         note_length = None
    #         for note in note_array:
    #             if desired_note_length is None or note.length == desired_note_length:
    #                 length = int(note.length * self.history['speed'])
    #                 center_y_end = note.position - length
    #                 center_y_start = note.position

    #                 if (center_y_start + self.history['y_range']) > self.history['key_y'] and \
    #                 (center_y_end - self.history['y_range']) < self.history['key_y']:
    #                     for line, direction in zip(note.lines, note.directions):
    #                         direction_idx = 0 if direction == 'down' else 1
    #                         pred_idx = self.action_size - (line * 2) - direction_idx - 2
    #                         n_notes += 1
    #                         correct_prediction[pred_idx] = 1

    #                     note_length = note.length

    #         reachable_notes_array.append(copy.deepcopy(correct_prediction))
    #         note_lengths.append(note_length)

    #     return np.array(reachable_notes_array), note_lengths

    def simulate_song(self, sim_model, _) -> float:
        '''Simulates playing a EMG Hero song with a given model

        Args:
            sim_model (_type_): model to play the song with
            _ (None): dummy input for training test set

        Returns:
            float: cummulative reward of the game
        '''
        print('WARN: likely wrong reward prediction, needs to be implemented properly')
        if self.song_dataset is None:
            logging.error('Please call load_data first')

        rewards_list = []
        for episode in self.song_dataset.episodes:
            rewards = []
            float_predictions = sim_model.predict(episode.observations)
            predictions = np.zeros_like(float_predictions)
            predictions[float_predictions > 0.5] = 1.
            for _note, _pred in zip(self.history['notes'], predictions):
                self.emg_hero.notes = _note
                action_line_dir = self.label_transformer.move_onehot_to_line(_pred)
                _, reward = self.emg_hero.check_note_hit(action_line_dir)
                rewards.append(reward)

            rewards_list.append(np.array(rewards).sum())
        return np.mean(rewards_list)

    def supervised_f1(self, model, _) -> float:
        '''Calculates F1 macro score for same dataset.

        This is used to get the supervised F1 score for a
        different (second) test set than used during training.
        '''
        return discrete_f1_macro(model, self.supervised_dataset)

    def load_data(self,
                    dataset: MDPDataset,
                    history: dict,
                    supervised_dataset: MDPDataset):
        '''Loads new data to calculate metrics

        Args:
            song (list): song to test on
            dataset (MDPDataset): RL dataset
            history (dict): history of RL dataset
            supervised_dataset (MDPDataset): supervised dataset
        '''
        self.emg_hero.song = history['song']
        self.song_dataset = dataset
        self.history = history
        self.supervised_dataset = supervised_dataset

# ---------------------
#   offline

def get_close_actions(history: dict,
                      note_time: float,
                      note_range: float = 0.1,
                      upper_lower_range: float = 0.5
                      ) -> tuple[np.ndarray, np.ndarray]:
    '''Get actions that are in the range of a given note time

    Args:
        history (dict): History of an EMG Hero song
        note_time (float): Desired timestamp
        note_range (float, optional): Range for correct note. Defaults to 0.1.
        upper_lower_range (float, optional): Range for close actions. Defaults to 0.5.

    Returns:
        tuple: Actions indices that belong to note and indices that are close to note
    '''
    start_time = note_time - upper_lower_range
    end_time = note_time + upper_lower_range
    close_action_inds = np.argwhere(
            (history['time'] > start_time) &
            (history['time'] < end_time))

    note_action_inds = np.argwhere(
        (history['time'] > (note_time - note_range)) &
        (history['time'] < (note_time + note_range)))

    # delete note actions from close actions
    close_action_inds = close_action_inds[~np.isin(close_action_inds, note_action_inds)]

    return note_action_inds, close_action_inds

def get_note_result(history: dict, note: Note, label_transformer: LabelTransformer) -> dict:
    '''Get metrics for a given notes

    Args:
        history (dict): History to analyze note
        note (Note): Note for which metrics should be calculated
        label_transformer (LabelTransformer): Label Transformer

    Returns:
        dict: Metric results
    '''
    # get range of predictions associated with note
    correct_window_s = history['y_range'] / history['speed'] # [s]
    note_inds, close_inds = get_close_actions(history,
                                              note.time,
                                              note_range = correct_window_s,
                                              upper_lower_range=0.5)

    note_actions =  history['actions'][note_inds].squeeze()
    close_actions = history['actions'][close_inds].squeeze()

    # calculate metrics
    onehot_idx = label_transformer.note_to_onehot(note)
    wanted_dofs = np.argwhere(onehot_idx==1)[0]

    # how many segments of note are hit?
    success_segments = note_actions[:,wanted_dofs]
    accuracy = success_segments.mean()

    # overshoot [%], smaller is better
    overshoot_segments = close_actions[:,wanted_dofs]
    overshoot = overshoot_segments.mean()

    # selection time, time to first correct detection
    first_success_seg = np.argwhere(success_segments==1)
    selection_time = None
    if len(first_success_seg) > 0:
        first_correct_time = history['time'][note_inds[first_success_seg][0,0]][0]
        dt_first_note_time = first_correct_time - note.time
        selection_time = 0 if dt_first_note_time < 0 else dt_first_note_time

    # how long it takes to move at all (from rest)
    actions_wo_rest = note_actions[:,:-1]
    first_prediction_seg = np.argwhere(np.any(actions_wo_rest == 1, axis = 1))
    first_movement_time = np.NaN
    if len(first_prediction_seg) > 0:
        first_prediction_time = history['time'][note_inds[first_prediction_seg][0,0]][0]
        dt_first_prediction_time = first_prediction_time - note.time
        first_movement_time = 0 if dt_first_prediction_time < 0 else dt_first_prediction_time

    result = {
        'onehot_note': onehot_idx,
        'predictions': note_actions,
        'accuracy': accuracy,
        'overshoot' : overshoot,
        'selection_time': selection_time,
        'first_movement_time': first_movement_time,
        'hit': success_segments.any(),
    }
    return result

# false positive
# filter notes for reachable ones
def get_reachable_notes(history: dict, label_transformer, switch_lines: bool = False) -> np.ndarray:
    '''Calculates reachable notes for each timestamp in history

    Args:
        history (dict): History to analyze
        n_actions (int): Number of actions

    Returns:
        np.ndarray: reachable notes
    '''
    reachable_notes_array = []
    for note_array in history['notes']:
        correct_prediction = np.zeros(label_transformer.move_config.n_actions)
        for note in note_array:

            length = int(note.length * history['speed'])
            center_y_end = note.position - length
            center_y_start = note.position

            if (center_y_start + history['y_range']) > history['key_y'] \
                and (center_y_end - history['y_range']) < history['key_y']:
                if switch_lines:
                    lines = (label_transformer.move_config.n_dof - 1) - np.array(note.lines)
                else:
                    lines = note.lines
                key = {'lines': lines, 'directions': note.directions}
                correct_prediction = label_transformer.keys_to_onehot(key)

        reachable_notes_array.append(copy.deepcopy(correct_prediction))

    return np.array(reachable_notes_array)

def get_false_positives(history: dict, n_actions: int) -> np.ndarray:
    '''Gets mean count of false positives.
    Here, each movement/entry in a onehot array is considered independently,
    so there can be more than one false positive in a prediction.
    The mean is taken over all predicted movements, so:

    mean false positives = n false positives / (n samples * n actions)

    Args:
        history (dict): History to analyze
        n_actions (int): Number of actions

    Returns:
        np.ndarray: Mean false positives
    '''
    reachable_notes_array = get_reachable_notes(history, n_actions=n_actions)
    predicted_actions = np.array(history['actions'])

    # find every 1 in predicted actions that is not in reachable_notes_array, these are errors
    false_positives_array = predicted_actions - reachable_notes_array
    false_positives_array[np.where(false_positives_array < 0)] = 0
    false_positives = false_positives_array.mean()
    return false_positives

def get_emr_and_f1(history_results: list) -> tuple[list, list, list, list]:
    '''Calculates exact match ratio and F1 score

    Args:
        history_results (list): history results obtained by get_note_result

    Returns:
        tuple: emrs, f1s, labels and predictions for histories
    '''
    histories_emrs = []
    histories_f1s = []
    histories_labels = []
    histories_preds = []

    for results in history_results:
        hist_emr = []
        hist_labels = []
        hist_preds = []
        for result in results:
            res_emr = []
            predictions = result['predictions']
            label = result['onehot_note']
            # calcualte EMR for each
            for pred in predictions:
                hist_labels.append(label)
                hist_preds.append(pred)
                matches = 1 if np.array_equal(pred, label) else 0

                res_emr.append(matches)
            hist_emr.append(np.mean(res_emr))
        histories_emrs.append(hist_emr)
        histories_labels.append(np.array(hist_labels))
        histories_preds.append(np.array(hist_preds))

    return histories_emrs, histories_f1s, histories_labels, histories_preds

# from https://www.kaggle.com/code/podsyp/population-stability-index
def get_feature_psi(expected: np.ndarray, actual: np.ndarray, bucket_type: str = "bins", n_bins: int = 10) -> float:
    """Calculate PSI metric for two arrays.
    
    Parameters
    ----------
        expected : list-like
            Array of expected values
        actual : list-like
            Array of actual values
        bucket_type : str
            Binning strategy. Accepts two options: 'bins' and 'quantiles'. Defaults to 'bins'.
            'bins': input arrays are splitted into bins with equal
                and fixed steps based on 'expected' array
            'quantiles': input arrays are binned according to 'expected' array
                with given number of n_bins
        n_bins : int
            Number of buckets for binning. Defaults to 10.

    Returns
    -------
        A single float number
    """
    breakpoints = np.arange(0, n_bins + 1) / (n_bins) * 100
    if bucket_type == "bins":
        breakpoints = np.histogram(expected, n_bins)[1]
    elif bucket_type == "quantiles":
        breakpoints = np.percentile(expected, breakpoints)

    # Calculate frequencies
    expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)
    # Clip freaquencies to avoid zero division
    expected_percents = np.clip(expected_percents, a_min=0.0001, a_max=None)
    actual_percents = np.clip(actual_percents, a_min=0.0001, a_max=None)
    # Calculate PSI
    psi_value = (expected_percents - actual_percents) * np.log(expected_percents / actual_percents)

    psi_value = sum(psi_value)

    return psi_value

def get_psi_sum(last_features, current_features):
    n_feats = last_features.shape[1]

    person_psis = np.zeros(n_feats)
    for i in range(n_feats):
        person_psis[i] = get_feature_psi(last_features[:,i], current_features[:,i], n_bins=10)

    return person_psis.sum()

def get_feature_entropy(prob_dist1, prob_dist2):

    prob_dist1 = np.array(prob_dist1)
    prob_dist2 = np.array(prob_dist2)

    last_idx = prob_dist1.shape[0] if prob_dist1.shape[0] < prob_dist2.shape[0] else prob_dist2.shape[0]

    prob_dist1 = prob_dist1[:last_idx, :]
    prob_dist2 = prob_dist2[:last_idx, :]
    
    prob_dist1 /= prob_dist1.sum(axis=1, keepdims=True)
    prob_dist2 /= prob_dist2.sum(axis=1, keepdims=True)

    epsilon = 1e-10
    prob_dist1 = np.maximum(prob_dist1, epsilon)
    prob_dist2 = np.maximum(prob_dist2, epsilon)

    # Calculate KL divergence
    return entropy(prob_dist1.T, prob_dist2.T, axis=1).mean()

def get_single_entropy(prob_dist):
    prob_dist = np.array(prob_dist)
    prob_dist /= prob_dist.sum(axis=1, keepdims=True)

    epsilon = 1e-10
    prob_dist = np.maximum(prob_dist, epsilon)

    # Calculate KL divergence
    return entropy(prob_dist).mean()

def mean_squared(x_vals):
    return np.mean(np.square(x_vals), axis=(0))

def get_signal_noise_ratio(raw_data, raw_labels):
    rest_data = raw_data[raw_labels == np.unique(raw_labels)[-1]].reshape(-1, 8)
    rest_rms = mean_squared(rest_data)

    person_snrs = []
    for label_idx in np.unique(raw_labels)[:-1]:
        move_data = raw_data[raw_labels == label_idx].reshape(-1, 8)
        move_rms = mean_squared(move_data)
        move_signal_noise_ratios = 10 * np.log10(move_rms / rest_rms)
        max_move_snr = move_signal_noise_ratios.max()
        person_snrs.append(max_move_snr)

    return np.mean(person_snrs)
