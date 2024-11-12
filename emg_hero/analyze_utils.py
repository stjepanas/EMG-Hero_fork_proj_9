import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.metrics import f1_score, accuracy_score
from sklearn.feature_selection import mutual_info_classif

from emg_hero.defs import MoveConfig
from emg_hero.label_transformer import LabelTransformer
from emg_hero.datasets import load_motion_test_data
from emg_hero.utils import actions_to_labels
from emg_hero.metrics import get_reachable_notes, \
                             get_psi_sum, get_feature_entropy, \
                             get_single_entropy, \
                             get_signal_noise_ratio, \
                             get_rewards, convert_rest


def motion_test_preds_to_onehot(mt_preds, move_config, return_flat_move_array=True):
    feat_len = len(mt_preds[0][0][0])
    moves_len = len(mt_preds[0][0])
    n_repetitions = len(mt_preds[0])

    all_moves_predictions_list = []
    for move_ind in range(moves_len):
        move_predictions_list = []
        for rep in range(n_repetitions):
            move_per_rep = []
            for feat_ind in range(feat_len):
                activated_moves = mt_preds[0][rep][move_ind][feat_ind]
                onehot_vec = np.zeros((move_config.n_actions))
                if isinstance(activated_moves, np.ndarray):
                    # check if still valid predictions or movement is done
                    if activated_moves.size == 0:
                        break

                    for act_move_ind in activated_moves:
                        onehot_vec[act_move_ind-1] = 1
                elif isinstance(activated_moves, int):
                    assert (activated_moves-1) > -1, 'index should never be negative'
                    onehot_vec[activated_moves-1] = 1
                else:
                    print('warn')
                
                if return_flat_move_array:
                    move_predictions_list.append(onehot_vec)
                else:
                    move_per_rep.append(onehot_vec)

            if not return_flat_move_array:
                move_predictions_list.append(move_per_rep)
        all_moves_predictions_list.append(np.array(move_predictions_list))
    return all_moves_predictions_list

def get_motion_test_labels(motion_test, move_config):
    individual_movements = dict(np.flip(motion_test['individual_movements'][0], 1))
    individual_movements['Rest'] = 7
    all_moves = motion_test['movements'][0]

    motion_test_labels = np.zeros((len(all_moves), move_config.n_actions))
    for i, moves_str in enumerate(all_moves):
        split_moves = moves_str.split(' + ')
        move_inds = [individual_movements[move] for move in split_moves]

        onehot_vec = np.zeros((move_config.n_actions))
        for move_ind in move_inds:
            onehot_vec[move_ind-1] = 1

        motion_test_labels[i, :] = onehot_vec

    return motion_test_labels

def get_num_action_changes(actions):
    row_diff = np.diff(actions, axis=0)
    return np.any(row_diff != 0, axis=1).sum()

def get_motion_test_metrics(mt_onehot: np.ndarray, mt_labels: np.ndarray, n_dof):

    assert len(mt_onehot.shape) == 2, 'Wrong length'
    assert len(mt_labels.shape) == 2, 'Wrong length'

    # remove rest
    mt_onehot = mt_onehot[:,:-1]
    mt_labels = mt_labels[:,:-1]
             
    metrics = {}
    # overall
    all_y_pred = np.vstack(mt_onehot)
    all_y_true = np.vstack(mt_labels)

    metrics['emr'] = accuracy_score(y_true=all_y_true, y_pred=all_y_pred)
    metrics['f1'] = f1_score(y_true=all_y_true, y_pred=all_y_pred, average='macro', zero_division=1)

    dof_keys = ['emr_1_dof', 'f1_1_dof', 'emr_2_dof', 'f1_2_dof', 'emr_3_dof', 'f1_3_dof']

    for dof_key in dof_keys:
        metrics[dof_key] = np.NaN

    metrics['emr_'+str(n_dof)+'_dof'] = accuracy_score(y_true=all_y_true, y_pred=all_y_pred)
    metrics['f1_'+str(n_dof)+'_dof'] = f1_score(y_true=all_y_true, y_pred=all_y_pred, average='macro', zero_division=1)
    # 1 DOF
    # dof_1_y_pred = np.vstack(mt_onehot[:6])
    # dof_1_y_true = np.vstack(mt_labels[:6])

    # metrics['emr_1_dof'] = accuracy_score(y_true=dof_1_y_true, y_pred=dof_1_y_pred)
    # metrics['f1_1_dof'] = f1_score(y_true=dof_1_y_true, y_pred=dof_1_y_pred, average='macro', zero_division=1)

    # 2 DOF
    # dof_2_y_pred = np.vstack(mt_onehot[6:10])
    # dof_2_y_true = np.vstack(mt_labels[6:10])

    # metrics['emr_2_dof'] = accuracy_score(y_true=dof_2_y_true, y_pred=dof_2_y_pred)
    # metrics['f1_2_dof'] = f1_score(y_true=dof_2_y_true, y_pred=dof_2_y_pred, average='macro', zero_division=1)

    # # 3 DOF
    # dof_3_y_pred = np.vstack(mt_onehot[10:12])
    # dof_3_y_true = np.vstack(mt_labels[10:12])

    # metrics['emr_3_dof'] = accuracy_score(y_true=dof_3_y_true, y_pred=dof_3_y_pred)
    # metrics['f1_3_dof'] = f1_score(y_true=dof_3_y_true, y_pred=dof_3_y_pred, average='macro', zero_division=1)

    return metrics

def get_emr_and_f1_score(y_true, y_pred):
    if y_pred.shape[0] > 0:
        _emr = accuracy_score(y_true=y_true, y_pred=y_pred)
        _f1 = f1_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=1)
    else:
        _emr = np.nan
        _f1 = np.nan

    return _emr, _f1

def get_history_metrics(all_y_pred_raw, all_y_true_raw, ideal_labels, features, move_config):
    metrics = {}

    # remove rest
    # all_y_pred = all_y_pred_raw[:,:-1]
    # all_y_true = all_y_true_raw[:,:-1]
    # convert rest
    all_y_pred = convert_rest(all_y_pred_raw)
    all_y_true = convert_rest(all_y_true_raw)

    metrics['emr'], metrics['f1'] = get_emr_and_f1_score(all_y_true, all_y_pred)
    metrics['mi'] = mutual_info_classif(features, ideal_labels).mean()

    miss_classification_array = all_y_pred - all_y_true
    miss_classification_array[np.where(miss_classification_array < 0)] = 0
    metrics['miss_classifications'] = miss_classification_array.mean()

    # without 'Rest'
    move_wo_rest = all_y_true.any(axis=1)
    move_wo_rest_y_true = all_y_true[move_wo_rest]
    move_wo_rest_y_pred = all_y_pred[move_wo_rest]

    metrics['emr_wo_rest'], metrics['f1_wo_rest'] = get_emr_and_f1_score(move_wo_rest_y_true, move_wo_rest_y_pred)

    # per DOF
    summed_y_true = all_y_true[:,:-1].sum(axis=1)
    inds_1_dof = np.argwhere(summed_y_true==1)
    inds_2_dof = np.argwhere(summed_y_true==2)
    inds_3_dof = np.argwhere(summed_y_true==3)

    # 1 DOF
    dof_1_y_pred = all_y_true[inds_1_dof[:,0], :]
    dof_1_y_true = all_y_pred[inds_1_dof[:,0], :]

    metrics['emr_1_dof'], metrics['f1_1_dof'] = get_emr_and_f1_score(dof_1_y_true, dof_1_y_pred)
    dof_1_feats = features[inds_1_dof[:,0], :]
    dof_1_labels = ideal_labels[inds_1_dof[:,0]]
    metrics['mi_1_dof'] = mutual_info_classif(dof_1_feats, dof_1_labels).mean()

    # 2 DOF
    dof_2_y_pred = all_y_true[inds_2_dof[:,0], :]
    dof_2_y_true = all_y_pred[inds_2_dof[:,0], :]

    metrics['emr_2_dof'], metrics['f1_2_dof'] = get_emr_and_f1_score(dof_2_y_true, dof_2_y_pred)

    if inds_2_dof.shape[0] > 0:
        dof_2_feats = features[inds_2_dof[:,0], :]
        dof_2_labels = ideal_labels[inds_2_dof[:,0]]
        metrics['mi_2_dof'] = mutual_info_classif(dof_2_feats, dof_2_labels).mean()
    else:
        metrics['mi_2_dof'] = np.nan

    # 3 DOF
    dof_3_y_pred = all_y_true[inds_3_dof[:,0], :]
    dof_3_y_true = all_y_pred[inds_3_dof[:,0], :]

    metrics['emr_3_dof'], metrics['f1_3_dof'] = get_emr_and_f1_score(dof_3_y_true, dof_3_y_pred)

    if inds_3_dof.shape[0] > 0:
        dof_3_feats = features[inds_3_dof[:,0], :]
        dof_3_labels = ideal_labels[inds_3_dof[:,0]]
        metrics['mi_3_dof'] = mutual_info_classif(dof_3_feats, dof_3_labels).mean()
    else:
        metrics['mi_3_dof'] = np.nan

    # per note length
    # possible_note_lengths = [0.5, 1.0, 1.5, 2.0]
    # for desired_note_length in possible_note_lengths:
    #     reachable_notes_array_note_length = get_reachable_notes(history, n_actions=move_config.n_actions, desired_note_length=desired_note_length)
    #     reachable_notes_array_note_length = reachable_notes_array_note_length[:,:6]
    #     relevant_notes_map = reachable_notes_array_note_length.any(axis=1)

    #     all_y_pred_note_length = all_y_pred[relevant_notes_map]
    #     all_y_true_note_length = reachable_notes_array_note_length[relevant_notes_map]

    #     metrics[('emr_note_length_'+''.join(str(desired_note_length).split('.')))] = accuracy_score(y_true=all_y_true_note_length, y_pred=all_y_pred_note_length)
    #     metrics[('f1_note_length_'+''.join(str(desired_note_length).split('.')))] = f1_score(y_true=all_y_true_note_length, y_pred=all_y_pred_note_length, average='macro', zero_division=1)

    # number of times action changes
    metrics['num_action_changes'] = get_num_action_changes(all_y_pred_raw)

    return metrics


def create_csv(history_summary, base_folder='./', save_path=None, save_file=False):
    motion_test_files = history_summary['motion_test_files']
    HISTORY_FILENAMES = history_summary['HISTORY_FILENAMES']
    # experiment_folder = './logs/' + history_summary['experiment_folder'] FIXME
    experiment_folder = history_summary['experiment_folder']
    subject_id = history_summary['subject_id']
    subject_date = history_summary['subject_date']
    subject_age = history_summary['subject_age']

    mat_net_file = experiment_folder / 'pretrained_network_params.mat'
    if mat_net_file.exists():
        mat_params = loadmat(mat_net_file.as_posix(), squeeze_me=True)
        if "movements" in mat_params.keys():
            movements = mat_params["movements"]
            move_config = MoveConfig(movements=movements)
        else:
            print('Using standard moves')
            move_config = MoveConfig()
    else:
        print('Using standard moves')
        move_config = MoveConfig()

    motion_test_exists = len(motion_test_files) > 1

    if motion_test_exists:
        motion_test_data = load_motion_test_data(motion_test_files)
        motion_test_metrics = {}

        for key, motion_test in motion_test_data.items():
            mt_onehot = motion_test_preds_to_onehot(motion_test['predictions'], move_config, return_flat_move_array=False)
            mt_single_labels = get_motion_test_labels(motion_test, move_config)

            person_mt_metrics = []
            for mt_label, mt_move_preds in zip(mt_single_labels, mt_onehot):
                for single_rep_mt_preds in mt_move_preds:
                    single_rep_mt_preds = np.array(single_rep_mt_preds)
                    y_true_move = np.repeat(np.expand_dims(mt_label, 0), single_rep_mt_preds.shape[0], axis=0)

                    n_dof = int(mt_label.sum())

                    rep_mt_metrics = get_motion_test_metrics(y_true_move[:,:6], single_rep_mt_preds[:,:6], n_dof=n_dof)
                    person_mt_metrics.append(rep_mt_metrics)

            mean_dict = {}
            for m_key in person_mt_metrics[0].keys():
                key_data = [person_m[m_key] for person_m in person_mt_metrics]
                mean_dict[m_key] = np.nanmean(key_data)

            motion_test_metrics[key] = mean_dict
    # for key, motion_test in motion_test_data.items():
    #     mt_onehot = motion_test_preds_to_onehot(motion_test['predictions'])
    #     mt_single_labels = get_motion_test_labels(motion_test)

    #     mt_labels = []
    #     for mt_label, mt_move_preds in zip(mt_single_labels, mt_onehot):
    #         y_true_move = np.repeat(np.expand_dims(mt_label, 0), mt_move_preds.shape[0], axis=0)
    #         mt_labels.append(y_true_move)

    #     mt_onehot = [mt_o[:,:6] for mt_o in mt_onehot]
    #     mt_labels = [mt_l[:,:6] for mt_l in mt_labels]

    #     motion_test_metrics[key] = get_motion_test_metrics(mt_onehot, mt_labels)

    # load supervised dataset
    supervised_filename = experiment_folder / 'supervised_data.mat'
    mat_data = loadmat(supervised_filename.as_posix(), squeeze_me = True)

    pretrain_train_observations = np.swapaxes(mat_data['person']['featsTrain'].take(0),
                                                0, 1).astype(np.float64)

    # pretrain_raw_data = mat_data['person']['rawData'].take(0).astype(np.float64)
    # pretrain_raw_labels = mat_data['person']['rawLabels'].take(0).astype(np.float64)

    signal_noise_ratio = np.nan #get_signal_noise_ratio(pretrain_raw_data, pretrain_raw_labels)

    last_feature_dist = pretrain_train_observations

    first_feature_dist = None

    label_transformer = LabelTransformer(move_config)

    # history results
    results_list = []
    for i, hist_filename in enumerate(HISTORY_FILENAMES):
        with open(hist_filename, 'rb') as file:
            file_history = pickle.load(file)
            file_history['actions'] = np.array(file_history['actions'])
            file_history['time'] = np.array(file_history['time'])

        hist_rewards = np.array(file_history['rewards'])
        reward_sum = hist_rewards.sum()
        positive_reward_sum = hist_rewards[hist_rewards>0.].sum()

        actions = np.array(file_history['actions'])
        ideal_actions = get_reachable_notes(file_history, label_transformer, switch_lines=history_summary['switch_lines'])
        features = np.array(file_history['features'])
        ideal_labels = actions_to_labels(ideal_actions)
        rl_metrics = get_history_metrics(actions, ideal_actions, ideal_labels, features, move_config)

        max_reward = get_rewards(ideal_actions, ideal_actions).sum()
        wrong_actions = np.ones_like(ideal_actions)
        min_reward = get_rewards(wrong_actions, ideal_actions).sum()
        sim_reward = get_rewards(actions, ideal_actions).sum()

        # create csv row for each history
        results_row = {}

        results_row['id'] = subject_id
        results_row['experiment_folder'] = experiment_folder
        results_row['date'] = subject_date
        results_row['age'] = subject_age
        results_row['filename'] = hist_filename
        results_row['max_reward'] = max_reward
        results_row['min_reward'] = min_reward
        results_row['sim_reward'] = sim_reward

        results_row['entropy_supervised'] = get_feature_entropy(pretrain_train_observations, file_history['features'])
        results_row['entropy_last_episode'] = get_feature_entropy(last_feature_dist, file_history['features'])
        results_row['entropy_last_episode_2'] = get_feature_entropy(file_history['features'], last_feature_dist)
        if first_feature_dist is None:
            first_feature_dist = file_history['features']
            results_row['entropy_first_episode'] = get_feature_entropy(first_feature_dist, first_feature_dist)
        else:
            results_row['entropy_first_episode'] = get_feature_entropy(first_feature_dist, file_history['features'])

        results_row['entropy'] = get_single_entropy(file_history['features'])
        results_row['signal_noise_ratio'] = signal_noise_ratio

        if i > 0:
            transition_psi = get_psi_sum(np.array(last_feature_dist), np.array(file_history['features']))
            transition_psi_actions = get_psi_sum(last_actions, file_history['actions'])
        else:
            transition_psi = np.NAN
            transition_psi_actions = np.NAN

        results_row['psi'] = transition_psi
        results_row['psi_actions'] = transition_psi_actions

        last_feature_dist = file_history['features']
        last_actions = file_history['actions']

        if i < 8:
            results_row['type'] = 'rl_' + str(i)
            motion_test_key = None
        elif i == 8:
            results_row['type'] = 'rl_final'
            motion_test_key = 'rl'
        elif i == 9:
            results_row['type'] = 'pretrained'
            motion_test_key = 'pretrained'
        elif i == 10:
            results_row['type'] = 'more_data'
            motion_test_key = 'more_data'

        if motion_test_key is not None and motion_test_exists:
            results_row['motion_test_emr'] = motion_test_metrics[motion_test_key]['emr']
            results_row['motion_test_emr_1_dof'] = motion_test_metrics[motion_test_key]['emr_1_dof']
            results_row['motion_test_emr_2_dof'] = motion_test_metrics[motion_test_key]['emr_2_dof']
            results_row['motion_test_emr_3_dof'] = motion_test_metrics[motion_test_key]['emr_3_dof']
            results_row['motion_test_f1'] = motion_test_metrics[motion_test_key]['f1']
            results_row['motion_test_f1_1_dof'] = motion_test_metrics[motion_test_key]['f1_1_dof']
            results_row['motion_test_f1_2_dof'] = motion_test_metrics[motion_test_key]['f1_2_dof']
            results_row['motion_test_f1_3_dof'] = motion_test_metrics[motion_test_key]['f1_3_dof']
            results_row['motion_test_filename'] = motion_test_files[motion_test_key]
        else:
            results_row['motion_test_emr'] = None
            results_row['motion_test_emr_1_dof'] = None
            results_row['motion_test_emr_2_dof'] = None
            results_row['motion_test_emr_3_dof'] = None
            results_row['motion_test_f1'] = None
            results_row['motion_test_f1_1_dof'] = None
            results_row['motion_test_f1_2_dof'] = None
            results_row['motion_test_f1_3_dof'] = None
            results_row['motion_test_filename'] = None

        results_row['rewards'] = reward_sum
        results_row['positive_rewards'] = positive_reward_sum

        for key, value in rl_metrics.items():
            results_row[key] = value
        results_list.append(results_row)

    df = pd.DataFrame(data=[list(r.values()) for r in results_list],
                  columns=list(results_list[0].keys()))
    


    if save_file:
        if save_path is None:
            raise ValueError('save_path is None')
        if not isinstance(save_path, Path):
            save_path = Path(save_path)
    
        save_folder = save_path / 'experiment_results_patients'
        save_folder.mkdir(exist_ok=True)

        df.to_csv((save_folder / f'subject_{subject_id}.csv'))
    return df
