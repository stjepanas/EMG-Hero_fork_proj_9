import pickle
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from emg_hero.datasets import load_histories
from emg_hero.label_transformer import LabelTransformer
from emg_hero.metrics import get_reachable_notes
from emg_hero.defs import MoveConfig
from emg_hero.model_utility import load_torch_policy_params

# TODO integrate pretrained weights and biases


def create_experiment_dataset(experiment_name, history_filenames, switch_lines):
    experiment_folder = Path(".") / "logs" / experiment_name
    mat_filename = experiment_folder / 'pretrained_network_params.mat'
    if mat_filename.exists():
        mat_params = loadmat(mat_filename, squeeze_me=True)
        all_weights = mat_params["allWeights"]
        all_biases = mat_params["allBiases"]
        movements = mat_params["movements"]
        move_config = MoveConfig(movements=movements)
    else:
        pt_torch_filename = experiment_folder / 'pretrained_policy_v2.pt'
        all_weights, all_biases = load_torch_policy_params(pt_torch_filename)
        move_config = MoveConfig()
    label_transformer = LabelTransformer(move_config)
    episodes_list = []
    for history_filename in history_filenames:
        history, _ = load_histories(history_filenames=[history_filename])
        ideal_actions = get_reachable_notes(history, label_transformer, switch_lines=switch_lines)
        episodes_list.append({
            'observations': np.array(history['features']),
            'recording_rewards': np.array(history['rewards']),
            'ideal_actions': ideal_actions,
            'recording_actions': np.array(history['actions']),
        })
    out_file = experiment_folder / 'experiment_dataset.pkl'
    with open(out_file, 'wb') as f:
        pickle.dump(episodes_list, f)

    dataset_dir = Path(".") / "datasets" / "online" 
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # count number of files in dataset_dir
    count = len(list(dataset_dir.glob("*.pkl")))
    dataset_file = dataset_dir / f"person_{count}.pkl"

    exp_data = {
        "episodes": episodes_list,
        "weights": all_weights,
        "biases": all_biases,
        "movements": move_config.movements,
    }

    with open(dataset_file, 'wb') as f:
        pickle.dump(exp_data, f)


def main():
    experiment_name = "p1"
    history_filenames = [
                    './logs/p1/emg_hero_history_2024_1_11_9_48_34.pkl',
                    './logs/p1/emg_hero_history_2024_1_11_9_51_10.pkl',
                    './logs/p1/emg_hero_history_2024_1_11_9_53_27.pkl',
                    './logs/p1/emg_hero_history_2024_1_11_9_55_53.pkl',
                    './logs/p1/emg_hero_history_2024_1_11_9_58_39.pkl',
                    './logs/p1/emg_hero_history_2024_1_11_10_1_13.pkl',
                    './logs/p1/emg_hero_history_2024_1_11_10_4_4.pkl',
                    './logs/p1/emg_hero_history_2024_1_11_10_6_37.pkl',
                    './logs/p1/emg_hero_history_2024_1_11_10_9_18.pkl',]
    switch_lines = True
    create_experiment_dataset(experiment_name, history_filenames, switch_lines)

    experiment_name = "p2"
    history_filenames = [
                    './logs/p2/emg_hero_history_2024_1_30_15_11_38.pkl',
                    './logs/p2/emg_hero_history_2024_1_30_15_14_7.pkl',
                    './logs/p2/emg_hero_history_2024_1_30_15_16_36.pkl',
                    './logs/p2/emg_hero_history_2024_1_30_15_19_10.pkl',
                    './logs/p2/emg_hero_history_2024_1_30_15_21_49.pkl',
                    './logs/p2/emg_hero_history_2024_1_30_15_24_32.pkl',
                    './logs/p2/emg_hero_history_2024_1_30_15_27_26.pkl',
                    './logs/p2/emg_hero_history_2024_1_30_15_30_18.pkl',
                    './logs/p2/emg_hero_history_2024_1_30_15_33_14.pkl',]
    switch_lines = False
    create_experiment_dataset(experiment_name, history_filenames, switch_lines)
    
    experiment_name = "3_2023_4_27_16_38_5"
    history_filenames = [
        './logs/3_2023_4_27_16_38_5/emg_hero_history_2023_4_27_16_47_53.pkl',
        './logs/3_2023_4_27_16_38_5/emg_hero_history_2023_4_27_16_53_30.pkl',
        './logs/3_2023_4_27_16_38_5/emg_hero_history_2023_4_27_16_57_11.pkl',
        './logs/3_2023_4_27_16_38_5/emg_hero_history_2023_4_27_17_0_55.pkl',
        './logs/3_2023_4_27_16_38_5/emg_hero_history_2023_4_27_17_4_50.pkl',
        './logs/3_2023_4_27_16_38_5/emg_hero_history_2023_4_27_17_9_20.pkl',
        './logs/3_2023_4_27_16_38_5/emg_hero_history_2023_4_27_17_13_45.pkl',
        './logs/3_2023_4_27_16_38_5/emg_hero_history_2023_4_27_17_18_27.pkl',
        './logs/3_2023_4_27_16_38_5/emg_hero_history_2023_4_27_17_23_40.pkl',
        ]
    switch_lines = False
    create_experiment_dataset(experiment_name, history_filenames, switch_lines)

    experiment_name = "2_2023_4_27_14_25_2"
    history_filenames = [
        './logs/2_2023_4_27_14_25_2/emg_hero_history_2023_4_27_14_37_49.pkl',
        './logs/2_2023_4_27_14_25_2/emg_hero_history_2023_4_27_14_41_8.pkl',
        './logs/2_2023_4_27_14_25_2/emg_hero_history_2023_4_27_14_44_39.pkl',
        './logs/2_2023_4_27_14_25_2/emg_hero_history_2023_4_27_14_48_33.pkl',
        './logs/2_2023_4_27_14_25_2/emg_hero_history_2023_4_27_14_53_47.pkl',
        './logs/2_2023_4_27_14_25_2/emg_hero_history_2023_4_27_14_58_56.pkl',
        './logs/2_2023_4_27_14_25_2/emg_hero_history_2023_4_27_15_6_18.pkl',
        './logs/2_2023_4_27_14_25_2/emg_hero_history_2023_4_27_15_10_55.pkl',
        './logs/2_2023_4_27_14_25_2/emg_hero_history_2023_4_27_15_16_53.pkl',
        ]

    switch_lines = False
    create_experiment_dataset(experiment_name, history_filenames, switch_lines)

    experiment_name = "4_2023_4_28_10_30_41"
    history_filenames = [
        './logs/4_2023_4_28_10_30_41/emg_hero_history_2023_4_28_10_40_46.pkl',
        './logs/4_2023_4_28_10_30_41/emg_hero_history_2023_4_28_10_47_29.pkl',
        './logs/4_2023_4_28_10_30_41/emg_hero_history_2023_4_28_10_50_54.pkl',
        './logs/4_2023_4_28_10_30_41/emg_hero_history_2023_4_28_10_54_39.pkl',
        './logs/4_2023_4_28_10_30_41/emg_hero_history_2023_4_28_10_58_36.pkl',
        './logs/4_2023_4_28_10_30_41/emg_hero_history_2023_4_28_11_2_47.pkl',
        './logs/4_2023_4_28_10_30_41/emg_hero_history_2023_4_28_11_7_6.pkl',
        './logs/4_2023_4_28_10_30_41/emg_hero_history_2023_4_28_11_11_42.pkl',
        './logs/4_2023_4_28_10_30_41/emg_hero_history_2023_4_28_11_16_46.pkl',
        ]
    switch_lines = False
    create_experiment_dataset(experiment_name, history_filenames, switch_lines)

    experiment_name = "5_2023_4_28_13_15_15"
    history_filenames = [
    './logs/5_2023_4_28_13_15_15/emg_hero_history_2023_4_28_13_25_26.pkl',
    './logs/5_2023_4_28_13_15_15/emg_hero_history_2023_4_28_13_28_41.pkl',
    './logs/5_2023_4_28_13_15_15/emg_hero_history_2023_4_28_13_32_23.pkl',
    './logs/5_2023_4_28_13_15_15/emg_hero_history_2023_4_28_13_36_12.pkl',
    './logs/5_2023_4_28_13_15_15/emg_hero_history_2023_4_28_13_40_29.pkl',
    './logs/5_2023_4_28_13_15_15/emg_hero_history_2023_4_28_13_45_35.pkl',
    './logs/5_2023_4_28_13_15_15/emg_hero_history_2023_4_28_13_50_12.pkl',
    './logs/5_2023_4_28_13_15_15/emg_hero_history_2023_4_28_13_54_53.pkl',
    './logs/5_2023_4_28_13_15_15/emg_hero_history_2023_4_28_13_59_51.pkl',
    ]
    switch_lines = False
    create_experiment_dataset(experiment_name, history_filenames, switch_lines)


    experiment_name = "6_2023_5_1_11_39_54"
    history_filenames = [
    './logs/6_2023_5_1_11_39_54/emg_hero_history_2023_5_1_11_50_11.pkl',
    './logs/6_2023_5_1_11_39_54/emg_hero_history_2023_5_1_11_53_58.pkl',
    './logs/6_2023_5_1_11_39_54/emg_hero_history_2023_5_1_11_57_43.pkl',
    './logs/6_2023_5_1_11_39_54/emg_hero_history_2023_5_1_12_1_34.pkl',
    './logs/6_2023_5_1_11_39_54/emg_hero_history_2023_5_1_12_5_51.pkl',
    './logs/6_2023_5_1_11_39_54/emg_hero_history_2023_5_1_12_10_30.pkl',
    './logs/6_2023_5_1_11_39_54/emg_hero_history_2023_5_1_12_18_16.pkl',
    './logs/6_2023_5_1_11_39_54/emg_hero_history_2023_5_1_12_23_1.pkl',
    './logs/6_2023_5_1_11_39_54/emg_hero_history_2023_5_1_12_28_0.pkl',
    ]
    switch_lines = False
    create_experiment_dataset(experiment_name, history_filenames, switch_lines)

    experiment_name = "7_2023_5_1_14_47_16"
    history_filenames = [
    './logs/7_2023_5_1_14_47_16/emg_hero_history_2023_5_1_14_57_17.pkl',
    './logs/7_2023_5_1_14_47_16/emg_hero_history_2023_5_1_15_0_55.pkl',
    './logs/7_2023_5_1_14_47_16/emg_hero_history_2023_5_1_15_4_58.pkl',
    './logs/7_2023_5_1_14_47_16/emg_hero_history_2023_5_1_15_8_45.pkl',
    './logs/7_2023_5_1_14_47_16/emg_hero_history_2023_5_1_15_12_45.pkl',
    './logs/7_2023_5_1_14_47_16/emg_hero_history_2023_5_1_15_17_6.pkl',
    './logs/7_2023_5_1_14_47_16/emg_hero_history_2023_5_1_15_21_28.pkl',
    './logs/7_2023_5_1_14_47_16/emg_hero_history_2023_5_1_15_26_15.pkl',
    './logs/7_2023_5_1_14_47_16/emg_hero_history_2023_5_1_15_31_8.pkl',
    ]
    switch_lines = False
    create_experiment_dataset(experiment_name, history_filenames, switch_lines)

    experiment_name = "8_2023_5_2_16_4_41"
    history_filenames = [
    './logs/8_2023_5_2_16_4_41/emg_hero_history_2023_5_2_16_18_3.pkl',
    './logs/8_2023_5_2_16_4_41/emg_hero_history_2023_5_2_16_21_37.pkl',
    './logs/8_2023_5_2_16_4_41/emg_hero_history_2023_5_2_16_25_1.pkl',
    './logs/8_2023_5_2_16_4_41/emg_hero_history_2023_5_2_16_29_16.pkl',
    './logs/8_2023_5_2_16_4_41/emg_hero_history_2023_5_2_16_33_43.pkl',
    './logs/8_2023_5_2_16_4_41/emg_hero_history_2023_5_2_16_37_58.pkl',
    './logs/8_2023_5_2_16_4_41/emg_hero_history_2023_5_2_16_42_25.pkl',
    './logs/8_2023_5_2_16_4_41/emg_hero_history_2023_5_2_16_47_11.pkl',
    './logs/8_2023_5_2_16_4_41/emg_hero_history_2023_5_2_16_52_9.pkl',
    ]
    switch_lines = False
    create_experiment_dataset(experiment_name, history_filenames, switch_lines)

    experiment_name = "9_2023_5_3_10_35_1"
    history_filenames = [
    './logs/9_2023_5_3_10_35_1/emg_hero_history_2023_5_3_10_45_13.pkl',
    './logs/9_2023_5_3_10_35_1/emg_hero_history_2023_5_3_10_48_34.pkl',
    './logs/9_2023_5_3_10_35_1/emg_hero_history_2023_5_3_10_52_3.pkl',
    './logs/9_2023_5_3_10_35_1/emg_hero_history_2023_5_3_10_56_3.pkl',
    './logs/9_2023_5_3_10_35_1/emg_hero_history_2023_5_3_11_0_19.pkl',
    './logs/9_2023_5_3_10_35_1/emg_hero_history_2023_5_3_11_4_52.pkl',
    './logs/9_2023_5_3_10_35_1/emg_hero_history_2023_5_3_11_9_26.pkl',
    './logs/9_2023_5_3_10_35_1/emg_hero_history_2023_5_3_11_14_4.pkl',
    './logs/9_2023_5_3_10_35_1/emg_hero_history_2023_5_3_11_19_5.pkl',
    ]
    switch_lines = False
    create_experiment_dataset(experiment_name, history_filenames, switch_lines)

    experiment_name = "10_2023_5_3_14_23_26"
    history_filenames = [
'./logs/10_2023_5_3_14_23_26/emg_hero_history_2023_5_3_14_51_56.pkl',
 './logs/10_2023_5_3_14_23_26/emg_hero_history_2023_5_3_15_0_32.pkl',
 './logs/10_2023_5_3_14_23_26/emg_hero_history_2023_5_3_15_4_0.pkl',
 './logs/10_2023_5_3_14_23_26/emg_hero_history_2023_5_3_15_7_44.pkl',
 './logs/10_2023_5_3_14_23_26/emg_hero_history_2023_5_3_15_11_43.pkl',
 './logs/10_2023_5_3_14_23_26/emg_hero_history_2023_5_3_15_15_58.pkl',
 './logs/10_2023_5_3_14_23_26/emg_hero_history_2023_5_3_15_20_21.pkl',
 './logs/10_2023_5_3_14_23_26/emg_hero_history_2023_5_3_15_25_8.pkl',
 './logs/10_2023_5_3_14_23_26/emg_hero_history_2023_5_3_15_30_47.pkl',
]
    switch_lines = False
    create_experiment_dataset(experiment_name, history_filenames, switch_lines)

    experiment_name = "11_2023_5_17_16_32_53"
    history_filenames = [
'./logs/11_2023_5_17_16_32_53/emg_hero_history_2023_5_17_16_46_44.pkl',
 './logs/11_2023_5_17_16_32_53/emg_hero_history_2023_5_17_16_49_59.pkl',
 './logs/11_2023_5_17_16_32_53/emg_hero_history_2023_5_17_16_53_29.pkl',
 './logs/11_2023_5_17_16_32_53/emg_hero_history_2023_5_17_16_57_27.pkl',
 './logs/11_2023_5_17_16_32_53/emg_hero_history_2023_5_17_17_1_57.pkl',
 './logs/11_2023_5_17_16_32_53/emg_hero_history_2023_5_17_17_6_34.pkl',
 './logs/11_2023_5_17_16_32_53/emg_hero_history_2023_5_17_17_11_16.pkl',
 './logs/11_2023_5_17_16_32_53/emg_hero_history_2023_5_17_17_15_53.pkl',
 './logs/11_2023_5_17_16_32_53/emg_hero_history_2023_5_17_17_20_58.pkl',
]
    switch_lines = False
    create_experiment_dataset(experiment_name, history_filenames, switch_lines)
    experiment_name = "12_2023_6_6_22_11_55"
    history_filenames = [
'./logs/12_2023_6_6_22_11_55/emg_hero_history_2023_6_6_22_22_37.pkl',
 './logs/12_2023_6_6_22_11_55/emg_hero_history_2023_6_6_22_25_56.pkl',
 './logs/12_2023_6_6_22_11_55/emg_hero_history_2023_6_6_22_29_46.pkl',
 './logs/12_2023_6_6_22_11_55/emg_hero_history_2023_6_6_22_33_37.pkl',
 './logs/12_2023_6_6_22_11_55/emg_hero_history_2023_6_6_22_37_59.pkl',
 './logs/12_2023_6_6_22_11_55/emg_hero_history_2023_6_6_22_42_29.pkl',
 './logs/12_2023_6_6_22_11_55/emg_hero_history_2023_6_6_22_47_19.pkl',
 './logs/12_2023_6_6_22_11_55/emg_hero_history_2023_6_6_22_52_27.pkl',
 './logs/12_2023_6_6_22_11_55/emg_hero_history_2023_6_6_22_58_28.pkl',
]
    switch_lines = False
    create_experiment_dataset(experiment_name, history_filenames, switch_lines)
    experiment_name = "13_2023_6_12_12_59_19"
    history_filenames = [
'./logs/13_2023_6_12_12_59_19/emg_hero_history_2023_6_12_13_9_32.pkl',
 './logs/13_2023_6_12_12_59_19/emg_hero_history_2023_6_12_13_12_47.pkl',
 './logs/13_2023_6_12_12_59_19/emg_hero_history_2023_6_12_13_16_17.pkl',
 './logs/13_2023_6_12_12_59_19/emg_hero_history_2023_6_12_13_20_0.pkl',
 './logs/13_2023_6_12_12_59_19/emg_hero_history_2023_6_12_13_23_56.pkl',
 './logs/13_2023_6_12_12_59_19/emg_hero_history_2023_6_12_13_28_11.pkl',
 './logs/13_2023_6_12_12_59_19/emg_hero_history_2023_6_12_13_32_34.pkl',
 './logs/13_2023_6_12_12_59_19/emg_hero_history_2023_6_12_13_37_10.pkl',
 './logs/13_2023_6_12_12_59_19/emg_hero_history_2023_6_12_13_41_59.pkl',
]
    switch_lines = False
    create_experiment_dataset(experiment_name, history_filenames, switch_lines)
    experiment_name = "14_2023_6_12_22_53_45"
    history_filenames = [
'./logs/14_2023_6_12_22_53_45/emg_hero_history_2023_6_12_23_3_16.pkl',
 './logs/14_2023_6_12_22_53_45/emg_hero_history_2023_6_12_23_6_38.pkl',
 './logs/14_2023_6_12_22_53_45/emg_hero_history_2023_6_12_23_11_1.pkl',
 './logs/14_2023_6_12_22_53_45/emg_hero_history_2023_6_12_23_14_51.pkl',
 './logs/14_2023_6_12_22_53_45/emg_hero_history_2023_6_12_23_19_26.pkl',
 './logs/14_2023_6_12_22_53_45/emg_hero_history_2023_6_12_23_24_4.pkl',
 './logs/14_2023_6_12_22_53_45/emg_hero_history_2023_6_12_23_28_32.pkl',
 './logs/14_2023_6_12_22_53_45/emg_hero_history_2023_6_12_23_33_21.pkl',
 './logs/14_2023_6_12_22_53_45/emg_hero_history_2023_6_12_23_38_28.pkl',
]
    switch_lines = False
    create_experiment_dataset(experiment_name, history_filenames, switch_lines)
    experiment_name = "15_2023_6_21_11_14_33"
    history_filenames = [
'./logs/15_2023_6_21_11_14_33/emg_hero_history_2023_6_21_11_24_58.pkl',
 './logs/15_2023_6_21_11_14_33/emg_hero_history_2023_6_21_11_29_0.pkl',
 './logs/15_2023_6_21_11_14_33/emg_hero_history_2023_6_21_11_32_35.pkl',
 './logs/15_2023_6_21_11_14_33/emg_hero_history_2023_6_21_11_36_22.pkl',
 './logs/15_2023_6_21_11_14_33/emg_hero_history_2023_6_21_11_40_21.pkl',
 './logs/15_2023_6_21_11_14_33/emg_hero_history_2023_6_21_11_44_43.pkl',
 './logs/15_2023_6_21_11_14_33/emg_hero_history_2023_6_21_11_49_45.pkl',
 './logs/15_2023_6_21_11_14_33/emg_hero_history_2023_6_21_11_54_37.pkl',
 './logs/15_2023_6_21_11_14_33/emg_hero_history_2023_6_21_11_59_58.pkl',
]
    switch_lines = False
    create_experiment_dataset(experiment_name, history_filenames, switch_lines)

    experiment_name = "16_2023_6_22_15_26_7"
    history_filenames = [
'./logs/16_2023_6_22_15_26_7/emg_hero_history_2023_6_22_15_36_6.pkl',
 './logs/16_2023_6_22_15_26_7/emg_hero_history_2023_6_22_15_39_29.pkl',
 './logs/16_2023_6_22_15_26_7/emg_hero_history_2023_6_22_15_43_0.pkl',
 './logs/16_2023_6_22_15_26_7/emg_hero_history_2023_6_22_15_46_58.pkl',
 './logs/16_2023_6_22_15_26_7/emg_hero_history_2023_6_22_15_51_13.pkl',
 './logs/16_2023_6_22_15_26_7/emg_hero_history_2023_6_22_15_55_52.pkl',
 './logs/16_2023_6_22_15_26_7/emg_hero_history_2023_6_22_16_0_29.pkl',
 './logs/16_2023_6_22_15_26_7/emg_hero_history_2023_6_22_16_5_23.pkl',
 './logs/16_2023_6_22_15_26_7/emg_hero_history_2023_6_22_16_11_8.pkl',
]
    switch_lines = False
    create_experiment_dataset(experiment_name, history_filenames, switch_lines)



if __name__ == "__main__":
    main()