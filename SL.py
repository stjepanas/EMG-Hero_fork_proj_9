import libemg
from libemg.gui import GUI
from libemg import streamers
from libemg.feature_extractor import FeatureExtractor
import numpy as np
from libemg.data_handler import OfflineDataHandler, RegexFilter
from libemg.data_handler import OnlineDataHandler
from d3rlpy.algos import BCConfig
from emg_hero.model_utility import pretrain_emg_hero_model
import argparse
from emg_hero.metrics import F1MacroEvaluator
from emg_hero.model_utility import get_newest_directory
import os
import re
from d3rlpy import load_learnable
from d3rlpy.logging import FileAdapterFactory
from create_MDPDatasets import get_training_dataset, get_testing_dataset
from d3rlpy.dataset import MDPDataset
from d3rlpy.constants import ActionSpace  
from emg_hero.model_utility import build_algo 
from emg_hero.configs import BaseConfig

if __name__ == "__main__":

    # streamer, smm = streamers.sifi_bioarmband_streamer(
    #                                             filtering= True,
    #                                             emg_bandpass=[20,500],  # since lowpass = 20 and highpass = 500
    #                                             emg_notch_freq=50,      # notch filter at 50hz
    #                                             #bridge_version="1.1.3",
    #                                             name="BioArmband",
    #                                             ecg=False, emg=True, eda=False, imu=False, ppg=False)
    # online_dh = OnlineDataHandler(smm)

    # training_ui = GUI(online_dh, width=700, height=700, gesture_height=300, gesture_width=300)
    # training_ui.download_gestures([0,1,2,3,4,5,6,7,8,9,10,11,12], r'images\\', download_imgs=False)
    # training_ui.start_gui()

    # where raw data from the armband is stored
    dataset_folder = 'data'
    # gestures picked out for training, this is according to the gesture_list json/collection json
    gestures = ["0","1","2","3","4","5","6","7","8","9","10","11","12"]
    reps = ["0","1","2","3","4","5","6","7"]
    regex_filters = [
    RegexFilter(left_bound = "C_", right_bound="_R", values = gestures, description='classes'),
    RegexFilter(left_bound = "R_", right_bound="_emg.csv", values = reps, description='reps'),
]
    WINDOW_SIZE = 300 
    WINDOW_INCREMENT = 75 

    offline_dh = OfflineDataHandler()
    offline_dh.get_data(folder_location = dataset_folder, regex_filters=regex_filters, delimiter=",")

    train_data = offline_dh.isolate_data("reps", [0,1,2,3]) # TODO: too little data, try eigth more reps
    test_data = offline_dh.isolate_data("reps", [4,5])

    train_windows, train_meta = train_data.parse_windows(WINDOW_SIZE, WINDOW_INCREMENT)#(windows x channels x samples)
    size_windows_train = np.shape(train_windows)[0]
    test_windows, test_meta = test_data.parse_windows(WINDOW_SIZE, WINDOW_INCREMENT) 
    size_windows_test = np.shape(test_windows)[0]
    print('train_meta unique moves:', np.unique(train_meta['classes']))

    fe = FeatureExtractor()
    feature_list = ['MAV', 'SSC', 'ZC', 'WL']
    training_features = fe.extract_features(feature_list, train_windows)
    testing_features = fe.extract_features(feature_list, test_windows)

    observations_train,actions_train,rewards_train,terminals_train,pretrain_timeouts = get_training_dataset(gestures, feature_list, size_windows_train, training_features, train_meta)
    observations_test,actions_test,rewards_test,terminals_test,pretest_timeouts = get_testing_dataset(gestures , feature_list, size_windows_test, testing_features, test_meta)

    dataset_pretrain = MDPDataset(observations_train,actions_train,rewards_train,terminals_train,pretrain_timeouts,action_space = ActionSpace.CONTINUOUS)
    dataset_pretest = MDPDataset(observations_test,actions_test,rewards_test,terminals_test,pretest_timeouts,action_space = ActionSpace.CONTINUOUS)
    
# ################ Behaviour cloning ################################################################
    #supervised_train = dataset_pretrain
    #supervised_test = dataset_pretest
    # print(np.shape(actions_train))
    # print(actions_train)
    #base_config = BaseConfig()

    #model, floornoise = build_algo(pt_weights=None, pt_biases=None, base_config=base_config)

#     n_samples = np.sum([e.observations.shape[0] for e in supervised_train.episodes])
#     LOGGER.info("Train dataset size: %i", n_samples)

    #f1_macro_evaluator = F1MacroEvaluator(supervised_test.episodes)
    
    # TODO: Copy config from model_utility for BCConfig
#     bc_model = model.fit(
#     supervised_train,
#     n_steps=1000,
#     n_steps_per_epoch=50,
#     evaluators={
#         "f1_macro": f1_macro_evaluator,
#     }
# )
    # observation = observations_test
    # action = actions_test
    # bc_model = load_learnable('d3rlpy_logs\BC_20241216134326\model_50.d3')
    # actions = bc_model.predict(observation)
    # print('predicted actions:',actions, 'actual actions:', actions_test)



    # f1s_array = [h[1]["f1_macro"] for h in bc_hist]

    # newest_dir_path = get_newest_directory('\EMG-Hero_fork_proj_9\d3rlpy_logs')
    

    # model_files = []
    # for _, _, files in os.walk(newest_dir_path):
    #     for file in files:
    #         if file.endswith(".d3"):
    #             model_files.append(file)

    # # sort model_files
    # numbers = [int(re.findall("[0-9]+", m_file)[0]) for m_file in model_files]
    # sorted_model_names = sorted(zip(numbers, model_files))

    # # pick best model
    # best_model_idx = np.argmax(f1s_array)
    # best_model_filename = sorted_model_names[best_model_idx][1]

    # # get path of best model
    # best_model_path = newest_dir_path / best_model_filename
    # print('The best models path is',best_model_path)
    # #LOGGER.info(
    #  #   "Best model is %s with index %i and F1 score %f",
    #  #   best_model_filename,
    #   #  best_model_idx,
    #   #  np.max(f1s_array),
    # #)
    # #LOGGER.info(best_model_path)
    # #print(f"Took model {best_model_idx}: {best_model_filename}")

    # bc_model = load_learnable(best_model_path)

    

# TODO
# add README
# validation process
