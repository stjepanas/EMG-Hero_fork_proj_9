import libemg
from libemg.gui import GUI
from libemg.data_handler import OnlineDataHandler
from libemg import streamers
from libemg.feature_extractor import FeatureExtractor
import numpy as np
from libemg.data_handler import OfflineDataHandler, RegexFilter
from d3rlpy.dataset import MDPDataset
from d3rlpy.constants import ActionSpace
from d3rlpy.algos import BCConfig
from gym_problem import action_mapping
from gym_env import EMGHeroEnv
from get_action_vector import get_action_vector
from emg_hero.model_utility import pretrain_emg_hero_model
import argparse
import gymnasium as gym
from emg_hero.metrics import F1MacroEvaluator
from emg_hero.bc_modified import pretrain_emg_hero_model_modified
from emg_hero.model_utility import get_newest_directory
import os
import re
from d3rlpy import load_learnable
from d3rlpy.logging import FileAdapterFactory

if __name__ == "__main__":

    #streamer, smm = streamers.sifi_bioarmband_streamer(emg_notch_freq=50,
                                                #bridge_version="1.1.3",
                                                #name="BioArmband",
                                                #ecg=False, emg=True, eda=False, imu=False, ppg=False)
    #online_dh = OnlineDataHandler(smm)

    #training_ui = GUI(online_dh, width=700, height=700, gesture_height=300, gesture_width=300)
    #training_ui.download_gestures([1,2,3,4,5,6,7,8,9,10,11,12,13], r'images\\', download_imgs=False)
    #training_ui.start_gui()

    dataset_folder = 'data'
    gestures = ["0","1","2","3","4","5","6","7","8","9","10","11","12"]
    reps = ["0","1","2"]
    regex_filters = [
    RegexFilter(left_bound = "C_", right_bound="_R", values = gestures, description='classes'),
    RegexFilter(left_bound = "R_", right_bound="_emg.csv", values = reps, description='reps'),
]
    WINDOW_SIZE = 300 # TODO comment where this comes from
    WINDOW_INCREMENT = 75 # TODO calculate increment 50 ms

    offline_dh = OfflineDataHandler()
    offline_dh.get_data(folder_location = dataset_folder, regex_filters=regex_filters, delimiter=",")

# if we have multiple reps, it stacks the same movements together, so rep 1 of the first move, rep 2 of the first
# move and so on
    train_data = offline_dh.isolate_data("reps", [0]) 
    test_data = offline_dh.isolate_data("reps", [1])

    train_windows, train_meta = train_data.parse_windows(WINDOW_SIZE, WINDOW_INCREMENT)#(488x8x300)
    test_windows, test_meta = test_data.parse_windows(WINDOW_SIZE, WINDOW_INCREMENT) 
    print('train_windows:', train_windows[-1])
    print('test_windows:', test_windows[-1])

    fe = FeatureExtractor()
    feature_list = ['MAV', 'SSC', 'ZC', 'WL']
    training_features = fe.extract_features(feature_list, train_windows)
    testing_features = fe.extract_features(feature_list, test_windows)
    print(np.shape(training_features['MAV'])) 
    print(np.shape(testing_features['MAV'])) 
    
    nr_windows = np.shape(test_windows)[0]
    nr_channels = np.shape(test_windows)[1]
    
    observations_train = np.zeros((nr_windows, len(feature_list)*nr_channels)) # so a 488X(4*8)
    observations_test = np.zeros((nr_windows, len(feature_list)*nr_channels)) 

    actions_train = np.zeros((nr_windows, 7))
    actions_test = np.zeros((nr_windows, 7))

    rewards = np.zeros(nr_windows)
    terminals = np.zeros((nr_windows))
   
   # checks how many zeros there are in the beginning to check for sample size for
   # each movement
    counter = 0
    for i in train_meta['classes']:
        if i == int(gestures[0]):
            counter += 1
        else:
            break

    print('counter=', counter)

    pretrain_timeouts = np.array([1 if ((i+1) % (counter) == 0 and i > 0) else 0 \
                                     for i in range(actions_train.shape[0])]) # maybe not correct

    for i in range(nr_windows):
    # Extract features for the current window
        feature_train_vector = []
        feature_test_vector = []
        for feature in feature_list:
            # Concatenate features from all channels
            feature_train_vector.extend(training_features[feature][i])
            feature_test_vector.extend(testing_features[feature][i])

        observations_train[i] = feature_train_vector
        observations_test[i] = feature_test_vector

        gesture_train_class = train_meta['classes'][i] 
        actions_train[i] = get_action_vector(gesture_train_class)

        gesture_test_class = test_meta['classes'][i] 
        actions_test[i] = get_action_vector(gesture_test_class)

    #print('get_action vector:',get_action_vector(1))
    #print(np.shape(observations_train))
    print('observations_train:', observations_train[0])   
    print('obsrrvation_size:', np.shape(observations_train[0]))                           
    dataset_pretrain = MDPDataset(observations_train,actions_train,rewards,terminals,pretrain_timeouts,action_space = ActionSpace.CONTINUOUS)
    dataset_pretest = MDPDataset(observations_test,actions_test,rewards,terminals,pretrain_timeouts,action_space = ActionSpace.CONTINUOUS)
    dataset_pretrain.dump('dataset_pretrain.h5')
################ Behaviour cloning ################################################################
    #bc = BCConfig().create(device=False) 
    #bc.build_with_dataset(dataset_pretrain)

    #f1_macro_evaluator = F1MacroEvaluator(dataset_pretest.episodes)
    # Offline training
    #bc.fit(
    #dataset_pretrain,
    #n_steps=10000,
    #n_steps_per_epoch=10000,
    #evaluators={"f1_macro": f1_macro_evaluator}
    #)

    #supervised_train = dataset_pretrain
    #supervised_test = dataset_pretest

    #bc = BCConfig().create(device="cpu")#.from_json(bc_config_file)
    #bc.build_with_dataset(supervised_train)

    #n_samples = np.sum([e.observations.shape[0] for e in supervised_train.episodes])
    #LOGGER.info("Train dataset size: %i", n_samples)

    #f1_macro_evaluator = F1MacroEvaluator(supervised_test.episodes)

#    bc_hist = bc_model.fit(
#    supervised_train,
#    n_steps=10000,
#    n_steps_per_epoch=50,
#    evaluators={
#        "f1_macro": f1_macro_evaluator,
#    }
#)

    #observation = observations_train
    #action = bc.predict([observation])
    #print()
    #print(action)

'''
    f1s_array = [h[1]["f1_macro"] for h in bc_hist]

    newest_dir_path = get_newest_directory('\EMG-Hero_fork_proj_9\d3rlpy_logs')
    

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
    print('The best models path is',best_model_path)
    #LOGGER.info(
     #   "Best model is %s with index %i and F1 score %f",
     #   best_model_filename,
      #  best_model_idx,
      #  np.max(f1s_array),
    #)
    #LOGGER.info(best_model_path)
    #print(f"Took model {best_model_idx}: {best_model_filename}")

    bc_model = load_learnable(best_model_path)
    '''
    

# TODO
# add README
# validation process
#mat_data = loadmat('afeRec_2023-04-21 15-03.mat', squeeze_me = True)
