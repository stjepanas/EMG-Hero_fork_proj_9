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
from d3rlpy.metrics import TDErrorEvaluator
from d3rlpy.metrics import EnvironmentEvaluator
from gym_env import EMGHeroEnv
from get_action_vector import get_action_vector
from emg_hero.model_utility import pretrain_emg_hero_model
import argparse
import gymnasium as gym

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
    train_data = offline_dh.isolate_data("reps", [0]) # should we have only one of these?
    test_data = offline_dh.isolate_data("reps", [1])

    train_windows, train_meta = train_data.parse_windows(WINDOW_SIZE, WINDOW_INCREMENT)#(488x8x300)
    test_windows, test_meta = test_data.parse_windows(WINDOW_SIZE, WINDOW_INCREMENT) 

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
    print(np.shape(observations_train))
                                         
    dataset_pretrain = MDPDataset(observations_train,actions_train,rewards,terminals,pretrain_timeouts,action_space = ActionSpace.CONTINUOUS)
    
    dataset_pretest = MDPDataset(observations_test,actions_test,rewards,terminals,pretrain_timeouts,action_space = ActionSpace.CONTINUOUS)

################ Behaviour cloning ################################################################
    bc = BCConfig().create(device=False) 
   # initialize neural networks with the given observation shape and action size.
   # this is not necessary when you directly call fit or fit_online method.
    bc.build_with_dataset(dataset_pretrain)
    # calculate metrics with training dataset


    #td_error_evaluator = TDErrorEvaluator()
    # set environment in scorer function
    #env = gym.make('Blackjack-v1')

    #env_evaluator = EnvironmentEvaluator(env)
    # evaluate algorithm on the environment
    # is it the pretrain or pretest dataset here????
    #rewards = env_evaluator(bc, dataset=dataset_pretrain)
    # Offline training
    bc.fit(
    dataset_pretrain,
    n_steps=100000,
    n_steps_per_epoch=10000
) 

# TODO
# add README
# validation process
#mat_data = loadmat('afeRec_2023-04-21 15-03.mat', squeeze_me = True)
