import libemg
from libemg.gui import GUI
from libemg import streamers
from libemg.feature_extractor import FeatureExtractor
import numpy as np
from libemg.data_handler import OfflineDataHandler, RegexFilter
from libemg.data_handler import OnlineDataHandler
from d3rlpy.algos import BCConfig
import argparse
from emg_hero.metrics import F1MacroEvaluator
from d3rlpy import load_learnable
from create_MDPDatasets import get_training_dataset, get_testing_dataset
from d3rlpy.dataset import MDPDataset
from d3rlpy.constants import ActionSpace  
from emg_hero.model_utility import build_algo 
from emg_hero.configs import BaseConfig
from gym_problem import reverse_mapping, mapping

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
    # # download function doesn't really need to be used if you want to train with the movements in the images map
    # #training_ui.download_gestures([10,11,12], r'images\\', download_imgs=False)
    # training_ui.start_gui()

    # folder where raw data from the armband is going to be stored
    dataset_folder = 'data_stjepana'
    # gestures picked out for training, this is according to the gesture_list.json/collection.json
    gestures = ["0","1","2","3","4","5","6","7","8","9","10","11","12"]
    reps = ["0","1","2","3","4","5"]
    # filters out data files that don't contain gestures/reps/types of signals specified above
    regex_filters = [
    RegexFilter(left_bound = "C_", right_bound="_R", values = gestures, description='classes'),
    RegexFilter(left_bound = "R_", right_bound="_emg.csv", values = reps, description='reps'),
]
    WINDOW_SIZE = 300 
    WINDOW_INCREMENT = 75 

    offline_dh = OfflineDataHandler()
    offline_dh.get_data(folder_location = dataset_folder, regex_filters=regex_filters, delimiter=",")

    # separates raw data into testing and training data
    train_data = offline_dh.isolate_data("reps", [0,1,2,3]) # more data for training than for testing
    test_data = offline_dh.isolate_data("reps", [4,5])

    # shapes data into windows x channels x nr of samples in a window, so nr_windows x 8 x 300
    # meta tells which movement is done for every window, gives a number from the gestures
    train_windows, train_meta = train_data.parse_windows(WINDOW_SIZE, WINDOW_INCREMENT)
    size_windows_train = np.shape(train_windows)[0]
    test_windows, test_meta = test_data.parse_windows(WINDOW_SIZE, WINDOW_INCREMENT) 
    size_windows_test = np.shape(test_windows)[0]

    fe = FeatureExtractor()
    feature_list = ['MAV', 'SSC', 'ZC', 'WL']
    # extracts the necessary features, returns a dictionary where keys are the features
    # this is done for each window, so each window gets a dictionary of features
    training_features = fe.extract_features(feature_list, train_windows)
    testing_features = fe.extract_features(feature_list, test_windows)

    # returns everything that goes into the MDPDataset function
    observations_train,actions_train,rewards_train,terminals_train,pretrain_timeouts = get_training_dataset(gestures, feature_list, size_windows_train, training_features, train_meta)
    
    observations_test,actions_test,rewards_test,terminals_test,pretest_timeouts = get_testing_dataset(gestures , feature_list, size_windows_test, testing_features, test_meta)

    # returns the MDPDatasets
    #dataset_pretrain = MDPDataset(observations_train,actions_train,rewards_train,terminals_train,pretrain_timeouts,action_space = ActionSpace.CONTINUOUS)
    #dataset_pretest = MDPDataset(observations_test,actions_test,rewards_test,terminals_test,pretest_timeouts,action_space = ActionSpace.CONTINUOUS)
    
################ Behaviour cloning ################################################################
#     supervised_train = dataset_pretrain
#     supervised_test = dataset_pretest
#     # takes the previous project's hyperparameters for the model
#     base_config = BaseConfig()

#     model, floornoise = build_algo(pt_weights=None, pt_biases=None, base_config=base_config)

#     f1_macro_evaluator = F1MacroEvaluator(supervised_test.episodes)
    
#     bc_model = model.fit(
#       supervised_train,
#       n_steps=1000,
#       n_steps_per_epoch=50,
#       evaluators={
#           "f1_macro": f1_macro_evaluator,
#      }
#  )
    # observation = observations_test
    # gt_actions = actions_test
    # # bc_model = load_learnable(r"d3rlpy_logs\BC_20241216175424\model_100000.d3")
    # model_path = r"d3rlpy_logs\BC_20241216161532\model_1000.d3"
    # bc_model = load_learnable(model_path)
    # actions = bc_model.predict(observation)




