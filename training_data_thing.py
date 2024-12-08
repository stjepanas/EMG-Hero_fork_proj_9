import libemg
from libemg.gui import GUI
from libemg.data_handler import OnlineDataHandler
from libemg import streamers
from libemg.feature_extractor import FeatureExtractor
from libemg.emg_predictor import OnlineEMGClassifier, EMGClassifier
import time
from libemg.filtering import Filter
from libemg.utils import get_windows
import numpy as np
import matplotlib.pyplot as plt
from libemg.data_handler import OfflineDataHandler, RegexFilter
from d3rlpy.dataset import MDPDataset
from scipy.io import loadmat
from d3rlpy.constants import ActionSpace

# how do we get the other images to the GUI so that we can get the data???

if __name__ == "__main__":
    streamer, smm = streamers.sifi_bioarmband_streamer(emg_notch_freq=50,
                                                #bridge_version="1.1.3",
                                                name="BioArmband",
                                                ecg=False, emg=True, eda=False, imu=False, ppg=False)
    #ondh = OnlineDataHandler(smm)

    #training_ui = GUI(ondh, width=700, height=700, gesture_height=300, gesture_width=300)
    #training_ui.download_gestures([1,2,3,4,5], "\images")
    #training_ui.start_gui()

    dataset_folder = 'data'
    gestures = ["0","1","2","3","4"]
    reps = ["0","1","2"]
    regex_filters = [
    RegexFilter(left_bound = "C_", right_bound="_R", values = gestures, description='classes'),
    RegexFilter(left_bound = "R_", right_bound="_emg.csv", values = reps, description='reps'),
]
    WINDOW_SIZE = 300 # TODO 200 ms window
    WINDOW_INCREMENT = 75 # TODO calculate increment 50 ms

    offline_dh = OfflineDataHandler()
    offline_dh.get_data(folder_location = dataset_folder, regex_filters=regex_filters, delimiter=",")

    train_data = offline_dh.isolate_data("reps", [0])
    test_data = offline_dh.isolate_data("reps", [1])

    train_windows, train_meta = train_data.parse_windows(WINDOW_SIZE, WINDOW_INCREMENT)#(488x8x300)
    test_windows, test_meta = test_data.parse_windows(WINDOW_SIZE, WINDOW_INCREMENT) 
    # this is the same length as the amount of windows, we can just take
    # the integer from the meta data for every step in the loop and one hot encode it

    fe = FeatureExtractor()
    feature_list = ['MAV', 'SSC', 'ZC', 'WL']
    training_features = fe.extract_features(feature_list, train_windows)
    testing_features = fe.extract_features(feature_list, test_windows)
    # print(np.shape(training_features['MAV'])) # This returns 488x8 array, so 1x8 array for every
    # window. Loop through the nr of windows and append
    # the training features for every feature
    
    nr_windows = np.shape(train_windows)[0]
    nr_channels = np.shape(train_windows)[1]
    
    observations_train = np.zeros((nr_windows, len(feature_list)*nr_channels)) # so a 488X(4*8)
    observations_test = np.zeros((nr_windows, len(feature_list)*nr_channels)) 

    actions_train = np.zeros((nr_windows, len(gestures)+1))
    actions_test = np.zeros((nr_windows, len(gestures)+1))

    rewards = np.zeros(nr_windows)
    terminals = np.zeros((nr_windows))
    
    # took this from datasets, DO NOT KNOW what this means
    pretrain_terminals = np.array([1 if ((i+1) % 47 == 0 and i > 0) else 0 \
                                         for i in range(actions_train.shape[0])]) # why 47???

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
        actions_train[i, int(gesture_train_class)] = 1  # One-hot encode the action based on class label,
        
        gesture_test_class = test_meta['classes'][i] 
        actions_test[i, int(gesture_test_class)] = 1 
                                        


    dataset_pretrain = MDPDataset(observations_train,actions_train,rewards,terminals,timeouts = pretrain_terminals,action_space = ActionSpace.CONTINUOUS)
    dataset_pretest = MDPDataset(observations_test,actions_test,rewards,terminals,timeouts = pretrain_terminals,action_space = ActionSpace.CONTINUOUS)

###############################################################################################
#mat_data = loadmat('afeRec_2023-04-21 15-03.mat', squeeze_me = True)
    #print(mat_data) 
    # this then probably not needed??
    # Step 4: Create the EMG Classifier
    #o_classifier = EMGClassifier("MLP") # there is an MLP classifier
    #o_classifier.fit(feature_dictionary=data_set)

    # Online classification

    #odh.analyze_hardware()
    #fe = FeatureExtractor()
    #offline_classifier = 0
   
    #classififer_thing = OnlineEMGClassifier(offline_classifier, window_size, window_increment, online_data_handler, features, file_path='.', file=False, smm=False, smm_items=None, port=12346, ip='127.0.0.1', std_out=False, tcp=False, output_format='predictions')