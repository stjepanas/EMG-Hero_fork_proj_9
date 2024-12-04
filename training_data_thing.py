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
    WINDOW_SIZE = 300 # todo: 200 ms window
    WINDOW_INCREMENT = 75 # todo calculate increment 50 ms

    offline_dh = OfflineDataHandler()
    offline_dh.get_data(folder_location = dataset_folder, regex_filters=regex_filters, delimiter=",")

    train_data = offline_dh.isolate_data("reps", [0])
    test_data = offline_dh.isolate_data("reps", [1])

    train_windows, train_meta = train_data.parse_windows(WINDOW_SIZE, WINDOW_INCREMENT)
    test_windows, test_meta = test_data.parse_windows(WINDOW_SIZE, WINDOW_INCREMENT)
    print(test_meta)

    print(train_windows)

    fe = FeatureExtractor()
    feature_list = ['MAV', 'SSC', 'ZC', 'WL']
    training_features = fe.extract_features(feature_list, train_windows)
    # print(np.shape(training_features['MAV']))
    
    nr_windows = np.shape(train_windows)[0]
    print(nr_windows)
    nr_channels = np.shape(train_windows)[-1]
    observations = np.zeros((nr_windows, len(feature_list)*nr_channels))
    # 1000 steps of actions with shape of (4,)
    actions = np.zeros((nr_windows, len(gestures)+1))
    # 1000 steps of rewards
    rewards = np.zeros(nr_windows)
    # 1000 steps of terminal flags
    terminals = np.zeros((2, 488))

    
    for i in range(nr_windows):
        feature_vector = []
        for feature in feature_list:
        # Collect features for the current window across all channels
            feature_vector.extend(training_features[feature][i])
            observations[i] = feature_vector  # Save feature vector to observations array

    # Example: Populate actions with dummy data (assumes class info is in train_meta)
        gesture_class = train_meta[i]["class"]  # Assuming train_meta has a 'class' field
        actions[i, int(gesture_class)] = 1  # One-hot encode the gesture class


    
    #print(np.shape(test_windows))
    #print(train_windows[0,:,:]) # need to extract the features for every window, 489 windows with 300 samples
    # the observations are the features and actions the intended movements
    dataset = MDPDataset(observations, actions, rewards, terminals)
    print(dataset)
    # 1000 steps of observations with shape of (100,)
 

###############################################################################################
    # Extract windows
    #train_windows, train_meta = train_data.parse_windows(WINDOW_SIZE, WINDOW_INCREMENT)
    #test_windows, test_meta = test_data.parse_windows(WINDOW_SIZE, WINDOW_INCREMENT)
    
    # don't know if we need the parse_windows thing?
    # train_windows, train_metadata = train_odh.parse_windows(WINDOW_SIZE,WINDOW_INCREMENT)

    # Extract features
    #fe = FeatureExtractor()
    #feature_list = ['MAV', 'SSC', 'ZC', 'WL']
    #training_features = fe.extract_features(feature_list, train_windows)

    # this then probably not needed??
    # Step 4: Create the EMG Classifier
    #o_classifier = EMGClassifier("LDA") # there is an MLP classifier
    #o_classifier.fit(feature_dictionary=data_set)

    # Online classification

    #odh.analyze_hardware()
    #fe = FeatureExtractor()
    #offline_classifier = 0
   

    #classififer_thing = OnlineEMGClassifier(offline_classifier, window_size, window_increment, online_data_handler, features, file_path='.', file=False, smm=False, smm_items=None, port=12346, ip='127.0.0.1', std_out=False, tcp=False, output_format='predictions')