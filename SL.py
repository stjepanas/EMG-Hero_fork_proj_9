from libemg.feature_extractor import FeatureExtractor
import numpy as np
from libemg.data_handler import OfflineDataHandler, RegexFilter
from emg_hero.metrics import F1MacroEvaluator
from d3rlpy import load_learnable
from create_datasets import get_training_dataset, get_testing_dataset
from d3rlpy.dataset import MDPDataset
from d3rlpy.constants import ActionSpace  
from emg_hero.model_utility import build_algo 
from emg_hero.configs import BaseConfig
from gym_defs import reverse_mapping, mapping
import os
import matplotlib.pyplot as plt
import csv
import matplotlib.pyplot as plt
from emg_hero.model_utility import get_newest_directory
from libemg import streamers
from libemg.gui import GUI
from libemg.data_handler import OnlineDataHandler 

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
    dataset_folder = 'data'
    # gestures picked out for training, this is according to the gesture_list.json/collection.json
    gestures = ["0","1","2","3","4","5","6","7","8","9","10","11","12"]
    reps = ["0","1","2","3","4","5","6"]
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
    train_data = offline_dh.isolate_data("reps", [0,1,2,3,4,5]) # more data for training than for testing
    test_data = offline_dh.isolate_data("reps", [6])

    # shapes data into windows x channels x nr of samples in a window, so nr_windows x 8 x 300
    # meta tells which movement is done for every window, gives a number from the gestures
    train_windows, train_meta = train_data.parse_windows(WINDOW_SIZE, WINDOW_INCREMENT)
    size_windows_train = np.shape(train_windows)[0]
    test_windows, test_meta = test_data.parse_windows(WINDOW_SIZE, WINDOW_INCREMENT) 
    size_windows_test = np.shape(test_windows)[0]

    fe = FeatureExtractor()
    feature_list = ['MAV', 'WL', 'ZC', 'SSC']
    # extracts the necessary features, returns a dictionary where keys are the features
    # this is done for each window, so each window gets a dictionary of features
    training_features = fe.extract_features(feature_list, train_windows)
    testing_features = fe.extract_features(feature_list, test_windows)

    # returns everything that goes into the MDPDataset function
    observations_train,actions_train,rewards_train,terminals_train,pretrain_timeouts = get_training_dataset(gestures, feature_list, size_windows_train, training_features, train_meta)
    observations_test,actions_test,rewards_test,terminals_test,pretest_timeouts = get_testing_dataset(gestures , feature_list, size_windows_test, testing_features, test_meta)

    # returns the MDPDatasets
    dataset_pretrain = MDPDataset(observations_train,actions_train,rewards_train,terminals_train,pretrain_timeouts,action_space = ActionSpace.CONTINUOUS)
    dataset_pretest = MDPDataset(observations_test,actions_test,rewards_test,terminals_test,pretest_timeouts,action_space = ActionSpace.CONTINUOUS)
    
    import pickle

    # save dataset_pretrain
    with open('dataset_pretrain.pkl', 'wb') as f:
        pickle.dump(dataset_pretrain, f)
    # save dataset_pretest
    with open('dataset_pretest.pkl', 'wb') as f:
        pickle.dump(dataset_pretest, f)

################ Behaviour cloning ################################################################
    #takes the previous project's hyperparameters for the model
    base_config = BaseConfig()
    model, floornoise = build_algo(pt_weights=None, pt_biases=None, base_config=base_config)

    f1_macro_evaluator = F1MacroEvaluator(dataset_pretest.episodes)
    f1_macro_evaluator_train = F1MacroEvaluator(dataset_pretrain.episodes)


    from d3rlpy.metrics.evaluators import ContinuousActionDiffEvaluator

    act_diff_train_evaluator = ContinuousActionDiffEvaluator(dataset_pretrain.episodes)
    act_diff_test_evaluator = ContinuousActionDiffEvaluator(dataset_pretest.episodes)
    
    bc_model = model.fit(
      dataset_pretrain,
      n_steps=50000,
      n_steps_per_epoch=500,
      evaluators={
          "f1_macro": f1_macro_evaluator,
          "f1_train": f1_macro_evaluator_train,
          "loss_train": act_diff_train_evaluator,
          "loss_test": act_diff_test_evaluator,
     }
 )

################### Testing and plotting the model results #################################    
    observation = observations_test
    gt_actions = actions_test
    actions = np.zeros((np.shape(observation)[0],7))
    model_map = get_newest_directory('d3rlpy_logs') # gets the latest generated map
    filenames = os.listdir(model_map)
    model_names = [f for f in filenames if f.startswith("model")]
    plot_stuff = np.zeros((len(model_names), 4))

    loss = np.zeros((len(model_names),1))
    with open(str(model_map)+'\loss.csv', mode ='r')as file: # gets the loss file from the newest map
        csvFile = csv.reader(file)
        for i,lines in enumerate(csvFile):
                loss[i] = lines[2]


# Loops through all the newly generated models, each model predicts the movements on the test dataset
# and the amount of correct, incorrect and invalid actions is recorded (loss as well) and plotted
    for idx, model in enumerate(model_names):
        bc_model = load_learnable(os.path.join(model_map, model))
        actions = bc_model.predict(observation)
        actions[actions < 0.5] = 0
        actions[actions >= 0.5] = 1
        correct = 0
        incorrect = 0
        total = 0
        invalid = 0
        for i in range(observation.shape[0]):
            total += 1
            if actions[i].tobytes() in reverse_mapping.keys():  # Check if valid action
                if np.array_equal(actions[i], gt_actions[i]):
                    correct += 1
                else:
                    incorrect += 1
            else:
                invalid += 1
        mod_loss = loss[idx]
        # Store results for this model as percentages
        plot_stuff[idx] = [correct/total, incorrect/total, invalid/total, float(mod_loss)]

    correct = plot_stuff[:, 0]
    incorrect = plot_stuff[:, 1]
    invalid = plot_stuff[:, 2]
    loss = plot_stuff[:, 3]
    # for model names
    model_numbers = [i + 1 for i in range(np.shape(plot_stuff)[0])] 

    # Create the figure and subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

    # Top plot with 3 signals
    ax1.plot(model_numbers, correct, marker='o', label='Correct', color='green')
    max_idx = np.argmax(correct)
    ax1.text(model_numbers[max_idx], correct[max_idx], f'Max: {correct[max_idx]:.2f} - Model{max_idx}',
         fontsize=12, color='green')


    ax1.plot(model_numbers, incorrect, marker='o', label='Incorrect', color='red')
    ax1.plot(model_numbers, invalid, marker='o', label='Invalid', color='orange')
    ax1.set_title('Prediction outcome')
    ax1.legend()
    ax1.grid()

    # Bottom plot with 1 signal
    ax2.plot(model_numbers, loss, label='Loss', color='blue')
    min_idx = np.argmin(loss)
    ax2.text(model_numbers[min_idx], loss[min_idx], f'Min: {loss[min_idx]:.2f} - Model{min_idx}',
         fontsize=12, color='blue')
    ax2.set_title('Training loss')
    ax2.legend()
    ax2.grid()

    # Label the axes
    ax2.set_xlabel('X-axis')
    ax1.set_ylabel('Amplitude')
    ax2.set_ylabel('Amplitude')

    # Adjust layout
    plt.tight_layout()
    plt.show()
                     

                 

            





