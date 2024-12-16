import numpy as np   
#from get_action_vector import get_action_vector   
from gym_problem import reverse_mapping, mapping

def get_training_dataset(gestures: list, feature_list: list, size_windows_train, training_features, train_meta):

    nr_windows = int(size_windows_train)
    nr_channels = 8 
    
    observations_train = np.zeros((nr_windows, len(feature_list)*nr_channels)) 
    actions_train = np.zeros((nr_windows, 7))
    rewards_train = np.zeros(nr_windows)
    terminals_train = np.zeros((nr_windows))
   
   # checks for how long a movement is going on to set timeouts
    counter = 0
    for i in train_meta['classes']:
        if i == int(gestures[0]):
            counter += 1
        else:
            break

    pretrain_timeouts = np.array([1 if ((i+1) % (counter) == 0 and i > 0) else 0 \
                                     for i in range(actions_train.shape[0])]) 

    # Build dataset
    for i in range(nr_windows): 
        feature_train_vector = []
        for feature in feature_list:
            # Concatenate features from all channels
            feature_train_vector.extend(training_features[feature][i])
  
        observations_train[i] = feature_train_vector

        gesture_train_class = train_meta['classes'][i] 
        action = mapping[gesture_train_class] #get_action_vector(gesture_train_class)
        actions_train[i] = action['one_hot_pred']

        #print('actions train:', actions_train[i], 'action meta:', train_meta['classes'][i] )

    return observations_train,actions_train,rewards_train,terminals_train,pretrain_timeouts
    


def get_testing_dataset(gestures:list , feature_list:list, size_windows_test, testing_features, test_meta):

    nr_windows = int(size_windows_test)
    nr_channels = 8
    
    observations_test = np.zeros((nr_windows, len(feature_list)*nr_channels)) 
    actions_test = np.zeros((nr_windows, 7))

    rewards_test = np.zeros(nr_windows)
    terminals_test = np.zeros((nr_windows))
   
   # checks how many zeros there are in the beginning to check for sample size for
    counter = 0
    for i in test_meta['classes']:
        if i == int(gestures[0]):
            counter += 1
        else:
            break

    pretest_timeouts = np.array([1 if ((i+1) % (counter) == 0 and i > 0) else 0 \
                                     for i in range(actions_test.shape[0])]) 

    # Build dataset
    for i in range(nr_windows): 
    # Extract features for the current window
        feature_test_vector = []
        for feature in feature_list:
            # Concatenate features from all channels
            feature_test_vector.extend(testing_features[feature][i])

        observations_test[i] = feature_test_vector

        gesture_test_class = test_meta['classes'][i] 
        action = mapping[gesture_test_class]#get_action_vector(gesture_test_class)
        actions_test[i] = action['one_hot_pred']
        #print('actions test:', actions_test[i], 'action meta:', test_meta['classes'][i] )

    return observations_test,actions_test,rewards_test,terminals_test,pretest_timeouts