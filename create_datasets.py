import numpy as np     
from gym_defs import reverse_mapping, mapping

def get_training_dataset(gestures: list, feature_list: list, size_windows_train, training_features, train_meta):
    ''' Returns the necessary components to build the MDPDataset for training
    Args: 
        gestures(list): list of gestures that we want to pretrain the model on
        feature_list(list): list of features that we want extracted
        size_windows_train(int): number of windows in the training data
        training_features: features from the feature list extracted from the raw data using libemgs 
        extract_features function
        train_meta: meta data for the training dataset from the parse_windows function    
    '''

    nr_windows = int(size_windows_train)
    nr_channels = 8 # this is always the same so it can be hard-coded
    
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
   # sets timeouts at the beginning of a new movement
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
        # maps the gesture from meta to the corresponding one hot pred vector
        action = mapping[gesture_train_class] 
        actions_train[i] = action['one_hot_pred']

    return observations_train,actions_train,rewards_train,terminals_train,pretrain_timeouts
    


def get_testing_dataset(gestures:list , feature_list:list, size_windows_test, testing_features, test_meta):
    ''' Returns the necessary components to build the MDPDataset for testing
        Args: 
            gestures(list): list of gestures that we want to pretrain the model on
            feature_list(list): list of features that we want extracted
            size_windows_test(int): number of windows in the testing data
            testing_features: features from the feature list extracted from the raw data using libemgs 
            extract_features function
            test_meta: meta data for the testing dataset from the parse_windows function    
    '''
    nr_windows = int(size_windows_test)
    nr_channels = 8
    
    observations_test = np.zeros((nr_windows, len(feature_list)*nr_channels)) 
    actions_test = np.zeros((nr_windows, 7))

    rewards_test = np.zeros(nr_windows)
    terminals_test = np.zeros((nr_windows))
   
    # checks for how long a movement is going on to set timeouts
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
        # maps the gesture from meta to the corresponding one hot pred vector
        action = mapping[gesture_test_class]
        actions_test[i] = action['one_hot_pred']

    return observations_test,actions_test,rewards_test,terminals_test,pretest_timeouts