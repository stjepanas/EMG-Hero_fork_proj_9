from enum import Enum
import numpy as np
from libemg.utils import get_windows

# [ 0 1   0 1   0 0    0]
#  thumb index middle rest

mapping = {
    0: {"movement": "INDEX_MIDDLE_EXTEND",         "one_hot_pred": np.array([0, 0, 1, 0, 1, 0, 0], dtype=np.float32)},
    1: {"movement": "INDEX_EXTEND",                "one_hot_pred": np.array([0, 0, 1, 0, 0, 0, 0], dtype=np.float32)},
    2: {"movement": "INDEX_MIDDLE_FLEX",           "one_hot_pred": np.array([0, 0, 0, 1, 0, 1, 0], dtype=np.float32)},
    3: {"movement": "INDEX_FLEX",                  "one_hot_pred": np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32)},
    4: {"movement": "MIDDLE_EXTEND",               "one_hot_pred": np.array([0, 0, 0, 0, 1, 0, 0], dtype=np.float32)},
    5: {"movement": "MIDDLE_FLEX",                 "one_hot_pred": np.array([0, 0, 0, 0, 0, 1, 0], dtype=np.float32)},
    6: {"movement": "REST",                        "one_hot_pred": np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)},
    7: {"movement": "THUMB_INDEX_MIDDLE_EXTEND",   "one_hot_pred": np.array([1, 0, 1, 0, 1, 0, 0], dtype=np.float32)},
    8: {"movement": "THUMB_INDEX_EXTEND",          "one_hot_pred": np.array([1, 0, 1, 0, 0, 0, 0], dtype=np.float32)},
    9: {"movement": "THUMB_EXTEND",                "one_hot_pred": np.array([1, 0, 0, 0, 0, 0, 0], dtype=np.float32)},
    10: {"movement": "THUMB_INDEX_MIDDLE_FLEX",     "one_hot_pred": np.array([0, 1, 0, 1, 0, 1, 0], dtype=np.float32)},
    11: {"movement": "THUMB_INDEX_FLEX",            "one_hot_pred": np.array([0, 1, 0, 1, 0, 0, 0], dtype=np.float32)},
    12: {"movement": "THUMB_FLEX",                  "one_hot_pred": np.array([0, 1, 0, 0, 0, 0, 0], dtype=np.float32)}
}


# Reverse Mapping
reverse_mapping = {
    value["one_hot_pred"].tobytes(): {
        "movement": value["movement"],
        "number": key
    }
    for key, value in mapping.items()
}


def get_bioarmband_data(online_data_handler, feature_extractor):

    # extract 300 data points from streamer (200ms windows => 300 sample windows at 1500hz sampling freq)
    data, count = online_data_handler.get_data(N=300)
    emg = data['emg']
    # Increments of windows is set to 75 samples => 50ms
    windows = get_windows(emg,300,75)
    features = feature_extractor.extract_features(feature_list = ['MAV', 'WL', 'ZC', 'SSC'], 
                                                  windows = windows,
                                                  )

    # print("data: ", data)
    # print("Features: ", features)

    mavs = features['MAV'].flatten()
    wls = features['WL'].flatten()
    zcs = features['ZC'].flatten()
    sscs = features['SSC'].flatten()

    # Stack the features in the correct order and flatten
    feat_data = np.ravel(np.column_stack((mavs, wls, zcs, sscs)))
    # print("mavs:",mavs)
    # print("wls:",wls)
    # print("zcs:",zcs)
    # print("sscs:",sscs)
    # print("---------------------")
    # print("feat_data", feat_data)

    mean_mav = np.mean(features['MAV'])

    return feat_data, mean_mav