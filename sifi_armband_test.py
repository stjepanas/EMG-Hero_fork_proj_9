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

if __name__ == "__main__":
    # Create data handler and streamer 
    streamer, smm = streamers.sifi_bioarmband_streamer(
                                            filtering= True,
                                            emg_bandpass=[20,500],
                                            emg_notch_freq=50,
                                             #bridge_version="1.1.3",
                                             name="BioArmband",
                                             ecg=False, emg=True, eda=False, imu=False, ppg=False)
    odh = OnlineDataHandler(smm)
    fe = FeatureExtractor()
    
    # fi = Filter(sampling_frequency=1500)

    # butter_filter   = { "name": "butterworth", "cutoff": 20}
    # notch_filter    = { "name": "notch", "cutoff": 50, "bandwidth":3, "order": 2}
    # low_pass_filter = { "name": "lowpass", "analog":True, "cutoff": 500, "order": 1}

    # fi.install_filters([butter_filter,notch_filter,low_pass_filter])
    # fi.install_filters(notch_filter)
    # fi.install_filters(low_pass_filter)

    # odh.install_filter(fi)

    odh.start_listening()   
    odh.visualize()



    # mavs = np.empty((8,0))
    # sscs = np.empty((8,0))
    # zcs = np.empty((8,0))
    # wls = np.empty((8,0))
    # i = 0
    # max_iter = 100
    # for iteration in range(max_iter):
    #     time.sleep(1)
    #     data, count = odh.get_data(N=300)
    #     emg = data['emg']
        # print("data: ", emg)
    #     # print("data type: ", type(emg))
    #     # print("data shape: ", np.shape(emg))
        # windows = get_windows(emg,300,75)
    #     # print('windows: ', windows)
    #     # print("windows type: ", type(windows))
    #     # print("windows shape: ", np.shape(windows))
        # features = fe.extract_features(feature_list = ['MAV', 'SSC', 'ZC', 'WL'],
                                        # windows = windows)

        # print(features['MAV'])
    #     print(np.shape(features['MAV'].T))
    #     print(np.shape(features['MAV']))
        # features = {'MAV': np.random.rand(1, 8), 
        #             'SSC': np.random.rand(1, 8), 
        #             'ZC': np.random.rand(1, 8) ,
        #             'WL': np.random.rand(1, 8)}
        # mavs = np.hstack((mavs,features['MAV'].T))
        # sscs = np.hstack((sscs,features['SSC'].T))
        # zcs = np.hstack((zcs,features['ZC'].T))
        # wls = np.hstack((wls,features['WL'].T))

    # Create a 2x2 grid of subplots
    # time_points = np.linspace(0, 60, max_iter)

    # fig,axs = plt.subplots(2,2)
    # axs[0, 0].plot(time_points, mavs.T)
    # axs[0, 0].set_title('MAV')
    # axs[0, 1].plot(time_points, sscs.T)
    # axs[0, 1].set_title('SSC')
    # axs[1, 0].plot(time_points, zcs.T)
    # axs[1, 0].set_title('ZC')
    # axs[1, 1].plot(time_points, wls.T)
    # axs[1, 1].set_title('WLS')
    # plt.show()