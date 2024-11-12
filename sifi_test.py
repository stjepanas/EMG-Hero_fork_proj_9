from libemg.gui import GUI
from libemg.data_handler import OnlineDataHandler
from libemg import streamers

if __name__ == "__main__":
    # Create data handler and streamer 
    streamer, smm = streamers.sifi_bioarmband_streamer(emg_notch_freq=60,
                                             bridge_version="1.1.3",
                                             name="BioArmband",
                                             ecg=False, emg=True, eda=False, imu=True, ppg=False)
    odh = OnlineDataHandler(smm)

    training_ui = GUI(odh, width=700, height=700, gesture_height=300, gesture_width=300)
    training_ui.download_gestures([1, 2, 3, 4, 5], "images/")
    training_ui.start_gui()
