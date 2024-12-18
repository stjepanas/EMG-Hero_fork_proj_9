from emg_hero.model_utility import ModelHandle, build_algo
from emg_hero.defs import MoveConfig
from emg_hero.configs import BaseConfig
from emg_hero.metrics import EMGHeroMetrics
from emg_hero.game import EMGHero
from emg_hero.label_transformer import LabelTransformer
import time
from gym_problem import mapping, reverse_mapping, get_bioarmband_data
from libemg.feature_extractor import FeatureExtractor
from libemg.data_handler import OnlineDataHandler
from libemg import streamers
from d3rlpy import load_learnable

if __name__ == "__main__":

    streamer, smm = streamers.sifi_bioarmband_streamer(
                                                    filtering= True,
                                                    emg_bandpass=[20,500],  # since lowpass = 20 and highpass = 500
                                                    emg_notch_freq=50,      # notch filter at 50hz
                                                    #bridge_version="1.1.3",
                                                    name="BioArmband",
                                                    ecg=False, emg=True, eda=False, imu=False, ppg=False)
            
    odh = OnlineDataHandler(smm)
    fe = FeatureExtractor()



    base_config = BaseConfig()
    move_config = MoveConfig()
    config = base_config.game

    model_path = r"d3rlpy_logs\BC_20241218170331\model_50000.d3"
    # model_path = r"d3rlpy_logs\BC_20241218132053\model_50000.d3"
    bc_model = load_learnable(model_path)

    model, floornoise = build_algo(pt_weights=None, pt_biases=None, base_config=base_config)

    label_transformer = LabelTransformer(move_config=move_config)

    emg_hero_dummy = EMGHero(canvas=None, song=None, experiment_folder='', config=config)
    emg_hero_metrics = EMGHeroMetrics(emg_hero = emg_hero_dummy,
                            label_transformer = label_transformer,
                            song_dataset = None,
                            history = None,
                            supervised_dataset = None,
                            action_size = move_config.n_actions)




    # initialize model handler
    model_handle = ModelHandle(model=bc_model,
                            model_path = model_path,
                            experiment_folder = "logs/test1",
                            play_with_emg = True,
                            n_actions=move_config.n_actions,
                            emg_hero_metrics=emg_hero_metrics,
                            label_transformer=label_transformer,
                            take_best_reward_model=base_config.algo.take_best_reward_model,
                            floornoise=floornoise,
                            n_features=config.n_feats)

    interval = 1

    while(True):
        start_time = time.time()
        feat_data, mean_mav = get_bioarmband_data(online_data_handler = odh, feature_extractor = fe)
        # print("channel 1:", feat_data[0:4])
        # print("channel 2:", feat_data[4:8])
        # print("channel 3:", feat_data[8:12])
        # print("channel 4:", feat_data[12:16])
        # print("channel 5:", feat_data[16:20])
        # print("channel 6:", feat_data[20:24])
        # print("channel 7:", feat_data[24:28])
        # print("channel 8:", feat_data[28:32])
        emg_keys, emg_one_hot_preds, _, new_features, too_high_values = model_handle.get_emg_keys(feat_data, mean_mav)

        key = emg_one_hot_preds.tobytes()

        if key in reverse_mapping.keys():
            print(reverse_mapping[key],f"              {emg_one_hot_preds}")
        else:
            print("INVALID MOVEMENT",f"              {emg_one_hot_preds}")

        elapsed_time = time.time() - start_time

        if elapsed_time < interval:
            time.sleep(interval - elapsed_time)
        else:
            # Log or handle overrun
            print(f"Loop overrun! Took {elapsed_time:.4f}s instead of {interval:.4f}s.")