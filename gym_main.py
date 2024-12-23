import os
import time
import copy
import pickle
import logging
import argparse
import datetime
from pathlib import Path
import numpy as np
from d3rlpy import load_learnable
from torch.nn import Sigmoid
import pygame

from emg_hero.model_utility import ModelHandle, build_algo
from emg_hero.datasets import get_history_filenames
from emg_hero.metrics import EMGHeroMetrics
from emg_hero.label_transformer import LabelTransformer
from emg_hero.defs import get_current_timestamp, SUPERVISED_DATA_FILENAME, MoveConfig
from emg_hero.generate_song import generate_song
from emg_hero.analyze_utils import create_csv
from emg_hero.game import EMGHero
from emg_hero.configs import BaseConfig

import gymnasium as gym
from gymnasium.envs.registration import register
import gymnasium.spaces as spaces

from gym_defs import reverse_mapping, mapping, get_bioarmband_data

from libemg import streamers
from libemg.feature_extractor import FeatureExtractor
from libemg.data_handler import OnlineDataHandler

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                prog='EMGHero',
                description='EMG Hero is a game to train EMG control')

    parser.add_argument('--experiment_folder',
                        type=str,
                        required=True,
                        default='./logs/emg_hero_test/',
                        help='Name of current experiment folder. \
                            Must contain pretrained model, config and supervised data')
    parser.add_argument('--no_emg',
                        action='store_true',
                        help='Flag to play with keyboard instead of EMG')
    parser.add_argument('-m', '--model',
                        default=None,
                        help='Specify model name if not pretrain')
    parser.add_argument('--matlab',
                        action='store_true',
                        help='Flag to load pretrained model from matlab')
    parser.add_argument('--reload',
                    action='store_true',
                    help='Flag to reload existing files in experiment folder')

    args = parser.parse_args()


    base_config = BaseConfig()
    config = base_config.game

    SCRIPT_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
    EXPERIMENT_FOLDER = SCRIPT_DIR / args.experiment_folder
    EXPERIMENT_FOLDER.mkdir(parents=True, exist_ok=True)
    SUPERVISED_PATH = EXPERIMENT_FOLDER / SUPERVISED_DATA_FILENAME
    PLAY_WITH_EMG = not args.no_emg
    FULL_SOUNDFILE = SCRIPT_DIR / 'song_files' / 'default_song.wav'
    CROPPED_SOUNDFILE = EXPERIMENT_FOLDER / 'song_audio.wav'
    floornoise = None
    N_ROUND = 1

    HISTORY_FILENAMES = []

    if args.matlab:
        logging.info('Loading matlab weights and biases')
        MAT_NETWORK_PARAMS_FILENAME = EXPERIMENT_FOLDER / 'pretrained_network_params.mat'
    else:
        MAT_NETWORK_PARAMS_FILENAME = None

    if args.reload:
        logging.info('Reloading model and histories')  
        # load latest network
        final_model_path = EXPERIMENT_FOLDER / "final_rl_model.d3"
        assert final_model_path.exists(), 'Trained model not found'

        model = load_learnable(final_model_path)
        MODEL_PATH = final_model_path # this is not ideal as it will get overwritten

        # # load histories
        HISTORY_FILENAMES = get_history_filenames(EXPERIMENT_FOLDER)
        N_ROUND = len(HISTORY_FILENAMES)


    move_config = MoveConfig()

    model, floornoise = build_algo(pt_weights=None, pt_biases=None, base_config=base_config)

    MODEL_PATH = EXPERIMENT_FOLDER / 'pretrained_model.d3'
    model.save(MODEL_PATH)

    if args.model is not None:
        MODEL_PATH = SCRIPT_DIR / args.model
        if args.matlab:
            logging.warning('matlab model is overwritten by provided model with --model')
            print(MODEL_PATH)
        model = load_learnable(MODEL_PATH)

    assert isinstance(model.impl.policy._final_act, Sigmoid), 'Last layer of model should be Sigmoid'

    logging.info('Using model %s', MODEL_PATH.as_posix())

    # load song
    # TODO move config
    time_between_notes = .5
    n_single_notes_per_class = 0
    n_repetitions_per_class = 1
    note_lenghts = [0.5, 1.0, 1.5, 2.0]

    SONG, NOTES_FILENAME = generate_song(FULL_SOUNDFILE, EXPERIMENT_FOLDER, move_config,
                                        n_single_notes_per_class, n_repetitions_per_class,
                                        note_lenghts, time_between_notes=time_between_notes)

    pygame.init()
    pygame.font.init()

    # clock = pygame.time.Clock()
    game_canvas = pygame.display.set_mode((config.window_width, config.window_height))
    pygame.display.set_caption("EMG Hero")

    emg_hero = EMGHero(canvas=game_canvas,
                    song=SONG,
                    experiment_folder=EXPERIMENT_FOLDER,
                    notes_filename=NOTES_FILENAME,
                    cropped_soundfile=CROPPED_SOUNDFILE,
                    config=config,
                    move_config=move_config)
    emg_hero.reset_canvas()

    label_transformer = LabelTransformer(move_config=move_config)

    # load metrics
    emg_hero_dummy = EMGHero(canvas=None, song=None, experiment_folder='', config=config)
    emg_hero_metrics = EMGHeroMetrics(emg_hero = emg_hero_dummy,
                                label_transformer = label_transformer,
                                song_dataset = None,
                                history = None,
                                supervised_dataset = None,
                                action_size = move_config.n_actions)
    
    if PLAY_WITH_EMG:
        # Libemg streamer for the bio armband
        streamer, smm = streamers.sifi_bioarmband_streamer(
                                                filtering= True,
                                                emg_bandpass=[20,500],  # since lowpass = 20 and highpass = 500
                                                emg_notch_freq=50,      # notch filter at 50hz
                                                #bridge_version="1.1.3",
                                                name="BioArmband",
                                                ecg=False, emg=True, eda=False, imu=False, ppg=False)
        
        odh = OnlineDataHandler(smm)
        fe = FeatureExtractor()


    # initialize model handler
    model_handle = ModelHandle(model=model,
                            model_path = MODEL_PATH,
                            experiment_folder = EXPERIMENT_FOLDER,
                            play_with_emg = PLAY_WITH_EMG,
                            n_actions=move_config.n_actions,
                            emg_hero_metrics=emg_hero_metrics,
                            label_transformer=label_transformer,
                            take_best_reward_model=base_config.algo.take_best_reward_model,
                            floornoise=floornoise,
                            n_features=config.n_feats)


    ################## TESTING ######################

    register(
        id = 'EMGHero-v0',
        entry_point = 'gym_env:EMGHeroEnv'
    )

    env = gym.make("EMGHero-v0", 
                history_filenames = HISTORY_FILENAMES,
                emg_hero = emg_hero,
                model_handle = model_handle,
                play_with_emg = PLAY_WITH_EMG,
                n_round = N_ROUND,
                experiment_folder = EXPERIMENT_FOLDER)

    observation,_= env.reset()

    interval = 0.05
    episode_over = False
    while not episode_over:
        start_time = time.time()
        
        if PLAY_WITH_EMG:
            # Extract the stacked feature vector used by the model as well as the mean_mav value if applicable
            feat_data, mean_mav = get_bioarmband_data(odh,fe)

            # env.pressed_keys, one_hot_preds, _, env.new_features, env.too_high_values = model_handle.get_emg_keys(feat_data,mean_mav)
            action = model_handle.get_emg_keys(feat_data,mean_mav)
            emg_hero.observation = feat_data
            input = action[1]
            # print("pressed keys main:", input)
            # if input.tobytes() in reverse_mapping.keys():
            #     print("action:", reverse_mapping[input.tobytes()]['movement'])
            # else:
            #     print("action:", input, "movement:","invalid movement")
        else:
            input = np.zeros((7,)) #dummy value, will be ignored in step()

        observation, reward, terminated, truncated, info = env.step(action=action)

        # if restarted reset the environment

        if not terminated:
            env.render()

        episode_over = terminated or truncated

        
        elapsed_time = time.time() - start_time

        if elapsed_time < interval:
            time.sleep(interval - elapsed_time)

    # -----------------------------------------
    # Game over

    print(info.keys())
    
    current_history_filenames = info["current_history_filenames"]

    # check if game score and history score adds up
    if np.sum(emg_hero.history['rewards']) != emg_hero.score:
        logging.warning("History score different from game score")

    logging.info('Reward sum: %f', emg_hero.score)

    NOW = get_current_timestamp()
    histories_save_path = EXPERIMENT_FOLDER / ('emg_hero_history_filenames_'+NOW+'.pkl')
    with open(histories_save_path, 'wb') as _file:
        pickle.dump(current_history_filenames, _file)
        logging.info('History filenames successfully saved to %s', histories_save_path)

    # calculate and save metrics
    history_summary = {
        'HISTORY_FILENAMES': current_history_filenames,
        'motion_test_files': {},
        'subject_id': None,
        'subject_date': datetime.date.today(),
        'subject_age': None,
        'experiment_folder': EXPERIMENT_FOLDER,
        'switch_lines': False,
    }
    df = create_csv(history_summary)
    short_df = df[['experiment_folder', 'rewards', 'emr', 'f1', 'num_action_changes']]

    results_file = EXPERIMENT_FOLDER / 'results.csv'
    short_results_file = EXPERIMENT_FOLDER / 'short_results.csv'

    # append in case exists, otherwise create new file
    if os.path.exists(results_file):
        df.to_csv(results_file, mode='a', index=False, header=False)
        short_df.to_csv(short_results_file, mode='a', index=False, header=False)
    else:
        df.to_csv(results_file)
        short_df.to_csv(short_results_file)

    # destruct game class
    del model_handle
    del emg_hero

    env.close()