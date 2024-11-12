''' Main executable to run EMG Hero

EMG Hero is a game that is designed to be played with EMG signals.
The score can be used to fine-tune an existing classifier via
Reinforcement Learning. This script can be run as demo when running
with --no_emg, for the scenario with EMG the corresponding MatLAB
script is needed to obtain EMG features.
'''
import os
import time
import copy
import pickle
import logging
import argparse
import datetime
from pathlib import Path

import pygame
import numpy as np
from d3rlpy import load_learnable
from torch.nn import Sigmoid

from emg_hero.model_utility import ModelHandle, build_algo
from emg_hero.datasets import get_history_filenames
from emg_hero.metrics import EMGHeroMetrics
from emg_hero.label_transformer import LabelTransformer
from emg_hero.defs import get_current_timestamp, SUPERVISED_DATA_FILENAME, MoveConfig
from emg_hero.generate_song import generate_song
from emg_hero.analyze_utils import create_csv
from emg_hero.game import EMGHero
from emg_hero.configs import BaseConfig


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%y-%m-%d %H:%M:%S')


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

    # episodes_list, pt_weights, pt_biases, move_config = load_data(data_path)
    move_config = MoveConfig()

    model, floornoise = build_algo(pt_weights=None, pt_biases=None, base_config=base_config)
    
    MODEL_PATH = EXPERIMENT_FOLDER / 'pretrained_model.d3'
    model.save(MODEL_PATH)

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


    if args.model is not None:
        MODEL_PATH = SCRIPT_DIR / args.model
        if args.matlab:
            logging.warning('matlab model is overwritten by provided model with --model')
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

    clock = pygame.time.Clock()
    game_canvas = pygame.display.set_mode((config.window_width, config.window_height))
    pygame.display.set_caption("EMG Hero")

    EXIT = False
    GAME_DONE = False
    GAME_RESTART = False

    # initialize emg hero class
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
    # initialize model handler
    model_handle = ModelHandle(model=model,
                               model_path = MODEL_PATH,
                               experiment_folder = EXPERIMENT_FOLDER,
                               play_with_emg = PLAY_WITH_EMG,
                               tcp_host=config.host,
                               tcp_port=config.port,
                               n_actions=move_config.n_actions,
                               emg_hero_metrics=emg_hero_metrics,
                               label_transformer=label_transformer,
                               take_best_reward_model=base_config.algo.take_best_reward_model,
                               floornoise=floornoise,
                               n_features=config.n_feats)

    # start timer and play sound
    game_start_time = time.time() + config.time_before_start
    last_pos_update_time = time.time()
    LAST_NOTE_TIME = None
    SONG_STARTED = False
    GAME_STARTED = False
    SUCCESS = False
    current_history_filenames = copy.deepcopy(HISTORY_FILENAMES)

    while not EXIT:
        clock.tick(config.fps)

        if GAME_RESTART:
            # ask if retraining
            RESTARTING = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    EXIT = True
                if event.type == pygame.KEYDOWN:
                    if event.unicode == 'q' or event.unicode == 'Q':
                        EXIT = True
                    if event.unicode == 'y' or event.unicode == 'Y':
                        # retrain model
                        logging.info('Retraining model')
                        # only take last n histories to retrain
                        if base_config.algo.n_histories_replay_buffer is None:
                            hists_in_buffer = current_history_filenames
                        else:
                            hists_in_buffer = current_history_filenames[-base_config.algo.n_histories_replay_buffer:]

                        model_handle.retrain_model(hists_in_buffer,
                                                supervised_filename = SUPERVISED_PATH,
                                                experiment_folder = EXPERIMENT_FOLDER,
                                                move_config=move_config,
                                                only_use_last_history = base_config.algo.only_use_last_history,
                                                wrong_note_randomization = base_config.algo.wrong_note_randomization,
                                                n_steps = base_config.algo.n_steps)
                        RESTARTING = True
                    if event.unicode == 'n' or event.unicode == 'N':
                        # don't retrain model
                        RESTARTING = True

            if RESTARTING:
                logging.info('Restarting game')
                GAME_RESTART = False
                GAME_STARTED = False
                game_start_time = time.time() + config.time_before_start
                last_pos_update_time = time.time()
                LAST_NOTE_TIME = None
                GAME_TIME = -100
                GAME_DONE = False
                SONG_STARTED = False
                N_ROUND += 1
                emg_hero.reset()

            emg_hero.draw_restarting_canvas()
            pygame.display.update()
            time.sleep(0.05)
            continue

        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                EXIT = True
            if event.type == pygame.KEYDOWN:
                if event.unicode == 'q' or event.unicode == 'Q':
                    HIST_FILENAME = emg_hero.save_history()
                    current_history_filenames.append(HIST_FILENAME)
                    EXIT = True
                if event.unicode == 'r' or event.unicode == 'R':
                    # restart game
                    HIST_FILENAME = emg_hero.save_history()
                    current_history_filenames.append(HIST_FILENAME)
                    GAME_RESTART = True
                if event.unicode == 'm' or event.unicode == 'M':
                    emg_hero.use_right_arm = not emg_hero.use_right_arm
                if event.unicode == 'h' or event.unicode == 'H':
                    emg_hero.help_mode = not emg_hero.help_mode
                if event.key == pygame.K_RETURN:
                    if not GAME_STARTED:
                        GAME_STARTED = True
                        game_start_time = time.time() + config.time_before_start
                        last_pos_update_time = time.time()

        if not GAME_STARTED:
            emg_hero.draw_start_canvas(N_ROUND)
            pygame.display.update()
            time.sleep(0.05)
            continue

        start_time = time.time()
        GAME_TIME = time.time() - game_start_time

        # if game done, show results screen
        if GAME_DONE:
            emg_hero.draw_end_canvas(N_ROUND)
            pygame.display.update()
            time.sleep(0.05)
            continue

        if GAME_TIME >= 0 and not config.silent_mode and not SONG_STARTED:
            emg_hero.play_song()
            SONG_STARTED = True

        # do step
        emg_hero.reset_canvas()
        emg_hero.update_notes(GAME_TIME)
        emg_hero.update_positions(last_pos_update_time)
        last_pos_update_time = time.time()
        emg_hero.draw()
        emg_hero.draw_score()

        # get pressed keys
        if PLAY_WITH_EMG:
            pressed_keys, one_hot_preds, features, new_features, too_high_values = model_handle.get_emg_keys()
        else:
            new_features = True
            too_high_values = False
            features = np.NaN
            keys = pygame.key.get_pressed()
            pressed_keys = {
                'lines': [],
                'directions': [],
                }

            for key_idx in range(min((move_config.n_dof * 2, 9))):
                key_pos = getattr(pygame, ('K_' + str(key_idx + 1)))
                if keys[key_pos]:
                    pressed_keys['lines'].append(int(key_idx/2))
                    pressed_keys['directions'].append('up' if key_idx % 2 == 0  else 'down')

            one_hot_preds = label_transformer.keys_to_onehot(pressed_keys)

        # flip lines if direction switched
        if emg_hero.use_right_arm:
            pressed_keys['lines'] = [((move_config.n_dof-1) - old_line) for old_line in pressed_keys['lines']]

        if new_features:
        # check if pressed keys hit note
            SUCCESS, new_score = emg_hero.check_note_hit(pressed_keys)
            emg_hero.append_history(GAME_TIME, new_score, one_hot_preds, pressed_keys, features)


        emg_hero.visualize_too_high_values(too_high_values)

        emg_hero.visualize_button_press(pressed_keys, is_success=SUCCESS)

        # set time when no notes are present anymore
        if len(emg_hero.song) == 0 and LAST_NOTE_TIME is None:
            LAST_NOTE_TIME = time.time()

        # end game 1 second after last note disappeared
        TIME_AFTER_LAST_NOTE = (config.key_y / config.speed) + config.time_before_start + 1
        if LAST_NOTE_TIME is not None and time.time() - LAST_NOTE_TIME > TIME_AFTER_LAST_NOTE:
            GAME_DONE = True
            logging.info('Reward sum: %f', emg_hero.score)

        # show fps
        if config.show_fps:
            emg_hero.draw_fps(start_time, GAME_TIME)

        pygame.display.update()

        # end of game while loop

    # -----------------------------------
    # Game done

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
