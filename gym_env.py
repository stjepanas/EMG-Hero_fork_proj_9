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

import gym_problem

class EMGHeroEnv(gym.Env):

    def __init__(self):
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

        self.base_config = BaseConfig()
        self.config = self.base_config.game

        SCRIPT_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
        self.EXPERIMENT_FOLDER = SCRIPT_DIR / args.experiment_folder
        self.EXPERIMENT_FOLDER.mkdir(parents=True, exist_ok=True)
        self.SUPERVISED_PATH = self.EXPERIMENT_FOLDER / SUPERVISED_DATA_FILENAME
        self.PLAY_WITH_EMG = not args.no_emg
        FULL_SOUNDFILE = SCRIPT_DIR / 'song_files' / 'default_song.wav'
        CROPPED_SOUNDFILE = self.EXPERIMENT_FOLDER / 'song_audio.wav'
        floornoise = None
        self.N_ROUND = 1

        HISTORY_FILENAMES = []

        if args.matlab:
            logging.info('Loading matlab weights and biases')
            MAT_NETWORK_PARAMS_FILENAME = self.EXPERIMENT_FOLDER / 'pretrained_network_params.mat'
        else:
            MAT_NETWORK_PARAMS_FILENAME = None

        # episodes_list, pt_weights, pt_biases, move_config = load_data(data_path)
        self.move_config = MoveConfig()

        model, floornoise = build_algo(pt_weights=None, pt_biases=None, base_config=self.base_config)
        
        MODEL_PATH = self.EXPERIMENT_FOLDER / 'pretrained_model.d3'
        model.save(MODEL_PATH)

        if args.reload:
            logging.info('Reloading model and histories')  
            # load latest network
            final_model_path = self.EXPERIMENT_FOLDER / "final_rl_model.d3"
            assert final_model_path.exists(), 'Trained model not found'

            model = load_learnable(final_model_path)
            MODEL_PATH = final_model_path # this is not ideal as it will get overwritten

            # # load histories
            HISTORY_FILENAMES = get_history_filenames(self.EXPERIMENT_FOLDER)
            self.N_ROUND = len(HISTORY_FILENAMES)


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

        SONG, NOTES_FILENAME = generate_song(FULL_SOUNDFILE, self.EXPERIMENT_FOLDER, self.move_config,
                                            n_single_notes_per_class, n_repetitions_per_class,
                                            note_lenghts, time_between_notes=time_between_notes)
        pygame.init()
        pygame.font.init()

        self.clock = pygame.time.Clock()
        game_canvas = pygame.display.set_mode((self.config.window_width, self.config.window_height))
        pygame.display.set_caption("EMG Hero")

        self.EXIT = False
        self.GAME_DONE = False
        self.GAME_RESTART = False

        # initialize emg hero class
        self.emg_hero = EMGHero(canvas=game_canvas,
                        song=SONG,
                        experiment_folder=self.EXPERIMENT_FOLDER,
                        notes_filename=NOTES_FILENAME,
                        cropped_soundfile=CROPPED_SOUNDFILE,
                        config=self.config,
                        move_config=self.move_config)
        self.emg_hero.reset_canvas()

        self.label_transformer = LabelTransformer(move_config=self.move_config)

        # load metrics
        emg_hero_dummy = EMGHero(canvas=None, song=None, experiment_folder='', config=self.config)
        emg_hero_metrics = EMGHeroMetrics(emg_hero = emg_hero_dummy,
                                    label_transformer = self.label_transformer,
                                    song_dataset = None,
                                    history = None,
                                    supervised_dataset = None,
                                    action_size = self.move_config.n_actions)
        # initialize model handler
        self.model_handle = ModelHandle(model=model,
                                model_path = MODEL_PATH,
                                experiment_folder = self.EXPERIMENT_FOLDER,
                                play_with_emg = self.PLAY_WITH_EMG,
                                tcp_host=self.config.host,
                                tcp_port=self.config.port,
                                n_actions=self.move_config.n_actions,
                                emg_hero_metrics=emg_hero_metrics,
                                label_transformer=self.label_transformer,
                                take_best_reward_model=self.base_config.algo.take_best_reward_model,
                                floornoise=floornoise,
                                n_features=self.config.n_feats)

        # start timer and play sound
        self.game_start_time = time.time() + self.config.time_before_start
        self.last_pos_update_time = time.time()
        self.LAST_NOTE_TIME = None
        self.SONG_STARTED = False
        self.GAME_STARTED = False
        self.SUCCESS = False
        self.current_history_filenames = copy.deepcopy(HISTORY_FILENAMES)

        # Gym render
        self.render_mode = None

        # Gym initialize action space
        self.action_space = spaces.Discrete(len(gym_problem.emgHeroActions))

        # Gym initialize observation space
        self.observation_space = spaces.Discrete(6)
        self.observation = [0,0,0,0,0,0]

        

    def step(self, predicted_action):
        self.clock.tick(self.config.fps)

        if self.GAME_RESTART:
            # ask if retraining
            RESTARTING = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.EXIT = True
                if event.type == pygame.KEYDOWN:
                    if event.unicode == 'q' or event.unicode == 'Q':
                        self.EXIT = True
                    if event.unicode == 'y' or event.unicode == 'Y':
                        # retrain model
                        logging.info('Retraining model')
                        # only take last n histories to retrain
                        if self.base_config.algo.n_histories_replay_buffer is None:
                            hists_in_buffer = self.current_history_filenames
                        else:
                            hists_in_buffer = self.current_history_filenames[-self.base_config.algo.n_histories_replay_buffer:]

                        self.model_handle.retrain_model(hists_in_buffer,
                                                supervised_filename = self.SUPERVISED_PATH,
                                                experiment_folder = self.EXPERIMENT_FOLDER,
                                                move_config=self.move_config,
                                                only_use_last_history = self.base_config.algo.only_use_last_history,
                                                wrong_note_randomization = self.base_config.algo.wrong_note_randomization,
                                                n_steps = self.base_config.algo.n_steps)
                        RESTARTING = True
                    if event.unicode == 'n' or event.unicode == 'N':
                        # don't retrain model
                        RESTARTING = True

            if RESTARTING:
                logging.info('Restarting game')
                self.GAME_RESTART = False
                self.GAME_STARTED = False
                self.game_start_time = time.time() + self.config.time_before_start
                self.last_pos_update_time = time.time()
                self.LAST_NOTE_TIME = None
                self.GAME_TIME = -100
                self.GAME_DONE = False
                self.SONG_STARTED = False
                self.N_ROUND += 1
                self.emg_hero.reset()

        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                self.EXIT = True
            if event.type == pygame.KEYDOWN:
                if event.unicode == 'q' or event.unicode == 'Q':
                    HIST_FILENAME = self.emg_hero.save_history()
                    self.current_history_filenames.append(HIST_FILENAME)
                    self.EXIT = True
                if event.unicode == 'r' or event.unicode == 'R':
                    # restart game
                    HIST_FILENAME = self.emg_hero.save_history()
                    self.current_history_filenames.append(HIST_FILENAME)
                    self.GAME_RESTART = True
                if event.unicode == 'm' or event.unicode == 'M':
                    self.emg_hero.use_right_arm = not self.emg_hero.use_right_arm
                if event.unicode == 'h' or event.unicode == 'H':
                    self.emg_hero.help_mode = not self.emg_hero.help_mode
                if event.key == pygame.K_RETURN:
                    if not self.GAME_STARTED:
                        self.GAME_STARTED = True
                        self.game_start_time = time.time() + self.config.time_before_start
                        self.last_pos_update_time = time.time()



        self.start_time = time.time()
        self.GAME_TIME = time.time() - self.game_start_time

        if self.GAME_TIME >= 0 and not self.config.silent_mode and not self.SONG_STARTED:
            self.emg_hero.play_song()
            self.SONG_STARTED = True

        # get pressed keys
        if self.PLAY_WITH_EMG:
            self.pressed_keys, one_hot_preds, features, new_features, self.too_high_values = self.model_handle.get_emg_keys()
        else:
            new_features = True
            self.too_high_values = False
            features = np.NaN
            keys = pygame.key.get_pressed()
            self.pressed_keys = {
                'lines': [],
                'directions': [],
                }

            for key_idx in range(min((self.move_config.n_dof * 2, 9))):
                key_pos = getattr(pygame, ('K_' + str(key_idx + 1)))
                if keys[key_pos]:
                    self.pressed_keys['lines'].append(int(key_idx/2))
                    self.pressed_keys['directions'].append('up' if key_idx % 2 == 0  else 'down')

            one_hot_preds = self.label_transformer.keys_to_onehot(self.pressed_keys)

        # flip lines if direction switched
        if self.emg_hero.use_right_arm:
            self.pressed_keys['lines'] = [((self.move_config.n_dof-1) - old_line) for old_line in self.pressed_keys['lines']]

        if new_features:
        # check if pressed keys hit note
            self.SUCCESS, new_score = self.emg_hero.check_note_hit(self.pressed_keys)
            self.emg_hero.append_history(self.GAME_TIME, new_score, one_hot_preds, self.pressed_keys, features)


        # set time when no notes are present anymore
        if len(self.emg_hero.song) == 0 and self.LAST_NOTE_TIME is None:
            self.LAST_NOTE_TIME = time.time()

        # end game 1 second after last note disappeared
        TIME_AFTER_LAST_NOTE = (self.config.key_y / self.config.speed) + self.config.time_before_start + 1
        if self.LAST_NOTE_TIME is not None and time.time() - self.LAST_NOTE_TIME > TIME_AFTER_LAST_NOTE:
            self.GAME_DONE = True
            logging.info('Reward sum: %f', self.emg_hero.score)

        
        
        self.terminated = self.EXIT
        self.observation, self.reward, self.truncated, self.info = [0,0,0,0,0,0],0,0,{}

        return self.observation, self.reward, self.terminated, self.truncated, self.info
    
    def reset(self, seed = None, options = None):
        super().reset(seed=seed)
        self.observation = [1,2,3,4,5,6]
        return self.observation

    def render(self):
        self.emg_hero.reset_canvas()

        if not self.GAME_STARTED:
            self.emg_hero.draw_start_canvas(self.N_ROUND)
            pygame.display.update()
            time.sleep(0.05)
            return

        # if game done, show results screen
        if self.GAME_DONE:
            self.emg_hero.draw_end_canvas(self.N_ROUND)
            pygame.display.update()
            time.sleep(0.05)
            return
        
        if self.GAME_RESTART:
            self.emg_hero.draw_restarting_canvas()
            pygame.display.update()
            time.sleep(0.05)
            return


        # do step
        self.emg_hero.reset_canvas()
        self.emg_hero.update_notes(self.GAME_TIME)
        self.emg_hero.update_positions(self.last_pos_update_time)
        self.last_pos_update_time = time.time()
        self.emg_hero.draw()
        self.emg_hero.draw_score()


        self.emg_hero.visualize_too_high_values(self.too_high_values)
        self.emg_hero.visualize_button_press(self.pressed_keys, is_success=self.SUCCESS)

        # show fps
        if self.config.show_fps:
            self.emg_hero.draw_fps(self.start_time, self.GAME_TIME)

        pygame.display.update()
        return

    def close(self):
        pass

register(
    id = 'EMGHero-v0',
    entry_point = 'gym_env:EMGHeroEnv'
)

################# TESTING #################


env = gym.make('EMGHero-v0')
# print(gym.spec('EMGHero-v0'))

observation= env.reset()

episode_over = False
while not episode_over:
    #action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(1)
    if not terminated:
        env.render()
    episode_over = terminated or truncated

env.close()