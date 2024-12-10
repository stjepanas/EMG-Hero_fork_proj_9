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

    def __init__(self, history_filenames, emg_hero, model_handle, play_with_emg, n_round):
        self.HISTORY_FILENAMES = history_filenames
        self.emg_hero = emg_hero  
        self.model_handle = model_handle
        self.PLAY_WITH_EMG = play_with_emg
        self.N_ROUND = n_round
        self.last_data_extract = time.time()

        self.base_config = BaseConfig()
        self.config = self.base_config.game
        self.move_config = MoveConfig()

        self.clock = pygame.time.Clock()

        self.EXIT = False
        self.GAME_DONE = False
        self.GAME_RESTART = False

        self.label_transformer = LabelTransformer(move_config=self.move_config)

        # start timer and play sound
        self.game_start_time = time.time() + self.config.time_before_start
        self.last_pos_update_time = time.time()
        self.LAST_NOTE_TIME = None
        self.SONG_STARTED = False
        self.GAME_STARTED = False
        self.SUCCESS = False
        self.current_history_filenames = copy.deepcopy(self.HISTORY_FILENAMES)

        # Gym initialize action space
        # 7 float values that will be passed through a sigmoid and threshold 
        self.action_space = spaces.Discrete(12)

        # Gym initialize observation space
        # 4 EMG features [MAW, WL, ZC, SSC]
        self.observation_space = spaces.Discrete(4)

        

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

        if self.GAME_STARTED:
            self.start_time = time.time()
            self.GAME_TIME = time.time() - self.game_start_time

            if not self.GAME_DONE:
                if self.GAME_TIME >= 0 and not self.config.silent_mode and not self.SONG_STARTED:
                    self.emg_hero.play_song()
                    self.SONG_STARTED = True

                # get pressed keys
                if self.PLAY_WITH_EMG:
                    #if time.time() - self.last_data_extract > 0.05:
                    self.pressed_keys, one_hot_preds, features, new_features, self.too_high_values = self.model_handle.get_emg_keys()
                    print("onehots: ", type(one_hot_preds))
                    print("time at get_emg_keys: ", time.time() - self.last_data_extract)
                    self.last_data_extract = time.time()
                    # else:
                    #     new_features = False
                    #     self.too_high_values
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
        if not self.EXIT:
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