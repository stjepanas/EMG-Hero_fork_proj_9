import time
import copy
import logging
import numpy as np
import pygame

from emg_hero.datasets import get_history_filenames
from emg_hero.label_transformer import LabelTransformer
from emg_hero.defs import get_current_timestamp, SUPERVISED_DATA_FILENAME, MoveConfig
from emg_hero.generate_song import generate_song
from emg_hero.analyze_utils import create_csv
from emg_hero.configs import BaseConfig

import gymnasium as gym
from gymnasium.envs.registration import register
import gymnasium.spaces as spaces

class EMGHeroEnv(gym.Env):

    def __init__(self, history_filenames, emg_hero, model_handle, play_with_emg, n_round, experiment_folder):
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

        self.EXPERIMENT_FOLDER = experiment_folder
        self.SUPERVISED_PATH = self.EXPERIMENT_FOLDER / SUPERVISED_DATA_FILENAME
        

        self.label_transformer = LabelTransformer(move_config=self.move_config)

        # start timer and play sound
        self.game_start_time = time.time() + self.config.time_before_start
        self.last_pos_update_time = time.time()
        self.LAST_NOTE_TIME = None
        self.SONG_STARTED = False
        self.GAME_STARTED = False
        self.SUCCESS = False
        self.current_history_filenames = copy.deepcopy(self.HISTORY_FILENAMES)
        self.new_features = False
        self.too_high_values = False
        self.pressed_keys = {
            'lines': [],
            'directions': [],
            }

        # Gym initialize action space
        # 7 float values that will be passed through a sigmoid and threshold 
        self.action_space = spaces.Box(low=0, high=1, shape=(7,), dtype=np.float64)

        # Gym initialize observation space
        # 4 EMG features [MAW, WL, ZC, SSC] with 8 channels, hence 32 elements
        self.observation_space = spaces.Box(low=-2**61, high=2**61-2, shape=(32,), dtype=np.float64)

        self.observation = np.zeros((32,))
        self.info = {}

        

    def step(self, action):
        print(self.GAME_RESTART)

        """ Performs a single update step for the game

        Args:
            action (np.array(int)): one hot prediction array from model (7,1)
        """
        self.clock.tick(self.config.fps)

        self.pressed_keys = action[0]
        one_hot_preds = action[1]
        features = action[2]
        self.new_features = action[3]
        self.too_high_values = action[4]

        # print("pressed keys env:", self.pressed_keys)

        if self.GAME_RESTART:
            # ask if retraining
            RESTARTING = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.EXIT = True
                if event.type == pygame.KEYDOWN:
                    if event.unicode == 'q' or event.unicode == 'Q':
                        print("--------------Q1-------------")
                        self.EXIT = True
                    if event.unicode == 'y' or event.unicode == 'Y':
                        # retrain model
                        logging.info('Retraining model')
                        # only take last n histories to retrain
                        if self.base_config.algo.n_histories_replay_buffer is None:
                            hists_in_buffer = self.current_history_filenames
                        else:
                            hists_in_buffer = self.current_history_filenames[-self.base_config.algo.n_histories_replay_buffer:]

                        print("SUPERVISED PATH: ", self.SUPERVISED_PATH)
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
                    print("--------------Q2-------------")
                if event.unicode == 'r' or event.unicode == 'R':
                    print("---------------R--------------")
                    # print(self.GAME_DONE)
                    # print(self.GAME_RESTART)
                    # print(self.GAME_STARTED)
                    # print(self.GAME_TIME)
                    # print(self.EXIT)
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
                    # one_hot_preds = action
                    self.last_data_extract = time.time()
                else:
                    self.new_features = True
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

                if self.new_features:
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

        self.info = {"current_history_filenames": self.current_history_filenames}
        
        self.terminated = self.EXIT

        self.reward = self.emg_hero.score 

        self.truncated = False

        return self.observation, self.reward, self.terminated, self.truncated, self.info
    
    def reset(self, seed = None, options = None):
        super().reset(seed=seed)

        self.observation = np.zeros((32,),dtype=np.float64)
        return self.observation, self.info

    def render(self):
        if not self.EXIT:
            self.emg_hero.reset_canvas()

            if not self.GAME_STARTED:
                self.emg_hero.draw_start_canvas(self.N_ROUND)
                pygame.display.update()
                time.sleep(0.05)
                return
            
            if self.GAME_RESTART:
                self.emg_hero.draw_restarting_canvas()
                pygame.display.update()
                time.sleep(0.05)
                return

            # if game done, show results screen
            if self.GAME_DONE:
                self.emg_hero.draw_end_canvas(self.N_ROUND)
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