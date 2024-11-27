import gymnasium as gym
from gymnasium.envs.registration import register
import gymnasium.spaces as spaces
import gym_problem
import emg_hero.configs as config
import copy
import pygame
import emg_hero.game as game

register(
    id = 'emg-hero-v0',
    entry_point = 'emgHeroEnv'
)

class EMGHeroEnv(gym.Env):

    metadata = {'render_modes': ['human'], 'render_fps': config.GameConfig.fps} # dont know if 30

    def __init__(self,canvas, song, experiment_folder, config, notes_filename=None, cropped_soundfile=None, move_config=None, record_history=True):

        # properties
        self.move_config = move_config
        self.record_history = record_history
        self.notes_filename = notes_filename
        self.experiment_folder = experiment_folder
        self.use_right_arm = False
        self.help_mode = False
        self.config = config
        self.full_song = song

        # scores etc
        self.score = 0
        self.visibile_score = 0
        self.notes = []
        self.song = copy.deepcopy(song)

        # pygame canvas
        self.canvas = canvas
        if canvas is not None:
            self.font = pygame.font.SysFont('Arial', self.config.font_size)
            self.big_font = pygame.font.SysFont('Arial', self.config.big_font_size)
            self.song_mixer = pygame.mixer.Sound(file=cropped_soundfile)
            self.move_imgs = game.load_move_imgs(move_config, experiment_folder, self.config)

        # History
        self.history = {
            'features': [],
            'actions': [],
            'pressed_keys': [],
            'rewards': [],
            'time': [],
            'notes': [],
            'key_y': self.config.key_y,
            'song': song,
            'song_filename': notes_filename,
            'y_range': self.config.circle_radius,
            'speed': self.config.speed,
            'move_config': move_config,
        }

        # Gym render
        self.render_mode = None

        # Gym initialize action space
        self.action_space = spaces.Discrete(len(gym_problem.emgHeroActions))

        # Gym initialize observation space
        self.observation_space = spaces.Discrete()

    def step(self, action):


        return self.observation, self.reward, self.terminated, self.truncated, self.info
    
    def reset(self):
        return self.observation

    def render(self, render_mode = 'human'):
        pass

    def close(self):
        pass