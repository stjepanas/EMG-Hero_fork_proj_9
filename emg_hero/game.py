import copy
import time
import pickle
import logging

import pygame
import numpy as np

from emg_hero.defs import Note, get_current_timestamp


NoneType = type(None)

def load_move_imgs(move_config, experiment_folder, config):
    # TODO add elbow right
    finger_keys = ['middle', 'index', 'thumb', 'elbow']

    base_dir = experiment_folder.parents[1]
    # right moves in pos 0 and left in pos 1
    move_imgs = []
    for arm_idx, arm_side in enumerate(['Left', 'Right']):
        arm_move_imgs = {}
        for ind_move in move_config.individual_movements[:-1]:
            if arm_idx == 1:
                # add key 'Right' for gross moves
                if any(finger_key in ind_move.lower() for finger_key in finger_keys):
                    file_name = base_dir / 'figures' / 'movements' / f'{ind_move}.png'
                else:
                    file_name = base_dir / 'figures' / 'movements' / f'{ind_move} {arm_side}.png'
            else:
                file_name = base_dir / 'figures' / 'movements' / f'{ind_move}.png'

            raw_move_img = pygame.image.load(file_name)
            raw_move_img.convert()
            angle = 0
            scale = config.img_scale
            move_img = pygame.transform.rotozoom(raw_move_img, angle, scale)
            arm_move_imgs[ind_move] = move_img

        move_imgs.append(arm_move_imgs)

    return move_imgs

class EMGHero:
    '''EMG Hero main game class that handles all logic and drawing'''
    def __init__(self, canvas, song, experiment_folder, config, notes_filename=None, cropped_soundfile=None, move_config=None, record_history=True):
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

        self.canvas = canvas
        if canvas is not None:
            self.font = pygame.font.SysFont('Arial', self.config.font_size)
            self.big_font = pygame.font.SysFont('Arial', self.config.big_font_size)
            self.song_mixer = pygame.mixer.Sound(file=cropped_soundfile)
            self.move_imgs = load_move_imgs(move_config, experiment_folder, self.config)

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

    def check_single_note_hit(self, note: Note,
                              curr_pressed_keys: dict,
                              give_partial_rewards: bool,
                              ) -> tuple[bool, float, bool]:
        '''Checks for a single note if it is correctly pressed

        Args:
            note (Note): note to be checked
            curr_pressed_keys (dict): current movement keys
            give_partial_rewards (bool): if partial rewards should

        Returns:
            bool: if note is in range and score given
            float: new score
            bool: if note was successful
        '''
        # FIXME predicts 0 when ideal movement is not Rest but the actual movement is Rest
        length = int(note.length * self.config.speed)
        center_y_end = note.position - length
        center_y_start = note.position

        note_new_score = 0
        note_in_range = False
        note_success = False
        if (center_y_start + self.config.circle_radius) > self.config.key_y and (center_y_end - self.config.circle_radius) < self.config.key_y \
            and len(curr_pressed_keys['lines']) > 0:
            # sort
            if self.use_right_arm:
                lines = (self.move_config.n_dof - 1) - np.array(note.lines)
            else:
                lines = note.lines
            sorted_notes = np.array(list(sorted(zip(lines, note.directions))))
            sorted_keys = np.array(list(sorted(zip(curr_pressed_keys['lines'],
                                                   curr_pressed_keys['directions']))))

            # check if lines are the same
            if np.array_equal(sorted_notes[:,0], sorted_keys[:,0]):
                # check if directions are the same
                if np.array_equal(sorted_notes[:,1], sorted_keys[:,1]):
                    if note.length > 0:
                        note_new_score += self.config.hit_repetition_score
                    else:
                        note_new_score += self.config.hit_note_score
                    note_in_range = True
                    note_success = True
                else:
                    note_new_score += self.config.correct_dof_wrong_direction_score

            # only give partial reward when less notes are predicted
            elif len(sorted_notes[:,0]) > len(sorted_keys[:,0]):
                if give_partial_rewards:
                    n_wanted_dof = len(note.lines)
                    correct_dofs = 0

                    # get correct lines and directions
                    for line, note_dir in zip(sorted_notes[:,0], sorted_notes[:,1]):
                        for key_line, key_dir in zip(sorted_keys[:,0], sorted_keys[:,1]):
                            if line == key_line and note_dir == key_dir:
                                correct_dofs += 1

                    if correct_dofs > 0:
                        note_in_range = True
                        correct_ratio = correct_dofs / \
                                        (n_wanted_dof + self.config.partial_score_penatly_denominator)
                        if note.length > 0:
                            note_new_score += self.config.hit_repetition_score * correct_ratio
                        else:
                            note_new_score += self.config.hit_note_score * correct_ratio
                        if correct_ratio >= 1.:
                            logging.warning("Ratio should never be greater than 1")

        # if not hit at all, give negative reward
        if (center_y_start) > self.config.key_y and (center_y_end) < self.config.key_y and not note_in_range:
            note_new_score += self.config.not_hit_score
            note_in_range = True

        return note_in_range, note_new_score, note_success

    def check_note_hit(self, curr_pressed_keys: dict,
                       give_partial_rewards: bool = False) -> tuple[bool, float]:
        '''Checks if note at given key index was hit

        Args:
            curr_pressed_keys (dict): dict of keys that are currently pressed

        Returns:
            bool: if any note was successful
            float: newly obtained score
        '''
        keys_success = False
        keys_in_range = False
        keys_new_score = 0.

        for note in self.notes:
            note_in_range, note_new_score, note_success = self.check_single_note_hit(note,
                                                        curr_pressed_keys,
                                                        give_partial_rewards=give_partial_rewards)

            if note_in_range:
                keys_new_score += note_new_score
            if not keys_success:
                keys_success = note_success
            if not keys_in_range:
                keys_in_range = note_in_range

        if not keys_in_range and len(curr_pressed_keys['lines']) > 0:
            keys_new_score += self.config.not_hit_score

        # add score to game score
        self.score += keys_new_score
        self.visibile_score += (keys_new_score + 1 if not keys_new_score == 0 else 0.01)

        return keys_success, keys_new_score

    def update_notes(self, current_game_time: float):
        '''Updates current visible notes, adds new ones when the time is right

        Args:
            current_game_time (float): current timestamp of the game
        '''
        time_to_button = self.config.key_y / self.config.speed

        for i, note in enumerate(self.song):
            # check time it needs to reach button
            appearance_time = note.time - time_to_button
            if appearance_time < current_game_time:
                self.notes.append(copy.deepcopy(note))
                # delete from song
                self.song.pop(i)

    def update_positions(self, last_update_time: float):
        '''Updates the position of every note

        Args:
            last_update_time (float): time of last update
        '''
        pos_dt = time.time() - last_update_time
        for note in self.notes:
            note.position += self.config.speed * pos_dt

    def draw_move_img(self, pos: list, movement):
        move_img = self.move_imgs[int(self.use_right_arm)][movement]
        rect = move_img.get_rect()
        rect.center = pos[0], pos[1]
        self.canvas.blit(move_img, rect)

    def draw_single_arrow(self,
                          pos: list,
                          direction: str = 'up',
                          color: list = (255,255,255),
                          width: int = 0,
                          scaling : float = 1):
        '''Draws single arrow in specified position and direction

        Args:
            pos (list): position of the arrow
            direction (str, optional): direction of the arrow. Defaults to 'up'.
            color (list, optional): color of the arrow. Defaults to (255,255,255).
            width (int, optional): line width. Defaults to 0.
            scaling (float, optional): size scaling. Defaults to 1.
        '''
        default_points = [(self.config.arrow_width/4, -self.config.arrow_height/2),
                        (self.config.arrow_width/4, 0),
                        (self.config.arrow_width/2, 0),
                        (0, self.config.arrow_height/2),
                        (-self.config.arrow_width/2, 0),
                        (-self.config.arrow_width/4, 0),
                        (-self.config.arrow_width/4, -self.config.arrow_height/2)]

        if direction == 'up':
            default_points = [(x, -y) for (x, y) in default_points]
        elif direction == 'right':
            default_points = [(y, x) for (x, y) in default_points]
        elif direction == 'left':
            default_points = [(-y, x) for (x, y) in default_points]
        elif direction == 'down':
            pass
        else:
            logging.error('Arrow direction not found')

        points = [(x * scaling + pos[0], y * scaling + pos[1]) for (x, y) in default_points]
        pygame.draw.polygon(surface=self.canvas, color=color, points=points, width=width)

    def draw_diamond(self, pos: list, color: list = (255,255,255)):
        '''Draw diamond shaped object

        Args:
            pos (list): position of diamond
            color (list, optional): color of diamond. Defaults to (255,255,255).
        '''
        default_points = [(0, self.config.arrow_height/2),
            (self.config.arrow_width/2, 0),
            (0, -self.config.arrow_height/2),
            (-self.config.arrow_width/2, 0)]

        points = [(x+pos[0], y+pos[1]) for (x, y) in default_points]
        pygame.draw.polygon(surface=self.canvas, color=color, points=points, width=5)

    def draw(self):
        '''Draw current notes to canvas
        '''
        for note in self.notes:
            onehot = note.onehot[:-1]
            note_move_list = np.array(self.move_config.individual_movements[:-1])[np.where(onehot == 1)]
    
            # if self.use_right_arm:
                # FIXME this is not working, is switiching hands properly working?
                # note_move_list = note_move_list[::-1]

            assert len(note.lines) == len(note_move_list), f'Note lines {note.lines} and {note_move_list} should have same length'
            for raw_line, direction, note_move in zip(note.lines, note.directions, note_move_list):
                if self.use_right_arm:
                    line = (self.move_config.n_dof - 1) - raw_line
                else:
                    line = raw_line

                if note.length > 0:
                    # draw note that has to be hold for x seconds
                    length = int(note.length * self.config.speed) # [s] * [px / s] = [px]
                    circle_color = self.config.circle_colors[line][direction]
                    center_x = int((self.config.window_width * (line + 1)) / (self.move_config.n_dof + 1))
                    center_y_end = note.position - length

                    if direction == 'up':
                        arrow_y = center_y_end
                    elif direction == 'down':
                        arrow_y = note.position

                    self.draw_single_arrow((center_x, arrow_y),
                                    direction=direction,
                                    color=circle_color)

                    rect = pygame.Rect((center_x-self.config.arrow_width/4),
                                       center_y_end,
                                       (self.config.arrow_width / 2 + 1),
                                       length)
                    pygame.draw.rect(self.canvas, circle_color, rect)

                    if self.help_mode:
                        self.draw_move_img(pos=(center_x, note.position-40), movement=note_move)

                else:
                    circle_color = self.config.circle_colors[line][direction]
                    center_x = int((self.config.window_width * (line + 1)) / (self.move_config.n_dof + 1))
                    center = (center_x, note.position)
                    self.draw_single_arrow(center, direction=direction, color=circle_color)

    def draw_score(self):
        '''Draw score to canvas
        '''
        score_text = self.font.render(("Score: " + str(round(self.visibile_score, 1))),
                                      False,
                                      (255,255,255))
        self.canvas.blit(score_text, (10, 10))

    def draw_current_arm(self):
        '''Draw current arm (left/right) to canvas
        '''
        arm_direction = 'Right' if self.use_right_arm else 'Left'
        arm_str = f"{arm_direction} arm"
        if self.help_mode:
            arm_str += ' [H]'
        arm_text = self.font.render(arm_str,
                                      False,
                                      (255,255,255))
        self.canvas.blit(arm_text, (self.config.window_width - 140, 10))

    def visualize_too_high_values(self, too_high_values):
        if too_high_values:
            pygame.draw.circle(self.canvas, (255,0,0), (self.config.window_width - 30, 30), 10)

    def visualize_button_press(self, vis_pressed_keys: dict, is_success: bool = False):
        '''Visualize which movement is predicted

        Args:
            vis_pressed_keys (dict): keys with line and directions that are predicted
            is_success (bool, optional): if last prediction was correct. Defaults to False.
        '''
        if len(vis_pressed_keys['lines']) < 1:
            return

        for line, direction in zip(vis_pressed_keys['lines'], vis_pressed_keys['directions']):
            if is_success:
                circle_color = (124,252,0)
                width = 0
                scaling = 1.3
            else:
                circle_color = (255,255,255)
                width = 5
                scaling = 1
            x_line = int((self.config.window_width * (line + 1)) / (self.move_config.n_dof + 1))
            center = (x_line, self.config.key_y)
            self.draw_single_arrow(pos = center,
                                   color = circle_color,
                                   direction = direction,
                                   width = width,
                                   scaling = scaling)

    def reset_canvas(self):
        '''Resets canvas to blank state with only lines
        '''
        self.canvas.fill((0,0,0))
        color = (255,255,255)
        for i in range(self.move_config.n_dof):
            x_line = int((self.config.window_width * (i + 1)) / (self.move_config.n_dof + 1))
            line = [(x_line, 0),(x_line, self.config.window_height)]

            # draw lines
            pygame.draw.aaline(self.canvas, color, *line)
            circle_color = (150,150,150)
            center = (x_line, self.config.key_y)
            self.draw_diamond(pos=center, color=circle_color)

            # display names of movements
            move_dof_idx = i if self.use_right_arm else -(i+1)
            line_dof = self.move_config.dof_movements[move_dof_idx]
            dof_text = self.font.render(line_dof, False, (255,255,255))
            self.canvas.blit(dof_text, (x_line+10, 50))

    def draw_end_canvas(self, n_round):
        '''Draw game end canvas with score
        '''
        self.canvas.fill((0,0,0))

        # show final score
        final_done_text = 'Round ' + str(n_round) + ' done'
        song_done_text = self.big_font.render(final_done_text, False, (255,255,255))
        text_rect = song_done_text.get_rect()
        text_rect.center = (self.config.window_width // 2, 100 + self.config.window_height // 4)
        self.canvas.blit(song_done_text, text_rect)

        # show final score
        final_score_text = self.big_font.render(('Final Score: ' + str(round(self.visibile_score))),
                                                False,
                                                (255,255,255))
        text_rect = final_score_text.get_rect()
        text_rect.center = (self.config.window_width // 2, 180 + self.config.window_height // 4)
        self.canvas.blit(final_score_text, text_rect)

        # show filename
        restart_text = self.font.render(('Press R to restart game or Q to quit'),
                                                False,
                                                (255,255,255))
        text_rect = restart_text.get_rect()
        text_rect.center = (self.config.window_width // 2, self.config.window_height * 3 // 4)
        self.canvas.blit(restart_text, text_rect)

    def draw_start_canvas(self, n_round):
        '''Draw game start canvas
        '''
        self.canvas.fill((0,0,0))

        # show final score
        title_start_text = 'EMG Hero'
        game_text = self.big_font.render(title_start_text, False, (255,255,255))
        text_rect = game_text.get_rect()
        text_rect.center = (self.config.window_width // 2, 100 + self.config.window_height // 4)
        self.canvas.blit(game_text, text_rect)

        # show round
        round_text = self.font.render((f'Round {n_round}'),
                                                False,
                                                (255,255,255))
        text_rect = round_text.get_rect()
        text_rect.center = (self.config.window_width // 2, 180 + self.config.window_height // 4)
        self.canvas.blit(round_text, text_rect)

        # show prompt
        start_text = self.font.render(('Press \'Enter\' to start game'),
                                                False,
                                                (255,255,255))
        text_rect = start_text.get_rect()
        text_rect.center = (self.config.window_width // 2, self.config.window_height * 3 // 4)
        self.canvas.blit(start_text, text_rect)

        self.draw_current_arm()

    def draw_restarting_canvas(self):
        '''Draw canvas that is shown when restarting the game
        '''
        self.canvas.fill((0,0,0))

        # show final score
        song_done_text = self.big_font.render('Restarting', False, (255,255,255))
        text_rect = song_done_text.get_rect()
        text_rect.center = (self.config.window_width // 2, 100 + self.config.window_height // 4)
        self.canvas.blit(song_done_text, text_rect)

        # Ask for retraining
        restart_text = self.font.render(('Do you want to retrain the model? Press [y] or [n]'),
                                                False,
                                                (255,255,255))
        text_rect = restart_text.get_rect()
        text_rect.center = (self.config.window_width // 2, self.config.window_height // 2)
        self.canvas.blit(restart_text, text_rect)

    def draw_fps(self, curr_start_time: float, game_time: float):
        '''Draws current FPS to canvas

        Args:
            curr_start_time (float): start time of iteration of game loop
            game_time (float): current game time
        '''
        fps_dt = time.time() - curr_start_time
        curr_fps = int(1. / fps_dt)
        text = self.font.render(("FPS: " + str(curr_fps)), False, (255,255,255))
        self.canvas.blit(text, (self.config.window_width - 98, 50))
        text = self.font.render(("Game time: " + str(round(game_time,1))),
                                    False,
                                    (255,255,255))
        self.canvas.blit(text, (self.config.window_width - 160, 10))

    def append_history(self,
                       current_game_time: float,
                       current_new_score: float,
                       current_one_hot_preds: np.ndarray,
                       current_pressed_keys: dict,
                       current_feat_data: NoneType | np.ndarray):
        '''Append history with newest entries

        Args:
            current_game_time (float): current game time
            current_new_score (float): score that was added in this iteration
            current_one_hot_preds (np.ndarray): latest prediction
            current_feat_data (NoneType | np.ndarray): latest features
        '''
        if self.record_history:
            self.history['rewards'].append(copy.deepcopy(current_new_score))
            self.history['time'].append(copy.deepcopy(current_game_time))
            self.history['features'].append(copy.deepcopy(current_feat_data))
            self.history['actions'].append(copy.deepcopy(current_one_hot_preds))
            self.history['pressed_keys'].append(copy.deepcopy(current_pressed_keys))
            self.history['notes'].append(copy.deepcopy(self.notes))

    def save_history(self, backup: bool = False) -> str:
        '''Saves current EMG Hero history to file

        Args:
            backup (bool, optional): append filename with backup. Defaults to False.

        Returns:
            str: filename of saved history
        '''
        now = get_current_timestamp()
        if backup:
            results_filename = self.experiment_folder / ('emg_hero_history_'+now+'_backup.pkl')
        else:
            results_filename = self.experiment_folder / ('emg_hero_history_'+now+'.pkl')

        with open(results_filename, 'wb') as _file:
            pickle.dump(self.history, _file)
            logging.info('History successfully saved to %s', results_filename)

        return results_filename

    def play_song(self):
        '''Function to play song
        '''
        self.song_mixer.play()

    def reset(self):
        self.save_history(backup=True)
        
        # FIXME self.song_mixer.stop()

        self.score = 0
        self.visibile_score = 0
        self.notes = []
        self.song = copy.deepcopy(self.full_song)

        self.history = {
            'features': [],
            'actions': [],
            'pressed_keys': [],
            'rewards': [],
            'time': [],
            'notes': [],
            'key_y': self.config.key_y,
            'song': self.song,
            'song_filename': self.notes_filename,
            'y_range': self.config.circle_radius,
            'speed': self.config.speed,
            'move_config': self.move_config,
        }

    def __del__(self):
        if self.canvas is not None:
            self.save_history(backup=True)
            self.song_mixer.stop()
