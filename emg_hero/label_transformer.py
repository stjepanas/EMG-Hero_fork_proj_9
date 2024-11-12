'''Label transformer for EMG Hero

Handles all label transformations that need to be performed.
'''

import numpy as np
from emg_hero.defs import Note

class LabelTransformer():
    '''Transforms labels from Move Name, to one hot to emg hero format with line and direction'''
    def __init__(self, move_config):
        self.n_actions = move_config.n_actions
        self.movement_mappings = move_config.mappings
        self.move_config = move_config
    
        self.move_dir_dict = {'down': ['flex', 'supination', 'close'],
                            'up': ['extend', 'pronation', 'open']}

    def get_move_lines(self, move):
        assert self.move_config is not None, 'Please provice move_config for label transformer to use this function'

        line_idxs = []
        for raw_line_idx, dof_move in enumerate(self.move_config.dof_movements):
            is_rot_move = (dof_move.lower() == 'wrist rotation' and move.lower() in ['pronation', 'supination'])
            move_in_dof_move = (dof_move.lower() in move.lower())
            if move_in_dof_move or is_rot_move:
                # switch direction
                line_idx = (len(self.move_config.dof_movements) - 1) - raw_line_idx
                line_idxs.append(line_idx)
        return line_idxs

    def get_move_directions(self, move: str):
        assert self.move_config is not None, 'Please provice move_config for label transformer to use this function'

        directions = []
        for direction in self.move_dir_dict.keys():
            for dir_key in self.move_dir_dict[direction]:
                if dir_key in move.lower():
                    directions.append(direction)

        return directions

    def move_name_to_onehot(self, move_name: str) -> np.ndarray:
        '''Converts move name to onehot

        Args:
            move_name (str): Name of movement

        Returns:
            np.ndarray: =nehot encoding of movement
        '''
        split_move_names = move_name.split(' + ')

        move_onehot = np.zeros(self.n_actions)
        for string_key in split_move_names:
            move_onehot[self.movement_mappings[string_key]] = 1
        return move_onehot

    def move_onehot_to_line(self, move_onehot: np.ndarray) -> tuple[list, list]:
        '''Converts onehot encoding to lines and directions

        Args:
            move_onehot (np.ndarray): Onehot encoded movement

        Returns:
            tuple[list, list]: Lines and directions of movement
        '''
        move_idxs = np.where(move_onehot)[0]
        lines = []
        directions = []
        for move_idx in move_idxs:
            for move, value in self.move_config.mappings.items():
                if move_idx == value:
                    lines += self.get_move_lines(move)
                    directions += self.get_move_directions(move)

        return {'lines': lines, 'directions': directions}

    def move_name_to_line(self, move_name: str) -> tuple[list, list]:
        '''Converts name of movement to lines and directnions

        Args:
            move_name (str): Movement name

        Returns:
            tuple[list, list]: Lines and directions of movement
        '''
        lines = []
        directions = []
        split_move_names = move_name.split(' + ')
        for single_move_name in split_move_names:
            lines += self.get_move_lines(single_move_name)
            directions += self.get_move_directions(single_move_name)
        # return self.move_onehot_to_line(self.move_name_to_onehot(move_name))
        assert len(lines) == len(directions), f'Line and direction missmatch for move {move_name}: {lines}, {directions}'
        return {'lines': lines, 'directions': directions}


    def note_to_onehot(self, note: Note) -> np.ndarray:
        return note_to_onehot(note, self.n_actions)

    def get_dof_move_count(self, dof_move):
        count = 0
        for move in self.move_config.mappings.keys():
            is_rot_move = (dof_move.lower() == 'wrist rotation' and move.lower() in ['pronation', 'supination'])
            move_in_dof_move = (dof_move.lower() in move.lower())
            if move_in_dof_move or is_rot_move:
                count += 1
        return count

    def keys_to_onehot(self, key: dict) -> np.ndarray:
        '''Converts keys to onehot encoded movement

        Args:
            key (dict): Pressed key

        Returns:
            np.ndarray: Onehot encoding of key
        '''
        dof_start_pos = {}
        count = 0
        for dof_move in self.move_config.dof_movements:
            dof_start_pos[dof_move] = count
            count += self.get_dof_move_count(dof_move)

        onehot_label = np.zeros(self.move_config.n_actions)
        for line, direction in zip(key['lines'], key['directions']):
            # reverse back
            raw_line_idx = (len(self.move_config.dof_movements) - 1) - line
            dof_move = self.move_config.dof_movements[raw_line_idx]
            dof_move_count = self.get_dof_move_count(dof_move)
            direction_idx = 1 if direction == 'down' else 0
            if dof_move_count == 1:
                direction_idx = 0

            start_pos = dof_start_pos[dof_move]
            pred_idx = start_pos + direction_idx
            onehot_label[pred_idx] = 1
        return onehot_label

def note_to_onehot(note: Note, n_actions) -> np.ndarray:
    '''Converts note to onehot encoded movement

    Args:
        note (Note): Note of EMG Hero song

    Returns:
        np.ndarray: Onehot encoding of song
    '''
    print('deprecated')
    onehot_label = np.zeros(n_actions)
    for line, direction in zip(note.lines, note.directions):
        direction_idx = 0 if direction == 'down' else 1
        pred_idx = n_actions - (line * 2) - direction_idx - 2
        onehot_label[pred_idx] = 1

    return onehot_label


class MultilabelConverter:
    '''Converts multilabel vectors to single label and back
    '''
    def __init__(self):
        self.mapping = {}
        self.next_label = 1

    def convert(self, multilabel_vector):
        key = tuple(multilabel_vector)
        if key not in self.mapping:
            self.mapping[key] = self.next_label
            self.next_label += 1
        return self.mapping[key]
    
    def convert_list(self, multilabel_vectors):
        return np.array([self.convert(v) for v in multilabel_vectors])
    
    def convert_back(self, label):
        for key, value in self.mapping.items():
            if value == label:
                return key
        return np.zeros(len(key))
    
    def convert_back_list(self, labels):
        return np.array([self.convert_back(l) for l in labels])
