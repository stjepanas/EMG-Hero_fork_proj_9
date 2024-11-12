'''Definitions for EMG Hero
'''
from dataclasses import dataclass, field
from typing import List, Optional
import datetime

import numpy as np

from emg_hero.utils import get_dof_movements
@dataclass
class Note():
    """Class for keeping track of note details."""
    lines: list[int]
    directions: list[str]
    time: float = 0.
    position: float = 0.
    length: float = 0.
    # onehot: np.ndarray = field(default_factory=lambda: np.array([0,0,0,0,0,0,1])) 
    onehot: Optional[np.ndarray] = None
    move_name: str = ''


SUPERVISED_DATA_FILENAME = 'supervised_data.mat'
MODEL_CONFIG_FILENAME = 'model_config.json'
PRETRAINED_MODEL_FILENAME = 'pretrained_model.pt'
PRETRAINED_POLICY_FILENAME = 'pretrained_policy.pt'


def default_movements():
    return ['Thumb Extend',
            'Thumb Flex',
            'Index Extend',
            'Index Flex',
            'Middle Extend',
            'Middle Flex',
            'Thumb Extend + Index Extend',
            'Thumb Flex + Index Flex',
            'Index Extend + Middle Extend',
            'Index Flex + Middle Flex',
            'Thumb Extend + Index Extend + Middle Extend',
            'Thumb Flex + Index Flex + Middle Flex',
            'Rest',]


@dataclass
class MoveConfig:
    movements: List[str] = field(default_factory=default_movements)

    def __post_init__(self):
        self._movements_wo_rest = [move for move in self.movements if 'rest' not in move.lower()]
        self._individual_movements = [move for move in self.movements if '+' not in move]
        self._n_actions = len(self._individual_movements)
        self._dof_movements = get_dof_movements(self._individual_movements)
        self._n_dof = len(self._dof_movements)

        mappings = {}
        for i, move in enumerate(self._individual_movements):
            mappings[move] = i
        self._mappings = mappings

    @property
    def movements_wo_rest(self):
        return self._movements_wo_rest
    
    @property
    def individual_movements(self):
        return self._individual_movements

    @property
    def n_actions(self):
        return self._n_actions

    @property
    def dof_movements(self):
        return self._dof_movements

    @property
    def n_dof(self):
        return self._n_dof

    @property
    def mappings(self):
        return self._mappings


def get_current_timestamp() -> str:
    '''Gets current timestamp as string

    Returns:
        str: Current timestamp
    '''
    now = datetime.datetime.now()
    curr_time = str(now.year) + '_' + \
                str(now.month) + '_' + \
                str(now.day) + '_' + \
                str(now.hour) + '_' + \
                str(now.minute) + '_' + \
                str(now.second)
    return curr_time
