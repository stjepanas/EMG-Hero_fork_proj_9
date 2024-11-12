import copy
from collections import OrderedDict
from typing import Dict, Any

import wandb
import d3rlpy
import numpy as np


def actions_to_labels(actions):
    unique_actions = np.unique(actions, axis=0)

    labels = []
    for i in range(actions.shape[0]):
        action = actions[i,:]
        action_idx = np.where((unique_actions == action).all(axis=1))[0][0]
        labels.append(action_idx)

    return np.array(labels)


def get_other_unique_moves(other_movements):
    other_moves_lower = [move.lower() for move in other_movements]
    other_key_pairs = [{'name': 'hand',
                        'keys': ['open hand', 'close hand']},
                    {'name': 'sub pro',
                        'keys': ['supination', 'pronation']}]

    other_unique_moves = []
    for key_pair in other_key_pairs:
        if key_pair['keys'][0] in other_moves_lower and key_pair['keys'][1] in other_moves_lower:
            for key in key_pair['keys']:
                other_moves_lower.remove(key)
            other_unique_moves.append(key_pair['name'].capitalize())
        else:
            for key in key_pair['keys']:
                if key in other_moves_lower:
                    other_unique_moves.append(key.capitalize())
                    other_moves_lower.remove(key)

    if len(other_moves_lower) > 0:
        print('Warn: not all movements considered')

    return other_unique_moves

def get_flex_extend_unique_moves(flex_extend_movements):
    fe_dof_moves = [move.lower().replace(" ", "").replace("flex", "").replace("extend", "").capitalize()
            for move in flex_extend_movements if 'rest' not in move.lower()]
    unique_fe_dof_moves = np.unique(fe_dof_moves).tolist()
    return unique_fe_dof_moves

def get_dof_movements(individual_movements):
    indv_move_copy = copy.deepcopy(individual_movements[:-1])
    to_remove_strs = ['Open', 'Close', 'Flex', 'Extend', ' ']
    for rm_str in to_remove_strs:
        indv_move_copy = [im.replace(rm_str, '') for im in indv_move_copy]

    unique_moves = list(OrderedDict.fromkeys(indv_move_copy))

    replaced = False
    if 'Supination' in unique_moves and 'Pronation' in unique_moves:
        for i, u_move in enumerate(unique_moves):
            if u_move == 'Pronation':
                unique_moves.remove('Pronation')
                if not replaced:
                    unique_moves.insert(1, 'Wrist Rotation')
                    replaced = True

            if u_move == 'Supination':
                unique_moves.remove('Supination')
                if not replaced:
                    unique_moves.insert(1, 'Wrist Rotation')
                    replaced = True

    return unique_moves


class WandbAdapter(d3rlpy.logging.LoggerAdapter):
    def write_params(self, params: Dict[str, Any]) -> None:
        # save dictionary as json file
        # with open("params.json", "w") as f:
        #     f.write(json.dumps(params, default=default_json_encoder, indent=2))
        pass

    def before_write_metric(self, epoch: int, step: int) -> None:
        pass

    def write_metric(self, epoch: int, step: int, name: str, value: float) -> None:
        # with open(f"{name}.csv", "a") as f:
        #     print(f"{epoch},{step},{value}", file=f)
        wandb.run.log(
            {name: value,},
            # step=step
        )

    def after_write_metric(self, epoch: int, step: int) -> None:
        pass

    def save_model(self, epoch: int, algo: Any) -> None:
        # algo.save(f"model_{epoch}.d3")
        pass

    def close(self) -> None:
        pass


class WandbAdapterFactory(d3rlpy.logging.LoggerAdapterFactory):
    def create(self, experiment_name: str) -> d3rlpy.logging.FileAdapter:
        return WandbAdapter()
