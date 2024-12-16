from pydantic import BaseModel
from typing import Optional, List

# TODO move config yaml info here

class MLPConfig(BaseModel):
    n_layers: int = 6
    hidden_size: int = 256
    use_batch_norm: bool = True
    dropout: float = 0.1


class CNNConfig(BaseModel):
    n_channels: int = 32
    dropout: float = 0.1
    kernel_size: int = 3
    padding: int = 1
    n_layers: int = 1
    hidden_size: int = 256


class TCNConfig(BaseModel):
    out_channels: int = 16
    kernel_size: int = 3
    channels: int = 1
    layers: int = 1
    n_mlp_layers: int = 2
    hidden_size: int = 256
    dropout: float = 0.1
    do_pooling: bool = True


# TODO only used for policy_comparison so far
class PolicyConfig(BaseModel):
    seed: int = 100
    deterministic: bool = True
    encoder_type: str = 'conformer' # 'mlp' 'cnn', 'tcn' 'conformer'
    criterion_key: str = 'mse' # 'bce' 'mse' # TODO crossentropy for tanh
    out_actiation_key: str = 'tanh' # 'tanh' 'sigmoid'
    n_stacked_features: int = 10
    lr: float = 1e-4 # 1e-3
    n_actions: int = 7
    epochs: int = 50#250
    batch_size: int = 512 # 128
    do_swag: bool = False
    n_workers: int = 11

    tcn: TCNConfig = TCNConfig()
    cnn: CNNConfig = CNNConfig()
    mlp: MLPConfig = MLPConfig()


class GameConfig(BaseModel):
    # TCP settings
    host: str = "127.0.0.1"
    port: int = 51234

    silent_mode: bool = False
    show_fps: bool = False
    fps: int = 60
    circle_radius: int = 20
    arrow_height: int = 100
    arrow_width: int = 100
    img_scale: float = 0.15
    font_size: int = 20
    big_font_size: int = 40
    window_height: int = 900
    window_width: int = 1200 # px
    n_feats: int = 4 # FIXME 
    key_y: int = 800 # px
    time_before_start: int = 5 # s
    speed: int = 200 # px / s
    no_negative_reward_after_success_time: float = .2 # s
    circle_colors: List[dict] = [
        {
        'up': (255,0,0), # red
        'down': (255,0,0), # red
        },
        {
        'up': (255,255,0), # yellow
        'down': (255,255,0), # yellow
        },
        {
        'up': (0,0,255), # blue
        'down': (0,0,255), # blue
        },
    ]

    # scores
    hit_note_score: float = 10
    hit_repetition_score: float = 1
    not_hit_score: float = -1
    correct_dof_wrong_direction_score: float = 0.
    partial_score_penatly_denominator: int = 6 # higher value -> lower partial score


# TODO combine with PolicyConfig
class AlgoConfig(BaseModel):
    type: str = 'bc' # 'bc' 'awac' 'td3bc' 'td3' 'dt'

    only_use_last_history: bool = False
    n_histories_replay_buffer: Optional[int] = None # how many histories to include in replay buffer
    take_best_reward_model: bool = True
    append_ideal_data: bool = True
    n_ideal_appends: int = 2
    wrong_note_randomization: float = 0.76 # 0.0 for td3bc and 0.9 for awac
    wrong_note_replacement: bool = False # if True, wrong notes are replaced with random notes (like in old version)
    # use_discrete_actions: bool = False

    # general algo params
    n_steps: int = 2_200
    actor_hidden_size: int = 64
    actor_n_layers: int = 6
    actor_dropout: float = 0.0159
    critic_hidden_size: int = 128
    critic_n_layers: int = 12
    critic_dropout: float = 0.338
    actor_learning_rate: float = 0.00034
    critic_learning_rate: float = 0.000873
    batch_size: int = 254
    gamma: float = 1.0
    buffer_size: int = 50_000
    tau: float = 0.00044
    n_critics: int = 3

    # awac config
    lam: float = 0.93
    n_action_samples: int = 2

    # td3 config
    target_smoothing_sigma: float = 0.2
    target_smoothing_clip: float = 0.5
    update_actor_interval: int = 4

    # plus bc
    alpha: float = 2.5

    # bc config
    bc_learning_rate: float = 0.001
    bc_dropout: float = 0.25
    bc_batch_size: int = 128
    bc_n_steps: int = 2_400


class DiscreteDT(BaseModel):
    ideal_data_only: bool = False
    context_size: int = 20
    batch_size: int = 128
    max_timestep: int = 3000
    learning_rate: float = 6e-4
    num_heads: int = 8
    num_layers: int = 1
    attn_dropout: float = 0.1
    resid_dropout: float = 0.1
    embed_dropout: float = 0.1
    activation_type: str = "gelu"
    embed_activation_type: str = "tanh"
    position_encoding_type: str = "global" # "simple"
    warmup_tokens: int = 1000
    final_tokens: int = 30_000_000
    clip_grad_norm: float = 1.0
    buffer_size: int = 50_000
    n_steps: int = 2_000


class BaseConfig(BaseModel):
    random_seed: int = 100
    device: str = "cpu"
    sigmoid_threshold: float = 0.5
    tanh_threshold: float = 0.0

    notes_filename: Optional[str] = None

    game: GameConfig = GameConfig()
    algo: AlgoConfig = AlgoConfig()
    
