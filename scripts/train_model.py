'''Script to train EMG Hero model. Both, for pretraining with supervised, labeled, data and RL from playing EMG Hero.
'''
import os
import logging
import argparse
from pathlib import Path
from d3rlpy.algos import AWAC
from main import EMGHero
from emg_hero.label_transformer import LabelTransformer
from emg_hero.metrics import EMGHeroMetrics
from emg_hero.model_utility import pretrain_emg_hero_model, \
                                    save_emg_hero_model, \
                                    train_emg_hero_model
from emg_hero.defs import get_current_timestamp, \
                        N_ACTIONS, \
                        MOVEMENT_MAPPINGS, \
                        SUPERVISED_DATA_FILENAME, \
                        MODEL_CONFIG_FILENAME

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%y-%m-%d %H:%M:%S')

    parser = argparse.ArgumentParser(
                    prog='EMG Hero Model Utility',
                    description='Contains training scripts for pretraining and RL with EMG Hero',
                    epilog='A project at CBPR')

    parser.add_argument('-p', '--pretrain',
                        action='store_true',
                        help='Flag to only pretrain a model')
    parser.add_argument("--data",
                        type=str,
                        help="Filename of training data.\
                            If --pretrain then this should be supervised data")
    parser.add_argument("--model_config",
                        type=str,
                        default='./config/awac_params.json',
                        help="Filename of model config")
    parser.add_argument("--experiment_name",
                        type=str,
                        default='experiment',
                        help="Name of current experiment")
    parser.add_argument("--experiment_folder",
                        default=None,
                        help="Folder of existing experiment")
    parser.add_argument("--model",
                        default=None,
                        help="Existing model to train")
    parser.add_argument("--history",
                        default=None,
                        help="EMG Hero history filename to train model")
    args = parser.parse_args()

    if args.pretrain:
        NOW = get_current_timestamp()
        LOG_FOLDER =  Path('./logs')
        EXPERIMENT_FOLDER = LOG_FOLDER / (args.experiment_name + '_' + NOW)

        if not os.path.exists(EXPERIMENT_FOLDER):
            os.makedirs(EXPERIMENT_FOLDER)

        logging.info('Starting pretraining on %s', args.data)
        pretrained_model = pretrain_emg_hero_model(args.data,
                                                   model_config_file = args.model_config,
                                                   experiment_folder = EXPERIMENT_FOLDER,
                                                   n_steps=1_000)
        save_emg_hero_model(model = pretrained_model,
                            # model_config_file = args.model_config,
                            experiment_folder = EXPERIMENT_FOLDER,
                            supervised_data_file = args.data)
        logging.info('Training done, play EMG Hero with the trained model by runnning \n\
                     python3 main.py --experiment_folder %s',
                     EXPERIMENT_FOLDER)
    else:
        if args.experiment_folder is None:
            logging.error('Please provide experiment folder when not pretraining')

        if args.model is None:
            logging.error('Please provide model when not pretraining')

        if args.history is None:
            logging.error('Please provide history filename when not pretraining')

        EXPERIMENT_FOLDER = args.experiment_folder

        train_history_filenames = [args.history]
        supervised_filename = EXPERIMENT_FOLDER + SUPERVISED_DATA_FILENAME
        model_config_path = EXPERIMENT_FOLDER + MODEL_CONFIG_FILENAME
        model_path = EXPERIMENT_FOLDER + args.model
        # load model
        model = AWAC.from_json(model_config_path)
        model.load_model(model_path)

        label_transformer = LabelTransformer(n_actions=N_ACTIONS,
                                         movement_mappings=MOVEMENT_MAPPINGS)

        # load metrics
        emg_hero_dummy = EMGHero(None, None, '')
        emg_hero_metrics = EMGHeroMetrics(emg_hero = emg_hero_dummy,
                                    label_transformer = label_transformer,
                                    song_dataset = None,
                                    history = None,
                                    supervised_dataset = None,
                                    action_size = N_ACTIONS)

        # load history

        # retrain
        trained_model, model_path = train_emg_hero_model(model,
                                           train_history_filenames,
                                           supervised_filename,
                                           experiment_folder=EXPERIMENT_FOLDER,
                                           emg_hero_metrics=emg_hero_metrics,
                                           take_best_reward_model = True,
                                           wrong_note_randomization=0.9)

        logging.info('Finished training, model saved to %s', model_path)
