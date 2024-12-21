#Internal Testing
```
python3 gym_env.py --no_emg --experiment_folder logs/test1
```
# EMGHero
A Guitar Hero inspire game, which can be played using EMG instead of buttons. The classifier can be retrained on the game data after playing using Offline Reinforcement Learning.

Checkout out our [website](https://sites.google.com/view/bionic-limb-rl) for a more detailed explanation and videos.

## Setup
To setup the game and install the necessary libraries run
```
git clone https://github.com/KilianFt/EMG-Hero.git
cd EMG-Hero
conda create -n emg_hero python=3.11
conda activate emg_hero
pip3 install -r requirements.txt
```
Additionally, you will have to install our own fork of d3rlpy
```
python3 -m pip install 'd3rlpy @ git+https://github.com/KilianFt/d3rlpy.git'
```
**NOTE** if d3rlpy was previously installed in the conda environment then deinstall it first with
```
# Only do this if d3rlpy was previously installed
pip3 uninstall d3rlpy
```
before installing the fork from github.

### Setup SifiLabs BioArmband
To install libemg with Sifi BioArmband run
```
pip install git+https://github.com/LibEMG/libemg.git@latest
```

## Testing
You can run a keyboard-controlled version with
```
python main.py --no_emg --experiment_folder logs/test1
```

## Game Control
Following buttons are available to control the game:
| Key | Function                  |
|-----|---------------------------|
| q   | Quit                      |
| r   | Restart                   |
| m   | Switch movement direction |
| h   | Turn on help mode (imgs)  |

## Standard training procedure
**To be done**

The main loop should eventually run with:
```
python3 main.py --experiment-folder ./logs/my_experiment_2023_1_1_10_10_10/
```
Now you can decide if you want to play again, if yes you can either retrain the classifier or play again with the same one.

Press Q to quit the game.

All histories and classifier names will be saved when quitting.

## Generate notes to song
You can generate notes to any song you want, you only need a *.wav file. Change the filename in *generate_emghero_song.py* and run it.

## Parameters
TODO

## Supervised learning 
Everything necessary to build the training and testing datasets is in the SL and create_datasets scripts.
SL script does the data gathering, separates it into the training and testing, extracts features and makes the MDPDatasets with functions from create_datasets.

To gather the raw data LibEMGs own GUI is used but their image database has been replaced with own images.
GUI shows every image in the images map, so every time gata is gathered, it is gathered for all the images in the map. LibEMGs function download_images could be modified so that only picking out specific movements is possible but that would entail forking the LibEMG git repo and their image repo. Moving the non-wanted movements from the images map also works but then the movements need to be remapped in gym_problem to match the collection.json in the data map. 

SL script also contains the behaviour cloning part where the models are being fit with the hyperparameters that were used in the previous project, as well as testing with the test dataset and plotting the success rate + the loss for every model. Every time the bc.fit is run, it creates a new map with the trained models. The get_newest_directory function from the previous project is used so that the directories don't need to be changed manually in the testing and plotting part of the SL script. This way, everything can be done in one run of the SL script.


