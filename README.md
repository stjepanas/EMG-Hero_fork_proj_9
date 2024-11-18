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
