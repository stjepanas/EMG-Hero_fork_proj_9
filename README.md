# EMGHero
A Guitar Hero inspire game, which can be played using EMG instead of buttons. The classifier can be retrained on the game data after playing using Offline Reinforcement Learning.

Checkout out our [website](https://sites.google.com/view/bionic-limb-rl) for a more detailed explanation and videos.

## Setup
Install necessary libraries by running
```
cd EMGHero
conda create -n emg_hero python=3.11
conda activate emg_hero
pip3 install -r requirements.txt
python3 -m pip install 'd3rlpy @ git+https://github.com/KilianFt/d3rlpy.git'
```
**NOTE** if d3rlpy was previously installed in the conda environment then deinstall it first with
```
pip3 uninstall d3rlpy
```
before installing the fork from github.

### Setup with MatLAB
Then run the following to get the path of the conda python installation (in active conda env)
```
which python
```
Copy the path and set in in app_EMGHero.m for variable *condaPythonPath*.

Next, update the path to the EMGHero python package.

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

## Standard training procedure with MatLAB
### In MatLAB
1. Run RecordMovements from NCAL fit.
2. Pretrain a model
3. Run RL training (EMG Hero) by pressing the button
4. Press 'Start' when button activates
5. Play game
6. Decide if you want to update the MatLAB model with the latest RL one

> logs of the python script are saved in the experiment folder under emg_hero_logs.txt

### In Python
First, record movements in matlab and create a *.mat file with the *supervised_python_preperation.m* script. 
To pretrain the RL model on supervised data run
```
conda activate emg_hero
python3 train_model.py --pretrain --data ../data/emg_hero/RL_pretrain_example.mat
```
This will print the next command to run the main script using the newly trained model.

Before executing the python script with the given command run the MatLAB TCP server *EMGHero.m* from the MiSMT Env.

The main python script can be run with
```
python3 main.py --experiment-folder ./emg_hero_logs/my_experiment_2023_1_1_10_10_10/
```
Now you can decide if you want to play again, if yes you can either retrain the classifier or play again with the same one.

Press Q to quit the game.

All histories and classifier names will be saved when quitting.

### Testing
For testing purpose you can play the game without EMG
```
python3 main.py --experiment-folder ./emg_hero_logs/my_experiment_2023_1_1_10_10_10/ --no_emg
```

## Generate notes to song
You can generate notes to any song you want, you only need a *.wav file. Change the filename in *generate_emghero_song.py* and run it.

## Parameters
TODO