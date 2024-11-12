''' Song generator for EMG Hero

This script generates a song that can be played using the main script.
This song has the same amount of notes for each movement, only its
position is determined randomely. Though, each note is put on a beat
of the song so that it somewhat fits to the music.
'''
import os
import pickle
import random
import copy
from pathlib import Path
from collections import Counter

import librosa
import librosa.display
import soundfile as sf
import numpy as np

from emg_hero.label_transformer import LabelTransformer
from emg_hero.defs import Note, MoveConfig

def get_individual_notes(move_config, label_transformer):
    individual_notes = []
    for move in move_config.movements_wo_rest:
        line_dir_dict = label_transformer.move_name_to_line(move)
        onehot_move = label_transformer.move_name_to_onehot(move)
        lines = line_dir_dict['lines']
        directions = line_dir_dict['directions']
        note = Note(lines = lines,
                    directions = directions,
                    move_name = move,
                    onehot = onehot_move)
        individual_notes.append(note)

    return individual_notes

def get_available_timestamps(_available_beat_timestamps, _note, _note_buffer, time_between_notes):
    '''Finds all currently available timestamps for a given note in a buffer'''
    if _note.length > 0:
        # get all notes where after the note
        _possible_note_timestamps = []
        for av_beat in _available_beat_timestamps:
            _start_time = av_beat - time_between_notes
            _end_time = av_beat + _note.length + time_between_notes
            # check if there is already a note in this timeframe
            is_free = True
            for other_note in _note_buffer:
                if _start_time < other_note.time < _end_time:
                    is_free = False
            if is_free:
                _possible_note_timestamps.append(av_beat)

        _possible_note_timestamps = np.array(_possible_note_timestamps)
    else:
        _possible_note_timestamps = _available_beat_timestamps

    return _possible_note_timestamps


def fade_and_save_song(data, sr, experiment_folder):
    # Apply a fade-out effect
    fade_out_duration = 2.0
    fade_out_samples = int(fade_out_duration * sr)
    fade_out_ramp = np.linspace(1.0, 0.0, fade_out_samples)

    data[-fade_out_samples:] *= fade_out_ramp

    sf.write(experiment_folder / 'song_audio.wav', data, sr)


def allocate_note_times(note_buffer, beat_timestamps, time_between_notes):
    # spread out notes over whole song
    available_beat_timestamps = copy.deepcopy(beat_timestamps)

    for note in note_buffer:
        possible_note_timestamps = get_available_timestamps(available_beat_timestamps, note,
                                                            note_buffer, time_between_notes)
        chosen_time  = random.choice(possible_note_timestamps)
        note.time = chosen_time
        start_time = chosen_time - time_between_notes
        end_time = chosen_time + note.length + time_between_notes

        # remove impossible times
        to_delete_idxs = np.where(
            (available_beat_timestamps < end_time) &
            (available_beat_timestamps >= start_time))
        available_beat_timestamps = np.delete(available_beat_timestamps, to_delete_idxs)

    return note_buffer

def generate_song(audio_filename, experiment_folder, move_config, n_single_notes_per_class,
                  n_repetitions_per_class, note_lenghts, time_between_notes=.5, additional_time_p=.4):
    results_notes_filename = experiment_folder / 'song_notes.pkl'
    
    # check if song already exists
    if os.path.exists(results_notes_filename):
        with open(results_notes_filename, 'rb') as file:
            song = pickle.load(file)

        if len(song['movements']) == len(move_config.movements) and \
              Counter(song['movements']) == Counter(move_config.movements):
            print('Loading existing song')
            return song['notes'], results_notes_filename
        else:
            # TODO 
            # raise NotImplementedError
            print('WARN: old song file will be overwritten with new movements')

    # load song
    y, sr = librosa.load(audio_filename)

    # find beat
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)
    _, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    beat_timestamps = librosa.frames_to_time(beats, sr=sr)

    label_transformer = LabelTransformer(move_config=move_config)

    individual_notes = get_individual_notes(move_config, label_transformer)

    # create repetition notes
    individual_repetitive_notes = []
    for note in copy.deepcopy(individual_notes):
        for note_length in note_lenghts:
            note.length = note_length
            individual_repetitive_notes.append(copy.deepcopy(note))

    repetition_notes = list(np.repeat(individual_repetitive_notes, n_repetitions_per_class))
    repetition_notes = [copy.deepcopy(note) for note in repetition_notes]

    single_notes = list(np.repeat(individual_notes, n_single_notes_per_class))
    single_notes = [copy.deepcopy(note) for note in single_notes]

    note_buffer = repetition_notes + single_notes

    # check if there is enough time
    time_needed = np.sum([note.length + 2 * time_between_notes for note in note_buffer])
    song_time = beat_timestamps[-1]
    error_msg = f"Error: time need for notes {time_needed} s while song has {song_time} s"
    assert time_needed < song_time, error_msg

    # cut song length if too long
    additional_time_factor = 1. + additional_time_p
    if time_needed * additional_time_factor < song_time:
        song_end_time = time_needed * additional_time_factor
        cutoff = int(len(y) * (song_end_time / song_time))
        y = y[:cutoff]
        beat_timestamps = beat_timestamps[beat_timestamps < song_end_time]

    fade_and_save_song(y, sr, experiment_folder)

    note_buffer = allocate_note_times(note_buffer, beat_timestamps, time_between_notes)

    # check if notes has correct length
    n_wanted_notes = len(move_config.movements_wo_rest) * \
                    (n_single_notes_per_class + len(note_lenghts) * n_repetitions_per_class)
    assert len(note_buffer) == n_wanted_notes, "Error: not all notes have been allocated a beat"

    song = {
        'notes': note_buffer,
        'movements': move_config.movements,
        'audio_filename': audio_filename,
    }

    with open(results_notes_filename, 'wb') as file:
        pickle.dump(song, file)
        print(f'Object successfully saved to "{results_notes_filename}"')

    return note_buffer, results_notes_filename


if __name__ == '__main__':
    audio_filename = './song_files/test_song_file2.wav'
    experiment_folder = Path('emg_hero_logs/dummy_folder/')

    time_between_notes = .5
    n_single_notes_per_class = 0
    n_repetitions_per_class = 1
    note_lenghts = [0.5, 1.0, 1.5, 2.0]

    movements = ['Thumb Extend', 'Thumb Flex', 'Index Extend', 'Index Flex', 'Middle Extend', 'Middle Flex', 'Thumb Extend + Index Extend',
                'Thumb Extend + Index Extend + Middle Extend', 'Thumb Flex + Index Flex', 'Thumb Flex + Index Flex + Middle Flex', 'Index Extend + Middle Extend',
                'Index Flex + Middle Flex', 'Rest']
    
    move_config = MoveConfig(movements=movements)

    song, result_filename = generate_song(audio_filename,
                                          experiment_folder,
                                          move_config,
                                          n_single_notes_per_class,
                                          n_repetitions_per_class,
                                          note_lenghts,
                                          time_between_notes=time_between_notes)
