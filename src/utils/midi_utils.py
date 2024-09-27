import pretty_midi
import numpy as np

def piano_roll_to_midi(piano_roll, fs=4):
    """Convert a piano roll array to a PrettyMIDI object"""
    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)

    # Create a note from each non-zero element
    for frame in range(frames):
        for note in range(notes):
            if piano_roll[note, frame] > 0:
                # Create Note object
                start = frame / fs
                end = (frame + 1) / fs
                note = pretty_midi.Note(
                    velocity=int(piano_roll[note, frame]),
                    pitch=note,
                    start=start,
                    end=end
                )
                instrument.notes.append(note)

    pm.instruments.append(instrument)
    return pm