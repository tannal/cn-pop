import torch
from torch.utils.data import Dataset
import numpy as np
import pretty_midi
import os

class MIDIDataset(Dataset):
    def __init__(self, data_dir, sequence_length=100):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.mid')]
        self.data = self.load_midi_files()

    def load_midi_files(self):
        data = []
        for file in self.file_list:
            try:
                midi_data = pretty_midi.PrettyMIDI(os.path.join(self.data_dir, file))
                piano_roll = midi_data.get_piano_roll(fs=4)  # 4 ticks per quarter note
                data.append(piano_roll.T)
            except:
                print(f"Error loading {file}")
        return np.concatenate(data)

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        sequence = self.data[idx:idx+self.sequence_length]
        target = self.data[idx+1:idx+self.sequence_length+1]
        return torch.FloatTensor(sequence), torch.FloatTensor(target)