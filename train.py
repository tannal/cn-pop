import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data.midi_dataset import MIDIDataset
from src.models.music_rnn import MusicRNN
from src.utils.midi_utils import piano_roll_to_midi
import os

# Hyperparameters
input_size = 128  # Number of MIDI notes
hidden_size = 256
num_layers = 2
output_size = 128
learning_rate = 0.001
batch_size = 64
num_epochs = 50
sequence_length = 100

# Load dataset
data_dir = "path/to/lakh/midi/dataset"
dataset = MIDIDataset(data_dir, sequence_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MusicRNN(input_size, hidden_size, num_layers, output_size).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}")

# Generate music
model.eval()
with torch.no_grad():
    # Start with a random sequence
    sequence = torch.randn(1, sequence_length, input_size).to(device)
    generated = []
    
    for _ in range(500):  # Generate 500 time steps
        output = model(sequence)
        generated.append(output[:, -1, :].cpu().numpy())
        sequence = torch.cat([sequence[:, 1:, :], output[:, -1:, :]], dim=1)

generated = np.array(generated).squeeze()

# Convert to MIDI and save
midi_output = piano_roll_to_midi(generated.T)
output_dir = "generated_music"
os.makedirs(output_dir, exist_ok=True)
midi_output.write(os.path.join(output_dir, "generated_music.mid"))

print("Music generation complete. Output saved in the 'generated_music' directory.")