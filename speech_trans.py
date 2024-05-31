import torch
from speech_transformer import SpeechTransformer

BATCH_SIZE, SEQ_LENGTH, DIM, NUM_CLASSES = 320, 200, 160, 4
OUTPUT_DIM = 40  # Desired output dimension

cuda = None#torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')

inputs = torch.rand(BATCH_SIZE, SEQ_LENGTH, DIM).to(device)
input_lengths = torch.randint(low=1, high=SEQ_LENGTH + 1, size=(BATCH_SIZE,), dtype=torch.int32)  # Random lengths for each sequence
targets = torch.randint(low=0, high=NUM_CLASSES, size=(BATCH_SIZE, OUTPUT_DIM), dtype=torch.long)  # Adjusted target tensor for desired output shape
target_lengths = torch.randint(low=1, high=SEQ_LENGTH + 1, size=(BATCH_SIZE,), dtype=torch.int32)  # Random lengths for each target sequence

# Ensure the model architecture is appropriate for your task
model = SpeechTransformer(num_classes=NUM_CLASSES, d_model=DIM, num_heads=8, input_dim=DIM)

# Move model to appropriate device
model = model.to(device)

# Ensure inputs and targets are on the same device
inputs = inputs.to(device)
targets = targets.to(device)

# Call model
predictions, logits = model(inputs, input_lengths, targets, target_lengths)

# Print output shapes
print("Predictions Shape:", predictions.shape)
print("Logits Shape:", logits.shape)
