from model import Model  # Import your model from Model.py
from torch.utils.data import DataLoader
from dataset import LRADataset  # Import the dataset class used in run_tasks.py
import torchlens as tl  # Assuming torchlens is correctly installed

# Load the configuration from run_tasks.py
config = {
    "embedding_dim": 512,
    "transformer_dim": 512,
    "transformer_hidden_dim": 2048,
    "vocab_size": 30522,
    "max_seq_len": 512,
    "dropout_prob": 0.1,
    "num_layers": 12,
    "tied_weights": False,
    "attn_type": "softmax",
    "is_butterfly": False,
    "fabnet_att_layer": -1,
    "attention_grad_checkpointing": False,  # Required for attention
    "head_dim": 64,  # Dimension of attention head
    "num_head": 8,   # Number of attention heads
    "attention_dropout": 0.1,  # Dropout rate for attention
    "is_quant": False  # Explicitly disable quantization
}

# Initialize the model from Model.py
model = Model(config)

# Load the same dataset as run_tasks.py
# Replace 'task' with the actual task name you are working on
task = 'text'  # or whatever task you're running
dataset_path = f"../datasets/long-range-arena/lra_benchmarks/{task}/data"
batch_size = 8

# Load the dataset using LRADataset and DataLoader
dataset = LRADataset(dataset_path, True)
dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True)

# Get one batch of data from the dataloader
for batch in dataloader:
    input_ids = batch['input_ids']  # Assuming input_ids is the key in your dataset
    mask = batch['mask']  # Assuming mask is the key for attention masks
    break  # We only need one batch for this forward pass

# Log the forward pass using TorchLens
model_history = tl.log_forward_pass(model, (input_ids, mask), layers_to_save='all', vis_opt='unrolled')

# Print the captured model history to visualize the layers and outputs
print(model_history)