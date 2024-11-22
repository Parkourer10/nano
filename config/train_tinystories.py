# Configuration for 22M parameter TinyStories model
out_dir = 'out-tinystories'
eval_interval = 1000 
eval_iters = 100
log_interval = 100 
always_save_checkpoint = True 

# Weights & Biases logging
wandb_log = False
wandb_project = 'tinystories-22m'
wandb_run_name = 'gpt-22m'

# Dataset config
dataset = 'tinystories'
gradient_accumulation_steps = 4 
batch_size = 64
block_size = 512 

n_layer = 12
n_head = 12
n_embd = 768  # d_model
dropout = 0.1 

# Training parameters
learning_rate = 3e-4 
max_iters = 10000  
lr_decay_iters = 50000
min_lr = 3e-5 
beta2 = 0.95 
warmup_iters = 1000 

device = 'cuda'
dtype = 'float16'  
compile = False    
