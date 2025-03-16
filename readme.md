During testing my "Thinkpad t440" Laptop was making strange noises. It is **highly recommended** to **not use this** on anything **near** or **like** a thinkpad... Or just **laptops** in general.

---

# TrAIner: Usage Guide

This version of TrAIner allows you to efficiently train and run inference with transformer-based language models on personal computers. Below is a guide on how to use this system.

## Environment Setup

First, install the required dependencies:

```bash
pip install torch safetensors tqdm numpy
```

For optimal performance with CUDA support (if you have an NVIDIA GPU):

```bash
pip install torch --extra-index-url https://download.pytorch.org/whl/cu118
```

## Project Structure

```
TrAIner/
├── data/ 
│   └── dialogues.txt (Training data)
│
├── model/ 
│   └── ... (Saved model checkpoints)
│
├── modeling.py (Core model architecture)
├── train.py (Training script)
├── chat.py (Inference script)
```

## Training a Model

### Preparing Training Data

Place your training data in the `data/` directory. The system supports two formats:

1. **Text format** (`dialogues.txt`): Each conversation should follow this pattern:
   ```
   <|user|>User message<|assistant|>Assistant response<|end|>
   ```

2. **JSONL format**: Each line contains a JSON object with conversation data.

### Running Training

Basic training command:

```bash
python train.py --data_path ./data/dialogues.txt
```

Advanced training options:

```bash
python train.py \
  --data_path ./data/dialogues.txt \
  --data_format txt \
  --output_dir ./model \
  --hidden_size 768 \
  --num_layers 12 \
  --num_heads 12 \
  --feed_forward_dim 3072 \
  --batch_size 4 \
  --epochs 30 \
  --learning_rate 5e-5 \
  --use_moe \
  --use_rotary \
  --use_rmsnorm \
  --fp16
```

For distributed training on multiple GPUs:

```bash
python -m torch.distributed.launch --nproc_per_node=2 train.py --local_rank=0
```

## Running Inference

After training, you can chat with your model:

```bash
python chat.py --model_path ./model/final
```

This launches an interactive chat interface where you can type messages and receive responses from the model.

### Inference Options

For better performance or different response styles:

```bash
python chat.py \
  --model_path ./model/final \
  --temperature 0.7 \
  --top_p 0.95 \
  --max_tokens 1024 \
  --repetition_penalty 1.2 \
  --use_speculative
```

For non-interactive use with a specific prompt:

```bash
python chat.py --model_path ./model/final --input "Write a poem about AI"
```

## Key Parameters

### Training Parameters

- `--hidden_size`: Size of hidden layers (default: 768)
- `--num_layers`: Number of transformer layers (default: 12)
- `--num_heads`: Number of attention heads (default: 12)
- `--learning_rate`: Learning rate (default: 5e-5)
- `--batch_size`: Batch size per device (default: 4)
- `--use_moe`: Enable Mixture of Experts for better quality
- `--fp16` or `--optimized_bf16`: Enable mixed precision training

### Inference Parameters

- `--temperature`: Controls randomness (lower = more deterministic)
- `--top_p`: Nucleus sampling probability threshold
- `--repetition_penalty`: Penalizes repetition (higher = less repetition)
- `--max_tokens`: Maximum new tokens to generate
- `--use_speculative`: Enable speculative decoding for faster inference

## Example Workflow

1. **Prepare your data**:
   - Create conversation samples in `data/dialogues.txt`

2. **Train the model**:
   ```bash
   python train.py --data_path ./data/dialogues.txt --epochs 20
   ```

3. **Chat with the model**:
   ```bash
   python chat.py --model_path ./model/final
   ```

For more detailed information about available options, use the help flag:
```bash
python train.py --help
python chat.py --help
```
