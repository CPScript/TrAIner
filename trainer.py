import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
import argparse
import random
import numpy as np
import time
import json
from pathlib import Path
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR
from modeling import ChatModel, ChatConfig, ChatTokenizer, ChatDataset, JsonlChatDataset

# === Setup Logging ===
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# === Training Arguments ===
def parse_args():
    parser = argparse.ArgumentParser(description="Train a model")
    # Model configuration
    parser.add_argument("--model_path", type=str, default=None, help="Path to load pretrained model")
    parser.add_argument("--output_dir", type=str, default="./model", help="Directory to save the model")
    parser.add_argument("--vocab_size", type=int, default=32000, help="Size of vocabulary")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--hidden_size", type=int, default=768, help="Hidden size")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--num_kv_heads", type=int, default=None, help="Number of key/value heads (for GQA)")
    parser.add_argument("--feed_forward_dim", type=int, default=3072, help="Feed forward dimension")
    parser.add_argument("--window_size", type=int, default=512, help="Sliding window size")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--use_moe", action="store_true", help="Use Mixture of Experts")
    parser.add_argument("--num_experts", type=int, default=8, help="Number of experts")
    parser.add_argument("--use_gqa", action="store_true", help="Use Grouped Query Attention")
    parser.add_argument("--use_rmsnorm", action="store_true", help="Use RMSNorm")
    parser.add_argument("--use_rotary", action="store_true", help="Use Rotary Embeddings")
    parser.add_argument("--use_flash_attn", action="store_true", help="Use Flash Attention")
    parser.add_argument("--use_sliding_window", action="store_true", help="Use Sliding Window Attention")
    
    # Training configuration
    parser.add_argument("--data_path", type=str, default="./data/dialogues.txt", help="Path to training data")
    parser.add_argument("--data_format", type=str, default="txt", choices=["txt", "jsonl"], help="Data format")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Number of warmup steps")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluation steps")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save steps")
    parser.add_argument("--scheduler", type=str, default="cosine", 
                        choices=["linear", "cosine", "plateau", "onecycle"], 
                        help="Learning rate scheduler")
    parser.add_argument("--optimized_bf16", action="store_true", help="Use BF16 mixed precision")
    
    # Distributed training
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument("--deepspeed", action="store_true", help="Use DeepSpeed")
    parser.add_argument("--deepspeed_config", type=str, default=None, help="DeepSpeed config file")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume_from_checkpoint", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--logging_steps", type=int, default=100, help="Logging steps")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 training")
    parser.add_argument("--quant_aware_training", action="store_true", help="Quantization-aware training")
    parser.add_argument("--check_dataset", action="store_true", help="Check and verify dataset")
    
    return parser.parse_args()

# === Training Setup ===
def setup_training(args):
    # Set random seed
    set_seed(args.seed)
    
    # Setup distributed training if needed
    is_distributed = (args.local_rank != -1)
    
    if is_distributed:
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        args.device = torch.device("cuda", args.local_rank)
        torch.cuda.set_device(args.device)
        args.world_size = torch.distributed.get_world_size()
        logger.info(f"Initialized distributed training with world size: {args.world_size}")
    else:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.world_size = 1
    
    # Setup data paths and output directory
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
        
    return args

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# === Dataset Handling ===
def prepare_dataset(args, tokenizer):
    if args.data_format == "txt":
        from modeling import read_texts
        texts = read_texts(args.data_path)
        dataset = ChatDataset(tokenizer, args.max_seq_length, texts)
    elif args.data_format == "jsonl":
        dataset = JsonlChatDataset(tokenizer, args.max_seq_length, args.data_path)
    else:
        raise ValueError(f"Unsupported data format: {args.data_format}")
    
    # Check dataset
    if args.check_dataset and args.local_rank in [-1, 0]:
        verify_dataset(dataset, tokenizer)
    
    # Split dataset
    train_size = int((1 - args.val_split) * len(dataset))
    val_size = len(dataset) - train_size
    
    if args.local_rank != -1:
        # For distributed training, use random split but with same seed for all processes
        generator = torch.Generator().manual_seed(args.seed)
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
        
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=min(8, os.cpu_count() or 1),
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            sampler=val_sampler,
            num_workers=min(8, os.cpu_count() or 1),
            pin_memory=True
        )
    else:
        # For single-process training, just use random split
        generator = torch.Generator().manual_seed(args.seed)
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=min(8, os.cpu_count() or 1),
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=min(8, os.cpu_count() or 1),
            pin_memory=True
        )
    
    return train_loader, val_loader, train_dataset, val_dataset

def verify_dataset(dataset, tokenizer):
    """Verify the dataset by checking a few examples"""
    logger.info("Verifying dataset...")
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        input_ids = sample["input_ids"]
        labels = sample["labels"]
        
        decoded_input = tokenizer.decode(input_ids)
        mask = labels != -100
        valid_labels = labels[mask]
        decoded_labels = tokenizer.decode(valid_labels)
        
        logger.info(f"Sample {i+1} input: {decoded_input[:100]}...")
        logger.info(f"Sample {i+1} expected output: {decoded_labels[:100]}...")
        
    logger.info("Dataset verification complete")

# === Training Functions ===
def train(args, model, train_dataloader, eval_dataloader, optimizer, scheduler, tokenizer):
    """Main training loop"""
    
    model.train()
    
    # Setup mixed precision training
    scaler = None
    if args.fp16:
        scaler = torch.cuda.amp.GradScaler()
    elif args.optimized_bf16:
        # BF16 doesn't need a scaler since it doesn't overflow
        pass
    
    # Training metrics
    total_steps = 0
    global_step = 0
    best_val_loss = float('inf')
    early_stopping_counter = 0
    gradient_accumulation_steps = args.gradient_accumulation_steps
    
    # Training loop
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        
        if isinstance(train_dataloader.sampler, DistributedSampler):
            train_dataloader.sampler.set_epoch(epoch)
        
        train_loss = 0
        train_steps = 0
        
        progress_bar = tqdm(
            train_dataloader, 
            desc=f"Epoch {epoch+1}/{args.epochs}",
            disable=args.local_rank not in [-1, 0]
        )
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(args.device) for k, v in batch.items()}
            
            # Forward pass
            if args.fp16:
                with torch.cuda.amp.autocast():
                    outputs = model(**batch)
                    loss = outputs["loss"] / gradient_accumulation_steps
            elif args.optimized_bf16:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    outputs = model(**batch)
                    loss = outputs["loss"] / gradient_accumulation_steps
            else:
                outputs = model(**batch)
                loss = outputs["loss"] / gradient_accumulation_steps
            
            # Backward pass
            if args.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update metrics
            train_loss += loss.item() * gradient_accumulation_steps
            train_steps += 1
            total_steps += 1
            
            # Accumulate gradients
            if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                # Gradient clipping
                if args.fp16:
                    scaler.unscale_(optimizer)
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                # Update weights
                if args.fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                    
                # Update learning rate
                scheduler.step()
                
                # Zero gradients
                optimizer.zero_grad()
                
                global_step += 1
                
                # Logging
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    avg_loss = train_loss / train_steps
                    lr = optimizer.param_groups[0]['lr']
                    progress_bar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'lr': f'{lr:.9f}',
                        'step': global_step
                    })
                    logger.info(f"Step {global_step}: loss={avg_loss:.4f}, lr={lr:.9f}")
                
                # Evaluation
                if args.eval_steps > 0 and global_step % args.eval_steps == 0:
                    eval_results = evaluate(args, model, eval_dataloader, tokenizer)
                    model.train()  # Set back to training mode
                    
                    val_loss = eval_results["loss"]
                    
                    if args.local_rank in [-1, 0]:
                        logger.info(f"Validation at step {global_step}: loss={val_loss:.4f}")
                        
                        # Save best model
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            early_stopping_counter = 0
                            save_checkpoint(args, model, tokenizer, f"checkpoint-best")
                        else:
                            early_stopping_counter += 1
                            logger.info(f"No improvement in validation loss. Early stopping counter: {early_stopping_counter}/{args.early_stopping_patience}")
                            
                            if args.early_stopping_patience > 0 and early_stopping_counter >= args.early_stopping_patience:
                                logger.info("Early stopping triggered")
                                return
                
                # Save checkpoint
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    save_checkpoint(args, model, tokenizer, f"checkpoint-{global_step}")
        
        # End of epoch
        epoch_time = time.time() - epoch_start_time
        avg_loss = train_loss / train_steps
        
        if args.local_rank in [-1, 0]:
            logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds")
            logger.info(f"Average training loss: {avg_loss:.4f}")
            
            # Save model for each epoch
            save_checkpoint(args, model, tokenizer, f"epoch-{epoch+1}")
            
            # Full evaluation at the end of each epoch
            eval_results = evaluate(args, model, eval_dataloader, tokenizer)
            val_loss = eval_results["loss"]
            logger.info(f"End of epoch {epoch+1} validation: loss={val_loss:.4f}")
            
            # Check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stopping_counter = 0
                save_checkpoint(args, model, tokenizer, f"checkpoint-best")
            else:
                early_stopping_counter += 1
                logger.info(f"No improvement in validation loss. Early stopping counter: {early_stopping_counter}/{args.early_stopping_patience}")
                
                if args.early_stopping_patience > 0 and early_stopping_counter >= args.early_stopping_patience:
                    logger.info("Early stopping triggered")
                    break
    
    # Save final model
    if args.local_rank in [-1, 0]:
        save_checkpoint(args, model, tokenizer, "final")
        
        # Log training completion
        logger.info(f"Training completed. Best validation loss: {best_val_loss:.4f}")
    
    # Make sure all processes are synchronized
    if args.local_rank != -1:
        torch.distributed.barrier()

def evaluate(args, model, eval_dataloader, tokenizer):
    """Evaluate the model on the validation dataset"""
    model.eval()
    
    total_loss = 0
    total_steps = 0
    total_correct_tokens = 0
    total_tokens = 0
    
    for batch in tqdm(eval_dataloader, desc="Evaluating", disable=args.local_rank not in [-1, 0]):
        batch = {k: v.to(args.device) for k, v in batch.items()}
        
        with torch.no_grad():
            outputs = model(**batch)
            
        total_loss += outputs["loss"].item()
        total_steps += 1
        
        # Calculate token prediction accuracy
        logits = outputs["logits"]
        labels = batch["labels"]
        predictions = logits.argmax(dim=-1)
        
        mask = labels != -100
        correct = (predictions == labels) & mask
        total_correct_tokens += correct.sum().item()
        total_tokens += mask.sum().item()
    
    # Average loss
    avg_loss = total_loss / total_steps
    
    # Token prediction accuracy
    token_acc = total_correct_tokens / total_tokens if total_tokens > 0 else 0
    
    # Gather metrics from all processes if distributed
    if args.local_rank != -1:
        # Convert to tensor for gather
        metrics = torch.tensor([avg_loss, token_acc, total_steps, total_tokens], device=args.device)
        gathered_metrics = [torch.zeros_like(metrics) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_metrics, metrics)
        
        # Average loss across all processes
        if args.local_rank == 0:
            avg_loss = sum(m[0] for m in gathered_metrics) / sum(m[2] for m in gathered_metrics)
            token_acc = sum(m[1] * m[3] for m in gathered_metrics) / sum(m[3] for m in gathered_metrics)
    
    results = {
        "loss": avg_loss,
        "token_accuracy": token_acc
    }
    
    return results

def save_checkpoint(args, model, tokenizer, checkpoint_name):
    """Save a model checkpoint"""
    checkpoint_dir = os.path.join(args.output_dir, checkpoint_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save model state
    if hasattr(model, "module"):  # Distributed training
        model.module.save_pretrained(checkpoint_dir)
    else:
        model.save_pretrained(checkpoint_dir)
    
    # Save tokenizer
    tokenizer.save(checkpoint_dir)
    
    # Save args
    with open(os.path.join(checkpoint_dir, "training_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    logger.info(f"Saved checkpoint: {checkpoint_dir}")
    
    return checkpoint_dir

def create_optimizer_and_scheduler(args, model, num_training_steps):
    """Create optimizer and learning rate scheduler"""
    
    # Prepare optimizer
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias", "layer_norm", "layernorm"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    
    # Prepare scheduler
    warmup_steps = args.warmup_steps
    
    if args.scheduler == "linear":
        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - warmup_steps))
            )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    elif args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=num_training_steps - warmup_steps,
            eta_min=args.learning_rate * 0.1
        )
        # Add warmup
        if warmup_steps > 0:
            def lr_lambda(current_step: int):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                return 1.0
            scheduler = torch.optim.lr_scheduler.ChainedScheduler([
                torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
                scheduler
            ])
    
    elif args.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode="min", 
            factor=0.5, 
            patience=2,
            min_lr=args.learning_rate * 0.01
        )
    
    elif args.scheduler == "onecycle":
        scheduler = OneCycleLR(
            optimizer,
            max_lr=args.learning_rate,
            total_steps=num_training_steps,
            pct_start=0.05,
            anneal_strategy="cos",
            cycle_momentum=True,
            div_factor=25.0,
            final_div_factor=10000.0
        )
    
    else:
        raise ValueError(f"Unsupported scheduler: {args.scheduler}")
    
    return optimizer, scheduler

def load_or_create_model(args, tokenizer):
    """Load a pretrained model or create a new one"""
    if args.model_path and os.path.exists(args.model_path):
        logger.info(f"Loading model from {args.model_path}")
        # Load config
        config = ChatConfig.from_pretrained(args.model_path)
        # Update config with CLI arguments if provided
        update_config_from_args(config, args)
        # Load model
        model = ChatModel.from_pretrained(args.model_path, config)
        logger.info(f"Loaded model with {sum(p.numel() for p in model.parameters())} parameters")
    else:
        logger.info("Creating new model")
        # Create new config
        config = ChatConfig(
            vocab_size=len(tokenizer.vocab),
            max_seq_length=args.max_seq_length,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            rope_dim=args.hidden_size // args.num_heads,
            feed_forward_dim=args.feed_forward_dim,
            window_size=args.window_size,
            dropout=args.dropout,
            num_experts=args.num_experts,
            expert_loss_weight=0.01,
            use_gqa=args.use_gqa,
            use_rmsnorm=args.use_rmsnorm,
            use_rotary=args.use_rotary,
            use_flash_attn=args.use_flash_attn,
            use_sliding_window=args.use_sliding_window,
            use_moe=args.use_moe
        )
        # Create model
        model = ChatModel(config)
        logger.info(f"Created new model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Check and handle quantization-aware training
    if args.quant_aware_training:
        try:
            import torch.quantization as quant
            model = quant.QuantWrapper(model)
            model.qconfig = quant.get_default_qat_qconfig("fbgemm")
            torch.quantization.prepare_qat(model, inplace=True)
            logger.info("Enabled quantization-aware training")
        except ImportError:
            logger.warning("Quantization-aware training requires PyTorch 1.8+. Disabling.")
    
    return model, config

def update_config_from_args(config, args):
    """Update config object with values from args if they're explicitly provided"""
    # Only update fields if they're explicitly provided in args
    if args.hidden_size is not None:
        config.hidden_size = args.hidden_size
    if args.num_layers is not None:
        config.num_layers = args.num_layers
    if args.num_heads is not None:
        config.num_heads = args.num_heads
    if args.num_kv_heads is not None:
        config.num_kv_heads = args.num_kv_heads
    if args.feed_forward_dim is not None:
        config.feed_forward_dim = args.feed_forward_dim
    if args.window_size is not None:
        config.window_size = args.window_size
    if args.dropout is not None:
        config.dropout = args.dropout
    if args.use_moe is not None:
        config.use_moe = args.use_moe
    if args.num_experts is not None:
        config.num_experts = args.num_experts
    if args.use_gqa is not None:
        config.use_gqa = args.use_gqa
    if args.use_rmsnorm is not None:
        config.use_rmsnorm = args.use_rmsnorm
    if args.use_rotary is not None:
        config.use_rotary = args.use_rotary
    if args.use_flash_attn is not None:
        config.use_flash_attn = args.use_flash_attn
    if args.use_sliding_window is not None:
        config.use_sliding_window = args.use_sliding_window
    
    # Update rope_dim based on hidden_size and num_heads
    config.rope_dim = config.hidden_size // config.num_heads

def main():
    # Parse arguments
    args = parse_args()
    
    # Setup distributed training and other configuration
    args = setup_training(args)
    
    # Create/load tokenizer
    if args.model_path and os.path.exists(args.model_path):
        tokenizer = ChatTokenizer.from_pretrained(args.model_path)
    else:
        tokenizer = ChatTokenizer()
        # Build vocabulary if it's a new tokenizer
        if args.data_format == "txt":
            from modeling import read_texts
            texts = read_texts(args.data_path)
            tokenizer.build_vocab(texts, max_vocab_size=args.vocab_size)
        # For JSONL, we need to read and extract text
        elif args.data_format == "jsonl":
            texts = []
            with open(args.data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        if isinstance(item, dict):
                            if "messages" in item:
                                for msg in item["messages"]:
                                    texts.append(msg.get("content", ""))
                            elif "input" in item and "output" in item:
                                texts.append(item["input"])
                                texts.append(item["output"])
                            else:
                                texts.append(str(item))
                        else:
                            texts.append(str(item))
            tokenizer.build_vocab(texts, max_vocab_size=args.vocab_size)
    
    logger.info(f"Vocabulary size: {len(tokenizer.vocab)}")
    
    # Prepare dataset
    train_dataloader, eval_dataloader, train_dataset, val_dataset = prepare_dataset(args, tokenizer)
    
    # Calculate training steps
    if args.local_rank != -1:
        num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    else:
        num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    
    num_training_steps = num_update_steps_per_epoch * args.epochs
    
    logger.info(f"Number of training examples: {len(train_dataset)}")
    logger.info(f"Number of validation examples: {len(val_dataset)}")
    logger.info(f"Number of training steps: {num_training_steps}")
    
    # Load or create model
    model, config = load_or_create_model(args, tokenizer)
    
    # Move model to device
    model = model.to(args.device)
    
    # Setup distributed training if needed
    if args.local_rank != -1:
        model = DDP(
            model, 
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True
        )
    
    # Log model configuration
    if args.local_rank in [-1, 0]:
        logger.info(f"Model configuration: {config.__dict__}")
    
    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(args, model, num_training_steps)
    
    # Train the model
    train(args, model, train_dataloader, eval_dataloader, optimizer, scheduler, tokenizer)
    
    # Final sync in distributed mode
    if args.local_rank != -1:
        torch.distributed.barrier()
    
    logger.info("Training complete!")

if __name__ == "__main__":
    main()