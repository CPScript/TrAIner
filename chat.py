import os
import time
import torch
import argparse
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from modeling import ChatModel, ChatTokenizer, ChatConfig

# === Setup Logging ===
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class SpeculativeDecoder:
    """Fast inference using speculative decoding"""
    
    def __init__(self, model, tokenizer, draft_model=None, spec_len=5, top_k=5):
        self.model = model
        self.tokenizer = tokenizer
        self.draft_model = draft_model or model  # Use same model if no draft provided
        self.spec_len = spec_len
        self.top_k = top_k
        self.device = next(model.parameters()).device
    
    def generate(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
        **kwargs
    ) -> torch.Tensor:
        """Generate tokens using speculative decoding"""
        # Initialize
        batch_size, seq_len = input_ids.shape
        past_key_values = None
        generated_tokens = []
        
        # Track token positions for repetition penalty
        all_tokens = set(input_ids[0].tolist())
        
        # Generate in chunks of spec_len
        for i in range(0, max_new_tokens, self.spec_len):
            # Adjust remaining length
            current_spec_len = min(self.spec_len, max_new_tokens - i)
            if current_spec_len <= 0:
                break
                
            # Generate draft tokens with draft model
            draft_tokens = self._generate_draft(
                input_ids, 
                current_spec_len, 
                temperature,
                top_p
            )
            
            # Verify draft tokens with main model
            verified_tokens, past_key_values = self._verify_draft(
                input_ids,
                draft_tokens,
                temperature,
                top_p,
                repetition_penalty,
                all_tokens,
                past_key_values
            )
            
            # Update input_ids with verified tokens
            input_ids = torch.cat([input_ids, verified_tokens], dim=1)
            
            # Update all_tokens for repetition penalty
            all_tokens.update(verified_tokens[0].tolist())
            
            # Add to generated tokens
            generated_tokens.append(verified_tokens)
            
            # Check for stopping criteria
            if verified_tokens.shape[1] < current_spec_len:
                # The verification stopped early (e.g., EOS token)
                break
                
        # Combine all generated tokens
        if generated_tokens:
            return torch.cat([input_ids[:, :seq_len], torch.cat(generated_tokens, dim=1)], dim=1)
        else:
            return input_ids
    
    def _generate_draft(
        self, 
        input_ids: torch.Tensor, 
        num_tokens: int, 
        temperature: float,
        top_p: float
    ) -> torch.Tensor:
        """Generate draft tokens with the draft model"""
        with torch.no_grad():
            # Start with the original input
            draft_input = input_ids.clone()
            draft_tokens = []
            
            for _ in range(num_tokens):
                # Get logits from draft model
                outputs = self.draft_model(draft_input, use_cache=True)
                logits = outputs["logits"][:, -1, :]
                
                # Temperature scaling
                if temperature > 0:
                    logits = logits / temperature
                
                # Top-p sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Keep the first token above the threshold
                    sorted_indices_to_remove[..., 0] = False
                    
                    # Scatter sorted tensors to original indexing
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        dim=-1, 
                        index=sorted_indices, 
                        src=sorted_indices_to_remove
                    )
                    logits = logits.masked_fill(indices_to_remove, float('-inf'))
                
                # Sample next token
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to draft tokens
                draft_tokens.append(next_token)
                
                # Update draft input for next iteration
                draft_input = torch.cat([draft_input, next_token], dim=1)
            
            return torch.cat(draft_tokens, dim=1)
    
    def _verify_draft(
        self, 
        input_ids: torch.Tensor, 
        draft_tokens: torch.Tensor,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        all_tokens: set,
        past_key_values: Optional[Tuple] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        """Verify draft tokens using the main model"""
        verified_tokens = []
        current_input = input_ids.clone()
        
        with torch.no_grad():
            # Iterate through draft tokens and verify them
            for i in range(draft_tokens.shape[1]):
                # Get next draft token
                draft_token = draft_tokens[:, i:i+1]
                
                # Forward pass with main model
                outputs = self.model(
                    current_input,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                logits = outputs["logits"][:, -1, :]
                past_key_values = outputs.get("past_key_values", None)
                
                # Apply temperature
                if temperature > 0:
                    logits = logits / temperature
                
                # Apply repetition penalty
                if repetition_penalty > 1.0:
                    for token_id in all_tokens:
                        logits[0, token_id] /= repetition_penalty
                
                # Apply top-p
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Keep the first token above the threshold
                    sorted_indices_to_remove[..., 0] = False
                    
                    # Scatter sorted tensors to original indexing
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        dim=-1, 
                        index=sorted_indices, 
                        src=sorted_indices_to_remove
                    )
                    logits = logits.masked_fill(indices_to_remove, float('-inf'))
                
                # Get top-k tokens from the predicted distribution
                probs = torch.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, k=self.top_k, dim=-1)
                
                # Check if the draft token is among the top-k tokens
                draft_token_value = draft_token[0, 0].item()
                if draft_token_value in topk_indices[0]:
                    # Accept draft token since it's in the top-k
                    verified_tokens.append(draft_token)
                    current_input = torch.cat([current_input, draft_token], dim=1)
                else:
                    # Sample from the model's distribution
                    next_token = torch.multinomial(probs, num_samples=1)
                    verified_tokens.append(next_token)
                    current_input = torch.cat([current_input, next_token], dim=1)
                    
                    # If we had to reject, we break out and continue with normal
                    # generation since the next tokens are likely to be rejected too
                    break
        
        # Return verified tokens
        return torch.cat(verified_tokens, dim=1) if verified_tokens else torch.zeros((input_ids.shape[0], 0), dtype=torch.long, device=input_ids.device), past_key_values


class KVCache:
    """Manages the KV cache for efficient inference"""
    
    def __init__(self, max_seq_len=4096, dtype=torch.float16):
        self.cache = {}
        self.max_seq_len = max_seq_len
        self.dtype = dtype
    
    def update(self, key, new_entries):
        """Update cache with new key-value entries"""
        if key not in self.cache:
            self.cache[key] = new_entries
        else:
            # Concatenate with existing cache
            current = self.cache[key]
            if isinstance(current, tuple) and isinstance(new_entries, tuple):
                self.cache[key] = tuple(torch.cat([c, n], dim=1) for c, n in zip(current, new_entries))
            else:
                # For other types of cache entries
                self.cache[key] = torch.cat([current, new_entries], dim=1)
            
            # Prune if too long
            if self.get_size(key) > self.max_seq_len:
                self.prune(key)
    
    def get(self, key):
        """Retrieve cache for a given key"""
        return self.cache.get(key, None)
    
    def get_size(self, key):
        """Get the sequence length in the cache"""
        if key not in self.cache:
            return 0
        
        cache_value = self.cache[key]
        if isinstance(cache_value, tuple):
            return cache_value[0].shape[1]  # Take sequence dim from first element
        else:
            return cache_value.shape[1]
    
    def prune(self, key, keep_last=1024):
        """Prune cache to keep only the most recent tokens"""
        if key not in self.cache:
            return
        
        cache_value = self.cache[key]
        if isinstance(cache_value, tuple):
            self.cache[key] = tuple(c[:, -keep_last:, ...] for c in cache_value)
        else:
            self.cache[key] = cache_value[:, -keep_last:, ...]
    
    def clear(self):
        """Clear the entire cache"""
        self.cache = {}


def load_model_and_tokenizer(model_path, device_map="auto", use_quantization=False, use_benchmark=True):
    """Load model and tokenizer with various optimizations"""
    # Set PyTorch benchmarking
    if use_benchmark and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    # Load config
    logger.info(f"Loading model from {model_path}")
    config = ChatConfig.from_pretrained(model_path)
    
    # Load tokenizer
    tokenizer = ChatTokenizer.from_pretrained(model_path)
    
    # Set up device
    if device_map == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU")
    else:
        device = torch.device(device_map)
    
    # Load model with optimizations
    if use_quantization and torch.cuda.is_available():
        try:
            # Try to use optimized kernels
            import torch.ao.quantization as quantization
            model = ChatModel.from_pretrained(model_path, config)
            model.eval()
            
            # Quantize the model (8-bit)
            quantized_model = quantization.quantize_dynamic(
                model, 
                {torch.nn.Linear}, 
                dtype=torch.qint8
            )
            model = quantized_model
            logger.info("Using 8-bit quantization")
        except Exception as e:
            logger.warning(f"Quantization failed: {e}")
            model = ChatModel.from_pretrained(model_path, config)
    else:
        model = ChatModel.from_pretrained(model_path, config)
    
    # Move to device and set to evaluation mode
    model = model.to(device)
    model.eval()
    
    return model, tokenizer, device


def chat(
    text: str,
    model: ChatModel,
    tokenizer: ChatTokenizer,
    device: torch.device,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 50,
    repetition_penalty: float = 1.2,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    use_cache: bool = True,
    use_speculative: bool = False,
    spec_len: int = 5,
    streamed: bool = True,
    convert_to_tensors: bool = True
):
    """Generate a chat response"""
    # Format the input prompt
    prompt = f"<|user|>{text}<|assistant|>"
    
    # Tokenize the prompt
    tokens = tokenizer.tokenize(prompt)
    input_ids = tokenizer.convert_tokens_to_ids(tokens, update_vocab=False)
    
    if convert_to_tensors:
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    # Set up stopping criteria - stop at end token
    end_token_id = tokenizer.vocab.get("<|end|>", -1)
    
    # Set up callback for streaming output
    output_text = ""
    
    def streaming_callback(token_ids):
        nonlocal output_text
        # Decode and print token
        new_text = tokenizer.decode(token_ids[0], skip_special_tokens=False)
        
        if new_text == "<|end|>":
            return
        
        output_text += new_text
        if streamed:
            print(new_text, end="", flush=True)
    
    # Generate response
    if streamed:
        streaming_callback = streaming_callback
    else:
        streaming_callback = None
    
    # Use speculative decoding if enabled
    if use_speculative:
        decoder = SpeculativeDecoder(
            model=model,
            tokenizer=tokenizer,
            spec_len=spec_len,
            top_k=top_k
        )
        
        start_time = time.time()
        output_ids = decoder.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        )
        generation_time = time.time() - start_time
        
        # Decode output
        generated_text = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=False)
        
        # Print output if not streaming
        if not streamed:
            print(generated_text)
        
        # For benchmarking
        tokens_per_second = output_ids.shape[1] - input_ids.shape[1] / generation_time if generation_time > 0 else 0
        logger.info(f"Generated {output_ids.shape[1] - input_ids.shape[1]} tokens in {generation_time:.2f}s ({tokens_per_second:.2f} tokens/s)")
        
        return generated_text
    
    # Standard generation
    else:
        start_time = time.time()
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                use_cache=use_cache,
                callback=streaming_callback
            )
        generation_time = time.time() - start_time
        
        # For benchmarking
        tokens_per_second = output_ids.shape[1] - input_ids.shape[1] / generation_time if generation_time > 0 else 0
        logger.info(f"Generated {output_ids.shape[1] - input_ids.shape[1]} tokens in {generation_time:.2f}s ({tokens_per_second:.2f} tokens/s)")
        
        # Decode output if we're not streaming
        if not streamed:
            generated_text = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=False)
            print(generated_text)
            return generated_text
        
        return output_text


def chat_interactive(
    model: ChatModel,
    tokenizer: ChatTokenizer,
    device: torch.device,
    **generation_kwargs
):
    """Interactive chat session"""
    print("\n" + "=" * 50)
    print("TrAIner Chat Interface")
    print("Type 'exit' to quit, 'clear' to start a new conversation")
    print("=" * 50 + "\n")
    
    conversation_history = ""
    
    while True:
        text = input("\n> ")
        
        if text.strip().lower() == "exit":
            break
        
        if text.strip().lower() == "clear":
            conversation_history = ""
            print("\nConversation cleared. Starting new chat.\n")
            continue
        
        # Append user input to conversation
        conversation_history += f"\n<|user|>{text}"
        
        # Generate response
        print("\n", end="")
        response = chat(
            conversation_history,
            model,
            tokenizer,
            device,
            **generation_kwargs
        )
        
        # Append response to conversation
        if isinstance(response, str):
            conversation_history += f"<|assistant|>{response}<|end|>"
        
        print("\n")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="TrAIner Chat Interface")
    
    # Model loading options
    parser.add_argument("--model_path", type=str, default="./model/final", help="Path to the model")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (cuda, cpu, or auto)")
    parser.add_argument("--quantize", action="store_true", help="Use quantization for faster inference")
    parser.add_argument("--no_benchmark", action="store_true", help="Disable cuDNN benchmarking")
    
    # Generation options
    parser.add_argument("--max_tokens", type=int, default=1024, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Nucleus sampling probability")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--repetition_penalty", type=float, default=1.2, help="Repetition penalty")
    parser.add_argument("--presence_penalty", type=float, default=0.0, help="Presence penalty")
    parser.add_argument("--frequency_penalty", type=float, default=0.0, help="Frequency penalty")
    
    # Advanced options
    parser.add_argument("--no_cache", action="store_true", help="Disable KV caching")
    parser.add_argument("--use_speculative", action="store_true", help="Use speculative decoding")
    parser.add_argument("--spec_len", type=int, default=5, help="Speculative decoding length")
    
    # Interface options
    parser.add_argument("--no_stream", action="store_true", help="Disable streaming output")
    parser.add_argument("--input", type=str, default=None, help="Input text (if not using interactive mode)")
    parser.add_argument("--no_interactive", action="store_true", help="Disable interactive mode")
    
    # Debug options
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load model and tokenizer
    model, tokenizer, device = load_model_and_tokenizer(
        args.model_path, 
        device_map=args.device,
        use_quantization=args.quantize,
        use_benchmark=not args.no_benchmark
    )
    
    # Set up generation parameters
    generation_kwargs = {
        "max_new_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "presence_penalty": args.presence_penalty,
        "frequency_penalty": args.frequency_penalty,
        "use_cache": not args.no_cache,
        "use_speculative": args.use_speculative,
        "spec_len": args.spec_len,
        "streamed": not args.no_stream
    }
    
    # Check if we're processing a single input or running in interactive mode
    if args.input is not None:
        # Process single input
        chat(args.input, model, tokenizer, device, **generation_kwargs)
    elif not args.no_interactive:
        # Run interactive mode
        chat_interactive(model, tokenizer, device, **generation_kwargs)
    else:
        logger.error("No input provided and interactive mode disabled. Nothing to do.")
        exit(1)


if __name__ == "__main__":
    main()