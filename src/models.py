"""
Model interfaces for uncertainty quantification experiments.
"""

import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    GPT2LMHeadModel, GPT2Tokenizer
)
from typing import List, Dict, Optional, Tuple
import numpy as np


class ModelInterface:
    """Base interface for language models."""
    
    def __init__(self, model_name: str, device: str = 'cpu'):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.load_model()
    
    def load_model(self):
        """Load model and tokenizer."""
        raise NotImplementedError
    
    def get_logits(self, input_text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get model logits for input text.
        
        Returns:
            logits: [seq_len, vocab_size]
            token_ids: [seq_len]
        """
        raise NotImplementedError
    
    def generate_with_logits(self, prompt: str, max_length: int = 50) -> Dict:
        """Generate text and return logits for uncertainty analysis."""
        raise NotImplementedError


class GPT2Interface(ModelInterface):
    """Interface for GPT-2 models."""
    
    def load_model(self):
        """Load GPT-2 model and tokenizer."""
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.model.to(self.device)
        self.model.eval()
        
        # Add pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def get_logits(self, input_text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get logits for input text."""
        inputs = self.tokenizer(input_text, return_tensors='pt', truncation=True)
        token_ids = inputs['input_ids'][0].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(token_ids.unsqueeze(0))
            logits = outputs.logits[0]  # [seq_len, vocab_size]
        
        return logits[:-1], token_ids[1:]  # Exclude last logit, first token
    
    def generate_with_logits(self, prompt: str, max_length: int = 50) -> Dict:
        """Generate text and collect logits."""
        inputs = self.tokenizer(prompt, return_tensors='pt')
        input_ids = inputs['input_ids'].to(self.device)
        
        generated_ids = []
        all_logits = []
        
        current_ids = input_ids
        
        for _ in range(max_length):
            with torch.no_grad():
                outputs = self.model(current_ids)
                logits = outputs.logits[0, -1]  # Last token logits
                
            # Sample next token (greedy for reproducibility)
            next_token = torch.argmax(logits).unsqueeze(0).unsqueeze(0)
            
            generated_ids.append(next_token.item())
            all_logits.append(logits)
            
            current_ids = torch.cat([current_ids, next_token], dim=1)
            
            # Stop at EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        generated_text = self.tokenizer.decode(generated_ids)
        logits_tensor = torch.stack(all_logits)
        
        return {
            'generated_text': generated_text,
            'generated_ids': torch.tensor(generated_ids),
            'logits': logits_tensor,
            'full_text': prompt + generated_text
        }


class HuggingFaceInterface(ModelInterface):
    """Generic interface for HuggingFace models."""
    
    def __init__(self, model_name: str, device: str = 'cpu'):
        self.hf_model_name = model_name
        super().__init__(model_name, device)
    
    def load_model(self):
        """Load HuggingFace model and tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.hf_model_name,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                device_map='auto' if self.device == 'cuda' else None
            )
            if self.device == 'cpu':
                self.model.to(self.device)
            self.model.eval()
            
            # Add pad token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
        except Exception as e:
            print(f"Error loading model {self.hf_model_name}: {e}")
            print("Falling back to GPT-2...")
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.model = GPT2LMHeadModel.from_pretrained('gpt2')
            self.model.to(self.device)
            self.model.eval()
    
    def get_logits(self, input_text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get logits for input text."""
        inputs = self.tokenizer(input_text, return_tensors='pt', truncation=True, max_length=512)
        token_ids = inputs['input_ids'][0].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(token_ids.unsqueeze(0))
            logits = outputs.logits[0]
        
        return logits[:-1], token_ids[1:]
    
    def generate_with_logits(self, prompt: str, max_length: int = 50) -> Dict:
        """Generate text and collect logits."""
        inputs = self.tokenizer(prompt, return_tensors='pt')
        input_ids = inputs['input_ids'].to(self.device)
        
        generated_ids = []
        all_logits = []
        current_ids = input_ids
        
        for _ in range(max_length):
            with torch.no_grad():
                outputs = self.model(current_ids)
                logits = outputs.logits[0, -1]
            
            next_token = torch.argmax(logits).unsqueeze(0).unsqueeze(0)
            generated_ids.append(next_token.item())
            all_logits.append(logits)
            
            current_ids = torch.cat([current_ids, next_token], dim=1)
            
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        logits_tensor = torch.stack(all_logits) if all_logits else torch.empty(0, self.model.config.vocab_size)
        
        return {
            'generated_text': generated_text,
            'generated_ids': torch.tensor(generated_ids),
            'logits': logits_tensor,
            'full_text': prompt + generated_text
        }


class ModelFactory:
    """Factory for creating model interfaces."""
    
    SUPPORTED_MODELS = {
        'gpt2': GPT2Interface,
        'microsoft/DialoGPT-medium': HuggingFaceInterface,
        'Qwen/Qwen2.5-3B': HuggingFaceInterface,
        'google/gemma-2-2b': HuggingFaceInterface,
        'HuggingFaceTB/SmolLM2-1.7B': HuggingFaceInterface,
    }
    
    @classmethod
    def create_model(cls, model_name: str, device: str = 'cpu') -> ModelInterface:
        """Create model interface."""
        if model_name in cls.SUPPORTED_MODELS:
            if model_name == 'gpt2':
                return cls.SUPPORTED_MODELS[model_name](model_name, device)
            else:
                return cls.SUPPORTED_MODELS[model_name](model_name, device)
        else:
            # Try as generic HuggingFace model
            return HuggingFaceInterface(model_name, device)
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """Get list of supported models."""
        return list(cls.SUPPORTED_MODELS.keys())