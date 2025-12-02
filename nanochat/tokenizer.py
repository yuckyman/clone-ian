"""
tokenizer wrapper - uses tiktoken (gpt-4 style)
"""

import tiktoken
from typing import List, Optional


class Tokenizer:
    """simple tokenizer wrapper around tiktoken"""
    
    def __init__(self, encoding_name: str = "cl100k_base"):
        """
        encoding_name: tiktoken encoding to use
        - cl100k_base: gpt-4 style (default)
        - p50k_base: gpt-3.5 style
        - r50k_base: older gpt-3 style
        """
        self.enc = tiktoken.get_encoding(encoding_name)
        self.vocab_size = self.enc.n_vocab
        
        # special tokens for chat
        # tiktoken already has these, but we'll define them for clarity
        self.im_start = "<|im_start|>"
        self.im_end = "<|im_end|>"
        
        # get token ids
        try:
            self.im_start_id = self.enc.encode(self.im_start)[0]
            self.im_end_id = self.enc.encode(self.im_end)[0]
        except:
            # fallback if tokens don't exist
            self.im_start_id = None
            self.im_end_id = None
    
    def encode(self, text: str) -> List[int]:
        """encode text to token ids"""
        return self.enc.encode(text, allowed_special="all")
    
    def decode(self, token_ids: List[int]) -> str:
        """decode token ids to text"""
        return self.enc.decode(token_ids)
    
    def encode_chatml(self, messages: List[dict]) -> List[int]:
        """
        encode chatml format conversation
        messages: list of {"role": "user|assistant", "content": "..."}
        """
        tokens = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # add role token
            if self.im_start_id is not None:
                tokens.extend(self.encode(self.im_start))
            tokens.extend(self.encode(role))
            tokens.append(self.enc._special_tokens["<|newline|>"])
            
            # add content
            tokens.extend(self.encode(content))
            
            # add end token
            if self.im_end_id is not None:
                tokens.extend(self.encode(self.im_end))
            tokens.append(self.enc._special_tokens["<|newline|>"])
        
        return tokens

