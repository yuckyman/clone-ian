"""
simple chat cli for trained model
"""

import torch
import argparse
from pathlib import Path

import sys
from pathlib import Path

# add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from nanochat.gpt import GPT
    from nanochat.config import GPTConfig
    from nanochat.tokenizer import Tokenizer
except ImportError:
    # fallback for direct execution
    from gpt import GPT
    from config import GPTConfig
    from tokenizer import Tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='out/ckpt.pt', help='checkpoint path')
    parser.add_argument('--temperature', type=float, default=0.8, help='sampling temperature')
    parser.add_argument('--top_k', type=int, default=200, help='top-k sampling')
    parser.add_argument('--max_tokens', type=int, default=256, help='max tokens to generate')
    args = parser.parse_args()
    
    # device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"using device: {device}")
    
    # load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"error: checkpoint not found at {checkpoint_path}")
        return
    
    print(f"loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # model
    model = GPT(config)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    print("model loaded!")
    
    # tokenizer
    tokenizer = Tokenizer()
    
    # chat loop
    print("\n" + "="*50)
    print("chat with your model! (type 'quit' to exit)")
    print("="*50 + "\n")
    
    conversation = []
    
    while True:
        # get user input
        user_input = input("you: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        
        if not user_input:
            continue
        
        # add to conversation
        conversation.append({"role": "user", "content": user_input})
        
        # format as chatml
        chatml_text = ""
        for msg in conversation:
            chatml_text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
        chatml_text += "<|im_start|>assistant\n"
        
        # encode
        tokens = tokenizer.encode(chatml_text)
        tokens_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
        
        # generate
        with torch.no_grad():
            generated = model.generate(
                tokens_tensor,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                stop_token=tokenizer.im_end_id
            )
        
        # decode response
        response_tokens = generated[0][len(tokens):].tolist()
        if tokenizer.im_end_id and tokenizer.im_end_id in response_tokens:
            response_tokens = response_tokens[:response_tokens.index(tokenizer.im_end_id)]
        
        response = tokenizer.decode(response_tokens).strip()
        print(f"model: {response}\n")
        
        # add to conversation
        conversation.append({"role": "assistant", "content": response})
        
        # limit conversation length
        if len(conversation) > 10:
            conversation = conversation[-10:]


if __name__ == "__main__":
    main()

