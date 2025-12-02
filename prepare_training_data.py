#!/usr/bin/env python3
"""
Convert iMessage corpus into training data for language model fine-tuning.
Formats data for LoRA personality grafting - training a model to sound like you.
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from collections import defaultdict

# Configuration
INPUT_JSONL = Path("imessage_corpus.jsonl")
OUTPUT_DIR = Path("training_data")
YOUR_NAME = "ian t."  # Your sender name to identify your responses
CONVERSATION_GAP_HOURS = 2  # Messages within this time are considered same conversation
MIN_MESSAGE_LENGTH = 3  # Minimum characters for a message to be included
MAX_CONVERSATION_LENGTH = 50  # Max messages per conversation thread

# Output formats
OUTPUT_DIR.mkdir(exist_ok=True)


def parse_timestamp(ts_str: str) -> Optional[datetime]:
    """Parse ISO8601 timestamp string."""
    try:
        return datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
    except Exception:
        return None


def is_valid_message(text: str) -> bool:
    """Check if message should be included in training data."""
    if not text or len(text.strip()) < MIN_MESSAGE_LENGTH:
        return False
    # Filter out very short responses that don't add much
    if len(text.strip()) < 3 and text.strip().lower() not in ['ok', 'yes', 'no', 'lol', 'haha']:
        return False
    return True


def group_into_conversations(messages: List[Dict]) -> List[List[Dict]]:
    """Group messages into conversation threads based on time gaps."""
    conversations = []
    current_conversation = []
    last_timestamp = None
    
    for msg in messages:
        timestamp = parse_timestamp(msg.get("timestamp", ""))
        if not timestamp:
            continue
            
        # If gap is too large, start new conversation
        if last_timestamp:
            gap = timestamp - last_timestamp
            if gap > timedelta(hours=CONVERSATION_GAP_HOURS):
                if current_conversation:
                    conversations.append(current_conversation)
                current_conversation = []
        
        current_conversation.append(msg)
        last_timestamp = timestamp
        
        # Limit conversation length
        if len(current_conversation) >= MAX_CONVERSATION_LENGTH:
            conversations.append(current_conversation)
            current_conversation = []
    
    if current_conversation:
        conversations.append(current_conversation)
    
    return conversations


def format_alpaca(conversations: List[List[Dict]]) -> List[Dict]:
    """Format as Alpaca instruction/response pairs."""
    examples = []
    
    for conv in conversations:
        # Extract pairs where user says something, then you respond
        for i in range(len(conv) - 1):
            user_msg = conv[i]
            your_msg = conv[i + 1]
            
            if user_msg.get("sender") == YOUR_NAME:
                continue
            if your_msg.get("sender") != YOUR_NAME:
                continue
            
            user_text = user_msg.get("text", "").strip()
            your_text = your_msg.get("text", "").strip()
            
            if not is_valid_message(user_text) or not is_valid_message(your_text):
                continue
            
            examples.append({
                "instruction": "Respond to this message in your typical conversational style:",
                "input": user_text,
                "output": your_text
            })
    
    return examples


def format_sharegpt(conversations: List[List[Dict]]) -> List[Dict]:
    """Format as ShareGPT conversational format."""
    examples = []
    
    for conv in conversations:
        # Filter to only valid messages
        valid_conv = [msg for msg in conv if is_valid_message(msg.get("text", ""))]
        if len(valid_conv) < 2:
            continue
        
        # Build conversation with role labels
        conversations_list = []
        for msg in valid_conv:
            role = "assistant" if msg.get("sender") == YOUR_NAME else "user"
            content = msg.get("text", "").strip()
            conversations_list.append({
                "from": role,
                "value": content
            })
        
        # Only include if there's at least one of your responses
        if any(item["from"] == "assistant" for item in conversations_list):
            examples.append({
                "conversations": conversations_list
            })
    
    return examples


def format_chatml(conversations: List[List[Dict]]) -> List[str]:
    """Format as ChatML text format."""
    examples = []
    
    for conv in conversations:
        valid_conv = [msg for msg in conv if is_valid_message(msg.get("text", ""))]
        if len(valid_conv) < 2:
            continue
        
        # Build ChatML format
        lines = []
        for msg in valid_conv:
            role = "assistant" if msg.get("sender") == YOUR_NAME else "user"
            content = msg.get("text", "").strip()
            lines.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        
        # Only include if there's at least one of your responses
        if any("assistant" in line for line in lines):
            examples.append("\n".join(lines))
    
    return examples


def format_instruction_pairs(conversations: List[List[Dict]]) -> List[Dict]:
    """Format as simple instruction/response pairs with context."""
    examples = []
    
    for conv in conversations:
        valid_conv = [msg for msg in conv if is_valid_message(msg.get("text", ""))]
        if len(valid_conv) < 2:
            continue
        
        # Extract all pairs where you respond
        for i in range(len(valid_conv) - 1):
            if valid_conv[i + 1].get("sender") != YOUR_NAME:
                continue
            
            # Build context from previous messages in conversation
            context_messages = []
            for j in range(max(0, i - 3), i + 1):  # Last 3 messages as context
                msg = valid_conv[j]
                role = "You" if msg.get("sender") == YOUR_NAME else "Them"
                context_messages.append(f"{role}: {msg.get('text', '').strip()}")
            
            context = "\n".join(context_messages)
            your_response = valid_conv[i + 1].get("text", "").strip()
            
            examples.append({
                "context": context,
                "response": your_response
            })
    
    return examples


def main():
    print("📱 loading messages...")
    
    if not INPUT_JSONL.exists():
        print(f"❌ error: {INPUT_JSONL} not found", file=sys.stderr)
        sys.exit(1)
    
    # Load all messages
    messages = []
    with INPUT_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                msg = json.loads(line.strip())
                if msg.get("text") and is_valid_message(msg.get("text", "")):
                    messages.append(msg)
            except json.JSONDecodeError:
                continue
    
    print(f"  ✓ loaded {len(messages):,} messages")
    
    # Count your messages
    your_messages = [m for m in messages if m.get("sender") == YOUR_NAME]
    print(f"  ✓ found {len(your_messages):,} messages from {YOUR_NAME}")
    
    # Group into conversations
    print(f"\n💬 grouping into conversations (max gap: {CONVERSATION_GAP_HOURS}h)...")
    conversations = group_into_conversations(messages)
    print(f"  ✓ created {len(conversations):,} conversation threads")
    
    # Filter conversations that have your responses
    conversations_with_you = [
        conv for conv in conversations 
        if any(msg.get("sender") == YOUR_NAME for msg in conv)
    ]
    print(f"  ✓ {len(conversations_with_you):,} conversations include your responses")
    
    # Generate different formats
    print(f"\n📝 generating training formats...")
    
    # Alpaca format
    alpaca_data = format_alpaca(conversations_with_you)
    alpaca_path = OUTPUT_DIR / "alpaca_format.jsonl"
    with alpaca_path.open("w", encoding="utf-8") as f:
        for item in alpaca_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  ✓ alpaca format: {len(alpaca_data):,} examples → {alpaca_path}")
    
    # ShareGPT format
    sharegpt_data = format_sharegpt(conversations_with_you)
    sharegpt_path = OUTPUT_DIR / "sharegpt_format.jsonl"
    with sharegpt_path.open("w", encoding="utf-8") as f:
        for item in sharegpt_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  ✓ sharegpt format: {len(sharegpt_data):,} examples → {sharegpt_path}")
    
    # ChatML format
    chatml_data = format_chatml(conversations_with_you)
    chatml_path = OUTPUT_DIR / "chatml_format.txt"
    with chatml_path.open("w", encoding="utf-8") as f:
        for item in chatml_data:
            f.write(item + "\n\n")
    print(f"  ✓ chatml format: {len(chatml_data):,} examples → {chatml_path}")
    
    # Instruction pairs with context
    instruction_data = format_instruction_pairs(conversations_with_you)
    instruction_path = OUTPUT_DIR / "instruction_pairs.jsonl"
    with instruction_path.open("w", encoding="utf-8") as f:
        for item in instruction_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  ✓ instruction pairs: {len(instruction_data):,} examples → {instruction_path}")
    
    # Stats
    print(f"\n📊 statistics:")
    print(f"  • total messages: {len(messages):,}")
    print(f"  • your messages: {len(your_messages):,}")
    print(f"  • conversation threads: {len(conversations_with_you):,}")
    print(f"  • training examples (alpaca): {len(alpaca_data):,}")
    print(f"  • training examples (sharegpt): {len(sharegpt_data):,}")
    
    # Sample output
    if alpaca_data:
        print(f"\n📄 sample alpaca example:")
        sample = alpaca_data[0]
        print(f"  instruction: {sample['instruction']}")
        print(f"  input: {sample['input'][:60]}...")
        print(f"  output: {sample['output'][:60]}...")
    
    print(f"\n✅ done! training data saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

