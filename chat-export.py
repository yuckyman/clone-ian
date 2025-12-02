#!/usr/bin/env python3
import sqlite3
import json
from pathlib import Path
from datetime import datetime, timezone
import sys
import re

# — adjust these if you made a copy of chat.db —
CHAT_DB = Path("chat.db")
OUTPUT_JSONL = Path("imessage_corpus.jsonl")
CONTACTS_FILE = Path("contacts.json")
MIN_YEAR = 2022  # Only export messages from this year onwards

def is_reaction_message(text):
    # Common patterns for reaction messages
    reaction_patterns = [
        # Basic reactions - consolidated pattern
        r'(loved|liked|emphasized|laughed at|disliked|questioned|reacted to)\s+(?:an\s+)?(?:image|message|video|attachment|audio message|movie|digital touch message)',
        # Reactions with quoted text - handle both straight and curly quotes
        r'(loved|liked|emphasized|laughed at|disliked|questioned)\s+[“”]'
    ]
    
    # Debug: Check if text contains common reaction words but isn't caught
    reaction_words = ['laughed', 'loved', 'liked', 'emphasized', 'disliked', 'questioned', 'reacted']
    text_lower = text.lower()
    print(text_lower)
    return any(re.search(pattern, text_lower) for pattern in reaction_patterns)

def load_contacts():
    """Load the contacts mapping file if it exists"""
    try:
        if CONTACTS_FILE.exists():
            with CONTACTS_FILE.open() as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def get_sender_name(sender, contacts):
    """Get the contact name for a sender, fallback to phone number"""
    if sender and sender.startswith('+'):
        name = contacts.get(sender, '').strip()
        return name if name else sender
    return sender or "ian t."

def convert_apple_timestamp(mac_time):
    """
    Convert Apple timestamp into an ISO8601 string in UTC. 
    Handles both old format (seconds since 2001) and new format (nanoseconds since 2001).
    Returns None on error.
    """
    try:
        if mac_time == 0:
            return None
            
        # if the number is huge (> 1e15), it's nanoseconds; otherwise seconds
        if mac_time > 1e15:
            # nanoseconds since 2001-01-01
            ts = mac_time / 1_000_000_000 + 978307200
        else:
            # seconds since 2001-01-01 (older format)
            ts = mac_time + 978307200
            
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    except Exception:
        return None

def main():
    if not CHAT_DB.exists():
        print(f"error: cannot find {CHAT_DB}", file=sys.stderr)
        sys.exit(1)

    with sqlite3.connect(str(CHAT_DB)) as conn, OUTPUT_JSONL.open("w", encoding="utf-8") as out:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                message.date    AS mac_time,
                message.text    AS text,
                handle.id       AS sender
            FROM message
            LEFT JOIN handle ON message.handle_id = handle.ROWID
            WHERE message.text IS NOT NULL
            ORDER BY message.date ASC
        """)

        contacts = load_contacts()
        reaction_count = 0
        year_filtered_count = 0
        total_count = 0
        min_date = datetime(MIN_YEAR, 1, 1, tzinfo=timezone.utc)

        for mac_time, text, sender in cursor:
            total_count += 1
            if is_reaction_message(text):
                reaction_count += 1
                continue
            
            timestamp_str = convert_apple_timestamp(mac_time)
            if not timestamp_str:
                continue
            
            # Check if message is before MIN_YEAR
            try:
                msg_date = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                if msg_date < min_date:
                    year_filtered_count += 1
                    continue
            except Exception:
                # Skip messages with invalid timestamps
                continue
                
            record = {
                "timestamp": timestamp_str,
                "sender":    get_sender_name(sender, contacts),
                "text":      text
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✔ exported {OUTPUT_JSONL} ({CHAT_DB.name})")
    print(f"  - filtered out {reaction_count} reaction messages out of {total_count} total messages")
    print(f"  - filtered out {year_filtered_count} messages before {MIN_YEAR}")

if __name__ == "__main__":
    main()