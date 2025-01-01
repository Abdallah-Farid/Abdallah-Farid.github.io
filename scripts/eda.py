import pandas as pd
from datetime import datetime
from langdetect import detect
from collections import defaultdict
import logging
import re
from .whatsapp_parser import parse_whatsapp_chat

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def is_franco_arabic(text):
    """
    Detect Franco-Arabic text based on common patterns.
    """
    # Common Franco-Arabic patterns
    patterns = [
        r'\b[23](?:nd|rd)\b',  # Numbers used as letters (e.g., '3nd' for 'and')
        r'\b7[ao]g[ao]\b',     # Common words like '7aga'
        r'\b[aeiou]?7[aeiou]\w*\b',  # Words with '7'
        r'\b[aeiou]?3[aeiou]\w*\b',  # Words with '3'
        r'\b[aeiou]?2[aeiou]\w*\b',  # Words with '2'
        r'\b[aeiou]?5[aeiou]\w*\b',  # Words with '5'
        r'\b[aeiou]?8[aeiou]\w*\b',  # Words with '8'
        r'\b[aeiou]?9[aeiou]\w*\b',  # Words with '9'
        r'\b(?:el|al)-?\w+\b',  # Words starting with 'el' or 'al'
        r'\b(?:ana|enta|enti|ehna)\b',  # Common pronouns
        r'\bm(?:sh|esh|4)\b',  # Negation forms
        r'\bf[iy]\b',          # Preposition 'fi'
        r'\bw[ae]l?l?[ae]h[iy]?\b',  # Various spellings of 'wallahi'
        r'\bb[ae]2[ae]\b',     # Various spellings of 'ba2a'
        r'\bkid[ae]\b',        # Various spellings of 'kida'
        r'\b3[ae]yz[ae]?\b',   # Various spellings of '3ayez/3ayza'
        r'\b[td]e3raf\b',      # Various spellings of 'te3raf'
    ]
    
    # Check if any Franco-Arabic pattern is found
    for pattern in patterns:
        if re.search(pattern, text.lower()):
            return True
            
    # Check for mix of English letters and numbers in Arabic context
    has_numbers = bool(re.search(r'\d', text))
    has_letters = bool(re.search(r'[a-zA-Z]', text))
    has_arabic_context = any(word in text.lower() for word in ['ya', 'el', 'al', 'fi', 'we', 'ana', 'enta', 'da', 'di', 'fe'])
    
    return has_numbers and has_letters and has_arabic_context

def detect_language_safe(text):
    """
    Safely detect language of text, handling Arabic, English, and Franco-Arabic.
    Returns 'unknown' if detection fails.
    """
    if not isinstance(text, str):
        return 'unknown'
    
    # Remove URLs and emojis for better detection
    text = re.sub(r'http\S+|www\S+|\S+\.\S+', '', text)
    text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', '', text)
    
    if not text.strip():
        return 'unknown'
    
    try:
        # Check for Arabic characters
        if any('\u0600' <= char <= '\u06FF' for char in text):
            return 'arabic'
        
        # Check for Franco-Arabic
        if is_franco_arabic(text):
            return 'franco'
        
        # Use langdetect for remaining text
        detected = detect(text)
        
        # Map similar languages to main categories
        if detected == 'ar':
            return 'arabic'
        elif detected == 'en':
            return 'english'
        # Consider most European language detections as potential Franco
        elif detected in ['sv', 'da', 'no', 'nl', 'de', 'af']:
            # Double check if it might be Franco
            if is_franco_arabic(text):
                return 'franco'
            if len(text.split()) <= 3:  # Short texts are often misclassified
                return 'franco'
        
        return 'english'  # Default to English for other cases
    except:
        return 'unknown'

def analyze_chat(file_path):
    """
    Perform exploratory data analysis on WhatsApp chat data.
    
    Args:
        file_path (str): Path to WhatsApp chat file
        
    Returns:
        tuple: (enriched_df, stats_dict)
    """
    logger.info("Starting chat analysis...")
    
    # Load and parse chat
    df = parse_whatsapp_chat(file_path)
    
    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Enrich DataFrame
    logger.info("Enriching data with additional features...")
    df['message_length'] = df['message'].astype(str).str.len()
    df['language'] = df['message'].astype(str).apply(detect_language_safe)
    df['date'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.hour
    
    # Basic statistics
    stats = {
        'messages_per_sender': df.groupby('sender').size().to_dict(),
        'avg_message_length': df.groupby('sender')['message_length'].mean().to_dict(),
        'daily_message_count': df.groupby('date').size().to_dict(),
        'language_distribution': df['language'].value_counts().to_dict(),
        'total_messages': len(df),
        'unique_senders': df['sender'].nunique(),
        'date_range': {
            'start': df['timestamp'].min().strftime('%Y-%m-%d'),
            'end': df['timestamp'].max().strftime('%Y-%m-%d')
        }
    }
    
    # Language statistics per sender
    language_stats = df.groupby(['sender', 'language']).size().unstack(fill_value=0)
    stats['language_per_sender'] = language_stats.to_dict()
    
    # Most active hours
    stats['messages_by_hour'] = df.groupby('hour').size().to_dict()
    
    logger.info("Analysis completed successfully")
    return df, stats

def save_results(df, stats, output_path='data/chat_analysis.csv'):
    """Save analysis results to CSV"""
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    # Save statistics to a separate file
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(output_path.replace('.csv', '_stats.csv'), index=False, encoding='utf-8')
    
if __name__ == "__main__":
    # Example usage
    # file_path = "data/chat.txt"
    # df, stats = analyze_chat(file_path)
    # print("Basic Stats:", stats)
    pass
