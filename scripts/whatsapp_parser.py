import pandas as pd
import re
from datetime import datetime
import logging
from .ai_analysis import enrich_dataframe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_system_events(df):
    """Extract group events (joins/leaves) from system messages."""
    joins = df[df['message'].str.contains(' joined ', case=False, na=False)]
    leaves = df[df['message'].str.contains(' left ', case=False, na=False)]
    return {
        'joins': len(joins),
        'leaves': len(leaves),
        'join_dates': joins['timestamp'].tolist(),
        'leave_dates': leaves['timestamp'].tolist()
    }

def parse_whatsapp_chat(content):
    """Parse WhatsApp chat content and return a DataFrame."""
    # Regular expression for WhatsApp message format
    pattern = r'(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}(?::\d{2})?\s(?:AM|PM)?) - (.*?): (.*)'
    
    messages = []
    current_date = None
    current_sender = None
    current_message = None
    
    for line in content.split('\n'):
        match = re.match(pattern, line)
        if match:
            if current_message:
                messages.append({
                    'timestamp': current_date,
                    'sender': current_sender,
                    'message': current_message
                })
            
            date_str, sender, message = match.groups()
            try:
                # Try to parse the date with seconds
                current_date = datetime.strptime(date_str, '%m/%d/%y, %I:%M:%S %p')
            except ValueError:
                try:
                    # Try without seconds
                    current_date = datetime.strptime(date_str, '%m/%d/%y, %I:%M %p')
                except ValueError:
                    # Skip if date parsing fails
                    continue
            
            current_sender = sender
            current_message = message
        elif current_message:
            # Append multi-line messages
            current_message += '\n' + line
    
    # Add the last message
    if current_message:
        messages.append({
            'timestamp': current_date,
            'sender': current_sender,
            'message': current_message
        })
    
    # Create DataFrame
    df = pd.DataFrame(messages)
    
    if not df.empty:
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Reset index
        df = df.reset_index(drop=True)
    
    # Enrich with AI analysis
    logger.info("Performing AI analysis...")
    df = enrich_dataframe(df)
    
    return df

def get_latest_messages(df, n=5):
    """Get the n most recent messages."""
    return df.tail(n)

if __name__ == "__main__":
    # Example usage
    # df = parse_whatsapp_chat("path/to/chat.txt")
    # print(df.head())
    # latest_msgs = get_latest_messages(df)
    # print(latest_msgs)
    pass
