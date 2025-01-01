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

def parse_whatsapp_chat(file_path):
    """Parse WhatsApp chat file and convert it to a pandas DataFrame."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='utf-16') as file:
            content = file.read()

    # Regular expression for matching WhatsApp message format
    pattern = r'(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}(?::\d{2})?(?:\s(?:AM|PM))?)\s-\s([^:]+):\s(.+)'
    
    messages = []
    system_messages = []
    
    for line in content.split('\n'):
        if not line.strip():
            continue
            
        match = re.match(pattern, line)
        if match:
            timestamp_str, sender, message = match.groups()
            
            try:
                # Try different date formats
                for fmt in ['%d/%m/%y, %H:%M', '%d/%m/%Y, %H:%M', 
                          '%m/%d/%y, %H:%M', '%m/%d/%Y, %H:%M',
                          '%d/%m/%y, %H:%M:%S', '%d/%m/%Y, %H:%M:%S']:
                    try:
                        timestamp = datetime.strptime(timestamp_str, fmt)
                        break
                    except ValueError:
                        continue
                
                # Check if it's a system message
                if "added" in message or "left" in message or "joined" in message:
                    system_messages.append({
                        'timestamp': timestamp,
                        'event': message
                    })
                else:
                    messages.append({
                        'timestamp': timestamp,
                        'sender': sender.strip(),
                        'message': message
                    })
                    
            except ValueError as e:
                logger.warning(f"Could not parse timestamp {timestamp_str}: {e}")
                continue
                
    # Convert to DataFrame
    if not messages:
        return pd.DataFrame(columns=['timestamp', 'sender', 'message']), pd.DataFrame(columns=['timestamp', 'event'])
    
    df = pd.DataFrame(messages)
    system_df = pd.DataFrame(system_messages) if system_messages else pd.DataFrame(columns=['timestamp', 'event'])
    
    # Ensure timestamp column is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    system_df['timestamp'] = pd.to_datetime(system_df['timestamp'])
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    system_df = system_df.sort_values('timestamp').reset_index(drop=True)
    
    # Enrich with AI analysis
    logger.info("Performing AI analysis...")
    df = enrich_dataframe(df)
    
    return df, system_df

def get_latest_messages(df, n=10):
    """Get the latest message from top N most active senders."""
    top_senders = df.groupby('sender').size().nlargest(n).index
    latest_messages = []
    
    for sender in top_senders:
        sender_msgs = df[df['sender'] == sender]
        if not sender_msgs.empty:
            latest = sender_msgs.iloc[-1]
            latest_messages.append({
                'sender': sender,
                'message': latest['message'],
                'timestamp': latest['timestamp'],
                'message_count': len(sender_msgs)
            })
    
    return pd.DataFrame(latest_messages)

if __name__ == "__main__":
    # Example usage
    # df, system_df = parse_whatsapp_chat("path/to/chat.txt")
    # print(df.head())
    # print(system_df.head())
    # latest_msgs = get_latest_messages(df)
    # print(latest_msgs)
    pass
