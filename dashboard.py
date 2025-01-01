import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from collections import Counter
import emoji
import humanize
from datetime import datetime
import os
import sys
import re

def parse_whatsapp_chat(content):
    """Parse WhatsApp chat content and return a DataFrame."""
    # Regular expressions for different WhatsApp message formats
    patterns = [
        # Format: 12/31/23, 11:59 PM - Name: Message
        r'(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s(?:AM|PM))\s-\s(.*?):\s(.*)',
        # Format: [31/12/23, 23:59:59] Name: Message
        r'\[(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}(?::\d{2})?)\]\s(.*?):\s(.*)',
        # Format: [31/12/2023 23:59:59] Name: Message
        r'\[(\d{1,2}/\d{1,2}/\d{4}\s\d{1,2}:\d{2}(?::\d{2})?)\]\s(.*?):\s(.*)',
        # Format: 31/12/23, 23:59 - Name: Message
        r'(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2})\s-\s(.*?):\s(.*)'
    ]
    
    date_formats = [
        '%m/%d/%y, %I:%M %p',  # 12/31/23, 11:59 PM
        '%d/%m/%y, %H:%M:%S',  # 31/12/23, 23:59:59
        '%d/%m/%Y %H:%M:%S',   # 31/12/2023 23:59:59
        '%d/%m/%y, %H:%M'      # 31/12/23, 23:59
    ]
    
    messages = []
    current_date = None
    current_sender = None
    current_message = None
    
    for line in content.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        matched = False
        for pattern, date_format in zip(patterns, date_formats):
            match = re.match(pattern, line)
            if match:
                if current_message:
                    messages.append({
                        'timestamp': current_date,
                        'sender': current_sender,
                        'message': current_message.strip()
                    })
                
                date_str, sender, message = match.groups()
                try:
                    current_date = datetime.strptime(date_str, date_format)
                    current_sender = sender.strip()
                    current_message = message.strip()
                    matched = True
                    break
                except ValueError:
                    continue
        
        if not matched and current_message is not None:
            # Append multi-line messages
            current_message += '\n' + line
    
    # Add the last message
    if current_message:
        messages.append({
            'timestamp': current_date,
            'sender': current_sender,
            'message': current_message.strip()
        })
    
    # Create DataFrame
    df = pd.DataFrame(messages)
    
    if df.empty:
        return None
        
    # Sort by timestamp
    df = df.sort_values('timestamp')
    df = df.reset_index(drop=True)
    
    # Add message length
    df['message_length'] = df['message'].str.len()
    
    return df

def get_latest_messages(df, n=5):
    """Get the n most recent messages."""
    return df.tail(n)

def format_time_ago(timestamp):
    """Format timestamp as time ago (e.g., '2 hours ago')."""
    now = datetime.now()
    diff = now - timestamp
    return humanize.naturaltime(diff)

def calculate_conversation_metrics(df):
    """Calculate advanced conversation metrics."""
    # Response times
    df_sorted = df.sort_values('timestamp')
    df_sorted['time_diff'] = df_sorted['timestamp'].diff()
    avg_response_time = df_sorted['time_diff'].mean()
    
    # Message length patterns
    avg_message_length = df['message_length'].mean()
    
    # Most active hours
    peak_hours = df['timestamp'].dt.hour.value_counts().nlargest(3)
    
    # Most active days
    peak_days = df['timestamp'].dt.day_name().value_counts().nlargest(3)
    
    # Early birds (5 AM - 11 AM)
    morning_starters = df[df['timestamp'].dt.hour.between(5, 11)]['sender'].value_counts().nlargest(3)
    
    # Night owls (10 PM - 5 AM)
    night_owls = df[df['timestamp'].dt.hour.between(22, 5)]['sender'].value_counts().nlargest(3)
    
    return {
        'avg_response_time': avg_response_time,
        'avg_message_length': avg_message_length,
        'peak_hours': peak_hours,
        'peak_days': peak_days,
        'morning_starters': morning_starters,
        'night_owls': night_owls
    }

def main():
    st.set_page_config(
        page_title="WhatsApp Chat Analyzer",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
        <style>
        .main {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        .stButton>button {
            background-color: #FF4B4B;
            color: white;
        }
        .uploadedFile {
            background-color: #262730;
        }
        </style>
        """, unsafe_allow_html=True)

    st.title("WhatsApp Chat Analyzer 💬")
    
    uploaded_file = st.file_uploader("Upload your WhatsApp chat export (.txt file)", type="txt")
    
    if uploaded_file is not None:
        try:
            # Read and decode the file content
            content = uploaded_file.read().decode('utf-8')
            
            # Parse the chat content
            df = parse_whatsapp_chat(content)
            
            if df is not None and not df.empty:
                # Calculate metrics
                metrics = calculate_conversation_metrics(df)
                
                # Display basic stats
                st.header("Chat Overview")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Messages", len(df))
                with col2:
                    st.metric("Participants", df['sender'].nunique())
                with col3:
                    st.metric("Time Span", f"{(df['timestamp'].max() - df['timestamp'].min()).days} days")
                
                # Activity Analysis
                st.header("Activity Analysis")
                
                # Message Volume Over Time
                st.subheader("Message Volume Over Time")
                daily_messages = df.groupby(df['timestamp'].dt.date).size().reset_index()
                daily_messages.columns = ['date', 'messages']
                fig = px.line(daily_messages, x='date', y='messages', 
                            title='Daily Message Volume')
                fig.update_layout(
                    plot_bgcolor='#262730',
                    paper_bgcolor='#262730',
                    font_color='#FAFAFA'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Peak Activity Times
                st.subheader("Peak Activity Times")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("Most Active Hours:")
                    for hour, count in metrics['peak_hours'].items():
                        st.write(f"• {hour:02d}:00 - {count} messages")
                
                with col2:
                    st.write("Most Active Days:")
                    for day, count in metrics['peak_days'].items():
                        st.write(f"• {day} - {count} messages")
                
                # Early Birds vs Night Owls
                st.subheader("Early Birds vs Night Owls")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("Early Birds (5 AM - 11 AM):")
                    for sender, count in metrics['morning_starters'].items():
                        st.write(f"• {sender}: {count} messages")
                
                with col2:
                    st.write("Night Owls (10 PM - 5 AM):")
                    for sender, count in metrics['night_owls'].items():
                        st.write(f"• {sender}: {count} messages")
                
                # Message Length Distribution
                st.subheader("Message Length Distribution")
                fig = px.histogram(df, x='message_length', 
                                 title='Distribution of Message Lengths',
                                 nbins=50)
                fig.update_layout(
                    plot_bgcolor='#262730',
                    paper_bgcolor='#262730',
                    font_color='#FAFAFA'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Latest Messages
                st.header("Latest Messages")
                latest_msgs = get_latest_messages(df, n=5)
                for _, msg in latest_msgs.iterrows():
                    with st.container():
                        st.markdown(f"""
                        **{msg['sender']}** - _{format_time_ago(msg['timestamp'])}_  
                        {msg['message']}
                        """)
                        st.markdown("---")
                
            else:
                st.error("Could not parse the chat file. Please make sure it's a valid WhatsApp chat export.")
                
        except Exception as e:
            st.error(f"An error occurred while processing the file: {str(e)}")
            st.error("Please make sure you've uploaded a valid WhatsApp chat export file.")

if __name__ == "__main__":
    main()
