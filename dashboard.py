import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
from pathlib import Path
import tempfile
import os
from wordcloud import WordCloud
import arabic_reshaper
from bidi.algorithm import get_display
import numpy as np
from collections import Counter
import emoji
import humanize

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from scripts.whatsapp_parser import parse_whatsapp_chat, get_latest_messages

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
    df['message_length'] = df['message'].str.len()
    avg_message_length = df['message_length'].mean()
    
    # Most active hours
    peak_hours = df['timestamp'].dt.hour.value_counts().nlargest(3)
    
    # Most active days
    peak_days = df['timestamp'].dt.day_name().value_counts().nlargest(3)
    
    # Conversation starters
    morning_starters = df[df['timestamp'].dt.hour.between(5, 11)]['sender'].value_counts().nlargest(3)
    
    # Night owls
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
    
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name
            
        try:
            # Parse chat file
            df, system_df = parse_whatsapp_chat(temp_path)
            os.unlink(temp_path)
            
            # Get system events
            total_joins = len(system_df[system_df['event'].str.contains('joined', case=False, na=False)])
            total_leaves = len(system_df[system_df['event'].str.contains('left', case=False, na=False)])
            
            # Calculate metrics
            total_messages = len(df)
            total_participants = df['sender'].nunique()
            avg_daily = df.groupby(df['timestamp'].dt.date).size().mean()
            
            # Calculate advanced metrics
            metrics = calculate_conversation_metrics(df)
            
            # Display key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                    <div class="stat-card">
                        <h3>{total_messages:,}</h3>
                        <p>Total Messages</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                    <div class="stat-card">
                        <h3>{int(avg_daily)}</h3>
                        <p>Avg. Daily Messages</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                    <div class="stat-card">
                        <h3>{metrics['avg_message_length']:.0f}</h3>
                        <p>Avg. Message Length</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col4:
                response_time = metrics['avg_response_time'].total_seconds() / 60
                st.markdown(f"""
                    <div class="stat-card">
                        <h3>{response_time:.1f}m</h3>
                        <p>Avg. Response Time</p>
                    </div>
                """, unsafe_allow_html=True)
            
            # Create tabs for different analyses
            tab1, tab2, tab3 = st.tabs(["📊 Activity", "🎯 Engagement", "🔍 Content"])
            
            with tab1:
                # Message Timeline
                daily_msgs = df.groupby(df['timestamp'].dt.date).size().reset_index()
                daily_msgs.columns = ['date', 'messages']
                
                fig_timeline = px.line(
                    daily_msgs,
                    x='date',
                    y='messages',
                    title="Message Volume Over Time",
                    labels={'date': 'Date', 'messages': 'Number of Messages'}
                )
                fig_timeline.update_traces(line_color='#00bcd4')
                fig_timeline.update_layout(
                    plot_bgcolor='#2d2d2d',
                    paper_bgcolor='#2d2d2d',
                    font_color='#e0e0e0',
                    title_font_color='#e0e0e0',
                    xaxis=dict(gridcolor='#3d3d3d', linecolor='#3d3d3d'),
                    yaxis=dict(gridcolor='#3d3d3d', linecolor='#3d3d3d')
                )
                st.plotly_chart(fig_timeline, use_container_width=True)
                
                # Activity Heatmap
                df['hour'] = df['timestamp'].dt.hour
                df['day'] = df['timestamp'].dt.day_name()
                
                activity_pivot = pd.pivot_table(
                    df,
                    values='message',
                    index='day',
                    columns='hour',
                    aggfunc='count',
                    fill_value=0
                )
                
                days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                activity_pivot = activity_pivot.reindex(days_order)
                
                fig_heatmap = px.imshow(
                    activity_pivot,
                    title="Activity Patterns by Day and Hour",
                    labels=dict(x="Hour of Day", y="Day of Week", color="Messages"),
                    color_continuous_scale="Viridis"
                )
                fig_heatmap.update_layout(
                    plot_bgcolor='#2d2d2d',
                    paper_bgcolor='#2d2d2d',
                    font_color='#e0e0e0',
                    title_font_color='#e0e0e0'
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
                
                # Weekly Activity Pattern
                df['week'] = df['timestamp'].dt.isocalendar().week
                weekly_pattern = df.groupby(['week', 'day']).size().reset_index()
                weekly_pattern.columns = ['week', 'day', 'messages']
                
                fig_weekly = px.line(
                    weekly_pattern,
                    x='week',
                    y='messages',
                    color='day',
                    title="Weekly Activity Patterns",
                    labels={'week': 'Week Number', 'messages': 'Messages', 'day': 'Day'}
                )
                fig_weekly.update_layout(
                    plot_bgcolor='#2d2d2d',
                    paper_bgcolor='#2d2d2d',
                    font_color='#e0e0e0',
                    title_font_color='#e0e0e0',
                    xaxis=dict(gridcolor='#3d3d3d', linecolor='#3d3d3d'),
                    yaxis=dict(gridcolor='#3d3d3d', linecolor='#3d3d3d')
                )
                st.plotly_chart(fig_weekly, use_container_width=True)
            
            with tab2:
                # Engagement Insights
                st.subheader("⭐ Chat Insights")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                        <div class="stat-card">
                            <h3>Peak Activity Times</h3>
                            <div style="text-align: left; padding: 1rem;">
                    """, unsafe_allow_html=True)
                    
                    for hour, count in metrics['peak_hours'].items():
                        st.write(f"• {hour:02d}:00 - {count:,} messages")
                    
                    st.markdown("</div></div>", unsafe_allow_html=True)
                    
                    st.markdown("""
                        <div class="stat-card">
                            <h3>Early Birds 🌅</h3>
                            <div style="text-align: left; padding: 1rem;">
                    """, unsafe_allow_html=True)
                    
                    for sender, count in metrics['morning_starters'].items():
                        st.write(f"• {sender}: {count:,} morning messages")
                    
                    st.markdown("</div></div>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                        <div class="stat-card">
                            <h3>Busiest Days</h3>
                            <div style="text-align: left; padding: 1rem;">
                    """, unsafe_allow_html=True)
                    
                    for day, count in metrics['peak_days'].items():
                        st.write(f"• {day}: {count:,} messages")
                    
                    st.markdown("</div></div>", unsafe_allow_html=True)
                    
                    st.markdown("""
                        <div class="stat-card">
                            <h3>Night Owls 🦉</h3>
                            <div style="text-align: left; padding: 1rem;">
                    """, unsafe_allow_html=True)
                    
                    for sender, count in metrics['night_owls'].items():
                        st.write(f"• {sender}: {count:,} late messages")
                    
                    st.markdown("</div></div>", unsafe_allow_html=True)
            
            with tab3:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Emoji Analysis
                    emoji_counts = Counter()
                    for message in df['message']:
                        emojis = ''.join(c for c in str(message) if c in emoji.EMOJI_DATA)
                        emoji_counts.update(emojis)
                    
                    if emoji_counts:
                        top_emojis = pd.DataFrame(
                            emoji_counts.most_common(10),
                            columns=['Emoji', 'Count']
                        )
                        
                        fig_emoji = px.bar(
                            top_emojis,
                            x='Count',
                            y='Emoji',
                            orientation='h',
                            title="Most Used Emojis",
                            color='Count',
                            color_continuous_scale='Viridis'
                        )
                        fig_emoji.update_layout(
                            plot_bgcolor='#2d2d2d',
                            paper_bgcolor='#2d2d2d',
                            font_color='#e0e0e0',
                            title_font_color='#e0e0e0',
                            xaxis=dict(gridcolor='#3d3d3d', linecolor='#3d3d3d'),
                            yaxis=dict(gridcolor='#3d3d3d', linecolor='#3d3d3d')
                        )
                        st.plotly_chart(fig_emoji, use_container_width=True)
                    else:
                        st.info("No emojis found in the chat")
                
                with col2:
                    # Language Distribution
                    lang_dist = df['language'].value_counts()
                    
                    fig_lang = px.pie(
                        values=lang_dist.values,
                        names=lang_dist.index,
                        title="Language Distribution",
                        hole=0.4,
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    fig_lang.update_layout(
                        plot_bgcolor='#2d2d2d',
                        paper_bgcolor='#2d2d2d',
                        font_color='#e0e0e0',
                        title_font_color='#e0e0e0'
                    )
                    st.plotly_chart(fig_lang, use_container_width=True)
                    
                    # Message Length Distribution
                    fig_length = px.histogram(
                        df,
                        x='message_length',
                        title="Message Length Distribution",
                        nbins=50,
                        color_discrete_sequence=['#00bcd4']
                    )
                    fig_length.update_layout(
                        plot_bgcolor='#2d2d2d',
                        paper_bgcolor='#2d2d2d',
                        font_color='#e0e0e0',
                        title_font_color='#e0e0e0',
                        xaxis=dict(gridcolor='#3d3d3d', linecolor='#3d3d3d', title="Message Length (characters)"),
                        yaxis=dict(gridcolor='#3d3d3d', linecolor='#3d3d3d', title="Number of Messages")
                    )
                    st.plotly_chart(fig_length, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error analyzing chat: {str(e)}")
            
    else:
        st.info("👆 Upload your WhatsApp chat export file to begin analysis!")
    
    # Footer
    st.markdown("---")
    st.markdown("Powered by Advanced Analytics • Made with ❤️")

if __name__ == "__main__":
    main()
