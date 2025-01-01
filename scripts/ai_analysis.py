import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from langdetect import detect, LangDetectException
import logging
from typing import Dict, Any
import re
from camel_tools.sentiment import SentimentAnalyzer

# Download required NLTK data
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
except Exception as e:
    logging.warning(f"Failed to download NLTK data: {str(e)}")

# Initialize sentiment analyzers
try:
    vader = SentimentIntensityAnalyzer()
except Exception as e:
    logging.error(f"Failed to initialize VADER: {str(e)}")
    vader = None

try:
    arabic_analyzer = SentimentAnalyzer.pretrained()
except Exception as e:
    logging.error(f"Failed to initialize Arabic analyzer: {str(e)}")
    arabic_analyzer = None

def is_franco_arabic(text: str) -> bool:
    """
    Detect Franco-Arabic text based on common patterns.
    """
    if not isinstance(text, str):
        return False
        
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

def detect_language(text: str) -> str:
    """
    Detect the language of a text, handling errors gracefully.
    """
    if not isinstance(text, str) or len(text.strip()) < 3:
        return 'unknown'

    try:
        # Clean text
        text = re.sub(r'http\S+|www\S+|\S+\.\S+', '', text)
        text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', '', text)
        
        # Check for Arabic script
        if any('\u0600' <= char <= '\u06FF' for char in text):
            return 'arabic'
            
        # Check for Franco-Arabic
        if is_franco_arabic(text):
            return 'franco'
            
        # Use langdetect for other cases
        detected = detect(text)
        
        # Map similar languages
        if detected == 'ar':
            return 'arabic'
        elif detected == 'en':
            return 'english'
        elif detected in ['sv', 'da', 'no', 'nl', 'de', 'af']:
            return 'franco' if is_franco_arabic(text) else 'english'
            
        return 'english'
    except LangDetectException:
        return 'unknown'
    except Exception as e:
        logging.error(f"Language detection error: {str(e)}")
        return 'unknown'

def analyze_sentiment_english(text: str) -> Dict[str, Any]:
    """
    Analyze sentiment for English text using VADER and TextBlob.
    """
    try:
        # VADER analysis
        if vader:
            vader_scores = vader.polarity_scores(text)
        else:
            vader_scores = {'compound': 0, 'pos': 0, 'neu': 0, 'neg': 0}
            
        # TextBlob analysis
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity
        
        # Combine scores
        return {
            'compound': vader_scores['compound'],
            'positive': vader_scores['pos'],
            'neutral': vader_scores['neu'],
            'negative': vader_scores['neg'],
            'textblob_polarity': textblob_polarity
        }
    except Exception as e:
        logging.error(f"English sentiment analysis error: {str(e)}")
        return {'compound': 0, 'positive': 0, 'neutral': 1, 'negative': 0, 'textblob_polarity': 0}

def analyze_sentiment_arabic(text: str) -> Dict[str, Any]:
    """
    Analyze sentiment for Arabic text using CAMeL Tools.
    """
    try:
        if arabic_analyzer:
            prediction = arabic_analyzer.predict(text)
            # Map CAMeL Tools output to our format
            sentiment_map = {'positive': 1, 'negative': -1, 'neutral': 0}
            score = sentiment_map.get(prediction, 0)
            
            return {
                'compound': score,
                'positive': max(0, score),
                'neutral': 1 if score == 0 else 0,
                'negative': abs(min(0, score)),
                'textblob_polarity': score
            }
    except Exception as e:
        logging.error(f"Arabic sentiment analysis error: {str(e)}")
        
    return {'compound': 0, 'positive': 0, 'neutral': 1, 'negative': 0, 'textblob_polarity': 0}

def analyze_sentiment(text: str, language: str) -> Dict[str, Any]:
    """
    Analyze sentiment based on detected language.
    """
    if not isinstance(text, str) or len(text.strip()) < 3:
        return {'compound': 0, 'positive': 0, 'neutral': 1, 'negative': 0, 'textblob_polarity': 0}
        
    try:
        if language == 'english':
            return analyze_sentiment_english(text)
        elif language in ['arabic', 'franco']:
            return analyze_sentiment_arabic(text)
        else:
            # Default to English analysis for unknown languages
            return analyze_sentiment_english(text)
    except Exception as e:
        logging.error(f"Sentiment analysis error: {str(e)}")
        return {'compound': 0, 'positive': 0, 'neutral': 1, 'negative': 0, 'textblob_polarity': 0}

def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add language detection and sentiment analysis to DataFrame.
    """
    # Detect languages
    df['language'] = df['message'].apply(detect_language)
    
    # Analyze sentiment
    sentiment_results = df.apply(
        lambda row: analyze_sentiment(row['message'], row['language']),
        axis=1
    )
    
    # Add sentiment columns
    df['sentiment_compound'] = sentiment_results.apply(lambda x: x['compound'])
    df['sentiment_positive'] = sentiment_results.apply(lambda x: x['positive'])
    df['sentiment_neutral'] = sentiment_results.apply(lambda x: x['neutral'])
    df['sentiment_negative'] = sentiment_results.apply(lambda x: x['negative'])
    df['sentiment_textblob'] = sentiment_results.apply(lambda x: x['textblob_polarity'])
    
    return df
