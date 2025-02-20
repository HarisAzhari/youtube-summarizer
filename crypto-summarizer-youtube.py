# All imports remain the same
from flask import Flask, jsonify, request
from datetime import datetime, timedelta
import threading
from pathlib import Path
import os
import requests
import pytz
import time
from youtube_transcript_api import YouTubeTranscriptApi
import google.generativeai as genai
from tqdm import tqdm
import json
from flask_cors import CORS
import sqlite3
import sys
from flask import Flask, jsonify, request
from flask_cors import CORS
import yfinance as yf
import numpy as np
import pandas as pd
import datetime as dt
import joblib
import pickle
import google.generativeai as palm
import re
from statsmodels.tsa.arima.model import ARIMA
from enum import Enum
import logging
import os
from typing import List, Optional,Dict, Any
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import traceback
from functools import wraps
from historical_price import HistoricalPriceService
from dataclasses import dataclass
import re

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": [
            "https://admin.moometrics.io",  # Admin frontend
            "https://api.moometrics.io",    # API domain
            "https://moometrics.io",        # Main domain
            "http://localhost:*",           # Local development (any port)
            "http://localhost:5176",
            "https://www.aifiqh.com"        # Specific local development port
        ],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True,       # Add this for credentials support
        "expose_headers": ["Content-Type", "Authorization"]  # Expose necessary headers
    }
})

# Configuration
DB_PATH = 'youtube_crypto.db'
API_KEY = 'AIzaSyCwVHt2_T0A3N39voVCCAieDgg0PctOSCs'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Combined channel list including Indonesian channels
ALL_CHANNELS = [
    # English channels
    '@CoinBureau',
    '@MeetKevin',
    'UCQglaVhGOBI0BR5S6IJnQPg',
    '@AltcoinDaily',
    '@CryptosRUs',
    '@elliotrades_official',
    '@DataDash',
    '@IvanOnTech',
    '@TheCryptoLark',
    '@CryptoCasey',
    '@AnthonyPompliano',
    '@alessiorastani',
    '@CryptoCapitalVenture',
    '@aantonop',
    '@Boxmining',
    '@CryptoZombie',
    '@tonevays',
    '@ScottMelker',
    '@CTOLARSSON',
    '@Bankless',
    '@gemgemcrypto',
    '@DappUniversity',
    '@EatTheBlocks',
    '@MilkRoadDaily',
    '@Coinsider',
    '@InvestAnswers',
    '@DigitalAssetNews',
    '@CoinGecko',
    '@CoinMarketCapOfficial',
    '@MaxMaher',
    '@when-shift-happens',
    '@intothecryptoverse',
    '@UnchainedCrypto',
    '@RealVisionFinance',
    '@Delphi_Digital',
    'UCFEHdhuB_BEUHA_eL9cjqHA',
    '@milesdeutscher1357',
    '@TheDefiant',
    '@CryptoBanterPlus',
    '@CryptoBanterGroup',
    
    # Indonesian channels
    '@AnggaAndinata',
    '@RepublikRupiahOfficial',
    '@AkademiCrypto',
    '@felicia.tjiasaka',
    '@Coinvestasi',
    '@TimothyRonald',
    '@AndySenjaya',
    '@leon.hartono'
]

# Status tracking
analysis_status = {
    "is_running": False,
    "current_channel": None,
    "current_stage": None,
    "channels_processed": 0,
    "total_channels": 0,
    "errors": [],
    "processed_videos": 0,
    "skipped_videos": 0
}

# Automation status
automation_status = {
    "is_scheduled": False,
    "next_run_time": None,
    "last_execution": None
}

# Update cache duration to 5 hours
CACHE_DURATION = timedelta(hours=5)

# Add these near the top of the file, after the imports but before the routes
_results_cache = {
    'data': None,
    'timestamp': None
}

def cache_response(duration):
    """Decorator to cache API responses"""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            # Check if we have a valid cached response
            now = datetime.now()
            if (_results_cache['data'] is not None and 
                _results_cache['timestamp'] is not None and
                now - _results_cache['timestamp'] < duration):
                return _results_cache['data']
            
            # Get fresh response
            response = f(*args, **kwargs)
            
            # Cache the response
            _results_cache['data'] = response
            _results_cache['timestamp'] = now
            
            return response
        return wrapper
    return decorator

def get_next_run_time():
    """Get next 6 AM MYT run time"""
    malaysia_tz = pytz.timezone('Asia/Kuala_Lumpur')
    now = datetime.now(malaysia_tz)
    next_run = now.replace(hour=6, minute=58, second=0, microsecond=0)
    
    # If it's already past 6 AM, schedule for next day
    if now >= next_run:
        next_run = next_run + timedelta(days=1)
    
    return next_run


def check_title_exists(title):
    """Check if a video with this title has already been processed"""
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute('SELECT 1 FROM videos WHERE title = ?', (title,))
        return c.fetchone() is not None

def init_db():
    """Initialize SQLite database with JSON storage for reasons"""
    db_dir = os.path.dirname(DB_PATH)
    Path(db_dir).mkdir(parents=True, exist_ok=True)
    
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        
        # Videos table
        c.execute('''
            CREATE TABLE IF NOT EXISTS videos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT UNIQUE NOT NULL,
                channel_id TEXT NOT NULL,
                channel_name TEXT NOT NULL,
                title TEXT NOT NULL,
                url TEXT NOT NULL,
                thumbnail_url TEXT,
                views TEXT,
                duration TEXT,
                transcript TEXT,
                published_at TIMESTAMP,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Coin analysis table
        c.execute('''
            CREATE TABLE IF NOT EXISTS coin_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT NOT NULL,
                coin_mentioned TEXT NOT NULL,
                reasons TEXT NOT NULL,  -- This will store JSON array of reasons
                indicator TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (video_id) REFERENCES videos(video_id)
            )
        ''')
        
        # Processing logs table
        c.execute('''
            CREATE TABLE IF NOT EXISTS processing_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel TEXT NOT NULL,
                status TEXT NOT NULL,
                message TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Coin edits table (new)
        c.execute('''
            CREATE TABLE IF NOT EXISTS coin_edits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                current_name TEXT NOT NULL,
                new_name TEXT NOT NULL,
                edited_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Video edits table (new)
        c.execute('''
            CREATE TABLE IF NOT EXISTS video_edits (
                edit_id INTEGER,
                video_id TEXT NOT NULL,
                FOREIGN KEY (edit_id) REFERENCES coin_edits(id),
                FOREIGN KEY (video_id) REFERENCES videos(video_id)
            )
        ''')

def get_caption_tracks(video_id, api_key):
    """Get available caption tracks for a video"""
    url = f"https://youtube.googleapis.com/youtube/v3/captions?part=snippet&videoId={video_id}&key={api_key}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        if 'items' in data:
            return data['items']
        return []
    except Exception as e:
        print(f"Error getting caption tracks: {str(e)}")
        return []

def get_transcript_v2(video_id, api_key):
    """Get transcript with support for Indonesian and English"""
    try:
        print("\nFetching transcript...")
        
        # Get transcript list
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Try to get English transcript first
        try:
            transcript = transcript_list.find_transcript(['en'])
        except:
            # If no English transcript, try Indonesian and translate
            try:
                transcript = transcript_list.find_transcript(['id'])
                transcript = transcript.translate('en')
            except:
                print("Could not find or translate any available transcripts")
                return None
        
        # Get the actual transcript
        transcript_data = transcript.fetch()
        
        # Sort and combine transcript
        transcript_data.sort(key=lambda x: x['start'])
        full_transcript = " ".join(entry['text'] for entry in transcript_data)
        
        return full_transcript.strip()
    except Exception as e:
        print(f"Error getting transcript: {str(e)}")
        return None

def get_channel_id_by_handle(handle, api_key):
    """Get channel ID from handle with improved error handling"""
    try:
        # Clean up handle and remove @ if present
        handle = handle.strip().replace('@', '')
        
        # First try searching by handle directly
        url = f"https://youtube.googleapis.com/youtube/v3/search?part=snippet&type=channel&q={handle}&maxResults=1&key={api_key}"
        response = requests.get(url)
        response.raise_for_status()  # Raise exception for bad status codes
        data = response.json()
        
        # Debug print
        print(f"\nDebug - Search response for {handle}:")
        print(json.dumps(data, indent=2))
        
        if 'items' in data and data['items']:
            for item in data['items']:
                if ('id' in item and 
                    'channelId' in item['id'] and 
                    'snippet' in item and 
                    'title' in item['snippet']):
                    return {
                        'channelId': item['id']['channelId'],
                        'channelName': item['snippet']['title']
                    }
        
        # If direct search fails, try using channels endpoint
        custom_url = handle.lower()
        url = f"https://youtube.googleapis.com/youtube/v3/channels?part=snippet&forHandle={custom_url}&key={api_key}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        if 'items' in data and data['items']:
            channel = data['items'][0]
            return {
                'channelId': channel['id'],
                'channelName': channel['snippet']['title']
            }
        
        # If both methods fail, try one more time with channel lookup
        url = f"https://youtube.googleapis.com/youtube/v3/channels?part=snippet&forUsername={handle}&key={api_key}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        if 'items' in data and data['items']:
            channel = data['items'][0]
            return {
                'channelId': channel['id'],
                'channelName': channel['snippet']['title']
            }
        
        print(f"Warning: Could not find channel for handle: {handle}")
        return None
        
    except requests.exceptions.RequestException as e:
        print(f"Network error for handle {handle}: {str(e)}")
        return None
    except KeyError as e:
        print(f"Unexpected response structure for handle {handle}: {str(e)}")
        return None
    except Exception as e:
        print(f"Unexpected error for handle {handle}: {str(e)}")
        return None

def get_uploads_playlist_id(channel_id, api_key):
    """Get uploads playlist ID"""
    url = f"https://youtube.googleapis.com/youtube/v3/channels?part=contentDetails&id={channel_id}&key={api_key}"
    response = requests.get(url)
    data = response.json()
    
    if 'items' in data and len(data['items']) > 0:
        return data['items'][0]['contentDetails']['relatedPlaylists']['uploads']
    return None

def get_video_details(video_id, api_key):
    """Get video metadata"""
    url = f"https://youtube.googleapis.com/youtube/v3/videos?part=snippet,contentDetails,statistics&id={video_id}&key={api_key}"
    response = requests.get(url)
    data = response.json()
    
    if 'items' in data and len(data['items']) > 0:
        video = data['items'][0]
        return {
            'duration': video['contentDetails']['duration'],
            'thumbnail': video['snippet']['thumbnails']['high']['url'],
            'views': video['statistics'].get('viewCount', 'N/A')
        }
    return None

def get_transcript(video_id):
    """Get transcript with support for Indonesian and English"""
    try:
        print("\nFetching transcript...")
        
        # First, check available transcripts
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Try to get English transcript first
        try:
            transcript = transcript_list.find_transcript(['en'])
        except:
            # If no English transcript, try to get Indonesian auto-generated one
            # and translate it to English
            try:
                transcript = transcript_list.find_transcript(['id'])
                transcript = transcript.translate('en')
            except:
                print("Could not find or translate any available transcripts")
                return None
        
        # Get the actual transcript
        transcript_data = transcript.fetch()
        
        # Sort transcript entries by start time and combine
        transcript_data.sort(key=lambda x: x['start'])
        full_transcript = " ".join(entry['text'] for entry in transcript_data)
        
        return full_transcript.strip()
    except Exception as e:
        print(f"Error getting transcript: {str(e)}")
        try:
            available_transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
            print("\nAvailable transcripts:")
            for transcript in available_transcripts:
                print(f"- {transcript.language_code} ({transcript.language})")
        except:
            print("Could not retrieve available transcripts")
        return None

def is_short(duration):
    """Check if video is a Short"""
    if not duration:
        return False
    if 'H' in duration or 'M' in duration:
        return False
    elif 'S' in duration:
        seconds = int(duration.replace('PT', '').replace('S', ''))
        return seconds < 90
    return True

def check_recent_videos(playlist_id, api_key, channel_info):
    """Check for videos from the past day"""
    videos = []
    next_page_token = None
    now = datetime.now(pytz.UTC)
    day_ago = now - timedelta(days=3)
    
    while True:
        base_url = f"https://youtube.googleapis.com/youtube/v3/playlistItems?part=snippet&playlistId={playlist_id}&maxResults=50&key={api_key}"
        url = f"{base_url}&pageToken={next_page_token}" if next_page_token else base_url
        
        try:
            response = requests.get(url)
            data = response.json()
            
            if 'items' not in data:
                break
                
            found_old_video = False
            for video in data['items']:
                publish_time = datetime.strptime(
                    video['snippet']['publishedAt'], 
                    '%Y-%m-%dT%H:%M:%SZ'
                ).replace(tzinfo=pytz.UTC)
                
                if publish_time > day_ago:
                    video_id = video['snippet']['resourceId']['videoId']
                    
                    # Check if video title exists before getting details
                    if check_title_exists(video['snippet']['title']):
                        print(f"Skipping duplicate video: {video['snippet']['title']}")
                        continue
                        
                    video_details = get_video_details(video_id, api_key)
                    
                    if video_details and not is_short(video_details['duration']):
                        videos.append({
                            'video_id': video_id,
                            'video_title': video['snippet']['title'],
                            'video_url': f"https://youtube.com/watch?v={video_id}",
                            'thumbnail_url': video_details['thumbnail'],
                            'channel_id': channel_info['channelId'],
                            'channel_name': channel_info['channelName'],
                            'views': video_details['views'],
                            'duration': video_details['duration'],
                            'published_at': publish_time.isoformat()
                        })
                else:
                    found_old_video = True
            
            if found_old_video:
                break
                
            next_page_token = data.get('nextPageToken')
            if not next_page_token:
                break
                
        except Exception as e:
            print(f"Error checking videos: {str(e)}")
            break
    
    if videos:
        videos.sort(key=lambda x: x['published_at'], reverse=True)
    
    return videos if videos else None

def chunk_transcript(text, max_length=15000):
    """Split transcript into manageable chunks"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 <= max_length:
            current_chunk.append(word)
            current_length += len(word) + 1
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = len(word) + 1
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def summarize_text(text, model, max_retries=3, retry_delay=60):
    """Analyze transcript with Gemini, with retry logic and chunk handling"""
    try:
        # Split transcript if too long
        chunks = chunk_transcript(text)
        all_analyses = []
        
        for chunk_index, chunk in enumerate(chunks):
            print(f"\nProcessing chunk {chunk_index + 1} of {len(chunks)}...")
            
            for attempt in range(max_retries):
                try:
                    prompt = f"""Analyze the given text and extract information about cryptocurrencies mentioned. Follow these rules EXACTLY:

RULES FOR ANALYSIS:
1. Each coin should be analyzed independently
2. For each coin:
   - List ONLY reasons specifically about that coin
   - The indicator field must ONLY describe that specific coin's outlook
   - Keep indicator field to 5 words maximum
   - Do NOT mention other coins in the indicator field

INDICATOR GUIDELINES:
- Use ONLY these formats:
  * "Bullish, or [with specific condition]"
  * "Bearish or [with specific condition]"

REASONS GUIDELINES:
- Each reason must be:
  * About ONLY the specific coin being analyzed
  * A detailed reasons with reasonable example(if there's any from the text)
  * Self-contained (no references to "it" or "this")
  * Not mentioning other coins' influences

Format your response EXACTLY like this example:
[
    {{
        "coin_mentioned": "CoinA",
        "reason": [
            "Direct reason about CoinA only",
            "Another specific reason about CoinA",
            "Third reason focusing on CoinA"
        ],
        "indicator": "Bullish [if there's specific condition, state it]"
    }}
]

IMPORTANT:
- Output ONLY valid JSON
- No text before or after the JSON
- Each coin's analysis must be completely independent
- Never mention relationships between coins in the indicator field

Analyze this text:
{chunk}"""

                    print(f"\nAnalyzing chunk {chunk_index + 1}...")
                    with tqdm(total=100, desc="Analyzing", bar_format='{l_bar}{bar}| {n_fmt}%') as pbar:
                        response = model.generate_content(prompt)
                        pbar.update(100)

                    if response:
                        clean_response = response.text.strip()
                        if clean_response.startswith('```json'):
                            clean_response = clean_response[7:]
                        if clean_response.startswith('```'):
                            clean_response = clean_response[3:]
                        if clean_response.endswith('```'):
                            clean_response = clean_response[:-3]
                        
                        chunk_analysis = json.loads(clean_response.strip())
                        if chunk_analysis and isinstance(chunk_analysis, list):
                            all_analyses.extend(chunk_analysis)
                        break  # Successful analysis, move to next chunk
                
                except Exception as e:
                    if "429" in str(e) and attempt < max_retries - 1:  # Rate limit error
                        print(f"\nRate limit reached. Waiting {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    elif attempt < max_retries - 1:  # Other errors
                        print(f"Error in analysis attempt {attempt + 1}: {str(e)}")
                        time.sleep(retry_delay)
                        continue
                    else:
                        print(f"Failed all retry attempts for chunk {chunk_index + 1}")
                        raise
        
        # Deduplicate and consolidate analyses
        seen_coins = set()
        final_analyses = []
        for analysis in all_analyses:
            if analysis['coin_mentioned'] not in seen_coins:
                seen_coins.add(analysis['coin_mentioned'])
                final_analyses.append(analysis)
        
        print(f"\nAnalysis completed. Found {len(final_analyses)} unique coins.")
        return final_analyses if final_analyses else None

    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        return None

def verify_analysis_storage(video_id, analysis):
    """Verify that analysis was stored correctly"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            
            # Check if analysis was stored
            c.execute('SELECT COUNT(*) FROM coin_analysis WHERE video_id = ?', (video_id,))
            count = c.fetchone()[0]
            
            if count != len(analysis):
                print(f"WARNING: Expected {len(analysis)} analyses for video {video_id}, but found {count}")
                
                # Print what was supposed to be stored
                print("\nAttempted to store:")
                print(json.dumps(analysis, indent=2))
                
                # Print what's actually in the database
                c.execute('''
                    SELECT coin_mentioned, reasons, indicator 
                    FROM coin_analysis 
                    WHERE video_id = ?
                ''', (video_id,))
                stored = c.fetchall()
                print("\nActually stored:")
                for row in stored:
                    print(json.dumps({
                        'coin_mentioned': row[0],
                        'reasons': json.loads(row[1]),
                        'indicator': row[2]
                    }, indent=2))
                    
            return count == len(analysis)
    except Exception as e:
        print(f"Error verifying analysis storage: {str(e)}")
        return False

# Then modify store_results to use this:
def store_results(video_info, transcript, analysis):
    """Store results in database with JSON reasons and verification"""
    print("\nStoring analysis results:")
    print(json.dumps(analysis, indent=2))
    
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        
        # Store video info
        c.execute('''
            INSERT OR REPLACE INTO videos 
            (video_id, channel_id, channel_name, title, url, thumbnail_url, views, duration, transcript, published_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            video_info['video_id'],
            video_info['channel_id'],
            video_info['channel_name'],
            video_info['video_title'],
            video_info['video_url'],
            video_info['thumbnail_url'],
            video_info['views'],
            video_info['duration'],
            transcript,
            video_info['published_at']
        ))
        
        # Store analysis results
        if analysis and len(analysis) > 0:
            for coin_data in analysis:
                try:
                    # Store reasons as JSON array
                    reasons_json = json.dumps(coin_data['reason'])
                    
                    c.execute('''
                        INSERT INTO coin_analysis
                        (video_id, coin_mentioned, reasons, indicator)
                        VALUES (?, ?, ?, ?)
                    ''', (
                        video_info['video_id'],
                        coin_data['coin_mentioned'],
                        reasons_json,
                        coin_data['indicator']
                    ))
                except Exception as e:
                    print(f"Error storing analysis for {coin_data['coin_mentioned']}: {str(e)}")
            
            # Verify storage
            if not verify_analysis_storage(video_info['video_id'], analysis):
                print(f"WARNING: Analysis storage verification failed for video {video_info['video_id']}")
        else:
            print(f"Warning: No analysis data for video {video_info['video_id']}")


def process_channels():
    """Main processing function with updated transcript retrieval"""
    global analysis_status, automation_status
    
    while True:
        try:
            # Get current time in MYT
            malaysia_tz = pytz.timezone('Asia/Kuala_Lumpur')
            now = datetime.now(malaysia_tz)
            
            # If automation is scheduled, check timing
            if automation_status["is_scheduled"]:
                next_run = automation_status["next_run_time"]
                
                # Wait until next run time
                if now < next_run:
                    time.sleep(60)  # Check every minute
                    continue
                
                # Update next run time for tomorrow
                automation_status["next_run_time"] = next_run + timedelta(days=1)
                automation_status["last_execution"] = now
            
            # Reset analysis status before starting
            analysis_status.update({
                "is_running": True,
                "current_channel": None,
                "current_stage": "initializing",
                "channels_processed": 0,
                "total_channels": len(ALL_CHANNELS),
                "errors": [],
                "processed_videos": 0,
                "skipped_videos": 0
            })
            
            # Configure Gemini
            genai.configure(api_key=os.getenv('GEMINI_API_KEY_3'))
            model = genai.GenerativeModel(model_name="gemini-1.5-flash")
            
            # Process each channel
            for handle in ALL_CHANNELS:
                try:
                    print(f"\nProcessing channel: {handle}")
                    analysis_status["current_channel"] = handle
                    analysis_status["current_stage"] = "Getting channel info"
                    
                    channel_info = get_channel_id_by_handle(handle, API_KEY)
                    if not channel_info:
                        raise Exception(f"Failed to get channel info for {handle}")
                    
                    analysis_status["current_stage"] = "Getting playlist"
                    playlist_id = get_uploads_playlist_id(channel_info['channelId'], API_KEY)
                    
                    if not playlist_id:
                        raise Exception(f"Could not get uploads playlist for {handle}")
                    
                    analysis_status["current_stage"] = "Checking recent videos"
                    videos = check_recent_videos(playlist_id, API_KEY, channel_info)
                    
                    if videos:
                        for video_info in videos:
                            analysis_status["current_stage"] = f"Processing video: {video_info['video_title']}"
                            
                            # Use new transcript method
                            transcript = get_transcript_v2(video_info['video_id'], API_KEY)
                            
                            if transcript:
                                analysis = summarize_text(transcript, model)
                                if analysis:
                                    store_results(video_info, transcript, analysis)
                                    print(f"Successfully processed: {video_info['video_title']}")
                                    analysis_status["processed_videos"] += 1
                            else:
                                print(f"No transcript available: {video_info['video_title']}")
                                analysis_status["skipped_videos"] += 1
                    else:
                        print(f"No recent videos found for {handle}")
                    
                    analysis_status["channels_processed"] += 1
                    
                except Exception as e:
                    error_msg = f"{handle}: {str(e)}"
                    print(f"Error processing channel {handle}: {str(e)}")
                    analysis_status["errors"].append(error_msg)
                
                time.sleep(1)  # Rate limiting
            
            # Reset running status but keep other stats for logging
            analysis_status["is_running"] = False
            analysis_status["current_channel"] = None
            analysis_status["current_stage"] = None
            
            # If not automated, break the loop
            if not automation_status["is_scheduled"]:
                break
                
            print(f"Completed run at {now}. Next run scheduled for {automation_status['next_run_time']}")
            
        except Exception as e:
            print(f"Error in process_channels: {str(e)}")
            if not automation_status["is_scheduled"]:
                break
            time.sleep(60)  # Wait before retrying
@app.route('/youtube/start')
def start_analysis():
    """Start the analysis process with automation"""
    global analysis_status, automation_status
    
    if analysis_status["is_running"]:
        current_progress = {
            "current_channel": analysis_status["current_channel"],
            "current_stage": analysis_status["current_stage"],
            "processed": {
                "channels": analysis_status["channels_processed"],
                "videos": analysis_status["processed_videos"]
            },
            "skipped_videos": analysis_status["skipped_videos"],
            "total_channels": analysis_status["total_channels"],
            "next_run": automation_status["next_run_time"].isoformat() if automation_status["next_run_time"] else None
        }
        
        return jsonify({
            "status": "error",
            "message": "Analysis is currently running",
            "progress": current_progress
        }), 409

    # Set up automation
    automation_status["is_scheduled"] = True
    automation_status["next_run_time"] = get_next_run_time()
    
    # Reset status
    analysis_status.update({
        "is_running": True,
        "current_channel": None,
        "current_stage": "initializing",
        "channels_processed": 0,
        "total_channels": len(ALL_CHANNELS),
        "errors": [],
        "processed_videos": 0,
        "skipped_videos": 0,
        "start_time": datetime.now().isoformat()
    })
    
    # Start processing in background
    thread = threading.Thread(target=process_channels)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        "status": "success",
        "message": "YouTube analysis process started with automation",
        "automation": {
            "is_scheduled": True,
            "next_run": automation_status["next_run_time"].isoformat(),
            "message": f"Next run scheduled for 6 AM MYT"
        }
    })

@app.route('/test/<video_id>')
def test_transcript(video_id):
    """Test endpoint for transcript retrieval using URL parameter"""
    try:
        if not video_id:
            return jsonify({
                "status": "error",
                "message": "Please provide a YouTube video ID."
            }), 400
            
        # Get video details first
        video_details = get_video_details(video_id, API_KEY)
        if not video_details:
            return jsonify({
                "status": "error",
                "message": "Could not fetch video details. Invalid video ID."
            }), 404
            
        # Get available caption tracks
        caption_tracks = get_caption_tracks(video_id, API_KEY)
        
        # Try to get transcript
        transcript = get_transcript_v2(video_id, API_KEY)
        
        return jsonify({
            "status": "success",
            "data": {
                "video_id": video_id,
                "video_details": video_details,
                "available_captions": [
                    {
                        "language": track['snippet']['language'],
                        "type": track['snippet']['trackKind'],
                        "last_updated": track['snippet']['lastUpdated']
                    } for track in caption_tracks
                ],
                "transcript": {
                    "available": transcript is not None,
                    "content": transcript if transcript else None,
                    "length": len(transcript) if transcript else 0
                }
            }
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/youtube/log')
def get_status():
    """Get current status including automation"""
    return jsonify({
        "status": "success",
        "data": {
            "is_running": analysis_status["is_running"],
            "current_channel": analysis_status["current_channel"],
            "current_stage": analysis_status["current_stage"],
            "progress": {
                "processed": {
                    "channels": analysis_status["channels_processed"],
                    "videos": analysis_status["processed_videos"]
                },
                "skipped_videos": analysis_status["skipped_videos"],
                "total_channels": analysis_status["total_channels"]
            },
            "errors": analysis_status["errors"],
            "automation": {
                "is_scheduled": automation_status["is_scheduled"],
                "next_run": automation_status["next_run_time"].isoformat() if automation_status["next_run_time"] else None,
                "last_execution": automation_status["last_execution"].isoformat() if automation_status["last_execution"] else None
            }
        }
    })

@app.route('/youtube/get')
def get_results():
    """Get all results with JSON reasons"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            
            # First get all videos
            c.execute('''
                SELECT 
                    id, video_id, channel_id, channel_name, 
                    title, url, thumbnail_url, views, 
                    duration, transcript, published_at, processed_at
                FROM videos 
                ORDER BY processed_at DESC
            ''')
            videos = [dict(row) for row in c.fetchall()]
            
            # Then get analyses for each video
            for video in videos:
                c.execute('''
                    SELECT 
                        coin_mentioned, reasons, indicator
                    FROM coin_analysis 
                    WHERE video_id = ?
                ''', (video['video_id'],))
                
                analyses = []
                for row in c.fetchall():
                    analyses.append({
                        'coin_mentioned': row['coin_mentioned'],
                        'reason': json.loads(row['reasons']),  # Parse JSON reasons back into array
                        'indicator': row['indicator']
                    })
                video['analyses'] = analyses
            
            return jsonify({
                "status": "success",
                "data": {
                    "total_videos": len(videos),
                    "videos": videos
                }
            })
            
    except Exception as e:
        print(f"Error in get_results: {str(e)}")  # Debug print
        return jsonify({
            "status": "error",
            "message": str(e)
        })
    # Add this as a new endpoint

@app.route('/check_time')
def check_time():
    """Get current local time in Malaysia timezone"""
    try:
        # Get current time in UTC
        utc_time = datetime.now(pytz.UTC)
        
        # Convert to Malaysia timezone (MYT - Malaysia Time)
        malaysia_tz = pytz.timezone('Asia/Kuala_Lumpur')
        local_time = utc_time.astimezone(malaysia_tz)
        
        return jsonify({
            "status": "success",
            "data": {
                "utc_time": utc_time.isoformat(),
                "local_time": local_time.isoformat(),
                "timezone": "Malaysia Time (MYT)",
                "formatted_time": local_time.strftime("%Y-%m-%d %H:%M:%S %Z"),
                # Adding a more human-readable format
                "readable_time": local_time.strftime("%A, %d %B %Y, %I:%M:%S %p")
            }
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
    
@app.route('/youtube/coins')
def get_unique_coins():
    """Get list of all unique coins mentioned in analyses"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            
            # Simple query to get just unique coin names
            c.execute('''
                SELECT DISTINCT coin_mentioned
                FROM coin_analysis 
                ORDER BY coin_mentioned ASC
            ''')
            
            coins = [row[0] for row in c.fetchall()]
            
            return jsonify({
                "status": "success",
                "data": {
                    "coins": coins
                }
            })
            
    except Exception as e:
        print(f"Error in get_unique_coins: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/youtube/clean-coins')
def clean_coin_names():
    """Analyze and suggest standardized coin names using Gemini AI without updating the database"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            
            # Get all unique coin names
            c.execute('SELECT DISTINCT coin_mentioned FROM coin_analysis ORDER BY coin_mentioned')
            coins = [row[0] for row in c.fetchall()]
            
            # Configure Gemini
            genai.configure(api_key="AIzaSyBks3X3tJ5md4vr_iRl5J9vi-DTjkjzQx8")
            model = genai.GenerativeModel(model_name="gemini-1.5-flash")
            
            # Process coins in batches to avoid token limits
            batch_size = 50
            all_suggestions = []
            
            for i in range(0, len(coins), batch_size):
                batch = coins[i:i + batch_size]
                coin_list = "\n".join(batch)
                
                prompt = """Clean and standardize these cryptocurrency names. Follow these rules:

1. For each coin, provide the standardized name and original name
2. Merge obvious duplicates (e.g., "Bitcoin", "BTC", "BITCOIN" should all map to "Bitcoin")
3. Keep proper capitalization (e.g., "Bitcoin" not "BITCOIN")
4. Remove redundant entries (e.g., "Bitcoin (BTC)" should just be "Bitcoin")
5. Identify and merge variations (e.g., "Eth", "ETH", "Ethereum" should all be "Ethereum")
6. For uncertain or unique entries, keep the original name
7. Keep special project names or unique identifiers intact
8. For AI-related coins, maintain their unique identifiers

Return a valid JSON array where each item has this exact structure:
{
    "original": "the original coin name",
    "standardized": "the cleaned coin name",
    "reason": "explanation of why this standardization was chosen"
}

Clean these coin names:
""" + coin_list

                try:
                    response = model.generate_content(prompt)
                    clean_response = response.text.strip()
                    
                    # Remove JSON code block markers if present
                    if clean_response.startswith('```json'):
                        clean_response = clean_response[7:]
                    elif clean_response.startswith('```'):
                        clean_response = clean_response[3:]
                    if clean_response.endswith('```'):
                        clean_response = clean_response[:-3]
                    
                    # Parse the JSON response
                    batch_suggestions = json.loads(clean_response.strip())
                    all_suggestions.extend(batch_suggestions)
                    
                except Exception as e:
                    print(f"Error processing batch: {str(e)}")
                    print(f"Raw response: {clean_response}")
                    continue
            
            # Process and organize the suggestions
            if all_suggestions:
                # Create summary of changes
                changes_summary = {
                    "total_original": len(coins),
                    "total_standardized": len(set(s['standardized'] for s in all_suggestions)),
                    "suggestions": all_suggestions,
                    "major_groups": {}
                }
                
                # Group similar coins
                for suggestion in all_suggestions:
                    std_name = suggestion['standardized']
                    if std_name not in changes_summary['major_groups']:
                        changes_summary['major_groups'][std_name] = []
                    if suggestion['original'] != std_name:
                        changes_summary['major_groups'][std_name].append(suggestion['original'])
                
                # Remove empty groups
                changes_summary['major_groups'] = {k: v for k, v in changes_summary['major_groups'].items() if v}
                
                return jsonify({
                    "status": "success",
                    "data": changes_summary
                })
            
            return jsonify({
                "status": "error",
                "message": "No cleaning suggestions were generated"
            }), 400
            
    except Exception as e:
        print(f"Error in clean_coin_names: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
    
@app.route('/test/new-reason/<video_id>')
def analyze_mentions_type(video_id):
    """Analyze mentions with standardized format, project/protocol distinction, and source information"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            
            # Get mentions and transcript for this video
            c.execute('''
                SELECT v.transcript, ca.coin_mentioned
                FROM videos v
                JOIN coin_analysis ca ON v.video_id = ca.video_id
                WHERE v.video_id = ?
            ''', (video_id,))
            
            results = c.fetchall()
            if not results:
                return jsonify({
                    "status": "error",
                    "message": "No data found for this video ID"
                }), 404
            
            transcript = results[0][0]  # Get transcript from first row
            mentions = list(set(row[1] for row in results))  # Get unique mentions
            
            # Configure Gemini
            genai.configure(api_key="AIzaSyBks3X3tJ5md4vr_iRl5J9vi-DTjkjzQx8")
            model = genai.GenerativeModel(model_name="gemini-1.5-pro")
            
            prompt = f"""Analyze the given text and extract information about cryptocurrencies mentioned. Follow these rules EXACTLY:

CLASSIFICATION RULES:
1. Distinguish between:
   - COINS: Actual cryptocurrencies with their own token/coin
   - PROTOCOLS: Blockchain infrastructure or DeFi platforms
   - PROJECTS: Blockchain applications, services, or AI projects
   - NETWORKS: Layer-1 blockchains
2. Add a "type" field to indicate if it's a "COIN", "PROTOCOL", "PROJECT", or "NETWORK"
3. NFT collections should be marked as PROJECTS
4. AI projects without tokens should be marked as PROJECTS
5. Layer-1 blockchains with native tokens should be marked as both NETWORK and COIN
6. IT IS A MUST THAT YOU PROVIDE SOURCE BASED ON WHAT THE TRNASCRIPT STATED WHETHER FROM WEBSITE, ARTICLE, ETC

RULES FOR ANALYSIS:
1. Each coin should be analyzed independently
2. For each coin:
   - List ONLY reasons specifically about that coin
   - For each reason, MUST include specific quotes or examples from the transcript IF AVAILABLE
   - Include source information when mentioned (e.g., data sources, experts, organizations)
   - The indicator field must ONLY describe that specific coin's outlook
   - Keep indicator field to 5 words maximum
   - Do NOT mention other coins in the indicator field

INDICATOR GUIDELINES:
- Use ONLY these formats:
  * "Bullish, or [with specific condition]"
  * "Bearish or [with specific condition]"

REASONS AND SOURCES GUIDELINES:
- Each reason must be:
  * About ONLY the specific coin being analyzed
  * Include DIRECT QUOTES from the transcript when available
  * Include any mentioned sources (e.g., "According to CoinGecko...", "Data from Glassnode shows...")
  * Include specific examples and data points mentioned
  * Self-contained (no references to "it" or "this")
  * Not mentioning other coins' influences

Format your response EXACTLY like this example:
[
    {{
        "coin_mentioned": "CoinA",
        "type": "COIN/PROTOCOL/PROJECT/NETWORK",
        "reason": [
            "According to Glassnode data mentioned in the video, CoinA saw a 25% increase in active addresses",
            "The host cites CoinMetrics report showing 'CoinA's transaction volume reached $5B in December'",
            "Direct quote from transcript: 'Our analysis shows CoinA's mining hashrate has doubled'"
        ],
        "indicator": "Bullish [if there's specific condition, state it]"
    }}
]

IMPORTANT:
- Output ONLY valid JSON
- No text before or after the JSON
- Each coin's analysis must be completely independent
- Never mention relationships between coins in the indicator field
- ALWAYS include direct quotes and sources when they appear in the transcript
- When data or statistics are mentioned, include them with their source
- If specific experts, organizations, or tools are cited, include them

Analyze this text:
{transcript}"""

            try:
                response = model.generate_content(prompt)
                clean_response = response.text.strip()
                
                # Remove JSON code block markers if present
                if clean_response.startswith('```json'):
                    clean_response = clean_response[7:]
                elif clean_response.startswith('```'):
                    clean_response = clean_response[3:]
                if clean_response.endswith('```'):
                    clean_response = clean_response[:-3]
                
                # Parse the JSON response
                analysis = json.loads(clean_response.strip())
                
                # Create summary statistics
                type_counts = {}
                for item in analysis:
                    item_type = item.get('type', 'UNKNOWN')
                    if item_type not in type_counts:
                        type_counts[item_type] = 0
                    type_counts[item_type] += 1
                
                return jsonify({
                    "status": "success",
                    "data": {
                        "video_id": video_id,
                        "total_mentions": len(mentions),
                        "type_distribution": type_counts,
                        "analysis": analysis
                    }
                })
                
            except Exception as e:
                print(f"Error processing analysis: {str(e)}")
                print(f"Raw response: {clean_response}")
                return jsonify({
                    "status": "error",
                    "message": f"Error processing analysis: {str(e)}"
                }), 500
            
    except Exception as e:
        print(f"Error in analyze_mentions_type: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
    
@app.route('/test/channel/<channel_handle>')
def test_channel(channel_handle):
    """Test endpoint to get recent video information for a specific channel"""
    try:
        # Clean up handle if needed
        if not channel_handle.startswith('@'):
            channel_handle = f'@{channel_handle}'
            
        # Get channel info
        channel_info = get_channel_id_by_handle(channel_handle, API_KEY)
        if not channel_info:
            return jsonify({
                "status": "error",
                "message": f"Could not find channel information for {channel_handle}"
            }), 404
            
        # Get playlist ID
        playlist_id = get_uploads_playlist_id(channel_info['channelId'], API_KEY)
        if not playlist_id:
            return jsonify({
                "status": "error",
                "message": f"Could not get uploads playlist for {channel_handle}"
            }), 404
            
        # Get recent videos (using existing function but modified for this endpoint)
        url = f"https://youtube.googleapis.com/youtube/v3/playlistItems?part=snippet&playlistId={playlist_id}&maxResults=5&key={API_KEY}"
        
        response = requests.get(url)
        data = response.json()
        
        if 'items' not in data:
            return jsonify({
                "status": "error",
                "message": "No videos found"
            }), 404
            
        # Process video information
        videos = []
        for video in data['items']:
            video_info = {
                'title': video['snippet']['title'],
                'video_id': video['snippet']['resourceId']['videoId'],
                'published_at': video['snippet']['publishedAt'],
                'channel_name': channel_info['channelName'],
                'video_url': f"https://youtube.com/watch?v={video['snippet']['resourceId']['videoId']}"
            }
            videos.append(video_info)
            
        return jsonify({
            "status": "success",
            "data": {
                "channel_name": channel_info['channelName'],
                "channel_id": channel_info['channelId'],
                "recent_videos": videos
            }
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
    
@app.route('/youtube/get-light')
@cache_response(CACHE_DURATION)
def get_light_results():
    """Get all results without transcripts for lighter payload"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            
            # Get ALL edit history for name overrides
            c.execute('''
                SELECT 
                    current_name,
                    new_name
                FROM coin_edits
                ORDER BY edited_at DESC
            ''')
            
            # Create a mapping of old names to new names
            name_mapping = {row['current_name']: row['new_name'] for row in c.fetchall()}
            
            # Get all videos except transcript field
            c.execute('''
                SELECT 
                    id, video_id, channel_id, channel_name, 
                    title, url, thumbnail_url, views, 
                    duration, published_at, processed_at
                FROM videos 
                ORDER BY processed_at DESC
            ''')
            videos = [dict(row) for row in c.fetchall()]
            
            # Get analyses for each video and apply name overrides
            for video in videos:
                c.execute('''
                    SELECT 
                        coin_mentioned, reasons, indicator
                    FROM coin_analysis 
                    WHERE video_id = ?
                ''', (video['video_id'],))
                
                analyses = []
                for row in c.fetchall():
                    # Apply name override if exists
                    coin_name = row['coin_mentioned']
                    if coin_name in name_mapping:
                        coin_name = name_mapping[coin_name]
                        
                    analyses.append({
                        'coin_mentioned': coin_name,
                        'reason': json.loads(row['reasons']),
                        'indicator': row['indicator']
                    })
                video['analyses'] = analyses
            
            return jsonify({
                "status": "success",
                "data": {
                    "total_videos": len(videos),
                    "videos": videos,
                    "cached": False
                }
            })
            
    except Exception as e:
        print(f"Error in get_light_results: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# Add a cache control endpoint (optional)
@app.route('/youtube/clear-cache')
def clear_cache():
    """Clear the API response cache"""
    global _results_cache
    _results_cache = {
        'data': None,
        'timestamp': None
    }
    return jsonify({
        "status": "success",
        "message": "Cache cleared successfully"
    })

# Optional: Add a new endpoint to get a single video's transcript if needed
@app.route('/youtube/transcript/<video_id>')
def get_video_transcript(video_id):
    """Get transcript for a specific video"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            
            c.execute('SELECT transcript FROM videos WHERE video_id = ?', (video_id,))
            result = c.fetchone()
            
            if result:
                return jsonify({
                    "status": "success",
                    "data": {
                        "video_id": video_id,
                        "transcript": result[0]
                    }
                })
            else:
                return jsonify({
                    "status": "error",
                    "message": "Video not found"
                }), 404
                
    except Exception as e:
        print(f"Error getting transcript: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

from flask import request, jsonify

from flask import request, jsonify

@app.route('/youtube/override-coin', methods=['POST'])
def override_coin_name():
    """Override a coin name and automatically find affected videos"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['current_name', 'new_name']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    "status": "error",
                    "message": f"Missing required field: {field}"
                }), 400

        current_name = data['current_name'].strip()
        new_name = data['new_name'].strip()

        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            
            # First check if the new name already exists
            c.execute('''
                SELECT COUNT(*) as count, 
                       GROUP_CONCAT(DISTINCT coin_mentioned) as variations
                FROM coin_analysis 
                WHERE LOWER(coin_mentioned) = LOWER(?)
            ''', (new_name,))
            
            existing = c.fetchone()
            existing_count = existing['count']
            existing_variations = existing['variations'].split(',') if existing['variations'] else []
            
            # Find videos with current name
            c.execute('''
                SELECT DISTINCT 
                    v.video_id,
                    v.title,
                    v.channel_name,
                    v.url,
                    ca.coin_mentioned
                FROM coin_analysis ca
                JOIN videos v ON ca.video_id = v.video_id
                WHERE ca.coin_mentioned = ?
            ''', (current_name,))
            
            affected_videos = [dict(row) for row in c.fetchall()]
            
            if not affected_videos:
                return jsonify({
                    "status": "error",
                    "message": f"No videos found with coin name: {current_name}",
                    "note": f"Existing variations of target name: {existing_variations}" if existing_variations else None
                }), 404
            
            # Insert the edit record
            c.execute('''
                INSERT INTO coin_edits (current_name, new_name)
                VALUES (?, ?)
            ''', (current_name, new_name))
            
            edit_id = c.lastrowid
            
            # Update coin names
            c.execute('''
                UPDATE coin_analysis 
                SET coin_mentioned = ?
                WHERE coin_mentioned = ?
            ''', (new_name, current_name))
            
            # Record affected videos
            for video in affected_videos:
                c.execute('''
                    INSERT INTO video_edits (edit_id, video_id)
                    VALUES (?, ?)
                ''', (edit_id, video['video_id']))

            # Get updated counts after the merge
            c.execute('''
                SELECT COUNT(*) as count
                FROM coin_analysis 
                WHERE LOWER(coin_mentioned) = LOWER(?)
            ''', (new_name,))
            
            new_count = c.fetchone()['count']

            return jsonify({
                "status": "success",
                "data": {
                    "current_name": current_name,
                    "new_name": new_name,
                    "timestamp": datetime.now().isoformat(),
                    "merge_stats": {
                        "existing_mentions": existing_count,
                        "mentions_added": len(affected_videos),
                        "total_after_merge": new_count,
                        "existing_variations": existing_variations
                    },
                    "videos_affected": len(affected_videos),
                    "videos": [{
                        "video_id": video["video_id"],
                        "title": video["title"],
                        "channel": video["channel_name"],
                        "url": video["url"],
                        "original_name": video["coin_mentioned"]
                    } for video in affected_videos]
                }
            })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/youtube/edit-history')
def get_edit_history():
    """Get the history of all coin name edits with details"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            
            # Get all edits with affected video counts
            c.execute('''
                SELECT 
                    ce.id as edit_id,
                    ce.current_name,
                    ce.new_name,
                    ce.edited_at,
                    COUNT(ve.video_id) as affected_videos
                FROM coin_edits ce
                LEFT JOIN video_edits ve ON ce.id = ve.edit_id
                GROUP BY ce.id
                ORDER BY ce.edited_at DESC
            ''')
            
            edits = [dict(row) for row in c.fetchall()]
            
            return jsonify({
                "status": "success",
                "data": {
                    "total_edits": len(edits),
                    "edits": [{
                        "edit_id": edit["edit_id"],
                        "from": edit["current_name"],
                        "to": edit["new_name"],
                        "timestamp": edit["edited_at"],
                        "affected_videos": edit["affected_videos"]
                    } for edit in edits]
                }
            })
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/youtube/undo-edit', methods=['POST'])
def undo_last_edit():
    """Undo the most recent coin name edit"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            
            # Get the most recent edit
            c.execute('''
                SELECT 
                    id as edit_id,
                    current_name,
                    new_name,
                    edited_at
                FROM coin_edits
                ORDER BY edited_at DESC
                LIMIT 1
            ''')
            
            last_edit = c.fetchone()
            
            if not last_edit:
                return jsonify({
                    "status": "error",
                    "message": "No edits to undo"
                }), 404
            
            # Get affected videos
            c.execute('''
                SELECT video_id
                FROM video_edits
                WHERE edit_id = ?
            ''', (last_edit['edit_id'],))
            
            affected_videos = [row[0] for row in c.fetchall()]
            
            # Revert the coin name changes
            c.execute('''
                UPDATE coin_analysis
                SET coin_mentioned = ?
                WHERE video_id IN (
                    SELECT video_id 
                    FROM video_edits 
                    WHERE edit_id = ?
                )
                AND coin_mentioned = ?
            ''', (last_edit['current_name'], last_edit['edit_id'], last_edit['new_name']))
            
            # Delete the edit records
            c.execute('DELETE FROM video_edits WHERE edit_id = ?', (last_edit['edit_id'],))
            c.execute('DELETE FROM coin_edits WHERE id = ?', (last_edit['edit_id'],))
            
            return jsonify({
                "status": "success",
                "data": {
                    "undone_edit": {
                        "edit_id": last_edit['edit_id'],
                        "from": last_edit['current_name'],
                        "to": last_edit['new_name'],
                        "timestamp": last_edit['edited_at'],
                        "affected_videos": len(affected_videos)
                    },
                    "message": f"Successfully reverted {len(affected_videos)} videos from '{last_edit['new_name']}' back to '{last_edit['current_name']}'"
                }
            })
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/youtube/affected-videos/<edit_id>')
def get_affected_videos(edit_id):
    """Get list of videos affected by a specific edit"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            
            c.execute('''
                SELECT 
                    v.video_id,
                    v.title,
                    v.channel_name
                FROM video_edits ve
                JOIN videos v ON ve.video_id = v.video_id
                WHERE ve.edit_id = ?
            ''', (edit_id,))
            
            videos = [dict(row) for row in c.fetchall()]
            
            return jsonify({
                "status": "success",
                "data": {
                    "videos_affected": videos
                }
            })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
    
@app.route('/youtube/daily-summary', methods=['POST'])
def generate_daily_summary():
    """Generate structured summary of cryptocurrency analysis for specific coins"""
    try:
        data = request.get_json()
        
        if 'date' not in data:
            return jsonify({
                "status": "error",
                "message": "Missing required field: date"
            }), 400

        target_date = data['date']
        # Define our target coins
        target_coins = ["Bitcoin", "Ethereum (ETH)", "XRP"]

        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            
            # Get all analyses for the specific date
            c.execute('''
                SELECT 
                    v.video_id,
                    v.title,
                    v.channel_name,
                    v.url,
                    v.processed_at,
                    ca.coin_mentioned,
                    ca.reasons,
                    ca.indicator
                FROM videos v
                JOIN coin_analysis ca ON v.video_id = ca.video_id
                WHERE DATE(v.processed_at) = DATE(?)
                ORDER BY v.processed_at ASC
            ''', (target_date,))
            
            analyses = [dict(row) for row in c.fetchall()]
            
            if not analyses:
                return jsonify({
                    "status": "error",
                    "message": f"No analyses found for {target_date}"
                }), 404

            # Organize data by coin
            coin_data = {coin: [] for coin in target_coins}
            
            for analysis in analyses:
                # Normalize coin names for matching
                current_coin = analysis['coin_mentioned'].strip()
                matched_coin = None
                
                # Manual matching logic
                if any(name.lower() in current_coin.lower() for name in ["bitcoin", "btc"]):
                    matched_coin = "Bitcoin"
                elif any(name.lower() in current_coin.lower() for name in ["ethereum", "eth"]):
                    matched_coin = "Ethereum (ETH)"
                elif any(name.lower() in current_coin.lower() for name in ["xrp", "ripple"]):
                    matched_coin = "XRP"
                
                if matched_coin in target_coins:
                    coin_data[matched_coin].append({
                        "indicator": analysis['indicator'],
                        "reasons": json.loads(analysis['reasons']),
                        "source": {
                            "channel": analysis['channel_name'],
                            "title": analysis['title'],
                            "url": analysis['url']
                        }
                    })

            # Configure Gemini for summary generation
            genai.configure(api_key=os.getenv('GEMINI_API_KEY_1'))
            model = genai.GenerativeModel(model_name="gemini-1.5-pro")
            
            # Prepare detailed analysis for each coin
            coin_summaries = {}
            
            for coin in target_coins:
                if not coin_data[coin]:
                    continue
                    
                coin_prompt = f"""Analyze this {coin} data and create a structured summary:

{json.dumps(coin_data[coin], indent=2)}

Provide analysis in this EXACT format:
{{
    "overall_sentiment": "Brief 1-2 sentence overview of general sentiment",
    "bullish_points": [
        "Clear, specific bullish points with source citations in parentheses",
        "Each point should be a complete, detailed thought"
    ],
    "bearish_points": [
        "Clear, specific bearish points with source citations in parentheses",
        "Each point should be a complete, detailed thought"
    ],
    "key_metrics": [
        "Any specific numbers, statistics, or data points mentioned",
        "Include source and specific values"
    ],
    "channels_mentioned": [
        "List of unique channels discussing this coin"
    ]
}}

Rules:
1. Each point must be specific and detailed
2. Include source citations in parentheses
3. Separate truly bullish and bearish points
4. Focus on factual statements and data
5. Include specific numbers when available"""

                try:
                    response = model.generate_content(coin_prompt)
                    clean_response = response.text.strip()
                    
                    # Remove JSON code block markers if present
                    if clean_response.startswith('```json'):
                        clean_response = clean_response[7:]
                    if clean_response.startswith('```'):
                        clean_response = clean_response[3:]
                    if clean_response.endswith('```'):
                        clean_response = clean_response[:-3]
                    
                    # Additional cleanup for potential JSON issues
                    clean_response = clean_response.strip()
                    
                    try:
                        parsed_response = json.loads(clean_response)
                        coin_summaries[coin] = parsed_response
                    except json.JSONDecodeError as je:
                        print(f"JSON parsing error for {coin}: {str(je)}")
                        print(f"Raw response: {clean_response}")
                        
                        # Attempt to fix common JSON issues
                        clean_response = clean_response.replace('\n', ' ').replace('\r', '')
                        clean_response = clean_response.replace('""', '"')
                        
                        try:
                            parsed_response = json.loads(clean_response)
                            coin_summaries[coin] = parsed_response
                        except:
                            # If still failing, provide structured error response
                            coin_summaries[coin] = {
                                "overall_sentiment": "Error processing analysis",
                                "bullish_points": [],
                                "bearish_points": [],
                                "key_metrics": [],
                                "channels_mentioned": [],
                                "error": "Failed to parse analysis response"
                            }
                            
                except Exception as e:
                    print(f"Error processing {coin}: {str(e)}")
                    coin_summaries[coin] = {
                        "overall_sentiment": "Error processing analysis",
                        "bullish_points": [],
                        "bearish_points": [],
                        "key_metrics": [],
                        "channels_mentioned": [],
                        "error": f"Failed to generate analysis: {str(e)}"
                    }

            # Generate overall market summary
            overall_prompt = """Create a brief overall market summary based on the coin-specific analyses:

{}

Provide a few key points about:
1. General market sentiment
2. Notable patterns across coins
3. Significant developments affecting multiple coins
4. Key disagreements or contrasting views

Keep it concise and fact-based.""".format(json.dumps(coin_summaries, indent=2))

            try:
                response = model.generate_content(overall_prompt)
                market_summary = response.text.strip()
            except Exception as e:
                market_summary = f"Error generating market summary: {str(e)}"

            return jsonify({
                "status": "success",
                "data": {
                    "date": target_date,
                    "metrics": {
                        "videos_analyzed": len(set(a['video_id'] for a in analyses)),
                        "total_mentions": {
                            coin: len(mentions) for coin, mentions in coin_data.items() if mentions
                        }
                    },
                    "market_summary": market_summary,
                    "coin_analysis": coin_summaries
                }
            })

    except Exception as e:
        print(f"Error in generate_daily_summary: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
    
@app.route('/youtube/today-channels')
def get_todays_channels():
    """Get channels that have uploaded videos today with their video details"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            
            # Get current date in Malaysia timezone
            malaysia_tz = pytz.timezone('Asia/Kuala_Lumpur')
            current_date = datetime.now(malaysia_tz).strftime("%Y-%m-%d")
            
            # Modified query to use published_at date and Malaysia timezone
            c.execute('''
                WITH video_data AS (
                    SELECT 
                        v.channel_name,
                        v.channel_id,
                        v.video_id,
                        v.title,
                        v.url,
                        v.thumbnail_url,
                        v.views,
                        v.published_at,
                        COALESCE(
                            (SELECT COUNT(*) 
                             FROM coin_analysis ca2 
                             WHERE ca2.video_id = v.video_id),
                            0
                        ) as coin_count
                    FROM videos v
                    LEFT JOIN coin_analysis ca ON v.video_id = ca.video_id
                    WHERE DATE(v.published_at) = ?
                    GROUP BY v.video_id
                )
                SELECT * FROM video_data
                ORDER BY published_at DESC
            ''', (current_date,))
            
            videos = [dict(row) for row in c.fetchall()]
            
            if not videos:
                return jsonify({
                    "status": "success",
                    "data": {
                        "date": current_date,
                        "channels": [],
                        "total_channels": 0,
                        "total_videos": 0,
                        "metrics": {
                            "average_videos_per_channel": 0,
                            "total_coins_analyzed": 0
                        }
                    }
                })
            
            # Organize videos by channel
            channels = {}
            for video in videos:
                channel_name = video['channel_name']
                if channel_name not in channels:
                    channels[channel_name] = {
                        "channel_name": channel_name,
                        "channel_id": video['channel_id'],
                        "videos": []
                    }
                
                # Add video details to channel
                channels[channel_name]['videos'].append({
                    "video_id": video['video_id'],
                    "title": video['title'],
                    "url": video['url'],
                    "thumbnail_url": video['thumbnail_url'],
                    "views": video['views'],
                    "published_at": video['published_at'],
                    "coins_analyzed": video['coin_count']
                })
            
            # Calculate statistics
            total_videos = sum(len(channel['videos']) for channel in channels.values())
            
            # Convert to list and sort by number of videos
            channels_list = sorted(
                channels.values(),
                key=lambda x: len(x['videos']),
                reverse=True
            )
            
            return jsonify({
                "status": "success",
                "data": {
                    "date": current_date,
                    "channels": channels_list,
                    "total_channels": len(channels),
                    "total_videos": total_videos,
                    "metrics": {
                        "average_videos_per_channel": round(total_videos / len(channels), 2),
                        "total_coins_analyzed": sum(
                            video['coins_analyzed'] 
                            for channel in channels_list 
                            for video in channel['videos']
                        )
                    }
                }
            })
            
    except Exception as e:
        print(f"Error in get_todays_channels: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/youtube/top-coins')
def get_top_coins():
    """Get daily sentiment analysis for specific coins (BTC, ETH, XRP, SOL, BNB) from the last 7 days"""
    try:
        # Define exact coin names to match
        target_coins = ["Bitcoin", "Ethereum", "XRP", "Solana", "BNB"]

        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            
            # Get current time in Malaysia timezone
            malaysia_tz = pytz.timezone('Asia/Kuala_Lumpur')
            current_date = datetime.now(malaysia_tz)
            seven_days_ago = current_date - timedelta(days=7)
            
            # Get analyses for the last 7 days with published_at date
            c.execute('''
                SELECT 
                    ca.coin_mentioned,
                    ca.indicator,
                    DATE(v.published_at) as pub_date
                FROM videos v
                JOIN coin_analysis ca ON v.video_id = ca.video_id
                WHERE 
                    v.published_at >= ? 
                    AND ca.coin_mentioned IN (?, ?, ?, ?, ?)
                ORDER BY v.published_at ASC
            ''', (seven_days_ago.strftime('%Y-%m-%d'), *target_coins))
            
            analyses = [dict(row) for row in c.fetchall()]
            
            if not analyses:
                return jsonify({
                    "status": "error",
                    "message": "No analyses found for the last 7 days"
                }), 404

            # Initialize results dictionary with dates
            date_range = [(current_date - timedelta(days=x)).strftime('%Y-%m-%d') 
                         for x in range(7, -1, -1)]
            results = {coin: {date: {"bullish": 0, "bearish": 0, "neutral": 0} 
                            for date in date_range} 
                      for coin in target_coins}
            
            # Process each analysis
            for analysis in analyses:
                coin = analysis['coin_mentioned']
                pub_date = analysis['pub_date']
                
                if pub_date in results[coin]:
                    # Count sentiment
                    indicator = analysis['indicator'].lower()
                    if 'bullish' in indicator:
                        results[coin][pub_date]['bullish'] += 1
                    elif 'bearish' in indicator:
                        results[coin][pub_date]['bearish'] += 1
                    else:
                        results[coin][pub_date]['neutral'] += 1

            # Format final response
            response_data = {}
            for coin, dates in results.items():
                response_data[coin] = {
                    date: {
                        "bullish": data['bullish'],
                        "bearish": data['bearish'],
                        "neutral": data['neutral'],
                        "total": data['bullish'] + data['bearish'] + data['neutral']
                    }
                    for date, data in dates.items()
                }

            return jsonify({
                "status": "success",
                "data": response_data,
                "date_range": date_range
            })

    except Exception as e:
        print(f"Error in get_top_coins: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

class ModelType(str, Enum):
    lstm = "lstm"
    nn = "nn" 
    gru = "gru"
    arima = "arima"

class CoinType(str, Enum):
    btc = "btc"
    eth = "eth"
    xrp = "xrp"
    bnb = "bnb"
    sol = "sol"

# Mapping for cryptocurrency symbols
COIN_SYMBOLS = {
    CoinType.btc: "BTC-USD",
    CoinType.eth: "ETH-USD", 
    CoinType.xrp: "XRP-USD",
    CoinType.bnb: "BNB-USD",
    CoinType.sol: "SOL-USD"
}

class PricePrediction:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.model_paths = {
            ModelType.lstm: {
                'model': os.path.join(base_dir, 'LSTM_price_prediction.keras'),
                'scaler': os.path.join(base_dir, 'LSTM_price_scaler.pkl')
            },
            ModelType.nn: {
                'model': os.path.join(base_dir, 'NN_price_prediction.keras'),
                'scaler': os.path.join(base_dir, 'NN_price_scaler.pkl')
            },
            ModelType.gru: {
                'model': os.path.join(base_dir, 'gru_model.h5'),
                'scaler': None
            },
            ModelType.arima: {
                'model': os.path.join(base_dir, 'ARIMA_price_prediction.pkl'),
                'scaler': None
            }
        }
        
    def load_crypto_data(self, coin: str, start_date: dt.datetime, end_date: dt.datetime) -> pd.DataFrame:
        """Load cryptocurrency data with proper error handling"""
        try:
            symbol = COIN_SYMBOLS.get(CoinType(coin))
            if not symbol:
                raise ValueError(f"Invalid coin type: {coin}")
            
            data = yf.download(symbol, start=start_date, end=end_date)
            if data.empty:
                raise ValueError(f"No data available for {symbol}")
            
            return data
        except Exception as e:
            logger.error(f"Error loading data for {coin}: {str(e)}")
            raise

    def prepare_sequences(self, data: np.ndarray, sequence_length: int) -> np.ndarray:
        """Prepare sequences for deep learning models"""
        sequences = []
        for i in range(sequence_length, len(data)):
            sequences.append(data[i-sequence_length:i])
        return np.array(sequences)

    def predict_deep_learning(
        self, 
        model_type: ModelType, 
        data: pd.DataFrame, 
        prediction_days: int = 60, 
        future_days: int = 30
    ) -> Dict[str, Any]:
        """Enhanced deep learning prediction with proper error handling"""
        try:
            # Load model and scaler
            model = load_model(self.model_paths[model_type]['model'])
            scaler = MinMaxScaler(feature_range=(0, 1))
            
            # Scale the data
            scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
            
            # Prepare sequences
            x_test = self.prepare_sequences(scaled_data, prediction_days)
            x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
            
            # Make predictions
            predictions = model.predict(x_test)
            predictions = scaler.inverse_transform(predictions)
            
            # Generate future predictions
            future_predictions = []
            last_sequence = scaled_data[-prediction_days:]
            
            for _ in range(future_days):
                next_pred = model.predict(
                    last_sequence.reshape(1, prediction_days, 1), 
                    verbose=0
                )
                next_pred = scaler.inverse_transform(next_pred)[0][0]
                future_predictions.append(float(next_pred))
                last_sequence = np.roll(last_sequence, -1)
                last_sequence[-1] = scaler.transform([[next_pred]])[0][0]
            
            return {
                'historical_predictions': predictions.flatten(),
                'future_predictions': future_predictions
            }
            
        except Exception as e:
            logger.error(f"Error in deep learning prediction: {str(e)}")
            raise

    def predict_arima(
        self, 
        data: pd.DataFrame, 
        prediction_days: int = 60, 
        future_days: int = 30
    ) -> Dict[str, Any]:
        """ARIMA prediction with specific test period starting from 2020"""
        try:
            # Load ARIMA model
            with open(self.model_paths[ModelType.arima]['model'], 'rb') as file:
                arima_model = pickle.load(file)

            # Prepare test data starting from 2020
            test_start = dt.datetime(2020, 1, 1)
            test_end = dt.datetime.now()
            symbol = data.index.name + "-USD" if data.index.name else "BTC-USD"  # Default to BTC if name not set
            test_data = yf.download(symbol, start=test_start, end=test_end)

            # Fit model with complete data
            model = ARIMA(data['Close'], order=arima_model.order)
            fitted_model = model.fit()

            # Generate historical predictions for test period
            historical_predictions = fitted_model.predict(
                start=len(data) - len(test_data),
                end=len(data) - 1
            )

            # Generate future predictions
            future_predictions = fitted_model.forecast(steps=future_days).tolist()

            return {
                'historical_predictions': historical_predictions,
                'future_predictions': future_predictions,
                'test_dates': test_data.index,
                'test_prices': test_data['Close'].values
            }

        except Exception as e:
            logger.error(f"Error in ARIMA prediction: {str(e)}")
            raise

    def format_predictions(
        self,
        dates: pd.DatetimeIndex,
        actual_prices: np.ndarray,
        predictions: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Format predictions into a consistent response structure"""
        result = []
        
        # Format historical predictions
        for date, actual, pred in zip(
            dates[-len(predictions['historical_predictions']):],
            actual_prices[-len(predictions['historical_predictions']):],
            predictions['historical_predictions']
        ):
            result.append({
                "date": date.strftime('%Y-%m-%d'),
                "actual_price": float(actual),
                "predicted_price": float(pred)
            })
        
        # Add future predictions
        future_dates = pd.date_range(
            start=dates[-1] + pd.Timedelta(days=1),
            periods=len(predictions['future_predictions'])
        )
        
        for date, pred in zip(future_dates, predictions['future_predictions']):
            result.append({
                "date": date.strftime('%Y-%m-%d'),
                "actual_price": None,
                "predicted_price": float(pred)
            })
        
        return result

def predict_endpoint(model_type: str, coin: str) -> Dict[str, Any]:
    """Enhanced prediction endpoint with proper validation and error handling"""
    try:
        # Validate input parameters
        if model_type not in [m.value for m in ModelType]:
            return jsonify({"error": f"Invalid model type: {model_type}"}), 400
        if coin not in [c.value for c in CoinType]:
            return jsonify({"error": f"Invalid coin type: {coin}"}), 400
        
        logger.info(f"Starting prediction for {coin} using {model_type} model")
        
        # Initialize prediction class
        predictor = PricePrediction(os.path.dirname(os.path.abspath(__file__)))
        
        # Load data
        start_date = dt.datetime(2016, 1, 1)
        end_date = dt.datetime.now()
        data = predictor.load_crypto_data(coin, start_date, end_date)
        
        # Make predictions based on model type
        if model_type == ModelType.arima.value:
            predictions = predictor.predict_arima(data)
        else:
            predictions = predictor.predict_deep_learning(ModelType(model_type), data)
        
        # Format results
        result = predictor.format_predictions(
            data.index,
            data['Close'].values,
            predictions
        )
        
        logger.info(f"Successfully generated predictions for {coin}")
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# Flask route implementation
@app.route("/predict/<model_type>/<coin>")
def predict(model_type: str, coin: str):
    return predict_endpoint(model_type, coin)

@app.route('/youtube/unique-coins')
def get_unique_coins_with_stats():
    """Get all unique coins with mention statistics and sentiment breakdown"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            
            # Get unique coins with mention counts and sentiment breakdown
            c.execute('''
                WITH sentiment_counts AS (
                    SELECT 
                        coin_mentioned,
                        COUNT(*) as total_mentions,
                        SUM(CASE WHEN lower(indicator) LIKE '%bullish%' THEN 1 ELSE 0 END) as bullish_count,
                        SUM(CASE WHEN lower(indicator) LIKE '%bearish%' THEN 1 ELSE 0 END) as bearish_count,
                        SUM(CASE 
                            WHEN lower(indicator) NOT LIKE '%bullish%' 
                            AND lower(indicator) NOT LIKE '%bearish%' 
                            THEN 1 ELSE 0 END) as neutral_count,
                        COUNT(DISTINCT video_id) as video_count,
                        MAX(created_at) as last_mentioned
                    FROM coin_analysis
                    GROUP BY coin_mentioned
                )
                SELECT 
                    coin_mentioned,
                    total_mentions,
                    bullish_count,
                    bearish_count,
                    neutral_count,
                    video_count,
                    last_mentioned,
                    ROUND(CAST(bullish_count AS FLOAT) / total_mentions * 100, 2) as bullish_percentage,
                    ROUND(CAST(bearish_count AS FLOAT) / total_mentions * 100, 2) as bearish_percentage,
                    ROUND(CAST(neutral_count AS FLOAT) / total_mentions * 100, 2) as neutral_percentage
                FROM sentiment_counts
                ORDER BY total_mentions DESC
            ''')
            
            coins = [dict(row) for row in c.fetchall()]
            
            # Calculate overall statistics
            total_unique_coins = len(coins)
            total_mentions = sum(coin['total_mentions'] for coin in coins)
            total_videos = len(set(
                row[0] for row in c.execute('SELECT video_id FROM coin_analysis').fetchall()
            ))
            
            # Get top mentioned coins (top 10)
            top_mentioned = coins[:10] if len(coins) > 10 else coins
            
            # Get most bullish coins (top 5 with at least 5 mentions)
            most_bullish = sorted(
                [coin for coin in coins if coin['total_mentions'] >= 5],
                key=lambda x: x['bullish_percentage'],
                reverse=True
            )[:5]
            
            # Get most bearish coins (top 5 with at least 5 mentions)
            most_bearish = sorted(
                [coin for coin in coins if coin['total_mentions'] >= 5],
                key=lambda x: x['bearish_percentage'],
                reverse=True
            )[:5]
            
            return jsonify({
                "status": "success",
                "data": {
                    "statistics": {
                        "total_unique_coins": total_unique_coins,
                        "total_mentions": total_mentions,
                        "total_videos": total_videos,
                        "average_mentions_per_video": round(total_mentions / total_videos, 2) if total_videos > 0 else 0
                    },
                    "insights": {
                        "top_mentioned": top_mentioned,
                        "most_bullish": most_bullish,
                        "most_bearish": most_bearish
                    },
                    "all_coins": coins
                }
            })
            
    except Exception as e:
        print(f"Error in get_unique_coins_with_stats: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route('/youtube/coins/names')
def get_coin_names():
    """Get just the list of unique coin names with edit history applied"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            
            # First get all coin name overrides
            c.execute('''
                SELECT 
                    current_name,
                    new_name
                FROM coin_edits
                ORDER BY edited_at DESC
            ''')
            name_mapping = {row[0]: row[1] for row in c.fetchall()}
            
            # Get unique coin names
            c.execute('''
                SELECT DISTINCT coin_mentioned
                FROM coin_analysis 
                ORDER BY coin_mentioned ASC
            ''')
            
            # Apply name overrides
            coins = []
            for row in c.fetchall():
                coin_name = row[0]
                # Use the override if it exists, otherwise use original name
                final_name = name_mapping.get(coin_name, coin_name)
                coins.append(final_name)
            
            # Remove duplicates that might have been created by overrides
            unique_coins = sorted(set(coins))
            
            return jsonify({
                "status": "success",
                "data": {
                    "coins": unique_coins,
                    "total": len(unique_coins),
                    "metadata": {
                        "overrides_applied": len(name_mapping)
                    }
                }
            })
            
    except Exception as e:
        print(f"Error in get_coin_names: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/youtube/reason/today')
def get_todays_reasons():
    """Get all reasons from today's analyses with their associated coins and sources"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            
            # Get current date in Malaysia timezone
            malaysia_tz = pytz.timezone('Asia/Kuala_Lumpur')
            current_date = datetime.now(malaysia_tz).strftime("%Y-%m-%d")
            
            # Modified query to handle the datetime format in the database
            c.execute('''
                SELECT 
                    v.channel_name,
                    v.title as video_title,
                    v.url as video_url,
                    ca.coin_mentioned,
                    ca.reasons,
                    ca.indicator,
                    v.published_at,
                    v.processed_at
                FROM videos v
                JOIN coin_analysis ca ON v.video_id = ca.video_id
                WHERE DATE(v.processed_at) = DATE('now')
                ORDER BY v.processed_at DESC, ca.coin_mentioned ASC
            ''')
            
            analyses = [dict(row) for row in c.fetchall()]
            
            if not analyses:
                return jsonify({
                    "status": "success",
                    "data": {
                        "date": current_date,
                        "total_analyses": 0,
                        "reasons": [],
                        "statistics": {
                            "total_coins": 0,
                            "total_channels": 0,
                            "total_reasons": 0
                        }
                    }
                })
            
            # Process and structure the reasons
            structured_reasons = []
            for analysis in analyses:
                # Parse the JSON reasons
                reasons_list = json.loads(analysis['reasons'])
                
                # Add each reason with its context
                for reason in reasons_list:
                    structured_reasons.append({
                        "coin": analysis['coin_mentioned'],
                        "reason": reason,
                        "sentiment": analysis['indicator'],
                        "source": {
                            "channel": analysis['channel_name'],
                            "video_title": analysis['video_title'],
                            "video_url": analysis['video_url'],
                            "published_at": analysis['published_at'],
                            "processed_at": analysis['processed_at']
                        }
                    })
            
            # Calculate statistics
            unique_coins = len(set(analysis['coin_mentioned'] for analysis in analyses))
            unique_channels = len(set(analysis['channel_name'] for analysis in analyses))
            
            return jsonify({
                "status": "success",
                "data": {
                    "date": current_date,
                    "total_analyses": len(analyses),
                    "reasons": structured_reasons,
                    "statistics": {
                        "total_coins": unique_coins,
                        "total_channels": unique_channels,
                        "total_reasons": len(structured_reasons),
                        "average_reasons_per_coin": round(len(structured_reasons) / unique_coins, 2) if unique_coins > 0 else 0
                    }
                }
            })
            
    except Exception as e:
        print(f"Error in get_todays_reasons: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/youtube/coin/mentions', methods=['POST'])
def get_coin_mentions():
    """Get all YouTube channels and their videos that mentioned a specific coin"""
    try:
        # Validate request data
        data = request.get_json()
        if not data or 'coin_name' not in data:
            return jsonify({
                "status": "error",
                "message": "Missing required field: coin_name"
            }), 400

        coin_name = data['coin_name']

        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            
            # Get all mentions of the coin with channel and video details
            c.execute('''
                SELECT DISTINCT
                    v.channel_name,
                    v.channel_id,
                    v.video_id,
                    v.title,
                    v.url,
                    v.views,
                    v.published_at,
                    ca.reasons,
                    ca.indicator
                FROM videos v
                JOIN coin_analysis ca ON v.video_id = ca.video_id
                WHERE LOWER(ca.coin_mentioned) LIKE LOWER(?)
                ORDER BY v.published_at DESC
            ''', (f"%{coin_name}%",))
            
            results = [dict(row) for row in c.fetchall()]
            
            if not results:
                return jsonify({
                    "status": "success",
                    "message": f"No mentions found for {coin_name}",
                    "data": []
                })

            # Group by channel
            channels = {}
            for row in results:
                channel_name = row['channel_name']
                if channel_name not in channels:
                    channels[channel_name] = {
                        "channel_name": channel_name,
                        "channel_id": row['channel_id'],
                        "mentions": []
                    }
                
                channels[channel_name]["mentions"].append({
                    "video_title": row['title'],
                    "video_url": row['url'],
                    "views": row['views'],
                    "published_at": row['published_at'],
                    "sentiment": row['indicator'],
                    "reasons": json.loads(row['reasons'])
                })

            return jsonify({
                "status": "success",
                "data": list(channels.values())
            })

    except Exception as e:
        print(f"Error in get_coin_mentions: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/youtube/today-analysis/<date>')
def get_todays_channel_analysis(date):
    try:
        # Validate date format
        try:
            target_date = datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            return jsonify({
                "status": "error",
                "message": "Invalid date format. Please use YYYY-MM-DD"
            }), 400

        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            
            # First get all coin name overrides
            c.execute('''
                SELECT 
                    current_name,
                    new_name
                FROM coin_edits
                ORDER BY edited_at DESC
            ''')
            name_mapping = {row['current_name']: row['new_name'] for row in c.fetchall()}
            
            # Get Malaysia timezone (UTC+8)
            malaysia_tz = pytz.timezone('Asia/Kuala_Lumpur')
            
            # Set the target date range in Malaysia time
            start_my = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_my = start_my.replace(hour=23, minute=59, second=59)
            
            # Convert to UTC for database query
            start_utc = malaysia_tz.localize(start_my).astimezone(pytz.UTC).strftime("%Y-%m-%d %H:%M:%S")
            end_utc = malaysia_tz.localize(end_my).astimezone(pytz.UTC).strftime("%Y-%m-%d %H:%M:%S")
            
            # Query with UTC timestamp range
            c.execute('''
                WITH video_analyses AS (
                    SELECT 
                        v.channel_name,
                        v.channel_id,
                        v.video_id,
                        v.title as video_title,
                        v.url as video_url,
                        v.thumbnail_url,
                        v.views,
                        v.published_at,
                        v.processed_at,
                        ca.coin_mentioned,
                        ca.reasons,
                        ca.indicator
                    FROM videos v
                    LEFT JOIN coin_analysis ca ON v.video_id = ca.video_id
                    WHERE v.published_at BETWEEN ? AND ?
                    AND ca.coin_mentioned IS NOT NULL
                )
                SELECT * FROM video_analyses
                ORDER BY published_at DESC
            ''', (start_utc, end_utc))
            
            rows = c.fetchall()
            
            if not rows:
                return jsonify({
                    "status": "success",
                    "data": {
                        "date": date,
                        "reasons": [],
                        "statistics": {
                            "total_videos": 0,
                            "total_coins": 0,
                            "total_reasons": 0
                        }
                    }
                })
            
            # Format the response with name overrides
            reasons = []
            coins_mentioned = set()
            
            for row in rows:
                try:
                    parsed_reasons = json.loads(row['reasons'])
                    
                    # Apply name override if exists
                    coin_name = row['coin_mentioned']
                    if coin_name in name_mapping:
                        coin_name = name_mapping[coin_name]
                    
                    coins_mentioned.add(coin_name)
                    
                    for reason in parsed_reasons:
                        reasons.append({
                            "coin": coin_name,  # Using overridden name
                            "reason": reason,
                            "sentiment": row['indicator'],
                            "source": {
                                "channel": row['channel_name'],
                                "video_title": row['video_title'],
                                "video_url": row['video_url'],
                                "published_at": row['published_at'],
                                "processed_at": row['processed_at']
                            }
                        })
                except Exception as e:
                    print(f"Error parsing reasons for video {row['video_id']}: {str(e)}")
                    continue
            
            # Calculate statistics with overridden names
            statistics = {
                "total_videos": len(set(row['video_id'] for row in rows)),
                "total_coins": len(coins_mentioned),
                "total_reasons": len(reasons),
                "coins_breakdown": {
                    coin: len([r for r in reasons if r['coin'] == coin])
                    for coin in coins_mentioned
                },
                "name_overrides_applied": len(name_mapping)  # Added for transparency
            }
            
            return jsonify({
                "status": "success",
                "data": {
                    "date": date,
                    "reasons": reasons,
                    "statistics": statistics,
                    "metadata": {
                        "timezone": "Asia/Kuala_Lumpur",
                        "period": {
                            "start": start_utc,
                            "end": end_utc
                        }
                    }
                }
            })
            
    except Exception as e:
        print(f"Error in get_todays_channel_analysis: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/youtube/sector-analysis/<date>')
def analyze_sectors_by_date(date):
    """Analyze cryptocurrency sectors mentioned on a specific date"""
    try:
        print(f"\nStarting sector analysis for date: {date}")
        
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            
            # Get all coins and their mentions for the specified date
            query = '''
                SELECT 
                    ca.coin_mentioned,
                    ca.reasons,
                    ca.indicator,
                    v.channel_name,
                    v.title,
                    v.url
                FROM videos v
                JOIN coin_analysis ca ON v.video_id = ca.video_id
                WHERE DATE(v.published_at) = ?
                ORDER BY v.published_at ASC
            '''
            print(f"\nExecuting query for date {date}...")
            c.execute(query, (date,))
            
            mentions = [dict(row) for row in c.fetchall()]
            print(f"\nFound {len(mentions)} mentions for date {date}")
            
            if not mentions:
                print(f"No data found for date: {date}")
                return jsonify({
                    "status": "error",
                    "message": f"No data found for date: {date}"
                }), 404

            # Configure Gemini
            print("\nConfiguring Gemini model...")
            genai.configure(api_key=os.getenv('GEMINI_API_KEY_1'))
            model = genai.GenerativeModel(model_name="gemini-1.5-pro")
            
            # Prepare data for analysis
            print("\nPreparing analysis data...")
            analysis_data = {
                "date": date,
                "mentions": []
            }
            
            # Process each mention with error handling
            for m in mentions:
                try:
                    mention_data = {
                        "coin": m["coin_mentioned"],
                        "reasons": json.loads(m["reasons"]),
                        "indicator": m["indicator"],
                        "source": {
                            "channel": m["channel_name"],
                            "title": m["title"],
                            "url": m["url"]
                        }
                    }
                    analysis_data["mentions"].append(mention_data)
                except json.JSONDecodeError as e:
                    print(f"Error parsing reasons for {m['coin_mentioned']}: {str(e)}")
                    print(f"Raw reasons data: {m['reasons']}")
                except Exception as e:
                    print(f"Error processing mention for {m['coin_mentioned']}: {str(e)}")

            print(f"\nProcessed {len(analysis_data['mentions'])} valid mentions")
            
            # Create the prompt
            prompt = """Analyze these cryptocurrency mentions and categorize them into SPECIFIC sectors.
CRITICAL RULES:
1. NEVER use generic categories like "Other", "Miscellaneous", or "Unclassified" - ALWAYS assign a specific sector
2. If a coin seems unclear, analyze its use case and assign it to the most relevant specific sector
3. ALWAYS provide detailed reasoning for EACH sector explaining why those coins belong there
4. Be consistent with sector naming
5. If a coin could fit multiple sectors, choose the most dominant use case

Categorize into specific sectors such as:
- AI & Machine Learning (Example coins: Fetch.ai, Ocean Protocol, SingularityNET)
- DeFi (Example coins: Aave, Uniswap, Curve)
- Gaming & Metaverse (Example coins: Axie Infinity, The Sandbox, Decentraland)
- Layer 1 Blockchains (Example coins: Ethereum, Solana, Cardano)
- Layer 2 Scaling (Example coins: Arbitrum, Optimism, Polygon)
- Privacy Focused (Example coins: Monero, Zcash, Secret)
- NFTs & Digital Collectibles (Example coins: Bored Ape, CryptoPunks)
- Web3 Infrastructure (Example coins: Chainlink, Graph Protocol, Filecoin)
- Exchange Platforms (Example coins: BNB, FTT, CRO)
- Cross-Chain Solutions (Example coins: Polkadot, Cosmos, THORChain)
- Decentralized Storage (Example coins: Filecoin, Arweave, Storj)
- Social Finance (Example coins: Friend.tech, Stars Arena)
- Green Blockchain (Example coins: IOTA, Energy Web Token)
- IoT & Supply Chain (Example coins: VeChain, IOTA, WaltonChain)
- Decentralized Identity (Example coins: Litentry, ONT ID)
- Governance Tokens (Example coins: UNI, AAVE, COMP)

Return a JSON response with this EXACT structure:
{
    "sectors": [
        {
            "name": "specific sector name",
            "coins": ["coin1", "coin2"],
            "sentiment": {
                "bullish": number,
                "bearish": number
            },
            "reasoning": "REQUIRED: Detailed explanation of why these specific coins belong in this sector. Example: 'These coins are categorized as Layer 1 because they operate their own independent blockchain networks with unique consensus mechanisms. TRX (Tron) and SUI both have their own blockchain networks, while Oasis Network provides its own layer 1 blockchain focused on privacy and scalability.'"
        }
    ]
}

IMPORTANT: Each sector MUST include detailed reasoning explaining why those specific coins belong in that category."""

            print("\nGenerating analysis with Gemini...")
            response = model.generate_content(prompt)
            print("\nRaw Gemini response:")
            print(response.text)
            
            # Clean and parse the response
            clean_response = response.text.strip()
            if clean_response.startswith('```json'):
                clean_response = clean_response[7:]
            if clean_response.startswith('```'):
                clean_response = clean_response[3:]
            if clean_response.endswith('```'):
                clean_response = clean_response[:-3]
            
            print("\nCleaned response:")
            print(clean_response)
            
            analysis = json.loads(clean_response.strip())
            print("\nSuccessfully parsed JSON response")
            
            # Add metadata
            analysis["metadata"] = {
                "total_videos": len(set(m["url"] for m in mentions)),
                "total_channels": len(set(m["channel_name"] for m in mentions)),
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            print("\nAnalysis complete!")
            return jsonify({
                "status": "success",
                "data": analysis
            })
            
    except Exception as e:
        print(f"\nError in sector analysis: {str(e)}")
        print("\nFull error details:")
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/youtube/weekly-sectors')
def analyze_weekly_sectors():
    try:
        print("\nStarting 7-day sector analysis...")
        
        # Calculate date range (last 3 days instead of 7)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        print(f"\nAnalyzing period: {start_date.date()} to {end_date.date()}")
        
        # Get all Gemini API keys from environment
        gemini_keys = [
            os.getenv('GEMINI_API_KEY_1'),
            os.getenv('GEMINI_API_KEY_2'),
            os.getenv('GEMINI_API_KEY_3'),
            os.getenv('GEMINI_API_KEY_4')
        ]
        current_key_index = 0
        
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            
            query = '''
                SELECT 
                    ca.coin_mentioned,
                    ca.reasons,
                    ca.indicator,
                    v.channel_name,
                    v.title,
                    v.url,
                    v.published_at
                FROM videos v
                JOIN coin_analysis ca ON v.video_id = ca.video_id
                WHERE v.published_at >= ? AND v.published_at <= ?
                ORDER BY v.published_at DESC
            '''
            c.execute(query, (start_date.isoformat(), end_date.isoformat()))
            mentions = [dict(row) for row in c.fetchall()]
            
            print(f"\nFound {len(mentions)} total mentions in the past 3 days")
            
            if not mentions:
                return jsonify({
                    "status": "error",
                    "message": "No data found for the past 3 days"
                }), 404

            def get_next_key():
                nonlocal current_key_index
                key = gemini_keys[current_key_index]
                current_key_index = (current_key_index + 1) % len(gemini_keys)
                return key

            print("\nConfiguring Gemini model...")
            genai.configure(api_key=get_next_key())
            model = genai.GenerativeModel(model_name="gemini-1.5-pro")
            
            CHUNK_SIZE = 50
            chunks = [mentions[i:i + CHUNK_SIZE] for i in range(0, len(mentions), CHUNK_SIZE)]
            
            print(f"\nProcessing {len(chunks)} chunks of data...")
            
            all_sectors = {}
            
            for chunk_index, chunk in enumerate(chunks):
                while True:
                    try:
                        print(f"\n{'='*50}")
                        print(f"PROCESSING CHUNK {chunk_index + 1} OF {len(chunks)}")
                        print(f"{'='*50}")
                        
                        chunk_data = {
                            "mentions": []
                        }
                        
                        for m in chunk:
                            try:
                                chunk_data["mentions"].append({
                                    "coin": m["coin_mentioned"],
                                    "reasons": json.loads(m["reasons"]),
                                    "indicator": m["indicator"],
                                    "source": {
                                        "channel": m["channel_name"],
                                        "title": m["title"],
                                        "url": m["url"]
                                    }
                                })
                            except Exception as e:
                                print(f"Error processing mention: {str(e)}")
                                continue
                        
                        print(f"\nProcessed {len(chunk_data['mentions'])} valid mentions in chunk")
                        print("Sample coins in this chunk:")
                        for mention in chunk_data['mentions'][:5]:
                            print(f"- {mention['coin']}")
                        
                        prompt = """Analyze these cryptocurrency mentions and categorize them into SPECIFIC sectors.
CRITICAL RULES:
1. NEVER use generic categories like "Other", "Miscellaneous", or "Unclassified" - ALWAYS assign a specific sector
2. If a coin seems unclear, analyze its use case and assign it to the most relevant specific sector
3. ALWAYS provide detailed reasoning for EACH sector explaining why those coins belong there
4. Be consistent with sector naming
5. If a coin could fit multiple sectors, choose the most dominant use case

Categorize into specific sectors such as:
- AI & Machine Learning (Example coins: Fetch.ai, Ocean Protocol, SingularityNET)
- DeFi (Example coins: Aave, Uniswap, Curve)
- Gaming & Metaverse (Example coins: Axie Infinity, The Sandbox, Decentraland)
- Layer 1 Blockchains (Example coins: Ethereum, Solana, Cardano)
- Layer 2 Scaling (Example coins: Arbitrum, Optimism, Polygon)
- Privacy Focused (Example coins: Monero, Zcash, Secret)
- NFTs & Digital Collectibles (Example coins: Bored Ape, CryptoPunks)
- Web3 Infrastructure (Example coins: Chainlink, Graph Protocol, Filecoin)
- Exchange Platforms (Example coins: BNB, FTT, CRO)
- Cross-Chain Solutions (Example coins: Polkadot, Cosmos, THORChain)
- Decentralized Storage (Example coins: Filecoin, Arweave, Storj)
- Social Finance (Example coins: Friend.tech, Stars Arena)
- Green Blockchain (Example coins: IOTA, Energy Web Token)
- IoT & Supply Chain (Example coins: VeChain, IOTA, WaltonChain)
- Decentralized Identity (Example coins: Litentry, ONT ID)
- Governance Tokens (Example coins: UNI, AAVE, COMP)

Return a JSON response with this EXACT structure:
{
    "sectors": [
        {
            "name": "specific sector name",
            "coins": ["coin1", "coin2"],
            "sentiment": {
                "bullish": number,
                "bearish": number
            },
            "reasoning": "REQUIRED: Detailed explanation of why these specific coins belong in this sector"
        }
    ]
}"""
                        
                        print("\nSending to Gemini for analysis...")
                        response = model.generate_content(prompt + "\n\nData to analyze:\n" + json.dumps(chunk_data, indent=2))
                        
                        print("\nRAW GEMINI RESPONSE:")
                        print(response.text)
                        
                        try:
                            clean_response = response.text.strip()
                            if clean_response.startswith('```json'):
                                clean_response = clean_response[7:]
                            if clean_response.startswith('```'):
                                clean_response = clean_response[3:]
                            if clean_response.endswith('```'):
                                clean_response = clean_response[:-3]
                            
                            print("\nCLEANED RESPONSE:")
                            print(clean_response)
                            
                            parsed_response = json.loads(clean_response)
                            
                            for sector in parsed_response['sectors']:
                                sector_name = sector['name']
                                if sector_name not in all_sectors:
                                    all_sectors[sector_name] = {
                                        "mention_count": 0,
                                        "sentiment": {"bullish": 0, "bearish": 0},
                                        "coins": set()
                                    }
                                
                                all_sectors[sector_name]["mention_count"] += (
                                    sector['sentiment']['bullish'] + 
                                    sector['sentiment']['bearish']
                                )
                                all_sectors[sector_name]["sentiment"]["bullish"] += sector['sentiment']['bullish']
                                all_sectors[sector_name]["sentiment"]["bearish"] += sector['sentiment']['bearish']
                                all_sectors[sector_name]["coins"].update(sector['coins'])
                            
                            print("\nSuccessfully aggregated chunk data!")
                            
                        except Exception as parse_error:
                            print(f"\nERROR PARSING RESPONSE: {str(parse_error)}")
                            print("Full response that failed to parse:")
                            print(clean_response)
                            continue
                        
                        print("\nSuccessfully processed chunk!")
                        break
                        
                    except Exception as e:
                        if "429" in str(e) or "Resource has been exhausted" in str(e):
                            print(f"API key exhausted, switching to next key...")
                            genai.configure(api_key=get_next_key())
                            model = genai.GenerativeModel(model_name="gemini-1.5-pro")
                            continue
                        else:
                            print(f"\nUnexpected error processing chunk {chunk_index + 1}: {str(e)}")
                            traceback.print_exc()
                            break
            
            if not all_sectors:
                return jsonify({
                    "status": "error",
                    "message": "Failed to process any chunks successfully"
                }), 500
            
            total_mentions = sum(s["mention_count"] for s in all_sectors.values())
            
            final_sectors = []
            for sector_name, sector_data in all_sectors.items():
                try:
                    final_sectors.append({
                        "name": sector_name,
                        "mention_count": sector_data["mention_count"],
                        "percentage": round((sector_data["mention_count"] / total_mentions * 100), 2),
                        "sentiment": {
                            "bullish": round((sector_data["sentiment"]["bullish"] / sector_data["mention_count"] * 100), 2),
                            "bearish": round((sector_data["sentiment"]["bearish"] / sector_data["mention_count"] * 100), 2)
                        },
                        "top_coins": list(sector_data["coins"])[:5]
                    })
                except Exception as e:
                    print(f"Error processing sector {sector_name}: {str(e)}")
                    continue
            
            final_sectors.sort(key=lambda x: x["mention_count"], reverse=True)
            
            analysis = {
                "period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                },
                "total_mentions": total_mentions,
                "sectors": final_sectors,
                "metadata": {
                    "total_videos": len(set(m["url"] for m in mentions)),
                    "total_channels": len(set(m["channel_name"] for m in mentions)),
                    "analysis_timestamp": datetime.now().isoformat()
                }
            }

            # Store in memory cache before database insertion
            app.last_sector_analysis = analysis

            # Then try database insertion
            try:
                with sqlite3.connect(DB_PATH) as conn:
                    c = conn.cursor()
                    
                    # Check if table exists
                    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sector_analysis'")
                    table_exists = c.fetchone() is not None
                    
                    if table_exists:
                        # Get current columns
                        c.execute("PRAGMA table_info(sector_analysis)")
                        columns = [column[1] for column in c.fetchall()]
                        
                        # If analysis_date column doesn't exist, add it
                        if 'analysis_date' not in columns:
                            c.execute('ALTER TABLE sector_analysis ADD COLUMN analysis_date TEXT')
                        
                        # Add any other missing columns
                        required_columns = {
                            'start_date': 'TEXT',
                            'end_date': 'TEXT',
                            'total_mentions': 'INTEGER',
                            'total_videos': 'INTEGER',
                            'total_channels': 'INTEGER',
                            'sectors_data': 'TEXT',
                            'created_at': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'
                        }
                        
                        for col_name, col_type in required_columns.items():
                            if col_name not in columns:
                                c.execute(f'ALTER TABLE sector_analysis ADD COLUMN {col_name} {col_type}')
                    else:
                        # Create new table with all required columns
                        c.execute('''
                            CREATE TABLE sector_analysis (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                analysis_date TEXT,
                                start_date TEXT,
                                end_date TEXT,
                                total_mentions INTEGER,
                                total_videos INTEGER,
                                total_channels INTEGER,
                                sectors_data TEXT,
                                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                            )
                        ''')
                    
                    # Insert the analysis results
                    c.execute('''
                        INSERT INTO sector_analysis (
                            analysis_date,
                            start_date,
                            end_date,
                            total_mentions,
                            total_videos,
                            total_channels,
                            sectors_data
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        datetime.now().isoformat(),
                        analysis['period']['start'],
                        analysis['period']['end'],
                        analysis['total_mentions'],
                        analysis['metadata']['total_videos'],
                        analysis['metadata']['total_channels'],
                        json.dumps(analysis['sectors'])
                    ))
                    conn.commit()
                    print("\nAnalysis results stored in database successfully!")
                    
            except Exception as db_error:
                print(f"\nError handling database: {str(db_error)}")
                traceback.print_exc()
            
            return jsonify({
                "status": "success",
                "data": analysis
            })
            
    except Exception as e:
        print(f"\nError in weekly sector analysis: {str(e)}")
        print("\nFull error details:")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

def weekly_analysis(prompt):  # Add prompt parameter
    api_keys = [
        "AIzaSyD2uaGDT5swFpWBLEKZhCnwAozJ8KrFtV4",
        "AIzaSyBVaovh2Cz9LU7gUJ_ft00UBEv26_vaaC0",
        "AIzaSyDU3yl8ZGdXUFN6rN2uKnl8dKuwwDQUdXg"
    ]
    current_key_index = 0
    max_retries = len(api_keys)
    retry_count = 0

    while retry_count < max_retries:
        try:
            # Configure Gemini with current API key
            genai.configure(api_key=api_keys[current_key_index])
            model = genai.GenerativeModel(model_name="gemini-1.5-pro")
            
            # Your existing weekly analysis code here
            response = model.generate_content(prompt)
            return jsonify({"response": response.text})

        except Exception as e:
            error_message = str(e).lower()
            if "resource exhausted" in error_message:
                print(f"API key {current_key_index + 1} exhausted, switching to next key...")
                current_key_index = (current_key_index + 1) % len(api_keys)
                retry_count += 1
            else:
                # If it's a different error, raise it
                raise e

    # If all API keys are exhausted, wait 60 seconds
    print("All API keys exhausted. Waiting 60 seconds...")
    time.sleep(60)
    
    # Try again with the first API key after waiting
    genai.configure(api_key=api_keys[0])
    model = genai.GenerativeModel(model_name="gemini-1.5-pro")
    response = model.generate_content(prompt)
    return jsonify({"response": response.text})

def merge_similar_sectors(sectors: List[Dict]) -> List[Dict]:
    """Merge sectors with the same normalized names"""
    merged = {}
    
    for sector in sectors:
        name = sector['name']
        if name not in merged:
            merged[name] = {
                'name': name,
                'mention_count': 0,
                'sentiment': {'bullish': 0, 'bearish': 0},
                'top_coins': set(),
                'reasoning': sector.get('reasoning', '')  # Add reasoning field
            }
        
        # Merge data
        merged[name]['mention_count'] += sector['mention_count']
        merged[name]['sentiment']['bullish'] += sector['sentiment']['bullish'] * sector['mention_count'] / 100
        merged[name]['sentiment']['bearish'] += sector['sentiment']['bearish'] * sector['mention_count'] / 100
        merged[name]['top_coins'].update(sector['top_coins'])
        
        # Combine reasoning if different
        if sector.get('reasoning') and sector['reasoning'] != merged[name]['reasoning']:
            merged[name]['reasoning'] = f"{merged[name]['reasoning']} | {sector['reasoning']}".strip(' |')
    
    # Calculate final percentages and format response
    result = []
    total_mentions = sum(s['mention_count'] for s in merged.values())
    
    for sector in merged.values():
        total_sector_mentions = sector['mention_count']
        result.append({
            'name': sector['name'],
            'mention_count': total_sector_mentions,
            'percentage': round((total_sector_mentions / total_mentions * 100), 2),
            'sentiment': {
                'bullish': round((sector['sentiment']['bullish'] / total_sector_mentions * 100), 2),
                'bearish': round((sector['sentiment']['bearish'] / total_sector_mentions * 100), 2)
            },
            'top_coins': list(sector['top_coins'])[:5],  # Keep top 5 coins
            'reasoning': sector['reasoning']  # Include reasoning in final output
        })
    
    return sorted(result, key=lambda x: x['mention_count'], reverse=True)

@app.route('/youtube/get-light-temp')
@cache_response(CACHE_DURATION)
def get_light_results_with_override():
    """Temporary endpoint to get results with coin name overrides from edit history"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            
            # Get ALL edit history (not filtered by date)
            c.execute('''
                SELECT 
                    current_name,
                    new_name
                FROM coin_edits
                ORDER BY edited_at DESC
            ''')
            
            # Create a mapping of old names to new names
            name_mapping = {row['current_name']: row['new_name'] for row in c.fetchall()}
            
            # Get videos only from the specified date range
            c.execute('''
                SELECT 
                    id, video_id, channel_id, channel_name, 
                    title, url, thumbnail_url, views, 
                    duration, published_at, processed_at
                FROM videos 
                WHERE DATE(published_at) BETWEEN '2025-01-31' AND '2025-02-03'
                ORDER BY processed_at DESC
            ''')
            videos = [dict(row) for row in c.fetchall()]
            
            # Get analyses and apply name overrides
            for video in videos:
                c.execute('''
                    SELECT 
                        coin_mentioned, reasons, indicator
                    FROM coin_analysis 
                    WHERE video_id = ?
                ''', (video['video_id'],))
                
                analyses = []
                for row in c.fetchall():
                    # Apply name override if exists
                    coin_name = row['coin_mentioned']
                    if coin_name in name_mapping:
                        coin_name = name_mapping[coin_name]
                        
                    analyses.append({
                        'coin_mentioned': coin_name,
                        'reason': json.loads(row['reasons']),
                        'indicator': row['indicator']
                    })
                video['analyses'] = analyses
            
            return jsonify({
                "status": "success",
                "data": {
                    "total_videos": len(videos),
                    "videos": videos,
                    "cached": False,
                    "_debug_name_mappings": name_mapping
                }
            })
            
    except Exception as e:
        print(f"Error in get_light_results_with_override: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/youtube/get-sector-analysis', methods=['GET'])
def get_sector_analysis():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            
            # First try to get from database
            try:
                c.execute('''
                    SELECT 
                        analysis_date,
                        period_start,  # Changed from start_date
                        period_end,    # Changed from end_date
                        total_mentions,
                        total_videos,
                        total_channels,
                        sectors_data,
                        created_at
                    FROM sector_analysis
                    ORDER BY created_at DESC
                    LIMIT 1
                ''')
                
                result = c.fetchone()
                
                if result:
                    analysis = {
                        "analysis_date": result['analysis_date'],
                        "period": {
                            "start": result['period_start'],
                            "end": result['period_end']
                        },
                        "total_mentions": result['total_mentions'],
                        "metadata": {
                            "total_videos": result['total_videos'],
                            "total_channels": result['total_channels'],
                            "created_at": result['created_at']
                        },
                        "sectors": json.loads(result['sectors_data'])
                    }
                    
                    return jsonify({
                        "status": "success",
                        "source": "database",
                        "data": analysis
                    })
                    
            except sqlite3.OperationalError as e:
                print(f"Database error: {str(e)}")
                
            # If database retrieval fails, try to get from memory cache
            # (You'll need to add this variable at the top of your file)
            if hasattr(app, 'last_sector_analysis'):
                return jsonify({
                    "status": "success",
                    "source": "memory",
                    "data": app.last_sector_analysis
                })
            
            return jsonify({
                "status": "error",
                "message": "No sector analysis data found in database or memory"
            }), 404
            
    except Exception as e:
        print(f"Error retrieving sector analysis: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/youtube/drop-sector-analysis', methods=['POST'])
def drop_sector_analysis():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute('DROP TABLE IF EXISTS sector_analysis')
            conn.commit()
            
            return jsonify({
                "status": "success",
                "message": "sector_analysis table dropped successfully"
            })
            
    except Exception as e:
        print(f"Error dropping table: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/youtube/today-coins/<date>')
def get_coins_by_date(date):
    """Get coin mention counts for a specific date (Malaysia timezone) with edit history applied"""
    try:
        # Validate date format
        try:
            target_date = datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            return jsonify({
                "status": "error",
                "message": "Invalid date format. Please use YYYY-MM-DD"
            }), 400

        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            
            # First get all coin name overrides
            c.execute('''
                SELECT 
                    current_name,
                    new_name
                FROM coin_edits
                ORDER BY edited_at DESC
            ''')
            name_mapping = {row['current_name']: row['new_name'] for row in c.fetchall()}
            
            # Get Malaysia timezone (UTC+8)
            malaysia_tz = pytz.timezone('Asia/Kuala_Lumpur')
            
            # Set the target date range in Malaysia time
            start_my = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_my = start_my.replace(hour=23, minute=59, second=59)
            
            # Convert to UTC for database query (subtract 8 hours since DB is in UTC)
            start_utc = malaysia_tz.localize(start_my).astimezone(pytz.UTC)
            end_utc = malaysia_tz.localize(end_my).astimezone(pytz.UTC)
            
            # Query with explicit timezone conversion in SQL
            c.execute('''
                WITH video_times AS (
                    SELECT 
                        v.video_id,
                        v.channel_name,
                        v.published_at,
                        datetime(v.published_at, '+8 hours') as my_time,
                        ca.coin_mentioned,
                        ca.indicator
                    FROM videos v
                    JOIN coin_analysis ca ON v.video_id = ca.video_id
                )
                SELECT *
                FROM video_times
                WHERE date(my_time) = ?
            ''', (date,))
            
            mentions = [dict(row) for row in c.fetchall()]
            
            if not mentions:
                return jsonify({
                    "status": "success",
                    "data": {
                        "date": date,
                        "total_unique_coins": 0,
                        "coins": [],
                        "metadata": {
                            "timezone": "Asia/Kuala_Lumpur",
                            "period": {
                                "start": start_utc.isoformat(),
                                "end": end_utc.isoformat()
                            }
                        }
                    }
                })

            # Process mentions with name overrides
            coin_stats = {}
            for mention in mentions:
                # Apply name override if exists
                coin_name = mention['coin_mentioned']
                if coin_name in name_mapping:
                    coin_name = name_mapping[coin_name]
                
                if coin_name not in coin_stats:
                    coin_stats[coin_name] = {
                        "coin_mentioned": coin_name,
                        "total_mentions": 0,
                        "unique_mentions": 0,
                        "video_ids": set(),
                        "channels": set(),
                        "channel_mentions": {},
                        "sentiment_counts": {
                            "bullish": 0,
                            "bearish": 0,
                            "neutral": 0
                        }
                    }
                
                stats = coin_stats[coin_name]
                stats["total_mentions"] += 1
                stats["video_ids"].add(mention["video_id"])
                
                # Track channel-specific mentions
                channel = mention["channel_name"]
                if channel not in stats["channel_mentions"]:
                    stats["channel_mentions"][channel] = 0
                    stats["unique_mentions"] += 1
                stats["channel_mentions"][channel] += 1
                stats["channels"].add(channel)
                
                # Track sentiment counts
                indicator = mention["indicator"].lower()
                if "bullish" in indicator:
                    stats["sentiment_counts"]["bullish"] += 1
                elif "bearish" in indicator:
                    stats["sentiment_counts"]["bearish"] += 1
                else:
                    stats["sentiment_counts"]["neutral"] += 1

            # Format the final response
            formatted_coins = []
            for coin_name, stats in coin_stats.items():
                formatted_coins.append({
                    "coin_mentioned": coin_name,
                    "total_mentions": stats["total_mentions"],
                    "unique_channel_mentions": stats["unique_mentions"],
                    "video_count": len(stats["video_ids"]),
                    "channel_count": len(stats["channels"]),
                    "sentiment": stats["sentiment_counts"],  # Now using raw counts instead of percentages
                    "channel_breakdown": {
                        channel: count 
                        for channel, count in stats["channel_mentions"].items()
                    }
                })
            
            # Sort by unique channel mentions first, then total mentions
            formatted_coins.sort(key=lambda x: (x["unique_channel_mentions"], x["total_mentions"]), reverse=True)

            return jsonify({
                "status": "success",
                "data": {
                    "date": date,
                    "total_unique_coins": len(formatted_coins),
                    "coins": formatted_coins,
                    "metadata": {
                        "timezone": "Asia/Kuala_Lumpur",
                        "period": {
                            "start": start_utc.isoformat(),
                            "end": end_utc.isoformat()
                        },
                        "name_overrides_applied": len(name_mapping)
                    }
                }
            })
            
    except Exception as e:
        print(f"Error in get_coins_by_date: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/youtube/coin-trends')
def get_coin_trends():
    """Get all historical coin mentions with sentiment counts (Malaysia timezone)"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            
            # Get coin name overrides
            c.execute('''
                SELECT 
                    current_name,
                    new_name
                FROM coin_edits
                ORDER BY edited_at DESC
            ''')
            name_mapping = {row['current_name']: row['new_name'] for row in c.fetchall()}
            
            # Query with explicit timezone conversion in SQL
            c.execute('''
                WITH video_times AS (
                    SELECT 
                        v.video_id,
                        v.channel_name,
                        v.published_at,
                        datetime(v.published_at, '+8 hours') as my_time,
                        ca.coin_mentioned,
                        ca.indicator
                    FROM videos v
                    JOIN coin_analysis ca ON v.video_id = ca.video_id
                )
                SELECT 
                    date(my_time) as my_date,
                    coin_mentioned,
                    indicator
                FROM video_times
                ORDER BY my_date DESC, coin_mentioned
            ''')
            
            results = {}
            
            for row in c.fetchall():
                date = row['my_date']
                
                # Apply name override if exists
                coin_name = row['coin_mentioned']
                if coin_name in name_mapping:
                    coin_name = name_mapping[coin_name]
                
                if date not in results:
                    results[date] = {}
                
                if coin_name not in results[date]:
                    results[date][coin_name] = {
                        "bullish": 0,
                        "bearish": 0
                    }
                
                # Count sentiments
                indicator = row['indicator'].lower()
                if "bullish" in indicator:
                    results[date][coin_name]["bullish"] += 1
                elif "bearish" in indicator:
                    results[date][coin_name]["bearish"] += 1
            
            # Format the response
            formatted_results = []
            for date in results:
                for coin in results[date]:
                    formatted_results.append({
                        "date": date,
                        "coin": coin,
                        "bullish": results[date][coin]["bullish"],
                        "bearish": results[date][coin]["bearish"]
                    })
            
            # Sort by date (newest first) and then by coin name
            formatted_results.sort(key=lambda x: (x["date"], x["coin"]), reverse=True)
            
            return jsonify({
                "status": "success",
                "data": formatted_results
            })
            
    except Exception as e:
        print(f"Error in get_coin_trends: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/youtube/coin-reasons')
def get_all_coin_reasons():
    """Get all reasons for all coins in the database"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            
            # Get coin name overrides
            c.execute('''
                SELECT 
                    current_name,
                    new_name
                FROM coin_edits
                ORDER BY edited_at DESC
            ''')
            name_mapping = {row['current_name']: row['new_name'] for row in c.fetchall()}
            
            # Query all coins and their reasons
            c.execute('''
                SELECT 
                    ca.coin_mentioned,
                    ca.reasons
                FROM coin_analysis ca
                JOIN videos v ON v.video_id = ca.video_id
                WHERE ca.reasons IS NOT NULL
                  AND datetime(v.published_at, '+8 hours') >= datetime('now', '-30 days')
            ''')
            
            results = {}
            
            for row in c.fetchall():
                try:
                    # Apply name override if exists
                    coin_name = row['coin_mentioned']
                    if coin_name in name_mapping:
                        coin_name = name_mapping[coin_name]
                    
                    if coin_name not in results:
                        results[coin_name] = []
                    
                    # Parse and add reasons
                    reasons = json.loads(row['reasons'])
                    if isinstance(reasons, list):
                        results[coin_name].extend(reasons)
                        
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    print(f"Error processing row for {coin_name}: {str(e)}")
                    continue
            
            # Convert to list format
            formatted_results = [
                {
                    "coin": coin,
                    "reasons": list(set(reasons))  # Remove duplicates
                }
                for coin, reasons in results.items()
            ]
            
            # Sort by coin name
            formatted_results.sort(key=lambda x: x["coin"])
            
            return jsonify({
                "status": "success",
                "data": formatted_results
            })
            
    except Exception as e:
        print(f"Error in get_all_coin_reasons: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/youtube/reason/<date>')
def get_reasons_by_date(date):
    """Get coin reasons for a specific date (Malaysia timezone)"""
    try:
        # Validate date format
        try:
            target_date = datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            return jsonify({
                "status": "error",
                "message": "Invalid date format. Please use YYYY-MM-DD"
            }), 400

        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            
            # Get coin name overrides
            c.execute('''
                SELECT 
                    current_name,
                    new_name
                FROM coin_edits
                ORDER BY edited_at DESC
            ''')
            name_mapping = {row['current_name']: row['new_name'] for row in c.fetchall()}
            
            # Query with Malaysia timezone adjustment
            c.execute('''
                WITH video_times AS (
                    SELECT 
                        ca.coin_mentioned,
                        ca.reasons,
                        datetime(v.published_at, '+8 hours') as my_time
                    FROM videos v
                    JOIN coin_analysis ca ON v.video_id = ca.video_id
                    WHERE date(datetime(v.published_at, '+8 hours')) = ?
                )
                SELECT coin_mentioned, reasons
                FROM video_times
                ORDER BY coin_mentioned
            ''', (date,))
            
            mentions = [dict(row) for row in c.fetchall()]
            
            if not mentions:
                return jsonify({
                    "status": "success",
                    "data": []
                })

            # Process mentions with name overrides and combine reasons
            coin_reasons = {}
            for mention in mentions:
                # Apply name override if exists
                coin_name = mention['coin_mentioned']
                if coin_name in name_mapping:
                    coin_name = name_mapping[coin_name]
                
                if coin_name not in coin_reasons:
                    coin_reasons[coin_name] = set()
                
                # Add reasons to set to avoid duplicates
                try:
                    reasons = json.loads(mention['reasons'])
                    coin_reasons[coin_name].update(reasons)
                except json.JSONDecodeError:
                    continue

            # Format the response
            formatted_results = [
                {
                    "coin": coin,
                    "reasons": list(reasons)
                }
                for coin, reasons in coin_reasons.items()
            ]
            
            # Sort by coin name
            formatted_results.sort(key=lambda x: x["coin"])

            return jsonify({
                "status": "success",
                "data": formatted_results
            })
            
    except Exception as e:
        print(f"Error in get_reasons_by_date: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/youtube/reason/summary')
def summarize_daily_reasons():
    """Analyze and summarize coin reasons for Feb 13-17"""
    try:
        dates = [
            "2025-02-20"
        ]
        
        # Configure Gemini
        genai.configure(api_key=os.getenv('GEMINI_API_KEY_4'))
        model = genai.GenerativeModel(model_name="gemini-1.5-pro")
        
        all_summaries = []
        
        for date in dates:
            print(f"Processing {date}...")
            
            with app.test_client() as client:
                response = client.get(f'/youtube/today-analysis/{date}')
                if response.status_code != 200:
                    print(f"No data found for {date}")
                    continue
                    
                raw_data = response.get_json()
                if not raw_data.get('data', {}).get('reasons'):
                    print(f"No reasons found for {date}")
                    continue
                
                coin_data = {}
                for reason in raw_data['data']['reasons']:
                    coin_name = reason['coin']
                    if coin_name not in coin_data:
                        coin_data[coin_name] = []
                    coin_data[coin_name].append(reason['reason'])

                with sqlite3.connect(DB_PATH) as conn:
                    c = conn.cursor()
                    
                    for coin_name, reasons in coin_data.items():
                        print(f"Processing {coin_name} for {date}...")
                        
                        prompt = f"""Analyze these reasons for {coin_name} and extract key points with sentiment.
                        
                        CRITICAL RULES:
                        1. Format output EXACTLY as shown in the example below
                        2. Each key point must have a "point" and "sentiment" field
                        3. Sentiment must be one of: "BULLISH", "BEARISH", or "NEUTRAL"
                        4. Combine similar points and remove duplicates
                        5. Keep specific numbers and statistics
                        6. Maintain accuracy - don't create new information
                        
                        Example output format:
                        {{
                            "key_points": [
                                {{
                                    "point": "Bitcoin ETFs saw $1.2B inflows in first week of trading",
                                    "sentiment": "BULLISH"
                                }},
                                {{
                                    "point": "Price dropped 5% on high volume selling",
                                    "sentiment": "BEARISH"
                                }}
                            ]
                        }}

                        Analyze this data:
                        {json.dumps(reasons, indent=2)}
                        """

                        try:
                            response = model.generate_content(prompt)
                            clean_response = response.text.strip()
                            
                            # Print raw response for debugging
                            print(f"Raw Gemini response for {coin_name}:")
                            print(clean_response)
                            
                            if clean_response.startswith('```json'):
                                clean_response = clean_response[7:]
                            if clean_response.startswith('```'):
                                clean_response = clean_response[3:]
                            if clean_response.endswith('```'):
                                clean_response = clean_response[:-3]

                            analysis = json.loads(clean_response.strip())
                            
                            # Store in database
                            c.execute('''
                                INSERT OR REPLACE INTO reason_summaries 
                                (analysis_date, coin_name, summary, created_at)
                                VALUES (?, ?, ?, datetime('now'))
                            ''', (date, coin_name, clean_response.strip()))
                            
                            all_summaries.append({
                                "coin_name": coin_name,
                                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "key_points": analysis.get('key_points', [])
                            })
                            
                            print(f"Successfully processed {coin_name} for {date}")
                            
                        except Exception as e:
                            print(f"Error processing {coin_name} for {date}: {str(e)}")
                            print(f"Clean response that failed: {clean_response}")
                            continue
                    
                    conn.commit()
                    print(f"Completed storing summaries for {date}")

        return jsonify({
            "status": "success",
            "data": {
                "date": dates[0],  # Using first date
                "summaries": all_summaries
            }
        })

    except Exception as e:
        print(f"Error in summarize_daily_reasons: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/youtube/reason-summaries/all/<coin>')
def get_all_reason_summaries(coin=None):
    """Get all entries from the reason_summaries table, optionally filtered by coin"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            
            # First, let's get the table schema
            c.execute("PRAGMA table_info(reason_summaries)")
            columns = [column[1] for column in c.fetchall()]
            
            # Prepare query based on whether coin parameter is provided
            if coin:
                # Case-insensitive search with partial matching
                query = '''
                    SELECT *
                    FROM reason_summaries
                    WHERE LOWER(coin_name) LIKE LOWER(?)
                    ORDER BY analysis_date DESC, coin_name
                '''
                search_param = f'%{coin}%'
                c.execute(query, (search_param,))
            else:
                # Get all entries if no coin specified
                c.execute('''
                    SELECT *
                    FROM reason_summaries
                    ORDER BY analysis_date DESC, coin_name
                ''')
            
            rows = c.fetchall()
            
            if not rows:
                return jsonify({
                    "status": "success",
                    "message": f"No entries found{f' for coin: {coin}' if coin else ''}",
                    "schema": columns,
                    "data": []
                })
            
            # Format the results
            summaries = []
            for row in rows:
                try:
                    summary_data = {
                        "analysis_date": row['analysis_date'],
                        "coin_name": row['coin_name'],
                        "summary": json.loads(row['summary']),
                        "created_at": row['created_at'] if 'created_at' in columns else None
                    }
                    summaries.append(summary_data)
                except Exception as e:
                    print(f"Error processing row: {str(e)}")
                    continue
            
            return jsonify({
                "status": "success",
                "schema": columns,
                "total_entries": len(summaries),
                "filter": {"coin": coin} if coin else None,
                "data": summaries
            })
            
    except Exception as e:
        print(f"Error retrieving reason summaries: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# Add an endpoint to retrieve stored summaries
@app.route('/youtube/reason/summary/stored/<date>')
def get_stored_summary(date):
    """Retrieve stored summary for a specific date"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            
            c.execute('''
                SELECT analysis_date, coin_name, summary, created_at
                FROM reason_summaries
                WHERE analysis_date = ?
                ORDER BY coin_name
            ''', (date,))
            
            rows = c.fetchall()
            
            if rows:
                summaries = []
                for row in rows:
                    summaries.append({
                        "coin_name": row['coin_name'],
                        "key_points": json.loads(row['summary'])['key_points'],
                        "created_at": row['created_at']
                    })
                
                return jsonify({
                    "status": "success",
                    "data": {
                        "date": date,
                        "summaries": summaries
                    }
                })
            else:
                return jsonify({
                    "status": "error",
                    "message": f"No summary found for date: {date}"
                }), 404
                
    except Exception as e:
        print(f"Error retrieving stored summary: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/test/summary/<date>')
def test_summary(date):
    """Test endpoint for summarizing crypto data with Gemini"""
    try:
        print(f"\nStarting test summary for date: {date}")
        
        # Configure Gemini
        api_key = os.getenv('GEMINI_API_KEY_1')
        if not api_key:
            return jsonify({
                "status": "error",
                "message": "GEMINI_API_KEY_1 not found in environment variables"
            }), 500
            
        print("\nConfiguring Gemini model...")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name="gemini-1.5-pro")
        
        # Get raw data using existing endpoint
        raw_data = get_reasons_by_date(date).get_json()
        
        if raw_data['status'] != 'success' or not raw_data['data']:
            return jsonify({
                "status": "error",
                "message": f"No data found for date: {date}"
            }), 404

        # Get coin name overrides
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute('''
                SELECT 
                    current_name,
                    new_name
                FROM coin_edits
                ORDER BY edited_at DESC
            ''')
            name_mapping = {row['current_name']: row['new_name'] for row in c.fetchall()}
            
        # Apply name overrides to raw data
        for item in raw_data['data']:
            if item['coin'] in name_mapping:
                item['coin'] = name_mapping[item['coin']]

        print(f"\nFound {len(raw_data['data'])} coin mentions for {date}")
        
        prompt = """Analyze these cryptocurrency mentions and create professional news-style summaries.

CRITICAL RULES:
1. Write in a professional news/journalism style (like Bloomberg, Reuters, or CoinDesk)
2. Be direct, concise, and factual - no speculation or vague statements
3. Prioritize concrete information: numbers, statistics, significant events
4. Use strong, clear language appropriate for financial news
5. Merge related points into cohesive statements
6. Maintain chronological order for events when relevant
7. Keep only the most impactful and newsworthy points
8. Do NOT fabricate or infer information
9. Use EXACTLY the coin names provided - do not modify them

Format the response in this EXACT JSON structure:
{
    "summaries": [
        {
            "coin_name": "coin name",
            "key_points": [
                "Professional news-style point 1",
                "Professional news-style point 2"
            ]
        }
    ]
}

Analyze this data:
"""

        print("\nSending to Gemini for analysis...")
        response = model.generate_content(prompt + json.dumps(raw_data['data'], indent=2))
        
        # Print raw response for debugging
        print("\nRAW GEMINI RESPONSE:")
        print(response.text)
        
        # Clean response
        clean_response = response.text.strip()
        if clean_response.startswith('```json'):
            clean_response = clean_response[7:]
        if clean_response.startswith('```'):
            clean_response = clean_response[3:]
        if clean_response.endswith('```'):
            clean_response = clean_response[:-3]
        
        print("\nCLEANED RESPONSE:")
        print(clean_response)
        
        try:
            analysis = json.loads(clean_response.strip())
        except json.JSONDecodeError as json_err:
            print(f"\nJSON PARSE ERROR: {str(json_err)}")
            print("Raw response that failed to parse:")
            print(clean_response)
            raise
        
        return jsonify({
            "status": "success",
            "date": date,
            "data": analysis
        })
        
    except Exception as e:
        print(f"\nError in test_summary: {str(e)}")
        print("\nFull error details:")
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/youtube/summary/prompt3/<date>')
def summary_prompt3(date):
    """Process stored summaries coin by coin using enhanced news-style prompt"""
    try:
        print(f"\nStarting Prompt 3 summary for date: {date}")
        
        # Get stored summaries directly from database
        try:
            with sqlite3.connect(DB_PATH) as conn:
                conn.row_factory = sqlite3.Row
                c = conn.cursor()
                
                c.execute('''
                    SELECT analysis_date, coin_name, summary, created_at
                    FROM reason_summaries
                    WHERE analysis_date = ?
                    ORDER BY coin_name
                ''', (date,))
                
                rows = c.fetchall()

                print(f"Found {len(rows)} stored summaries for date: {date}")
                
                if not rows:
                    return jsonify({
                        "status": "error",
                        "message": f"No stored summaries found for date: {date}"
                    }), 404
                
                stored_data = {
                    "data": {
                        "date": date,
                        "summaries": [{
                            "coin_name": row['coin_name'],
                            "key_points": json.loads(row['summary'])['key_points'],
                            "created_at": row['created_at']
                        } for row in rows]
                    }
                }
                
        except Exception as db_error:
            print(f"Database error: {str(db_error)}")
            return jsonify({
                "status": "error",
                "message": f"Database error: {str(db_error)}"
            }), 500

        # Configure Gemini
        api_key = os.getenv('GEMINI_API_KEY_1')
        if not api_key:
            return jsonify({
                "status": "error",
                "message": "GEMINI_API_KEY_1 not found in environment variables"
            }), 500
            
        print("\nConfiguring Gemini model...")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name="gemini-1.5-pro")
        
        processed_coins = []
        
        # Process each coin's summary
        for coin_summary in stored_data['data']['summaries']:
            coin_name = coin_summary['coin_name']
            print(f"\nProcessing summary for {coin_name}")
            
            prompt = f"""Analyze this {coin_name} market data and create a professional Bloomberg/Reuters-style market analysis.

CRITICAL RULES:
1. Write in authoritative financial journalism style
2. Focus on market-moving events, price action, and significant developments
3. Include specific numbers, percentages, and market statistics when available
4. Explain market impact and implications clearly
5. Maintain professional, confident tone throughout
6. Structure insights from most to least impactful
7. Be direct and factual - no speculation
8. Use strong, precise financial language

Example high-quality response:
{{
    "coin_name": "Bitcoin",
    "analysis": [
        {{
            "insight": "Bitcoin surges 8.5% to $48,250 as ETF inflows hit record $1.1B weekly volume, signaling strong institutional demand",
            "impact": "BULLISH"
        }},
        {{
            "insight": "Network activity jumps 40% MoM with 325,000 daily active addresses, indicating growing adoption",
            "impact": "BULLISH"
        }}
    ]
}}

Original data to analyze:
{json.dumps(coin_summary['key_points'], indent=2)}

Return analysis in EXACT format shown above."""

            try:
                print("Sending to Gemini...")
                response = model.generate_content(prompt)
                
                # Clean response
                clean_response = response.text.strip()
                if clean_response.startswith('```json'):
                    clean_response = clean_response[7:]
                if clean_response.startswith('```'):
                    clean_response = clean_response[3:]
                if clean_response.endswith('```'):
                    clean_response = clean_response[:-3]
                
                analysis = json.loads(clean_response.strip())
                processed_coins.append(analysis)
                
                # Store in database
                try:
                    with sqlite3.connect(DB_PATH) as conn:
                        c = conn.cursor()
                        # First check if table exists
                        c.execute('''CREATE TABLE IF NOT EXISTS prompt3_summaries
                                   (id INTEGER PRIMARY KEY AUTOINCREMENT,
                                    analysis_date TEXT,
                                    coin_name TEXT,
                                    analysis_data TEXT,
                                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
                        
                        # Delete any existing analysis for this coin/date
                        c.execute('''DELETE FROM prompt3_summaries 
                                   WHERE analysis_date = ? AND coin_name = ?''', 
                                (date, coin_name))
                        
                        # Insert new analysis
                        c.execute('''INSERT INTO prompt3_summaries 
                                   (analysis_date, coin_name, analysis_data)
                                   VALUES (?, ?, ?)''',
                                (date, coin_name, json.dumps(analysis)))
                        conn.commit()
                        print(f"Stored analysis in database for {coin_name}")
                except Exception as db_error:
                    print(f"Database error for {coin_name}: {str(db_error)}")
                
                print(f"Successfully processed {coin_name}")
                
            except Exception as e:
                print(f"Error processing {coin_name}: {str(e)}")
                continue

        return jsonify({
            "status": "success",
            "data": {
                "date": date,
                "coins": processed_coins,
                "total_processed": len(processed_coins)
            }
        })
        
    except Exception as e:
        print(f"\nError in summary_prompt3: {str(e)}")
        print("\nFull error details:")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# Add a new endpoint to process the entire date range
@app.route('/youtube/summary/prompt3/process-range')
def process_date_range():
    """Process summaries for date range Feb 11, 2025 to Feb 13, 2025 (Malaysia timezone)"""
    try:
        # Set up Malaysia timezone
        malaysia_tz = pytz.timezone('Asia/Kuala_Lumpur')
        
        # Set date range in Malaysia timezone (Feb 11-13, 2025)
        start_date = datetime(2025, 2, 13, tzinfo=malaysia_tz)
        end_date = datetime(2025, 2, 14, tzinfo=malaysia_tz)
        current_date = start_date
        
        print(f"\nAnalyzing period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} (Malaysia timezone)")
        
        processed_dates = []
        
        # Add one day to end_date to ensure we process the full last day
        end_date = end_date + timedelta(days=1)
        
        while current_date < end_date:  # Changed <= to < since we added an extra day
            date_str = current_date.strftime("%Y-%m-%d")
            print(f"\n{'='*50}")
            print(f"Processing date: {date_str}")
            print(f"{'='*50}")
            
            try:
                # Call the summary_prompt3 function for each date
                response = summary_prompt3(date_str)
                
                # Handle the response correctly based on its type
                if isinstance(response, tuple):
                    response_data = response[0]
                    status_code = response[1]
                else:
                    # If it's a Response object, get the json data
                    response_data = response.json if hasattr(response, 'json') else response
                    status_code = response.status_code if hasattr(response, 'status_code') else 200
                
                if status_code == 200:
                    processed_dates.append({
                        "date": date_str,
                        "status": "success"
                    })
                else:
                    error_message = response_data.get('message') if isinstance(response_data, dict) else str(response_data)
                    processed_dates.append({
                        "date": date_str,
                        "status": "error",
                        "message": error_message
                    })
            except Exception as e:
                processed_dates.append({
                    "date": date_str,
                    "status": "error",
                    "message": str(e)
                })
            
            current_date += timedelta(days=1)
            
            # Add explicit check to ensure we process Feb 13
            if date_str == "2025-02-13":
                print("\nEnsuring Feb 13 is fully processed...")
        
        print("\nProcessing complete!")
        return jsonify({
            "status": "success",
            "data": {
                "date_range": {
                    "start": start_date.strftime("%Y-%m-%d"),
                    "end": (end_date - timedelta(days=1)).strftime("%Y-%m-%d"),  # Adjust end date in response
                    "timezone": "Asia/Kuala_Lumpur"
                },
                "processed_dates": processed_dates
            }
        })
        
    except Exception as e:
        print(f"\nError processing date range: {str(e)}")
        print("\nFull error details:")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/youtube/summary/prompt3/stored/<date>')
def get_prompt3_summaries(date):
    """Retrieve stored Prompt 3 summaries for a specific date"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            
            c.execute('''
                SELECT analysis_date, coin_name, analysis_data, created_at
                FROM prompt3_summaries
                WHERE analysis_date = ?
                ORDER BY coin_name
            ''', (date,))
            
            rows = c.fetchall()
            
            if not rows:
                return jsonify({
                    "status": "error",
                    "message": f"No Prompt 3 summaries found for date: {date}"
                }), 404
            
            summaries = [{
                "coin_name": row['coin_name'],
                "analysis": json.loads(row['analysis_data'])['analysis'],
                "created_at": row['created_at']
            } for row in rows]
            
            return jsonify({
                "status": "success",
                "data": {
                    "date": date,
                    "summaries": summaries
                }
            })
            
    except Exception as e:
        print(f"Error retrieving Prompt 3 summaries: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/youtube/summary/prompt4/<date>')
def summary_prompt4(date):
    """Generate comprehensive market analysis including historical patterns and future predictions"""
    try:
        print(f"\nStarting Prompt 4 comprehensive analysis for date: {date}")
        
        # Get today's summaries from database
        try:
            with sqlite3.connect(DB_PATH) as conn:
                conn.row_factory = sqlite3.Row
                c = conn.cursor()
                
                print("\nGetting today's data from database...")
                c.execute('''
                    SELECT analysis_date, coin_name, summary
                    FROM reason_summaries
                    WHERE analysis_date = ?
                    ORDER BY coin_name
                ''', (date,))
                
                today_rows = c.fetchall()
                
                if not today_rows:
                    print(f"No data found for date: {date}")
                    return jsonify({
                        "status": "error",
                        "message": f"No summaries found for date: {date}"
                    }), 404
                
                print(f"\nFound {len(today_rows)} coin summaries for today")
                
                # Get historical data (last 30 days)
                current_date = datetime.strptime(date, '%Y-%m-%d')
                past_date = current_date - timedelta(days=30)
                
                print(f"\nFetching historical data from {past_date.strftime('%Y-%m-%d')} to {date}")
                c.execute('''
                    SELECT analysis_date, coin_name, summary
                    FROM reason_summaries
                    WHERE analysis_date BETWEEN ? AND ?
                    ORDER BY analysis_date DESC, coin_name
                ''', (past_date.strftime('%Y-%m-%d'), date))
                
                historical_rows = c.fetchall()
                print(f"Found {len(historical_rows)} historical summaries")
                
                # Format data for analysis
                print("\nFormatting data for analysis...")
                today_data = {
                    "date": date,
                    "summaries": []
                }
                
                for row in today_rows:
                    try:
                        summary_json = json.loads(row['summary'])
                        today_data['summaries'].append({
                            "coin_name": row['coin_name'],
                            "key_points": summary_json.get('key_points', [])
                        })
                    except Exception as e:
                        print(f"Error processing summary for {row['coin_name']}: {str(e)}")
                        continue
                
                historical_data = []
                for row in historical_rows:
                    try:
                        summary_json = json.loads(row['summary'])
                        historical_data.append({
                            "date": row['analysis_date'],
                            "coin_name": row['coin_name'],
                            "key_points": summary_json.get('key_points', [])
                        })
                    except Exception as e:
                        print(f"Error processing historical data for {row['coin_name']}: {str(e)}")
                        continue

        except Exception as db_error:
            print(f"Database error: {str(db_error)}")
            return jsonify({
                "status": "error",
                "message": f"Database error: {str(db_error)}"
            }), 500

        # Get historical prices for major coins
        major_coins = ['btc', 'eth', 'sol', 'xrp']
        historical_price_data = {}
        for coin in major_coins:
            try:
                price_data = price_service.get_historical_prices(coin)
                historical_price_data[coin.upper()] = price_data
            except Exception as e:
                print(f"Error getting {coin} price data: {str(e)}")

        # Configure Gemini
        api_key = os.getenv('GEMINI_API_KEY_1')
        if not api_key:
            return jsonify({
                "status": "error",
                "message": "GEMINI_API_KEY_1 not found in environment variables"
            }), 500
            
        print("\nConfiguring Gemini model...")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name="gemini-1.5-pro")
        
        prompt = """Analyze the cryptocurrency market comprehensively, with a deep focus on historical patterns and relationships.

CRITICAL RULES:
1. Write in professional financial analysis style (Bloomberg/Reuters)
2. Use specific price data to support your analysis
3. Connect price movements with news events
4. Provide data-backed technical analysis
5. Include exact prices, percentages, and date-specific movements
6. Focus on actionable insights and clear market narratives
7. Maintain objectivity while highlighting key patterns
8. In each of the analysis, provide a confidence score and explanation for the score.

Required Categories:

4.0 CURRENT MARKET OVERVIEW
- Current market conditions with specific price levels
- Major market-moving events and their direct price impact
- Overall market sentiment backed by data
- Key support/resistance levels for major coins
- Notable volume and liquidity metrics
- Correlation patterns between major assets

4.1 HISTORICAL PATTERN ANALYSIS (Since December 1st)

Market-Wide Patterns:
- Comprehensive analysis of major market events since December 1st
- Identification of recurring price patterns and their triggers
- Cross-asset correlation analysis during significant events
- Impact patterns of different types of news/events
- Volume and liquidity trend analysis
- Historical support/resistance effectiveness
- Market sentiment cycles and their duration
- Institutional activity patterns
- Regulatory news impact assessment
- Geographic market influence patterns
- Macro event correlation analysis

Bitcoin (BTC) Historical Analysis:
- Detailed timeline of key price turning points since December
- News events that preceded major price moves (both positive and negative)
- Typical price reaction patterns to different news types
- Historical support/resistance levels and their effectiveness
- Pattern of institutional involvement and its price impact
- Volume profile during major moves
- Market sentiment cycles
- Correlation with traditional markets during key events

Ethereum (ETH) Historical Analysis:
- Historical correlation patterns with BTC
- Impact analysis of network upgrades and developments
- Gas fee patterns and their price relationship
- Developer activity correlation with price movements
- DeFi TVL correlation with price
- Historical response to market stress events
- Network usage metrics correlation
- Layer 2 development impact

Solana (SOL) Historical Analysis:
- Network performance impact on price movements
- Historical recovery patterns from network issues
- Correlation patterns with other L1 blockchains
- DeFi activity impact on price movements
- Developer activity correlation
- NFT market impact analysis
- Institutional investment patterns
- Technical pattern effectiveness

XRP Historical Analysis:
- Legal news impact patterns on price
- Historical volume analysis by region
- Price action patterns in different jurisdictions
- Correlation with traditional finance developments
- Regulatory news impact assessment
- Partnership announcement impact patterns
- Cross-border transaction volume correlation
- Historical support/resistance effectiveness

Cross-Market Historical Analysis:
- Crypto market correlation with traditional markets
- Inter-cryptocurrency correlation patterns
- Impact of global economic events
- Regional market influence analysis
- Regulatory impact patterns across regions
- Trading volume distribution patterns
- Liquidity flow patterns between assets
- Institutional money flow patterns 

4.2 LAYER 1 BLOCKCHAIN ANALYSIS

CRITICAL RULES:
1. Write in professional financial analysis style (Bloomberg/Reuters)
2. Use specific price data to support your analysis
3. Connect price movements with news events
4. Provide data-backed technical analysis
5. Include exact prices, percentages, and date-specific movements
6. Focus on actionable insights and clear market narratives
7. Maintain objectivity while highlighting key patterns
8. In each of the analysis, provide a confidence score and explanation for the score.

Layer 1 Market Overview:
- Current state of Layer 1 ecosystem
- Major market-moving events affecting L1s
- Comparative performance analysis
- TVL distribution across major L1s
- Developer activity metrics
- Cross-chain bridge volume analysis
- Network adoption metrics comparison
- Fee structure and revenue analysis

For Each Layer 1 Project:

Technical Infrastructure Analysis:
- Network performance metrics
- Scalability solutions progress
- Transaction processing capabilities
- Validator/node distribution
- Network security measures
- Fee structure effectiveness
- Network upgrade impacts
- Technical architecture advantages

Ecosystem Development:
- Smart contract platform adoption
- DeFi ecosystem growth metrics
- NFT and gaming platform status
- Developer activity trends
- Project launch success rate
- Cross-chain bridge integration
- Partnership developments
- Community growth metrics

Market Performance:
- Price action analysis
- Volume profile patterns
- Liquidity depth analysis
- Market maker participation
- Institutional involvement
- Trading pattern analysis
- Support/resistance levels
- Volatility comparison

Network Usage:
- Active address growth
- Transaction volume trends
- Smart contract interactions
- Fee revenue analysis
- Staking participation
- Governance activity
- Protocol revenue metrics
- User adoption patterns

Investment & Development:
- Venture capital activity
- Developer contribution metrics
- Ecosystem fund allocation
- Grant program progress
- Institutional investment flows
- Partnership announcement impacts
- Technology adoption metrics
- Innovation pipeline analysis

Cross-L1 Comparative Analysis:
- Performance metrics comparison
- Fee structure effectiveness
- Developer activity distribution
- TVL growth patterns
- Institutional investment flow
- Cross-chain interoperability
- Security incident analysis
- Market share evolution

"""

        # Convert PriceData objects to dictionaries
        formatted_price_data = {}
        for coin, price_info in historical_price_data.items():
            formatted_price_data[coin] = {
                "symbol": price_info["symbol"],
                "prices": [
                    {"date": price.date, "price": price.price}
                    for price in price_info["prices"]
                ]
            }

        # Add historical price data and market news to the prompt
        prompt += f"\n\nHistorical Price Data:\n{json.dumps(formatted_price_data, indent=2)}"
        prompt += f"\n\nMarket News and Sentiment:\n{json.dumps(today_data, indent=2)}"

        try:
            print("\nSending to Gemini for analysis...")
            response = model.generate_content(prompt)
            
            print("\nRAW GEMINI RESPONSE:")
            print(response.text)
            
            # Return the raw response without JSON parsing
            return jsonify({
                "status": "success",
                "gemini_response": response.text
            })
            
        except Exception as e:
            print(f"\nError generating analysis: {str(e)}")
            print("\nFull error details:")
            traceback.print_exc()
            return jsonify({
                "status": "error",
                "message": str(e)
            }), 500
            
    except Exception as e:
        print(f"\nError in summary_prompt4: {str(e)}")
        print("\nFull error details:")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/youtube/summary/prompt4/stored/<date>')
def get_prompt4_summary(date):
    """Retrieve stored Prompt 4 summary for a specific date"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            
            c.execute('''
                SELECT analysis_date, market_analysis, created_at
                FROM prompt4_summaries
                WHERE analysis_date = ?
            ''', (date,))
            
            row = c.fetchone()
            
            if not row:
                return jsonify({
                    "status": "error",
                    "message": f"No Prompt 4 summary found for date: {date}"
                }), 404
            
            return jsonify({
                "status": "success",
                "data": {
                    "date": row['analysis_date'],
                    "analysis": json.loads(row['market_analysis']),
                    "created_at": row['created_at']
                }
            })
            
    except Exception as e:
        print(f"Error retrieving Prompt 4 summary: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# Initialize the service (do this at the top level with other initializations)
price_service = HistoricalPriceService()

@app.route('/youtube/historical-price/<coin>')
def get_historical_price(coin):
    """Get historical price data for a specific coin starting from December 1, 2024"""
    try:
        result = price_service.get_historical_prices(coin)
        return jsonify({
            "status": "success",
            "data": result
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/youtube/summary/prompt4/ada/<date>')
def summary_prompt4_ada(date):
    """Generate comprehensive ADA market analysis using historical reasons and data and store in database"""
    try:
        print(f"\nStarting Prompt 4 ADA analysis for date: {date}")
        
        # Get historical ADA reasons from database
        try:
            with sqlite3.connect(DB_PATH) as conn:
                conn.row_factory = sqlite3.Row
                c = conn.cursor()
                
                # Get historical data (last 30 days)
                current_date = datetime.strptime(date, '%Y-%m-%d')
                past_date = current_date - timedelta(days=30)
                
                print(f"\nFetching ADA historical data from {past_date.strftime('%Y-%m-%d')} to {date}")
                c.execute('''
                    SELECT analysis_date, summary
                    FROM reason_summaries
                    WHERE analysis_date BETWEEN ? AND ?
                    AND LOWER(coin_name) IN ('ada', 'cardano')
                    ORDER BY analysis_date DESC
                ''', (past_date.strftime('%Y-%m-%d'), date))
                
                historical_rows = c.fetchall()
                print(f"Found {len(historical_rows)} historical ADA summaries")
                
                # Format historical data
                historical_data = []
                for row in historical_rows:
                    try:
                        summary_json = json.loads(row['summary'])
                        historical_data.append({
                            "date": row['analysis_date'],
                            "key_points": summary_json.get('key_points', [])
                        })
                    except Exception as e:
                        print(f"Error processing historical data: {str(e)}")
                        continue

        except Exception as db_error:
            print(f"Database error: {str(db_error)}")
            return jsonify({
                "status": "error",
                "message": f"Database error: {str(db_error)}"
            }), 500

        # Get ADA price data and convert to serializable format
        try:
            ada_price_data = price_service.get_historical_prices('ada')
            
            formatted_price_data = {
                "symbol": ada_price_data["symbol"],
                "prices": [
                    {
                        "date": price.date,
                        "price": price.price
                    } for price in ada_price_data["prices"]
                ]
            }
            
            current_price = ada_price_data['prices'][-1].price if ada_price_data['prices'] else 0
            previous_price = ada_price_data['prices'][-2].price if len(ada_price_data['prices']) > 1 else current_price
            price_change = ((current_price - previous_price) / previous_price) * 100
        except Exception as e:
            print(f"Error getting ADA price data: {str(e)}")
            current_price = 0
            price_change = 0
            formatted_price_data = {"symbol": "ADA", "prices": []}

        # Configure Gemini
        api_key = os.getenv('GEMINI_API_KEY_1')
        if not api_key:
            return jsonify({
                "status": "error",
                "message": "GEMINI_API_KEY_1 not found in environment variables"
            }), 500
            
        print("\nConfiguring Gemini model...")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name="gemini-2.0-flash-lite-preview-02-05")
        
        prompt = f"""Analyze Cardano (ADA) market conditions and provide a detailed report matching the exact format below.

ANALYSIS REQUIREMENTS:

1. MARKET DYNAMICS
- Network Performance:
  * Ouroboros metrics
  * Epoch statistics
  * Transaction throughput
  * Stake pool metrics
  * Network saturation

- Price Action Analysis:
  * Support/resistance levels
  * Volume profile
  * Market depth
  * Technical indicators
  * Trading patterns

- Network Fundamentals:
  * Total stake statistics
  * Plutus script activity
  * Transaction volumes
  * Active addresses
  * Network security

2. ECOSYSTEM LANDSCAPE
- Platform Metrics:
  * DeFi TVL trends
  * dApp deployment stats
  * Hydra scalability data
  * Cross-chain bridges
  * NFT marketplace activity

- Development Activity:
  * Vasil/Voltaire progress
  * Smart contract adoption
  * Partnership expansions
  * Infrastructure growth

3. MARKET SENTIMENT
- Technical Indicators:
  * Fear & Greed index
  * Long/short ratios
  * Funding rates
  * Options market sentiment

- On-Chain Metrics:
  * Whale wallet movements
  * Exchange flows
  * Stake delegation
  * Pool operator activity

4. FORWARD-LOOKING INDICATORS
- Upcoming Catalysts:
  * Hard fork events
  * Governance updates
  * Partnership announcements
  * Market structure changes

Current Market Data:
Price: ${current_price:.2f}
24h Change: {price_change:.1f}%

REQUIRED OUTPUT FORMAT:
{{
  "symbol": "ADA",
  "name": "Cardano",
  "price": "{current_price:.2f}",
  "priceChange": {price_change:.1f},
  "targetPrice": "[Target Price]",
  "sentiment": "[Bullish/Neutral/Bearish]",
  "sentimentScore": [1-10],
  "market": "[Single detailed paragraph combining key market insights]",
  "prediction": {{
    "direction": "[Clear trend statement]",
    "reasoning": [
      "1. [Ouroboros/Network insight]",
      "2. [Technical/Staking insight]",
      "3. [Development/Ecosystem insight]",
      "4. [Catalyst/Timeline insight]"
    ]
  }},
  "timeline": "[3-month window]",
  "confidence": [percentage],
  "status": "In Progress"
}}

CRITICAL REQUIREMENTS:
1. All analysis must be data-driven with specific numbers
2. Market analysis must be one detailed paragraph
3. Prediction points must be specific and quantifiable
4. Maintain exact UI format
5. Target price must have technical and fundamental basis
6. Timeline must consider known upcoming events
7. Confidence score must reflect data reliability
8. All monetary values in USD

Historical ADA Analysis Data:
{json.dumps(historical_data, indent=2)}

Price Data:
{json.dumps(formatted_price_data, indent=2)}
"""

        try:
            print("\nSending to Gemini for analysis...")
            response = model.generate_content(prompt)
            
            print("\nRAW GEMINI RESPONSE:")
            print(response.text)
            
            # Clean and parse response - IMPROVED PARSING
            clean_response = response.text.strip()
            
            # Find the JSON part
            json_start = clean_response.find('{')
            json_end = clean_response.find('\n\n', json_start)
            if json_end == -1:  # If no double newline, try to find the end of the JSON
                json_end = clean_response.rfind('}') + 1
            
            # Extract just the JSON part
            json_content = clean_response[json_start:json_end].strip()
            
            # Remove any markdown code block markers
            json_content = json_content.replace('```json', '').replace('```', '')
            
            print("\nExtracted JSON content:")
            print(json_content)
            
            analysis = json.loads(json_content)
            
            # Store the analysis in the database
            
            return jsonify({
                "status": "success",
                "data": analysis
            })
            
        except Exception as e:
            print(f"\nError generating analysis: {str(e)}")
            print("\nFull error details:")
            traceback.print_exc()
            return jsonify({
                "status": "error",
                "message": str(e)
            }), 500
            
    except Exception as e:
        print(f"\nError in summary_prompt4_ada: {str(e)}")
        print("\nFull error details:")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/market-analysis/insert', methods=['POST'])
def insert_market_analysis():
    """Insert initial market analysis data"""
    try:
        initial_data = [
            {
                "symbol": "BTC",
                "name": "Bitcoin",
                "price": "$96,168",
                "priceChange": 2.8,
                "targetPrice": "$120,000",
                "sentiment": "Bullish",
                "sentimentScore": 8,
                "market": "Bitcoin's market dynamics show significant institutional momentum following ETF approvals. Daily trading volumes across spot markets exceed $45B, with ETF volume maintaining $2B+ daily average. Key support levels established at $85,000-$89,000 range, with major resistance at $98,000-$102,000. Institutional holdings increased by 12% in the past month, while mining difficulty reached ATH at 81.55T. Options market shows strong call bias for June-July expiries, indicating bullish sentiment among derivatives traders. Network health metrics remain robust with hashrate stabilizing above 500 EH/s.",
                "prediction": {
                    "direction": "Upward trend likely",
                    "reasoning": [
                        "1. ETF inflows continue to exceed expectations with daily volume averaging $2B+",
                        "2. Mining difficulty at all-time high, indicating strong network security",
                        "3. Institutional adoption accelerating with major banks offering crypto custody",
                        "4. Supply shock expected from upcoming halving event in April 2024"
                    ]
                },
                "timeline": "Jul-Sep 2025",
                "confidence": 75,
                "status": "In Progress"
            },
            {
                "symbol": "ETH",
                "name": "Ethereum",
                "price": "$2,613",
                "priceChange": -1.5,
                "targetPrice": "$3,500",
                "sentiment": "Neutral",
                "sentimentScore": 5,
                "market": "Ethereum's market structure shows increasing complexity with major hedge funds building significant short positions (aggregate value $1.2B+). Recent price action contained within $2,400-$2,800 range with declining volatility. Layer 2 adoption surge continues with TVL reaching $27B across major solutions. Staking participation rate stabilized at 25.3% of total supply, yielding average 4.8% APR. DEX volume trending upward with 15% MoM growth. Gas fees moderating post-EIP-4844 implementation, averaging 20 gwei for standard transfers.",
                "prediction": {
                    "direction": "Potential breakout",
                    "reasoning": [
                        "1. Record short positions could trigger significant short squeeze",
                        "2. EIP-4844 upgrade improving scalability and reducing fees",
                        "3. Growing institutional interest in staking services",
                        "4. Spot ETF approval speculation driving market sentiment"
                    ]
                },
                "timeline": "Oct-Dec 2025",
                "confidence": 60,
                "status": "In Progress"
            },
            {
                "symbol": "SOL",
                "name": "Solana",
                "price": "$197",
                "priceChange": 4.2,
                "targetPrice": "$250",
                "sentiment": "Bullish",
                "sentimentScore": 7,
                "market": "Solana ecosystem metrics demonstrate robust growth with daily active addresses reaching 1.2M+. Network performance maintains 99.98% uptime over past quarter. DeFi TVL surged 45% YTD to $12.8B, dominated by lending and DEX protocols. NFT marketplace volume showing steady recovery with 28% MoM increase. Developer activity remains strong with 350+ monthly active contributors. Transaction costs stable at ~$0.002 average, processing 3,000 TPS at peak times. Institutional inflows primarily targeting liquid staking derivatives and DeFi yield opportunities.",
                "prediction": {
                    "direction": "Sustained growth expected",
                    "reasoning": [
                        "1. DeFi TVL growing consistently month-over-month",
                        "2. Network stability improvements reducing downtime incidents",
                        "3. Rising institutional adoption in NFT and DeFi sectors",
                        "4. Competitive advantage in transaction speed and costs"
                    ]
                },
                "timeline": "Apr-Jun 2025",
                "confidence": 65,
                "status": "In Progress"
            },
            {
                "symbol": "XRP",
                "name": "XRP",
                "price": "$2.41",
                "priceChange": -0.8,
                "targetPrice": "$3.00",
                "sentiment": "Neutral",
                "sentimentScore": 6,
                "market": "XRP market sentiment improving following regulatory clarity achievements. Network seeing increased adoption in cross-border payment corridors, with 24h transfer volume reaching $450M+. Banking partnerships expanded to 35+ institutions across APAC and LATAM regions. ODL (On-Demand Liquidity) corridors showing 25% QoQ growth in transaction volume. Ripple's enterprise solutions gaining traction with 15 new financial institutions onboarded in Q1 2025. Network upgrades focused on smart contract functionality and DeFi ecosystem development.",
                "prediction": {
                    "direction": "Moderate upside potential",
                    "reasoning": [
                        "1. Ongoing partnerships with major financial institutions",
                        "2. Improved regulatory clarity post-SEC case resolution",
                        "3. Expansion in cross-border payment solutions",
                        "4. Growing adoption in emerging markets"
                    ]
                },
                "timeline": "Jan-Mar 2026",
                "confidence": 55,
                "status": "In Progress"
            }
        ]

        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            
            # Create table if it doesn't exist
            c.execute('''
                CREATE TABLE IF NOT EXISTS market_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    name TEXT NOT NULL,
                    price TEXT NOT NULL,
                    price_change REAL NOT NULL,
                    target_price TEXT NOT NULL,
                    sentiment TEXT NOT NULL,
                    sentiment_score INTEGER NOT NULL,
                    market TEXT NOT NULL,
                    prediction_direction TEXT NOT NULL,
                    prediction_reasoning TEXT NOT NULL,
                    timeline TEXT NOT NULL,
                    confidence INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol)
                )
            ''')
            
            # Insert data
            for item in initial_data:
                c.execute('''
                    INSERT OR REPLACE INTO market_analysis 
                    (symbol, name, price, price_change, target_price, 
                     sentiment, sentiment_score, market, prediction_direction,
                     prediction_reasoning, timeline, confidence, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    item['symbol'],
                    item['name'],
                    item['price'],
                    item['priceChange'],
                    item['targetPrice'],
                    item['sentiment'],
                    item['sentimentScore'],
                    item['market'],
                    item['prediction']['direction'],
                    json.dumps(item['prediction']['reasoning']),
                    item['timeline'],
                    item['confidence'],
                    item['status']
                ))

        return jsonify({
            "status": "success",
            "message": "Market analysis data inserted successfully"
        })

    except Exception as e:
        print(f"Error inserting market analysis: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/market-analysis/get', methods=['GET'])
def get_market_analysis():
    """Get market analysis data"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            
            c.execute('''
                SELECT *
                FROM market_analysis
                ORDER BY confidence DESC
            ''')
            
            rows = c.fetchall()
            analyses = []
            
            for row in rows:
                analysis = {
                    "symbol": row['symbol'],
                    "name": row['name'],
                    "price": row['price'],
                    "priceChange": row['price_change'],
                    "targetPrice": row['target_price'],
                    "sentiment": row['sentiment'],
                    "sentimentScore": row['sentiment_score'],
                    "market": row['market'],
                    "prediction": {
                        "direction": row['prediction_direction'],
                        "reasoning": json.loads(row['prediction_reasoning'])
                    },
                    "timeline": row['timeline'],
                    "confidence": row['confidence'],
                    "status": row['status'],
                    "lastUpdated": row['last_updated']
                }
                analyses.append(analysis)

            return jsonify({
                "status": "success",
                "data": analyses
            })

    except Exception as e:
        print(f"Error getting market analysis: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/v1/combined-analysis', methods=['GET'])
def get_combined_analysis():
    """Fetch comprehensive market analysis"""
    try:
        # Get market analysis data
        response = requests.get('https://api.moometrics.io/news/market-analysis/get', timeout=30)
        response.raise_for_status()
        analyses = response.json().get('data', [])
        
        processed_analyses = []
        for analysis in analyses:
            # Get coin details
            coin_search = requests.get(
                f"https://api.heifereum.com/api/cryptocurrency/search",
                params={"query": analysis['symbol']},
                timeout=30
            )
            coin_search.raise_for_status()
            coin_data = coin_search.json().get('coins', [])
            if not coin_data:
                continue
            
            coin_details = coin_data[0]
            
            # Get crypto info
            crypto_info = requests.get(
                f"https://api.heifereum.com/api/cryptocurrency/info",
                params={"id": coin_details['id']},
                timeout=30
            )
            crypto_info.raise_for_status()
            coin_info = crypto_info.json()
            market_data = coin_info.get('market_data', {})
            
            # Format response
            processed_analysis = {
                "confidence": analysis['confidence'],
                "lastUpdated": analysis['lastUpdated'],
                "market": analysis['market'],
                "id": coin_details['id'],
                "symbol": analysis['symbol'],
                "prediction": analysis['prediction'],
                "current_price": float(market_data.get('current_price', {}).get('usd', '0')),
                "price_change_24h": float(market_data.get('price_change_percentage_24h', '0')),
                "sentiment": analysis['sentiment'],
                "sentiment_score": analysis['sentimentScore'],
                "status": analysis['status'],
                "target_price": float(analysis['targetPrice'].replace('$', '').replace(',', '')),
                "timeline": analysis['timeline'],
                "image": coin_info.get('image', {}).get('thumb', '')
            }
            processed_analyses.append(processed_analysis)
            
        return jsonify({
            "status": "success",
            "data": processed_analyses
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/v1/combined-analysis/<symbol>', methods=['GET'])
def get_single_combined_analysis(symbol):
    """Fetch analysis for a specific cryptocurrency"""
    try:
        # Get market analysis data
        response = requests.get('https://api.moometrics.io/news/market-analysis/get', timeout=30)
        response.raise_for_status()
        analyses = response.json().get('data', [])
        
        # Find matching analysis
        analysis = next((a for a in analyses if a['symbol'].upper() == symbol.upper()), None)
        if not analysis:
            return jsonify({
                "status": "error",
                "message": f"No analysis found for {symbol}"
            }), 404
        
        # Get coin details
        coin_search = requests.get(
            f"https://api.heifereum.com/api/cryptocurrency/info",
            params={"query": analysis['symbol']},
            timeout=30
        )
        coin_search.raise_for_status()
        coin_data = coin_search.json().get('coins', [])
        if not coin_data:
            return jsonify({
                "status": "error",
                "message": f"Coin not found: {symbol}"
            }), 404
        
        coin_details = coin_data[0]
        
        # Get crypto info
        crypto_info = requests.get(
            f"https://api.heifereum.com/api/cryptocurrency/info",
            params={"id": coin_details['id']},
            timeout=30
        )
        crypto_info.raise_for_status()
        coin_info = crypto_info.json()
        market_data = coin_info.get('market_data', {})
        
        # Format response
        processed_analysis = {
            "confidence": analysis['confidence'],
            "lastUpdated": analysis['lastUpdated'],
            "market": analysis['market'],
            "id": coin_details['id'],
            "symbol": analysis['symbol'],
            "prediction": analysis['prediction'],
            "current_price": float(market_data.get('current_price', {}).get('usd', '0')),
            "price_change_24h": float(market_data.get('price_change_percentage_24h', '0')),
            "sentiment": analysis['sentiment'],
            "sentiment_score": analysis['sentimentScore'],
            "status": analysis['status'],
            "target_price": float(analysis['targetPrice'].replace('$', '').replace(',', '')),
            "timeline": analysis['timeline'],
            "image": coin_info.get('image', {}).get('thumb', '')
        }
        
        return jsonify({
            "status": "success",
            "data": processed_analysis
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/respond', methods=['POST'])
def respond():
    data = request.json
    prompt = data.get('message')
    
    # Use the provided API key directly
    API_KEY = 'AIzaSyBdMrLHWWswcUjJABblf2E1n2s1TZqYu9w'  # Your actual API key

    try:
        # Configure the model with the API key
        genai.configure(api_key=API_KEY)  # Configure the model with the API key
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Structured prompt for Fiqh questions
        system_prompt = """
        You are a Fiqh guidance system. When users ask about Zakat or other Fiqh topics:
        1. Instead of giving direct answers, provide 3-4 clarifying questions
        2. Focus on understanding which specific aspect they need help with
        3. Questions should cover different aspects like:
           - Practical implementation
           - Conditions and requirements
           - Specific situations or cases
           - Timing and scheduling
        4. Format your response as numbered questions only
        5. Each question should start with "Are you asking about..."

        For example, if someone asks about Zakat, respond with questions like:
        1. Are you asking about how to calculate the minimum amount (nisab) for Zakat?
        2. Are you asking about which types of wealth are subject to Zakat?
        3. Are you asking about when Zakat becomes obligatory?

        Provide 6 clarifying questions

        Provide respond according to the language of the user if its english or urdu or arabic or malay. The question is not only about zakat but also about other fiqh topics.

        Current user question\n: {user_question}
        """
        
        final_prompt = system_prompt.format(user_question=prompt)
        response = model.generate_content(final_prompt)
        return jsonify({"clarifying_questions": response.text})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/answer-detailed', methods=['POST'])
def answer_detailed():
    try:
        # Get the question from request body
        request_data = request.get_json()
        if not request_data or 'question' not in request_data:
            return jsonify({
                "error": "Please provide a question in the request body",
                "status": "error"
            }), 400

        question = request_data['question']
        print(f"Received question: {question}")

        # Use Gemini to identify relevant keywords and analyze the question
        print("Analyzing question with Gemini...")
        api_key = "AIzaSyCrH4y3hiGHd8UnsXkWvC2_OvNcJb4jWZo"
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-pro')

        analysis_prompt = f"""
        Given this question: "{question}"
        
        Identify the most important keywords or concepts that would help find relevant information in an Islamic text.
        Return your response in this JSON format:
        {{
            "keywords": ["keyword1", "keyword2", ...]
        }}
        """

        print("Sending analysis prompt to Gemini...")
        keywords_response = model.generate_content(analysis_prompt)
        keywords_data = json.loads(keywords_response.text)
        keywords = keywords_data.get('keywords', [])
        print(f"Keywords identified: {keywords}")

        # Read batch.json
        print("Reading batch.json file...")
        try:
            with open('batch.json', 'r', encoding='utf-8') as f:
                batch_data = json.load(f)
        except Exception as e:
            print(f"Error reading batch.json: {str(e)}")
            return jsonify({
                "error": "Error reading batch analysis file",
                "status": "error"
            }), 500

        # Search through batch analyses for relevant content based on keywords
        print("Searching through batch analyses...")
        relevant_analyses = []

        for batch in batch_data['batch_analyses']:
            analysis_text = batch.get('analysis', '').lower()
            # Check if any keywords match in the analysis
            if any(keyword.lower() in analysis_text for keyword in keywords):
                print(f"Found relevant content in batch: {batch['batch']}")
                relevant_analyses.append(batch)

        if not relevant_analyses:
            print("No relevant content found")
            return jsonify({
                "status": "no_results",
                "message": "No relevant content found for your question"
            })

        # Use Gemini to generate a final answer based on the found content
        print("Generating final answer...")
        answer_prompt = f"""
        Question: {question}

        Based on these relevant excerpts from an Islamic text:

        {json.dumps(relevant_analyses, indent=2)}

        Please provide:
        1. A direct answer to the question
        2. The specific batches where this information was found
        3. Any additional context that might be helpful
        """

        final_answer = model.generate_content(answer_prompt)

        return jsonify({
            "status": "success",
            "question": question,
            "keywords_used": keywords,
            "relevant_analyses": relevant_analyses,
            "answer": final_answer.text
        })

    except Exception as e:
        print(f"Error in /answer-detailed endpoint: {str(e)}")
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500
def remove_backticks(text):
    """Remove three consecutive backticks from the text."""
    return text.replace('```', '').replace('`', '')


class ExpiringCache:
    def __init__(self, expiration_hours=24):
        self.data = None
        self.timestamp = None
        self.expiration_hours = expiration_hours
    
    def set(self, data):
        self.data = data
        self.timestamp = datetime.now()
    
    def get(self):
        if not self.data or not self.timestamp:
            return None
        
        # Check if cache has expired
        if datetime.now() - self.timestamp > timedelta(hours=self.expiration_hours):
            self.data = None
            self.timestamp = None
            return None
        
        return self.data

# Global cache instance
_thumbs_cache = ExpiringCache()

def get_coin_thumb(coin_name):
    """Get thumbnail for a specific coin with robust error handling"""
    try:
        # Clean coin name and create search URL
        search_term = coin_name.lower().strip()
        if '(' in search_term:
            search_term = search_term.split('(')[0].strip()
        
        url = f'https://api.heifereum.com/api/cryptocurrency/search?query={search_term}'
        
        # Make request with proper headers and timeout
        headers = {
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/json'
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('coins') and len(data['coins']) > 0:
                    thumb = data['coins'][0].get('thumb')
                    if thumb:
                        print(f"✅ Successfully found thumbnail for {coin_name}")
                        return {'coin': coin_name, 'thumb': thumb, 'status': 'success'}
                
                print(f"❌ No thumbnail found for {coin_name}")
                return {'coin': coin_name, 'thumb': 'not found', 'status': 'not_found'}
            
            print(f"❌ Failed to fetch data for {coin_name}. Status code: {response.status_code}")
            return {'coin': coin_name, 'thumb': 'not found', 'status': 'not_found'}
        
        except requests.RequestException as e:
            print(f"❌ Request error for {coin_name}: {str(e)}")
            return {'coin': coin_name, 'thumb': 'error', 'error': str(e), 'status': 'error'}
    
    except Exception as e:
        print(f"❌ Unexpected error for {coin_name}: {str(e)}")
        return {'coin': coin_name, 'thumb': 'error', 'error': str(e), 'status': 'error'}

def process_coins_sequentially(coins):
    """Process coins sequentially"""
    results = []
    success_count = 0
    not_found_count = 0
    error_count = 0
    
    for i, coin in enumerate(coins, 1):
        result = get_coin_thumb(coin)
        results.append(result)
        
        # Update counts based on status
        if result['status'] == 'success':
            success_count += 1
        elif result['status'] == 'not_found':
            not_found_count += 1
        else:
            error_count += 1
        
        # Optional: Print progress with success ratio
        print(f"Progress: {i}/{len(coins)} coins processed | Success: {success_count}/{i} ({(success_count/i)*100:.1f}%)")
    
    return results, success_count, not_found_count, error_count

@app.route('/crypto/test/<coin>')
def test_coin(coin):
    """Test endpoint to check single coin"""
    result = get_coin_thumb(coin)
    return jsonify({
        'coin': coin,
        'result': result,
        'timestamp': time.time()
    })

@app.route('/crypto/start')
def fetch_thumbs():
    """Start fetching thumbnails sequentially"""
    total_start_time = time.time()
    
    try:
        # Get coins list
        print("Fetching list of coins...")
        response = requests.get('https://api.moometrics.io/news/youtube/coins/names')
        coins = response.json()['data']['coins']
        print(f"Total coins to process: {len(coins)}")
        
        # Process coins
        results, success_count, not_found_count, error_count = process_coins_sequentially(coins)
        
        # Store in cache
        cache_data = {
            'results': results,
            'total_coins': len(coins),
            'successful': success_count,
            'not_found': not_found_count,
            'errors': error_count
        }
        _thumbs_cache.set(cache_data)
        
        total_time = time.time() - total_start_time
        
        return jsonify({
            "execution_time": f"{total_time:.2f} seconds",
            "total_coins": len(coins),
            "successful": success_count,
            "not_found": not_found_count,
            "errors": error_count,
            "success_rate": f"{(success_count/len(coins))*100:.1f}%",
            "results": results
        })
        
    except Exception as e:
        print(f"Error during thumbnail fetch: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/crypto/get')
def get_thumbs():
    """Get cached thumbnails"""
    cached_data = _thumbs_cache.get()
    if cached_data:
        return jsonify(cached_data)
    return jsonify({"message": "No data. Run /crypto/start first"}), 404

@dataclass
class CoinTrendResponse:
    bearish: int
    bullish: int
    symbol: str
    id: str
    image: str
    date: str
    current_price: float
    price_24h: float
    price_30d: float
    price_7d: float
    dominance: float
    bull_percentage: float
    bear_percentage: float

@dataclass
class SourceModel:
    channel: str
    processed_at: str
    published_at: str
    video_title: str
    video_url: str

@dataclass
class ReasonModel:
    coin: str
    reason: str
    sentiment: str
    source: SourceModel

@dataclass
class PredictionModel:
    direction: str
    reasoning: List[str]

@dataclass
class MarketAnalysisResponse:
    confidence: int
    lastUpdated: str
    market: str
    id: str
    symbol: str
    image: str
    prediction: PredictionModel
    current_price: float
    price_change_24h: float
    sentiment: str
    sentiment_score: int
    status: str
    target_price: float
    timeline: str

def convert_target_price(target_price: str) -> float:
    return float(target_price.replace('$', '').replace(',', ''))

def calculate_bull_bear_percentage(bullish: int, bearish: int) -> float:
    total = bullish + bearish
    if total == 0:
        return 0
    return (bullish / total) * 100

class CoinTrendProcessor:
    def __init__(self, date: str):
        self.coin_trend_url = "https://api.moometrics.io/news/youtube/coin-trends"
        self.heifereum_base_url = "https://api.heifereum.com/api"
        self.default_limit = 10
        self.max_processed_coins = 100
        self.date = date
        
    def search_cryptocurrency(self, symbol: str) -> Optional[Dict[str, Any]]:
        try:
            response = requests.get(
                f"{self.heifereum_base_url}/cryptocurrency/search",
                params={"query": symbol},
                timeout=30
            )
            response.raise_for_status()
            data = response.json().get('coins', [])
            return data[0] if data else None
        except requests.RequestException as e:
            return None

    def get_crypto_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        try:
            # Cache the coin details to avoid multiple API calls
            coin_details = self.search_cryptocurrency(symbol)
            if not coin_details:
                return None
                
            response = requests.get(   
                f"{self.heifereum_base_url}/cryptocurrency/info",
                params={"id": coin_details['id']},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            data['coin_details'] = coin_details
            return data
        except requests.RequestException as e:
            return None

    def get_coin_trend(self, skip: int = 0, limit: int = None) -> List[CoinTrendResponse]:
        try:
            if limit is None:
                limit = self.default_limit

            response = requests.get(self.coin_trend_url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if not isinstance(data, dict) or 'data' not in data:
                return []
                
            # Filter trends by date
            trends = [
                trend for trend in data['data'] 
                if trend.get('date') == self.date
            ]
            
            # Apply pagination to the filtered trends
            paginated_trends = trends[skip:skip + limit]
            
            coin_trend_responses = []
            
            for trend in paginated_trends:
                if not isinstance(trend, dict):
                    continue
                    
                coin_symbol = trend.get('coin')
                if not coin_symbol:
                    continue
                    
                coin_info = self.get_crypto_info(coin_symbol)
                if not coin_info:
                    continue
                        
                market_data = coin_info.get('market_data', {})
                coin_details = coin_info['coin_details']
                image = coin_info.get('image', {}).get('thumb', '')
                
                coin_trend_responses.append(CoinTrendResponse(
                    bearish=trend.get('bearish', 0),
                    bullish=trend.get('bullish', 0),
                    symbol=coin_symbol,
                    id=coin_details['id'],
                    image=image,
                    date=self.date,
                    current_price=float(market_data.get('current_price', {}).get('usd', 0)),
                    price_24h=float(market_data.get('price_change_percentage_24h', 0)),
                    price_30d=float(market_data.get('price_change_percentage_30d', 0)),
                    price_7d=float(market_data.get('price_change_percentage_7d', 0)),
                    dominance=float(market_data.get('dominance', {}).get('usd', 0)),
                    bull_percentage=calculate_bull_bear_percentage(trend.get('bullish', 0), trend.get('bearish', 0)),
                    bear_percentage=calculate_bull_bear_percentage(trend.get('bearish', 0), trend.get('bullish', 0))
                ))
                    
            return coin_trend_responses
        except requests.RequestException as e:
            return []

class MarketDataProcessor:
    def __init__(self):
        self.moometrics_url = "https://api.moometrics.io/news/market-analysis/get"
        self.heifereum_base_url = "https://api.heifereum.com/api"

    def search_cryptocurrency(self, symbol: str) -> Optional[Dict[str, Any]]:
        try:
            response = requests.get(
                f"{self.heifereum_base_url}/cryptocurrency/search",
                params={"query": symbol},
                timeout=30
            )
            response.raise_for_status()
            data = response.json().get('coins', [])
            return data[0] if data else None
        except requests.RequestException as e:
            return None

    def get_crypto_info(self, coin_id: str) -> Dict[str, Any]:
        try:
            response = requests.get(   
                f"{self.heifereum_base_url}/cryptocurrency/info",
                params={"id": coin_id},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {}

    def get_market_analysis(self) -> List[Dict[str, Any]]:
        try:
            response = requests.get(self.moometrics_url, timeout=30)
            response.raise_for_status()
            return response.json().get('data', [])
        except requests.RequestException as e:
            return []

    def process_combined_data(self, analysis: Dict[str, Any]) -> MarketAnalysisResponse:
        coin_details = self.search_cryptocurrency(analysis['symbol'])
        if not isinstance(coin_details, dict):
            raise ValueError(f"Coin not found: {analysis['symbol']}")

        coin_info = self.get_crypto_info(coin_details['id'])
        if not isinstance(coin_info, dict):
            raise ValueError(f"Error fetching coin info for {analysis['symbol']}")

        market_data = coin_info.get('market_data', {})

        return MarketAnalysisResponse(
            confidence=analysis['confidence'],
            lastUpdated=analysis['lastUpdated'],
            market=analysis['market'],
            id=coin_details['id'],
            symbol=analysis['symbol'],
            prediction=analysis['prediction'],
            current_price=float(market_data.get('current_price', {}).get('usd', '0')),
            price_change_24h=float(market_data.get('price_change_percentage_24h', '0')),
            sentiment=analysis['sentiment'],
            sentiment_score=analysis['sentimentScore'],
            status=analysis['status'],
            target_price=convert_target_price(analysis['targetPrice']),
            timeline=analysis['timeline'],
            image=coin_info.get('image', {}).get('thumb', '')
        )


trend_processor = CoinTrendProcessor(date=datetime.now().strftime("%Y-%m-%d"))

@app.route("/api/v1/coin-trend")
def get_coin_trend():
    """
    Get coin trends with pagination support.
    Default limit is 10 coins per page.
    Use skip parameter to paginate through results.
    """
    try:
        skip = request.args.get('skip', default=0, type=int)
        limit = request.args.get('limit', default=None, type=int)
        result = trend_processor.get_coin_trend(skip=skip, limit=limit)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/youtube/mentions/<date>')
def get_daily_coin_mentions(date):
    """Get count of unique coins mentioned for a specific date (Malaysia timezone)"""
    try:
        # Validate date format
        try:
            target_date = datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            return jsonify({
                "status": "error",
                "message": "Invalid date format. Please use YYYY-MM-DD"
            }), 400

        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            
            # Get all coin name overrides
            c.execute('''
                SELECT 
                    current_name,
                    new_name
                FROM coin_edits
                ORDER BY edited_at DESC
            ''')
            name_mapping = {row['current_name']: row['new_name'] for row in c.fetchall()}
            
            # Get all mentions for processing
            c.execute('''
                SELECT v.published_at, ca.coin_mentioned
                FROM videos v
                LEFT JOIN coin_analysis ca ON v.video_id = ca.video_id
                WHERE ca.coin_mentioned IS NOT NULL
            ''')
            
            # Process results in Python
            malaysia_tz = pytz.timezone('Asia/Kuala_Lumpur')
            target_date_my = malaysia_tz.localize(target_date)
            start_my = target_date_my.replace(hour=0, minute=0, second=0, microsecond=0)
            end_my = target_date_my.replace(hour=23, minute=59, second=59)
            
            # Filter and process mentions
            unique_coins = set()
            for row in c.fetchall():
                try:
                    # Parse ISO format timestamp
                    published_at = datetime.fromisoformat(row['published_at'].replace('Z', '+00:00'))
                    published_at = pytz.utc.localize(published_at) if published_at.tzinfo is None else published_at
                    published_my = published_at.astimezone(malaysia_tz)
                    
                    if start_my <= published_my <= end_my:
                        coin_name = row['coin_mentioned']
                        if coin_name in name_mapping:
                            coin_name = name_mapping[coin_name]
                        unique_coins.add(coin_name)
                except Exception as e:
                    print(f"Error processing row: {str(e)}")
                    continue
            
            return jsonify({
                "status": "success",
                "date": date,
                "total_coins": len(unique_coins)
            })
            
    except Exception as e:
        print(f"Error getting coin mentions count: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/youtube/reason/prompt5')
def market_summary_today():
    """Get market summary for today (Malaysia timezone)"""
    try:
        # Get current date in Malaysia timezone
        malaysia_tz = pytz.timezone('Asia/Kuala_Lumpur')
        malaysia_time = datetime.now(malaysia_tz)
        print("Malayisa: ", malaysia_time)
        date = malaysia_time.strftime('%Y-%m-%d')
        
        print(f"Getting market summary for {date} (Malaysia timezone)")
        
        # Get raw data using getlight endpoint
        with app.test_client() as client:
            response = client.get(f'/youtube/today-analysis/{date}')
            print(response)
            if response.status_code != 200:
                return jsonify({
                    "status": "error",
                    "message": f"No data found for {date}"
                }), 404
                
            raw_data = response.get_json()
            if not raw_data.get('data', {}).get('reasons'):
                return jsonify({
                    "status": "error",
                    "message": f"No reasons found for {date}"
                }), 404

        # Configure Gemini
        genai.configure(api_key=os.getenv('GEMINI_API_KEY_1'))
        model = genai.GenerativeModel(model_name="gemini-1.5-pro")
        
        # Collect all reasons
        all_reasons = []
        for reason in raw_data['data']['reasons']:
            all_reasons.append({
                'coin': reason['coin'],
                'reason': reason['reason']
            })

        prompt = f"""Create a concise, engaging 4-5 minute broadcast script summarizing the most important crypto market developments. Make it exciting while maintaining credibility and realistic analysis.

        CRITICAL REQUIREMENTS:
        1. Structure the script in this exact order:
           - Opening hook (5-10 seconds)
             * Use verified, significant developments ("BlackRock's new $2B position")
             * Focus on actual institutional movements
             * Example: "Major banks are rapidly entering crypto - here's what it means"
           
           - Global economic context (45-60 seconds)
             * Connect crypto to broader financial trends
             * Use realistic institutional comparisons
             * Example: "Traditional finance is allocating 1-2% to crypto, potentially moving billions into the market"
             * Include specific but credible numbers
           
           - Regulatory landscape (60-75 seconds)
             * Focus on actual policy impacts
             * Connect regulations to market structure
             * Example: "New regulations could open crypto to pension funds, accessing a $50 trillion market"
             * Highlight genuine institutional interest
           
           - Market movements & technical analysis (45-60 seconds)
             * Use historical data for context
             * Include reasonable price analyses
             * Example: "Bitcoin's pattern mirrors the 2020 accumulation phase"
             * Focus on market structure and volume
           
           - Forward-looking conclusion (30-45 seconds)
             * Summarize key opportunities with realistic timeframes
             * Highlight genuine market catalysts
        
        2. Writing style must be:
           - Professional yet accessible
           - Back excitement with data
           - Use clear comparisons to traditional markets
           - Explain complex concepts simply
           - Focus on structural changes over price predictions

        3. KEY GUIDELINES:
           - Keep price predictions within historical ranges
           - Focus more on market structure than specific targets
           - Use institutional data to support analysis
           - Highlight genuine adoption metrics
           - Connect predictions to actual market catalysts
           
        4. CREDIBILITY RULES:
           - All gains must be based on historical precedent
           - Major predictions need supporting evidence
           - Focus on verified institutional movements
           - Use actual market data and volumes
           - Maintain professional skepticism
        
        5. Format the output exactly as shown:
        {{
            "broadcast_script": {{
                "title": "Crypto Market Update for [DATE]",
                "estimated_duration": "4:30",
                "script_sections": [
                    {{
                        "section": "Opening",
                        "content": "This is [Network] Crypto with your market intelligence update. Today we're tracking major institutional movements and regulatory developments that could reshape the crypto landscape."
                    }},
                    {{
                        "section": "Body",
                        "content": "[Main content with balanced analysis and credible developments]"
                    }},
                    {{
                        "section": "Conclusion",
                        "content": "[Evidence-based summary and key developments to watch]"
                    }}
                ]
            }}
        }}

        Analyze and transform these market events into a balanced, credible broadcast script:
        {json.dumps(all_reasons, indent=2)}
        """

        response = model.generate_content(prompt)
        clean_response = response.text.strip()
        
        # Print raw response for debugging
        print(f"Raw Gemini response:")
        print(clean_response)
        
        # Clean up JSON response
        if clean_response.startswith('```json'):
            clean_response = clean_response[7:]
        if clean_response.startswith('```'):
            clean_response = clean_response[3:]
        if clean_response.endswith('```'):
            clean_response = clean_response[:-3]

        # Parse the cleaned response
        analysis = json.loads(clean_response.strip())
        
        # Store in database
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            
            # Create table if it doesn't exist
            c.execute('''
                CREATE TABLE IF NOT EXISTS market_summaries (
                    analysis_date TEXT PRIMARY KEY,
                    summary TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Insert or update the summary
            c.execute('''
                INSERT OR REPLACE INTO market_summaries 
                (analysis_date, summary, created_at, updated_at)
                VALUES (?, ?, datetime('now'), datetime('now'))
            ''', (date, clean_response.strip()))
            
            conn.commit()

        return jsonify({
            "status": "success",
            "data": analysis
        })

    except json.JSONDecodeError as e:
        print(f"JSON parsing error in market_summary_today: {str(e)}")
        print(f"Problematic response: {clean_response}")
        return jsonify({
            "status": "error",
            "message": "Error parsing AI response",
            "details": str(e)
        }), 500
        
    except sqlite3.Error as e:
        print(f"Database error in market_summary_today: {str(e)}")
        return jsonify({
            "status": "error",
            "message": "Database operation failed",
            "details": str(e)
        }), 500
        
    except Exception as e:
        print(f"Error in market_summary_today: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": "An unexpected error occurred",
            "details": str(e)
        }), 500
    

# Configure Gemini clients
# Configure Gemini clients
EMBEDDING_API_KEY = "AIzaSyAJkVH1OkkhIJIvkQ4_zj7MvbwOgcvJifA"
GEMINI_API_KEY = "AIzaSyAOogW_ZTPgDniIc0ecGSQk_4L9U_y7dno"
GEMINI_API_KEY_2 = "AIzaSyBxeQKYExn_2Mu2wkg9ExfQr_yn7RiJ6Ow"
GENERAL_API_KEY = "AIzaSyDNWfFmywWgydEI0NxL9xbCTjdlnYlOoKE"

# Initialize the API
palm.configure(api_key=EMBEDDING_API_KEY)

def clean_frontend_query(query: str) -> str:
    patterns = [
        r"^Are you asking about ",
        r"^Would you like to know ",
        r"^Do you want to learn ",
        r"^Are you interested in ",
    ]
    cleaned = query.lower()
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("?", "").replace("the", "").replace("and", "")
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def get_embeddings(text: str):
    try:
        # Configure the model
        genai.configure(api_key=EMBEDDING_API_KEY)
        model = genai.GenerativeModel('text-embedding-004')  # Use the embedding model
        
        # Generate embeddings
        result = model.embed_content(text=text)
        return result.embedding
    except Exception as e:
        print(f"Error generating embedding: {str(e)}")
        return None

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_similar_texts(query: str, top_k: int = 5):
    """Shared function to get similar texts"""
    try:
        with open("validated_content/embedded_toharoh_entries.json", 'r', encoding='utf-8') as f:
            toharoh_data = json.load(f)
            # Flatten the madhab-structured data
            embedded_entries = []
            for madhab, entries in toharoh_data.items():
                for entry in entries:
                    entry['madhab'] = madhab  # Add madhab info to entry
                    embedded_entries.append(entry)
    except FileNotFoundError:
        return None, "Database file not found"
    
    cleaned_query = clean_frontend_query(query)
    query_embedding = get_embeddings(cleaned_query)
    
    if not query_embedding:
        return None, "Failed to generate embeddings"
    
    results = []
    for entry in embedded_entries:
        similarity = cosine_similarity(query_embedding, entry['english_translation_embedding'])
        results.append({
            'madhab': entry['madhab'],
            'page': entry['page_number'],
            'similarity': similarity,
            'arabic': entry['arabic_text'],
            'english': entry['english_translation']
        })
    
    results.sort(key=lambda x: x['similarity'], reverse=True)
    return results[:top_k], cleaned_query

def generate_content(prompt: str, api_key: str):
    palm.configure(api_key=api_key)
    model = palm.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content(prompt)
    return response.text

def analyze_first_three(results):
    """Analyze definition, ruling, and evidence"""
    content_for_analysis = "Analyze these Islamic texts for definition, ruling, and evidence:\n\n"
    for idx, result in enumerate(results, 1):
        content_for_analysis += f"Text {idx}:\nOriginal:\n{result['arabic']}\n\nTranslation:\n{result['english']}\n\nPage: {result['page']}\n\n"

    prompt = f"""{content_for_analysis}
Please analyze these texts and provide a response in the following JSON format:
{{
    "definition": {{
        "explanation": "General explanation of the concept",
        "schools": [
            {{
                "name": "Hanafi",
                "definition": "How the Hanafi school defines this concept",
                "source": "Text number and page reference if available, otherwise 'No specific definition found'"
            }},
            {{
                "name": "Maliki",
                "definition": "How the Maliki school defines this concept",
                "source": "Text number and page reference if available, otherwise 'No specific definition found'"
            }},
            {{
                "name": "Shafi",
                "definition": "How the Shafi school defines this concept",
                "source": "Text number and page reference if available, otherwise 'No specific definition found'"
            }},
            {{
                "name": "Hanbali",
                "definition": "How the Hanbali school defines this concept",
                "source": "Text number and page reference if available, otherwise 'No specific definition found'"
            }},
            {{
                "name": "OTHER",
                "definition": "How other schools or scholars define this concept",
                "source": "Text number and page reference if available, otherwise 'No specific definition found'"
            }}
        ]
    }},
    "ruling": {{
        "explanation": "General explanation of the ruling",
        "schools": [
            {{
                "name": "Hanafi",
                "ruling": "The Hanafi ruling on this matter",
                "source": "Text number and page reference if available, otherwise 'No specific ruling found'"
            }},
            {{
                "name": "Maliki",
                "ruling": "The Maliki ruling on this matter",
                "source": "Text number and page reference if available, otherwise 'No specific ruling found'"
            }},
            {{
                "name": "Shafi",
                "ruling": "The Shafi ruling on this matter",
                "source": "Text number and page reference if available, otherwise 'No specific ruling found'"
            }},
            {{
                "name": "Hanbali",
                "ruling": "The Hanbali ruling on this matter",
                "source": "Text number and page reference if available, otherwise 'No specific ruling found'"
            }},
            {{
                "name": "OTHER",
                "ruling": "Rulings from other schools or scholars",
                "source": "Text number and page reference if available, otherwise 'No specific ruling found'"
            }}
        ]
    }},
    "evidence": {{
        "explanation": "Overview of the evidences presented",
        "schools": [
            {{
                "name": "Hanafi",
                "arabic_text": "Original Arabic/Urdu text supporting this evidence, strictly in arabic/urdu, which is evidence from Hanafi school if not available, return 'No specific evidence found' ",
                "translation": "English translation of the evidence text"
            }},
            {{
                "name": "Maliki",
                "arabic_text": "Original Arabic/Urdu text supporting this evidence, strictly in arabic/urdu, which is evidence from Maliki school if not available, return 'No specific evidence found' ",
                "translation": "English translation of the evidence text"
            }},
            {{
                "name": "Shafi",
                "arabic_text": "Original Arabic/Urdu text supporting this evidence, strictly in arabic/urdu, which is evidence from Shafi school if not available, return 'No specific evidence found' ",
                "translation": "English translation of the evidence text"
            }},
            {{
                "name": "Hanbali",
                "arabic_text": "Original Arabic/Urdu text supporting this evidence, strictly in arabic/urdu, which is evidence from Hanbali school if not available, return 'No specific evidence found' ",
                "translation": "English translation of the evidence text"
            }},
        ]
    }}
}}"""

    try:
        response_text = generate_content(prompt, GEMINI_API_KEY)
        if '```json' in response_text:
            json_str = response_text.split('```json')[1].split('```')[0].strip()
        elif '```' in response_text:
            json_str = response_text.split('```')[1].strip()
        else:
            json_str = response_text.strip()
            
        return json.loads(json_str)
    except Exception as e:
        return {"error": "Analysis failed", "message": str(e)}

def analyze_last_three(results):
    """Analyze reasoning, application, and examples"""
    content_for_analysis = "Analyze these Islamic texts for reasoning, practical applications, and examples:\n\n"
    for idx, result in enumerate(results, 1):
        content_for_analysis += f"Text {idx}:\nOriginal:\n{result['arabic']}\n\nTranslation:\n{result['english']}\n\nPage: {result['page']}\n\n"

    prompt = f"""{content_for_analysis}
Please analyze these texts and provide a response in the following JSON format:
{{
    "reasoning": {{
        "explanation": "Overview of the underlying wisdom and reasoning",
        "schools": [
            {{
                "name": "Hanafi",
                "reasoning": "The underlying wisdom according to Hanafi school",
                "arabic_text": "Original Arabic/Urdu text showing this reasoning",
                "translation": "English translation of the reasoning text"
            }},
            {{
                "name": "Maliki",
                "reasoning": "The underlying wisdom according to Maliki school",
                "arabic_text": "Original Arabic/Urdu text showing this reasoning",
                "translation": "English translation of the reasoning text"
            }},
            {{
                "name": "Shafi",
                "reasoning": "The underlying wisdom according to Shafi school",
                "arabic_text": "Original Arabic/Urdu text showing this reasoning",
                "translation": "English translation of the reasoning text"
            }},
            {{
                "name": "Hanbali",
                "reasoning": "The underlying wisdom according to Hanbali school",
                "arabic_text": "Original Arabic/Urdu text showing this reasoning",
                "translation": "English translation of the reasoning text"
            }}
        ]
    }},
    "application": {{
        "explanation": "Overview of practical applications",
        "schools": [
            {{
                "name": "Hanafi",
                "application": "Practical application according to Hanafi school",
                "arabic_text": "Original Arabic/Urdu text about this application",
                "translation": "English translation of the application text"
            }},
            {{
                "name": "Maliki",
                "application": "Practical application according to Maliki school",
                "arabic_text": "Original Arabic/Urdu text about this application",
                "translation": "English translation of the application text"
            }},
            {{
                "name": "Shafi",
                "application": "Practical application according to Shafi school",
                "arabic_text": "Original Arabic/Urdu text about this application",
                "translation": "English translation of the application text"
            }},
            {{
                "name": "Hanbali",
                "application": "Practical application according to Hanbali school",
                "arabic_text": "Original Arabic/Urdu text about this application",
                "translation": "English translation of the application text"
            }}
        ]
    }},
    "examples": {{
        "explanation": "Overview of practical examples",
        "schools": [
            {{
                "name": "Hanafi",
                "example": "Examples from Hanafi school",
                "arabic_text": "Original Arabic/Urdu text of the examples",
                "translation": "English translation of the examples"
            }},
            {{
                "name": "Maliki",
                "example": "Examples from Maliki school",
                "arabic_text": "Original Arabic/Urdu text of the examples",
                "translation": "English translation of the examples"
            }},
            {{
                "name": "Shafi",
                "example": "Examples from Shafi school",
                "arabic_text": "Original Arabic/Urdu text of the examples",
                "translation": "English translation of the examples"
            }},
            {{
                "name": "Hanbali",
                "example": "Examples from Hanbali school",
                "arabic_text": "Original Arabic/Urdu text of the examples",
                "translation": "English translation of the examples"
            }}
        ]
    }}
}}"""

    try:
        response = palm.Client(api_key=GEMINI_API_KEY_2).models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        response_text = response.text
        if '```json' in response_text:
            json_str = response_text.split('```json')[1].split('```')[0].strip()
        elif '```' in response_text:
            json_str = response_text.split('```')[1].strip()
        else:
            json_str = response_text.strip()
            
        return json.loads(json_str)
    except Exception as e:
        return {"error": "Analysis failed", "message": str(e)}

@app.route('/api/search/first', methods=['POST'])
def handle_first_three():
    """Handle definition, ruling, and evidence"""
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400
    
    data = request.get_json()
    if 'query' not in data:
        return jsonify({"error": "Query parameter is required"}), 400
    
    results, cleaned_query = get_similar_texts(data['query'])
    if results is None:
        return jsonify({"error": "Search failed", "message": cleaned_query}), 500
    
    analysis = analyze_first_three(results)
    
    response = {
        "query": {
            "original": data['query'],
            "cleaned": cleaned_query
        },
        "similar_texts": [
            {
                "page": result['page'],
                "similarity": f"{result['similarity']:.2%}",
                "arabic": result['arabic'],
                "english": result['english']
            }
            for result in results
        ],
        "analysis": analysis
    }
    
    return jsonify(response)

@app.route('/api/search/last', methods=['POST'])
def handle_last_three():
    """Handle reasoning, application, and examples"""
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400
    
    data = request.get_json()
    if 'query' not in data:
        return jsonify({"error": "Query parameter is required"}), 400
    
    results, cleaned_query = get_similar_texts(data['query'])
    if results is None:
        return jsonify({"error": "Search failed", "message": cleaned_query}), 500
    
    analysis = analyze_last_three(results)
    
    response = {
        "query": {
            "original": data['query'],
            "cleaned": cleaned_query
        },
        "similar_texts": [
            {
                "page": result['page'],
                "similarity": f"{result['similarity']:.2%}",
                "arabic": result['arabic'],
                "english": result['english']
            }
            for result in results
        ],
        "analysis": analysis
    }
    
    return jsonify(response)

@app.route('/api/query/general', methods=['POST'])
def handle_general_query():
    """Handle initial query by returning both query and extracted keywords"""
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400
    
    data = request.get_json()
    if 'query' not in data:
        return jsonify({"error": "Query parameter is required"}), 400

    try:
        palm.configure(api_key=GENERAL_API_KEY)
        
        answer_prompt = f"""As an Islamic scholar, what you get is a formulated question. Just directly answer the question in general. Make sure at least there's some detailness".

This is the question user clicked that he wants to know about: {data['query']}

Return only the answer not another question, don't start the question with yes or no. Directly answer the question. Nothing else."""

        answer_response = generate_content(answer_prompt, GENERAL_API_KEY)

        keyword_prompt = f"""From this Islamic question: "{data['query']}"
Extract only the most important keywords related to Islamic terms, concepts, or practices. 
Return just 2-4 keywords separated by spaces, nothing else. For example: "wudu prayer fasting" """

        keyword_response = generate_content(keyword_prompt, GENERAL_API_KEY)
        
        return jsonify({
            "query": answer_response,
            "keyword": keyword_response
        })

    except Exception as e:
        error_msg = f"Failed to generate response: {str(e)}"
        return jsonify({"error": error_msg}), 500


if __name__ == '__main__':
    init_db()
    app.run(port=8080, host='0.0.0.0')