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
from statsmodels.tsa.arima.model import ARIMA
from enum import Enum
import logging
import os
from typing import List, Optional,Dict, Any
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
CORS(app)

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

def get_next_run_time():
    """Get next 6 AM MYT run time"""
    malaysia_tz = pytz.timezone('Asia/Kuala_Lumpur')
    now = datetime.now(malaysia_tz)
    next_run = now.replace(hour=6, minute=0, second=0, microsecond=0)
    
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
            genai.configure(api_key="AIzaSyAyQ4DGoHTIDWgfUE5qXl8FNYgBS3hMG_g")
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
            genai.configure(api_key="AIzaSyAyQ4DGoHTIDWgfUE5qXl8FNYgBS3hMG_g")
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
            genai.configure(api_key="AIzaSyAyQ4DGoHTIDWgfUE5qXl8FNYgBS3hMG_g")
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
def get_light_results():
    """Get all results without transcripts for lighter payload"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            
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
            
            # Get analyses for each video
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
        print(f"Error in get_light_results: {str(e)}")  # Debug print
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

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

        current_name = data['current_name']
        new_name = data['new_name']

        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            
            # First find all affected videos with their details
            c.execute('''
                SELECT DISTINCT 
                    v.video_id,
                    v.title,
                    v.channel_name,
                    v.url
                FROM coin_analysis ca
                JOIN videos v ON ca.video_id = v.video_id
                WHERE ca.coin_mentioned = ?
            ''', (current_name,))
            
            affected_videos = [dict(row) for row in c.fetchall()]
            
            if not affected_videos:
                return jsonify({
                    "status": "error",
                    "message": f"No videos found with coin name: {current_name}"
                }), 404
            
            # Insert the edit record
            c.execute('''
                INSERT INTO coin_edits (current_name, new_name)
                VALUES (?, ?)
            ''', (current_name, new_name))
            
            edit_id = c.lastrowid
            
            # Update coin names in all affected videos
            c.execute('''
                UPDATE coin_analysis 
                SET coin_mentioned = ?
                WHERE coin_mentioned = ?
            ''', (new_name, current_name))
            
            # Record all affected videos
            for video in affected_videos:
                c.execute('''
                    INSERT INTO video_edits (edit_id, video_id)
                    VALUES (?, ?)
                ''', (edit_id, video['video_id']))

            # Get the timestamp of the edit
            c.execute('SELECT edited_at FROM coin_edits WHERE id = ?', (edit_id,))
            timestamp = c.fetchone()[0]

            return jsonify({
                "status": "success",
                "data": {
                    "current_name": current_name,
                    "new_name": new_name,
                    "timestamp": timestamp,
                    "videos_affected": len(affected_videos),
                    "videos": [{
                        "video_id": video["video_id"],
                        "title": video["title"],
                        "channel": video["channel_name"],
                        "url": video["url"]
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
    """Get the simplified history of all coin name edits"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            
            # Get all edits with simplified format
            c.execute('''
                SELECT 
                    current_name,
                    new_name,
                    strftime('%H:%M:%S', edited_at) as timestamp
                FROM coin_edits
                ORDER BY edited_at DESC
            ''')
            
            edits = [dict(row) for row in c.fetchall()]
            
            return jsonify({
                "status": "success",
                "data": {
                    "log": edits
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
            genai.configure(api_key="AIzaSyAyQ4DGoHTIDWgfUE5qXl8FNYgBS3hMG_g")
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

if __name__ == '__main__':
    init_db()
    app.run(port=8080, host='0.0.0.0')