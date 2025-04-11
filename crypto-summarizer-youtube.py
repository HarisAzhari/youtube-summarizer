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
from sklearn.preprocessing import MinMaxScaler
import traceback
from functools import wraps
from historical_price import HistoricalPriceService
from dataclasses import dataclass
import re

# Add the new imports at the top with other imports
from google_search import (
    # Original functions
    search_feature_releases,
    search_whitepaper_updates,
    search_testnet_mainnet_milestones,
    search_platform_integration,
    search_team_changes,
    search_new_partnerships,
    search_onchain_activity,
    search_active_addresses_growth,
    search_real_world_adoption,
    search_developer_activity,
    search_community_sentiment,
    search_liquidity_changes,
    
    # New functions
    search_market_cap,
    search_circulating_vs_total_supply,
    search_tokenomics,
    search_inflation_emission_rate,
    search_token_utility,
    search_network_activity_usage,
    search_transaction_fees_revenue,
    search_governance_decentralization,
    search_exchange_listings,
    search_regulatory_status,
    search_security_audits,
    search_competitor_analysis,
    search_volatility_price_swings,
    search_unusual_trading_patterns,
    search_divergence_market_trends,
    search_catalysts,
    search_technical_signals,
    search_comparative_strength_weakness,
    search_trading_volume_anomalies,
    search_exchange_inflows_outflows,
    search_open_interest_futures,
    
    # Utility functions
    analyze_cryptocurrency_full,
    save_analysis_to_file
)

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": [
            "https://admin.moometrics.io",  # Admin frontend
            "https://api.moometrics.io",    # API domain
            "https://moometrics.io",        # Main domain
            "http://localhost:*",           # Local development (any port)
            "http://localhost:5176",
            "https://www.aifiqh.com",
            "https://admin.aifiqh.com",
            "https://admin.moometrics.io/prediciton ",
            "https://admin.moometrics.io/graph"
            "https://admin.moometrics.io/news"
            "https://admin.moometrics.io/news-scraper"
        ],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True,       # Add this for credentials support
        "expose_headers": ["Content-Type", "Authorization"]  # Expose necessary headers
    }
})

# Configuration
DB_PATH = 'youtube_crypto.db'
API_KEY = 'AIzaSyBB1F_3gIWDmC-jZj5kIicdHVhCtZuK-dA'

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
    next_run = now.replace(hour=11, minute=35, second=0, microsecond=0)
    
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
    day_ago = now - timedelta(days=5)
    
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

# At the top of the file, add this cache variable
COIN_NAMES_CACHE = {
    "timestamp": None,
    "data": None,
    "cache_duration": timedelta(hours=1)  # Cache for 1 hour
}

@app.route('/youtube/coins/names')
def get_coin_names():
    """Get just the list of unique coin names with edit history applied"""
    try:
        global COIN_NAMES_CACHE
        current_time = datetime.now()

        # Check if we have valid cached data
        if (COIN_NAMES_CACHE["timestamp"] and 
            COIN_NAMES_CACHE["data"] and 
            current_time - COIN_NAMES_CACHE["timestamp"] < COIN_NAMES_CACHE["cache_duration"]):
            print("✨ Returning cached coin names")
            return jsonify(COIN_NAMES_CACHE["data"])

        print("🔄 Cache expired or empty, fetching fresh coin names")
        
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
            print(f"📝 Found {len(name_mapping)} name overrides")
            
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
            print(f"🪙 Found {len(unique_coins)} unique coins")
            
            response_data = {
                "status": "success",
                "data": {
                    "coins": unique_coins,
                    "total": len(unique_coins),
                    "metadata": {
                        "overrides_applied": len(name_mapping),
                        "cached_at": current_time.isoformat(),
                        "cache_expires": (current_time + COIN_NAMES_CACHE["cache_duration"]).isoformat()
                    }
                }
            }
            
            # Update cache
            COIN_NAMES_CACHE["timestamp"] = current_time
            COIN_NAMES_CACHE["data"] = response_data
            print("💾 Updated coin names cache")
            
            return jsonify(response_data)
            
    except Exception as e:
        print(f"❌ Error in get_coin_names: {str(e)}")
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

def retry_with_delay(max_retries=3, delay_seconds=60):
    """Decorator to retry a function with delay when resources are exhausted"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    error_str = str(e).lower()
                    
                    # Check if error is related to resource exhaustion
                    if any(msg in error_str for msg in ['resource', 'rate limit', 'quota']):
                        if retries < max_retries:
                            print(f"\n⏳ Resource limit hit. Waiting {delay_seconds} seconds before retry {retries}/{max_retries}...")
                            time.sleep(delay_seconds)
                            continue
                    raise e
            return None
        return wrapper
    return decorator

@app.route('/youtube/reason/summary')
def summarize_daily_reasons():
    """Analyze and summarize coin reasons for Feb 13-17"""
    try:
        dates = [
            "2025-04-10",
            "2025-04-11",
        ]
        
        print("\n=== Starting Daily Reason Summary Process ===")
        print(f"Target dates: {dates}")
        
        # Configure Gemini
        api_key = os.getenv('GEMINI_API_KEY_4')
        print(f"\nGemini API Key present: {bool(api_key)}")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name="gemini-2.0-flash")
        
        all_summaries = []
        
        @retry_with_delay(max_retries=3, delay_seconds=60)
        def process_with_gemini(prompt, coin_name):
            """Process prompt with Gemini with retry logic"""
            response = model.generate_content(prompt)
            return response.text.strip()
        
        for date in dates:
            print(f"\n=== Processing Date: {date} ===")
            
            with app.test_client() as client:
                print(f"\nFetching data from /youtube/today-analysis/{date}")
                response = client.get(f'/youtube/today-analysis/{date}')
                print(f"Response status code: {response.status_code}")
                
                if response.status_code != 200:
                    print(f"❌ No data found for {date}")
                    print(f"Response data: {response.get_data(as_text=True)}")
                    continue
                    
                raw_data = response.get_json()
                print(f"\nRaw data keys: {raw_data.keys()}")
                print(f"Data section keys: {raw_data.get('data', {}).keys()}")
                
                if not raw_data.get('data', {}).get('reasons'):
                    print(f"❌ No reasons found for {date}")
                    print(f"Raw data structure: {json.dumps(raw_data, indent=2)}")
                    continue
                
                print(f"\nFound {len(raw_data['data']['reasons'])} reason entries")
                
                # Group reasons by coin
                coin_data = {}
                for reason in raw_data['data']['reasons']:
                    coin_name = reason['coin']
                    if coin_name not in coin_data:
                        coin_data[coin_name] = []
                    coin_data[coin_name].append(reason['reason'])
                
                print(f"\nProcessed coins: {list(coin_data.keys())}")
                print(f"Total unique coins: {len(coin_data)}")

                with sqlite3.connect(DB_PATH) as conn:
                    c = conn.cursor()
                    
                    for coin_name, reasons in coin_data.items():
                        print(f"\n--- Processing {coin_name} for {date} ---")
                        print(f"Total reasons for {coin_name}: {len(reasons)}")
                        print(f"Sample reasons: {reasons[:2]}")  # Show first 2 reasons
                        
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
                            print(f"\nSending prompt to Gemini for {coin_name}")
                            
                            # Use retry logic for Gemini API calls
                            clean_response = process_with_gemini(prompt, coin_name)
                            if not clean_response:
                                print(f"❌ Failed to get response from Gemini for {coin_name} after retries")
                                continue
                            
                            print(f"\nRaw Gemini response for {coin_name}:")
                            print("=" * 50)
                            print(clean_response)
                            print("=" * 50)
                            
                            if clean_response.startswith('```json'):
                                clean_response = clean_response[7:]
                            if clean_response.startswith('```'):
                                clean_response = clean_response[3:]
                            if clean_response.endswith('```'):
                                clean_response = clean_response[:-3]

                            print("\nCleaned response:")
                            print(clean_response)
                            
                            try:
                                analysis = json.loads(clean_response.strip())
                                print(f"\nParsed JSON successfully for {coin_name}")
                                print(f"Number of key points: {len(analysis.get('key_points', []))}")
                            except json.JSONDecodeError as json_err:
                                print(f"❌ JSON parsing error for {coin_name}: {str(json_err)}")
                                print("Problem section:")
                                print(clean_response.strip())
                                continue
                            
                            # Store in database with retry
                            @retry_with_delay(max_retries=3, delay_seconds=30)
                            def store_in_db():
                                c.execute('''
                                    INSERT OR REPLACE INTO reason_summaries 
                                    (analysis_date, coin_name, summary, created_at)
                                    VALUES (?, ?, ?, datetime('now'))
                                ''', (date, coin_name, clean_response.strip()))
                            
                            try:
                                store_in_db()
                                print(f"\n✅ Successfully stored {coin_name} in database")
                            except sqlite3.Error as db_err:
                                print(f"❌ Database error for {coin_name}: {str(db_err)}")
                                continue
                            
                            all_summaries.append({
                                "coin_name": coin_name,
                                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "key_points": analysis.get('key_points', [])
                            })
                            
                            print(f"✅ Successfully processed {coin_name} for {date}")
                            
                        except Exception as e:
                            print(f"❌ Error processing {coin_name} for {date}: {str(e)}")
                            print(f"Error type: {type(e).__name__}")
                            print("Full traceback:")
                            traceback.print_exc()
                            continue
                    
                    conn.commit()
                    print(f"\n✅ Completed storing summaries for {date}")
                    print(f"Total summaries stored: {len(all_summaries)}")

        print("\n=== Process Complete ===")
        print(f"Total summaries generated: {len(all_summaries)}")
        
        return jsonify({
            "status": "success",
            "data": {
                "date": dates[0],
                "summaries": all_summaries,
                "debug_info": {
                    "total_processed": len(all_summaries),
                    "dates_processed": dates,
                    "completion_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            }
        })

    except Exception as e:
        print("\n❌ CRITICAL ERROR in summarize_daily_reasons:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e),
            "error_type": type(e).__name__,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }), 500
    
@app.route('/youtube/reason/search')
def search_coin_reasons():
    """Search for coin reasons by coin name and date with flexible matching"""
    try:
        # Get search parameters
        coin = request.args.get('coin')
        date = request.args.get('date')
        
        if not coin and not date:
            return jsonify({
                "status": "error",
                "message": "Please provide either 'coin' or 'date' parameter"
            }), 400

        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            
            # Get coin name overrides first
            c.execute('''
                SELECT 
                    current_name,
                    new_name
                FROM coin_edits
                ORDER BY edited_at DESC
            ''')
            name_mapping = {row['current_name']: row['new_name'] for row in c.fetchall()}
            
            # Build the query based on provided parameters
            query = '''
                SELECT 
                    analysis_date,
                    coin_name,
                    summary,
                    created_at
                FROM reason_summaries
                WHERE 1=1
            '''
            params = []
            
            if coin:
                # Case-insensitive search with partial matching
                query += ''' 
                    AND (
                        LOWER(coin_name) LIKE LOWER(?)
                        OR LOWER(coin_name) LIKE LOWER(?)
                    )
                '''
                params.extend([f'%{coin}%', f'%{name_mapping.get(coin, coin)}%'])
            
            if date:
                query += ' AND analysis_date = ?'
                params.append(date)
            
            query += ' ORDER BY analysis_date DESC, coin_name'
            
            c.execute(query, params)
            rows = c.fetchall()
            
            if not rows:
                return jsonify({
                    "status": "success",
                    "message": "No matching records found",
                    "data": {
                        "search_params": {
                            "coin": coin,
                            "date": date
                        },
                        "results": []
                    }
                })
            
            # Process results
            results = []
            for row in rows:
                try:
                    summary_data = json.loads(row['summary'])
                    
                    # Apply name override if exists
                    coin_name = row['coin_name']
                    if coin_name in name_mapping:
                        coin_name = name_mapping[coin_name]
                    
                    results.append({
                        "date": row['analysis_date'],
                        "coin": coin_name,
                        "key_points": summary_data.get('key_points', []),
                        "created_at": row['created_at'],
                        "sentiment_summary": {
                            "bullish": len([p for p in summary_data.get('key_points', []) 
                                          if p.get('sentiment') == 'BULLISH']),
                            "bearish": len([p for p in summary_data.get('key_points', []) 
                                          if p.get('sentiment') == 'BEARISH']),
                            "neutral": len([p for p in summary_data.get('key_points', []) 
                                          if p.get('sentiment') == 'NEUTRAL'])
                        }
                    })
                except json.JSONDecodeError:
                    print(f"Error parsing summary for {row['coin_name']} on {row['analysis_date']}")
                    continue
            
            # Add summary statistics
            total_points = sum(len(r['key_points']) for r in results)
            total_bullish = sum(r['sentiment_summary']['bullish'] for r in results)
            total_bearish = sum(r['sentiment_summary']['bearish'] for r in results)
            total_neutral = sum(r['sentiment_summary']['neutral'] for r in results)
            
            return jsonify({
                "status": "success",
                "data": {
                    "search_params": {
                        "coin": coin,
                        "date": date
                    },
                    "summary": {
                        "total_records": len(results),
                        "unique_coins": len(set(r['coin'] for r in results)),
                        "date_range": {
                            "earliest": min(r['date'] for r in results),
                            "latest": max(r['date'] for r in results)
                        },
                        "total_points": total_points,
                        "sentiment_distribution": {
                            "bullish": total_bullish,
                            "bearish": total_bearish,
                            "neutral": total_neutral,
                            "overall_sentiment": "BULLISH" if total_bullish > total_bearish else 
                                               "BEARISH" if total_bearish > total_bullish else 
                                               "NEUTRAL"
                        }
                    },
                    "results": results
                }
            })

    except Exception as e:
        print(f"Error in search_coin_reasons: {str(e)}")
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
        1. Instead of giving direct answers, provide 4 clarifying questions
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

        Provide 4 clarifying questions

        Provide respond according to the language of the user if its english or urdu or arabic or malay. The question is not only about zakat but also about other fiqh topics.

        Current user question\n: {user_question}
        """
        
        final_prompt = system_prompt.format(user_question=prompt)
        response = model.generate_content(final_prompt)
        return jsonify({"clarifying_questions": response.text})
        
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


@app.route('/draft', methods=['POST'])
def save_draft():
    """Save template editor draft"""
    try:
        data = request.get_json()
        
        # Create drafts table if it doesn't exist
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            
            
            # Create table with correct schema
            c.execute('''
                CREATE TABLE IF NOT EXISTS template_drafts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    template_name TEXT,
                    description TEXT,
                    coin_symbol TEXT,
                    historical_period INTEGER,
                    analysis_date TEXT,
                    gemini_model TEXT,
                    requirements TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Insert new draft
            c.execute('''
                INSERT INTO template_drafts 
                (template_name, description, coin_symbol, historical_period, 
                 analysis_date, gemini_model, requirements)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                data.get('template_name'),
                data.get('description'),
                data.get('coin_symbol'),
                data.get('historical_period'),
                data.get('analysis_date'),
                data.get('gemini_model'),
                json.dumps(data.get('requirements', []))
            ))
            
            draft_id = c.lastrowid
            
            return jsonify({
                "status": "success",
                "message": "Draft saved successfully",
                "data": {
                    "draft_id": draft_id
                }
            })
            
    except Exception as e:
        print(f"Error saving draft: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/draft', methods=['GET'])
def get_drafts():
    """Get all template editor drafts"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            
            c.execute('''
                SELECT 
                    id,
                    template_name,
                    description,
                    coin_symbol,
                    historical_period,
                    analysis_date,
                    gemini_model,
                    requirements,
                    datetime(created_at) as created_at,
                    datetime(updated_at) as updated_at
                FROM template_drafts 
                ORDER BY updated_at DESC
            ''')
            
            drafts = []
            for row in c.fetchall():
                draft = dict(row)
                draft['requirements'] = json.loads(draft['requirements'])
                drafts.append(draft)
            
            return jsonify({
                "status": "success",
                "data": drafts
            })
            
    except Exception as e:
        print(f"Error getting drafts: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/draft/<int:draft_id>', methods=['GET'])
def get_draft(draft_id):
    """Get specific draft by ID"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            
            c.execute('''
                SELECT 
                    id,
                    template_name,
                    description,
                    coin_symbol,
                    historical_period,
                    analysis_date,
                    gemini_model,
                    requirements,
                    datetime(created_at) as created_at,
                    datetime(updated_at) as updated_at
                FROM template_drafts 
                WHERE id = ?
            ''', (draft_id,))
            
            result = c.fetchone()
            
            if result:
                draft = dict(result)
                draft['requirements'] = json.loads(draft['requirements'])
                return jsonify({
                    "status": "success",
                    "data": draft
                })
            
            return jsonify({
                "status": "error",
                "message": "Draft not found"
            }), 404
            
    except Exception as e:
        print(f"Error getting draft: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/draft/<int:draft_id>', methods=['DELETE'])
def delete_draft(draft_id):
    """Delete a specific draft by ID"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            
            # Delete the draft
            c.execute('DELETE FROM template_drafts WHERE id = ?', (draft_id,))
            
            if c.rowcount == 0:
                return jsonify({
                    "status": "error",
                    "message": f"Draft with ID {draft_id} not found"
                }), 404
            
            conn.commit()
            
            return jsonify({
                "status": "success",
                "message": f"Draft {draft_id} deleted successfully"
            })
            
    except Exception as e:
        print(f"Error deleting draft: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/draft/run', methods=['POST'])
def run_draft():
    """Run analysis based on provided template data"""
    try:
        print("\n🚀 Starting draft/run analysis...")
        
        data = request.get_json()
        print(f"📥 Received request data: {json.dumps(data, indent=2)}")
        
        # Get historical data for the selected coin
        try:
            print(f"\n🔍 Parsing analysis date: {data.get('analysis_date')}")
            current_date = datetime.strptime(data['analysis_date'], '%B %dth, %Y')
            past_date = current_date - timedelta(days=data['historical_period'])
            
            print(f"📅 Analysis period:")
            print(f"  - From: {past_date.strftime('%Y-%m-%d')}")
            print(f"  - To: {current_date.strftime('%Y-%m-%d')}")
            print(f"  - Duration: {data['historical_period']} days")
            
            with sqlite3.connect(DB_PATH) as conn:
                print("\n🗃️ Connected to database")
                conn.row_factory = sqlite3.Row
                c = conn.cursor()
                
                print(f"🔎 Querying historical data for {data['coin_symbol']}")
                c.execute('''
                    SELECT analysis_date, summary
                    FROM reason_summaries
                    WHERE analysis_date BETWEEN ? AND ?
                    AND LOWER(coin_name) = LOWER(?)
                    ORDER BY analysis_date DESC
                ''', (past_date.strftime('%Y-%m-%d'), 
                      current_date.strftime('%Y-%m-%d'), 
                      data['coin_symbol']))
                
                historical_rows = c.fetchall()
                print(f"✨ Found {len(historical_rows)} historical summaries")
                
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
            print(f"\n❌ Database error:")
            print(f"Error type: {type(db_error).__name__}")
            print(f"Error details: {str(db_error)}")
            traceback.print_exc()
            return jsonify({
                "status": "error",
                "message": f"Database error: {str(db_error)}"
            }), 500

        # Get price data
        try:
            print(f"\n💰 Fetching price data for {data['coin_symbol']}")
            price_data = price_service.get_historical_prices(data['coin_symbol'].lower())
            
            formatted_price_data = {
                "symbol": price_data["symbol"],
                "prices": [
                    {
                        "date": price.date,
                        "price": price.price
                    } for price in price_data["prices"]
                ]
            }
            
            current_price = price_data['prices'][-1].price if price_data['prices'] else 0
            previous_price = price_data['prices'][-2].price if len(price_data['prices']) > 1 else current_price
            price_change = ((current_price - previous_price) / previous_price) * 100
            
            print(f"📊 Got {len(price_data['prices'])} price points")
            
        except Exception as price_error:
            print(f"\n❌ Price data error:")
            print(f"Error type: {type(price_error).__name__}")
            print(f"Error details: {str(price_error)}")
            traceback.print_exc()
            return jsonify({
                "status": "error",
                "message": f"Price data error: {str(price_error)}"
            }), 500

        # Configure Gemini
        try:
            print("\n🤖 Configuring Gemini model...")
            api_key = "AIzaSyBVaovh2Cz9LU7gUJ_ft00UBEv26_vaaC0"
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name=data['gemini_model'])
            print(f"✅ Configured model: {data['gemini_model']}")
            
        except Exception as gemini_error:
            print(f"\n❌ Gemini configuration error:")
            print(f"Error type: {type(gemini_error).__name__}")
            print(f"Error details: {str(gemini_error)}")
            traceback.print_exc()
            return jsonify({
                "status": "error",
                "message": f"Gemini configuration error: {str(gemini_error)}"
            }), 500

        # Build the prompt
        try:
            print("\n📝 Building analysis prompt...")
            requirements_text = "\n".join([f"{i+1}. {req}" for i, req in enumerate(data['requirements'])])
            
            # Format timeline using target dates
            timeline_format = f"[{data.get('target_start')} to {data.get('target_end')}]"
            
            prompt = f"""Analyze {data['coin_symbol']} market conditions and provide a detailed report matching the exact format below.

ANALYSIS REQUIREMENTS:

{requirements_text}

Current Market Data:
Price: ${current_price:.2f}
24h Change: {price_change:.1f}%

REQUIRED OUTPUT FORMAT:
{{
  "symbol": "{data['coin_symbol']}",
  "name": "{data['template_name']}",
  "price": "{current_price:.2f}",
  "priceChange": {price_change:.1f},
  "targetPrice": "[Target Price]",
  "sentiment": "[Bullish/Neutral/Bearish]",
  "sentimentScore": [1-10],
  "market": "[Single detailed paragraph combining key market insights]",
  "prediction": {{
    "direction": "[Clear trend statement]",
    "reasoning": [
      "1. [Network insight]",
      "2. [Technical insight]",
      "3. [Development insight]",
      "4. [Timeline insight]"
    ]
  }},
  "timeline": "{timeline_format} but strictly convert into this date format: DD-name of month-year",
  "confidence": [percentage],
  "status": "Complete"
}}

CRITICAL REQUIREMENTS:
1. All analysis must be data-driven with specific numbers
2. Market analysis must be one detailed paragraph
3. Prediction points must be specific and quantifiable
4. Maintain exact UI format
5. Target price must have technical and fundamental basis
6. Timeline must consider known upcoming events within the specified date range
7. Confidence score must reflect data reliability
8. All monetary values in USD

Historical Analysis Data:
{json.dumps(historical_data, indent=2)}

Price Data:
{json.dumps(formatted_price_data, indent=2)}
"""
            print("✅ Prompt built successfully")

        except Exception as prompt_error:
            print(f"\n❌ Prompt building error:")
            print(f"Error type: {type(prompt_error).__name__}")
            print(f"Error details: {str(prompt_error)}")
            traceback.print_exc()
            return jsonify({
                "status": "error",
                "message": f"Prompt building error: {str(prompt_error)}"
            }), 500

        # Generate analysis
        try:
            print("\n🤖 Sending to Gemini for analysis...")
            response = model.generate_content(prompt)
            
            print("\n📄 Raw Gemini response:")
            print(response.text)
            
            # Clean and parse response
            clean_response = response.text.strip()
            json_start = clean_response.find('{')
            json_end = clean_response.rfind('}') + 1
            json_content = clean_response[json_start:json_end].strip()
            
            print("\n🔍 Extracted JSON content:")
            print(json_content)
            
            analysis = json.loads(json_content)
            print("\n✅ Successfully parsed analysis")
            
            return jsonify({
                "status": "success",
                "data": analysis
            })
            
        except Exception as analysis_error:
            print(f"\n❌ Analysis generation error:")
            print(f"Error type: {type(analysis_error).__name__}")
            print(f"Error details: {str(analysis_error)}")
            traceback.print_exc()
            return jsonify({
                "status": "error",
                "message": f"Analysis generation error: {str(analysis_error)}"
            }), 500
            
    except Exception as e:
        print(f"\n❌ Unexpected error in run_draft:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error details: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/template', methods=['POST'])
def save_template():
    """Save analysis template"""
    try:
        data = request.get_json()
        
        # Create templates table if it doesn't exist
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            
            # Drop existing table to reset schema
            
            # Create table with same structure as drafts
            c.execute('''
                CREATE TABLE IF NOT EXISTS analysis_templates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    template_name TEXT,
                    description TEXT,
                    coin_symbol TEXT,
                    historical_period INTEGER,
                    analysis_date TEXT,
                    gemini_model TEXT,
                    requirements TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Insert new template
            c.execute('''
                INSERT INTO analysis_templates 
                (template_name, description, coin_symbol, historical_period, 
                 analysis_date, gemini_model, requirements)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                data.get('template_name'),
                data.get('description'),
                data.get('coin_symbol'),
                data.get('historical_period'),
                data.get('analysis_date'),
                data.get('gemini_model'),
                json.dumps(data.get('requirements', []))
            ))
            
            template_id = c.lastrowid
            
            return jsonify({
                "status": "success",
                "message": "Template saved successfully",
                "data": {
                    "template_id": template_id
                }
            })
            
    except Exception as e:
        print(f"Error saving template: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/template', methods=['GET'])
def get_templates():
    """Get all analysis templates"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            
            c.execute('''
                SELECT 
                    id,
                    template_name,
                    description,
                    coin_symbol,
                    historical_period,
                    analysis_date,
                    gemini_model,
                    requirements,
                    datetime(created_at) as created_at,
                    datetime(updated_at) as updated_at
                FROM analysis_templates 
                ORDER BY updated_at DESC
            ''')
            
            templates = []
            for row in c.fetchall():
                template = dict(row)
                template['requirements'] = json.loads(template['requirements'])
                templates.append(template)
            
            return jsonify({
                "status": "success",
                "data": templates
            })
            
    except Exception as e:
        print(f"Error getting templates: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/template/<int:template_id>', methods=['DELETE'])
def delete_template(template_id):
    """Delete a specific template by ID"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            
            c.execute('DELETE FROM analysis_templates WHERE id = ?', (template_id,))
            
            if c.rowcount == 0:
                return jsonify({
                    "status": "error",
                    "message": f"Template with ID {template_id} not found"
                }), 404
            
            return jsonify({
                "status": "success",
                "message": f"Template {template_id} deleted successfully"
            })
            
    except Exception as e:
        print(f"Error deleting template: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/analysis-result', methods=['POST'])
def save_analysis_result():
    """Save analysis result"""
    try:
        data = request.get_json()
        
        # Create analysis_results table if it doesn't exist
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            
            # Create table if not exists
            c.execute('''
                CREATE TABLE IF NOT EXISTS analysis_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    name TEXT,
                    price TEXT,
                    price_change REAL,
                    target_price TEXT,
                    sentiment TEXT,
                    sentiment_score INTEGER,
                    market TEXT,
                    prediction_direction TEXT,
                    prediction_reasoning TEXT,
                    timeline TEXT,
                    confidence INTEGER,
                    status TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Insert new analysis result
            c.execute('''
                INSERT INTO analysis_results 
                (symbol, name, price, price_change, target_price, sentiment, 
                 sentiment_score, market, prediction_direction, prediction_reasoning, 
                 timeline, confidence, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data.get('symbol'),
                data.get('name'),
                data.get('price'),
                data.get('priceChange'),
                data.get('targetPrice'),
                data.get('sentiment'),
                data.get('sentimentScore'),
                data.get('market'),
                data.get('prediction', {}).get('direction'),
                json.dumps(data.get('prediction', {}).get('reasoning', [])),
                data.get('timeline'),
                data.get('confidence'),
                data.get('status')
            ))
            
            result_id = c.lastrowid
            
            print(f"✅ Saved analysis result with ID: {result_id}")
            
            return jsonify({
                "status": "success",
                "message": "Analysis result saved successfully",
                "data": {
                    "result_id": result_id
                }
            })
            
    except Exception as e:
        print(f"❌ Error saving analysis result: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/analysis-result', methods=['GET'])
def get_analysis_results():
    """Get all analysis results"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            
            c.execute('''
                SELECT 
                    id,
                    symbol,
                    name,
                    price,
                    price_change,
                    target_price,
                    sentiment,
                    sentiment_score,
                    market,
                    prediction_direction,
                    prediction_reasoning,
                    timeline,
                    confidence,
                    status,
                    datetime(created_at) as created_at
                FROM analysis_results 
                ORDER BY created_at DESC
            ''')
            
            results = []
            for row in c.fetchall():
                result = dict(row)
                
                # Format the result to match the expected structure
                formatted_result = {
                    "id": result['id'],
                    "symbol": result['symbol'],
                    "name": result['name'],
                    "price": result['price'],
                    "priceChange": result['price_change'],
                    "targetPrice": result['target_price'],
                    "sentiment": result['sentiment'],
                    "sentimentScore": result['sentiment_score'],
                    "market": result['market'],
                    "prediction": {
                        "direction": result['prediction_direction'],
                        "reasoning": json.loads(result['prediction_reasoning'])
                    },
                    "timeline": result['timeline'],
                    "confidence": result['confidence'],
                    "status": result['status'],
                    "timestamp": result['created_at']
                }
                
                results.append(formatted_result)
            
            return jsonify({
                "status": "success",
                "data": results
            })
            
    except Exception as e:
        print(f"❌ Error getting analysis results: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/analysis-result/<int:result_id>', methods=['DELETE'])
def delete_analysis_result(result_id):
    """Delete a specific analysis result by ID"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            
            c.execute('DELETE FROM analysis_results WHERE id = ?', (result_id,))
            
            if c.rowcount == 0:
                return jsonify({
                    "status": "error",
                    "message": f"Analysis result with ID {result_id} not found"
                }), 404
            
            return jsonify({
                "status": "success",
                "message": f"Analysis result {result_id} deleted successfully"
            })
            
    except Exception as e:
        print(f"❌ Error deleting analysis result: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
    
@app.route('/publish', methods=['POST'])
def publish_analysis():
    """Publish analysis result"""
    try:
        data = request.get_json()
        
        # Validate category
        category = data.get('category', '').lower()
        if category not in ['large cap', 'low cap']:
            return jsonify({
                "status": "error",
                "message": "Category must be either 'large cap' or 'low cap'"
            }), 400
        
        # Create or update published_analyses table
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            
            # Create table if not exists
            c.execute('''
                CREATE TABLE IF NOT EXISTS published_analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    name TEXT,
                    price TEXT,
                    price_change REAL,
                    target_price TEXT,
                    sentiment TEXT,
                    sentiment_score INTEGER,
                    market TEXT,
                    prediction_direction TEXT,
                    prediction_reasoning TEXT,
                    timeline TEXT,
                    confidence INTEGER,
                    status TEXT,
                    published_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Check if category column exists, if not add it
            try:
                c.execute("SELECT category FROM published_analyses LIMIT 1")
            except sqlite3.OperationalError:
                print("Adding category column to published_analyses table")
                c.execute("ALTER TABLE published_analyses ADD COLUMN category TEXT")
            
            # Insert new published analysis
            c.execute('''
                INSERT INTO published_analyses 
                (symbol, name, price, price_change, target_price, sentiment, 
                 sentiment_score, market, prediction_direction, prediction_reasoning, 
                 timeline, confidence, status, category)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data.get('symbol'),
                data.get('name'),
                data.get('price'),
                data.get('priceChange'),
                data.get('targetPrice'),
                data.get('sentiment'),
                data.get('sentimentScore'),
                data.get('market'),
                data.get('prediction', {}).get('direction'),
                json.dumps(data.get('prediction', {}).get('reasoning', [])),
                data.get('timeline'),
                data.get('confidence'),
                data.get('status'),
                category
            ))
            
            publish_id = c.lastrowid
            
            print(f"✅ Published analysis with ID: {publish_id}")
            
            return jsonify({
                "status": "success",
                "message": "Analysis published successfully",
                "data": {
                    "publish_id": publish_id
                }
            })
            
    except Exception as e:
        print(f"❌ Error publishing analysis: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/publish', methods=['GET'])
def get_published_analyses():
    """Get all published analyses"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            
            # Check if category column exists, if not add it
            try:
                c.execute("SELECT category FROM published_analyses LIMIT 1")
            except sqlite3.OperationalError:
                print("Adding category column to published_analyses table")
                c.execute("ALTER TABLE published_analyses ADD COLUMN category TEXT")
            
            c.execute('''
                SELECT 
                    id,
                    symbol,
                    name,
                    price,
                    price_change,
                    target_price,
                    sentiment,
                    sentiment_score,
                    market,
                    prediction_direction,
                    prediction_reasoning,
                    timeline,
                    confidence,
                    status,
                    category,
                    datetime(published_at) as published_at
                FROM published_analyses 
                ORDER BY published_at DESC
            ''')
            
            results = []
            for row in c.fetchall():
                result = dict(row)
                
                # Format the result to match the expected structure
                formatted_result = {
                    "id": result['id'],
                    "symbol": result['symbol'],
                    "name": result['name'],
                    "price": result['price'],
                    "priceChange": result['price_change'],
                    "targetPrice": result['target_price'],
                    "sentiment": result['sentiment'],
                    "sentimentScore": result['sentiment_score'],
                    "market": result['market'],
                    "prediction": {
                        "direction": result['prediction_direction'],
                        "reasoning": json.loads(result['prediction_reasoning'])
                    },
                    "timeline": result['timeline'],
                    "confidence": result['confidence'],
                    "status": result['status'],
                    "category": result['category'] or "large cap",  # Default to large cap if null
                    "timestamp": result['published_at']
                }
                
                results.append(formatted_result)
            
            return jsonify({
                "status": "success",
                "data": results
            })
            
    except Exception as e:
        print(f"❌ Error getting published analyses: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/publish/<int:publish_id>', methods=['GET'])
def get_published_analysis(publish_id):
    """Get a specific published analysis by ID"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            
            # Check if category column exists, if not add it
            try:
                c.execute("SELECT category FROM published_analyses LIMIT 1")
            except sqlite3.OperationalError:
                print("Adding category column to published_analyses table")
                c.execute("ALTER TABLE published_analyses ADD COLUMN category TEXT")
            
            c.execute('''
                SELECT 
                    id,
                    symbol,
                    name,
                    price,
                    price_change,
                    target_price,
                    sentiment,
                    sentiment_score,
                    market,
                    prediction_direction,
                    prediction_reasoning,
                    timeline,
                    confidence,
                    status,
                    category,
                    datetime(published_at) as published_at
                FROM published_analyses 
                WHERE id = ?
            ''', (publish_id,))
            
            row = c.fetchone()
            
            if not row:
                return jsonify({
                    "status": "error",
                    "message": f"Published analysis with ID {publish_id} not found"
                }), 404
            
            result = dict(row)
            
            # Format the result to match the expected structure
            formatted_result = {
                "id": result['id'],
                "symbol": result['symbol'],
                "name": result['name'],
                "price": result['price'],
                "priceChange": result['price_change'],
                "targetPrice": result['target_price'],
                "sentiment": result['sentiment'],
                "sentimentScore": result['sentiment_score'],
                "market": result['market'],
                "prediction": {
                    "direction": result['prediction_direction'],
                    "reasoning": json.loads(result['prediction_reasoning'])
                },
                "timeline": result['timeline'],
                "confidence": result['confidence'],
                "status": result['status'],
                "category": result['category'] or "large cap",  # Default to large cap if null
                "timestamp": result['published_at']
            }
            
            return jsonify({
                "status": "success",
                "data": formatted_result
            })
            
    except Exception as e:
        print(f"❌ Error getting published analysis: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route('/publish/<int:publish_id>', methods=['DELETE'])
def delete_published_analysis(publish_id):
    """Delete a specific published analysis by ID"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            
            # First check if the analysis exists
            c.execute('SELECT id FROM published_analyses WHERE id = ?', (publish_id,))
            if not c.fetchone():
                return jsonify({
                    "status": "error",
                    "message": f"Published analysis with ID {publish_id} not found"
                }), 404
            
            # Delete the analysis
            c.execute('DELETE FROM published_analyses WHERE id = ?', (publish_id,))
            
            if c.rowcount > 0:
                print(f"✅ Deleted published analysis with ID: {publish_id}")
                return jsonify({
                    "status": "success",
                    "message": f"Published analysis with ID {publish_id} deleted successfully"
                })
            else:
                return jsonify({
                    "status": "error",
                    "message": f"Failed to delete published analysis with ID {publish_id}"
                }), 500
            
    except Exception as e:
        print(f"❌ Error deleting published analysis: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/v1/market-analysis/<date>')
def get_daily_market_analysis(date):
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
                        v.video_id,
                        ca.coin_mentioned,
                        ca.reasons,
                        ca.indicator
                    FROM videos v
                    LEFT JOIN coin_analysis ca ON v.video_id = ca.video_id
                    WHERE v.published_at BETWEEN ? AND ?
                    AND ca.coin_mentioned IS NOT NULL
                )
                SELECT * FROM video_analyses
            ''', (start_utc, end_utc))
            
            rows = c.fetchall()
            
            if not rows:
                return jsonify({
                    "status": "success",
                    "data": {
                        "date": date,
                        "coins": []
                    }
                })
            
            # Group analyses by coin
            coin_analyses = {}
            
            for row in rows:
                try:
                    parsed_reasons = json.loads(row['reasons'])
                    
                    # Apply name override if exists
                    coin_name = row['coin_mentioned']
                    if coin_name in name_mapping:
                        coin_name = name_mapping[coin_name]
                    
                    # Initialize coin entry if it doesn't exist
                    if coin_name not in coin_analyses:
                        coin_analyses[coin_name] = []
                    
                    # Add each reason with its sentiment
                    for reason in parsed_reasons:
                        coin_analyses[coin_name].append({
                            "reason": reason,
                            "sentiment": row['indicator']
                        })
                        
                except Exception as e:
                    print(f"Error parsing reasons for video {row['video_id']}: {str(e)}")
                    continue
            
            # Convert dictionary to list for response
            coins_list = []
            for coin_name, analyses in coin_analyses.items():
                coins_list.append({
                    "coin": coin_name,
                    "analyses": analyses
                })
            
            return jsonify({
                "status": "success",
                "data": {
                    "date": date,
                    "coins": coins_list
                }
            })
            
    except Exception as e:
        print(f"Error in get_daily_market_analysis: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
    
@app.route('/api/v1/coin-trends/<date>')
def get_coin_trends_by_date(date):
    """Get simplified coin sentiment counts for a specific date (Malaysia timezone)"""
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
                        "coins": []
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
                        "coin": coin_name,
                        "sentiment": {
                            "bullish": 0,
                            "bearish": 0,
                            "neutral": 0
                        }
                    }
                
                # Track sentiment counts
                indicator = mention["indicator"].lower()
                if "bullish" in indicator:
                    coin_stats[coin_name]["sentiment"]["bullish"] += 1
                elif "bearish" in indicator:
                    coin_stats[coin_name]["sentiment"]["bearish"] += 1
                else:
                    coin_stats[coin_name]["sentiment"]["neutral"] += 1

            # Convert to list and sort by total mentions
            coins_list = list(coin_stats.values())
            
            # Sort by total mentions (sum of all sentiment counts)
            coins_list.sort(
                key=lambda x: (
                    x["sentiment"]["bullish"] + 
                    x["sentiment"]["bearish"] + 
                    x["sentiment"]["neutral"]
                ), 
                reverse=True
            )

            return jsonify({
                "status": "success",
                "data": {
                    "date": date,
                    "coins": coins_list
                }
            })
            
    except Exception as e:
        print(f"Error in get_coin_trends_by_date: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/v1/market-prediction', methods=['GET'])
def get_market_predictions():
    """Get simplified market predictions with specific fields only"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            
            # Check if category column exists, if not add it
            try:
                c.execute("SELECT category FROM published_analyses LIMIT 1")
            except sqlite3.OperationalError:
                print("Adding category column to published_analyses table")
                c.execute("ALTER TABLE published_analyses ADD COLUMN category TEXT")
            
            c.execute('''
                SELECT 
                    symbol,
                    target_price,
                    sentiment,
                    sentiment_score,
                    market,
                    prediction_direction,
                    prediction_reasoning,
                    timeline,
                    confidence,
                    status,
                    category,
                    datetime(published_at) as published_at
                FROM published_analyses 
                ORDER BY published_at DESC
            ''')
            
            results = []
            for row in c.fetchall():
                result = dict(row)
                
                # Format the result with only the requested fields
                formatted_result = {
                    "symbol": result['symbol'],
                    "targetPrice": result['target_price'],
                    "sentiment": result['sentiment'],
                    "sentimentScore": result['sentiment_score'],
                    "market": result['market'],
                    "prediction": {
                        "direction": result['prediction_direction'],
                        "reasoning": json.loads(result['prediction_reasoning'])
                    },
                    "timeline": result['timeline'],
                    "confidence": result['confidence'],
                    "status": result['status'],
                    "category": result['category'] or "large cap",  # Default to large cap if null
                    "timestamp": result['published_at']
                }
                
                results.append(formatted_result)
            
            return jsonify({
                "status": "success",
                "data": results
            })
            
    except Exception as e:
        print(f"❌ Error getting market predictions: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/publish/delete-all', methods=['DELETE'])
def delete_all_published():
    """Delete all published analyses by dropping the table"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            
            # Drop the table completely
            c.execute('DROP TABLE IF EXISTS published_analyses')
            conn.commit()
            
            return jsonify({
                "status": "success",
                "message": "Successfully dropped published analyses table"
            })
            
    except Exception as e:
        print(f"❌ Error dropping published analyses table: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
    
@app.route('/publish/<int:publish_id>/status', methods=['PUT'])
def update_publish_status(publish_id):
    """Update the status of a published analysis"""
    try:
        data = request.get_json()
        new_status = data.get('status')
        
        if not new_status:
            return jsonify({
                "status": "error",
                "message": "Status is required"
            }), 400

        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute('''
                UPDATE published_analyses 
                SET status = ? 
                WHERE id = ?
            ''', (new_status, publish_id))
            
            if c.rowcount == 0:
                return jsonify({
                    "status": "error",
                    "message": "Published analysis not found"
                }), 404

            return jsonify({
                "status": "success",
                "message": "Status updated successfully"
            })

    except Exception as e:
        print(f"❌ Error updating publish status: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
    
@app.route('/youtube/reason/summary/q1/<symbol>')
def get_coin_q1_summary(symbol):
    """Retrieve all stored summaries for a specific coin during Q1 2025"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            
            # Query for Q1 2025 (January 1st to March 31st)
            c.execute('''
                SELECT analysis_date, coin_name, summary, created_at
                FROM reason_summaries
                WHERE coin_name = ?
                AND analysis_date >= '2025-01-01' 
                AND analysis_date <= '2025-03-31'
                ORDER BY analysis_date
            ''', (symbol.upper(),))
            
            rows = c.fetchall()
            
            if rows:
                summaries = []
                for row in rows:
                    summary_data = {
                        "date": row['analysis_date'],
                        "key_points": json.loads(row['summary'])['key_points'],
                        "created_at": row['created_at']
                    }
                    summaries.append(summary_data)
                
                return jsonify({
                    "status": "success",
                    "data": {
                        "coin": symbol.upper(),
                        "period": "Q1 2025",
                        "total_days_with_mentions": len(summaries),
                        "summaries": summaries
                    }
                })
            else:
                return jsonify({
                    "status": "error",
                    "message": f"No summaries found for {symbol.upper()} in Q1 2025"
                }), 404
                
    except Exception as e:
        print(f"Error retrieving Q1 summaries: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/youtube/process-news/<symbol>')
def process_coin_news(symbol):
    """Process Q1 2025 data for a specific coin and generate comprehensive insights in Messari-style format"""
    try:
        print(f"\nStarting comprehensive analysis for {symbol}")
        
        # Get all stored summaries for this coin in Q1 2025
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            
            print(f"\nFetching data for {symbol.upper()} from Q1 2025 (Jan 1 - Mar 31)...")
            
            # Query for Q1 2025 (January 1st to March 31st)
            c.execute('''
                SELECT analysis_date, coin_name, summary, created_at
                FROM reason_summaries
                WHERE coin_name = ?
                AND analysis_date >= '2025-01-01' 
                AND analysis_date <= '2025-03-31'
                ORDER BY analysis_date
            ''', (symbol.upper(),))
            
            rows = c.fetchall()
            
            if not rows:
                print(f"No data found for {symbol.upper()} in Q1 2025")
                return jsonify({
                    "status": "error",
                    "message": f"No data found for {symbol.upper()} in Q1 2025"
                }), 404
            
            print(f"\nFound {len(rows)} days of data for {symbol.upper()}")
            print(f"Date range: {rows[0]['analysis_date']} to {rows[-1]['analysis_date']}")
            
            # Prepare data for analysis
            summaries = []
            for row in rows:
                summary_data = {
                    "date": row['analysis_date'],
                    "key_points": json.loads(row['summary'])['key_points'],
                    "created_at": row['created_at']
                }
                summaries.append(summary_data)
            
            print(f"\nTotal key points collected: {sum(len(s['key_points']) for s in summaries)}")
            print(f"Average key points per day: {sum(len(s['key_points']) for s in summaries) / len(summaries):.1f}")
            
            # Configure Gemini
            api_key = "AIzaSyBdMrLHWWswcUjJABblf2E1n2s1TZqYu9w"
            if not api_key:
                print("Error: GEMINI_API_KEY_1 not found in environment variables")
                return jsonify({
                    "status": "error",
                    "message": "GEMINI_API_KEY_1 not found in environment variables"
                }), 500
                
            print("\nConfiguring Gemini model...")
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name="gemini-2.0-flash")
            
            prompt = f"""Analyze this Q1 2025 data for {symbol.upper()} and create a detailed market analysis article in the style of Messari.

CRITICAL RULES:
1. Write in professional financial journalism style (like Messari, Bloomberg, or CoinDesk)
2. Focus on concrete data points and specific numbers
3. Provide percentage-based insights where possible
4. Structure the article with clear sections and transitions
5. Include both technical and fundamental analysis
6. Highlight key trends and patterns
7. Maintain objectivity while providing clear insights
8. Include specific dates and timeframes for all data points
9. Write in a narrative, engaging style that flows naturally

Required Article Structure:
{{
    "overall_title": "Main title for the entire analysis (e.g., 'Q1 2025 Market Analysis: {symbol.upper()} Performance and Outlook')",
    "article": {{
        "key_metrics": {{
            "price": "Current price with 24h change",
            "market_cap": "Current market cap",
            "volume_24h": "24-hour trading volume",
            "dominance": "Market dominance percentage",
            "volatility": "Current volatility metrics"
        }},
        "title": "Engaging, data-driven title that captures the main insight",
        "subtitle": "Brief subtitle providing context",
        "sections": [
            {{
                "title": "Executive Summary",
                "content": "2-3 paragraphs providing key takeaways and overall market position"
            }},
            {{
                "title": "Market Overview",
                "content": "Detailed analysis of current market conditions with specific data points"
            }},
            {{
                "title": "Key Developments",
                "content": "Chronological analysis of major events and their market impact"
            }},
            {{
                "title": "Technical Analysis",
                "content": "In-depth technical analysis with specific levels and indicators"
            }},
            {{
                "title": "On-Chain Metrics",
                "content": "Analysis of relevant on-chain data and network activity"
            }},
            {{
                "title": "Market Sentiment",
                "content": "Analysis of market sentiment with specific data points and trends"
            }},
            {{
                "title": "Risk Factors",
                "content": "Comprehensive analysis of potential risks and their implications"
            }},
            {{
                "title": "Future Outlook",
                "content": "Forward-looking analysis with specific predictions and confidence levels"
            }},
            {{
                "title": "Key Takeaways",
                "content": "Bullet points summarizing the most important insights"
            }}
        ]
    }}
}}

Analyze this data:
{json.dumps(summaries, indent=2)}"""

            print("\nSending to Gemini for analysis...")
            response = model.generate_content(prompt)
            
            print("\n=== Raw Response from Gemini ===")
            print(response.text)
            
            # Clean response - handle various response formats
            clean_response = response.text.strip()
            print("\n=== After Initial Strip ===")
            print(clean_response)
            
            # Remove any markdown code block markers
            if clean_response.startswith('```json'):
                clean_response = clean_response[7:]
                print("\n=== After Removing ```json ===")
                print(clean_response)
            elif clean_response.startswith('```'):
                clean_response = clean_response[3:]
                print("\n=== After Removing ``` ===")
                print(clean_response)
            
            if clean_response.endswith('```'):
                clean_response = clean_response[:-3]
                print("\n=== After Removing Ending ``` ===")
                print(clean_response)
            
            # Remove any leading/trailing whitespace and newlines
            clean_response = clean_response.strip()
            print("\n=== After Final Strip ===")
            print(clean_response)
            
            try:
                # Try to parse the JSON
                print("\n=== Attempting First JSON Parse ===")
                analysis = json.loads(clean_response)
                print(f"\nGenerated overall title: {analysis.get('overall_title', 'No title generated')}")
            except json.JSONDecodeError as e:
                print(f"\n=== First JSON Parse Failed ===")
                print(f"Error: {str(e)}")
                print("Error location:", e.pos)
                print("Error line:", e.lineno)
                print("Error column:", e.colno)
                print("Raw response:", clean_response)
                
                # Enhanced cleaning of the response
                try:
                    print("\n=== Starting Enhanced Cleaning ===")
                    
                    # Remove control characters and normalize line endings
                    clean_response = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', clean_response)
                    print("\nAfter removing control characters:")
                    print(clean_response)
                    
                    clean_response = clean_response.replace('\r\n', '\n').replace('\r', '\n')
                    print("\nAfter normalizing line endings:")
                    print(clean_response)
                    
                    # Fix common JSON formatting issues
                    clean_response = re.sub(r'(\w+):', r'"\1":', clean_response)
                    print("\nAfter fixing unquoted keys:")
                    print(clean_response)
                    
                    clean_response = re.sub(r',\s*}', '}', clean_response)
                    clean_response = re.sub(r',\s*]', ']', clean_response)
                    print("\nAfter removing trailing commas:")
                    print(clean_response)
                    
                    # Handle escaped quotes and special characters
                    clean_response = clean_response.replace('\\"', '"')
                    clean_response = clean_response.replace('\\n', ' ')
                    clean_response = clean_response.replace('\\t', ' ')
                    print("\nAfter handling escaped characters:")
                    print(clean_response)
                    
                    # Remove any double spaces
                    clean_response = re.sub(r'\s+', ' ', clean_response)
                    print("\nAfter removing double spaces:")
                    print(clean_response)
                    
                    print("\n=== Attempting Second JSON Parse ===")
                    analysis = json.loads(clean_response)
                    print(f"\nGenerated overall title: {analysis.get('overall_title', 'No title generated')}")
                except json.JSONDecodeError as e:
                    print(f"\n=== Second JSON Parse Failed ===")
                    print(f"Error: {str(e)}")
                    print("Error location:", e.pos)
                    print("Error line:", e.lineno)
                    print("Error column:", e.colno)
                    print("Final cleaned response:", clean_response)
                    raise Exception("Failed to parse Gemini response as valid JSON")
            
            # Store the analysis in database
            try:
                with sqlite3.connect(DB_PATH) as conn:
                    c = conn.cursor()
                    # Create table if it doesn't exist
                    c.execute('''CREATE TABLE IF NOT EXISTS comprehensive_analysis
                               (id INTEGER PRIMARY KEY AUTOINCREMENT,
                                coin_name TEXT,
                                analysis_data TEXT,
                                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
                    
                    # Delete any existing analysis for this coin
                    c.execute('DELETE FROM comprehensive_analysis WHERE coin_name = ?', 
                            (symbol.upper(),))
                    
                    # Insert new analysis
                    c.execute('''INSERT INTO comprehensive_analysis 
                               (coin_name, analysis_data)
                               VALUES (?, ?)''',
                            (symbol.upper(), json.dumps(analysis)))
                    conn.commit()
                    print(f"\nStored comprehensive analysis for {symbol.upper()}")
                    print(f"Article title: {analysis['article']['title']}")
                    print(f"Number of sections: {len(analysis['article']['sections'])}")
            except Exception as db_error:
                print(f"Database error: {str(db_error)}")
            
            return jsonify({
                "status": "success",
                "data": analysis
            })
            
    except Exception as e:
        print(f"\nError in process_coin_news: {str(e)}")
        print("\nFull error details:")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/youtube/mentions/all-dates')
def get_all_dates_coin_mentions():
    """Get list of all dates with their coin mention counts"""
    try:
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
            
            # Get all mentions with dates
            c.execute('''
                SELECT v.published_at, ca.coin_mentioned
                FROM videos v
                LEFT JOIN coin_analysis ca ON v.video_id = ca.video_id
                WHERE ca.coin_mentioned IS NOT NULL
            ''')
            
            # Process results in Python
            malaysia_tz = pytz.timezone('Asia/Kuala_Lumpur')
            date_counts = {}  # Dictionary to store date -> unique coins mapping
            
            for row in c.fetchall():
                try:
                    # Parse ISO format timestamp
                    published_at = datetime.fromisoformat(row['published_at'].replace('Z', '+00:00'))
                    published_at = pytz.utc.localize(published_at) if published_at.tzinfo is None else published_at
                    published_my = published_at.astimezone(malaysia_tz)
                    
                    # Get date in YYYY-MM-DD format
                    date_str = published_my.strftime('%Y-%m-%d')
                    
                    # Initialize set for this date if not exists
                    if date_str not in date_counts:
                        date_counts[date_str] = set()
                    
                    # Add coin to set (with name mapping if applicable)
                    coin_name = row['coin_mentioned']
                    if coin_name in name_mapping:
                        coin_name = name_mapping[coin_name]
                    date_counts[date_str].add(coin_name)
                    
                except Exception as e:
                    print(f"Error processing row: {str(e)}")
                    continue
            
            # Convert sets to counts and sort by date
            result = {date: len(coins) for date, coins in date_counts.items()}
            sorted_result = dict(sorted(result.items()))
            
            return jsonify(sorted_result)
            
    except Exception as e:
        print(f"Error getting coin mentions by date: {str(e)}")
        return jsonify({})
    
# New endpoints for detailed cryptocurrency analysis by category
@app.route('/api/crypto/features/<symbol>')
def get_crypto_features(symbol):
    """Get feature releases information for a cryptocurrency"""
    try:
        result = search_feature_releases(symbol)
        return jsonify({"success": True, "data": result}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/crypto/whitepaper/<symbol>')
def get_crypto_whitepaper(symbol):
    """Get whitepaper updates information for a cryptocurrency"""
    try:
        result = search_whitepaper_updates(symbol)
        return jsonify({"success": True, "data": result}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/crypto/milestones/<symbol>')
def get_crypto_milestones(symbol):
    """Get testnet/mainnet milestones information for a cryptocurrency"""
    try:
        result = search_testnet_mainnet_milestones(symbol)
        return jsonify({"success": True, "data": result}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/crypto/integration/<symbol>')
def get_crypto_integration(symbol):
    """Get platform integration information for a cryptocurrency"""
    try:
        result = search_platform_integration(symbol)
        return jsonify({"success": True, "data": result}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/crypto/team/<symbol>')
def get_crypto_team(symbol):
    """Get team changes information for a cryptocurrency"""
    try:
        result = search_team_changes(symbol)
        return jsonify({"success": True, "data": result}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/crypto/partnerships/<symbol>')
def get_crypto_partnerships(symbol):
    """Get new partnerships information for a cryptocurrency"""
    try:
        result = search_new_partnerships(symbol)
        return jsonify({"success": True, "data": result}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/crypto/onchain/<symbol>')
def get_crypto_onchain(symbol):
    """Get on-chain activity information for a cryptocurrency"""
    try:
        result = search_onchain_activity(symbol)
        return jsonify({"success": True, "data": result}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/crypto/addresses/<symbol>')
def get_crypto_addresses(symbol):
    """Get active addresses growth information for a cryptocurrency"""
    try:
        result = search_active_addresses_growth(symbol)
        return jsonify({"success": True, "data": result}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/crypto/adoption/<symbol>')
def get_crypto_adoption(symbol):
    """Get real-world adoption information for a cryptocurrency"""
    try:
        result = search_real_world_adoption(symbol)
        return jsonify({"success": True, "data": result}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/crypto/development/<symbol>')
def get_crypto_development(symbol):
    """Get developer activity information for a cryptocurrency"""
    try:
        result = search_developer_activity(symbol)
        return jsonify({"success": True, "data": result}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/crypto/sentiment/<symbol>')
def get_crypto_sentiment(symbol):
    """Get community sentiment information for a cryptocurrency"""
    try:
        result = search_community_sentiment(symbol)
        return jsonify({"success": True, "data": result}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/crypto/liquidity/<symbol>')
def get_crypto_liquidity(symbol):
    """Get liquidity changes information for a cryptocurrency"""
    try:
        result = search_liquidity_changes(symbol)
        return jsonify({"success": True, "data": result}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/crypto/full/<symbol>')
def get_crypto_analysis_full(symbol):
    """Get full analysis for a cryptocurrency (all categories)"""
    try:
        results = analyze_cryptocurrency_full(symbol)
        return jsonify({"success": True, "data": results}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/crypto/save/<symbol>', methods=['POST'])
def save_crypto_analysis(symbol):
    """Save analysis results to a file"""
    try:
        data = request.json
        if not data or 'results' not in data:
            return jsonify({"success": False, "error": "No results provided"}), 400
        
        filename = save_analysis_to_file(symbol, data['results'])
        return jsonify({"success": True, "filename": filename}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# Additional endpoints for the new cryptocurrency analysis categories
@app.route('/api/crypto/marketcap/<symbol>')
def get_crypto_marketcap(symbol):
    """Get market cap information for a cryptocurrency"""
    try:
        result = search_market_cap(symbol)
        return jsonify({"success": True, "data": result}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/crypto/supply/<symbol>')
def get_crypto_supply(symbol):
    """Get circulating vs total supply information for a cryptocurrency"""
    try:
        result = search_circulating_vs_total_supply(symbol)
        return jsonify({"success": True, "data": result}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/crypto/tokenomics/<symbol>')
def get_crypto_tokenomics(symbol):
    """Get tokenomics information for a cryptocurrency"""
    try:
        result = search_tokenomics(symbol)
        return jsonify({"success": True, "data": result}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/crypto/emission/<symbol>')
def get_crypto_emission(symbol):
    """Get inflation/emission rate information for a cryptocurrency"""
    try:
        result = search_inflation_emission_rate(symbol)
        return jsonify({"success": True, "data": result}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/crypto/utility/<symbol>')
def get_crypto_utility(symbol):
    """Get token utility information for a cryptocurrency"""
    try:
        result = search_token_utility(symbol)
        return jsonify({"success": True, "data": result}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/crypto/usage/<symbol>')
def get_crypto_usage(symbol):
    """Get network activity & usage information for a cryptocurrency"""
    try:
        result = search_network_activity_usage(symbol)
        return jsonify({"success": True, "data": result}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/crypto/fees/<symbol>')
def get_crypto_fees(symbol):
    """Get transaction fees and revenue information for a cryptocurrency"""
    try:
        result = search_transaction_fees_revenue(symbol)
        return jsonify({"success": True, "data": result}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/crypto/governance/<symbol>')
def get_crypto_governance(symbol):
    """Get governance and decentralization information for a cryptocurrency"""
    try:
        result = search_governance_decentralization(symbol)
        return jsonify({"success": True, "data": result}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/crypto/exchanges/<symbol>')
def get_crypto_exchanges(symbol):
    """Get exchange listings information for a cryptocurrency"""
    try:
        result = search_exchange_listings(symbol)
        return jsonify({"success": True, "data": result}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/crypto/regulation/<symbol>')
def get_crypto_regulation(symbol):
    """Get regulatory status information for a cryptocurrency"""
    try:
        result = search_regulatory_status(symbol)
        return jsonify({"success": True, "data": result}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/crypto/security/<symbol>')
def get_crypto_security(symbol):
    """Get security audits information for a cryptocurrency"""
    try:
        result = search_security_audits(symbol)
        return jsonify({"success": True, "data": result}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/crypto/competitors/<symbol>')
def get_crypto_competitors(symbol):
    """Get competitor analysis information for a cryptocurrency"""
    try:
        result = search_competitor_analysis(symbol)
        return jsonify({"success": True, "data": result}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/crypto/volatility/<symbol>')
def get_crypto_volatility(symbol):
    """Get volatility spikes / price swings information for a cryptocurrency"""
    try:
        result = search_volatility_price_swings(symbol)
        return jsonify({"success": True, "data": result}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/crypto/unusual/<symbol>')
def get_crypto_unusual(symbol):
    """Get unusual trading patterns information for a cryptocurrency"""
    try:
        result = search_unusual_trading_patterns(symbol)
        return jsonify({"success": True, "data": result}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/crypto/divergence/<symbol>')
def get_crypto_divergence(symbol):
    """Get divergence from market trends information for a cryptocurrency"""
    try:
        result = search_divergence_market_trends(symbol)
        return jsonify({"success": True, "data": result}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/crypto/catalysts/<symbol>')
def get_crypto_catalysts(symbol):
    """Get catalysts (news/events) information for a cryptocurrency"""
    try:
        result = search_catalysts(symbol)
        return jsonify({"success": True, "data": result}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/crypto/technical/<symbol>')
def get_crypto_technical(symbol):
    """Get technical signals information for a cryptocurrency"""
    try:
        result = search_technical_signals(symbol)
        return jsonify({"success": True, "data": result}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/crypto/strength/<symbol>')
def get_crypto_strength(symbol):
    """Get comparative strength/weakness information for a cryptocurrency"""
    try:
        result = search_comparative_strength_weakness(symbol)
        return jsonify({"success": True, "data": result}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/crypto/volume/<symbol>')
def get_crypto_volume(symbol):
    """Get trading volume anomalies information for a cryptocurrency"""
    try:
        result = search_trading_volume_anomalies(symbol)
        return jsonify({"success": True, "data": result}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/crypto/flows/<symbol>')
def get_crypto_flows(symbol):
    """Get exchange inflows/outflows information for a cryptocurrency"""
    try:
        result = search_exchange_inflows_outflows(symbol)
        return jsonify({"success": True, "data": result}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/crypto/futures/<symbol>')
def get_crypto_futures(symbol):
    """Get open interest & futures activity information for a cryptocurrency"""
    try:
        result = search_open_interest_futures(symbol)
        return jsonify({"success": True, "data": result}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# Additional utility endpoint for category-specific analysis
@app.route('/api/crypto/category/<symbol>', methods=['POST'])
def get_crypto_by_category(symbol):
    """Get information for a specific category"""
    try:
        data = request.json
        if not data or 'category' not in data:
            return jsonify({"success": False, "error": "No category specified"}), 400
        
        category = data['category']
        # Create a mapping of categories to their respective search functions
        category_map = {
            # Original categories
            "Feature Releases or Updates": search_feature_releases,
            "Whitepaper Updates": search_whitepaper_updates,
            "Testnet or Mainnet Milestones": search_testnet_mainnet_milestones,
            "Platform Integration": search_platform_integration,
            "Team Changes": search_team_changes,
            "New Partnerships": search_new_partnerships,
            "On-chain Activity": search_onchain_activity,
            "Active Addresses Growth": search_active_addresses_growth,
            "Real World Adoption": search_real_world_adoption,
            "Developer Activity": search_developer_activity,
            "Community Sentiment": search_community_sentiment,
            "Liquidity Changes": search_liquidity_changes,
            
            # New categories
            "Market Cap": search_market_cap,
            "Circulating Supply vs Total Supply": search_circulating_vs_total_supply,
            "Tokenomics": search_tokenomics,
            "Inflation/Emission Rate": search_inflation_emission_rate,
            "Token Utility": search_token_utility,
            "Network Activity & Usage": search_network_activity_usage,
            "Transaction Fees and Revenue": search_transaction_fees_revenue,
            "Governance and Decentralization": search_governance_decentralization,
            "Exchange Listings": search_exchange_listings,
            "Regulatory Status": search_regulatory_status,
            "Security Audits": search_security_audits,
            "Competitor Analysis": search_competitor_analysis,
            "Volatility Spikes / Price Swings": search_volatility_price_swings,
            "Unusual Trading Patterns": search_unusual_trading_patterns,
            "Divergence from Market Trends": search_divergence_market_trends,
            "Catalysts (News/Events)": search_catalysts,
            "Technical Signals": search_technical_signals,
            "Comparative Strength/Weakness": search_comparative_strength_weakness,
            "Trading Volume Anomalies": search_trading_volume_anomalies,
            "Exchange Inflows/Outflows": search_exchange_inflows_outflows,
            "Open Interest & Futures Activity": search_open_interest_futures
        }
        
        if category not in category_map:
            return jsonify({"success": False, "error": f"Unknown category: {category}"}), 400
        
        # Call the appropriate search function
        result = category_map[category](symbol)
        return jsonify({"success": True, "data": result, "category": category}), 200
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# Endpoint to get all available categories
@app.route('/api/crypto/categories')
def get_crypto_categories():
    """Get all available cryptocurrency analysis categories"""
    try:
        categories = [
            # Original categories
            "Feature Releases or Updates",
            "Whitepaper Updates",
            "Testnet or Mainnet Milestones",
            "Platform Integration",
            "Team Changes",
            "New Partnerships",
            "On-chain Activity",
            "Active Addresses Growth",
            "Real World Adoption",
            "Developer Activity",
            "Community Sentiment",
            "Liquidity Changes",
            
            # New categories
            "Market Cap",
            "Circulating Supply vs Total Supply",
            "Tokenomics",
            "Inflation/Emission Rate",
            "Token Utility",
            "Network Activity & Usage",
            "Transaction Fees and Revenue",
            "Governance and Decentralization",
            "Exchange Listings",
            "Regulatory Status",
            "Security Audits",
            "Competitor Analysis",
            "Volatility Spikes / Price Swings",
            "Unusual Trading Patterns",
            "Divergence from Market Trends",
            "Catalysts (News/Events)",
            "Technical Signals",
            "Comparative Strength/Weakness",
            "Trading Volume Anomalies",
            "Exchange Inflows/Outflows",
            "Open Interest & Futures Activity"
        ]
        
        return jsonify({"success": True, "categories": categories}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
    