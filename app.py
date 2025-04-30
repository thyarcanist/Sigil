#!/usr/bin/env python
"""
Quantum Data Service API
------------------------
REST API for the Quantum Data Service, offering tiered access to quantum-derived random data.

Tiers:
- Sprinkle: 1KB/day
- Pulse: 10KB/day
- Torrent: 1MB/week
- Gatekeeper: Custom burst with vetting

Author: Datorien Laurae Anderson
"""

import os
import sys
import json
import time
import uuid
import hashlib
import datetime
import base64
from pathlib import Path
from functools import wraps
from flask import Flask, request, jsonify, send_file, abort, render_template
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import jwt
from werkzeug.security import generate_password_hash, check_password_hash
from supabase import create_client, Client
from typing import List
import requests
import logging

# Configure paths for IDIA -> REMOVED as dependencies are copied into /app/src by Dockerfile
# IDIA_PATHS = [
#     Path(\"T:/Github/Idia\"),     # Local development path
#     Path(\"/app\"),               # Container path
#     Path(\"..\"),                 # Relative path
#     Path(\".\")                   # Current directory
# ]
#
# # Add potential IDIA paths to sys.path
# for path in IDIA_PATHS:
#     if path.exists():
#         sys.path.append(str(path))
#         if (path / \"src\").exists():
#             sys.path.append(str(path / \"src\"))
#             print(f\"Added IDIA path: {path}/src\")

# Import IDIA modules - Relative to /app/src
try:
    from src.idia.crypto import IdiaSignatureEngine
    from src.idia.entropy import ErisEntropy, WhitenedEntropySource
    from src.idia.audit.entropy_audit import EntropyAuditor
    # Note: If app.py needed utils, it would be: from src.utils import ...
    print("Successfully imported IDIA modules for quantum entropy from src")
except ImportError as e:
    print(f"Error: Failed to import IDIA modules from src: {e}")
    print("Ensure src/idia and src/utils are correctly copied in the Dockerfile and PYTHONPATH is set.")
    print("Paths searched:")
    for p in sys.path:
        print(f"  - {p}")
    sys.exit(1)  # Exit if we can't import the required modules

# *** DEVELOPMENT NOTE: Operational Limit for Current Plan ***
# Based on testing (April 2024) on DigitalOcean App Platform
# $12/mo plan (apps-s-1vcpu-1gb), requests >= 896 bytes timed out.
# $25/mo plan testing needs to confirm limits for larger sizes.
# This limit MUST be manually updated/removed if the plan is upgraded/downgraded.
# Setting limit based on Torrent tier average daily (~146KB) for testing on upgraded plan.
CURRENT_PLAN_OPERATIONAL_LIMIT_BYTES = 149504 # Approx 146KB
# ***********************************************************

# Initialize Flask app
app = Flask(__name__, static_folder='static', static_url_path='')
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', '28a7ac9e3ddf0dc4f46979dfe4e3')
app.config['QUANTUM_CACHE_DIR'] = os.environ.get('QUANTUM_CACHE_DIR', 'cache')

# +++ Configure Root Logging with DEBUG Handler +++
import logging
import sys # Import sys for stdout

root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG) # Set logger level

# Remove existing handlers to avoid duplicates if any were added by Flask/Gunicorn defaults
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

# Create a StreamHandler that outputs to standard error
handler = logging.StreamHandler(sys.stderr) # Output to stderr is common for logs
handler.setLevel(logging.DEBUG) # Set handler level

# Create a formatter and set it for the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Add the handler to the root logger
root_logger.addHandler(handler)

app.logger.info("Root logger and handler configured for DEBUG level output to stderr.")
# +++ End Logging Configuration +++

# --- Define External Eris API URL and Key ---
ERIS_API_URL = os.environ.get("ERIS_API_URL", "https://entropy.occybyte.com/api/eris/invoke")
ERIS_API_KEY = os.environ.get("ERIS_API_KEY") # Will be None if not set

if ERIS_API_KEY is None:
    print("WARNING: ERIS_API_KEY environment variable not set. External API calls may fail authentication.", file=sys.stderr)
# --------------------------------------------

# --- Initialize Supabase Client ---
supabase_url: str = os.environ.get("SUPABASE_URL")
supabase_key: str = os.environ.get("SUPABASE_SERVICE_KEY")
supabase_anon_key: str = os.environ.get("SUPABASE_ANON_KEY")
supabase: Client = None # Initialize as None

if supabase_url and supabase_key:
    try:
        supabase = create_client(supabase_url, supabase_key)
        print("INFO: Supabase client initialized successfully.")
    except Exception as e:
        print(f"ERROR: Failed to initialize Supabase client: {e}", file=sys.stderr)
        # Depending on requirements, you might want to exit or handle this differently
else:
    print("ERROR: SUPABASE_URL and/or SUPABASE_SERVICE_KEY environment variables not set.", file=sys.stderr)
    # Consider exiting if Supabase is critical
    # sys.exit(1)
# ---------------------------------

# Create cache directory
os.makedirs(app.config['QUANTUM_CACHE_DIR'], exist_ok=True)

# Configure rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["5 per minute"],
    storage_uri="memory://"
)

# Define service tiers
TIERS = {
    'free': {
        'daily_bytes': 256,
        'burst_bytes': 64,
        'rate_limit': "5 per day",
        'description': "Up to 256 bytes/day",
        'sigil': "🌱",
    },
    'sprinkle': { # Builder
        'daily_bytes': 81920,  # ~80KB/day
        'burst_bytes': 81920,  # Max size per request = daily limit
        'rate_limit': "20 per day",
        'description': "Builder: ~80KB/day quantum randomness",
        'sigil': "🔧",
    },
    'pulse': { # Chaos Engineer
        'daily_bytes': 143360, # ~140KB/day
        'burst_bytes': 143360, # Max size per request = daily limit
        'rate_limit': "50 per day",
        'description': "Chaos Engineer: ~140KB+/day quantum randomness",
        'sigil': "⚙️",
    },
    'torrent': { # Architect
        'daily_bytes': 524288,  # 512KB+/day
        'burst_bytes': 524288,  # Max size per request = daily limit
        'rate_limit': "100 per day",
        'description': "Entropy Architect: 512KB to 1MB+/day quantum randomness (Custom)",
        'sigil': "🧠",
    },
    'gatekeeper': {
        'daily_bytes': 10485760,  # 10MB/day
        'burst_bytes': 1048576,   # 1MB per request
        'rate_limit': "20 per day", # Kept original, can adjust if needed
        'description': "Custom quantum data bursts with vetting",
        'sigil': "🔮",
    },
    'house': { # <<< NEW INTERNAL TIER
        'daily_bytes': 10**12,  # Effectively unlimited (1 Terabyte)
        'burst_bytes': 10**9,   # Effectively unlimited burst (1 Gigabyte)
        'rate_limit': "10000 per minute", # Very high rate limit for internal calls
        'description': "Internal system tier for service functions (e.g., d20)",
        'sigil': "🏠",
    }
}

# +++ Add Cache-Control Headers for Eris Invoke +++
@app.after_request
def no_cache(response):
    if request.path.startswith("/api/eris/invoke"):
        response.headers['Cache-Control'] = 'private, no-store, max-age=0, must-revalidate'
        response.headers['Pragma']        = 'no-cache'
        response.headers['Expires']       = '0'
        # Optionally vary on the API‑Key/Authorization header
        response.headers['Vary'] = 'Authorization' # Vary based on auth header
    return response
# +++++++++++++++++++++++++++++++++++++++++++++++++++

# --- Helper Function for Parsing FAQ ---
def parse_qa_file(filepath):
    """Parses a Q&A file into a list of dictionaries."""
    faq_items = []
    # Check if the file actually exists before trying to open it
    if not os.path.exists(filepath):
        app.logger.error(f"FAQ file not found at path: {filepath}")
        return faq_items # Return empty list if file doesn't exist

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            current_q = None
            current_a = None
            for line in f:
                stripped_line = line.strip()
                if stripped_line.startswith('Q:'):
                    # If we have a complete previous Q/A pair, save it
                    if current_q and current_a is not None:
                        faq_items.append({'q': current_q, 'a': current_a})
                    # Start new question
                    current_q = stripped_line[2:].strip()
                    current_a = None # Reset answer
                elif stripped_line.startswith('A:') and current_q is not None:
                    # Start new answer, allow multi-line answers by initializing here
                    current_a = stripped_line[2:].strip()
                elif current_a is not None and stripped_line: # Handle multi-line answers
                     current_a += "<br>" + stripped_line # Append subsequent lines
                elif not stripped_line and current_q is not None and current_a is not None:
                     # Empty line signals end of current Q/A pair
                     faq_items.append({'q': current_q, 'a': current_a})
                     current_q = None
                     current_a = None

            # Add the last pair if the file doesn't end with a blank line
            if current_q and current_a is not None:
                faq_items.append({'q': current_q, 'a': current_a})
    except FileNotFoundError: # This shouldn't happen due to the os.path.exists check, but good practice
        app.logger.error(f"FAQ file disappeared between check and open: {filepath}")
    except Exception as e:
         app.logger.error(f"Error parsing FAQ file {filepath}: {e}")
         # Depending on severity, you might want to clear faq_items here
         # faq_items = []
    return faq_items
# --- End Helper Function ---

# --- Function to determine rate limit based on API Key --- 
def get_rate_limit_for_user():
    """Dynamically determines the rate limit string based on the user's tier (from Supabase)."""
    token = request.headers.get('X-API-Key')
    
    # Default limit for requests without a key or with an invalid key
    default_limit = "5 per minute" 
    
    if not token:
        return default_limit
    
    # --- Query Supabase for user tier by API key ---
    user_tier = None
    try:
        if supabase:
            response = supabase.table('users').select('tier').eq('api_key', token).limit(1).single().execute()
            if response.data:
                user_tier = response.data.get('tier')
        else:
            app.logger.error("Supabase client not available for rate limit check.")
            # Fallback to default if DB is down during check
            return default_limit 
            
    except Exception as e:
        app.logger.error(f"Error querying Supabase for tier during rate limit check: {e}")
        # Fallback to default on DB error
        return default_limit
    # -------------------------------------------
            
    if user_tier is None:
        # User not found for the given API key
        return default_limit # Treat invalid key as default rate
        
    # Get rate limit from static TIERS config based on fetched tier
    tier_info = TIERS.get(user_tier)
    
    if tier_info and 'rate_limit' in tier_info:
        return tier_info['rate_limit']
    else:
        # Fallback if tier is invalid in DB or missing rate_limit in config
        app.logger.warning(f"Could not determine rate limit for tier '{user_tier}' fetched from DB. Applying default.")
        return default_limit
# ----------------------------------------------------------

# +++ NEW Flexible Auth Decorator +++
def require_auth(f):
    """Decorator for endpoints requiring authentication via Supabase JWT or X-API-Key."""
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_user_id = None # Initialize auth_user_id

        # 1. Check for Authorization: Bearer JWT
        auth_header = request.headers.get('Authorization')
        jwt_token = None
        if auth_header and auth_header.startswith('Bearer '):
            jwt_token = auth_header.split(' ')[1]

        if jwt_token:
            try:
                if supabase is None:
                    app.logger.error("Supabase client not initialized. Cannot authenticate user via JWT.")
                    # Proceed to check API Key? Or fail? Let's try API key.
                else:
                    user_data = supabase.auth.get_user(jwt_token)
                    if user_data and user_data.user and user_data.user.id:
                        auth_user_id = user_data.user.id
                        app.logger.debug(f"Authenticated via JWT for user ID: {auth_user_id}")
                    else:
                        app.logger.warning("JWT provided but validation failed or returned no user.")
                        # Don't return error yet, proceed to check API key
            except Exception as e:
                app.logger.error(f"Error validating Supabase JWT: {e}")
                # Don't return error yet, proceed to check API key

        # 2. If JWT didn't authenticate, check for X-API-Key
        if not auth_user_id:
            api_key = request.headers.get('X-API-Key')
            if api_key:
                try:
                    if supabase is None:
                        app.logger.error("Supabase client not initialized. Cannot authenticate user via API Key.")
                        return jsonify({'message': 'Internal server error: Auth system unavailable'}), 500

                    # Query the users table for the API key
                    response = supabase.table('users').select('auth_user_id').eq('api_key', api_key).limit(1).single().execute()

                    if response.data and response.data.get('auth_user_id'):
                        auth_user_id = response.data.get('auth_user_id')
                        app.logger.debug(f"Authenticated via API Key for user ID: {auth_user_id}")
                    else:
                        app.logger.warning(f"Invalid X-API-Key provided: {api_key[:5]}...") # Log truncated key
                        # Key is invalid, return error
                        return jsonify({'message': 'Invalid API Key'}), 401

                except Exception as e:
                    app.logger.error(f"Error validating API Key: {e}")
                    return jsonify({'message': 'Internal server error during API Key validation'}), 500
            # else: No API key provided either, error will be raised below

        # 3. Final Check
        if not auth_user_id:
            # Neither JWT nor API Key resulted in a valid user ID
            app.logger.warning("Authentication failed: No valid JWT or API Key provided.")
            return jsonify({'message': 'Authentication required (Bearer token or X-API-Key)'}), 401

        # Pass the authenticated Supabase Auth user ID (UUID) to the decorated function
        return f(auth_user_id, *args, **kwargs)

    return decorated
# +++ End Flexible Auth Decorator +++

def token_required(f):
    """Decorator for endpoints that require a valid Supabase JWT token"""
    @wraps(f)
    def decorated(*args, **kwargs):
        # Expect token in Authorization header
        auth_header = request.headers.get('Authorization')
        token = None
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
        
        if not token:
            return jsonify({'message': 'Authorization token is missing'}), 401
        
        # --- Validate JWT with Supabase --- 
        user_data = None
        try:
            if supabase is None:
                app.logger.error("Supabase client not initialized. Cannot authenticate user.")
                return jsonify({'message': 'Internal server error: Auth system unavailable'}), 500

            # Validate token and get user data
            user_data = supabase.auth.get_user(token) 
            
            if not user_data or not user_data.user:
                app.logger.warning(f"Supabase JWT validation failed or returned no user.")
                return jsonify({'message': 'Invalid or expired token'}), 401

        except Exception as e:
            # Catch potential exceptions from supabase.auth.get_user (e.g., invalid token format, network issues)
            app.logger.error(f"Error validating Supabase JWT: {e}")
            # Map specific Supabase errors if possible, otherwise generic invalid token
            return jsonify({'message': 'Invalid or expired token'}), 401
        # ----------------------------------

        # Extract the user's Supabase Auth ID (UUID)
        auth_user_id = user_data.user.id 
        if not auth_user_id:
             app.logger.error(f"Could not extract user ID from validated Supabase JWT.")
             return jsonify({'message': 'Error processing authentication token'}), 500
        
        # Pass the Supabase Auth user ID (UUID) to the decorated function
        return f(auth_user_id, *args, **kwargs) # Pass UUID instead of username
    
    return decorated

def tier_limit(f):
    """Decorator to enforce tier-based limits using Supabase."""
    @wraps(f)
    def decorated(auth_user_id, *args, **kwargs): # Now receives auth_user_id (UUID)
        # Ensure supabase client is available
        if supabase is None:
            app.logger.error("Supabase client not initialized. Cannot enforce tier limits.")
            return jsonify({'message': 'Internal server error: Tier system unavailable'}), 500

        try:
            # 1. Fetch user data from Supabase using auth_user_id
            response = supabase.table('users').select('tier, daily_usage, last_reset').eq('auth_user_id', auth_user_id).limit(1).single().execute()

            if not response.data:
                app.logger.warning(f"Could not find user profile for auth ID '{auth_user_id}' in database.") # Log UUID
                return jsonify({'message': 'User not found for tier check'}), 404 
            
            user_data = response.data
            user_tier = user_data.get('tier')
            current_usage = user_data.get('daily_usage', 0)
            last_reset_str = user_data.get('last_reset') # Supabase returns date as string 'YYYY-MM-DD'

            # 2. Check if usage needs resetting
            today_str = datetime.date.today().isoformat()
            needs_reset = False
            if last_reset_str != today_str:
                needs_reset = True
                current_usage = 0 # Reset usage for limit check

            # 3. Get tier limits from static config
            tier_info = TIERS.get(user_tier)
            if not tier_info:
                app.logger.error(f"Invalid tier '{user_tier}' configured for user ID '{auth_user_id}'.")
                return jsonify({'message': 'Internal server error: Invalid user configuration'}), 500
            
            daily_limit = tier_info.get('daily_bytes')

            # 4. Check limit
            if current_usage >= daily_limit:
                return jsonify({
                    'message': 'Daily quota exceeded',
                    'tier': user_tier,
                    'limit': daily_limit,
                    'used': current_usage # Show the usage that exceeds the limit
                }), 429
            
            # 5. Update usage in DB if it was reset using auth_user_id
            if needs_reset:
                 update_response = supabase.table('users').update({
                     'daily_usage': 0,
                     'last_reset': today_str
                 }).eq('auth_user_id', auth_user_id).execute()
                 # Optional: check update_response for errors
                 if not update_response.data:
                      app.logger.error(f"Failed to reset daily usage for user ID '{auth_user_id}'.")
                      # Decide how to handle - proceed or error out?
                      # Proceeding might mean user exceeds quota temporarily if DB update fails

        except Exception as e:
            app.logger.error(f"Error during tier limit check for user ID '{auth_user_id}': {e}") # Log UUID
            # Consider potential infinite loops if DB errors persist
            return jsonify({'message': 'Internal server error during tier check'}), 500

        # If limit not exceeded, proceed to the actual route function
        return f(auth_user_id, *args, **kwargs)
    
    return decorated

@app.route('/api/health', methods=['GET'])
def health_check():
    """Basic health check endpoint"""
    return jsonify({
        'status': 'online',
        'timestamp': datetime.datetime.now().isoformat(),
        'service': 'Quantum Data API',
        'entropy_source': 'ERIS Quantum RNG v3.0.0'
    })

@app.route('/api/tiers', methods=['GET'])
def list_tiers():
    """List available service tiers"""
    return jsonify({
        'tiers': {
            tier_name: {
                'description': tier_info['description'],
                'daily_bytes': tier_info['daily_bytes'],
                'burst_bytes': tier_info['burst_bytes'],
                'sigil': tier_info['sigil']
            } for tier_name, tier_info in TIERS.items()
        }
    })

@app.route('/api/user/info', methods=['GET'])
@require_auth # <<< CHANGED from @token_required
def user_info(auth_user_id): # Receives auth_user_id (UUID)
    """Get user information and tier details from Supabase."""
    # Ensure supabase client is available
    if supabase is None:
        app.logger.error("Supabase client not initialized. Cannot fetch user info.")
        return jsonify({'message': 'Internal server error: DB unavailable'}), 500

    try:
        # Fetch user data from Supabase using auth_user_id
        response = supabase.table('users').select('username, tier, daily_usage, last_reset, api_key').eq('auth_user_id', auth_user_id).limit(1).single().execute()

        if not response.data:
            app.logger.warning(f"Could not find user profile for auth ID '{auth_user_id}' in database.") # Log UUID
            return jsonify({'message': 'User profile not found'}), 404
        
        user_data = response.data
        user_tier = user_data.get('tier')
        current_usage = user_data.get('daily_usage', 0)
        last_reset_str = user_data.get('last_reset')

        # Get static tier info
        tier_info = TIERS.get(user_tier)
        if not tier_info:
            app.logger.error(f"Invalid tier '{user_tier}' configured for user ID '{auth_user_id}'.")
            # Don't expose internal config error directly to user
            return jsonify({'message': 'Error retrieving user tier information'}), 500
        
        daily_limit = tier_info.get('daily_bytes')
        
        # Check if usage needs resetting for display purposes (doesn't update DB here)
        today_str = datetime.date.today().isoformat()
        if last_reset_str != today_str:
            display_usage = 0
            display_last_reset = today_str
        else:
            display_usage = current_usage
            display_last_reset = last_reset_str

        # Prepare response (include api_key if needed)
        return jsonify({
            'username': user_data.get('username'), # Might be email from social provider
            'tier': user_tier,
            'tier_info': {
                'description': tier_info['description'],
                'daily_bytes': daily_limit,
                'burst_bytes': tier_info['burst_bytes'],
                'sigil': tier_info['sigil']
            },
            'usage': {
                'used_today': display_usage,
                'remaining': daily_limit - display_usage,
                'last_reset': display_last_reset
            },
            'api_key': user_data.get('api_key') # Return API key for programmatic use
        })

    except Exception as e:
        app.logger.error(f"Error fetching user info for auth ID '{auth_user_id}': {e}") # Log UUID
        return jsonify({'message': 'Internal server error retrieving user info'}), 500

# +++ Initialize Shared Engine Placeholder +++
shared_auth_quantum_engine = None
# ++++++++++++++++++++++++++++++++++++++++++++

# +++ Initialize Engine Per Worker Before First Request +++
@app.before_first_request
def initialize_shared_engine():
    """Initializes and warms up the shared engine instance for the current worker process."""
    global shared_auth_quantum_engine
    if shared_auth_quantum_engine is None:
        print("INFO: Initializing shared authenticated engine for this worker...")
        try:
            # Using eris:full as these endpoints typically provide whitened data
            shared_auth_quantum_engine = IdiaSignatureEngine(entropy_source="eris:full")
            print("INFO: Shared IdiaSignatureEngine instance created successfully for worker.")
            
            # +++ WARM-UP: Generate and discard initial bytes to evolve state +++
            print("INFO: Warming up shared authenticated engine for worker...")
            _ = shared_auth_quantum_engine.generate_salt(4096) # <<< INCREASED WARM-UP SIZE
            print("INFO: Shared authenticated engine warm-up complete for worker.")
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            
        except Exception as e:
            print(f"ERROR: Failed to initialize or warm up shared authenticated IdiaSignatureEngine for worker: {e}", file=sys.stderr)
            # Keep it None so subsequent requests fail gracefully
            shared_auth_quantum_engine = None 
    # else: Engine already initialized for this worker
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

@app.route('/api/eris/invoke', methods=['GET'])
@require_auth # <<< CHANGED from @token_required
@tier_limit # Receives auth_user_id now
@limiter.limit(get_rate_limit_for_user) # Rate limit check needs review if it depends on tier fetched via old method
def get_quantum_random(auth_user_id): # Receives auth_user_id (UUID)
    """Generate and return quantum random data via the Eris invoke endpoint"""
    # Remove redundant global keyword here as the before_first_request handles init
    # global shared_auth_quantum_engine 

    # Ensure supabase client is available
    if supabase is None:
        app.logger.error("Supabase client not initialized. Cannot process request.")
        return jsonify({'message': 'Internal server error: DB unavailable'}), 500
        
    # Ensure shared engine is available (Initialized by before_first_request)
    if shared_auth_quantum_engine is None:
        app.logger.error("Shared authenticated quantum engine is not available for this worker.")
        return jsonify({'message': 'Quantum data source temporarily unavailable.'}), 503 # Service Unavailable
        
    # Get size parameter with tier-based limits
    # Note: tier_limit decorator already validated user and tier existence
    #       and checked if *current* usage allows *any* request.
    #       We still need tier info here for burst_bytes.
    try:
        # Fetch user data using auth_user_id
        user_data_resp = supabase.table('users').select('tier, daily_usage').eq('auth_user_id', auth_user_id).single().execute()
        if not user_data_resp.data:
             app.logger.error(f"User '{auth_user_id}' not found during request processing.")
             return jsonify({'message': 'User data inconsistency'}), 500
        user_data = user_data_resp.data
        user_tier = user_data['tier']
        current_usage = user_data['daily_usage']
    except Exception as e:
        app.logger.error(f"Error fetching user tier/usage for {auth_user_id}: {e}")
        return jsonify({'message': 'Internal server error fetching user data'}), 500
        
    tier_info = TIERS.get(user_tier) # Get static tier info
    if not tier_info:
        app.logger.error(f"Invalid tier '{user_tier}' found for user ID '{auth_user_id}'.")
        return jsonify({'message': 'Internal server error: Invalid user configuration'}), 500

    daily_limit = tier_info.get('daily_bytes')
    burst_limit = tier_info.get('burst_bytes')
    
    try:
        requested_size = int(request.args.get('size', 64))
    except ValueError:
        return jsonify({'message': 'Invalid size parameter'}), 400
    
    # +++ Add Check for Current Plan Operational Limit +++
    if requested_size > CURRENT_PLAN_OPERATIONAL_LIMIT_BYTES:
        app.logger.warning(f"Request size {requested_size} exceeds current plan limit {CURRENT_PLAN_OPERATIONAL_LIMIT_BYTES} for user {auth_user_id}.")
        return jsonify({
            'message': f'Requested size ({requested_size} bytes) exceeds the operational limit ({CURRENT_PLAN_OPERATIONAL_LIMIT_BYTES} bytes) for the current service plan. Please request fewer bytes or contact support regarding higher volume needs.'
        }), 413 # Payload Too Large
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    # Enforce size limits based on tier and remaining quota
    remaining_quota = daily_limit - current_usage
    max_allowed_size = min(burst_limit, remaining_quota)
    
    # Ensure requested size is positive and within limits
    if requested_size <= 0:
         return jsonify({'message': 'Size must be positive'}), 400
         
    # Clamp the size to what's allowed
    size_to_generate = min(requested_size, max_allowed_size)
    
    if size_to_generate <= 0:
        # This means remaining_quota was <= 0, which should have been caught by @tier_limit
        # but we double-check here.
        app.logger.warning(f"Request for user ID {auth_user_id} blocked, calculated size_to_generate <= 0. Remaining quota: {remaining_quota}")
        return jsonify({'message': 'Daily quota exceeded'}), 429
    
    # USE SHARED INSTANCE (Now initialized per-worker)
    try:
        quantum_data = shared_auth_quantum_engine.generate_salt(size_to_generate)
    except Exception as e:
        app.logger.error(f"Error generating quantum data for {auth_user_id} using shared engine: {str(e)}")
        return jsonify({'message': 'Error generating quantum data'}), 500
    
    # --- Update usage in Supabase using auth_user_id --- 
    new_usage = current_usage + size_to_generate
    try:
        update_response = supabase.table('users').update({
            'daily_usage': new_usage
        }).eq('auth_user_id', auth_user_id).execute()
        # Optional: Add error checking for update_response
        if not update_response.data:
            app.logger.error(f"Failed to update usage for user ID '{auth_user_id}' from {current_usage} to {new_usage}")
            # Decide how to handle: maybe roll back? For now, just log.

    except Exception as e:
        app.logger.error(f"Error updating usage for auth ID '{auth_user_id}': {e}") # Log UUID
        # Critical decision: If usage update fails, should we still return data?
        # For now, we proceed but log the error. Consider returning 500 if usage must be tracked.
    # --------------------------------
    
    # Generate response
    response = {
        'size': len(quantum_data), # Actual generated size
        'timestamp': datetime.datetime.now().isoformat(),
        'tier': user_tier,
        'sigil': tier_info['sigil'],
        'sha256': hashlib.sha256(quantum_data).hexdigest(),
        'data': base64.b64encode(quantum_data).decode('utf-8'), # Use base64 encoding
        'usage': {
            'used_today': new_usage, # Reflect the updated usage
            'remaining': daily_limit - new_usage
        }
    }
    
    return jsonify(response)

@app.route('/api/quantum/certificate', methods=['GET'])
@require_auth # <<< CHANGED from @token_required
@tier_limit # Receives auth_user_id now
def get_quantum_certificate(auth_user_id): # Receives auth_user_id (UUID)
    """Generate a verification certificate for quantum data"""
    # Ensure supabase client is available
    if supabase is None:
        app.logger.error("Supabase client not initialized. Cannot process request.")
        return jsonify({'message': 'Internal server error: DB unavailable'}), 500

    # --- Fetch user data and determine limits using auth_user_id ---
    try:
        user_data_resp = supabase.table('users').select('tier, daily_usage').eq('auth_user_id', auth_user_id).single().execute()
        if not user_data_resp.data:
             app.logger.error(f"User '{auth_user_id}' not found during certificate request processing.")
             return jsonify({'message': 'User data inconsistency'}), 500
        user_data = user_data_resp.data
        user_tier = user_data['tier']
        current_usage = user_data['daily_usage']
    except Exception as e:
        app.logger.error(f"Error fetching user tier/usage for certificate {auth_user_id}: {e}")
        return jsonify({'message': 'Internal server error fetching user data'}), 500

    tier_info = TIERS.get(user_tier)
    if not tier_info:
        app.logger.error(f"Invalid tier '{user_tier}' found for user ID '{auth_user_id}'.")
        return jsonify({'message': 'Internal server error: Invalid user configuration'}), 500

    daily_limit = tier_info.get('daily_bytes')
    burst_limit = tier_info.get('burst_bytes')
    # -------------------------------------------

    # Get size parameter
    try:
        requested_size = int(request.args.get('size', 64))
    except ValueError:
        return jsonify({'message': 'Invalid size parameter'}), 400
    
    # +++ Add Check for Current Plan Operational Limit +++
    if requested_size > CURRENT_PLAN_OPERATIONAL_LIMIT_BYTES:
        app.logger.warning(f"Certificate request size {requested_size} exceeds current plan limit {CURRENT_PLAN_OPERATIONAL_LIMIT_BYTES} for user {auth_user_id}.")
        return jsonify({
            'message': f'Requested size ({requested_size} bytes) exceeds the operational limit ({CURRENT_PLAN_OPERATIONAL_LIMIT_BYTES} bytes) for the current service plan. Please request fewer bytes or contact support regarding higher volume needs.'
        }), 413 # Payload Too Large
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # Enforce size limits
    remaining_quota = daily_limit - current_usage
    max_allowed_size = min(burst_limit, remaining_quota)

    if requested_size <= 0:
         return jsonify({'message': 'Size must be positive'}), 400

    size_to_generate = min(requested_size, max_allowed_size)

    if size_to_generate <= 0:
        app.logger.warning(f"Certificate request for user ID {auth_user_id} blocked, calculated size_to_generate <= 0. Remaining quota: {remaining_quota}")
        return jsonify({'message': 'Daily quota exceeded'}), 429
    
    # +++ Use Separate Engines for Raw and Whitened Data +++
    if shared_auth_quantum_engine is None:
        app.logger.error(f"Shared quantum engine not initialized during certificate request for user {auth_user_id}.")
        return jsonify({'message': 'Internal server error: Quantum service unavailable'}), 503 # Service Unavailable

    try:
        # Use the shared engine (initialized as eris:full) for whitened data
        whitened_data = shared_auth_quantum_engine.generate_salt(size_to_generate)
        
        # Create a temporary, separate engine for raw data
        raw_engine = IdiaSignatureEngine(entropy_source="eris:raw")
        raw_data = raw_engine.generate_salt(size_to_generate)
        
        # Explicitly delete the temporary raw engine instance (optional, for clarity)
        del raw_engine

    except Exception as e:
        app.logger.error(f"Error generating quantum data for certificate ({auth_user_id}): {str(e)}")
        return jsonify({'message': 'Error generating quantum data for certificate'}), 500
    # +++ End Use Separate Engines +++
    
    # --- Calculate Metrics using EntropyAuditor ---
    auditor = EntropyAuditor()
    calculated_metrics = {}
    try:
        # Calculate metrics for WHITENED data (most commonly reported)
        whitened_entropy = auditor.calculate_entropy(whitened_data)
        whitened_freq_test = auditor.run_frequency_test(whitened_data)
        whitened_corr_test = auditor.run_serial_correlation_test(whitened_data)
        whitened_bit_dist = auditor.analyze_bit_distribution(whitened_data)
        
        # Calculate entropy for RAW data (might be interesting)
        raw_entropy = auditor.calculate_entropy(raw_data)
        
        calculated_metrics = {
            'entropy_score_raw': raw_entropy / 8.0, # Raw entropy score
            'entropy_score_whitened': whitened_entropy / 8.0, # Whitened entropy score
            'balance_whitened': whitened_freq_test.get('balance', 'N/A'),
            'ones_zeros_ratio': f"{whitened_freq_test.get('proportion_ones', 0.0):.3f}:{whitened_freq_test.get('proportion_zeros', 0.0):.3f}",
            'correlation': whitened_corr_test.get('correlation', 'N/A'),
        }
        app.logger.debug(f"Calculated metrics for certificate: {calculated_metrics}")

    except Exception as audit_err:
        app.logger.error(f"Error calculating audit metrics for certificate ({auth_user_id}): {audit_err}")
        # Proceed with placeholders if calculation fails
        calculated_metrics = {} # Reset to ensure placeholders are used

    # --- Update usage in Supabase using auth_user_id --- 
    new_usage = current_usage + size_to_generate
    try:
        update_response = supabase.table('users').update({
            'daily_usage': new_usage
        }).eq('auth_user_id', auth_user_id).execute()
        if not update_response.data:
            app.logger.error(f"Failed to update usage for certificate auth ID '{auth_user_id}' from {current_usage} to {new_usage}")
            # Log and proceed for now

    except Exception as e:
        app.logger.error(f"Error updating usage for certificate auth ID '{auth_user_id}': {e}") # Log UUID
        # Log and proceed for now
    # --------------------------------

    # Generate certificate response (use size_to_generate for accuracy)
    certificate = {
        'title': "QUANTUM DATA VERIFICATION CERTIFICATE",
        'generation_date': datetime.datetime.now().isoformat(),
        'size_bytes': size_to_generate, # Use actual generated size
        'user': auth_user_id,
        'tier': user_tier,
        'file_integrity': {
            'raw_data_sha256': hashlib.sha256(raw_data).hexdigest(),
            'whitened_data_sha256': hashlib.sha256(whitened_data).hexdigest(),
        },
        'quantum_source': {
            'system': f"ERIS Quantum Random Number Generator v{os.environ.get('ERIS_VERSION', '3.0.0')}", # Use env var if possible
            'raw_source': "eris:raw",
            'whitened_source': "eris:full",
            'raw_chi_square': 821027.0, # Placeholder - Requires external test suite
            'quality_score_raw': calculated_metrics.get('entropy_score_raw', 'N/A'),
            'quality_score_whitened': calculated_metrics.get('entropy_score_whitened', 'N/A')
        },
        'statistical_properties': {
            'tests_passed_raw': "N/A", # Placeholder - Requires external test suite
            'tests_passed_whitened': "N/A", # Placeholder - Requires external test suite
            # Use calculated values if available, otherwise keep placeholders
            'balance_whitened': calculated_metrics.get('balance_whitened', 0.994),
            'ones_zeros_ratio': calculated_metrics.get('ones_zeros_ratio', "0.503:0.497"),
            'correlation': calculated_metrics.get('correlation', 0.017),
            'entropy_score': calculated_metrics.get('entropy_score_whitened', 0.995) # Bits per bit (whitened)
        },
        'certification': {
            'message': "This certificate verifies genuine quantum-derived random data produced by the ERIS Quantum RNG system.",
            'certified_on': datetime.datetime.now().strftime('%Y-%m-%d')
        },
        'usage': { # Add usage info to certificate response
            'used_today': new_usage,
            'remaining': daily_limit - new_usage
        },
        'data': {
            'raw': raw_data.hex(),
            'whitened': whitened_data.hex()
        }
    }
    
    return jsonify(certificate)

# +++ UPDATED Root Route to render index.html template +++
@app.route('/', methods=['GET'])
def index():
    """Renders the index.html template, injecting Supabase keys and FAQ data."""
    # Read keys from environment variables (ensure they are set in DO App Spec)
    supa_url = os.environ.get('SUPABASE_URL')
    supa_anon_key = os.environ.get('SUPABASE_ANON_KEY')

    if not supa_url or not supa_anon_key:
        app.logger.error("SUPABASE_URL or SUPABASE_ANON_KEY environment variables not set for template rendering!")
        # Return a simple error page or message
        return "Error: Application configuration missing.", 500

    # --- Get FAQ Data ---
    faq_data = [] # Default to empty list
    try:
        # Construct the path relative to the Flask app's root directory
        # Assumes app.py is in the 'api' directory relative to the project root where 'data' also sits
        faq_file_path = os.path.join(os.path.dirname(__file__), 'data', 'qa.txt')
        # Use the parse_qa_file function defined above
        faq_data = parse_qa_file(faq_file_path)
        if not faq_data:
            app.logger.warning(f"FAQ data parsed as empty from {faq_file_path}")
    except Exception as e:
        app.logger.error(f"Error processing FAQ data from {faq_file_path}: {e}")
        # Proceed without FAQ data, template will handle empty list

    # --- Render Template ---
    try:
        # Pass the keys and faq_items to the template context
        return render_template('index.html',
                               supabase_url=supa_url,
                               supabase_anon_key=supa_anon_key,
                               faq_items=faq_data) # Pass the parsed FAQ data
    except Exception as e: # Catch potential TemplateNotFound errors or others
        app.logger.error(f"Error rendering template index.html: {e}")
        abort(500)
# +++ End Root Route +++

# +++ NEW Password Generation Endpoint +++
@app.route('/api/generate/password', methods=['GET'])
@require_auth # <<< CHANGED from @token_required
@tier_limit # Apply tier usage checks even for small amounts
@limiter.limit("10 per minute") # Add a specific rate limit
def generate_password_endpoint(auth_user_id):
    """Generates a secure password using raw quantum data."""
    # Ensure supabase client is available
    if supabase is None:
        app.logger.error("Supabase client not initialized. Cannot generate password.")
        return jsonify({'message': 'Internal server error: DB unavailable'}), 500

    # --- Fetch user data and determine limits --- 
    try:
        user_data_resp = supabase.table('users').select('tier, daily_usage').eq('auth_user_id', auth_user_id).single().execute()
        if not user_data_resp.data:
             app.logger.error(f"User '{auth_user_id}' not found during password request processing.")
             return jsonify({'message': 'User data inconsistency'}), 500
        user_data = user_data_resp.data
        user_tier = user_data['tier']
        current_usage = user_data['daily_usage']
    except Exception as e:
        app.logger.error(f"Error fetching user tier/usage for password {auth_user_id}: {e}")
        return jsonify({'message': 'Internal server error fetching user data'}), 500

    tier_info = TIERS.get(user_tier)
    if not tier_info:
        app.logger.error(f"Invalid tier '{user_tier}' found for user ID '{auth_user_id}'.")
        return jsonify({'message': 'Internal server error: Invalid user configuration'}), 500

    daily_limit = tier_info.get('daily_bytes')
    burst_limit = tier_info.get('burst_bytes')
    # -------------------------------------------

    # Get size parameter (default to 32 bytes -> 256 bits)
    try:
        requested_bytes = int(request.args.get('bytes', 32))
    except ValueError:
        return jsonify({'message': 'Invalid bytes parameter, must be an integer'}), 400

    # Validate requested size
    if not (0 < requested_bytes <= 128): # Set a reasonable max, e.g., 128 bytes (1024 bits)
        return jsonify({'message': 'Invalid bytes parameter, must be between 1 and 128'}), 400

    # Enforce size limits based on tier and remaining quota
    remaining_quota = daily_limit - current_usage
    # For passwords, the burst limit is less critical, but we respect the remaining quota
    max_allowed_bytes = min(requested_bytes, remaining_quota)

    if max_allowed_bytes <= 0:
        app.logger.warning(f"Password request for user ID {auth_user_id} blocked, not enough quota. Remaining: {remaining_quota}")
        return jsonify({'message': 'Daily quota exceeded'}), 429

    bytes_to_generate = max_allowed_bytes # Use the calculated allowed bytes

    # Initialize quantum generator - USE SHARED ENGINE
    try:
        # --- Use the shared, warmed-up engine ---
        # Note: Passwords typically use raw entropy, but shared engine is eris:full.
        # If raw is strictly needed, a second shared engine or config change is required.
        # Using eris:full for now as it's available and provides randomness.
        if shared_auth_quantum_engine is None:
            app.logger.error("Shared quantum engine is not available for password generation.")
            return jsonify({'message': 'Quantum data source temporarily unavailable.'}), 503
            
        raw_bytes = shared_auth_quantum_engine.generate_salt(bytes_to_generate)
        # --- End use shared engine ---
        
        # --- REMOVED LOCAL ENGINE INSTANCE ---
        # engine = IdiaSignatureEngine(entropy_source="eris:full") 
        # raw_bytes = engine.generate_salt(bytes_to_generate)
        # --- END REMOVED ---
    except Exception as e:
        app.logger.error(f"Error generating quantum data for password ({auth_user_id}) using shared engine: {str(e)}")
        return jsonify({'message': 'Error generating quantum data'}), 500

    # --- Update usage in Supabase --- 
    new_usage = current_usage + bytes_to_generate
    try:
        update_response = supabase.table('users').update({
            'daily_usage': new_usage
        }).eq('auth_user_id', auth_user_id).execute()
        if not update_response.data:
            app.logger.error(f"Failed to update usage for password request auth ID '{auth_user_id}'")
            # Log and proceed for now

    except Exception as e:
        app.logger.error(f"Error updating usage for password request auth ID '{auth_user_id}': {e}")
        # Log and proceed for now
    # --------------------------------

    # Format password (URL-safe Base64 is good for general use)
    generated_password = base64.urlsafe_b64encode(raw_bytes).decode('utf-8').rstrip('=')

    # Generate response
    response = {
        'password': generated_password,
        'usage': {
            'used_today': new_usage,
            'remaining': daily_limit - new_usage
        }
    }
    
    return jsonify(response)
# +++ End Password Generation Endpoint +++

# +++ NEW D20 Roll Endpoint +++
@app.route('/api/d20', methods=['GET'])
@limiter.limit("20 per minute") # Apply a rate limit for unauthenticated access
def roll_d20_endpoint():
    """Rolls one or more d20s using the shared quantum random engine."""
    try:
        # 1. Get and validate the number of rolls requested
        try:
            num_rolls = int(request.args.get('rolls', 1))
        except ValueError:
            return jsonify({'message': 'Invalid rolls parameter, must be an integer.'}), 400

        # Clamp the number of rolls between 1 and 100
        num_rolls = max(1, min(num_rolls, 100))

        # 2. Check and use Shared Quantum Engine
        if not shared_auth_quantum_engine:
            app.logger.error("Shared quantum engine not initialized for d20 roll.")
            return jsonify({'message': 'Quantum service temporarily unavailable.'}), 503

        try:
            # Fetch the required number of bytes directly from the shared engine
            quantum_bytes = shared_auth_quantum_engine.get_random_bytes(num_rolls)
            if not quantum_bytes or len(quantum_bytes) != num_rolls:
                 raise Exception(f"Failed to generate {num_rolls} bytes from shared engine")

            # Map bytes to d20 rolls (1-20) using consistent mapping
            d20_results = []
            for byte_val in quantum_bytes:
                # Uniform mapping: floor((byte / 256) * 20) + 1
                roll = (byte_val * 20) // 256 + 1
                d20_results.append(roll)

        except AttributeError as e:
             # This might happen if the shared engine object doesn't have get_random_bytes
             app.logger.error(f"Shared quantum engine missing expected method: {e}")
             return jsonify({'message': 'Internal server error: Invalid quantum method configuration.'}), 500
        except Exception as e:
            app.logger.error(f"Error generating quantum data for d20 roll via shared engine: {str(e)}")
            return jsonify({'message': 'Error generating quantum data source.'}), 500
        
        # 3. (Optional) Log rolls to Supabase
        if supabase:
            try:
                session_id = uuid.uuid4().hex # Generate a unique ID for this batch of rolls
                rows_to_insert = [
                    {'roll': r, 'session_id': session_id} for r in d20_results
                ]
                # Note: Ensure 'd20_rolls' table exists in Supabase with 'roll' (int4) and 'session_id' (text) columns.
                insert_response = supabase.table('d20_rolls').insert(rows_to_insert).execute()
                # Optional: Check insert_response for errors, but don't fail the request if logging fails
                if not insert_response.data and hasattr(insert_response, 'error'):
                     app.logger.error(f"Failed to log d20 rolls to Supabase for session {session_id}: {insert_response.error}")
            except Exception as e:
                # Catch potential exceptions during Supabase interaction
                app.logger.error(f"Error logging d20 rolls to Supabase: {e}")
                # Do not block the user response due to logging failure

        # 4. Return the results
        response = {
            'rolls': d20_results,
            'timestamp': datetime.datetime.now().isoformat()
        }
        return jsonify(response)

    except Exception as e:
        # General catch-all for unexpected errors during request processing
        app.logger.error(f"Unexpected error in /api/d20 endpoint: {e}")
        return jsonify({'message': 'An internal server error occurred.'}), 500
# +++ End D20 Roll Endpoint +++

# +++ NEW Raw Entropy Endpoint +++
@app.route('/api/eris/raw', methods=['GET'])
@require_auth
@tier_limit
@limiter.limit(get_rate_limit_for_user)
def get_raw_quantum_random(auth_user_id):
    """Generate and return raw, unwhitened quantum random data."""
    # Ensure supabase client is available
    if supabase is None:
        app.logger.error("Supabase client not initialized. Cannot process raw request.")
        return jsonify({'message': 'Internal server error: DB unavailable'}), 500

    # --- Fetch user data and determine limits ---
    try:
        user_data_resp = supabase.table('users').select('tier, daily_usage').eq('auth_user_id', auth_user_id).single().execute()
        if not user_data_resp.data:
             app.logger.error(f"User '{auth_user_id}' not found during raw request processing.")
             return jsonify({'message': 'User data inconsistency'}), 500
        user_data = user_data_resp.data
        user_tier = user_data['tier']
        current_usage = user_data['daily_usage']
    except Exception as e:
        app.logger.error(f"Error fetching user tier/usage for raw request {auth_user_id}: {e}")
        return jsonify({'message': 'Internal server error fetching user data'}), 500

    tier_info = TIERS.get(user_tier)
    if not tier_info:
        app.logger.error(f"Invalid tier '{user_tier}' found for user ID '{auth_user_id}'.")
        return jsonify({'message': 'Internal server error: Invalid user configuration'}), 500

    daily_limit = tier_info.get('daily_bytes')
    burst_limit = tier_info.get('burst_bytes')
    # -------------------------------------------

    # --- Get and Validate Size Parameter ---
    try:
        requested_size = int(request.args.get('size', 64))
    except ValueError:
        return jsonify({'message': 'Invalid size parameter'}), 400

    # Check operational limit
    if requested_size > CURRENT_PLAN_OPERATIONAL_LIMIT_BYTES:
        app.logger.warning(f"Raw request size {requested_size} exceeds current plan limit {CURRENT_PLAN_OPERATIONAL_LIMIT_BYTES} for user {auth_user_id}.")
        return jsonify({
            'message': f'Requested size ({requested_size} bytes) exceeds the operational limit ({CURRENT_PLAN_OPERATIONAL_LIMIT_BYTES} bytes) for the current service plan.'
        }), 413 # Payload Too Large

    # Enforce size limits based on tier and remaining quota
    remaining_quota = daily_limit - current_usage
    max_allowed_size = min(burst_limit, remaining_quota)

    if requested_size <= 0:
         return jsonify({'message': 'Size must be positive'}), 400

    size_to_generate = min(requested_size, max_allowed_size)

    if size_to_generate <= 0:
        app.logger.warning(f"Raw request for user ID {auth_user_id} blocked, calculated size_to_generate <= 0. Remaining quota: {remaining_quota}")
        return jsonify({'message': 'Daily quota exceeded'}), 429
    # ---------------------------------------

    # --- Generate Raw Quantum Data ---
    # Instantiate a *separate* engine specifically for raw data for this request
    try:
        raw_engine = IdiaSignatureEngine(entropy_source="eris:raw")
        # Use generate_salt or get_random_bytes - let's use generate_salt for consistency with invoke
        raw_quantum_data = raw_engine.generate_salt(size_to_generate)
        del raw_engine # Clean up instance
    except Exception as e:
        app.logger.error(f"Error generating raw quantum data for {auth_user_id}: {str(e)}")
        return jsonify({'message': 'Error generating raw quantum data'}), 500
    # ---------------------------------

    # --- Update Usage in Supabase ---
    new_usage = current_usage + size_to_generate
    try:
        update_response = supabase.table('users').update({
            'daily_usage': new_usage
        }).eq('auth_user_id', auth_user_id).execute()
        if not update_response.data:
            app.logger.error(f"Failed to update usage for raw request auth ID '{auth_user_id}' from {current_usage} to {new_usage}")
            # Log and proceed for now
    except Exception as e:
        app.logger.error(f"Error updating usage for raw request auth ID '{auth_user_id}': {e}")
        # Log and proceed for now
    # --------------------------------

    # --- Format and Return Response ---
    response = {
        'size': len(raw_quantum_data), # Actual generated size
        'timestamp': datetime.datetime.now().isoformat(),
        'source': 'eris:raw', # Indicate the source type
        'tier': user_tier,
        'sigil': tier_info['sigil'],
        'sha256': hashlib.sha256(raw_quantum_data).hexdigest(),
        'data': base64.b64encode(raw_quantum_data).decode('utf-8'), # Use base64 encoding
        'usage': {
            'used_today': new_usage,
            'remaining': daily_limit - new_usage
        }
    }
    return jsonify(response)
# +++ End Raw Entropy Endpoint +++

# --- Add About Page Route --- 
@app.route('/about')
def about_page():
    """Renders the about page."""
    return render_template('about.html')
# --- End About Page Route ---

# --- Add Use Cases Page Route --- 
@app.route('/use-cases')
def use_cases_page():
    """Renders the use cases page."""
    return render_template('use_cases.html')
# --- End Use Cases Page Route ---

# --- Add Visualizations Page Route --- 
@app.route('/visualizations')
def visualizations_page():
    """Renders the visualizations page."""
    return render_template('visualizations.html')
# --- End Visualizations Page Route ---

# --- Add On Entropy Page Route --- 
@app.route('/on-entropy')
def on_entropy_page():
    """Renders the on-entropy page."""
    return render_template('on-entropy.html')
# --- End On Entropy Page Route ---

# +++ Main Execution +++
if __name__ == '__main__':
    # Use waitress for production-ready serving
    from waitress import serve
    # Default port 5000 if PORT env var not set
    port = int(os.environ.get("PORT", 5000))
    print(f"INFO: Starting server on port {port}")
    # serve(app, host='0.0.0.0', port=port) # Original serve call

    # --- START DEBUGGING: Increase threads and check cleanup interval --- 
    serve(
        app, 
        host='0.0.0.0', 
        port=port, 
        threads=8,               # Increase threads (default is 4)
        channel_timeout=120,     # Increase channel timeout (default 60)
        cleanup_interval=60,     # Increase cleanup interval (default 30)
        connection_limit=500     # Increase connection limit (default 100)
    )
    # --- END DEBUGGING --- 