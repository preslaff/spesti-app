# app.py

import os
import re
import json
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from google.cloud import vision
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials
from PIL import Image
import io
import openai
import googlemaps
import requests
from dotenv import load_dotenv
import pandas as pd
from rapidfuzz import fuzz, process
import time
from functools import wraps
from threading import Lock
from collections import defaultdict
import hashlib
import logging
import logging.config
from contextlib import contextmanager
from config import config

# Load environment variables from .env file
if not load_dotenv():
    print("Warning: .env file not found. Using system environment variables.")

# --- START: LOGGING CONFIGURATION ---

def setup_logging():
    """Setup structured logging configuration."""
    import sys
    
    # Configure sys.stdout to handle UTF-8 on Windows
    if sys.platform.startswith('win'):
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
            'detailed': {
                'format': '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(funcName)s(): %(message)s'
            },
            'json': {
                'format': '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s", "module": "%(module)s", "function": "%(funcName)s", "line": %(lineno)d}'
            }
        },
        'handlers': {
            'console': {
                'level': 'INFO',
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'level': 'DEBUG',
                'class': 'logging.FileHandler',
                'formatter': 'detailed',
                'filename': 'price_checker.log',
                'mode': 'a',
                'encoding': 'utf-8'
            },
            'error_file': {
                'level': 'ERROR',
                'class': 'logging.FileHandler',
                'formatter': 'json',
                'filename': 'price_checker_errors.log',
                'mode': 'a',
                'encoding': 'utf-8'
            }
        },
        'loggers': {
            'price_checker': {
                'handlers': ['console', 'file', 'error_file'],
                'level': 'DEBUG',
                'propagate': False
            },
            'cache': {
                'handlers': ['file'],
                'level': 'DEBUG',
                'propagate': False
            },
            'performance': {
                'handlers': ['file'],
                'level': 'INFO',
                'propagate': False
            }
        },
        'root': {
            'level': 'INFO',
            'handlers': ['console']
        }
    }
    
    logging.config.dictConfig(logging_config)

# Initialize logging
setup_logging()
logger = logging.getLogger('price_checker')
cache_logger = logging.getLogger('cache')
perf_logger = logging.getLogger('performance')

class PerformanceMonitor:
    """Simple performance monitoring utility."""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.lock = Lock()
    
    def record_timing(self, operation, duration):
        """Record timing for an operation."""
        with self.lock:
            self.metrics[f"{operation}_duration"].append(duration)
            perf_logger.info(f"Operation: {operation}, Duration: {duration:.3f}s")
    
    def record_event(self, event_type, data=None):
        """Record an event with optional data."""
        with self.lock:
            self.metrics[f"{event_type}_count"].append(1)
            if data:
                perf_logger.info(f"Event: {event_type}, Data: {data}")
            else:
                perf_logger.info(f"Event: {event_type}")
    
    def get_stats(self):
        """Get performance statistics."""
        with self.lock:
            stats = {}
            for key, values in self.metrics.items():
                if '_duration' in key:
                    stats[key] = {
                        'avg': sum(values) / len(values) if values else 0,
                        'min': min(values) if values else 0,
                        'max': max(values) if values else 0,
                        'count': len(values)
                    }
                else:
                    stats[key] = {'count': len(values)}
            return stats

# Global performance monitor
performance = PerformanceMonitor()

@contextmanager
def measure_time(operation_name):
    """Context manager to measure operation time."""
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        performance.record_timing(operation_name, duration)

# --- END: LOGGING CONFIGURATION ---

# Validate required environment variables
def validate_environment():
    """Validate required environment variables and provide helpful error messages."""
    required_vars = {
        'OPENAI_API_KEY': 'OpenAI API key for product name extraction',
        'SPREADSHEET_ID': 'Google Sheets ID for data storage'
    }
    
    missing_vars = []
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing_vars.append(f"- {var}: {description}")
    
    if missing_vars:
        error_msg = "Missing required environment variables:\n" + "\n".join(missing_vars)
        error_msg += "\n\nPlease create a .env file with these variables or set them in your environment."
        raise ValueError(error_msg)

# Validate environment before proceeding
validate_environment()

# INITIAL SETUP
# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Try different credential paths in order of preference
DEFAULT_CREDENTIALS_PATH = os.path.join(SCRIPT_DIR, 'gcp_credentials.json')
GCP_CREDENTIALS_PATH = os.getenv('GCP_CREDENTIALS_PATH', DEFAULT_CREDENTIALS_PATH)

# If the default doesn't work, try alternative paths
if not os.path.exists(GCP_CREDENTIALS_PATH):
    alternative_paths = [
        'gcp_credentials.json',  # Current working directory
        os.path.join(os.getcwd(), 'backend', 'gcp_credentials.json'),  # backend subdir
    ]
    
    for alt_path in alternative_paths:
        if os.path.exists(alt_path):
            GCP_CREDENTIALS_PATH = alt_path
            break
    else:
        raise FileNotFoundError(f"Google Cloud credentials file not found. Tried: {DEFAULT_CREDENTIALS_PATH}, {', '.join(alternative_paths)}")

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = GCP_CREDENTIALS_PATH
app = Flask(__name__)
CORS(app)

try:
    vision_client = vision.ImageAnnotatorClient()
except Exception as e:
    raise RuntimeError(f"Failed to initialize Google Vision client: {e}")

# Configuration from environment variables and config
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GOOGLE_MAPS_API_KEY = os.getenv('GOOGLE_MAPS_API_KEY')

# OpenAI setup for version >= 1.0.0
try:
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    raise RuntimeError(f"Failed to initialize OpenAI client: {e}")

# Google Maps setup
gmaps = None
if GOOGLE_MAPS_API_KEY:
    gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)
else:
    logger.warning("GOOGLE_MAPS_API_KEY not found. Geolocation-based store detection will be disabled.")

# Google Sheets setup - using config
SPREADSHEET_ID = os.getenv('SPREADSHEET_ID', '1z1lclOhKeNEKbSGr3CULc2dMhER3btruLC510CN24bM')

# Initialize Google Sheets client
try:
    credentials = Credentials.from_service_account_file(GCP_CREDENTIALS_PATH, scopes=config.sheets.scopes)
    sheets_service = build('sheets', 'v4', credentials=credentials)
except Exception as e:
    raise RuntimeError(f"Failed to initialize Google Sheets client: {e}")

# Global variables for product dictionary with thread safety
PRODUCT_DICTIONARY = []
PRODUCT_DICTIONARY_NORMALIZED = []
PRODUCT_DICT_LOCK = Lock()

# Caching system
class CacheManager:
    def __init__(self, default_ttl=300):  # 5 minutes default TTL
        self.cache = {}
        self.ttl_map = {}
        self.lock = Lock()
        self.default_ttl = default_ttl
        
    def get(self, key):
        with self.lock:
            if key in self.cache:
                # Check if still valid
                if time.time() < self.ttl_map[key]:
                    cache_logger.debug(f"Cache hit for key: {key}")
                    return self.cache[key]
                else:
                    # Expired, remove
                    cache_logger.debug(f"Cache expired for key: {key}")
                    del self.cache[key]
                    del self.ttl_map[key]
            cache_logger.debug(f"Cache miss for key: {key}")
            return None
    
    def set(self, key, value, ttl=None):
        with self.lock:
            if ttl is None:
                ttl = self.default_ttl
            self.cache[key] = value
            self.ttl_map[key] = time.time() + ttl
            cache_logger.debug(f"Cache set for key: {key}, TTL: {ttl}s")
    
    def invalidate(self, pattern=None):
        with self.lock:
            if pattern is None:
                self.cache.clear()
                self.ttl_map.clear()
            else:
                # Pattern-based invalidation
                keys_to_remove = [k for k in self.cache.keys() if pattern in k]
                for key in keys_to_remove:
                    del self.cache[key]
                    del self.ttl_map[key]
    
    def stats(self):
        with self.lock:
            return {
                'size': len(self.cache),
                'keys': list(self.cache.keys())
            }

# Initialize cache managers using config
sheets_cache = CacheManager(default_ttl=config.cache.sheets_cache_ttl)
price_history_cache = CacheManager(default_ttl=config.cache.price_history_ttl)
llm_cache = CacheManager(default_ttl=config.cache.llm_cache_ttl)

# Get column indices from config
columns = config.get_column_indices()

# --- START: PRODUCT DICTIONARY FUNCTIONS ---

def load_product_dictionary():
    """Load and preprocess the Bulgarian product dictionary from CSV with thread safety."""
    global PRODUCT_DICTIONARY, PRODUCT_DICTIONARY_NORMALIZED
    
    try:
        # Load the CSV file using config - make path relative to script directory
        csv_path = os.path.join(SCRIPT_DIR, config.products.PRODUCT_DICT_FILE)
        df = pd.read_csv(csv_path)
        
        # Extract product names and clean them
        product_names = df['product_name'].dropna().tolist()
        
        # Remove duplicates and empty entries
        product_names = list(set([name.strip() for name in product_names if name.strip()]))
        
        # Create normalized versions for better matching
        normalized_names = [normalize_text_for_matching(name) for name in product_names]
        
        # Thread-safe update of global variables
        with PRODUCT_DICT_LOCK:
            PRODUCT_DICTIONARY = product_names
            PRODUCT_DICTIONARY_NORMALIZED = normalized_names
        
        logger.info(f"Loaded {len(product_names)} products from dictionary")
        
    except Exception as e:
        logger.error(f"Error loading product dictionary: {e}")
        with PRODUCT_DICT_LOCK:
            PRODUCT_DICTIONARY = []
            PRODUCT_DICTIONARY_NORMALIZED = []

def normalize_text_for_matching(text):
    """Normalize text for better fuzzy matching (handles common OCR errors)."""
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower().strip()
    
    # Extreme OCR misrecognition patterns (specific cases)
    extreme_replacements = {
        'kloomema': 'кюфтета',
        'klyfteta': 'кюфтета',
        'kyfteta': 'кюфтета',
        'klqfteta': 'кюфтета',
        'ke6an4eta': 'кебапчета',
        'kebapчeta': 'кебапчета',
        'kebap4eta': 'кебапчета',
    }
    
    for wrong, correct in extreme_replacements.items():
        if wrong in text:
            text = text.replace(wrong, correct)
    
    # Common Latin->Cyrillic OCR error corrections
    cyrillic_replacements = {
        'a': 'а', 'e': 'е', 'p': 'р', 'o': 'о', 'c': 'с', 
        'x': 'х', 'y': 'у', 't': 'т', 'n': 'н', 'm': 'м', 'k': 'к',
        'h': 'н', 'b': 'в', 'u': 'у', 'i': 'и', 'j': 'ј', 'l': 'л',
        'q': 'ю', 'w': 'ш', 'f': 'ф', 'g': 'г', 'd': 'д', 'z': 'з',
        '6': 'б', '4': 'ч', '3': 'з', '9': 'д'
    }
    
    for latin, cyrillic in cyrillic_replacements.items():
        text = text.replace(latin, cyrillic)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def fuzzy_match_product(ocr_text, threshold=None, limit=None):
    """
    Find the best fuzzy match for OCR text in the product dictionary with thread safety.
    Returns tuple: (best_match, confidence_score, all_matches)
    """
    # Use config defaults if not provided
    if threshold is None:
        threshold = config.products.FUZZY_THRESHOLD
    if limit is None:
        limit = config.products.FUZZY_LIMIT
        
    # Thread-safe access to global dictionaries
    with PRODUCT_DICT_LOCK:
        if not PRODUCT_DICTIONARY or not ocr_text:
            return None, 0, []
        
        # Create local copies for processing
        local_dict = PRODUCT_DICTIONARY.copy()
        local_normalized = PRODUCT_DICTIONARY_NORMALIZED.copy()
    
    # Normalize the OCR text
    normalized_ocr = normalize_text_for_matching(ocr_text)
    
    # First pass: Try different matching strategies
    strategies = [
        ('exact', fuzz.ratio),
        ('partial', fuzz.partial_ratio), 
        ('token_sort', fuzz.token_sort_ratio),
        ('token_set', fuzz.token_set_ratio)
    ]
    
    best_matches = []
    
    for strategy_name, scorer in strategies:
        # Match against normalized dictionary
        matches = process.extract(
            normalized_ocr, 
            local_normalized, 
            scorer=scorer, 
            limit=limit
        )
        
        for match, score, _ in matches:
            if score >= threshold:
                # Find the original product name
                try:
                    original_idx = local_normalized.index(match)
                    original_name = local_dict[original_idx]
                    best_matches.append((original_name, score, strategy_name))
                except ValueError:
                    continue
    
    # Second pass: If no good matches, try with original OCR text at lower threshold
    if not best_matches and threshold > 50:
        logger.debug(f"No matches found for normalized '{normalized_ocr}', trying original '{ocr_text}' with lower threshold")
        for strategy_name, scorer in strategies:
            matches = process.extract(
                ocr_text, 
                local_dict, 
                scorer=scorer, 
                limit=limit
            )
            
            for match, score, _ in matches:
                if score >= (threshold - 20):  # Lower threshold for second pass
                    best_matches.append((match, score, f"{strategy_name}_original"))
    
    if not best_matches:
        return None, 0, []
    
    # Sort by score and return best match
    best_matches.sort(key=lambda x: x[1], reverse=True)
    best_match = best_matches[0]
    
    logger.debug(f"Best match for '{ocr_text}' -> '{best_match[0]}' (score: {best_match[1]}, strategy: {best_match[2]})")
    
    return best_match[0], best_match[1], best_matches[:limit]

def enhance_product_candidates_with_dictionary(candidates):
    """Enhance product candidates with dictionary matches."""
    enhanced_candidates = []
    
    for candidate in candidates:
        # Get original candidate info
        enhanced_candidate = candidate.copy()
        
        # Try fuzzy matching with lower threshold for extreme OCR errors
        match, confidence, all_matches = fuzzy_match_product(candidate["text"], threshold=config.products.FUZZY_FALLBACK_THRESHOLD)
        
        if match and confidence > 80:
            # High confidence dictionary match - boost this candidate
            enhanced_candidate["dictionary_match"] = match
            enhanced_candidate["dictionary_confidence"] = confidence
            enhanced_candidate["dictionary_all_matches"] = all_matches
            
            # Boost prominence score for dictionary matches
            enhanced_candidate["weighted_area"] *= 1.5
            enhanced_candidate["has_dictionary_match"] = True
            
            logger.debug(f"Dictionary match: '{candidate['text']}' -> '{match}' (confidence: {confidence})")
        else:
            enhanced_candidate["dictionary_match"] = None
            enhanced_candidate["dictionary_confidence"] = 0
            enhanced_candidate["has_dictionary_match"] = False
        
        enhanced_candidates.append(enhanced_candidate)
    
    return enhanced_candidates

# --- END: PRODUCT DICTIONARY FUNCTIONS ---

# --- START: API RATE LIMITING UTILITIES ---

def retry_on_rate_limit(max_retries=3, base_delay=1):
    """Decorator to retry API calls with exponential backoff on rate limits."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_str = str(e).lower()
                    
                    # Check for rate limit indicators
                    if any(indicator in error_str for indicator in [
                        'quota', 'rate limit', 'too many requests', '429', 
                        'resource exhausted', 'deadline exceeded'
                    ]):
                        if attempt < max_retries:
                            delay = base_delay * (2 ** attempt)  # Exponential backoff
                            print(f"Rate limit hit, retrying in {delay}s (attempt {attempt + 1}/{max_retries + 1})")
                            time.sleep(delay)
                            continue
                        else:
                            print(f"Max retries exceeded for {func.__name__}")
                            raise RuntimeError(f"API rate limit exceeded after {max_retries} retries: {e}")
                    else:
                        # Non-rate-limit error, don't retry
                        raise e
            return None
        return wrapper
    return decorator

# --- END: API RATE LIMITING UTILITIES ---

# --- START: INPUT VALIDATION ---

class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass

def validate_file_upload(file):
    """Validate uploaded file for security and correctness."""
    if not file:
        raise ValidationError("No file provided")
    
    if file.filename == '':
        raise ValidationError("No file selected")
    
    # Check file size using config
    file.seek(0, 2)  # Seek to end
    size = file.tell()
    file.seek(0)  # Reset to beginning
    
    max_size = config.validation.max_file_size_mb * 1024 * 1024
    min_size = config.validation.min_file_size_kb * 1024
    
    if size > max_size:
        raise ValidationError(f"File too large. Maximum size is {config.validation.max_file_size_mb}MB")
    
    if size < min_size:
        raise ValidationError(f"File too small. Minimum size is {config.validation.min_file_size_kb}KB")
    
    # Check file extension using config
    file_ext = os.path.splitext(file.filename.lower())[1]
    
    if file_ext not in config.validation.allowed_image_extensions:
        raise ValidationError(f"Invalid file type. Allowed: {', '.join(config.validation.allowed_image_extensions)}")
    
    # Check MIME type
    file_content = file.read(512)  # Read first 512 bytes for header check
    file.seek(0)  # Reset
    
    # Basic magic number validation using config
    image_signatures = config.security.IMAGE_SIGNATURES
    
    is_valid_image = False
    for signature, format_name in image_signatures.items():
        if file_content.startswith(signature):
            is_valid_image = True
            break
    
    if not is_valid_image:
        raise ValidationError("File does not appear to be a valid image")
    
    return True

def validate_coordinates(latitude, longitude):
    """Validate GPS coordinates."""
    if latitude is None and longitude is None:
        return None, None  # Both None is acceptable
    
    if latitude is None or longitude is None:
        raise ValidationError("Both latitude and longitude must be provided together")
    
    try:
        lat = float(latitude)
        lng = float(longitude)
    except (ValueError, TypeError):
        raise ValidationError("Coordinates must be valid numbers")
    
    # Basic range validation
    if not (-90 <= lat <= 90):
        raise ValidationError("Latitude must be between -90 and 90 degrees")
    
    if not (-180 <= lng <= 180):
        raise ValidationError("Longitude must be between -180 and 180 degrees")
    
    # Additional validation for Bulgarian region using config
    if not (config.validation.min_latitude <= lat <= config.validation.max_latitude and 
            config.validation.min_longitude <= lng <= config.validation.max_longitude):
        print(f"Warning: Coordinates outside Bulgaria region: {lat}, {lng}")
        # Don't reject, just warn
    
    return lat, lng

def validate_store_name(store_name):
    """Validate store name input."""
    if not store_name:
        return "Неизвестен магазин"
    
    # Sanitize input
    store_name = store_name.strip()
    
    # Check length using config
    if len(store_name) > config.validation.max_store_name_length:
        raise ValidationError(f"Store name too long (max {config.validation.max_store_name_length} characters)")
    
    # Check for valid characters (allow Cyrillic, Latin, numbers, basic punctuation)
    import re
    if not re.match(r'^[а-яА-ЯёЁa-zA-Z0-9\s\-\.]+$', store_name):
        raise ValidationError("Store name contains invalid characters")
    
    return store_name

def validate_price_range(price, field_name="price"):
    """Validate price is within reasonable range."""
    if price is None:
        raise ValidationError(f"{field_name} cannot be None")
    
    try:
        price_val = float(price)
    except (ValueError, TypeError):
        raise ValidationError(f"{field_name} must be a valid number")
    
    if price_val < 0:
        raise ValidationError(f"{field_name} cannot be negative")
    
    if price_val > config.currency.max_price:
        raise ValidationError(f"{field_name} seems unreasonably high (max {config.currency.max_price})")
    
    if price_val < config.currency.min_price:
        raise ValidationError(f"{field_name} too small (min {config.currency.min_price})")
    
    return price_val

def sanitize_text_input(text, max_length=None):
    """Sanitize text input to prevent injection attacks."""
    if not text:
        return ""
    
    # Use config default if not provided
    if max_length is None:
        max_length = config.validation.max_text_length
    
    # Convert to string and strip
    text = str(text).strip()
    
    # Check length
    if len(text) > max_length:
        text = text[:max_length]
    
    # Remove potential script tags and SQL injection patterns using config
    dangerous_patterns = config.security.DANGEROUS_PATTERNS
    
    text_lower = text.lower()
    for pattern in dangerous_patterns:
        if pattern in text_lower:
            raise ValidationError(f"Text contains potentially dangerous content: {pattern}")
    
    return text

# --- END: INPUT VALIDATION ---

# --- START: SPATIAL ANALYSIS FUNCTIONS ---

def get_block_height(vertices):
    """Calculates the height of a bounding box."""
    return max(v.y for v in vertices) - min(v.y for v in vertices)

def get_block_width(vertices):
    """Calculates the width of a bounding box."""
    return max(v.x for v in vertices) - min(v.x for v in vertices)

def get_block_center(vertices):
    """Gets the center point of a bounding box."""
    x_coords = [v.x for v in vertices]
    y_coords = [v.y for v in vertices]
    return {
        'x': sum(x_coords) / len(x_coords),
        'y': sum(y_coords) / len(y_coords)
    }

def is_price_like(text):
    """Checks if a string resembles a price or currency-related noise."""
    text = text.strip().lower()
    # Matches numbers like 12,99, 1.45, or text with currency symbols
    if re.search(r'^\d+([.,]\d{1,2})?$', text):
        return True
    if re.search(r'(лв|bgn|eur|€|st|ct|ст)', text):
        return True
    # Weight/quantity indicators
    if re.search(r'(кг|kg|гр|gr|g|мл|ml|л|l|бр|br|шт|pc)$', text):
        return True
    # Barcode patterns
    if re.search(r'^\d{8,13}$', text):
        return True
    return False

def is_store_info(text):
    """Detects store-related information."""
    text = text.strip().lower()
    store_keywords = ['kaufland', 'lidl', 'metro', 'billa', 'fantastico', 't-market', 'пикадили', 'cba']
    for keyword in store_keywords:
        if keyword in text:
            return True
    return False

def contains_excluded_phrase(text):
    """Check if text contains any excluded phrases."""
    text_lower = text.lower().strip()
    for phrase in config.products.EXCLUDED_PHRASES:
        if phrase.lower() in text_lower:
            return True
    return False

def remove_excluded_phrases_from_text(text):
    """Remove excluded phrases from text while preserving structure."""
    if not text:
        return text
    
    # Split into lines to preserve structure
    lines = text.split('\n')
    filtered_lines = []
    
    for line in lines:
        line_lower = line.lower().strip()
        should_exclude = False
        
        # Check if entire line contains excluded phrases
        for phrase in config.products.EXCLUDED_PHRASES:
            if phrase.lower() in line_lower:
                should_exclude = True
                break
        
        if not should_exclude:
            filtered_lines.append(line)
    
    return '\n'.join(filtered_lines)

def get_text_style_weight(annotation):
    """
    Analyze text style properties to determine importance weight.
    Uses bounding box height as proxy for font size/emphasis.
    """
    weight = 1.0
    
    # Get bounding box height as proxy for font size
    if hasattr(annotation, 'bounding_poly') and annotation.bounding_poly.vertices:
        vertices = annotation.bounding_poly.vertices
        height = get_block_height(vertices)
        
        # Larger text gets higher weight (height-based emphasis detection)
        if height > 50:  # Very large text (likely headers/product names)
            weight *= 2.0
        elif height > 35:  # Large text (likely important)
            weight *= 1.6
        elif height > 25:  # Medium-large text
            weight *= 1.3
        elif height > 15:  # Standard text
            weight *= 1.0
        else:  # Very small text (likely fine print)
            weight *= 0.7
    
    # Check if we have access to text style information
    if hasattr(annotation, 'property') and annotation.property:
        # Google Vision API sometimes provides text style info
        if hasattr(annotation.property, 'detected_break'):
            break_type = annotation.property.detected_break.type_
            # Text with strong breaks (paragraphs) might be more important
            if break_type in ['LINE_BREAK', 'EOL_SURE_SPACE']:
                weight *= 1.1
        
        # Check for font properties if available
        if hasattr(annotation.property, 'detected_text') and annotation.property.detected_text:
            # This is rarely available but worth checking
            detected_text = annotation.property.detected_text
            if hasattr(detected_text, 'is_bold') and detected_text.is_bold:
                weight *= 1.8
            if hasattr(detected_text, 'font_size') and detected_text.font_size:
                # Larger font sizes get higher weight
                if detected_text.font_size > 14:
                    weight *= 1.4
                elif detected_text.font_size > 12:
                    weight *= 1.2
    
    return weight

def is_date_like(text):
    """Detects date patterns."""
    text = text.strip()
    if re.search(r'\d{1,2}[./]\d{1,2}[./]\d{2,4}', text):
        return True
    return False

def analyze_text_blocks_spatial(annotations):
    """
    Analyzes text blocks with spatial relationships to identify product name candidates.
    """
    if not annotations or len(annotations) < 2:
        return []

    blocks = []
    for annotation in annotations[1:]:  # Skip the first full-text annotation
        text = annotation.description.strip()
        if len(text) > 1:
            vertices = annotation.bounding_poly.vertices
            
            # Get style weight for this text block
            style_weight = get_text_style_weight(annotation)
            
            block_info = {
                "text": text,
                "height": get_block_height(vertices),
                "width": get_block_width(vertices),
                "center": get_block_center(vertices),
                "area": get_block_height(vertices) * get_block_width(vertices),
                "vertices": vertices,
                "style_weight": style_weight,
                "is_price_like": is_price_like(text),
                "is_store_info": is_store_info(text),
                "is_date_like": is_date_like(text),
                "has_excluded_phrase": contains_excluded_phrase(text)
            }
            blocks.append(block_info)
    
    # Filter out obvious non-product-name blocks
    product_candidates = []
    for block in blocks:
        # Skip blocks with excluded phrases, prices, store info, dates
        if not (block["is_price_like"] or block["is_store_info"] or 
                block["is_date_like"] or block["has_excluded_phrase"]):
            # Apply style weight to the area score
            block["weighted_area"] = block["area"] * block["style_weight"]
            product_candidates.append(block)
    
    # Sort by weighted area (style-adjusted visual prominence)
    product_candidates.sort(key=lambda x: x["weighted_area"], reverse=True)
    
    return product_candidates

def find_connected_text_blocks(candidates, max_distance=50):
    """
    Groups text blocks that are spatially close to each other (likely multi-line product names).
    """
    if not candidates:
        return []
    
    groups = []
    used_indices = set()
    
    for i, block in enumerate(candidates):
        if i in used_indices:
            continue
            
        # Start a new group with this block
        group = [block]
        used_indices.add(i)
        
        # Find nearby blocks
        for j, other_block in enumerate(candidates):
            if j in used_indices or j == i:
                continue
            
            # Calculate distance between block centers
            dx = abs(block["center"]["x"] - other_block["center"]["x"])
            dy = abs(block["center"]["y"] - other_block["center"]["y"])
            
            # If blocks are close vertically (multi-line) or horizontally (continuation)
            if dy < max_distance or (dx < max_distance and dy < block["height"]):
                group.append(other_block)
                used_indices.add(j)
        
        if group:
            groups.append(group)
    
    return groups

def extract_product_name_advanced(annotations):
    """
    Enhanced product name extraction with spatial analysis and LLM verification.
    Returns tuple: (product_name, bounding_box_vertices)
    """
    if not annotations or len(annotations) < 2:
        return "Неизвестен продукт", None

    # Get all text lines for context, filtering out excluded phrases
    full_text = annotations[0].description
    filtered_full_text = remove_excluded_phrases_from_text(full_text)
    
    # Analyze spatial relationships
    product_candidates = analyze_text_blocks_spatial(annotations)
    
    if not product_candidates:
        return "Неизвестен продукт", None
    
    # Enhance candidates with dictionary matching
    product_candidates = enhance_product_candidates_with_dictionary(product_candidates)
    
    # Group spatially connected blocks
    text_groups = find_connected_text_blocks(product_candidates)
    
    # Prepare candidates for LLM analysis
    llm_candidates = []
    
    # Add individual blocks (sorted by prominence)
    for block in product_candidates[:8]:  # Top 8 most prominent blocks
        candidate_info = {
            "text": block["text"],
            "type": "individual",
            "prominence_score": block["weighted_area"],  # Use weighted area
            "style_weight": block["style_weight"],
            "vertices": block["vertices"]
        }
        
        # Add dictionary match information if available
        if block.get("has_dictionary_match", False):
            candidate_info["dictionary_match"] = block["dictionary_match"]
            candidate_info["dictionary_confidence"] = block["dictionary_confidence"]
        
        llm_candidates.append(candidate_info)
    
    # Add grouped blocks (potential multi-line names)
    for group in text_groups:
        if len(group) > 1:  # Only multi-block groups
            # Sort blocks in group by vertical position (top to bottom)
            group.sort(key=lambda x: x["center"]["y"])
            combined_text = " ".join([block["text"] for block in group])
            total_weighted_area = sum([block["weighted_area"] for block in group])
            avg_style_weight = sum([block["style_weight"] for block in group]) / len(group)
            
            # Create combined bounding box for grouped text
            all_vertices = []
            for block in group:
                all_vertices.extend(block["vertices"])
            
            # Try dictionary matching on the combined text
            group_match, group_confidence, group_all_matches = fuzzy_match_product(combined_text, threshold=60)
            
            group_candidate = {
                "text": combined_text,
                "type": "grouped",
                "prominence_score": total_weighted_area,
                "component_count": len(group),
                "style_weight": avg_style_weight,
                "vertices": all_vertices,
                "group_blocks": group
            }
            
            # Add dictionary match information if available
            if group_match and group_confidence > 75:
                group_candidate["dictionary_match"] = group_match
                group_candidate["dictionary_confidence"] = group_confidence
                group_candidate["has_dictionary_match"] = True
                # Boost prominence for dictionary matches
                group_candidate["prominence_score"] *= 1.8  # Higher boost for grouped matches
                print(f"Dictionary match for group: '{combined_text}' -> '{group_match}' (confidence: {group_confidence})")
            else:
                group_candidate["dictionary_match"] = None
                group_candidate["dictionary_confidence"] = 0
                group_candidate["has_dictionary_match"] = False
            
            llm_candidates.append(group_candidate)
    
    # Sort all candidates by prominence
    llm_candidates.sort(key=lambda x: x["prominence_score"], reverse=True)
    
    # Prepare enhanced prompt for LLM
    # Create list of excluded phrases for LLM prompt
    excluded_phrases_text = ', '.join(config.products.EXCLUDED_PHRASES)
    
    prompt = f"""
    You are analyzing a Bulgarian price label to extract the product name. Below are text candidates extracted from the label, including both individual text blocks and spatially-grouped multi-line text combinations. The final product name MUST be written ONLY in Bulgarian Cyrillic script. If the OCR text contains Latin characters that look like Bulgarian words, convert them to proper Bulgarian Cyrillic.

    FULL LABEL TEXT (for context - promotional phrases already filtered):
    {filtered_full_text}

    CANDIDATE TEXTS (ordered by visual prominence):
    {json.dumps([{
        "text": c["text"], 
        "type": c["type"],
        "prominence_score": c["prominence_score"],
        "dictionary_match": c.get("dictionary_match", None),
        "dictionary_confidence": c.get("dictionary_confidence", 0)
    } for c in llm_candidates[:10]], ensure_ascii=False, indent=2)}

    ANALYSIS INSTRUCTIONS:
    1. Identify which candidate represents the complete product name
    2. CRITICAL: Strongly prefer "grouped" type candidates over "individual" ones - they represent complete multi-word product names
    3. Multi-line product names are common - look for candidates that combine related text blocks
    4. Ignore: prices, weights (кг, гр, мл), store names, dates, barcodes, promotional text
    5. CRITICAL: Completely avoid and reject ANY text containing these promotional phrases: {excluded_phrases_text}
    6. IMPORTANT: If a candidate has a dictionary_match with high dictionary_confidence (>75), strongly prefer it as it matches a real Bulgarian product
    7. The product name should be meaningful and coherent in Bulgarian
    8. Prefer candidates that appear to be emphasized/bold (higher style weight and prominence_score)
    9. If multiple candidates seem valid, choose the most complete/descriptive one (grouped > individual)
    10. Clean up the final name (proper capitalization, remove extra spaces)
    11. If choosing a dictionary match, use the dictionary_match text, not the original OCR text

    EXAMPLES:
    - "ЛУКАНКА ОРЕХИТЕ ТРАДИЦИОННА" → "Луканка Орехите Традиционна"
    - "МЛЕЧЕН ШОКОЛАД MILKA" → "Млечен Шоколад Milka"
    - "ХЛЯБ ЧЕРЕН ПЪЛНОЗЪРНЕСТ" → "Хляб Черен Пълнозърнест"

    Respond with a JSON object containing:
    {{
        "product_name": "extracted product name",
        "confidence": "high/medium/low",
        "reasoning": "brief explanation of your choice"
    }}
    """

    # Create cache key based on prompt content
    cache_key = hashlib.md5(prompt.encode('utf-8')).hexdigest()
    
    # Check LLM cache first
    cached_result = llm_cache.get(cache_key)
    if cached_result:
        print(f"LLM cache hit for product extraction")
        result_json = cached_result
    else:
        print(f"LLM cache miss - making API call")
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing Bulgarian product labels and extracting product names from OCR text."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.1
            )
            
            result_json = json.loads(response.choices[0].message.content)
            
            # Cache the result
            llm_cache.set(cache_key, result_json)
            print(f"LLM response cached with key: {cache_key[:8]}...")
            
        except Exception as api_error:
            print(f"LLM API error: {api_error}")
            raise api_error
    
    try:
        product_name = result_json.get("product_name", "").strip()
        confidence = result_json.get("confidence", "low")
        reasoning = result_json.get("reasoning", "")
        
        print(f"LLM Product Extraction - Name: {product_name}, Confidence: {confidence}, Reasoning: {reasoning}")
        
        # Find the matching candidate to get bounding box
        selected_candidate = None
        if product_name and product_name.lower() not in ["неизвестен продукт", "unknown", ""]:
            # Try to find the candidate that matches the selected product name
            for candidate in llm_candidates:
                if candidate["text"].lower() == product_name.lower():
                    selected_candidate = candidate
                    break
        
        # Fallback if LLM returns empty or invalid result
        if not product_name or product_name.lower() in ["неизвестен продукт", "unknown", ""]:
            # Try to use dictionary matches first
            dictionary_candidates = [c for c in llm_candidates if c.get("dictionary_match")]
            if dictionary_candidates:
                best_dict_candidate = max(dictionary_candidates, key=lambda x: x.get("dictionary_confidence", 0))
                product_name = best_dict_candidate["dictionary_match"]
                selected_candidate = best_dict_candidate
                print(f"Using dictionary fallback: {product_name}")
            elif llm_candidates:
                product_name = llm_candidates[0]["text"]
                selected_candidate = llm_candidates[0]
            else:
                product_name = "Неизвестен продукт"
                selected_candidate = None
        
        # If still no match, use the first candidate
        if not selected_candidate and llm_candidates:
            selected_candidate = llm_candidates[0]
        
        # Extract bounding box vertices
        bounding_box = None
        if selected_candidate and "vertices" in selected_candidate:
            vertices = selected_candidate["vertices"]
            if selected_candidate["type"] == "grouped":
                # For grouped text, create a bounding box that encompasses all blocks
                x_coords = [v.x for v in vertices]
                y_coords = [v.y for v in vertices]
                bounding_box = [
                    {'x': min(x_coords), 'y': min(y_coords)},
                    {'x': max(x_coords), 'y': min(y_coords)},
                    {'x': max(x_coords), 'y': max(y_coords)},
                    {'x': min(x_coords), 'y': max(y_coords)}
                ]
            else:
                # For individual text blocks
                bounding_box = [{'x': v.x, 'y': v.y} for v in vertices]
        
        return product_name, bounding_box

    except Exception as e:
        print(f"LLM extraction error: {e}")
        
        # Enhanced fallback strategy
        if llm_candidates:
            best_candidate = None
            best_score = 0
            
            for candidate in llm_candidates:
                text = candidate["text"]
                score = candidate["prominence_score"]
                
                # Boost score for longer, more descriptive text
                if len(text) > 10:
                    score *= 1.5
                
                # Boost score for grouped (multi-line) candidates
                if candidate["type"] == "grouped":
                    score *= 1.3
                
                # Boost score for text with higher style weight (bold/emphasized)
                style_weight = candidate.get("style_weight", 1.0)
                if style_weight > 1.2:
                    score *= 1.2
                
                # Penalize very short text
                if len(text) < 4:
                    score *= 0.5
                
                if score > best_score:
                    best_score = score
                    best_candidate = candidate
            
            if best_candidate:
                # Extract bounding box for fallback candidate
                bounding_box = None
                if "vertices" in best_candidate:
                    vertices = best_candidate["vertices"]
                    if best_candidate["type"] == "grouped":
                        # For grouped text, create a bounding box that encompasses all blocks
                        x_coords = [v.x for v in vertices]
                        y_coords = [v.y for v in vertices]
                        bounding_box = [
                            {'x': min(x_coords), 'y': min(y_coords)},
                            {'x': max(x_coords), 'y': min(y_coords)},
                            {'x': max(x_coords), 'y': max(y_coords)},
                            {'x': min(x_coords), 'y': max(y_coords)}
                        ]
                    else:
                        # For individual text blocks
                        bounding_box = [{'x': v.x, 'y': v.y} for v in vertices]
                
                return best_candidate["text"], bounding_box
            else:
                return "Неизвестен продукт", None
        
        return "Неизвестен продукт", None

# --- END: SPATIAL ANALYSIS FUNCTIONS ---

# --- START: UTILITY FUNCTIONS ---

def get_cached_sheet_data():
    """Get sheet data with caching."""
    cache_key = f"sheet_data_{SPREADSHEET_ID}"
    cached_data = sheets_cache.get(cache_key)
    
    if cached_data:
        logger.debug("Using cached sheet data")
        performance.record_event("sheets_cache_hit")
        return cached_data
    
    try:
        logger.debug("Fetching fresh sheet data")
        performance.record_event("sheets_cache_miss")
        
        with measure_time("sheets_api_call"):
            result = sheets_service.spreadsheets().values().get(
                spreadsheetId=SPREADSHEET_ID, 
                range=config.sheets.range_name
            ).execute()
            values = result.get('values', [])
        
        # Cache the data
        sheets_cache.set(cache_key, values)
        logger.info(f"Fetched {len(values)} rows from sheets")
        return values
    except Exception as e:
        logger.error(f"Error fetching sheet data: {e}")
        performance.record_event("sheets_api_error")
        return []

@retry_on_rate_limit(max_retries=2, base_delay=1)
def find_last_price_in_sheets(product_name):
    """Find the last recorded price for a given product with caching."""
    # Check price history cache first
    cache_key = f"price_history_{product_name.lower()}"
    cached_result = price_history_cache.get(cache_key)
    
    if cached_result:
        print(f"Using cached price history for {product_name}")
        return cached_result
    
    try:
        values = get_cached_sheet_data()
        if not values or len(values) <= 1: 
            return None
        
        # Search through data (most recent first)
        for row in reversed(values[1:]):
            if len(row) > columns['bgn_price'] and len(row) > columns['product_name']:
                if row[columns['product_name']].strip().lower() == product_name.lower():
                    result = {
                        "last_price_bgn": float(row[columns['bgn_price']]), 
                        "last_scan_date": row[columns['timestamp']].split(' ')[0]
                    }
                    # Cache the result
                    price_history_cache.set(cache_key, result)
                    return result
        
        # Cache negative result to avoid repeated searches
        price_history_cache.set(cache_key, None)
        return None
    except Exception as e:
        print(f"Error finding historical price: {e}")
        return None

def calculate_area(vertices):
    """Calculate the area of a bounding box."""
    x_coords = [v.x for v in vertices]
    y_coords = [v.y for v in vertices]
    return (max(x_coords) - min(x_coords)) * (max(y_coords) - min(y_coords))

def parse_price_string(s):
    """Parse a price string into a float."""
    if not s: 
        return None
    try: 
        return float(s.replace(',', '.'))
    except (ValueError, TypeError): 
        return None

def get_fallback_coordinates(store_name, latitude=None, longitude=None):
    """
    Get fallback coordinates when GPS is not available.
    Priority order:
    1. Use provided GPS coordinates if valid
    2. Use store-specific default coordinates
    3. Use Sofia city center as ultimate fallback
    """
    # If we have valid GPS coordinates, use them
    if latitude is not None and longitude is not None:
        try:
            lat = float(latitude)
            lng = float(longitude)
            
            # Enhanced coordinate validation
            # Check if coordinates are in reasonable Bulgarian bounds
            if (41.2 <= lat <= 44.2 and 22.3 <= lng <= 28.6):
                # Additional check: not in water or obviously invalid locations
                # Sofia metropolitan area: more precise bounds
                if (42.0 <= lat <= 43.0 and 23.0 <= lng <= 24.0):
                    return lat, lng
                # Other major Bulgarian cities (Plovdiv, Varna, Burgas, etc.)
                elif (41.5 <= lat <= 44.0 and 23.0 <= lng <= 28.0):
                    return lat, lng
                # Border regions - be more cautious
                elif is_valid_bulgaria_location(lat, lng):
                    return lat, lng
        except (ValueError, TypeError):
            print(f"Invalid coordinate format: lat={latitude}, lng={longitude}")
    
    # Try store-specific coordinates using config
    if store_name and store_name in config.stores.STORE_DEFAULT_COORDINATES:
        coords = config.stores.STORE_DEFAULT_COORDINATES[store_name]
        return coords['lat'], coords['lng']
    
    # Ultimate fallback using config
    return config.stores.DEFAULT_LOCATION['lat'], config.stores.DEFAULT_LOCATION['lng']

def is_valid_bulgaria_location(lat, lng):
    """
    Additional validation for Bulgarian locations using known invalid areas.
    Excludes major water bodies, neighboring countries, etc.
    """
    # Exclude Black Sea (rough bounds)
    if lat >= 42.0 and lng >= 27.5:
        return False
    
    # Exclude obvious neighboring countries
    # Serbia/North Macedonia (west of Bulgaria)
    if lng < 22.5:
        return False
    
    # Turkey (south of Bulgaria)  
    if lat < 41.5 and lng > 26.0:
        return False
        
    # Romania (north, but allow border regions)
    if lat > 44.0:
        return False
        
    return True

def add_coordinate_spread(latitude, longitude, existing_coordinates, min_distance=0.001):
    """
    Add small random offset to coordinates if they're too close to existing ones.
    This prevents overlapping markers on the map.
    """
    import random
    
    # Check if coordinates are too close to existing ones
    for existing_lat, existing_lng in existing_coordinates:
        if existing_lat and existing_lng:
            distance = ((latitude - existing_lat) ** 2 + (longitude - existing_lng) ** 2) ** 0.5
            if distance < min_distance:
                # Add small random offset
                latitude += random.uniform(-0.002, 0.002)
                longitude += random.uniform(-0.002, 0.002)
                break
    
    return latitude, longitude

@retry_on_rate_limit(max_retries=2, base_delay=2)
def find_nearby_stores(latitude, longitude, radius=1000):
    """Find nearby stores using Google Places API (New) via direct HTTP requests."""
    if not GOOGLE_MAPS_API_KEY:
        return []
    
    try:
        # Use the new Places API (New) Nearby Search endpoint
        url = "https://places.googleapis.com/v1/places:searchNearby"
        
        headers = {
            'Content-Type': 'application/json',
            'X-Goog-Api-Key': GOOGLE_MAPS_API_KEY,
            'X-Goog-FieldMask': 'places.displayName,places.rating,places.formattedAddress'
        }
        
        data = {
            "includedTypes": ["grocery_store", "supermarket"],
            "maxResultCount": config.api.PLACES_MAX_RESULTS,
            "locationRestriction": {
                "circle": {
                    "center": {
                        "latitude": latitude,
                        "longitude": longitude
                    },
                    "radius": radius
                }
            },
            "languageCode": "bg"
        }
        
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code != 200:
            print(f"Places API error: {response.status_code} - {response.text}")
            return []
        
        result = response.json()
        nearby_stores = []
        
        for place in result.get('places', []):
            store_name = place.get('displayName', {}).get('text', '').lower()
            
            # Check if the store name matches our known store list using config
            for store_option in config.stores.STORE_OPTIONS:
                if store_option.lower() in store_name:
                    nearby_stores.append({
                        'name': store_option,
                        'distance': place.get('formattedAddress', ''),
                        'rating': place.get('rating', 0)
                    })
                    break
        
        # Sort by rating (higher is better)
        nearby_stores.sort(key=lambda x: x['rating'], reverse=True)
        return nearby_stores
        
    except Exception as e:
        print(f"Error finding nearby stores: {e}")
        return []

def extract_store_name_from_text(ocr_text, latitude=None, longitude=None):
    """Extract store name from OCR text, with optional geolocation fallback."""
    if not ocr_text: 
        return "Неизвестен магазин"
    
    lines = ocr_text.split('\n')
    store_keywords = [keyword.lower() for keyword in config.stores.STORE_KEYWORDS]
    
    # First try to find store name in OCR text
    for line in lines:
        for keyword in store_keywords:
            if keyword in line.strip().lower(): 
                return keyword.title()
    
    # If not found in OCR and we have geolocation, try nearby stores
    if latitude is not None and longitude is not None:
        nearby_stores = find_nearby_stores(latitude, longitude)
        if nearby_stores:
            print(f"Found nearby stores: {[store['name'] for store in nearby_stores]}")
            return nearby_stores[0]['name']  # Return the highest-rated nearby store
    
    return "Неизвестен магазин"

@retry_on_rate_limit(max_retries=2, base_delay=1)
def save_to_sheets(product_name, store_name, bgn_price, eur_price, expected_eur, status, latitude=None, longitude=None):
    """Save the price verification result to Google Sheets."""
    try:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        difference = round(eur_price - expected_eur, 4)
        values = [[timestamp, product_name, store_name, bgn_price, eur_price, expected_eur, difference, status, latitude, longitude]]
        body = {'values': values, 'majorDimension': 'ROWS'}
        
        sheets_service.spreadsheets().values().append(
            spreadsheetId=SPREADSHEET_ID, 
            range=config.sheets.range_name, 
            valueInputOption='RAW', 
            insertDataOption='INSERT_ROWS', 
            body=body
        ).execute()
        
        # Invalidate caches after successful write
        sheets_cache.invalidate()
        price_history_cache.invalidate(f"price_history_{product_name.lower()}")
        print("Caches invalidated after sheet update")
        
        return True
    except Exception as e:
        print(f"Error saving to sheets: {e}")
        return False

def setup_sheet_headers():
    """Setup the headers for the Google Sheet."""
    try:
        headers = [['Timestamp', 'Product Name', 'Store Name', 'BGN Price', 'EUR Price', 'Expected EUR', 'Difference', 'Status', 'Latitude', 'Longitude']]
        body = {'values': headers, 'majorDimension': 'ROWS'}
        
        sheets_service.spreadsheets().values().update(
            spreadsheetId=SPREADSHEET_ID, 
            range='Sheet1!A1:J1', 
            valueInputOption='RAW', 
            body=body
        ).execute()
        
        return True
    except Exception as e:
        print(f"Error setting up headers: {e}")
        return False

# --- END: UTILITY FUNCTIONS ---

# --- START: API ENDPOINTS ---

@app.route('/api/stores', methods=['GET'])
def get_stores():
    """Get the list of available stores."""
    return jsonify({"stores": config.stores.STORE_OPTIONS})

@app.route('/api/prices', methods=['GET'])
def get_prices():
    """Get price data for map visualization with caching."""
    try:
        # Check cache first
        cache_key = "api_prices_data"
        cached_prices = sheets_cache.get(cache_key)
        
        if cached_prices:
            print("Returning cached price data for API")
            return jsonify({"prices": cached_prices})
        
        # Get all data from Google Sheets
        values = get_cached_sheet_data()
        
        if not values or len(values) <= 1:
            return jsonify({"prices": []})
        
        prices = []
        existing_coordinates = []
        
        # Skip header row
        for row in values[1:]:
            if len(row) >= 8:  # Ensure we have all required columns
                try:
                    # Get coordinates using smart fallback system
                    stored_lat = row[8] if len(row) >= 10 and row[8] else None
                    stored_lng = row[9] if len(row) >= 10 and row[9] else None
                    store_name = row[2]
                    
                    # Use fallback coordinate system
                    latitude, longitude = get_fallback_coordinates(store_name, stored_lat, stored_lng)
                    
                    # Add slight spread to prevent overlapping markers
                    latitude, longitude = add_coordinate_spread(latitude, longitude, existing_coordinates)
                    existing_coordinates.append((latitude, longitude))
                    
                    price_data = {
                        "product_name": row[1],
                        "store_name": store_name,
                        "price_bgn": float(row[3]),
                        "price_eur": float(row[4]),
                        "expected_eur": float(row[5]),
                        "scan_date": row[0].split(' ')[0],  # Extract date part
                        "status": row[7],
                        "latitude": latitude,
                        "longitude": longitude,
                        "is_best_price": row[7] == "CORRECT"
                    }
                    prices.append(price_data)
                except (ValueError, IndexError) as e:
                    print(f"Error parsing row: {row}, error: {e}")
                    continue
        
        # Limit to recent entries (last 100 for performance)
        prices = prices[-100:]
        
        # Cache the processed prices
        sheets_cache.set(cache_key, prices, ttl=300)  # Cache for 5 minutes
        
        return jsonify({"prices": prices})
        
    except Exception as e:
        print(f"Error getting price data: {e}")
        return jsonify({"prices": []})

@app.route('/api/verify-prices', methods=['POST'])
def verify_prices():
    """Main endpoint for price verification with comprehensive validation."""
    request_start = time.time()
    performance.record_event("price_verification_request")
    
    try:
        # Validate file upload
        if 'file' not in request.files:
            return jsonify({"status": "ГРЕШКА", "message": "Не е качен файл."}), 400
        
        file = request.files['file']
        
        # Comprehensive file validation
        try:
            validate_file_upload(file)
        except ValidationError as e:
            return jsonify({"status": "ГРЕШКА", "message": f"Невалиден файл: {str(e)}"}), 400
        
        image_content = file.read()
        
        # Validate and sanitize form inputs
        try:
            selected_store = validate_store_name(request.form.get('store', ''))
            latitude, longitude = validate_coordinates(
                request.form.get('latitude'), 
                request.form.get('longitude')
            )
            
            if latitude is not None and longitude is not None:
                print(f"Received valid geolocation: {latitude}, {longitude}")
            
        except ValidationError as e:
            return jsonify({"status": "ГРЕШКА", "message": f"Невалидни данни: {str(e)}"}), 400

        # Get image dimensions for validation
        try:
            image_for_dims = Image.open(io.BytesIO(image_content))
            image_width, image_height = image_for_dims.size
        except Exception as e:
            return jsonify({"status": "ГРЕШКА", "message": f"Невалиден файл с изображение: {e}"})

        # Perform OCR using Google Vision API with config
        image = vision.Image(content=image_content)
        image_context = vision.ImageContext(language_hints=config.ocr.language_hints)
        response = vision_client.document_text_detection(image=image, image_context=image_context)
        annotations = response.text_annotations

        if not annotations:
            return jsonify({"status": "ГРЕШКА", "message": "Не можах да разчета текст."})

        # Extract product name using advanced spatial analysis
        product_name, product_name_box = extract_product_name_advanced(annotations)
        
        print(f"Extracted product name: {product_name}")
        print(f"Product name bounding box: {product_name_box}")
        
        # Store detection
        if selected_store == "Неизвестен магазин" or not selected_store:
            full_ocr_text = annotations[0].description
            store_name = extract_store_name_from_text(full_ocr_text, latitude, longitude)
        else:
            store_name = selected_store

        # Price extraction
        price_candidates = []
        price_pattern = re.compile(r'^\d+([.,]\d{1,2})?$')
        
        for annotation in annotations[1:]:
            if price_pattern.match(annotation.description):
                value = parse_price_string(annotation.description)
                if value is not None:
                    price_candidates.append({
                        'value': value,
                        'area': calculate_area(annotation.bounding_poly.vertices),
                        'box': [{'x': v.x, 'y': v.y} for v in annotation.bounding_poly.vertices]
                    })

        if len(price_candidates) < 2:
            return jsonify({"status": "ERROR", "message": f"Намерени са по-малко от две цени."})

        # Validate extracted prices
        valid_prices = []
        for candidate in price_candidates:
            try:
                validated_price = validate_price_range(candidate['value'], "extracted price")
                candidate['value'] = validated_price
                valid_prices.append(candidate)
            except ValidationError as e:
                print(f"Invalid price {candidate['value']}: {e}")
                continue
        
        if len(valid_prices) < 2:
            return jsonify({"status": "ERROR", "message": "Намерени са твърде малко валидни цени."})

        # Sort price candidates by area (visual prominence)
        price_candidates = valid_prices
        price_candidates.sort(key=lambda p: p['area'], reverse=True)
        
        # Check if the image is taken from appropriate distance
        largest_price_box = price_candidates[0]['box']
        box_height = max(v['y'] for v in largest_price_box) - min(v['y'] for v in largest_price_box)
        relative_height = box_height / image_height
        
        if relative_height < config.ocr.min_relative_height_threshold:
            return jsonify({"status": "TOO-FAR", "message": "Приближете се до етикета и опитайте отново."})

        # Smart BGN/EUR price detection
        def determine_bgn_eur_prices(candidates):
            """
            Intelligently determine which price is BGN and which is EUR.
            Uses multiple heuristics for better accuracy.
            """
            price1, price2 = candidates[0], candidates[1]
            
            # Heuristic 1: Check ratio - BGN/EUR should be close to exchange rate
            ratio1_2 = price1['value'] / price2['value'] if price2['value'] > 0 else 0
            ratio2_1 = price2['value'] / price1['value'] if price1['value'] > 0 else 0
            
            expected_ratio = config.currency.bgn_to_eur_rate
            
            # Check which assignment gives a ratio closer to expected
            error1 = abs(ratio1_2 - expected_ratio)  # price1=BGN, price2=EUR
            error2 = abs(ratio2_1 - expected_ratio)  # price2=BGN, price1=EUR
            
            # Heuristic 2: BGN prices are typically higher in absolute value
            size_suggests_bgn_1 = price1['value'] > price2['value']
            
            # Heuristic 3: EUR prices typically range 0.50-50.00, BGN prices 1.00-100.00
            eur_range_1 = 0.1 <= price1['value'] <= 100.0
            eur_range_2 = 0.1 <= price2['value'] <= 100.0
            bgn_range_1 = 0.5 <= price1['value'] <= 200.0
            bgn_range_2 = 0.5 <= price2['value'] <= 200.0
            
            # Score each assignment
            score_1_bgn = 0  # price1=BGN, price2=EUR
            score_2_bgn = 0  # price2=BGN, price1=EUR
            
            # Ratio scoring (most important)
            if error1 < error2:
                score_1_bgn += 3
            else:
                score_2_bgn += 3
                
            # Size scoring
            if size_suggests_bgn_1:
                score_1_bgn += 1
            else:
                score_2_bgn += 1
                
            # Range scoring
            if bgn_range_1 and eur_range_2:
                score_1_bgn += 1
            if bgn_range_2 and eur_range_1:
                score_2_bgn += 1
                
            # Additional validation: reject impossible ratios
            if ratio1_2 < 1.5 or ratio1_2 > 2.5:  # Outside reasonable BGN/EUR range
                score_1_bgn -= 2
            if ratio2_1 < 1.5 or ratio2_1 > 2.5:
                score_2_bgn -= 2
                
            print(f"Price detection: P1={price1['value']}, P2={price2['value']}")
            print(f"Ratios: {ratio1_2:.3f}, {ratio2_1:.3f}, Expected: {expected_ratio:.3f}")
            print(f"Scores: P1=BGN: {score_1_bgn}, P2=BGN: {score_2_bgn}")
            
            if score_1_bgn >= score_2_bgn:
                return price1, price2  # price1=BGN, price2=EUR
            else:
                return price2, price1  # price2=BGN, price1=EUR
        
        bgn_price_data, eur_price_data = determine_bgn_eur_prices(price_candidates[:2])
            
        price_bgn = bgn_price_data['value']
        price_eur = eur_price_data['value']
        expected_eur = price_bgn / config.currency.bgn_to_eur_rate
        price_difference = price_eur - expected_eur

        # Determine if the price is correct using config tolerance
        if price_difference > config.currency.price_tolerance:
            status, message = "INCORRECT", "Цената в EUR изглежда несправедливо завишена."
        else:
            status, message = "CORRECT", "Цената е правилна и съответства на официалния курс."

        # Get historical price information
        historical_price_info = None
        if product_name != "Неизвестен продукт":
            historical_data = find_last_price_in_sheets(product_name)
            if historical_data:
                last_price = historical_data['last_price_bgn']
                price_change = price_bgn - last_price
                historical_price_info = {
                    "last_price_bgn": last_price, 
                    "last_scan_date": historical_data['last_scan_date'], 
                    "price_change_bgn": round(price_change, 2)
                }

        # Get fallback coordinates if GPS is not available
        final_latitude, final_longitude = get_fallback_coordinates(store_name, latitude, longitude)
        
        # Save results to Google Sheets
        save_success = save_to_sheets(product_name, store_name, price_bgn, price_eur, round(expected_eur, 2), status, final_latitude, final_longitude)

        # Prepare response
        response_data = {
            "status": status, 
            "message": message,
            "data": {
                "product_name": product_name, 
                "store_name": store_name, 
                "found_bgn": price_bgn,
                "found_eur": price_eur, 
                "expected_eur": round(expected_eur, 2), 
                "difference_eur": round(price_difference, 4),
                "bgn_box": bgn_price_data['box'], 
                "eur_box": eur_price_data['box'], 
                "product_name_box": product_name_box,
                "saved_to_sheets": save_success,
                "historical_price": historical_price_info
            }
        }
        
        # Record successful completion
        request_duration = time.time() - request_start
        performance.record_timing("price_verification_total", request_duration)
        performance.record_event("price_verification_success")
        logger.info(f"Price verification completed in {request_duration:.3f}s for product: {product_name}")
        
        return jsonify(response_data)
    
    except ValidationError as e:
        performance.record_event("price_verification_validation_error")
        logger.warning(f"Validation error in verify_prices: {e}")
        return jsonify({"status": "ГРЕШКА", "message": f"Валидационна грешка: {str(e)}"}), 400
    except Exception as e:
        performance.record_event("price_verification_error")
        logger.error(f"Unexpected error in verify_prices: {e}")
        return jsonify({"status": "ГРЕШКА", "message": "Възникна неочаквана грешка при обработката."}), 500

@app.route('/api/setup-sheet', methods=['POST'])
def setup_sheet():
    """Setup the Google Sheet with proper headers."""
    success = setup_sheet_headers()
    return jsonify({
        "status": "SUCCESS" if success else "ERROR", 
        "message": "Sheet headers created successfully" if success else "Failed to create sheet headers"
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "OK",
        "message": "Price verification service is running",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/maps-key', methods=['GET'])
def get_maps_key():
    """Get Google Maps API key for frontend."""
    if not GOOGLE_MAPS_API_KEY:
        return jsonify({"error": "Google Maps API key not configured"}), 500
    return jsonify({"api_key": GOOGLE_MAPS_API_KEY})

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Get performance metrics and cache statistics."""
    try:
        metrics = {
            "performance": performance.get_stats(),
            "cache": {
                "sheets_cache": sheets_cache.stats(),
                "price_history_cache": price_history_cache.stats(),
                "llm_cache": llm_cache.stats()
            },
            "config": {
                "cache_ttl": config.cache.default_ttl,
                "max_file_size_mb": config.validation.max_file_size_mb,
                "exchange_rate": config.currency.bgn_to_eur_rate
            }
        }
        return jsonify(metrics)
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        return jsonify({"error": "Failed to get metrics"}), 500

# --- END: API ENDPOINTS ---

# Load product dictionary on startup
load_product_dictionary()

if __name__ == '__main__':
    app.run(debug=True)