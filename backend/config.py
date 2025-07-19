# config.py - Centralized Business Logic Configuration

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class CurrencyConfig:
    """Currency exchange and validation configuration."""
    bgn_to_eur_rate: float = 1.95583
    min_price: float = 0.01
    max_price: float = 10000.0
    price_tolerance: float = 0.01  # EUR difference threshold for "correct" prices
    
@dataclass 
class ValidationConfig:
    """Input validation configuration."""
    max_file_size_mb: int = 10
    min_file_size_kb: int = 1
    allowed_image_extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    max_text_length: int = 1000
    max_store_name_length: int = 100
    
    # Coordinate bounds for Bulgaria
    min_latitude: float = 41.2
    max_latitude: float = 44.2
    min_longitude: float = 22.3
    max_longitude: float = 28.6
    
@dataclass
class OCRConfig:
    """OCR and image processing configuration."""
    min_relative_height_threshold: float = 0.03
    language_hints: list = None
    
    def __post_init__(self):
        if self.language_hints is None:
            self.language_hints = ['bg', 'bg-BG']

@dataclass
class CacheConfig:
    """Caching configuration."""
    default_ttl: int = 300  # 5 minutes
    sheets_cache_ttl: int = 600  # 10 minutes  
    price_history_ttl: int = 1800  # 30 minutes
    api_prices_ttl: int = 300  # 5 minutes
    llm_cache_ttl: int = 3600  # 1 hour - LLM responses are stable for identical inputs

@dataclass
class GoogleSheetsConfig:
    """Google Sheets configuration."""
    range_name: str = 'Sheet1!A:J'
    scopes: list = None
    batch_size: int = 100  # For pagination
    
    def __post_init__(self):
        if self.scopes is None:
            self.scopes = ['https://www.googleapis.com/auth/spreadsheets']

class StoreConfig:
    """Store detection and location configuration."""
    
    # Available store options
    STORE_OPTIONS = [
        'Kaufland', 'Lidl', 'Metro', 'Billa', 'Fantastico', 
        'T-Market', 'Пикадили', 'CBA', 'Други'
    ]
    
    # Default coordinates for major store chains in Sofia
    STORE_DEFAULT_COORDINATES = {
        'Kaufland': {'lat': 42.6864, 'lng': 23.3238},
        'Lidl': {'lat': 42.6506, 'lng': 23.3806},
        'Metro': {'lat': 42.6234, 'lng': 23.3844},
        'Billa': {'lat': 42.6977, 'lng': 23.3219},
        'Fantastico': {'lat': 42.6584, 'lng': 23.3486},
        'T-Market': {'lat': 42.6977, 'lng': 23.3219},
        'Пикадили': {'lat': 42.6977, 'lng': 23.3219},
        'CBA': {'lat': 42.6977, 'lng': 23.3219},
        'Други': {'lat': 42.6977, 'lng': 23.3219}
    }
    
    # Store name keywords for detection
    STORE_KEYWORDS = [
        'kaufland', 'lidl', 'metro', 'billa', 'fantastico', 
        't-market', 'пикадили', 'cba'
    ]
    
    # Default fallback location (Sofia city center)
    DEFAULT_LOCATION = {'lat': 42.6977, 'lng': 23.3219}

class ProductDictionaryConfig:
    """Product dictionary and matching configuration."""
    
    # CSV file path for product dictionary
    PRODUCT_DICT_FILE = 'final_product_name.csv'
    
    # Fuzzy matching parameters
    FUZZY_THRESHOLD = 80
    FUZZY_LIMIT = 3
    FUZZY_FALLBACK_THRESHOLD = 60
    
    # Phrases to exclude from product names
    EXCLUDED_PHRASES = [
        'специална цена', 'special price', 'промоция', 'promo', 'намаление', 'отстъпка',
        'discount', 'sale', 'акция', 'валиден до', 'valid until', 'expires', 'изтича',
        'fresh', 'свеж', 'daily', 'дневен', 'ново', 'new', 'топ качество', 'top quality',
        'препоръчваме', 'recommend', 'популярен', 'popular', 'hit', 'хит', 'organic',
        'био', 'eco', 'еко', 'natural', 'натурален', 'premium', 'премиум', 'luxury',
        'луксозен', 'imported', 'внос', 'местен', 'local', 'regional', 'регионален', 
        'порция свежест', 'грил майстор', 'стара планина', 'xxl', 'xl', 'l', 'размер',
        'size', 'опаковка', 'package', 'pack', 'бройка', 'брой', 'count', 'cantidad', 'планина'
    ]

class APIConfig:
    """API rate limiting and retry configuration."""
    
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_BASE_DELAY = 1
    PLACES_API_MAX_RETRIES = 2
    PLACES_API_BASE_DELAY = 2
    SHEETS_API_MAX_RETRIES = 2
    SHEETS_API_BASE_DELAY = 1
    
    # Places API configuration
    PLACES_SEARCH_RADIUS = 1000  # meters
    PLACES_MAX_RESULTS = 10

class SecurityConfig:
    """Security and validation patterns."""
    
    # Dangerous patterns for text input validation
    DANGEROUS_PATTERNS = [
        '<script', '</script>', 'javascript:', 'vbscript:',
        'onload=', 'onerror=', 'onclick=',
        "';", '";', '--', '/*', '*/',
        'DROP TABLE', 'DELETE FROM', 'INSERT INTO', 'UPDATE SET'
    ]
    
    # Image file magic numbers for validation
    IMAGE_SIGNATURES = {
        b'\xff\xd8\xff': 'JPEG',
        b'\x89\x50\x4e\x47': 'PNG', 
        b'\x42\x4d': 'BMP',
        b'\x52\x49\x46\x46': 'WEBP'
    }

class AppConfig:
    """Main application configuration that combines all configs."""
    
    def __init__(self):
        self.currency = CurrencyConfig()
        self.validation = ValidationConfig()
        self.ocr = OCRConfig()
        self.cache = CacheConfig()
        self.sheets = GoogleSheetsConfig()
        self.stores = StoreConfig()
        self.products = ProductDictionaryConfig()
        self.api = APIConfig()
        self.security = SecurityConfig()
        
        # Override with environment variables if available
        self._load_from_env()
    
    def _load_from_env(self):
        """Load configuration overrides from environment variables."""
        
        # Currency config
        if os.getenv('BGN_TO_EUR_RATE'):
            self.currency.bgn_to_eur_rate = float(os.getenv('BGN_TO_EUR_RATE'))
        
        if os.getenv('PRICE_TOLERANCE'):
            self.currency.price_tolerance = float(os.getenv('PRICE_TOLERANCE'))
            
        # Validation config  
        if os.getenv('MAX_FILE_SIZE_MB'):
            self.validation.max_file_size_mb = int(os.getenv('MAX_FILE_SIZE_MB'))
            
        # OCR config
        if os.getenv('MIN_HEIGHT_THRESHOLD'):
            self.ocr.min_relative_height_threshold = float(os.getenv('MIN_HEIGHT_THRESHOLD'))
            
        # Cache config
        if os.getenv('CACHE_TTL'):
            self.cache.default_ttl = int(os.getenv('CACHE_TTL'))
            
        # Sheets config
        if os.getenv('SHEET_RANGE'):
            self.sheets.range_name = os.getenv('SHEET_RANGE')
    
    def get_column_indices(self):
        """Get standardized column indices for sheets."""
        return {
            'timestamp': 0,
            'product_name': 1, 
            'store_name': 2,
            'bgn_price': 3,
            'eur_price': 4,
            'expected_eur': 5,
            'difference': 6,
            'status': 7,
            'latitude': 8,
            'longitude': 9
        }

# Global configuration instance
config = AppConfig()