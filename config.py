# config.py - OPTIMIZED FOR 1.4GB DATASET
# ====================
# DATA PATHS & MEMORY SETTINGS
# ====================

# Kenyan Vehicle Dataset (Craigslist data)
RAW_DATA_PATH = 'vehicles.csv'  # Make sure this file is in your project folder

# Memory optimization settings
SAMPLE_SIZE = 20000              # Reduced from 50000 for memory
USE_CHUNKS = False               # Set to True if still having memory issues
CHUNK_SIZE = 5000               # Process in chunks if needed

# Features to load (only essential columns to save memory)
FEATURES_TO_KEEP = [
    'price', 'year', 'manufacturer', 'model', 'odometer',
    'condition', 'fuel', 'transmission', 'drive', 'type', 'cylinders'
]

# Optimized data types for memory efficiency
DTYPE_SPEC = {
    'price': 'float32',
    'year': 'int16',
    'odometer': 'float32',
    'manufacturer': 'category',
    'model': 'category',
    'condition': 'category',
    'fuel': 'category',
    'transmission': 'category',
    'drive': 'category',
    'type': 'category',
    'cylinders': 'category'
}

# Dataset type
DATASET_TYPE = 'kenyan'

# ====================
# CURRENCY & UNITS CONVERSION
# ====================

# Exchange rate (USD to KES) - current approximate rate
USD_TO_KES_RATE = 130.0

# Miles to Kilometers conversion
MILES_TO_KM = 1.60934

# Currency settings for Kenya
CURRENCY = {
    'symbol': 'KSh',
    'name': 'KES',
    'usd_symbol': '$',
    'usd_name': 'USD',
    'conversion_rate': USD_TO_KES_RATE,
    'miles_to_km': MILES_TO_KM
}

# ====================
# DATA CLEANING SETTINGS
# ====================

# Data cleaning thresholds
CLEANING_THRESHOLDS = {
    'min_year': 1990,
    'max_year': 2024,
    'min_price_usd': 1000,
    'max_price_usd': 100000,
    'min_odometer': 0,
    'max_odometer': 300000,
    'valid_conditions': ['excellent', 'good', 'fair', 'salvage', 'like new', 'new']
}

# ====================
# MODEL TRAINING SETTINGS
# ====================

# Train-test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Model to use
MODEL_TYPE = 'random_forest'

# Dynamic model parameters based on dataset size
def get_model_params(dataset_size):
    """Get optimal model parameters based on dataset size"""
    n_estimators = min(100, max(10, dataset_size // 200))
    max_depth = min(20, max(5, dataset_size // 1000))
    
    return {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'max_features': 'sqrt'
    }

# Default model params (will be adjusted based on actual data size)
DEFAULT_MODEL_PARAMS = get_model_params(20000)

# ====================
# APPLICATION SETTINGS
# ====================

# Default values for the web form (Kenyan market preferences)
DEFAULT_VALUES = {
    'year': 2018,
    'odometer_km': 50000,        # Changed from miles to km for clarity
    'manufacturer': 'toyota',
    'model': 'corolla',
    'condition': 'good',
    'fuel': 'gas',
    'transmission': 'automatic',
    'cylinders': '4 cylinders',
    'drive': 'fwd',
    'type': 'sedan',
    'size': 'mid-size'
}

# Available options for dropdowns
AVAILABLE_OPTIONS = {
    'manufacturers': ['toyota', 'nissan', 'subaru', 'mitsubishi', 'honda', 
                      'mazda', 'ford', 'mercedes-benz', 'bmw', 'audi', 
                      'volkswagen', 'isuzu', 'suzuki', 'kia', 'hyundai',
                      'chevrolet', 'jeep', 'ram', 'dodge', 'lexus'],
    'fuel_types': ['gas', 'diesel', 'hybrid', 'electric', 'other'],
    'transmissions': ['automatic', 'manual', 'other'],
    'vehicle_types': ['sedan', 'SUV', 'truck', 'hatchback', 'coupe', 
                      'convertible', 'wagon', 'minivan', 'pickup', 'van',
                      'offroad', 'bus', 'other'],
    'conditions': ['excellent', 'good', 'fair', 'salvage', 'like new', 'new'],
    'drive_types': ['fwd', 'rwd', '4wd', 'awd', 'other'],
    'cylinder_options': ['4 cylinders', '6 cylinders', '8 cylinders', 
                        '10 cylinders', '12 cylinders', 'other']
}

# ====================
# APP CONFIGURATION
# ====================

DEBUG = True
HOST = '127.0.0.1'
PORT = 5000
SECRET_KEY = 'kenyan-car-price-prediction-2025'

# Image search (enabled for Kenyan market)
ENABLE_IMAGE_SEARCH = True
UNSPLASH_ACCESS_KEY = 'yPHgXJrWyINAgdbm1dP_MuFAD3yFU2fl21JMkZXtStY'
PEXELS_API_KEY = 'b9z182Xa4XOT63WilFJ3fixA13b97KikRdykyQdgsVWCQrHzXaUSnvHp'

# ====================
# PRICE RANGES (Kenyan Market)
# ====================

PRICE_RANGES_KES = {
    'budget': (200000, 800000),
    'mid_range': (800000, 2500000),
    'premium': (2500000, 8000000),
    'luxury': (8000000, 20000000)
}

# ====================
# PERFORMANCE SETTINGS
# ====================

# Enable/disable features for performance
ENABLE_ML_MODEL = True          # Set to False if ML model causes issues
ENABLE_RULE_BASED_FALLBACK = True
ENABLE_IMAGE_APIS = True
ENABLE_DATASET_STATS = True

# Cache settings
MODEL_CACHE_DIR = 'models'
CACHE_MODEL = True

# ====================
# LOGGING CONFIG
# ====================

LOG_LEVEL = 'INFO'
LOG_FILE = 'logs/kenyan_car_predictor.log'

# ====================
# VALIDATION SETTINGS
# ====================

# Input validation ranges
VALIDATION_RANGES = {
    'year_min': 1990,
    'year_max': 2024,
    'odometer_km_min': 0,
    'odometer_km_max': 500000,  # 500,000 km maximum
    'price_min_kes': 100000,    # 100,000 KES minimum
    'price_max_kes': 50000000   # 50 million KES maximum
}

# ====================
# ERROR HANDLING
# ====================

# Maximum retries for external APIs
MAX_API_RETRIES = 3
API_TIMEOUT_SECONDS = 10

# Model fallback confidence threshold
ML_MODEL_CONFIDENCE_THRESHOLD = 0.6  # Use rule-based if ML confidence < 60%