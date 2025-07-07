import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

class Config:
    """Application configuration"""
    
    # Base paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    EXPORTS_DIR = DATA_DIR / "exports"
    LOGS_DIR = BASE_DIR / "logs"
    
    # API Configuration
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
    
    # Application
    APP_NAME = os.getenv('APP_NAME', 'AI Customer Intelligence Agent')
    APP_VERSION = os.getenv('APP_VERSION', '1.0.0')
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    # Data
    DEFAULT_DATA_PATH = os.getenv('DEFAULT_DATA_PATH', 'data/raw/customer_data.csv')
    MAX_RECORDS = int(os.getenv('MAX_RECORDS', 100000))
    CACHE_TTL = int(os.getenv('CACHE_TTL', 3600))
    
    # UI
    THEME = os.getenv('THEME', 'dark')
    SIDEBAR_STATE = os.getenv('SIDEBAR_STATE', 'expanded')
    LAYOUT = os.getenv('LAYOUT', 'wide')
    
    @classmethod
    def validate(cls):
        """Validate configuration"""
        errors = []
        
        if not cls.OPENAI_API_KEY or cls.OPENAI_API_KEY == 'your_openai_api_key_here':
            errors.append("OPENAI_API_KEY not configured")
        
        # Create directories if they don't exist
        for directory in [cls.DATA_DIR, cls.RAW_DATA_DIR, cls.PROCESSED_DATA_DIR, 
                         cls.EXPORTS_DIR, cls.LOGS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
        
        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")
        
        return True

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False
    LOG_LEVEL = 'WARNING'

# Configuration selector
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
