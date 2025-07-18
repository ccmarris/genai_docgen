import logging
import configparser
import os
import clients.response_cache

__author__ = "Chris Marrison"
__copyright__ = "Copyright 2025, Chris Marrison / Infoblox"
__license__ = "BSD2"
__version__ = "0.3.5"
__email__ = "chris@infoblox.com"

_logger = logging.getLogger(__name__)

class IniFileSectionError(Exception):
    """Exception raised when a required section is missing in the ini file."""
    pass

class IniFileKeyError(Exception):
    """Exception raised when a required key is missing in the ini file."""
    pass

class APIKeyFormatError(Exception):
    """Exception raised when the API key format is incorrect."""
    pass    

class LLMClient:
    """Abstract base class for LLM providers."""
    def get_response(self, prompt, **kwargs):
        raise NotImplementedError("Subclasses must implement get_response()")


# ** Facilitate ini file for basic configuration including API Key
"""Exception raised when the API key format is incorrect."""
def read_ini(filename:str, section:str, ini_keys:list) -> dict:
    '''
    Open and parse ini file

    Parameters:
        filename (str): name of inifile

    Returns:
        config (dict): Dictionary of BloxOne configuration elements

    Raises:
        IniFileSectionError
        IniFileKeyError
        APIKeyFormatError
        FileNotFoundError

    '''
    # Local Variables
    cfg = configparser.ConfigParser()
    config = {}

    # Check for inifile and raise exception if not found
    if os.path.isfile(filename):
        # Attempt to read api_key from ini file
        try:
            cfg.read(filename)
        except configparser.Error as err:
            _logger.error(err)

        # Look for BloxOne section
        if section in cfg:
            for key in ini_keys:
                # Check for key in section
                if key in cfg[section]:
                    config[key] = cfg[section][key].strip("'\"")
                    _logger.debug(f'Key {key} found in {filename}: {config[key]}')
                else:
                    _logger.debug(f'Key {key} not found in {section} section.')
                    # raise IniFileKeyError(f'Key "{key}" not found within' +
                    #        f'[{section}] section of ini file {filename}')
                    
        else:
            _logger.error(f'No {section} Section in config file: {filename}')
            raise IniFileSectionError(f'No [{section}] section found in ini file {filename}')
        
    else:
        raise FileNotFoundError('ini file "{filename}" not found.')

    return config


def get_llm_client(provider="openai", **kwargs):
    if provider == "openai":
        try:
            from clients.openai_client import OpenAIClient
        except ImportError:
            _logger.error("OpenAI client module not found. Please install it.")
            raise ImportError("OpenAI client module not found. Please install it.")
        return OpenAIClient(kwargs.get('inifile', ''), use_cache=kwargs.get('use_cache', True))
    elif provider == "anthropic":
        try:
            from clients.anthropic_client import AnthropicClient
        except ImportError:
            _logger.error("Anthropic client module not found. Please install it.")
            raise ImportError("Anthropic client module not found. Please install it.")
        return AnthropicClient(kwargs.get('inifile',''), use_cache=kwargs.get('use_cache', True))
    elif provider == "gemini":
        try:
            from clients.gemini_client import GeminiClient
        except ImportError:
            _logger.error("Gemini client module not found. Please install it.")
            raise ImportError("Gemini client module not found. Please install it.")
        return AnthropicClient(kwargs.get('inifile',''), use_cache=kwargs.get('use_cache', True))
    else:
        raise ValueError(f"Unknown provider: {provider}")

def clear_cache(provider="openai", inifile=''):
    """
    Clear the response cache for the specified provider.
    Parameters:
        provider (str): The LLM provider to clear the cache for.
        inifile (str): Path to the ini file containing configuration.
    """
    success = False
    try:
        cache = response_cache.RESPONSECACHE(cache_type=provider)
        cache.reset()
        success = True
        _logger.info(f"Cache cleared for provider: {provider}")
    except Exception as e:
        _logger.error(f"Error clearing cache for provider {provider}: {e}")
        success = False
    
    return success

# Example usage in main.py:
# client = get_llm_client(provider=args.provider, inifile=args.ini)
# response = client.get_response(prompt, temperature=args.temperature, ...)