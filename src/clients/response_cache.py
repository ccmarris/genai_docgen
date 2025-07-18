import logging
import os
import json
from platformdirs import user_cache_dir
from datetime import datetime, timedelta

__author__ = "Chris Marrison"
__copyright__ = "Copyright 2025, Chris Marrison / Infoblox"
__license__ = "BSD2"
__email__ = "chris@infoblox.com"
__version__ = "0.2.0"

_logger = logging.getLogger(__name__)

class RESPONSECACHE():
    '''
    RESPONSECACHE is a generic prompt/response cache for GenAI clients.
    This class provides a flexible caching mechanism for storing and retrieving prompt-response pairs, along with associated parameters and timestamps, for various GenAI API clients such as OpenAI ChatGPT, OpenAI API, Anthropic, Azure OpenAI, or custom configurations.
    Attributes:
        cache_type (str): The type of cache (e.g., "chatgpt", "openai", "anthropic", "azure", "custom").
        cache_dir (str): The directory where the cache file is stored.
        cache_file (str): The full path to the cache file.
        key_fields (list): The list of fields used as keys for cache entries.
        cache (list): The in-memory list of cached entries.
    Methods:
        __init__(cache_type: str = "chatgpt", fields: list = None):
            Initializes the cache for the specified cache_type, setting up the cache file and key fields.
        load():
            Loads the cache from the cache file, validating its structure.
        _validate(cache):
            Validates the cache structure, ensuring it is a list of dictionaries with required fields.
        clean_cache(days=28):
            Removes cache entries older than the specified number of days.
        reset():
            Clears the cache by removing the cache file and reinitializing an empty cache.
        write():
            Writes the current in-memory cache to the cache file in JSON format.
        get_entry(**params):
            Retrieves a cached response matching the provided parameters.
        add_entry(**kwargs):
            Adds a new entry to the cache with the specified parameters.
    Usage:
        cache = RESPONSECACHE(cache_type="chatgpt")
        cache.add_entry(prompt="Hello", model="gpt-3.5-turbo", response="Hi!", ...)
        response = cache.get_entry(prompt="Hello", model="gpt-3.5-turbo")
    '''

    def __init__(self, cache_type:str="chatgpt", fields:list=None):
        '''
        Initializes the Response Cache with support for different cache types.
        Supported cache types:
        - chatgpt: Cache for OpenAI ChatGPT responses.
        - openai: Cache for OpenAI API responses.
        - anthropic: Cache for Anthropic API responses.
        - azure: Cache for Azure OpenAI API responses.
        - custom: Custom cache type, fields must be provided.

        The cache is stored in a user-specific directory using platformdirs.
        The cache file is named based on the cache type.
        The cache is loaded from the cache file on initialization.

        If the cache file does not exist, it will be created.
        '''
        key_fields:list = None
        cache_file:str = None

        match cache_type: 
            case "openai":
                cache_file = "openai_cache.json"
                key_fields = [ "cache_type", "prompt", "model", "temperature", 
                              "top_p", "frequency_penalty", "presence_penalty", 
                              "response", "timestamp" ]

            case "anthropic":
                cache_file = "anthropic_cache.json"
                key_fields=[ "cache_type", "prompt", "model", "max_tokens", 
                            "temperature", "top_k", "top_p" ]

            case "azure":
                cache_file = "azure_cache.json"
                key_fields = fields if key_fields is not None else []

            case "custom":
                cache_file = "custom_cache.json"
                key_fields = fields if key_fields is not None else []

            case _:
                raise ValueError(f"Unsupported cache type: {cache_type}")


        self.cache_type = cache_type
        self.cache_dir = user_cache_dir("genai_docgen", "Chris Marrison")
        self.cache_file = self.cache_dir + '/' + cache_file
        self.key_fields = key_fields if key_fields is not None else []
        self.search_keys = self._search_keys()
        self.cache = self.load()

        return


    def load(self):
        '''
        Load the cache from the cache file.

        Returns:
            list: List of cached responses.
        '''
        cache = []

        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r", encoding="utf-8") as f:
                try:
                    cache = json.load(f)
                    cache = self._validate(cache)
                except json.JSONDecodeError as e:
                    _logger.error(f'Error decoding JSON from cache file {self.cache_file}: {e}')
                    # If the cache file is not valid JSON, reset the cache
                    self.reset()
                    cache = []
                    _logger.info(f'Reset cache due to invalid JSON in {self.cache_file}')
        else:
            # Create the cache directory if it doesn't exist
            os.makedirs(self.cache_dir, exist_ok=True)
            _logger.debug(f'Cache file not found, creating new cache file: {self.cache_file}')
            # Initialize an empty cache
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=2)
            
        _logger.debug(f'Cache loaded with {len(cache)} entries.')

        return cache


    def _validate(self, cache):
        '''
        Ensure cache is a valid list of dictionaries with required fields.
        Args:
            cache (list): The cache to validate.
        Returns:
            list: The validated cache or an empty list if invalid.
        Raises:
            ValueError: If the cache is not valid.
        '''

        # Ensure the cache is a list
        if not isinstance(cache, list):
            _logger.error(f'Cache file {self.cache_file} is not a valid JSON list. Resetting cache.')
            cache = []
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=2)
        # Ensure the cache is a list of dictionaries
        if not all(isinstance(entry, dict) for entry in cache):
            _logger.error(f'Cache file {self.cache_file} contains non-dictionary entries. Resetting cache.')
            cache = []
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=2)
        # Ensure each entry has the required fields
        for entry in cache:
            if not all(key in entry for key in self.key_fields):
                # If any entry is missing required fields, reset the cache
                _logger.error(f'Cache entry {entry} is missing required fields. Resetting cache.')
                cache = []
                with open(self.cache_file, "w", encoding="utf-8") as f:
                    json.dump(cache, f, indent=2)
                break
        _logger.debug(f'Cache loaded successfully with {len(cache)} entries.')
        # Return the loaded cache
        
        return cache


    def clean_cache(self, days=28):

        cutoff = datetime.now() - timedelta(days=days)
        _logger.debug(f'Cleaning cache. Removing entries older than {days} days. Cutoff date: {cutoff.isoformat()}')
        cleaned_cache = [
            entry for entry in self.cache
                if "timestamp" in entry and (datetime.fromisoformat(entry["timestamp"]) >= cutoff)
        ]

        _logger.debug(f"Cleaned cache. {len(self.cache) - len(cleaned_cache)} entries removed.")
        self.cache = cleaned_cache
        self.write()

        return cleaned_cache


    def reset(self):
        '''
        Clear the cache by removing the cache file.
        '''
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)
            self.cache = self.load()  # Reload the cache to ensure it's empty
            _logger.info(f'Cache cleared: {self.cache_file}')
        else:
            _logger.info('No cache file to clear.')
        return


    def write(self):
        '''
        Write the current cache to the cache file.
        This method serializes the cache to JSON format and writes it to the cache file.
        Returns:
            None
        '''
        with open(self.cache_file, "w", encoding="utf-8") as f:
            try:
                json.dump(self.cache, f, indent=2)
            except TypeError as e:
                _logger.error(f'Error writing cache to file {self.cache_file}: {e}')
                raise ValueError(f'Cache entry contains non-serializable data: {e}')
        _logger.debug(f'Cache written to {self.cache_file}. Total entries: {len(self.cache)}')

        return

    def _search_keys(self):
        '''
        Get the keys used for searching cache entries.
        Returns:
            list: The list of keys used for searching cache entries.
        '''
        unneeded_keys = ['cache_type', 'timestamp', 'response']
        search_keys = self.key_fields.copy()
        for k in unneeded_keys:
            if k in search_keys:
                search_keys.remove(k)
        return search_keys


    def check_cache(self, **params):
        '''
        Retrieve a cached response based on the provided parameters.
        Args:
            **params: Key-value pairs to match against the cache entries.
        Returns:
            dict: The cached response entry if found, otherwise None.
        '''
        response = None

        for entry in self.cache:
            if all(entry.get(k) == params.get(k) for k in self.search_keys):
                _logger.debug(f'Cache hit for entry: {entry}')
                # If a matching entry is found, return the response
                response = entry.get("response")
                break

        return response


    def add_entry(self, **kwargs):
        '''
        Add a new entry to the cache.
        Args:
            **kwargs: Key-value pairs for the cache entry. Must include all key fields.
        Raises:
            ValueError: If any key field is missing from kwargs.
        Returns:
            None
        '''
        entry = {k: kwargs.get(k) for k in self.key_fields}
        entry["cache_type"] = self.cache_type
        entry["timestamp"] = datetime.now().isoformat()
        self.cache.append(entry)
        self.write()
        return 