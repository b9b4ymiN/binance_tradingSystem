import time
import json
import logging
from typing import Dict, Any, Optional
from functools import wraps

logger = logging.getLogger(__name__)

class MemoryCache:
    """Simple in-memory cache with TTL support"""

    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}

    def set(self, key: str, value: Any, ttl_seconds: int = 300) -> None:
        """Set a cache entry with TTL"""
        expires_at = time.time() + ttl_seconds
        self._cache[key] = {
            'value': value,
            'expires_at': expires_at,
            'created_at': time.time()
        }
        logger.debug(f"Cache SET: {key} (TTL: {ttl_seconds}s)")

    def get(self, key: str) -> Optional[Any]:
        """Get a cache entry, return None if expired or missing"""
        if key not in self._cache:
            return None

        entry = self._cache[key]

        if time.time() > entry['expires_at']:
            del self._cache[key]
            logger.debug(f"Cache EXPIRED: {key}")
            return None

        logger.debug(f"Cache HIT: {key}")
        return entry['value']

    def delete(self, key: str) -> bool:
        """Delete a cache entry"""
        if key in self._cache:
            del self._cache[key]
            logger.debug(f"Cache DELETE: {key}")
            return True
        return False

    def clear(self) -> None:
        """Clear all cache entries"""
        count = len(self._cache)
        self._cache.clear()
        logger.info(f"Cache CLEARED: {count} entries removed")

    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys containing the pattern"""
        keys_to_delete = [key for key in self._cache.keys() if pattern in key]
        for key in keys_to_delete:
            del self._cache[key]
        logger.info(f"Cache PATTERN INVALIDATE: {pattern} - {len(keys_to_delete)} entries removed")
        return len(keys_to_delete)

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_entries = len(self._cache)
        expired_entries = 0
        current_time = time.time()

        for entry in self._cache.values():
            if current_time > entry['expires_at']:
                expired_entries += 1

        return {
            'total_entries': total_entries,
            'active_entries': total_entries - expired_entries,
            'expired_entries': expired_entries,
            'cache_keys': list(self._cache.keys())
        }

# Global cache instance
cache = MemoryCache()

def cached(ttl_seconds: int = 300, key_prefix: str = ""):
    """Decorator for caching function results"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{key_prefix}{func.__name__}"
            if args:
                cache_key += f"_args_{hash(str(args))}"
            if kwargs:
                cache_key += f"_kwargs_{hash(str(sorted(kwargs.items())))}"

            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute function and cache result
            try:
                result = func(*args, **kwargs)
                cache.set(cache_key, result, ttl_seconds)
                return result
            except Exception as e:
                logger.error(f"Error in cached function {func.__name__}: {e}")
                raise

        # Add cache management methods to the wrapper
        wrapper.invalidate_cache = lambda: cache.invalidate_pattern(f"{key_prefix}{func.__name__}")
        wrapper.clear_cache = lambda: cache.clear()

        return wrapper
    return decorator

def cache_key(prefix: str, *args, **kwargs) -> str:
    """Generate a consistent cache key"""
    key_parts = [prefix]

    if args:
        key_parts.append(f"args_{hash(str(args))}")

    if kwargs:
        sorted_kwargs = sorted(kwargs.items())
        key_parts.append(f"kwargs_{hash(str(sorted_kwargs))}")

    return "_".join(key_parts)