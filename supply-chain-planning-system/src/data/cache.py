"""
Data Cache for Supply Chain Planning System.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A single cache entry."""
    key: str
    value: Any
    created_at: datetime
    expires_at: Optional[datetime] = None
    hits: int = 0


class DataCache:
    """
    Simple in-memory cache for planning data.
    
    Supports TTL-based expiration and LRU-style eviction.
    """
    
    def __init__(self, max_size: int = 1000, default_ttl_seconds: int = 3600):
        """
        Initialize the cache.
        
        Args:
            max_size: Maximum number of entries
            default_ttl_seconds: Default time-to-live in seconds
        """
        self.max_size = max_size
        self.default_ttl = timedelta(seconds=default_ttl_seconds)
        self._cache: Dict[str, CacheEntry] = {}
        logger.info("DataCache initialized with max_size=%d, ttl=%ds", max_size, default_ttl_seconds)
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        entry = self._cache.get(key)
        
        if entry is None:
            return None
        
        # Check expiration
        if entry.expires_at and datetime.now() > entry.expires_at:
            del self._cache[key]
            return None
        
        entry.hits += 1
        return entry.value
    
    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None
    ) -> None:
        """
        Set a value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Optional custom TTL
        """
        # Evict if necessary
        if len(self._cache) >= self.max_size:
            self._evict_lru()
        
        ttl = timedelta(seconds=ttl_seconds) if ttl_seconds else self.default_ttl
        expires_at = datetime.now() + ttl
        
        self._cache[key] = CacheEntry(
            key=key,
            value=value,
            created_at=datetime.now(),
            expires_at=expires_at
        )
    
    def delete(self, key: str) -> bool:
        """Delete a key from cache."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        logger.info("Cache cleared")
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._cache:
            return
        
        # Find entry with lowest hits
        lru_key = min(self._cache.keys(), key=lambda k: self._cache[k].hits)
        del self._cache[lru_key]
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_hits = sum(e.hits for e in self._cache.values())
        
        return {
            'size': len(self._cache),
            'max_size': self.max_size,
            'total_hits': total_hits,
            'keys': list(self._cache.keys())
        }
    
    def cleanup_expired(self) -> int:
        """Remove expired entries and return count removed."""
        now = datetime.now()
        expired_keys = [
            k for k, v in self._cache.items()
            if v.expires_at and now > v.expires_at
        ]
        
        for key in expired_keys:
            del self._cache[key]
        
        return len(expired_keys)
