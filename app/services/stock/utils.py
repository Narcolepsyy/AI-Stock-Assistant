import numpy as np
from datetime import datetime
from typing import Any, Optional
import threading
from cachetools import TTLCache

class ThreadSafeCache:
    """A thread-safe wrapper around TTLCache to reduce lock contention."""
    
    def __init__(self, maxsize: int, ttl: int):
        self._cache = TTLCache(maxsize=maxsize, ttl=ttl)
        self._lock = threading.RLock()
    
    def get(self, key, default=None):
        """Get item from cache in a thread-safe manner."""
        try:
            with self._lock:
                return self._cache.get(key, default)
        except Exception:
            return default
    
    def set(self, key, value):
        """Set item in cache in a thread-safe manner."""
        try:
            with self._lock:
                self._cache[key] = value
        except Exception:
            pass
    
    def __contains__(self, key):
        """Check if key exists in cache."""
        try:
            with self._lock:
                return key in self._cache
        except Exception:
            return False

def safe_float(value: Any) -> Optional[float]:
    """Coerce a value to float if possible, guarding against NaN/inf."""
    try:
        if value is None:
            return None
        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            return None
        if isinstance(value, (int, np.integer)):
            return float(value)
        if isinstance(value, (np.floating,)):
            val = float(value)
            if np.isnan(val) or np.isinf(val):
                return None
            return val
        val = float(value)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    except Exception:
        return None

def safe_int(value: Any) -> Optional[int]:
    """Coerce a value to int if possible, guarding against NaN."""
    try:
        if value is None:
            return None
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, (float, np.floating)):
            if np.isnan(value) or np.isinf(value):
                return None
            return int(round(float(value)))
        return int(value)
    except Exception:
        return None

def to_timestamp_str(value: Any) -> Optional[str]:
    """Normalize timestamps to ISO 8601 strings with 'T' separator."""
    if value is None:
        return None
    try:
        if hasattr(value, "to_pydatetime"):
            dt = value.to_pydatetime()
        elif isinstance(value, datetime):
            dt = value
        else:
            return str(value)
        iso = dt.isoformat()
        return iso.replace(" ", "T")
    except Exception:
        try:
            return str(value)
        except Exception:
            return None
