import redis
import hashlib 
import streamlit as st
import time
from typing import List, Optional

class LLMLoadBalancer:
    def __init__(self):
        #key pool
        self.api_keys = st.secrets.get("GOOGLE_API_KEYS", [])
        if not self.api_keys:
            self.api_keys = [st.secrets["GOOGLE_API_KEY"]]
            
        # 2. Redis Configuration
        try:
            self.redis_client = redis.Redis(
                host=st.secrets["REDIS_HOST"],
                port=st.secrets["REDIS_PORT"],
                password=st.secrets["REDIS_PASSWORD"],
                decode_responses=True,
                ssl=True
            )
            self.use_redis = True
        except Exception as e:
            st.warning("Redis not connected. Falling back to basic Round-Robin without global rate limiting.")
            self.use_redis = False

        self.rpm_limit = 15  # Google Free Tier limit is ~15 RPM
        self.current_index = 0

    def _get_key_rpm(self, key_hash: str) -> int:
        """Checks the current request count for a specific key in the last 60 seconds."""
        if not self.use_redis:
            return 0
        current_minute = int(time.time() / 60)
        redis_key = f"rpm:{key_hash}:{current_minute}"
        return int(self.redis_client.get(redis_key) or 0)

    def _increment_key_usage(self, key_hash: str):
        """Increments the usage count for a key."""
        if not self.use_redis:
            return
        current_minute = int(time.time() / 60)
        redis_key = f"rpm:{key_hash}:{current_minute}"
        pipe = self.redis_client.pipeline()
        pipe.incr(redis_key)
        pipe.expire(redis_key, 120)  # Keep for 2 minutes
        pipe.execute()

    def get_next_available_key(self) -> str:
        attempts = 0
        while attempts < len(self.api_keys):
            key = self.api_keys[self.current_index]
            
            # FIX: Ensure key is a string and use a stable hash for Redis
            if isinstance(key, list):
                key = key[0] # Grab the first string if it's accidentally a nested list
            
            key_hash = hashlib.md5(str(key).encode()).hexdigest()
            
            current_rpm = self._get_key_rpm(key_hash)
            
            if current_rpm < self.rpm_limit:
                self._increment_key_usage(key_hash)
                self.current_index = (self.current_index + 1) % len(self.api_keys)
                return str(key) # Ensure we return the string key
            
            self.current_index = (self.current_index + 1) % len(self.api_keys)
            attempts += 1
            
        return str(self.api_keys[0])