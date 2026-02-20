import redis


REDIS_URL = "rediss://default:ASkAAAIncDE1MDc0ZWRkZGIxMWI0OWQ1OTBkMzQ3YzRiYzczY2IwY3AxMTA0OTY@creative-gazelle-10496.upstash.io:6379"

def test_redis():
    try:
        # Use from_url to handle the protocol, password, and host automatically
        r = redis.from_url(REDIS_URL, decode_responses=True)
        
        r.set("ping", "pong")
        result = r.get("ping")
        print(f"✅ Connection Successful! Response: {result}")
        
    except Exception as e:
        print(f"Failed to connect: {e}")
        print("\nDEBUG TIPS:")
        print("- Ensure the URL starts with 'rediss://' (with two 's' for SSL)")
        print("- Ensure there are no spaces at the start or end of the URL")

if __name__ == "__main__":
    test_redis()