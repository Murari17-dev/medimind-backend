import os

# parser.py requires a key at import time. Tests never reach the provider.
os.environ.setdefault("OPENROUTER_API_KEY", "test-key")
