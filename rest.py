import os
import base64

# 32-byte secure random key
key = base64.urlsafe_b64encode(os.urandom(32)).decode()

print("Your HMAC key:", key)
