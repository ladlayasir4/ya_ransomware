import os
import sys
import base64
import string
import time
from ctypes import windll
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# ========== CONFIGURATION ==========
def get_drives():
    drives = []
    bitmask = windll.kernel32.GetLogicalDrives()
    for letter in string.ascii_uppercase:
        if bitmask & 1: drives.append(letter + ":\\")
        bitmask >>= 1
    return drives

TARGET_PATHS = get_drives()

# MUST MATCH FILEGUARDIAN EXACTLY
MASTER_PASSWORD = b"yasirunlock"
SALT = b'fixed_salt_1234' 

class FileRestorer:
    def __init__(self):
        self.key = None
        self.files_restored = 0
        self.errors = 0

    def derive_key(self):
        """Derive 32-byte URL-safe base64 key from password"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=SALT,
            iterations=100000,
        )
        self.key = base64.urlsafe_b64encode(kdf.derive(MASTER_PASSWORD))
        return self.key

    def decrypt_file(self, file_path):
        """Decrypt a single file"""
        try:
            if not file_path.endswith(".LOCKED"): return False
            
            with open(file_path, "rb") as f: data = f.read()
            fernet = Fernet(self.key)
            decrypted_data = fernet.decrypt(data)
            
            original_path = file_path[:-7] 
            with open(original_path, "wb") as f: f.write(decrypted_data)
            
            os.remove(file_path)
            print(f"RESTORED: {os.path.basename(original_path)}")
            self.files_restored += 1
            return True
        except:
            self.errors += 1
            return False

    def scan_and_restore(self):
        self.derive_key()
        print(f"🚑 RESTORER STARTED - AUTO MODE")
        
        for root_path in TARGET_PATHS:
            if not os.path.exists(root_path): continue
            for root, dirs, files in os.walk(root_path):
                for file in files:
                    if file.endswith(".LOCKED"):
                        self.decrypt_file(os.path.join(root, file))
            
            # Remove Note
            try:
                msg_path = os.path.join(root_path, "READ_ME.txt")
                if os.path.exists(msg_path): os.remove(msg_path)
            except: pass

        print(f"\n✅ DONE. Restored: {self.files_restored}")
        # Keep window open briefly so user sees result
        time.sleep(5)

if __name__ == "__main__":
    FileRestorer().scan_and_restore()
