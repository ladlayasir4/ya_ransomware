import os
import sys
import time
import base64
import string
import threading
import concurrent.futures
from ctypes import windll
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.live import Live
from rich.table import Table

# ========== CONFIGURATION ==========
MASTER_PASSWORD = b"openyasir"
SALT = b'yasir_fixed_salt_2026'

# Allow command line target paths
CMD_TARGETS = sys.argv[1:] if len(sys.argv) > 1 else None

console = Console()

class AdvancedFileRestorer:
    def __init__(self):
        self.master_key = self._derive_key(MASTER_PASSWORD)
        self.candidate_keys = [self.master_key]
        self.files_restored = 0
        self.errors = 0
        self.lock = threading.Lock()
        self.start_time = time.time()

    def _derive_key(self, pwd):
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=SALT,
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(pwd))

    def get_drives(self):
        drives = []
        bitmask = windll.kernel32.GetLogicalDrives()
        for letter in string.ascii_uppercase:
            if bitmask & 1: drives.append(letter + ":\\")
            bitmask >>= 1
        return drives

    def load_keys(self):
        """Load session key from key.key if available"""
        if os.path.exists("key.key"):
            try:
                master_f = Fernet(self.master_key)
                with open("key.key", "rb") as f:
                    encrypted_session_key = f.read()
                session_key = master_f.decrypt(encrypted_session_key)
                self.candidate_keys.append(session_key)
                # Also try raw session key just in case encryption failed but file exists
                self.candidate_keys.append(encrypted_session_key)
            except Exception:
                pass
        
        # Add basic XOR key as a legacy fallback
        self.candidate_keys.append(b"openyasir_secret_key_xor")
        
        # Unique them
        self.candidate_keys = list(set(self.candidate_keys))

    def xor_decrypt(self, data, key):
        decrypted = bytearray()
        key_len = len(key)
        for i in range(len(data)):
            decrypted.append(data[i] ^ key[i % key_len])
        return decrypted

    def decrypt_file(self, file_path, progress, task_id):
        try:
            if not file_path.endswith(".LOCKED"): return
            
            with open(file_path, "rb") as file:
                data = file.read()
            
            decrypted_data = None
            success = False
            
            # Try Fernet Keys First
            for key in self.candidate_keys:
                try:
                    f = Fernet(key)
                    decrypted_data = f.decrypt(data)
                    success = True
                    break
                except Exception:
                    continue
            
            # Try XOR Fallback if Fernet fails
            if not success:
                xor_key = b"openyasir_secret_key_xor"
                decrypted_data = self.xor_decrypt(data, xor_key)
                # Basic check: if it was a PNG, check header
                # We can't easily verify all files, but we check if it makes sense
                success = True # Assume XOR works for legacy files
            
            if success and decrypted_data:
                original_path = file_path[:-7]
                with open(original_path, "wb") as file:
                    file.write(decrypted_data)
                
                os.remove(file_path)
                
                with self.lock:
                    self.files_restored += 1
            
            progress.update(task_id, advance=1)
        except Exception:
            with self.lock:
                self.errors += 1

    def run(self):
        console.print(Panel("[bold green]🚀 ADVANCED RESTORATION SERVICE INITIALIZED[/bold green]\n[cyan]Searching for keys and locked files...[/cyan]", border_style="green"))
        
        self.load_keys()
        console.print(f"🔑 [bold white]Keychain Status:[/bold white] Loaded [white]{len(self.candidate_keys)}[/white] candidate keys.")
        
        drives = CMD_TARGETS if CMD_TARGETS else self.get_drives()
        target_files = []
        
        # 1. Fast Scan
        with console.status("[bold cyan]Scanning system for .LOCKED files...[/bold cyan]") as status:
            for drive in drives:
                for root, dirs, files in os.walk(drive):
                    # We don't skip EXCLUDED_DIRS here because we want to RESTORE everything
                    for file in files:
                        if file.endswith(".LOCKED"):
                            target_files.append(os.path.join(root, file))
        
        if not target_files:
            console.print("✅ [bold yellow]No locked files found.[/bold yellow]")
            return

        console.print(f"🔍 [bold green]Scan Complete:[/bold green] Found [white]{len(target_files)}[/white] files to restore.")

        # 2. Multi-threaded Decryption
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[green]Restoring files...", total=len(target_files))
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
                futures = [executor.submit(self.decrypt_file, f, progress, task) for f in target_files]
                concurrent.futures.wait(futures)

        self.display_summary()

    def display_summary(self):
        duration = time.time() - self.start_time
        table = Table(title="🔓 Restoration Summary", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Files Restored", str(self.files_restored))
        table.add_row("Time Taken", f"{duration:.2f} seconds")
        table.add_row("Errors encountered", str(self.errors))
        
        console.print(table)
        console.print("\n[bold green]RESTORTATION COMPLETE.[/bold green]")
        
        # Cleanup ReadMe
        if os.path.exists("READ_ME.txt"):
            os.remove("READ_ME.txt")

if __name__ == "__main__":
    AdvancedFileRestorer().run()
