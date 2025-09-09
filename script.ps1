# PowerShell Script to Set Up and Run the ADVANCED AI Risk Prediction Engine
# This script handles the PyTorch model and the new API server.

# --- Step 1: Check for Python ---
Write-Host "Step 1: Checking for Python..." -ForegroundColor Green
try {
    $pythonVersion = python --version
    Write-Host "Python is installed: $pythonVersion"
} catch {
    Write-Host "Python not found. Please install Python 3.8+ and add it to your PATH." -ForegroundColor Red
    exit 1
}


# --- Step 4: Install Required Packages ---
Write-Host "Step 4: Installing dependencies from requirements.txt..." -ForegroundColor Green
# Ensure pip is up to date
python -m pip install --upgrade pip
pip install -r requirements.txt

# --- Step 5: Train Model and Launch Server ---
Write-Host "Step 5: Training model and launching the API server..." -ForegroundColor Cyan
Write-Host "(This single command will now handle everything.)"
Write-Host "The server will be running at http://127.0.0.1:5001" -ForegroundColor Cyan
Write-Host "Open your 'dashboard.html' file in a browser to view the application." -ForegroundColor Cyan
Write-Host "Press CTRL+C in this terminal to stop the server." -ForegroundColor Yellow

# This single command now runs the entire process from your Python script.
python chronic.py 

# --- End of Script ---
Write-Host "Server stopped. To deactivate the virtual environment, type 'deactivate'." -ForegroundColor Green

