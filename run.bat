@echo off
echo ==========================================
echo   Health Insurance Fraud Detection
echo ==========================================

echo [1/3] Installing dependencies...
call npm install

echo [2/3] Generating synthetic dataset...
call npm run generate-data

echo [3/3] Starting the application...
echo The model will be trained on startup.
call npm run dev

pause
