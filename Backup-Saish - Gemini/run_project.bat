@echo off
echo ====================================================
echo Starting NeuroLens Web Server...
echo Please KEEP THIS BLACK WINDOW OPEN while using the app!
echo ====================================================
echo.
echo Opening your web browser to http://localhost:8080...

:: Wait 2 seconds before opening browser to ensure server starts
timeout /t 2 /nobreak > NUL
start http://localhost:8080

:: Start the python server
python -m http.server 8080
