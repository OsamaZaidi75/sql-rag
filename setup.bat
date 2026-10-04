@echo off
echo ===================================================
echo   Setting up SQL RAG Studio Environment
echo ===================================================

echo Installing required Python packages...
pip install -r requirements.txt

echo.
echo Running self-diagnostic tests...
python -m pytest -v

echo.
echo ===================================================
echo   Setup Complete!
echo   To start the app, run:
echo   streamlit run app.py
echo ===================================================
pause
