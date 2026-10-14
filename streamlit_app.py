"""
streamlit_app.py
Root entry point for Streamlit Community Cloud deployment.
Dispatches execution to app/app.py while ensuring all local imports
and paths resolve seamlessly.
"""

from pathlib import Path
import sys
import runpy

# Ensure app directory is at the front of sys.path
APP_DIR = Path(__file__).resolve().parent / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# Launch main app
APP_SCRIPT = APP_DIR / "app.py"
runpy.run_path(str(APP_SCRIPT), run_name="__main__")
