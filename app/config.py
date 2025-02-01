# config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Google Cloud Configuration
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION")

# Ensure necessary environment variables are set
if not all([GOOGLE_APPLICATION_CREDENTIALS, PROJECT_ID, LOCATION]):
    raise ValueError(
        "Some environment variables are missing. Please check your .env file."
    )
