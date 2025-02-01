# config/settings.py

from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Access the variables from the environment
PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Check if the required environment variables are set
if not PROJECT_ID:
    raise ValueError(
        "ERROR: PROJECT_ID environment variable is missing. Please set it in the .env file."
    )
if not LOCATION:
    raise ValueError(
        "ERROR: LOCATION environment variable is missing. Please set it in the .env file."
    )
if not GOOGLE_APPLICATION_CREDENTIALS:
    raise ValueError(
        "ERROR: GOOGLE_APPLICATION_CREDENTIALS environment variable is missing. Please set it in the .env file."
    )

# Optional: Set default paths for the progress note and HCC codes if not specified
PROGRESS_NOTE_PATH = os.getenv("PROGRESS_NOTE_PATH", "data/progress_notes/")
HCC_CODES_PATH = os.getenv("HCC_CODES_PATH", "data/hcc_codes/HCC_relevant_codes.csv")
