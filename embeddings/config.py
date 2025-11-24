import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    JSON_FOLDER = "data/json/"
    OPENAI_KEY = os.getenv("OPENAI_API_KEY")
    EMBEDDING_MODEL = "text-embedding-3-large"

settings = Settings()
