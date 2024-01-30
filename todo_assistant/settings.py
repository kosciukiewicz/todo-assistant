import os
from pathlib import Path

from pydantic import BaseSettings

_ENV_FILE = os.getenv("_ENV_FILE", default=str(Path(__file__).parent.parent / ".env"))


class Settings(BaseSettings):
    OPENAI_API_KEY: str
    NOTION_API_KEY: str
    NOTION_DATABASE_ID: str
    MODEL_NAME: str
    MODEL_VERBOSE: bool = False

    class Config:
        env_file = _ENV_FILE
        env_file_encoding = "utf-8"
