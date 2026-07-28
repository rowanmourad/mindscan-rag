"""
config.py - Centralized configuration for the Brain Tumor Detection API.

Reads settings from environment variables (with local-dev defaults) so that
secrets and per-environment values (server name, credentials, allowed
origins) never live directly in source code.

To use, create a `.env` file next to app.py (and add it to .gitignore):

    DB_DRIVER=ODBC Driver 17 for SQL Server
    DB_SERVER=YOUR_SERVER_NAME
    DB_NAME=MindScanDB
    DB_TRUSTED_CONNECTION=yes
    # If not using trusted connection:
    # DB_TRUSTED_CONNECTION=no
    # DB_USER=sa
    # DB_PASSWORD=your_password
    ALLOWED_ORIGINS=https://preview--neural-scan-guard.lovable.app,https://neural-scan-guard.lovable.app,http://localhost:5173
"""
from __future__ import annotations

import os

# Optional: load a local .env file if python-dotenv is installed.
# (pip install python-dotenv). Safe no-op if the package isn't present.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class Settings:
    # ── SQL Server connection ────────────────────────────────────────────
    DB_DRIVER: str = os.environ.get("DB_DRIVER", "ODBC Driver 17 for SQL Server")
    DB_SERVER: str = os.environ.get("DB_SERVER", "YOUR_SERVER_NAME")
    DB_NAME: str = os.environ.get("DB_NAME", "MindScanDB")
    DB_TRUSTED_CONNECTION: bool = os.environ.get(
        "DB_TRUSTED_CONNECTION", "yes"
    ).lower() in ("yes", "true", "1")
    DB_USER: str = os.environ.get("DB_USER", "")
    DB_PASSWORD: str = os.environ.get("DB_PASSWORD", "")

    @property
    def connection_string(self) -> str:
        base = f"DRIVER={{{self.DB_DRIVER}}};SERVER={self.DB_SERVER};DATABASE={self.DB_NAME};"
        if self.DB_TRUSTED_CONNECTION:
            return base + "Trusted_Connection=yes;"
        return base + f"UID={self.DB_USER};PWD={self.DB_PASSWORD};"

    # ── CORS ──────────────────────────────────────────────────────────────
    # Comma-separated list in env; falls back to the two known frontend
    # origins plus localhost for local dev. NOTE: do not combine "*" with
    # allow_credentials=True — browsers reject that combination.
    ALLOWED_ORIGINS: list[str] = [
        o.strip()
        for o in os.environ.get(
            "ALLOWED_ORIGINS",
            "https://preview--neural-scan-guard.lovable.app,"
            "https://neural-scan-guard.lovable.app,"
            "http://localhost:5173",
        ).split(",")
        if o.strip()
    ]


settings = Settings()
