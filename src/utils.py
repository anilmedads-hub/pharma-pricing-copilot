"""
utils.py
========
Shared utilities for the Pharma Pricing Copilot project.

Provides:
  - Centralised logging configuration (``get_logger``)
  - Environment / configuration loading (``load_config``)
  - Common type aliases used across modules
  - Data validation helpers
  - DataFrame I/O helpers (CSV / Parquet read-write wrappers)
  - Currency and percentage formatting helpers

All public symbols are re-exported from ``src.__init__`` for convenience.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

PriceHistoryDF = pd.DataFrame   # alias — DataFrame conforming to price history schema
CatalogDF = pd.DataFrame        # alias — DataFrame conforming to drug catalog schema
CompetitiveDF = pd.DataFrame    # alias — DataFrame conforming to competitive landscape schema
JSONDict = dict[str, Any]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_LOG_FORMAT = "%(asctime)s [%(levelname)-8s] %(name)s — %(message)s"
_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_LOGGING_CONFIGURED = False


def configure_logging(level: int | str = logging.INFO) -> None:
    """Configure root logger with a consistent format.

    Safe to call multiple times — configuration is applied only once.

    Parameters
    ----------
    level:
        Logging level (e.g. ``logging.DEBUG`` or the string ``'DEBUG'``).
    """
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT))

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)

    # Quieten noisy third-party loggers
    for lib in ("httpx", "httpcore", "anthropic", "urllib3", "faiss"):
        logging.getLogger(lib).setLevel(logging.WARNING)

    _LOGGING_CONFIGURED = True


def get_logger(name: str, level: int | str = logging.INFO) -> logging.Logger:
    """Return a named logger, ensuring root logging is configured first.

    Parameters
    ----------
    name:
        Logger name — typically ``__name__`` of the calling module.
    level:
        Logging level for this specific logger.

    Returns
    -------
    logging.Logger
    """
    configure_logging()
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


# ---------------------------------------------------------------------------
# Environment / configuration
# ---------------------------------------------------------------------------


def load_config(env_file: str | Path = ".env") -> dict[str, str]:
    """Load environment variables from *env_file* and return relevant config.

    Parameters
    ----------
    env_file:
        Path to the ``.env`` file.  Relative paths are resolved from the
        current working directory.

    Returns
    -------
    dict[str, str]
        Dictionary of relevant config keys (never exposes secrets as values).
    """
    load_dotenv(dotenv_path=env_file, override=False)
    config = {
        "ANTHROPIC_API_KEY": "***set***" if os.environ.get("ANTHROPIC_API_KEY") else "MISSING",
        "LOG_LEVEL": os.environ.get("LOG_LEVEL", "INFO"),
        "DATA_DIR": os.environ.get("DATA_DIR", "data"),
        "RAW_DATA_DIR": os.environ.get("RAW_DATA_DIR", "data/raw"),
        "PROCESSED_DATA_DIR": os.environ.get("PROCESSED_DATA_DIR", "data/processed"),
        "VECTOR_STORE_PATH": os.environ.get("VECTOR_STORE_PATH", "data/processed/vector_store"),
        "STREAMLIT_PORT": os.environ.get("STREAMLIT_PORT", "8501"),
        "RANDOM_SEED": os.environ.get("RANDOM_SEED", "42"),
    }
    return config


def require_env(key: str) -> str:
    """Return the value of an environment variable or raise :class:`RuntimeError`.

    Parameters
    ----------
    key:
        Environment variable name.

    Returns
    -------
    str
        The variable's value.

    Raises
    ------
    RuntimeError
        If *key* is not set.
    """
    value = os.environ.get(key)
    if not value:
        raise RuntimeError(
            f"Required environment variable '{key}' is not set. "
            f"Copy .env.example → .env and fill in your values."
        )
    return value


# ---------------------------------------------------------------------------
# Data I/O helpers
# ---------------------------------------------------------------------------


def save_dataframe(df: pd.DataFrame, path: str | Path, fmt: str = "parquet") -> Path:
    """Persist *df* to disk in the requested format.

    Parameters
    ----------
    df:
        DataFrame to save.
    path:
        Destination file path (without extension — extension is added based on *fmt*).
    fmt:
        ``'parquet'`` (default) or ``'csv'``.

    Returns
    -------
    pathlib.Path
        Resolved path where the file was written.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        full_path = p.with_suffix(".parquet")
        df.to_parquet(full_path, index=False)
    elif fmt == "csv":
        full_path = p.with_suffix(".csv")
        df.to_csv(full_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {fmt!r}. Choose 'parquet' or 'csv'.")
    return full_path


def load_dataframe(path: str | Path) -> pd.DataFrame:
    """Load a DataFrame from a Parquet or CSV file.

    Parameters
    ----------
    path:
        Path to the file.

    Returns
    -------
    pandas.DataFrame

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file extension is not recognised.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {p}")
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    if p.suffix == ".csv":
        return pd.read_csv(p)
    raise ValueError(f"Cannot load '{p.suffix}' files. Supported: .parquet, .csv")


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_price_history_schema(df: pd.DataFrame) -> list[str]:
    """Check that *df* has the expected price history columns.

    Returns
    -------
    list[str]
        List of validation error messages.  Empty list means valid.
    """
    required = {"ndc", "brand_name", "period", "wac_per_unit"}
    missing = required - set(df.columns)
    errors: list[str] = []
    if missing:
        errors.append(f"Missing required columns: {missing}")
    if "wac_per_unit" in df.columns and df["wac_per_unit"].lt(0).any():
        errors.append("wac_per_unit contains negative values.")
    if "gross_to_net_pct" in df.columns:
        oob = df["gross_to_net_pct"].dropna()
        if (oob < 0).any() or (oob > 1).any():
            errors.append("gross_to_net_pct must be between 0 and 1.")
    return errors


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def fmt_usd(value: float, decimals: int = 2) -> str:
    """Format *value* as a USD currency string.

    >>> fmt_usd(1234.5)
    '$1,234.50'
    """
    return f"${value:,.{decimals}f}"


def fmt_pct(value: float, decimals: int = 1) -> str:
    """Format *value* (a decimal fraction) as a percentage string.

    >>> fmt_pct(0.1523)
    '15.2%'
    """
    return f"{value * 100:.{decimals}f}%"


def fmt_date(d: date | datetime | str) -> str:
    """Return a consistently formatted date string (``YYYY-MM-DD``)."""
    if isinstance(d, (date, datetime)):
        return d.strftime("%Y-%m-%d")
    return str(d)[:10]


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def project_root() -> Path:
    """Return the project root directory (two levels up from this file)."""
    return Path(__file__).resolve().parent.parent


def data_dir(subdir: str = "") -> Path:
    """Return the absolute path to ``data/`` (or a subdirectory thereof)."""
    base = project_root() / "data"
    return (base / subdir) if subdir else base
