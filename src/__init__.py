"""
pharma-pricing-copilot — src package
=====================================
Top-level package for the Pharma Pricing Copilot project.

Exposes the main public API surface so that downstream modules and the
Streamlit app can import components from a single, stable namespace.

Modules
-------
data_simulator   : Synthetic pharmaceutical pricing data generation.
anomaly_detection: Statistical and ML-based pricing anomaly detection.
rag_pipeline     : Retrieval-Augmented Generation pipeline for regulatory docs.
agent            : Claude-powered conversational pricing analysis agent.
utils            : Shared helpers (logging, config, type aliases, etc.).
"""

__version__ = "0.1.0"
__author__ = "Pharma Pricing Copilot Team"

from src.utils import get_logger  # noqa: F401 — re-export for convenience

__all__ = [
    "data_simulator",
    "anomaly_detection",
    "rag_pipeline",
    "agent",
    "utils",
]
