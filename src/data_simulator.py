"""
data_simulator.py
=================
Realistic pharmaceutical pricing data simulator for PBM / healthcare analytics.

Generates a 15,000-row transactional dataset that covers five real-world drug
pricing benchmarks across 50 branded drugs, 200 pharmacies, and 15 US states.

Pricing benchmarks simulated
-----------------------------
WAC   — Wholesale Acquisition Cost (manufacturer list price)
ASP   — Average Selling Price (Medicare Part B benchmark, ≈ WAC × 0.95)
AMP   — Average Manufacturer Price (Medicaid benchmark, ≈ WAC × 0.85)
340B  — 340B Drug Pricing Program price (safety-net hospitals, ≈ WAC × 0.70)
GTN   — Gross-to-Net price (WAC minus all rebates/discounts, ≈ WAC × 0.60–0.75)
NADAC — National Average Drug Acquisition Cost (≈ AMP × 1.01–1.05)

Drug classes
------------
Diabetes · Cardiovascular · Oncology · Immunology · Respiratory  (10 drugs each)

Pharmacy chains
---------------
CVS · Walgreens · Rite Aid · Walmart · Kroger · Independent  (200 pharmacies / 15 states)

Anomaly types injected (≈ 7 % of records)
------------------------------------------
price_spike    — actual_price > WAC × 2.8
margin_break   — margin_percent < −15 %
gtn_deviation  — gtn_price deviates > 30 % from expected WAC ratio
volume_outlier — volume_units > drug mean + 4 × std

Usage
-----
>>> from src.data_simulator import PricingDataSimulator
>>> sim = PricingDataSimulator(seed=42)
>>> df  = sim.generate()
>>> anom = sim.get_anomalies()
>>> stats = sim.get_summary_stats()
>>> sim.build_sqlite_db()
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from faker import Faker

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------

logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Static reference data
# ---------------------------------------------------------------------------

# 50 drugs: 10 per class — (drug_name, manufacturer, wac_low, wac_high)
_DRUG_CATALOG: list[dict[str, Any]] = [
    # ── Diabetes ──────────────────────────────────────────────────────────
    {"drug_name": "Metformin",       "manufacturer": "Teva Pharmaceuticals",    "drug_class": "Diabetes",       "wac_low": 25,    "wac_high": 55},
    {"drug_name": "Ozempic",         "manufacturer": "Novo Nordisk",            "drug_class": "Diabetes",       "wac_low": 850,   "wac_high": 960},
    {"drug_name": "Jardiance",       "manufacturer": "Boehringer Ingelheim",    "drug_class": "Diabetes",       "wac_low": 530,   "wac_high": 620},
    {"drug_name": "Trulicity",       "manufacturer": "Eli Lilly",               "drug_class": "Diabetes",       "wac_low": 720,   "wac_high": 820},
    {"drug_name": "Lantus",          "manufacturer": "Sanofi",                  "drug_class": "Diabetes",       "wac_low": 290,   "wac_high": 380},
    {"drug_name": "Humalog",         "manufacturer": "Eli Lilly",               "drug_class": "Diabetes",       "wac_low": 270,   "wac_high": 340},
    {"drug_name": "Victoza",         "manufacturer": "Novo Nordisk",            "drug_class": "Diabetes",       "wac_low": 640,   "wac_high": 720},
    {"drug_name": "Farxiga",         "manufacturer": "AstraZeneca",             "drug_class": "Diabetes",       "wac_low": 490,   "wac_high": 570},
    {"drug_name": "Januvia",         "manufacturer": "Merck",                   "drug_class": "Diabetes",       "wac_low": 420,   "wac_high": 500},
    {"drug_name": "Mounjaro",        "manufacturer": "Eli Lilly",               "drug_class": "Diabetes",       "wac_low": 980,   "wac_high": 1070},
    # ── Cardiovascular ────────────────────────────────────────────────────
    {"drug_name": "Lipitor",         "manufacturer": "Pfizer",                  "drug_class": "Cardiovascular", "wac_low": 15,    "wac_high": 45},
    {"drug_name": "Crestor",         "manufacturer": "AstraZeneca",             "drug_class": "Cardiovascular", "wac_low": 20,    "wac_high": 55},
    {"drug_name": "Eliquis",         "manufacturer": "Bristol-Myers Squibb",    "drug_class": "Cardiovascular", "wac_low": 480,   "wac_high": 570},
    {"drug_name": "Xarelto",         "manufacturer": "Janssen Pharmaceuticals", "drug_class": "Cardiovascular", "wac_low": 490,   "wac_high": 580},
    {"drug_name": "Lisinopril",      "manufacturer": "Lupin Pharmaceuticals",   "drug_class": "Cardiovascular", "wac_low": 10,    "wac_high": 30},
    {"drug_name": "Metoprolol",      "manufacturer": "AstraZeneca",             "drug_class": "Cardiovascular", "wac_low": 12,    "wac_high": 35},
    {"drug_name": "Amlodipine",      "manufacturer": "Pfizer",                  "drug_class": "Cardiovascular", "wac_low": 8,     "wac_high": 25},
    {"drug_name": "Entresto",        "manufacturer": "Novartis",                "drug_class": "Cardiovascular", "wac_low": 550,   "wac_high": 650},
    {"drug_name": "Brilinta",        "manufacturer": "AstraZeneca",             "drug_class": "Cardiovascular", "wac_low": 320,   "wac_high": 390},
    {"drug_name": "Repatha",         "manufacturer": "Amgen",                   "drug_class": "Cardiovascular", "wac_low": 580,   "wac_high": 680},
    # ── Oncology ──────────────────────────────────────────────────────────
    {"drug_name": "Keytruda",        "manufacturer": "Merck",                   "drug_class": "Oncology",       "wac_low": 9500,  "wac_high": 11000},
    {"drug_name": "Opdivo",          "manufacturer": "Bristol-Myers Squibb",    "drug_class": "Oncology",       "wac_low": 8200,  "wac_high": 9800},
    {"drug_name": "Herceptin",       "manufacturer": "Genentech",               "drug_class": "Oncology",       "wac_low": 5500,  "wac_high": 6800},
    {"drug_name": "Avastin",         "manufacturer": "Genentech",               "drug_class": "Oncology",       "wac_low": 4800,  "wac_high": 5900},
    {"drug_name": "Revlimid",        "manufacturer": "Bristol-Myers Squibb",    "drug_class": "Oncology",       "wac_low": 18000, "wac_high": 24000},
    {"drug_name": "Ibrance",         "manufacturer": "Pfizer",                  "drug_class": "Oncology",       "wac_low": 10500, "wac_high": 12000},
    {"drug_name": "Xtandi",          "manufacturer": "Astellas Pharma",         "drug_class": "Oncology",       "wac_low": 7200,  "wac_high": 8500},
    {"drug_name": "Imbruvica",       "manufacturer": "AbbVie",                  "drug_class": "Oncology",       "wac_low": 12000, "wac_high": 14500},
    {"drug_name": "Rituxan",         "manufacturer": "Genentech",               "drug_class": "Oncology",       "wac_low": 6000,  "wac_high": 7400},
    {"drug_name": "Kadcyla",         "manufacturer": "Genentech",               "drug_class": "Oncology",       "wac_low": 8800,  "wac_high": 10500},
    # ── Immunology ────────────────────────────────────────────────────────
    {"drug_name": "Humira",          "manufacturer": "AbbVie",                  "drug_class": "Immunology",     "wac_low": 5500,  "wac_high": 7200},
    {"drug_name": "Enbrel",          "manufacturer": "Amgen",                   "drug_class": "Immunology",     "wac_low": 4800,  "wac_high": 6200},
    {"drug_name": "Remicade",        "manufacturer": "Janssen Pharmaceuticals", "drug_class": "Immunology",     "wac_low": 4000,  "wac_high": 5500},
    {"drug_name": "Stelara",         "manufacturer": "Janssen Pharmaceuticals", "drug_class": "Immunology",     "wac_low": 14000, "wac_high": 17000},
    {"drug_name": "Dupixent",        "manufacturer": "Sanofi",                  "drug_class": "Immunology",     "wac_low": 3200,  "wac_high": 4000},
    {"drug_name": "Skyrizi",         "manufacturer": "AbbVie",                  "drug_class": "Immunology",     "wac_low": 12000, "wac_high": 15000},
    {"drug_name": "Rinvoq",          "manufacturer": "AbbVie",                  "drug_class": "Immunology",     "wac_low": 2800,  "wac_high": 3600},
    {"drug_name": "Tremfya",         "manufacturer": "Janssen Pharmaceuticals", "drug_class": "Immunology",     "wac_low": 10000, "wac_high": 13000},
    {"drug_name": "Cosentyx",        "manufacturer": "Novartis",                "drug_class": "Immunology",     "wac_low": 5000,  "wac_high": 6500},
    {"drug_name": "Taltz",           "manufacturer": "Eli Lilly",               "drug_class": "Immunology",     "wac_low": 4500,  "wac_high": 5800},
    # ── Respiratory ───────────────────────────────────────────────────────
    {"drug_name": "Advair Diskus",   "manufacturer": "GlaxoSmithKline",         "drug_class": "Respiratory",    "wac_low": 380,   "wac_high": 460},
    {"drug_name": "Symbicort",       "manufacturer": "AstraZeneca",             "drug_class": "Respiratory",    "wac_low": 350,   "wac_high": 430},
    {"drug_name": "Spiriva",         "manufacturer": "Boehringer Ingelheim",    "drug_class": "Respiratory",    "wac_low": 420,   "wac_high": 500},
    {"drug_name": "Breo Ellipta",    "manufacturer": "GlaxoSmithKline",         "drug_class": "Respiratory",    "wac_low": 360,   "wac_high": 440},
    {"drug_name": "Trelegy Ellipta", "manufacturer": "GlaxoSmithKline",         "drug_class": "Respiratory",    "wac_low": 580,   "wac_high": 680},
    {"drug_name": "Nucala",          "manufacturer": "GlaxoSmithKline",         "drug_class": "Respiratory",    "wac_low": 2800,  "wac_high": 3400},
    {"drug_name": "Fasenra",         "manufacturer": "AstraZeneca",             "drug_class": "Respiratory",    "wac_low": 2600,  "wac_high": 3200},
    {"drug_name": "Xolair",          "manufacturer": "Genentech",               "drug_class": "Respiratory",    "wac_low": 1200,  "wac_high": 1600},
    {"drug_name": "Albuterol",       "manufacturer": "Perrigo Company",         "drug_class": "Respiratory",    "wac_low": 50,    "wac_high": 90},
    {"drug_name": "Montelukast",     "manufacturer": "Teva Pharmaceuticals",    "drug_class": "Respiratory",    "wac_low": 20,    "wac_high": 55},
]

# Pharmacy chain seat counts — total 200
_CHAIN_SEATS: dict[str, int] = {
    "CVS":         40,
    "Walgreens":   40,
    "Rite Aid":    30,
    "Walmart":     35,
    "Kroger":      25,
    "Independent": 30,
}

_STATES: list[str] = ["CA", "TX", "FL", "NY", "IL", "PA", "OH", "GA", "NC", "MI", "WA", "AZ", "MA", "CO", "NJ"]

# Volume ranges (units per transaction) by drug class
_VOLUME_RANGES: dict[str, tuple[int, int]] = {
    "Diabetes":       (30, 300),
    "Cardiovascular": (30, 300),
    "Oncology":       (1,  10),
    "Immunology":     (1,  15),
    "Respiratory":    (30, 200),
}

# Anomaly types and their approximate share of the 7 % anomaly budget
_ANOMALY_TYPES: list[str] = ["price_spike", "margin_break", "gtn_deviation", "volume_outlier"]

_TOTAL_ROWS   = 15_000
_ANOMALY_RATE = 0.07

# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


class PricingDataSimulator:
    """Generate realistic pharmaceutical pricing transactional datasets.

    Parameters
    ----------
    seed:
        Integer random seed for full reproducibility.
    data_dir:
        Root directory for data outputs.  Sub-folders ``raw/`` and
        ``processed/`` are created automatically.
    n_rows:
        Total number of transaction rows to generate (default 15 000).
    anomaly_rate:
        Fraction of rows to mark as anomalies (default 0.07 → 7 %).
    """

    def __init__(
        self,
        seed: int = 42,
        data_dir: str | Path = "data",
        n_rows: int = _TOTAL_ROWS,
        anomaly_rate: float = _ANOMALY_RATE,
    ) -> None:
        self.seed = seed
        self.data_dir = Path(data_dir)
        self.n_rows = n_rows
        self.anomaly_rate = anomaly_rate

        self._rng = np.random.default_rng(seed)
        self._fake = Faker()
        Faker.seed(seed)

        self._raw_dir = self.data_dir / "raw"
        self._processed_dir = self.data_dir / "processed"
        self._raw_dir.mkdir(parents=True, exist_ok=True)
        self._processed_dir.mkdir(parents=True, exist_ok=True)

        self._df: pd.DataFrame | None = None   # cached after generate()
        self._drug_df: pd.DataFrame | None = None
        self._pharmacy_df: pd.DataFrame | None = None

        logger.info(
            "PricingDataSimulator initialised (seed=%d, n_rows=%d, anomaly_rate=%.1f%%)",
            seed, n_rows, anomaly_rate * 100,
        )

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def generate(self) -> pd.DataFrame:
        """Generate the full 15 000-row pharmaceutical pricing dataset.

        Steps
        -----
        1. Build drug and pharmacy catalogues.
        2. Sample transactions (drug × pharmacy × date × price).
        3. Derive all pricing benchmarks with realistic correlations.
        4. Inject anomalies at the configured rate.
        5. Persist to ``data/raw/drug_pricing.csv``.

        Returns
        -------
        pandas.DataFrame
            Full transaction table with schema documented in the module
            docstring.  Also cached internally for subsequent method calls.
        """
        logger.info("Starting data generation (%d rows)…", self.n_rows)

        drug_df     = self._build_drug_catalog()
        pharmacy_df = self._build_pharmacy_catalog()

        self._drug_df     = drug_df
        self._pharmacy_df = pharmacy_df

        rows = self._generate_transactions(drug_df, pharmacy_df)
        df   = self._compute_prices(rows)
        df   = self._inject_anomalies(df)

        # Final column ordering
        col_order = [
            "drug_id", "drug_name", "ndc_code", "manufacturer", "drug_class",
            "pharmacy_id", "pharmacy_name", "pharmacy_chain", "state",
            "wac_price", "asp_price", "amp_price", "price_340b", "gtn_price",
            "nadac_price", "actual_price", "margin_percent",
            "transaction_date", "volume_units",
            "is_anomaly", "anomaly_type",
        ]
        df = df[col_order]

        out_path = self._raw_dir / "drug_pricing.csv"
        df.to_csv(out_path, index=False)
        logger.info("Saved %d rows → %s", len(df), out_path)

        self._df = df
        return df

    def get_anomalies(self) -> pd.DataFrame:
        """Return only the anomalous rows from the generated dataset.

        Calls :meth:`generate` automatically if data has not yet been
        generated in this session.

        Returns
        -------
        pandas.DataFrame
            Subset of the full dataset where ``is_anomaly == True``.
            Saved to ``data/raw/anomalies.csv``.
        """
        if self._df is None:
            logger.info("Data not yet generated — calling generate() first.")
            self.generate()

        anomaly_df = self._df[self._df["is_anomaly"]].copy()
        out_path   = self._raw_dir / "anomalies.csv"
        anomaly_df.to_csv(out_path, index=False)
        logger.info(
            "Anomaly extract: %d rows (%.2f%%) → %s",
            len(anomaly_df),
            100 * len(anomaly_df) / len(self._df),
            out_path,
        )
        return anomaly_df

    def get_summary_stats(self) -> dict[str, Any]:
        """Compute and return a summary statistics dictionary.

        Keys
        ----
        total_records         : int   — total row count
        anomaly_count         : int   — number of anomalous rows
        anomaly_rate          : float — fraction of anomalous rows
        avg_margin            : float — mean margin_percent across all rows
        avg_gtn_discount_pct  : float — mean (WAC − GTN) / WAC across all rows
        top_5_anomalous_drugs : list  — drug names with most anomalies
        price_range_by_class  : dict  — min/max WAC per drug class

        Returns
        -------
        dict[str, Any]
        """
        if self._df is None:
            logger.info("Data not yet generated — calling generate() first.")
            self.generate()

        df = self._df

        gtn_discount = (df["wac_price"] - df["gtn_price"]) / df["wac_price"]

        top5 = (
            df[df["is_anomaly"]]
            .groupby("drug_name")
            .size()
            .nlargest(5)
            .index.tolist()
        )

        price_range: dict[str, dict[str, float]] = {}
        for cls, grp in df.groupby("drug_class"):
            price_range[cls] = {
                "wac_min": round(float(grp["wac_price"].min()), 2),
                "wac_max": round(float(grp["wac_price"].max()), 2),
                "wac_mean": round(float(grp["wac_price"].mean()), 2),
            }

        stats: dict[str, Any] = {
            "total_records":        int(len(df)),
            "anomaly_count":        int(df["is_anomaly"].sum()),
            "anomaly_rate":         round(float(df["is_anomaly"].mean()), 4),
            "avg_margin":           round(float(df["margin_percent"].mean()), 4),
            "avg_gtn_discount_pct": round(float(gtn_discount.mean()), 4),
            "top_5_anomalous_drugs": top5,
            "price_range_by_class": price_range,
        }

        logger.info(
            "Summary stats — total=%d | anomalies=%d | avg_margin=%.2f%% | avg_gtn_disc=%.2f%%",
            stats["total_records"],
            stats["anomaly_count"],
            stats["avg_margin"],
            stats["avg_gtn_discount_pct"] * 100,
        )
        return stats

    def build_sqlite_db(self) -> None:
        """Persist the dataset to a normalised SQLite database.

        Tables created
        --------------
        drugs          — drug master with base WAC and class
        pharmacies     — pharmacy master with chain and state
        transactions   — fact table (all 15 000 rows, FKs to drugs/pharmacies)
        anomalies      — subset view of transactions where is_anomaly = 1

        Database saved to ``data/processed/pricing.db``.
        """
        if self._df is None:
            logger.info("Data not yet generated — calling generate() first.")
            self.generate()

        db_path = self._processed_dir / "pricing.db"
        logger.info("Building SQLite database → %s", db_path)

        con = sqlite3.connect(db_path)
        try:
            self._create_db_schema(con)
            self._insert_drug_table(con)
            self._insert_pharmacy_table(con)
            self._insert_transaction_table(con)
            self._insert_anomaly_table(con)
            con.commit()
            logger.info("SQLite database committed — 4 tables written.")
        except Exception:
            con.rollback()
            logger.exception("Database build failed — rolled back.")
            raise
        finally:
            con.close()

    # -----------------------------------------------------------------------
    # Private — catalogue builders
    # -----------------------------------------------------------------------

    def _build_drug_catalog(self) -> pd.DataFrame:
        """Return a DataFrame of 50 drugs with assigned IDs and NDC codes.

        Returns
        -------
        pandas.DataFrame
            Columns: drug_id, drug_name, ndc_code, manufacturer, drug_class,
            wac_low, wac_high.
        """
        records = []
        for i, entry in enumerate(_DRUG_CATALOG, start=1):
            labeler = 10000 + i * 317          # deterministic 5-digit labeler
            product = 1000 + i * 13            # 4-digit product code
            package = 10 + (i % 8)            # 2-digit package code
            ndc     = f"{labeler:05d}-{product:04d}-{package:02d}"
            records.append({
                "drug_id":      f"DRG{i:03d}",
                "drug_name":    entry["drug_name"],
                "ndc_code":     ndc,
                "manufacturer": entry["manufacturer"],
                "drug_class":   entry["drug_class"],
                "wac_low":      entry["wac_low"],
                "wac_high":     entry["wac_high"],
            })
        df = pd.DataFrame(records)
        logger.debug("Drug catalog built: %d drugs across %d classes.", len(df), df["drug_class"].nunique())
        return df

    def _build_pharmacy_catalog(self) -> pd.DataFrame:
        """Return a DataFrame of 200 pharmacies distributed across chains and states.

        Pharmacy names are synthetically generated with Faker to look realistic.

        Returns
        -------
        pandas.DataFrame
            Columns: pharmacy_id, pharmacy_name, pharmacy_chain, state.
        """
        records = []
        pid     = 1
        for chain, count in _CHAIN_SEATS.items():
            states_cycle = (
                self._rng.choice(_STATES, size=count, replace=True).tolist()
            )
            for j in range(count):
                st   = states_cycle[j]
                city = self._fake.city()
                if chain == "Independent":
                    name = f"{self._fake.last_name()} Pharmacy"
                else:
                    name = f"{chain} — {city}"
                records.append({
                    "pharmacy_id":    f"PH{pid:04d}",
                    "pharmacy_name":  name,
                    "pharmacy_chain": chain,
                    "state":          st,
                })
                pid += 1
        df = pd.DataFrame(records)
        logger.debug(
            "Pharmacy catalog built: %d pharmacies across %d states.",
            len(df), df["state"].nunique(),
        )
        return df

    # -----------------------------------------------------------------------
    # Private — transaction generation
    # -----------------------------------------------------------------------

    def _generate_transactions(
        self,
        drug_df: pd.DataFrame,
        pharmacy_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Sample raw transaction rows (drug × pharmacy × date).

        Transactions are sampled with replacement from the drug and pharmacy
        catalogues, weighted by drug class volume profiles.

        Parameters
        ----------
        drug_df:
            Drug master table from :meth:`_build_drug_catalog`.
        pharmacy_df:
            Pharmacy master from :meth:`_build_pharmacy_catalog`.

        Returns
        -------
        pandas.DataFrame
            Raw rows; prices not yet computed.
        """
        n   = self.n_rows
        rng = self._rng

        # Random drug indices (uniform across all 50 drugs)
        drug_idx      = rng.integers(0, len(drug_df), size=n)
        pharmacy_idx  = rng.integers(0, len(pharmacy_df), size=n)

        # Transaction dates over the last 18 months
        today     = date.today()
        start_dt  = today - timedelta(days=548)   # ≈ 18 months
        date_offsets = rng.integers(0, 548, size=n)
        tx_dates  = [start_dt + timedelta(days=int(d)) for d in date_offsets]

        # Volume units — draw from class-specific range
        volume_units = np.empty(n, dtype=np.int64)
        for i, didx in enumerate(drug_idx):
            cls              = drug_df.iloc[didx]["drug_class"]
            lo, hi           = _VOLUME_RANGES[cls]
            volume_units[i]  = int(rng.integers(lo, hi + 1))

        rows = pd.DataFrame({
            "drug_id":          drug_df.iloc[drug_idx]["drug_id"].values,
            "drug_name":        drug_df.iloc[drug_idx]["drug_name"].values,
            "ndc_code":         drug_df.iloc[drug_idx]["ndc_code"].values,
            "manufacturer":     drug_df.iloc[drug_idx]["manufacturer"].values,
            "drug_class":       drug_df.iloc[drug_idx]["drug_class"].values,
            "_wac_low":         drug_df.iloc[drug_idx]["wac_low"].values.astype(float),
            "_wac_high":        drug_df.iloc[drug_idx]["wac_high"].values.astype(float),
            "pharmacy_id":      pharmacy_df.iloc[pharmacy_idx]["pharmacy_id"].values,
            "pharmacy_name":    pharmacy_df.iloc[pharmacy_idx]["pharmacy_name"].values,
            "pharmacy_chain":   pharmacy_df.iloc[pharmacy_idx]["pharmacy_chain"].values,
            "state":            pharmacy_df.iloc[pharmacy_idx]["state"].values,
            "transaction_date": tx_dates,
            "volume_units":     volume_units,
        }).reset_index(drop=True)

        logger.debug("Transaction skeleton: %d rows sampled.", len(rows))
        return rows

    def _compute_prices(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Derive all pricing benchmarks from WAC with realistic correlations.

        Correlation structure
        ---------------------
        WAC   — base list price drawn uniformly from [wac_low, wac_high]
                with small row-level noise (±3 %)
        ASP   — WAC × U(0.92, 0.97)     → Medicare average ≈ 0.95 × WAC
        AMP   — WAC × U(0.82, 0.88)     → Medicaid benchmark ≈ 0.85 × WAC
        340B  — WAC × U(0.67, 0.73)     → Safety-net ceiling ≈ 0.70 × WAC
        GTN   — WAC × U(0.60, 0.75)     → Net-of-rebates ≈ 0.67 × WAC
        NADAC — AMP × U(1.01, 1.06)     → Acquisition cost proxy
        actual— U(AMP, ASP)              → What pharmacy actually pays
        margin— (ASP − actual) / ASP    → Pharmacy margin on the transaction

        Parameters
        ----------
        rows:
            Raw transaction rows from :meth:`_generate_transactions`.

        Returns
        -------
        pandas.DataFrame
            Rows with all pricing columns populated.
        """
        rng = self._rng
        n   = len(rows)

        lo  = rows["_wac_low"].values
        hi  = rows["_wac_high"].values

        # Base WAC with ±3 % row-level noise
        wac_base  = rng.uniform(lo, hi)
        noise     = rng.uniform(0.97, 1.03, size=n)
        wac       = np.round(wac_base * noise, 2)

        asp       = np.round(wac * rng.uniform(0.92, 0.97, size=n), 2)
        amp       = np.round(wac * rng.uniform(0.82, 0.88, size=n), 2)
        b340      = np.round(wac * rng.uniform(0.67, 0.73, size=n), 2)
        gtn       = np.round(wac * rng.uniform(0.60, 0.75, size=n), 2)
        nadac     = np.round(amp  * rng.uniform(1.01, 1.06, size=n), 2)

        # actual_price ∈ [AMP, ASP] — what the pharmacy pays
        actual    = np.round(rng.uniform(amp, asp), 2)

        # margin_percent = (reimbursement − acquisition) / reimbursement × 100
        margin    = np.round((asp - actual) / asp * 100, 4)

        df = rows.drop(columns=["_wac_low", "_wac_high"]).copy()
        df["wac_price"]      = wac
        df["asp_price"]      = asp
        df["amp_price"]      = amp
        df["price_340b"]     = b340
        df["gtn_price"]      = gtn
        df["nadac_price"]    = nadac
        df["actual_price"]   = actual
        df["margin_percent"] = margin
        df["is_anomaly"]     = False
        df["anomaly_type"]   = None

        logger.debug("Prices computed for %d rows.", n)
        return df

    # -----------------------------------------------------------------------
    # Private — anomaly injection
    # -----------------------------------------------------------------------

    def _inject_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Corrupt ~7 % of rows with one of four anomaly types.

        Anomaly definitions
        -------------------
        price_spike
            ``actual_price`` set to ``wac_price × U(2.8, 4.0)``.
            Simulates data entry errors or fraudulent billing.

        margin_break
            ``margin_percent`` forced to ``U(−40, −15) %`` and
            ``actual_price`` adjusted accordingly.
            Simulates reimbursement below acquisition cost.

        gtn_deviation
            ``gtn_price`` set to ``wac_price × U(0.05, 0.25)`` (too low)
            or ``wac_price × U(0.85, 0.99)`` (too high — >30 % from expected).
            Simulates rebate reporting errors.

        volume_outlier
            ``volume_units`` set to ``drug_mean + U(4, 7) × drug_std``.
            Simulates suspicious bulk purchases or data duplication.

        Parameters
        ----------
        df:
            Full dataset with clean prices.

        Returns
        -------
        pandas.DataFrame
            Dataset with anomalies injected and ``is_anomaly`` / ``anomaly_type``
            columns populated.
        """
        rng     = self._rng
        n_total = len(df)
        n_anom  = int(n_total * self.anomaly_rate)

        # Per-drug volume statistics (for volume_outlier threshold)
        vol_stats = (
            df.groupby("drug_id")["volume_units"]
            .agg(["mean", "std"])
            .rename(columns={"mean": "vol_mean", "std": "vol_std"})
        )
        vol_stats["vol_std"] = vol_stats["vol_std"].fillna(1.0)

        # Select anomaly indices
        anomaly_idx = rng.choice(n_total, size=n_anom, replace=False)

        # Distribute types as evenly as possible
        type_assignments = _ANOMALY_TYPES * (n_anom // len(_ANOMALY_TYPES) + 1)
        rng.shuffle(type_assignments)
        type_assignments = type_assignments[:n_anom]

        for idx, atype in zip(anomaly_idx, type_assignments):
            df.at[idx, "is_anomaly"]   = True
            df.at[idx, "anomaly_type"] = atype

            wac    = float(df.at[idx, "wac_price"])
            asp    = float(df.at[idx, "asp_price"])
            did    = df.at[idx, "drug_id"]
            vmean  = float(vol_stats.at[did, "vol_mean"])
            vstd   = float(vol_stats.at[did, "vol_std"])

            if atype == "price_spike":
                # actual_price > WAC × 2.8
                spike_factor          = float(rng.uniform(2.8, 4.0))
                new_actual            = round(wac * spike_factor, 2)
                df.at[idx, "actual_price"]   = new_actual
                df.at[idx, "margin_percent"] = round((asp - new_actual) / asp * 100, 4)

            elif atype == "margin_break":
                # margin_percent < −15 %
                new_margin            = float(rng.uniform(-40.0, -15.0))
                # actual_price derived from margin: actual = asp × (1 − margin/100)
                new_actual            = round(asp * (1 - new_margin / 100), 2)
                df.at[idx, "margin_percent"] = round(new_margin, 4)
                df.at[idx, "actual_price"]   = new_actual

            elif atype == "gtn_deviation":
                # gtn_price deviates > 30 % from expected range (WAC × 0.60–0.75)
                if rng.random() < 0.5:
                    # Too low (< 0.30 × WAC)
                    new_gtn = round(wac * float(rng.uniform(0.05, 0.28)), 2)
                else:
                    # Too high (> 0.85 × WAC)
                    new_gtn = round(wac * float(rng.uniform(0.85, 0.99)), 2)
                df.at[idx, "gtn_price"] = new_gtn

            elif atype == "volume_outlier":
                # volume_units > mean + 4 × std
                multiplier            = float(rng.uniform(4.0, 7.0))
                new_vol               = int(vmean + multiplier * vstd)
                df.at[idx, "volume_units"] = max(new_vol, int(vmean) + 1)

        n_by_type = pd.Series(type_assignments).value_counts().to_dict()
        logger.info(
            "Injected %d anomalies (%.1f%%) — by type: %s",
            n_anom, self.anomaly_rate * 100, n_by_type,
        )
        return df

    # -----------------------------------------------------------------------
    # Private — SQLite helpers
    # -----------------------------------------------------------------------

    def _create_db_schema(self, con: sqlite3.Connection) -> None:
        """Create all four tables in the SQLite database.

        Parameters
        ----------
        con:
            Open SQLite connection.
        """
        con.executescript("""
            PRAGMA foreign_keys = ON;

            DROP TABLE IF EXISTS anomalies;
            DROP TABLE IF EXISTS transactions;
            DROP TABLE IF EXISTS pharmacies;
            DROP TABLE IF EXISTS drugs;

            CREATE TABLE drugs (
                drug_id      TEXT PRIMARY KEY,
                drug_name    TEXT NOT NULL,
                ndc_code     TEXT NOT NULL UNIQUE,
                manufacturer TEXT NOT NULL,
                drug_class   TEXT NOT NULL,
                wac_low      REAL NOT NULL,
                wac_high     REAL NOT NULL
            );

            CREATE TABLE pharmacies (
                pharmacy_id    TEXT PRIMARY KEY,
                pharmacy_name  TEXT NOT NULL,
                pharmacy_chain TEXT NOT NULL,
                state          TEXT NOT NULL
            );

            CREATE TABLE transactions (
                tx_id            INTEGER PRIMARY KEY AUTOINCREMENT,
                drug_id          TEXT NOT NULL REFERENCES drugs(drug_id),
                pharmacy_id      TEXT NOT NULL REFERENCES pharmacies(pharmacy_id),
                transaction_date TEXT NOT NULL,
                wac_price        REAL NOT NULL,
                asp_price        REAL NOT NULL,
                amp_price        REAL NOT NULL,
                price_340b       REAL NOT NULL,
                gtn_price        REAL NOT NULL,
                nadac_price      REAL NOT NULL,
                actual_price     REAL NOT NULL,
                margin_percent   REAL NOT NULL,
                volume_units     INTEGER NOT NULL,
                is_anomaly       INTEGER NOT NULL DEFAULT 0,
                anomaly_type     TEXT
            );

            CREATE TABLE anomalies (
                tx_id        INTEGER PRIMARY KEY REFERENCES transactions(tx_id),
                drug_id      TEXT NOT NULL,
                pharmacy_id  TEXT NOT NULL,
                anomaly_type TEXT NOT NULL,
                actual_price REAL NOT NULL,
                wac_price    REAL NOT NULL,
                margin_percent REAL NOT NULL,
                gtn_price    REAL NOT NULL,
                volume_units INTEGER NOT NULL,
                transaction_date TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_tx_drug     ON transactions(drug_id);
            CREATE INDEX IF NOT EXISTS idx_tx_pharmacy ON transactions(pharmacy_id);
            CREATE INDEX IF NOT EXISTS idx_tx_date     ON transactions(transaction_date);
            CREATE INDEX IF NOT EXISTS idx_tx_anomaly  ON transactions(is_anomaly);
        """)
        logger.debug("SQLite schema created.")

    def _insert_drug_table(self, con: sqlite3.Connection) -> None:
        """Populate the ``drugs`` table from the cached drug catalogue.

        Parameters
        ----------
        con:
            Open SQLite connection.
        """
        assert self._drug_df is not None, "Drug catalogue not built."
        cols = ["drug_id", "drug_name", "ndc_code", "manufacturer", "drug_class", "wac_low", "wac_high"]
        self._drug_df[cols].to_sql("drugs", con, if_exists="append", index=False)
        logger.debug("drugs table: %d rows inserted.", len(self._drug_df))

    def _insert_pharmacy_table(self, con: sqlite3.Connection) -> None:
        """Populate the ``pharmacies`` table.

        Parameters
        ----------
        con:
            Open SQLite connection.
        """
        assert self._pharmacy_df is not None, "Pharmacy catalogue not built."
        self._pharmacy_df.to_sql("pharmacies", con, if_exists="append", index=False)
        logger.debug("pharmacies table: %d rows inserted.", len(self._pharmacy_df))

    def _insert_transaction_table(self, con: sqlite3.Connection) -> None:
        """Populate the ``transactions`` fact table.

        Converts ``transaction_date`` to ISO-format strings and
        ``is_anomaly`` to SQLite integers (0/1) before insertion.

        Parameters
        ----------
        con:
            Open SQLite connection.
        """
        assert self._df is not None, "Main dataset not generated."
        tx = self._df.copy()
        tx["transaction_date"] = tx["transaction_date"].astype(str)
        tx["is_anomaly"]       = tx["is_anomaly"].astype(int)

        tx_cols = [
            "drug_id", "pharmacy_id", "transaction_date",
            "wac_price", "asp_price", "amp_price", "price_340b", "gtn_price",
            "nadac_price", "actual_price", "margin_percent",
            "volume_units", "is_anomaly", "anomaly_type",
        ]
        tx[tx_cols].to_sql("transactions", con, if_exists="append", index=False)
        logger.debug("transactions table: %d rows inserted.", len(tx))

    def _insert_anomaly_table(self, con: sqlite3.Connection) -> None:
        """Populate the ``anomalies`` table from the SQLite transactions.

        Uses a SQL ``INSERT … SELECT`` so the auto-incremented ``tx_id``
        foreign key is resolved without re-loading the DataFrame.

        Parameters
        ----------
        con:
            Open SQLite connection.
        """
        con.execute("""
            INSERT INTO anomalies
                (tx_id, drug_id, pharmacy_id, anomaly_type,
                 actual_price, wac_price, margin_percent,
                 gtn_price, volume_units, transaction_date)
            SELECT
                tx_id, drug_id, pharmacy_id, anomaly_type,
                actual_price, wac_price, margin_percent,
                gtn_price, volume_units, transaction_date
            FROM transactions
            WHERE is_anomaly = 1
        """)
        count = con.execute("SELECT COUNT(*) FROM anomalies").fetchone()[0]
        logger.debug("anomalies table: %d rows inserted.", count)
