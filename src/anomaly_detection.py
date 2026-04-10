"""
anomaly_detection.py
====================
Ensemble anomaly detection engine for pharmaceutical pricing data.

Runs six complementary detection methods — three statistical, one time-series,
two machine-learning, and one domain-rule engine — then merges all signals into
a single deduplicated result table with severity scores, human-readable
descriptions, and recommended compliance actions.

Detection methods
-----------------
1. Z-Score          statistical     |z| > 3.0 on actual_price / margin_percent / gtn_price
2. IQR Fence        statistical     Q1 − 2.5·IQR / Q3 + 2.5·IQR fence
3. MoM Change       time-series     >25 % rise or >20 % drop between consecutive transactions
4. Isolation Forest ML              contamination=0.07, n_estimators=200
5. Local Outlier Factor ML          n_neighbors=20, contamination=0.07
6. Regulatory Rules domain          WAC breach / ASP-WAC / 340B ceiling / GTN floor / margin

Data flow
---------
    data/raw/drug_pricing.csv
        └─► AnomalyDetectionEngine.detect_all()
                └─► data/processed/anomaly_results.csv

Usage
-----
>>> from src.anomaly_detection import AnomalyDetectionEngine
>>> engine = AnomalyDetectionEngine()
>>> results = engine.detect_all()
>>> report  = engine.get_anomaly_report("DRG001")
>>> risks   = engine.get_pharmacy_risk_scores()
>>> summary = engine.get_drug_class_summary()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler

# ---------------------------------------------------------------------------
# Module logger
# ---------------------------------------------------------------------------

logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------

# Regulatory anomaly types — carry extra weight in severity / action logic
_REGULATORY_TYPES: frozenset[str] = frozenset({
    "wac_breach",
    "asp_wac_violation",
    "340b_ceiling_breach",
    "gtn_floor_breach",
    "margin_critical",
})

# Priority used when selecting the "primary" anomaly type after dedup
# Higher number = shown first in the final record
_TYPE_PRIORITY: dict[str, int] = {
    "wac_breach":           10,
    "asp_wac_violation":     9,
    "340b_ceiling_breach":   8,
    "gtn_floor_breach":      7,
    "margin_critical":       6,
    "isolation_forest":      5,
    "lof_outlier":           4,
    "mom_spike":             3,
    "mom_drop":              2,
    "zscore_outlier":        1,
    "iqr_outlier":           1,
}

# Human-readable method names (for the detection_method column)
_METHOD_LABELS: dict[str, str] = {
    "zscore_outlier":    "Z-Score",
    "iqr_outlier":       "IQR Fence",
    "mom_spike":         "MoM Change",
    "mom_drop":          "MoM Change",
    "isolation_forest":  "Isolation Forest",
    "lof_outlier":       "LOF",
    "wac_breach":        "Regulatory Rules",
    "asp_wac_violation": "Regulatory Rules",
    "340b_ceiling_breach": "Regulatory Rules",
    "gtn_floor_breach":  "Regulatory Rules",
    "margin_critical":   "Regulatory Rules",
}

# Columns carried forward from the source data to every detection result
_PASSTHROUGH_COLS: list[str] = [
    "drug_id", "drug_name", "pharmacy_id", "pharmacy_name",
    "pharmacy_chain", "state", "drug_class", "transaction_date",
    "wac_price", "actual_price", "margin_percent", "gtn_price",
    "asp_price", "amp_price", "price_340b", "volume_units",
]

# Final output column order
_OUTPUT_COLS: list[str] = [
    "drug_id", "drug_name", "pharmacy_id", "pharmacy_name",
    "pharmacy_chain", "state", "drug_class", "transaction_date",
    "wac_price", "actual_price", "margin_percent", "gtn_price",
    "anomaly_type", "detection_method", "severity",
    "anomaly_score", "description", "recommended_action",
]


# ---------------------------------------------------------------------------
# Helper: empty flag frame
# ---------------------------------------------------------------------------

def _empty_flags() -> pd.DataFrame:
    """Return an empty DataFrame conforming to the internal flag schema."""
    return pd.DataFrame(columns=[
        "_orig_idx", "anomaly_type", "detection_method",
        "anomaly_score", "description",
    ])


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------


class AnomalyDetectionEngine:
    """Six-method ensemble anomaly detection engine for pharmaceutical pricing.

    Parameters
    ----------
    data_dir:
        Project data root.  Expects ``{data_dir}/raw/drug_pricing.csv`` to
        exist after running :class:`~src.data_simulator.PricingDataSimulator`.
    z_threshold:
        Z-score cut-off (Method 1). Default 3.0.
    iqr_multiplier:
        IQR fence multiplier (Method 2). Default 2.5.
    mom_spike_pct:
        Month-over-month price increase threshold (Method 3). Default 0.25.
    mom_drop_pct:
        Month-over-month price decrease threshold (Method 3). Default 0.20.
    if_contamination:
        Isolation Forest contamination (Method 4). Default 0.07.
    lof_n_neighbors:
        LOF neighbours (Method 5). Default 20.
    lof_contamination:
        LOF contamination (Method 5). Default 0.07.
    random_state:
        Seed for ML reproducibility.
    """

    def __init__(
        self,
        data_dir: str | Path = "data",
        z_threshold: float = 3.0,
        iqr_multiplier: float = 2.5,
        mom_spike_pct: float = 0.25,
        mom_drop_pct: float = 0.20,
        if_contamination: float = 0.07,
        lof_n_neighbors: int = 20,
        lof_contamination: float = 0.07,
        random_state: int = 42,
    ) -> None:
        self._data_dir       = Path(data_dir)
        self._raw_dir        = self._data_dir / "raw"
        self._processed_dir  = self._data_dir / "processed"
        self._processed_dir.mkdir(parents=True, exist_ok=True)

        self.z_threshold       = z_threshold
        self.iqr_multiplier    = iqr_multiplier
        self.mom_spike_pct     = mom_spike_pct
        self.mom_drop_pct      = mom_drop_pct
        self.if_contamination  = if_contamination
        self.lof_n_neighbors   = lof_n_neighbors
        self.lof_contamination = lof_contamination
        self.random_state      = random_state

        self._data: pd.DataFrame | None    = None   # source data cache
        self._results: pd.DataFrame | None = None   # detect_all() cache

        logger.info(
            "AnomalyDetectionEngine ready "
            "(z=%.1f, iqr=%.1f×, mom=+%.0f%%/−%.0f%%, IF_cont=%.2f, LOF_k=%d)",
            z_threshold, iqr_multiplier,
            mom_spike_pct * 100, mom_drop_pct * 100,
            if_contamination, lof_n_neighbors,
        )

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def detect_all(self) -> pd.DataFrame:
        """Run all six detection methods and return a merged anomaly table.

        Each source row is deduplicated so it appears at most once in the
        output, even if multiple methods flagged it.  Severity is derived
        from the number of methods that independently flagged the same
        transaction.

        Severity rules
        --------------
        high   — ≥ 3 methods flagged it **or** any regulatory rule fired
        medium — exactly 2 methods flagged it
        low    — only 1 method flagged it

        Returns
        -------
        pandas.DataFrame
            Anomaly result table saved to
            ``data/processed/anomaly_results.csv``.
        """
        logger.info("=== detect_all() — starting 6-method ensemble ===")
        df = self._load_data()

        detectors = [
            ("Method 1 — Z-Score",          self._detect_zscore),
            ("Method 2 — IQR Fence",         self._detect_iqr),
            ("Method 3 — MoM Change",        self._detect_mom),
            ("Method 4 — Isolation Forest",  self._detect_isolation_forest),
            ("Method 5 — LOF",               self._detect_lof),
            ("Method 6 — Regulatory Rules",  self._detect_regulatory_rules),
        ]

        all_flags: list[pd.DataFrame] = []
        for label, fn in detectors:
            logger.info("Running %s…", label)
            try:
                flags = fn(df)
                n = len(flags)
                logger.info("  ✓ %s → %d flags", label, n)
                if n > 0:
                    all_flags.append(flags)
            except Exception:
                logger.exception("  ✗ %s raised an exception — skipping.", label)

        if not all_flags:
            logger.warning("No anomalies detected by any method.")
            self._results = pd.DataFrame(columns=_OUTPUT_COLS)
            return self._results

        results = self._merge_detections(df, all_flags)

        out_path = self._processed_dir / "anomaly_results.csv"
        results.to_csv(out_path, index=False)
        logger.info(
            "=== detect_all() complete — %d anomalies saved → %s ===",
            len(results), out_path,
        )

        self._results = results
        return results

    def get_anomaly_report(self, drug_id: str) -> dict[str, Any]:
        """Return a structured anomaly report for a single drug.

        Parameters
        ----------
        drug_id:
            Drug identifier (e.g. ``"DRG001"``).

        Returns
        -------
        dict with keys:
            drug_id, drug_name, anomaly_history, price_trend,
            severity_breakdown, top_violations, recommended_action.
        """
        if self._results is None:
            logger.info("Results not cached — running detect_all() first.")
            self.detect_all()

        df_src = self._data
        results = self._results

        drug_anomalies = results[results["drug_id"] == drug_id].copy()
        drug_rows      = df_src[df_src["drug_id"] == drug_id].copy()

        if drug_rows.empty:
            logger.warning("drug_id '%s' not found in source data.", drug_id)
            return {"drug_id": drug_id, "error": "drug_id not found"}

        drug_name = drug_rows["drug_name"].iloc[0]

        # ── Price trend: last 90 days ──────────────────────────────────────
        drug_rows["transaction_date"] = pd.to_datetime(drug_rows["transaction_date"])
        cutoff  = drug_rows["transaction_date"].max() - pd.Timedelta(days=90)
        recent  = drug_rows[drug_rows["transaction_date"] >= cutoff].sort_values("transaction_date")
        price_trend: dict[str, float] = {
            str(row["transaction_date"].date()): round(float(row["actual_price"]), 2)
            for _, row in recent.iterrows()
        }

        # ── Severity breakdown ─────────────────────────────────────────────
        severity_breakdown: dict[str, int] = {
            "low":    int((drug_anomalies["severity"] == "low").sum()),
            "medium": int((drug_anomalies["severity"] == "medium").sum()),
            "high":   int((drug_anomalies["severity"] == "high").sum()),
        }

        # ── Top violations ─────────────────────────────────────────────────
        top_violations: list[str] = (
            drug_anomalies["anomaly_type"].value_counts().head(5).index.tolist()
        )

        # ── Recommended action ─────────────────────────────────────────────
        has_regulatory = drug_anomalies["anomaly_type"].isin(_REGULATORY_TYPES).any()
        if severity_breakdown["high"] > 0 and has_regulatory:
            recommended_action = "regulatory_review"
        elif severity_breakdown["high"] > 0:
            recommended_action = "escalate"
        elif severity_breakdown["medium"] > 0:
            recommended_action = "investigate"
        else:
            recommended_action = "monitor"

        logger.info(
            "Anomaly report for %s (%s): %d anomalies, action=%s",
            drug_id, drug_name, len(drug_anomalies), recommended_action,
        )

        return {
            "drug_id":            drug_id,
            "drug_name":          drug_name,
            "anomaly_history":    drug_anomalies.to_dict(orient="records"),
            "price_trend":        price_trend,
            "severity_breakdown": severity_breakdown,
            "top_violations":     top_violations,
            "recommended_action": recommended_action,
        }

    def get_pharmacy_risk_scores(self) -> pd.DataFrame:
        """Aggregate anomaly signals into a per-pharmacy risk score.

        Risk score formula (0–100)
        --------------------------
        weighted_pts = high × 15  +  medium × 7  +  low × 2
                     + regulatory_breach × 25
        risk_score   = min(100,  weighted_pts / total_pharmacy_rows × 500)

        Risk tiers
        ----------
        green  — risk_score  < 25
        yellow — risk_score ∈ [25, 60)
        red    — risk_score ≥ 60

        Returns
        -------
        pandas.DataFrame
            Columns: pharmacy_id, pharmacy_name, pharmacy_chain, state,
            total_anomalies, high_severity_count, regulatory_breach_count,
            risk_score, risk_tier.
        """
        if self._results is None:
            logger.info("Results not cached — running detect_all() first.")
            self.detect_all()

        results = self._results
        df_src  = self._data

        # Total transactions per pharmacy (denominator for ratio)
        tx_counts = (
            df_src.groupby("pharmacy_id")
            .size()
            .rename("total_tx")
        )

        # Aggregate anomaly counts
        agg = (
            results.groupby("pharmacy_id")
            .agg(
                pharmacy_name    = ("pharmacy_name",   "first"),
                pharmacy_chain   = ("pharmacy_chain",  "first"),
                state            = ("state",            "first"),
                total_anomalies  = ("anomaly_type",    "count"),
                high_severity_count = (
                    "severity",
                    lambda s: int((s == "high").sum())
                ),
                medium_severity_count = (
                    "severity",
                    lambda s: int((s == "medium").sum())
                ),
                low_severity_count = (
                    "severity",
                    lambda s: int((s == "low").sum())
                ),
                regulatory_breach_count = (
                    "anomaly_type",
                    lambda s: int(s.isin(_REGULATORY_TYPES).sum())
                ),
            )
            .reset_index()
        )

        agg = agg.join(tx_counts, on="pharmacy_id")
        agg["total_tx"] = agg["total_tx"].fillna(1)

        # Risk score
        weighted = (
            agg["high_severity_count"]      * 15
            + agg["medium_severity_count"]  *  7
            + agg["low_severity_count"]     *  2
            + agg["regulatory_breach_count"] * 25
        )
        agg["risk_score"] = (weighted / agg["total_tx"] * 500).clip(upper=100).round(1)

        # Risk tier
        agg["risk_tier"] = pd.cut(
            agg["risk_score"],
            bins=[-1, 25, 60, 101],
            labels=["green", "yellow", "red"],
        ).astype(str)

        out_cols = [
            "pharmacy_id", "pharmacy_name", "pharmacy_chain", "state",
            "total_anomalies", "high_severity_count",
            "regulatory_breach_count", "risk_score", "risk_tier",
        ]
        result_df = agg[out_cols].sort_values("risk_score", ascending=False).reset_index(drop=True)

        logger.info(
            "Pharmacy risk scores: %d pharmacies — red=%d yellow=%d green=%d",
            len(result_df),
            (result_df["risk_tier"] == "red").sum(),
            (result_df["risk_tier"] == "yellow").sum(),
            (result_df["risk_tier"] == "green").sum(),
        )
        return result_df

    def get_drug_class_summary(self) -> pd.DataFrame:
        """Return anomaly summary statistics grouped by drug class.

        Returns
        -------
        pandas.DataFrame
            Columns: drug_class, anomaly_rate, avg_severity_score,
            most_common_violation, avg_gtn_discount_pct.

        Notes
        -----
        avg_severity_score maps low→1, medium→2, high→3.
        avg_gtn_discount_pct = mean((WAC − GTN) / WAC) per class.
        """
        if self._results is None:
            logger.info("Results not cached — running detect_all() first.")
            self.detect_all()

        results = self._results
        df_src  = self._data

        # Severity numeric map
        sev_map = {"low": 1.0, "medium": 2.0, "high": 3.0}

        # Rows per class in source data (denominator)
        class_counts = df_src.groupby("drug_class").size().rename("total_tx")

        # GTN discount per class from source
        df_src = df_src.copy()
        df_src["gtn_discount_pct"] = (
            (df_src["wac_price"] - df_src["gtn_price"]) / df_src["wac_price"]
        ).clip(0, 1)
        avg_gtn = df_src.groupby("drug_class")["gtn_discount_pct"].mean().rename("avg_gtn_discount_pct")

        if results.empty:
            summary = class_counts.reset_index().rename(columns={"drug_class": "drug_class"})
            summary["anomaly_rate"]          = 0.0
            summary["avg_severity_score"]    = 0.0
            summary["most_common_violation"] = None
            summary = summary.join(avg_gtn, on="drug_class")
            logger.warning("No anomalies detected — returning zero-valued summary.")
            return summary[["drug_class", "anomaly_rate", "avg_severity_score",
                             "most_common_violation", "avg_gtn_discount_pct"]]

        agg = (
            results.groupby("drug_class")
            .agg(
                anomaly_count        = ("anomaly_type", "count"),
                avg_severity_score   = ("severity", lambda s: s.map(sev_map).mean()),
                most_common_violation = ("anomaly_type", lambda s: s.value_counts().idxmax()),
            )
            .reset_index()
        )

        agg = agg.join(class_counts, on="drug_class")
        agg["anomaly_rate"] = (agg["anomaly_count"] / agg["total_tx"]).round(4)
        agg = agg.join(avg_gtn, on="drug_class")

        out_cols = [
            "drug_class", "anomaly_rate", "avg_severity_score",
            "most_common_violation", "avg_gtn_discount_pct",
        ]
        result_df = agg[out_cols].sort_values("anomaly_rate", ascending=False).reset_index(drop=True)

        logger.info(
            "Drug class summary: %d classes — highest anomaly rate: %s (%.1f%%)",
            len(result_df),
            result_df.iloc[0]["drug_class"],
            result_df.iloc[0]["anomaly_rate"] * 100,
        )
        return result_df

    # -----------------------------------------------------------------------
    # Private — data loading
    # -----------------------------------------------------------------------

    def _load_data(self) -> pd.DataFrame:
        """Load and validate the source pricing CSV.

        Returns
        -------
        pandas.DataFrame
            Source data with ``transaction_date`` parsed as datetime.

        Raises
        ------
        FileNotFoundError
            If ``data/raw/drug_pricing.csv`` does not exist.
        """
        if self._data is not None:
            logger.debug("Using cached source data (%d rows).", len(self._data))
            return self._data

        path = self._raw_dir / "drug_pricing.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Source data not found: {path}\n"
                "Run PricingDataSimulator().generate() first."
            )

        df = pd.read_csv(path, parse_dates=["transaction_date"])
        df = df.reset_index(drop=True)   # ensures clean 0-based RangeIndex

        required = {
            "drug_id", "drug_name", "pharmacy_id", "pharmacy_name",
            "wac_price", "asp_price", "amp_price", "price_340b",
            "gtn_price", "actual_price", "margin_percent", "volume_units",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Source CSV missing columns: {missing}")

        logger.info("Loaded source data: %d rows, %d columns from %s", len(df), len(df.columns), path)
        self._data = df
        return df

    # -----------------------------------------------------------------------
    # Private — detection methods
    # -----------------------------------------------------------------------

    def _detect_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        """Method 1: Z-Score — flag rows where |z| > threshold on any feature.

        Features evaluated: actual_price, margin_percent, gtn_price.
        The maximum absolute z-score across features determines the flag
        and the anomaly_score.

        Parameters
        ----------
        df:
            Full source DataFrame.

        Returns
        -------
        pandas.DataFrame
            Internal flag frame (one row per flagged source index).
        """
        features = ["actual_price", "margin_percent", "gtn_price"]
        z_matrix = pd.DataFrame(index=df.index, dtype=float)

        for feat in features:
            col    = df[feat].fillna(df[feat].median())
            mu     = col.mean()
            sigma  = col.std(ddof=1)
            if sigma < 1e-9:
                z_matrix[feat] = 0.0
            else:
                z_matrix[feat] = (col - mu) / sigma

        max_z      = z_matrix.abs().max(axis=1)
        worst_feat = z_matrix.abs().idxmax(axis=1)
        mask       = max_z > self.z_threshold

        if not mask.any():
            return _empty_flags()

        flagged = df[mask].copy()
        records = []
        for idx in flagged.index:
            z     = float(max_z[idx])
            feat  = worst_feat[idx]
            val   = float(df.at[idx, feat])
            records.append({
                "_orig_idx":        idx,
                "anomaly_type":     "zscore_outlier",
                "detection_method": _METHOD_LABELS["zscore_outlier"],
                "anomaly_score":    float(min(abs(z) / (self.z_threshold * 3), 1.0)),
                "description": (
                    f"Z-score outlier: '{feat}' has z={z:.2f} "
                    f"(value={val:,.2f}, threshold ±{self.z_threshold})"
                ),
            })
        return pd.DataFrame(records)

    def _detect_iqr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Method 2: IQR Fence — flag rows outside Q1−k·IQR / Q3+k·IQR.

        Features evaluated: actual_price, wac_price, margin_percent.

        Parameters
        ----------
        df:
            Full source DataFrame.

        Returns
        -------
        pandas.DataFrame
            Internal flag frame.
        """
        features = ["actual_price", "wac_price", "margin_percent"]
        flagged_idx: dict[int, dict[str, Any]] = {}

        for feat in features:
            col = df[feat].fillna(df[feat].median())
            q1, q3 = col.quantile(0.25), col.quantile(0.75)
            iqr    = q3 - q1
            lower  = q1 - self.iqr_multiplier * iqr
            upper  = q3 + self.iqr_multiplier * iqr
            mask   = (col < lower) | (col > upper)

            for idx in df.index[mask]:
                val = float(df.at[idx, feat])
                dist = max(abs(val - lower), abs(val - upper)) / (iqr + 1e-9)
                if idx not in flagged_idx or dist > flagged_idx[idx]["_dist"]:
                    direction = "above upper fence" if val > upper else "below lower fence"
                    flagged_idx[idx] = {
                        "_orig_idx":        idx,
                        "_dist":            dist,
                        "anomaly_type":     "iqr_outlier",
                        "detection_method": _METHOD_LABELS["iqr_outlier"],
                        "anomaly_score":    float(min(dist / 10.0, 1.0)),
                        "description": (
                            f"IQR fence breach on '{feat}': value={val:,.2f} "
                            f"{direction} (fence=[{lower:,.2f}, {upper:,.2f}])"
                        ),
                    }

        if not flagged_idx:
            return _empty_flags()

        records = [{k: v for k, v in d.items() if k != "_dist"} for d in flagged_idx.values()]
        return pd.DataFrame(records)

    def _detect_mom(self, df: pd.DataFrame) -> pd.DataFrame:
        """Method 3: Month-over-Month Change — flag large consecutive price moves.

        Groups by (drug_id, pharmacy_id), sorts by transaction_date, and
        computes pct_change on actual_price.  Flags if change > mom_spike_pct
        (rise) or < −mom_drop_pct (drop).

        Parameters
        ----------
        df:
            Full source DataFrame.

        Returns
        -------
        pandas.DataFrame
            Internal flag frame with anomaly_type "mom_spike" or "mom_drop".
        """
        records = []
        grouped = df.groupby(["drug_id", "pharmacy_id"], sort=False)

        for (drug_id, pharm_id), grp in grouped:
            if len(grp) < 2:
                continue
            grp_sorted = grp.sort_values("transaction_date")
            pct        = grp_sorted["actual_price"].pct_change()

            for idx, chg in pct.items():
                if pd.isna(chg):
                    continue
                if chg > self.mom_spike_pct:
                    atype = "mom_spike"
                    desc  = (
                        f"Price spike of {chg*100:.1f}% detected "
                        f"(drug={drug_id}, pharmacy={pharm_id}); "
                        f"threshold +{self.mom_spike_pct*100:.0f}%"
                    )
                    score = float(min(chg / (self.mom_spike_pct * 4), 1.0))
                elif chg < -self.mom_drop_pct:
                    atype = "mom_drop"
                    desc  = (
                        f"Price drop of {abs(chg)*100:.1f}% detected "
                        f"(drug={drug_id}, pharmacy={pharm_id}); "
                        f"threshold −{self.mom_drop_pct*100:.0f}%"
                    )
                    score = float(min(abs(chg) / (self.mom_drop_pct * 4), 1.0))
                else:
                    continue

                records.append({
                    "_orig_idx":        idx,
                    "anomaly_type":     atype,
                    "detection_method": _METHOD_LABELS[atype],
                    "anomaly_score":    score,
                    "description":      desc,
                })

        if not records:
            return _empty_flags()
        return pd.DataFrame(records).drop_duplicates(subset="_orig_idx", keep="last")

    def _detect_isolation_forest(self, df: pd.DataFrame) -> pd.DataFrame:
        """Method 4: Isolation Forest — ML-based outlier detection.

        Trains on seven price/margin features then predicts on the same data
        (transductive mode).  Rows predicted as −1 (outlier) are flagged.
        Anomaly score is derived from ``decision_function`` (lower = more
        anomalous), normalised to [0, 1].

        Features
        --------
        wac_price, asp_price, amp_price, price_340b,
        gtn_price, actual_price, margin_percent

        Parameters
        ----------
        df:
            Full source DataFrame.

        Returns
        -------
        pandas.DataFrame
            Internal flag frame.
        """
        feature_cols = [
            "wac_price", "asp_price", "amp_price", "price_340b",
            "gtn_price", "actual_price", "margin_percent",
        ]
        X = df[feature_cols].fillna(df[feature_cols].median())

        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)

        clf = IsolationForest(
            n_estimators=200,
            contamination=self.if_contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        preds  = clf.fit_predict(X_scaled)          # −1 = outlier
        scores = clf.decision_function(X_scaled)    # lower = more anomalous

        # Normalise scores to [0, 1] where 1 = most anomalous
        s_min, s_max = scores.min(), scores.max()
        norm_scores  = 1.0 - (scores - s_min) / (s_max - s_min + 1e-9)

        mask    = preds == -1
        records = []
        for i, idx in enumerate(df.index):
            if not mask[i]:
                continue
            records.append({
                "_orig_idx":        idx,
                "anomaly_type":     "isolation_forest",
                "detection_method": _METHOD_LABELS["isolation_forest"],
                "anomaly_score":    float(norm_scores[i]),
                "description": (
                    f"Isolation Forest outlier (normalised score={norm_scores[i]:.3f}); "
                    f"actual_price={df.at[idx, 'actual_price']:,.2f}, "
                    f"margin={df.at[idx, 'margin_percent']:.1f}%"
                ),
            })

        if not records:
            return _empty_flags()
        return pd.DataFrame(records)

    def _detect_lof(self, df: pd.DataFrame) -> pd.DataFrame:
        """Method 5: Local Outlier Factor — density-based outlier detection.

        Uses three features that capture both price and volume behaviour.
        Anomaly score is derived from ``negative_outlier_factor_`` and
        normalised to [0, 1].

        Features
        --------
        actual_price, margin_percent, volume_units

        Parameters
        ----------
        df:
            Full source DataFrame.

        Returns
        -------
        pandas.DataFrame
            Internal flag frame.
        """
        feature_cols = ["actual_price", "margin_percent", "volume_units"]
        X = df[feature_cols].fillna(df[feature_cols].median())

        scaler   = RobustScaler()
        X_scaled = scaler.fit_transform(X)

        n_neighbors = min(self.lof_n_neighbors, len(df) - 1)
        lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=self.lof_contamination,
            novelty=False,
        )
        preds  = lof.fit_predict(X_scaled)             # −1 = outlier
        # negative_outlier_factor_ is negative LOF score (larger abs = more anomalous)
        factors = -lof.negative_outlier_factor_         # flip sign; higher = more anomalous

        f_min, f_max = factors.min(), factors.max()
        norm_scores  = (factors - f_min) / (f_max - f_min + 1e-9)

        mask    = preds == -1
        records = []
        for i, idx in enumerate(df.index):
            if not mask[i]:
                continue
            records.append({
                "_orig_idx":        idx,
                "anomaly_type":     "lof_outlier",
                "detection_method": _METHOD_LABELS["lof_outlier"],
                "anomaly_score":    float(norm_scores[i]),
                "description": (
                    f"LOF density outlier (LOF factor={factors[i]:.3f}); "
                    f"actual_price={df.at[idx, 'actual_price']:,.2f}, "
                    f"volume_units={int(df.at[idx, 'volume_units'])}"
                ),
            })

        if not records:
            return _empty_flags()
        return pd.DataFrame(records)

    def _detect_regulatory_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """Method 6: Regulatory Rules — domain-specific hard-rule checks.

        Rules
        -----
        wac_breach          actual_price > wac_price × 2.5
        asp_wac_violation   asp_price    > wac_price × 1.05
        340b_ceiling_breach price_340b   > amp_price  × 0.855
        gtn_floor_breach    gtn_price    < wac_price  × 0.45
        margin_critical     margin_percent < −20 %

        Each row can trigger multiple rules; all are recorded separately,
        and the highest-priority rule is selected during deduplication.

        Parameters
        ----------
        df:
            Full source DataFrame.

        Returns
        -------
        pandas.DataFrame
            Internal flag frame (may contain multiple rows per source index
            if multiple rules fire on the same transaction).
        """
        rules: list[tuple[str, pd.Series, str]] = [
            (
                "wac_breach",
                df["actual_price"] > df["wac_price"] * 2.5,
                lambda r: (
                    f"WAC breach: actual_price=${r['actual_price']:,.2f} exceeds "
                    f"WAC × 2.5 threshold (${r['wac_price']*2.5:,.2f})"
                ),
            ),
            (
                "asp_wac_violation",
                df["asp_price"] > df["wac_price"] * 1.05,
                lambda r: (
                    f"ASP-WAC violation: asp_price=${r['asp_price']:,.2f} exceeds "
                    f"WAC × 1.05 (${r['wac_price']*1.05:,.2f})"
                ),
            ),
            (
                "340b_ceiling_breach",
                df["price_340b"] > df["amp_price"] * 0.855,
                lambda r: (
                    f"340B ceiling breach: price_340b=${r['price_340b']:,.2f} exceeds "
                    f"AMP × 0.855 ceiling (${r['amp_price']*0.855:,.2f})"
                ),
            ),
            (
                "gtn_floor_breach",
                df["gtn_price"] < df["wac_price"] * 0.45,
                lambda r: (
                    f"GTN floor breach: gtn_price=${r['gtn_price']:,.2f} is below "
                    f"WAC × 0.45 floor (${r['wac_price']*0.45:,.2f})"
                ),
            ),
            (
                "margin_critical",
                df["margin_percent"] < -20.0,
                lambda r: (
                    f"Critical margin: margin_percent={r['margin_percent']:.1f}% "
                    "is below the −20% threshold"
                ),
            ),
        ]

        records: list[dict[str, Any]] = []
        for atype, mask, desc_fn in rules:
            count = int(mask.sum())
            if count == 0:
                continue
            logger.debug("  Regulatory rule '%s': %d violations", atype, count)
            for idx in df.index[mask]:
                row = df.loc[idx]
                records.append({
                    "_orig_idx":        idx,
                    "anomaly_type":     atype,
                    "detection_method": _METHOD_LABELS[atype],
                    "anomaly_score":    1.0,   # regulatory violations always score maximum
                    "description":      desc_fn(row),
                })

        if not records:
            return _empty_flags()
        return pd.DataFrame(records)

    # -----------------------------------------------------------------------
    # Private — merge & enrich
    # -----------------------------------------------------------------------

    def _merge_detections(
        self,
        df: pd.DataFrame,
        all_flags: list[pd.DataFrame],
    ) -> pd.DataFrame:
        """Merge flags from all methods; deduplicate by source row.

        Deduplication strategy
        ----------------------
        For each unique ``_orig_idx``:
          - Count unique detection methods that fired.
          - Select the highest-priority ``anomaly_type``.
          - Join detection method names with " | ".
          - Compute ``anomaly_score`` as the maximum across methods.
          - Assign severity and recommended_action.
          - Join passthrough columns from the source DataFrame.

        Parameters
        ----------
        df:
            Full source DataFrame (used for passthrough columns).
        all_flags:
            List of internal flag DataFrames from each detector.

        Returns
        -------
        pandas.DataFrame
            Final anomaly result table conforming to ``_OUTPUT_COLS``.
        """
        combined = pd.concat(all_flags, ignore_index=True)

        if combined.empty:
            return pd.DataFrame(columns=_OUTPUT_COLS)

        # Group by source row index
        def _agg_group(grp: pd.DataFrame) -> pd.Series:
            atypes = grp["anomaly_type"].tolist()
            # Primary anomaly = highest priority
            primary = max(atypes, key=lambda t: _TYPE_PRIORITY.get(t, 0))
            # Unique method labels, preserving encounter order
            seen: dict[str, None] = {}
            for m in grp["detection_method"].tolist():
                seen[m] = None
            methods_str = " | ".join(seen.keys())

            n_methods     = grp["detection_method"].nunique()
            has_reg       = any(t in _REGULATORY_TYPES for t in atypes)
            max_score     = float(grp["anomaly_score"].max())
            primary_desc  = grp.loc[
                grp["anomaly_type"] == primary, "description"
            ].iloc[0]

            return pd.Series({
                "anomaly_type":     primary,
                "detection_method": methods_str,
                "anomaly_score":    max_score,
                "description":      primary_desc,
                "_n_methods":       n_methods,
                "_has_reg":         has_reg,
            })

        agg = (
            combined.groupby("_orig_idx", sort=True)
            .apply(_agg_group, include_groups=False)
            .reset_index()
        )

        # Severity
        def _severity(row: pd.Series) -> str:
            if row["_n_methods"] >= 3 or row["_has_reg"]:
                return "high"
            if row["_n_methods"] == 2:
                return "medium"
            return "low"

        agg["severity"] = agg.apply(_severity, axis=1)

        # Recommended action
        def _action(row: pd.Series) -> str:
            if row["anomaly_type"] in _REGULATORY_TYPES:
                return "regulatory_review"
            if row["severity"] == "high":
                return "escalate"
            if row["severity"] == "medium":
                return "investigate"
            return "monitor"

        agg["recommended_action"] = agg.apply(_action, axis=1)

        # Join passthrough columns from source
        passthrough = df[_PASSTHROUGH_COLS].copy()
        passthrough.index.name = "_orig_idx"
        passthrough = passthrough.reset_index()

        result = agg.merge(passthrough, on="_orig_idx", how="left")
        result = result.drop(columns=["_orig_idx", "_n_methods", "_has_reg"])

        # Enforce output column order (keep only those present)
        present = [c for c in _OUTPUT_COLS if c in result.columns]
        result  = result[present].sort_values(
            ["severity", "anomaly_score"],
            ascending=[True, False],      # high first (alphabetically last), score desc
            key=lambda s: s if s.name == "anomaly_score" else s.map({"high": 0, "medium": 1, "low": 2}),
        ).reset_index(drop=True)

        logger.info(
            "Merge complete — %d unique anomalous transactions "
            "(high=%d, medium=%d, low=%d)",
            len(result),
            (result["severity"] == "high").sum(),
            (result["severity"] == "medium").sum(),
            (result["severity"] == "low").sum(),
        )
        return result
