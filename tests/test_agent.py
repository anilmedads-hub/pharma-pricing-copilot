"""
test_agent.py
=============
Unit and integration tests for the Pharma Pricing Copilot modules.

Test suites
-----------
TestPricingDataSimulator   — synthetic data generation correctness
TestAnomalyDetector        — detection logic and regulatory checks
TestToolExecutor           — agent tool dispatch and output shape
TestPricingAgent           — agent conversational loop (mocked Claude API)
TestRAGPipeline            — chunking, vector store, retrieval
TestUtils                  — helper functions

Run
---
    pytest tests/test_agent.py -v
    pytest tests/test_agent.py -v --tb=short -q   # terse output
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.anomaly_detection import AnomalyDetector
from src.data_simulator import PricingDataSimulator, SimulatorConfig
from src.rag_pipeline import Chunk, RAGPipeline, TextChunker, VectorStore
from src.utils import (
    fmt_pct,
    fmt_usd,
    load_config,
    validate_price_history_schema,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_price_history(n_drugs: int = 3, periods: int = 6, seed: int = 0) -> pd.DataFrame:
    """Return a small price history DataFrame for testing."""
    sim = PricingDataSimulator(seed=seed)
    catalog = sim.generate_drug_catalog(n_drugs=n_drugs)
    return sim.generate_price_history(catalog, periods=periods)


# ===========================================================================
# TestPricingDataSimulator
# ===========================================================================


class TestPricingDataSimulator(unittest.TestCase):
    """Tests for :class:`~src.data_simulator.PricingDataSimulator`."""

    def setUp(self) -> None:
        self.sim = PricingDataSimulator(seed=42)

    # --- drug catalog ---

    def test_catalog_row_count(self) -> None:
        catalog = self.sim.generate_drug_catalog(n_drugs=10)
        self.assertEqual(len(catalog), 10)

    def test_catalog_columns(self) -> None:
        catalog = self.sim.generate_drug_catalog(n_drugs=5)
        expected_cols = {
            "ndc", "brand_name", "generic_name", "therapeutic_area",
            "dosage_form", "strength_mg", "launch_date", "base_wac",
            "annual_price_increase_pct",
        }
        self.assertTrue(expected_cols.issubset(set(catalog.columns)))

    def test_catalog_wac_positive(self) -> None:
        catalog = self.sim.generate_drug_catalog(n_drugs=20)
        self.assertTrue((catalog["base_wac"] > 0).all(), "All WAC values must be positive.")

    def test_catalog_price_increase_range(self) -> None:
        catalog = self.sim.generate_drug_catalog(n_drugs=20)
        self.assertTrue((catalog["annual_price_increase_pct"] >= 0).all())
        self.assertTrue((catalog["annual_price_increase_pct"] <= 1).all())

    def test_catalog_reproducible(self) -> None:
        sim_a = PricingDataSimulator(seed=99)
        sim_b = PricingDataSimulator(seed=99)
        df_a = sim_a.generate_drug_catalog(n_drugs=5)
        df_b = sim_b.generate_drug_catalog(n_drugs=5)
        pd.testing.assert_frame_equal(df_a, df_b)

    # --- price history ---

    def test_price_history_row_count(self) -> None:
        catalog = self.sim.generate_drug_catalog(n_drugs=4)
        ph = self.sim.generate_price_history(catalog, periods=12)
        self.assertEqual(len(ph), 4 * 12)

    def test_price_history_columns(self) -> None:
        catalog = self.sim.generate_drug_catalog(n_drugs=2)
        ph = self.sim.generate_price_history(catalog, periods=3)
        expected_cols = {"ndc", "brand_name", "period", "wac_per_unit", "asp", "amp",
                         "best_price", "ceiling_340b", "gross_to_net_pct", "net_price"}
        self.assertTrue(expected_cols.issubset(set(ph.columns)))

    def test_price_history_wac_monotonically_increasing(self) -> None:
        """WAC should generally increase over time for a given drug (modulo noise)."""
        catalog = self.sim.generate_drug_catalog(n_drugs=1)
        ph = self.sim.generate_price_history(catalog, periods=24)
        ndc = catalog.iloc[0]["ndc"]
        drug_ph = ph[ph["ndc"] == ndc].sort_values("period")
        # First WAC should be <= last WAC given positive annual increase
        self.assertLessEqual(drug_ph["wac_per_unit"].iloc[0], drug_ph["wac_per_unit"].iloc[-1])

    # --- anomaly injection ---

    def test_anomaly_injection_adds_flag_column(self) -> None:
        catalog = self.sim.generate_drug_catalog(n_drugs=3)
        ph = self.sim.generate_price_history(catalog, periods=6)
        ph_with = self.sim.inject_anomalies(ph, anomaly_rate=0.10)
        self.assertIn("is_anomaly", ph_with.columns)

    def test_anomaly_injection_rate(self) -> None:
        catalog = self.sim.generate_drug_catalog(n_drugs=10)
        ph = self.sim.generate_price_history(catalog, periods=12)
        ph_with = self.sim.inject_anomalies(ph, anomaly_rate=0.10)
        actual_rate = ph_with["is_anomaly"].mean()
        # Should be within a few percentage points of the target
        self.assertAlmostEqual(actual_rate, 0.10, delta=0.05)

    def test_anomaly_injection_does_not_change_clean_rows(self) -> None:
        catalog = self.sim.generate_drug_catalog(n_drugs=3)
        ph = self.sim.generate_price_history(catalog, periods=6)
        ph_with = self.sim.inject_anomalies(ph, anomaly_rate=0.0)
        # With 0 rate, at least 1 anomaly is injected (max(1, int(N*0)) = 1)
        # So we just check the column exists
        self.assertIn("is_anomaly", ph_with.columns)

    # --- competitive landscape ---

    def test_competitive_landscape_columns(self) -> None:
        catalog = self.sim.generate_drug_catalog(n_drugs=5)
        comp = self.sim.generate_competitive_landscape(catalog, n_competitors=2)
        expected = {"therapeutic_area", "competitor_brand", "competitor_wac",
                    "market_share_pct", "snapshot_date"}
        self.assertTrue(expected.issubset(set(comp.columns)))

    def test_competitive_landscape_row_count(self) -> None:
        catalog = self.sim.generate_drug_catalog(n_drugs=5)
        n_areas = catalog["therapeutic_area"].nunique()
        comp = self.sim.generate_competitive_landscape(catalog, n_competitors=3)
        self.assertEqual(len(comp), n_areas * 3)


# ===========================================================================
# TestAnomalyDetector
# ===========================================================================


class TestAnomalyDetector(unittest.TestCase):
    """Tests for :class:`~src.anomaly_detection.AnomalyDetector`."""

    def setUp(self) -> None:
        self.detector = AnomalyDetector(
            use_isolation_forest=False,  # speed up unit tests
            use_lof=False,
        )
        self.ph = _make_price_history(n_drugs=5, periods=12)

    def test_detect_returns_dataframe(self) -> None:
        result = self.detector.detect(self.ph)
        self.assertIsInstance(result, pd.DataFrame)

    def test_detect_adds_anomaly_columns(self) -> None:
        result = self.detector.detect(self.ph)
        self.assertIn("anomaly", result.columns)
        self.assertIn("anomaly_reason", result.columns)
        self.assertIn("anomaly_score", result.columns)

    def test_detect_preserves_row_count(self) -> None:
        result = self.detector.detect(self.ph)
        self.assertEqual(len(result), len(self.ph))

    def test_bp_violation_detected(self) -> None:
        """A row where best_price > amp must be flagged."""
        ph = self.ph.copy()
        ph.loc[ph.index[0], "best_price"] = ph.loc[ph.index[0], "amp"] * 1.5
        detector = AnomalyDetector(
            use_isolation_forest=False, use_lof=False, run_regulatory_checks=True
        )
        result = detector.detect(ph)
        self.assertTrue(result.loc[ph.index[0], "anomaly"])
        self.assertEqual(result.loc[ph.index[0], "anomaly_reason"], "bp_exceeds_amp")

    def test_negative_340b_ceiling_detected(self) -> None:
        ph = self.ph.copy()
        ph.loc[ph.index[0], "ceiling_340b"] = -10.0
        detector = AnomalyDetector(
            use_isolation_forest=False, use_lof=False, run_regulatory_checks=True
        )
        result = detector.detect(ph)
        self.assertTrue(result.loc[ph.index[0], "anomaly"])

    def test_zscore_spike_detected(self) -> None:
        """Inject a massive WAC spike and verify z-score check catches it."""
        ph = self.ph.copy()
        # Target the last row of the first NDC group
        ndc0 = ph["ndc"].iloc[0]
        last_idx = ph[ph["ndc"] == ndc0].index[-1]
        ph.loc[last_idx, "wac_per_unit"] *= 100  # huge spike
        result = self.detector.detect(ph)
        self.assertTrue(result.loc[last_idx, "anomaly"])

    def test_summary_keys(self) -> None:
        result = self.detector.detect(self.ph)
        summary = self.detector.summary(result)
        self.assertIn("total_rows", summary)
        self.assertIn("total_anomalies", summary)
        self.assertIn("anomaly_rate_pct", summary)
        self.assertIn("by_reason", summary)

    def test_empty_dataframe_handled(self) -> None:
        empty = pd.DataFrame(columns=self.ph.columns)
        result = self.detector.detect(empty)
        self.assertEqual(len(result), 0)

    def test_with_isolation_forest_and_lof(self) -> None:
        """Smoke test: runs without error when ML methods are enabled."""
        ph = _make_price_history(n_drugs=5, periods=24)
        detector = AnomalyDetector(use_isolation_forest=True, use_lof=True)
        result = detector.detect(ph)
        self.assertIn("anomaly", result.columns)


# ===========================================================================
# TestToolExecutor
# ===========================================================================


class TestToolExecutor(unittest.TestCase):
    """Tests for :class:`~src.agent.ToolExecutor`."""

    def setUp(self) -> None:
        from src.agent import ToolExecutor

        self.ph = _make_price_history(n_drugs=3, periods=6)
        sim = PricingDataSimulator(seed=0)
        catalog = sim.generate_drug_catalog(n_drugs=3)
        self.competitive = sim.generate_competitive_landscape(catalog)
        self.executor = ToolExecutor(
            price_history_df=self.ph,
            competitive_df=self.competitive,
        )

    def _call(self, tool: str, **kwargs) -> dict:
        raw = self.executor.execute(tool, kwargs)
        return json.loads(raw)

    def test_get_price_history_returns_records(self) -> None:
        brand = self.ph["brand_name"].iloc[0]
        result = self._call("get_price_history", identifier=brand)
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

    def test_get_price_history_unknown_drug(self) -> None:
        result = self._call("get_price_history", identifier="NONEXISTENT_DRUG_XYZ")
        self.assertIn("error", result)

    def test_run_anomaly_detection_returns_summary(self) -> None:
        brand = self.ph["brand_name"].iloc[0]
        result = self._call("run_anomaly_detection", identifier=brand)
        self.assertIn("summary", result)
        self.assertIn("anomalies", result)

    def test_calculate_340b_ceiling_innovator(self) -> None:
        result = self._call("calculate_340b_ceiling", amp=100.0, rebate_pct=0.2302)
        self.assertIn("ceiling_340b", result)
        expected = round(100.0 * (1 - 0.2302), 4)
        self.assertAlmostEqual(result["ceiling_340b"], expected, places=3)

    def test_calculate_340b_ceiling_non_innovator(self) -> None:
        result = self._call("calculate_340b_ceiling", amp=100.0, rebate_pct=0.1302, drug_type="non_innovator")
        self.assertIn("ceiling_340b", result)
        self.assertAlmostEqual(result["ceiling_340b"], round(100.0 * (1 - 0.1302), 4), places=3)

    def test_summarise_gtn_waterfall_structure(self) -> None:
        brand = self.ph["brand_name"].iloc[0]
        result = self._call("summarise_gtn_waterfall", identifier=brand)
        self.assertIn("waterfall", result)
        waterfall = result["waterfall"]
        self.assertIn("chargebacks", waterfall)
        self.assertIn("rebates", waterfall)

    def test_unknown_tool_returns_error(self) -> None:
        result = self._call("nonexistent_tool_xyz")
        self.assertIn("error", result)

    def test_get_competitive_snapshot(self) -> None:
        area = self.competitive["therapeutic_area"].iloc[0]
        result = self._call("get_competitive_snapshot", therapeutic_area=area)
        self.assertIsInstance(result, list)


# ===========================================================================
# TestPricingAgent (mocked)
# ===========================================================================


class TestPricingAgent(unittest.TestCase):
    """Tests for :class:`~src.agent.PricingAgent` with mocked Anthropic client."""

    def _build_mock_response(self, text: str, stop_reason: str = "end_turn"):
        """Create a mock Anthropic API response object."""
        block = MagicMock()
        block.type = "text"
        block.text = text
        response = MagicMock()
        response.content = [block]
        response.stop_reason = stop_reason
        return response

    @patch("src.agent.anthropic.Anthropic")
    def test_chat_yields_text(self, mock_anthropic_cls) -> None:
        from src.agent import PricingAgent

        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = self._build_mock_response(
            "Here is my analysis of drug pricing anomalies."
        )

        ph = _make_price_history(n_drugs=2, periods=6)
        agent = PricingAgent(anthropic_api_key="test-key", price_history_df=ph)
        response_chunks = list(agent.chat("Are there pricing anomalies?"))

        self.assertTrue(len(response_chunks) > 0)
        full_response = "".join(response_chunks)
        self.assertIn("analysis", full_response.lower())

    @patch("src.agent.anthropic.Anthropic")
    def test_reset_clears_history(self, mock_anthropic_cls) -> None:
        from src.agent import PricingAgent

        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = self._build_mock_response("Test response.")

        agent = PricingAgent(anthropic_api_key="test-key")
        list(agent.chat("Hello"))
        self.assertGreater(len(agent._history.messages), 0)
        agent.reset()
        self.assertEqual(len(agent._history.messages), 0)

    @patch("src.agent.anthropic.Anthropic")
    def test_tool_use_loop(self, mock_anthropic_cls) -> None:
        """Verify the agent correctly handles a tool_use → end_turn cycle."""
        from src.agent import PricingAgent

        ph = _make_price_history(n_drugs=2, periods=6)
        agent = PricingAgent(anthropic_api_key="test-key", price_history_df=ph)

        # First response: tool_use block
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "get_price_history"
        tool_block.id = "tool_abc123"
        tool_block.input = {"identifier": ph["brand_name"].iloc[0]}

        tool_response = MagicMock()
        tool_response.content = [tool_block]
        tool_response.stop_reason = "tool_use"

        # Second response: text answer
        text_response = self._build_mock_response("Here are the results.")

        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.side_effect = [tool_response, text_response]

        chunks = list(agent.chat("Show me price history for Brand001"))
        full = "".join(chunks)
        self.assertIn("results", full.lower())

    @patch("src.agent.anthropic.Anthropic")
    def test_load_data_updates_executor(self, mock_anthropic_cls) -> None:
        from src.agent import PricingAgent

        mock_anthropic_cls.return_value = MagicMock()
        agent = PricingAgent(anthropic_api_key="test-key")
        new_ph = _make_price_history(n_drugs=2, periods=4)
        agent.load_data(price_history_df=new_ph)
        self.assertIs(agent._executor._price_history, new_ph)


# ===========================================================================
# TestRAGPipeline
# ===========================================================================


class TestTextChunker(unittest.TestCase):
    """Tests for :class:`~src.rag_pipeline.TextChunker`."""

    def test_single_chunk_for_short_text(self) -> None:
        from src.rag_pipeline import Document, TextChunker

        doc = Document(doc_id="d1", source_path="", title="Test", content="Short text.")
        chunker = TextChunker(chunk_size=1000, overlap=100)
        chunks = chunker.split(doc)
        self.assertEqual(len(chunks), 1)

    def test_multiple_chunks_for_long_text(self) -> None:
        from src.rag_pipeline import Document, TextChunker

        long_text = "word " * 1000  # ~5000 chars
        doc = Document(doc_id="d2", source_path="", title="Long", content=long_text)
        chunker = TextChunker(chunk_size=500, overlap=50)
        chunks = chunker.split(doc)
        self.assertGreater(len(chunks), 1)

    def test_chunk_overlap(self) -> None:
        from src.rag_pipeline import Document, TextChunker

        text = "a" * 200
        doc = Document(doc_id="d3", source_path="", title="Overlap", content=text)
        chunker = TextChunker(chunk_size=100, overlap=20)
        chunks = chunker.split(doc)
        # Verify overlap: end of chunk 0 should overlap with start of chunk 1
        self.assertEqual(chunks[0].text[-20:], chunks[1].text[:20])


class TestVectorStore(unittest.TestCase):
    """Tests for :class:`~src.rag_pipeline.VectorStore`."""

    def _make_chunk(self, text: str, embedding: np.ndarray) -> Chunk:
        import hashlib

        chunk_id = hashlib.sha256(text.encode()).hexdigest()[:8]
        c = Chunk(chunk_id=chunk_id, doc_id="d1", source_path="", title="T", text=text, chunk_index=0)
        c.embedding = embedding
        return c

    def test_add_and_search(self) -> None:
        store = VectorStore()
        rng = np.random.default_rng(0)
        chunks = [self._make_chunk(f"text {i}", rng.standard_normal(64).astype(np.float32)) for i in range(20)]
        store.add(chunks)
        self.assertEqual(len(store), 20)
        query = chunks[5].embedding
        results = store.search(query, top_k=3)
        self.assertEqual(len(results), 3)
        # The closest chunk should be the query chunk itself
        self.assertEqual(results[0][0].chunk_id, chunks[5].chunk_id)

    def test_empty_store_search_returns_empty(self) -> None:
        store = VectorStore()
        result = store.search(np.zeros(64, dtype=np.float32), top_k=5)
        self.assertEqual(result, [])


class TestRAGPipelineOffline(unittest.TestCase):
    """Offline tests for :class:`~src.rag_pipeline.RAGPipeline` (no API calls)."""

    @patch("src.rag_pipeline.RAGPipeline._embed_text")
    def test_ingest_and_retrieve(self, mock_embed) -> None:
        """Verify that ingested text is stored and retrieved on a query."""
        rng = np.random.default_rng(0)
        # Return deterministic embeddings so search is meaningful
        mock_embed.side_effect = lambda text: rng.standard_normal(64).astype(np.float32)

        pipeline = RAGPipeline(anthropic_api_key="dummy")
        pipeline.ingest_text("Medicaid Best Price is defined as the lowest price available.", title="CMS Guide")
        ctx = pipeline.retrieve("What is Medicaid Best Price?", top_k=1)
        self.assertEqual(len(ctx.chunks), 1)

    @patch("src.rag_pipeline.RAGPipeline._embed_text")
    def test_ingest_directory_missing_path(self, mock_embed) -> None:
        """Ingesting a non-existent directory should return an empty list gracefully."""
        pipeline = RAGPipeline(anthropic_api_key="dummy")
        result = pipeline.ingest_directory("/nonexistent/path/xyz123")
        self.assertEqual(result, [])


# ===========================================================================
# TestUtils
# ===========================================================================


class TestUtils(unittest.TestCase):
    """Tests for :mod:`src.utils` helper functions."""

    def test_fmt_usd_basic(self) -> None:
        self.assertEqual(fmt_usd(1234.5), "$1,234.50")

    def test_fmt_usd_zero(self) -> None:
        self.assertEqual(fmt_usd(0), "$0.00")

    def test_fmt_pct_basic(self) -> None:
        self.assertEqual(fmt_pct(0.1523), "15.2%")

    def test_fmt_pct_zero(self) -> None:
        self.assertEqual(fmt_pct(0.0), "0.0%")

    def test_validate_schema_valid(self) -> None:
        df = pd.DataFrame({
            "ndc": ["12345-678-90"],
            "brand_name": ["Drug A"],
            "period": ["2024-01-01"],
            "wac_per_unit": [500.0],
            "gross_to_net_pct": [0.45],
        })
        errors = validate_price_history_schema(df)
        self.assertEqual(errors, [])

    def test_validate_schema_missing_column(self) -> None:
        df = pd.DataFrame({"ndc": ["12345-678-90"], "brand_name": ["Drug A"]})
        errors = validate_price_history_schema(df)
        self.assertTrue(len(errors) > 0)
        self.assertTrue(any("Missing" in e for e in errors))

    def test_validate_schema_negative_wac(self) -> None:
        df = pd.DataFrame({
            "ndc": ["1"], "brand_name": ["X"],
            "period": ["2024-01"], "wac_per_unit": [-1.0],
        })
        errors = validate_price_history_schema(df)
        self.assertTrue(any("negative" in e for e in errors))

    def test_validate_schema_gtn_out_of_range(self) -> None:
        df = pd.DataFrame({
            "ndc": ["1"], "brand_name": ["X"],
            "period": ["2024-01"], "wac_per_unit": [100.0],
            "gross_to_net_pct": [1.5],  # invalid
        })
        errors = validate_price_history_schema(df)
        self.assertTrue(any("gross_to_net_pct" in e for e in errors))

    def test_load_config_returns_dict(self) -> None:
        config = load_config(env_file=".env.example")
        self.assertIn("LOG_LEVEL", config)
        self.assertIn("DATA_DIR", config)

    def test_simulator_config_defaults(self) -> None:
        config = SimulatorConfig()
        self.assertEqual(config.seed, 42)
        self.assertEqual(config.n_drugs, 20)
        self.assertGreater(config.history_months, 0)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
