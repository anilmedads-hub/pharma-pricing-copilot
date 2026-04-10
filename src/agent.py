"""
agent.py
========
Multi-turn pharmaceutical pricing intelligence agent built on Claude Haiku.

PharmaPricingAgent orchestrates six tools — data retrieval, SQL querying,
causal analysis, pharmacy risk profiling, RAG-based regulatory search, and
action recommendation — within a conversational loop that supports both
blocking and streaming response modes.

Tool-use loop (both modes)
--------------------------
1.  User message → appended to conversation_history
2.  Claude called with tools + system prompt + full history
3.  If stop_reason == "tool_use": execute all tool blocks, append results,
    repeat (up to max_tool_calls=5 rounds)
4.  Streaming mode: after all tool calls resolve, make a final streaming
    call so the analyst sees tokens arrive in real time
5.  Blocking mode: extract text from the end_turn response directly
6.  Trim history to max_history_turns × 2 messages after each chat turn

Model: claude-haiku-4-5-20251001
API key: loaded from .env via python-dotenv

Usage
-----
>>> from src.agent import PharmaPricingAgent
>>> agent = PharmaPricingAgent()
>>> print(agent.chat("Are there any high-severity anomalies for DRG001?"))
>>> for chunk in agent.chat_streaming("Explain the 340B breach risk at PH0042"):
...     print(chunk, end="", flush=True)
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Generator

# Ensure project root is on sys.path for imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import anthropic
import pandas as pd
from dotenv import load_dotenv

from src.rag_pipeline import RAGPipeline

load_dotenv()

logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MODEL:          str = "claude-haiku-4-5-20251001"
_MAX_TOKENS:     int = 2048
_MAX_TOOL_CALLS: int = 5
_MAX_HISTORY:    int = 10   # turns (each turn = 1 user + 1 assistant message)

_DATA_DIR      = Path("data")
_RAW_DIR       = _DATA_DIR / "raw"
_PROCESSED_DIR = _DATA_DIR / "processed"

_SYSTEM_PROMPT: str = """\
You are a Senior Pharmaceutical Pricing Intelligence Analyst at a major PBM company.
You have deep expertise in:
  - Drug pricing benchmarks: WAC, ASP, AMP, 340B, GTN
  - Regulatory compliance: CMS, OIG, state Medicaid best price rules
  - Anomaly detection: statistical, ML-based, and regulatory violations
  - Causal inference and pre/post pricing impact analysis

Your job is to help pricing analysts investigate anomalies, understand
regulatory violations, and recommend actions. Always cite specific pricing
benchmarks and be precise with percentages and dollar amounts.
When severity is HIGH or regulatory_review is needed, always escalate.\
"""

# ---------------------------------------------------------------------------
# SQL safety: block any mutation keyword in the first token
# ---------------------------------------------------------------------------

_SQL_BLOCKED: frozenset[str] = frozenset({
    "insert", "update", "delete", "drop", "create",
    "alter", "truncate", "replace", "attach", "pragma",
})

# ---------------------------------------------------------------------------
# Action templates keyed by anomaly_type
# ---------------------------------------------------------------------------

_ACTION_TEMPLATES: dict[str, str] = {
    "wac_breach":          "Escalate to pricing committee. Flag pharmacy for WAC audit.",
    "340b_ceiling_breach": "Immediate regulatory review required. Notify 340B compliance team.",
    "asp_wac_violation":   "Review Medicare Part B billing. CMS notification may be required.",
    "gtn_floor_breach":    "Investigate rebate stacking. Review contract terms.",
    "margin_critical":     "Emergency margin review. Check claims data integrity.",
}
_DEFAULT_ACTION = "Flag for standard review cycle."

# ---------------------------------------------------------------------------
# Tool JSON schemas (Anthropic tool_use format)
# ---------------------------------------------------------------------------

_TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_anomaly_details",
        "description": (
            "Retrieve the top-5 most severe pricing anomalies for a specific drug. "
            "Returns anomaly type, severity, score, recommended action, transaction date, "
            "WAC price, actual price, and margin."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "drug_id": {
                    "type": "string",
                    "description": "Drug identifier from the dataset, e.g. 'DRG001'.",
                },
            },
            "required": ["drug_id"],
        },
    },
    {
        "name": "query_pricing_database",
        "description": (
            "Execute a read-only SQL SELECT query against the pharmaceutical pricing "
            "SQLite database (tables: drugs, pharmacies, transactions, anomalies). "
            "Returns a markdown-formatted table (max 20 rows). "
            "Only SELECT statements are permitted."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sql_query": {
                    "type": "string",
                    "description": "A SQL SELECT query string.",
                },
            },
            "required": ["sql_query"],
        },
    },
    {
        "name": "get_causal_summary",
        "description": (
            "Generate a pre/post causal price-impact summary for a drug by splitting "
            "its transaction history at the dataset midpoint. Returns average prices, "
            "margins, price trend (increasing/decreasing/stable), and volatility."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "drug_id": {
                    "type": "string",
                    "description": "Drug identifier, e.g. 'DRG001'.",
                },
            },
            "required": ["drug_id"],
        },
    },
    {
        "name": "get_pharmacy_risk_profile",
        "description": (
            "Return a risk profile for a specific pharmacy including total anomalies, "
            "high-severity count, regulatory breach count, most common violation type, "
            "risk tier (green/yellow/red), and the top 3 drugs flagged."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pharmacy_id": {
                    "type": "string",
                    "description": "Pharmacy identifier, e.g. 'PH0001'.",
                },
            },
            "required": ["pharmacy_id"],
        },
    },
    {
        "name": "search_regulatory_knowledge",
        "description": (
            "Search the pharmaceutical pricing compliance knowledge base using "
            "natural language. Returns the top-4 most relevant regulatory passages "
            "with similarity scores covering WAC rules, 340B, NADAC, GTN, and "
            "OIG/CMS compliance guidance."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "Natural-language compliance or pricing question.",
                },
            },
            "required": ["question"],
        },
    },
    {
        "name": "suggest_pricing_action",
        "description": (
            "Generate a prioritised action recommendation for a specific drug and "
            "anomaly type. Combines anomaly details with regulatory context to "
            "produce a concrete, escalation-aware recommendation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "drug_id": {
                    "type": "string",
                    "description": "Drug identifier.",
                },
                "anomaly_type": {
                    "type": "string",
                    "description": (
                        "Anomaly type string, e.g. 'wac_breach', "
                        "'340b_ceiling_breach', 'asp_wac_violation', "
                        "'gtn_floor_breach', 'margin_critical'."
                    ),
                },
            },
            "required": ["drug_id", "anomaly_type"],
        },
    },
]


# ===========================================================================
# Main Agent Class
# ===========================================================================


class PharmaPricingAgent:
    """Multi-turn pharmaceutical pricing intelligence agent.

    Wraps Claude Haiku with six domain-specific tools and maintains a
    rolling conversation history across turns.

    Parameters
    ----------
    model:
        Claude model identifier (default ``claude-haiku-4-5-20251001``).
    max_history_turns:
        Maximum number of full turns (user + assistant) to retain in
        the conversation window before trimming.
    """

    def __init__(
        self,
        model: str = _MODEL,
        max_history_turns: int = _MAX_HISTORY,
    ) -> None:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self._client          = anthropic.Anthropic(api_key=api_key)
        self._model           = model
        self.max_history_turns = max_history_turns

        # Conversation state
        self.conversation_history: list[dict[str, Any]] = []
        self._tools_called:        list[str]             = []

        # RAG pipeline — load existing vectorstore index
        logger.info("Loading RAG pipeline…")
        self._rag = RAGPipeline()
        try:
            self._rag._store.load_index()
            logger.info("RAG index loaded (%d chunks).", len(self._rag._store.chunks))
        except FileNotFoundError:
            logger.warning(
                "RAG index not found — call RAGPipeline().build_knowledge_base() "
                "before querying regulatory knowledge."
            )

        self._tools = _TOOLS
        logger.info(
            "PharmaPricingAgent ready (model=%s, %d tools, max_history=%d turns)",
            model, len(self._tools), max_history_turns,
        )

    # -----------------------------------------------------------------------
    # Public — chat interface
    # -----------------------------------------------------------------------

    def chat(self, user_message: str) -> str:
        """Run a full non-streaming multi-turn chat turn.

        Parameters
        ----------
        user_message:
            The analyst's question or instruction.

        Returns
        -------
        str
            The agent's final text response after all tool calls resolve.
        """
        self.conversation_history.append({"role": "user", "content": user_message})
        logger.info("chat() — user: '%s'", user_message[:80])

        final_text = ""
        for tool_round in range(_MAX_TOOL_CALLS):
            response = self._client.messages.create(
                model=self._model,
                max_tokens=_MAX_TOKENS,
                system=_SYSTEM_PROMPT,
                tools=self._tools,
                messages=self.conversation_history,
            )
            logger.debug(
                "chat() round %d: stop_reason=%s, content_blocks=%d",
                tool_round, response.stop_reason, len(response.content),
            )

            serialised = self._serialise_content(response.content)
            self.conversation_history.append({"role": "assistant", "content": serialised})

            if response.stop_reason == "end_turn":
                final_text = self._extract_text(response.content)
                break

            if response.stop_reason == "tool_use":
                tool_results = self._execute_all_tool_blocks(response.content)
                self.conversation_history.append({"role": "user", "content": tool_results})
                continue

            # Unexpected stop reason — extract whatever text exists
            final_text = self._extract_text(response.content)
            break

        self._trim_history()
        logger.info("chat() — response length: %d chars", len(final_text))
        return final_text

    def chat_streaming(
        self, user_message: str
    ) -> Generator[str, None, None]:
        """Run a streaming multi-turn chat turn.

        Tool-use rounds are resolved with blocking API calls; the final
        answer is streamed token-by-token so the analyst sees it arrive
        in real time.

        Parameters
        ----------
        user_message:
            The analyst's question or instruction.

        Yields
        ------
        str
            Incremental text fragments of the agent's final response.
        """
        self.conversation_history.append({"role": "user", "content": user_message})
        logger.info("chat_streaming() — user: '%s'", user_message[:80])

        # ── Phase 1: Resolve all tool calls with blocking calls ────────────
        for tool_round in range(_MAX_TOOL_CALLS):
            response = self._client.messages.create(
                model=self._model,
                max_tokens=_MAX_TOKENS,
                system=_SYSTEM_PROMPT,
                tools=self._tools,
                messages=self.conversation_history,
            )
            logger.debug(
                "chat_streaming() tool round %d: stop_reason=%s",
                tool_round, response.stop_reason,
            )

            if response.stop_reason == "end_turn":
                # No tool calls needed — history ready for streaming
                break

            if response.stop_reason == "tool_use":
                serialised   = self._serialise_content(response.content)
                self.conversation_history.append({"role": "assistant", "content": serialised})
                tool_results = self._execute_all_tool_blocks(response.content)
                self.conversation_history.append({"role": "user", "content": tool_results})
                continue

            break  # unexpected stop reason

        # ── Phase 2: Stream the final answer ──────────────────────────────
        full_chunks: list[str] = []
        with self._client.messages.stream(
            model=self._model,
            max_tokens=_MAX_TOKENS,
            system=_SYSTEM_PROMPT,
            tools=self._tools,
            messages=self.conversation_history,
        ) as stream:
            for text in stream.text_stream:
                full_chunks.append(text)
                yield text

        # Append completed response to history
        complete_text = "".join(full_chunks)
        self.conversation_history.append({"role": "assistant", "content": complete_text})
        self._trim_history()
        logger.info(
            "chat_streaming() — streamed %d chars", len(complete_text)
        )

    def reset_conversation(self) -> None:
        """Clear conversation history and tool-call log for a fresh session."""
        self.conversation_history = []
        self._tools_called        = []
        logger.info("Conversation history cleared.")

    def get_conversation_summary(self) -> dict[str, Any]:
        """Return a lightweight summary of the current conversation state.

        Returns
        -------
        dict with keys:
            total_turns (int), tools_called (list[str]),
            last_user_message (str), last_assistant_preview (str, first 100 chars).
        """
        history = self.conversation_history

        # Last user message (string content only)
        last_user = next(
            (
                m["content"] if isinstance(m["content"], str) else ""
                for m in reversed(history)
                if m["role"] == "user" and isinstance(m["content"], str)
            ),
            "",
        )

        # Last assistant message text preview
        last_asst_preview = ""
        for m in reversed(history):
            if m["role"] != "assistant":
                continue
            content = m["content"]
            if isinstance(content, str):
                last_asst_preview = content[:100]
                break
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        last_asst_preview = block["text"][:100]
                        break
                if last_asst_preview:
                    break

        # Turn count: pair of user/assistant messages = 1 turn
        user_msgs = sum(1 for m in history if m["role"] == "user"
                        and isinstance(m["content"], str))

        return {
            "total_turns":           user_msgs,
            "tools_called":          list(self._tools_called),
            "last_user_message":     last_user,
            "last_assistant_preview": last_asst_preview,
        }

    # -----------------------------------------------------------------------
    # Private — tool execution dispatcher
    # -----------------------------------------------------------------------

    def _execute_all_tool_blocks(
        self, content: list[Any]
    ) -> list[dict[str, Any]]:
        """Execute every tool_use block in *content* and return tool results.

        Parameters
        ----------
        content:
            List of content blocks from an Anthropic API response.

        Returns
        -------
        list[dict]
            List of ``{"type": "tool_result", "tool_use_id": ..., "content": ...}``
            dicts ready to be added as a user message.
        """
        results: list[dict[str, Any]] = []
        for block in content:
            if not hasattr(block, "type") or block.type != "tool_use":
                continue
            tool_result = self._execute_tool(block.name, block.input)
            results.append({
                "type":        "tool_result",
                "tool_use_id": block.id,
                "content":     tool_result,
            })
        return results

    def _execute_tool(self, name: str, tool_input: dict[str, Any]) -> str:
        """Dispatch a single tool call and return the string result.

        Parameters
        ----------
        name:
            Tool name (must match one of the registered _TOOLS names).
        tool_input:
            Keyword arguments dict for the tool.

        Returns
        -------
        str
            Human-readable tool output, or an error message if execution fails.
        """
        self._tools_called.append(name)
        logger.info("Tool called: %s | input=%s", name, json.dumps(tool_input)[:120])

        handlers: dict[str, Any] = {
            "get_anomaly_details":          self._tool_get_anomaly_details,
            "query_pricing_database":       self._tool_query_pricing_database,
            "get_causal_summary":           self._tool_get_causal_summary,
            "get_pharmacy_risk_profile":    self._tool_get_pharmacy_risk_profile,
            "search_regulatory_knowledge":  self._tool_search_regulatory_knowledge,
            "suggest_pricing_action":       self._tool_suggest_pricing_action,
        }

        handler = handlers.get(name)
        if handler is None:
            msg = f"Unknown tool: '{name}'"
            logger.error(msg)
            return msg

        try:
            result = handler(**tool_input)
            logger.info("Tool '%s' → %d chars", name, len(result))
            return result
        except Exception as exc:
            msg = f"Tool '{name}' raised {type(exc).__name__}: {exc}"
            logger.exception(msg)
            return msg

    # -----------------------------------------------------------------------
    # Private — tool implementations
    # -----------------------------------------------------------------------

    def _tool_get_anomaly_details(self, drug_id: str) -> str:
        """Tool 1: Load anomaly_results.csv and return top-5 anomalies for drug_id.

        Parameters
        ----------
        drug_id:
            Drug identifier string (e.g. ``"DRG001"``).

        Returns
        -------
        str
            Formatted anomaly listing or a "no anomalies" message.
        """
        csv_path = _PROCESSED_DIR / "anomaly_results.csv"
        if not csv_path.exists():
            return (
                f"Anomaly results file not found at {csv_path}. "
                "Run AnomalyDetectionEngine().detect_all() first."
            )

        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            return f"Failed to load anomaly results: {exc}"

        drug_rows = df[df["drug_id"] == drug_id]
        if drug_rows.empty:
            return f"No anomalies found for {drug_id}."

        # Severity ordering for sorting
        sev_order = {"high": 0, "medium": 1, "low": 2}
        top5 = (
            drug_rows.assign(_sev_rank=drug_rows["severity"].map(sev_order).fillna(3))
            .sort_values(["_sev_rank", "anomaly_score"], ascending=[True, False])
            .head(5)
        )

        lines = [f"=== Anomaly Details for {drug_id} (top {len(top5)}) ==="]
        for i, (_, row) in enumerate(top5.iterrows(), 1):
            lines.append(
                f"\n[{i}] Drug: {row.get('drug_name', drug_id)} | "
                f"Type: {row.get('anomaly_type', '?')} | "
                f"Severity: {row.get('severity', '?')}\n"
                f"     Score: {row.get('anomaly_score', 0):.2f} | "
                f"Action: {row.get('recommended_action', '?')}\n"
                f"     Date: {row.get('transaction_date', '?')} | "
                f"WAC: ${row.get('wac_price', 0):.2f}\n"
                f"     Actual: ${row.get('actual_price', 0):.2f} | "
                f"Margin: {row.get('margin_percent', 0):.1f}%\n"
                f"     {row.get('description', '')}"
            )
        return "\n".join(lines)

    def _tool_query_pricing_database(self, sql_query: str) -> str:
        """Tool 2: Execute a read-only SELECT query against pricing.db.

        Parameters
        ----------
        sql_query:
            SQL SELECT statement.  Any non-SELECT statement is rejected.

        Returns
        -------
        str
            Markdown-formatted results table or an error/rejection message.
        """
        first_word = sql_query.strip().split()[0].lower() if sql_query.strip() else ""
        if first_word in _SQL_BLOCKED:
            return (
                f"Rejected: '{first_word.upper()}' statements are not permitted. "
                "Only SELECT queries are allowed."
            )

        db_path = _PROCESSED_DIR / "pricing.db"
        if not db_path.exists():
            return (
                f"Database not found at {db_path}. "
                "Run PricingDataSimulator().build_sqlite_db() first."
            )

        try:
            con = sqlite3.connect(db_path)
            con.row_factory = sqlite3.Row
            cursor = con.execute(sql_query)
            rows   = cursor.fetchmany(20)
            con.close()
        except sqlite3.Error as exc:
            return f"SQL error: {exc}"
        except Exception as exc:
            return f"Database error: {exc}"

        if not rows:
            return "Query returned no results."

        # Build markdown table
        cols      = list(rows[0].keys())
        header    = " | ".join(cols)
        separator = " | ".join("---" for _ in cols)
        body      = "\n".join(
            " | ".join(str(row[c]) for c in cols) for row in rows
        )
        return f"| {header} |\n| {separator} |\n" + "\n".join(
            f"| {' | '.join(str(row[c]) for c in cols)} |" for row in rows
        )

    def _tool_get_causal_summary(self, drug_id: str) -> str:
        """Tool 3: Pre/post causal analysis for a drug split at the dataset midpoint.

        Parameters
        ----------
        drug_id:
            Drug identifier string.

        Returns
        -------
        str
            Structured causal summary string.
        """
        csv_path = _RAW_DIR / "drug_pricing.csv"
        if not csv_path.exists():
            return f"Pricing CSV not found at {csv_path}."

        try:
            df = pd.read_csv(csv_path, parse_dates=["transaction_date"])
        except Exception as exc:
            return f"Failed to load pricing CSV: {exc}"

        drug_rows = df[df["drug_id"] == drug_id].sort_values("transaction_date")
        if drug_rows.empty:
            return f"No data found for drug_id '{drug_id}'."

        drug_name = drug_rows["drug_name"].iloc[0]
        n         = len(drug_rows)
        midpoint  = n // 2

        pre  = drug_rows.iloc[:midpoint]
        post = drug_rows.iloc[midpoint:]

        pre_avg_price    = float(pre["actual_price"].mean())
        post_avg_price   = float(post["actual_price"].mean())
        pre_avg_margin   = float(pre["margin_percent"].mean())
        post_avg_margin  = float(post["margin_percent"].mean())
        volatility       = float(drug_rows["actual_price"].std())

        price_change_pct = (
            (post_avg_price - pre_avg_price) / pre_avg_price * 100
            if pre_avg_price else 0.0
        )
        margin_change_pct = (
            (post_avg_margin - pre_avg_margin) / abs(pre_avg_margin) * 100
            if pre_avg_margin else 0.0
        )

        if abs(price_change_pct) < 5:
            trend = "stable"
        elif price_change_pct > 0:
            trend = "increasing"
        else:
            trend = "decreasing"

        pre_start  = str(pre["transaction_date"].min().date())
        pre_end    = str(pre["transaction_date"].max().date())
        post_start = str(post["transaction_date"].min().date())
        post_end   = str(post["transaction_date"].max().date())

        return (
            f"=== Causal Summary: {drug_name} ({drug_id}) ===\n"
            f"Total transactions analysed : {n}\n"
            f"\nPRE-PERIOD  ({pre_start} → {pre_end}):\n"
            f"  Avg actual price : ${pre_avg_price:,.2f}\n"
            f"  Avg margin       : {pre_avg_margin:.1f}%\n"
            f"\nPOST-PERIOD ({post_start} → {post_end}):\n"
            f"  Avg actual price : ${post_avg_price:,.2f}\n"
            f"  Avg margin       : {post_avg_margin:.1f}%\n"
            f"\nCHANGES:\n"
            f"  Price change     : {price_change_pct:+.1f}%\n"
            f"  Margin change    : {margin_change_pct:+.1f}%\n"
            f"  Price trend      : {trend}\n"
            f"  Price volatility : ${volatility:,.2f} (std)\n"
        )

    def _tool_get_pharmacy_risk_profile(self, pharmacy_id: str) -> str:
        """Tool 4: Compute risk profile for a specific pharmacy.

        Parameters
        ----------
        pharmacy_id:
            Pharmacy identifier string (e.g. ``"PH0001"``).

        Returns
        -------
        str
            Formatted risk profile string.
        """
        csv_path = _PROCESSED_DIR / "anomaly_results.csv"
        if not csv_path.exists():
            return (
                f"Anomaly results not found at {csv_path}. "
                "Run detect_all() first."
            )

        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            return f"Failed to load anomaly results: {exc}"

        pharm_rows = df[df["pharmacy_id"] == pharmacy_id]
        if pharm_rows.empty:
            return f"No anomaly records found for pharmacy_id '{pharmacy_id}'."

        total_anomalies    = len(pharm_rows)
        high_severity_count = int((pharm_rows["severity"] == "high").sum())
        reg_keywords       = ("breach", "violation")
        regulatory_breach_count = int(
            pharm_rows["anomaly_type"]
            .fillna("")
            .apply(lambda t: any(k in t for k in reg_keywords))
            .sum()
        )

        most_common = (
            pharm_rows["anomaly_type"].mode().iloc[0]
            if not pharm_rows["anomaly_type"].mode().empty
            else "N/A"
        )

        top3_drugs = (
            pharm_rows["drug_name"]
            .value_counts()
            .head(3)
            .index.tolist()
        )

        if high_severity_count > 5:
            risk_tier = "red"
        elif high_severity_count > 2:
            risk_tier = "yellow"
        else:
            risk_tier = "green"

        pharm_name  = pharm_rows["pharmacy_name"].iloc[0]  if "pharmacy_name"  in pharm_rows.columns else pharmacy_id
        pharm_chain = pharm_rows["pharmacy_chain"].iloc[0] if "pharmacy_chain" in pharm_rows.columns else "Unknown"
        state       = pharm_rows["state"].iloc[0]          if "state"          in pharm_rows.columns else "?"

        return (
            f"=== Risk Profile: {pharm_name} ({pharmacy_id}) ===\n"
            f"Chain      : {pharm_chain} | State: {state}\n"
            f"Risk Tier  : {risk_tier.upper()}\n"
            f"\nANOMALY COUNTS:\n"
            f"  Total anomalies        : {total_anomalies}\n"
            f"  High-severity          : {high_severity_count}\n"
            f"  Regulatory breaches    : {regulatory_breach_count}\n"
            f"  Most common violation  : {most_common}\n"
            f"\nTOP 3 DRUGS FLAGGED:\n"
            + "\n".join(f"  • {d}" for d in top3_drugs)
        )

    def _tool_search_regulatory_knowledge(self, question: str) -> str:
        """Tool 5: Retrieve top-4 regulatory knowledge base passages for a question.

        Parameters
        ----------
        question:
            Natural-language compliance or pricing question.

        Returns
        -------
        str
            Formatted context string with source metadata and similarity scores.
        """
        try:
            chunks = self._rag._store.search(question, k=4)
        except RuntimeError:
            # Index not loaded — attempt to load
            try:
                self._rag._store.load_index()
                chunks = self._rag._store.search(question, k=4)
            except Exception as exc:
                return (
                    f"Regulatory knowledge base not available: {exc}. "
                    "Call RAGPipeline().build_knowledge_base() first."
                )
        except Exception as exc:
            return f"Knowledge base search failed: {exc}"

        if not chunks:
            return "No relevant regulatory passages found."

        parts: list[str] = []
        for i, chunk in enumerate(chunks, 1):
            score   = chunk.get("similarity_score", chunk.get("similarity", 0.0))
            topic   = chunk.get("metadata", {}).get("topic", "")
            title   = chunk.get("metadata", {}).get("title", "")
            snippet = chunk.get("text", "")[:500]
            parts.append(
                f"Source {i}: [score: {score:.3f}] {topic} — {title}\n{snippet}"
            )

        return "\n\n---\n\n".join(parts)

    def _tool_suggest_pricing_action(
        self, drug_id: str, anomaly_type: str
    ) -> str:
        """Tool 6: Build a prioritised action recommendation.

        Parameters
        ----------
        drug_id:
            Drug identifier string.
        anomaly_type:
            Anomaly type string (e.g. ``"wac_breach"``).

        Returns
        -------
        str
            Action recommendation combining anomaly details and regulatory context.
        """
        anomaly_details = self._tool_get_anomaly_details(drug_id)
        reg_context     = self._tool_search_regulatory_knowledge(
            f"{anomaly_type.replace('_', ' ')} pharmaceutical pricing rule"
        )
        action_template = _ACTION_TEMPLATES.get(anomaly_type, _DEFAULT_ACTION)

        return (
            f"=== Pricing Action Recommendation ===\n"
            f"Drug       : {drug_id}\n"
            f"Anomaly    : {anomaly_type}\n"
            f"\nRECOMMENDED ACTION:\n  {action_template}\n"
            f"\nANOMALY CONTEXT:\n{anomaly_details}\n"
            f"\nREGULATORY CONTEXT:\n{reg_context[:800]}\n"
        )

    # -----------------------------------------------------------------------
    # Private — helpers
    # -----------------------------------------------------------------------

    def _serialise_content(self, content: list[Any]) -> list[dict[str, Any]]:
        """Convert Anthropic SDK content blocks to plain dicts for history storage.

        Parameters
        ----------
        content:
            List of Anthropic content block objects (TextBlock, ToolUseBlock, etc.).

        Returns
        -------
        list[dict]
            JSON-serialisable list of content block dicts.
        """
        result: list[dict[str, Any]] = []
        for block in content:
            if not hasattr(block, "type"):
                continue
            if block.type == "text":
                result.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                result.append({
                    "type":  "tool_use",
                    "id":    block.id,
                    "name":  block.name,
                    "input": block.input,
                })
        return result

    @staticmethod
    def _extract_text(content: list[Any]) -> str:
        """Extract concatenated text from a list of content blocks.

        Parameters
        ----------
        content:
            List of Anthropic content block objects.

        Returns
        -------
        str
            Concatenated text from all TextBlock items.
        """
        parts: list[str] = []
        for block in content:
            if hasattr(block, "type") and block.type == "text":
                parts.append(block.text)
        return "".join(parts)

    def _trim_history(self) -> None:
        """Trim conversation_history to the last ``max_history_turns`` turns.

        Keeps the most recent ``max_history_turns * 2`` messages (each turn
        is one user message + one assistant message).
        """
        limit = self.max_history_turns * 2
        if len(self.conversation_history) > limit:
            self.conversation_history = self.conversation_history[-limit:]
            logger.debug("History trimmed to %d messages.", limit)
