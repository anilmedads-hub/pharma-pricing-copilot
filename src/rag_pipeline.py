"""
rag_pipeline.py
===============
Retrieval-Augmented Generation pipeline for pharmaceutical pricing compliance.

Three-class architecture:

TextChunker
    Splits raw documents into overlapping fixed-size character chunks and
    attaches metadata and a deterministic chunk ID to each piece.

VectorStore
    Sentence-transformer embedding engine backed by a plain numpy matrix.
    Cosine similarity is computed as a dot product on unit-normalised vectors.
    The index is persisted to ``data/vectorstore/`` as .npy / .json files so
    it can be rebuilt once and reloaded on subsequent runs.

RAGPipeline
    Orchestrates end-to-end Q&A:
      1. Calls ``build_knowledge_base()`` to create and index 60 synthetic
         pharmaceutical pricing documents across 6 compliance topics.
      2. Retrieves the top-k relevant chunks for a user query.
      3. Sends the retrieved context + conversation history to Claude Haiku
         (``claude-haiku-4-5-20251001``) via the Anthropic API and returns the
         grounded answer.
    Supports both blocking and streaming answer generation.

Embedding model : sentence-transformers/all-MiniLM-L6-v2  (384-dim, local)
Generation model: claude-haiku-4-5-20251001               (Anthropic API)

Usage
-----
>>> from src.rag_pipeline import RAGPipeline
>>> rag = RAGPipeline()
>>> rag.build_knowledge_base()           # run once; index is cached to disk
>>> answer = rag.answer("What is the 340B ceiling price formula?")
>>> print(answer)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Generator

import anthropic
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EMBEDDING_MODEL:  str = "all-MiniLM-L6-v2"
_GENERATION_MODEL: str = "claude-haiku-4-5-20251001"
_VECTORSTORE_DIR:  str = "data/vectorstore"
_MAX_HISTORY_MSGS: int = 6     # last N messages included in context window
_MAX_CONTEXT_CHARS: int = 6000  # approximate cap on retrieved context fed to LLM

_SYSTEM_PROMPT: str = """\
You are a pharmaceutical pricing compliance expert with deep knowledge of:
  - WAC (Wholesale Acquisition Cost) pricing rules and manufacturer obligations
  - ASP (Average Selling Price) Medicare Part B benchmarks
  - AMP (Average Manufacturer Price) and Medicaid rebate calculations
  - 340B Drug Pricing Program eligibility, ceiling prices, and audit triggers
  - NADAC (National Average Drug Acquisition Cost) methodology
  - Gross-to-Net rebate structures and pricing transparency requirements
  - OIG, CMS, FTC regulatory frameworks and enforcement actions

You assist PBM analysts, pharmacy compliance teams, and pricing managers.
When answering:
  - Cite specific regulatory references (statute, programme name) when available.
  - Be precise about numeric thresholds (e.g. AMP × 0.855 for 340B ceiling).
  - Flag compliance risks clearly.
  - If the retrieved context does not cover the question, say so explicitly.\
"""


# ===========================================================================
# CLASS 1 — TextChunker
# ===========================================================================


class TextChunker:
    """Split documents into overlapping character-level chunks for embedding.

    Parameters
    ----------
    chunk_size:
        Maximum character length of each chunk (default 400).
    overlap:
        Number of characters shared between consecutive chunks (default 80).
        Must be less than ``chunk_size``.
    """

    def __init__(self, chunk_size: int = 400, overlap: int = 80) -> None:
        if overlap >= chunk_size:
            raise ValueError("overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.overlap    = overlap
        logger.debug("TextChunker init (chunk_size=%d, overlap=%d)", chunk_size, overlap)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_text(self, text: str, metadata: dict[str, Any]) -> list[dict[str, Any]]:
        """Split *text* into overlapping chunks.

        Parameters
        ----------
        text:
            Full document text to split.
        metadata:
            Arbitrary key-value pairs attached verbatim to every chunk
            (e.g. ``{"doc_id": "WAC-001", "topic": "WAC Policy"}``).

        Returns
        -------
        list[dict]
            Each element has keys ``text``, ``metadata``, ``chunk_id``.
            ``chunk_id`` is a 12-character hex digest of
            ``"{doc_id}-{chunk_index}"``.
        """
        text   = text.strip()
        chunks: list[dict[str, Any]] = []
        start  = 0
        idx    = 0
        stride = self.chunk_size - self.overlap
        doc_id = str(metadata.get("doc_id", "doc"))

        while start < len(text):
            end        = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk_id = hashlib.sha256(
                    f"{doc_id}-{idx}".encode()
                ).hexdigest()[:12]
                chunks.append({
                    "text":     chunk_text,
                    "metadata": {**metadata, "chunk_index": idx},
                    "chunk_id": chunk_id,
                })
                idx += 1
            start += stride

        return chunks

    def chunk_documents(self, docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Chunk a list of documents and return a flat list of all chunks.

        Parameters
        ----------
        docs:
            Each document dict must have a ``"content"`` key with the raw
            text.  All other keys are forwarded as chunk metadata.

        Returns
        -------
        list[dict]
            Flat list of chunk dicts from all documents.
        """
        all_chunks: list[dict[str, Any]] = []
        for doc in docs:
            text     = doc.get("content", "")
            metadata = {k: v for k, v in doc.items() if k != "content"}
            chunks   = self.chunk_text(text, metadata)
            all_chunks.extend(chunks)

        logger.info(
            "chunk_documents: %d docs → %d chunks",
            len(docs), len(all_chunks),
        )
        return all_chunks


# ===========================================================================
# CLASS 2 — VectorStore
# ===========================================================================


class VectorStore:
    """Numpy-backed vector store with sentence-transformer embeddings.

    Similarity is computed as a dot product on L2-normalised embedding
    vectors, which is equivalent to cosine similarity.

    Parameters
    ----------
    embedding_model:
        HuggingFace sentence-transformers model name.
        Default: ``"all-MiniLM-L6-v2"`` (384-dim, fast, good quality).
    vectorstore_dir:
        Directory for persisting ``index.npy`` and ``chunks.json``.
    """

    def __init__(
        self,
        embedding_model: str = _EMBEDDING_MODEL,
        vectorstore_dir: str | Path = _VECTORSTORE_DIR,
    ) -> None:
        self._vs_dir = Path(vectorstore_dir)
        self._vs_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Loading embedding model '%s'…", embedding_model)
        self._model: SentenceTransformer = SentenceTransformer(embedding_model)
        self._dim: int = self._model.get_sentence_embedding_dimension()
        logger.info("Embedding model ready (dim=%d)", self._dim)

        self.index:  np.ndarray | None   = None   # shape (N, dim)
        self.chunks: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, texts: list[str]) -> np.ndarray:
        """Encode *texts* into L2-normalised embedding vectors.

        Parameters
        ----------
        texts:
            List of strings to embed.

        Returns
        -------
        np.ndarray
            Shape ``(len(texts), embedding_dim)`` with unit-norm rows.
        """
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=64,
        )
        return np.array(embeddings, dtype=np.float32)

    def build_index(self, chunks: list[dict[str, Any]]) -> None:
        """Embed all chunks and persist the index to disk.

        Parameters
        ----------
        chunks:
            List of chunk dicts (each must contain a ``"text"`` key).
            Typically the output of :meth:`TextChunker.chunk_documents`.
        """
        if not chunks:
            raise ValueError("Cannot build index from empty chunk list.")

        logger.info("Building index for %d chunks…", len(chunks))
        texts          = [c["text"] for c in chunks]
        self.index     = self.embed(texts)
        self.chunks    = list(chunks)

        # ── Persist ───────────────────────────────────────────────────
        idx_path    = self._vs_dir / "index.npy"
        chunks_path = self._vs_dir / "chunks.json"

        np.save(str(idx_path), self.index)
        with chunks_path.open("w", encoding="utf-8") as fh:
            json.dump(self.chunks, fh, ensure_ascii=False, indent=2)

        logger.info(
            "Index persisted → %s (%.1f MB) + %s",
            idx_path,
            idx_path.stat().st_size / 1e6,
            chunks_path,
        )

    def load_index(self) -> None:
        """Load a previously persisted index from ``vectorstore_dir``.

        Raises
        ------
        FileNotFoundError
            If either ``index.npy`` or ``chunks.json`` is missing.
        """
        idx_path    = self._vs_dir / "index.npy"
        chunks_path = self._vs_dir / "chunks.json"

        if not idx_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(
                f"Vector store not found in '{self._vs_dir}'. "
                "Call build_index() or RAGPipeline.build_knowledge_base() first."
            )

        self.index  = np.load(str(idx_path))
        with chunks_path.open("r", encoding="utf-8") as fh:
            self.chunks = json.load(fh)

        logger.info(
            "Index loaded: %d vectors (dim=%d) from %s",
            len(self.chunks), self.index.shape[1], self._vs_dir,
        )

    def search(self, query: str, k: int = 4) -> list[dict[str, Any]]:
        """Return the top-*k* most relevant chunks for *query*.

        Parameters
        ----------
        query:
            Natural-language question or search phrase.
        k:
            Number of results to return.

        Returns
        -------
        list[dict]
            Each element is a chunk dict with two added keys:
            ``"similarity_score"`` and ``"similarity"`` (both identical,
            float in [−1, 1], higher = better).

        Raises
        ------
        RuntimeError
            If the index has not been built or loaded.
        """
        if self.index is None or not self.chunks:
            raise RuntimeError(
                "Index is empty. Call build_index() or load_index() first."
            )

        q_emb = self.embed([query])                          # (1, dim)
        sims  = (self.index @ q_emb.T).flatten()             # (N,)

        k         = min(k, len(self.chunks))
        top_idx   = np.argsort(sims)[::-1][:k]

        results: list[dict[str, Any]] = []
        for i in top_idx:
            chunk = dict(self.chunks[i])
            score = float(sims[i])
            chunk["similarity_score"] = score   # canonical name
            chunk["similarity"] = score         # convenience alias
            results.append(chunk)

        return results


# ===========================================================================
# CLASS 3 — RAGPipeline
# ===========================================================================


class RAGPipeline:
    """End-to-end RAG pipeline for pharmaceutical pricing compliance Q&A.

    Initialises a :class:`TextChunker` and :class:`VectorStore`, then
    exposes ``build_knowledge_base()``, ``retrieve()``, ``answer()``, and
    ``answer_streaming()`` as the public interface.

    The Anthropic API key is loaded automatically from the ``.env`` file
    via :mod:`python-dotenv`.
    """

    def __init__(self) -> None:
        self._chunker     = TextChunker(chunk_size=400, overlap=80)
        self._store       = VectorStore()
        self._vs_dir      = Path(_VECTORSTORE_DIR)
        self._api_key     = os.environ.get("ANTHROPIC_API_KEY", "")
        self._client      = anthropic.Anthropic(api_key=self._api_key)
        self._gen_model   = _GENERATION_MODEL
        logger.info("RAGPipeline ready (generation_model=%s)", self._gen_model)

    # ------------------------------------------------------------------
    # Knowledge base
    # ------------------------------------------------------------------

    def build_knowledge_base(self) -> None:
        """Generate 60 synthetic compliance documents and build the FAISS index.

        Documents cover six topics (10 each):
          1. WAC Pricing Policy
          2. NADAC Benchmarks
          3. 340B Program Rules
          4. GTN & Rebate Structures
          5. Anomaly Case Studies
          6. Regulatory & Compliance

        The embedded index is saved to ``data/vectorstore/``.  Subsequent
        calls reuse the cached index without re-embedding.
        """
        idx_path = self._vs_dir / "index.npy"
        if idx_path.exists():
            logger.info("Cached index found — loading from %s", self._vs_dir)
            self._store.load_index()
            return

        logger.info("Building knowledge base from 60 synthetic documents…")
        docs   = self._generate_knowledge_docs()
        chunks = self._chunker.chunk_documents(docs)
        self._store.build_index(chunks)
        logger.info("Knowledge base ready (%d chunks indexed).", len(chunks))

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, k: int = 4) -> list[dict[str, Any]]:
        """Return the top-*k* most relevant knowledge base chunks.

        Parameters
        ----------
        query:
            User question or keyword phrase.
        k:
            Number of chunks to retrieve (default 4).

        Returns
        -------
        list[dict]
            Chunk dicts with ``text``, ``metadata``, ``chunk_id``, and
            ``similarity_score``.
        """
        if self._store.index is None:
            self.build_knowledge_base()
        results = self._store.search(query, k=k)
        logger.debug("Retrieved %d chunks for query: '%s'", len(results), query[:60])
        return results

    # ------------------------------------------------------------------
    # Answer generation — blocking
    # ------------------------------------------------------------------

    def answer(
        self,
        query: str,
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> str:
        """Answer *query* using retrieved context and Claude Haiku.

        Parameters
        ----------
        query:
            The user's pharmaceutical pricing question.
        conversation_history:
            Optional list of prior ``{"role": "user"|"assistant", "content": str}``
            messages.  The last :data:`_MAX_HISTORY_MSGS` are included.

        Returns
        -------
        str
            The model's grounded answer.
        """
        messages = self._build_messages(query, conversation_history)
        response = self._client.messages.create(
            model=self._gen_model,
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            messages=messages,
        )
        answer_text: str = response.content[0].text if response.content else ""
        logger.info("answer() — %d input tokens, %d output tokens",
                    response.usage.input_tokens, response.usage.output_tokens)
        return answer_text

    # ------------------------------------------------------------------
    # Answer generation — streaming
    # ------------------------------------------------------------------

    def answer_streaming(
        self,
        query: str,
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> Generator[str, None, None]:
        """Stream the answer token-by-token using the Anthropic streaming API.

        Parameters
        ----------
        query:
            The user's pharmaceutical pricing question.
        conversation_history:
            Optional prior conversation turns (last ``_MAX_HISTORY_MSGS``
            messages are used).

        Yields
        ------
        str
            Incremental text fragments as they arrive from the API.
        """
        messages = self._build_messages(query, conversation_history)
        logger.info("answer_streaming() — starting stream for query: '%s'", query[:60])
        with self._client.messages.stream(
            model=self._gen_model,
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                yield text

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        query: str,
        conversation_history: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        """Compose the Anthropic messages list for a given query.

        Retrieves relevant context, prepends up to
        :data:`_MAX_HISTORY_MSGS` prior turns, and appends the current
        user message with context inline.

        Parameters
        ----------
        query:
            Current user question.
        conversation_history:
            Optional prior conversation turns.

        Returns
        -------
        list[dict]
            Ready-to-send ``messages`` list for the Anthropic API.
        """
        chunks = self.retrieve(query, k=4)

        # Build context block — cap at _MAX_CONTEXT_CHARS
        context_parts: list[str] = []
        budget = _MAX_CONTEXT_CHARS
        for chunk in chunks:
            topic  = chunk["metadata"].get("topic", "")
            title  = chunk["metadata"].get("title", "")
            header = f"[{topic} — {title}]" if title else f"[{topic}]"
            body   = chunk["text"][:budget]
            context_parts.append(f"{header}\n{body}")
            budget -= len(body)
            if budget <= 0:
                break
        context_str = "\n\n---\n\n".join(context_parts)

        # Build conversation history slice
        history = list(conversation_history or [])
        if len(history) > _MAX_HISTORY_MSGS:
            # Keep the most recent N messages while preserving role alternation
            history = history[-_MAX_HISTORY_MSGS:]

        messages: list[dict[str, Any]] = list(history)
        messages.append({
            "role":    "user",
            "content": (
                f"RETRIEVED KNOWLEDGE BASE CONTEXT:\n"
                f"{context_str}\n\n"
                f"---\n\n"
                f"QUESTION: {query}"
            ),
        })
        return messages

    # ------------------------------------------------------------------
    # Knowledge document generation (60 synthetic docs)
    # ------------------------------------------------------------------

    def _generate_knowledge_docs(self) -> list[dict[str, Any]]:
        """Return 60 synthetic pharmaceutical pricing compliance documents.

        Each document is a dict with keys:
          doc_id, title, topic, content

        Topics (10 docs each)
        ----------------------
        1. WAC Pricing Policy
        2. NADAC Benchmarks
        3. 340B Program Rules
        4. GTN & Rebate Structures
        5. Anomaly Case Studies
        6. Regulatory & Compliance
        """
        docs: list[dict[str, Any]] = []

        # ── Topic 1: WAC Pricing Policy ───────────────────────────────

        docs.append({"doc_id": "WAC-001", "title": "WAC Definition and Statutory Framework", "topic": "WAC Pricing Policy", "content": """
Wholesale Acquisition Cost (WAC) is defined under 42 U.S.C. § 1395w-3a as the manufacturer's list price for a drug to wholesalers or direct purchasers in the United States, not including prompt-pay or other discounts, rebates, or reductions in price. WAC serves as the foundational list price from which all other pricing benchmarks—ASP, AMP, Best Price, and 340B ceiling prices—are derived. Manufacturers are legally required to report WAC to data aggregators such as Medi-Span, First DataBank, and RED BOOK within 30 days of any price change. WAC is distinct from Average Wholesale Price (AWP), which is a historically inflated benchmark that typically runs 20-25% above WAC. Unlike AWP, WAC represents an actual transaction price (pre-discount) rather than a published benchmark. Federal regulations prohibit manufacturers from misrepresenting WAC; doing so can trigger False Claims Act liability. State transparency laws in California (SB 17), Oregon, and Nevada additionally require advance notification of WAC increases exceeding 16% over a two-year period, accompanied by a manufacturer justification letter.
""".strip()})

        docs.append({"doc_id": "WAC-002", "title": "AWP vs WAC: Historical Relationship and Spread", "topic": "WAC Pricing Policy", "content": """
Average Wholesale Price (AWP) has historically been priced at a fixed spread above WAC—typically AWP = WAC × 1.20 for brand-name drugs and AWP = WAC × 1.25 for generics, though this varies by publisher. The AWP-WAC spread, known as the "spread," created significant pricing arbitrage opportunities exploited in the 1990s and 2000s. Major litigation, including the AWP multidistrict litigation (MDL 1456), resulted in settlements exceeding $350 million paid by multiple manufacturers for publishing inflated AWP benchmarks. CMS moved away from AWP-based reimbursement for Medicare Part B toward ASP + 6% in 2005 (MMA 2003 provision). However, many commercial payer and PBM contracts continue to reference AWP as a benchmark. Pharmacies typically acquire drugs at WAC minus a wholesaler discount (1-3%), making the AWP-WAC spread a key driver of pharmacy gross margin, particularly for specialty products where spread can reach $500-$2,000 per unit.
""".strip()})

        docs.append({"doc_id": "WAC-003", "title": "Manufacturer WAC Reporting Obligations", "topic": "WAC Pricing Policy", "content": """
Under the Drug Supply Chain Security Act (DSCSA) and CMS price transparency requirements, manufacturers must report WAC changes to commercial drug pricing compendia within 30 days of the effective date. Failure to report timely can result in exclusion from federal healthcare programs. The three major compendia—Medi-Span (Wolters Kluwer), First DataBank, and Elsevier Gold Standard—each publish WAC based on manufacturer submissions. Manufacturers may set different WAC levels for different package sizes, dosage forms, and National Drug Codes (NDCs). Each unique 11-digit NDC carries its own WAC. When a new formulation or package size is introduced, the manufacturer must submit WAC for the new NDC within the required reporting window. PBMs and payers typically lock WAC at the point of dispensing using compendia data pulled daily or weekly. Discrepancies between compendia can create pricing inconsistencies in adjudication systems, which may trigger anomaly detection flags.
""".strip()})

        docs.append({"doc_id": "WAC-004", "title": "Annual WAC Price Increase Trends and State Notification Laws", "topic": "WAC Pricing Policy", "content": """
Annual WAC price increases for brand-name drugs averaged 9.1% per year from 2008 to 2020, far exceeding the Consumer Price Index (CPI). Specialty biologics saw even larger increases, with some exceeding 15% annually. In response, multiple states enacted price increase notification laws. California SB 17 (2017) requires manufacturers to provide 60-day advance notice for increases exceeding 16% over a two-year period. Oregon requires 60-day notice for increases of 10% or more per year. Nevada requires 45-day advance notice for any increase on essential diabetes medications. The federal Inflation Reduction Act (IRA) of 2022 introduced Medicare inflation rebates: manufacturers must pay CMS rebates when a drug's price increases faster than the CPI-U. For drugs covered under Medicare Part B, the IRA inflation rebate is calculated quarterly. Manufacturers that exceed inflation-based price increase caps face per-unit rebates payable to the federal government, creating new financial incentives to moderate WAC increases.
""".strip()})

        docs.append({"doc_id": "WAC-005", "title": "WAC in PBM Contracting and MAC Pricing", "topic": "WAC Pricing Policy", "content": """
In PBM (Pharmacy Benefit Manager) contracts, WAC serves as the reference benchmark for establishing pharmacy reimbursement rates. Retail pharmacy reimbursement is commonly expressed as AWP minus a percentage (e.g., AWP - 16%) or WAC plus a percentage (e.g., WAC + 1%). Maximum Allowable Cost (MAC) pricing for generic drugs is set by PBMs independently and may bear no relationship to WAC, often being far below it. For brand drugs, WAC-based reimbursement is standard. Specialty drugs are frequently reimbursed at WAC + 0% to WAC + 8% depending on channel and competitive dynamics. In employer-sponsored health plans, WAC-plus contracts allow employers to audit pharmacy purchases against WAC to verify that pharmacies are not overbilling. Any actual_price exceeding WAC by more than 2.5x is considered a WAC breach anomaly and should trigger immediate investigation. Contractual clawback provisions typically allow PBMs to recoup overpayments identified within 18 months of the transaction date.
""".strip()})

        docs.append({"doc_id": "WAC-006", "title": "WAC and Medicaid Best Price Interaction", "topic": "WAC Pricing Policy", "content": """
Medicaid Best Price (BP) is defined under 42 U.S.C. § 1396r-8 as the lowest price a manufacturer charges any purchaser in the United States, with limited statutory exceptions (340B, VA, DoD, IHS, etc.). Best Price cannot exceed AMP, and it creates a floor below which manufacturers cannot effectively price without triggering additional Medicaid rebate liability. The Medicaid rebate for brand drugs is the greater of: (a) 23.1% of AMP, or (b) AMP minus Best Price. If a manufacturer offers deep discounts to certain commercial customers, those prices may become the new Best Price, dramatically increasing Medicaid rebate liability. WAC increases can raise AMP proportionally, which—if Best Price remains stable—can reduce the AMP-minus-Best-Price component of the rebate. Manufacturers sometimes manage WAC strategically to balance the WAC/AMP relationship and control Medicaid rebate exposure. Discrepancies between a manufacturer's reported Best Price and prices found in transaction data are a primary OIG audit target.
""".strip()})

        docs.append({"doc_id": "WAC-007", "title": "WAC vs Net Price: The Gross-to-Net Gap", "topic": "WAC Pricing Policy", "content": """
The gross-to-net (GTN) gap is the difference between a drug's WAC (list price) and the net price actually received by the manufacturer after all rebates, chargebacks, discounts, and allowances. For many specialty and biologic products, the GTN gap has grown dramatically: in 2022, the average GTN discount across all brand drugs exceeded 50%, meaning manufacturers received less than 50 cents per dollar of WAC. For some heavily rebated products in competitive therapeutic categories (e.g., insulins, GLP-1 receptor agonists, TNF inhibitors), GTN discounts can reach 70-80% of WAC. This creates a paradox where WAC increases may not translate to higher manufacturer revenue if PBMs extract corresponding rebate increases. The GTN bubble distorts price signals for patients who pay cost-sharing based on WAC or AWP rather than net price, particularly those in high-deductible health plans. Monitoring GTN price deviation is a key compliance activity: if GTN price falls below WAC × 0.45, it may indicate improper rebate stacking or data errors.
""".strip()})

        docs.append({"doc_id": "WAC-008", "title": "WAC Transparency Legislation: Federal and State Overview", "topic": "WAC Pricing Policy", "content": """
Drug pricing transparency legislation at both federal and state levels mandates disclosure of WAC and price increase information. At the federal level, the Drug Price Transparency in Communication Act (H.R. 1038) would require WAC disclosure in direct-to-consumer advertising. The American Rescue Plan Act (ARPA) of 2021 eliminated the Medicaid rebate cap of 100% of AMP, allowing rebates to exceed AMP. At the state level, as of 2024, 27 states have enacted some form of drug pricing transparency legislation. Maryland's Drug Price Transparency Act requires manufacturers to justify price increases above 50% over five years. Colorado's Prescription Drug Affordability Review Board (PDAB) can set upper payment limits for high-cost drugs. Ohio requires PBMs to register and disclose spread pricing. These laws create significant compliance obligations for manufacturers, PBMs, and pharmacy chains. Failure to comply can result in state-level fines ranging from $10,000 to $1,000,000 per violation, depending on jurisdiction.
""".strip()})

        docs.append({"doc_id": "WAC-009", "title": "WAC Data Sources and Compendia Access", "topic": "WAC Pricing Policy", "content": """
Three primary commercial compendia maintain WAC databases used in drug pricing adjudication: Medi-Span (Wolters Kluwer Health), First DataBank (FDB), and Elsevier Gold Standard (GSDD). Each compendium receives manufacturer price submissions and publishes updates on different schedules—some daily, some weekly. Subscription fees for full compendia access can range from $50,000 to $500,000+ annually for enterprise licensees such as PBMs, payers, and hospital systems. CMS publishes NADAC data publicly and free of charge at data.medicaid.gov. The FDA NDC directory provides product identifiers but not pricing. Pharmacy data vendors such as IQVIA (formerly IMS Health), Symphony Health, and MMIT aggregate transaction-level data that can be cross-referenced against WAC to identify pricing anomalies. Data discrepancies between compendia are common: up to 3% of NDCs may show different WAC values across publishers at any given time, creating adjudication errors that require reconciliation.
""".strip()})

        docs.append({"doc_id": "WAC-010", "title": "WAC-Based Anomaly Detection Thresholds", "topic": "WAC Pricing Policy", "content": """
Industry-standard thresholds for WAC-based anomaly detection have been developed through OIG guidance, PBM audit frameworks, and NCPDP standards. Key thresholds include: (1) actual_price > WAC × 2.5 constitutes a WAC breach anomaly, indicating overbilling or data error; (2) actual_price < AMP × 0.70 may indicate diversion or unreported price concessions; (3) ASP > WAC × 1.05 triggers an ASP-WAC violation, as ASP must not exceed WAC in practice given ASP is a post-rebate benchmark; (4) GTN price < WAC × 0.45 flags extreme discount levels that may indicate rebate stacking or misclassification; (5) month-over-month WAC changes exceeding 25% should be flagged for manufacturer notification verification. Statistical methods such as z-score analysis (threshold |z| > 3.0) and IQR fencing (Q1 − 2.5·IQR to Q3 + 2.5·IQR) are applied to detect distributional outliers in price time series. Ensemble methods combining statistical, ML-based (Isolation Forest, LOF), and rule-based detectors achieve higher precision and recall than any single method alone.
""".strip()})

        # ── Topic 2: NADAC Benchmarks ─────────────────────────────────

        docs.append({"doc_id": "NADAC-001", "title": "NADAC Methodology and CMS Survey Process", "topic": "NADAC Benchmarks", "content": """
The National Average Drug Acquisition Cost (NADAC) is published weekly by CMS (Centers for Medicare & Medicaid Services) based on a voluntary survey of retail community pharmacies. The survey, administered by Myers and Stauffer LC under contract with CMS, collects invoice-level data from a stratified sample of pharmacies representing different chain sizes, geographic regions, and pharmacy types. NADAC reflects the actual invoice price paid by pharmacies to wholesalers, net of prompt-pay discounts but gross of performance-based rebates and DIR fees (Direct and Indirect Remuneration). NADAC is calculated separately for brand and generic drugs and is updated every Wednesday. CMS publishes NADAC data publicly at data.medicaid.gov, making it one of the most transparent drug pricing benchmarks available. NADAC is used by many state Medicaid programs as a reimbursement benchmark, replacing the historically inflated AWP-based methodology. NADAC values typically fall between AMP and WAC—closer to AMP for generics (often within 2%) and slightly above AMP for brand drugs (typically AMP × 1.01-1.06).
""".strip()})

        docs.append({"doc_id": "NADAC-002", "title": "NADAC vs AWP: Drug Pricing Transparency Shift", "topic": "NADAC Benchmarks", "content": """
The transition from AWP-based to NADAC-based Medicaid reimbursement represents one of the most significant changes in pharmacy pricing policy over the past decade. AWP, often called "Ain't What's Paid," consistently overstated actual acquisition costs by 20-25% for brand drugs and by much larger margins for generics. NADAC, by contrast, reflects actual invoice prices reported by pharmacies. Under AWP-based Medicaid reimbursement (historically AWP - 12% to AWP - 15%), state Medicaid programs overpaid significantly for generic drugs where the spread between acquisition cost and AWP was often 50-90%. The shift to NADAC + a dispensing fee has reduced generic drug reimbursement substantially in states that have adopted it. However, NADAC does not account for DIR fees charged back to pharmacies by PBMs under Medicare Part D, which can reduce net pharmacy reimbursement by 2-8% post-point-of-sale. This means NADAC overstates net pharmacy revenue for Medicare Part D transactions.
""".strip()})

        docs.append({"doc_id": "NADAC-003", "title": "NADAC Deviations: Causes and Detection", "topic": "NADAC Benchmarks", "content": """
Significant deviations between a pharmacy's actual acquisition cost and the published NADAC can arise from several causes. (1) Wholesaler contract tiers: Large chain pharmacies negotiating direct manufacturer contracts may achieve acquisition costs 5-15% below NADAC for high-volume generics. Independent pharmacies typically pay closer to NADAC or above it. (2) Specialty drug pricing: Specialty products distributed through limited specialty pharmacy networks may have negotiated acquisition costs substantially different from NADAC, which is primarily surveyed from retail channels. (3) Drug shortages: During shortage periods, acquisition costs may spike above NADAC as secondary market suppliers charge premiums. (4) Survey lag: NADAC surveys occur over several weeks before publication; rapid market price changes may cause temporary NADAC-market gaps. (5)340B pricing: Covered entities purchasing at 340B ceiling prices should never use NADAC for 340B inventory—340B and non-340B inventory must be strictly separated. Pharmacies billing Medicaid at NADAC while acquiring product at 340B prices without proper disclosure may trigger duplicate discount violations.
""".strip()})

        docs.append({"doc_id": "NADAC-004", "title": "NADAC for Generic vs Brand Drug Reimbursement", "topic": "NADAC Benchmarks", "content": """
NADAC functions differently as a benchmark for generic versus brand drugs, reflecting fundamental differences in market structure. For generic drugs, NADAC is highly responsive to market dynamics—during periods of generic price erosion, NADAC can drop 20-30% within weeks as competition intensifies. The NADAC for a generic drug often falls to near the cost of raw active pharmaceutical ingredient (API) plus manufacturing margin for highly competitive markets. For brand drugs, NADAC typically tracks closely to WAC minus a small wholesaler discount (0.5-2%), as brand manufacturers maintain strict price discipline through distribution agreements. State Medicaid programs using NADAC-based reimbursement must handle the brand/generic differential carefully: generic NADAC can be volatile, requiring frequent MAC list updates. The CMS NADAC database separates brand and generic entries using the FDA's therapeutic equivalence codes. Pharmacies should monitor NADAC weekly for high-volume generics to ensure acquisition costs remain aligned with reimbursement benchmarks and avoid negative margin transactions.
""".strip()})

        docs.append({"doc_id": "NADAC-005", "title": "State Medicaid NADAC Adoption and Reimbursement Models", "topic": "NADAC Benchmarks", "content": """
As of 2024, over 35 state Medicaid programs have adopted NADAC as the primary benchmark for pharmacy reimbursement, replacing AWP-based models. The typical NADAC-based reimbursement formula is: NADAC + a dispensing fee (ranging from $8 to $15 per prescription depending on state). Some states apply NADAC × a markup percentage for certain drug categories. The CMS Medicaid pharmacy reimbursement final rule (2020) established that ingredient cost reimbursement must be based on acquisition cost or its equivalent—with NADAC being an approved proxy. States that have not adopted NADAC must demonstrate their alternative methodology produces equivalent results. Managed care organizations (MCOs) contracting with state Medicaid programs are required to ensure their pharmacy benefit managers reimburse pharmacies at NADAC-equivalent levels under the 2020 Medicaid rule. Non-compliance by MCOs can result in federal financial participation (FFP) disallowances. PBMs operating in NADAC states must maintain auditable records demonstrating NADAC-aligned reimbursement for each transaction.
""".strip()})

        docs.append({"doc_id": "NADAC-006", "title": "NADAC API Usage and Data Integration", "topic": "NADAC Benchmarks", "content": """
CMS provides NADAC data through the Medicaid.gov Data API, which allows programmatic access to current and historical NADAC values by NDC. The API returns NADAC per unit price, as_of_date, pricing_unit, and whether the drug is classified as brand or generic. Integration workflow: (1) Pull weekly NADAC file from data.medicaid.gov, (2) Join on 11-digit NDC to internal pricing database, (3) Flag NDCs where actual acquisition cost deviates from NADAC by more than 10% in either direction, (4) Escalate drugs with NADAC above WAC (possible data error), (5) Update MAC lists for generics where NADAC has declined more than 15% from previous week. NADAC historical data is available back to 2013, enabling long-term price trend analysis. Data quality checks: NADAC should generally be ≤ WAC; NADAC > WAC × 1.02 indicates a potential data error requiring verification with the manufacturer and compendium. NADAC should generally be ≥ AMP; NADAC < AMP × 0.95 for a brand drug warrants investigation.
""".strip()})

        docs.append({"doc_id": "NADAC-007", "title": "DIR Fees and Their Impact on NADAC Accuracy", "topic": "NADAC Benchmarks", "content": """
Direct and Indirect Remuneration (DIR) fees are post-point-of-sale adjustments charged back to pharmacies by PBMs under Medicare Part D contracts. DIR fees, which averaged $12.6 billion annually in recent years, significantly reduce pharmacy net revenue below NADAC. The CMS rule effective January 1, 2024, requires that all price concessions (including DIR fees) be applied at the point of sale rather than retroactively, fundamentally changing how DIR impacts NADAC alignment. Pre-2024, pharmacies could receive NADAC-based reimbursement at the point of sale but have their net revenue reduced retroactively by DIR fees totaling 2-8% of revenue. Post-2024 point-of-sale DIR, the benchmark price paid to pharmacies is lower upfront but stable. For anomaly detection purposes: pharmacies with high DIR fee exposure in pre-2024 periods may appear to have unusual margin patterns. NADAC-based comparisons for Medicare Part D transactions should account for the DIR timing difference. Any pharmacy reporting actual_price significantly above NADAC + DIR fee equivalent should be investigated for overbilling.
""".strip()})

        docs.append({"doc_id": "NADAC-008", "title": "NADAC During Drug Shortages: Price Spike Detection", "topic": "NADAC Benchmarks", "content": """
Drug shortages create significant distortions between published NADAC and actual pharmacy acquisition costs. When primary wholesalers run out of stock, pharmacies may purchase from secondary market distributors at prices 20-500% above NADAC. FDA's drug shortage database lists hundreds of products in shortage at any given time, with oncology injectable drugs and pediatric formulations being particularly vulnerable. During the 2022-2023 shortage of amoxicillin suspension and other pediatric antibiotics, secondary market prices reached 300-400% of NADAC. For anomaly detection systems: month-over-month actual price changes exceeding 25% on a shortage-listed NDC should be flagged as "shortage-related volatility" rather than fraud. Anomaly classification should incorporate FDA shortage data as a contextual signal to reduce false positive rates. Conversely, if a drug is not listed in FDA's shortage database but shows a sudden 25%+ price increase, that warrants immediate fraud investigation. Shortage pricing anomalies should be documented and reported to state pharmacy boards and PBM audit departments.
""".strip()})

        docs.append({"doc_id": "NADAC-009", "title": "NADAC Auditing and Pharmacy Compliance Requirements", "topic": "NADAC Benchmarks", "content": """
Pharmacies participating in Medicaid must maintain acquisition cost documentation sufficient to support NADAC-based reimbursement claims. CMS and state Medicaid agencies conduct post-payment audits comparing claimed ingredient costs against actual wholesaler invoices. Standard audit documentation requirements include: (1) Wholesaler invoices for the specific NDC and lot dispensed, (2) 340B purchase records if applicable (with strict separation from non-340B inventory), (3) Direct manufacturer purchase contracts with pricing tiers, (4) Proof of prompt-pay discounts applied. Pharmacies that cannot produce invoice-level documentation face recoupment of the difference between claimed NADAC and lowest documented acquisition cost. Multi-state pharmacy chains face audit coordination complexity when purchase contracts span multiple distribution centers. Automated compliance systems should reconcile every dispensing event against the corresponding purchase invoice within 30 days of dispensing. Failure to maintain adequate records can result in Medicaid provider agreement termination and exclusion from federal healthcare programs under 42 C.F.R. § 1002.210.
""".strip()})

        docs.append({"doc_id": "NADAC-010", "title": "NADAC Benchmark Alignment Best Practices for Specialty Pharmacies", "topic": "NADAC Benchmarks", "content": """
Specialty pharmacies face unique NADAC alignment challenges because NADAC is primarily surveyed from retail channels and may not reflect specialty acquisition costs. Key best practices: (1) Maintain separate cost-basis tracking for specialty products by therapeutic category and channel, (2) Apply a specialty acquisition cost premium of 2-5% above standard NADAC for limited distribution drugs where secondary market sourcing is required, (3) Document manufacturer hub services and patient support program costs separately from drug acquisition costs, (4) Monitor specialty NADAC weekly and flag any drug where your acquisition cost deviates more than 15% from published NADAC, (5) For cell and gene therapies with single-occurrence pricing, NADAC is not applicable—use WAC as the benchmark instead. Specialty pharmacies billing Medicaid managed care should negotiate NADAC-plus provisions that account for specialty acquisition cost differentials. Documentation of specialty acquisition costs should be maintained for a minimum of 10 years given extended Medicaid audit windows. Specialty pharmacies with 340B contract pharmacy agreements must maintain meticulous replenishment records to avoid duplicate discount allegations.
""".strip()})

        # ── Topic 3: 340B Program Rules ───────────────────────────────

        docs.append({"doc_id": "340B-001", "title": "340B Program Overview and Statutory Basis", "topic": "340B Program Rules", "content": """
The 340B Drug Pricing Program was established by Section 602 of the Veterans Health Care Act of 1992 (Public Law 102-585), codified at 42 U.S.C. § 256b. The program requires pharmaceutical manufacturers participating in Medicaid to offer covered outpatient drugs at or below the ceiling price to covered entities—a diverse group of safety-net healthcare providers. The program's stated purpose is to stretch scarce federal resources, enable covered entities to reach more eligible patients, and provide comprehensive services. HRSA (Health Resources & Services Administration) administers the program through its Office of Pharmacy Affairs (OPA). As of 2024, over 50,000 covered entity sites participate, with annual 340B drug purchases exceeding $54 billion. The program has grown significantly since its inception due to expanded covered entity eligibility, the shift from hospital outpatient to contract pharmacy dispensing, and rising drug prices that amplify the WAC-to-ceiling-price spread. Manufacturers participating in Medicaid (virtually all) must offer 340B ceiling prices; refusal or overcharging constitutes a program violation subject to civil monetary penalties of up to $5,000 per instance.
""".strip()})

        docs.append({"doc_id": "340B-002", "title": "340B Ceiling Price Calculation: AMP × 0.855", "topic": "340B Program Rules", "content": """
The 340B ceiling price is calculated using the formula: Ceiling Price = AMP − Unit Rebate Amount (URA). For innovator (brand) drugs, URA = AMP × 23.1% (the base Medicaid rebate). Therefore, Ceiling Price = AMP × (1 − 0.231) = AMP × 0.769 for drugs without additional inflation rebates. However, the commonly cited approximation of AMP × 0.855 reflects the blended average across drug types and years, as inflation rebates can reduce the ceiling further. The precise formula for any given drug and quarter is: Ceiling Price = AMP − URA, where URA = AMP × Base Rebate Percentage + inflation penalty rebate. For non-innovator (generic) drugs, the base rebate percentage is 13.02%, yielding a ceiling of approximately AMP × 0.8698. Manufacturers calculate and report 340B ceiling prices to HRSA quarterly. HRSA's 340B OPAIS (Office of Pharmacy Affairs Information System) database allows covered entities to verify ceiling prices. A 340B ceiling breach occurs when a covered entity's actual purchase price exceeds the calculated ceiling—this is a manufacturer compliance violation, not a covered entity violation, unless the covered entity submitted false data.
""".strip()})

        docs.append({"doc_id": "340B-003", "title": "Covered Entity Eligibility Requirements", "topic": "340B Program Rules", "content": """
Six categories of covered entities are eligible for the 340B program under 42 U.S.C. § 256b: (1) Federally Qualified Health Centers (FQHCs) and FQHC Look-Alikes, (2) Ryan White HIV/AIDS Program grantees, (3) Black Lung Clinics, (4) Comprehensive Hemophilia Diagnostic Treatment Centers, (5) Native Hawaiian Health Centers, (6) Urban Indian Organizations. Additionally, certain hospitals qualify if they meet specific criteria: Disproportionate Share Hospitals (DSH) with a DSH adjustment percentage > 11.75%, children's hospitals, critical access hospitals, rural referral centers, and sole community hospitals. To maintain eligibility, covered entities must: register annually with HRSA and maintain active OPAIS registration, operate within their four walls (or through approved contract pharmacy arrangements), not use 340B drugs for Medicaid fee-for-service patients (to avoid duplicate discounts), and maintain patient eligibility documentation. HRSA audits covered entities annually, with a focus on patient definition compliance, duplicate discount prevention, and drug diversion prevention.
""".strip()})

        docs.append({"doc_id": "340B-004", "title": "Contract Pharmacy Arrangements Under 340B", "topic": "340B Program Rules", "content": """
Contract pharmacy arrangements allow covered entities without on-site pharmacies to access 340B pricing through third-party pharmacy partners. HRSA issued guidance in 2010 permitting unlimited contract pharmacy relationships, leading to explosive program growth. By 2024, there were over 30,000 contract pharmacy arrangements. The arrangement requires: a written contract between the covered entity and pharmacy, a third-party administrator (TPA) to manage claims processing and 340B replenishment, and a mechanism to identify eligible patients and transactions. Key compliance requirements: (1) Only patients "of" the covered entity—those receiving services from the covered entity's providers—are eligible for 340B-priced drugs; (2) Separate inventory accumulation models (virtual inventory) or physical separation must prevent 340B drug from being dispensed to non-eligible patients; (3) Medicaid fee-for-service patients must be excluded to prevent duplicate discounts. Several large manufacturers (AstraZeneca, Eli Lilly, Novartis, etc.) began restricting 340B pricing to in-house pharmacies only in 2020-2021, triggering litigation that has been partially resolved in favor of HRSA's interpretation requiring manufacturers to honor all covered entity 340B purchases.
""".strip()})

        docs.append({"doc_id": "340B-005", "title": "340B Drug Diversion Prevention", "topic": "340B Program Rules", "content": """
Drug diversion in the 340B context occurs when 340B-priced drugs are dispensed to patients who are not eligible for the program, effectively converting the 340B discount to profit rather than patient benefit. Common diversion scenarios include: (1) Contract pharmacy dispensing 340B drugs to walk-in patients with no covered entity encounter, (2) 340B drugs appearing in distribution channels outside the covered entity network, (3) Duplicate inventory mixing 340B and WAC-priced product without proper allocation. HRSA's Office of Pharmacy Affairs conducts both announced and unannounced audits. The audit process examines patient definition records, pharmacy dispensing logs, and TPA reconciliation reports. Common audit findings include inadequate patient definition documentation (the patient must have received a service from a provider employed by or under contract with the covered entity), improper contract pharmacy utilization, and lack of written policies. Manufacturers may conduct their own 340B integrity audits under HRSA guidelines. Penalties for diversion include repayment of the difference between 340B purchase price and WAC, program termination, and referral to OIG.
""".strip()})

        docs.append({"doc_id": "340B-006", "title": "340B Ceiling Breach Detection and Remediation", "topic": "340B Program Rules", "content": """
A 340B ceiling breach occurs when a covered entity pays more than the calculated ceiling price for a covered outpatient drug. Covered entities should implement automated monitoring to compare actual purchase prices against HRSA-published ceiling prices quarterly. Detection methodology: (1) Pull 340B OPAIS ceiling prices for all purchased NDCs, (2) Compare against actual invoice prices from wholesaler or manufacturer invoices, (3) Flag any transaction where actual price > ceiling price + a tolerance of $0.01 per unit, (4) Aggregate findings by manufacturer and submit dispute to HRSA's 340B Prime Vendor Program. The 340B Prime Vendor (PVP), currently Apexus, assists covered entities in accessing ceiling prices and resolving overcharge disputes. Manufacturers must refund overcharges within 120 days of a verified dispute. Systematic overcharging by a manufacturer across multiple covered entities can trigger HRSA sanctions and OIG investigation. For anomaly detection systems: the rule price_340b > amp_price × 0.855 serves as a quick proxy—any 340B price exceeding this threshold should be flagged as a potential ceiling breach requiring verification against the actual HRSA ceiling calculation.
""".strip()})

        docs.append({"doc_id": "340B-007", "title": "Duplicate Discounts: 340B and Medicaid Overlap", "topic": "340B Program Rules", "content": """
Duplicate discounts arise when a drug purchased at 340B ceiling price is subsequently billed to Medicaid fee-for-service (FFS), triggering a Medicaid rebate from the manufacturer for the same unit. This effectively gives the government two discounts on the same drug—a prohibited practice under 42 U.S.C. § 256b(a)(5)(A). Medicaid managed care (MMC) organizations are generally not subject to the duplicate discount prohibition (confirmed by CMS in 2010), though some states have implemented carve-in and carve-out policies that affect this analysis. Prevention strategies: (1) Carved-out Medicaid: States carve 340B products out of Medicaid FFS billing—covered entities must track Medicaid FFS eligibility per transaction, (2) Carved-in Medicaid: 340B drugs can be billed to Medicaid MMC only, avoiding duplicate discounts, (3) TPA systems must maintain eligibility matrices for all state Medicaid programs in the covered entity's service area. HRSA audits specifically examine duplicate discount controls. Covered entities identified with duplicate discounts must repay CMS and potentially face program termination.
""".strip()})

        docs.append({"doc_id": "340B-008", "title": "IRA and 340B: Inflation Rebates Impact on Ceiling Prices", "topic": "340B Program Rules", "content": """
The Inflation Reduction Act (IRA) of 2022 modified 340B ceiling price calculations by enhancing the inflation penalty rebate component. Under the IRA, manufacturers whose drug prices increase faster than CPI-U must pay additional Medicaid inflation rebates, which are incorporated into the URA calculation. A higher URA directly lowers the 340B ceiling price: if AMP inflation rebates increase the URA, the ceiling price (AMP − URA) decreases correspondingly. For covered entities, this means that drugs subject to IRA inflation penalties may have 340B ceiling prices that are lower than the pre-IRA calculation. HRSA updates ceiling prices quarterly to reflect current AMP and URA values. Covered entities monitoring for ceiling breaches must use the most current HRSA pricing, not internally cached values from prior quarters. The IRA also established Medicare price negotiation for certain high-cost drugs; drugs subject to negotiated Maximum Fair Prices (MFPs) may need specific 340B ceiling price guidance, as the interaction between MFP and 340B ceiling calculations is still being clarified by HRSA as of 2024.
""".strip()})

        docs.append({"doc_id": "340B-009", "title": "HRSA Audits: Risk Factors and Preparation", "topic": "340B Program Rules", "content": """
HRSA conducts approximately 200 covered entity audits annually and publishes findings in annual reports. High-risk indicators that increase audit probability: (1) Large and growing 340B purchase volume relative to covered entity size, (2) Numerous contract pharmacy arrangements (> 10 locations), (3) Prior audit findings that were not fully remediated, (4) Complaints from manufacturers or patient advocacy groups, (5) Inconsistencies in OPAIS registration data. Audit preparation checklist: maintain a current written policies and procedures manual for 340B compliance; conduct quarterly self-audits of patient definition records; reconcile 340B purchase volumes against patient encounter data monthly; train all patient access and pharmacy staff on 340B eligibility criteria annually; maintain TPA contracts with audit rights provisions; document all changes to the covered entity's 340B eligible clinics and providers within 5 business days. HRSA audit findings are made public. "Major" findings (affecting program eligibility) can result in repayment demands and 30-day cure letters before program termination. Covered entities should engage 340B legal counsel immediately upon receiving an audit notification.
""".strip()})

        docs.append({"doc_id": "340B-010", "title": "340B Accumulator Adjustment Programs and Manufacturer Restrictions", "topic": "340B Program Rules", "content": """
Manufacturer restrictions on 340B contract pharmacy access represent the most significant 340B policy controversy of the 2020s. Starting in 2020, AstraZeneca, Eli Lilly, Novartis, Sanofi, and others announced policies limiting 340B pricing to in-house pharmacies only, or requiring covered entities to provide claims data as a condition of 340B access. HRSA issued Advisory Opinion 20-06 stating that such restrictions violate the statutory requirement that manufacturers "offer" 340B pricing to covered entities without conditions. Federal district courts have issued conflicting rulings; the D.C. Circuit in May 2023 held that HRSA's authority to sanction manufacturers for contract pharmacy restrictions was limited. Congress has not yet amended the statute to clarify. The practical impact: covered entities in restrictive manufacturer states must use wholesaler replenishment or establish direct purchase agreements; contract pharmacies using virtual inventory models face access disruptions. Covered entities should maintain contingency sourcing arrangements and monitor HRSA advisory opinions quarterly for policy updates.
""".strip()})

        # ── Topic 4: GTN & Rebate Structures ─────────────────────────

        docs.append({"doc_id": "GTN-001", "title": "Understanding the Gross-to-Net Bubble", "topic": "GTN & Rebate Structures", "content": """
The gross-to-net (GTN) bubble describes the growing divergence between pharmaceutical list prices (WAC) and the net prices manufacturers actually receive after all rebates, discounts, chargebacks, and allowances. The GTN gap has more than tripled since 2012: from approximately 25% in 2012 to over 50% of WAC for brand drugs in 2023, according to IQVIA analysis. For insulin products specifically, GTN discounts have reached 70-80% of WAC. The GTN gap creates significant problems: patients paying cost-sharing based on WAC-linked prices pay far more than the net price; the CBO and economists cannot accurately model the true cost of drugs using list prices; and manufacturers face pressure to keep raising WAC to maintain nominal revenue as rebates escalate. The GTN waterfall—the cascade of deductions from WAC to net revenue—typically includes: government-mandated rebates (Medicaid, AMP adjustments ~15-20% of WAC), commercial channel rebates (PBM formulary rebates ~20-30% of WAC), chargebacks and off-invoice allowances (~5-10% of WAC), patient assistance program costs (~2-5% of WAC), distribution fees (~2-3% of WAC), and returns/allowances (~1-2% of WAC).
""".strip()})

        docs.append({"doc_id": "GTN-002", "title": "Types of Pharmaceutical Rebates: Formulary, Market Share, Compliance", "topic": "GTN & Rebate Structures", "content": """
Pharmaceutical rebates fall into three primary categories with distinct compliance and reporting implications. (1) Formulary/tier placement rebates: Paid by manufacturers to PBMs to secure preferred or exclusive formulary status. A drug with Tier 2 (preferred brand) formulary placement may pay 20-35% of WAC in formulary rebates, while a Tier 1 (generic) equivalent pays nothing. Formulary rebates are negotiated confidentially and have been the subject of antitrust scrutiny. (2) Market share rebates: Also called performance or compliance rebates, these increase in value as a drug's market share within a therapeutic class exceeds specified thresholds (e.g., additional 5% rebate if market share exceeds 40%). Market share rebates can incentivize formulary exclusivity but may raise antitrust concerns when they effectively block competitor access. (3) Compliance rebates: Paid for achieving specific utilization management outcomes, such as step therapy compliance rates, prior authorization approval rates, or adherence metrics. Under the 2023 HHS rebate rule (subsequently blocked by litigation), these rebates were proposed to be treated as price concessions subject to Best Price inclusion. Understanding rebate types is critical for GTN analysis: different rebate categories have different AMP inclusion rules, Best Price implications, and 340B interactions.
""".strip()})

        docs.append({"doc_id": "GTN-003", "title": "GTN Discount Ranges by Drug Class", "topic": "GTN & Rebate Structures", "content": """
Gross-to-net discounts vary significantly by therapeutic class, reflecting competitive dynamics, formulary leverage, and payer concentration. Typical GTN ranges as of 2023: Diabetes (GLP-1 agonists, insulins): 60-75% GTN discount. These classes have multiple branded competitors competing intensely for formulary placement, driving record rebate levels. Cardiovascular (statins, anticoagulants): 40-60% GTN for branded products; generics have minimal GTN once market matures. Oncology: 15-35% GTN for novel checkpoint inhibitors with limited competition; can rise to 40-50% as class matures and biosimilars enter. Immunology (TNF inhibitors, IL-17 inhibitors): 50-65% GTN driven by intense formulary competition; biosimilar entry pushing brand GTN to 70%+ in some markets. Respiratory (ICS/LABA inhalers): 45-60% GTN for branded inhalers. Rare disease/orphan drugs: 10-25% GTN due to limited patient population, smaller payer leverage, and often no therapeutic alternatives. For anomaly detection: gtn_price < WAC × 0.45 may indicate rebate stacking error or data corruption; gtn_price > WAC × 0.80 for a heavily-rebated class may indicate underreporting of rebates.
""".strip()})

        docs.append({"doc_id": "GTN-004", "title": "Rebate Aggregators and PBM Pass-Through", "topic": "GTN & Rebate Structures", "content": """
PBMs (Pharmacy Benefit Managers) occupy the central role in the GTN ecosystem, negotiating rebates from manufacturers and distributing a portion to plan sponsors. Historically, the "spread" model allowed PBMs to retain a portion of rebates without disclosure to plan sponsors. The "pass-through" model (now required for Medicare Part D per CMS) contractually requires PBMs to remit 100% of negotiated rebates to the plan sponsor. Major PBMs—CVS Caremark, Express Scripts (Cigna), OptumRx (UnitedHealth)—each process hundreds of millions of prescriptions annually. Rebate aggregators such as Zinc Health Services and Ascent Health Services act as intermediaries that pool smaller PBMs' negotiating power. For plan sponsors seeking transparency: demand pass-through contracts with explicit rebate audit rights; require quarterly rebate reconciliation reports by drug and therapeutic class; audit reconciliation against manufacturer remittance data when possible. GTN visibility is improving as CMS and state regulators mandate rebate reporting: Medicare Part D Prescription Drug Plans (PDPs) must report per-drug rebate data to CMS annually under the Part D rebate reporting requirement.
""".strip()})

        docs.append({"doc_id": "GTN-005", "title": "IRA Impact on GTN Structures and Manufacturer Strategy", "topic": "GTN & Rebate Structures", "content": """
The Inflation Reduction Act of 2022 is reshaping pharmaceutical GTN structures in profound ways. Key IRA provisions affecting GTN: (1) Medicare Drug Price Negotiation: CMS negotiates Maximum Fair Prices (MFPs) for high-expenditure Medicare drugs; MFPs are expected to be 40-60% below WAC for the first 10 selected drugs (effective 2026). Manufacturers must offer MFPs to all Medicare beneficiaries, fundamentally changing the WAC-to-net relationship for negotiated drugs. (2) Inflation Rebates: Manufacturers pay rebates to CMS when prices increase faster than CPI-U; these inflation penalties reduce the effective WAC by 2-15% for drugs with above-CPI price increases. (3) Medicare Part D Redesign: The elimination of the coverage gap ("donut hole") and the introduction of a $2,000 annual OOP cap shifts cost to manufacturers through increased catastrophic coverage contributions, effectively increasing manufacturer GTN contributions by 3-7% for high-cost Part D drugs. (4) Rebate Reform: While the proposed safe harbor rule for rebates was not finalized, CMS continues to explore point-of-sale rebate requirements that would redirect rebate value to patients.
""".strip()})

        docs.append({"doc_id": "GTN-006", "title": "GTN Reporting Requirements for Manufacturers", "topic": "GTN & Rebate Structures", "content": """
Manufacturers face complex GTN reporting requirements across multiple regulatory programs. AMP Reporting: AMP must be calculated and reported to CMS monthly (within 30 days of month-end) and used for quarterly Medicaid rebate invoicing. AMP reflects the manufacturer's average net price to retail community pharmacies, including chargebacks and rebates to wholesalers but excluding certain items (prompt-pay discounts, nominal price sales, 340B sales). Best Price Reporting: Best Price must be reported quarterly and reflects the lowest price offered to any customer, with limited exceptions. Manufacturers must maintain systems to capture all price concessions across commercial, government, and specialty channels. Non-Medicaid Price Reporting: Certain states (e.g., Vermont, California) require additional GTN disclosure directly to state agencies. CMS Gross Price Data: Under the ACA § 1302, manufacturers in certain markets must report list and net price data. Manufacturer gross price data submitted to IQVIA and Symphony Health is used by researchers and payers for market analysis. Penalties for inaccurate AMP or Best Price reporting include False Claims Act liability, program exclusion, and civil monetary penalties of $100,000 per misrepresentation plus triple damages.
""".strip()})

        docs.append({"doc_id": "GTN-007", "title": "Rebate Stack Analysis for Specialty Drugs", "topic": "GTN & Rebate Structures", "content": """
Specialty drug rebate stacks are complex multi-layer structures that compound across various payer and channel agreements. A typical specialty drug rebate stack for a biologic in a competitive immunology class: Layer 1 — Medicaid statutory rebate: 23.1% of AMP (~17-18% of WAC). Layer 2 — Commercial formulary rebate: 25-35% of WAC paid to top-3 PBMs for Tier 2 placement. Layer 3 — Market share rebate escalator: Additional 5-10% of WAC if market share exceeds 35% threshold. Layer 4 — Pull-through/compliance rebates: 3-5% of WAC for achieving step therapy bypass rates. Layer 5 — GPO rebates: 2-4% of WAC for hospital group purchasing agreements. Total stacked discount: 55-70%+ of WAC, leaving net revenue of $0.30-$0.45 per dollar of WAC. GTN deviation anomaly detection threshold: if gtn_price for a specialty biologic falls below WAC × 0.25, it may indicate rebate stacking errors, double-counting of rebate layers, or intentional manipulation to deflate reportable net price. Manufacturers must ensure that rebate stacking does not create a Best Price below an appropriate floor.
""".strip()})

        docs.append({"doc_id": "GTN-008", "title": "Patient Assistance Programs and GTN Impact", "topic": "GTN & Rebate Structures", "content": """
Manufacturer patient assistance programs (PAPs) and copay assistance cards (copay coupons) have complex GTN implications. Copay coupons, offered to commercially-insured patients to reduce OOP costs, are classified by CMS as price concessions that may affect Best Price if structured incorrectly. The "bona fide service fee" safe harbor allows manufacturers to exclude certain patient assistance costs from Best Price calculations. AMP exclusion: Drugs provided through nominal price programs (price ≤ 10% of AMP) to certain safety net entities are excluded from AMP calculations. PAP drugs provided free to uninsured patients may also be excluded. Copay accumulator programs implemented by many PBMs strip copay card value from patients' deductible accumulators, effectively eliminating the copay assistance benefit mid-year. Copay maximizer programs redirect the full copay card benefit value to offset plan costs rather than patient OOP. These programs have been challenged in litigation by manufacturer groups. For GTN modeling: copay assistance costs should be captured in the GTN waterfall as a separate line item, typically ranging from 1-5% of WAC for specialty products with active copay programs.
""".strip()})

        docs.append({"doc_id": "GTN-009", "title": "Chargeback Mechanics and Their GTN Role", "topic": "GTN & Rebate Structures", "content": """
Chargebacks are manufacturer payments to wholesalers compensating for the difference between the wholesaler's contract price and the lower price negotiated with specific end-customers (hospitals, clinics, pharmacies under GPO agreements). Chargeback mechanics: (1) Manufacturer sets WAC as the wholesaler sell-in price, (2) End-customer has a contract price (e.g., GPO pricing at WAC − 15%), (3) Wholesaler sells to end-customer at the contract price, (4) Wholesaler submits a chargeback request to the manufacturer for the difference between WAC and contract price, (5) Manufacturer pays chargeback, reducing effective net revenue. Chargebacks are a significant GTN component—for hospital-distributed products, they may represent 15-25% of WAC. AMP inclusion: most chargebacks reduce AMP, as they represent price concessions to customers in the retail pharmacy chain. GPO chargebacks to hospital or clinic customers may be excluded from AMP depending on their classification. For anomaly detection: sudden changes in chargeback volumes or rates may indicate contract pricing errors, unauthorized price extensions, or diversion of discounted product into non-qualifying channels.
""".strip()})

        docs.append({"doc_id": "GTN-010", "title": "Gross-to-Net Forecasting and Variance Analysis", "topic": "GTN & Rebate Structures", "content": """
Pharmaceutical manufacturers maintain complex GTN forecasting models to project net revenue, guide pricing decisions, and manage Medicaid rebate accruals. A robust GTN model includes: (1) Channel-level volume forecasting (retail, specialty, hospital, government), (2) Rebate rate forecasting by program type and payer, (3) Chargeback accrual rates by customer segment, (4) Medicaid mix assumptions (what percentage of volume is Medicaid FFS vs. managed care), (5) Return/allowance reserves. GTN variance analysis compares forecast to actual GTN on a monthly basis. Key variance drivers: changes in channel mix (e.g., 340B volume higher than forecast), rebate rate escalation due to market share threshold triggers, unexpected Medicaid utilization increases, and new state Medicaid supplemental rebate agreements. GTN accrual errors can misstate manufacturer earnings: overaccrued rebates reduce reported net revenue; underaccrued rebates create large one-time charges when settled. For compliance teams: GTN reporting should align with AMP-reported values within 2-3% to detect potential AMP underreporting. Large discrepancies between internal GTN and AMP-derived net price warrant investigation.
""".strip()})

        # ── Topic 5: Anomaly Case Studies ────────────────────────────

        docs.append({"doc_id": "CASE-001", "title": "Case Study: Pharmacy Chain Overbilling Above WAC", "topic": "Anomaly Case Studies", "content": """
SCENARIO: A regional pharmacy chain (220 locations across 6 states) was identified by a PBM audit as systematically billing Medicaid managed care plans at prices exceeding WAC for 12 specialty oncology products over an 18-month period. DETECTION METHOD: The PBM's price verification system flagged 4,847 claims where actual_billed_price exceeded WAC × 1.15, triggering a WAC breach alert. Initial investigation used z-score analysis on the specialty claims subset (|z| > 3.5 on billed price vs. WAC ratio), confirmed by Isolation Forest ML model trained on 24 months of historical specialty claims. ROOT CAUSE: The pharmacy's price file update process had a mapping error that caused 12 specialty NDCs to be priced against AWP rather than WAC in their adjudication system, resulting in billing at AWP (≈ WAC × 1.20) instead of WAC. RESOLUTION: The chain identified and corrected 4,847 claims totaling $2.3M in overbilling. Full repayment was made to the PBM within 90 days. The pharmacy implemented automated daily price verification against compendium WAC data and a dual-control review process for specialty price file updates. LESSON: Price file integrity controls are essential; WAC should be verified against compendia data daily, and specialty products should have additional validation layers.
""".strip()})

        docs.append({"doc_id": "CASE-002", "title": "Case Study: 340B Ceiling Breach at Safety-Net Hospital", "topic": "Anomaly Case Studies", "content": """
SCENARIO: A large urban safety-net hospital (340B-qualified DSH) discovered through its annual self-audit that it had been overcharged by a specialty manufacturer on 3 high-cost oncology biologics over a 6-quarter period. DETECTION METHOD: The hospital's 340B TPA performed a retrospective price verification comparing actual invoice prices against HRSA OPAIS ceiling prices. Algorithm: for each 340B purchase, compare invoice_price_per_unit against ceil_price (AMP × (1 − URA%)) from HRSA's quarterly database. Flagged transactions: price_340b > AMP × 0.855 proxy threshold, then verified against exact OPAIS ceiling. 847 purchase orders totaling $1.2M in overcharges were identified across 3 NDCs. ROOT CAUSE: The manufacturer's 340B pricing system failed to incorporate the most recent AMP inflation rebate adjustments in the URA calculation, causing ceiling prices to be set too high for 6 consecutive quarters. RESOLUTION: Hospital submitted overcharge disputes through HRSA's 340B Prime Vendor dispute resolution process. Manufacturer acknowledged the error and issued credits totaling $1.2M within 90 days. LESSON: Covered entities must verify 340B ceiling prices quarterly against OPAIS, not only annually. AMP-adjusted ceilings can change meaningfully when inflation rebates are updated.
""".strip()})

        docs.append({"doc_id": "CASE-003", "title": "Case Study: GTN Manipulation Through Rebate Stacking", "topic": "Anomaly Case Studies", "content": """
SCENARIO: A mid-size specialty pharmaceutical manufacturer was investigated by OIG for potential AMP underreporting after a competitor filed a qui tam complaint alleging the manufacturer had structured complex multi-party rebate arrangements that reduced its reported AMP below actual market prices. DETECTION METHOD: OIG analysts compared the manufacturer's publicly reported AMP trajectory against IQVIA transaction-level price data and CMS Best Price reports. A gtn_deviation anomaly was identified: the manufacturer's net price as derived from AMP was approximately 28% below industry benchmarks for the same therapeutic class, and the AMP decline was not correlated with any WAC changes. The manufacturer's gtn_price was below WAC × 0.35, exceeding the 30% GTN deviation threshold. ROOT CAUSE: The manufacturer had structured "value-based" agreements with certain PBMs that included retrospective outcome-based rebates, some of which should have been included in AMP calculations but were instead classified as administrative service fees. RESOLUTION: After a 3-year investigation, the manufacturer paid $42M to resolve FCA allegations. AMP calculations were restated for 8 quarters. The manufacturer restructured its value-based agreement documentation to properly classify all payments as AMP-includable rebates.
""".strip()})

        docs.append({"doc_id": "CASE-004", "title": "Case Study: ASP Inflation for Medicare Part B Drug", "topic": "Anomaly Case Studies", "content": """
SCENARIO: A manufacturer of an injectable oncology product (administered in physician offices, billed under Medicare Part B) was flagged by a CMS ASP audit for reporting ASP values that consistently exceeded WAC × 1.05—a violation of the ASP methodology, which should yield values below WAC given market discounts. DETECTION METHOD: CMS's quarterly ASP validation system compared reported ASP against WAC data from compendia. The automated check flagged the product with an asp_wac_violation: asp_price = WAC × 1.07 for three consecutive quarters. Additional analysis using LOF (Local Outlier Factor) on the class of oncology injectables identified this NDC as a density-based outlier in the ASP/WAC ratio distribution. ROOT CAUSE: The manufacturer's ASP calculation excluded a significant volume of discounted product sold through a specialty distributor channel—these discounted sales should have been included in the ASP denominator, which would have reduced ASP below WAC. The exclusion was characterized as a "distribution fee" rather than a price concession. RESOLUTION: CMS issued a correction notice requiring ASP restatement for 3 quarters. The manufacturer paid $8.7M in excess Medicare Part B reimbursement clawback. CMS updated the ASP calculation methodology guidance to clarify treatment of specialty distributor arrangements.
""".strip()})

        docs.append({"doc_id": "CASE-005", "title": "Case Study: Month-over-Month Price Spike During Drug Shortage", "topic": "Anomaly Case Studies", "content": """
SCENARIO: A multi-state pharmacy cooperative identified a 340% price spike for preservative-free methotrexate injection (25 mg/mL) across 15 member pharmacies over a 3-week period during a documented FDA drug shortage. DETECTION METHOD: The cooperative's pricing surveillance system flagged the product using MoM price change monitoring: actual_price increased from NADAC baseline of $12.40 to $54.80 per vial—a 342% increase—triggering a mom_spike alert (threshold: >25% MoM increase). Cross-referencing against FDA's shortage database confirmed an active shortage declaration. Isolation Forest model also flagged these transactions as outliers with IF anomaly score = 0.91. ROOT CAUSE: The cooperative's pharmacies were sourcing product from secondary market distributors (not primary wholesalers) during the shortage period. Secondary market pricing reflected genuine supply scarcity rather than fraud. RESOLUTION: The cooperative disclosed the shortage-related cost increase to its MCO payer, which granted a temporary dispensing fee supplement of $25/vial for shortage-period transactions. Anomaly detection system was updated to cross-reference FDA shortage data as a contextual modifier: shortage-confirmed drugs receive a higher MoM spike threshold (75% vs. 25%) to reduce false positives during genuine shortage events.
""".strip()})

        docs.append({"doc_id": "CASE-006", "title": "Case Study: Medicaid Best Price Misreporting Discovery", "topic": "Anomaly Case Studies", "content": """
SCENARIO: A manufacturer's internal compliance team discovered during a routine AMP reconciliation that its Best Price for a cardiovascular brand drug had been materially understated for 6 quarters due to a failure to capture a significant commercial customer's net price following a mid-contract amendment. DETECTION METHOD: Internal audit compared quarter-end Best Price reports against a comprehensive customer price file. The gtn_deviation flag was triggered when a specific managed care customer's net price (WAC × 0.48) was found to be lower than the reported Best Price (WAC × 0.62) for the same period. The 27% gap between actual net and reported Best Price exceeded the 30% deviation threshold. ROOT CAUSE: A large health system had negotiated an amendment to its GPO agreement mid-quarter that reduced the manufacturer's net price, but the price database feed to the AMP/Best Price calculation system was not updated to reflect the amendment. The price amendment was documented only in a contract addendum not linked to the price reporting workflow. RESOLUTION: The manufacturer self-disclosed to CMS, restated Best Price for 6 quarters, and paid approximately $31M in additional Medicaid rebates. Internal controls were enhanced with automated contract-to-price-database reconciliation and quarterly Best Price reasonableness testing.
""".strip()})

        docs.append({"doc_id": "CASE-007", "title": "Case Study: Volume Outlier Indicating Data Duplication", "topic": "Anomaly Case Studies", "content": """
SCENARIO: A national PBM identified a volume outlier anomaly for a single specialty pharmacy location dispensing Humira (adalimumab) biosimilar: the pharmacy reported dispensing 847 units in a single month versus a monthly average of 42 units (>4 standard deviations above mean). DETECTION METHOD: Volume outlier detection algorithm: volume_units > drug_mean + 4 × drug_std, flagged as "volume_outlier" anomaly. The LOF model independently confirmed the transaction cluster as a density outlier (LOF factor = 8.4). The pharmacy's total monthly specialty drug volume also appeared anomalous relative to its historical dispensing patterns. ROOT CAUSE: Investigation revealed that the pharmacy had experienced a dispensing system data migration error that duplicated 805 claims from a prior month. The duplicate claims had passed initial adjudication checks because each duplicated claim had a unique claim number generated by the new system, but underlying member IDs, drug NDCs, and dates of service were repeated. RESOLUTION: PBM reversed 805 duplicate claims totaling $1.7M in erroneous payments. The pharmacy implemented mandatory duplicate claim detection logic requiring member ID + NDC + DOS + quantity to be unique within a 30-day rolling window. The PBM tightened its automated duplicate detection rules to catch cross-system migration duplicates.
""".strip()})

        docs.append({"doc_id": "CASE-008", "title": "Case Study: Independent Pharmacy Margin Abuse", "topic": "Anomaly Case Studies", "content": """
SCENARIO: A state Medicaid program identified a pattern of margin_break anomalies at 14 independent pharmacy locations operated by a single owner in a rural state. The pharmacies systematically billed Medicaid for brand drugs at NADAC-equivalent prices while purchasing through a related-party wholesaler at inflated acquisition costs, creating negative margins that were subsidized through fraudulent management fee payments. DETECTION METHOD: Statistical margin analysis (margin_percent < −15% threshold for > 30% of transactions at each location) triggered margin_break flags across all 14 locations. Isolation Forest applied to pharmacy-level margin distribution identified all 14 locations as outliers (IF contamination score > 0.85). Cross-referencing with wholesaler invoice data from the related-party distributor revealed purchase prices 35-60% above market WAC. ROOT CAUSE: The scheme involved artificially inflating the cost of drugs purchased from the related-party wholesaler, reducing pharmacy apparent margin, and then paying kickback-structured "management fees" back to the pharmacy owner through a shell entity—effectively laundering the spread between NADAC reimbursement and actual low-cost drug purchases. RESOLUTION: OIG investigation resulted in exclusion of all 14 pharmacies and criminal charges against the owner. $4.2M in fraudulent Medicaid payments were recovered through civil settlement.
""".strip()})

        docs.append({"doc_id": "CASE-009", "title": "Case Study: WAC Price Cascade Error in PBM System", "topic": "Anomaly Case Studies", "content": """
SCENARIO: A major PBM discovered through its monthly pricing quality review that a compendia data integration failure had caused 847 NDCs to receive WAC updates from an incorrect compendia version, resulting in WAC values inflated by a uniform $24.50 per unit for all affected drugs over a 72-hour period. DETECTION METHOD: The PBM's WAC integrity monitoring system runs daily z-score analysis on WAC change percentage across all NDCs. Day-over-day WAC changes exceeding 50% for more than 50 NDCs simultaneously triggered a system-wide WAC cascade alert. IQR fence analysis on the WAC change distribution showed the affected NDCs forming a distinct cluster at the upper fence (Q3 + 2.5 × IQR). ROOT CAUSE: The compendia data vendor published an out-of-band test file to the production FTP directory during a system validation exercise. The PBM's automated ingestion process picked up the test file and applied it to production pricing tables before the data quality validation step could flag the anomaly. RESOLUTION: All 847 NDCs were reverted to correct WAC within 4 hours of detection. Claims adjudicated during the 72-hour window with incorrect WAC were identified (41,000 claims) and retroactively reprocessed where the WAC error caused overpayment. Total overpayment recovery: $1.1M. The vendor implemented strict file naming conventions and digital signatures to prevent test files from being ingested by production systems.
""".strip()})

        docs.append({"doc_id": "CASE-010", "title": "Case Study: Regulatory Rule Ensemble Detection Success", "topic": "Anomaly Case Studies", "content": """
SCENARIO: A hospital system's pharmacy compliance team deployed an ensemble anomaly detection system combining z-score, IQR, Isolation Forest, LOF, MoM change analysis, and regulatory rule checks. Within the first quarter, the system detected a cluster of transactions with three concurrent anomaly flags—a regulatory rule breach (340b_ceiling_breach), a volume outlier, and an IQR fence breach—on a single high-cost oncology biologic purchased through a contract pharmacy. DETECTION: The ensemble flagged these transactions with severity = "high" (3+ methods, plus regulatory rule). severity_score = 0.97. The detection_method field showed "IQR Fence | LOF | Regulatory Rules." ROOT CAUSE: Investigation revealed that a contract pharmacy was acquiring the oncology product at 340B prices but accidentally dispensing it to non-covered-entity patients due to a patient eligibility lookup failure in their TPA integration. The covered entity had also ordered an unusually large quantity (volume outlier) in anticipation of starting a new clinical protocol that was subsequently delayed. RESOLUTION: The contract pharmacy's eligibility lookup failure was corrected; 23 non-eligible dispensing events were identified and the manufacturer received a Best Price notification. The covered entity's over-purchase was returned to the wholesaler under restocking provisions. The case validated the ensemble approach: no single detection method would have identified the full scope of the compliance violation.
""".strip()})

        # ── Topic 6: Regulatory & Compliance ─────────────────────────

        docs.append({"doc_id": "REG-001", "title": "CMS Price Reporting Deadlines and Penalties", "topic": "Regulatory & Compliance", "content": """
Manufacturers participating in Medicaid must comply with strict CMS price reporting deadlines or face significant penalties. AMP Reporting: monthly AMP must be submitted to CMS via the Medicaid Drug Program (MDP) system within 30 days of the end of each calendar month (e.g., January AMP due by March 1). Quarterly AMP (used for Medicaid rebate invoicing) is due 30 days after the close of each calendar quarter. Best Price reporting follows the same quarterly timeline as AMP. Penalties for late or inaccurate reporting: failure to report timely subjects manufacturers to civil monetary penalties of $10,000 per day per drug during the delinquency period. Knowing misrepresentation of AMP or Best Price can trigger FCA liability with treble damages plus $5,500-$11,000 per false claim. CMS may exclude manufacturers from Medicaid participation for systematic non-compliance. ASP Reporting (Medicare Part B): manufacturers of Part B drugs must submit quarterly ASP data within 30 days of each quarter's close. CMS uses ASP to set Medicare reimbursement rates (ASP + 6% for most Part B drugs). ASP reporting errors have resulted in CMS retroactively adjusting Medicare reimbursement rates, requiring healthcare providers to refund overpayments.
""".strip()})

        docs.append({"doc_id": "REG-002", "title": "OIG Audit Risk Factors for Pharmaceutical Pricing", "topic": "Regulatory & Compliance", "content": """
The Office of Inspector General (OIG) of HHS issues an annual Work Plan and publishes advisory opinions that define high-risk areas for pharmaceutical pricing compliance. Top OIG audit risk factors: (1) Best Price reporting — particularly drugs with complex rebate structures or multiple customer tiers with different net prices; OIG has issued multiple advisory opinions on value-based arrangements and their Best Price implications. (2) AMP calculation — specifically the treatment of prompt-pay discounts, bundled sales, and nominal price transactions. (3) 340B ceiling price compliance — both manufacturer overcharging of covered entities and covered entity diversion/duplicate discounts. (4) WAC reporting accuracy — particularly rapid price increases without corresponding notification to state agencies. (5) ASP manipulation — excluding volume discounts from ASP calculations to inflate Medicare reimbursement. (6) Anti-kickback compliance in rebate arrangements — rebates structured as payments for formulary access that may exceed "bona fide" service fee safe harbor amounts. Risk mitigation: implement a comprehensive Pharmaceutical Pricing Compliance Program (PPCP) with designated compliance officer, annual training, written policies, internal audit schedule, and clear escalation procedures for identified discrepancies.
""".strip()})

        docs.append({"doc_id": "REG-003", "title": "State Medicaid Best Price Rules and Supplemental Rebates", "topic": "Regulatory & Compliance", "content": """
Beyond the federal Medicaid rebate program, many states negotiate supplemental rebate agreements with pharmaceutical manufacturers that further reduce net drug costs. These state-level supplemental rebates are negotiated separately from CMS and are not included in the federal AMP/Best Price framework. As of 2024, 44 states have active supplemental rebate programs, with aggregate supplemental rebates exceeding $8 billion annually. State-level Best Price rules: some states (notably California, Texas, and New York) have enacted laws requiring manufacturers to offer the state's Medicaid program the lowest price offered to any other payer within the state, going beyond the federal definition. California's Drug Pricing Transparency Act requires manufacturers to notify DHCS of all price concessions exceeding 15% offered to California commercial payers. Non-compliance with state supplemental rebate agreements can result in drug exclusion from state Medicaid formularies. Manufacturers must track and comply with both federal and state price reporting requirements simultaneously, creating significant data management complexity. State Medicaid supplemental rebate agreements typically include audit rights provisions allowing states to verify that the offered supplemental rebate represents the manufacturer's best commercial offer.
""".strip()})

        docs.append({"doc_id": "REG-004", "title": "FTC Pharmaceutical Pricing Guidelines and Antitrust Enforcement", "topic": "Regulatory & Compliance", "content": """
The Federal Trade Commission (FTC) has significantly intensified pharmaceutical pricing antitrust enforcement since 2021 under the leadership of Chair Lina Khan. Key FTC enforcement areas relevant to drug pricing: (1) Pay-for-delay settlements: "reverse payment" agreements where brand manufacturers pay generic competitors to delay market entry. The Supreme Court's FTC v. Actavis (2013) decision confirmed that such arrangements are subject to antitrust scrutiny; FTC has challenged numerous such settlements. (2) Product hopping: manufacturers making minor formulation changes (e.g., tablet to capsule) to extend market exclusivity and impede generic substitution. (3) Formulary exclusion rebates: FTC has investigated whether PBM-manufacturer rebate arrangements that require formulary exclusion of competitors violate antitrust law. (4) PBM vertical integration: FTC's 2024 report on PBM practices highlighted concerns about spread pricing, rebate opacity, and conflicts of interest in vertically integrated PBM-insurer-pharmacy entities. (5) Drug pricing cartel behavior: price-fixing among generic manufacturers resulted in criminal charges and hundreds of millions in fines starting in 2019 (United States v. Glazer). Compliance teams should conduct annual antitrust risk assessments of pricing strategies, rebate structures, and distribution agreements.
""".strip()})

        docs.append({"doc_id": "REG-005", "title": "Whistleblower Cases: Fictional But Realistic Pharma Pricing Settlements", "topic": "Regulatory & Compliance", "content": """
Case A — "Cascade Biopharmaceuticals AMP Settlement" (fictional): A former pricing analyst filed a qui tam complaint alleging that Cascade Bio had excluded approximately $340M in annual managed care rebates from AMP calculations over a 6-year period by reclassifying them as "outcomes-based service fees." OIG investigation confirmed the misclassification. Settlement: $215M (treble damages on $71.7M in underpaid Medicaid rebates). The relator received $36M (17% of recovery) as whistleblower share. Case B — "Meridian Pharmacy Chain WAC Overbilling" (fictional): A former compliance officer reported that Meridian was billing Medicare Part B for oncology injectables at AWP rather than ASP+6%, exploiting a legacy adjudication system bug. The overbilling amounted to $128M over 4 years. Settlement: $95M plus corporate integrity agreement (CIA) with 5-year OIG oversight. Case C — "Synergy Specialty 340B Diversion" (fictional): Former pharmacy technician reported that Synergy was dispensing 340B-priced product to cash-pay patients unaffiliated with any covered entity, capturing $18M in 340B margin. Settlement: $27M repayment plus exclusion of 6 pharmacy locations. LESSON: Whistleblower incentives under the FCA (15-30% of government recovery) create powerful incentives for insiders to report pricing irregularities.
""".strip()})

        docs.append({"doc_id": "REG-006", "title": "False Claims Act Liability in Drug Pricing", "topic": "Regulatory & Compliance", "content": """
The False Claims Act (31 U.S.C. §§ 3729-3733) is the primary federal statute used to pursue pharmaceutical pricing fraud. The FCA imposes liability for: knowingly presenting a false claim to the government, knowingly making a false statement material to a false claim, or conspiring to commit such acts. "Knowingly" includes deliberate ignorance or reckless disregard. In drug pricing, FCA claims arise from: (1) Inflated AMP submissions that understate Medicaid rebate obligations, (2) Best Price underreporting that reduces Medicaid rebates, (3) WAC overreporting to inflate ASP Medicare payments, (4) 340B claims where product was purchased outside eligible channels, (5) False pharmacy invoices to PBMs or Medicaid programs. Penalties: $13,946-$27,894 per false claim (2024 figures, adjusted annually) plus treble damages. Relators (qui tam whistleblowers) may receive 15-30% of government recovery. Statute of limitations: 6 years from violation, or 3 years from when government knew or should have known, whichever is later (maximum 10 years). Voluntary self-disclosure to OIG under the Provider Self-Disclosure Protocol may significantly reduce penalty exposure.
""".strip()})

        docs.append({"doc_id": "REG-007", "title": "Anti-Kickback Statute and Pharmaceutical Rebates", "topic": "Regulatory & Compliance", "content": """
The Anti-Kickback Statute (AKS), 42 U.S.C. § 1320a-7b(b), prohibits offering, paying, soliciting, or receiving anything of value to induce or reward referrals of items or services covered by federal health care programs. Pharmaceutical rebates potentially implicate the AKS because they may constitute remuneration paid to PBMs and plan sponsors to induce formulary placement decisions that affect which drugs are reimbursed by Medicare and Medicaid. The safe harbor regulation at 42 C.F.R. § 1001.952(h) protects "discounts" from AKS liability if they are: (1) Properly disclosed and reflected in the cost claimed by the beneficiary to the federal program, and (2) The discount is earned based on purchases of the same good. The "bona fide service fee" safe harbor (42 C.F.R. § 1001.952(d)) protects payments for legitimate administrative services from PBMs, provided fees are fair market value and not tied to volume. In 2020, HHS published a final rule removing the safe harbor for rebates paid by manufacturers to plan sponsors or PBMs under Medicare Part D, but courts enjoined enforcement. The interaction between rebate reform, AKS, and the IRA continues to evolve. Compliance recommendation: document the legitimate service value received for each rebate payment and ensure no conditioned formulary coverage arrangement exists.
""".strip()})

        docs.append({"doc_id": "REG-008", "title": "IRA Inflation Penalties: Calculation and Compliance", "topic": "Regulatory & Compliance", "content": """
The Inflation Reduction Act's Medicare Drug Inflation Rebate Program requires manufacturers to pay rebates to CMS when the price of a Part B or Part D drug increases faster than the Consumer Price Index for All Urban Consumers (CPI-U). Calculation methodology for Part B drugs: Inflation rebate per unit = (ASP − Benchmark ASP) × units reimbursed under Medicare. Benchmark ASP = 2021 ASP (reference period) × cumulative CPI-U adjustment since 2021. If current ASP > Benchmark ASP, the manufacturer pays the difference per unit to CMS quarterly. For Part D drugs, a similar calculation applies based on AMP versus 2021 baseline adjusted by CPI-U. Example: Drug A had AMP of $100 in Q3 2021. CPI-U has risen 18% cumulatively by Q3 2024. Benchmark AMP = $118. If current AMP = $145 (45% increase), inflation rebate = ($145 − $118) per unit × quarterly Part D units. Compliance requirements: manufacturers must maintain systems to calculate and pay inflation rebates quarterly, aligned with AMP reporting. The IRA exempts certain drug categories: biologics in the last year before biosimilar entry, drugs with CMS negotiated prices under the negotiation program, and certain other exclusions. Non-payment triggers additional civil monetary penalties equal to 125% of the rebate owed.
""".strip()})

        docs.append({"doc_id": "REG-009", "title": "Drug Pricing Transparency Laws: State-by-State Overview", "topic": "Regulatory & Compliance", "content": """
State-level drug pricing transparency laws have proliferated significantly since 2016, creating a complex compliance landscape for manufacturers, PBMs, and pharmacies operating across multiple states. Key state requirements as of 2024: CALIFORNIA (AB 2185, SB 1281): Requires 60-day advance notice for WAC increases ≥ 16% over 24 months, with a written justification letter to DHCS. Applies to drugs with WAC > $40 per month. VERMONT (H.476): Annual price reporting for drugs with WAC > $10,000 per year, including detailed GTN data and R&D cost justifications. OREGON (SB 1529): 60-day notice for ≥ 10% annual WAC increases; public reporting through the Drug Price Transparency Program. NEVADA (SB 539): Real-time price reporting for essential diabetes medications (insulins, test strips); allows legislative override of price increases deemed excessive. MARYLAND (HB 631): 45-day notice for drug price increases ≥ 50% over 5 years. NEW YORK (L.2020, ch. 19): Requires PBMs to register with DFS and report spread pricing practices. COLORADO (HB 22-1372): Established PDAB with authority to set upper payment limits. Manufacturers operating in multiple states should implement a centralized state transparency compliance calendar with automated notice generation triggered by WAC change thresholds.
""".strip()})

        docs.append({"doc_id": "REG-010", "title": "CMS Value-Based Arrangements and Pricing Compliance", "topic": "Regulatory & Compliance", "content": """
Value-based arrangements (VBAs), also called outcomes-based contracts or indication-specific pricing, link drug reimbursement to real-world clinical outcomes. These innovative payment models create significant pharmaceutical pricing compliance complexity. Types of VBAs: (1) Outcomes-based rebates: manufacturer pays additional rebate if the drug fails to achieve specified clinical outcomes (e.g., HbA1c reduction below target). (2) Indication-specific pricing: different net prices for the same drug based on the clinical indication treated. (3) Subscription models: payer pays a fixed annual fee for unlimited access to a drug (used for some hepatitis C and CAR-T therapies). Compliance challenges: (a) AMP/Best Price treatment: CMS issued guidance (2019 CMS VBP Guidance) permitting manufacturers to exclude certain VBA payments from AMP/Best Price if specific conditions are met, including reporting to CMS under a VBP Agreement. (b) Medicaid supplemental rebate interaction: VBA terms must be disclosed to state Medicaid programs to avoid double-counting. (c) FCA risk: VBAs structured as sham outcomes programs that function as undisclosed rebates may trigger FCA liability. (d) Anti-kickback: VBA payments must meet the service fee or discount safe harbor. CMS's VBP protocol (42 C.F.R. § 447.502) provides a formal pathway for manufacturers to seek AMP/Best Price exclusion for qualifying VBAs.
""".strip()})

        logger.info("Generated %d knowledge documents across 6 topics.", len(docs))
        return docs
