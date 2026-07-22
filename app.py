"""
Stock Intelligence Analyst — single-file cloud build.
Deploy on Streamlit Community Cloud. All secrets read via st.secrets.
"""

from __future__ import annotations
import io, json, re, time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pdfplumber

try:
    import anthropic
    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False

try:
    import yfinance as yf
    _HAS_YFINANCE = True
except ImportError:
    _HAS_YFINANCE = False

# =============================================================================
# 1. CONFIG — all secrets via st.secrets, never hardcoded
# =============================================================================
def secret(key: str, default: str = "") -> str:
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default

ANTHROPIC_API_KEY = secret("ANTHROPIC_API_KEY")
CLAUDE_MODEL = secret("CLAUDE_MODEL", "claude-sonnet-5")
CLAUDE_MAX_TOKENS = 4096
CLAUDE_TEMPERATURE = 0.2
PDF_CHUNK_CHAR_SIZE = 12000
PDF_CHUNK_OVERLAP = 500
MAX_CHUNKS_SENT_TO_CLAUDE = 8

MC_CONFIG = {
    "base_url": secret("MC_BASE_URL", "https://www.moneycontrol.com"),
    "api_base_url": secret("MC_API_BASE_URL", "https://api.moneycontrol.com"),
    "priceapi_base_url": secret("MC_PRICEAPI_BASE_URL", "https://priceapi.moneycontrol.com"),
    "cookie": secret("MC_COOKIE", ""),
    "x_auth_token": secret("MC_X_AUTH_TOKEN", ""),
    "user_agent": secret("MC_USER_AGENT",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"),
    "request_timeout": 15, "max_retries": 3, "retry_backoff_seconds": 1.5,
}

@dataclass
class Thresholds:
    min_revenue_cagr: float = 10.0
    min_eps_cagr: float = 10.0
    min_roe: float = 15.0
    min_roce: float = 15.0
    max_debt_to_equity: float = 1.0
    min_interest_coverage: float = 3.0
    net_margin_floor: float = 5.0

THRESHOLDS = Thresholds()

THEME = {"bg": "#0E1117", "panel": "#161B22", "border": "#262D38", "text": "#E6EDF3",
          "muted": "#8B949E", "green": "#26A69A", "red": "#EF5350", "amber": "#D29922",
          "accent": "#4C8DFF"}
GRADE_COLORS = {"EXCELLENT": "#26A69A", "GOOD": "#4C8DFF", "BAD/RISKY": "#D29922",
                 "LOSS-MAKING / DESTRUCTIVE": "#EF5350"}
GRADES_ORDER = ["LOSS-MAKING / DESTRUCTIVE", "BAD/RISKY", "GOOD", "EXCELLENT"]


# =============================================================================
# 2. PDF PARSER — multi-page extraction -> token-optimized, page-tagged chunks
# =============================================================================
@dataclass
class PageContent:
    page_number: int
    text: str
    tables: List[List[List[Optional[str]]]] = field(default_factory=list)

@dataclass
class DocChunk:
    chunk_id: int
    start_page: int
    end_page: int
    text: str
    table_count: int = 0

@dataclass
class ParsedDocument:
    filename: str
    total_pages: int
    pages: List[PageContent]
    chunks: List[DocChunk]

    def provenance_index(self) -> str:
        lines = [f"Document: {self.filename} ({self.total_pages} pages)"]
        for p in self.pages:
            tbl_note = f", {len(p.tables)} table(s)" if p.tables else ""
            preview = (p.text[:80] + "...") if len(p.text) > 80 else p.text
            lines.append(f"  Page {p.page_number}{tbl_note}: {preview.replace(chr(10), ' ').strip()}")
        return "\n".join(lines)


def extract_pages(file_bytes: bytes, filename: str) -> List[PageContent]:
    pages: List[PageContent] = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            try:
                tables = page.extract_tables() or []
            except Exception:
                tables = []
            pages.append(PageContent(page_number=i, text=text, tables=tables))
    if not any(p.text.strip() for p in pages):
        raise ValueError(f"Could not extract text from '{filename}' — it may be a scanned image PDF.")
    return pages


def _clean(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _table_md(table: List[List[Optional[str]]]) -> str:
    if not table:
        return ""
    rows = [[c if c is not None else "" for c in row] for row in table]
    header = rows[0]
    md = ["| " + " | ".join(str(c) for c in header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows[1:]:
        md.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(md)


def build_page_blocks(pages: List[PageContent]) -> List[str]:
    blocks = []
    for p in pages:
        block = f"\n[PAGE {p.page_number}]\n{_clean(p.text)}\n"
        for t_idx, table in enumerate(p.tables, start=1):
            md = _table_md(table)
            if md:
                block += f"\n[PAGE {p.page_number} - TABLE {t_idx}]\n{md}\n"
        blocks.append(block)
    return blocks


def chunk_document(pages: List[PageContent], chunk_size=PDF_CHUNK_CHAR_SIZE,
                    overlap=PDF_CHUNK_OVERLAP, max_chunks=MAX_CHUNKS_SENT_TO_CLAUDE) -> List[DocChunk]:
    blocks = build_page_blocks(pages)
    chunks: List[DocChunk] = []
    current_text, current_start, chunk_id = "", None, 0

    for page in pages:
        block = blocks[page.page_number - 1]
        if current_start is None:
            current_start = page.page_number
        if len(current_text) + len(block) > chunk_size and current_text:
            chunks.append(DocChunk(chunk_id, current_start, page.page_number - 1, current_text,
                                    current_text.count("- TABLE")))
            chunk_id += 1
            current_text = current_text[-overlap:] + block
            current_start = page.page_number
        else:
            current_text += block

    if current_text.strip():
        chunks.append(DocChunk(chunk_id, current_start, pages[-1].page_number, current_text,
                                current_text.count("- TABLE")))

    if len(chunks) <= max_chunks:
        return chunks
    table_chunks = [c for c in chunks if c.table_count > 0]
    other_chunks = [c for c in chunks if c.table_count == 0]
    slots = max(max_chunks - len(table_chunks), 0)
    sampled = other_chunks[::max(len(other_chunks) // slots, 1)][:slots] if slots and other_chunks else []
    return sorted(table_chunks + sampled, key=lambda c: c.start_page)[:max_chunks]


def parse_pdf(file_bytes: bytes, filename: str) -> ParsedDocument:
    pages = extract_pages(file_bytes, filename)
    return ParsedDocument(filename, len(pages), pages, chunk_document(pages))


# =============================================================================
# 3. CLAUDE ENGINE — all Anthropic API calls + strict JSON contracts
# =============================================================================
@dataclass
class StockAnalysis:
    company_name: str = "Unknown"
    grade: str = "GOOD"
    grade_rationale: str = ""
    fundamental_metrics: Dict[str, Any] = field(default_factory=dict)
    qualitative_flags: List[Dict[str, str]] = field(default_factory=list)
    cash_flow_flag: Optional[str] = None
    growth_trajectory: str = ""
    methodology: List[Dict[str, str]] = field(default_factory=list)
    provenance: List[Dict[str, str]] = field(default_factory=list)
    raw_model_output: str = ""


SYSTEM_PROMPT = """You are a rigorous equity research analyst operating a dual-engine review \
framework: (1) Financial Health & Compound Growth, and (2) Qualitative Risk & Corporate \
Governance. You extract only what is explicitly supported by the provided source text/data \
— you never invent numbers. When a metric cannot be computed, omit it and note it in \
`data_gaps`. You always respond with a single valid JSON object and nothing else (no \
markdown fences, no commentary). Every numeric claim must be traceable to a page number, \
table, or payload field — populate `provenance` accordingly."""

CHUNK_INSTRUCTIONS = """Analyze the SOURCE TEXT below (a chunk of a larger financial \
document, pages noted in [PAGE n] tags) and extract ONLY what is explicitly present. \
Return JSON with this schema:
{{
  "company_name": string or null,
  "fundamental_findings": {{
    "revenue": [{{"period": string, "value": string, "page": string}}],
    "eps": [{{"period": string, "value": string, "page": string}}],
    "gross_margin_pct": [{{"period": string, "value": number, "page": string}}],
    "operating_margin_pct": [{{"period": string, "value": number, "page": string}}],
    "net_margin_pct": [{{"period": string, "value": number, "page": string}}],
    "roe_pct": [{{"period": string, "value": number, "page": string}}],
    "roce_pct": [{{"period": string, "value": number, "page": string}}],
    "debt_to_equity": [{{"period": string, "value": number, "page": string}}],
    "interest_coverage": [{{"period": string, "value": number, "page": string}}],
    "operating_cash_flow": [{{"period": string, "value": string, "page": string}}],
    "capex": [{{"period": string, "value": string, "page": string}}],
    "free_cash_flow": [{{"period": string, "value": string, "page": string}}],
    "net_profit_reported": [{{"period": string, "value": string, "page": string}}]
  }},
  "governance_findings": {{
    "management_changes": [{{"description": string, "page": string}}],
    "board_changes": [{{"description": string, "page": string}}],
    "executive_comp_notes": [{{"description": string, "page": string}}],
    "accounting_policy_changes": [{{"description": string, "page": string}}],
    "related_party_transactions": [{{"description": string, "page": string}}],
    "auditor_qualifications": [{{"description": string, "page": string}}],
    "pending_litigation": [{{"description": string, "page": string}}],
    "institutional_holding_changes": [{{"description": string, "page": string}}],
    "promoter_pledge_changes": [{{"description": string, "page": string}}]
  }},
  "data_gaps": [string]
}}
Only include entries you can directly cite from the SOURCE TEXT. Empty list if none found.

SOURCE TEXT (pages {start_page}-{end_page}):
{chunk_text}
"""

SYNTHESIS_INSTRUCTIONS = """You are given per-chunk extraction results (JSON array) from \
one financial document, plus threshold benchmarks. Merge into ONE final structured \
verdict. Compute 3-5yr CAGR for revenue/EPS where possible (state formula and periods). \
Cross-check profitability against cash flow (flag if net profit is positive/growing while \
operating cash flow or FCF is negative). Weigh governance findings into the grade. Assign \
exactly one grade: "EXCELLENT", "GOOD", "BAD/RISKY", "LOSS-MAKING / DESTRUCTIVE".

Benchmarks: Revenue/EPS CAGR {min_revenue_cagr}%/{min_eps_cagr}% | ROE/ROCE {min_roe}%/{min_roce}% \
| Max D/E {max_de}x | Min Interest Coverage {min_ic}x | Net margin floor {net_margin_floor}%

Return ONLY this JSON schema:
{{
  "company_name": string, "grade": string, "grade_rationale": string,
  "fundamental_metrics": {{
     "revenue_cagr_pct": number or null, "revenue_cagr_period": string or null,
     "eps_cagr_pct": number or null, "eps_cagr_period": string or null,
     "latest_gross_margin_pct": number or null, "latest_operating_margin_pct": number or null,
     "latest_net_margin_pct": number or null, "latest_roe_pct": number or null,
     "latest_roce_pct": number or null, "latest_debt_to_equity": number or null,
     "latest_interest_coverage": number or null, "latest_free_cash_flow": string or null,
     "margin_trend_notes": string
  }},
  "cash_flow_flag": string or null,
  "qualitative_flags": [{{"category": string, "severity": "low"|"medium"|"high", "description": string, "page": string}}],
  "growth_trajectory": string,
  "methodology": [{{"metric": string, "formula": string}}],
  "provenance": [{{"fact": string, "source": string}}],
  "data_gaps": [string]
}}

PER-CHUNK EXTRACTIONS:
{chunk_json}
"""

SCAN_INSTRUCTIONS = """You are given a cleaned JSON payload of fundamental + market data \
for one stock from a Moneycontrol premium endpoint. Apply the identical dual-engine \
framework and benchmarks. Return the SAME JSON schema as document synthesis. For \
`provenance`, cite the payload field name instead of a page number.

Benchmarks: Revenue/EPS CAGR {min_revenue_cagr}%/{min_eps_cagr}% | ROE/ROCE {min_roe}%/{min_roce}% \
| Max D/E {max_de}x | Min Interest Coverage {min_ic}x | Net margin floor {net_margin_floor}%

PAYLOAD:
{payload_json}
"""


def _extract_json(raw: str) -> Dict[str, Any]:
    raw = raw.strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, re.DOTALL)
    if m:
        raw = m.group(1)
    else:
        a, b = raw.find("{"), raw.rfind("}")
        if a != -1 and b != -1:
            raw = raw[a:b + 1]
    return json.loads(raw)


def get_client():
    if not _HAS_ANTHROPIC:
        raise RuntimeError("The 'anthropic' package is not installed — check requirements.txt.")
    if not ANTHROPIC_API_KEY:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. In Streamlit Cloud go to your app -> Settings -> "
            "Secrets and add: ANTHROPIC_API_KEY = \"sk-ant-...\" (with quotes), then save and reboot the app."
        )
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def _friendly_claude_error(e: Exception) -> str:
    """Translate raw Anthropic SDK exceptions into a plain-English, actionable message."""
    if not _HAS_ANTHROPIC:
        return str(e)
    if isinstance(e, anthropic.AuthenticationError):
        return ("Anthropic rejected your API key (401 Authentication error). Double-check the "
                "ANTHROPIC_API_KEY value pasted into Streamlit Secrets — it should start with "
                "'sk-ant-' and have no extra spaces or quotes issues.")
    if isinstance(e, anthropic.PermissionDeniedError):
        return "Your API key does not have permission to use this model (403). Check your Anthropic Console plan/limits."
    if isinstance(e, anthropic.RateLimitError):
        return "Anthropic rate limit or usage cap hit (429). Wait a minute and try again, or check your usage limits in the Anthropic Console."
    if isinstance(e, anthropic.APIConnectionError):
        return "Could not reach the Anthropic API (network/connection error). This is usually transient — try again."
    if isinstance(e, anthropic.BadRequestError):
        return f"Anthropic rejected the request as malformed (400): {e}"
    if isinstance(e, anthropic.APIStatusError):
        return f"Anthropic API returned an error (HTTP {getattr(e, 'status_code', '?')}): {e}"
    return f"Unexpected error calling Claude: {e}"


def _call_claude(system: str, user: str, retries: int = 3) -> str:
    client = get_client()
    last_exc = None
    for attempt in range(retries):
        try:
            resp = client.messages.create(
                model=CLAUDE_MODEL, max_tokens=CLAUDE_MAX_TOKENS, temperature=CLAUDE_TEMPERATURE,
                system=system, messages=[{"role": "user", "content": user}],
            )
            return "\n".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        except Exception as e:
            last_exc = e
            # Don't waste retries on errors that will never succeed on retry
            if _HAS_ANTHROPIC and isinstance(e, (anthropic.AuthenticationError, anthropic.PermissionDeniedError,
                                                    anthropic.BadRequestError)):
                break
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(_friendly_claude_error(last_exc))


def test_claude_connection() -> Tuple[bool, str]:
    """Minimal 1-token round trip used by the sidebar diagnostic button."""
    try:
        client = get_client()
        client.messages.create(model=CLAUDE_MODEL, max_tokens=10,
                                 messages=[{"role": "user", "content": "Say OK"}])
        return True, f"Connected successfully using model '{CLAUDE_MODEL}'."
    except Exception as e:
        return False, _friendly_claude_error(e) if _HAS_ANTHROPIC else str(e)


def extract_from_chunk(chunk: DocChunk) -> Dict[str, Any]:
    prompt = CHUNK_INSTRUCTIONS.format(start_page=chunk.start_page, end_page=chunk.end_page,
                                        chunk_text=chunk.text)
    raw = _call_claude(SYSTEM_PROMPT, prompt)
    try:
        return _extract_json(raw)
    except json.JSONDecodeError:
        return {"company_name": None, "fundamental_findings": {}, "governance_findings": {},
                "data_gaps": [f"Parse failure pages {chunk.start_page}-{chunk.end_page}"], "_raw": raw}


def synthesize(chunk_extractions: List[Dict[str, Any]]) -> StockAnalysis:
    prompt = SYNTHESIS_INSTRUCTIONS.format(
        min_revenue_cagr=THRESHOLDS.min_revenue_cagr, min_eps_cagr=THRESHOLDS.min_eps_cagr,
        min_roe=THRESHOLDS.min_roe, min_roce=THRESHOLDS.min_roce,
        max_de=THRESHOLDS.max_debt_to_equity, min_ic=THRESHOLDS.min_interest_coverage,
        net_margin_floor=THRESHOLDS.net_margin_floor,
        chunk_json=json.dumps(chunk_extractions, indent=2)[:60000],
    )
    return _parse_analysis(_call_claude(SYSTEM_PROMPT, prompt))


def analyze_document_chunks(chunks: List[DocChunk], progress_callback=None) -> StockAnalysis:
    extractions = []
    for i, chunk in enumerate(chunks):
        result = extract_from_chunk(chunk)
        result["_pages"] = f"{chunk.start_page}-{chunk.end_page}"
        extractions.append(result)
        if progress_callback:
            progress_callback((i + 1) / len(chunks))
    return synthesize(extractions)


def analyze_scan_payload(payload: Dict[str, Any]) -> StockAnalysis:
    prompt = SCAN_INSTRUCTIONS.format(
        min_revenue_cagr=THRESHOLDS.min_revenue_cagr, min_eps_cagr=THRESHOLDS.min_eps_cagr,
        min_roe=THRESHOLDS.min_roe, min_roce=THRESHOLDS.min_roce,
        max_de=THRESHOLDS.max_debt_to_equity, min_ic=THRESHOLDS.min_interest_coverage,
        net_margin_floor=THRESHOLDS.net_margin_floor,
        payload_json=json.dumps(payload, indent=2)[:60000],
    )
    return _parse_analysis(_call_claude(SYSTEM_PROMPT, prompt))


def _parse_analysis(raw: str) -> StockAnalysis:
    try:
        data = _extract_json(raw)
    except json.JSONDecodeError:
        return StockAnalysis(company_name="Parse Error", grade="GOOD",
                              grade_rationale="Model output could not be parsed as JSON.",
                              raw_model_output=raw)
    return StockAnalysis(
        company_name=data.get("company_name") or "Unknown", grade=data.get("grade", "GOOD"),
        grade_rationale=data.get("grade_rationale", ""),
        fundamental_metrics=data.get("fundamental_metrics", {}),
        qualitative_flags=data.get("qualitative_flags", []), cash_flow_flag=data.get("cash_flow_flag"),
        growth_trajectory=data.get("growth_trajectory", ""), methodology=data.get("methodology", []),
        provenance=data.get("provenance", []), raw_model_output=raw,
    )


# =============================================================================
# 4. FUNDAMENTAL MATH — pure, auditable formulas
# =============================================================================
def series_cagr(values: List[Tuple[str, float]]) -> Optional[dict]:
    clean = [(p, v) for p, v in values if v is not None]
    if len(clean) < 2:
        return None
    periods = len(clean) - 1
    (b_label, b_val), (e_label, e_val) = clean[0], clean[-1]
    if b_val <= 0:
        return None
    rate = (((e_val / b_val) ** (1 / periods)) - 1) * 100
    return {"cagr_pct": round(rate, 2), "period_label": f"{b_label} → {e_label} ({periods}yr)",
            "formula": f"CAGR = (({e_val} / {b_val}) ^ (1/{periods})) - 1"}


def cash_profit_divergence_flag(net_profit: float, ocf: float) -> Optional[str]:
    if net_profit is None or ocf is None:
        return None
    if net_profit > 0 and ocf < 0:
        return (f"CRITICAL: Reported net profit of {net_profit:,.0f} is positive while operating "
                f"cash flow is negative ({ocf:,.0f}). Profit is not cash-backed.")
    if net_profit > 0 and 0 < ocf < 0.5 * net_profit:
        return f"CAUTION: Operating cash flow ({ocf:,.0f}) is under 50% of net profit ({net_profit:,.0f})."
    return None


METHODOLOGY_FORMULAS = [
    {"metric": "Revenue / EPS CAGR (3-5yr)", "formula": "((End / Begin) ^ (1/N years)) - 1"},
    {"metric": "Gross Margin", "formula": "Gross Profit / Revenue × 100"},
    {"metric": "Operating Margin", "formula": "EBIT / Revenue × 100"},
    {"metric": "Net Margin", "formula": "Net Profit / Revenue × 100"},
    {"metric": "ROE", "formula": "Net Income / Average Shareholders' Equity × 100"},
    {"metric": "ROCE", "formula": "EBIT / (Total Assets - Current Liabilities) × 100"},
    {"metric": "Debt-to-Equity", "formula": "Total Debt / Total Shareholders' Equity"},
    {"metric": "Interest Coverage", "formula": "EBIT / Interest Expense"},
    {"metric": "Free Cash Flow", "formula": "Operating Cash Flow - Capex"},
    {"metric": "Cash-Profit Divergence Flag", "formula": "Flag if Net Profit > 0 AND OCF < 0"},
]


# =============================================================================
# 5. GRADING GUARDRAIL — deterministic override of Claude's proposed grade
# =============================================================================
@dataclass
class GradeResult:
    grade: str
    color: str
    overridden: bool
    override_reason: Optional[str]
    model_rationale: str


def finalize_grade(analysis: StockAnalysis) -> GradeResult:
    model_grade = analysis.grade if analysis.grade in GRADES_ORDER else "GOOD"
    fm = analysis.fundamental_metrics or {}
    final_grade, override_reason = model_grade, None

    burning_cash = bool(analysis.cash_flow_flag) and "CRITICAL" in (analysis.cash_flow_flag or "")
    if burning_cash and model_grade in ("EXCELLENT", "GOOD"):
        final_grade = "LOSS-MAKING / DESTRUCTIVE"
        override_reason = "Downgraded: profit not cash-backed (positive profit, negative operating cash flow)."

    high_flags = sum(1 for f in analysis.qualitative_flags if f.get("severity") == "high")
    if high_flags >= 1 and final_grade == "EXCELLENT":
        final_grade = "GOOD" if high_flags == 1 else "BAD/RISKY"
        override_reason = (override_reason + " " if override_reason else "") + f"Capped due to {high_flags} high-severity governance flag(s)."
    if high_flags >= 2 and final_grade != "LOSS-MAKING / DESTRUCTIVE":
        final_grade = "BAD/RISKY"
        override_reason = (override_reason + " " if override_reason else "") + f"Multiple ({high_flags}) high-severity governance red flags."

    de = fm.get("latest_debt_to_equity")
    if de is not None and de > (THRESHOLDS.max_debt_to_equity * 2.5) and final_grade == "EXCELLENT":
        final_grade = "BAD/RISKY"
        override_reason = (override_reason + " " if override_reason else "") + f"D/E of {de}x exceeds 2.5x policy ceiling."

    return GradeResult(final_grade, GRADE_COLORS.get(final_grade, "#8B949E"),
                        final_grade != model_grade, override_reason, analysis.grade_rationale)


# =============================================================================
# 6. MONEYCONTROL CLIENT — session-driven wrapper w/ mock fallback
# =============================================================================
ENDPOINTS = {
    "stock_fundamentals": "{api_base}/mcapi/v1/stock/fundamentals",
    "shareholding": "{api_base}/mcapi/v1/stock/shareholding-pattern",
    "historical_ohlc": "{priceapi_base}/techCharts/indianMarket/stock/history",
    "screener_scan": "{api_base}/mcapi/v1/screener/scan",
}

class MoneycontrolClient:
    def __init__(self, cfg=None):
        self.cfg = cfg or MC_CONFIG
        self.session = requests.Session()

    def _headers(self):
        h = {"User-Agent": self.cfg.get("user_agent", ""), "Accept": "application/json, text/plain, */*",
             "Referer": self.cfg.get("base_url", "")}
        if self.cfg.get("cookie"):
            h["Cookie"] = self.cfg["cookie"]
        if self.cfg.get("x_auth_token"):
            h["X-Auth-Token"] = self.cfg["x_auth_token"]
            h["Authorization"] = f"Bearer {self.cfg['x_auth_token']}"
        return h

    def is_authenticated(self) -> bool:
        return bool(self.cfg.get("cookie") or self.cfg.get("x_auth_token"))

    def _request(self, url, params=None):
        """
        Returns (data, error_message). Never raises — Moneycontrol's endpoints are
        unofficial/placeholder paths (see module docstring), so any failure here is
        expected and must degrade gracefully rather than crash the app.
        """
        retries, backoff = self.cfg.get("max_retries", 3), self.cfg.get("retry_backoff_seconds", 1.5)
        last_err = None
        for attempt in range(retries):
            try:
                resp = self.session.get(url, headers=self._headers(), params=params,
                                         timeout=self.cfg.get("request_timeout", 15))
                if resp.status_code in (401, 403):
                    return None, (f"Moneycontrol rejected the session (HTTP {resp.status_code}). "
                                  "Your cookie/token likely expired or the endpoint requires different auth.")
                resp.raise_for_status()
                return resp.json(), None
            except Exception as e:
                last_err = str(e)
                time.sleep(backoff * (attempt + 1))
        return None, (f"Moneycontrol request failed after {retries} attempts ({last_err}). "
                       "This endpoint path is an unverified placeholder — Moneycontrol has no public API, "
                       "so this URL likely doesn't exist as written. See the scanner's data-source note.")

    def get_fundamentals(self, sc_id: str) -> Dict[str, Any]:
        if not self.is_authenticated():
            return _mock_fundamentals(sc_id)
        url = ENDPOINTS["stock_fundamentals"].format(api_base=self.cfg["api_base_url"])
        data, err = self._request(url, params={"scId": sc_id, "type": "consolidated"})
        if err or not data:
            result = _mock_fundamentals(sc_id)
            result["_source_error"] = err
            return result
        return {"sc_id": sc_id, "company_name": data.get("companyName", sc_id),
                "financials": data.get("financials", data), "_source": "moneycontrol_live"}

    def get_shareholding_pattern(self, sc_id: str) -> Dict[str, Any]:
        if not self.is_authenticated():
            return _mock_shareholding(sc_id)
        url = ENDPOINTS["shareholding"].format(api_base=self.cfg["api_base_url"])
        data, err = self._request(url, params={"scId": sc_id})
        return data if (data and not err) else _mock_shareholding(sc_id)

    def get_historical_ohlc(self, sc_id: str, resolution="D", from_ts=None, to_ts=None) -> pd.DataFrame:
        to_ts = to_ts or int(time.time())
        from_ts = from_ts or to_ts - 365 * 24 * 3600
        if not self.is_authenticated():
            return _mock_ohlc(sc_id, from_ts, to_ts)
        url = ENDPOINTS["historical_ohlc"].format(priceapi_base=self.cfg["priceapi_base_url"])
        raw, err = self._request(url, params={"symbol": sc_id, "resolution": resolution, "from": from_ts, "to": to_ts})
        if err or not raw or "t" not in raw:
            return _mock_ohlc(sc_id, from_ts, to_ts)
        df = pd.DataFrame({"date": pd.to_datetime(raw["t"], unit="s"), "open": raw.get("o", []),
                            "high": raw.get("h", []), "low": raw.get("l", []),
                            "close": raw.get("c", []), "volume": raw.get("v", [])})
        return df.sort_values("date").reset_index(drop=True)

    def scan_universe(self, filters: Dict[str, Any], limit=50) -> pd.DataFrame:
        if not self.is_authenticated():
            return _mock_scan(filters, limit)
        url = ENDPOINTS["screener_scan"].format(api_base=self.cfg["api_base_url"])
        data, err = self._request(url, params={**filters, "limit": limit})
        if err or not data:
            return _mock_scan(filters, limit)
        return pd.DataFrame(data.get("results", []))


def _mock_fundamentals(sc_id: str) -> Dict[str, Any]:
    return {"sc_id": sc_id, "company_name": f"{sc_id} (MOCK — connect Moneycontrol session for live data)",
            "financials": {
                "revenue": [{"period": "FY21", "value": "38000"}, {"period": "FY22", "value": "44000"},
                            {"period": "FY23", "value": "51000"}, {"period": "FY24", "value": "59500"},
                            {"period": "FY25", "value": "68000"}],
                "eps": [{"period": "FY21", "value": "42.1"}, {"period": "FY22", "value": "48.6"},
                        {"period": "FY23", "value": "55.3"}, {"period": "FY24", "value": "63.8"},
                        {"period": "FY25", "value": "72.4"}],
                "gross_margin_pct": 58.2, "operating_margin_pct": 24.1, "net_margin_pct": 17.6,
                "roe_pct": 22.4, "roce_pct": 26.1, "debt_to_equity": 0.31, "interest_coverage": 14.2,
                "operating_cash_flow": "12400", "capex": "3100", "free_cash_flow": "9300",
                "net_profit_reported": "11980"}, "_source": "mock"}


def _mock_shareholding(sc_id: str) -> Dict[str, Any]:
    return {"sc_id": sc_id,
            "promoter_holding_pct": [{"period": "Q1FY25", "value": 51.2}, {"period": "Q4FY25", "value": 50.8}],
            "promoter_pledge_pct": [{"period": "Q1FY25", "value": 0.0}, {"period": "Q4FY25", "value": 0.0}],
            "institutional_holding_pct": [{"period": "Q1FY25", "value": 32.4}, {"period": "Q4FY25", "value": 34.1}],
            "_source": "mock"}


def _mock_ohlc(sc_id: str, from_ts: int, to_ts: int) -> pd.DataFrame:
    days = pd.date_range(pd.to_datetime(from_ts, unit="s"), pd.to_datetime(to_ts, unit="s"), freq="B")
    n = len(days)
    rng = np.random.default_rng(abs(hash(sc_id)) % (2**32))
    base = 1000 + (abs(hash(sc_id)) % 2000)
    drift = rng.normal(0.0006, 0.017, n).cumsum()
    close = base * (1 + drift)
    open_ = close * (1 + rng.normal(0, 0.006, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.008, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.008, n)))
    volume = rng.integers(200000, 4000000, n)
    return pd.DataFrame({"date": days, "open": open_, "high": high, "low": low, "close": close, "volume": volume})


def _mock_scan(filters: Dict[str, Any], limit: int) -> pd.DataFrame:
    names = [("RIL", "Reliance Industries", "Energy"), ("TCS", "Tata Consultancy Services", "IT"),
             ("HDFCBANK", "HDFC Bank", "Banking"), ("INFY", "Infosys", "IT"),
             ("ASIANPAINT", "Asian Paints", "FMCG"), ("TITAN", "Titan Company", "Consumer"),
             ("DIVISLAB", "Divi's Laboratories", "Pharma"), ("PIIND", "PI Industries", "Chemicals"),
             ("BAJFINANCE", "Bajaj Finance", "NBFC"), ("LTIM", "LTIMindtree", "IT")]
    rng = np.random.default_rng(42)
    rows = [{"sc_id": s, "company_name": n, "sector": sec,
             "revenue_cagr_3y": round(rng.uniform(6, 28), 1), "eps_cagr_3y": round(rng.uniform(4, 32), 1),
             "roe_pct": round(rng.uniform(8, 34), 1), "roce_pct": round(rng.uniform(8, 34), 1),
             "debt_to_equity": round(rng.uniform(0.0, 1.8), 2), "net_margin_pct": round(rng.uniform(3, 26), 1),
             "cmp": round(rng.uniform(300, 4500), 1), "pe_ratio": round(rng.uniform(12, 65), 1)}
            for s, n, sec in names]
    df = pd.DataFrame(rows)
    if filters.get("min_roe"):
        df = df[df["roe_pct"] >= filters["min_roe"]]
    if filters.get("min_roce"):
        df = df[df["roce_pct"] >= filters["min_roce"]]
    if filters.get("max_de") is not None:
        df = df[df["debt_to_equity"] <= filters["max_de"]]
    if filters.get("min_revenue_cagr_3y"):
        df = df[df["revenue_cagr_3y"] >= filters["min_revenue_cagr_3y"]]
    return df.sort_values("roce_pct", ascending=False).head(limit).reset_index(drop=True)


# =============================================================================
# 6B. YAHOO FINANCE CLIENT — free, no-login, reliable default data source
# =============================================================================
# Moneycontrol has no public/documented API — the ENDPOINTS above are unverified
# placeholders that will not resolve to real data even with valid premium
# credentials. Yahoo Finance (via the `yfinance` package) is used as the
# default, no-auth-required data source for real OHLCV and headline
# fundamentals. Indian NSE tickers use a ".NS" suffix (e.g. "RELIANCE.NS").

NIFTY50_TICKERS = [
    ("RELIANCE.NS", "Reliance Industries", "Energy"), ("TCS.NS", "Tata Consultancy Services", "IT"),
    ("HDFCBANK.NS", "HDFC Bank", "Banking"), ("INFY.NS", "Infosys", "IT"),
    ("ICICIBANK.NS", "ICICI Bank", "Banking"), ("HINDUNILVR.NS", "Hindustan Unilever", "FMCG"),
    ("ITC.NS", "ITC", "FMCG"), ("LT.NS", "Larsen & Toubro", "Infrastructure"),
    ("BAJFINANCE.NS", "Bajaj Finance", "NBFC"), ("ASIANPAINT.NS", "Asian Paints", "FMCG"),
    ("MARUTI.NS", "Maruti Suzuki", "Auto"), ("TITAN.NS", "Titan Company", "Consumer"),
    ("SUNPHARMA.NS", "Sun Pharma", "Pharma"), ("DIVISLAB.NS", "Divi's Laboratories", "Pharma"),
    ("BHARTIARTL.NS", "Bharti Airtel", "Telecom"), ("AXISBANK.NS", "Axis Bank", "Banking"),
    ("KOTAKBANK.NS", "Kotak Mahindra Bank", "Banking"), ("HCLTECH.NS", "HCL Technologies", "IT"),
    ("WIPRO.NS", "Wipro", "IT"), ("NESTLEIND.NS", "Nestle India", "FMCG"),
]


def yf_get_ohlc(ticker: str, period: str = "1y") -> Tuple[pd.DataFrame, Optional[str]]:
    """Fetch real historical OHLCV data from Yahoo Finance. No auth required."""
    if not _HAS_YFINANCE:
        return pd.DataFrame(), "The 'yfinance' package is not installed — check requirements.txt."
    try:
        hist = yf.Ticker(ticker).history(period=period, auto_adjust=False)
        if hist is None or hist.empty:
            return pd.DataFrame(), f"Yahoo Finance returned no data for '{ticker}'. Check the ticker symbol (NSE tickers need a .NS suffix, e.g. RELIANCE.NS)."
        df = hist.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
        df.columns = ["date", "open", "high", "low", "close", "volume"]
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        return df, None
    except Exception as e:
        return pd.DataFrame(), f"Yahoo Finance request failed for '{ticker}': {e}"


def yf_get_fundamentals(ticker: str) -> Tuple[Dict[str, Any], Optional[str]]:
    """Fetch headline fundamentals from Yahoo Finance's ticker.info payload."""
    if not _HAS_YFINANCE:
        return {}, "The 'yfinance' package is not installed."
    try:
        info = yf.Ticker(ticker).info
        if not info or info.get("regularMarketPrice") is None and info.get("longName") is None:
            return {}, f"Yahoo Finance returned no fundamentals for '{ticker}'."
        financials = {
            "roe_pct": round((info.get("returnOnEquity") or 0) * 100, 2) if info.get("returnOnEquity") else None,
            "net_margin_pct": round((info.get("profitMargins") or 0) * 100, 2) if info.get("profitMargins") else None,
            "operating_margin_pct": round((info.get("operatingMargins") or 0) * 100, 2) if info.get("operatingMargins") else None,
            "gross_margin_pct": round((info.get("grossMargins") or 0) * 100, 2) if info.get("grossMargins") else None,
            "debt_to_equity": round((info.get("debtToEquity") or 0) / 100, 2) if info.get("debtToEquity") else None,
            "revenue_growth_pct": round((info.get("revenueGrowth") or 0) * 100, 2) if info.get("revenueGrowth") else None,
            "earnings_growth_pct": round((info.get("earningsGrowth") or 0) * 100, 2) if info.get("earningsGrowth") else None,
            "trailing_pe": info.get("trailingPE"),
            "market_cap": info.get("marketCap"),
        }
        return {"sc_id": ticker, "company_name": info.get("longName", ticker),
                "financials": financials, "_source": "yahoo_finance"}, None
    except Exception as e:
        return {}, f"Yahoo Finance fundamentals request failed for '{ticker}': {e}"


def yf_scan_universe(filters: Dict[str, Any], limit: int = 50,
                      tickers: List[Tuple[str, str, str]] = NIFTY50_TICKERS) -> Tuple[pd.DataFrame, List[str]]:
    """
    Real (non-mock) scan across a preset NSE ticker list using Yahoo Finance
    headline fundamentals. Slower than a real screener API (one request per
    ticker) so it's capped to a small curated universe by default.
    """
    rows, errors = [], []
    for sc_id, name, sector in tickers:
        fund, err = yf_get_fundamentals(sc_id)
        if err or not fund:
            errors.append(f"{sc_id}: {err}")
            continue
        f = fund["financials"]
        rows.append({
            "sc_id": sc_id, "company_name": name, "sector": sector,
            "revenue_cagr_3y": f.get("revenue_growth_pct"), "eps_cagr_3y": f.get("earnings_growth_pct"),
            "roe_pct": f.get("roe_pct"), "roce_pct": f.get("roe_pct"),  # yfinance has no direct ROCE; ROE used as proxy
            "debt_to_equity": f.get("debt_to_equity"), "net_margin_pct": f.get("net_margin_pct"),
            "pe_ratio": f.get("trailing_pe"),
        })
    if not rows:
        return pd.DataFrame(columns=["sc_id", "company_name", "sector", "revenue_cagr_3y", "eps_cagr_3y",
                                       "roe_pct", "roce_pct", "debt_to_equity", "net_margin_pct", "pe_ratio"]), errors
    df = pd.DataFrame(rows).dropna(subset=["roe_pct"], how="all")
    if df.empty:
        return df, errors
    if filters.get("min_roe"):
        df = df[df["roe_pct"].fillna(-1) >= filters["min_roe"]]
    if filters.get("max_de") is not None:
        df = df[df["debt_to_equity"].fillna(0) <= filters["max_de"]]
    if filters.get("min_revenue_cagr_3y"):
        df = df[df["revenue_cagr_3y"].fillna(-100) >= filters["min_revenue_cagr_3y"]]
    return df.sort_values("roe_pct", ascending=False, na_position="last").head(limit).reset_index(drop=True), errors


# =============================================================================
# 7. TECHNICAL ENGINE — candlestick patterns + volume breakout (rule-based)
# =============================================================================
@dataclass
class PatternSignal:
    date: pd.Timestamp
    pattern: str
    direction: str
    rule: str
    index: int

@dataclass
class TechnicalVerdict:
    signals: List[PatternSignal]
    breakout_flag: bool
    breakout_note: str
    trend_20d: str
    trend_50d: str
    volume_trend: str
    confirms_fundamentals: bool
    confirmation_note: str


def detect_patterns(df: pd.DataFrame, lookback=60) -> List[PatternSignal]:
    d = df.tail(lookback).reset_index(drop=True)
    signals = []
    body = (d["close"] - d["open"]).abs()
    rng = (d["high"] - d["low"]).replace(0, np.nan)

    for i in range(2, len(d)):
        o, h, l, c = d.loc[i, ["open", "high", "low", "close"]]
        po, ph, pl, pc = d.loc[i - 1, ["open", "high", "low", "close"]]
        upper_wick, lower_wick = h - max(o, c), min(o, c) - l
        b = body.iloc[i]
        r = rng.iloc[i] if not np.isnan(rng.iloc[i]) else 1e-9

        if r > 0 and lower_wick >= 2 * b and upper_wick <= 0.15 * r and b <= 0.35 * r:
            if d.loc[max(0, i - 5):i - 1, "close"].mean() > c:
                signals.append(PatternSignal(d.loc[i, "date"], "Hammer", "bullish",
                    "Lower wick ≥ 2× body, small upper wick, after a decline.", i))

        if r > 0 and upper_wick >= 2 * b and lower_wick <= 0.15 * r and b <= 0.35 * r:
            uptrend = d.loc[max(0, i - 5):i - 1, "close"].mean() < c
            direction, pattern = ("bearish", "Shooting Star") if uptrend else ("bullish", "Inverted Hammer")
            signals.append(PatternSignal(d.loc[i, "date"], pattern, direction,
                "Upper wick ≥ 2× body, small lower wick.", i))

        if pc < po and c > o and o <= pc and c >= po and b > body.iloc[i - 1]:
            signals.append(PatternSignal(d.loc[i, "date"], "Bullish Engulfing", "bullish",
                "Green body fully engulfs prior red body.", i))

        if pc > po and c < o and o >= pc and c <= po and b > body.iloc[i - 1]:
            signals.append(PatternSignal(d.loc[i, "date"], "Bearish Engulfing", "bearish",
                "Red body fully engulfs prior green body.", i))

        if r > 0 and b <= 0.08 * r:
            signals.append(PatternSignal(d.loc[i, "date"], "Doji", "neutral",
                "Open ≈ close (body ≤ 8% of range).", i))

        if i >= 2:
            o0, c0 = d.loc[i - 2, ["open", "close"]]
            o1, c1 = d.loc[i - 1, ["open", "close"]]
            b0, b1 = abs(c0 - o0), abs(c1 - o1)
            first_bear, mid_small = c0 < o0 and b0 > 0, (b1 <= 0.4 * b0 if b0 > 0 else False)
            if first_bear and mid_small and c > o and c > (o0 + c0) / 2:
                signals.append(PatternSignal(d.loc[i, "date"], "Morning Star", "bullish",
                    "Large bearish → small indecision → large bullish closing past midpoint.", i))
            first_bull = c0 > o0 and b0 > 0
            if first_bull and mid_small and c < o and c < (o0 + c0) / 2:
                signals.append(PatternSignal(d.loc[i, "date"], "Evening Star", "bearish",
                    "Large bullish → small indecision → large bearish closing past midpoint.", i))
    return signals


def detect_volume_breakout(df: pd.DataFrame, window=20, vol_multiple=1.5):
    if len(df) < window + 2:
        return False, "Insufficient history to evaluate breakout window."
    d = df.reset_index(drop=True)
    rolling_high = d["high"].shift(1).rolling(window).max()
    avg_vol = d["volume"].shift(1).rolling(window).mean()
    last, last_high, last_avg_vol = d.iloc[-1], rolling_high.iloc[-1], avg_vol.iloc[-1]
    if pd.isna(last_high) or pd.isna(last_avg_vol):
        return False, "Insufficient trailing data."
    price_breakout = last["close"] > last_high
    vol_confirmed = last["volume"] > vol_multiple * last_avg_vol
    if price_breakout and vol_confirmed:
        return True, (f"Close of {last['close']:.2f} breaks the {window}-day high of {last_high:.2f} "
                       f"(+{(last['close']/last_high-1)*100:.1f}%) on {last['volume']/last_avg_vol:.1f}x "
                       f"average volume — a volume-confirmed breakout.")
    if price_breakout:
        return False, f"Price broke the {window}-day high but volume did not confirm."
    return False, "No breakout above the rolling high on the latest candle."


def _trend_label(df: pd.DataFrame, window: int) -> str:
    if len(df) < window + 1:
        return "insufficient data"
    ma = df["close"].rolling(window).mean()
    if pd.isna(ma.iloc[-1]) or pd.isna(ma.iloc[-window]):
        return "insufficient data"
    slope = ma.iloc[-1] - ma.iloc[-window]
    if slope > 0 and df["close"].iloc[-1] > ma.iloc[-1]:
        return "uptrend"
    if slope < 0 and df["close"].iloc[-1] < ma.iloc[-1]:
        return "downtrend"
    return "sideways / consolidating"


def evaluate_technicals(df: pd.DataFrame, fundamental_grade: str) -> TechnicalVerdict:
    signals = detect_patterns(df)
    breakout_flag, breakout_note = detect_volume_breakout(df)
    trend_20, trend_50 = _trend_label(df, 20), _trend_label(df, 50)
    recent_vol = df["volume"].tail(10).mean()
    baseline_vol = df["volume"].tail(60).mean() if len(df) >= 60 else df["volume"].mean()
    volume_trend = ("rising" if baseline_vol and recent_vol > 1.2 * baseline_vol else
                     "falling" if baseline_vol and recent_vol < 0.8 * baseline_vol else "stable")
    bullish = [s for s in signals if s.direction == "bullish"]
    price_supportive = trend_20 == "uptrend" or breakout_flag or len(bullish) > 0
    fundamentally_strong = fundamental_grade in ("EXCELLENT", "GOOD")
    confirms = fundamentally_strong and price_supportive
    if confirms:
        note = (f"Technical structure ({trend_20}, {len(bullish)} bullish pattern(s)) is consistent "
                f"with the {fundamental_grade} grade — price appears to confirm the fundamentals.")
    elif fundamentally_strong:
        note = (f"Fundamentals are {fundamental_grade}, but price shows a {trend_20} with no "
                f"confirming signal — may be a lag or an unpriced risk.")
    else:
        note = f"Fundamental grade is {fundamental_grade}; technical confirmation is secondary here."
    return TechnicalVerdict(signals, breakout_flag, breakout_note, trend_20, trend_50, volume_trend, confirms, note)


# =============================================================================
# 8. PLOTLY CHARTS
# =============================================================================
def candlestick_with_volume(df, title="Price Action", signals=None, breakout_flag=False):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25],
                         vertical_spacing=0.03, subplot_titles=(title, "Volume"))
    fig.add_trace(go.Candlestick(x=df["date"], open=df["open"], high=df["high"], low=df["low"],
        close=df["close"], increasing_line_color=THEME["green"], decreasing_line_color=THEME["red"],
        increasing_fillcolor=THEME["green"], decreasing_fillcolor=THEME["red"], name="OHLC"), row=1, col=1)
    if len(df) >= 20:
        fig.add_trace(go.Scatter(x=df["date"], y=df["close"].rolling(20).mean(), mode="lines",
            line=dict(color=THEME["accent"], width=1.4), name="MA 20"), row=1, col=1)
    if len(df) >= 50:
        fig.add_trace(go.Scatter(x=df["date"], y=df["close"].rolling(50).mean(), mode="lines",
            line=dict(color=THEME["amber"], width=1.4), name="MA 50"), row=1, col=1)
    vol_colors = [THEME["green"] if c >= o else THEME["red"] for o, c in zip(df["open"], df["close"])]
    fig.add_trace(go.Bar(x=df["date"], y=df["volume"], marker_color=vol_colors, name="Volume", opacity=0.75), row=2, col=1)
    if signals:
        for s in signals[-15:]:
            row = df[df["date"] == s.date]
            if row.empty:
                continue
            y = row["high"].values[0] if s.direction != "bearish" else row["low"].values[0]
            color = THEME["green"] if s.direction == "bullish" else (THEME["red"] if s.direction == "bearish" else THEME["muted"])
            fig.add_annotation(x=s.date, y=y, xref="x", yref="y", text=s.pattern, showarrow=True,
                arrowhead=2, ax=0, ay=(-30 if s.direction == "bullish" else 30), font=dict(size=10, color=color),
                bgcolor=THEME["panel"], bordercolor=color, borderwidth=1, row=1, col=1)
    if breakout_flag and len(df) > 0:
        last = df.iloc[-1]
        fig.add_annotation(x=last["date"], y=last["high"], xref="x", yref="y", text="⚡ Volume Breakout",
            showarrow=True, arrowhead=2, ax=0, ay=-45, font=dict(size=11, color=THEME["accent"]),
            bgcolor=THEME["panel"], bordercolor=THEME["accent"], borderwidth=1.5, row=1, col=1)
    fig.update_layout(template="plotly_dark", paper_bgcolor=THEME["bg"], plot_bgcolor=THEME["bg"],
        font=dict(color=THEME["text"]), height=620, margin=dict(l=40, r=20, t=50, b=30),
        xaxis_rangeslider_visible=False, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified")
    fig.update_xaxes(showgrid=True, gridcolor=THEME["border"], row=2, col=1, rangeslider_visible=False)
    fig.update_xaxes(showgrid=True, gridcolor=THEME["border"], row=1, col=1)
    fig.update_yaxes(showgrid=True, gridcolor=THEME["border"], row=1, col=1, title="Price")
    fig.update_yaxes(showgrid=False, row=2, col=1, title="Volume")
    return fig


def margin_comparison_chart(labels, values, benchmark=None):
    colors = [THEME["green"] if (v or 0) >= (benchmark or 0) else THEME["amber"] for v in values]
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors, name="Margin %"))
    if benchmark is not None:
        fig.add_hline(y=benchmark, line_dash="dash", line_color=THEME["muted"],
                       annotation_text=f"Benchmark {benchmark}%", annotation_position="top left")
    fig.update_layout(template="plotly_dark", paper_bgcolor=THEME["bg"], plot_bgcolor=THEME["bg"],
        font=dict(color=THEME["text"]), height=320, margin=dict(l=40, r=20, t=30, b=30),
        title="Margin & Capital-Efficiency Snapshot")
    return fig


def scan_scatter_chart(df):
    fig = go.Figure(go.Scatter(x=df["revenue_cagr_3y"], y=df["roce_pct"], mode="markers+text",
        text=df["sc_id"], textposition="top center",
        marker=dict(size=df["roe_pct"].clip(lower=1), sizemode="area",
                    sizeref=2. * df["roe_pct"].max() / (40. ** 2), sizemin=6,
                    color=df["debt_to_equity"], colorscale="RdYlGn_r", showscale=True, colorbar=dict(title="D/E")),
        name="Universe"))
    fig.update_layout(template="plotly_dark", paper_bgcolor=THEME["bg"], plot_bgcolor=THEME["bg"],
        font=dict(color=THEME["text"]), height=480, margin=dict(l=40, r=20, t=30, b=30),
        xaxis_title="Revenue CAGR (3yr, %)", yaxis_title="ROCE (%)",
        title="Scanner Universe: Growth vs Capital Efficiency (bubble=ROE, color=D/E)")
    return fig


# =============================================================================
# 9. STREAMLIT UI
# =============================================================================
st.set_page_config(page_title="Stock Intelligence Analyst", page_icon="📈", layout="wide",
                    initial_sidebar_state="expanded")

st.markdown(f"""<style>
    .stApp {{ background-color: {THEME['bg']}; }}
    section[data-testid="stSidebar"] {{ background-color: {THEME['panel']}; border-right: 1px solid {THEME['border']}; }}
    .badge {{ display:inline-block; padding:10px 22px; border-radius:8px; font-weight:700;
              font-size:1.15rem; letter-spacing:.03em; color:#0E1117; margin-bottom:6px; }}
    .flag-high {{ border-left:4px solid {THEME['red']}; padding-left:10px; margin-bottom:8px; }}
    .flag-medium {{ border-left:4px solid {THEME['amber']}; padding-left:10px; margin-bottom:8px; }}
    .flag-low {{ border-left:4px solid {THEME['muted']}; padding-left:10px; margin-bottom:8px; }}
    div[data-baseweb="tab-list"] {{ gap:6px; }}
</style>""", unsafe_allow_html=True)


def grade_badge(grade: str) -> str:
    return f'<span class="badge" style="background-color:{GRADE_COLORS.get(grade, THEME["muted"])};">{grade}</span>'


with st.sidebar:
    st.title("📈 Stock Intelligence")
    st.caption("Fundamental & Technical Analyst — powered by Claude")
    mode = st.radio("Choose module", ["📄 Document Upload Analysis", "🔎 Moneycontrol Premium Scanner"],
                     label_visibility="collapsed")
    st.divider()
    st.subheader("Policy Thresholds")
    st.write(f"- Revenue/EPS CAGR ≥ **{THRESHOLDS.min_revenue_cagr}%**")
    st.write(f"- ROE / ROCE ≥ **{THRESHOLDS.min_roe}% / {THRESHOLDS.min_roce}%**")
    st.write(f"- Debt/Equity ≤ **{THRESHOLDS.max_debt_to_equity}x**")
    st.write(f"- Interest Coverage ≥ **{THRESHOLDS.min_interest_coverage}x**")
    st.divider()
    st.subheader("Market Data Source")
    data_source = st.radio(
        "Where should price/fundamentals data come from?",
        ["Yahoo Finance (free, no login, recommended)", "Moneycontrol (experimental, needs your session)"],
        index=0, label_visibility="collapsed",
    )
    mc_client = MoneycontrolClient()
    if data_source.startswith("Moneycontrol"):
        st.caption(("🟢 Live Moneycontrol session configured" if mc_client.is_authenticated()
                    else "🟡 No session configured — Moneycontrol calls will fall back to mock data"))
    else:
        st.caption("🟢 Yahoo Finance selected — real market data, no credentials needed." if _HAS_YFINANCE
                   else "🔴 'yfinance' package missing — add it to requirements.txt.")

    st.divider()
    if not ANTHROPIC_API_KEY:
        st.error("ANTHROPIC_API_KEY not set — add it under Settings → Secrets to enable analysis.")
    else:
        if st.button("🔧 Test Claude API connection"):
            ok, msg = test_claude_connection()
            (st.success if ok else st.error)(msg)


def render_analysis(analysis: StockAnalysis, ohlc_df, provenance_extra=None):
    grade_result = finalize_grade(analysis)
    fm = analysis.fundamental_metrics or {}
    tech_verdict = evaluate_technicals(ohlc_df, grade_result.grade) if ohlc_df is not None and not ohlc_df.empty else None

    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown(f"### {analysis.company_name}")
        st.markdown(grade_badge(grade_result.grade), unsafe_allow_html=True)
        if grade_result.overridden:
            st.warning(f"⚠️ Grade guardrail override: {grade_result.override_reason}")
        st.write(analysis.grade_rationale or grade_result.model_rationale)
    with c2:
        if analysis.cash_flow_flag:
            st.error(f"💸 **Cash Flag:** {analysis.cash_flow_flag}")
        else:
            st.success("💵 No profit/cash-flow divergence detected.")

    tabs = st.tabs(["📊 Fundamentals", "🏛️ Governance & Risk", "🕯️ Technical Chart",
                     "🔮 Growth Trajectory", "🧾 Methodology & Provenance"])

    with tabs[0]:
        cols = st.columns(4)
        metric_map = [("Revenue CAGR", fm.get("revenue_cagr_pct"), "%", fm.get("revenue_cagr_period")),
                      ("EPS CAGR", fm.get("eps_cagr_pct"), "%", fm.get("eps_cagr_period")),
                      ("ROE", fm.get("latest_roe_pct"), "%", None), ("ROCE", fm.get("latest_roce_pct"), "%", None),
                      ("Net Margin", fm.get("latest_net_margin_pct"), "%", None),
                      ("Operating Margin", fm.get("latest_operating_margin_pct"), "%", None),
                      ("Debt/Equity", fm.get("latest_debt_to_equity"), "x", None),
                      ("Interest Coverage", fm.get("latest_interest_coverage"), "x", None)]
        for i, (label, val, unit, sub) in enumerate(metric_map):
            with cols[i % 4]:
                st.metric(label, f"{val}{unit}" if val is not None else "N/A", help=sub)
        if fm.get("margin_trend_notes"):
            st.info(fm["margin_trend_notes"])
        margin_vals = [fm.get("latest_gross_margin_pct"), fm.get("latest_operating_margin_pct"), fm.get("latest_net_margin_pct")]
        if any(v is not None for v in margin_vals):
            st.plotly_chart(margin_comparison_chart(["Gross", "Operating", "Net"], margin_vals, THRESHOLDS.net_margin_floor),
                             use_container_width=True)
        if fm.get("latest_free_cash_flow"):
            st.metric("Free Cash Flow (latest)", fm["latest_free_cash_flow"])

    with tabs[1]:
        if not analysis.qualitative_flags:
            st.success("No governance or qualitative red flags surfaced from the source material.")
        for flag in analysis.qualitative_flags:
            sev = flag.get("severity", "low")
            st.markdown(f'<div class="flag-{sev}"><b>[{sev.upper()}] {flag.get("category","")}:</b> '
                        f'{flag.get("description","")}<br><span style="color:{THEME["muted"]};font-size:0.85em;">'
                        f'Source: {flag.get("page","N/A")}</span></div>', unsafe_allow_html=True)

    with tabs[2]:
        if ohlc_df is None or ohlc_df.empty:
            st.info("No OHLC data available (document-only mode). Use the Moneycontrol Scanner tab instead.")
        else:
            st.plotly_chart(candlestick_with_volume(ohlc_df, f"{analysis.company_name} — Price Action",
                             tech_verdict.signals, tech_verdict.breakout_flag), use_container_width=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("20-day Trend", tech_verdict.trend_20d)
            c2.metric("50-day Trend", tech_verdict.trend_50d)
            c3.metric("Volume Trend", tech_verdict.volume_trend)
            (st.success if tech_verdict.breakout_flag else st.caption)(tech_verdict.breakout_note)
            if tech_verdict.signals:
                st.write("**Detected candlestick patterns (last 60 sessions):**")
                st.dataframe(pd.DataFrame([{"Date": s.date.date(), "Pattern": s.pattern,
                    "Direction": s.direction, "Rule": s.rule} for s in tech_verdict.signals]),
                    use_container_width=True, hide_index=True)
            else:
                st.caption("No classic reversal/continuation patterns detected in the recent window.")
            conf_color = THEME["green"] if tech_verdict.confirms_fundamentals else THEME["amber"]
            st.markdown(f'<div style="border-left:4px solid {conf_color};padding-left:10px;">'
                        f'<b>Fundamentals ↔ Price Confirmation:</b> {tech_verdict.confirmation_note}</div>',
                        unsafe_allow_html=True)

    with tabs[3]:
        st.write(analysis.growth_trajectory or "No forward trajectory commentary generated.")
        st.caption("This is a model-generated projection based on observed trends — not a guarantee.")

    with tabs[4]:
        st.subheader("Formulas used")
        st.dataframe(pd.DataFrame(METHODOLOGY_FORMULAS), use_container_width=True, hide_index=True)
        if analysis.methodology:
            st.write("**Model-reported calculation notes for this analysis:**")
            st.dataframe(pd.DataFrame(analysis.methodology), use_container_width=True, hide_index=True)
        st.subheader("Provenance — source of every extracted fact")
        prov = list(analysis.provenance) + (provenance_extra or [])
        if prov:
            st.dataframe(pd.DataFrame(prov), use_container_width=True, hide_index=True)
        else:
            st.caption("No provenance entries were returned by the model.")
        if tech_verdict:
            st.subheader("Technical pattern-matching logic (deterministic, not LLM)")
            st.dataframe(pd.DataFrame({
                "Pattern": ["Hammer", "Inverted Hammer/Shooting Star", "Bullish/Bearish Engulfing", "Doji",
                            "Morning/Evening Star", "Volume Breakout"],
                "Rule": ["Lower wick ≥ 2× body, small upper wick, after a decline",
                         "Upper wick ≥ 2× body, small lower wick",
                         "Current body fully engulfs prior opposite-color body",
                         "Body ≤ 8% of the day's range",
                         "Large → small indecision → large opposite closing past midpoint",
                         "Close > rolling N-day high AND volume > 1.5x N-day average"]}),
                use_container_width=True, hide_index=True)
        with st.expander("Raw model output (debug)"):
            st.code(analysis.raw_model_output or "(empty)", language="json")


def document_analysis_view():
    st.header("📄 Document Upload Analysis")
    st.caption("Upload Annual Reports, 10-Ks, financial statements, or MD&A sections (PDF).")
    uploaded = st.file_uploader("Upload a PDF financial document", type=["pdf"])
    ticker_hint = st.text_input(
        "Optional: ticker/symbol for live technical chart overlay "
        "(Yahoo Finance format, e.g. RELIANCE.NS, TCS.NS, INFY.NS)",
        value="",
    )

    if uploaded and st.button("🚀 Run Dual-Engine Analysis", type="primary"):
        file_bytes = uploaded.read()
        with st.status("Parsing PDF…", expanded=True) as status:
            parsed = parse_pdf(file_bytes, uploaded.name)
            status.write(f"Extracted {parsed.total_pages} pages → {len(parsed.chunks)} analysis chunk(s).")
            progress = st.progress(0.0, text="Running fundamental & governance extraction…")

            def _cb(pct):
                progress.progress(pct, text=f"Analyzing chunk {int(pct*len(parsed.chunks))}/{len(parsed.chunks)}…")

            try:
                analysis = analyze_document_chunks(parsed.chunks, progress_callback=_cb)
            except Exception as e:
                status.update(label="Analysis failed", state="error")
                st.error(str(e))
                return
            status.update(label="Analysis complete", state="complete")

        ohlc_df = None
        if ticker_hint.strip():
            if data_source.startswith("Moneycontrol"):
                try:
                    ohlc_df = mc_client.get_historical_ohlc(ticker_hint.strip().upper())
                except Exception as e:
                    st.warning(f"Could not fetch Moneycontrol chart data for '{ticker_hint}': {e}")
            else:
                ohlc_df, err = yf_get_ohlc(ticker_hint.strip())
                if err:
                    st.warning(err)

        provenance_extra = [{"fact": "Document page/table index", "source": parsed.provenance_index()[:2000]}]
        st.session_state["last_analysis"] = (analysis, ohlc_df, provenance_extra)

    if "last_analysis" in st.session_state:
        st.divider()
        render_analysis(*st.session_state["last_analysis"])


def moneycontrol_scanner_view():
    using_mc = data_source.startswith("Moneycontrol")
    st.header("🔎 Stock Scanner" + (" — Moneycontrol (experimental)" if using_mc else " — Yahoo Finance"))

    if using_mc:
        st.caption("Programmatic scan using your configured Moneycontrol premium session. "
                   "Falls back to illustrative mock data when no session is configured or a live "
                   "request fails (Moneycontrol has no public API — endpoint paths are unverified placeholders).")
        with st.expander("⚙️ Session status & configuration help", expanded=not mc_client.is_authenticated()):
            if mc_client.is_authenticated():
                st.success("A Moneycontrol session is configured — live endpoints will be attempted first.")
            else:
                st.warning("No Moneycontrol session configured. Add MC_COOKIE / MC_X_AUTH_TOKEN under "
                          "Streamlit Secrets to attempt live premium data (not guaranteed to work — see README).")
            st.code('MC_COOKIE = "mc_authToken=...; mc_session=...;"\nMC_X_AUTH_TOKEN = ""', language="toml")
    else:
        st.caption("Real, free scan across a curated NIFTY50 watchlist using Yahoo Finance — no login required. "
                   "Note: Yahoo Finance has no direct ROCE figure, so ROE is used as a proxy for it below.")

    st.subheader("Scan filters")
    c1, c2, c3, c4 = st.columns(4)
    universe = c1.selectbox("Universe", ["NIFTY50", "NIFTY500", "NIFTYMIDCAP150", "NIFTYSMALLCAP250"], index=0,
                             disabled=not using_mc, help="Only the Moneycontrol path supports universes beyond NIFTY50.")
    min_roe = c2.slider("Min ROE %", 0, 40, int(THRESHOLDS.min_roe))
    min_roce = c3.slider("Min ROCE %", 0, 40, int(THRESHOLDS.min_roce), disabled=not using_mc)
    max_de = c4.slider("Max Debt/Equity", 0.0, 3.0, THRESHOLDS.max_debt_to_equity, step=0.1)
    min_rev_cagr = st.slider("Min 3yr Revenue CAGR %", 0, 40, int(THRESHOLDS.min_revenue_cagr))

    if st.button("🔍 Run Scan", type="primary"):
        filters = {"universe": universe, "min_roe": min_roe, "min_roce": min_roce,
                   "max_de": max_de, "min_revenue_cagr_3y": min_rev_cagr}
        with st.spinner("Scanning universe…"):
            if using_mc:
                st.session_state["scan_results"] = mc_client.scan_universe(filters, limit=50)
            else:
                results, errors = yf_scan_universe(filters, limit=50)
                st.session_state["scan_results"] = results
                if errors:
                    st.session_state["scan_errors"] = errors

    if st.session_state.get("scan_errors"):
        with st.expander(f"⚠️ {len(st.session_state['scan_errors'])} ticker(s) failed to fetch"):
            for e in st.session_state["scan_errors"]:
                st.caption(e)

    if "scan_results" in st.session_state:
        results = st.session_state["scan_results"]
        st.divider()
        if results.empty:
            st.info("No stocks matched the current filter set.")
        else:
            st.subheader(f"Top-ranked matches ({len(results)})")
            st.dataframe(results.style.background_gradient(subset=["roce_pct", "roe_pct"], cmap="Greens"),
                        use_container_width=True, hide_index=True)
            st.plotly_chart(scan_scatter_chart(results), use_container_width=True)

            st.divider()
            st.subheader("Deep-dive: full dual-engine analysis on a scanned stock")
            pick = st.selectbox("Select a stock to analyze", results["sc_id"].tolist())
            if st.button(f"🚀 Analyze {pick} with Claude"):
                with st.status(f"Fetching fundamentals & chart data for {pick}…", expanded=True) as status:
                    if using_mc:
                        fundamentals_payload = mc_client.get_fundamentals(pick)
                        shareholding_payload = mc_client.get_shareholding_pattern(pick)
                        ohlc_df = mc_client.get_historical_ohlc(pick)
                    else:
                        fundamentals_payload, fund_err = yf_get_fundamentals(pick)
                        if fund_err:
                            status.write(f"⚠️ {fund_err}")
                            fundamentals_payload = fundamentals_payload or {"sc_id": pick, "company_name": pick, "financials": {}}
                        shareholding_payload = {}
                        ohlc_df, ohlc_err = yf_get_ohlc(pick)
                        if ohlc_err:
                            status.write(f"⚠️ {ohlc_err}")

                    combined_payload = {**fundamentals_payload, "shareholding": shareholding_payload,
                                        "scan_row": results[results["sc_id"] == pick].to_dict("records")[0]}
                    status.write("Running dual-engine analysis via Claude…")
                    try:
                        analysis = analyze_scan_payload(combined_payload)
                    except Exception as e:
                        status.update(label="Analysis failed", state="error")
                        st.error(str(e))
                        return
                    status.update(label="Analysis complete", state="complete")
                provenance_extra = [{"fact": "Data source", "source": fundamentals_payload.get("_source", "unknown")}]
                st.session_state["scan_deep_dive"] = (analysis, ohlc_df, provenance_extra)

    if "scan_deep_dive" in st.session_state:
        st.divider()
        render_analysis(*st.session_state["scan_deep_dive"])


try:
    if mode.startswith("📄"):
        document_analysis_view()
    else:
        moneycontrol_scanner_view()
except Exception as e:
    st.error(f"Something went wrong: {e}")
    with st.expander("Technical details"):
        st.exception(e)
    st.info("If this persists, click '🔧 Test Claude API connection' in the sidebar to rule out an API-key "
            "issue, or check 'Manage app' → logs on Streamlit Cloud for the full traceback.")
