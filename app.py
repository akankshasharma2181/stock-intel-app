"""
Stock Intelligence Analyst — single-file cloud build.
Dual-engine equity analysis (Fundamental + Qualitative/Governance) over
uploaded PDFs or a Moneycontrol premium-session scan, cross-checked against
a rule-based technical engine, and graded EXCELLENT / GOOD / BAD-RISKY /
LOSS-MAKING-DESTRUCTIVE.
"""

from __future__ import annotations

import io
import json
import re
import time
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
    from pypdf import PdfReader
    _HAS_PYPDF = True
except ImportError:
    _HAS_PYPDF = False

import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

st.set_page_config(page_title="Stock Intelligence Analyst", layout="wide", page_icon="📈")

# ============================================================================
# CONFIG — secrets pulled from st.secrets first, env vars as fallback
# ============================================================================
def _get_secret(key: str, default: str = "") -> str:
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    import os
    return os.environ.get(key, default)

ANTHROPIC_API_KEY = _get_secret("ANTHROPIC_API_KEY")
CLAUDE_MODEL = _get_secret("CLAUDE_MODEL", "claude-sonnet-5")
CLAUDE_MAX_TOKENS = 4096
CLAUDE_TEMPERATURE = 0.2

PDF_CHUNK_CHAR_SIZE = 12000
PDF_CHUNK_OVERLAP = 500
MAX_CHUNKS_SENT_TO_CLAUDE = 8

@dataclass
class FundamentalThresholds:
    min_revenue_cagr: float = 10.0
    min_eps_cagr: float = 10.0
    min_roe: float = 15.0
    min_roce: float = 15.0
    max_debt_to_equity: float = 1.0
    min_interest_coverage: float = 3.0
    net_margin_floor: float = 5.0

THRESHOLDS = FundamentalThresholds()

MONEYCONTROL_CONFIG = {
    "base_url": _get_secret("MC_BASE_URL", "https://www.moneycontrol.com"),
    "api_base_url": _get_secret("MC_API_BASE_URL", "https://api.moneycontrol.com"),
    "priceapi_base_url": _get_secret("MC_PRICEAPI_BASE_URL", "https://priceapi.moneycontrol.com"),
    "cookie": _get_secret("MC_COOKIE", ""),
    "x_auth_token": _get_secret("MC_X_AUTH_TOKEN", ""),
    "user_agent": _get_secret(
        "MC_USER_AGENT",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    ),
    "request_timeout": 15,
    "max_retries": 3,
    "retry_backoff_seconds": 1.5,
}

DEFAULT_SCAN_UNIVERSE = "NIFTY500"
SCAN_RESULT_LIMIT = 50

THEME = {
    "bg": "#0E1117", "panel": "#161B22", "border": "#262D38", "text": "#E6EDF3",
    "muted": "#8B949E", "green": "#26A69A", "red": "#EF5350", "amber": "#D29922",
    "accent": "#4C8DFF",
}
GRADE_COLORS = {
    "EXCELLENT": "#26A69A", "GOOD": "#4C8DFF",
    "BAD/RISKY": "#D29922", "LOSS-MAKING / DESTRUCTIVE": "#EF5350",
}

# ============================================================================
# PDF PARSER
# ============================================================================
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

def _extract_with_pdfplumber(file_bytes: bytes) -> List[PageContent]:
    pages = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            try:
                tables = page.extract_tables() or []
            except Exception:
                tables = []
            pages.append(PageContent(page_number=i, text=text, tables=tables))
    return pages

def _extract_with_pypdf(file_bytes: bytes) -> List[PageContent]:
    reader = PdfReader(io.BytesIO(file_bytes))
    return [PageContent(page_number=i, text=p.extract_text() or "", tables=[])
            for i, p in enumerate(reader.pages, start=1)]

def _clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def extract_pages(file_bytes: bytes, filename: str) -> List[PageContent]:
    try:
        pages = _extract_with_pdfplumber(file_bytes)
        if any(p.text.strip() for p in pages):
            return pages
    except Exception:
        pass
    if _HAS_PYPDF:
        try:
            return _extract_with_pypdf(file_bytes)
        except Exception:
            pass
    raise ValueError(f"Could not extract text from '{filename}'. It may be a scanned image PDF requiring OCR.")

def table_to_markdown(table: List[List[Optional[str]]]) -> str:
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
        block = f"\n[PAGE {p.page_number}]\n{_clean_text(p.text)}\n"
        for t_idx, table in enumerate(p.tables, start=1):
            md = table_to_markdown(table)
            if md:
                block += f"\n[PAGE {p.page_number} - TABLE {t_idx}]\n{md}\n"
        blocks.append(block)
    return blocks

def chunk_document(pages, chunk_size=PDF_CHUNK_CHAR_SIZE, overlap=PDF_CHUNK_OVERLAP,
                    max_chunks=MAX_CHUNKS_SENT_TO_CLAUDE) -> List[DocChunk]:
    blocks = build_page_blocks(pages)
    chunks: List[DocChunk] = []
    current_text = ""
    current_start = None
    chunk_id = 0
    for page in pages:
        block = blocks[page.page_number - 1]
        if current_start is None:
            current_start = page.page_number
        if len(current_text) + len(block) > chunk_size and current_text:
            chunks.append(DocChunk(chunk_id, current_start, page.page_number - 1,
                                    current_text, current_text.count("- TABLE")))
            chunk_id += 1
            current_text = current_text[-overlap:] + block
            current_start = page.page_number
        else:
            current_text += block
    if current_text.strip():
        chunks.append(DocChunk(chunk_id, current_start, pages[-1].page_number,
                                current_text, current_text.count("- TABLE")))
    if len(chunks) <= max_chunks:
        return chunks
    table_chunks = [c for c in chunks if c.table_count > 0]
    other_chunks = [c for c in chunks if c.table_count == 0]
    slots_left = max(max_chunks - len(table_chunks), 0)
    sampled = other_chunks[::max(len(other_chunks) // slots_left, 1)][:slots_left] if slots_left and other_chunks else []
    return sorted(table_chunks + sampled, key=lambda c: c.start_page)[:max_chunks]

def parse_pdf(file_bytes: bytes, filename: str) -> Tuple[List[PageContent], List[DocChunk]]:
    pages = extract_pages(file_bytes, filename)
    return pages, chunk_document(pages)

# ============================================================================
# CLAUDE ENGINE
# ============================================================================
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

def get_client() -> anthropic.Anthropic:
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it in Streamlit Secrets.")
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

SYSTEM_PROMPT = """You are a rigorous equity research analyst operating a dual-engine \
review framework: (1) Financial Health & Compound Growth, and (2) Qualitative Risk & \
Corporate Governance. You extract only what is explicitly supported by the provided \
source text/data — you do not invent numbers. When a metric cannot be computed from the \
provided material, you omit it rather than guess, and you say so in `data_gaps`.
You always respond with a single valid JSON object and nothing else (no markdown fences, \
no commentary before or after)."""

CHUNK_ANALYSIS_INSTRUCTIONS = """
Analyze the SOURCE TEXT below (one chunk of a larger financial document, pages noted in \
[PAGE n] tags) and extract ONLY what is explicitly present. Return JSON:
{
  "company_name": string or null,
  "fundamental_findings": {
    "revenue": [{"period": string, "value": string, "page": string}],
    "eps": [{"period": string, "value": string, "page": string}],
    "net_margin_pct": [{"period": string, "value": number, "page": string}],
    "roe_pct": [{"period": string, "value": number, "page": string}],
    "roce_pct": [{"period": string, "value": number, "page": string}],
    "debt_to_equity": [{"period": string, "value": number, "page": string}],
    "operating_cash_flow": [{"period": string, "value": string, "page": string}],
    "net_profit_reported": [{"period": string, "value": string, "page": string}]
  },
  "governance_findings": {
    "management_changes": [{"description": string, "page": string}],
    "related_party_transactions": [{"description": string, "page": string}],
    "auditor_qualifications": [{"description": string, "page": string}],
    "pending_litigation": [{"description": string, "page": string}],
    "promoter_pledge_changes": [{"description": string, "page": string}]
  },
  "data_gaps": [string]
}
Only include entries you can directly cite from the SOURCE TEXT."""

SYNTHESIS_INSTRUCTIONS = """
You are given per-chunk extraction results from one financial document, plus threshold \
benchmarks. Merge into ONE final verdict. Compute CAGR for revenue/EPS where possible. \
Flag if net_profit_reported is positive while operating_cash_flow is negative. Assign \
exactly one grade: "EXCELLENT", "GOOD", "BAD/RISKY", "LOSS-MAKING / DESTRUCTIVE".

Benchmarks: Revenue/EPS CAGR {min_revenue_cagr}%/{min_eps_cagr}%, ROE/ROCE {min_roe}%/{min_roce}%, \
Max D/E {max_de}x, Min Interest Coverage {min_ic}x, Net margin floor {net_margin_floor}%.

Return JSON only:
{{
  "company_name": string,
  "grade": "EXCELLENT"|"GOOD"|"BAD/RISKY"|"LOSS-MAKING / DESTRUCTIVE",
  "grade_rationale": string,
  "fundamental_metrics": {{"revenue_cagr_pct": number|null, "eps_cagr_pct": number|null,
    "latest_net_margin_pct": number|null, "latest_roe_pct": number|null,
    "latest_roce_pct": number|null, "latest_debt_to_equity": number|null,
    "margin_trend_notes": string}},
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

SCAN_ANALYSIS_INSTRUCTIONS = """
You are given a cleaned JSON payload of fundamental/market data for one stock. Apply the \
same dual-engine framework and benchmarks. Return the SAME schema as document synthesis. \
For `provenance`, cite the payload field name instead of a page number.

Benchmarks: Revenue/EPS CAGR {min_revenue_cagr}%/{min_eps_cagr}%, ROE/ROCE {min_roe}%/{min_roce}%, \
Max D/E {max_de}x, Min Interest Coverage {min_ic}x, Net margin floor {net_margin_floor}%.

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

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def _call_claude(system: str, user: str) -> str:
    client = get_client()
    resp = client.messages.create(
        model=CLAUDE_MODEL, max_tokens=CLAUDE_MAX_TOKENS, temperature=CLAUDE_TEMPERATURE,
        system=system, messages=[{"role": "user", "content": user}],
    )
    return "\n".join(b.text for b in resp.content if getattr(b, "type", "") == "text")

def extract_from_chunk(chunk: DocChunk) -> Dict[str, Any]:
    user_msg = CHUNK_ANALYSIS_INSTRUCTIONS + f"\n\nSOURCE TEXT (pages {chunk.start_page}-{chunk.end_page}):\n{chunk.text}"
    raw = _call_claude(SYSTEM_PROMPT, user_msg)
    try:
        return _extract_json(raw)
    except json.JSONDecodeError:
        return {"company_name": None, "fundamental_findings": {}, "governance_findings": {},
                "data_gaps": [f"Parse failure on pages {chunk.start_page}-{chunk.end_page}"]}

def _parse_analysis(raw: str) -> StockAnalysis:
    try:
        data = _extract_json(raw)
    except json.JSONDecodeError:
        return StockAnalysis(company_name="Parse Error", grade="GOOD",
                              grade_rationale="Model output could not be parsed as JSON.", raw_model_output=raw)
    return StockAnalysis(
        company_name=data.get("company_name") or "Unknown",
        grade=data.get("grade", "GOOD"),
        grade_rationale=data.get("grade_rationale", ""),
        fundamental_metrics=data.get("fundamental_metrics", {}),
        qualitative_flags=data.get("qualitative_flags", []),
        cash_flow_flag=data.get("cash_flow_flag"),
        growth_trajectory=data.get("growth_trajectory", ""),
        methodology=data.get("methodology", []),
        provenance=data.get("provenance", []),
        raw_model_output=raw,
    )

def analyze_document_chunks(chunks: List[DocChunk], progress_callback=None) -> StockAnalysis:
    extractions = []
    for i, chunk in enumerate(chunks):
        result = extract_from_chunk(chunk)
        result["_pages"] = f"{chunk.start_page}-{chunk.end_page}"
        extractions.append(result)
        if progress_callback:
            progress_callback((i + 1) / len(chunks))
    prompt = SYNTHESIS_INSTRUCTIONS.format(
        min_revenue_cagr=THRESHOLDS.min_revenue_cagr, min_eps_cagr=THRESHOLDS.min_eps_cagr,
        min_roe=THRESHOLDS.min_roe, min_roce=THRESHOLDS.min_roce,
        max_de=THRESHOLDS.max_debt_to_equity, min_ic=THRESHOLDS.min_interest_coverage,
        net_margin_floor=THRESHOLDS.net_margin_floor,
        chunk_json=json.dumps(extractions, indent=2)[:60000],
    )
    return _parse_analysis(_call_claude(SYSTEM_PROMPT, prompt))

def analyze_scan_payload(payload: Dict[str, Any]) -> StockAnalysis:
    prompt = SCAN_ANALYSIS_INSTRUCTIONS.format(
        min_revenue_cagr=THRESHOLDS.min_revenue_cagr, min_eps_cagr=THRESHOLDS.min_eps_cagr,
        min_roe=THRESHOLDS.min_roe, min_roce=THRESHOLDS.min_roce,
        max_de=THRESHOLDS.max_debt_to_equity, min_ic=THRESHOLDS.min_interest_coverage,
        net_margin_floor=THRESHOLDS.net_margin_floor,
        payload_json=json.dumps(payload, indent=2)[:60000],
    )
    return _parse_analysis(_call_claude(SYSTEM_PROMPT, prompt))

# ============================================================================
# GRADING ENGINE (deterministic guardrail)
# ============================================================================
GRADES_ORDER = ["LOSS-MAKING / DESTRUCTIVE", "BAD/RISKY", "GOOD", "EXCELLENT"]

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
    override_reason = None
    final_grade = model_grade

    burning_cash = bool(analysis.cash_flow_flag) and "CRITICAL" in (analysis.cash_flow_flag or "")
    if burning_cash and model_grade in ("EXCELLENT", "GOOD"):
        final_grade = "LOSS-MAKING / DESTRUCTIVE"
        override_reason = "Downgraded: profit is not cash-backed (positive profit, negative operating cash flow)."

    high_sev = sum(1 for f in analysis.qualitative_flags if f.get("severity") == "high")
    if high_sev >= 1 and final_grade == "EXCELLENT":
        final_grade = "GOOD" if high_sev == 1 else "BAD/RISKY"
        override_reason = ((override_reason + " ") if override_reason else "") + f"Capped due to {high_sev} high-severity governance flag(s)."
    if high_sev >= 2 and final_grade != "LOSS-MAKING / DESTRUCTIVE":
        final_grade = "BAD/RISKY"
        override_reason = ((override_reason + " ") if override_reason else "") + f"Multiple ({high_sev}) high-severity governance red flags."

    de = fm.get("latest_debt_to_equity")
    if de is not None and de > (THRESHOLDS.max_debt_to_equity * 2.5) and final_grade == "EXCELLENT":
        final_grade = "BAD/RISKY"
        override_reason = ((override_reason + " ") if override_reason else "") + f"D/E of {de}x exceeds 2.5x the policy ceiling."

    return GradeResult(final_grade, GRADE_COLORS.get(final_grade, "#8B949E"),
                        final_grade != model_grade, override_reason, analysis.grade_rationale)

# ============================================================================
# MONEYCONTROL CLIENT (config-driven placeholder + mock fallback)
# ============================================================================
ENDPOINTS = {
    "stock_fundamentals": "{api_base}/mcapi/v1/stock/fundamentals",
    "historical_ohlc": "{priceapi_base}/techCharts/indianMarket/stock/history",
    "screener_scan": "{api_base}/mcapi/v1/screener/scan",
}

class MoneycontrolAuthError(Exception):
    pass

class MoneycontrolClient:
    def __init__(self, cfg=None):
        self.cfg = cfg or MONEYCONTROL_CONFIG
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
        retries, backoff, last_exc = self.cfg.get("max_retries", 3), self.cfg.get("retry_backoff_seconds", 1.5), None
        for attempt in range(retries):
            try:
                resp = self.session.get(url, headers=self._headers(), params=params, timeout=self.cfg.get("request_timeout", 15))
                if resp.status_code in (401, 403):
                    raise MoneycontrolAuthError("Session rejected — cookie/token likely expired.")
                resp.raise_for_status()
                return resp.json()
            except MoneycontrolAuthError:
                raise
            except Exception as exc:
                last_exc = exc
                time.sleep(backoff * (attempt + 1))
        raise ConnectionError(f"Moneycontrol request failed: {last_exc}")

    def get_fundamentals(self, sc_id: str) -> Dict[str, Any]:
        if not self.is_authenticated():
            return _mock_fundamentals(sc_id)
        url = ENDPOINTS["stock_fundamentals"].format(api_base=self.cfg["api_base_url"])
        data = self._request(url, params={"scId": sc_id, "type": "consolidated"}) or {}
        return {"sc_id": sc_id, "company_name": data.get("companyName", sc_id),
                "financials": data.get("financials", data), "_source": "moneycontrol_live"}

    def get_historical_ohlc(self, sc_id, resolution="D", from_ts=None, to_ts=None) -> pd.DataFrame:
        to_ts = to_ts or int(time.time())
        from_ts = from_ts or to_ts - 365 * 24 * 3600
        if not self.is_authenticated():
            return _mock_ohlc(sc_id, from_ts, to_ts)
        url = ENDPOINTS["historical_ohlc"].format(priceapi_base=self.cfg["priceapi_base_url"])
        data = self._request(url, params={"symbol": sc_id, "resolution": resolution, "from": from_ts, "to": to_ts}) or {}
        if "t" not in data:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
        df = pd.DataFrame({"date": pd.to_datetime(data["t"], unit="s"), "open": data.get("o", []),
                            "high": data.get("h", []), "low": data.get("l", []),
                            "close": data.get("c", []), "volume": data.get("v", [])})
        return df.sort_values("date").reset_index(drop=True)

    def scan_universe(self, filters: Dict[str, Any], limit: int = 50) -> pd.DataFrame:
        if not self.is_authenticated():
            return _mock_scan(filters, limit)
        url = ENDPOINTS["screener_scan"].format(api_base=self.cfg["api_base_url"])
        data = self._request(url, params={**filters, "limit": limit}) or {}
        return pd.DataFrame(data.get("results", []))

def _mock_fundamentals(sc_id: str) -> Dict[str, Any]:
    return {"sc_id": sc_id, "company_name": f"{sc_id} (MOCK — connect Moneycontrol session for live data)",
            "financials": {
                "revenue": [{"period": f"FY{21+i}", "value": str(38000 + i * 7500)} for i in range(5)],
                "eps": [{"period": f"FY{21+i}", "value": str(round(42.1 + i * 7.5, 1))} for i in range(5)],
                "net_margin_pct": 17.6, "roe_pct": 22.4, "roce_pct": 26.1, "debt_to_equity": 0.31,
                "operating_cash_flow": "12400", "net_profit_reported": "11980",
            }, "_source": "mock"}

def _mock_ohlc(sc_id, from_ts, to_ts) -> pd.DataFrame:
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
    rows = [{"sc_id": sc, "company_name": name, "sector": sec,
             "revenue_cagr_3y": round(rng.uniform(6, 28), 1), "roe_pct": round(rng.uniform(8, 34), 1),
             "roce_pct": round(rng.uniform(8, 34), 1), "debt_to_equity": round(rng.uniform(0.0, 1.8), 2),
             "net_margin_pct": round(rng.uniform(3, 26), 1), "cmp": round(rng.uniform(300, 4500), 1),
             "pe_ratio": round(rng.uniform(12, 65), 1)} for sc, name, sec in names]
    df = pd.DataFrame(rows)
    if filters.get("min_roe"):
        df = df[df["roe_pct"] >= filters["min_roe"]]
    if filters.get("min_roce"):
        df = df[df["roce_pct"] >= filters["min_roce"]]
    if filters.get("max_de") is not None:
        df = df[df["debt_to_equity"] <= filters["max_de"]]
    return df.sort_values("roce_pct", ascending=False).head(limit).reset_index(drop=True)

# ============================================================================
# TECHNICAL ENGINE
# ============================================================================
@dataclass
class PatternSignal:
    date: pd.Timestamp
    pattern: str
    direction: str
    rule: str

def detect_patterns(df: pd.DataFrame, lookback: int = 60) -> List[PatternSignal]:
    d = df.tail(lookback).reset_index(drop=True)
    signals = []
    body = (d["close"] - d["open"]).abs()
    rng_ = (d["high"] - d["low"]).replace(0, np.nan)
    for i in range(2, len(d)):
        o, h, l, c = d.loc[i, ["open", "high", "low", "close"]]
        po, pc = d.loc[i - 1, ["open", "close"]]
        upper_wick, lower_wick = h - max(o, c), min(o, c) - l
        b = body.iloc[i]
        r = rng_.iloc[i] if not np.isnan(rng_.iloc[i]) else 1e-9
        if r > 0 and lower_wick >= 2 * b and upper_wick <= 0.15 * r and b <= 0.35 * r:
            if d.loc[max(0, i - 5):i - 1, "close"].mean() > c:
                signals.append(PatternSignal(d.loc[i, "date"], "Hammer", "bullish", "Long lower wick after decline."))
        if pc < po and c > o and o <= pc and c >= po and b > body.iloc[i - 1]:
            signals.append(PatternSignal(d.loc[i, "date"], "Bullish Engulfing", "bullish", "Green body engulfs prior red body."))
        if pc > po and c < o and o >= pc and c <= po and b > body.iloc[i - 1]:
            signals.append(PatternSignal(d.loc[i, "date"], "Bearish Engulfing", "bearish", "Red body engulfs prior green body."))
        if r > 0 and b <= 0.08 * r:
            signals.append(PatternSignal(d.loc[i, "date"], "Doji", "neutral", "Open ≈ close → indecision."))
    return signals

def detect_volume_breakout(df: pd.DataFrame, window: int = 20, vol_multiple: float = 1.5):
    if len(df) < window + 2:
        return False, "Insufficient history."
    d = df.reset_index(drop=True)
    rolling_high = d["high"].shift(1).rolling(window).max()
    avg_vol = d["volume"].shift(1).rolling(window).mean()
    last, lh, av = d.iloc[-1], rolling_high.iloc[-1], avg_vol.iloc[-1]
    if pd.isna(lh) or pd.isna(av):
        return False, "Insufficient trailing data."
    if last["close"] > lh and last["volume"] > vol_multiple * av:
        return True, f"Close {last['close']:.2f} breaks {window}-day high {lh:.2f} on {last['volume']/av:.1f}x avg volume."
    return False, "No confirmed breakout."

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
    return "sideways"

# ============================================================================
# CHARTS
# ============================================================================
def candlestick_with_volume(df: pd.DataFrame, title="Price Action", signals=None, breakout_flag=False) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25],
                         vertical_spacing=0.03, subplot_titles=(title, "Volume"))
    fig.add_trace(go.Candlestick(x=df["date"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
                                  increasing_line_color=THEME["green"], decreasing_line_color=THEME["red"],
                                  increasing_fillcolor=THEME["green"], decreasing_fillcolor=THEME["red"], name="OHLC"),
                  row=1, col=1)
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
            fig.add_annotation(x=s.date, y=y, text=s.pattern, showarrow=True, arrowhead=2, ax=0,
                                ay=-30 if s.direction == "bullish" else 30,
                                font=dict(size=10, color=color), bgcolor=THEME["panel"],
                                bordercolor=color, borderwidth=1, row=1, col=1)
    if breakout_flag and len(df) > 0:
        last = df.iloc[-1]
        fig.add_annotation(x=last["date"], y=last["high"], text="⚡ Volume Breakout", showarrow=True,
                            arrowhead=2, ax=0, ay=-45, font=dict(size=11, color=THEME["accent"]),
                            bgcolor=THEME["panel"], bordercolor=THEME["accent"], borderwidth=1.5, row=1, col=1)
    fig.update_layout(template="plotly_dark", paper_bgcolor=THEME["bg"], plot_bgcolor=THEME["bg"],
                       font=dict(color=THEME["text"]), height=620, margin=dict(l=40, r=20, t=50, b=30),
                       xaxis_rangeslider_visible=False,
                       legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                       hovermode="x unified")
    return fig

def growth_trend_chart(series: List[dict], value_key: str, label: str) -> go.Figure:
    periods = [s.get("period") for s in series]
    values = [s.get(value_key) for s in series]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=periods, y=values, marker_color=THEME["accent"], name=label, opacity=0.85))
    fig.add_trace(go.Scatter(x=periods, y=values, mode="lines+markers", line=dict(color=THEME["green"], width=2), name=f"{label} trend"))
    fig.update_layout(template="plotly_dark", paper_bgcolor=THEME["bg"], plot_bgcolor=THEME["bg"],
                       font=dict(color=THEME["text"]), height=320, margin=dict(l=40, r=20, t=30, b=30), title=label)
    return fig

def scan_scatter_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["revenue_cagr_3y"], y=df["roce_pct"], mode="markers+text",
                              text=df["sc_id"], textposition="top center",
                              marker=dict(size=df["roe_pct"].clip(lower=1), sizemode="area",
                                          sizeref=2. * df["roe_pct"].max() / (40. ** 2), sizemin=6,
                                          color=df["debt_to_equity"], colorscale="RdYlGn_r", showscale=True,
                                          colorbar=dict(title="D/E")), name="Universe"))
    fig.update_layout(template="plotly_dark", paper_bgcolor=THEME["bg"], plot_bgcolor=THEME["bg"],
                       font=dict(color=THEME["text"]), height=480, margin=dict(l=40, r=20, t=30, b=30),
                       xaxis_title="Revenue CAGR (3yr, %)", yaxis_title="ROCE (%)",
                       title="Growth vs Capital Efficiency (bubble=ROE, color=D/E)")
    return fig

# ============================================================================
# UI HELPERS
# ============================================================================
def render_grade_badge(result: GradeResult):
    st.markdown(
        f"""<div style="background:{result.color}22;border:1px solid {result.color};
        border-radius:8px;padding:14px 18px;margin-bottom:10px;">
        <span style="font-size:1.4em;font-weight:700;color:{result.color}">{result.grade}</span>
        </div>""", unsafe_allow_html=True)
    if result.overridden:
        st.warning(f"Guardrail override: {result.override_reason}")
    st.markdown(f"**Model rationale:** {result.model_rationale}")

def render_analysis(analysis: StockAnalysis):
    grade_result = finalize_grade(analysis)
    st.subheader(analysis.company_name)
    render_grade_badge(grade_result)

    tabs = st.tabs(["Fundamentals", "Governance Flags", "Growth Trajectory", "Methodology & Provenance", "Raw Output"])
    fm = analysis.fundamental_metrics or {}
    with tabs[0]:
        cols = st.columns(4)
        metric_map = [
            ("Revenue CAGR", fm.get("revenue_cagr_pct"), "%"), ("EPS CAGR", fm.get("eps_cagr_pct"), "%"),
            ("Net Margin", fm.get("latest_net_margin_pct"), "%"), ("ROE", fm.get("latest_roe_pct"), "%"),
            ("ROCE", fm.get("latest_roce_pct"), "%"), ("Debt/Equity", fm.get("latest_debt_to_equity"), "x"),
        ]
        for i, (label, val, unit) in enumerate(metric_map):
            with cols[i % 4]:
                st.metric(label, f"{val}{unit}" if val is not None else "N/A")
        if analysis.cash_flow_flag:
            st.error(analysis.cash_flow_flag)
        if fm.get("margin_trend_notes"):
            st.info(fm["margin_trend_notes"])

    with tabs[1]:
        if not analysis.qualitative_flags:
            st.write("No governance flags extracted.")
        for f in analysis.qualitative_flags:
            sev = f.get("severity", "low")
            color = {"high": THEME["red"], "medium": THEME["amber"], "low": THEME["muted"]}.get(sev, THEME["muted"])
            st.markdown(f"<span style='color:{color}'>●</span> **{f.get('category','')}** ({sev}) — {f.get('description','')} "
                        f"<span style='color:{THEME['muted']}'>[{f.get('page','')}]</span>", unsafe_allow_html=True)

    with tabs[2]:
        st.write(analysis.growth_trajectory or "No projection generated.")

    with tabs[3]:
        st.write("**Methodology**")
        for m in analysis.methodology:
            st.markdown(f"- **{m.get('metric')}**: {m.get('formula')}")
        st.write("**Provenance**")
        for p in analysis.provenance:
            st.markdown(f"- {p.get('fact')} — *{p.get('source')}*")
        if analysis.data_gaps if hasattr(analysis, "data_gaps") else False:
            st.write(analysis.data_gaps)

    with tabs[4]:
        st.code(analysis.raw_model_output or "(empty)", language="json")

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    st.title("📈 Fundamental & Technical Stock Intelligence Analyst")
    st.caption("Dual-engine equity analysis, cross-validated against a rule-based technical engine.")

    if not ANTHROPIC_API_KEY:
        st.warning("No ANTHROPIC_API_KEY found in Secrets. Add it in Streamlit Cloud → Settings → Secrets to enable Claude analysis.")

    mode = st.sidebar.radio("Analysis mode", ["Upload PDF (Annual Report)", "Moneycontrol Scanner"])
    client = MoneycontrolClient()
    if not client.is_authenticated():
        st.sidebar.info("Running on mock Moneycontrol data — add MC_COOKIE / MC_X_AUTH_TOKEN in Secrets for live data.")

    if mode == "Upload PDF (Annual Report)":
        uploaded = st.file_uploader("Upload an annual report / financial statement PDF", type=["pdf"])
        if uploaded and st.button("Run Dual-Engine Analysis", type="primary"):
            with st.spinner("Extracting PDF content..."):
                try:
                    pages, chunks = parse_pdf(uploaded.read(), uploaded.name)
                except ValueError as e:
                    st.error(str(e))
                    return
            st.success(f"Parsed {len(pages)} pages into {len(chunks)} chunk(s).")
            progress = st.progress(0.0)
            with st.spinner("Running Claude analysis..."):
                try:
                    analysis = analyze_document_chunks(chunks, progress_callback=progress.progress)
                except RuntimeError as e:
                    st.error(str(e))
                    return
            render_analysis(analysis)

    else:
        st.sidebar.subheader("Scanner Filters")
        min_roe = st.sidebar.slider("Min ROE %", 0, 40, 15)
        min_roce = st.sidebar.slider("Min ROCE %", 0, 40, 15)
        max_de = st.sidebar.slider("Max Debt/Equity", 0.0, 3.0, 1.0)
        min_rev_cagr = st.sidebar.slider("Min Revenue CAGR 3y %", 0, 40, 10)

        if st.sidebar.button("Run Scan", type="primary"):
            with st.spinner("Scanning universe..."):
                df = client.scan_universe({
                    "universe": DEFAULT_SCAN_UNIVERSE, "min_roe": min_roe, "min_roce": min_roce,
                    "max_de": max_de, "min_revenue_cagr_3y": min_rev_cagr,
                }, limit=SCAN_RESULT_LIMIT)
            st.session_state["scan_df"] = df

        df = st.session_state.get("scan_df")
        if df is not None and not df.empty:
            st.plotly_chart(scan_scatter_chart(df), use_container_width=True)
            st.dataframe(df, use_container_width=True)
            pick = st.selectbox("Select a stock for deep-dive analysis", df["sc_id"].tolist())
            if pick and st.button("Analyze Selected Stock", type="primary"):
                with st.spinner("Fetching fundamentals + running Claude analysis..."):
                    payload = client.get_fundamentals(pick)
                    try:
                        analysis = analyze_scan_payload(payload)
                    except RuntimeError as e:
                        st.error(str(e))
                        return
                render_analysis(analysis)

                ohlc = client.get_historical_ohlc(pick)
                if not ohlc.empty:
                    signals = detect_patterns(ohlc)
                    breakout, note = detect_volume_breakout(ohlc)
                    st.plotly_chart(candlestick_with_volume(ohlc, title=f"{pick} — Price Action",
                                                             signals=signals, breakout_flag=breakout),
                                     use_container_width=True)
                    st.caption(note)
                    st.caption(f"20d trend: {_trend_label(ohlc, 20)} | 50d trend: {_trend_label(ohlc, 50)}")
        elif df is not None:
            st.info("No stocks matched your filters. Try loosening them.")

if __name__ == "__main__":
    main()
