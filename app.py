"""
Stock Intelligence Analyst — fully offline build.
No external AI calls, no API keys. All financial analysis is done with
deterministic PDF parsing (pdfplumber), regex/table extraction, and
plain Python math for ratios and grading.

Optional: a free market-data scanner (Yahoo Finance / Moneycontrol) is
included for live price/fundamentals lookups. That still needs internet
access to fetch quotes, but makes no AI calls of any kind.
"""

from __future__ import annotations
import io, re, time
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
    import yfinance as yf
    _HAS_YFINANCE = True
except ImportError:
    _HAS_YFINANCE = False

# =============================================================================
# 1. CONFIG — no API keys anywhere in this app
# =============================================================================
def secret(key: str, default: str = "") -> str:
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default

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
    min_revenue_growth: float = 10.0
    min_eps_growth: float = 10.0
    min_roe: float = 15.0
    min_roce: float = 15.0
    max_debt_to_equity: float = 1.0
    min_interest_coverage: float = 3.0
    net_margin_floor: float = 5.0
    min_current_ratio: float = 1.0

THRESHOLDS = Thresholds()

THEME = {"bg": "#0E1117", "panel": "#161B22", "border": "#262D38", "text": "#E6EDF3",
          "muted": "#8B949E", "green": "#26A69A", "red": "#EF5350", "amber": "#D29922",
          "accent": "#4C8DFF"}
GRADE_COLORS = {"EXCELLENT": "#26A69A", "GOOD": "#4C8DFF", "BAD/RISKY": "#D29922",
                 "LOSS-MAKING / DESTRUCTIVE": "#EF5350"}


# =============================================================================
# 2. PDF PARSER — pdfplumber text + table extraction, page-tagged
# =============================================================================
@dataclass
class PageContent:
    page_number: int
    text: str
    tables: List[List[List[Optional[str]]]] = field(default_factory=list)

@dataclass
class ParsedDocument:
    filename: str
    total_pages: int
    pages: List[PageContent]

    def provenance_index(self) -> str:
        lines = [f"Document: {self.filename} ({self.total_pages} pages)"]
        for p in self.pages:
            tbl_note = f", {len(p.tables)} table(s)" if p.tables else ""
            preview = (p.text[:80] + "...") if len(p.text) > 80 else p.text
            lines.append(f"  Page {p.page_number}{tbl_note}: {preview.replace(chr(10), ' ').strip()}")
        return "\n".join(lines)


def parse_pdf(file_bytes: bytes, filename: str) -> ParsedDocument:
    pages: List[PageContent] = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            try:
                tables = page.extract_tables() or []
            except Exception:
                tables = []
            pages.append(PageContent(i, text, tables))
    if not any(p.text.strip() for p in pages):
        raise ValueError(f"Could not extract text from '{filename}' — it may be a scanned image PDF "
                          f"(this app does not include OCR).")
    return ParsedDocument(filename, len(pages), pages)


# =============================================================================
# 3. FINANCIAL LINE-ITEM EXTRACTION — regex + table based, no AI
# =============================================================================
@dataclass
class PeriodValue:
    value: float
    page: int
    snippet: str
    period_label: str  # "latest" | "previous"

NUMBER_RE = re.compile(r"\(?-?₹?\s?\d[\d,]*(?:\.\d+)?\)?")

def parse_number(tok: str) -> Optional[float]:
    tok = tok.strip()
    neg = tok.startswith("(") and tok.endswith(")")
    tok = tok.strip("()")
    tok = tok.replace(",", "").replace("₹", "").replace("Rs.", "").replace("INR", "").replace("%", "").strip()
    if tok in ("", "-", "--", "NA", "N/A", "."):
        return None
    try:
        val = float(tok)
    except ValueError:
        return None
    return -val if neg else val


# Patterns are tried IN ORDER per metric — the first pattern that produces any
# hit wins, so more specific labels (e.g. "revenue from operations") take
# priority over generic ones (e.g. "total income") when both appear.
LINE_ITEM_PATTERNS: Dict[str, List[str]] = {
    "revenue": [r"revenue\s+from\s+operations", r"net\s+sales", r"\bturnover\b", r"total\s+income"],
    "gross_profit": [r"gross\s+profit"],
    "operating_profit": [r"operating\s+profit", r"\bebit\b", r"profit\s+before\s+interest\s+and\s+tax"],
    "net_profit": [r"profit\s+for\s+the\s+year", r"profit\s+after\s+tax", r"\bpat\b", r"net\s+profit"],
    "total_assets": [r"total\s+assets"],
    "total_liabilities": [r"total\s+liabilities"],
    "total_equity": [r"total\s+equity", r"shareholders'?\s+funds", r"\bnet\s+worth\b"],
    "current_assets": [r"total\s+current\s+assets", r"\bcurrent\s+assets\b"],
    "current_liabilities": [r"total\s+current\s+liabilities", r"\bcurrent\s+liabilities\b"],
    "total_debt": [r"total\s+borrowings", r"total\s+debt"],
    "interest_expense": [r"finance\s+costs?", r"interest\s+expense"],
    "operating_cash_flow": [r"net\s+cash\s+(?:generated\s+from|from)\s+operating\s+activities",
                             r"cash\s+flow\s+from\s+operating\s+activities"],
    "capex": [r"purchase\s+of\s+property,?\s*plant", r"capital\s+expenditure"],
    "eps": [r"earnings\s+per\s+share"],
}

METRIC_LABELS = {
    "revenue": "Revenue", "gross_profit": "Gross Profit", "operating_profit": "Operating Profit (EBIT)",
    "net_profit": "Net Profit", "total_assets": "Total Assets", "total_liabilities": "Total Liabilities",
    "total_equity": "Total Equity / Net Worth", "current_assets": "Current Assets",
    "current_liabilities": "Current Liabilities", "total_debt": "Total Debt / Borrowings",
    "interest_expense": "Interest Expense (Finance Costs)", "operating_cash_flow": "Operating Cash Flow",
    "capex": "Capital Expenditure", "eps": "Earnings Per Share (EPS)",
}


def _find_with_patterns(pages: List[PageContent], patterns: List[str]) -> List[PeriodValue]:
    for pat_str in patterns:
        pat = re.compile(pat_str, re.IGNORECASE)
        hits: List[PeriodValue] = []
        for page in pages:
            lines = page.text.split("\n") if page.text else []
            for idx, line in enumerate(lines):
                low = line.lower()
                if "page " in low and " of " in low:
                    continue
                m = pat.search(line)
                if not m:
                    continue
                remainder = line[m.end():]
                nums = [n for n in (parse_number(t) for t in NUMBER_RE.findall(remainder)) if n is not None]
                if not nums and idx + 1 < len(lines):
                    nums = [n for n in (parse_number(t) for t in NUMBER_RE.findall(lines[idx + 1])) if n is not None]
                for j, n in enumerate(nums[:2]):
                    hits.append(PeriodValue(n, page.page_number, line.strip()[:140], "latest" if j == 0 else "previous"))
            for table in page.tables:
                for row in table:
                    if not row or row[0] is None:
                        continue
                    if not pat.search(str(row[0])):
                        continue
                    nums = [n for n in (parse_number(str(c)) for c in row[1:] if c is not None) if n is not None]
                    for j, n in enumerate(nums[:2]):
                        hits.append(PeriodValue(n, page.page_number,
                                                 " | ".join(str(c) for c in row if c is not None)[:140],
                                                 "latest" if j == 0 else "previous"))
        if hits:
            return hits
    return []


@dataclass
class ExtractedField:
    metric: str
    label: str
    latest: Optional[float] = None
    previous: Optional[float] = None
    latest_provenance: Optional[PeriodValue] = None
    previous_provenance: Optional[PeriodValue] = None
    found: bool = False


def extract_all_metrics(pages: List[PageContent]) -> Dict[str, ExtractedField]:
    results: Dict[str, ExtractedField] = {}
    for metric, patterns in LINE_ITEM_PATTERNS.items():
        hits = _find_with_patterns(pages, patterns)
        field_result = ExtractedField(metric=metric, label=METRIC_LABELS[metric])
        latest_hits = [h for h in hits if h.period_label == "latest"]
        prev_hits = [h for h in hits if h.period_label == "previous"]
        if latest_hits:
            field_result.latest = latest_hits[0].value
            field_result.latest_provenance = latest_hits[0]
            field_result.found = True
        if prev_hits:
            field_result.previous = prev_hits[0].value
            field_result.previous_provenance = prev_hits[0]
        results[metric] = field_result
    return results


# =============================================================================
# 4. GOVERNANCE / RED-FLAG KEYWORD SCANNER — no AI, plain keyword matching
# =============================================================================
@dataclass
class GovernanceFlag:
    category: str
    severity: str  # "low" | "medium" | "high"
    keyword: str
    page: int
    snippet: str

GOVERNANCE_KEYWORDS = [
    ("Auditor Qualification", "high", [r"qualified\s+opinion", r"adverse\s+opinion", r"disclaimer\s+of\s+opinion",
                                        r"except\s+for", r"material\s+uncertainty"]),
    ("Related Party Transactions", "medium", [r"related\s+part(?:y|ies)\s+transactions?"]),
    ("Pending Litigation", "medium", [r"pending\s+litigation", r"material\s+litigation", r"show[\s-]cause\s+notice",
                                       r"contingent\s+liabilit(?:y|ies)"]),
    ("Management Change", "medium", [r"resigned\s+as", r"resignation\s+of", r"ceased\s+to\s+be\s+a?\s*director",
                                      r"removal\s+of\s+director"]),
    ("Promoter Pledge", "high", [r"pledge\s+of\s+shares", r"shares?\s+pledged", r"encumbrance\s+of\s+shares"]),
    ("Accounting Policy Change", "medium", [r"change\s+in\s+accounting\s+policy", r"restat(?:ed|ement)\s+of"]),
]

def scan_governance_flags(pages: List[PageContent]) -> List[GovernanceFlag]:
    flags: List[GovernanceFlag] = []
    seen = set()
    for page in pages:
        text = page.text or ""
        for category, severity, patterns in GOVERNANCE_KEYWORDS:
            for pat_str in patterns:
                for m in re.finditer(pat_str, text, re.IGNORECASE):
                    start = max(0, m.start() - 80)
                    end = min(len(text), m.end() + 80)
                    snippet = text[start:end].replace("\n", " ").strip()
                    key = (category, page.page_number, snippet[:50])
                    if key in seen:
                        continue
                    seen.add(key)
                    flags.append(GovernanceFlag(category, severity, m.group(0), page.page_number, snippet))
    return flags


# =============================================================================
# 5. RATIO CALCULATIONS & DETERMINISTIC GRADING — pure Python, no AI
# =============================================================================
@dataclass
class RatioResult:
    metrics: Dict[str, Any]
    grade: str
    grade_rationale: str
    growth_commentary: str
    cash_flow_flag: Optional[str]


def pct_growth(latest: Optional[float], previous: Optional[float]) -> Optional[float]:
    if latest is None or previous is None or previous == 0:
        return None
    return round(((latest / previous) - 1) * 100, 2)


def safe_div(n: Optional[float], d: Optional[float]) -> Optional[float]:
    if n is None or d is None or d == 0:
        return None
    return n / d


def grade_from_metrics(revenue_growth_pct=None, roe_pct=None, roce_pct=None, net_margin_pct=None,
                        interest_coverage=None, debt_to_equity=None, net_profit=None,
                        cash_flow_flag=None) -> Tuple[str, str]:
    """
    Shared deterministic grading rubric. Works whether the metrics came from
    parsing raw PDF line items or directly from a ratio-based data provider
    (e.g. Yahoo Finance, which supplies ratios rather than statement rows).
    """
    if net_profit is not None and net_profit < 0:
        return "LOSS-MAKING / DESTRUCTIVE", f"The company reported a net loss of {net_profit:,.2f} for the period."
    if cash_flow_flag and "CRITICAL" in cash_flow_flag:
        return "LOSS-MAKING / DESTRUCTIVE", "Reported profit is not backed by operating cash flow (see Cash Flag)."

    checks = [
        (revenue_growth_pct, THRESHOLDS.min_revenue_growth, "revenue growth"),
        (roe_pct, THRESHOLDS.min_roe, "ROE"),
        (roce_pct, THRESHOLDS.min_roce, "ROCE"),
        (net_margin_pct, THRESHOLDS.net_margin_floor, "net margin"),
        (interest_coverage, THRESHOLDS.min_interest_coverage, "interest coverage"),
    ]
    available = [(v, t, name) for v, t, name in checks if v is not None]
    if not available:
        return "GOOD", "Not enough metrics were available to score fundamentals confidently; showing a neutral default grade."

    passes = [name for v, t, name in available if v >= t]
    de_ok = debt_to_equity is None or debt_to_equity <= THRESHOLDS.max_debt_to_equity
    pass_ratio = len(passes) / len(available)

    if pass_ratio >= 0.8 and de_ok:
        grade = "EXCELLENT"
    elif pass_ratio >= 0.5:
        grade = "GOOD"
    else:
        grade = "BAD/RISKY"
    if debt_to_equity is not None and debt_to_equity > THRESHOLDS.max_debt_to_equity * 2.5 and grade == "EXCELLENT":
        grade = "BAD/RISKY"

    rationale = f"{len(passes)}/{len(available)} tracked metrics ({', '.join(passes) if passes else 'none'}) met policy thresholds."
    if debt_to_equity is not None:
        rationale += f" Debt/Equity is {debt_to_equity}x (policy ceiling {THRESHOLDS.max_debt_to_equity}x)."
    return grade, rationale


def compute_ratios_and_grade(fields: Dict[str, ExtractedField],
                              market_price: Optional[float] = None,
                              shares_outstanding: Optional[float] = None) -> RatioResult:
    def latest(m):
        return fields[m].latest if m in fields else None
    def previous(m):
        return fields[m].previous if m in fields else None

    revenue, revenue_prev = latest("revenue"), previous("revenue")
    gross_profit = latest("gross_profit")
    op_profit = latest("operating_profit")
    net_profit, net_profit_prev = latest("net_profit"), previous("net_profit")
    total_assets = latest("total_assets")
    total_liabilities = latest("total_liabilities")
    total_equity = latest("total_equity")
    current_assets = latest("current_assets")
    current_liabilities = latest("current_liabilities")
    total_debt = latest("total_debt")
    interest_expense = latest("interest_expense")
    ocf = latest("operating_cash_flow")
    capex = latest("capex")
    eps, eps_prev = latest("eps"), previous("eps")

    revenue_growth_pct = pct_growth(revenue, revenue_prev)
    eps_growth_pct = pct_growth(eps, eps_prev)
    gross_margin_pct = round(safe_div(gross_profit, revenue) * 100, 2) if safe_div(gross_profit, revenue) is not None else None
    operating_margin_pct = round(safe_div(op_profit, revenue) * 100, 2) if safe_div(op_profit, revenue) is not None else None
    net_margin_pct = round(safe_div(net_profit, revenue) * 100, 2) if safe_div(net_profit, revenue) is not None else None
    roe_pct = round(safe_div(net_profit, total_equity) * 100, 2) if safe_div(net_profit, total_equity) is not None else None
    capital_employed = (total_assets - current_liabilities) if (total_assets is not None and current_liabilities is not None) else None
    roce_pct = round(safe_div(op_profit, capital_employed) * 100, 2) if safe_div(op_profit, capital_employed) is not None else None
    debt_to_equity = round(safe_div(total_debt, total_equity), 2) if safe_div(total_debt, total_equity) is not None else None
    interest_coverage = round(safe_div(op_profit, interest_expense), 2) if safe_div(op_profit, interest_expense) is not None else None
    current_ratio = round(safe_div(current_assets, current_liabilities), 2) if safe_div(current_assets, current_liabilities) is not None else None
    free_cash_flow = (ocf - capex) if (ocf is not None and capex is not None) else None
    pe_ratio = round(safe_div(market_price, eps), 2) if (market_price and eps) else None

    cash_flow_flag = None
    if net_profit is not None and ocf is not None:
        if net_profit > 0 and ocf < 0:
            cash_flow_flag = (f"CRITICAL: Reported net profit of {net_profit:,.2f} is positive while operating "
                               f"cash flow is negative ({ocf:,.2f}). Profit is not cash-backed.")
        elif net_profit > 0 and 0 < ocf < 0.5 * net_profit:
            cash_flow_flag = (f"CAUTION: Operating cash flow ({ocf:,.2f}) is under 50% of net profit "
                               f"({net_profit:,.2f}) — weak cash conversion.")

    metrics = {
        "revenue_growth_pct": revenue_growth_pct, "eps_growth_pct": eps_growth_pct,
        "gross_margin_pct": gross_margin_pct, "operating_margin_pct": operating_margin_pct,
        "net_margin_pct": net_margin_pct, "roe_pct": roe_pct, "roce_pct": roce_pct,
        "debt_to_equity": debt_to_equity, "interest_coverage": interest_coverage,
        "current_ratio": current_ratio, "free_cash_flow": free_cash_flow, "pe_ratio": pe_ratio,
        "net_profit": net_profit, "revenue": revenue,
    }

    grade, rationale = grade_from_metrics(revenue_growth_pct=revenue_growth_pct, roe_pct=roe_pct, roce_pct=roce_pct,
                                          net_margin_pct=net_margin_pct, interest_coverage=interest_coverage,
                                          debt_to_equity=debt_to_equity, net_profit=net_profit,
                                          cash_flow_flag=cash_flow_flag)

    parts = []
    if revenue_growth_pct is not None:
        parts.append(f"Revenue moved {revenue_growth_pct:+.1f}% year-over-year.")
    if eps_growth_pct is not None:
        parts.append(f"EPS moved {eps_growth_pct:+.1f}% year-over-year.")
    if net_margin_pct is not None:
        parts.append(f"Net margin stands at {net_margin_pct:.1f}%.")
    if roe_pct is not None:
        parts.append(f"ROE of {roe_pct:.1f}% is {'above' if roe_pct >= THRESHOLDS.min_roe else 'below'} "
                      f"the {THRESHOLDS.min_roe}% structural benchmark.")
    if not parts:
        parts.append("Not enough periods/line items were found in this document to describe a trend — "
                      "only single-period figures were available.")
    growth_commentary = " ".join(parts) + (" Note: this is only a year-over-year comparison, not a "
                                            "multi-year (3-5yr) CAGR, since this document contains at most two periods "
                                            "per line item.")

    return RatioResult(metrics=metrics, grade=grade, grade_rationale=rationale,
                        growth_commentary=growth_commentary, cash_flow_flag=cash_flow_flag)


def apply_governance_override(result: RatioResult, flags: List[GovernanceFlag]) -> RatioResult:
    high = sum(1 for f in flags if f.severity == "high")
    if high >= 2 and result.grade not in ("LOSS-MAKING / DESTRUCTIVE",):
        result.grade = "BAD/RISKY"
        result.grade_rationale += f" Capped at BAD/RISKY due to {high} high-severity governance flags."
    elif high == 1 and result.grade == "EXCELLENT":
        result.grade = "GOOD"
        result.grade_rationale += " Capped at GOOD due to 1 high-severity governance flag."
    return result


METHODOLOGY_FORMULAS = [
    {"metric": "Revenue / EPS growth (YoY)", "formula": "((Latest / Previous) - 1) × 100"},
    {"metric": "Gross Margin", "formula": "Gross Profit / Revenue × 100"},
    {"metric": "Operating Margin", "formula": "Operating Profit (EBIT) / Revenue × 100"},
    {"metric": "Net Margin", "formula": "Net Profit / Revenue × 100"},
    {"metric": "ROE", "formula": "Net Profit / Total Equity × 100"},
    {"metric": "ROCE", "formula": "EBIT / (Total Assets - Current Liabilities) × 100"},
    {"metric": "Debt-to-Equity", "formula": "Total Debt / Total Equity"},
    {"metric": "Interest Coverage", "formula": "EBIT / Interest Expense"},
    {"metric": "Current Ratio", "formula": "Current Assets / Current Liabilities"},
    {"metric": "Free Cash Flow", "formula": "Operating Cash Flow - Capex"},
    {"metric": "P/E Ratio", "formula": "Market Price per Share / EPS (requires manual market price input)"},
    {"metric": "Cash-Profit Divergence Flag", "formula": "Flag if Net Profit > 0 AND Operating Cash Flow < 0"},
]


# =============================================================================
# 6. MARKET DATA (optional) — Yahoo Finance / Moneycontrol. No AI calls.
#    Still requires internet access to fetch live quotes.
# =============================================================================
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
    if not _HAS_YFINANCE:
        return pd.DataFrame(), "The 'yfinance' package is not installed."
    try:
        hist = yf.Ticker(ticker).history(period=period, auto_adjust=False)
        if hist is None or hist.empty:
            return pd.DataFrame(), f"Yahoo Finance returned no data for '{ticker}' (NSE tickers need a .NS suffix)."
        df = hist.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
        df.columns = ["date", "open", "high", "low", "close", "volume"]
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        return df, None
    except Exception as e:
        return pd.DataFrame(), f"Yahoo Finance request failed for '{ticker}': {e}"


def yf_get_fundamentals(ticker: str) -> Tuple[Dict[str, Any], Optional[str]]:
    if not _HAS_YFINANCE:
        return {}, "The 'yfinance' package is not installed."
    try:
        info = yf.Ticker(ticker).info
        if not info or (info.get("regularMarketPrice") is None and info.get("longName") is None):
            return {}, f"Yahoo Finance returned no fundamentals for '{ticker}'."
        f = {
            "roe_pct": round((info.get("returnOnEquity") or 0) * 100, 2) if info.get("returnOnEquity") else None,
            "net_margin_pct": round((info.get("profitMargins") or 0) * 100, 2) if info.get("profitMargins") else None,
            "operating_margin_pct": round((info.get("operatingMargins") or 0) * 100, 2) if info.get("operatingMargins") else None,
            "gross_margin_pct": round((info.get("grossMargins") or 0) * 100, 2) if info.get("grossMargins") else None,
            "debt_to_equity": round((info.get("debtToEquity") or 0) / 100, 2) if info.get("debtToEquity") else None,
            "revenue_growth_pct": round((info.get("revenueGrowth") or 0) * 100, 2) if info.get("revenueGrowth") else None,
            "eps_growth_pct": round((info.get("earningsGrowth") or 0) * 100, 2) if info.get("earningsGrowth") else None,
            "trailing_pe": info.get("trailingPE"), "market_cap": info.get("marketCap"),
        }
        return {"sc_id": ticker, "company_name": info.get("longName", ticker), "financials": f}, None
    except Exception as e:
        return {}, f"Yahoo Finance fundamentals request failed for '{ticker}': {e}"


def yf_scan_universe(filters: Dict[str, Any], limit: int = 50,
                      tickers=NIFTY50_TICKERS) -> Tuple[pd.DataFrame, List[str]]:
    rows, errors = [], []
    for sc_id, name, sector in tickers:
        fund, err = yf_get_fundamentals(sc_id)
        if err or not fund:
            errors.append(f"{sc_id}: {err}")
            continue
        f = fund["financials"]
        rows.append({"sc_id": sc_id, "company_name": name, "sector": sector,
                     "revenue_growth_pct": f.get("revenue_growth_pct"), "eps_growth_pct": f.get("eps_growth_pct"),
                     "roe_pct": f.get("roe_pct"), "roce_pct": f.get("roe_pct"),
                     "debt_to_equity": f.get("debt_to_equity"), "net_margin_pct": f.get("net_margin_pct"),
                     "pe_ratio": f.get("trailing_pe")})
    if not rows:
        return pd.DataFrame(columns=["sc_id", "company_name", "sector", "revenue_growth_pct", "eps_growth_pct",
                                       "roe_pct", "roce_pct", "debt_to_equity", "net_margin_pct", "pe_ratio"]), errors
    df = pd.DataFrame(rows).dropna(subset=["roe_pct"], how="all")
    if df.empty:
        return df, errors
    if filters.get("min_roe"):
        df = df[df["roe_pct"].fillna(-1) >= filters["min_roe"]]
    if filters.get("max_de") is not None:
        df = df[df["debt_to_equity"].fillna(0) <= filters["max_de"]]
    if filters.get("min_revenue_growth"):
        df = df[df["revenue_growth_pct"].fillna(-100) >= filters["min_revenue_growth"]]
    return df.sort_values("roe_pct", ascending=False, na_position="last").head(limit).reset_index(drop=True), errors


ENDPOINTS = {
    "stock_fundamentals": "{api_base}/mcapi/v1/stock/fundamentals",
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
        retries, backoff = self.cfg.get("max_retries", 3), self.cfg.get("retry_backoff_seconds", 1.5)
        last_err = None
        for attempt in range(retries):
            try:
                resp = self.session.get(url, headers=self._headers(), params=params,
                                         timeout=self.cfg.get("request_timeout", 15))
                if resp.status_code in (401, 403):
                    return None, f"Moneycontrol rejected the session (HTTP {resp.status_code})."
                resp.raise_for_status()
                return resp.json(), None
            except Exception as e:
                last_err = str(e)
                time.sleep(backoff * (attempt + 1))
        return None, f"Moneycontrol request failed after {retries} attempts ({last_err})."

    def get_historical_ohlc(self, sc_id, resolution="D", from_ts=None, to_ts=None) -> pd.DataFrame:
        to_ts = to_ts or int(time.time())
        from_ts = from_ts or to_ts - 365 * 24 * 3600
        if not self.is_authenticated():
            return pd.DataFrame()
        url = ENDPOINTS["historical_ohlc"].format(priceapi_base=self.cfg["priceapi_base_url"])
        raw, err = self._request(url, params={"symbol": sc_id, "resolution": resolution, "from": from_ts, "to": to_ts})
        if err or not raw or "t" not in raw:
            return pd.DataFrame()
        df = pd.DataFrame({"date": pd.to_datetime(raw["t"], unit="s"), "open": raw.get("o", []),
                            "high": raw.get("h", []), "low": raw.get("l", []),
                            "close": raw.get("c", []), "volume": raw.get("v", [])})
        return df.sort_values("date").reset_index(drop=True)


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


def detect_patterns(df: pd.DataFrame, lookback=60) -> List[PatternSignal]:
    d = df.tail(lookback).reset_index(drop=True)
    signals = []
    body = (d["close"] - d["open"]).abs()
    rng = (d["high"] - d["low"]).replace(0, np.nan)
    for i in range(2, len(d)):
        o, h, l, c = d.loc[i, ["open", "high", "low", "close"]]
        po, pc = d.loc[i - 1, ["open", "close"]]
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
            signals.append(PatternSignal(d.loc[i, "date"], pattern, direction, "Upper wick ≥ 2× body.", i))
        if pc < po and c > o and o <= pc and c >= po and b > body.iloc[i - 1]:
            signals.append(PatternSignal(d.loc[i, "date"], "Bullish Engulfing", "bullish", "Body engulfs prior red body.", i))
        if pc > po and c < o and o >= pc and c <= po and b > body.iloc[i - 1]:
            signals.append(PatternSignal(d.loc[i, "date"], "Bearish Engulfing", "bearish", "Body engulfs prior green body.", i))
        if r > 0 and b <= 0.08 * r:
            signals.append(PatternSignal(d.loc[i, "date"], "Doji", "neutral", "Body ≤ 8% of range.", i))
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
        return True, (f"Close of {last['close']:.2f} breaks the {window}-day high of {last_high:.2f} on "
                       f"{last['volume']/last_avg_vol:.1f}x average volume.")
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


def evaluate_technicals(df: pd.DataFrame) -> TechnicalVerdict:
    signals = detect_patterns(df)
    breakout_flag, breakout_note = detect_volume_breakout(df)
    trend_20, trend_50 = _trend_label(df, 20), _trend_label(df, 50)
    recent_vol = df["volume"].tail(10).mean()
    baseline_vol = df["volume"].tail(60).mean() if len(df) >= 60 else df["volume"].mean()
    volume_trend = ("rising" if baseline_vol and recent_vol > 1.2 * baseline_vol else
                     "falling" if baseline_vol and recent_vol < 0.8 * baseline_vol else "stable")
    return TechnicalVerdict(signals, breakout_flag, breakout_note, trend_20, trend_50, volume_trend)


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
    fig = go.Figure(go.Scatter(x=df["revenue_growth_pct"], y=df["roce_pct"], mode="markers+text",
        text=df["sc_id"], textposition="top center",
        marker=dict(size=df["roe_pct"].clip(lower=1), sizemode="area",
                    sizeref=2. * df["roe_pct"].max() / (40. ** 2), sizemin=6,
                    color=df["debt_to_equity"], colorscale="RdYlGn_r", showscale=True, colorbar=dict(title="D/E")),
        name="Universe"))
    fig.update_layout(template="plotly_dark", paper_bgcolor=THEME["bg"], plot_bgcolor=THEME["bg"],
        font=dict(color=THEME["text"]), height=480, margin=dict(l=40, r=20, t=30, b=30),
        xaxis_title="Revenue Growth (%)", yaxis_title="ROCE (%)",
        title="Scanner Universe: Growth vs Capital Efficiency (bubble=ROE, color=D/E)")
    return fig


# =============================================================================
# 9. STREAMLIT UI
# =============================================================================
st.set_page_config(page_title="Stock Intelligence Analyst (Offline)", page_icon="📈", layout="wide",
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
    st.caption("100% offline analysis — no API keys, no external AI calls.")
    mode = st.radio("Choose module", ["📄 Document Upload Analysis", "🔎 Market Data Scanner (optional, needs internet)"],
                     label_visibility="collapsed")
    st.divider()
    st.subheader("Policy Thresholds")
    st.write(f"- Revenue/EPS growth ≥ **{THRESHOLDS.min_revenue_growth}%**")
    st.write(f"- ROE / ROCE ≥ **{THRESHOLDS.min_roe}% / {THRESHOLDS.min_roce}%**")
    st.write(f"- Debt/Equity ≤ **{THRESHOLDS.max_debt_to_equity}x**")
    st.write(f"- Interest Coverage ≥ **{THRESHOLDS.min_interest_coverage}x**")
    st.divider()
    st.success("✅ No API key required. All document analysis runs locally in this app.")


def render_metrics_and_grade(result: RatioResult, flags: List[GovernanceFlag], fields: Dict[str, ExtractedField],
                              ohlc_df: Optional[pd.DataFrame], provenance_extra=None):
    m = result.metrics
    tech_verdict = evaluate_technicals(ohlc_df) if ohlc_df is not None and not ohlc_df.empty else None

    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown(grade_badge(result.grade), unsafe_allow_html=True)
        st.write(result.grade_rationale)
    with c2:
        if result.cash_flow_flag:
            st.error(f"💸 **Cash Flag:** {result.cash_flow_flag}")
        else:
            st.success("💵 No profit/cash-flow divergence detected.")

    tabs = st.tabs(["📊 Fundamentals", "🏛️ Governance & Risk", "🕯️ Technical Chart",
                     "🔮 Growth Commentary", "🧾 Methodology & Provenance"])

    with tabs[0]:
        cols = st.columns(4)
        metric_map = [("Revenue Growth (YoY)", m.get("revenue_growth_pct"), "%"),
                      ("EPS Growth (YoY)", m.get("eps_growth_pct"), "%"),
                      ("ROE", m.get("roe_pct"), "%"), ("ROCE", m.get("roce_pct"), "%"),
                      ("Net Margin", m.get("net_margin_pct"), "%"), ("Operating Margin", m.get("operating_margin_pct"), "%"),
                      ("Debt/Equity", m.get("debt_to_equity"), "x"), ("Interest Coverage", m.get("interest_coverage"), "x"),
                      ("Current Ratio", m.get("current_ratio"), "x"), ("P/E Ratio", m.get("pe_ratio"), "x")]
        for i, (label, val, unit) in enumerate(metric_map):
            with cols[i % 4]:
                st.metric(label, f"{val}{unit}" if val is not None else "N/A")
        margin_vals = [m.get("gross_margin_pct"), m.get("operating_margin_pct"), m.get("net_margin_pct")]
        if any(v is not None for v in margin_vals):
            st.plotly_chart(margin_comparison_chart(["Gross", "Operating", "Net"], margin_vals, THRESHOLDS.net_margin_floor),
                             use_container_width=True)
        if m.get("free_cash_flow") is not None:
            st.metric("Free Cash Flow (latest period)", f"{m['free_cash_flow']:,.2f}")

    with tabs[1]:
        if not flags:
            st.success("No governance/red-flag keywords matched in this document.")
        else:
            st.caption("Automated keyword scan — always verify context by reading the cited snippet in the source document.")
        for flag in flags:
            st.markdown(f'<div class="flag-{flag.severity}"><b>[{flag.severity.upper()}] {flag.category}:</b> '
                        f'"...{flag.snippet}..."<br><span style="color:{THEME["muted"]};font-size:0.85em;">'
                        f'Page {flag.page}, matched: "{flag.keyword}"</span></div>', unsafe_allow_html=True)

    with tabs[2]:
        if ohlc_df is None or ohlc_df.empty:
            st.info("No OHLC data available. Provide a ticker (Yahoo Finance format, e.g. RELIANCE.NS) to overlay a live chart — this requires internet access but not AI.")
        else:
            st.plotly_chart(candlestick_with_volume(ohlc_df, "Price Action", tech_verdict.signals, tech_verdict.breakout_flag),
                            use_container_width=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("20-day Trend", tech_verdict.trend_20d)
            c2.metric("50-day Trend", tech_verdict.trend_50d)
            c3.metric("Volume Trend", tech_verdict.volume_trend)
            (st.success if tech_verdict.breakout_flag else st.caption)(tech_verdict.breakout_note)
            if tech_verdict.signals:
                st.dataframe(pd.DataFrame([{"Date": s.date.date(), "Pattern": s.pattern, "Direction": s.direction,
                    "Rule": s.rule} for s in tech_verdict.signals]), use_container_width=True, hide_index=True)

    with tabs[3]:
        st.write(result.growth_commentary)
        st.caption("Generated with fixed rule-based templates from the extracted numbers — no AI model involved.")

    with tabs[4]:
        st.subheader("Formulas used")
        st.dataframe(pd.DataFrame(METHODOLOGY_FORMULAS), use_container_width=True, hide_index=True)
        st.subheader("Extracted line items — source page & matched text")
        prov_rows = []
        for key, fld in fields.items():
            if fld.latest_provenance:
                prov_rows.append({"Metric": fld.label, "Period": "Latest", "Value": fld.latest,
                                  "Page": fld.latest_provenance.page, "Matched text": fld.latest_provenance.snippet})
            if fld.previous_provenance:
                prov_rows.append({"Metric": fld.label, "Period": "Previous", "Value": fld.previous,
                                  "Page": fld.previous_provenance.page, "Matched text": fld.previous_provenance.snippet})
        if prov_rows:
            st.dataframe(pd.DataFrame(prov_rows), use_container_width=True, hide_index=True)
        else:
            st.caption("No line items were matched in this document.")
        if provenance_extra:
            st.subheader("Document index")
            st.text(provenance_extra)


def document_analysis_view():
    st.header("📄 Document Upload Analysis (Offline)")
    st.caption("Upload an annual report, financial statement, or balance sheet (PDF). All extraction and "
               "ratio math runs locally in this app — nothing is sent to any external AI service.")

    uploaded = st.file_uploader("Upload a PDF financial document", type=["pdf"])

    if uploaded:
        if st.session_state.get("_last_uploaded_name") != uploaded.name:
            file_bytes = uploaded.read()
            try:
                parsed = parse_pdf(file_bytes, uploaded.name)
            except Exception as e:
                st.error(str(e))
                return
            fields = extract_all_metrics(parsed.pages)
            flags = scan_governance_flags(parsed.pages)
            st.session_state["_last_uploaded_name"] = uploaded.name
            st.session_state["_parsed_doc"] = parsed
            st.session_state["_extracted_fields"] = fields
            st.session_state["_gov_flags"] = flags

        parsed = st.session_state["_parsed_doc"]
        fields: Dict[str, ExtractedField] = st.session_state["_extracted_fields"]
        flags: List[GovernanceFlag] = st.session_state["_gov_flags"]

        st.success(f"Extracted {parsed.total_pages} pages. Found {sum(f.found for f in fields.values())}/"
                  f"{len(fields)} line items automatically — review and correct below before calculating.")

        st.subheader("Review & confirm extracted figures")
        st.caption("Pre-filled from the document where a match was found. Edit any field that looks wrong "
                  "(regex extraction is best-effort, not perfect) — the ratios below are calculated from "
                  "whatever values are in these boxes when you click Calculate.")

        edited: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        cols = st.columns(2)
        for i, (key, fld) in enumerate(fields.items()):
            with cols[i % 2]:
                st.markdown(f"**{fld.label}**" + (" ✅" if fld.found else " ⚠️ not found"))
                c1, c2 = st.columns(2)
                latest_val = c1.number_input(f"Latest", value=float(fld.latest) if fld.latest is not None else 0.0,
                                              key=f"latest_{key}", format="%.2f")
                previous_val = c2.number_input(f"Previous", value=float(fld.previous) if fld.previous is not None else 0.0,
                                               key=f"previous_{key}", format="%.2f")
                edited[key] = (latest_val if latest_val != 0.0 or fld.latest is not None else None,
                              previous_val if previous_val != 0.0 or fld.previous is not None else None)

        st.subheader("Optional manual inputs")
        c1, c2 = st.columns(2)
        market_price = c1.number_input("Current market price per share (for P/E)", min_value=0.0, value=0.0, step=1.0)
        ticker_hint = c2.text_input("Optional: ticker for live chart overlay (Yahoo Finance format, e.g. RELIANCE.NS)")

        if st.button("🚀 Calculate Analysis", type="primary"):
            updated_fields = {}
            for key, fld in fields.items():
                latest_v, previous_v = edited[key]
                updated_fields[key] = ExtractedField(metric=fld.metric, label=fld.label, latest=latest_v,
                                                      previous=previous_v, latest_provenance=fld.latest_provenance,
                                                      previous_provenance=fld.previous_provenance, found=fld.found)
            result = compute_ratios_and_grade(updated_fields, market_price=market_price or None)
            result = apply_governance_override(result, flags)

            ohlc_df = None
            if ticker_hint.strip():
                ohlc_df, err = yf_get_ohlc(ticker_hint.strip())
                if err:
                    st.warning(err)

            provenance_extra = parsed.provenance_index()[:2000]
            st.session_state["_analysis_result"] = (result, flags, updated_fields, ohlc_df, provenance_extra)

    if "_analysis_result" in st.session_state:
        st.divider()
        render_metrics_and_grade(*st.session_state["_analysis_result"])


def market_scanner_view():
    st.header("🔎 Market Data Scanner")
    st.caption("Optional module — fetches live price/fundamentals from Yahoo Finance (or your own Moneycontrol "
               "session if configured). This needs internet access to reach the data provider, but makes no "
               "calls to any AI model — all grading below is the same deterministic rules engine used for documents.")

    mc_client = MoneycontrolClient()
    using_mc = mc_client.is_authenticated() and st.checkbox("Use Moneycontrol session instead of Yahoo Finance", value=False)

    st.subheader("Scan filters")
    c1, c2, c3 = st.columns(3)
    min_roe = c1.slider("Min ROE %", 0, 40, int(THRESHOLDS.min_roe))
    max_de = c2.slider("Max Debt/Equity", 0.0, 3.0, THRESHOLDS.max_debt_to_equity, step=0.1)
    min_rev_growth = c3.slider("Min Revenue Growth %", 0, 40, int(THRESHOLDS.min_revenue_growth))

    if st.button("🔍 Run Scan", type="primary"):
        filters = {"min_roe": min_roe, "max_de": max_de, "min_revenue_growth": min_rev_growth}
        with st.spinner("Scanning universe…"):
            results, errors = yf_scan_universe(filters, limit=50)
            st.session_state["scan_results"] = results
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
            st.dataframe(results, use_container_width=True, hide_index=True)
            st.plotly_chart(scan_scatter_chart(results), use_container_width=True)

            st.divider()
            st.subheader("Deep-dive on a scanned stock (deterministic rules, no AI)")
            pick = st.selectbox("Select a stock", results["sc_id"].tolist())
            if st.button(f"📊 Analyze {pick}"):
                row = results[results["sc_id"] == pick].iloc[0]
                st.info("Yahoo Finance supplies ratios directly (not raw statement line items), so this "
                        "deep-dive grades those ratios directly with the same rules used for documents. "
                        "ROCE is approximated using ROE since Yahoo Finance has no direct ROCE figure.")
                grade, rationale = grade_from_metrics(
                    revenue_growth_pct=row["revenue_growth_pct"], roe_pct=row["roe_pct"],
                    roce_pct=row["roce_pct"], net_margin_pct=row["net_margin_pct"],
                    debt_to_equity=row["debt_to_equity"],
                )
                result = RatioResult(
                    metrics={"revenue_growth_pct": row["revenue_growth_pct"], "eps_growth_pct": row["eps_growth_pct"],
                             "roe_pct": row["roe_pct"], "roce_pct": row["roce_pct"],
                             "net_margin_pct": row["net_margin_pct"], "debt_to_equity": row["debt_to_equity"],
                             "pe_ratio": row["pe_ratio"], "gross_margin_pct": None, "operating_margin_pct": None,
                             "interest_coverage": None, "current_ratio": None, "free_cash_flow": None},
                    grade=grade, grade_rationale=rationale,
                    growth_commentary=(f"Revenue growth of {row['revenue_growth_pct']}% and ROE of {row['roe_pct']}% "
                                       f"are sourced directly from Yahoo Finance's trailing fundamentals." if row["revenue_growth_pct"] is not None
                                       else "Limited trend data available from this data source."),
                    cash_flow_flag=None,
                )
                ohlc_df, err = yf_get_ohlc(pick)
                if err:
                    st.warning(err)
                empty_fields = {k: ExtractedField(metric=k, label=METRIC_LABELS[k], found=False) for k in LINE_ITEM_PATTERNS}
                st.session_state["_scan_deep_dive"] = (result, [], empty_fields, ohlc_df, None)

    if "_scan_deep_dive" in st.session_state:
        st.divider()
        render_metrics_and_grade(*st.session_state["_scan_deep_dive"])


try:
    if mode.startswith("📄"):
        document_analysis_view()
    else:
        market_scanner_view()
except Exception as e:
    st.error(f"Something went wrong: {e}")
    with st.expander("Technical details"):
        st.exception(e)
