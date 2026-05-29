"""
Umbrella RAG — Streamlit UI
5G NR Telecom Root Cause Analysis Assistant
"""
from __future__ import annotations

import time
from typing import Any

import requests
import streamlit as st

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

API_BASE_URL = "http://localhost:8000"
RCA_ENDPOINT = f"{API_BASE_URL}/rca"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"
REQUEST_TIMEOUT = 60  # seconds

APP_TITLE = "Umbrella RAG"
APP_TAGLINE = "5G NR Root Cause Analysis Assistant"
APP_ICON = "☂️"

COLOR_BG_PRIMARY = "#0A0E1A"
COLOR_BG_SECONDARY = "#111827"
COLOR_BG_TERTIARY = "#1A2235"
COLOR_BG_SIDEBAR = "#0D1117"
COLOR_BG_ERROR = "#1A0F0F"
COLOR_ACCENT_PRIMARY = "#00D4FF"
COLOR_ACCENT_SECONDARY = "#0EA5E9"
COLOR_ACCENT_SUCCESS = "#10B981"
COLOR_ACCENT_WARNING = "#F59E0B"
COLOR_ACCENT_ERROR = "#EF4444"
COLOR_ACCENT_VIOLET = "#A78BFA"
COLOR_ACCENT_PINK = "#F472B6"
COLOR_TEXT_PRIMARY = "#F1F5F9"
COLOR_TEXT_SECONDARY = "#94A3B8"
COLOR_TEXT_MUTED = "#475569"
COLOR_TEXT_SUBTLE = "#334155"
COLOR_TEXT_DIM = "#64748B"
COLOR_TEXT_LIGHT = "#CBD5E1"
COLOR_TEXT_STRONG = "#E2E8F0"
COLOR_BORDER = "#1E293B"
COLOR_BORDER_ACCENT = "#00D4FF22"
COLOR_ACCENT_SOFT = "#00D4FF11"
COLOR_ACCENT_HOVER = "#00D4FF44"
COLOR_ACCENT_GLOW_LIGHT = "#00D4FF33"
COLOR_ACCENT_GLOW_STRONG = "#00D4FF66"
COLOR_ERROR_BORDER = "#EF444433"
COLOR_ERROR_TEXT_SOFT = "#FCA5A5"

CHUNK_TYPE_COLORS = {
    "procedure": COLOR_ACCENT_PRIMARY,
    "definition": COLOR_ACCENT_VIOLET,
    "parameter": COLOR_ACCENT_WARNING,
    "summary": COLOR_ACCENT_SUCCESS,
    "example": COLOR_ACCENT_PINK,
    "general": COLOR_TEXT_SECONDARY,
}

SAMPLE_QUERIES = [
    "Why does handover fail in 5G NR?",
    "What causes RLF and how does the UE recover?",
    "What triggers T310 timer expiry?",
    "Why is PDSCH throughput degrading when CQI is low?",
    "What happens during RRC reconfiguration for handover?",
    "Why does RACH procedure fail?",
    "What causes radio link failure in NR?",
    "How does the gNB detect too-early handover?",
]


# ─────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────

def inject_css() -> None:
    """Inject custom CSS for the NOC dashboard aesthetic."""
    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Base ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: __COLOR_BG_PRIMARY__;
        color: __COLOR_TEXT_PRIMARY__;
    }

    .stApp {
        background-color: __COLOR_BG_PRIMARY__;
    }

    /* Hide Streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }

    /* ── Animations ── */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50%       { opacity: 0.4; }
    }

    @keyframes blink {
        0%, 100% { opacity: 1; }
        50%       { opacity: 0; }
    }

    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-16px); }
        to   { opacity: 1; transform: translateX(0); }
    }

    @keyframes scanline {
        0%   { transform: translateY(-100%); }
        100% { transform: translateY(100vh); }
    }

    @keyframes glow {
        0%, 100% { box-shadow: 0 0 8px __COLOR_ACCENT_GLOW_LIGHT__; }
        50%       { box-shadow: 0 0 24px __COLOR_ACCENT_GLOW_STRONG__; }
    }

    /* ── Hero Header ── */
    .hero-container {
        text-align: center;
        padding: 3rem 1rem 2rem;
        animation: fadeInUp 0.8s ease both;
        position: relative;
    }

    .hero-icon {
        font-size: 3rem;
        margin-bottom: 0.5rem;
        display: block;
        animation: fadeInUp 0.6s ease both;
    }

    .hero-title {
        font-family: 'Inter', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: __COLOR_TEXT_PRIMARY__;
        margin: 0;
    }

    .hero-title span {
        color: __COLOR_ACCENT_PRIMARY__;
    }

    .hero-tagline {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: __COLOR_TEXT_MUTED__;
        margin-top: 0.5rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .hero-cursor {
        display: inline-block;
        width: 2px;
        height: 1em;
        background: __COLOR_ACCENT_PRIMARY__;
        margin-left: 2px;
        vertical-align: middle;
        animation: blink 1s step-end infinite;
    }

    /* ── Status Indicator ── */
    .status-bar {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
        margin: 1rem auto;
        padding: 0.4rem 1rem;
        background: __COLOR_BG_SECONDARY__;
        border: 1px solid __COLOR_BORDER__;
        border-radius: 999px;
        width: fit-content;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: __COLOR_TEXT_SECONDARY__;
        animation: fadeInUp 1s ease both;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: __COLOR_ACCENT_SUCCESS__;
        animation: pulse 2s ease-in-out infinite;
    }

    .status-dot.offline {
        background: __COLOR_ACCENT_ERROR__;
        animation: none;
    }

    /* ── Query Input ── */
    .query-container {
        animation: fadeInUp 0.9s ease both;
    }

    /* Style Streamlit text area */
    .stTextArea textarea {
        background-color: __COLOR_BG_SECONDARY__ !important;
        border: 1px solid __COLOR_BORDER__ !important;
        border-radius: 12px !important;
        color: __COLOR_TEXT_PRIMARY__ !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 1rem !important;
        padding: 1rem !important;
        resize: none !important;
        transition: border-color 0.2s ease !important;
    }

    .stTextArea textarea:focus {
        border-color: __COLOR_ACCENT_PRIMARY__ !important;
        box-shadow: 0 0 0 2px __COLOR_BORDER_ACCENT__ !important;
        outline: none !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, __COLOR_ACCENT_PRIMARY__ 0%, __COLOR_ACCENT_SECONDARY__ 100%) !important;
        color: __COLOR_BG_PRIMARY__ !important;
        border: none !important;
        border-radius: 10px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 0.6rem 1.8rem !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        letter-spacing: 0.01em !important;
    }

    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 20px __COLOR_ACCENT_HOVER__ !important;
    }

    .stButton > button:active {
        transform: translateY(0) !important;
    }

    /* Secondary button (sample queries) */
    .sample-btn > button {
        background: __COLOR_BG_SECONDARY__ !important;
        color: __COLOR_TEXT_SECONDARY__ !important;
        border: 1px solid __COLOR_BORDER__ !important;
        border-radius: 8px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.75rem !important;
        padding: 0.3rem 0.8rem !important;
    }

    .sample-btn > button:hover {
        border-color: __COLOR_ACCENT_PRIMARY__ !important;
        color: __COLOR_ACCENT_PRIMARY__ !important;
        background: __COLOR_ACCENT_SOFT__ !important;
    }

    /* ── Answer Card ── */
    .answer-card {
        background: __COLOR_BG_SECONDARY__;
        border: 1px solid __COLOR_BORDER__;
        border-left: 3px solid __COLOR_ACCENT_PRIMARY__;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        animation: slideInLeft 0.5s ease both;
        position: relative;
        overflow: hidden;
    }

    .answer-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg, __COLOR_ACCENT_HOVER__, transparent);
    }

    .answer-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: __COLOR_ACCENT_PRIMARY__;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .answer-label::after {
        content: '';
        flex: 1;
        height: 1px;
        background: __COLOR_BORDER__;
    }

    .answer-text {
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        line-height: 1.7;
        color: __COLOR_TEXT_STRONG__;
    }

    .answer-text strong {
        color: __COLOR_ACCENT_PRIMARY__;
        font-weight: 600;
    }

    /* ── Metrics Row ── */
    .metrics-row {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
        flex-wrap: wrap;
    }

    .metric-chip {
        background: __COLOR_BG_TERTIARY__;
        border: 1px solid __COLOR_BORDER__;
        border-radius: 8px;
        padding: 0.4rem 0.8rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: __COLOR_TEXT_SECONDARY__;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }

    .metric-chip .value {
        color: __COLOR_TEXT_PRIMARY__;
        font-weight: 600;
    }

    /* ── Chunk Cards ── */
    .chunks-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: __COLOR_TEXT_MUTED__;
        margin: 1.5rem 0 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .chunks-label::after {
        content: '';
        flex: 1;
        height: 1px;
        background: __COLOR_BORDER__;
    }

    .chunk-card {
        background: __COLOR_BG_SECONDARY__;
        border: 1px solid __COLOR_BORDER__;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.75rem;
        animation: slideInLeft 0.4s ease both;
        transition: border-color 0.2s ease;
    }

    .chunk-card:hover {
        border-color: __COLOR_ACCENT_HOVER__;
    }

    .chunk-meta {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 0.6rem;
        flex-wrap: wrap;
    }

    .chunk-spec {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        font-weight: 600;
        color: __COLOR_ACCENT_PRIMARY__;
        background: __COLOR_ACCENT_SOFT__;
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
    }

    .chunk-section {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: __COLOR_TEXT_SECONDARY__;
    }

    .chunk-type-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        padding: 0.1rem 0.45rem;
        border-radius: 4px;
        border: 1px solid;
    }

    .chunk-score {
        margin-left: auto;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: __COLOR_TEXT_MUTED__;
    }

    .chunk-score .score-val {
        color: __COLOR_ACCENT_WARNING__;
        font-weight: 600;
    }

    .chunk-title {
        font-size: 0.85rem;
        font-weight: 500;
        color: __COLOR_TEXT_LIGHT__;
        margin-bottom: 0.4rem;
    }

    .chunk-text {
        font-size: 0.82rem;
        line-height: 1.6;
        color: __COLOR_TEXT_DIM__;
        display: -webkit-box;
        -webkit-line-clamp: 3;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }

    .chunk-page {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        color: __COLOR_TEXT_SUBTLE__;
        margin-top: 0.4rem;
    }

    /* ── Spinner / Loading ── */
    .analyzing-container {
        text-align: center;
        padding: 3rem 1rem;
        animation: fadeInUp 0.3s ease both;
    }

    .analyzing-icon {
        font-size: 2.5rem;
        animation: pulse 1.2s ease-in-out infinite;
        display: block;
        margin-bottom: 1rem;
    }

    .analyzing-text {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: __COLOR_TEXT_SECONDARY__;
        letter-spacing: 0.08em;
    }

    .analyzing-dots::after {
        content: '';
        animation: dots 1.5s steps(4, end) infinite;
    }

    @keyframes dots {
        0%   { content: ''; }
        25%  { content: '.'; }
        50%  { content: '..'; }
        75%  { content: '...'; }
    }

    /* ── Error Card ── */
    .error-card {
        background: __COLOR_BG_ERROR__;
        border: 1px solid __COLOR_ERROR_BORDER__;
        border-left: 3px solid __COLOR_ACCENT_ERROR__;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        color: __COLOR_ERROR_TEXT_SOFT__;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
    }

    /* ── Sidebar ── */
    .css-1d391kg, [data-testid="stSidebar"] {
        background-color: __COLOR_BG_SIDEBAR__ !important;
        border-right: 1px solid __COLOR_BORDER__ !important;
    }

    .sidebar-section {
        margin-bottom: 1.5rem;
    }

    .sidebar-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: __COLOR_TEXT_MUTED__;
        margin-bottom: 0.5rem;
    }

    /* ── History Item ── */
    .history-item {
        background: __COLOR_BG_SECONDARY__;
        border: 1px solid __COLOR_BORDER__;
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.4rem;
        cursor: pointer;
        transition: all 0.15s ease;
    }

    .history-item:hover {
        border-color: __COLOR_ACCENT_HOVER__;
        background: __COLOR_BG_TERTIARY__;
    }

    .history-query {
        font-size: 0.78rem;
        color: __COLOR_TEXT_SECONDARY__;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .history-time {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        color: __COLOR_TEXT_SUBTLE__;
        margin-top: 0.2rem;
    }

    /* ── Divider ── */
    hr {
        border: none;
        border-top: 1px solid __COLOR_BORDER__;
        margin: 1.5rem 0;
    }

    /* ── Selectbox / inputs ── */
    .stSelectbox select {
        background: __COLOR_BG_SECONDARY__ !important;
        color: __COLOR_TEXT_PRIMARY__ !important;
        border: 1px solid __COLOR_BORDER__ !important;
    }
    </style>
    """

    color_tokens = {
        "COLOR_BG_PRIMARY": COLOR_BG_PRIMARY,
        "COLOR_BG_SECONDARY": COLOR_BG_SECONDARY,
        "COLOR_BG_TERTIARY": COLOR_BG_TERTIARY,
        "COLOR_BG_SIDEBAR": COLOR_BG_SIDEBAR,
        "COLOR_BG_ERROR": COLOR_BG_ERROR,
        "COLOR_ACCENT_PRIMARY": COLOR_ACCENT_PRIMARY,
        "COLOR_ACCENT_SECONDARY": COLOR_ACCENT_SECONDARY,
        "COLOR_ACCENT_SUCCESS": COLOR_ACCENT_SUCCESS,
        "COLOR_ACCENT_WARNING": COLOR_ACCENT_WARNING,
        "COLOR_ACCENT_ERROR": COLOR_ACCENT_ERROR,
        "COLOR_TEXT_PRIMARY": COLOR_TEXT_PRIMARY,
        "COLOR_TEXT_SECONDARY": COLOR_TEXT_SECONDARY,
        "COLOR_TEXT_MUTED": COLOR_TEXT_MUTED,
        "COLOR_TEXT_SUBTLE": COLOR_TEXT_SUBTLE,
        "COLOR_TEXT_DIM": COLOR_TEXT_DIM,
        "COLOR_TEXT_LIGHT": COLOR_TEXT_LIGHT,
        "COLOR_TEXT_STRONG": COLOR_TEXT_STRONG,
        "COLOR_BORDER": COLOR_BORDER,
        "COLOR_BORDER_ACCENT": COLOR_BORDER_ACCENT,
        "COLOR_ACCENT_SOFT": COLOR_ACCENT_SOFT,
        "COLOR_ACCENT_HOVER": COLOR_ACCENT_HOVER,
        "COLOR_ACCENT_GLOW_LIGHT": COLOR_ACCENT_GLOW_LIGHT,
        "COLOR_ACCENT_GLOW_STRONG": COLOR_ACCENT_GLOW_STRONG,
        "COLOR_ERROR_BORDER": COLOR_ERROR_BORDER,
        "COLOR_ERROR_TEXT_SOFT": COLOR_ERROR_TEXT_SOFT,
    }

    for token, value in color_tokens.items():
        css = css.replace(f"__{token}__", value)

    st.markdown(css, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def render_hero() -> None:
    """Render the animated hero header."""
    st.markdown(
        f"""
        <div class="hero-container">
            <span class="hero-icon">{APP_ICON}</span>
            <h1 class="hero-title">
                Umbrella <span>RAG</span>
            </h1>
            <p class="hero-tagline">
                {APP_TAGLINE}
                <span class="hero-cursor"></span>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def check_api_health() -> tuple[bool, dict]:
    """Check if the FastAPI backend is reachable and return component health."""
    try:
        resp = requests.get(HEALTH_ENDPOINT, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return True, data.get("components", {})
        return False, {}
    except Exception:
        return False, {}


def render_status_bar(online: bool, components: dict) -> None:
    """Render the API status indicator."""
    if online:
        all_ok = all(components.values()) if components else True
        dot_class = "status-dot" if all_ok else "status-dot offline"
        label = "API ONLINE" if all_ok else "API DEGRADED"
        color = COLOR_ACCENT_SUCCESS if all_ok else COLOR_ACCENT_WARNING
    else:
        dot_class = "status-dot offline"
        label = "API OFFLINE"
        color = COLOR_ACCENT_ERROR

    comp_text = ""
    if components:
        parts = []
        for name, status in components.items():
            icon = "✓" if status else "✗"
            parts.append(f"{icon} {name}")
        comp_text = "  ·  " + "  ·  ".join(parts)

    st.markdown(
        f"""
        <div class="status-bar">
            <div class="{dot_class}"></div>
            <span style="color: {color}; font-weight: 600;">{label}</span>
            <span style="color: {COLOR_TEXT_SUBTLE};">{comp_text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics(response: dict) -> None:
    """Render the response metrics row."""
    latency_ms = response.get("latency_ms", 0)
    latency_s = f"{latency_ms / 1000:.2f}s"
    retrieved = response.get("retrieval_count", 0)
    reranked = response.get("reranked_count", 0)
    model = response.get("model", "unknown")
    expanded = response.get("expanded_query", "")
    original = response.get("query", "")
    was_expanded = expanded != original

    st.markdown(
        f"""
        <div class="metrics-row">
            <div class="metric-chip">
                ⚡ latency <span class="value">{latency_s}</span>
            </div>
            <div class="metric-chip">
                🔍 retrieved <span class="value">{retrieved}</span>
            </div>
            <div class="metric-chip">
                ⭐ reranked <span class="value">{reranked}</span>
            </div>
            <div class="metric-chip">
                🤖 model <span class="value">{model}</span>
            </div>
            {"" if not was_expanded else f'<div class="metric-chip">✨ expanded <span class="value" title="{expanded}">query</span></div>'}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_answer(response: dict) -> None:
    """Render the main RCA answer card."""
    answer = response.get("answer", "No answer returned.")

    st.markdown(
        f"""
        <div class="answer-card">
            <div class="answer-label">⚙ RCA Analysis</div>
            <div class="answer-text">{answer.replace(chr(10), "<br>")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chunk_type_badge(chunk_type: str) -> str:
    """Return HTML for a chunk type badge with appropriate color."""
    color = CHUNK_TYPE_COLORS.get(chunk_type, COLOR_TEXT_SECONDARY)
    return (
        f'<span class="chunk-type-badge" '
        f'style="color: {color}; border-color: {color}44; '
        f'background: {color}11;">'
        f"{chunk_type}</span>"
    )


def render_chunks(chunks: list[dict]) -> None:
    """Render the retrieved and reranked source chunks."""
    if not chunks:
        return

    st.markdown(
        '<div class="chunks-label">📄 Source Evidence</div>',
        unsafe_allow_html=True,
    )

    for i, chunk in enumerate(chunks):
        spec = chunk.get("spec_number") or chunk.get("source", "").replace(".pdf", "")
        section = chunk.get("section", "")
        title = chunk.get("section_title", "")
        ctype = chunk.get("chunk_type", "general")
        page = chunk.get("page", "?")
        source = chunk.get("source", "")
        text = chunk.get("text", "")
        sim = chunk.get("similarity", 0)
        rerank = chunk.get("rerank_score")

        if rerank is not None:
            score_html = (
                f'<div class="chunk-score">'
                f'rerank <span class="score-val">{rerank:.2f}</span>'
                f"</div>"
            )
        else:
            score_html = (
                f'<div class="chunk-score">'
                f'sim <span class="score-val">{sim:.3f}</span>'
                f"</div>"
            )

        badge_html = render_chunk_type_badge(ctype)

        title_html = f'<div class="chunk-title">{title}</div>' if title else ""

        preview = text[:300].replace("<", "&lt;").replace(">", "&gt;")

        st.markdown(
            f"""
            <div class="chunk-card" style="animation-delay: {i * 0.08}s">
                <div class="chunk-meta">
                    <span class="chunk-spec">TS {spec}</span>
                    <span class="chunk-section">§ {section}</span>
                    {badge_html}
                    {score_html}
                </div>
                {title_html}
                <div class="chunk-text">{preview}{'...' if len(text) > 300 else ''}</div>
                <div class="chunk-page">
                    {source} · page {page}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def call_rca_api(query: str) -> dict | None:
    """Call the RCA API and return the response dict or None on error."""
    try:
        resp = requests.post(
            RCA_ENDPOINT,
            json={"query": query},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.session_state["last_error"] = (
            "Cannot connect to API. Is the FastAPI server running?\n"
            "Run: uvicorn umbrella_rag.api.app:app --host 0.0.0.0 --port 8000"
        )
        return None
    except requests.exceptions.Timeout:
        st.session_state["last_error"] = (
            f"Request timed out after {REQUEST_TIMEOUT}s. "
            "The model may be loading — try again."
        )
        return None
    except requests.exceptions.HTTPError as exc:
        detail = ""
        try:
            detail = exc.response.json().get("detail", str(exc))
        except Exception:
            detail = str(exc)
        st.session_state["last_error"] = f"API error: {detail}"
        return None
    except Exception as exc:
        st.session_state["last_error"] = f"Unexpected error: {exc}"
        return None


def render_sidebar(online: bool, components: dict) -> None:
    """Render the sidebar with health, history, and sample queries."""
    with st.sidebar:
        st.markdown(
            f"""
            <div style="padding: 1rem 0 0.5rem;">
                <span style="font-size: 1.3rem;">{APP_ICON}</span>
                <span style="font-family: 'Inter', sans-serif; font-weight: 700;
                      font-size: 1.1rem; color: {COLOR_TEXT_PRIMARY}; margin-left: 0.4rem;">
                    Umbrella
                </span>
                <span style="color: {COLOR_ACCENT_PRIMARY}; font-weight: 700; font-size: 1.1rem;">RAG</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # ── System Health ──
        st.markdown(
            '<div class="sidebar-label">System Status</div>',
            unsafe_allow_html=True,
        )

        if online:
            for name, status in components.items():
                icon = "✅" if status else "❌"
                color = COLOR_ACCENT_SUCCESS if status else COLOR_ACCENT_ERROR
                st.markdown(
                    f'<div style="font-family: JetBrains Mono, monospace; '
                    f'font-size: 0.75rem; color: {color}; '
                    f'margin-bottom: 0.25rem;">'
                    f"{icon} {name}</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                f'<div style="font-family: JetBrains Mono, monospace; '
                f'font-size: 0.75rem; color: {COLOR_ACCENT_ERROR};">❌ API unreachable</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # ── Sample Queries ──
        st.markdown(
            '<div class="sidebar-label">Sample Queries</div>',
            unsafe_allow_html=True,
        )

        for query in SAMPLE_QUERIES:
            if st.button(
                query[:55] + ("…" if len(query) > 55 else ""),
                key=f"sample_{hash(query)}",
                use_container_width=True,
            ):
                st.session_state["prefill_query"] = query
                st.session_state["query_input"] = query
                st.rerun()

        st.markdown("---")

        # ── Query History ──
        history = st.session_state.get("query_history", [])
        if history:
            st.markdown(
                '<div class="sidebar-label">Recent Queries</div>',
                unsafe_allow_html=True,
            )
            for item in reversed(history[-8:]):
                q = item["query"]
                ts = item["timestamp"]
                lat = item.get("latency_ms", 0)
                st.markdown(
                    f"""
                    <div class="history-item"
                         onclick="void(0)"
                         title="{q}">
                        <div class="history-query">{q[:50]}{'…' if len(q)>50 else ''}</div>
                        <div class="history-time">{ts} · {lat/1000:.1f}s</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        # ── Clear History ──
        if history:
            if st.button("🗑 Clear History", use_container_width=True):
                st.session_state["query_history"] = []
                st.rerun()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main() -> None:
    """Main Streamlit application entry point."""

    # ── Page config ──
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_css()

    # ── Session state init ──
    if "query_history" not in st.session_state:
        st.session_state["query_history"] = []
    if "last_error" not in st.session_state:
        st.session_state["last_error"] = None
    if "last_response" not in st.session_state:
        st.session_state["last_response"] = None
    if "prefill_query" not in st.session_state:
        st.session_state["prefill_query"] = ""

    # ── Health check ──
    online, components = check_api_health()

    # ── Sidebar ──
    render_sidebar(online, components)

    # ── Main content ──
    render_hero()
    render_status_bar(online, components)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Query input ──
    prefill = st.session_state.pop("prefill_query", "")
    if prefill:
        st.session_state["query_input"] = prefill

    col_input, col_btn = st.columns([5, 1])

    with col_input:
        query = st.text_area(
            label="Query",
            placeholder=(
                "Describe the anomaly or ask a technical question...\n\n"
                "Example: Why does handover fail when RSRP drops below -110 dBm?"
            ),
            height=110,
            label_visibility="collapsed",
            key="query_input",
        )

    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        submit = st.button(
            "Analyze",
            use_container_width=True,
            disabled=not online,
            type="primary",
        )

    # ── Show last error if any ──
    if st.session_state["last_error"]:
        st.markdown(
            f'<div class="error-card">⚠ {st.session_state["last_error"]}</div>',
            unsafe_allow_html=True,
        )
        st.session_state["last_error"] = None

    # ── Handle submission ──
    if submit and query and query.strip():
        st.session_state["last_response"] = None

        # Show analyzing state
        analyzing_placeholder = st.empty()
        analyzing_placeholder.markdown(
            f"""
            <div class="analyzing-container">
                <span class="analyzing-icon">{APP_ICON}</span>
                <div class="analyzing-text">
                    Analyzing query
                    <span class="analyzing-dots"></span>
                </div>
                <div style="font-family: JetBrains Mono, monospace;
                            font-size: 0.7rem; color: {COLOR_TEXT_SUBTLE}; margin-top: 0.5rem;">
                    retrieve → rerank → generate
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        response = call_rca_api(query.strip())
        analyzing_placeholder.empty()

        if response:
            st.session_state["last_response"] = response

            # Add to history
            st.session_state["query_history"].append({
                "query": query.strip(),
                "timestamp": time.strftime("%H:%M:%S"),
                "latency_ms": response.get("latency_ms", 0),
            })

    # ── Render last response ──
    response = st.session_state.get("last_response")
    if response:
        render_metrics(response)
        render_answer(response)

        chunks = response.get("chunks", [])
        if chunks:
            with st.expander(
                f"📄 Source Evidence ({len(chunks)} chunks)",
                expanded=True,
            ):
                render_chunks(chunks)

    elif not submit:
        # Empty state — show hint
        if not st.session_state.get("query_history"):
            st.markdown(
                f"""
                <div style="text-align: center; padding: 4rem 1rem;
                            color: {COLOR_BORDER}; font-family: JetBrains Mono, monospace;
                            font-size: 0.8rem; letter-spacing: 0.05em;">
                    Enter a query above or select a sample from the sidebar
                </div>
                """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
