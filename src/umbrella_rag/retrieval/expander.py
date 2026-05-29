from __future__ import annotations

import logging
import re


logger = logging.getLogger(__name__)

# Full telecom acronym dictionary
# Keys are uppercase acronyms, values are full expansions
TELECOM_GLOSSARY: dict[str, str] = {
    # Mobility
    "HO": "Handover",
    "RLF": "Radio Link Failure",
    "HOF": "Handover Failure",
    "MRO": "Mobility Robustness Optimization",
    "MLB": "Mobility Load Balancing",
    "A3": "A3 Event",
    "A5": "A5 Event",

    # Radio / Signal
    "RSRP": "Reference Signal Received Power",
    "RSRQ": "Reference Signal Received Quality",
    "SINR": "Signal to Interference and Noise Ratio",
    "CQI": "Channel Quality Indicator",
    "MCS": "Modulation and Coding Scheme",
    "RSSI": "Received Signal Strength Indicator",

    # Nodes
    "gNB": "Next Generation NodeB",
    "UE": "User Equipment",
    "AMF": "Access and Mobility Management Function",
    "SMF": "Session Management Function",
    "UPF": "User Plane Function",
    "CU": "Central Unit",
    "DU": "Distributed Unit",
    "RU": "Radio Unit",

    # Channels / Resources
    "PRB": "Physical Resource Block",
    "PDSCH": "Physical Downlink Shared Channel",
    "PUSCH": "Physical Uplink Shared Channel",
    "PDCCH": "Physical Downlink Control Channel",
    "PUCCH": "Physical Uplink Control Channel",
    "PRACH": "Physical Random Access Channel",
    "SSB": "Synchronization Signal Block",
    "CSI": "Channel State Information",
    "CSI-RS": "Channel State Information Reference Signal",
    "DMRS": "Demodulation Reference Signal",

    # Protocols
    "RRC": "Radio Resource Control",
    "NAS": "Non-Access Stratum",
    "PDCP": "Packet Data Convergence Protocol",
    "RLC": "Radio Link Control",
    "MAC": "Medium Access Control",
    "HARQ": "Hybrid Automatic Repeat Request",
    "ARQ": "Automatic Repeat Request",

    # Procedures
    "RA": "Random Access",
    "RACH": "Random Access Channel",
    "TAU": "Tracking Area Update",
    "SR": "Scheduling Request",
    "BSR": "Buffer Status Report",
    "PHR": "Power Headroom Report",

    # KPIs / Measurements
    "KPI": "Key Performance Indicator",
    "PRB util": "Physical Resource Block Utilization",
    "BLER": "Block Error Rate",
    "IBLER": "Initial Block Error Rate",
    "RBLER": "Residual Block Error Rate",
    "DRB": "Data Radio Bearer",
    "SRB": "Signaling Radio Bearer",
    "QoS": "Quality of Service",
    "QFI": "QoS Flow Identifier",
    "GBR": "Guaranteed Bit Rate",
    "AMBR": "Aggregate Maximum Bit Rate",

    # Interfaces
    "NGAP": "Next Generation Application Protocol",
    "XnAP": "Xn Application Protocol",
    "F1AP": "F1 Application Protocol",
    "E1AP": "E1 Application Protocol",
    "N2": "N2 Interface",
    "N3": "N3 Interface",
    "Xn": "Xn Interface",
    "F1": "F1 Interface",

    # Power / Coverage
    "EIRP": "Equivalent Isotropically Radiated Power",
    "TRP": "Total Radiated Power",
    "UL": "Uplink",
    "DL": "Downlink",
    "TDD": "Time Division Duplex",
    "FDD": "Frequency Division Duplex",
    "NR": "New Radio",
    "LTE": "Long Term Evolution",
    "CA": "Carrier Aggregation",
    "DC": "Dual Connectivity",
    "EN-DC": "E-UTRA NR Dual Connectivity",

    # Architecture
    "RAN": "Radio Access Network",
    "CN": "Core Network",
    "EPC": "Evolved Packet Core",
    "5GC": "5G Core Network",
    "NG-RAN": "Next Generation Radio Access Network",
    "NSA": "Non-Standalone",
    "SA": "Standalone",
    "NWDAF": "Network Data Analytics Function",
    "RIC": "RAN Intelligent Controller",
    "ORAN": "Open RAN",

    # Timers / Parameters
    "TTI": "Transmission Time Interval",
    "RTT": "Round Trip Time",
    "T310": "T310 Timer",
    "T311": "T311 Timer",
    "N310": "N310 Counter",
    "N311": "N311 Counter",
}

_NORMALIZED_GLOSSARY: dict[str, str] = {
    key.upper(): value for key, value in TELECOM_GLOSSARY.items()
}
_PHRASE_KEYS: tuple[str, ...] = tuple(
    key for key in TELECOM_GLOSSARY if re.search(r"\s", key)
)
_PHRASE_PATTERN = re.compile(
    r"(?<!\w)(?:" + "|".join(re.escape(key) for key in sorted(_PHRASE_KEYS, key=len, reverse=True)) + r")(?!\w)",
    flags=re.IGNORECASE,
) if _PHRASE_KEYS else None
_TOKEN_PATTERN = re.compile(r"\s+|[^\w\s]+|\w+(?:[-']\w+)*")


def get_glossary() -> dict[str, str]:
    """Return a copy of the telecom glossary.

    Args:
        None.

    Returns:
        A shallow copy of the module glossary dictionary.

    Notes:
        The returned dictionary is safe to mutate by callers.
    """
    return dict(TELECOM_GLOSSARY)


def expand_tokens(tokens: list[str]) -> list[str]:
    """Expand a list of pre-tokenized terms.

    Args:
        tokens: List of string tokens to expand.

    Returns:
        A new list of tokens with glossary entries expanded.

    Notes:
        Tokens not found in the glossary are returned unchanged.
    """
    return [_expand_token(token) for token in tokens]


def expand_query(query: str) -> str:
    """Expand telecom acronyms inside a raw engineer query.

    Args:
        query: Raw query string from an engineer.

    Returns:
        Query string with recognized acronyms expanded.

    Notes:
        Matching is case-insensitive, punctuation and spacing are preserved
        as much as possible, and the function is stateless and thread-safe.
    """
    if not query:
        logger.debug("Glossary expansion: original=%r expanded=%r", query, query)
        return query

    expanded_query = query
    phrase_replacements = 0
    if _PHRASE_PATTERN is not None:
        expanded_query, phrase_replacements = _PHRASE_PATTERN.subn(
            lambda match: _NORMALIZED_GLOSSARY[match.group(0).upper()],
            expanded_query,
        )

    tokens = _TOKEN_PATTERN.findall(expanded_query)
    expanded_tokens = expand_tokens(tokens)
    token_replacements = sum(1 for original, expanded in zip(tokens, expanded_tokens) if original != expanded)
    expanded_query = "".join(expanded_tokens)

    total_replacements = phrase_replacements + token_replacements
    logger.debug(
        "Glossary expansion: original=%r expanded=%r",
        query,
        expanded_query,
    )
    if total_replacements > 0:
        logger.info("Expanded %s glossary term(s)", total_replacements)

    return expanded_query


def _expand_token(token: str) -> str:
    """Expand a single token if it matches a glossary entry.

    Args:
        token: Input token to inspect.

    Returns:
        The expanded token when a glossary match exists, otherwise the input.

    Notes:
        Matching is case-insensitive and only applies to exact token values.
    """
    lookup_key = token.upper()
    return _NORMALIZED_GLOSSARY.get(lookup_key, token)