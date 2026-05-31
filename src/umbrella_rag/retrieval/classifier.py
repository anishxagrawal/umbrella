from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Maps keyword sets to preferred source documents.
# Order matters - first match wins.
# Keys are tuples of lowercase keywords/phrases.
_ROUTING_TABLE: list[tuple[tuple[str, ...], list[str]]] = [
    (
        (
            "xnap",
            "xn application",
            "xn interface",
            "sn addition",
            "sn change",
            "secondary node",
            "ue context retrieve",
            "ue context release",
            "xn handover",
        ),
        ["38423-i40.pdf"],
    ),
    (
        (
            "ngap",
            "ng application",
            "amf",
            "pdu session",
            "path switch",
            "ng handover",
            "ng-ran",
            "initial context",
            "ue radio capability",
            "paging",
            "nas transport",
        ),
        ["38413-j00.pdf"],
    ),
    (
        (
            "pdsch",
            "pusch",
            "throughput",
            "scheduling",
            "mcs",
            "tbs",
            "transport block",
            "rank indicator",
            "cqi",
            "link adaptation",
            "modulation order",
            "code rate",
            "pdsch mapping",
            "physical downlink shared",
            "physical uplink shared",
        ),
        ["38214-i40.pdf", "38213-j30.pdf"],
    ),
    (
        (
            "pdcch",
            "dci",
            "downlink control",
            "search space",
            "coreset",
            "blind decoding",
            "control channel",
        ),
        ["38213-j30.pdf", "38214-i40.pdf"],
    ),
    (
        (
            "rsrp",
            "rsrq",
            "sinr",
            "rrm",
            "measurement gap",
            "radio link monitoring",
            "rlm",
            "beam management",
            "cell reselection",
            "serving cell",
            "reference signal received",
            "measurement threshold",
            "handover threshold",
            "a3 offset",
            "hysteresis",
            "time to trigger",
        ),
        ["38133-i90.pdf"],
    ),
    (
        (
            "bs threshold",
            "base station output power",
            "reference sensitivity",
            "adjacent channel",
            "blocking",
            "spurious",
            "emissions",
            "conducted",
            "radiated",
        ),
        ["38104-j40.pdf"],
    ),
    (
        (
            "rrc reconfiguration",
            "rrc setup",
            "rrc release",
            "rrc connection",
            "measurement config",
            "handover command",
            "master information block",
            "system information",
            "rrc reestablishment",
            "reducedmaxbw",
            "reducedmimo",
            "reducedccs",
            "overheating",
            "ue assistance",
            "mib",
            "sib",
        ),
        ["38331-hg0.pdf"],
    ),
    (
        (
            "mac",
            "harq",
            "bsr",
            "buffer status",
            "scheduling request",
            "sr",
            "random access",
            "rach",
            "phr",
            "power headroom",
            "timing advance",
            "logical channel",
            "priority",
        ),
        ["38321-j10.pdf"],
    ),
    (
        (
            "architecture",
            "functional split",
            "cu-du",
            "f1 interface",
            "ng interface",
            "overall description",
            "protocol stack",
            "user plane",
            "control plane separation",
        ),
        ["38300-j20.pdf"],
    ),
]


def classify_query(query: str) -> list[str] | None:
    """Map a query to a list of preferred source documents.

    Args:
        query: Raw or expanded query string.

    Returns:
        List of source filenames to filter retrieval by,
        or None if no routing rule matches (search all sources).

    Notes:
        First match wins. Returns None on empty query.
        Matching is case-insensitive and whole-word aware.
    """
    if not query:
        return None

    query_lower = query.lower()

    for keywords, sources in _ROUTING_TABLE:
        for keyword in keywords:
            pattern = r"(?<![a-z0-9])" + re.escape(keyword) + r"(?![a-z0-9])"
            if re.search(pattern, query_lower):
                logger.info(
                    "Query classifier matched keyword=%r -> sources=%s",
                    keyword,
                    sources,
                )
                return sources

    logger.debug("Query classifier: no match, searching all sources")
    return None
