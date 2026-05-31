#!/usr/bin/env python3
"""
Patch ragas llms/base.py to handle removed langchain_community.chat_models.vertexai.
Run this after any ragas reinstall.
"""
import pathlib, site, sys

candidates = [
    pathlib.Path(p) / "ragas/llms/base.py"
    for p in site.getsitepackages()
]
target = next((p for p in candidates if p.exists()), None)

if target is None:
    print("ERROR: ragas/llms/base.py not found")
    sys.exit(1)

content = target.read_text()
old = "from langchain_community.chat_models.vertexai import ChatVertexAI"
new = (
    "try:\n"
    "    from langchain_community.chat_models.vertexai import ChatVertexAI\n"
    "except ImportError:\n"
    "    ChatVertexAI = None  # removed in langchain-community>=0.3"
)

if old not in content:
    print("Already patched or line not found — nothing to do")
    sys.exit(0)

target.write_text(content.replace(old, new))
print(f"Patched: {target}")
