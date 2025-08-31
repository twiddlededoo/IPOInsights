#!/usr/bin/env python3
"""
Minimal IPO Extractor
---------------------
Fetch IPO pages and use a Generative AI model to extract:
- Company name
- Open date
- Close date
- Grey Market Premium (GMP)
- Lot size

Outputs:
- ipo_list.csv
- ipo_list.md

Usage:
  python ipo_ai_extract.py
"""

import asyncio
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import List, Dict, Any

import httpx

# ---------------- Config ----------------

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT_SECONDS", "30"))
CSV_PATH = "ipo_list.csv"
MD_PATH = "ipo_list.md"
URLS_FILE = "urls.txt"

# ---------------- Data ----------------

@dataclass
class IPORecord:
    source_url: str
    company_name: str = ""
    open_date: str = ""
    close_date: str = ""
    gmp: str = ""
    lot_size: str = ""

FIELDS = ["company_name", "open_date", "close_date", "gmp", "lot_size", "source_url"]

SYSTEM_PROMPT = """You are a precise financial data extractor.
Only return the required fields, no explanations, no guesses."""

USER_PROMPT = """Extract IPO details from the following page text.
Return valid JSON with this schema:

{
  "company_name": str|null,
  "open_date": str|null,
  "close_date": str|null,
  "gmp": str|null,
  "lot_size": str|null
}

Now extract from this content:
"""

# ---------------- Utils ----------------

def simplify_text(html: str, limit: int = 50_000) -> str:
    html = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
    html = re.sub(r"(?is)<style.*?>.*?</style>", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]

async def fetch_text(client: httpx.AsyncClient, url: str) -> str:
    r = await client.get(url, follow_redirects=True, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    r.encoding = r.encoding or r.apparent_encoding
    return r.text

async def call_openai_json(content: str) -> Dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if OPENAI_API_KEY:
        headers["Authorization"] = f"Bearer {OPENAI_API_KEY}"

    payload = {
        "model": DEFAULT_MODEL,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT + content},
        ],
        "temperature": 0,
    }

    async with httpx.AsyncClient(base_url=OPENAI_BASE_URL, timeout=HTTP_TIMEOUT) as client:
        resp = await client.post("/chat/completions", headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return json.loads(data["choices"][0]["message"]["content"])

def coerce_record(url: str, payload: Dict[str, Any]) -> IPORecord:
    return IPORecord(
        source_url=url,
        company_name=payload.get("company_name") or "",
        open_date=payload.get("open_date") or "",
        close_date=payload.get("close_date") or "",
        gmp=payload.get("gmp") or "",
        lot_size=payload.get("lot_size") or "",
    )

def write_csv(records: List[IPORecord], path: str = CSV_PATH):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for r in records:
            writer.writerow({k: getattr(r, k) for k in FIELDS})

def write_md(records: List[IPORecord], path: str = MD_PATH):
    lines = ["| Company | Open Date | Close Date | GMP | Lot Size | URL |",
             "|---------|-----------|------------|-----|----------|-----|"]
    for r in records:
        lines.append(f"| {r.company_name} | {r.open_date} | {r.close_date} | {r.gmp} | {r.lot_size} | {r.source_url} |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# ---------------- Runner ----------------

async def process_url(url: str) -> IPORecord:
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers={"User-Agent": "ipo-ai-extractor/1.0"}) as client:
        html = await fetch_text(client, url)
    text = simplify_text(html)
    try:
        ai_json = await call_openai_json(text)
        return coerce_record(url, ai_json)
    except Exception as e:
        return IPORecord(source_url=url, company_name=f"ERROR: {e}")

async def main():
    if not os.path.exists(URLS_FILE):
        print(f"ERROR: {URLS_FILE} not found. Create it with one URL per line.")
        sys.exit(1)

    with open(URLS_FILE, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]

    if not urls:
        print(f"{URLS_FILE} is empty. Add some URLs first.")
        sys.exit(1)

    records = await asyncio.gather(*[process_url(u) for u in urls])
    write_csv(records, CSV_PATH)
    write_md(records, MD_PATH)

    print(f"\n✅ Done. Wrote {CSV_PATH} and {MD_PATH}")
    print("Preview:")
    for r in records:
        print(f"- {r.company_name} ({r.open_date} → {r.close_date}) GMP: {r.gmp}, Lot: {r.lot_size}")

if __name__ == "__main__":
    asyncio.run(main())
