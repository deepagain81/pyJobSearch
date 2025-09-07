#!/usr/bin/env python3
"""
react_jobs_finder.py — Find CURRENT “React” / “React Native” job openings from common ATS providers.

Outputs:
  1) CSV file with columns: Company | Career portal link | Job title | Location
  2) Printed Markdown table (top N rows) to stdout

What’s new in this refactor:
1) Added more official/public ATS integrations:
   • Greenhouse  -> Job Board API
   • Lever       -> Postings API
   • Ashby       -> Public Job Postings API (no auth)
   • SmartRecruiters -> Customer Posting API (optional; requires API key), else falls back to JSON-LD on public job pages
2) Discovery is now **Google Programmable Search (CSE) first**, with **SerpAPI** as the only fallback.
   (Both queries are locked to ATS domains.)

Providers & strategy:
- Prefer official/public endpoints where available (Greenhouse, Lever, Ashby, SmartRecruiters-with-key).
- Otherwise, fetch a single job page and read its JSON-LD (e.g., Workday/Ashby/SmartRecruiters) as a last-mile fallback.
- Discovery via search, limited to ATS domains:
    boards.greenhouse.io, jobs.lever.co, myworkdayjobs.com, ashbyhq.com, smartrecruiters.com

Ethics & safety:
- Respects robots.txt for every host before fetching HTML pages.
- Polite rate limiting + retries with exponential backoff.
- Identifying User-Agent.
- Only uses stdlib + requests + beautifulsoup4 + pandas.

CLI:
  --keywords   (repeatable; default: ["React", "React Native"])
  --limit      (max number of rows to output; default: 50)
  --out        (CSV output path; default: ./react_jobs.csv)

Example:
  python3 react_jobs_finder.py --keywords React "React Native" --limit 40 --out jobs.csv
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse, urlencode
from urllib import robotparser

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ----------------------------- Config ---------------------------------

ALLOWED_DOMAINS = {
    "boards.greenhouse.io",
    "jobs.lever.co",
    "myworkdayjobs.com",
    "ashbyhq.com",
    "jobs.ashbyhq.com",  # common hostname
    "smartrecruiters.com",
    "www.smartrecruiters.com",
    "careers.smartrecruiters.com",
}

SEARCH_QUERY_DOMAINS = [
    "boards.greenhouse.io",
    "jobs.lever.co",
    "myworkdayjobs.com",
    "ashbyhq.com",
    "smartrecruiters.com",
]

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (compatible; ReactJobsFinder/1.2; +https://example.com)"
)

REQUEST_TIMEOUT = 20  # seconds
BASE_SLEEP = 1.0      # polite rate limit base (seconds)

# ----------------------------- Data model -----------------------------

@dataclass
class JobPosting:
    company: str
    link: str
    title: str
    location: str

# ----------------------------- Helpers --------------------------------

def slug_to_company(slug: str) -> str:
    slug = slug.strip().strip("/").split("/")[0]
    slug = slug.replace("-", " ").replace("_", " ").strip()
    return slug.title() if slug else ""


def build_keyword_regex(keywords: List[str]) -> re.Pattern:
    escaped = [rf"\b{re.escape(k)}\b" for k in keywords]
    return re.compile("(" + "|".join(escaped) + ")", flags=re.IGNORECASE)


def is_allowed_by_robots(session: requests.Session, url: str, user_agent: str) -> bool:
    """
    Check robots.txt for the host in URL.
    Cache per-host robotparser instances on session for efficiency.
    """
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

    if not hasattr(session, "_robots_cache"):
        session._robots_cache = {}  # type: ignore[attr-defined]

    cache: Dict[str, robotparser.RobotFileParser] = session._robots_cache  # type: ignore[attr-defined]
    rp = cache.get(robots_url)

    if rp is None:
        rp = robotparser.RobotFileParser()
        try:
            resp = session.get(robots_url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": user_agent})
            if resp.status_code == 200:
                rp.parse(resp.text.splitlines())
            else:
                # If robots.txt missing or error, default to allowing (you may choose stricter behavior if desired).
                rp.parse([])
        except requests.RequestException:
            rp.parse([])
        cache[robots_url] = rp

    return rp.can_fetch(user_agent, url)


def sleep_with_backoff(attempt: int) -> None:
    # 1s, 2s, 4s, 8s (capped)
    delay = min(BASE_SLEEP * (2 ** attempt), 8.0)
    time.sleep(delay)


def safe_get_json(session: requests.Session, url: str, headers: Dict[str, str]) -> Optional[dict]:
    for attempt in range(4):
        try:
            # Allow JSON APIs regardless of robots, but keep politeness for HTML requests separately.
            r = session.get(url, timeout=REQUEST_TIMEOUT, headers=headers)
            if r.status_code == 200:
                return r.json()
            elif r.status_code in (429, 500, 502, 503, 504):
                sleep_with_backoff(attempt)
                continue
            else:
                return None
        except requests.RequestException:
            sleep_with_backoff(attempt)
    return None


def safe_get_html(session: requests.Session, url: str, headers: Dict[str, str]) -> Optional[str]:
    for attempt in range(4):
        try:
            if not is_allowed_by_robots(session, url, headers.get("User-Agent", DEFAULT_USER_AGENT)):
                return None
            r = session.get(url, timeout=REQUEST_TIMEOUT, headers=headers)
            if r.status_code == 200 and "text/html" in r.headers.get("Content-Type", ""):
                return r.text
            elif r.status_code in (429, 500, 502, 503, 504):
                sleep_with_backoff(attempt)
                continue
            else:
                return None
        except requests.RequestException:
            sleep_with_backoff(attempt)
    return None


def to_markdown_table(rows: List[JobPosting], limit: int) -> str:
    cols = ["Company", "Career portal link", "Job title", "Location"]
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body_lines = []
    for jp in rows[:limit]:
        body_lines.append(
            f"| {jp.company} | {jp.link} | {jp.title} | {jp.location} |"
        )
    return "\n".join([head, sep] + body_lines)


def normalize_domain(url: str) -> str:
    return urlparse(url).netloc.lower()


def is_allowed_domain(url: str) -> bool:
    host = normalize_domain(url)
    # allow subdomains
    for d in ALLOWED_DOMAINS:
        if host == d or host.endswith("." + d):
            return True
    return False


def make_session(user_agent: str = DEFAULT_USER_AGENT) -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent, "Accept-Language": "en-US,en;q=0.9"})
    return session


def first_path_segment(url: str) -> str:
    path = urlparse(url).path.strip("/")
    return (path.split("/")[0] if path else "").strip()

# ----------------------------- Search layer ---------------------------

def build_search_query(keywords: List[str]) -> str:
    kq = " OR ".join([f"\"{k}\"" for k in keywords])
    dq = " OR ".join([f"site:{d}" for d in SEARCH_QUERY_DOMAINS])
    return f"({kq}) ({dq})"


def search_google_cse(session: requests.Session, query: str, limit: int) -> List[str]:
    """
    Google Programmable Search Engine (Custom Search JSON API)
    Env vars:
      - GOOGLE_CSE_KEY : API key
      - GOOGLE_CSE_CX  : Search engine ID (must be configured to search ATS domains)
    """
    key = os.getenv("GOOGLE_CSE_KEY")
    cx = os.getenv("GOOGLE_CSE_CX")
    if no
