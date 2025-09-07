#!/usr/bin/env python3
"""
react_jobs_finder.py — Find CURRENT “React” / “React Native” job openings from common ATS providers.

Outputs:
  1) CSV file with columns: Company | Job title | Location | Career portal link
  2) Printed Markdown table (top N rows) to stdout

What’s new in this refactor:
- All static constants are sourced from env/.env (no in-script defaults for config).
- Logging & debug utilities moved to debug_constants.py.
- Debugger mode (--debug=true|false) + milestone logs via debug_constants.

Discovery priority:
- Google CSE (free 100/day) → SerpAPI fallback.
- Uses official/public ATS where possible (Greenhouse, Lever, Ashby; SmartRecruiters if token provided),
  otherwise a single-page JSON-LD parse (e.g., Workday).

CLI:
  --keywords   (repeatable; default: ["React", "React Native"])
  --limit      (max number of rows to output; default: 50)
  --out        (CSV output path; default: ./react_jobs.csv)
  --quiet      (suppress milestone logs)
  --progress   (force-enable milestone logs; default behavior)
  --debug      (true|false; default: false) → enables verbose debug logs & tracebacks

Example:
  python3 react_jobs_finder.py --keywords React "React Native" --limit 40 --out jobs.csv --debug=true
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
from urllib.parse import urlparse
from urllib import robotparser
from datetime import datetime
from zoneinfo import ZoneInfo


import pandas as pd
import requests
from bs4 import BeautifulSoup

# Logging / debug
from debug_constants import log, dlog, step, set_modes

# ============================ .env loader ==============================

def load_dotenv(path: str = ".env", override: bool = False) -> bool:
    """Minimal .env loader (no deps)."""
    try:
        changed = False
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key, val = key.strip(), val.strip().strip('"').strip("'")
                if override or key not in os.environ:
                    os.environ[key] = val
                    changed = True
        return changed
    except FileNotFoundError:
        return False

ENV_LOADED = load_dotenv()

# ============================ Config (from ENV) ========================

def _require(name: str) -> str:
    v = os.environ.get(name, "").strip()
    if not v:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return v

def _require_list(name: str) -> List[str]:
    vals = [x.strip() for x in _require(name).split(",") if x.strip()]
    if not vals:
        raise RuntimeError(f"{name} is empty — provide at least one value")
    return vals

def _parse_bool(s: str) -> bool:
    return s.strip().lower() in {"1", "true", "t", "yes", "y", "on"}

# Date and Time zone for output file
OUT_DATE_FMT: str
OUT_TZ: str

# Populated by init_config()
JOBS_USER_AGENT: str
HTTP_ACCEPT_LANGUAGE: str
REQUEST_TIMEOUT: int
BASE_SLEEP: float
BACKOFF_MAX_SECONDS: float
MAX_RETRIES: int

# Discovery
SEARCH_BACKENDS: List[str]
ATS_ALLOWED_DOMAINS: Set[str]
ATS_SEARCH_DOMAINS: List[str]

# Backends
GOOGLE_CSE_ENDPOINT: str
GOOGLE_CSE_KEY: str
GOOGLE_CSE_CX: str
CSE_PAGE_SIZE: int
CSE_MAX_START: int

SERPAPI_ENDPOINT: str
SERPAPI_ENGINE: str
SERPAPI_KEY: str

# Providers
ATS_PROVIDERS: Set[str]
GREENHOUSE_API_TEMPLATE: str
LEVER_API_TEMPLATE: str
ASHBY_API_TEMPLATE: str
SMARTRECRUITERS_API_TEMPLATE: str
SMARTRECRUITERS_TOKEN: str

# Defaults for CLI (also from env)
DEFAULT_LIMIT: int
DEFAULT_OUT_PATH: str
DEFAULT_KEYWORDS: List[str]

def init_config() -> None:
    global JOBS_USER_AGENT, HTTP_ACCEPT_LANGUAGE, REQUEST_TIMEOUT, BASE_SLEEP, BACKOFF_MAX_SECONDS, MAX_RETRIES
    global SEARCH_BACKENDS, ATS_ALLOWED_DOMAINS, ATS_SEARCH_DOMAINS
    global GOOGLE_CSE_ENDPOINT, GOOGLE_CSE_KEY, GOOGLE_CSE_CX, CSE_PAGE_SIZE, CSE_MAX_START
    global SERPAPI_ENDPOINT, SERPAPI_ENGINE, SERPAPI_KEY
    global ATS_PROVIDERS, GREENHOUSE_API_TEMPLATE, LEVER_API_TEMPLATE, ASHBY_API_TEMPLATE, SMARTRECRUITERS_API_TEMPLATE, SMARTRECRUITERS_TOKEN
    global DEFAULT_LIMIT, DEFAULT_OUT_PATH, DEFAULT_KEYWORDS

    # Date and Time zone (from env)
    global OUT_TZ, OUT_DATE_FMT
    OUT_DATE_FMT = _require("OUT_DATE_FMT")
    OUT_TZ = _require("OUT_TZ")

    # Core runtime
    JOBS_USER_AGENT = _require("JOBS_USER_AGENT")
    HTTP_ACCEPT_LANGUAGE = _require("HTTP_ACCEPT_LANGUAGE")
    REQUEST_TIMEOUT = int(_require("REQUEST_TIMEOUT_SECONDS"))
    BASE_SLEEP = float(_require("BASE_SLEEP_SECONDS"))
    BACKOFF_MAX_SECONDS = float(_require("BACKOFF_MAX_SECONDS"))
    MAX_RETRIES = int(_require("MAX_RETRIES"))

    # Discovery config
    SEARCH_BACKENDS = [b.strip().lower() for b in _require_list("SEARCH_BACKENDS")]
    ATS_ALLOWED_DOMAINS = set(_require_list("ATS_ALLOWED_DOMAINS"))
    ATS_SEARCH_DOMAINS = _require_list("ATS_SEARCH_DOMAINS")

    # Backends: Google CSE (if enabled)
    if "google_cse" in SEARCH_BACKENDS:
        GOOGLE_CSE_ENDPOINT = _require("GOOGLE_CSE_ENDPOINT")
        GOOGLE_CSE_KEY = _require("GOOGLE_CSE_KEY")
        GOOGLE_CSE_CX  = _require("GOOGLE_CSE_CX")
        CSE_PAGE_SIZE  = int(_require("CSE_PAGE_SIZE"))      # Google cap 10
        CSE_MAX_START  = int(_require("CSE_MAX_START"))      # Google cap 100
    else:
        GOOGLE_CSE_ENDPOINT = GOOGLE_CSE_KEY = GOOGLE_CSE_CX = ""
        CSE_PAGE_SIZE = CSE_MAX_START = 0

    # Backends: SerpAPI (if enabled)
    if "serpapi" in SEARCH_BACKENDS:
        SERPAPI_ENDPOINT = _require("SERPAPI_ENDPOINT")
        SERPAPI_ENGINE   = _require("SERPAPI_ENGINE")
        SERPAPI_KEY      = _require("SERPAPI_KEY")
    else:
        SERPAPI_ENDPOINT = SERPAPI_ENGINE = SERPAPI_KEY = ""

    # Providers
    ATS_PROVIDERS = set([p.strip().lower() for p in _require_list("ATS_PROVIDERS")])
    if "greenhouse" in ATS_PROVIDERS:
        GREENHOUSE_API_TEMPLATE = _require("GREENHOUSE_API_TEMPLATE")
    else:
        GREENHOUSE_API_TEMPLATE = ""
    if "lever" in ATS_PROVIDERS:
        LEVER_API_TEMPLATE = _require("LEVER_API_TEMPLATE")
    else:
        LEVER_API_TEMPLATE = ""
    if "ashby" in ATS_PROVIDERS:
        ASHBY_API_TEMPLATE = _require("ASHBY_API_TEMPLATE")
    else:
        ASHBY_API_TEMPLATE = ""
    if "smartrecruiters" in ATS_PROVIDERS:
        SMARTRECRUITERS_API_TEMPLATE = _require("SMARTRECRUITERS_API_TEMPLATE")
        SMARTRECRUITERS_TOKEN       = _require("SMARTRECRUITERS_TOKEN")
    else:
        SMARTRECRUITERS_API_TEMPLATE = ""
        SMARTRECRUITERS_TOKEN = ""

    # CLI defaults (from env so no in-script static)
    DEFAULT_LIMIT    = int(_require("DEFAULT_LIMIT"))
    DEFAULT_OUT_PATH = _require("DEFAULT_OUT_PATH")
    DEFAULT_KEYWORDS = [x.strip() for x in _require("DEFAULT_KEYWORDS").split(",") if x.strip()]

# ============================ Data model ===============================

@dataclass
class JobPosting:
    company: str
    link: str
    title: str
    location: str

# ============================ Helpers =================================

def slug_to_company(slug: str) -> str:
    slug = slug.strip().strip("/").split("/")[0]
    slug = slug.replace("-", " ").replace("_", " ").strip()
    return slug.title() if slug else ""

def build_keyword_regex(keywords: List[str]) -> re.Pattern:
    escaped = [rf"\b{re.escape(k)}\b" for k in keywords]
    return re.compile("(" + "|".join(escaped) + ")", flags=re.IGNORECASE)

def is_allowed_by_robots(session: requests.Session, url: str) -> bool:
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

    if not hasattr(session, "_robots_cache"):
        session._robots_cache = {}  # type: ignore[attr-defined]
    cache: Dict[str, robotparser.RobotFileParser] = session._robots_cache  # type: ignore[attr-defined]

    rp = cache.get(robots_url)
    if rp is None:
        rp = robotparser.RobotFileParser()
        try:
            dlog(f"Fetching robots.txt: {robots_url}")
            resp = session.get(robots_url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": JOBS_USER_AGENT})
            if resp.status_code == 200:
                rp.parse(resp.text.splitlines())
            else:
                rp.parse([])
                log(f"robots.txt unavailable ({resp.status_code}) for {parsed.netloc}; proceeding cautiously")
        except requests.RequestException as e:
            rp.parse([])
            log(f"robots.txt fetch failed for {parsed.netloc}; proceeding cautiously ({e})")
        cache[robots_url] = rp

    allowed = rp.can_fetch(JOBS_USER_AGENT, url)
    if not allowed:
        log(f"Disallowed by robots.txt: {url}")
    return allowed

def sleep_with_backoff(attempt: int) -> None:
    # 1, 2, 4, 8... capped by BACKOFF_MAX_SECONDS
    delay = min(BASE_SLEEP * (2 ** attempt), BACKOFF_MAX_SECONDS)
    log(f"Backing off {delay:.1f}s (attempt {attempt+1})")
    time.sleep(delay)

def safe_get_json(session: requests.Session, url: str, headers: Dict[str, str]) -> Optional[dict]:
    for attempt in range(MAX_RETRIES):
        try:
            dlog(f"GET JSON: {url}")
            r = session.get(url, timeout=REQUEST_TIMEOUT, headers=headers)
            dlog(f"→ status {r.status_code}")
            if r.status_code == 200:
                return r.json()
            elif r.status_code in (429, 500, 502, 503, 504):
                sleep_with_backoff(attempt)
                continue
            else:
                log(f"JSON fetch non-200 ({r.status_code}) for {url}")
                return None
        except requests.RequestException as e:
            log(f"JSON fetch error: {e} for {url}")
            sleep_with_backoff(attempt)
    return None

def safe_get_html(session: requests.Session, url: str, headers: Dict[str, str]) -> Optional[str]:
    for attempt in range(MAX_RETRIES):
        try:
            if not is_allowed_by_robots(session, url):
                return None
            dlog(f"GET HTML: {url}")
            r = session.get(url, timeout=REQUEST_TIMEOUT, headers=headers)
            dlog(f"→ status {r.status_code} content-type {r.headers.get('Content-Type')}")
            if r.status_code == 200 and "text/html" in (r.headers.get("Content-Type") or ""):
                return r.text
            elif r.status_code in (429, 500, 502, 503, 504):
                sleep_with_backoff(attempt)
                continue
            else:
                log(f"HTML fetch non-200 ({r.status_code}) or non-HTML for {url}")
                return None
        except requests.RequestException as e:
            log(f"HTML fetch error: {e} for {url}")
            sleep_with_backoff(attempt)
    return None

# def to_markdown_table(rows: List[JobPosting], limit: int) -> str:
#     cols = ["Company", "Career portal link", "Job title", "Location"]
#     head = "| " + " | ".join(cols) + " |"
#     sep = "| " + " | ".join(["---"] * len(cols)) + " |"
#     body = [f"| {r.company} | {r.link} | {r.title} | {r.location} |" for r in rows[:limit]]
#     return "\n".join([head, sep] + body)

def normalize_domain(url: str) -> str:
    return urlparse(url).netloc.lower()

def is_allowed_domain(url: str) -> bool:
    host = normalize_domain(url)
    for d in ATS_ALLOWED_DOMAINS:
        if host == d or host.endswith("." + d):
            return True
    return False

def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": JOBS_USER_AGENT, "Accept-Language": HTTP_ACCEPT_LANGUAGE})
    return s

def first_path_segment(url: str) -> str:
    path = urlparse(url).path.strip("/")
    return (path.split("/")[0] if path else "").strip()

def fmt(template: str, **kwargs: str) -> str:
    return template.format(**kwargs)

def expand_out_path(template: str, tz: str, fmt: str) -> str:
    """Replace {date} with today's date in OUT_TZ using OUT_DATE_FMT."""
    now = datetime.now(ZoneInfo(tz))
    return template.replace("{date}", now.strftime(fmt))


# ============================ Search layer ============================

def build_search_query(keywords: List[str]) -> str:
    kq = " OR ".join([f"\"{k}\"" for k in keywords])
    dq = " OR ".join([f"site:{d}" for d in ATS_SEARCH_DOMAINS])
    return f"({kq}) ({dq})"

def search_google_cse(session: requests.Session, query: str, limit: int) -> List[str]:
    urls: List[str] = []
    start = 1
    per_page = CSE_PAGE_SIZE
    log("Discovery: Google CSE → querying…")
    while len(urls) < limit and start <= CSE_MAX_START:
        params = {"key": GOOGLE_CSE_KEY, "cx": GOOGLE_CSE_CX, "q": query,
            "num": min(per_page, max(1, limit - len(urls))), "start": start}
        try:
            dlog(f"CSE request start={start} q={query}")
            r = session.get(GOOGLE_CSE_ENDPOINT, params=params, timeout=REQUEST_TIMEOUT)
            dlog(f"CSE status {r.status_code}")
            if r.status_code != 200:
                log(f"Google CSE returned status {r.status_code}")
                break
            data = r.json()
            items = data.get("items", []) if isinstance(data, dict) else []
            batch = [it.get("link") for it in items if it.get("link")]
            before = len(urls)
            for u in batch:
                if u and is_allowed_domain(u):
                    urls.append(u)
            log(f"Google CSE page start={start}: +{len(urls)-before} (total {len(urls)})")
            if not items:
                break
            start += per_page
        except requests.RequestException as e:
            log(f"Google CSE error: {e}")
            break
    return urls

def search_serpapi(session: requests.Session, query: str, limit: int) -> List[str]:
    params = {"engine": SERPAPI_ENGINE, "q": query, "num": min(limit, 50), "api_key": SERPAPI_KEY}
    log("Discovery: SerpAPI → querying…")
    try:
        dlog(f"SerpAPI request q={query}")
        r = session.get(SERPAPI_ENDPOINT, params=params, timeout=REQUEST_TIMEOUT)
        dlog(f"SerpAPI status {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            items = data.get("organic_results", [])
            urls = [it.get("link") for it in items if it.get("link")]
            allowed = [u for u in urls if u and is_allowed_domain(u)]
            log(f"SerpAPI returned {len(allowed)} allowed URLs")
            return allowed
        else:
            log(f"SerpAPI returned status {r.status_code}")
    except requests.RequestException as e:
        log(f"SerpAPI error: {e}")
    return []

def discover_posting_urls(session: requests.Session, keywords: List[str], limit: int) -> List[str]:
    query = build_search_query(keywords)
    with step("DISCOVERY"):
        log(f"Keywords: {keywords}")
        log(f"Search domains: {ATS_SEARCH_DOMAINS}")
        log(f"Built query: {query}")
        urls: List[str] = []
        for backend in SEARCH_BACKENDS:
            if backend == "google_cse":
                urls.extend(search_google_cse(session, query, limit * 3))
            elif backend == "serpapi":
                urls.extend(search_serpapi(session, query, limit * 3))
        # Deduplicate + filter
        seen: Set[str] = set()
        out: List[str] = []
        for u in urls:
            if u not in seen and is_allowed_domain(u):
                out.append(u); seen.add(u)
            if len(out) >= limit * 2:
                break
        log(f"Discovery: {len(urls)} candidates → {len(out)} after dedupe/filter")
        return out

# ===================== Provider detection & parsers ====================

def detect_provider(url: str) -> str:
    host = normalize_domain(url)
    if "greenhouse.io" in host:
        return "greenhouse"
    if "lever.co" in host:
        return "lever"
    if "myworkdayjobs.com" in host:
        return "workday"
    if "ashbyhq.com" in host:
        return "ashby"
    if "smartrecruiters.com" in host:
        return "smartrecruiters"
    return "unknown"

# ---- Official APIs (enabled by ATS_PROVIDERS) -------------------------

def parse_greenhouse_company_jobs(session: requests.Session, url: str, kw_re: re.Pattern) -> List[JobPosting]:
    slug = first_path_segment(url)
    if not slug: return []
    api = fmt(GREENHOUSE_API_TEMPLATE, company=slug)
    log(f"Greenhouse: {slug} → API")
    data = safe_get_json(session, api, headers={"User-Agent": JOBS_USER_AGENT, "Accept": "application/json"})
    if not data or "jobs" not in data:
        log(f"Greenhouse: {slug} → no data"); return []
    company, jobs, matches = slug_to_company(slug), [], 0
    for job in data.get("jobs", []):
        title = (job.get("title") or "").strip()
        if not title or not kw_re.search(title): continue
        location = (job.get("location", {}) or {}).get("name", "") or ""
        link = job.get("absolute_url") or ""
        if link:
            jobs.append(JobPosting(company=company, link=link, title=title, location=location)); matches += 1
    log(f"Greenhouse: {slug} → {matches} matches"); return jobs

def parse_lever_company_jobs(session: requests.Session, url: str, kw_re: re.Pattern) -> List[JobPosting]:
    slug = first_path_segment(url)
    if not slug: return []
    api = fmt(LEVER_API_TEMPLATE, company=slug)
    log(f"Lever: {slug} → API")
    data = safe_get_json(session, api, headers={"User-Agent": JOBS_USER_AGENT, "Accept": "application/json"})
    if not data or not isinstance(data, list):
        log(f"Lever: {slug} → no data"); return []
    company, jobs, matches = slug_to_company(slug), [], 0
    for item in data:
        title = item.get("text") or item.get("title") or item.get("name") or ""
        if not title or not kw_re.search(str(title)): continue
        loc = ""
        cats = item.get("categories") or {}
        if isinstance(cats, dict): loc = cats.get("location") or ""
        if not loc: loc = item.get("country") or item.get("workplaceType") or ""
        link = item.get("hostedUrl") or item.get("applyUrl") or item.get("url") or ""
        if link:
            jobs.append(JobPosting(company=company, link=link, title=str(title), location=str(loc))); matches += 1
    log(f"Lever: {slug} → {matches} matches"); return jobs

def parse_ashby_company_jobs(session: requests.Session, url: str, kw_re: re.Pattern) -> List[JobPosting]:
    board = first_path_segment(url)
    if not board: return []
    api = fmt(ASHBY_API_TEMPLATE, board=board)
    log(f"Ashby: {board} → API")
    data = safe_get_json(session, api, headers={"User-Agent": JOBS_USER_AGENT, "Accept": "application/json"})
    if not data or "jobs" not in data:
        log(f"Ashby: {board} → no data"); return []
    company, jobs, matches = slug_to_company(board), [], 0
    for j in data.get("jobs", []):
        title = (j.get("title") or "").strip()
        if not title or not kw_re.search(title): continue
        location = (j.get("location") or "").strip()
        link = (j.get("jobUrl") or "").strip()
        if link:
            jobs.append(JobPosting(company=company, link=link, title=title, location=location)); matches += 1
    log(f"Ashby: {board} → {matches} matches"); return jobs

def parse_smartrecruiters_company_jobs_api(session: requests.Session, url: str, kw_re: re.Pattern) -> List[JobPosting]:
    comp = first_path_segment(url)
    if not comp: return []
    api = fmt(SMARTRECRUITERS_API_TEMPLATE, company=comp)
    log(f"SmartRecruiters API: {comp} → API")
    headers = {"User-Agent": JOBS_USER_AGENT, "Accept": "application/json", "X-SmartToken": SMARTRECRUITERS_TOKEN}
    data = safe_get_json(session, api, headers=headers)
    if not data or "content" not in data:
        log(f"SmartRecruiters API: {comp} → no data"); return []
    company, jobs, matches = slug_to_company(comp), [], 0
    for item in data.get("content", []):
        title = (item.get("name") or "").strip()
        if not title or not kw_re.search(title): continue
        link = (item.get("jobAd") or {}).get("url") if isinstance(item.get("jobAd"), dict) else ""
        link = link or item.get("jobAdUrl") or ""
        loc_obj = item.get("location") or {}
        if isinstance(loc_obj, dict):
            parts = [loc_obj.get("city"), loc_obj.get("region"), loc_obj.get("country")]
            location = ", ".join([p for p in parts if p])
        else:
            location = ""
        if link:
            jobs.append(JobPosting(company=company, link=link, title=title, location=location)); matches += 1
    log(f"SmartRecruiters API: {comp} → {matches} matches"); return jobs

# ---- JSON-LD fallback (generic single-page parse) --------------------

def parse_jsonld_jobposting(html: str) -> Optional[Tuple[str, str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    for s in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            data = json.loads(s.string or "")
        except Exception:
            continue
        nodes = data if isinstance(data, list) else [data]
        for d in nodes:
            if not isinstance(d, dict): continue
            t = d.get("@type")
            if t == "JobPosting" or (isinstance(t, list) and "JobPosting" in t):
                title = (d.get("title") or "").strip()
                org = d.get("hiringOrganization") or {}
                company = (org.get("name") or "").strip() if isinstance(org, dict) else ""
                location = ""
                jl = d.get("jobLocation")
                if isinstance(jl, list) and jl:
                    addr = jl[0].get("address") if isinstance(jl[0], dict) else {}
                    if isinstance(addr, dict):
                        parts = [addr.get("addressLocality"), addr.get("addressRegion"), addr.get("addressCountry")]
                        location = ", ".join([p for p in parts if p])
                elif isinstance(jl, dict):
                    addr = jl.get("address") or {}
                    if isinstance(addr, dict):
                        parts = [addr.get("addressLocality"), addr.get("addressRegion"), addr.get("addressCountry")]
                        location = ", ".join([p for p in parts if p])
                if not location:
                    jtype = d.get("jobLocationType")
                    if isinstance(jtype, str) and jtype:
                        location = jtype
                if title or company:
                    return title, company, location
    return None

def parse_generic_page(session: requests.Session, url: str, kw_re: re.Pattern) -> List[JobPosting]:
    # log(f"HTML parse: {url}")
    html = safe_get_html(session, url, headers={"User-Agent": JOBS_USER_AGENT})
    if not html: return []
    extracted = parse_jsonld_jobposting(html)
    if not extracted:
        log("HTML parse: JSON-LD not found"); return []
    title, company, location = extracted
    if title and kw_re.search(title):
        comp = company or slug_to_company(first_path_segment(url))
        log("HTML parse: match ✔")
        return [JobPosting(company=comp, link=url, title=title, location=location or "")]
    log("HTML parse: no keyword match"); return []

# ============================ Orchestrator ============================

def harvest_jobs(session: requests.Session, discovered_urls: List[str], keywords: List[str], limit: int) -> List[JobPosting]:
    with step("HARVEST"):
        kw_re = build_keyword_regex(keywords)
        out: List[JobPosting] = []

        processed: Dict[str, Set[str]] = {"greenhouse": set(), "lever": set(), "ashby": set(), "smartrecruiters": set()}
        counts = {"greenhouse": 0, "lever": 0, "ashby": 0, "smartrecruiters": 0, "generic": 0}

        log(f"Limit: {limit} | Keywords: {keywords} | UA: {JOBS_USER_AGENT}")
        for url in discovered_urls:
            if len(out) >= limit: break
            provider = detect_provider(url)
            try:
                if provider == "greenhouse" and "greenhouse" in ATS_PROVIDERS:
                    slug = first_path_segment(url)
                    if slug and slug not in processed["greenhouse"]:
                        rows = parse_greenhouse_company_jobs(session, url, kw_re)
                        processed["greenhouse"].add(slug); out.extend(rows); counts["greenhouse"] += len(rows)

                elif provider == "lever" and "lever" in ATS_PROVIDERS:
                    slug = first_path_segment(url)
                    if slug and slug not in processed["lever"]:
                        rows = parse_lever_company_jobs(session, url, kw_re)
                        processed["lever"].add(slug); out.extend(rows); counts["lever"] += len(rows)

                elif provider == "ashby" and "ashby" in ATS_PROVIDERS:
                    board = first_path_segment(url)
                    if board and board not in processed["ashby"]:
                        rows = parse_ashby_company_jobs(session, url, kw_re)
                        processed["ashby"].add(board); out.extend(rows); counts["ashby"] += len(rows)
                    if "generic" in ATS_PROVIDERS and len(out) < limit:
                        rows = parse_generic_page(session, url, kw_re); out.extend(rows); counts["generic"] += len(rows)

                elif provider == "smartrecruiters" and "smartrecruiters" in ATS_PROVIDERS:
                    comp = first_path_segment(url); used_api = False
                    if comp and comp not in processed["smartrecruiters"]:
                        api_rows = parse_smartrecruiters_company_jobs_api(session, url, kw_re)
                        if api_rows:
                            out.extend(api_rows); counts["smartrecruiters"] += len(api_rows); used_api = True
                        processed["smartrecruiters"].add(comp)
                    if "generic" in ATS_PROVIDERS and (not used_api) and len(out) < limit:
                        rows = parse_generic_page(session, url, kw_re); out.extend(rows); counts["generic"] += len(rows)

                elif "generic" in ATS_PROVIDERS:
                    rows = parse_generic_page(session, url, kw_re); out.extend(rows); counts["generic"] += len(rows)

            except Exception as e:
                log(f"Harvest error on {url}: {e}")
                continue

            time.sleep(BASE_SLEEP * 0.6)
            if len(out) >= limit: break

        # Deduplicate by link
        seen: Set[str] = set(); uniq: List[JobPosting] = []
        for j in out:
            if j.link not in seen:
                uniq.append(j); seen.add(j.link)

        log("---------- SUMMARY ----------")
        log(f"Greenhouse: {counts['greenhouse']} | Lever: {counts['lever']} | Ashby: {counts['ashby']} | SmartRecruiters: {counts['smartrecruiters']} | HTML(JSON-LD): {counts['generic']}")
        log(f"Total matches (pre-unique): {len(out)} → unique: {len(uniq)}")
        return uniq[:limit]

def write_csv(rows: List[JobPosting], out_path: str) -> None:
    with step("WRITE CSV"):
        df = pd.DataFrame([{
            "Company": r.company,
            "Job title": r.title,
            "Location": r.location,
            "Career portal link": r.link
        } for r in rows])
        df.to_csv(out_path, index=False)
        log(f"CSV written → {out_path} ({len(rows)} rows)")

# ================================ Main ================================

def parse_args() -> argparse.Namespace:
    # Defaults are env-driven to avoid in-script constants
    default_limit = int(os.environ.get("DEFAULT_LIMIT"))
    default_out_template = _require("DEFAULT_OUT_PATH")
    default_keywords = [x.strip() for x in os.environ.get("DEFAULT_KEYWORDS", "React,React Native").split(",") if x.strip()]

    p = argparse.ArgumentParser(description="Find current React/React Native job openings from ATS providers.")
    p.add_argument("--keywords", nargs="+", default=default_keywords, help="Keywords to match in job titles")
    p.add_argument("--limit", type=int, default=default_limit, help="Maximum number of results to output")
    p.add_argument("--out", type=str, default=default_out_template, help="Path to write CSV output (supports {date})")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--quiet", action="store_true", help="Suppress milestone logs")
    g.add_argument("--progress", action="store_true", help="Show milestone logs (default behavior)")
    p.add_argument("--debug", type=str, default=os.getenv("DEBUG_DEFAULT", "false"), help="Enable debugger mode: true|false")
    return p.parse_args()

def main() -> None:
    with step("STARTUP"):
        log(f".env loaded: {'yes' if ENV_LOADED else 'no'}")
        init_config()
        log(f"Backends: {SEARCH_BACKENDS}")
        log(f"Providers: {sorted(ATS_PROVIDERS)}")
        log(f"Allowed domains: {sorted(ATS_ALLOWED_DOMAINS)}")
        log(f"Search domains: {ATS_SEARCH_DOMAINS}")

    args = parse_args()
    debug_mode = args.debug.strip().lower() in {"1","true","t","yes","y","on"}
    progress_mode = (not args.quiet) or debug_mode
    set_modes(progress=progress_mode, debug=debug_mode)

    with step("INIT SESSION"):
        session = make_session()

    with step("DISCOVER URLS"):
        discovered = discover_posting_urls(session, args.keywords, args.limit)
        if not discovered:
            log("No postings discovered via search (check env config and quotas).")
            print("No postings discovered via search (check env config and quotas).", file=sys.stderr)
            sys.exit(2)

    rows = harvest_jobs(session, discovered, args.keywords, args.limit)
    if not rows:
        log("No matching jobs found after parsing.")
        print("No matching jobs found after parsing.", file=sys.stderr)
        sys.exit(3)
    
    # Write CSV
    out_path = expand_out_path(args.out, OUT_TZ, OUT_DATE_FMT)
    write_csv(rows, out_path)

    # Print markdown table to stdout (capped by --limit)
    # with step("PRINT TABLE"):
    #     print(to_markdown_table(rows, min(len(rows), args.limit)))

    log("DONE ✅")
    log(f"Saved CSV: {out_path}")

if __name__ == "__main__":
    main()


# ================================ USAGE ===============================
"""
USAGE

0) Keep secrets & config out of git:
   - Create a local `.env` next to this script and add it to `.gitignore`:
       echo ".env" >> .gitignore
   - Example `.env` (edit values):
       # Search APIs (Google CSE preferred; then SerpAPI fallback)
       GOOGLE_CSE_KEY=your_google_key
       GOOGLE_CSE_CX=your_google_cx
       SERPAPI_KEY=your_serpapi_key

       # Optional SmartRecruiters token (if you have one)
       SMARTRECRUITERS_TOKEN=your_smart_token

       # Domains (comma-separated)
       ATS_ALLOWED_DOMAINS=boards.greenhouse.io,jobs.lever.co,myworkdayjobs.com,ashbyhq.com,smartrecruiters.com,www.smartrecruiters.com,careers.smartrecruiters.com
       ATS_SEARCH_DOMAINS=boards.greenhouse.io,jobs.lever.co,myworkdayjobs.com,ashbyhq.com,smartrecruiters.com

       # Official API URL templates (override only if needed)
       GREENHOUSE_API_TEMPLATE=https://boards-api.greenhouse.io/v1/boards/{company}/jobs?content=true
       LEVER_API_TEMPLATE=https://api.lever.co/v0/postings/{company}?mode=json
       ASHBY_API_TEMPLATE=https://api.ashbyhq.com/posting-api/job-board/{board}
       SMARTRECRUITERS_API_TEMPLATE=https://api.smartrecruiters.com/v1/companies/{company}/postings?limit=100&offset=0

       # Politeness / runtime
       JOBS_USER_AGENT=ReactJobsFinderBot/1.4 (+https://yourdomain.example/jobsfinder; contact: mailto:you@example.com)
       REQUEST_TIMEOUT_SECONDS=20
       BASE_SLEEP_SECONDS=1.0

1) Install dependencies (Python 3.9+):
   pip install requests beautifulsoup4 pandas

2) Run (progress on stderr, results to stdout):
   python3 react_jobs_finder.py --keywords React "React Native" --limit 50 --out jobs.csv

3) Debugger mode (more logs + tracebacks):
   python3 react_jobs_finder.py --debug=true --limit 20

4) Quiet mode (minimal logs):
   python3 react_jobs_finder.py --quiet --limit 50

Output:
- CSV file: Company | Career portal link | Job title | Location
- Markdown table printed to stdout (clean to pipe into README/notes).
"""
