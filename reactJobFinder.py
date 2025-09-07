#!/usr/bin/env python3
"""
react_jobs_finder.py — Find CURRENT “React” / “React Native” job openings from common ATS providers.

Outputs:
  1) CSV file with columns: Company | Career portal link | Job title | Location
  2) Printed Markdown table (top N rows) to stdout

What’s new in this refactor:
- Milestone progress output to the terminal (stderr) so you can see the script’s progression (already added).
- **Debugger mode** via `--debug=true|false` to pinpoint where errors occur:
    • Logs step START/END with durations
    • Includes request URLs, response codes, retry/backoff details
    • On exceptions, prints full tracebacks to stderr

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
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse, urlencode
from urllib import robotparser

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ============================ .env loader ==============================

def load_dotenv(path: str = ".env", override: bool = False) -> bool:
    """
    Minimal .env loader (no dependencies).
    - KEY=VALUE (quotes optional); ignores blank/# lines.
    - Existing environment variables win unless override=True.
    Returns True if a file was loaded, else False.
    """
    try:
        changed = False
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if override or key not in os.environ:
                    os.environ[key] = val
                    changed = True
        return changed
    except FileNotFoundError:
        return False

ENV_LOADED = load_dotenv()

# ============================ Config via ENV ===========================

def _getenv(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()

def _getenv_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default

def _getenv_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, str(default))))
    except ValueError:
        return default

def _getenv_list(name: str, default_list: List[str]) -> List[str]:
    raw = os.environ.get(name)
    if not raw:
        return default_list
    return [x.strip() for x in raw.split(",") if x.strip()]

# Domains (keep in sync with your CSE patterns; apex domains are fine here)
ATS_ALLOWED_DOMAINS = set(_getenv_list(
    "ATS_ALLOWED_DOMAINS",
    [
        "boards.greenhouse.io",
        "jobs.lever.co",
        "myworkdayjobs.com",
        "ashbyhq.com",
        "smartrecruiters.com",
        "www.smartrecruiters.com",
        "careers.smartrecruiters.com",
    ],
))

ATS_SEARCH_DOMAINS = _getenv_list(
    "ATS_SEARCH_DOMAINS",
    [
        "boards.greenhouse.io",
        "jobs.lever.co",
        "myworkdayjobs.com",
        "ashbyhq.com",
        "smartrecruiters.com",
    ],
)

# API URL templates (customizable in .env)
GREENHOUSE_API_TEMPLATE = _getenv(
    "GREENHOUSE_API_TEMPLATE",
    "https://boards-api.greenhouse.io/v1/boards/{company}/jobs?content=true",
)
LEVER_API_TEMPLATE = _getenv(
    "LEVER_API_TEMPLATE",
    "https://api.lever.co/v0/postings/{company}?mode=json",
)
ASHBY_API_TEMPLATE = _getenv(
    "ASHBY_API_TEMPLATE",
    "https://api.ashbyhq.com/posting-api/job-board/{board}",
)
SMARTRECRUITERS_API_TEMPLATE = _getenv(
    "SMARTRECRUITERS_API_TEMPLATE",
    "https://api.smartrecruiters.com/v1/companies/{company}/postings?limit=100&offset=0",
)

# Other runtime knobs
DEFAULT_USER_AGENT = _getenv(
    "JOBS_USER_AGENT",
    "ReactJobsFinderBot/1.4 (+https://yourdomain.example/jobsfinder; contact: mailto:you@example.com)"
)
REQUEST_TIMEOUT = _getenv_int("REQUEST_TIMEOUT_SECONDS", 20)
BASE_SLEEP = _getenv_float("BASE_SLEEP_SECONDS", 1.0)

# API credentials (from env or .env)
GOOGLE_CSE_KEY = os.environ.get("GOOGLE_CSE_KEY", "")
GOOGLE_CSE_CX = os.environ.get("GOOGLE_CSE_CX", "")
SERPAPI_KEY = os.environ.get("SERPAPI_KEY", "")
SMARTRECRUITERS_TOKEN = os.environ.get("SMARTRECRUITERS_TOKEN", "")

# ============================ Progress / Debug =========================

PROGRESS = True   # set by CLI
DEBUG = False     # set by CLI

def parse_bool(s: Optional[str]) -> bool:
    if s is None:
        return False
    return str(s).strip().lower() in {"1", "true", "t", "yes", "y", "on"}

def log(msg: str) -> None:
    """Milestone logger to stderr with HH:MM:SS timestamps."""
    if PROGRESS or DEBUG:
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] {msg}", file=sys.stderr)

def dlog(msg: str) -> None:
    """Verbose debug logger."""
    if DEBUG:
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] [DEBUG] {msg}", file=sys.stderr)

@contextmanager
def step(title: str):
    """Context manager to log step start/end and capture tracebacks when --debug=true."""
    log(f"▶ {title} — start")
    t0 = time.time()
    try:
        yield
    except SystemExit:
        # Let sys.exit() propagate without masking
        raise
    except Exception as e:
        log(f"✖ {title} — ERROR: {e}")
        if DEBUG:
            traceback.print_exc(file=sys.stderr)
        raise
    else:
        dt = time.time() - t0
        log(f"✔ {title} — done in {dt:.2f}s")

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

def is_allowed_by_robots(session: requests.Session, url: str, user_agent: str) -> bool:
    """Check robots.txt for HTML page fetches."""
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
            resp = session.get(robots_url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": user_agent})
            if resp.status_code == 200:
                rp.parse(resp.text.splitlines())
                dlog(f"robots.txt loaded for {parsed.netloc}")
            else:
                rp.parse([])
                log(f"robots.txt unavailable ({resp.status_code}) for {parsed.netloc}; proceeding cautiously")
        except requests.RequestException as e:
            rp.parse([])
            log(f"robots.txt fetch failed for {parsed.netloc}; proceeding cautiously ({e})")
        cache[robots_url] = rp

    allowed = rp.can_fetch(user_agent, url)
    if not allowed:
        log(f"Disallowed by robots.txt: {url}")
    return allowed

def sleep_with_backoff(attempt: int) -> None:
    delay = min(BASE_SLEEP * (2 ** attempt), 8.0)
    log(f"Backing off {delay:.1f}s (attempt {attempt+1})")
    time.sleep(delay)

def safe_get_json(session: requests.Session, url: str, headers: Dict[str, str]) -> Optional[dict]:
    for attempt in range(4):
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
    for attempt in range(4):
        try:
            if not is_allowed_by_robots(session, url, headers.get("User-Agent", DEFAULT_USER_AGENT)):
                return None
            dlog(f"GET HTML: {url}")
            r = session.get(url, timeout=REQUEST_TIMEOUT, headers=headers)
            dlog(f"→ status {r.status_code} content-type {r.headers.get('Content-Type')}")
            if r.status_code == 200 and "text/html" in r.headers.get("Content-Type", ""):
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

def to_markdown_table(rows: List[JobPosting], limit: int) -> str:
    cols = ["Company", "Career portal link", "Job title", "Location"]
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body_lines = []
    for jp in rows[:limit]:
        body_lines.append(f"| {jp.company} | {jp.link} | {jp.title} | {jp.location} |")
    return "\n".join([head, sep] + body_lines)

def normalize_domain(url: str) -> str:
    return urlparse(url).netloc.lower()

def is_allowed_domain(url: str) -> bool:
    host = normalize_domain(url)
    for d in ATS_ALLOWED_DOMAINS:
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

def fmt(template: str, **kwargs: str) -> str:
    return template.format(**kwargs)

# ============================ Search layer ============================

def build_search_query(keywords: List[str]) -> str:
    kq = " OR ".join([f"\"{k}\"" for k in keywords])
    dq = " OR ".join([f"site:{d}" for d in ATS_SEARCH_DOMAINS])
    return f"({kq}) ({dq})"

def search_google_cse(session: requests.Session, query: str, limit: int) -> List[str]:
    if not GOOGLE_CSE_KEY or not GOOGLE_CSE_CX:
        log("Google CSE not configured; skipping")
        return []
    endpoint = "https://www.googleapis.com/customsearch/v1"
    urls: List[str] = []
    start = 1
    per_page = 10
    log("Discovery: Google CSE → querying…")
    while len(urls) < limit and start <= 100:
        params = {"key": GOOGLE_CSE_KEY, "cx": GOOGLE_CSE_CX, "q": query, "num": min(per_page, limit - len(urls)), "start": start}
        try:
            dlog(f"CSE request start={start} q={query}")
            r = session.get(endpoint, params=params, timeout=REQUEST_TIMEOUT)
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
            log(f"Google CSE page start={start}: got {len(urls)-before} allowed URLs (total {len(urls)})")
            if not items:
                break
            start += per_page
        except requests.RequestException as e:
            log(f"Google CSE error: {e}")
            break
    return urls

def search_serpapi(session: requests.Session, query: str, limit: int) -> List[str]:
    if not SERPAPI_KEY:
        log("SerpAPI not configured; skipping")
        return []
    endpoint = "https://serpapi.com/search.json"
    params = {"engine": "google", "q": query, "num": min(limit, 50), "api_key": SERPAPI_KEY}
    log("Discovery: SerpAPI fallback → querying…")
    try:
        dlog(f"SerpAPI request q={query}")
        r = session.get(endpoint, params=params, timeout=REQUEST_TIMEOUT)
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
        for fn in (search_google_cse, search_serpapi):
            try:
                results = fn(session, query, limit * 3)  # oversample
                urls.extend(results)
            except Exception as e:
                log(f"Discovery function error: {e}")
                if DEBUG:
                    traceback.print_exc(file=sys.stderr)
                continue
            if len(urls) >= limit * 2:
                break
        # Deduplicate, keep order
        seen: Set[str] = set()
        deduped: List[str] = []
        for u in urls:
            if u not in seen and is_allowed_domain(u):
                deduped.append(u)
                seen.add(u)
            if len(deduped) >= limit * 2:
                break
        log(f"Discovery: found {len(urls)} candidates → {len(deduped)} after dedupe/filter")
        return deduped

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

# ---- Official APIs

def parse_greenhouse_company_jobs(session: requests.Session, url: str, kw_re: re.Pattern) -> List[JobPosting]:
    slug = first_path_segment(url)
    if not slug:
        return []
    api = fmt(GREENHOUSE_API_TEMPLATE, company=slug)
    log(f"Greenhouse: {slug} → API")
    data = safe_get_json(session, api, headers={"User-Agent": DEFAULT_USER_AGENT, "Accept": "application/json"})
    if not data or "jobs" not in data:
        log(f"Greenhouse: {slug} → no data")
        return []
    jobs: List[JobPosting] = []
    company = slug_to_company(slug)
    matches = 0
    for job in data.get("jobs", []):
        title = (job.get("title") or "").strip()
        if not title or not kw_re.search(title):
            continue
        location = (job.get("location", {}) or {}).get("name", "") or ""
        link = job.get("absolute_url") or ""
        if link:
            jobs.append(JobPosting(company=company, link=link, title=title, location=location))
            matches += 1
    log(f"Greenhouse: {slug} → {matches} matches")
    return jobs

def parse_lever_company_jobs(session: requests.Session, url: str, kw_re: re.Pattern) -> List[JobPosting]:
    slug = first_path_segment(url)
    if not slug:
        return []
    api = fmt(LEVER_API_TEMPLATE, company=slug)
    log(f"Lever: {slug} → API")
    data = safe_get_json(session, api, headers={"User-Agent": DEFAULT_USER_AGENT, "Accept": "application/json"})
    if not data or not isinstance(data, list):
        log(f"Lever: {slug} → no data")
        return []
    jobs: List[JobPosting] = []
    company = slug_to_company(slug)
    matches = 0
    for item in data:
        title = item.get("text") or item.get("title") or item.get("name") or ""
        if not title or not kw_re.search(str(title)):
            continue
        loc = ""
        cats = item.get("categories") or {}
        if isinstance(cats, dict):
            loc = cats.get("location") or ""
        if not loc:
            loc = item.get("country") or item.get("workplaceType") or ""
        link = item.get("hostedUrl") or item.get("applyUrl") or item.get("url") or ""
        if link:
            jobs.append(JobPosting(company=company, link=link, title=str(title), location=str(loc)))
            matches += 1
    log(f"Lever: {slug} → {matches} matches")
    return jobs

def parse_ashby_company_jobs(session: requests.Session, url: str, kw_re: re.Pattern) -> List[JobPosting]:
    board = first_path_segment(url)
    if not board:
        return []
    api = fmt(ASHBY_API_TEMPLATE, board=board)
    log(f"Ashby: {board} → API")
    data = safe_get_json(session, api, headers={"User-Agent": DEFAULT_USER_AGENT, "Accept": "application/json"})
    if not data or "jobs" not in data:
        log(f"Ashby: {board} → no data")
        return []
    company = slug_to_company(board)
    jobs: List[JobPosting] = []
    matches = 0
    for j in data.get("jobs", []):
        title = (j.get("title") or "").strip()
        if not title or not kw_re.search(title):
            continue
        location = (j.get("location") or "").strip()
        link = (j.get("jobUrl") or "").strip()
        if link:
            jobs.append(JobPosting(company=company, link=link, title=title, location=location))
            matches += 1
    log(f"Ashby: {board} → {matches} matches")
    return jobs

def parse_smartrecruiters_company_jobs_api(session: requests.Session, url: str, kw_re: re.Pattern) -> List[JobPosting]:
    if not SMARTRECRUITERS_TOKEN:
        return []
    company_id = first_path_segment(url)
    if not company_id:
        return []
    api = fmt(SMARTRECRUITERS_API_TEMPLATE, company=company_id)
    log(f"SmartRecruiters API: {company_id} → API")
    headers = {"User-Agent": DEFAULT_USER_AGENT, "Accept": "application/json", "X-SmartToken": SMARTRECRUITERS_TOKEN}
    data = safe_get_json(session, api, headers=headers)
    if not data or "content" not in data:
        log(f"SmartRecruiters API: {company_id} → no data")
        return []
    company = slug_to_company(company_id)
    jobs: List[JobPosting] = []
    matches = 0
    for item in data.get("content", []):
        title = (item.get("name") or "").strip()
        if not title or not kw_re.search(title):
            continue
        link = ""
        if isinstance(item.get("jobAd"), dict):
            link = (item.get("jobAd") or {}).get("url") or ""
        link = link or item.get("jobAdUrl") or ""
        loc_obj = item.get("location") or {}
        if isinstance(loc_obj, dict):
            parts = [loc_obj.get("city"), loc_obj.get("region"), loc_obj.get("country")]
            location = ", ".join([p for p in parts if p])
        else:
            location = ""
        if link:
            jobs.append(JobPosting(company=company, link=link, title=title, location=location))
            matches += 1
    log(f"SmartRecruiters API: {company_id} → {matches} matches")
    return jobs

# ---- JSON-LD fallback (e.g., Workday, generic)

def parse_jsonld_jobposting(html: str) -> Optional[Tuple[str, str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    scripts = soup.find_all("script", attrs={"type": "application/ld+json"})
    for s in scripts:
        try:
            data = json.loads(s.string or "")
        except Exception:
            continue
        candidates = data if isinstance(data, list) else [data]
        for d in candidates:
            if not isinstance(d, dict):
                continue
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
    log(f"HTML parse: {url}")
    html = safe_get_html(session, url, headers={"User-Agent": DEFAULT_USER_AGENT})
    if not html:
        return []
    extracted = parse_jsonld_jobposting(html)
    if not extracted:
        log("HTML parse: JSON-LD not found")
        return []
    title, company, location = extracted
    if title and kw_re.search(title):
        comp = company or slug_to_company(first_path_segment(url))
        log("HTML parse: match ✔")
        return [JobPosting(company=comp, link=url, title=title, location=location or "")]
    log("HTML parse: no keyword match")
    return []

# ============================ Orchestrator ============================

def harvest_jobs(session: requests.Session, discovered_urls: List[str], keywords: List[str], limit: int) -> List[JobPosting]:
    with step("HARVEST"):
        kw_re = build_keyword_regex(keywords)
        out: List[JobPosting] = []

        processed_company_slugs: Dict[str, Set[str]] = {
            "greenhouse": set(),
            "lever": set(),
            "ashby": set(),
            "smartrecruiters": set(),
        }

        counts = {"greenhouse": 0, "lever": 0, "ashby": 0, "smartrecruiters": 0, "generic": 0}

        log(f"Limit: {limit} | Keywords: {keywords} | UA: {DEFAULT_USER_AGENT}")
        for url in discovered_urls:
            if len(out) >= limit:
                break
            provider = detect_provider(url)
            dlog(f"URL provider={provider} → {url}")
            try:
                if provider == "greenhouse":
                    slug = first_path_segment(url)
                    if slug and slug not in processed_company_slugs["greenhouse"]:
                        rows = parse_greenhouse_company_jobs(session, url, kw_re)
                        processed_company_slugs["greenhouse"].add(slug)
                        out.extend(rows)
                        counts["greenhouse"] += len(rows)

                elif provider == "lever":
                    slug = first_path_segment(url)
                    if slug and slug not in processed_company_slugs["lever"]:
                        rows = parse_lever_company_jobs(session, url, kw_re)
                        processed_company_slugs["lever"].add(slug)
                        out.extend(rows)
                        counts["lever"] += len(rows)

                elif provider == "ashby":
                    board = first_path_segment(url)
                    if board and board not in processed_company_slugs["ashby"]:
                        rows = parse_ashby_company_jobs(session, url, kw_re)
                        processed_company_slugs["ashby"].add(board)
                        out.extend(rows)
                        counts["ashby"] += len(rows)
                    # Fallback: single-page JSON-LD (only if needed)
                    if len(out) < limit:
                        rows = parse_generic_page(session, url, kw_re)
                        out.extend(rows)
                        counts["generic"] += len(rows)

                elif provider == "smartrecruiters":
                    comp = first_path_segment(url)
                    used_api = False
                    if comp and comp not in processed_company_slugs["smartrecruiters"]:
                        api_rows = parse_smartrecruiters_company_jobs_api(session, url, kw_re)
                        if api_rows:
                            out.extend(api_rows)
                            counts["smartrecruiters"] += len(api_rows)
                            used_api = True
                        processed_company_slugs["smartrecruiters"].add(comp)
                    if not used_api and len(out) < limit:
                        rows = parse_generic_page(session, url, kw_re)
                        out.extend(rows)
                        counts["generic"] += len(rows)

                else:
                    # Workday / unknown → single-page JSON-LD
                    rows = parse_generic_page(session, url, kw_re)
                    out.extend(rows)
                    counts["generic"] += len(rows)

            except Exception as e:
                log(f"Harvest error on {url}: {e}")
                if DEBUG:
                    traceback.print_exc(file=sys.stderr)
                continue

            # polite spacing between requests
            time.sleep(BASE_SLEEP * 0.6)

            if len(out) >= limit:
                break

        # Deduplicate by link (stable order)
        seen_links: Set[str] = set()
        unique: List[JobPosting] = []
        for j in out:
            if j.link not in seen_links:
                unique.append(j)
                seen_links.add(j.link)

        log("---------- SUMMARY ----------")
        log(f"Greenhouse: {counts['greenhouse']} | Lever: {counts['lever']} | Ashby: {counts['ashby']} | SmartRecruiters: {counts['smartrecruiters']} | HTML(JSON-LD): {counts['generic']}")
        log(f"Total matches (pre-unique): {len(out)} → unique: {len(unique)}")
        return unique[:limit]

def write_csv(rows: List[JobPosting], out_path: str) -> None:
    with step("WRITE CSV"):
        df = pd.DataFrame([{
            "Company": r.company,
            "Career portal link": r.link,
            "Job title": r.title,
            "Location": r.location
        } for r in rows])
        df.to_csv(out_path, index=False)
        log(f"CSV written → {out_path} ({len(rows)} rows)")

# ================================ Main ================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Find current React/React Native job openings from common ATS providers.")
    p.add_argument("--keywords", nargs="+", default=["React", "React Native"], help='Keywords to match in job titles')
    p.add_argument("--limit", type=int, default=50, help="Maximum number of results to output")
    p.add_argument("--out", type=str, default="react_jobs.csv", help="Path to write CSV output")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--quiet", action="store_true", help="Suppress milestone logs")
    g.add_argument("--progress", action="store_true", help="Show milestone logs (default behavior)")
    p.add_argument("--debug", type=str, default="false", help="Enable debugger mode: true|false (default false)")
    return p.parse_args()

def main() -> None:
    global PROGRESS, DEBUG
    args = parse_args()
    DEBUG = parse_bool(args.debug)
    PROGRESS = (not args.quiet) or DEBUG  # if debug, always show logs

    with step("STARTUP"):
        log(f".env loaded: {'yes' if ENV_LOADED else 'no'}")
        log(f"User-Agent: {DEFAULT_USER_AGENT}")
        log(f"Allowed domains: {sorted(ATS_ALLOWED_DOMAINS)}")
        if DEBUG:
            dlog(f"REQUEST_TIMEOUT={REQUEST_TIMEOUT}s BASE_SLEEP={BASE_SLEEP}s")
            dlog(f"GOOGLE_CSE configured? {'yes' if (GOOGLE_CSE_KEY and GOOGLE_CSE_CX) else 'no'} | SERPAPI configured? {'yes' if SERPAPI_KEY else 'no'} | SMARTRECRUITERS token? {'yes' if SMARTRECRUITERS_TOKEN else 'no'}")

    with step("INIT SESSION"):
        session = make_session()

    with step("DISCOVER URLS"):
        discovered = discover_posting_urls(session, args.keywords, args.limit)
        if not discovered:
            log("No postings discovered via search (check GOOGLE_CSE_* or SERPAPI_KEY).")
            print("No postings discovered via search (check GOOGLE_CSE_* or SERPAPI_KEY).", file=sys.stderr)
            sys.exit(2)

    rows = harvest_jobs(session, discovered, args.keywords, args.limit)
    if not rows:
        log("No matching jobs found after parsing.")
        print("No matching jobs found after parsing.", file=sys.stderr)
        sys.exit(3)

    write_csv(rows, args.out)

    with step("PRINT TABLE"):
        print(to_markdown_table(rows, min(len(rows), args.limit)))

    log("DONE ✅")
    log(f"Saved CSV: {args.out}")

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
