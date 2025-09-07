#!/usr/bin/env python3
"""
react_jobs_finder.py — Find CURRENT “React” / “React Native” job openings from common ATS providers.

Outputs:
  1) CSV file with columns: Company | Career portal link | Job title | Location
  2) Printed Markdown table (top N rows) to stdout

What’s new in this refactor:
1) **.env support (no third-party libs):** loads a local `.env` file (if present) so you can keep keys & config out of source control.
2) **Static ATS config moved to env:** allowed/search domains and official API URL templates are now read from environment variables (or `.env`), with safe defaults in code.
3) **Discovery priority:** Google CSE (free 100/day) first, SerpAPI fallback.

Providers & strategy:
- Prefer official/public endpoints where available:
  • Greenhouse  -> Job Board API (company → job list)
  • Lever       -> Postings API (company → job list)
  • Ashby       -> Public Job Board API (board → job list)
  • SmartRecruiters -> Customer Posting API (optional; needs token); else parse JSON-LD on the job page
- Otherwise (e.g., Workday): fetch the specific discovered job page and read JSON-LD JobPosting (no broad crawling).

Ethics & safety:
- Respects robots.txt for HTML page fetches.
- Polite rate limiting + retries with exponential backoff.
- Identifying User-Agent.
- Only stdlib + requests + beautifulsoup4 + pandas.

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

# ============================ .env loader ==============================

def load_dotenv(path: str = ".env", override: bool = False) -> None:
    """
    Minimal .env loader (no dependencies).
    - Lines like KEY=VALUE (quotes optional) are supported.
    - Ignores blank lines and lines starting with #.
    - By default, existing environment variables WIN; set override=True to overwrite.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if override or key not in os.environ:
                    os.environ[key] = val
    except FileNotFoundError:
        pass

# Load .env early so os.environ is populated before reading config
load_dotenv()

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

# Domains (can be customized in .env)
ATS_ALLOWED_DOMAINS = set(_getenv_list(
    "ATS_ALLOWED_DOMAINS",
    [
        "boards.greenhouse.io",
        "jobs.lever.co",
        "myworkdayjobs.com",
        "ashbyhq.com",
        "jobs.ashbyhq.com",
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
    "Mozilla/5.0 (compatible; ReactJobsFinder/1.3; +https://example.com)"
)
REQUEST_TIMEOUT = _getenv_int("REQUEST_TIMEOUT_SECONDS", 20)
BASE_SLEEP = _getenv_float("BASE_SLEEP_SECONDS", 1.0)

# API credentials (from env or .env)
GOOGLE_CSE_KEY = os.environ.get("GOOGLE_CSE_KEY", "")
GOOGLE_CSE_CX = os.environ.get("GOOGLE_CSE_CX", "")
SERPAPI_KEY = os.environ.get("SERPAPI_KEY", "")
SMARTRECRUITERS_TOKEN = os.environ.get("SMARTRECRUITERS_TOKEN", "")

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
            resp = session.get(robots_url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": user_agent})
            if resp.status_code == 200:
                rp.parse(resp.text.splitlines())
            else:
                rp.parse([])
        except requests.RequestException:
            rp.parse([])
        cache[robots_url] = rp

    return rp.can_fetch(user_agent, url)

def sleep_with_backoff(attempt: int) -> None:
    delay = min(BASE_SLEEP * (2 ** attempt), 8.0)
    time.sleep(delay)

def safe_get_json(session: requests.Session, url: str, headers: Dict[str, str]) -> Optional[dict]:
    for attempt in range(4):
        try:
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
    """Small helper to format API templates safely."""
    return template.format(**kwargs)

# ============================ Search layer ============================

def build_search_query(keywords: List[str]) -> str:
    kq = " OR ".join([f"\"{k}\"" for k in keywords])
    dq = " OR ".join([f"site:{d}" for d in ATS_SEARCH_DOMAINS])
    return f"({kq}) ({dq})"

def search_google_cse(session: requests.Session, query: str, limit: int) -> List[str]:
    if not GOOGLE_CSE_KEY or not GOOGLE_CSE_CX:
        return []
    endpoint = "https://www.googleapis.com/customsearch/v1"
    urls: List[str] = []
    start = 1
    per_page = 10
    while len(urls) < limit and start <= 100:
        params = {"key": GOOGLE_CSE_KEY, "cx": GOOGLE_CSE_CX, "q": query, "num": min(per_page, limit - len(urls)), "start": start}
        try:
            r = session.get(endpoint, params=params, timeout=REQUEST_TIMEOUT)
            if r.status_code != 200:
                break
            data = r.json()
            items = data.get("items", []) if isinstance(data, dict) else []
            batch = [it.get("link") for it in items if it.get("link")]
            for u in batch:
                if u and is_allowed_domain(u):
                    urls.append(u)
            if not items:
                break
            start += per_page
        except requests.RequestException:
            break
    return urls

def search_serpapi(session: requests.Session, query: str, limit: int) -> List[str]:
    if not SERPAPI_KEY:
        return []
    endpoint = "https://serpapi.com/search.json"
    params = {"engine": "google", "q": query, "num": min(limit, 50), "api_key": SERPAPI_KEY}
    try:
        r = session.get(endpoint, params=params, timeout=REQUEST_TIMEOUT)
        if r.status_code == 200:
            data = r.json()
            items = data.get("organic_results", [])
            urls = [it.get("link") for it in items if it.get("link")]
            return [u for u in urls if u and is_allowed_domain(u)]
    except requests.RequestException:
        return []
    return []

def discover_posting_urls(session: requests.Session, keywords: List[str], limit: int) -> List[str]:
    """
    Discovery priority:
      1) Google CSE (free 100/day)
      2) SerpAPI (fallback)
    """
    query = build_search_query(keywords)
    urls: List[str] = []
    for fn in (search_google_cse, search_serpapi):
        try:
            results = fn(session, query, limit * 3)  # oversample
            urls.extend(results)
        except Exception:
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
    data = safe_get_json(session, api, headers={"User-Agent": DEFAULT_USER_AGENT, "Accept": "application/json"})
    if not data or "jobs" not in data:
        return []
    jobs: List[JobPosting] = []
    company = slug_to_company(slug)
    for job in data.get("jobs", []):
        title = (job.get("title") or "").strip()
        if not title or not kw_re.search(title):
            continue
        location = (job.get("location", {}) or {}).get("name", "") or ""
        link = job.get("absolute_url") or ""
        if link:
            jobs.append(JobPosting(company=company, link=link, title=title, location=location))
    return jobs

def parse_lever_company_jobs(session: requests.Session, url: str, kw_re: re.Pattern) -> List[JobPosting]:
    slug = first_path_segment(url)
    if not slug:
        return []
    api = fmt(LEVER_API_TEMPLATE, company=slug)
    data = safe_get_json(session, api, headers={"User-Agent": DEFAULT_USER_AGENT, "Accept": "application/json"})
    if not data or not isinstance(data, list):
        return []
    jobs: List[JobPosting] = []
    company = slug_to_company(slug)
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
    return jobs

def parse_ashby_company_jobs(session: requests.Session, url: str, kw_re: re.Pattern) -> List[JobPosting]:
    board = first_path_segment(url)
    if not board:
        return []
    api = fmt(ASHBY_API_TEMPLATE, board=board)
    data = safe_get_json(session, api, headers={"User-Agent": DEFAULT_USER_AGENT, "Accept": "application/json"})
    if not data or "jobs" not in data:
        return []
    company = slug_to_company(board)
    jobs: List[JobPosting] = []
    for j in data.get("jobs", []):
        title = (j.get("title") or "").strip()
        if not title or not kw_re.search(title):
            continue
        location = (j.get("location") or "").strip()
        link = (j.get("jobUrl") or "").strip()
        if link:
            jobs.append(JobPosting(company=company, link=link, title=title, location=location))
    return jobs

def parse_smartrecruiters_company_jobs_api(session: requests.Session, url: str, kw_re: re.Pattern) -> List[JobPosting]:
    if not SMARTRECRUITERS_TOKEN:
        return []
    company_id = first_path_segment(url)
    if not company_id:
        return []
    api = fmt(SMARTRECRUITERS_API_TEMPLATE, company=company_id)
    headers = {"User-Agent": DEFAULT_USER_AGENT, "Accept": "application/json", "X-SmartToken": SMARTRECRUITERS_TOKEN}
    data = safe_get_json(session, api, headers=headers)
    if not data or "content" not in data:
        return []
    company = slug_to_company(company_id)
    jobs: List[JobPosting] = []
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
    html = safe_get_html(session, url, headers={"User-Agent": DEFAULT_USER_AGENT})
    if not html:
        return []
    extracted = parse_jsonld_jobposting(html)
    if not extracted:
        return []
    title, company, location = extracted
    if title and kw_re.search(title):
        comp = company or slug_to_company(first_path_segment(url))
        return [JobPosting(company=comp, link=url, title=title, location=location or "")]
    return []

# ============================ Orchestrator ============================

def harvest_jobs(session: requests.Session, discovered_urls: List[str], keywords: List[str], limit: int) -> List[JobPosting]:
    kw_re = build_keyword_regex(keywords)
    out: List[JobPosting] = []

    processed_company_slugs: Dict[str, Set[str]] = {
        "greenhouse": set(),
        "lever": set(),
        "ashby": set(),
        "smartrecruiters": set(),
    }

    for url in discovered_urls:
        if len(out) >= limit:
            break
        provider = detect_provider(url)
        try:
            if provider == "greenhouse":
                slug = first_path_segment(url)
                if slug and slug not in processed_company_slugs["greenhouse"]:
                    out.extend(parse_greenhouse_company_jobs(session, url, kw_re))
                    processed_company_slugs["greenhouse"].add(slug)

            elif provider == "lever":
                slug = first_path_segment(url)
                if slug and slug not in processed_company_slugs["lever"]:
                    out.extend(parse_lever_company_jobs(session, url, kw_re))
                    processed_company_slugs["lever"].add(slug)

            elif provider == "ashby":
                board = first_path_segment(url)
                if board and board not in processed_company_slugs["ashby"]:
                    found = parse_ashby_company_jobs(session, url, kw_re)
                    out.extend(found)
                    processed_company_slugs["ashby"].add(board)
                # If nothing found via API, try single-page JSON-LD fallback
                if not any(j.link.startswith("https://jobs.ashbyhq.com/") for j in out):
                    out.extend(parse_generic_page(session, url, kw_re))

            elif provider == "smartrecruiters":
                comp = first_path_segment(url)
                used_api = False
                if comp and comp not in processed_company_slugs["smartrecruiters"]:
                    api_rows = parse_smartrecruiters_company_jobs_api(session, url, kw_re)
                    if api_rows:
                        out.extend(api_rows)
                        used_api = True
                    processed_company_slugs["smartrecruiters"].add(comp)
                if not used_api:
                    out.extend(parse_generic_page(session, url, kw_re))

            else:
                # Workday / unknown -> parse page via JSON-LD (single page only)
                out.extend(parse_generic_page(session, url, kw_re))

        except Exception:
            continue

        time.sleep(BASE_SLEEP * 0.6)
        if len(out) >= limit:
            break

    # Deduplicate by link
    seen_links: Set[str] = set()
    unique: List[JobPosting] = []
    for j in out:
        if j.link not in seen_links:
            unique.append(j)
            seen_links.add(j.link)

    return unique[:limit]

def write_csv(rows: List[JobPosting], out_path: str) -> None:
    df = pd.DataFrame([{
        "Company": r.company,
        "Career portal link": r.link,
        "Job title": r.title,
        "Location": r.location
    } for r in rows])
    df.to_csv(out_path, index=False)

# ================================ Main ================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Find current React/React Native job openings from common ATS providers.")
    p.add_argument("--keywords", nargs="+", default=["React", "React Native"], help='Keywords to match in job titles')
    p.add_argument("--limit", type=int, default=50, help="Maximum number of results to output")
    p.add_argument("--out", type=str, default="react_jobs.csv", help="Path to write CSV output")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    session = make_session()

    discovered = discover_posting_urls(session, args.keywords, args.limit)
    if not discovered:
        print("No postings discovered via search (check GOOGLE_CSE_* or SERPAPI_KEY).", file=sys.stderr)
        sys.exit(2)

    rows = harvest_jobs(session, discovered, args.keywords, args.limit)
    if not rows:
        print("No matching jobs found after parsing.", file=sys.stderr)
        sys.exit(3)

    write_csv(rows, args.out)
    print(to_markdown_table(rows, min(len(rows), args.limit)))
    print(f"\nSaved CSV: {args.out}", file=sys.stderr)

if __name__ == "__main__":
    main()

# ================================ USAGE ===============================
"""
USAGE

0) Keep secrets & config out of git:
   - Create a local `.env` file next to this script and add it to `.gitignore`:
       echo ".env" >> .gitignore
   - Example `.env` (edit values as needed):
       # Search APIs
       GOOGLE_CSE_KEY=your_google_key
       GOOGLE_CSE_CX=your_google_cx
       SERPAPI_KEY=your_serpapi_key

       # Optional SmartRecruiters token (if you have one)
       SMARTRECRUITERS_TOKEN=your_smart_token

       # Domains (comma-separated)
       ATS_ALLOWED_DOMAINS=boards.greenhouse.io,jobs.lever.co,myworkdayjobs.com,ashbyhq.com,jobs.ashbyhq.com,smartrecruiters.com,www.smartrecruiters.com,careers.smartrecruiters.com
       ATS_SEARCH_DOMAINS=boards.greenhouse.io,jobs.lever.co,myworkdayjobs.com,ashbyhq.com,smartrecruiters.com

       # Official API URL templates (override only if needed)
       GREENHOUSE_API_TEMPLATE=https://boards-api.greenhouse.io/v1/boards/{company}/jobs?content=true
       LEVER_API_TEMPLATE=https://api.lever.co/v0/postings/{company}?mode=json
       ASHBY_API_TEMPLATE=https://api.ashbyhq.com/posting-api/job-board/{board}
       SMARTRECRUITERS_API_TEMPLATE=https://api.smartrecruiters.com/v1/companies/{company}/postings?limit=100&offset=0

       # Politeness / runtime
       JOBS_USER_AGENT=Mozilla/5.0 (compatible; ReactJobsFinder/1.3; +https://example.com)
       REQUEST_TIMEOUT_SECONDS=20
       BASE_SLEEP_SECONDS=1.0

1) Install dependencies (Python 3.9+):
   pip install requests beautifulsoup4 pandas

2) Run:
   python3 react_jobs_finder.py --keywords React "React Native" --limit 50 --out jobs.csv

3) Output:
   - CSV file: Company | Career portal link | Job title | Location
   - Markdown table printed to stdout.

Notes
- **Discovery priority:** Google CSE (free 100/day) → SerpAPI fallback.
- **Ethical scraping:** Respects robots.txt for HTML; uses official ATS APIs where available.
- **Config in env:** You can also export env vars in your shell instead of using `.env`.
"""
