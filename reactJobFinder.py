#!/usr/bin/env python3
"""
react_jobs_finder.py — Find CURRENT “React” / “React Native” job openings from common ATS providers.

Outputs:
  1) CSV file with columns: Company | Career portal link | Job title | Location
  2) Printed Markdown table (top N rows) to stdout

Providers & strategy:
- Prefer official/public endpoints when available:
  • Greenhouse  -> https://boards-api.greenhouse.io/v1/boards/{company}/jobs?content=true
  • Lever       -> https://api.lever.co/v0/postings/{company}?mode=json
  • SmartRecruiters -> JSON-LD present on posting pages (and public API exists per-company)
  • Ashby / Workday -> parse JSON-LD on posting pages
- Discovery via search (limited to ATS domains). Uses, in order of availability:
  • Bing Web Search API (env: BING_API_KEY)
  • SerpAPI (env: SERPAPI_KEY)
  • DuckDuckGo HTML fallback (only if robots.txt allows)
- Respect robots.txt for every host before fetching.
- Polite rate limiting + retries with exponential backoff.
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
from typing import Dict, Iterable, List, Optional, Set, Tuple
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
    "Mozilla/5.0 (compatible; ReactJobsFinder/1.0; +https://example.com)"
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
    cache_key = ("robots", robots_url)

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
                # If robots.txt missing or error, default to allowing (conservative alternative would be disallow).
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
            if not is_allowed_by_robots(session, url, headers.get("User-Agent", DEFAULT_USER_AGENT)):
                return None
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
    # allow subdomains of myworkdayjobs.com, smartrecruiters.com, ashbyhq.com
    for d in ALLOWED_DOMAINS:
        if host == d or host.endswith("." + d):
            return True
    return False

# ----------------------------- Search layer ---------------------------

def make_session(user_agent: str = DEFAULT_USER_AGENT) -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent, "Accept-Language": "en-US,en;q=0.9"})
    return session


def build_search_query(keywords: List[str]) -> str:
    kq = " OR ".join([f"\"{k}\"" for k in keywords])
    dq = " OR ".join([f"site:{d}" for d in SEARCH_QUERY_DOMAINS])
    return f"({kq}) ({dq})"


def search_bing(session: requests.Session, query: str, limit: int) -> List[str]:
    api_key = os.getenv("BING_API_KEY")
    if not api_key:
        return []
    endpoint = "https://api.bing.microsoft.com/v7.0/search"
    params = {
        "q": query,
        "count": min(limit, 50),
        "responseFilter": "Webpages",
        "safeSearch": "Moderate",
    }
    headers = {"Ocp-Apim-Subscription-Key": api_key, "User-Agent": DEFAULT_USER_AGENT}
    try:
        r = session.get(endpoint, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
        if r.status_code == 200:
            data = r.json()
            items = data.get("webPages", {}).get("value", []) if isinstance(data, dict) else []
            urls = [it.get("url") for it in items if it.get("url")]
            return [u for u in urls if u and is_allowed_domain(u)]
    except requests.RequestException:
        return []
    return []


def search_serpapi(session: requests.Session, query: str, limit: int) -> List[str]:
    api_key = os.getenv("SERPAPI_KEY")
    if not api_key:
        return []
    endpoint = "https://serpapi.com/search.json"
    # Use Google engine with domain filters inside the query
    params = {
        "engine": "google",
        "q": query,
        "num": min(limit, 50),
        "api_key": api_key,
    }
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


def search_duckduckgo_html(session: requests.Session, query: str, limit: int) -> List[str]:
    # HTML fallback (only if robots allows). This is a last resort.
    base = "https://duckduckgo.com/html/"
    params = {"q": query}
    qurl = f"{base}?{urlencode(params)}"
    if not is_allowed_by_robots(session, qurl, DEFAULT_USER_AGENT):
        return []
    html = safe_get_html(session, qurl, headers={"User-Agent": DEFAULT_USER_AGENT}) or ""
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    links: List[str] = []
    for a in soup.select("a.result__a, a.result__url"):
        href = a.get("href")
        if href and href.startswith("http") and is_allowed_domain(href):
            links.append(href)
        if len(links) >= limit:
            break
    return links


def discover_posting_urls(session: requests.Session, keywords: List[str], limit: int) -> List[str]:
    query = build_search_query(keywords)
    urls: List[str] = []
    # Try providers in order
    for fn in (search_bing, search_serpapi, search_duckduckgo_html):
        try:
            results = fn(session, query, limit * 3)  # oversample a bit
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

# -------------------------- Provider detection ------------------------

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


def first_path_segment(url: str) -> str:
    path = urlparse(url).path.strip("/")
    return (path.split("/")[0] if path else "").strip()

# -------------------------- Parsers -----------------------------------

def parse_greenhouse_company_jobs(
    session: requests.Session, url: str, kw_re: re.Pattern
) -> List[JobPosting]:
    slug = first_path_segment(url)
    if not slug:
        return []
    api = f"https://boards-api.greenhouse.io/v1/boards/{slug}/jobs?content=true"
    data = safe_get_json(session, api, headers={"User-Agent": DEFAULT_USER_AGENT, "Accept": "application/json"})
    if not data or "jobs" not in data:
        return []
    jobs = []
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


def parse_lever_company_jobs(
    session: requests.Session, url: str, kw_re: re.Pattern
) -> List[JobPosting]:
    slug = first_path_segment(url)
    if not slug:
        return []
    api = f"https://api.lever.co/v0/postings/{slug}?mode=json"
    data = safe_get_json(session, api, headers={"User-Agent": DEFAULT_USER_AGENT, "Accept": "application/json"})
    if not data or not isinstance(data, list):
        return []
    jobs = []
    company = slug_to_company(slug)
    for item in data:
        # title fields vary across Lever tenants
        title = (
            item.get("text")
            or item.get("title")
            or item.get("name")
            or ""
        )
        if not title or not kw_re.search(str(title)):
            continue
        # location might be under categories.location, or "workplaceType"/"country"
        loc = ""
        cats = item.get("categories") or {}
        if isinstance(cats, dict):
            loc = cats.get("location") or ""
        if not loc:
            # fallback strings
            loc = item.get("country") or item.get("workplaceType") or ""
        link = item.get("hostedUrl") or item.get("applyUrl") or item.get("url") or ""
        if link:
            jobs.append(JobPosting(company=company, link=link, title=str(title), location=str(loc)))
    return jobs


def parse_jsonld_jobposting(html: str) -> Optional[Tuple[str, str, str]]:
    """
    Try to extract (title, company, location) from JSON-LD JobPosting blocks.
    """
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
                if isinstance(org, dict):
                    company = (org.get("name") or "").strip()
                else:
                    company = ""
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
                    # Try jobLocationType: "TELECOMMUTE"
                    jtype = d.get("jobLocationType")
                    if isinstance(jtype, str) and jtype:
                        location = jtype
                if title or company:
                    return title, company, location
    return None


def parse_generic_page(
    session: requests.Session, url: str, kw_re: re.Pattern
) -> List[JobPosting]:
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


# -------------------------- Orchestrator -------------------------------

def harvest_jobs(
    session: requests.Session,
    discovered_urls: List[str],
    keywords: List[str],
    limit: int,
) -> List[JobPosting]:
    kw_re = build_keyword_regex(keywords)
    out: List[JobPosting] = []

    processed_company_slugs: Dict[str, Set[str]] = {
        "greenhouse": set(),
        "lever": set(),
    }

    for url in discovered_urls:
        if len(out) >= limit:
            break
        provider = detect_provider(url)
        try:
            if provider == "greenhouse":
                slug = first_path_segment(url)
                if slug and slug not in processed_company_slugs["greenhouse"]:
                    jobs = parse_greenhouse_company_jobs(session, url, kw_re)
                    processed_company_slugs["greenhouse"].add(slug)
                    out.extend(jobs)
            elif provider == "lever":
                slug = first_path_segment(url)
                if slug and slug not in processed_company_slugs["lever"]:
                    jobs = parse_lever_company_jobs(session, url, kw_re)
                    processed_company_slugs["lever"].add(slug)
                    out.extend(jobs)
            else:
                # Workday / Ashby / SmartRecruiters / unknown -> parse page via JSON-LD
                out.extend(parse_generic_page(session, url, kw_re))
        except Exception:
            # be resilient; skip bad entries
            continue

        # rate limit between sources
        time.sleep(BASE_SLEEP * 0.6)

        # stop if we have enough
        if len(out) >= limit:
            break

    # Deduplicate by (link)
    seen_links: Set[str] = set()
    unique: List[JobPosting] = []
    for j in out:
        if j.link not in seen_links:
            unique.append(j)
            seen_links.add(j.link)

    return unique[:limit]


def write_csv(rows: List[JobPosting], out_path: str) -> None:
    df = pd.DataFrame([{"Company": r.company,
                        "Career portal link": r.link,
                        "Job title": r.title,
                        "Location": r.location} for r in rows])
    df.to_csv(out_path, index=False)


# -------------------------- Main --------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Find current React/React Native job openings from common ATS providers.")
    p.add_argument(
        "--keywords",
        nargs="+",
        default=["React", "React Native"],
        help='Keywords to match in job titles (default: ["React","React Native"])',
    )
    p.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum number of results to output (default: 50)",
    )
    p.add_argument(
        "--out",
        type=str,
        default="react_jobs.csv",
        help="Path to write CSV output (default: ./react_jobs.csv)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    session = make_session()

    # 1) Discover candidate posting URLs using the best available search backend
    discovered = discover_posting_urls(session, args.keywords, args.limit)
    if not discovered:
        print("No postings discovered via search (check API keys or network).", file=sys.stderr)
        print("Tip: set BING_API_KEY or SERPAPI_KEY in environment for stronger discovery.", file=sys.stderr)
        sys.exit(2)

    # 2) Harvest jobs using provider APIs (Greenhouse/Lever) or page JSON-LD fallbacks
    rows = harvest_jobs(session, discovered, args.keywords, args.limit)
    if not rows:
        print("No matching jobs found after parsing.", file=sys.stderr)
        sys.exit(3)

    # 3) Write CSV
    write_csv(rows, args.out)

    # 4) Print Markdown table of top N
    print(to_markdown_table(rows, min(len(rows), args.limit)))
    print(f"\nSaved CSV: {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()


# -------------------------- USAGE -------------------------------------
"""
USAGE

1) (Optional) Set a search API key for better discovery:
   - Bing Web Search API:
       export BING_API_KEY="YOUR_KEY"
   - or SerpAPI:
       export SERPAPI_KEY="YOUR_KEY"

   The script *only* searches these ATS domains:
     boards.greenhouse.io, jobs.lever.co, myworkdayjobs.com, ashbyhq.com, smartrecruiters.com

2) Install dependencies (Python 3.9+ recommended):
   pip install requests beautifulsoup4 pandas

3) Run:
   python3 react_jobs_finder.py --keywords React "React Native" --limit 50 --out jobs.csv

   Arguments:
     --keywords   One or more phrases to match in job titles. Default: ["React","React Native"].
     --limit      Max number of rows to output (default 50).
     --out        CSV output path (default ./react_jobs.csv).

4) Output:
   - Writes a CSV with columns: Company | Career portal link | Job title | Location
   - Prints a Markdown table (top N) to stdout.

Ethical scraping safeguards built-in:
   - Respects robots.txt for every host before fetching.
   - Uses official/public ATS endpoints where available (Greenhouse, Lever).
   - Polite rate limiting and bounded retries with backoff.
   - Clear User-Agent string.
"""
