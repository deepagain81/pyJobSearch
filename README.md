# React Jobs Finder

Find **current “React” / “React Native” job openings** from major ATS providers, save them to CSV, and print a clean Markdown table.

* **Ethical & polite:** identifies with a bot UA, respects `robots.txt`, rate-limits, retries.
* **ATS-first:** prefers official/public ATS endpoints (Greenhouse, Lever, Ashby, SmartRecruiters), with a generic JSON-LD fallback (e.g., Workday pages).
* **Search discovery:** uses your **Google Programmable Search Engine (CSE)** first (free tier), with optional **SerpAPI**.
* **Configurable constants:** everything is configured via environment variables (see `.env.example`).
* **Clear logs & debugging:** progress + step timings via `debug_constants.py`; `--debug=true` prints verbose details and tracebacks.

---

## Table of Contents

* [Requirements](#requirements)
* [Quick Start](#quick-start)
* [Configuration](#configuration)
* [Providers & Strategy](#providers--strategy)
* [Discovery Priority](#discovery-priority)
* [CLI](#cli)
* [Usage Examples](#usage-examples)
* [Output](#output)
* [US-Only Filtering (Optional)](#usonly-filtering-optional)
* [Ethics & Compliance](#ethics--compliance)
* [Troubleshooting](#troubleshooting)

---

## Requirements

* **Python 3.9+** (for `zoneinfo`).
  If you must use Python 3.8, install `backports.zoneinfo`.
* Packages:

  ```bash
  pip install requests beautifulsoup4 pandas
  ```

---

## Quick Start

```bash
# 1) Copy and edit your env
cp .env.example .env
# (fill in your Google CSE key/cx, domains, providers, UA, etc.)

# 2) Run
python3 react_jobs_finder.py

# 3) See results
# - CSV saved to path from DEFAULT_OUT_PATH (supports {date})
```

> Keep `.env` out of source control; `.env.example` is safe to commit.

---

## Configuration

All configuration lives in environment variables. This repo includes **`.env.example`** with every required key and documentation. Create your own `.env` next to the script.

Highlights you’ll likely edit:

* **User-Agent** (be transparent & contactable)

  * `JOBS_USER_AGENT=ReactJobsFinderBot/1.x (+https://yourdomain; contact: mailto:you@domain)`
* **Discovery backends** (order matters)

  * `SEARCH_BACKENDS=google_cse,serpapi`
* **Google CSE** (required if enabled)

  * `GOOGLE_CSE_KEY`, `GOOGLE_CSE_CX`, `CSE_PAGE_SIZE`, `CSE_MAX_START`
* **Domains**

  * `ATS_ALLOWED_DOMAINS=boards.greenhouse.io,jobs.lever.co,myworkdayjobs.com,ashbyhq.com,smartrecruiters.com,...`
  * `ATS_SEARCH_DOMAINS=boards.greenhouse.io,jobs.lever.co,myworkdayjobs.com,ashbyhq.com,smartrecruiters.com`
* **Providers**

  * `ATS_PROVIDERS=greenhouse,lever,ashby,smartrecruiters,generic`
  * Provide templates/tokens for enabled providers (see `.env.example`).
* **Output with date placeholder**

  * `DEFAULT_OUT_PATH=react_jobs_{date}.csv`
  * `OUT_DATE_FMT=%Y-%m-%d`
  * `OUT_TZ=America/Chicago`

> In your CSE control panel, set **Sites to search** (one per line):
> `boards.greenhouse.io/*`
> `jobs.lever.co/*`
> `*.myworkdayjobs.com/*`
> `jobs.ashbyhq.com/*`
> `*.smartrecruiters.com/*`

---

## Providers & Strategy

* **Greenhouse** → Job Board API (`GREENHOUSE_API_TEMPLATE`)
* **Lever** → Postings API (`LEVER_API_TEMPLATE`)
* **Ashby** → Public Job Board API (`ASHBY_API_TEMPLATE`)
* **SmartRecruiters** → Customer Posting API (`SMARTRECRUITERS_API_TEMPLATE` + `SMARTRECRUITERS_TOKEN`)
* **Generic** → Single-page **JSON-LD JobPosting** parse (for Workday/others) — no site-wide crawling.

The script prefers **official/public APIs** first and falls back to JSON-LD only for pages without public endpoints.

---

## Discovery Priority

1. **Google CSE** (Custom Search JSON API; you supply `key` + `cx`)
2. **SerpAPI** (optional; used if listed in `SEARCH_BACKENDS`)

Both are **restrictable by domain** using `ATS_SEARCH_DOMAINS`, so discovery stays inside ATS hosts.

---

## CLI

```
usage: react_jobs_finder.py [-h] [--keywords KEYWORDS [KEYWORDS ...]] [--limit LIMIT]
                            [--out OUT] [--quiet | --progress] [--debug DEBUG]
```

* `--keywords`  One or more title keywords (default from `.env`: `DEFAULT_KEYWORDS`, e.g., `React,React Native`)
* `--limit`     Max rows to output (default from `.env`: `DEFAULT_LIMIT`)
* `--out`       CSV path (supports `{date}` placeholder; default from `.env`: `DEFAULT_OUT_PATH`)
* `--quiet`     Suppress milestone logs (stderr)
* `--progress`  Force-enable milestone logs (stderr)
* `--debug`     `true|false` — verbose request/traceback mode (stderr)

---

## Usage Examples

```bash
# Default run (uses .env settings)
python3 react_jobs_finder.py

# Custom keywords & limit
python3 react_jobs_finder.py --keywords React "React Native" --limit 80

# Dated output filename (timezone/format from OUT_TZ/OUT_DATE_FMT)
python3 react_jobs_finder.py --out "exports/react_{date}.csv"

# Debug mode (verbose logs, request statuses, tracebacks on error)
python3 react_jobs_finder.py --debug=true

# Quiet mode (stdout only shows the Markdown table)
python3 react_jobs_finder.py --quiet
```

---

## Output

* **CSV** with columns: `Company | Career portal link | Job title | Location`
  Path is taken from `--out` (supports `{date}`) or `.env` `DEFAULT_OUT_PATH`.
* **Progress & debug logs** print to **stderr** (clean piping to files/pagers).

---


## Ethics & Compliance

* **User-Agent** clearly identifies the tool + contact info.
* **robots.txt** is checked for HTML fetches; API calls use public endpoints.
* **Politeness:** configurable rate limiting & retries with exponential backoff.
* **Scope control:** domain allowlist + ATS provider APIs to avoid broad crawling.
* **Cost-aware:** Google CSE first (free 100/day), SerpAPI optional.

---

## Troubleshooting

* **No dates in filename / NameError around `OUT_TZ`**
  Ensure `.env` has:

  ```
  DEFAULT_OUT_PATH=react_jobs_{date}.csv
  OUT_DATE_FMT=%Y-%m-%d
  OUT_TZ=America/Chicago
  ```

  And you’re on Python 3.9+ (or installed `backports.zoneinfo` for 3.8).

* **Google CSE only returns \~100 max results**
  That’s a product limit (pages of 10 up to start≈100). Increase breadth by adding more keywords, or rely on multiple providers.

* **Hitting CSE daily quota (100/day)**
  That’s enforced by Google. Add billing to raise limits or run fewer queries.

* **Quiet vs Debug logs**
  `--quiet` suppresses milestone logs; `--debug=true` turns on verbose request logging and tracebacks. Logging helpers live in `debug_constants.py`.

---
## US-Only Filtering (Not Covered yet)

Because ATS hosts serve global jobs under the same domains, CSE filtering is imperfect. Best results:

1. Bias/Restrict CSE:

* `CSE_CR=countryUS` (country restrict), `CSE_GL=us` (geo bias), `CSE_LR=lang_en` (language)

2. Enforce **post-discovery filtering** using provider location fields (already parsed):

* Greenhouse: `job.location.name`
* Lever: `categories.location`
* Ashby: `job.location`
* SmartRecruiters: `location.country` (and `city/region`)

You can use an allowlist in `.env` (see `.env.example`: `LOCATION_COUNTRY_ALLOWLIST=US,United States,USA,U.S.`) to keep U.S. roles only.

* **Getting non-US jobs**
  Use `CSE_CR/CSE_GL/CSE_LR` and the post-filter allowlist in `.env`. The provider location fields are the reliable filter.

---

**Happy hunting!** If you open an issue/PR, please redact any keys and logs that may include private tokens or URLs.
