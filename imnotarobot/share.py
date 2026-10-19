"""Results people choose to share, and the table of recommendations built from them.

Nothing is sent by the app: it opens a pre-filled GitHub issue in the browser, the person reads it and
presses Submit. The report holds model names and scores only — no keys, addresses, pictures or answers.

    python -m imnotarobot.share            # rebuild docs/community-results.md from shared issues
"""

import json
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from . import __version__

REPO = "zigansh1n/imnotarobot"
# GitHub refuses longer new-issue links
MAX_URL = 7500
MARK = "imnotarobot-results"


def report(bench: dict[str, Any], providers: dict[str, str]) -> dict[str, Any]:
    """`bench` is the app's comparison state; `providers` maps a key name to its maker (custom = reseller)."""
    rows = []
    for r in bench.get("rows") or []:
        key = r["name"].split("/", 1)[0]
        rows.append({"model": r["model"], "provider": providers.get(key, "custom"), "accuracy": round(r["accuracy"], 3),
                     "seconds": round(r["seconds"], 1), "errors": r.get("errors", 0),
                     "sets": {k: [v["ok"], v["n"]] for k, v in r.get("by_set", {}).items()}})
    return {"app": __version__, "date": time.strftime("%Y-%m-%d"), "kind": "models", "rows": rows}


def issue_url(rep: dict[str, Any]) -> str:
    best = max(rep["rows"], key=lambda r: r["accuracy"])["model"] if rep["rows"] else "—"
    table = "\n".join(f"| {r['model']} | {r['provider']} | {round(r['accuracy'] * 100)}% | {r['seconds']} s |" for r in rep["rows"])
    body = (f"Shared from I'm Not a Robot {rep['app']}. Model names and scores only, no keys or pictures.\n\n"
            f"| Model | Provider | Accuracy | Time per captcha |\n|---|---|---|---|\n{table}\n\n"
            f"<!-- {MARK}\n{json.dumps(rep, ensure_ascii=False)}\n-->\n")
    url = f"https://github.com/{REPO}/issues/new?" + urlencode(
        {"title": f"Results: {best} and {len(rep['rows']) - 1} more" if len(rep["rows"]) > 1 else f"Results: {best}",
         "labels": "results", "body": body})
    if len(url) > MAX_URL and len(rep["rows"]) > 1:
        # keep the most accurate half until the link fits
        rows = sorted(rep["rows"], key=lambda r: -r["accuracy"])
        return issue_url({**rep, "rows": rows[:max(1, len(rows) // 2)]})
    return url


def parse(body: str) -> dict[str, Any] | None:
    m = re.search(rf"<!-- {MARK}\n(.*?)\n-->", body or "", re.S)
    try:
        return json.loads(m.group(1)) if m else None
    except ValueError:
        return None


def aggregate(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per model: how many people shared it, median accuracy and time; most accurate first."""
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rep in reports:
        for r in rep.get("rows", []):
            by_model[r["model"]].append(r)
    out = [{"model": m, "reports": len(rs), "accuracy": statistics.median(r["accuracy"] for r in rs),
            "seconds": statistics.median(r["seconds"] for r in rs),
            "providers": sorted({r["provider"] for r in rs})} for m, rs in by_model.items()]
    return sorted(out, key=lambda x: (-x["accuracy"], x["seconds"]))


def markdown(rows: list[dict[str, Any]]) -> str:
    lines = ["# Community results", "",
             "Which vision models solve captchas best, from results people shared with the app's "
             "**Share results** button. Median over all reports; more reports = more trustworthy.", "",
             "| Model | Accuracy | Time per captcha | Reports | Providers |", "|---|---|---|---|---|"]
    lines += [f"| {r['model']} | {round(r['accuracy'] * 100)}% | {r['seconds']} s | {r['reports']} | {', '.join(r['providers'])} |"
              for r in rows]
    return "\n".join(lines) + "\n"


def fetch_reports() -> list[dict[str, Any]]:
    import httpx

    reports, page = [], 1
    while True:
        r = httpx.get(f"https://api.github.com/repos/{REPO}/issues",
                      params={"labels": "results", "state": "all", "per_page": 100, "page": page}, timeout=30)
        r.raise_for_status()
        items = r.json()
        reports += [rep for rep in (parse(i.get("body", "")) for i in items) if rep]
        if len(items) < 100:
            return reports
        page += 1


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs/community-results.md")
    out.write_text(markdown(aggregate(fetch_reports())))
    print(f"written {out}")


if __name__ == "__main__":
    main()
