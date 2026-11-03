"""Download the open-access papers listed in literature/papers.json and rebuild literature/INDEX.md.

    python scripts/fetch_literature.py            # download what is missing, rewrite the index
    python scripts/fetch_literature.py --index    # only rewrite the index

Only open-access / author-hosted / preprint copies are listed with a pdf link; every other paper is indexed
with its citation and identifier only. Each PDF goes into its own folder, literature/pdfs/<key>/<key>.pdf,
and is kept only if it really starts with %PDF. Downloaded files are never opened or executed here.
"""

import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIT = ROOT / "literature"
UA = {"User-Agent": "Mozilla/5.0 (research literature fetch; STML project)"}


def fetch(url: str) -> bytes | None:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
            data = r.read()
        return data if data[:4] == b"%PDF" else None
    except Exception as e:
        print("   failed:", repr(e)[:90])
        return None


def main() -> None:
    papers = json.loads((LIT / "papers.json").read_text())
    status_path = LIT / "fetch_status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    if "--index" not in sys.argv:
        for p in papers:
            if not p["pdf"]:
                continue
            f = LIT / "pdfs" / p["key"] / f"{p['key']}.pdf"
            if f.exists():
                status[p["key"]] = "downloaded"
                continue
            print("fetching", p["key"])
            data = fetch(p["pdf"])
            if data:
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_bytes(data)
                status[p["key"]] = "downloaded"
            else:
                status[p["key"]] = "download failed (not a PDF or blocked)"
            time.sleep(1.0)
        status_path.write_text(json.dumps(status, indent=1))
    topics: dict[str, list] = {}
    for p in papers:
        topics.setdefault(p["topic"], []).append(p)
    lines = ["# Literature for STML", "",
             "Neuroscience and cognitive science of learning, learning from errors, metacognition, and the AI work closest to STML. "
             "Source of truth: `papers.json`; rebuild with `python scripts/fetch_literature.py`. "
             "A PDF is stored only for open-access / author-hosted / preprint copies; the rest are cited by identifier. "
             "Details marked 'not verified' came from memory or a single search snippet.", ""]
    for t, ps in topics.items():
        lines += [f"## {t}", ""]
        for p in ps:
            have = status.get(p["key"], "").startswith("downloaded")
            loc = f"[pdf](pdfs/{p['key']}/{p['key']}.pdf)" if have else (f"[page]({p['page']})" if p.get("page") else "citation only")
            lines.append(f"- **{p['authors']} ({p['year']})**. {p['title']}. *{p['venue']}*. {p['id']} - {loc}  ")
            lines.append(f"  {p['why']}")
        lines.append("")
    (LIT / "INDEX.md").write_text("\n".join(lines))
    print({k: sum(1 for v in status.values() if v == k) for k in set(status.values())})


main()
