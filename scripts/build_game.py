"""Build the human pilot game: game/index.html with every critique precomputed.

Usage:  python scripts/build_game.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stml.episodes import game_payload  # noqa: E402


def main() -> None:
    payload = game_payload()
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    template = (ROOT / "game" / "template.html").read_text()
    if "/*__DATA__*/" not in template:
        raise SystemExit("template is missing the /*__DATA__*/ placeholder")
    out = ROOT / "game" / "index.html"
    out.write_text(template.replace("/*__DATA__*/", data))
    (ROOT / "game" / "episodes.json").write_text(json.dumps(payload, indent=1))
    n = len(payload["episodes"])
    print(f"wrote {out} ({out.stat().st_size // 1024} KB, {n} worlds)")


if __name__ == "__main__":
    main()
