"""Download ARCTraj from Hugging Face into data/raw/arctraj/.

    python scripts/download_arctraj.py

The dataset is a single CSV (ARCTraj.csv, ~236 MB) plus a README. Using
load_dataset() only returns the Croissant metadata card, not the trajectories,
so the files are fetched directly with huggingface_hub. If Hugging Face asks
for a login, run `huggingface-cli login` first.
"""

from pathlib import Path

from huggingface_hub import hf_hub_download

OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "arctraj"
OUT.mkdir(parents=True, exist_ok=True)
for name in ("ARCTraj.csv", "README.md"):
    print(hf_hub_download("SejinKimm/ARCTraj", name, repo_type="dataset", local_dir=str(OUT)))
