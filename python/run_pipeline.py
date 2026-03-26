#!/usr/bin/env python3
"""
Full pipeline runner:
1. generate_embeddings
2. process_metadata
3. combine_features
4. generate_all_similarities (multiple modes)
"""

import sys
import subprocess
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY_DIR = ROOT / "python"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(PY_DIR / 'pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def run(step_args):
    """helper to run a python script with args in the python/ dir"""
    cmd = [sys.executable] + step_args
    logger.info(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=PY_DIR, check=True)

def main():
    logger.info("=== Pipeline start ===")

    # Step 1: embeddings
    run(["generate_embeddings.py"])

    # Step 2: metadata processing
    run(["process_metadata.py"])

    # Step 3: combine CLIP + metadata
    run(["combine_features.py"])

    # Step 4: similarity runs
    similarity_modes = [
        ["generate_all_similarities.py","--mode","combined","--top-k","5"],
        ["generate_all_similarities.py","--mode","visual_only","--top-k","5"],
        ["generate_all_similarities.py","--mode","combined","--exclude-same-artist","--top-k","5","--output-suffix","cross_artist"],
        ["generate_all_similarities.py","--mode","visual_only","--exclude-same-artist","--top-k","5","--output-suffix","visual_cross_artist"],
        ["generate_all_similarities.py","--mode","combined","--exclude-same-artist","--min-threshold","0.7","--top-k","3","--output-suffix","high_quality"]
    ]

    for args in similarity_modes:
        try:
            run(args)
        except subprocess.CalledProcessError as e:
            logger.warning(f"Similarity generation failed for {' '.join(args)}: {e}")

    logger.info("=== Pipeline complete ===")
    logger.info("Artifacts:")
    logger.info(" - embeddings/: *.npy, similarity stats, summary csvs")
    logger.info(" - images/: downloaded artwork images")
    logger.info(" - metadata/: per-object json + artwork_metadata.csv")
    logger.info(" - data/: link/similarity/reduced csv for the web app")

if __name__ == "__main__":
    main()
