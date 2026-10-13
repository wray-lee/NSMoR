"""Write an exploratory publication supplement from frozen PHASE-2 JSON."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Establish CPU1 before package imports (nsmor imports torch at package level).
os.environ.update(CUDA_VISIBLE_DEVICES="", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nsmor.analysis.publication_robustness import write_supplement  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Frozen realdata_phase2_family.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; existing paths are rejected")
    args = parser.parse_args(argv)
    output = write_supplement(args.input, args.output_dir,
                              sys.argv if argv is None else [str(Path(__file__)), *argv])
    print(f"Wrote exploratory supplement to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
