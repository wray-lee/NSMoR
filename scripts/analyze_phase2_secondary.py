"""Run the phase-2 secondary descriptive battery for one saved checkpoint."""
import argparse
import os
from pathlib import Path
import sys

os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                  NUMEXPR_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nsmor.analysis.secondary_battery import run_battery


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--arm", choices=("R0", "R1", "R2", "R3"), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--nested_prior", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = run_battery(args.checkpoint, args.arm, args.dataset, args.nested_prior, args.output_dir)
    print(f"{args.arm} seed {summary['seed']}: {summary['population']['n_analyzed']} validation trials -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
