"""
Compare two training runs file by file.

    python src/compare_runs.py <project_root_A> <project_root_B>

Each root is a project folder containing results/ (and optionally models/).
Reports, for every results file present in both: byte-identical or not, and if not,
the maximum absolute numeric difference. Also compares models (if both have them).
Nothing is changed or written; it only prints.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def files(folder: Path) -> set[str]:
    return {p.name for p in folder.iterdir() if p.is_file()} if folder.exists() else set()


def csv_difference(a: Path, b: Path) -> str:
    x, y = pd.read_csv(a), pd.read_csv(b)
    if x.shape != y.shape or list(x.columns) != list(y.columns):
        return f"DIFFERENT SHAPE/COLUMNS {x.shape} vs {y.shape}"
    num = x.select_dtypes("number").columns
    d = (x[num] - y[num]).abs().to_numpy()
    text_same = x.drop(columns=num).equals(y.drop(columns=num))
    return f"same shape {x.shape}; max abs numeric difference = {np.nanmax(d) if d.size else 0.0:.3e}; text columns identical = {text_same}"


def main(a_root: str, b_root: str) -> int:
    A, B = Path(a_root), Path(b_root)
    print(f"A = {A}\nB = {B}\n")
    ra, rb = files(A / "results"), files(B / "results")
    common = sorted(ra & rb)
    print(f"results files in A: {len(ra)} | in B: {len(rb)} | compared (present in both): {len(common)}")
    n_identical = 0
    for name in common:
        pa, pb = A / "results" / name, B / "results" / name
        same = sha(pa) == sha(pb)
        n_identical += same
        detail = "" if same else (" -> " + csv_difference(pa, pb) if name.endswith(".csv") else " -> content differs")
        print(f"  {name:46s} byte-identical = {same}{detail}")
    print(f"\n{n_identical} of {len(common)} common results files are byte-identical")
    if ra - rb:
        print("only in A:", sorted(ra - rb))
    if rb - ra:
        print("only in B:", sorted(rb - ra))

    ma, mb = files(A / "models"), files(B / "models")
    if ma and mb:
        print("\nModels present in both folders:")
        for name in sorted(ma & mb):
            print(f"  {name:36s} byte-identical = {sha(A / 'models' / name) == sha(B / 'models' / name)}")
    else:
        print("\nModels: not compared (one folder has no models/ directory)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2]))
