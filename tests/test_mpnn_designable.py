"""Tests for mpnn_designable_residues: letting ProteinMPNN redesign a
SUPERSET of what RFDiffusion generated (RFDiff's design set becomes a
subset of MPNN's design set).

Synthetic setup:
  - input PDB: chain A, 22 residues (A1..A22)
  - contig "[A1-10/5-5/A11-20]": RFDiff generates 5 new residues between
    A10 and A11; A1-10 and A11-20 keep their backbone (A21-A22 not in contig)
  - output PDB: chain A, 25 residues (fixed 10 + new 5 + fixed 10)
  - default MPNN fixed set: per-chain indices 1-10 and 16-25 (20 residues)
"""
import os
import sys
import json
import shutil
import pickle

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
os.chdir(PROJ)

from omegaconf import OmegaConf
import prosculpt

TMP = "/tmp/mpnn_designable_test"
shutil.rmtree(TMP, ignore_errors=True)
os.makedirs(TMP)

failures = []


def check(name, cond, extra=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  -- {extra}" if extra and not cond else ""))
    if not cond:
        failures.append(name)


def write_protein_pdb(path, chains):
    """chains: {chain_id: n_residues}; writes a minimal all-MET PDB."""
    serial = 0
    lines = []
    for chain_id, n in chains.items():
        for resseq in range(1, n + 1):
            serial += 1
            lines.append(
                f"ATOM  {serial:5d}  CA  MET {chain_id}{resseq:4d}    "
                f"   1.000   2.000   3.000  1.00  0.00           C"
            )
    lines.append("END")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# --- fixtures ---------------------------------------------------------------
input_pdb = os.path.join(TMP, "input.pdb")
write_protein_pdb(input_pdb, {"A": 22})  # A1..A22

workdir = os.path.join(TMP, "work", "cycle_dir")  # input_mpnn (cycle 0)
os.makedirs(workdir, exist_ok=True)
out_pdb = os.path.join(workdir, "_0.pdb")
write_protein_pdb(out_pdb, {"A": 25})  # RFDiff output: 25 residues

CONTIG = "[A1-10/5-5/A11-20]"
inpaint_seq = [True] * 10 + [False] * 5 + [True] * 10
trb = {
    "config": {"contigmap": {"contigs": [CONTIG], "provide_seq": None}},
    "inpaint_seq": inpaint_seq,
    "inpaint_str": inpaint_seq,
    # PDB-mapped contig positions in contig order
    "con_ref_idx0": list(range(20)),          # ref 0-9 (A1-10), 10-19 (A11-20)
    "con_hal_idx0": list(range(0, 10)) + list(range(15, 25)),
}
with open(os.path.join(workdir, "_0.trb"), "wb") as f:
    pickle.dump(trb, f)

base_cfg = {
    "skipRfDiff": False,
    "pdb_path": input_pdb,
    "contig": CONTIG,
    "inference": {"symmetry": None},
    "partial_diffusion": False,
    "chains_to_design": "A",
    "designable_residues": None,
}


def run(mpnn_designable=None):
    cfg = OmegaConf.create(dict(base_cfg))
    if mpnn_designable is not None:
        cfg.mpnn_designable_residues = mpnn_designable
    out_dir = os.path.join(TMP, "mpnn_out")
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)
    prosculpt.process_pdb_files(workdir, out_dir, cfg, None, cycle=0)
    with open(os.path.join(out_dir, "fixed_pdbs.jsonl")) as f:
        fixpos = json.load(f)
    return fixpos["_0"]


# --- Test 1: regression - without the option, MPNN fixes all 20 PDB residues
res0 = run(None)
check(
    "regression: no option -> 20 fixed positions (1-10, 16-25)",
    sorted(res0["A"]) == list(range(1, 11)) + list(range(16, 26)),
    str(res0),
)
check("regression: design set == RFDiff's 5 generated positions",
      sorted(set(range(1, 26)) - set(res0["A"])) == [11, 12, 13, 14, 15])

# --- Test 2: single residues, range, and a residue in a fixed chain region
res2 = run(["A5", "A11-A13", "A20"])
expected = sorted((set(range(1, 11)) | set(range(16, 26))) - {5, 16, 17, 18, 25})
check(
    "A5, A11-A13, A20 unfix per-chain 5, 16-18, 25",
    sorted(res2["A"]) == expected,
    str(res2["A"]),
)
check("design set is now the superset (10 positions)",
      sorted(set(range(1, 26)) - set(res2["A"]))
      == [5, 11, 12, 13, 14, 15, 16, 17, 18, 25])

# --- Test 3: whole-chain spec "A" unfixes every fixed residue
res3 = run(["A"])
check("whole chain 'A' -> nothing fixed", res3["A"] == [], str(res3["A"]))

# --- Test 4: residue outside the contig (A21) -> NOTE, no change
res4 = run(["A21"])
check(
    "A21 (not in contig) -> unchanged fixed set",
    sorted(res4["A"]) == sorted(res0["A"]),
    str(res4["A"]),
)

# --- Test 5: unknown residue -> WARNING, no change
res5 = run(["Z99", "A12"])
check(
    "unknown Z99 ignored, A12 unfixes per-chain 17",
    sorted(res5["A"]) == sorted((set(range(1, 11)) | set(range(16, 26))) - {17}),
    str(res5["A"]),
)

print()
if failures:
    print(f"{len(failures)} FAILURES: {failures}")
    sys.exit(1)
print("ALL TESTS PASSED")
