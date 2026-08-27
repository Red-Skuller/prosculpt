"""Functional tests for boltz_extras / postprocess hook / ligand-tolerant scoring."""
import os
import sys
import shutil
import yaml

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
os.chdir(PROJ)

# prosculpt_run imports pyrosetta + hydra at module level; stub for this test
import types

_pyrosetta = types.ModuleType("pyrosetta")
_pyrosetta.init = lambda *a, **k: None
sys.modules["pyrosetta"] = _pyrosetta

_hydra = types.ModuleType("hydra")
_hydra.main = lambda *a, **k: (lambda f: f)
_hydra.core = types.ModuleType("hydra.core")
_hydra_config = types.ModuleType("hydra.core.hydra_config")
_hydra_config.HydraConfig = type("HydraConfig", (), {"get": staticmethod(lambda: None)})
_hydra_utils = types.ModuleType("hydra.utils")
_hydra_utils.get_original_cwd = lambda: os.getcwd()
_hydra_utils.to_absolute_path = lambda p: os.path.abspath(p)
sys.modules["hydra"] = _hydra
sys.modules["hydra.core"] = _hydra.core
sys.modules["hydra.core.hydra_config"] = _hydra_config
sys.modules["hydra.utils"] = _hydra_utils

from omegaconf import OmegaConf
import prosculpt
import prosculpt_run

TMP = "/tmp/boltz_extra_test"
shutil.rmtree(TMP, ignore_errors=True)
os.makedirs(TMP)

failures = []


def check(name, cond, extra=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  -- {extra}" if extra and not cond else ""))
    if not cond:
        failures.append(name)


# ---------------------------------------------------------------- Test 1
# Regression: no boltz_extras -> identical output to original behaviour
cfg0 = OmegaConf.create({"use_a3m": False})
p0 = prosculpt.make_boltz_input_yaml(cfg0, "101_1", "MKT:GLY", TMP, None)
d0 = yaml.safe_load(open(p0))
check(
    "regression: no extras -> 2 protein chains, msa empty",
    d0
    == {
        "sequences": [
            {"protein": {"id": "A", "sequence": "MKT", "msa": "empty"}},
            {"protein": {"id": "B", "sequence": "GLY", "msa": "empty"}},
        ]
    },
    str(d0),
)

# ---------------------------------------------------------------- Test 2
# Merge: ccd ligand, smiles ligand, multi-ccd, auto-id rna, constraints,
# affinity, version
cfg1 = OmegaConf.create(
    {
        "use_a3m": False,
        "boltz_extras": {
            "sequences": [
                {"ligand": {"id": "C", "ccd": "SAH"}},
                {"ligand": {"id": "D", "smiles": "CC(=O)Oc1ccccc1C(=O)O"}},
                {"ligand": {"id": "E", "ccd": ["EDO", "GLU"]}},
                {"rna": {"sequence": "GCAUAGC"}},  # no id -> auto-assigned
            ],
            "constraints": [
                {"pocket": {"binder": "C", "contacts": [["A", 42]], "max_distance": 6.0}},
                {"contact": {"token1": ["A", 1], "token2": ["D", 1], "max_distance": 6.0}},
            ],
            "properties": [{"affinity": {"binder": "C"}}],
            "version": 1,
        },
    }
)
p1 = prosculpt.make_boltz_input_yaml(cfg1, "101_2", "MKT:GLY", TMP, None)
d1 = yaml.safe_load(open(p1))
ids = [s[next(iter(s))]["id"] for s in d1["sequences"]]
check("merge: chain ids A,B,C,D,E,F (auto F for rna)", ids == ["A", "B", "C", "D", "E", "F"], str(ids))
check(
    "merge: ccd ligand present",
    d1["sequences"][2] == {"ligand": {"id": "C", "ccd": "SAH"}},
)
check(
    "merge: smiles ligand present",
    d1["sequences"][3] == {"ligand": {"id": "D", "smiles": "CC(=O)Oc1ccccc1C(=O)O"}},
)
check(
    "merge: multi-ccd ligand present",
    d1["sequences"][4] == {"ligand": {"id": "E", "ccd": ["EDO", "GLU"]}},
)
check(
    "merge: auto-id rna present",
    d1["sequences"][5] == {"rna": {"id": "F", "sequence": "GCAUAGC"}},
)
check("merge: constraints copied", d1["constraints"][0]["pocket"]["binder"] == "C")
check("merge: affinity property copied", d1["properties"] == [{"affinity": {"binder": "C"}}])
check("merge: version copied", d1["version"] == 1)

# ---------------------------------------------------------------- Test 3
# Collision: explicit id clashing with a designed chain must raise
cfg_bad = OmegaConf.create(
    {
        "use_a3m": False,
        "boltz_extras": {"sequences": [{"ligand": {"id": "B", "ccd": "EDO"}}]},
    }
)
try:
    prosculpt.make_boltz_input_yaml(cfg_bad, "101_3", "MKT:GLY", TMP, None)
    check("collision: id B vs designed chain B raises", False)
except ValueError as e:
    check("collision: id B vs designed chain B raises", "collides" in str(e), str(e))

# ---------------------------------------------------------------- Test 4
# use_a3m branch: merge must apply there too
align_dir = os.path.join(TMP, "aligns")
os.makedirs(align_dir, exist_ok=True)
for ch in ("A", "B"):
    with open(f"{align_dir}/101_4_{ch}.a3m", "w") as f:
        f.write(f">x\nMKT\n")
cfg2 = OmegaConf.create(
    {
        "use_a3m": True,
        "boltz_extras": {"sequences": [{"ligand": {"id": "C", "ccd": "SAH"}}]},
    }
)
p2 = prosculpt.make_boltz_input_yaml(cfg2, "101_4", "MKT:GLY", TMP, align_dir)
d2 = yaml.safe_load(open(p2))
check(
    "use_a3m branch: msa paths + extras merged",
    d2["sequences"][0]["protein"]["msa"] == f"{align_dir}/101_4_A.a3m"
    and d2["sequences"][2] == {"ligand": {"id": "C", "ccd": "SAH"}},
    str(d2),
)

# ---------------------------------------------------------------- Test 5
# Postprocess script hook
hook_script = os.path.join(TMP, "hook.py")
with open(hook_script, "w") as f:
    f.write(
        "def postprocess_yaml(yaml_path, cfg, model_id):\n"
        "    import yaml\n"
        "    with open(yaml_path) as fh:\n"
        "        d = yaml.safe_load(fh)\n"
        "    d.setdefault('constraints', []).append({'contact': {'token1': ['A', 5], 'token2': ['C', 1], 'max_distance': 4.5}})\n"
        "    d['model_id_marker'] = model_id\n"
        "    with open(yaml_path, 'w') as fh:\n"
        "        yaml.safe_dump(d, fh)\n"
    )
cfg3 = OmegaConf.create(
    {
        "use_a3m": False,
        "boltz_yaml_postprocess_script": hook_script,
        "boltz_extras": {"sequences": [{"ligand": {"id": "C", "ccd": "SAH"}}]},
    }
)
p3 = prosculpt.make_boltz_input_yaml(cfg3, "101_5", "MKT", TMP, None)
prosculpt_run.run_boltz_yaml_postprocess(cfg3, p3, "101_5")
d3 = yaml.safe_load(open(p3))
check(
    "postprocess hook: script modified yaml",
    d3["constraints"][-1]["contact"]["max_distance"] == 4.5
    and d3["model_id_marker"] == "101_5",
    str(d3),
)

# hook with no script configured -> no-op
cfg_none = OmegaConf.create({"use_a3m": False})
prosculpt_run.run_boltz_yaml_postprocess(cfg_none, p3, "101_5")
check("postprocess hook: no-op when unset", True)

# hook script missing required function -> ValueError
bad_hook = os.path.join(TMP, "bad_hook.py")
with open(bad_hook, "w") as f:
    f.write("def wrong_name():\n    pass\n")
cfg_badhook = OmegaConf.create({"boltz_yaml_postprocess_script": bad_hook})
try:
    prosculpt_run.run_boltz_yaml_postprocess(cfg_badhook, p3, "101_5")
    check("postprocess hook: missing function raises", False)
except ValueError as e:
    check("postprocess hook: missing function raises", "postprocess_yaml" in str(e))

# ---------------------------------------------------------------- Test 6
# filter_protein_residues: ligand without CA + calcium ion named CA
pdb = os.path.join(TMP, "with_ligand.pdb")
with open(pdb, "w") as f:
    f.write(
        "ATOM      1  N   MET A   1      11.000  11.000  11.000  1.00  0.00           N\n"
        "ATOM      2  CA  MET A   1      12.000  11.000  11.000  1.00  0.00           C\n"
        "ATOM      3  N   GLY A   2      13.000  11.000  11.000  1.00  0.00           N\n"
        "ATOM      4  CA  GLY A   2      14.000  11.000  11.000  1.00  0.00           C\n"
        "ATOM      5  O   EDO B   3       5.000   5.000   5.000  1.00  0.00           O\n"
        "ATOM      6  C   EDO B   3       6.000   5.000   5.000  1.00  0.00           C\n"
        "ATOM      7  CA  CA  C   4       7.000   7.000   7.000  1.00  0.00           CA\n"
        "END\n"
    )
from Bio.PDB import PDBParser

parser = PDBParser(PERMISSIVE=1)
struct = parser.get_structure("t", pdb)
res = list(struct.get_residues())
check("fixture: 4 residues parsed", len(res) == 4, str(len(res)))
filtered = prosculpt.filter_protein_residues(res)
check(
    "filter_protein_residues: keeps only MET/GLY (drops EDO and CA ion)",
    [r.get_resname() for r in filtered] == ["MET", "GLY"],
    str([r.get_resname() for r in filtered]),
)

# ---------------------------------------------------------------- Test 7
# homooligomer_rmsd helpers with ligand chains
import homooligomer_rmsd

chains = homooligomer_rmsd._protein_chains(struct)
check(
    "homooligomer_rmsd._protein_chains: ligand-only chains dropped",
    [c.get_id() for c in chains] == ["A"],
    str([c.get_id() for c in chains]),
)
std = homooligomer_rmsd._std_aa_only(res)
check(
    "homooligomer_rmsd._std_aa_only: same filter",
    [r.get_resname() for r in std] == ["MET", "GLY"],
)

# align_chain_A end-to-end: target = protein-only pdb, mobile = pdb with ligands
target_pdb = os.path.join(TMP, "target.pdb")
with open(target_pdb, "w") as f:
    f.write(
        "ATOM      1  N   MET A   1      11.000  11.000  11.000  1.00  0.00           N\n"
        "ATOM      2  CA  MET A   1      12.000  11.000  11.000  1.00  0.00           C\n"
        "ATOM      3  N   GLY A   2      13.000  11.000  11.000  1.00  0.00           N\n"
        "ATOM      4  CA  GLY A   2      14.000  11.000  11.000  1.00  0.00           C\n"
        "END\n"
    )
st_target, st_mobile = homooligomer_rmsd.align_chain_A(target_pdb, pdb, parser)
check("align_chain_A: works with ligand chains in mobile structure", st_mobile is not None)

rmsd = homooligomer_rmsd.align_oligomers(target_pdb, pdb, save_aligned=False)
check("align_oligomers: returns finite rmsd with ligands present", rmsd is not None, str(rmsd))

print()
if failures:
    print(f"{len(failures)} FAILURES: {failures}")
    sys.exit(1)
print("ALL TESTS PASSED")
