"""Validate a generated boltz.yaml against the Boltz v2 schema rules
(mirroring the checks in boltz/src/boltz/data/parse/schema.py:
parse_boltz_schema), using the shipped Examples/ligand_boltz.yaml config."""
import os
import sys
import yaml

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
os.chdir(PROJ)

from omegaconf import OmegaConf
import prosculpt

# Load the shipped example config (plain YAML; hydra 'defaults' ignored here)
with open(os.path.join(PROJ, "Examples", "ligand_boltz.yaml")) as f:
    example = yaml.safe_load(f)

cfg = OmegaConf.create(
    {
        "use_a3m": False,
        "boltz_extras": example["boltz_extras"],
    }
)

TMP = "/tmp/boltz_schema_test"
os.makedirs(TMP, exist_ok=True)
# 2 designed chains -> A, B ; example uses C, D, E + auto F
path = prosculpt.make_boltz_input_yaml(cfg, "201_1", "MKT:GLY", TMP, None)
data = yaml.safe_load(open(path))

errors = []

# --- top level -------------------------------------------------------------
if data.get("version", 1) != 1:
    errors.append(f"bad version: {data.get('version')}")

chain_to_entity = {}
for i, item in enumerate(data["sequences"]):
    if len(item) != 1:
        errors.append(f"sequences[{i}] must have exactly one key: {item}")
        continue
    etype = next(iter(item)).lower()
    if etype not in {"protein", "dna", "rna", "ligand"}:
        errors.append(f"sequences[{i}] invalid entity type {etype}")
        continue
    spec = item[etype]
    cid = spec["id"]
    ids = cid if isinstance(cid, list) else [cid]
    for c in ids:
        if c in chain_to_entity:
            errors.append(f"chain id {c} used twice")
        chain_to_entity[c] = etype
    if etype in {"protein", "dna", "rna"}:
        if not isinstance(spec.get("sequence"), str):
            errors.append(f"sequences[{i}] missing sequence")
    elif etype == "ligand":
        has_smiles = "smiles" in spec
        has_ccd = "ccd" in spec
        if has_smiles == has_ccd:
            errors.append(f"sequences[{i}] ligand needs exactly one of smiles/ccd")
        if has_ccd and not isinstance(spec["ccd"], (str, list)):
            errors.append(f"sequences[{i}] ccd must be str or list")

# --- constraints ------------------------------------------------------------
for i, c in enumerate(data.get("constraints", [])):
    keys = [k.lower() for k in c]
    if keys == ["bond"]:
        if "atom1" not in c["bond"] or "atom2" not in c["bond"]:
            errors.append(f"constraints[{i}] bond missing atom1/atom2")
    elif keys == ["pocket"]:
        for req in ("binder", "contacts"):
            if req not in c["pocket"]:
                errors.append(f"constraints[{i}] pocket missing {req}")
        else:
            if c["pocket"]["binder"] not in chain_to_entity:
                errors.append(
                    f"constraints[{i}] pocket binder {c['pocket']['binder']} unknown"
                )
            for ch, res in c["pocket"]["contacts"]:
                if ch not in chain_to_entity:
                    errors.append(f"constraints[{i}] contact chain {ch} unknown")
    elif keys == ["contact"]:
        for tok in ("token1", "token2"):
            if tok not in c["contact"]:
                errors.append(f"constraints[{i}] contact missing {tok}")
            else:
                if c["contact"][tok][0] not in chain_to_entity:
                    errors.append(
                        f"constraints[{i}] contact {tok} chain unknown"
                    )
    else:
        errors.append(f"constraints[{i}] unknown constraint type: {keys}")

# --- properties -------------------------------------------------------------
for i, p in enumerate(data.get("properties", [])):
    if list(p)[0].lower() != "affinity":
        errors.append(f"properties[{i}] unknown property")
        continue
    binder = p["affinity"]["binder"]
    if binder not in chain_to_entity:
        errors.append(f"properties[{i}] binder {binder} unknown")
    elif chain_to_entity[binder] != "ligand":
        errors.append(f"properties[{i}] binder {binder} is not a ligand")

print("Generated YAML:")
print(open(path).read())
if errors:
    print("SCHEMA ERRORS:")
    for e in errors:
        print(" -", e)
    sys.exit(1)
print("SCHEMA VALIDATION PASSED")
