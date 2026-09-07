import pickle
from Bio.PDB import PDBParser, PDBIO, Superimposer, PPBuilder, MMCIFParser
from Bio.SVDSuperimposer import SVDSuperimposer
from Bio import SeqIO, Align
from Bio.Seq import Seq
import json
import glob
import os
import numpy as np
import pandas as pd
import shutil
from pathlib import Path
import string
import homooligomer_rmsd
from Bio.Align import PairwiseAligner
from Bio.Data import IUPACData
import re
import io
import yaml
import copy
from omegaconf import OmegaConf

# 3-letter residue names (uppercase, as in PDB files) of the 20 standard
# amino acids. Used to filter protein residues out of predicted structures
# that may also contain non-polymer chains (ligands, metal ions, water)
# added via boltz_extras.
STD_AA_RESNAMES = {k.upper() for k in IUPACData.protein_letters_3to1}


def filter_protein_residues(residues):
    """Keep only standard amino-acid residues.

    A predicted PDB that included ligands/metals in the input (see
    `boltz_extras`) contains non-protein residues without CA atoms (a calcium
    ion ligand is even a single atom literally named 'CA'), so we filter by
    standard residue name instead. Protein residues keep their original
    order, so index-based mapping (e.g. from the RFDiffusion .trb file)
    stays aligned.
    """
    return [r for r in residues if r.get_resname() in STD_AA_RESNAMES]


def _boltz_next_free_id(used_ids):
    """Return the next free single-letter chain ID (A..Z) not in used_ids."""
    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        if ch not in used_ids:
            return ch
    raise ValueError(
        "boltz_extras: no free single-letter chain IDs left (A-Z all in use)"
    )


def _remap_boltz_ref(ref, remap):
    """Remap a single chain/residue reference from input PDB coordinates to the
    generated (designed) sequence coordinates.

    ref = [chain, resnum] (pocket contact / contact token) or
          [chain, resnum, atom_name] (bond constraint).
    remap(chain, resnum) -> (out_chain, out_resnum) or None.

    If remap is None, the residue is not a PDB-mapped input residue (e.g. it is
    a static ligand/metal chain), or the residue is not numeric, the reference
    is returned unchanged.
    """
    if remap is None or not isinstance(ref, (list, tuple)) or len(ref) < 2:
        return ref
    chain, resnum = ref[0], ref[1]
    try:
        resnum_int = int(resnum)
    except (TypeError, ValueError):
        return ref
    mapped = remap(chain, resnum_int)
    if mapped is None:
        return ref
    out_chain, out_resnum = mapped
    rest = list(ref[2:])  # e.g. atom name for bond constraints
    return [out_chain, out_resnum] + rest


def _remap_boltz_ref_or_drop(ref, remap, valid_ids):
    """Remap a constraint reference; return None (drop it) when it cannot be
    remapped AND its chain is not present in the yaml.

    This is _remap_boltz_ref plus dangling-reference handling: a reference to
    a chain that is in the yaml but not in the remap (ligands, metals, static
    extra proteins) is kept unchanged; a reference to a chain that is in
    neither (e.g. a C2-image chain that is absent from a monomer-only yaml)
    would make Boltz fail validation, so it is dropped with a warning.
    """
    if not isinstance(ref, (list, tuple)) or len(ref) < 2:
        return ref
    chain = ref[0]
    try:
        resnum_int = int(ref[1])
    except (TypeError, ValueError):
        return ref
    mapped = remap(chain, resnum_int)
    if mapped is not None:
        out_chain, out_resnum = mapped
        return [out_chain, out_resnum] + list(ref[2:])
    if chain in valid_ids:
        return list(ref)
    print(
        f"WARNING: boltz_extras constraint reference {ref} cannot be remapped "
        f"and chain '{chain}' is not in the yaml; dropping it."
    )
    return None


def build_boltz_ref_to_pos(cfg, trb_file):
    """Build a mapping {(input_chain, input_resseq): flat_position} for one
    RFDiffusion design, used to remap boltz_extras constraint references from
    input PDB coordinates to the generated (designed) sequence coordinates.

    flat_position is the 0-based position in the generated sequence (the
    concatenation of the designed chains), matching the trb 'hal' indices.
    Because it is derived from the actual trb mapping, it correctly handles
    arbitrary contigs - including length changes (e.g. [A1-25/1-5/A30-50])
    and chain re-mapping - not just equal-length replacements.

    The mapping is content-based: the trb stores the PDB-mapped contig
    positions in contig order as two parallel lists,
        complex_con_ref_pdb_idx[j] = (chain, resseq) in the input PDB
        complex_con_hal_idx0[j]    = position in the generated sequence
    so it does not assume RFDiff's and Biopython's residue lists line up
    positionally (they don't for multi-model PDBs, where RFDiff reads every
    model and Biopython only the first).

    Returns None when the mapping cannot be built (missing trb, or missing
    trb fields) - callers should then leave references as-is.
    """
    if not trb_file or not os.path.exists(trb_file):
        return None
    try:
        with open(trb_file, "rb") as f:
            trb_data = pickle.load(f)
        trb_ref_pdb = trb_data.get(
            "complex_con_ref_pdb_idx", trb_data.get("con_ref_pdb_idx", None)
        )
        hal_idx_list = list(
            trb_data.get("complex_con_hal_idx0", trb_data.get("con_hal_idx0", []))
        )
    except Exception as e:
        print(f"WARNING: could not build boltz_extras residue remap: {e}")
        return None
    if trb_ref_pdb is None:
        # older trb without a content-based (chain, resseq) list: fall back to
        # positional indexing into the input PDB residue list (works when
        # RFDiff's and Biopython's parsers agree on the residues)
        ref_path = cfg.get("pdb_path", None)
        if ref_path is None or not os.path.exists(ref_path):
            return None
        try:
            all_res, _ = get_all_residues(ref_path)
            ref_idx_list = list(
                trb_data.get("complex_con_ref_idx0", trb_data.get("con_ref_idx0", []))
            )
        except Exception as e:
            print(f"WARNING: could not build boltz_extras residue remap: {e}")
            return None
        if not ref_idx_list or len(ref_idx_list) != len(hal_idx_list):
            return None
        mapping = {
            all_res[ref_i]: int(p)
            for ref_i, p in zip(ref_idx_list, hal_idx_list)
            if ref_i < len(all_res)
        }
    else:
        # Content-based mapping (preferred): the trb's (chain, resseq) list
        # pairs directly with the generated-position list, no PDB parsing
        # involved.
        mapping = {}
        for entry, p in zip(trb_ref_pdb, hal_idx_list):
            mapping[(str(entry[0]), int(entry[1]))] = int(p)

    # C2 symmetry: the generated output is the monomer plus its strict 180-
    # degree image, so the flat sequence is [monomer tokens, image tokens].
    # Constraint references that use an IMAGE input chain (e.g. the B peptide
    # when the monomer contig only contains the A peptide) are not in the trb
    # mapping; map them onto the image chain at the SAME per-chain position
    # (the image is an exact copy, so position p in the image chain is the
    # C2 counterpart of position p in the monomer chain).
    # c2_chain_pairs: [[monomer_chain, image_chain], ...] in input PDB coords.
    try:
        inference = cfg.get("inference", None)
        symmetry = inference.get("symmetry", None) if inference is not None else None
    except AttributeError:
        symmetry = None
    if symmetry == "c2":
        # monomer flat length = full contig length (incl. new linker tokens)
        try:
            L_mono = len(trb_data["inpaint_seq"])
        except (KeyError, TypeError):
            L_mono = len(hal_idx_list)
        pairs = cfg.get("c2_chain_pairs", None)
        if pairs is None:
            print(
                "WARNING: inference.symmetry is c2 but c2_chain_pairs is not "
                "set; boltz_extras constraint references to C2-image input "
                "chains will not be remapped."
            )
        else:
            pairs = OmegaConf.to_container(pairs, resolve=True)
            for mon, img in pairs:
                for (ch, rs), p in list(mapping.items()):
                    if ch == str(mon):
                        mapping[(str(img), rs)] = L_mono + p
    return mapping


def build_boltz_full_complex(cfg, trb_data, mpnn_sequence, rfdiff_pdb):
    """Build the Boltz input chains for `boltz_full_complex` mode: the full
    input PDB chains with the designed (RFDiff/MPNN) segments spliced in at
    the contig positions, so Boltz predicts the whole complex (e.g. a full
    tetramer) while RFDiff/MPNN only ran on a trimmed contig.

    For each contig chain k, with host input chain X (the input chain with
    the most contig-mapped residues in k):
      - if k contains NEW positions (a fused design, e.g. linker + peptide
        inserted into a subunit):
            boltz chain = [native prefix of X before X's first anchor in k,
                           only if the design sequence starts with an X
                           residue]
                        + [the design sequence of k, exactly as MPNN
                           produced it, including new insertions and
                           cross-chain segments]
                        + [native suffix of X after X's last anchor in k,
                           only if the design sequence ends with an X
                           residue]
        Native residues that the contig skipped BETWEEN two anchors (e.g. a
        loop that the short designed linkers jump over) stay excluded: the
        designed linkers cannot physically span them.
      - if k has no new positions (pure fixed context, e.g. a whole subunit
        or interface windows of one): boltz chain = X's FULL native sequence
        (all residues of X present in the input PDB).
    Input chains not referenced by the contig at all are appended as full
    native chains (alphabetical chain order) after the contig chains.

    mpnn_sequence: the colon-separated designed-chains sequence (MPNN fasta
    line). rfdiff_pdb: the (re-chained) RFDiff output PDB, used to map
    design flat positions onto the fasta segments.

    Returns a dict
        chains:       [{"letter", "sequence", "host", "contig_chain"}, ...]
        res_to_token: {(input_chain, resseq): (letter, 1-based position)}
        flat_to_token: {design flat pos: 0-based protein token index}
        new_tokens:   [token indices of the new (inpaint==False) positions]
    or None when it cannot be built (callers should fail the run).
    """
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    inpaint_seq = list(trb_data.get("inpaint_seq", []))
    n_flat = len(inpaint_seq)
    if n_flat == 0:
        print("WARNING: boltz_full_complex: empty inpaint_seq in trb")
        return None

    # Input PDB: standard-AA residues per chain, in native (resseq) order.
    try:
        structure = parse_pdb_structure(cfg["pdb_path"])
    except Exception as e:
        print(f"WARNING: boltz_full_complex: could not parse input PDB: {e}")
        return None
    t2o = {k.upper(): v for k, v in IUPACData.protein_letters_3to1.items()}
    input_chains = {}  # chain letter -> [(resseq, aa1)] in resseq order
    for ch in structure.get_chains():
        cid = ch.get_id()
        entries = []
        for res in ch.get_residues():
            rn = res.get_resname().upper()
            if rn in STD_AA_RESNAMES and any(
                a.get_name() == "CA" for a in res.get_atoms()
            ):
                entries.append((res.get_id()[1], rn))  # 3-letter, converted below
        if entries:
            entries = [(rs, t2o.get(rn, "X")) for rs, rn in entries]
            input_chains.setdefault(cid, []).extend(entries)
    for cid in input_chains:
        # de-duplicate (resseq) - multi-model PDBs can repeat chains across
        # models - and sort by resseq
        by_resseq = {}
        for rs, aa in input_chains[cid]:
            by_resseq.setdefault(rs, aa)
        input_chains[cid] = sorted(by_resseq.items())

    # Design flat position -> input (chain, resseq), content-based from the trb
    ref_pdb = trb_data.get(
        "complex_con_ref_pdb_idx", trb_data.get("con_ref_pdb_idx", None)
    )
    hal_idx = list(
        trb_data.get("complex_con_hal_idx0", trb_data.get("con_hal_idx0", []))
    )
    flat_to_input = {}
    if ref_pdb is not None and len(ref_pdb) == len(hal_idx):
        for res, pos in zip(ref_pdb, hal_idx):
            flat_to_input[int(pos)] = (str(res[0]), int(res[1]))
    else:
        # older trb: fall back to positional indexing into the input PDB
        try:
            all_res, _ = get_all_residues(cfg["pdb_path"])
            ref_idx = list(
                trb_data.get(
                    "complex_con_ref_idx0", trb_data.get("con_ref_idx0", [])
                )
            )
            for ri, pos in zip(ref_idx, hal_idx):
                if ri < len(all_res):
                    flat_to_input[int(pos)] = all_res[ri]
        except Exception as e:
            print(f"WARNING: boltz_full_complex: no residue mapping: {e}")
            return None

    # Contig -> per-chain flat ranges. The contig string gives the ORDER of
    # segments (PDB ranges and new/generated segments) and the chain breaks;
    # the actual lengths of the segments come from the trb itself, because
    # RFDiff's numeric segment syntax (e.g. 1-5) does not always equal the
    # number of generated positions in the design (observed: 1-5 produced
    # 5, 3 or 4 positions depending on position). We walk the design flat
    # sequence token by token.
    contig = trb_data["config"]["contigmap"]["contigs"][0]
    body = contig.strip().strip("[]")
    tokens = []  # ("pdb", cid, lo, hi) | ("new",) | ("break",)
    for part in body.split(" "):
        for rng in part.split("/"):
            rng = rng.strip()
            if not rng:
                continue
            if rng == "0":  # chain separator
                tokens.append(("break",))
            elif rng[:1].isalpha():  # PDB range, e.g. C219-692
                lo_s, hi_s = rng.split("-")
                tokens.append(("pdb", lo_s[0], int(lo_s[1:]), int(hi_s)))
            else:  # generated segment, e.g. 1-5
                tokens.append(("new",))

    pos = 0
    chain_bounds = [(0, None)]  # (b0, b1) per contig chain; closed at breaks
    for tok in tokens:
        if tok[0] == "break":
            chain_bounds[-1] = (chain_bounds[-1][0], pos)
            chain_bounds.append((pos, None))
            continue
        if tok[0] == "new":
            count = 0
            while pos < n_flat and not inpaint_seq[pos]:
                count += 1
                pos += 1
            if count == 0:
                print(
                    f"WARNING: boltz_full_complex: expected a generated "
                    f"segment at flat position {pos}, found none"
                )
                return None
        else:  # pdb
            _, cid, lo, hi = tok
            count = 0
            last_rs = None
            while pos < n_flat and inpaint_seq[pos]:
                mapped = flat_to_input.get(pos)
                if (
                    mapped is not None
                    and mapped[0] == cid
                    and lo <= mapped[1] <= hi
                    and (last_rs is None or mapped[1] == last_rs + 1)
                ):
                    count += 1
                    last_rs = mapped[1]
                    pos += 1
                else:
                    break
            if count == 0:
                print(
                    f"WARNING: boltz_full_complex: contig range {cid}{lo}-{hi} "
                    f"matches no design positions at flat {pos}"
                )
                return None
    if chain_bounds and chain_bounds[-1][1] is None:
        chain_bounds[-1] = (chain_bounds[-1][0], pos)
    if pos != n_flat:
        print(
            f"WARNING: boltz_full_complex: contig walk consumed {pos} of "
            f"{n_flat} design positions; giving up"
        )
        return None
    chain_bounds = [(a, b) for a, b in chain_bounds if a < b]
    if not chain_bounds:
        print("WARNING: boltz_full_complex: no contig chains found")
        return None

    # Design AA at each flat position (from the MPNN fasta where the chain is
    # designed; native AA for fixed residues of non-designed chains).
    try:
        rechain_offsets, _ = getChainResidOffsets(rfdiff_pdb, None)
    except Exception as e:
        print(f"WARNING: boltz_full_complex: could not read RFDiff PDB: {e}")
        return None
    ctd = cfg.get("chains_to_design", None)
    if ctd:
        design_letters = sorted(str(ctd).split())
    else:
        design_letters = list(letters[:len(chain_bounds)])
    segments = [s for s in mpnn_sequence.split(":")]
    if len(segments) != len(design_letters):
        print(
            f"WARNING: boltz_full_complex: MPNN sequence has "
            f"{len(segments)} chains, expected {len(design_letters)} "
            f"(chains_to_design)"
        )
        return None

    def design_aa(f):
        for c in rechain_offsets:
            o = rechain_offsets[c]
            if f >= o:
                end = n_flat
                # find the end of this re-chained chain
                for c2 in rechain_offsets:
                    if rechain_offsets[c2] > o and rechain_offsets[c2] < end:
                        end = rechain_offsets[c2]
                if o <= f < end:
                    if c in design_letters:
                        seg = segments[design_letters.index(c)]
                        if f - o < len(seg):
                            return seg[f - o]
                        return None
                    else:
                        mapped = flat_to_input.get(f)
                        if mapped is not None:
                            for rs, aa in input_chains.get(mapped[0], []):
                                if rs == mapped[1]:
                                    return aa
                        return None
        return None

    design_aa_by_flat = {}
    for f in range(n_flat):
        aa = design_aa(f)
        if aa is None:
            print(
                f"WARNING: boltz_full_complex: no design amino acid for flat "
                f"position {f} (not in the MPNN sequence?)"
            )
            return None
        design_aa_by_flat[f] = aa

    # Build the boltz chains
    chains = []
    res_to_token = {}
    flat_to_token = {}
    new_tokens = []
    referenced_input_chains = set(flat_to_input.values())

    for k, (b0, b1) in enumerate(chain_bounds):
        positions = range(b0, b1)
        has_new = any(not inpaint_seq[f] for f in positions)
        # host = input chain with the most mapped positions in this contig chain
        counts = {}
        for f in positions:
            mapped = flat_to_input.get(f)
            if mapped is not None:
                counts[mapped[0]] = counts.get(mapped[0], 0) + 1
        if not counts:
            print(
                f"WARNING: boltz_full_complex: contig chain {k} maps no input "
                f"residues (all-new chain); skipping it in the full complex"
            )
            continue
        host = max(counts, key=counts.get)
        design_seq = [design_aa_by_flat[f] for f in positions]
        prefix, suffix = [], []
        if has_new:
            anchors = sorted(
                r for f in positions
                if (m := flat_to_input.get(f)) is not None and m[0] == host
                for r in [m[1]]
            )
            native = input_chains.get(host, [])
            if design_seq and flat_to_input.get(b0) is not None and \
                    flat_to_input[b0][0] == host:
                prefix = [aa for rs, aa in native if rs < min(anchors)]
            if design_seq and flat_to_input.get(b1 - 1) is not None and \
                    flat_to_input[b1 - 1][0] == host:
                suffix = [aa for rs, aa in native if rs > max(anchors)]
            seq = prefix + design_seq + suffix
        else:
            # pure fixed context: the full native host chain
            seq = [aa for _, aa in input_chains.get(host, [])]
        letter = letters[len(chains)]
        chains.append(
            {
                "letter": letter,
                "sequence": "".join(seq),
                "host": host,
                "contig_chain": k,
            }
        )
        base = len(prefix)
        chain_offset = sum(len(c["sequence"]) for c in chains[:-1])
        for i, f in enumerate(positions):
            token = base + i  # 0-based within this chain
            flat_to_token[f] = token + chain_offset
            if not inpaint_seq[f]:
                new_tokens.append(flat_to_token[f])
            mapped = flat_to_input.get(f)
            if mapped is not None:
                key = (mapped[0], mapped[1])
                if key in res_to_token:
                    print(
                        f"NOTE: boltz_full_complex: input residue {key} appears "
                        f"in the contig more than once; the first occurrence is "
                        f"used for constraint remapping"
                    )
                else:
                    res_to_token[key] = (letter, token + 1)
        if not has_new:
            for i, (rs, _) in enumerate(input_chains.get(host, [])):
                key = (host, rs)
                if key not in res_to_token:
                    res_to_token[key] = (letter, i + 1)
        else:
            # native prefix/suffix residues
            for i, (rs, _) in enumerate(native if has_new else []):
                key = (host, rs)
                if key not in res_to_token:
                    if prefix and rs < min(anchors):
                        res_to_token[key] = (letter, i + 1)
                    elif suffix and rs > max(anchors):
                        res_to_token[key] = (
                            letter,
                            len(prefix) + len(design_seq) + (i - (len(native) - len(suffix))) + 1,
                        )

    # input chains not referenced by the contig: full native chains
    for cid in sorted(set(input_chains) - {c for c, _ in referenced_input_chains}):
        letter = letters[len(chains)]
        seq = "".join(aa for _, aa in input_chains[cid])
        chains.append({"letter": letter, "sequence": seq, "host": cid, "contig_chain": None})
        offset = sum(len(c["sequence"]) for c in chains[:-1])
        for i, (rs, _) in enumerate(input_chains[cid]):
            key = (cid, rs)
            if key not in res_to_token:
                res_to_token[key] = (letter, i + 1)

    if len(chains) > 26:
        print("WARNING: boltz_full_complex: more than 26 chains; not supported")
        return None
    return {
        "chains": chains,
        "res_to_token": res_to_token,
        "flat_to_token": flat_to_token,
        "new_tokens": new_tokens,
    }


def _parse_residue_ranges(residues):
    """Parse a residue selection into a set of resseq ints.

    Accepts '219-736', '219', a list of such strings, or an int.
    """
    if isinstance(residues, int):
        return {residues}
    if isinstance(residues, str):
        residues = [residues]
    out = set()
    for part in residues:
        part = str(part).strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return out


def _seqres_chains(pdb_path):
    """Parse SEQRES records -> {chain_id: [3-letter residue names]}.

    Biopython's PDBParser discards SEQRES lines, so read them directly.
    Format: chain id in column 12 (1-based), a residue count, then three-letter
    names (up to ~13 per line; continuation lines repeat the chain id). Names
    are whitespace-split, which copes with both the standard two-space groups
    and single-spaced writers (e.g. ChimeraX exports). Repeated identical
    header blocks (e.g. from concatenated input files) are not double-counted.
    """
    seqres = {}
    with open(pdb_path) as f:
        for line in f:
            if line[:6] != "SEQRES":
                continue
            ch = line[11:12].strip()
            if not ch:
                continue
            m = re.match(r"^SEQRES\s+\d+\s+\S?\s+(\d+)\s+(.*)$", line)
            names = m.group(2).split() if m else []
            names = [n for n in names if len(n) >= 2]
            if not names:
                continue
            existing = seqres.get(ch)
            if existing is None:
                seqres[ch] = names
            elif names == existing[: len(names)]:
                continue  # duplicate block (concatenated file headers)
            else:
                existing.extend(names)
    return seqres


def _protein_sequence_from_pdb(pdb_path, chain_id, residues=None):
    """Extract the one-letter sequence of one chain from a PDB file.

    Used for boltz_extras `protein` entries that reference the input PDB by
    chain id (`pdb_chain`) instead of spelling out the sequence. Modified
    amino acids (ATOM records with non-standard names) map to 'X'; non-polymer
    residues that Biopython merges into the chain (HETATM ligands, metals,
    water - e.g. a CU ion on a protein chain id) are skipped. `residues`
    optionally selects a subset, e.g. "693-720" or ["219-692", "721-736"];
    default is the whole chain.

    Residues without coordinates (e.g. a PDB trimmed in ChimeraX that keeps
    the full sequence) are taken from the chain's SEQRES record when present;
    for a whole-chain request the full SEQRES sequence is returned. The
    SEQRES numbering is anchored at the chain's first coordinated residue
    (or 1 if the chain has no coordinates at all).
    """
    structure = parse_pdb_structure(pdb_path)
    wanted = _parse_residue_ranges(residues) if residues is not None else None
    three2one = {k.upper(): v for k, v in IUPACData.protein_letters_3to1.items()}
    atom_seq = {}  # resseq -> one-letter (residue has coordinates)
    found_chain = False
    for chain in structure.get_chains():
        if chain.get_id() != chain_id:
            continue
        found_chain = True
        for res in chain.get_residues():
            hetflag, resseq, _icode = res.get_id()
            resname = res.get_resname().upper()
            if resname in three2one:
                atom_seq[resseq] = three2one[resname]
            elif hetflag == " ":
                atom_seq[resseq] = "X"  # modified amino acid (polymer residue)
            # else: non-polymer HETATM residue (ligand/metal/water) - skip
    # SEQRES fallback for positions without coordinates
    seqres_map = {}
    seqres_names = _seqres_chains(pdb_path).get(chain_id)
    if not found_chain and seqres_names is None:
        raise ValueError(
            f"boltz_extras: chain '{chain_id}' not found in input PDB "
            f"({pdb_path})"
        )
    if seqres_names:
        first = min(atom_seq) if atom_seq else 1
        seqres_map = {first + i: n for i, n in enumerate(seqres_names)}
    if wanted is None:
        wanted = set(atom_seq) | set(seqres_map)
    if not wanted:
        raise ValueError(
            f"boltz_extras: the residue selection '{residues}' matched no "
            f"residues of chain '{chain_id}' in {pdb_path}"
        )
    seq_parts = []
    missing = []
    for resseq in sorted(wanted):
        if resseq in atom_seq:
            seq_parts.append(atom_seq[resseq])
        elif resseq in seqres_map:
            seq_parts.append(three2one.get(seqres_map[resseq].upper(), "X"))
        else:
            missing.append(resseq)
    if missing:
        shown = ", ".join(map(str, missing[:8]))
        if len(missing) > 8:
            shown += f" ... ({len(missing)} total)"
        raise ValueError(
            f"boltz_extras: chain '{chain_id}' of {pdb_path} has neither "
            f"coordinates nor a SEQRES record for residues: {shown}"
        )
    return "".join(seq_parts)


def seqres_has_chain(pdb_path, chain_id):
    """True if the PDB has a SEQRES record for the chain (no coordinates needed)."""
    return chain_id in _seqres_chains(pdb_path)


def _merge_boltz_extras(data, cfg, remap=None):
    """Merge static additions from cfg.boltz_extras into a boltz input dict.

    The config block mirrors the Boltz input schema (see boltz docs), plus
    two prosculpt conveniences:

        boltz_extras:
          sequences:   # extra non-protein (or static protein) chains
            - ligand: {id: C, ccd: SAH}          # or: smiles: 'CCO'
            - ligand: {id: D, ccd: [EDO, GLU]}   # multi-residue ligand
            - rna:    {id: E, sequence: GCAUAGC}
            - dna:    {id: F, sequence: ATCG}
            # static protein: give the sequence explicitly ...
            - protein: {id: G, sequence: MKTAYIA...}
            # ... or take it from a chain of the input PDB (pdb_path),
            # optionally with a residue selection. If the input PDB was
            # trimmed (e.g. in ChimeraX), point 'pdb_file' at the FULL,
            # untrimmed structure instead:
            - protein: {id: G, pdb_chain: C, residues: "693-720"}
            - protein: {id: H, pdb_chain: C, pdb_file: /path/to/full.pdb}
          constraints:  # bond / pocket / contact
            - pocket: {binder: C, contacts: [[A, 42]], max_distance: 6.0}
          templates:    # Boltz templates anchoring the prediction to a structure
            - cif: /path/to/template.cif           # or: pdb: /path/to/template.pdb
            # prosculpt convenience: use the input PDB as the template:
            - input_pdb: true
              chain_id: [G, H]        # optional: model chain ids to seed
              template_id: [C, D]     # optional: template chain ids (paired 1:1)
          properties:   # [{affinity: {binder: C}}]
          version: 1

    - Entries without an `id` get the next free chain letter auto-assigned.
    - Explicit `id`s are checked against the designed protein chain IDs
      (A, B, C... by position in the colon-separated MPNN sequence) and
      raise a clear error on collision.
    - If cfg has no boltz_extras (or it is null), data is returned unchanged.
    """
    if cfg is None or cfg.get("boltz_extras", None) is None:
        return data

    extras = OmegaConf.to_container(cfg.boltz_extras, resolve=True)

    used_ids = set()
    for seq in data["sequences"]:
        cid = next(iter(seq.values()))["id"]
        used_ids.update(cid if isinstance(cid, list) else [cid])

    for entry in extras.get("sequences", []) or []:
        etype = next(iter(entry))
        if etype not in {"protein", "dna", "rna", "ligand"}:
            raise ValueError(
                f"boltz_extras.sequences: invalid entry type '{etype}' "
                f"(expected protein, dna, rna or ligand)"
            )
        spec = dict(entry[etype])
        if "id" not in spec:
            spec["id"] = _boltz_next_free_id(used_ids)
        else:
            ids = spec["id"] if isinstance(spec["id"], list) else [spec["id"]]
            for cid in ids:
                if cid in used_ids:
                    raise ValueError(
                        f"boltz_extras: chain id '{cid}' collides with an "
                        f"already used chain id. Designed protein chains are "
                        f"labeled A..Z by position in the colon-separated "
                        f"MPNN sequence; pick a free letter or omit the id "
                        f"to auto-assign."
                    )
        used_ids.update(ids if isinstance(spec["id"], list) else [spec["id"]])
        if etype == "protein" and "pdb_chain" in spec:
            if "sequence" in spec:
                raise ValueError(
                    "boltz_extras: protein entry has both 'sequence' and "
                    "'pdb_chain'; use one or the other."
                )
            pdb_path = spec.pop("pdb_file", None) or cfg.get("pdb_path", None)
            if pdb_path is None:
                raise ValueError(
                    "boltz_extras: a protein entry with 'pdb_chain' requires "
                    "pdb_path (or an explicit 'pdb_file') - the PDB the chain "
                    "is taken from."
                )
            spec["sequence"] = _protein_sequence_from_pdb(
                pdb_path, spec.pop("pdb_chain"), spec.pop("residues", None)
            )
        data["sequences"].append({etype: spec})

    # All chain ids present in this yaml (designed + extra chains). Used to
    # tell "static reference to a known chain" (keep) from "reference to a
    # chain that is not in this yaml" (drop) during constraint remapping.
    valid_ids = set()
    for seq in data["sequences"]:
        cid = next(iter(seq.values()))["id"]
        valid_ids.update(cid if isinstance(cid, list) else [cid])

    # prosculpt convenience: templates may reference the input PDB directly
    if extras.get("templates") is not None:
        resolved_templates = []
        for t in extras["templates"]:
            t = dict(t)
            if t.pop("input_pdb", False):
                pdb_path = cfg.get("pdb_path", None)
                if pdb_path is None:
                    raise ValueError(
                        "boltz_extras: a template with 'input_pdb: true' "
                        "requires pdb_path (the input PDB)."
                    )
                if "cif" in t or "pdb" in t:
                    raise ValueError(
                        "boltz_extras: template entry has both 'input_pdb' "
                        "and an explicit 'cif'/'pdb' path."
                    )
                t["pdb"] = pdb_path
            # Filter chain_id/template_id pairs down to chains that exist in
            # this yaml (needed for monomer-only yamls in symmetry mode, where
            # the C2-image chains are absent; Boltz requires every chain_id to
            # be an input protein chain).
            cids = t.get("chain_id", None)
            tids = t.get("template_id", None)
            if cids is not None and tids is not None:
                keep_c, keep_t = [], []
                for cid, tid in zip(cids, tids):
                    if cid in valid_ids:
                        keep_c.append(cid)
                        keep_t.append(tid)
                    else:
                        print(
                            f"WARNING: template chain_id '{cid}' is not a "
                            f"chain of this yaml; dropping template pair "
                            f"({cid} -> {tid})."
                        )
                if keep_c:
                    t["chain_id"] = keep_c
                    t["template_id"] = keep_t
                else:
                    del t["chain_id"]
                    del t["template_id"]
            resolved_templates.append(t)
        extras["templates"] = resolved_templates

    # Remap chain/residue references in constraints from input PDB coordinates
    # to the generated (designed) sequence coordinates, so that contacts stay
    # attached to the right residues even when the contig changes lengths or
    # re-maps chains. Static references (ligand/metal chain ids) are left
    # as-is; references to chains that are neither remappable nor present in
    # the yaml are dropped (they would fail Boltz input validation).
    if remap is not None:
        kept_constraints = []
        for c in extras.get("constraints", []) or []:
            dropped = False
            if "pocket" in c and "contacts" in c["pocket"]:
                contacts = [
                    _remap_boltz_ref_or_drop(r, remap, valid_ids)
                    for r in c["pocket"]["contacts"]
                ]
                contacts = [r for r in contacts if r is not None]
                if not contacts:
                    dropped = True  # no usable contacts left
                c["pocket"]["contacts"] = contacts
            if not dropped and "contact" in c:
                for tok in ("token1", "token2"):
                    if tok in c["contact"]:
                        new_tok = _remap_boltz_ref_or_drop(
                            c["contact"][tok], remap, valid_ids
                        )
                        if new_tok is None:
                            dropped = True
                            break
                        c["contact"][tok] = new_tok
            if not dropped and "bond" in c:
                for atom in ("atom1", "atom2"):
                    if atom in c["bond"]:
                        new_atom = _remap_boltz_ref_or_drop(
                            c["bond"][atom], remap, valid_ids
                        )
                        if new_atom is None:
                            dropped = True
                            break
                        c["bond"][atom] = new_atom
            if dropped:
                print(
                    f"WARNING: dropping boltz_extras constraint with no valid "
                    f"references: {c}"
                )
            else:
                kept_constraints.append(c)
        extras["constraints"] = kept_constraints

    for key in ("constraints", "templates", "properties", "version"):
        value = extras.get(key, None)
        if value is not None and value != []:
            data[key] = value
    return data


def _realign_extra_protein_msas(data, n_design_seqs, model_id, alignment_dir):
    """Re-project boltz_extras protein MSAs onto their chain sequences.

    Extra static proteins (e.g. other native subunits) normally reference a
    raw input a3m file that prosculpt does not re-project; such files carry
    lowercase insertions (ragged lines) and occasionally X. Every a3m file
    written into a generated boltz.yaml is therefore re-projected through
    recalculate_a3m, which guarantees a rectangular (query-width), X-free
    file whose query line equals the chain sequence.

    Boltz requires that all proteins with the same sequence share one MSA
    file, so a sequence already present on another chain of the yaml (a
    designed chain or another extra) keeps that chain's MSA path instead of
    getting a new file.
    """
    seq_to_msa = {}
    for seq in data["sequences"]:
        if next(iter(seq)) != "protein":
            continue
        p = seq["protein"]
        if p.get("msa") not in (None, "empty"):
            seq_to_msa.setdefault(p["sequence"], p["msa"])

    for seq in data["sequences"][n_design_seqs:]:
        if next(iter(seq)) != "protein":
            continue
        p = seq["protein"]
        msa = p.get("msa", None)
        if not msa or msa == "empty" or "://" in str(msa):
            continue
        cid = str(p.get("id"))
        # identical sequence on another chain: Boltz requires ONE shared MSA
        if seq_to_msa.get(p["sequence"]) not in (None, p["msa"]):
            shared = seq_to_msa[p["sequence"]]
            print(
                f"boltz_extras chain {cid}: sequence identical to another "
                f"chain of this yaml; sharing its MSA {shared}"
            )
            p["msa"] = shared
            continue
        out_path = os.path.join(alignment_dir, f"{model_id}_extras_{cid}.a3m")
        try:
            recalculate_a3m(os.path.expanduser(str(msa)), p["sequence"], out_path)
        except Exception as e:
            print(
                f"WARNING: could not re-align boltz_extras MSA {msa} for "
                f"chain {cid} ({e}); keeping the original path."
            )
            continue
        print(f"boltz_extras chain {cid}: re-aligned MSA {msa} -> {out_path}")
        seq_to_msa[p["sequence"]] = out_path
        p["msa"] = out_path


def make_boltz_input_yaml(
    cfg,
    model_id,
    mpnn_sequence,
    output_dir,
    input_alignment_dir,
    ref_to_pos=None,
    full_complex=None,
):
    chain_ids = []
    sequences = []
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if full_complex is not None:
        # boltz_full_complex mode: the Boltz protein chains are the full input
        # PDB chains with the designed (RFDiff/MPNN) segments spliced in, not
        # the bare MPNN design chains (see build_boltz_full_complex).
        for c in full_complex["chains"]:
            chain_ids.append(c["letter"])
            sequences.append(c["sequence"])
            print(
                f"boltz_full_complex chain {c['letter']} (host {c['host']}): "
                f"{len(c['sequence'])} residues"
            )
        split_chains = list(sequences)
    else:
        # Split multi-chain sequences by colon
        print(
            f"splitting mpnn sequence {mpnn_sequence} by colon for Boltz yaml generation..."
        )
        split_chains = mpnn_sequence.split(":")
        for i, chain_seq in enumerate(split_chains):
            chain_id = f"{letters[i]}"
            chain_ids.append(chain_id)
            print(f"Chain ID: {chain_id}, Sequence: {chain_seq}")
            sequences.append(chain_seq)

    # Make boltz yaml
    data = dict(sequences=dict())
    data["sequences"] = []
    if cfg.use_a3m and input_alignment_dir is not None:
        sequence_to_msa = {}

        for idxsequence, sequence in enumerate(sequences):
            chain_id = chain_ids[idxsequence]
            cleaned = "".join(c for c in sequence if c.isalpha())

            print(
                f"Chain ID: {chain_id}, Sequence: {sequence}, Cleaned Sequence: {cleaned}"
            )

            # If we've already seen this exact sequence, reuse its MSA
            if cleaned in sequence_to_msa:
                msa_path = sequence_to_msa[cleaned]
            else:
                msa_path = f"{input_alignment_dir}/{model_id}_{chain_id}.a3m"
                sequence_to_msa[cleaned] = msa_path

            data["sequences"].append(
                {
                    "protein": {
                        "id": chain_id,
                        "sequence": cleaned,
                        "msa": msa_path,
                    }
                }
            )
    else:
        for idxsequence, sequence in enumerate(sequences):
            cleaned = "".join(c for c in sequence if c.isalpha())
            print(
                f"Chain ID: {chain_ids[idxsequence]}, Sequence: {sequence}, Cleaned Sequence: {cleaned}"
            )
            chain_id = chain_ids[idxsequence]
            data["sequences"].append(
                {
                    "protein": {
                        "id": chain_id,
                        "sequence": cleaned,
                        "msa": "empty",
                    }
                }
            )

    # Build the boltz_extras constraint remap: input-PDB (chain, resseq)
    # references -> (Boltz chain id, 1-based residue number) in the chains
    # written above (see _remap_boltz_ref).
    remap = None
    if full_complex is not None:
        res_to_token = full_complex["res_to_token"]

        def remap(chain, resnum):
            return res_to_token.get((str(chain), int(resnum)))

    elif ref_to_pos:
        pos_to_chain_res = {}
        flat = 0
        for i, chain_seq in enumerate(split_chains):
            cleaned_len = sum(1 for c in chain_seq if c.isalpha())
            for j in range(cleaned_len):
                pos_to_chain_res[flat + j] = (chain_ids[i], j + 1)
            flat += cleaned_len

        def remap(chain, resnum, _pos=pos_to_chain_res, _ref=ref_to_pos):
            pos = _ref.get((chain, resnum))
            if pos is None:
                return None
            return _pos.get(pos)

    # Add user-specified static additions (ligands, RNA/DNA, constraints,
    # templates, affinity properties) from the boltz_extras config block.
    # Constraint chain/residue references are remapped from input PDB
    # coordinates to the newly formed chains (handles length changes).
    n_design_seqs = len(data["sequences"])
    data = _merge_boltz_extras(data, cfg, remap=remap)

    if cfg.use_a3m and input_alignment_dir is not None:
        _realign_extra_protein_msas(
            data, n_design_seqs, model_id, input_alignment_dir
        )

    with open(f"{output_dir}/{model_id}.yaml", "w") as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

    if full_complex is not None:
        # Sidecar for the scoring stage: maps design flat positions onto the
        # (longer, spliced) Boltz protein tokens, so plDDT/RMSD can be
        # evaluated on the designed region only.
        with open(f"{output_dir}/{model_id}.full_complex.json", "w") as f:
            json.dump(
                {
                    "flat_to_token": {
                        str(k): v
                        for k, v in full_complex["flat_to_token"].items()
                    },
                    "new_tokens": full_complex["new_tokens"],
                    "chains": [
                        {
                            "letter": c["letter"],
                            "host": c["host"],
                            "length": len(c["sequence"]),
                        }
                        for c in full_complex["chains"]
                    ],
                },
                f,
            )
    return f"{output_dir}/{model_id}.yaml"


def make_AF3_input_json(cfg, model_id, mpnn_sequence, output_dir, input_alignment_dir):
    chain_ids = []
    sequences = []
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    # Split multi-chain sequences by colon
    print(
        f"splitting mpnn sequence {mpnn_sequence} by colon for AF3 yaml generation..."
    )
    split_chains = mpnn_sequence.split(":")
    for i, chain_seq in enumerate(split_chains):
        chain_id = f"{letters[i]}"
        chain_ids.append(chain_id)
        print(f"Chain ID: {chain_id}, Sequence: {chain_seq}")
        sequences.append(chain_seq)

    # Make boltz yaml
    data = dict(sequences=dict())
    data["name"] = model_id
    data["sequences"] = []
    if cfg.use_a3m and input_alignment_dir is not None:
        for idxsequence, sequence in enumerate(sequences):
            chain_id = chain_ids[idxsequence]
            cleaned = "".join(c for c in sequence if c.isalpha())
            print(
                f"Chain ID: {chain_ids[idxsequence]}, Sequence: {sequence}, Cleaned Sequence: {cleaned}"
            )
            data["sequences"].append(
                {
                    "protein": {
                        "id": chain_id,
                        "sequence": cleaned,
                        "unpairedMsaPath": f"{input_alignment_dir}/{model_id}_{chain_id}.a3m",
                        "pairedMsa": "",
                        "templates": [],
                    }
                }
            )
            data["modelSeeds"] = [1]
            data["dialect"] = "alphafold3"
            data["version"] = 1
    else:
        for idxsequence, sequence in enumerate(sequences):
            cleaned = "".join(c for c in sequence if c.isalpha())
            print(
                f"Chain ID: {chain_ids[idxsequence]}, Sequence: {sequence}, Cleaned Sequence: {cleaned}"
            )
            chain_id = chain_ids[idxsequence]
            data["sequences"].append(
                {
                    "protein": {
                        "id": chain_id,
                        "sequence": cleaned,
                        "unpairedMsa": "",
                        "pairedMsa": "",
                        "templates": [],
                    }
                }
            )
            data["modelSeeds"] = [1]
            data["dialect"] = "alphafold3"
            data["version"] = 1

    with open(f"{output_dir}/{model_id}.json", "w") as outfile:
        json.dump(data, outfile)
    return f"{output_dir}/{model_id}.json"


def get_rmsd_from_coords(native_coords, model_coords, rot, tran):
    model_coords_rotated = np.dot(model_coords, rot) + tran
    diff = native_coords - model_coords_rotated
    RMSD = np.sqrt(sum(sum(diff**2)) / native_coords.shape[0])
    return RMSD


def extract_chain_letter(filename):
    match = re.search(r"Chain_([A-Z])", filename)
    return match.group(1) if match else ""


def masked_positions(seq1, seq2, min_block=3):
    aligner = PairwiseAligner()
    aligner.mode = "local"

    alignment = aligner.align(seq1, seq2)[0]
    aligned1, aligned2 = alignment[0], alignment[1]

    masked = []
    positions = []
    seq1_index = 0

    for a, b in zip(aligned1, aligned2):
        if a != "-":
            masked.append(a == b)
            seq1_index += 1

    # Apply min_block filtering
    n = len(masked)
    i = 0
    while i < n:
        if not masked[i]:
            i += 1
            continue
        j = i
        while j < n and masked[j]:
            j += 1
        if (j - i) < min_block:
            for k in range(i, j):
                masked[k] = False
        i = j

    # Collect positions that remain True
    positions = [idx for idx, keep in enumerate(masked) if keep]

    return positions


def calculate_RMSD_linker_len(
    cfg, trb_path, af2_pdb, starting_pdb, rfdiff_pdb_path, symmetry, model_monomer,
    flat_to_token=None,
):
    # First calculate RMSD between input protein and AF2 generated protein
    # Second calcualte number of total generated AA by RFDIFF
    #   - if designing only in one location the number is equal linker length

    parser = PDBParser(PERMISSIVE=1)
    structure_af2 = parser.get_structure("af2", af2_pdb)

    skipRfDiff = cfg.get("skipRfDiff", False)
    designable_residues = cfg.get("designable_residues", None)
    # Skip if trb does not exist

    if skipRfDiff:
        print(
            "skipping rfdiffusion. RMSD_Sculpted is that of the designable residues. RMSD_Motif is that of the non-designable residues"
        )

        if symmetry != None or model_monomer:
            rmsd = homooligomer_rmsd.align_oligomers(
                starting_pdb, af2_pdb, save_aligned=False
            )
            return ([round(rmsd, 1), -1, -1, -1, -1], -1)
        else:  # There is a lot of code duplication here from when RFDiff is used, but I don't want to deal with the problem of making the skipped one go into the non-skip correctly
            # rmsd = homooligomer_rmsd.align_monomer(starting_pdb, af2_pdb, save_aligned=False)

            chainResidOffset, con_hal_pdb_idx_complete = getChainResidOffsets(
                cfg.pdb_path, designable_residues
            )

            selected_residue_data = [
                x + (chainResidOffset[chain] - 1)
                for chain, x in con_hal_pdb_idx_complete
            ]
            selected_residues_in_designed_chains = selected_residue_data  # This will be a problem if there are fixed chains. TODO:FIX

            # Filter to standard AAs: predicted PDBs may contain ligand/metal
            # chains added via boltz_extras (see filter_protein_residues).
            all_af2_res = filter_protein_residues(
                structure_af2.get_residues()
            )
            all_af2_res_ca = [ind["CA"] for ind in all_af2_res]
            af2_all_fixed_res = [
                all_af2_res[ind]["CA"] for ind in selected_residue_data
            ]
            af2_sculpted_res = [
                ind["CA"] for ind in all_af2_res if ind["CA"] not in af2_all_fixed_res
            ]
            # af2_fixed_chain_res=[all_af2_res[ind]['CA'] for ind in selected_residues_in_fixed_chains]
            # af2_motif_res=[all_af2_res[ind]['CA'] for ind in selected_residues_in_designed_chains]
            structure_rfdiff = parser.get_structure("control", rfdiff_pdb_path)
            all_rfdiff_res = list(
                structure_rfdiff.get_residues()
            )  # obtain a list of all the residues in the structure, structure_control is object
            all_rfdiff_res_ca = [ind["CA"] for ind in all_rfdiff_res]

            superimposer = SVDSuperimposer()
            rfdiff_all_coords = np.array([a.coord for a in all_rfdiff_res_ca])
            af2_all_coords = np.array([a.coord for a in all_af2_res_ca])

            superimposer.set(rfdiff_all_coords, af2_all_coords)
            superimposer.run()
            rmsd = get_rmsd_from_coords(
                rfdiff_all_coords, af2_all_coords, superimposer.rot, superimposer.tran
            )

            rfdiff_all_fixed_res = [
                all_rfdiff_res[ind]["CA"] for ind in selected_residue_data
            ]  # retrieve the residue with the corresponding index from rfdiff_res
            rfdiff_sculpted_res = [
                ind["CA"]
                for ind in all_rfdiff_res
                if ind["CA"] not in rfdiff_all_fixed_res
            ]

            superimposer = SVDSuperimposer()
            rfdiff_all_fixed_coords = np.array([a.coord for a in rfdiff_all_fixed_res])
            af2_all_fixed_coords = np.array([a.coord for a in af2_all_fixed_res])
            superimposer.set(rfdiff_all_fixed_coords, af2_all_fixed_coords)
            superimposer.run()
            rmsd_all_fixed = get_rmsd_from_coords(
                rfdiff_all_fixed_coords,
                af2_all_fixed_coords,
                superimposer.rot,
                superimposer.tran,
            )
            # rfdiff_fixed_chain_res=[all_rfdiff_res[ind]['CA'] for ind in selected_residues_in_fixed_chains]
            # rfdiff_motif_res=[all_rfdiff_res[ind]['CA'] for ind in selected_residues_in_designed_chains]
            rfdiff_sculpted_coords = [a.coord for a in rfdiff_sculpted_res]
            af2_sculpted_coords = [a.coord for a in af2_sculpted_res]
            rfdiff_sculpted_coords = np.array(rfdiff_sculpted_coords)
            af2_sculpted_coords = np.array(af2_sculpted_coords)

            rmsd_sculpted = get_rmsd_from_coords(
                rfdiff_sculpted_coords,
                af2_sculpted_coords,
                superimposer.rot,
                superimposer.tran,
            )
            print(
                [
                    round(rmsd, 1),
                    round(rmsd_all_fixed, 1),
                    round(rmsd_sculpted, 1),
                    -1,
                    round(rmsd_all_fixed, 1),
                ]
            )
            return (
                [
                    round(rmsd, 1),
                    round(rmsd_all_fixed, 1),
                    round(rmsd_sculpted, 1),
                    -1,
                    round(rmsd_all_fixed, 1),
                ],
                -1,
            )

    with open(trb_path, "rb") as f:
        trb_dict = pickle.load(f)

    # Get different data from trb file depending on the fact if designing a monomer (one chain) or heteromer
    # complex_con_rex_idx present only if there are chains that are completely fixed.
    # Data structure: con_ref_idx0 = [0, 1, 2, 3, ...]
    #   Info: array of input pdb AA indices starting 0 (con_ref_pdb_idx), and where they are in the output pdb (con_hal_pdb_idx)
    #   In complex_con_hal_idx0 there is no chain info however RFDIFF changes pdb indeces to go from 1 to n (e.g. 1st AA in chain B has idx 34)

    # selected_residues_data will hold the information only of those residues that were selected from the reference structure to be used in the final design
    if "complex_con_ref_idx0" in trb_dict:
        selected_residues_data = trb_dict["complex_con_hal_idx0"]
        selected_residues_in_designed_chains = trb_dict["con_hal_idx0"]
        if (
            trb_dict["config"]["contigmap"]["provide_seq"] != None
        ):
            provide_seq_residues = np.where(       
                [
                    a != b
                    for a, b in zip(trb_dict["inpaint_seq"], trb_dict["inpaint_str"])
                ]
            )[0]  # if this works...
            print(
                f"DEBUG: PARTIAL DIFUSSION KEEPING RESIDUES {provide_seq_residues}"
            )
            selected_residues_in_designed_chains=copy.deepcopy(sorted(list(selected_residues_in_designed_chains)+list(provide_seq_residues))) #We gotta add the provide_seq residues to both lists
            selected_residues_data=copy.deepcopy(sorted(list(selected_residues_data)+list(provide_seq_residues)))
            # print(selected_residues_in_designed_chains)

        # selected_residues_in_fixed_chains=trb_dict['receptor_con_hal_idx0'] #This actually doesn't work and I think it's a bug in RFDiff
        selected_residues_in_fixed_chains = [
            res
            for res in selected_residues_data
            if res not in selected_residues_in_designed_chains
        ]
    else:
        partial_diffusion = cfg.get("partial_diffusion", False)
        if not partial_diffusion:
            selected_residues_data = trb_dict["con_hal_idx0"]
        else:  # When using partial diffusion, RFDiffusion doesn't put anything of the diffused chain on the
            # con_hal_pdb_idx (because nothing is technically fixed). This means that we need to recompile it
            # based on the inpaint_seq
            selected_residues_data = []

            chainResidOffset, con_hal_pdb_idx_complete = getChainResidOffsets(
                cfg.pdb_path, designable_residues
            )
            for id0, value in enumerate(trb_dict["inpaint_seq"]):
                if value == True:
                    abeceda = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    for key in abeceda:
                        if key in chainResidOffset:
                            if id0 >= chainResidOffset[key]:
                                currentResidueChain = key
                    # residue_data_control_1.append((currentResidueChain,id0+chainResidOffset[currentResidueChain]))
                    selected_residues_data.append(id0)

        selected_residues_in_fixed_chains = []
        selected_residues_in_designed_chains = selected_residues_data

    # Filter to standard AAs: predicted PDBs may contain ligand/metal
    # chains added via boltz_extras (see filter_protein_residues).
    all_af2_res = filter_protein_residues(structure_af2.get_residues())
    if (
        not flat_to_token
        and len(all_af2_res) < len(trb_dict["inpaint_seq"])
    ):
        # Bare-chain mode: the predicted PDB should contain exactly the
        # designed (MPNN fasta) chains, in RFDiff output chain order, so its
        # residue indices equal the trb flat positions. If rechain produced
        # more chains than chains_to_design covers, the extra (always
        # trailing) chain(s) never reached the prediction input, so the
        # predicted PDB is a flat PREFIX of the RFDiff output. Clip all index
        # lists to the predicted part so the stats cover what was actually
        # predicted (instead of raising IndexError downstream).
        n_pred = len(all_af2_res)
        n_trb = len(trb_dict["inpaint_seq"])
        print(
            f"WARNING: predicted PDB has {n_pred} protein residues but the "
            f"RFDiff output has {n_trb}: rechain created more chains than "
            f"chains_to_design covers, so the extra chain(s) are missing from "
            f"the prediction input. RMSD/plDDT stats cover the predicted part "
            f"only. To predict the full complex, enable boltz_full_complex (or "
            f"add the extra chain letters to chains_to_design - and rename "
            f"any colliding boltz_extras chain ids)."
        )
        selected_residues_data = [
            i for i in selected_residues_data if i < n_pred
        ]
        selected_residues_in_fixed_chains = [
            i for i in selected_residues_in_fixed_chains if i < n_pred
        ]
        selected_residues_in_designed_chains = [
            i for i in selected_residues_in_designed_chains if i < n_pred
        ]
    if flat_to_token:
        # boltz_full_complex mode: the predicted PDB holds the FULL spliced
        # chains (native + designed residues, plus extra chains), so the
        # trb design-flat indices do NOT equal its residue indices. Select
        # the predicted design-region atoms through the design-flat ->
        # protein-token map saved next to the boltz yaml by
        # build_boltz_full_complex (protein token == index into
        # all_af2_res, since protein chains come first in the yaml).
        fixed_set = set(selected_residues_data)
        all_flat_positions = sorted(flat_to_token)

        def _token_ca(f):
            return all_af2_res[flat_to_token[f]]["CA"]

        all_af2_res_ca = [_token_ca(f) for f in all_flat_positions]
        af2_all_fixed_res = [
            _token_ca(f) for f in selected_residues_data if f in flat_to_token
        ]
        af2_sculpted_res = [
            _token_ca(f)
            for f in all_flat_positions
            if f not in fixed_set and f in flat_to_token
        ]
        af2_fixed_chain_res = [
            _token_ca(f)
            for f in selected_residues_in_fixed_chains
            if f in flat_to_token
        ]
        af2_motif_res = [
            _token_ca(f)
            for f in selected_residues_in_designed_chains
            if f in flat_to_token
        ]
    else:
        all_af2_res_ca = [ind["CA"] for ind in all_af2_res]

        af2_all_fixed_res = [all_af2_res[ind]["CA"] for ind in selected_residues_data]
        af2_sculpted_res = [
            ind["CA"] for ind in all_af2_res if ind["CA"] not in af2_all_fixed_res
        ]
        af2_fixed_chain_res = [
            all_af2_res[ind]["CA"] for ind in selected_residues_in_fixed_chains
        ]
        af2_motif_res = [
            all_af2_res[ind]["CA"] for ind in selected_residues_in_designed_chains
        ]

    trb_help = list(trb_dict["inpaint_str"])
    linker_indeces = [
        boolean for boolean in trb_help if boolean == False
    ]  # calculate linker length here - convenient
    linker_length = len(linker_indeces)

    # io=PDBIO()
    # io.set_structure(structure_af2)
    # io.save("af2_pdb_2.pdb") #This is not necessary and might be slowing down everything a bit.
    rmsd = -1
    rmsd_all_fixed = (
        -1
    )  # If there's no starting structure, we cannot compare it. RMSD is undefined (-1)
    rmsd_sculpted = -1
    rmsd_fixed_chains = -1
    rmsd_motif = -1
    # if starting_pdb:
    structure_rfdiff = parser.get_structure("control", rfdiff_pdb_path)

    all_rfdiff_res = list(
        structure_rfdiff.get_residues()
    )  # obtain a list of all the residues in the structure, structure_control is object
    if not flat_to_token and len(all_rfdiff_res) > len(all_af2_res):
        # keep the RFDiff side consistent with the clipped predicted part
        # (see the prefix warning above)
        all_rfdiff_res = all_rfdiff_res[: len(all_af2_res)]
    all_rfdiff_res_ca = [ind["CA"] for ind in all_rfdiff_res]

    rfdiff_all_fixed_res = [
        all_rfdiff_res[ind]["CA"] for ind in selected_residues_data
    ]  # retrieve the residue with the corresponding index from rfdiff_res
    rfdiff_sculpted_res = [
        ind["CA"] for ind in all_rfdiff_res if ind["CA"] not in rfdiff_all_fixed_res
    ]
    rfdiff_fixed_chain_res = [
        all_rfdiff_res[ind]["CA"] for ind in selected_residues_in_fixed_chains
    ]
    rfdiff_motif_res = [
        all_rfdiff_res[ind]["CA"] for ind in selected_residues_in_designed_chains
    ]

    if len(rfdiff_all_fixed_res) != len(af2_all_fixed_res):
        print(
            "Fixed and moving atom lists differ in size"
        )  # for now, this is when input pdb and output are different length
        print(rfdiff_all_fixed_res, af2_all_fixed_res)
        return (-1, -1)

    # Align all and get RMSD of all
    superimposer = SVDSuperimposer()
    rfdiff_all_coords = np.array([a.coord for a in all_rfdiff_res_ca])
    af2_all_coords = np.array([a.coord for a in all_af2_res_ca])

    if len(rfdiff_all_coords) == len(af2_all_coords):
        superimposer.set(rfdiff_all_coords, af2_all_coords)
        superimposer.run()
        rmsd = get_rmsd_from_coords(
            rfdiff_all_coords,
            af2_all_coords,
            superimposer.rot,
            superimposer.tran,
        )
    else:
        # e.g. boltz_full_complex predictions contain extra native (spliced)
        # residues, so a whole-structure superimposition is not defined
        print(
            f"NOTE: RFDiff and predicted PDB have different numbers of "
            f"protein residues ({len(rfdiff_all_coords)} vs "
            f"{len(af2_all_coords)}); 'all' RMSD set to -1."
        )
        rmsd = -1

    # Align all reference residues if there are no fixed chains. Otherwise, align only fixed chains.  (Very nice because fully fixed chains should be a stable reference)
    # then get rmsd_all_fixed and rmsd_sculpted (If any)
    # superimposer = SVDSuperimposer()
    rfdiff_all_fixed_coords = np.array([a.coord for a in rfdiff_all_fixed_res])
    af2_all_fixed_coords = np.array([a.coord for a in af2_all_fixed_res])

    rfdiff_fixed_chain_coords = np.array([a.coord for a in rfdiff_fixed_chain_res])
    rfdiff_motif_res_coords = np.array([a.coord for a in rfdiff_motif_res])

    af2_fixed_chain_coords = np.array([a.coord for a in af2_fixed_chain_res])
    af2_motif_res_coords = np.array([a.coord for a in af2_motif_res])

    if len(rfdiff_fixed_chain_coords) == 0:  # (there are no fixed chains)
        if len(rfdiff_all_fixed_coords) != 0:  # (There are fixed residues at all)
            superimposer.set(rfdiff_all_fixed_coords, af2_all_fixed_coords)
            superimposer.run()
            rmsd_all_fixed = get_rmsd_from_coords(
                rfdiff_all_fixed_coords,
                af2_all_fixed_coords,
                superimposer.rot,
                superimposer.tran,
            )
        else:
            rmsd_all_fixed = -1
    else:
        superimposer.set(rfdiff_fixed_chain_coords, af2_fixed_chain_coords)
        superimposer.run()
        rmsd_fixed_chains = get_rmsd_from_coords(
            rfdiff_fixed_chain_coords,
            af2_fixed_chain_coords,
            superimposer.rot,
            superimposer.tran,
        )
        rmsd_all_fixed = get_rmsd_from_coords(
            rfdiff_all_fixed_coords,
            af2_all_fixed_coords,
            superimposer.rot,
            superimposer.tran,
        )

    if True in trb_dict["inpaint_seq"]:  # There are non-redesigned residues
        rfdiff_sculpted_coords = [a.coord for a in rfdiff_sculpted_res]
        af2_sculpted_coords = [a.coord for a in af2_sculpted_res]

        rfdiff_sculpted_coords = np.array(rfdiff_sculpted_coords)
        af2_sculpted_coords = np.array(af2_sculpted_coords)

        # superimposer.set(rfdiff_all_coords, af2_all_coords)
        # superimposer.run()

        rmsd_sculpted = get_rmsd_from_coords(
            rfdiff_sculpted_coords,
            af2_sculpted_coords,
            superimposer.rot,
            superimposer.tran,
        )
        if len(rfdiff_motif_res_coords) != 0:
            rmsd_motif = get_rmsd_from_coords(
                rfdiff_motif_res_coords,
                af2_motif_res_coords,
                superimposer.rot,
                superimposer.tran,
            )

    # If we do symmetry, we align af2 model to rfdiffusion structure. Should we control that, or hardcode it?

    if symmetry != None:
        rmsd = homooligomer_rmsd.align_oligomers(
            rfdiff_pdb_path, af2_pdb, save_aligned=False
        )

    return (
        [
            round(rmsd, 1),
            round(rmsd_all_fixed, 1),
            round(rmsd_sculpted, 1),
            round(rmsd_fixed_chains, 1),
            round(rmsd_motif, 1),
        ],
        linker_length,
    )

def sanitize_string(text):
    """Removes null bytes and non-printable control characters."""
    text = text.replace('\x00', '')
    return "".join(c for c in text if c in string.printable)


# The 20 standard amino acids. Boltz maps every OTHER uppercase letter in an
# a3m line through prot_letter_to_token (src/boltz/data/const.py): X/J/B/Z/U/O
# map to the UNK token, anything else (digits, ...) raises a KeyError. To keep
# the generated files clean and always parsable, homolog lines are normalized
# to the 20 AAs + gaps only.
A3M_STD_AA = set("ACDEFGHIKLMNPQRSTVWY")
# Non-20AA letters that Boltz accepts (as UNK). Kept verbatim in the QUERY
# line (it must match the chain sequence of the yaml); normalized to gaps in
# homolog lines.
A3M_UNK_LETTERS = set("XJBUOZ")


def _a3m_aligner():
    """Query-vs-query aligner. Input sequences are gapless, so a
    substitution matrix can be used."""
    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    try:
        from Bio.Align import substitution_matrices
        aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    except ImportError:
        aligner.match_score = 2
        aligner.mismatch_score = -1
    aligner.open_gap_score = -5
    aligner.extend_gap_score = -1
    return aligner


def _a3m_row_aligner():
    """Aligner for projecting one (gap-containing) homolog line onto the
    query. Uses match/mismatch scores because a substitution matrix rejects
    gap characters in its inputs."""
    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = 1
    aligner.mismatch_score = -1
    aligner.open_gap_score = -4
    aligner.extend_gap_score = -1
    return aligner


def _parse_a3m_blocks(path):
    """Parse an a3m file into a list of (header, sequence) blocks.

    Supports both common formats:
      - standard a3m: one '>' header per sequence; a sequence may be wrapped
        over several lines (the lines are joined);
      - single-header a3m: ONE '>' line followed by one independent sequence
        per line (line 1 = query, the rest = homologs). Those lines must NOT
        be joined - joining would merge all homologs into one sequence.

    NUL bytes and other non-printable characters are stripped (a stray NUL
    would crash Boltz's a3m token map).
    """
    raw_lines = []
    with open(path, 'r', errors='replace') as f:
        for line in f:
            line = sanitize_string(line.strip())
            if line:
                raw_lines.append(line)

    n_hdr = sum(1 for l in raw_lines if l.startswith('>'))
    seq_lines = [l for l in raw_lines if not l.startswith('>')]

    if n_hdr == 1 and len(seq_lines) >= 2:
        total = sum(len(l) for l in seq_lines)
        # Heuristic: wrapped lines of ONE sequence contain a dominant line
        # (>= half of all sequence characters); independent MSA lines do not.
        if max(len(l) for l in seq_lines) * 2 < total:
            header = raw_lines[0]
            return [(header, s) for s in seq_lines]

    # standard a3m (also covers the single-header/single-sequence case)
    blocks = []
    header, seq = None, []
    for line in raw_lines:
        if line.startswith('>'):
            if header is not None:
                blocks.append((header, "".join(seq)))
            header, seq = line, []
        else:
            seq.append(line)
    if header is not None:
        blocks.append((header, "".join(seq)))
    return blocks


def _a3m_guide(line):
    """Guide (match-state) characters of an a3m line, normalized for Boltz:
    uppercase 20 AAs kept, gaps kept, every other uppercase letter
    (X/J/B/Z/U/O/digits/...) -> gap, lowercase insertion characters dropped
    (they are not aligned to any query position)."""
    out = []
    for c in sanitize_string(line):
        if c == '-':
            out.append('-')
        elif c.isupper():
            out.append(c if c in A3M_STD_AA else '-')
        # lowercase = insertion relative to the query: dropped
    return "".join(out)


def project_a3m_row_to_query(row, old_query):
    """Project one homolog a3m line onto the old query columns.

    Returns a gap-normalized guide string of EXACTLY len(old_query) characters:
    position i corresponds to old_query[i]. Well-aligned rows (same number of
    guide columns as the query) pass through the character-normalization only;
    ragged rows (different guide-column count - some a3m files contain them)
    are re-aligned against the query first, so the column mapping stays sane.
    """
    guide = _a3m_guide(row)
    if len(guide) == len(old_query):
        return guide
    if not guide or not old_query:
        return '-' * len(old_query)
    aln = _a3m_row_aligner().align(guide, old_query)[0]
    g_str, q_str = str(aln[0]), str(aln[1])
    return "".join(
        g if (g != '-' and q != '-') else '-' for g, q in zip(g_str, q_str) if q != '-'
    )


def recalculate_a3m(input_a3m_path: str, new_query_seq: str, output_a3m_path: str) -> None:
    """
    Recalculates an A3M MSA matrix against a new query sequence.

    The output file is guaranteed to be:
      - rectangular: the query line and EVERY homolog line are exactly
        len(query) characters wide (lowercase insertion states are dropped,
        so the width equals the number of query positions);
      - Boltz-safe: the query line equals the (gapless, uppercase) new query
        so it matches the chain sequence of the generated boltz.yaml, and
        every homolog line contains only the 20 standard AAs and gaps (X and
        other non-standard letters are normalized to gaps; NUL bytes and
        other non-printables are stripped).

    Accepts both standard a3m (one '>' header per sequence) and
    single-header a3m (one '>' line, one sequence per line) input files.

    Args:
        input_a3m_path: Path to the source A3M file.
        new_query_seq: The new unaligned query sequence string.
        output_a3m_path: Path to write the transformed A3M file.
    """
    blocks = _parse_a3m_blocks(input_a3m_path)
    if not blocks:
        raise ValueError(f"No sequences found in A3M file: {input_a3m_path}")

    old_query_seq = _a3m_guide(blocks[0][1]).replace('-', '')
    new_query_clean = sanitize_string(new_query_seq).replace('-', '').upper()

    if not new_query_clean:
        raise ValueError("recalculate_a3m: empty new query sequence")
    bad_query = sorted(set(new_query_clean) - A3M_STD_AA)
    if bad_query:
        print(
            f"WARNING: recalculate_a3m: new query contains non-standard "
            f"residue(s) {bad_query}; they are kept in the query line (it "
            f"must match the chain sequence of the yaml). Boltz maps "
            f"{'/'.join(sorted(A3M_UNK_LETTERS & set(bad_query))) or 'these'} "
            f"to the UNK token - check the source structure/MPNN output."
        )
    if not old_query_seq:
        raise ValueError(f"recalculate_a3m: input a3m query has no residues: {input_a3m_path}")

    # Global pairwise alignment between the new and the old query
    try:
        best_aln = _a3m_aligner().align(new_query_clean, old_query_seq)[0]
    except ValueError:
        # new query carries a letter that is not in the BLOSUM62 alphabet
        # (e.g. X): fall back to the score-based aligner
        best_aln = _a3m_row_aligner().align(new_query_clean, old_query_seq)[0]
    t_str = str(best_aln[0])  # Target: new query path
    q_str = str(best_aln[1])  # Query: old query path

    n_ragged = 0
    with open(output_a3m_path, 'w') as out:
        out.write(f">query\n{new_query_clean}\n")

        for name, seq in blocks[1:]:
            row = project_a3m_row_to_query(seq, old_query_seq)
            if len(row) != len(old_query_seq):
                n_ragged += 1
                row = row[:len(old_query_seq)]  # defensive; keeps the file rectangular

            # Re-map the old-query columns onto the new-query positions.
            # - retained column  -> homolog character at the old position
            # - new-only column  -> gap (the homolog has no residue there)
            # - old-only column  -> dropped (the position is not in the query)
            new_row = []
            old_idx = 0
            for t_char, q_char in zip(t_str, q_str):
                if t_char != '-' and q_char != '-':
                    new_row.append(row[old_idx])
                    old_idx += 1
                elif t_char != '-' and q_char == '-':
                    new_row.append('-')
                else:  # t_char == '-' and q_char != '-': old-only column
                    old_idx += 1

            out.write(f">{name}\n{''.join(new_row)}\n")

    if n_ragged:
        print(
            f"WARNING: recalculate_a3m: {n_ragged} homolog line(s) of "
            f"{input_a3m_path} had a different number of guide columns than "
            f"the query; they were re-aligned and truncated (best effort)."
        )


def make_alignment_file_boltz(sequence_id, sequence, alignment_dir, output_dir):
    """
    Creates A3M alignment files mapping sequence inputs against MSA targets,
    accounting for insertions, deletions, and linkers.
    """
    os.makedirs(output_dir, exist_ok=True)
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    sequence = sanitize_string(sequence)
    split_chains = sequence.split(":")

    alignment_files = []
    if os.path.exists(alignment_dir):
        alignment_files = [f for f in os.listdir(alignment_dir) if f.endswith(".a3m")]

    for i, mpnn_chain_seq in enumerate(split_chains):
        mpnn_chain_seq = "".join(c for c in mpnn_chain_seq if c.isalpha())
        chain_id = letters[i]

        print(f"Processing chain {chain_id} for {sequence_id} with sequence {mpnn_chain_seq}...")

        matching_file = next(
            (f for f in alignment_files if f"Chain_{chain_id}" in f or f"auth_{chain_id}" in f),
            None
        )

        output_path = os.path.join(output_dir, f"{sequence_id}_{chain_id}.a3m")

        if not matching_file:
            print(f"Warning: No alignment file found for chain {chain_id}. Writing base FASTA only.")
            with open(output_path, "w") as f:
                f.write(f">{sequence_id}_{chain_id}\n{mpnn_chain_seq}\n")
            continue

        a3m_path = os.path.join(alignment_dir, matching_file)

        # Execute mapping and gap projection computation
        recalculate_a3m(a3m_path, mpnn_chain_seq, output_path)



def make_alignment_file(cfg, trb_path, pdb_file, mpnn_seq, alignments_path, output):

    skipRfDiff = cfg.get("skipRfDiff", False)
    designable_residues = cfg.get("designable_residues", None)
    chainResidOffset, con_hal_pdb_idx_complete = getChainResidOffsets(
        pdb_file, designable_residues
    )

    if not skipRfDiff:
        with open(trb_path, "rb") as f:
            trb_dict = pickle.load(f)

        if "complex_con_ref_idx0" in trb_dict:
            # residue_data_control_0 = trb_dict['complex_con_ref_idx0']
            residue_data_af2_0 = trb_dict["complex_con_hal_idx0"]
            residue_data_control_1 = trb_dict["complex_con_ref_pdb_idx"]
            # residue_data_af2_1 = trb_dict['complex_con_hal_pdb_idx']
        else:

            partial_diffusion = cfg.get("partial_diffusion", False)
            if not partial_diffusion:
                residue_data_af2_0 = trb_dict["con_hal_idx0"]
                residue_data_control_1 = trb_dict["con_ref_pdb_idx"]
            else:  # When using partial diffusion, RFDiffusion doesn't put anything of the diffused chain on the
                # con_hal_pdb_idx (because nothing is technically fixed). This means that we need to recompile it
                # based on the inpaint_seq
                residue_data_control_1 = []
                residue_data_af2_0 = []

                for id0, value in enumerate(trb_dict["inpaint_seq"]):
                    if value == True:
                        abeceda = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                        for key in abeceda:
                            if key in chainResidOffset:
                                if id0 >= chainResidOffset[key]:
                                    currentResidueChain = key
                        residue_data_control_1.append(
                            (
                                currentResidueChain,
                                id0 + chainResidOffset[currentResidueChain],
                            )
                        )
                        residue_data_af2_0.append(id0)

            # residue_data_control_0 = trb_dict['con_ref_idx0']

            # residue_data_af2_1 = trb_dict['con_hal_pdb_idx']
    else:
        residue_data_af2_0 = [
            x + (chainResidOffset[chain] - 1) for chain, x in con_hal_pdb_idx_complete
        ]
        residue_data_control_1 = con_hal_pdb_idx_complete
        print(residue_data_af2_0)
        print(con_hal_pdb_idx_complete)

    if mpnn_seq[-1:] == "\n":
        mpnn_seq = mpnn_seq[:-1]
    mpnn_sequence_no_colons = mpnn_seq.replace(":", "")

    used_chains = list(set([i[0] for i in residue_data_control_1]))

    mpnn_sequences_list = mpnn_seq.split(":")
    sequences_limits = []

    for seq_num, sequence in enumerate(mpnn_sequences_list):
        seq_start = mpnn_sequence_no_colons.find(sequence)

        if seq_num != len(mpnn_sequences_list) - 1:
            next_sequence = mpnn_sequences_list[seq_num + 1]
            seq_end = mpnn_sequence_no_colons.find(next_sequence)
        else:
            seq_end = len(mpnn_sequence_no_colons)

        sequences_limits.append((seq_start, seq_end))

    with open(output, "w") as f:
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # used for naming chains

        # write the header line
        first_line = "#"
        for seq in mpnn_sequences_list:
            first_line += str(len(seq))
            if seq != mpnn_sequences_list[len(mpnn_sequences_list) - 1]:
                first_line += ","
        first_line += "\t"
        for seq in mpnn_sequences_list:
            first_line += "1"
            if seq != mpnn_sequences_list[len(mpnn_sequences_list) - 1]:
                first_line += ","
        f.write(first_line + "\n")

        # write the whole sequence once
        all_names = ""
        for seq_num, sequence in enumerate(mpnn_sequences_list):
            # all_names+=str(101+seq_num)
            all_names += letters[seq_num]
            if sequence != mpnn_sequences_list[len(mpnn_sequences_list) - 1]:
                all_names += "\t"

        f.write(">" + all_names + "\n")
        f.write(mpnn_sequence_no_colons + "\n")

        # write the sequences to be modelled:
        for seq_num, sequence in enumerate(mpnn_sequences_list):
            f.write(">" + letters[seq_num] + "\n")
            sequence_line = (
                "-" * sequences_limits[seq_num][0]
            )  # Add a gap for each position before the sequence
            sequence_line += sequence  # add sequence
            sequence_line += "-" * (
                (len(mpnn_sequence_no_colons) - sequences_limits[seq_num][1])
            )  # Add a gap for each position after the sequence.
            f.write(sequence_line + "\n")  # write padded sequence

        # now write the aligned sequences
        for chain in letters:
            if chain in used_chains:
                # LEt's get the correct file for this chain
                for file in os.listdir(alignments_path):
                    if "auth_" + chain in file or "Chain_" + chain in file:
                        alignment_file = file
                        print(
                            "Alignment file for chain "
                            + chain
                            + " is "
                            + alignment_file
                        )

                with open(
                    os.path.join(alignments_path, alignment_file), "r"
                ) as chain_alignment_file:
                    for line_id, line in enumerate(chain_alignment_file):
                        if (
                            line_id >= 3
                        ):  # skip first three lines, since they contain the original sequence.
                            if line[0] == ">":
                                f.write(line)
                            else:
                                table = str.maketrans(
                                    "", "", string.ascii_lowercase
                                )  # This deletes lowercase characters from the string
                                line_without_insertions = line.translate(table)

                                new_aligned_seq = "-" * (
                                    len(mpnn_sequence_no_colons)
                                )  # Make a gap sequence of the length of the sequence..
                                trb_chain = [
                                    x
                                    for x in residue_data_control_1
                                    if x[0][0] == chain
                                ]
                                first_residue_in_trb = trb_chain[0][1]
                                for id, pos in enumerate(residue_data_control_1):
                                    if (
                                        pos[0] == chain
                                    ):  # If position chain corresponds to the chain we're looking at

                                        position_to_copy = (
                                            residue_data_control_1[id][1] - 1
                                        )  # minus 1 because this is 1-indexed while the sequence is 0 indexed
                                        new_aligned_seq = (
                                            new_aligned_seq[: residue_data_af2_0[id]]
                                            + line_without_insertions[
                                                position_to_copy
                                                - first_residue_in_trb
                                                + 1
                                            ]
                                            + new_aligned_seq[
                                                residue_data_af2_0[id] + 1 :
                                            ]
                                        )

                                f.write(new_aligned_seq + "\n")

    # delete empty lines that are generated for weird reasons beyond my comprehension. This should be fixed and this section removed, but it doesn't really slow things that much.
    with open(output, "r+") as output_file:
        with open(output + "_tmp", "w") as temp_file:
            for line in output_file:
                if not line.isspace():
                    temp_file.write(line)

    os.remove(output)
    os.rename(output + "_tmp", output)
    # shutil.copyfile(output, output+"_backup") #this is for debug only, to see the file before it goes to AF2


def get_token_value(
    astr, token, regular_expression
):  # "(\d*\.\d+|\d+\.?\d*)" # (-?\d*\.\d+|-?\d+\.?\d*) to allow negative RMSD (-1 = undefined)
    """returns value next to token"""
    import re

    regexp = re.compile(f"{token}{regular_expression}")
    match = regexp.search(astr)
    # if match == None:
    #   match = "/"
    return match.group(1)


def merge_csv(output_dir, output_csv, scores_csv):
    # read csv files
    df1 = pd.read_csv(scores_csv)
    df2 = pd.read_csv(output_csv)

    # merge dataframes on 'model_path' column
    merged_df = pd.merge(df1, df2, on="model_path")
    # drop duplicate 'model_path' column (if it exists)
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

    # save merged dataframe to csv file
    merged_df.to_csv(
        f'{os.path.join(output_dir, "final_output.csv")}',
        index=False,
        float_format="%.1f",
    )

    # Select best ones and copy to another csv. commented out for now
    # if (output_best):
    # best_df=merged_df[(merged_df["RMSD"] <= rmsd_threshold) | (merged_df["plddt"] >= plddt_threshold)] #select best based on thresholds
    # best_df.to_csv(f'{os.path.join(output_dir, "final_output_best.csv")}', index=False)

    # dir_best_pdbs = os.path.join(output_dir, "best_pdbs")
    # os.makedirs(dir_best_pdbs, exist_ok=True) # directory is created even if some or all of the intermediate directories in the path do not exist

    # for file in best_df["model_path"]:
    #    shutil.copy(file,os.path.join(output_dir, "best_pdbs"))


def rename_pdb_create_csv_colabfold(
    cfg,
    output_dir,
    rfdiff_out_dir,
    trb_num,
    model_i,
    control_structure_path,
    symmetry=None,
    model_monomer=False,
):

    # Preparing paths to acces correct files
    model_i = os.path.join(model_i, "")  # add / to path to access json files within

    # dir_renamed_pdb = os.path.join(os.path.dirname(output_dir), "final_pdbs") #Why is this done to the parent folder? It's annoying if running multiple jobs on the same folder
    dir_renamed_pdb = os.path.join(output_dir, "final_pdbs")
    os.makedirs(
        dir_renamed_pdb, exist_ok=True
    )  # directory is created even if some or all of the intermediate directories in the path do not exist

    trb_file = os.path.join(
        rfdiff_out_dir, f"_{trb_num}.trb"
    )  # name of corresponding trb file
    skipRfDiff = cfg.get("skipRfDiff", False)
    designable_residues = cfg.get("designable_residues", None)
    partial_diffusion = cfg.get("partial_diffusion", False)

    if not skipRfDiff:
        with open(trb_file, "rb") as f:
            trb_dict = pickle.load(f)

        if not partial_diffusion:
            if "complex_con_ref_idx0" in trb_dict:
                residue_data_af2 = trb_dict["complex_con_hal_idx0"]
            else:
                residue_data_af2 = trb_dict["con_hal_idx0"]
        else:  # When using partial diffusion, RFDiffusion doesn't put anything of the diffused chain on the
            # con_hal_pdb_idx (because nothing is technically fixed). This means that we need to recompile it
            # based on the inpaint_seq
            residue_data_af2 = []
            abeceda = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            chainResidOffset, con_hal_pdb_idx_complete = getChainResidOffsets(
                control_structure_path, designable_residues
            )
            for id0, value in enumerate(trb_dict["inpaint_seq"]):

                if value == True:

                    for key in abeceda:
                        if key in chainResidOffset:

                            if id0 >= chainResidOffset[key]:
                                currentResidueChain = key
                    residue_data_af2.append(id0)
                    # residue_data_af2.append((currentResidueChain,id0+chainResidOffset[currentResidueChain]))

    else:
        chainResidOffset, con_hal_pdb_idx_complete = getChainResidOffsets(
            control_structure_path, designable_residues
        )
        residue_data_af2 = [
            x + (chainResidOffset[chain] - 1) for chain, x in con_hal_pdb_idx_complete
        ]

    json_files = glob.glob(os.path.join(model_i, "T*000.json"))
    rfdiff_pdb_path = os.path.join(rfdiff_out_dir, f"_{trb_num}.pdb")

    for (
        json_file
    ) in (
        json_files
    ):  # in af2 model_i directory for T_...json file in [all T_...json files]
        # This is done for each model_i directory therefore for each rfdiff pdb
        # There are 5 T_...jsons per 1 mpnn seq of the rfdiff model

        with open(json_file, "r") as f:
            params = json.load(f)

        # Handle filenames correctly to get the T_...pdb that corresponds to the T_...json
        json_filename = os.path.basename(json_file)
        json_dirname = os.path.dirname(json_file)
        json_newname = os.path.join(
            json_dirname, json_filename.replace("scores", "unrelaxed")
        )
        model_pdb_file = (
            os.path.splitext(json_newname)[0] + ".pdb"
        )  # T_0.1__sample_2__score_0.5830__global_score_0.8339__seq_recovery_02000_unrelaxed_rank_004_alphafold2_multimer_v3_model_3_seed_000.pdb

        # Extract relevant data. Files used: json file of specific af2 model, specific af2 pdb,  trb file of rfdiff model (1 for all AF2 models from same rfdiff pdb)
        plddt_list = params["plddt"]
        plddt = int(np.mean(plddt_list))

        try:
            plddt_sculpted_list = [
                plddt_list[i]
                for i in range(0, len(plddt_list))
                if i not in residue_data_af2
            ]

            plddt_sculpted = int(np.mean(plddt_sculpted_list))
        except NameError:
            plddt_sculpted = -1

        rmsd_list, linker_length = calculate_RMSD_linker_len(
            cfg,
            trb_file,
            model_pdb_file,
            control_structure_path,
            rfdiff_pdb_path,
            symmetry,
            model_monomer,
        )
        pae = round((np.mean(params["pae"])), 2)

        # if we are doing symmetry or monomer modelling we also want to add monomer rmsd to the output
        if symmetry:
            monomers_dirname = os.path.join(model_i, "monomers")
            basename = os.path.basename(model_pdb_file)

            # prefix up to seq_recovery_XXXXX
            prefix = re.search(r"^(T_.*?seq_recovery_\d+)", basename).group(1)

            # extract model number
            model_number = re.search(r"_model_(\d+)", basename).group(1)

            # glob pattern: ignore rank completely
            pattern = os.path.join(
                monomers_dirname, f"monomer_{prefix}*model_{model_number}_*.pdb"
            )
            matches = glob.glob(pattern)
            if not matches:
                raise FileNotFoundError(f"No matching monomer found for {basename}")
            monomer_pdb_file = matches[0]  # take the first match
            monomer_rmsd = homooligomer_rmsd.align_monomer(
                rfdiff_pdb_path, monomer_pdb_file, save_aligned=False
            )
            monomer_params_json = os.path.join(
                monomers_dirname, "monomer_" + os.path.basename(json_file)
            )

            pattern = os.path.join(
                monomers_dirname, f"monomer_{prefix}*model_{model_number}_*.json"
            )
            matches = glob.glob(pattern)
            if not matches:
                raise FileNotFoundError(f"No matching monomer found for {basename}")
            monomer_params_json = matches[0]  # take the first match
            with open(monomer_params_json, "r") as f:
                monomer_params = json.load(f)

            monomer_plddt_list = monomer_params["plddt"]
            monomer_plddt = int(np.mean(monomer_plddt_list))

        if model_monomer:
            monomers_dirname = os.path.join(model_i, "monomers")
            basename = os.path.basename(model_pdb_file)

            # prefix up to seq_recovery_XXXXX
            prefix = re.search(r"^(T_.*?seq_recovery_\d+)", basename).group(1)

            # extract model number
            model_number = re.search(r"_model_(\d+)", basename).group(1)

            # glob pattern: ignore rank completely
            pattern = os.path.join(
                monomers_dirname, f"monomer_{prefix}*model_{model_number}_*.pdb"
            )

            matches = glob.glob(pattern)

            if not matches:
                raise FileNotFoundError(f"No matching monomer found for {basename}")

            monomer_pdb_file = matches[0]  # take the first match
            parser = PDBParser(PERMISSIVE=1)

            structure_target = parser.get_structure("target", rfdiff_pdb_path)
            structure_mobile = parser.get_structure("mobile", monomer_pdb_file)

            target_chain = list(structure_target.get_chains())[0]
            mobile_chain_res = list(structure_mobile.get_residues())
            mobile_chain_res = [ind["CA"] for ind in mobile_chain_res]
            list_rmsd_chains = []

            target_chain_res = list(target_chain.get_residues())
            target_chain_res = [ind["CA"] for ind in target_chain_res]

            superimposer = Superimposer()
            superimposer.set_atoms(target_chain_res, mobile_chain_res)
            superimposer.apply(structure_mobile.get_atoms())
            list_rmsd_chains.append(superimposer.rms)
            monomer_rmsd = np.min(list_rmsd_chains)
            pattern = os.path.join(
                monomers_dirname, f"monomer_{prefix}*model_{model_number}_*.json"
            )
            matches = glob.glob(pattern)
            if not matches:
                raise FileNotFoundError(f"No matching monomer found for {basename}")
            monomer_params_json = matches[0]  # take the first match
            with open(monomer_params_json, "r") as f:
                monomer_params = json.load(f)

            monomer_plddt_list = monomer_params["plddt"]
            monomer_plddt = int(np.mean(monomer_plddt_list))

        # tracebility
        output_num = os.path.basename(output_dir)
        af2_model = get_token_value(
            json_filename, "_model_", "(\\d*\\.\\d+|\\d+\\.?\\d*)"
        )
        mpnn_sample = get_token_value(
            json_filename, "_sample_", "(\\d*\\.\\d+|\\d+\\.?\\d*)"
        )
        task_id = os.environ.get("SLURM_ARRAY_TASK_ID", 1)

        # Create a new name an copy te af2 model under that name into the output directory
        new_pdb_file = f"{task_id}.{trb_num}.{mpnn_sample}.{af2_model}__link_{linker_length}__plddt_{plddt}__plddt_sculpted_{plddt_sculpted}__rmsd_{rmsd_list[0]:.1f}__rmsd_sculpted_{rmsd_list[2]:.1f}__rmsd_fixedchains_{rmsd_list[3]:.1f}__rmsd_motif_{rmsd_list[4]:.1f}__pae_{pae}__out_{output_num}_.pdb"
        # out -> 00 -> number of task
        # rf -> 01 -> number of corresponding rf difff model
        # af_model -> 4 -> number of the af model (1-5), can be set using --model_order flag
        new_pdb_path = os.path.join(dir_renamed_pdb, new_pdb_file)

        try:
            shutil.copy2(model_pdb_file, new_pdb_path)
        except OSError as e:
            print(f"Error copying {model_pdb_file} to {new_pdb_file}: {e}")

        p = PDBParser()

        structure = p.get_structure("model_seq", new_pdb_path)

        ppb = PPBuilder()

        seq = ""
        for pp in ppb.build_peptides(structure):
            seq += f":{pp.get_sequence().__str__()}"

        print("new_pdb_file", new_pdb_file)
        dictionary = {
            "id": f"{task_id}.{trb_num}.{mpnn_sample}.{af2_model}",
            "link_lenght": (linker_length),
            "plddt": (plddt),
            "plddt_sculpted": (plddt_sculpted),
            "RMSD": f"{rmsd_list[0]:.1f}",
            #'Rmsd_all_fixed': get_token_value(new_pdb_file, '__rmsd_all_fixed_', "(-?\\d*\\.\\d+|-?\\d+\\.?\\d*)"),
            "RMSD_sculpted": f"{rmsd_list[2]:.1f}",
            "RMSD_fixed_chains": f"{rmsd_list[3]:.1f}",
            "RMSD_motif": f"{rmsd_list[4]:.1f}",
            "pae": (pae),
            "model_path": new_pdb_path,
            "sequence": seq[1:],
            "af2_json": json_file,
            "af2_pdb": model_pdb_file,
            "path_rfdiff": rfdiff_pdb_path,
        }  # MODEL PATH for scoring_rg_... #jsonfilename for traceability

        if symmetry or model_monomer:
            dictionary["monomer_rmsd"] = monomer_rmsd
            dictionary["monomer_plddt"] = monomer_plddt

        df = pd.json_normalize(dictionary)
        path_csv = os.path.join(output_dir, "output.csv")
        df.to_csv(
            path_csv,
            mode="a",
            header=not os.path.exists(path_csv),
            index=False,
            float_format="%.1f",
        )


def monomer_prediction_dirs(model_i, model_name):
    """Candidate directories holding the monomer prediction for one model
    (symmetry / monomer scoring). The first EXISTING directory is used.

    - Boltz2 flow: the monomer yaml is written into the same yaml_dir as the
      main yaml, so its prediction lands in the same predictions tree (no
      'monomers/' prefix).
    - AF3/ColabFold flow: monomer predictions are written under
      {model_i}/monomers/...
    """
    return [
        os.path.join(
            model_i,
            "boltz_results_yaml_inputs",
            "predictions",
            "monomer_" + model_name,
        ),
        os.path.join(
            model_i,
            "monomers",
            "boltz_results_yaml_inputs",
            "predictions",
            "monomer_" + model_name,
        ),
    ]


def rename_pdb_create_csv_boltz(
    cfg,
    output_dir,
    rfdiff_out_dir,
    trb_num,
    model_i,
    control_structure_path,
    symmetry=None,
    model_monomer=False,
):

    # Preparing paths to acces correct files
    model_i = os.path.join(model_i, "")  # add / to path to access json files within

    # dir_renamed_pdb = os.path.join(os.path.dirname(output_dir), "final_pdbs") #Why is this done to the parent folder? It's annoying if running multiple jobs on the same folder
    dir_renamed_pdb = os.path.join(output_dir, "final_pdbs")
    os.makedirs(
        dir_renamed_pdb, exist_ok=True
    )  # directory is created even if some or all of the intermediate directories in the path do not exist

    trb_file = os.path.join(
        rfdiff_out_dir, f"_{trb_num}.trb"
    )  # name of corresponding trb file
    skipRfDiff = cfg.get("skipRfDiff", False)
    designable_residues = cfg.get("designable_residues", None)
    partial_diffusion = cfg.get("partial_diffusion", False)

    if not skipRfDiff:
        with open(trb_file, "rb") as f:
            trb_dict = pickle.load(f)

        if not partial_diffusion:
            if "complex_con_ref_idx0" in trb_dict:
                residue_data_af2 = trb_dict["complex_con_hal_idx0"]
            else:
                residue_data_af2 = trb_dict["con_hal_idx0"]
        else:  # When using partial diffusion, RFDiffusion doesn't put anything of the diffused chain on the
            # con_hal_pdb_idx (because nothing is technically fixed). This means that we need to recompile it
            # based on the inpaint_seq
            residue_data_af2 = []
            abeceda = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            chainResidOffset, con_hal_pdb_idx_complete = getChainResidOffsets(
                control_structure_path, designable_residues
            )
            for id0, value in enumerate(trb_dict["inpaint_seq"]):

                if value == True:

                    for key in abeceda:
                        if key in chainResidOffset:

                            if id0 >= chainResidOffset[key]:
                                currentResidueChain = key
                    residue_data_af2.append(id0)
                    # residue_data_af2.append((currentResidueChain,id0+chainResidOffset[currentResidueChain]))

    else:
        chainResidOffset, con_hal_pdb_idx_complete = getChainResidOffsets(
            control_structure_path, designable_residues
        )
        residue_data_af2 = [
            x + (chainResidOffset[chain] - 1) for chain, x in con_hal_pdb_idx_complete
        ]

    individual_directories = glob.glob(
        os.path.join(model_i, "boltz_results_yaml_inputs", "predictions", "*")
    )
    rfdiff_pdb_path = os.path.join(rfdiff_out_dir, f"_{trb_num}.pdb")

    # boltz_full_complex mode: the sidecar map (design flat position ->
    # protein token in the full spliced Boltz chains) is written next to the
    # boltz yaml by make_boltz_input_yaml. Used to score only the designed
    # region of the (much longer) full-complex prediction.
    fc_files = (
        sorted(
            glob.glob(os.path.join(model_i, "yaml_inputs", "*.full_complex.json"))
        )
        if not skipRfDiff
        else []
    )

    def _load_full_complex_map(pred_name):
        if not fc_files:
            return None
        if len(fc_files) == 1:
            path = fc_files[0]
        else:  # several sequences per model dir: match by prediction name
            path = os.path.join(
                model_i, "yaml_inputs", f"{pred_name}.full_complex.json"
            )
            if not os.path.exists(path):
                path = fc_files[0]
        with open(path) as f:
            fc = json.load(f)
        print(
            f"boltz_full_complex: scoring the designed region using the "
            f"full-complex token map from {path}"
        )
        return fc

    for (
        directory
    ) in individual_directories:  # for each directory in the predictions folder
        dir_path = Path(directory)

        model_name = dir_path.parent.name if not dir_path.is_dir() else dir_path.name

        fc = _load_full_complex_map(model_name)
        flat_to_token = (
            {int(k): v for k, v in fc["flat_to_token"].items()} if fc else None
        )
        fc_new_tokens = fc.get("new_tokens", []) if fc else []

        model_pdb_files = glob.glob(os.path.join(directory, "*.pdb"))

        for model_pdb_file in model_pdb_files:
            basename = os.path.basename(model_pdb_file)
            match = re.search(r"_model_(\d+)", basename)
            if match:
                model_number = match.group(1)
            else:
                raise ValueError(f"Could not find model number in {basename}")
            pdb_stem = Path(model_pdb_file).stem

            json_file = glob.glob(
                os.path.join(directory, f"confidence_{pdb_stem}.json")
            )[0]
            pae_file = glob.glob(os.path.join(directory, f"pae_{pdb_stem}.npz"))[0]
            pde_file = glob.glob(os.path.join(directory, f"pde_{pdb_stem}.npz"))[0]
            plddt_file = glob.glob(os.path.join(directory, f"plddt_{pdb_stem}.npz"))[0]
            # model_pdb_file = glob.glob(os.path.join(directory, "*.pdb"))[0]

            with open(json_file, "r") as f:
                params = json.load(f)
            plddt_list = np.load(plddt_file)["plddt"].tolist()
            plddt = int(np.mean(plddt_list) * 100)
            pae_list = np.load(pae_file)["pae"].tolist()
            pae = np.mean(pae_list)
            pde_list = np.load(pde_file)["pde"].tolist()
            pde = np.mean(pde_list)

            # print(f"DEBUG:residue_data_af2 {residue_data_af2}")
            try:
                if flat_to_token and fc_new_tokens:
                    # boltz_full_complex: plDDT of the NEW (sculpted) design
                    # tokens only - the native spliced-in residues are not
                    # part of the design and must not enter this metric.
                    plddt_sculpted_list = [
                        plddt_list[i]
                        for i in fc_new_tokens
                        if i < len(plddt_list)
                    ]
                    plddt_sculpted = (
                        int(np.mean(plddt_sculpted_list) * 100)
                        if plddt_sculpted_list
                        else -1
                    )
                else:
                    plddt_sculpted_list = [
                        plddt_list[i]
                        for i in range(0, len(plddt_list))
                        if i not in residue_data_af2
                    ]

                    plddt_sculpted = int(np.mean(plddt_sculpted_list) * 100)
            except NameError:
                plddt_sculpted = -1

            rmsd_list, linker_length = calculate_RMSD_linker_len(
                cfg,
                trb_file,
                model_pdb_file,
                control_structure_path,
                rfdiff_pdb_path,
                symmetry,
                model_monomer,
                flat_to_token=flat_to_token,
            )

            # if we are doing symmetry or monomer modelling we also want to add monomer rmsd to the output
            if symmetry:
                monomer_rmsd = None
                monomer_plddt = None
                # AF3/ColabFold put monomer predictions under
                # {model_i}/monomers/...; in the Boltz2 flow the monomer yaml
                # is written into the same yaml_dir, so its prediction lands
                # in the same predictions tree (no 'monomers/' prefix). Try
                # both layouts.
                monomer_candidates = monomer_prediction_dirs(
                    model_i, model_name
                )
                current_monomer_dirname = next(
                    (d for d in monomer_candidates if os.path.isdir(d)),
                    monomer_candidates[0],
                )
                try:
                    monomer_pdb_file = os.path.join(
                        current_monomer_dirname,
                        "monomer_" + os.path.basename(model_pdb_file),
                    )
                    monomer_rmsd = homooligomer_rmsd.align_monomer(
                        rfdiff_pdb_path, monomer_pdb_file, save_aligned=False
                    )
                    monomer_json_file = glob.glob(
                        os.path.join(current_monomer_dirname, "*.json")
                    )[0]
                    monomer_pae_file = glob.glob(
                        os.path.join(current_monomer_dirname, f"pae_monomer_{pdb_stem}.npz")
                    )[0]
                    monomer_pde_file = glob.glob(
                        os.path.join(current_monomer_dirname, f"pde_monomer_{pdb_stem}.npz")
                    )[0]
                    monomer_plddt_file = glob.glob(
                        os.path.join(
                            current_monomer_dirname, f"plddt_monomer_{pdb_stem}.npz"
                        )
                    )[0]
                    with open(monomer_json_file, "r") as f:
                        monomer_params = json.load(f)
                    monomer_plddt_list = np.load(monomer_plddt_file)["plddt"].tolist()
                    monomer_plddt = int(np.mean(monomer_plddt_list) * 100)
                    monomer_pae_list = np.load(monomer_pae_file)["pae"].tolist()
                    monomer_pae = np.mean(monomer_pae_list)
                    monomer_pde_list = np.load(monomer_pde_file)["pde"].tolist()
                    monomer_pde = np.mean(monomer_pde_list)
                except Exception as e:
                    print(
                        f"WARNING: monomer scoring for {model_name} skipped "
                        f"(missing monomer prediction files?): {e}"
                    )
                    monomer_rmsd = None
                    monomer_plddt = None

            if model_monomer:
                monomers_dirname = os.path.join(model_i, "monomers")
                current_monomer_dirname = os.path.join(
                    monomers_dirname,
                    "boltz_results_yaml_inputs",
                    "predictions",
                    "monomer_" + model_name,
                )
                monomer_pdb_file = os.path.join(
                    current_monomer_dirname,
                    "monomer_" + os.path.basename(model_pdb_file),
                )

                parser = PDBParser(PERMISSIVE=1)

                structure_target = parser.get_structure("target", rfdiff_pdb_path)
                structure_mobile = parser.get_structure("mobile", monomer_pdb_file)

                target_chain = list(structure_target.get_chains())[0]
                mobile_chain_res = list(structure_mobile.get_residues())
                mobile_chain_res = [ind["CA"] for ind in mobile_chain_res]
                list_rmsd_chains = []

                target_chain_res = list(target_chain.get_residues())
                target_chain_res = [ind["CA"] for ind in target_chain_res]

                superimposer = Superimposer()
                superimposer.set_atoms(target_chain_res, mobile_chain_res)
                superimposer.apply(structure_mobile.get_atoms())
                list_rmsd_chains.append(superimposer.rms)
                monomer_rmsd = np.min(list_rmsd_chains)
                monomer_json_file = glob.glob(
                    os.path.join(current_monomer_dirname, "*.json")
                )[0]
                monomer_pae_file = glob.glob(
                    os.path.join(current_monomer_dirname, "pae*.npz")
                )[0]
                monomer_pde_file = glob.glob(
                    os.path.join(current_monomer_dirname, "pde*.npz")
                )[0]
                monomer_plddt_file = glob.glob(
                    os.path.join(current_monomer_dirname, "plddt*.npz")
                )[0]
                with open(monomer_json_file, "r") as f:
                    monomer_params = json.load(f)
                monomer_plddt_list = np.load(monomer_plddt_file)["plddt"].tolist()
                monomer_plddt = int(np.mean(monomer_plddt_list) * 100)
                monomer_pae_list = np.load(monomer_pae_file)["pae"].tolist()
                monomer_pae = np.mean(monomer_pae_list)
                monomer_pde_list = np.load(monomer_pde_file)["pde"].tolist()
                monomer_pde = np.mean(monomer_pde_list)

            # tracebility
            output_num = os.path.basename(output_dir)
            boltz_model = model_number  # Boltz only runs one model, so we can just set this to 1. If we wanted to run multiple AF2 models per RFDiffusion model, we would need to change the Boltz code to output the model number in the json filename and then extract it here like we do for the regular ColabFold runs.
            mpnn_sample = get_token_value(
                model_name, "_sample_", "(\\d*\\.\\d+|\\d+\\.?\\d*)"
            )
            task_id = os.environ.get("SLURM_ARRAY_TASK_ID", 1)

            # Create a new name an copy te af2 model under that name into the output directory
            new_pdb_file = f"{task_id}.{trb_num}.{mpnn_sample}.{boltz_model}__link_{linker_length}__plddt_{plddt}__plddt_sculpted_{plddt_sculpted}__rmsd_{rmsd_list[0]:.1f}__rmsd_sculpted_{rmsd_list[2]:.1f}__rmsd_fixedchains_{rmsd_list[3]:.1f}__rmsd_motif_{rmsd_list[4]:.1f}__pae_{pae}__out_{output_num}_.pdb"
            # out -> 00 -> number of task
            # rf -> 01 -> number of corresponding rf difff model
            # af_model -> 4 -> number of the af model (1-5), can be set using --model_order flag
            new_pdb_path = os.path.join(dir_renamed_pdb, new_pdb_file)

            try:
                shutil.copy2(model_pdb_file, new_pdb_path)
            except OSError as e:
                print(f"Error copying {model_pdb_file} to {new_pdb_file}: {e}")

            p = PDBParser()

            structure = p.get_structure("model_seq", new_pdb_path)

            ppb = PPBuilder()

            seq = ""
            for pp in ppb.build_peptides(structure):
                seq += f":{pp.get_sequence().__str__()}"

            print("new_pdb_file", new_pdb_file)
            dictionary = {
                "id": f"{task_id}.{trb_num}.{mpnn_sample}.{boltz_model}",
                "link_lenght": (linker_length),
                "plddt": (plddt),
                "plddt_sculpted": (plddt_sculpted),
                "RMSD": f"{rmsd_list[0]:.1f}",
                #'Rmsd_all_fixed': get_token_value(new_pdb_file, '__rmsd_all_fixed_', "(-?\\d*\\.\\d+|-?\\d+\\.?\\d*)"),
                "RMSD_sculpted": f"{rmsd_list[2]:.1f}",
                "RMSD_fixed_chains": f"{rmsd_list[3]:.1f}",
                "RMSD_motif": f"{rmsd_list[4]:.1f}",
                "pae": (pae),
                "model_path": new_pdb_path,
                "sequence": seq[1:],
                "af2_json": json_file,
                "af2_pdb": model_pdb_file,
                "path_rfdiff": rfdiff_pdb_path,
            }  # MODEL PATH for scoring_rg_... #jsonfilename for traceability
            dictionary.update(params)
            if (symmetry or model_monomer) and monomer_rmsd is not None:
                dictionary["monomer_rmsd"] = monomer_rmsd
                dictionary["monomer_plddt"] = monomer_plddt

            df = pd.json_normalize(dictionary)
            path_csv = os.path.join(output_dir, "output.csv")
            df.to_csv(
                path_csv,
                mode="a",
                header=not os.path.exists(path_csv),
                index=False,
                float_format="%.1f",
            )


def rename_pdb_create_csv_AF3(
    cfg,
    output_dir,
    rfdiff_out_dir,
    trb_num,
    model_i,
    control_structure_path,
    symmetry=None,
    model_monomer=False,
):

    # Preparing paths to acces correct files
    model_i = os.path.join(model_i, "")  # add / to path to access json files within

    # dir_renamed_pdb = os.path.join(os.path.dirname(output_dir), "final_pdbs") #Why is this done to the parent folder? It's annoying if running multiple jobs on the same folder
    dir_renamed_pdb = os.path.join(output_dir, "final_pdbs")
    os.makedirs(
        dir_renamed_pdb, exist_ok=True
    )  # directory is created even if some or all of the intermediate directories in the path do not exist

    trb_file = os.path.join(
        rfdiff_out_dir, f"_{trb_num}.trb"
    )  # name of corresponding trb file
    skipRfDiff = cfg.get("skipRfDiff", False)
    designable_residues = cfg.get("designable_residues", None)
    partial_diffusion = cfg.get("partial_diffusion", False)

    if not skipRfDiff:
        with open(trb_file, "rb") as f:
            trb_dict = pickle.load(f)

        if not partial_diffusion:
            if "complex_con_ref_idx0" in trb_dict:
                residue_data_af2 = trb_dict["complex_con_hal_idx0"]
            else:
                residue_data_af2 = trb_dict["con_hal_idx0"]
        else:  # When using partial diffusion, RFDiffusion doesn't put anything of the diffused chain on the
            # con_hal_pdb_idx (because nothing is technically fixed). This means that we need to recompile it
            # based on the inpaint_seq
            residue_data_af2 = []
            abeceda = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            chainResidOffset, con_hal_pdb_idx_complete = getChainResidOffsets(
                control_structure_path, designable_residues
            )
            for id0, value in enumerate(trb_dict["inpaint_seq"]):

                if value == True:

                    for key in abeceda:
                        if key in chainResidOffset:

                            if id0 >= chainResidOffset[key]:
                                currentResidueChain = key
                    residue_data_af2.append(id0)
                    # residue_data_af2.append((currentResidueChain,id0+chainResidOffset[currentResidueChain]))

    else:
        chainResidOffset, con_hal_pdb_idx_complete = getChainResidOffsets(
            control_structure_path, designable_residues
        )
        residue_data_af2 = [
            x + (chainResidOffset[chain] - 1) for chain, x in con_hal_pdb_idx_complete
        ]

    individual_directories = glob.glob(os.path.join(model_i, "T_*"))
    rfdiff_pdb_path = os.path.join(rfdiff_out_dir, f"_{trb_num}.pdb")

    for (
        directory
    ) in individual_directories:  # for each directory in the predictions folder
        dir_path = Path(directory)
        model_name = dir_path.parent.name if not dir_path.is_dir() else dir_path.name

        summary_confidences_file = glob.glob(
            os.path.join(directory, "*summary_confidences.json")
        )[0]

        confidences_files = glob.glob(os.path.join(directory, "*confidences.json"))

        confidences_file = [
            f for f in confidences_files if not f.endswith("summary_confidences.json")
        ][0]

        with open(summary_confidences_file, "r") as f:
            summary = json.load(f)

        with open(confidences_file, "r") as f:
            confidences = json.load(f)

        plddt_list = confidences["atom_plddts"]
        plddt = int(np.mean(plddt_list))
        pae_list = confidences["pae"]
        pae = np.mean(pae_list)

        model_cif_file_path = glob.glob(os.path.join(directory, "*.cif"))[0]
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("structure", model_cif_file_path)
        model_pdb_file = Path(model_cif_file_path).with_suffix(".pdb")

        io = PDBIO()
        io.set_structure(structure)
        io.save(str(model_pdb_file))

        # print(f"DEBUG:residue_data_af2 {residue_data_af2}")
        try:
            plddt_sculpted_list = [
                plddt_list[i]
                for i in range(0, len(plddt_list))
                if i not in residue_data_af2
            ]

            plddt_sculpted = int(np.mean(plddt_sculpted_list))
        except NameError:
            plddt_sculpted = -1

        rmsd_list, linker_length = calculate_RMSD_linker_len(
            cfg,
            trb_file,
            model_pdb_file,
            control_structure_path,
            rfdiff_pdb_path,
            symmetry,
            model_monomer,
        )

        # if we are doing symmetry or monomer modelling we also want to add monomer rmsd to the output
        if symmetry:
            monomers_dirname = os.path.join(model_i, "monomers")
            monomer_cif_file = os.path.join(
                monomers_dirname,
                "monomer_" + model_name,
                "monomer_" + os.path.basename(model_cif_file_path),
            )
            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure("structure", monomer_cif_file)
            monomer_pdb_file = Path(monomer_cif_file).with_suffix(".pdb")
            io = PDBIO()
            io.set_structure(structure)
            io.save(str(monomer_pdb_file))

            monomer_rmsd = homooligomer_rmsd.align_monomer(
                rfdiff_pdb_path, monomer_pdb_file, save_aligned=False
            )
            monomer_summary_confidences_file = glob.glob(
                os.path.join(
                    monomers_dirname,
                    "monomer_" + model_name,
                    "*summary_confidences.json",
                )
            )[0]

            monomer_confidences_files = glob.glob(
                os.path.join(
                    monomers_dirname, "monomer_" + model_name, "*confidences.json"
                )
            )

            monomer_confidences_file = [
                f
                for f in monomer_confidences_files
                if not f.endswith("summary_confidences.json")
            ][0]

            with open(monomer_summary_confidences_file, "r") as f:
                summary = json.load(f)

            with open(monomer_confidences_file, "r") as f:
                monomer_confidences = json.load(f)

            monomer_plddt_list = monomer_confidences["atom_plddts"]
            monomer_plddt = int(np.mean(monomer_plddt_list))
            monomer_pae_list = confidences["pae"]
            monomer_pae = np.mean(monomer_pae_list)

        if model_monomer:
            monomers_dirname = os.path.join(model_i, "monomers")
            monomer_cif_file = os.path.join(
                monomers_dirname,
                "monomer_" + model_name,
                "monomer_" + os.path.basename(model_cif_file_path),
            )

            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure("structure", monomer_cif_file)
            monomer_pdb_file = Path(monomer_cif_file).with_suffix(".pdb")
            io = PDBIO()
            io.set_structure(structure)
            io.save(str(monomer_pdb_file))

            parser = PDBParser(PERMISSIVE=1)

            structure_target = parser.get_structure("target", rfdiff_pdb_path)
            structure_mobile = parser.get_structure("mobile", monomer_pdb_file)

            target_chain = list(structure_target.get_chains())[0]
            mobile_chain_res = list(structure_mobile.get_residues())
            mobile_chain_res = [ind["CA"] for ind in mobile_chain_res]
            list_rmsd_chains = []

            target_chain_res = list(target_chain.get_residues())
            target_chain_res = [ind["CA"] for ind in target_chain_res]

            superimposer = Superimposer()
            superimposer.set_atoms(target_chain_res, mobile_chain_res)
            superimposer.apply(structure_mobile.get_atoms())
            list_rmsd_chains.append(superimposer.rms)
            monomer_rmsd = np.min(list_rmsd_chains)
            monomer_summary_confidences_file = glob.glob(
                os.path.join(
                    monomers_dirname,
                    "monomer_" + model_name,
                    "*summary_confidences.json",
                )
            )[0]
            print(
                f"DEBUG: files in monomer folder {glob.glob(os.path.join(monomers_dirname, "monomer_" + model_name, "*"))}"
            )
            monomer_confidences_files = glob.glob(
                os.path.join(
                    monomers_dirname, "monomer_" + model_name, "*confidences.json"
                )
            )

            monomer_confidences_file = [
                f
                for f in monomer_confidences_files
                if not f.endswith("summary_confidences.json")
            ][0]

            with open(monomer_summary_confidences_file, "r") as f:
                summary = json.load(f)

            with open(monomer_confidences_file, "r") as f:
                monomer_confidences = json.load(f)

            monomer_plddt_list = monomer_confidences["atom_plddts"]
            monomer_plddt = int(np.mean(monomer_plddt_list))
            monomer_pae_list = confidences["pae"]
            monomer_pae = np.mean(monomer_pae_list)

        # tracebility
        output_num = os.path.basename(output_dir)
        af2_model = "0"  # Right now we're just taking the final model from AF3 instead of the individual ones
        mpnn_sample = get_token_value(
            model_name, "_sample_", "(\\d*\\.\\d+|\\d+\\.?\\d*)"
        )
        task_id = os.environ.get("SLURM_ARRAY_TASK_ID", 1)

        # Create a new name an copy te af2 model under that name into the output directory
        new_pdb_file = f"{task_id}.{trb_num}.{mpnn_sample}.{af2_model}__link_{linker_length}__plddt_{plddt}__plddt_sculpted_{plddt_sculpted}__rmsd_{rmsd_list[0]:.1f}__rmsd_sculpted_{rmsd_list[2]:.1f}__rmsd_fixedchains_{rmsd_list[3]:.1f}__rmsd_motif_{rmsd_list[4]:.1f}__pae_{pae}__out_{output_num}_.pdb"
        # out -> 00 -> number of task
        # rf -> 01 -> number of corresponding rf difff model
        # af_model -> 4 -> number of the af model (1-5), can be set using --model_order flag
        new_pdb_path = os.path.join(dir_renamed_pdb, new_pdb_file)

        try:
            shutil.copy2(model_pdb_file, new_pdb_path)
        except OSError as e:
            print(f"Error copying {model_pdb_file} to {new_pdb_file}: {e}")

        p = PDBParser()

        structure = p.get_structure("model_seq", new_pdb_path)

        ppb = PPBuilder()

        seq = ""
        for pp in ppb.build_peptides(structure):
            seq += f":{pp.get_sequence().__str__()}"

        print("new_pdb_file", new_pdb_file)
        dictionary = {
            "id": f"{task_id}.{trb_num}.{mpnn_sample}.{af2_model}",
            "link_lenght": (linker_length),
            "plddt": (plddt),
            "plddt_sculpted": (plddt_sculpted),
            "RMSD": f"{rmsd_list[0]:.1f}",
            #'Rmsd_all_fixed': get_token_value(new_pdb_file, '__rmsd_all_fixed_', "(-?\\d*\\.\\d+|-?\\d+\\.?\\d*)"),
            "RMSD_sculpted": f"{rmsd_list[2]:.1f}",
            "RMSD_fixed_chains": f"{rmsd_list[3]:.1f}",
            "RMSD_motif": f"{rmsd_list[4]:.1f}",
            "pae": (pae),
            "model_path": new_pdb_path,
            "sequence": seq[1:],
            "af3_json": summary_confidences_file,
            "af3_pdb": model_pdb_file,
            "path_rfdiff": rfdiff_pdb_path,
        }  # MODEL PATH for scoring_rg_... #jsonfilename for traceability
        dictionary.update(summary)
        if symmetry or model_monomer:
            dictionary["monomer_rmsd"] = monomer_rmsd
            dictionary["monomer_plddt"] = monomer_plddt

        df = pd.json_normalize(dictionary)
        path_csv = os.path.join(output_dir, "output.csv")
        df.to_csv(
            path_csv,
            mode="a",
            header=not os.path.exists(path_csv),
            index=False,
            float_format="%.1f",
        )


# vsaka tabelca za svoj task
# funkcija na koncu, ki vse združi
# sestavljanje pathov je ok, dodaj nov column s pathom do rfdif


def create_dataframe(path_to_files, output_dir):  # path = r'content/*partial.pdb'
    # takes path  to renamed pdbs
    all_files = glob.glob(os.path.join(path_to_files, "*.pdb"))
    list_of_dicts = []

    for file_name in all_files:
        p = PDBParser()
        structure = p.get_structure("model_seq", file_name)
        ppb = PPBuilder()

        seq = ""
        for pp in ppb.build_peptides(structure):
            seq += f":{pp.get_sequence().__str__()}"
        print(seq)
        dictionary = {
            "link_lenght": get_token_value(
                file_name, "link_", "(\\d*\\.\\d+|\\d+\\.?\\d*)"
            ),
            "plddt": get_token_value(
                file_name, "__plddt_", "(\\d*\\.\\d+|\\d+\\.?\\d*)"
            ),
            "RMSD": get_token_value(
                file_name, "__rmsd_", "(-?\\d*\\.\\d+|-?\\d+\\.?\\d*)"
            ),
            #'Rmsd_all_fixed': get_token_value(file_name, '__rmsd_all_fixed_', "(-?\\d*\\.\\d+|-?\\d+\\.?\\d*)"),
            "RMSD_sculpted": get_token_value(
                file_name, "__rmsd_sculpted_", "(-?\\d*\\.\\d+|-?\\d+\\.?\\d*)"
            ),
            "pae": get_token_value(file_name, "__pae_", "(\\d*\\.\\d+|\\d+\\.?\\d*)"),
            "model_path": file_name,
            "sequence": seq,
            "rfdiff model": get_token_value(
                file_name, "__rf_", "(\\d*\\.\\d+|\\d+\\.?\\d*)"
            ),
        }  # MODEL PATH for scoring_rg_... #jsonfilename for traceability

        list_of_dicts.append(dictionary)

    # columns = ['link_length', 'plddt', 'loop_plddt', 'RMSD', 'model_path', 'sequence', 'score_traceb']
    df = pd.DataFrame(list_of_dicts)
    path_csv = os.path.join(os.path.dirname(output_dir), "output.csv")
    df.to_csv(
        path_csv,
        mode="a",
        header=not os.path.exists(path_csv),
        index=False,
        float_format="%.1f",
    )


class NumpyInt64Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int64):
            return int(obj)
        return super(NumpyInt64Encoder, self).default(obj)


def parse_pdb_structure(pdb_file):
    """Parse a PDB with Biopython, moving CONECT records to the end of the
    file first.

    Biopython's PDBParser treats the first CONECT record as the END of the
    atomic coordinates. Some tools write CONECT records in the middle of the
    file (e.g. between MODEL sections), which silently truncates the parse -
    every residue after the CONECT block is dropped (RFDiff's line-based
    parser does not have this limitation, so the two tools disagree on the
    residue list). Moving the CONECT block to its canonical position (before
    the END record) makes the full file readable.
    """
    with open(pdb_file) as f:
        lines = f.readlines()
    conect = [l for l in lines if l[:6] == "CONECT"]
    if not conect:
        return PDBParser(QUIET=True).get_structure("protein", pdb_file)
    rest = [l for l in lines if l[:6] != "CONECT"]
    for i in range(len(rest) - 1, -1, -1):
        if rest[i][:3] == "END" and rest[i][3:4] == " ":
            rest[i:i] = conect
            break
    else:
        rest = rest + conect
    return PDBParser(QUIET=True).get_structure("protein", io.StringIO("".join(rest)))

def getChainResidOffsets(pdb_file, designable_residues):
    chainResidOffset = {}
    con_hal_idx = []

    structure = parse_pdb_structure(pdb_file)

    global_residue_index = 1

    for chain in structure.get_chains():
        chain_id = chain.get_id()

        first_residue_seen = False

        for residue in chain.get_residues():

            # Store the global index of the first residue in this chain
            if not first_residue_seen:
                chainResidOffset[chain_id] = global_residue_index - 1
                first_residue_seen = True

            if designable_residues:
                if f"{chain_id}{residue.get_id()[1]}" not in designable_residues:
                    con_hal_idx.append((chain_id, residue.get_id()[1]))

            global_residue_index += 1

    return chainResidOffset, con_hal_idx

def get_all_residues(pdb_file):
    structure = parse_pdb_structure(pdb_file)

    residues = []
    residue_indices=[]
    counter=0
    for chain in structure.get_chains():
        chain_id = chain.get_id()

        for residue in chain.get_residues():

            # residue.get_id() = (hetflag, resseq, insertion_code)
            hetflag, resseq, icode = residue.get_id()
            residues.append((chain_id, resseq))
            residue_indices.append(counter)
            counter+=1  
    return residues, residue_indices

def process_pdb_files(pdb_path: str, out_path: str, cfg, trb_paths=None, cycle=0):
    skipRfDiff = cfg.get("skipRfDiff", False)
    designable_residues = cfg.get("designable_residues", None)

    fixpos = {}
    pdb_files = Path(pdb_path).glob("*.pdb")

    contig = cfg.contig
    abeceda = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    ref_path=cfg.get("pdb_path", None)
    if ref_path is not None:
        print("DEBUG: pdb_path found")
        all_residues_reference, all_residue_indices_reference = get_all_residues(ref_path)
        print(f"DEBUG all_residues_reference: {all_residues_reference}")    

    for pdb_file in pdb_files:

        pdb_basename = pdb_file.stem

        ## We get RfDiff model number
        rf_model_num = get_token_value(
            os.path.basename(pdb_file), "_" if cycle == 0 else "rf_", "(\\d+)"
        )  # get 0 from _0.fa using reg exp #get 0 from rf_0__model_1__cycle_2__itr_0__.pdb
        # (We could also just get the first _# and exit regex early -- RfDiff model number is always the first element of the filename)
        print(f"RfDiff model number: {rf_model_num}")

        fixed_res = {}

        # We need to renumber fixed resids: each chain should start with 1
        chainResidOffset, con_hal_pdb_idx_complete = getChainResidOffsets(
            pdb_file, designable_residues
        )
        print(f"ChainResidOffset: {chainResidOffset}")

        with open(
            f"{pdb_path}/../chainResidOffset_{rf_model_num}.json",
            "w" if cycle == 0 else "r",
        ) as f:
            if cycle == 0:
                json.dump(chainResidOffset, f)
            else:
                chainResidOffset = json.load(f)
        print(f"Dumped or read chainResidOffset_{rf_model_num}.json")
        print(f"ChainResidOffset: {chainResidOffset}")

        if not skipRfDiff:
            trb_file = pdb_file.with_suffix(".trb")

            if not trb_file.exists():
                rf_model_num = get_token_value(
                    os.path.basename(pdb_file), "rf_", "(\\d+)"
                )
                trb_file = os.path.join(
                    os.path.dirname(pdb_file), f"_{rf_model_num}.trb"
                )
                trb_file = trb_file.replace("2_1_cycle_directory", "1_rfdiff")
                print(
                    f"TRB file not found for {pdb_basename}. CAUTION, using composed path"
                )

            print(pdb_file, trb_file)

            with open(trb_file, "rb") as f:
                trb_data = pickle.load(f)

            contig = trb_data["config"]["contigmap"]["contigs"][0]
            partial_diffusion = cfg.get("partial_diffusion", False)
            

            con_hal_idx = []
            for id0, value in enumerate(trb_data["inpaint_seq"]):  # Just define con_hal_idx from inpaint_seq and ignore everything else. This should work
                if value == True:
                    for key in abeceda:
                        if key in chainResidOffset:

                            if id0 >= chainResidOffset[key]:
                                currentResidueChain = key
                    con_hal_idx.append(
                        (
                            currentResidueChain,
                            id0 +1
                        )
                    )
            #print(f"DEBUG con_hal_idx: {con_hal_idx}")

            if (trb_data["config"]["contigmap"]["provide_seq"] != None):
                provide_seq_residues = np.where(       
                [
                    a != b
                    for a, b in zip(trb_data["inpaint_seq"], trb_data["inpaint_str"])
                ]
                )[
                    0
                ]  # if this works...
                #print(f"DEBUG: PARTIAL DIFUSSION KEEPING RESIDUES {list(provide_seq_residues)}")
            else:
                provide_seq_residues = []

            # Process con_hal_idx to extract chain ids and indices
        else:
            con_hal_idx = con_hal_pdb_idx_complete
            provide_seq_residues = []

        # fixed_res should never be empty, otherwise ProteinMPNN will throw a KeyError fixed_position_dict[b['name']][letter]
        # We need to set blank fixed_res for each generated chain (based on contig).

        if cfg.inference.symmetry != None:  # if we find symmetry:
            breaks = int(cfg.inference.symmetry[1:])
        else:
            breaks = contig.count("/0 ") + 1

        fixed_res = dict(zip(abeceda, [[] for _ in range(breaks)]))
        print(f"DEBUG: Fixed res (according to contig chain breaks): {fixed_res}")

        # This is only good if multiple chains due to symmetry: all of them are equal; ProteinMPNN expects fixed_res as 1-based, resetting for each chain.
        # TODO: Fix for multi-chain receptors 

        if not skipRfDiff:
            con_ref_idx0 = trb_data.get('con_ref_idx0', []) 
            complex_con_ref_idx0 = trb_data.get('complex_con_ref_idx0', con_ref_idx0)
        else:
            con_set = set(con_hal_idx)

            complex_con_ref_idx0 = [
                i for i, res in enumerate(all_residue_indices_reference)
                if res not in con_set
            ]
        complex_con_ref_idx0 = copy.deepcopy(sorted(list(complex_con_ref_idx0) + list(provide_seq_residues)))
        print(f"DEBUG: complex_con_ref_idx0 (combined con_ref_idx0 and provide_seq_residues): {complex_con_ref_idx0}")
        if not skipRfDiff:
            # Content-based (chain, resseq) for all PDB-mapped contig
            # positions, in contig order - straight from the trb. The old code
            # derived these by indexing the Biopython residue list
            # (all_residues_reference) with RFDiff's own pdb_idx indices
            # (complex_con_ref_idx0). That only works if both parsers see
            # exactly the same residues in the same order, which is not true
            # for e.g. multi-model PDBs (RFDiff reads every model, Biopython
            # only the first one) and caused an IndexError.
            trb_ref_pdb_idx = trb_data.get(
                "complex_con_ref_pdb_idx", trb_data.get("con_ref_pdb_idx", None)
            )
            if trb_ref_pdb_idx is not None:
                complex_con_ref_pdb_idx = [
                    (str(r[0]), int(r[1])) for r in trb_ref_pdb_idx
                ]
                # provide_seq (partial diffusion) positions are inpainted but
                # have no PDB-mapped (chain, resseq); add placeholders so the
                # list is as long as con_hal_idx for the zip below (values are
                # not used there).
                complex_con_ref_pdb_idx += [("_", 0)] * len(provide_seq_residues)
            else:
                # older trb without a content-based (chain, resseq) list: fall
                # back to positional indexing (works when the parsers agree)
                try:
                    complex_con_ref_pdb_idx = [
                        all_residues_reference[id0]
                        for id0 in complex_con_ref_idx0
                    ]
                except IndexError:
                    print(
                        "WARNING: trb indices do not line up with the Biopython "
                        "residue list and the trb has no con_ref_pdb_idx; "
                        "using placeholders (values are unused downstream)."
                    )
                    complex_con_ref_pdb_idx = [
                        ("_", 0) for _ in complex_con_ref_idx0
                    ]
            while len(complex_con_ref_pdb_idx) < len(con_hal_idx):
                complex_con_ref_pdb_idx.append(("_", 0))
        else:
            # safe here: complex_con_ref_idx0 was built by enumerating
            # all_residue_indices_reference itself
            complex_con_ref_pdb_idx = [
                all_residues_reference[id0] for id0 in complex_con_ref_idx0
            ]
        print(f"DEBUG complex_con_ref_pdb_idx: {complex_con_ref_pdb_idx}")
        print(f"DEBUG con_hal_idx: {con_hal_idx}")
        for (chain, idx), (chain_from_input, idx_from_input) in zip(con_hal_idx, complex_con_ref_pdb_idx):
            if not skipRfDiff:
                #print(f"DEBUG: {(chain, idx), (chain_from_input, idx_from_input)} in con_hal_idx and complex_con_ref_pdb_idx")
                if trb_data["inpaint_seq"][
                    idx - 1
                ]:  # skip residues with FALSE in the inpaint_seq array
                    #print(f"DEBUG: Residue {chain}{idx} is fixed (True in inpaint_seq)") 
                    fixed_res.setdefault(chain, list()).append(
                        idx - chainResidOffset[chain]
                    )
            else:
                fixed_res.setdefault(chain, list()).append(
                    idx
                )
            # RfDiff outputs multiple chains if contig has /0 (chain break)

        print(f"Fixed res: ${fixed_res}")

        # Optionally unfix residues for ProteinMPNN.
        #
        # By default MPNN's design set == RFDiffusion's design set: exactly the
        # positions whose backbone was generated (inpaint_seq == False). To let
        # RFDiffusion design a SUBSET of the residues the user wants changed,
        # set mpnn_designable_residues (input PDB coordinates) to a list of
        # residues whose backbones stay fixed in the contig but which MPNN
        # should re-sequence anyway. Accepted specs: "A12" (single residue),
        # "A12-A45" (range, same chain), "B" (whole chain). A YAML list or a
        # single string like "[A1-A4, A6-A8, B]" both work.
        mpnn_designable = cfg.get("mpnn_designable_residues", None)
        if isinstance(mpnn_designable, str):
            mpnn_designable = [
                t for t in re.split(r"[,\[\]\s]+", mpnn_designable) if t
            ]
        if mpnn_designable and skipRfDiff:
            print(
                "NOTE: mpnn_designable_residues is ignored when skipRfDiff is True; "
                "use designable_residues instead."
            )
        if not skipRfDiff and mpnn_designable:
            if ref_path is None:
                print(
                    "WARNING: mpnn_designable_residues requires pdb_path (the input "
                    "PDB the residue numbers refer to). Ignoring it."
                )
            else:
                # Input PDB residue (chain, resseq) -> flat contig position.
                # Content-based: the trb stores the PDB-mapped contig
                # positions in contig order as two parallel lists:
                #   complex_con_ref_pdb_idx[j] = (chain, resseq) in the input PDB
                #   complex_con_hal_idx0[j]    = position in the generated sequence
                # This does not assume RFDiff's and Biopython's residue lists
                # line up positionally (they don't for multi-model PDBs, where
                # RFDiff reads every model and Biopython only the first).
                trb_ref_pdb = trb_data.get(
                    "complex_con_ref_pdb_idx",
                    trb_data.get("con_ref_pdb_idx", None),
                )
                hal_idx_list = list(
                    trb_data.get(
                        "complex_con_hal_idx0",
                        trb_data.get("con_hal_idx0", []),
                    )
                )
                if trb_ref_pdb is not None:
                    residue_to_pos = {
                        (str(r[0]), int(r[1])): int(p)
                        for r, p in zip(trb_ref_pdb, hal_idx_list)
                    }
                else:
                    # older trb without content-based (chain,resseq) lists:
                    # fall back to positional indexing
                    ref_to_pos = dict(
                        zip(
                            (int(x) for x in complex_con_ref_idx0),
                            (int(x) for x in hal_idx_list),
                        )
                    )
                    res_to_ref_idx = {
                        res: i for i, res in enumerate(all_residues_reference)
                    }

                def _parse_designable_spec(spec):
                    """'A12' -> [('A',12)]; 'A12-45' or 'A12-A45' -> range on
                    chain A; 'B' -> ('CHAIN','B') (resolved vs the contig's
                    fixed backbone). Returns None if unparseable."""
                    spec = str(spec).strip()
                    if not spec:
                        return None
                    if "-" in spec:
                        ch = spec[0]
                        if not ch.isalpha():
                            return None
                        parts = spec[1:].split("-")
                        if len(parts) != 2:
                            return None
                        try:
                            lo = int(parts[0])
                            hi_part = parts[1]
                            # allow both 'A12-45' and 'A12-A45'
                            hi = int(hi_part[1:]) if hi_part[:1].isalpha() else int(hi_part)
                        except ValueError:
                            return None
                        return [(ch, i) for i in range(lo, hi + 1)]
                    if len(spec) == 1 and spec.isalpha():
                        return ("CHAIN", spec)
                    try:
                        return [(spec[0], int(spec[1:]))]
                    except (ValueError, IndexError):
                        return None

                for spec in mpnn_designable:
                    parsed = _parse_designable_spec(spec)
                    if parsed is None:
                        print(
                            f"WARNING: mpnn_designable_residues: could not "
                            f"parse '{spec}'; ignoring."
                        )
                        continue
                    if isinstance(parsed, tuple) and parsed[0] == "CHAIN":
                        if trb_ref_pdb is not None:
                            targets = sorted(
                                (ch, rn)
                                for (ch, rn) in residue_to_pos
                                if ch == parsed[1]
                            )
                        else:
                            targets = [
                                r
                                for r in all_residues_reference
                                if r[0] == parsed[1]
                            ]
                    else:
                        targets = parsed
                    for ich, resn in targets:
                        if trb_ref_pdb is not None:
                            pos = residue_to_pos.get((ich, resn))
                        else:
                            ref_i = res_to_ref_idx.get((ich, resn))
                            pos = ref_to_pos.get(ref_i) if ref_i is not None else None
                        if pos is None:
                            print(
                                f"WARNING: mpnn_designable_residues: {ich}{resn} "
                                f"not found in the contig's fixed backbone "
                                f"(input PDB: {ref_path}); ignoring."
                            )
                            continue
                        # Which output chain contains this flat position, and
                        # what is its 1-based per-chain index (MPNN convention)?
                        ochain = None
                        for key in abeceda:
                            if key in chainResidOffset and pos >= chainResidOffset[key]:
                                ochain = key
                        if ochain is None:
                            print(
                                f"WARNING: mpnn_designable_residues: {ich}{resn} "
                                f"(position {pos}) does not map to any output "
                                f"chain; ignoring."
                            )
                            continue
                        perchain_idx = pos - chainResidOffset[ochain] + 1
                        if perchain_idx in fixed_res.get(ochain, []):
                            fixed_res[ochain].remove(perchain_idx)
                            print(
                                f"mpnn_designable_residues: unfixing {ich}{resn} "
                                f"(output {ochain}{perchain_idx}) for ProteinMPNN"
                            )
                        else:
                            print(
                                f"NOTE: {ich}{resn} (output {ochain}{perchain_idx}) "
                                f"was not in the fixed list; nothing to unfix."
                            )
                        ctd = str(cfg.get("chains_to_design", None) or "")
                        if ctd and ochain not in ctd.split():
                            print(
                                f"WARNING: output chain {ochain} (of {ich}{resn}) "
                                f"is not in chains_to_design ('{ctd}'); "
                                f"ProteinMPNN would still fix the whole chain. "
                                f"Add {ochain} to chains_to_design."
                            )

        fixpos[pdb_basename] = fixed_res

    #print("_________trb data____", trb_data)

    #print("_________ fix pos_________", fixpos)
    file_path = os.path.join(out_path, "fixed_pdbs.jsonl")
    # Save the fixpos dict as a JSON file
    with open(file_path, "w") as outfile:
        json.dump(fixpos, outfile, cls=NumpyInt64Encoder)

    return file_path


def get_chains_seq(pdb_file):
    """
    Extract sequences of all chains in a protein from a PDB file.
    Args:
        pdb_file (str): The path to the PDB file.

    Returns:
        list: A list of strings of seqenceces of all chains.

        Assumptions:
        - RfDIFF forms the new pdb in a way that the connected helices are now chain A, regardless of how they were identified before
        - Other chains are now B, C,... regardles if before they were A

    """
    # Parse the PDB file
    parser = PDBParser()
    structure = parser.get_structure("protein", pdb_file)

    # Extract polypeptides and their sequences
    pp_builder = PPBuilder()
    other_chains_sequences = []

    for pp in pp_builder.build_peptides(structure):
        sequence = pp.get_sequence()
        seqs = Seq(sequence)
        other_chains_sequences.append(seqs)

    return other_chains_sequences


def read_fasta_file(fasta_file):
    """
    Read a FASTA file using Biopython.

    Args:
        fasta_file (str): The path to the FASTA file.

    Returns:
        list: A list of SeqRecord objects containing the sequences from the FASTA file.
    """
    sequences = []

    with open(fasta_file, "r") as file_handle:
        for record in SeqIO.parse(file_handle, "fasta"):
            sequences.append(record)
        # sequences = list(SeqIO.parse(file_handle, "fasta")) # better, for loop ni potreben
    return sequences


def change_sequence_in_fasta(pdb_file, mpnn_fasta):
    """
        #function adds : for chainbrakes and chain sequences of all other chains
    sequences_all_chains = get_chains_seq(pdb_file)
    sequences_other_chains = sequences_all_chains[1:]
    #sequences of other chains because mpnn fasta has only chain A and no other chain seqs

    sequences_mpnn = read_fasta_file(mpnn_fasta)
    #funkcija za drop duplicates, napiši funkcijo
    for seq_record in sequences_mpnn:
            new_sequence = seq_record.seq
            for other_chain in sequences_other_chains:
                    new_sequence += f":{other_chain}"
            seq_record.seq = Seq(new_sequence)

    with open(mpnn_fasta, "w") as output:
            SeqIO.write(sequences_mpnn, output, "fasta")
    """
    print(os.path.exists(pdb_file))
    i = 0
    seq_dict = {}
    for record in SeqIO.parse(mpnn_fasta, "fasta"):
        if i == 0:
            # Skip if record.description doesn't contain sample=. First seq is actually input to mpnn. However, after restart, we should not remove it again.
            if "sample=" not in record.description:
                print(
                    f"Skipping {record.description} for it does not contain sample=, it is input to and not output of mpnn."
                )
                i += 1
                continue
        i += 1
        print(record.seq, record.description)
        seq_dict[record.seq] = record.description

    print(seq_dict)
    sequences = []
    for record in SeqIO.parse(mpnn_fasta, "fasta"):
        if record.description in seq_dict.values():
            newseq = record.seq.replace("/", ":")
            record.seq = newseq
            sequences.append(record)
    print(sequences)
    SeqIO.write(
        sequences, mpnn_fasta, "fasta-2line"
    )  # This needs to be fasta-2line for MSA code to work


def match_linker_length(trb_path):
    import re

    with open(trb_path, "rb") as f:
        trb_dict = pickle.load(f)

    input_string = " ".join(
        trb_dict["sampled_mask"]
    )  #' '.join(['A1-30/6-6/C1-30/0', 'D1-30/0', 'B1-30/0'])

    pattern = r"(?<=/)(\\d+)(?=-\\d+/)"
    """
        (?<=/): Positive lookbehind assertion -> sub-pattern is preceded by a '/'.
        (\\d+): Capture group that matches one or more digits.
        (?=-\\d+/): Positive lookahead assertion -> sub-pattern is followed by a '-' and one or more digits, then a '/'.
    """
    match = re.search(pattern, input_string)
    if match:
        link = match.group(1)
        return link  # Output: 6
    else:
        print("No match found in linker length function")


def find_correct_trb(model_i):
    json_files = glob.glob(os.path.join(model_i, "_*000.json"))

    print(json_files)
    for json_file in json_files:
        json_filename = os.path.basename(json_file)
        number_trb = get_token_value(
            json_filename, "_", "(\\d+)(?=_)"
        )  # stop at first digit -> (?=_)
        trb_file = os.path.join(f"_{number_trb}.trb")
    return trb_file
