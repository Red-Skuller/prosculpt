# Prosculpt
Protein design and sculpting using Rosetta and Deep learning methods (RFDiff and Alphafold2)

![image](pipeline_pic.png)
## Description 
The script `prosculpt_run.py` runs a pipeline to automate the processes of generating protein structures with RFdiffusion, sequence generation with ProteinMPNN, and folding and evaluation with AF2 and Rosetta. Specifically, the script uses motif scaffolding to generate structures (see [RFdiffusion github repository](https://github.com/RosettaCommons/RFdiffusion/blob/main/README.md)).

The main steps are as follows:
1. The main inputs are a protein PDB file and a yaml file that specifies how to generate the new structure around the original protein (which parts to keep, etc.). 
2. The RFdiffusion module generates new protein structures (backbone atoms only) based on the options provided.
4. Generated PDB files are preprocessed using helper scripts and ProteinMPNN generates sequences (aminoacid residues) for these structures.
5. Sequences are prepared for the AF2, which folds them into structures.
6. Additional evaluation parameters are calculated for these structures using a separate script with Rosetta.
7. The final output is a CSV file containing AF2 structure path, evaluation parameters etc.

A working Collab can be found [here](https://colab.research.google.com/github/ajasja/prosculpt/blob/main/Prosculpt_Colab.ipynb)

## Requirements  
The script requires the prosculpt package. It assumes that RFdiffusion, proteinMPNN, and AF2 are installed and that the correct paths are provided in the `installation.yaml` config file. Additionall biopython, hydra-core, pandas and scipy and pyRosetta are required.  

For running tests, Python version must be >= 3.7 (it needs the `capture_output` arg).

## Installation
The easiest way to install prosculpt is using a container manager like [apptainer](https://apptainer.org/) (or singularity). To do so, follow these instructions

Begin by cloning prosculpt:
```bash
git clone https://github.com/ajasja/prosculpt.git
cd prosculpt
```
Create a new environment called prosculpt (or any name of your choice) using python or any venv manager and activate it:
```bash
python -m venv prosculpt_venv/
source prosculpt_venv/bin/activate
```
Install it and its dependencies
```bash
pip install .
```
Install pyrosetta into the environment
```bash
pip install pyrosetta --find-links https://west.rosettacommons.org/pyrosetta/quarterly/release.cxx11thread.serialization
```

At a location of your choice download the appropriate SIF files, executing the following commands:
```bash
singularity pull rfdiff.sif docker://rosettacommons/rfdiffusion
singularity pull proteinmpnn.sif docker://rosettacommons/proteinmpnn
singularity pull pymol.sif docker://jysgro/pymol:3.1.0_amd_arm
```

Depending on the prediction model, follow the appropriate instructions (Multiple instructions can be followed and the prediction model selected as desired in the configuration)

For using Boltz2, install boltz2 into the environment
```bash
pip install boltz[cuda] -U
```
it might be necessary to install compilers for boltz2 to run correctly. The exact command for this is system-dependent. If using conda, it can be done by
```bash
conda install -c conda-forge compilers
```


For Alphafold 3 use, request the model parameters following the instructions on the [Alphafold 3 github](https://github.com/google-deepmind/alphafold3). Then download the sif file using
```bash
singularity pull af3_2.sif docker://kosinskilab/alphafold3
```

For using Colabfold (Note: Colabfold is slower and less convenient than using Boltz2 or AF3 and is not recommended)
```bash
singularity pull colabfold.sif docker://unionbio/colabfold:w4KMVR7WrKDlCbdQ1BYrjQ-test
```


Copy or rename the file config/installation.yaml.singularity_template as config/installation.yaml. Then replace the default values with the ones corresponding to your system. Mainly:

- Replace the run command of the sif files with the appropriate commands and paths for your system.
- Replace the prosculpt_python_path with the python path of your prosculpt env (You can get it by running "which python" with the env active)
- Replace the slurm options with those that correspond with your cluster (these can then be overwritten by individual job settings)
- If multiple users will be using the same installation, it is good to set the boltz2 cache parameter to the current user so that only one set of parameters are downloaded.

If using Boltz2 or Colabfold it is necessary to run a job to allow them to download the parameters. In a slurm cluster you can run:

Colabfold:
```bash
python run_tests.py -t unconditional
```
Boltz2:
```bash
python run_tests.py -t unconditional_boltz
```

## Usage

To run Prosculpt on Slurm, use the `slurm_runner.py` passing it an input yaml file with the job options. Note that slurm_runner no longer supports passing arguments inline with the exception of ++output_folder.
```bash
python slurm_runner.py job_parameters.yaml
```

If you want to run Prosculpt locally, it can be called as
```bash
python prosculpt_run.py -cd config_directory -cn config_file_name
```

(note that the config file name must be written without the yaml extension)

## Examples
For usage examples see the /Examples or /Minimal_Prosculpt_script folders

## Skipping RfDiff
Sometimes you may want to skip RfDiff and only pass the pdb to ProteinMPNN and AF with a few residues to change. 

Add `skipRfDiff: True` and `designable_residues: [A8, A9, A10, A13, A85]` to the yaml config, where designable_residues contains the residues you want to change.
If there are non-redesigned chains in the model that you want included in the final model, add the letter chain (without any numbers) to the designable residues.

In order to actually produce diverse results, `--sampling_temp` and `--backbone_noise` passed to ProteinMPNN are increased to `0.3` and `1`, respectively.  

## Working with natural proteins
To be modelled correctly, natural proteins require structure prediction softwares to work with multiple sequence alignments. In order to obtain good results when working with natural proteins, the option "use_a3m" needs to be set to True. This option requires the user to input an MSA file for each chain in the reference structure for which the user wants alignment information to be used. To obtain these MSAs just run each chain in the reference structure separately through an MSA server and use the A3M file it produces. The easiest way is to run the sequence for your input structure through Alphafold. Each file needs to contain within the name "Chain_X" where X is the chain ID of the corresponding chain in the reference structure file. These alignment files need to be placed together in a directory and the "a3m_dir" option needs to be set to this directory in the yaml config file or passed through command line. For clarification, look at the binder example in the Examples directory. 

Prosculpt generates new "partial alignment" files for each design where natural proteins are modelled using MSAs and designed proteins (or designed portions of hybrid proteins) are not.

## Backbone filtering
For backbone filtering, custom scripts called plugins are used. A plugin is a python script containing a function called "filter_backbone" which takes a pdb file and any set of arguments and returns true if the pdb passes the filter and false if it doesn't. The content of the function can be defined as needed by the user. To call filters add the following code to your configuration yaml. Examples can be found in the plugins folder and usage example can be found in the examples folder

```
rfdiff_backbone_filters:
    - filter_name: [filter_name]
      filter_script: [filter script file path]
      delete_failed: [true/false]
```

## Metrics and scoring script

Prosculpt calculates the following metrics

| Metric  | Meaning |
| ------------- | ------------- |
| RMSD  | Full model RMSD to RFDiff backbone  |
| pLDDT  | full model pLDDT  |
| RMSD_sculpted  | RMSD of any residues not present in original structure to RFDiff backbone |
| RMSD_fixed_chains  | RMSD of residues in non-redesigned chains to RFDiff backbone  |
| RMSD_motif  | RMSD of non-redesigned residues in chains containing redesigned residues  |
| pae  | Predicted alignment error  |

In addition, a scoring script can be added to calculate extra metrics. The scoring script is a python file called "scoring_script.py" in the scripts folder that outputs a CSV called "rosetta_scores.csv", with a "model_path" column. This CSV file gets merged with the prosculpt metrics for the final output. A default scoring script is included in the scripts folder, which calculates radius of gyration, charge and sap.

## Inputs: 
Most input parameters are documented in the examples, as well as the `run.yaml` config file. However, here's some additional info about them:
- `contig`: explained in detail in the [RFdiffusion github repository](https://github.com/RosettaCommons/RFdiffusion/blob/main/README.md).
    - Please note: The whole argument must be enclosed in double quotes and square brackets!
- `output_dir`: output directory. Along the pipeline, each module will create its own subdirectory. The AF2 models are in the end renamed and copied to a subdirectory called `final_pdbs` 
- `num_designs_rfdiff`: number of structures generated by RFdiffusion
- `num_seq_per_target_mpnn`: number of sequences generated per one RFdiffusion structure (each sequence is by default folded 5 separate times by AF2)

A note on outputs: when running on a cluster for each task a separate `final_outputs.csv` will be created. Run `merge_csv.py` along with the argument `--output_dir` to merge all csv files into one file.

## Passing additional arguments to ProteinMPNN
Add parameters as required by ProteinMPNN to the `pass_to_mpnn` group in run.yaml (including trailing `--`).

## Passing additional arguments to AlphaFold
Add parameters as required by AlphaFold to the `pass_to_af` group in run.yaml (including trailing `--`). Switches should be passed with an empty string value, like this `--templates: ""`. 

## Adding ligands and other static additions to Boltz runs (Boltz2 only)
By default, the generated `boltz.yaml` files only contain the protein sequences designed by RFDiffusion + ProteinMPNN. To include ligands, RNA/DNA, constraints, templates, or affinity properties in every generated YAML, set the `boltz_extras` option in your run config. It mirrors the [Boltz input schema](https://github.com/jwohlwend/boltz) (`sequences`, `constraints`, `templates`, `properties`, `version`).

**Chain ID convention:** the designed protein chains are labeled `A`, `B`, `C`, ... by their position in the colon-separated MPNN sequence (first segment = `A`). Any extra entry must use a chain letter that the design does NOT use, or omit the `id` entirely to have one auto-assigned. Explicit IDs that collide with a designed chain raise an error before Boltz is run.

```yaml
boltz_extras:
  sequences:
    - ligand: {id: C, ccd: SAH}                        # CCD ligand
    - ligand: {id: D, smiles: 'CC(=O)Oc1ccccc1C(=O)O'} # SMILES ligand
    - ligand: {id: E, ccd: [EDO, GLU]}                 # multi-residue ligand
    - rna:    {id: F, sequence: GCAUAGC}               # RNA (or dna:)
  constraints:
    - pocket: {binder: C, contacts: [[A, 42], [B, 10]], max_distance: 6.0}
  properties:
    - affinity: {binder: C}                            # Boltz2 affinity scoring
```

See `Examples/ligand_boltz.yaml` for a complete working example. Note that ligand chains are present in the predicted PDBs; the RMSD/plDDT scoring steps automatically ignore non-protein chains.

**Trimmed design, full-complex Boltz prediction.** When the input structure is large and most of it is irrelevant to the design, let RFDiffusion/ProteinMPNN work only on the small `contig` region and add the rest of the complex to Boltz as **static protein chains**. `protein` entries can take their sequence straight from a PDB via `pdb_chain` (from your `pdb_path` by default, or from a different file with `pdb_file` - e.g. the **full, untrimmed** structure when you trimmed the input PDB by hand, with an optional `residues` selection) instead of spelling it out, and `templates` can reference a structure directly (`input_pdb: true` for the input PDB, or an explicit `pdb:`/`cif:` path) so Boltz anchors the static chains to their native coordinates:

```yaml
boltz_extras:
  sequences:
    - protein: {id: B, pdb_chain: A, residues: "27-99"}   # rest of antigen A
    - protein: {id: C, pdb_chain: A, residues: "121-544"}
    - protein: {id: D, pdb_chain: H}                      # whole antibody chain
    - protein: {id: E, pdb_chain: H, pdb_file: full.pdb}  # from a different file
  templates:
    - input_pdb: true            # or: pdb: /path/to/full_structure.pdb
      chain_id: [A, B, C, D]     # Boltz input chains to seed (optional)
      template_id: [A, A, A, H]  # template chains, paired 1:1 (optional)
```

Keep the static chains and the contig **complementary** (don't add a static chain overlapping residues the contig already designs). Omit `chain_id`/`template_id` to let Boltz match template chains to input chains by sequence alignment; the same template chain may be mapped to several input chains. See `Examples/full_complex_boltz.yaml`.

**Contact coordinates are auto-remapped onto the newly formed chains.** Write the *protein* references in `pocket.contacts`, `contact.token1/token2`, and `bond.atom1/atom2` in **input PDB coordinates** (chain + residue of your `pdb_path` file). Prosculpt remaps each one to the corresponding residue in the designed (RFDiffusion/MPNN) output using the RFDiffusion trb mapping, so contacts stay attached to the right residues even when the contig changes chain lengths (e.g. `[A1-25/1-5/A30-50]`) or re-maps chains (e.g. `[B1-10/0 A1-10]`). Ligand/metal references (the `binder`, or a token pointing at a ligand chain) use the static chain IDs you assigned in `boltz_extras.sequences` and are passed through unchanged. If a referenced residue was removed by the contig: the reference is kept when its chain is a chain of the generated yaml (e.g. a static extra protein), and **dropped from the constraint (with a warning)** when the chain is not in the yaml at all - a dangling chain reference would make Boltz reject the whole input. A constraint whose contacts are all dropped is removed entirely. If there is no input PDB (unconditional design), references are left as-is.

For edits that go beyond the static block (per-model or computed values), set `boltz_yaml_postprocess_script` to a Python file defining `postprocess_yaml(yaml_path: str, cfg: dict, model_id: str) -> None`. It is called once per generated YAML after it is written.

## Fused full-length complexes in Boltz (Boltz2 only: `boltz_full_complex`)
The `boltz_extras`-based full-complex setup above adds the rest of the structure as **separate static chains**. With `boltz_full_complex: true`, Prosculpt instead **splices the designed segments (including the new linker/peptide insertions) into the full native sequences** of the input chains, so Boltz folds the complete full-length chains in one complex - the recommended setup when the design is a covalently fused patch (e.g. peptide linkers inserted into a protein, full tetramers, ...). It requires `pdb_path`.

Rules, per contig chain:
- **Fused chains** (contig chains containing new positions): the MPNN design is spliced into the **full native sequence** of the host input chain (the input chain contributing the most contig positions). Native N-/C-terminal regions outside the contig are added back, and cross-chain segments stay where the contig put them.
- **Pure context chains** (contig chains without new positions, e.g. a whole subunit or interface windows): the host chain is taken **full length** from the input PDB (all residues, native sequence).
- **Native residues the contig skipped between two anchors stay excluded.** A short designed linker (a few residues) cannot physically span a large gap, so e.g. a 28-residue loop that the contig jumps over does not reappear in the fused chain. If a skipped region is essential, redesign the contig so it is not skipped (and make the designed linkers long enough to span it).
- **Input chains not referenced by the contig** are appended as full native chains (alphabetical chain order) after the contig chains, so a tetramer stays a tetramer even if only two subunits were in the contig.

Chain letters stay `A`, `B`, `C`, ... by order (contig chains first, then unreferenced chains), so `boltz_extras` ligand IDs and `use_a3m`/`a3m_dir` work unchanged - but give the (longer) spliced chains real MSAs, since the a3m realignment now matches the full sequences. Constraint contacts in `boltz_extras` are remapped onto the spliced chains automatically (input PDB coordinates, as for the designed chains). Scoring (RMSD, plDDT_sculpted) is evaluated on the **designed region only**, using a per-model map saved next to the YAML (`<model>.full_complex.json`); native spliced-in residues do not enter those metrics.

```yaml
boltz_full_complex: true     # Boltz predicts the full input complex
use_a3m: true                # MSAs for the (longer) spliced chains
# boltz_extras: { sequences: [{ligand: {id: E, ccd: CU}}], constraints: [...] }
```

Example: AAV9 tetramer (chains C/D/E/F, 518 residues each) + two 27-mer peptides fused into C and D. The RFDiff contig only carries the binding interfaces + linkers + peptides; with `boltz_full_complex: true`, Boltz receives four full-length subunit chains (peptide fusions spliced in) + the two remaining subunits as full native chains + the three Cu ligands with auto-remapped pocket contacts. See `Examples/fused_full_complex_boltz.yaml`.

## Giving ProteinMPNN a different design set than RFDiffusion
By default, ProteinMPNN re-sequences exactly the residues whose backbone RFDiffusion generated (the non-PDB part of the `contig`) - the two "design sets" are identical. To let RFDiffusion design a **subset** of the residues you want changed, keep the rest of the region fixed in the contig and list it in `mpnn_designable_residues` (input PDB coordinates; accepted: `A12`, `A12-A45` or `A12-45`, `B` for a whole chain; a YAML list or a single quoted string like `"[A1-A4, A6-A8, B]"`):

```yaml
pdb_path: input.pdb
# RFDiffusion only generates a new 25-mer backbone (A31-55 region)
contig: "[A1-30/25-25/A56-100]"
# ...but MPNN also re-sequences these fixed-backbone residues
mpnn_designable_residues: [A20-A29, A56-A70]
```

Now RFDiff's design set (25 new residues) is a strict subset of MPNN's design set (25 + 10 + 15). Residues inside the newly generated region are always re-sequenced and need not be listed. See `Examples/redesign_subset_boltz.yaml`.

The opposite direction (MPNN designs a **subset** of what RFDiffusion diffused) is the built-in partial-diffusion mode: set `partial_diffusion: True` and `contigmap.provide_seq: [ranges]` (zero-indexed over the whole flat contig sequence) to keep the original sequence of selected diffused positions - see `Examples/partial_diffusion_AF3.yaml`. With `skipRfDiff: True`, `designable_residues` defines MPNN's design set explicitly (RFDiffusion is not run).

## C2 (dimer) symmetry with `boltz_extras`
With `inference.symmetry: c2`, RFDiffusion designs the **asymmetric unit** (the contig) and adds its exact 180-degree image about the **z axis through the origin** - so the input PDB must already be in that frame (rotate/translate the structure so its biological C2 axis *is* the z axis; RFDiffusion loads the input without re-centering). The two output chains are strict C2 copies, and ProteinMPNN is run with `--homooligomer 1` (tied positions), so **both chains get the identical sequence** - the design is a symmetric homooligomer by construction.

Practical rules:
- The asymmetric-unit contig must be a **single chain** (RFDiff labels the flat output as `A` = first half, `B` = second half; a multi-chain asymmetric unit would break the labeling and the MPNN tied positions).
- Chains you want in the final complex but not in the asymmetric unit (e.g. other native subunits) are added at the Boltz stage as static `boltz_extras` proteins (`pdb_chain: ...` + an `msa` whose query equals the chain sequence) and/or as a template.
- **`c2_chain_pairs`** (top-level config, input PDB coordinates) tells the constraint remapper how the input chains pair up under C2: `c2_chain_pairs: [[C, D], [A, B]]`. References written on the image side (e.g. `[B, 23]`) are then remapped onto the image output chain at the same per-chain position (the image is an exact copy). Only needed for constraints; set it whenever `inference.symmetry: c2` and your `boltz_extras` constraints reference image-side chains.
- Prosculpt also generates a **monomer-only yaml** (first chain) per design and predicts it separately for monomer scoring (`monomer_rmsd`/`monomer_plddt` CSV columns). In that yaml, image-side constraint references and template `chain_id`/`template_id` pairs pointing at absent chains are dropped automatically, so the monomer yaml always passes Boltz validation.

See `Examples/c2_symmetry_extras_boltz.yaml` (motif scaffolding + static subunits + ligand + constraints + template) and `Examples/c2_symmetry_boltz.yaml` (unconditional C2 design).

## Examples for contigs
Contigs are the most important input (for now, guiding potentials are another input to be explored in the future). Here are a few guidelines to ease the start of your projects.
- Ranges starting with letters will be taken from the input.pdb
- Ranges without letters will generate that many residues
- `/0 ` (mind the space!) will create a chain break
- For connecting two parts, you could use a contig such as this: `'[A1-37/30-60/A42-70]'`. If the original PDB has another chain B, it will not be given to RFdiffusion in this case.
- For connecting two parts with respect to another structure: `'[C33-60/4-7/B1-30/0 C61-120]'`. Chain B is given to RFdiffusion but no new structures are generated on it. Note: in the generated structure PDB, the chains will be labeled as A (connected A and B) and B (C in original). 
- For connecting multiple chains: `'[E10-68/5-15/C4-72/5-15/D78-145/5-15/A1-60/]'`
- For leveraging symmetry: `'[10/A29-41/10/0 10/B29-41/10/0 10/C29-41/10]'`. In this case, the additional symmetry parameter must be: `symmmetry=C3` (three subunits). Also, you cannot use contigs with interval lengths to be generated e.g. `'[10-20/A29-41/10-20]'`. 
- Important: if you wish to force a number of newly generated residues, pass it as range: `[A1-7/3-3/A11-12/1-1/A14-84/1-1/A86-96]` (and not `[A1-7/3/A11-12/1/A14-84/1/A86-96]`)
- More info is available in the [RFdiff repo](https://github.com/RosettaCommons/RFdiffusion/blob/main/README.md#motif-scaffolding).

## Automatic restart 
In case of errors in any of the sub programs (RfDiffusion, ProteinMPNN, AlphaFold) prosculpt can automatically restart, so that the previous steps are not lost.
To do so, pass `auto_restart: n`, where `n` ... number of allowed restarts, to the .yaml config.

## Caveats/Troubleshooting
### Multi-model input PDBs (MODEL/ENDMDL)
Input PDBs with several MODEL/ENDMDL sections (e.g. a complex assembled from separately computed parts, one per model) are supported: RFDiffusion reads every model, and prosculpt derives the residue mappings it needs (ProteinMPNN fixed positions, `mpnn_designable_residues`, `boltz_extras` contact remapping) from RFDiffusion's own trb mapping, so the two tools agree on the residues. PDBs with CONECT records in the middle of the file (e.g. between models) are also parsed completely - Biopython would otherwise stop reading coordinates at the first CONECT record and silently drop every residue after it.

### rechain splits a designed chain into more chains than `chains_to_design` covers
rechain.py re-splits the RFDiffusion output at large CA-CA distances. If that produces more chains than `chains_to_design` lists (e.g. a contig of 4 chains comes back as 6), the trailing chain(s) are **silently dropped from the prediction input**: the MPNN fasta only holds the first `chains_to_design` chains, so Boltz/AF2 never sees them. ProSculpt now
* warns at the rechain step ("rechain produced N chains but chains_to_design covers only M"),
* still scores the predicted part (RMSD/plDDT cover the flat prefix of the design, with a warning) instead of crashing with `IndexError`.
Fixes for the next run, in order of preference:
1. Enable `boltz_full_complex: true` - Boltz then predicts the full input complex (all subunits) with the designed segments spliced in, regardless of what rechain did.
2. Add the extra chain letters to `chains_to_design` (then rename any `boltz_extras` chain ids that collide with them).
3. Lower `chain_break_cutoff_A` so rechain keeps the designed chains intact.

### Boltz predicts nothing ("ran out of memory, skipping batch")
For large complexes, Boltz can run out of GPU memory on **every** input and still exit with code 0 - the job then fails later with a `FileNotFoundError` on `output.csv` in the scoring step. ProSculpt now warns right after the `boltz predict` call ("Boltz produced no PDBs") and raises an actionable error at the end instead of the `FileNotFoundError`. Levers, roughly in order of impact:
* `--sampling_steps 300` -> `200` (memory scales with the number of diffusion steps), `--recycling_steps 10` -> `5`, drop `--use_potentials` if you don't need the extra scores.
* Use a larger GPU (an A100 80 GB handles ~2000-residue complexes comfortably; an A40 48 GB is borderline for ~1300-2100 residue complexes plus ligands).
* Run fewer jobs in parallel: each job loads PyRosetta + torch + a 10-thread Boltz dataloader; 40+ concurrent jobs exhaust host RAM (jobs get SIGKILL, exit code -9) and `/dev/shm` (torch allocation errors). Add an explicit `--mem` to the SLURM section of your `installation.yaml` and/or lower `num_seq_per_design`.

### Binder design
* In the input pdb, target must start with chain `B`, and use subsequent chain letters (C, D ...) if it has multiple chains. 
    * Otherwise, errors arise when passing into ProteinMPNN.
* Targets should not have missing residues s. t. distance between two residues would be larger than `chain_break_cutoff_A`
    * Otherwise, rechain.py will start a new chain at that point, causing errors when passing into ProteinMPNN.
    * Note that even without missing residues, calculated distance after the RfDiffusion step may sometimes be just slightly above 2 Å around prolines, so adjust accordingly. 
* The `chains_to_design` parameter should not be present at all in the .yaml config file! 
    * Otherwise, target will be fixed incorrectly and thus its aminoacids changed. 

