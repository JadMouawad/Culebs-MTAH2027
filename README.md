# Team Culebs: ball possession from tracking data (More Than A Hack 2027)

This repository regenerates our possession file for Barcelona vs Athletic from the provided tactical tracking (ATD).
It contains only the inference code, the frozen configurations, the model weights and the official validator.
The match data is not included and must not be committed.

## Reproduce the submission

```
python -m venv .venv
.venv\Scripts\pip install -r requirements-lock.txt          # Linux / macOS: .venv/bin/pip
set MTAH_DATA_DIR=<folder with the competition data bundle>   # Linux / macOS: export MTAH_DATA_DIR=...
.venv\Scripts\python scripts/make_submission.py --candidate shared49_robust_L
.venv\Scripts\python evaluation/validate_submission.py outputs/submission/Culebs.csv
```

- Output: `outputs/submission/Culebs.csv` (199,319 rows), plus `Culebs.manifest.json` with every input and code hash.
- Expected sha256: `7839169700c0cc0060dc6168ea5e414f486fbdfd615c025f379c519649529838`. The script refuses to write a file whose bytes differ.
- The previous model, without the added ownership and gap networks, is `--candidate x_t1_50_dead37` (sha256 `d707e942b4a1a45c038e792b19e69e363373ba1132a3159539cd1c093edb74a1`); it needs the same files.
- Input: `MTAH_DATA_DIR` must contain these two files (the organisers' tactical tracking, unmodified):
  - `tactical/BAR-ATH_tactical_atd.txt`
  - `tactical/BAR-ATH_tactical_atd.xml`
- Requirements: numpy and pandas only (pinned in `requirements-lock.txt`), CPU, no GPU. Inference takes approximately 3 minutes on the machine where we verified it (165.8 s).
- Only the weights of these two candidates are in this repository; other registered candidates cannot be regenerated from it, except the rule-based ones.

**Checking the reproduction.** The output line `OK ... sha256 <value> ... frozen match True` is the check: the file is byte-identical to the
submitted one. This public copy does not contain the team's internal approval record, so the script also prints
`NOTE NOT A FINAL RUN`. That note concerns our internal release gate only; it does not change a single byte of the file.

## What is inside

| path | what |
|---|---|
| `culebs/` | tracking parsing, features, the learned-model inference (`culebs/learned.py`, `culebs/rich49.py`, `culebs/dl/`) and the release checks |
| `submission/models/x_t1_50_dead37/` | the thirteen weight files of the previous model and their attribution |
| `submission/models/shared49_L/` | fourteen further weight files (two ownership networks, six gap specialists exported twice) and their attribution |
| `experiments/configs/` | frozen configurations, hash-checked at run time |
| `submission/contenders.json` | the registry of frozen candidates and their expected output hashes |
| `evaluation/` | the organisers' scorer and validator, unchanged |
| `TECHNICAL_DOCUMENT.md` | the technical document |

## Licences and attribution

The model weights are derived from: Bassek, M., Rein, R., Weber, H. & Memmert, D. (2025). An integrated dataset of spatiotemporal and event data in elite soccer.
Scientific Data 12. doi:10.1038/s41597-025-04505-y (CC BY 4.0, (c) DFL / Sportec Solutions), and from the SkillCorner Open Data (MIT notice reproduced; credit SkillCorner).
See `submission/models/x_t1_50_dead37/ATTRIBUTION.md` and `submission/models/shared49_L/ATTRIBUTION.md` for the changes made and `docs/SPRINT37_SKILLCORNER_LICENCE.md` for our reading of the SkillCorner terms.
Culebs code: MIT (`LICENSE`). Files taken unchanged from the organisers' repository keep their MIT licence (`LICENSE-ORGANISERS`, Metrica Sports).
