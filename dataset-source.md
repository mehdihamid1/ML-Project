# Dataset source

User-provided repository: https://github.com/fabriciojoc/brazilian-malware-dataset

Inspected revision: `9f0d8a65a42a87eb0944a67fecc03934bace2d20`.

The project uses `brazilian-malware.zip` → `brazilian-malware.csv`, stored
locally at `data/raw/`. Both files are excluded from Git and Docker image
builds, and are accessible in the development container through its mount.
Run `python3 scripts/fetch_dataset.py` to fetch the pinned archive and verify
its ZIP and extracted CSV checksums. An existing CSV is verified without
downloading it again.

Verified against the local files on 2026-10-05:

- ZIP SHA-256: `657136309532868b78b646b14716013d5c36681f5d4bd008536523c1912ea7b7`.
- CSV SHA-256: `d13e54cf1970ffedbf1043196c135da3187c90a82e397f681d843ed320249242`.
- 50,181 rows and 28 columns: `Label` plus 27 input columns.
- Labels: 0 = goodware (21,116 rows); 1 = malware (29,065 rows).
- Missing values: 14,223 empty `Identify` cells; no other empty cells.
- 43,411 unique `SHA1` values; 3,542 hashes appear more than once.
- 18 hashes have conflicting labels. Repeated hashes differ only in
  `FirstSeenDate` and, for those conflicting hashes, `Label`.
- Constant columns: `Magic`, `PE_TYPE`, `SizeOfOptionalHeader`.

The earlier inspection of the separate goodware and daily malware CSVs did
not describe the CSV packaged in the ZIP. The packaged CSV matches the
assignment's stated dimensions and already contains labels.

Before training, resolve conflicting labels and deduplicate or group by
`SHA1` before splitting. Exclude `SHA1` and `FirstSeenDate` from model inputs
to avoid identifier and collection-time leakage. Preserve source row identity
for batch results; fit text encoders and other learned preprocessing only
inside training folds. See [AGENTS.md](AGENTS.md) for the shared project rules.

Dataset paper: F. Ceschin et al., "The Need for Speed: An Analysis of Brazilian
Malware Classifiers," IEEE Security & Privacy, 16(6), 31–41, 2018.
https://doi.org/10.1109/MSEC.2018.2875369
