# Runtime Backup and Recovery Policy

This project is run in Google Colab, where files under `/content/` are temporary. A successful copy command alone is not proof that a persistent backup is complete.

## Backup destinations

### GitHub: source of truth for code and research records

Repository: `AbhiDevLab/Brain-Tumor`

Keep version-controlled:
- Source code, tests, environment/dependency specifications, and notebook-independent scripts.
- Research plans, dataset audit findings, preparation reports, experiment protocols, and status summaries.
- Small JSON/CSV metrics, configuration files, plot-generation scripts, and a text backup manifest.
- Model checkpoints through Git LFS or a versioned GitHub Release (never silently add large weights to ordinary Git history). Record each checkpoint's SHA-256, size, experiment settings, and matching Drive path in the manifest.

Do not commit Kaggle credentials, access tokens, passwords, or other secrets.
Do not push the raw BRISC MRI dataset into Git history. Keep the dataset in Google Drive and/or re-download it from its official Kaggle source, subject to dataset terms.

### Google Drive: persistent data and experiment-artifact backup

Use this consistent structure:

```text
MyDrive/BrainTumorResearch/
├── data/
│   ├── raw/                 # optional; only if storage permits
│   ├── processed_brisc/     # train/validation/test and masks
│   └── segmentation_unique_eval/
├── models/
│   ├── classifier/
│   └── segmenter/
├── metrics/
├── logs/
├── reports/
└── manifests/
```

The folder names can be adapted once at notebook setup, then must be reused throughout the notebook.

## Required backup checkpoints

1. **After dataset preparation:** copy the complete prepared dataset, clean segmentation evaluation set, manifest, preparation report, and duplicate-audit reports to Drive.
2. **Verify the backup:** compare source and destination counts for all class folders and image/mask folders; ensure each image has its expected mask; check file sizes and SHA-256 hashes where practical. Re-read the Drive copies and confirm expected counts. A mere `exists()` check is insufficient.
3. **Before training:** save a run configuration (Git commit SHA, seed, image size, batch size, learning rate, loss, epochs, patience, and data paths) to both Drive and the repository.
4. **During training:** save a best-validation checkpoint and a resumable/latest checkpoint periodically (at least every epoch where feasible) directly to Drive. Include optimizer state, epoch, best validation score, and RNG state when resuming exact training is supported.
5. **After each run:** save the final/best checkpoint, per-epoch history, evaluation JSON, classification report/confusion matrix or segmentation per-image metrics, plots, and a short experiment summary to Drive.
6. **Back up to GitHub:** version source changes and concise reports/metrics on a dedicated commit. Upload checkpoint weights to Git LFS or a versioned Release asset where repository permissions and quota allow. Do not assume a checkpoint is in GitHub until it can be listed/downloaded from the remote.
7. **Disconnect only after verification:** print a backup inventory with item counts, byte sizes, and hashes; confirm both the Drive artifact copy and GitHub code/artifact references exist. If any mandatory item is missing, stop and fix the backup first.

## Dataset quality and evaluation protocol

- Run the raw BRISC audit and preparation scripts before training.
- Record exact-byte duplicates across splits and duplicate images with conflicting segmentation masks.
- Create the clean segmentation evaluation set using the documented rule: keep one representative only when duplicate images have identical masks; exclude all members of a duplicate-image group when their masks conflict. Save the audit and exclusion list.
- Patient-level isolation is unconfirmed unless a verified patient identifier and split audit establish it. Exact SHA-256 duplicate checks do not rule out near-duplicates.
- Use the validation split for model selection and the clean evaluation set for final testing. Never tune hyperparameters against the test results.

## Recovery after a Colab runtime reset

1. Mount Drive.
2. Clone `AbhiDevLab/Brain-Tumor` using the Colab secret `GITHUB_TOKEN`; never print the token.
3. Check out and record the intended commit SHA.
4. Restore data from `MyDrive/BrainTumorResearch/data/`; verify its manifest and hashes before training.
5. Restore model files and metrics from Drive. If absent there, check the recorded Git LFS/Release reference and checksum.
6. Reinstall dependencies from repository files, then continue from the last verified checkpoint if the training script supports resumption. Otherwise restart only that incomplete experiment; do not retrain completed baselines unnecessarily.

## Experiment labeling

Keep these distinct:
- **Completed baseline:** trained, selected using validation data, and evaluated on the final clean test set.
- **Interrupted run:** training did not complete its planned stopping rule; any checkpoint is interim.
- **Validation metrics:** for selection/tuning only.
- **Final test metrics:** report once for the fixed clean test set and preserve the exact protocol.

Never present published-paper metrics as our own results or describe this research prototype as clinically validated.
