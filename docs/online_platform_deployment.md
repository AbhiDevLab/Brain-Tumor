# Online MRI Research Platform — Deployment and Artifact Setup

**Status date:** 24 October 2026  
**Application entry point:** `app.py`  
**Preferred initial host:** Streamlit Community Cloud, conditional on secure model access and available memory.

This application is a research demonstration for de-identified **2D image files**. It is not clinically validated and must not be used for medical decisions. It does not support DICOM, NIfTI, full 3D volumes, or patient records.

## Current checkpoint inventory

Model binaries are deliberately excluded from ordinary Git history. The verified best checkpoints are already attached to the private draft GitHub Release [Brain Tumor Baseline Checkpoints — 2026-10-21](https://github.com/AbhiDevLab/Brain-Tumor/releases/tag/untagged-43fd0c2fb84e8d525a8b). The original Colab/Drive paths are retained here for local research:

- Classifier: `/content/drive/MyDrive/BrainTumorResearch/models/classifier/efficientnet_b0_best_20261020T192345Z.pt`
- Classifier recovery checkpoint: corresponding `.latest.pt`
- Default segmenter: `/content/drive/MyDrive/BrainTumorResearch/models/segmenter/unet_bce_best_20261020T195800Z.pt`
- BCE + Dice research comparison: `/content/drive/MyDrive/BrainTumorResearch/models/segmenter/unet_bce_dice_best_20261020T205900Z.pt`

These paths are local to the researcher's Google Drive/Colab setup; a hosted application cannot assume that Drive is mounted. The hosted path can instead download the best assets directly from the private GitHub Release using `BRAIN_MRI_GITHUB_TOKEN`. The BCE U-Net is the platform default; the app now offers BCE + Dice as an optional comparison only, not a proven improvement.

### Research metrics recorded from the completed runs

The following results were provided by the researcher for documentation. They have **not been independently recomputed in the hosting environment**.

| Model | Evaluation set | Reported result |
|---|---|---|
| EfficientNet-B0 | Classification test set, 1,000 images | Accuracy 99.40%; macro-F1 0.993945; weighted-F1 0.994000 |
| BCE U-Net | Cleaned segmentation evaluation set, 854 image-mask pairs | Mean Dice 0.799568; mean IoU 0.718709 |
| BCE + soft Dice U-Net | Same 854 image-mask pairs | Mean Dice 0.799782; mean IoU 0.716567 |

The paired bootstrap comparison reportedly found no convincing overall BCE + Dice improvement; both 95% confidence intervals included zero. None of these metrics establishes performance on arbitrary scans, patient-independent generalization, external validation, or clinical validity.

## Configure model artifacts

Choose one of the following approaches per model.

### Option A — local or mounted checkpoint path

Set the following server-side environment variables to files the Python process can read:

- `BRAIN_MRI_CLASSIFIER_CHECKPOINT`
- `BRAIN_MRI_SEGMENTER_CHECKPOINT`

For example, in PowerShell on a machine where you have securely copied the model artifacts:

```powershell
$env:BRAIN_MRI_CLASSIFIER_CHECKPOINT = "D:\private-models\efficientnet_b0_best_20261020T192345Z.pt"
$env:BRAIN_MRI_SEGMENTER_CHECKPOINT = "D:\private-models\unet_bce_best_20261020T195800Z.pt"
streamlit run app.py
```

Do not put these files in the Git repository. A Colab `/content/drive/...` path is only valid while that Drive is actually mounted in that Colab runtime.

### Option B — existing private GitHub Release (preferred)

The trained best checkpoints already exist in the private draft release [Brain Tumor Baseline Checkpoints — 2026-10-21](https://github.com/AbhiDevLab/Brain-Tumor/releases/tag/untagged-43fd0c2fb84e8d525a8b). The application is configured to retrieve the two pinned best assets directly through the GitHub Release asset API. No second artifact store or model upload is required.

| Model | Release asset ID | Expected SHA-256 |
|---|---:|---|
| EfficientNet-B0 | `660318425` | `c0775f865d8a4811f10b50818fb44ce62d8b08c1ecb366cf317f8242eb38f9d8` |
| BCE U-Net | `660464605` | `7b7cac96ceebdccf892ecfd7bfbe36a14fdda1bb5ce8899716a1db6ed2624357` |
| BCE + soft Dice U-Net (optional) | `661318090` | `346c701760b56d27def9d779f2d6e96f740579454fcc8a2db623ccfc28edc008` |

The API returned these three best-asset IDs, names, sizes and SHA-256 digests; each digest matched the value supplied for the release. The release is still marked as a **draft**. The inference loader pins the asset IDs, expected sizes and hashes; it does not select a `.latest.pt` recovery checkpoint.

#### Hosted app setup

1. Create a **fine-grained personal access token** on GitHub. Restrict it to only the `AbhiDevLab/Brain-Tumor` repository and grant **Contents: Read-only**. The GitHub [Get a release asset API](https://docs.github.com/en/rest/releases/assets) specifies read-only Contents permission for fine-grained tokens. Never give the token write access for this app.
2. In Streamlit Community Cloud, open your app's **Settings → Secrets** and add only:
   ```toml
   BRAIN_MRI_GITHUB_TOKEN = "YOUR_FINE_GRAINED_READ_ONLY_TOKEN"
   ```
3. Save the secret and restart/redeploy the app. When a user requests inference, it downloads the required best checkpoints, validates the response size and SHA-256, and then passes them to the existing strict model loaders.
4. For GitHub Actions, the dedicated smoke-test job uses its short-lived repository `GITHUB_TOKEN` with `contents: read`; this token is not exposed to the Streamlit browser or included in the application output.

#### Redirect and integrity handling

The download first requests the authenticated API asset endpoint over HTTPS using `Accept: application/octet-stream`. GitHub documents that the endpoint may return binary bytes directly (HTTP 200) or redirect (HTTP 302). The loader follows redirects manually, validates HTTPS and an allow-list of GitHub release CDN hosts, and **never forwards the Authorization header after leaving `api.github.com`**. It then checks the pinned byte size and computes SHA-256 before saving/loading. Redirect URLs, query strings, tokens, response bodies and local file paths are not emitted in user-facing errors.

The application caches verified files in a restricted temporary directory; that cache is runtime-local and can be downloaded again after a restart. The model binaries remain private Release assets and are never committed to Git.

### Option C — alternative private HTTPS artifact endpoint

For a hosted server without Drive access, the application can retrieve each checkpoint from a private HTTPS download endpoint. The endpoint must return the raw model file to a server-side GET request authenticated by a separate, narrowly scoped **read-only Bearer token**. The URL must not contain credentials or query parameters, and the endpoint must return the model bytes directly without redirecting. Configure all six values in the host's secret manager only if you choose this alternative rather than the existing GitHub Release:

```toml
BRAIN_MRI_CLASSIFIER_URL = "https://artifacts.example.org/brain-tumor/classifier.pt"
BRAIN_MRI_CLASSIFIER_TOKEN = "REPLACE_WITH_READ_ONLY_TOKEN"
BRAIN_MRI_CLASSIFIER_SHA256 = "REPLACE_WITH_64_HEX_SHA256"

BRAIN_MRI_SEGMENTER_URL = "https://artifacts.example.org/brain-tumor/segmenter-bce.pt"
BRAIN_MRI_SEGMENTER_TOKEN = "REPLACE_WITH_READ_ONLY_TOKEN"
BRAIN_MRI_SEGMENTER_SHA256 = "REPLACE_WITH_64_HEX_SHA256"
```

The example hostname and values are placeholders, not working credentials. Obtain the private storage and token from a provider/account you control. Do not publish the real token, model URL if access-bearing, or weights. The app checks the URL scheme, uses the token only in a server-side Authorization header, limits the download to 2 GiB, verifies SHA-256 before loading, and writes the cached artifact into a restricted temporary directory. The cache is runtime-local and can be downloaded again after the host restarts. The expected hash must come from a trusted source separate from the file download.

Generate a local checksum after obtaining the original file:

```powershell
Get-FileHash .\efficientnet_b0_best_20261020T192345Z.pt -Algorithm SHA256
Get-FileHash .\unet_bce_best_20261020T195800Z.pt -Algorithm SHA256
```

Copy only the 64-character hash into the relevant secret. Restart the application after changing model secrets; the model path/resource cache is process-scoped.

Option C is only a fallback for hosts that cannot access private GitHub Release assets. It does not require moving the existing model files if you use Option B.

## Streamlit Community Cloud setup

Official documentation:
- Deploy from a GitHub repository: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- Secrets management: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management
- App sharing / private repository behaviour: https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app
- Resource limits and platform constraints: https://docs.streamlit.io/deploy/streamlit-community-cloud/status
- Upload handling in memory: https://docs.streamlit.io/knowledge-base/using-streamlit/where-file-uploader-store-when-deleted

Recommended manual steps:

1. Sign in to Streamlit Community Cloud and authorize access to the **private** `AbhiDevLab/Brain-Tumor` repository. Select branch `feature/online-mri-research-platform`, entry point `app.py`, and a supported Python version matching the dependency environment (Python 3.11 or 3.12 is a reasonable starting point).
2. Add `BRAIN_MRI_GITHUB_TOKEN` from Option B in the app's **Secrets** settings (or the six endpoint values from Option C only if you are using a separate endpoint). Do not add secrets to Git, the README, or the browser. If using mounted file paths instead, select a host that actually makes those files accessible; Community Cloud cannot mount your Google Drive by default.
3. Deploy and wait for the host to complete dependency installation and application startup.
4. Inspect runtime logs privately if startup fails. The UI intentionally does not reveal full paths, stack traces, tokens, or remote provider responses.
5. Verify both checkpoints load by their expected architecture and state dictionaries, then run inference on an approved de-identified sample image. Confirm classifier output, binary mask, aligned overlay, Grad-CAM distinction, checksum metadata and JSON download.
6. Check the live URL in a separate browser session. A private GitHub repository and a private Streamlit app are separate settings; make the app public or explicitly invite viewers depending on your sharing choice.

Community Cloud resource limits vary and may be too restrictive for PyTorch + both models, especially during startup or concurrent requests. Start on CPU, measure peak memory and cold-start latency, and move to a host with more predictable RAM/compute if the app is throttled or fails. Treat temporary filesystem storage as ephemeral; do not depend on it to persist user uploads or artifacts between restarts. HTTPS and secret management are provided by the hosting platform, but must be verified in the deployed settings.

## Runtime safety and privacy

- Upload limit: 12 MiB; accepted suffix/content pairs: PNG, JPEG, BMP and TIFF.
- Images are decoded in memory. The app does not create a permanent upload directory, image history, or patient database.
- User-provided filenames are not included in the JSON report.
- The result JSON contains pixel-based segmentation statistics, uncalibrated model scores, checkpoint names/hashes, timestamps, and review flags—not the uploaded image or personal identity fields.
- Model weights are application-controlled artifacts only. The uploaded MRI is never loaded as a checkpoint or executed as code.
- If a checkpoint is absent, corrupted, has an unexpected architecture, or fails strict state-dictionary loading, the app reports an unavailable result rather than making up a prediction.
- The consistency rules (no-tumor label with a large mask, or a tumor label with a negligible mask) are preliminary heuristics. They have not been calibrated or validated and must not be described as a reliability gate.

## Local validation before sharing

Use the same Python major/minor version as deployment:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
streamlit run app.py
```

The inference unit tests use small controlled model doubles for software-contract checks; those tests are not medical-image performance results. Final acceptance still requires the actual saved classifier checkpoint and the actual BCE U-Net checkpoint to load and produce outputs in the intended hosted runtime. Until that real-artifact check and live smoke test happen, the deployment is not considered verified.


## Hosting alternatives considered

| Host | Advantages for this repository | Trade-offs / when to choose it |
|---|---|---|
| **Streamlit Community Cloud (first candidate)** | Existing Streamlit UI can deploy directly from the private GitHub repo; app secrets and explicit viewer-sharing controls are built in. | Resource quotas can be restrictive for PyTorch and two models. Choose it only after measuring cold-start RAM, latency and actual checkpoint loading. The app may be private by default, so configure viewer access deliberately. |
| **Hugging Face Spaces (Docker/Gradio)** | Official Spaces documentation describes a 16 GB RAM / 2 CPU default hardware profile and private Spaces; GPU upgrades and mounted model/storage volumes are available on supported plans. | A first-class Gradio UI would require a frontend adaptation, or Docker packaging for Streamlit. Hardware and storage features/pricing depend on current plan. Space-local disk is ephemeral unless a supported volume is attached. |
| **Render web service** | Can run Streamlit/Python from a private Git provider connection, has managed HTTPS, environment variables/secrets, and a paid persistent disk option. This is a good alternative if hosting needs more explicit service/resource control. | A suitable always-on service and persistent disk may incur cost. Persistent disks have service/scaling constraints; model artifacts must still be stored and access-controlled separately. |

References: [Streamlit Community Cloud status and limits](https://docs.streamlit.io/deploy/streamlit-community-cloud/status), [Streamlit sharing](https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app), [Hugging Face Spaces overview](https://huggingface.co/docs/hub/spaces-overview), [Spaces storage](https://huggingface.co/docs/hub/main/spaces-storage), [Render web services](https://render.com/docs/web-services), and [Render persistent disks](https://render.com/docs/disks).

**Selection:** Start with Streamlit Community Cloud because it matches the existing app and avoids a rewrite, but treat it as conditional rather than guaranteed. If checkpoint loading, memory or cold-start behaviour exceeds its current limits, move the same inference module to a host with a documented RAM budget such as a suitably configured Render service or a Docker-based Spaces deployment. In either case, private model artifacts must not be placed in ordinary Git history or exposed through public URLs.
