# LeafGuard

LeafGuard is a plant disease image-classification research project. It compares four frozen PlantVillage classifiers and studies whether heterogeneous MobileNetV3 and DeiT-Tiny feature fusion transfers to field-oriented imagery. The project reports controlled-dataset results separately from external-domain results and uses Grad-CAM++ for qualitative model inspection.

The CNN–Transformer combination is an engineering choice in this project, not a claim of architectural novelty. PlantVillage accuracy alone does not establish field robustness or deployment readiness.

## Dataset

The current processed PlantVillage subset contains ten classes at 224×224 input size. The documented split counts are 7,647 training, 845 validation, and 2,054 test images. Class folders and the existing held-out split are under `data/processed/{train,val,test}`.

PlantDoc is the external test source. The selected upstream revision reports 2,598 images across 13 species in its paper/README; the checked-out test split contains 236 files. The development workspace materialized 83 reliably mapped test images from nine classes; dataset images are not included in this Git repository. The upstream repository states CC BY 4.0; preserve attribution and check its `LICENSE.txt`. PlantDoc's images are internet-sourced and cropped, so they offer more image-condition variation than PlantVillage without representing an unrestricted field survey.

## Models

| Model | Checkpoint | Input processing |
|---|---|---|
| MobileNetV3-Small | `models/mobilenet_v3_best.pth` | Resize 224×224; ImageNet normalization |
| MobileNetV3-Small + CLAHE | `models/mobilenet_v3_clahe_best.pth` | Resize, LAB lightness CLAHE (clip 2.0, grid 8×8), ImageNet normalization |
| DeiT-Tiny | `models/deit_tiny_best.pth` | Resize 224×224; ImageNet normalization |
| MobileNetV3 + DeiT-Tiny | `models/hybrid_best.pth` | 576 + 192 features; 768→512→256→10 classifier |

The four trained checkpoints are included in `models/` (about 84 MiB total), so the web app can run inference immediately after installation. No training data is needed for inference. Dataset images are intentionally excluded from this repository; download and prepare datasets separately to reproduce training or evaluation.

## Environment

For a lightweight CPU web-server install, use Python 3.12 and `requirements-app.txt`. The full `requirements.txt` is for the research/evaluation environment and pins a CUDA-enabled PyTorch build; it is not required to run the web app.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-app.txt
```

For NVIDIA GPU inference, install the matching PyTorch and torchvision wheels for the machine's CUDA setup using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) instead of the CPU-wheel command, then install `requirements-app.txt`. The included model checkpoints are integrity-checked against `research/reproducibility_manifest.json`.

The Git repository intentionally excludes all dataset files under `data/`, generated Grad-CAM/example images, local accounts, logs, virtual environments, and local PDF/DOCX documents. The web app and packaged checkpoints do not depend on those datasets. To reproduce research or train models, obtain PlantVillage and PlantDoc separately, follow their respective license/attribution requirements, and use the preparation instructions below.

## Re-evaluate PlantVillage checkpoints

The verification run writes new records under `results/audit/`; it does not overwrite the legacy `results/metrics/*_test_results.json` files or `results/plots/` figures.

```powershell
python -m src.evaluation.finalize_results
```

To compute reproducible class-stratified percentile bootstrap intervals from the saved confusion matrices (10,000 resamples; fixed observed class supports), run:

```powershell
python -m src.evaluation.uncertainty
```

These intervals describe sampling variation conditional on this small mapped sample and do not establish statistical significance between models.

It checks all four checkpoints against the existing PlantVillage test split, measures batch-one inference time and parameter/file size, and generates comparison figures in `results/final_figures/`. FLOPs are estimates from the PyTorch profiler and can omit unsupported operators. Latency depends on the device and runtime.

## Cross-domain evaluation

The mapped test-only image subset is not included in this repository. For reproducibility, the source is [PlantDoc](https://github.com/pratikkayal/PlantDoc-Dataset), commit `5467f6012d78d1c446145d5f582da6096f852ae8`, licensed CC BY 4.0 by the upstream repository. Windows cannot check out several upstream paths whose filenames contain `?`; the materializer fetches only selected test blobs from a partial clone and writes Windows-safe copies. To obtain and rebuild the research subset, use:

```powershell
git clone --depth 1 --filter=blob:none --no-checkout https://github.com/pratikkayal/PlantDoc-Dataset data/external/PlantDoc-Dataset
python -m src.data.materialize_plantdoc_test --repo data/external/PlantDoc-Dataset
python -m src.data.prepare_plantdoc --source-root data/external/plantdoc_source_subset
python -m src.evaluation.domain_shift
python -m src.evaluation.finalize_results
```

The mapping is explicit in `research/class_mapping.md` and `src/data/prepare_plantdoc.py`. Unmatched or ambiguous source labels are excluded. Every external prediction is against the model's original ten outputs, so a prediction into an unmatched PlantVillage class is still an error. The mapped test set contains 83 images in nine classes; exact source and per-class counts are in `data/external/plantdoc_prepared/preparation_manifest.json`.

## Explainability

Grad-CAM++ hooks target the final MobileNet convolutional feature map and the final DeiT block's normalization output. DeiT patch tokens (excluding the class token) are reshaped to their 14×14 patch grid. The hybrid tool computes a map for each branch and averages the separately normalized maps for display. Branch maps and selected examples are written under `results/explainability/`.

```powershell
python -m src.explainability.generate_examples
```

The generated examples include a correctly classified PlantVillage case, a low-margin PlantVillage case, a PlantVillage failure, and correct/failure PlantDoc cases. PlantDoc classification data does not provide pixel masks for these samples, so localization accuracy is not reported. Grad-CAM++ is a qualitative attribution method and is not a causal explanation.

## Inference

```powershell
python -m src.inference.predict --image path/to/leaf.jpg
python -m src.inference.predict --image path/to/leaf.jpg --explanation results/explainability/prediction.png
```

The command prints the predicted class, confidence, top three classes, and checkpoint integrity status. The integrity utility compares the hybrid checkpoint SHA-256 with `configs/model_integrity.json`:

```powershell
python -m src.security.integrity
```

SHA-256 comparison detects changes against the expected value; it is not a digital signature or proof of publisher identity.

## Run the LeafGuard browser application

After installing the web dependencies above, start the app from the project root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.server:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` and create an account. The first inference request loads and verifies the four included checkpoints. User credentials and analysis history are stored in a local SQLite database; passwords are salted and PBKDF2-hashed. Browser sessions use signed, HttpOnly, SameSite cookies and CSRF tokens. Account, session, and database files are stored under `app/state/` and excluded from version control.

### Clone and start on Windows

Install Git and Python 3.12, then run these commands in PowerShell:

```powershell
git clone https://github.com/madarauchiha-095/leafguard.git
cd leafguard
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-app.txt
.\.venv\Scripts\python.exe -m uvicorn app.server:app --host 127.0.0.1 --port 8000
```

Then visit `http://127.0.0.1:8000`. The included checkpoints make model inference available without downloading datasets or connecting to PostgreSQL. SQLite is initialized automatically on first startup.

### Temporary public demo link from your laptop

The existing app serves both its frontend and API from one origin, so a separate Streamlit rewrite or PostgreSQL service is not needed to share a working demo. Streamlit Community Cloud runs the deployed app in Streamlit's hosted environment; it does not keep this Starlette server running on your laptop. Railway's free plan is usage-limited and is not a suitable always-on host for these local model checkpoints. To keep inference on the laptop, use a Cloudflare Quick Tunnel:

1. Install the Cloudflare Tunnel client once:

   ```powershell
   winget install --id Cloudflare.cloudflared -e
   ```

2. From the project root, run:

   ```powershell
   .\scripts\start_public.ps1
   ```

The script starts the app and tunnel, checks that the local app is ready, then prints the website URL. Quick Tunnel links are temporary and change each time they are restarted. The laptop must stay on, awake, and connected; stopping the script stops the website. Anyone with the URL can reach the site and create an account. Each account is limited to 120 submitted images per hour to protect the laptop's inference resources. Use a named Cloudflare Tunnel and a domain you control for a stable hostname; Quick Tunnels are intended for development and demos, not production uptime.

Public access exposes the laptop-backed service to internet traffic. Keep Windows and dependencies updated, use strong account passwords, do not expose port 8000 directly or configure router port forwarding, and stop the script when the demo is over. Analysis history contains per-account prediction metadata in the local SQLite database; uploaded image bytes are not retained. PostgreSQL is optional for this single-laptop demo and would not make the laptop server available while it is offline.

After signing in, the dashboard summarizes prior analyses and lets you revisit, export, or delete saved predictions. Analysis history stores filenames, image dimensions, SHA-256 digests, model outputs, scores, and timing in the local SQLite database; original image content is not retained. The analysis workspace accepts a folder or multiple files, streams requests in chunks of eight, and limits each account to 120 submitted images per hour. Resource safeguards cap each image at 10 MB and 20 megapixels. It reports four model predictions, class scores, model agreement, latency, class-distribution charts, and CSV/JSON exports. Grad-CAM++ explanations are generated on demand for selected images. The research pages display saved metrics, comparison charts, and confusion matrices.

Image content is processed per request and is not retained; multipart handling may use temporary spool storage while a request is active. Checkpoint SHA-256 values are checked against `research/reproducibility_manifest.json` before model loading. The local service binds to loopback by default. Before network deployment, configure HTTPS, a managed identity provider, operational rate limits, and an isolated inference boundary. Grad-CAM++ is qualitative attribution, not lesion localization; softmax scores are not calibrated probabilities.

## Training

The existing scripts are in `src/training/`. They save to the existing checkpoint paths, so back up checkpoints before intentionally running any training script. Training was not repeated for this research extension. To reproduce a training run when intentionally replacing a checkpoint:

```powershell
python -m src.training.train_mobilenet
python -m src.training.train_mobilenet_clahe
python -m src.training.train_deit_tiny
python -m src.training.train_hybrid
```

## Results and research material

- `results/final_results.json` and `results/final_results.md`: consolidated verified results and limitations.
- `results/final_figures/`: accuracy, macro F1, resource comparisons, hybrid confusion matrix, and an external-comparison placeholder until field evaluation runs.
- `results/domain_shift/`: one JSON result and confusion-matrix PNG per model after PlantDoc evaluation.
- `research/`: outline, methodology, results, discussion, and limitations.
- `research/paper.md`: full student-conference manuscript draft based on the verified results. Add author details and apply the target venue's template before submission.
- `results/explainability/`: original images, hybrid/branch Grad-CAM++ maps, overlays, and example metadata.

## Citation and limits

PlantDoc: Singh, D. et al. (2020), “PlantDoc: A Dataset for Visual Plant Disease Detection,” CoDS-COMAD 2020, [doi:10.1145/3371158.3371196](https://doi.org/10.1145/3371158.3371196). Cite the dataset repository and preserve the dataset license/attribution terms when redistributing material.

The PlantVillage test results are controlled-domain evidence. On the mapped PlantDoc subset, DeiT-Tiny scored 50.60% accuracy and 46.15% macro F1; the hybrid scored 49.40% accuracy and 42.39% macro F1. The hybrid did not lead this cross-domain comparison. Results on these benchmarks do not establish real-world deployment readiness.
