# DEFEAT: Stitching Fragmented File I/O Contexts for Early Ransomware Detection

This repository contains the code and dataset for our ACSAC 2026 paper:

> M. E. Ahmed, H. Kim, M. A. Alawami, A. Abuadbba, S. Camtepe, S. Nepal, and J. Qadir.
> **DEFEAT: Stitching Fragmented File I/O Contexts for Early Ransomware Detection.**
> In *Proceedings of the Annual Computer Security Applications Conference (ACSAC)*, 2026.

DEFEAT groups causally related Windows file I/O events into **File Event Gadgets (FEGs)**, models each FEG as an **attributed control flow graph (ACFG)** of file operations, embeds the graphs with an unsupervised graph neural network, and clusters them so analysts can label whole behaviour clusters instead of individual samples.

---

## Contents

| File | Description |
|---|---|
| `file_gadgets.json.zip` | FEGs extracted from ransomware and benign executions (compressed). |
| `enc_file_gadgets.json` | FEGs of interest from encryption-related activity. |
| `screen_locker_like_file_gadgets.json(.zip)` | FEGs from screen-locker-like samples. |
| `testdataset.json` | Held-out test set used in the evaluation. |
| `evaluate_unseen.py` | Evaluates DEFEAT on the unseen ransomware families. |
| `main_bert_file_io_patterns.py` | <!-- TODO: describe what this script does --> |

<!-- TODO: check each description above against the actual file contents. -->

## Dataset

The dataset was collected with a kernel-level ETW (Event Tracing for Windows) monitor in isolated VirtualBox VMs running Windows 10 (64-bit).

| Dataset | Samples | File I/O events | FEGs | FEGs of interest |
|---|---|---|---|---|
| Malicious | 292 | 36,727,555 | 505,203 | 19,988 |
| Benign | 46 | 43,431,886 | 51,048 | 956 |
| Unseen | 49 | 17,657,030 | 118,889 | 6,701 |
| **Total** | **387** | **97,816,471** | **675,140** | **27,548** |

The ransomware samples cover 67 families (see Table 6 in the paper's appendix), plus 24 unseen families used only for evaluation.

**What the data contains:** only kernel-level file operation metadata (event type, file key, timestamps, I/O sizes). It contains no file contents, no personally identifiable information, and no malware binaries.

<!-- TODO: if the full raw ETW logs (~38 GB) are hosted elsewhere (e.g., Zenodo or CSIRO DAP), add the link here. -->

### Data format

Each FEG is a JSON object. <!-- TODO: replace with a real example from file_gadgets.json -->

```json
{
  "feg_id": "...",
  "label": "ransomware | benign",
  "family": "Cerber",
  "events": [
    {"pid": 3496, "tid": 1512, "event": "Read", "time": 325784, "file_key": "49C160", "io_size": 4096}
  ]
}
```

## Getting started

### Requirements

- Python 3.8+
- <!-- TODO: list packages, ideally in a requirements.txt -->

```bash
git clone https://github.com/ejaz629/defeat.git
cd defeat
pip install -r requirements.txt
unzip file_gadgets.json.zip
```

### Usage

```bash
# Evaluate on unseen ransomware families
python evaluate_unseen.py
```

<!-- TODO: add the commands for training the graph embedding model and running clustering, with expected outputs. -->

## Pipeline overview

1. **FEG extraction** – group related file I/O events across files sharing an operational context.
2. **FEG-of-interest filtering** – keep FEGs with at least one write plus a read, delete, or rename.
3. **ACFG construction** – nodes are event types; edges are observed transitions.
4. **Graph embedding** – UGraphEmb with a Laplacian spectral proximity metric.
5. **Clustering** – PCA (98 → 5 dimensions) followed by HDBSCAN.
6. **Annotation** – analysts label clusters; labels propagate to all members.

## Ethics and responsible use

All ransomware samples were executed in isolated virtual machines with no network access. This repository releases only file I/O metadata and analysis code, intended for **defensive research**. It does not include malware samples.

## Citation

If you use DEFEAT or its dataset, please cite:

```bibtex
@inproceedings{ahmed2026defeat,
  title     = {{DEFEAT}: Stitching Fragmented File {I/O} Contexts for Early Ransomware Detection},
  author    = {Ahmed, Muhammad Ejaz and Kim, Hyoungshick and Alawami, Mohsen Ali and
               Abuadbba, Alsharif and Camtepe, Seyit and Nepal, Surya and Qadir, Junaid},
  booktitle = {Proceedings of the Annual Computer Security Applications Conference (ACSAC)},
  year      = {2026}
}
```

## Contact

Muhammad Ejaz Ahmed, CSIRO — ejaz.ahmed@csiro.au
