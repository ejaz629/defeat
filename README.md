# DEFEAT Dataset: Fragmented File I/O Contexts for Ransomware Detection

This repository contains the dataset for our ACSAC 2026 paper:

> M. E. Ahmed, H. Kim, M. A. Alawami, A. Abuadbba, S. Camtepe, S. Nepal, and J. Qadir.
> **DEFEAT: Stitching Fragmented File I/O Contexts for Early Ransomware Detection.**
> In *Proceedings of the Annual Computer Security Applications Conference (ACSAC)*, 2026.

The dataset captures Windows file I/O behaviour of ransomware and benign applications, organised into **File Event Gadgets (FEGs)**: groups of related file I/O events that span multiple files sharing a common operational context. It supports research on behaviour-based ransomware detection, file I/O analysis, and graph-based malware modelling.

---

## Contents

| File | Description |
|---|---|
| `file_gadgets.json.zip` | FEGs extracted from ransomware and benign executions. |
| `enc_file_gadgets.json` | FEGs of interest from encryption-related activity. |
| `screen_locker_like_file_gadgets.json.zip` | FEGs from screen-locker-like samples. |
| `testdataset.json` | Held-out test set used in the evaluation. |
| `dataset/mal_graphs_features.zip` | Graph features of malicious FEGs (attributed control flow graphs). |
| `dataset/unseen_graphs_features.zip` | Graph features of FEGs from unseen ransomware families. |

<!-- TODO: check each description against the actual file contents, and remove rows for files not in the repository. -->

Unzip the compressed files before use:

```bash
unzip file_gadgets.json.zip
unzip dataset/mal_graphs_features.zip -d dataset/
```

## Collection

The data was collected with a kernel-level monitor built on Event Tracing for Windows (ETW), in isolated VirtualBox virtual machines running Windows 10 (64-bit) with no network access. Each ransomware sample ran for ten minutes or until all user files were encrypted. Benign traces were collected on real user environments while running common applications, including encryption and compression tools that produce ransomware-like file I/O.

| Dataset | Samples | File I/O events | FEGs | FEGs of interest |
|---|---|---|---|---|
| Malicious | 292 | 36,727,555 | 505,203 | 19,988 |
| Benign | 46 | 43,431,886 | 51,048 | 956 |
| Unseen | 49 | 17,657,030 | 118,889 | 6,701 |
| **Total** | **387** | **97,816,471** | **675,140** | **27,548** |

The malicious set covers 67 ransomware families; the unseen set covers 24 further families held out from training. The full family and application lists are in Appendix B of the paper.

## Data format

Each FEG is a JSON object containing the related files and their ordered file I/O events. Event fields follow the ETW file I/O schema: process and thread IDs, timestamp, event type (`Create`, `FileCreate`, `Read`, `Write`, `Rename`, `Delete`, `FileDelete`), file key, file object, and I/O size.

<!-- TODO: paste a short real example from file_gadgets.json here. -->

## What the data does and does not contain

The dataset contains **only file operation metadata** (event types, file keys, timestamps, I/O sizes). It contains **no file contents, no personally identifiable information, and no malware binaries**.

## Responsible use

This dataset is released for **defensive research**, such as developing and evaluating ransomware detection methods. Please do not use it to develop or improve malicious software.

## Citation

If you use this dataset, please cite:

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

Muhammad Ejaz Ahmed, CSIRO, ejaz.ahmed@csiro.au
