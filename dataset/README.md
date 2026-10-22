# Dataset Files

## Core FEG Datasets

### file_gadgets.json.zip
Complete collection of File Event Gadgets (FEGs) extracted from the study:
- **Size**: 5.4 MB (compressed), ~243 MB (uncompressed)
- **Contains**: FEGs from both ransomware and benign Windows processes
- **Format**: JSON, one FEG per line
- **Usage**: Main dataset for training graph embedding models

**To extract:**
```bash
unzip file_gadgets.json.zip
```

### enc_file_gadgets.json
Filtered subset of FEGs of particular interest - encryption-related file operations:
- **Size**: 296 KB
- **Contains**: FEGs with write operations followed by read/delete/rename
- **Format**: JSON
- **Usage**: Focused analysis on encryption-like behavior patterns

### screen_locker_like_file_gadgets.json.zip
FEGs extracted specifically from screen-locker ransomware samples:
- **Size**: 27 KB (compressed)
- **Contains**: File gadgets from screen-locker behavior patterns
- **Format**: JSON
- **Usage**: Analysis of display-manipulation ransomware families

## Graph Embeddings

These are pre-computed graph neural network embeddings for efficient analysis:

### cln_graphs_features.json
Graph embeddings for clean/benign samples:
- **Size**: 19 MB
- **Dimensions**: Features from UGraphEmb embedding model
- **Samples**: All benign FEGs
- **Usage**: Baseline embeddings for classification and clustering

### mal_graphs_features.json
Graph embeddings for malware samples:
- **Size**: 212 MB
- **Dimensions**: Features from UGraphEmb embedding model
- **Samples**: All malicious FEGs from known ransomware families
- **Usage**: Malware behavior clustering and analysis

### unseen_graphs_features.json
Graph embeddings for unseen ransomware families:
- **Size**: 51 MB
- **Dimensions**: Features from UGraphEmb embedding model
- **Samples**: FEGs from 24 unseen ransomware families
- **Usage**: Evaluation on novel/unknown threats

## Evaluation Dataset

### testdataset.json
Held-out test set for evaluation:
- **Size**: 164 KB
- **Contains**: Balanced samples from training and unseen families
- **Format**: JSON
- **Usage**: Benchmark evaluation of the DEFEAT model

---

## Data Format Example

Each FEG is a JSON object with the following structure:

```json
{
  "feg_id": "unique_identifier",
  "label": "ransomware",
  "family": "Cerber",
  "events": [
    {
      "pid": 3496,
      "tid": 1512,
      "event": "Read",
      "time": 325784,
      "file_key": "49C160",
      "io_size": 4096
    },
    {
      "pid": 3496,
      "tid": 1512,
      "event": "Write",
      "time": 325800,
      "file_key": "49C160",
      "io_size": 8192
    }
  ]
}
```

### Field Descriptions
- **feg_id**: Unique identifier for this FEG
- **label**: "ransomware" or "benign"
- **family**: Ransomware family name (or "clean" for benign)
- **events**: Array of file I/O events
  - **pid**: Process ID
  - **tid**: Thread ID
  - **event**: Type of operation (Read, Write, Delete, Rename, etc.)
  - **time**: Timestamp (in milliseconds)
  - **file_key**: Internal identifier for the file
  - **io_size**: Number of bytes transferred

---

## Dataset Statistics

| Category | Count |
|---|---|
| Total samples executed | 387 |
| Malicious samples | 292 |
| Benign samples | 46 |
| Unseen families (evaluation) | 49 |
| Total FEGs generated | 675,140 |
| FEGs of interest (filtered) | 27,548 |
| Known ransomware families | 67 |
| Unknown families (test) | 24 |
| Total file I/O events | 97,816,471 |

---

## Privacy & Ethics

- **No file contents**: Only metadata about file operations
- **No binaries**: No malware samples included
- **No PII**: No personally identifiable information
- **Sandbox only**: All execution was in isolated virtual machines
- **Research use**: Intended for defensive security research
