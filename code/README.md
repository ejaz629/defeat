# DEFEAT Analysis Code

This directory contains Python scripts for training, evaluating, and analyzing the DEFEAT model for ransomware detection using file I/O behavior.

## Scripts

### evaluate_unseen.py
Evaluates the DEFEAT model on unseen ransomware families.

**Purpose**: 
- Tests the generalization capability of the model
- Evaluates clustering quality on 24 unknown ransomware families
- Generates performance metrics on held-out test data

**Usage**:
```bash
python evaluate_unseen.py
```

**Expected Inputs**:
- `../dataset/unseen_graphs_features.json` - Graph embeddings for unseen samples
- `../dataset/testdataset.json` - Test set with labels

**Output**:
- Classification metrics (precision, recall, F1-score)
- Clustering evaluation (silhouette score, homogeneity, etc.)
- Performance statistics on unseen families

**Key Functions**:
- Loads pre-computed graph embeddings
- Applies PCA dimensionality reduction (98 → 5 dimensions)
- Performs HDBSCAN clustering
- Evaluates against ground truth labels

---

### main_bert_file_io_patterns.py
Graph embedding and pattern analysis for file I/O behavior.

**Purpose**:
- Extracts file I/O patterns from FEGs
- Trains or applies graph neural network embeddings
- Analyzes behavioral patterns in ransomware vs. benign processes

**Usage**:
```bash
python main_bert_file_io_patterns.py
```

**Expected Inputs**:
- `../dataset/file_gadgets.json` - Raw FEGs (or `file_gadgets.json.zip` if uncompressed first)
- Configuration parameters (embedding model, dimensions, etc.)

**Output**:
- Trained embedding model (if training mode)
- Graph embeddings for all FEGs
- Pattern analysis and visualization data
- Cluster assignments

**Key Functions**:
- FEG to ACFG (Attributed Control Flow Graph) conversion
- Graph feature extraction
- UGraphEmb model training/inference
- Dimensionality reduction with PCA
- Clustering with HDBSCAN
- Behavioral pattern analysis

---

## Pipeline Overview

```
Raw ETW Logs
      ↓
FEG Extraction (groups related I/O events)
      ↓
ACFG Construction (control flow graph of operations)
      ↓
Graph Embedding (UGraphEmb unsupervised model)
      ↓
Dimensionality Reduction (PCA: 98 → 5 dimensions)
      ↓
Clustering (HDBSCAN)
      ↓
Annotation (analysts label clusters)
      ↓
Early Ransomware Detection
```

---

## Requirements

- Python 3.8+
- NumPy
- Pandas
- scikit-learn
- networkx (for graph operations)
- hdbscan (for clustering)

Install dependencies:
```bash
pip install numpy pandas scikit-learn networkx hdbscan
```

---

## Configuration

Both scripts support configuration through:
- Command-line arguments
- Configuration files
- Hardcoded defaults

Common parameters:
- `--embedding-dim`: Target embedding dimension (default: 98)
- `--pca-dim`: PCA reduction target (default: 5)
- `--min-cluster-size`: HDBSCAN parameter
- `--input-path`: Path to input FEG data
- `--output-path`: Where to save results

---

## Output Files

Scripts generate the following artifacts:

- `.json` files: Graph embeddings and cluster assignments
- `.pkl` files: Trained models (if training mode)
- `.csv` files: Performance metrics and analysis results
- Console output: Real-time progress and logging

---

## Troubleshooting

**Memory issues with large datasets**:
- Process files in batches
- Use pre-computed embeddings instead of training
- Consider using `.zip` files and processing in chunks

**Missing dependencies**:
```bash
pip install --upgrade numpy pandas scikit-learn networkx hdbscan
```

**File not found errors**:
- Ensure dataset files are extracted from `.zip` archives
- Check relative paths from the project root
- Verify file permissions

---

## Performance Notes

- **FEG Processing**: ~1 million FEGs per hour (CPU dependent)
- **Graph Embedding**: ~500K embeddings per hour
- **Clustering**: Real-time, scales to 100K+ samples
- **Memory**: ~8GB for full dataset processing

---

## References

For implementation details, see the paper:
> M. E. Ahmed et al. "DEFEAT: Stitching Fragmented File I/O Contexts for Early Ransomware Detection." ACSAC 2026.
