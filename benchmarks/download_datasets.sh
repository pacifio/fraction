#!/usr/bin/env bash
# Download benchmark datasets for Fraction evaluation
# Datasets: LoCoMo (already downloaded), LongMemEval (cleaned version)
# Skipping: ConvoMem (25.6GB, not used by mem0, not needed for supermemory headline numbers)
#
# Usage: bash benchmarks/download_datasets.sh
# Requires: huggingface-cli (pip install huggingface_hub) OR wget + git-lfs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATASET_DIR="$SCRIPT_DIR/dataset"

mkdir -p "$DATASET_DIR"

echo "=== Fraction Benchmark Dataset Downloader ==="
echo ""
echo "Disk space available: $(df -h "$PROJECT_ROOT" | tail -1 | awk '{print $4}')"
echo "Download directory:   $DATASET_DIR"
echo ""

# ---------------------------------------------------------------------------
# 1. LoCoMo (mem0's version) — ~5MB total
# ---------------------------------------------------------------------------
echo "--- [1/2] LoCoMo Dataset ---"

LOCOMO_SRC="$PROJECT_ROOT/locomo10.json"
LOCOMO_RAG_SRC="$PROJECT_ROOT/locomo10_rag.json"
LOCOMO_DST="$DATASET_DIR/locomo10.json"
LOCOMO_RAG_DST="$DATASET_DIR/locomo10_rag.json"

if [ -f "$LOCOMO_SRC" ]; then
    if [ ! -f "$LOCOMO_DST" ] || [ "$LOCOMO_SRC" -nt "$LOCOMO_DST" ]; then
        echo "Copying locomo10.json from project root..."
        cp "$LOCOMO_SRC" "$LOCOMO_DST"
    else
        echo "locomo10.json already in dataset dir, skipping."
    fi
else
    echo "WARNING: locomo10.json not found at $LOCOMO_SRC"
    echo "  Download it manually from mem0's Google Drive:"
    echo "  https://drive.google.com/drive/folders/1pZMHSTWdvHCqMfNUL63b_z4xRwDY0Gqq"
fi

if [ -f "$LOCOMO_RAG_SRC" ]; then
    if [ ! -f "$LOCOMO_RAG_DST" ] || [ "$LOCOMO_RAG_SRC" -nt "$LOCOMO_RAG_DST" ]; then
        echo "Copying locomo10_rag.json from project root..."
        cp "$LOCOMO_RAG_SRC" "$LOCOMO_RAG_DST"
    else
        echo "locomo10_rag.json already in dataset dir, skipping."
    fi
else
    echo "WARNING: locomo10_rag.json not found at $LOCOMO_RAG_SRC"
    echo "  (Optional — only needed for RAG baseline comparison)"
fi

echo ""

# ---------------------------------------------------------------------------
# 2. LongMemEval (cleaned version) — ~3GB total
#    Source: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
#    Note: using cleaned version per upstream recommendation (original is deprecated,
#          contains noisy history sessions that interfere with answer correctness)
#
#    Files:
#      longmemeval_oracle  ~15 MB   (ground-truth oracle contexts)
#      longmemeval_s       ~278 MB  (small: 500 questions, 115K+ tokens)
#      longmemeval_m       ~2.75 GB (medium: larger context windows)
# ---------------------------------------------------------------------------
echo "--- [2/2] LongMemEval Dataset (cleaned) ---"

HF_REPO="xiaowu0162/longmemeval-cleaned"
HF_BASE_URL="https://huggingface.co/datasets/$HF_REPO/resolve/main"

LONGMEM_DIR="$DATASET_DIR/longmemeval"
mkdir -p "$LONGMEM_DIR"

download_hf_file() {
    local filename="$1"
    local dest="$LONGMEM_DIR/$filename"
    local url="$HF_BASE_URL/$filename"

    if [ -f "$dest" ]; then
        echo "  $filename already exists, skipping."
        return 0
    fi

    echo "  Downloading $filename..."
    if command -v wget &> /dev/null; then
        wget -q --show-progress -O "$dest" "$url" || {
            echo "  ERROR: Failed to download $filename"
            rm -f "$dest"
            return 1
        }
    elif command -v curl &> /dev/null; then
        curl -L --progress-bar -o "$dest" "$url" || {
            echo "  ERROR: Failed to download $filename"
            rm -f "$dest"
            return 1
        }
    else
        echo "  ERROR: Neither wget nor curl found. Install one and retry."
        return 1
    fi
}

# Always download oracle (tiny) and small (needed for benchmarks)
download_hf_file "longmemeval_oracle.json"
download_hf_file "longmemeval_s_cleaned.json"

# Medium is 2.73GB — download only if user opts in
if [ -f "$LONGMEM_DIR/longmemeval_m_cleaned.json" ]; then
    echo "  longmemeval_m_cleaned.json already exists, skipping."
elif [ "${DOWNLOAD_MEDIUM:-0}" = "1" ]; then
    download_hf_file "longmemeval_m_cleaned.json"
else
    echo "  Skipping longmemeval_m_cleaned.json (2.73 GB) — set DOWNLOAD_MEDIUM=1 to include it."
    echo "  Example: DOWNLOAD_MEDIUM=1 bash benchmarks/download_datasets.sh"
fi

echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "=== Download Summary ==="
echo ""
echo "Dataset directory: $DATASET_DIR"
echo ""

echo "Files:"
if [ -d "$DATASET_DIR" ]; then
    find "$DATASET_DIR" -type f -exec ls -lh {} \; 2>/dev/null | awk '{print "  " $NF " (" $5 ")"}'
fi

echo ""
echo "Total size:"
du -sh "$DATASET_DIR" 2>/dev/null | awk '{print "  " $1}'

echo ""
echo "=== Done ==="
echo ""
echo "Benchmarks available:"
echo "  1. LoCoMo     — 10 conversations, ~300 turns, 5 question categories"
echo "     Used by: mem0, supermemory"
echo "  2. LongMemEval — 500 questions, multi-session, 5 capability areas"
echo "     Used by: supermemory (85.86% overall)"
echo ""
echo "Not downloaded (skipped):"
echo "  - ConvoMem (Salesforce) — 25.6 GB, not used by mem0, optional for supermemory"
echo "  - longmemeval_m_cleaned.json — 2.73 GB, use DOWNLOAD_MEDIUM=1 to include"
