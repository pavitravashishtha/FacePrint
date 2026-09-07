# 🔍 FacePrint: Face Identification & Blockchain Verification Pipeline

An end-to-end verification pipeline that takes a probe face scan as input, identifies matching content on the web/social media via live reverse image search, and cryptographically verifies the discovered data using a Proof-of-Work blockchain ledger with Merkle Tree inclusion proofs.

---

## 🚀 Architecture & Pipeline Shape

```
   [ Input Image (Face Scan) ]
                │
                ▼
   ┌───────────────────────────┐
   │ 1. Face Identification    │ (face_engine.py)
   │  - Deep YuNet & Haar Det. │
   │  - 64x64 Crop & HOG Vector│
   │  - SHA-256 Fingerprinting │
   └────────────┬──────────────┘
                │
                ▼
   ┌───────────────────────────┐
   │ 2. Web / Social Search    │ (search.py)
   │  - Live Reverse Image API │
   │  - Keyless Yandex Search  │
   │  - SerpAPI / Bing Support │
   └────────────┬──────────────┘
                │
                ▼
   ┌───────────────────────────┐
   │ 3. Match Verification     │ (main.py)
   │  - Cosine Similarity Test │
   │  - Threshold Validation   │
   └────────────┬──────────────┘
                │
                ▼
   ┌───────────────────────────┐
   │ 4. Blockchain Ledger      │ (block-chain.py + merkle.py)
   │  - Proof-of-Work (0000...)│
   │  - Merkle Root & Proofs   │
   │  - SHA-256 Block Chaining │
   │  - JSON Disk Persistence  │
   │  - Re-Verification Check  │
   └───────────────────────────┘
```

---

## 📦 Core Components & Features

1. **Face Identification (`face_engine.py`)**
   - **Multi-Backend Detection**: Supports modern OpenCV YuNet Deep Learning face detection (`FaceDetectorYN`) and Haar Cascades.
   - **Feature Extraction**: 1764-dimensional L2-normalized HOG (Histogram of Oriented Gradients) descriptor.
   - **Biometric Fingerprinting**: SHA-256 cryptographic image hashing.
   - **Biometric Comparator**: Normalized cosine similarity & Euclidean distance metric.

2. **Live Web / Social Media Reverse Search (`search.py`)**
   - **Genuine Live Search**: Executes real reverse image search against live web engines (no hardcoded data).
   - **Zero-Config Keyless Default**: Yandex Reverse Image Search works out of the box with zero API keys required.
   - **Supported Engines**: SerpAPI (Google Lens), Azure Bing Visual Search, and Keyless Yandex with automatic fallback chaining.

3. **Blockchain & Merkle Ledger (`block-chain.py` + `merkle.py`)**
   - **Proof-of-Work (PoW)**: Mines blocks with cryptographic difficulty target (`0000...`).
   - **Merkle Tree Inclusion Proofs**: Computes Merkle Roots for single and batch records, providing $O(\log N)$ inclusion verification.
   - **Audit & Re-Verification**: Independent `verify_chain()` and `verify_record()` functions to detect tampering.
   - **Persistence**: Persists ledger state to `chain_data.json` across program restarts.

4. **Biometric Face Comparison Tool**
   - Direct 1-to-1 comparison between two image files with visual confidence meter.

5. **Batch Directory Ingestion**
   - Batch processes an entire directory of photos and anchors them into a single multi-record Merkle block.

---

## 🛠️ Installation & Setup

### 1. Prerequisites
- Python 3.8+
- Dependencies in `requirements.txt`

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Optional Search API Key
You can pass your `SERPAPI_KEY` in `.env` (copy from `.env.example`), or via CLI flag `--serpapi-key`. If omitted, the pipeline automatically uses keyless Yandex reverse search.

---

## 🖥️ How to Run

### 1. Run Full End-to-End Pipeline
```bash
python main.py --image sample_face.jpg
```

Specify a particular search provider:
```bash
python main.py --image sample_face.jpg --provider yandex
```

Adjust similarity threshold (default `0.40`):
```bash
python main.py --image sample_face.jpg --threshold 0.50
```

### 2. 1-to-1 Biometric Face Comparison
Compare any two faces directly without web search:
```bash
python main.py --compare sample_face.jpg sample_face.jpg
```

### 3. Batch Directory Ingestion (Merkle Tree Block)
Process an entire folder of faces into a single block:
```bash
python main.py --batch sample_faces/
```

### 4. Audit Cryptographic Merkle Proof
Verify the $O(\log N)$ Merkle proof for any discovered URL on the blockchain:
```bash
python main.py --merkle-proof <post_url>
```

### 5. Verify Blockchain Ledger Integrity
Recompute all cryptographic hashes and verify ledger integrity:
```bash
python main.py --verify-chain
```

### 6. Run Adversarial Tamper Detection Demo
Simulate a malicious alteration of on-chain records and demonstrate instant cryptographic rejection:
```bash
python main.py --tamper-demo
```

### 7. Launch Interactive Visual Web Dashboard
Launch the zero-dependency, modern dark-mode browser dashboard:
```bash
python app.py
```
*(Opens `http://127.0.0.1:5000` automatically in your browser with drag-and-drop face scanning, live search match cards, and visual blockchain/Merkle tree explorer).*

---

## ⛓️ Which Blockchain is Used?

This project uses a **Local Proof-of-Work (PoW) Cryptographic Blockchain Ledger with Merkle Trees**, stored persistently in `chain_data.json`.

### Key Blockchain Characteristics:
- **Proof-of-Work Consensus**: Every block undergoes mining requiring nonces that produce a SHA-256 hash starting with `0000`.
- **Merkle Tree Rooting**: Every block contains a Merkle Root summarizing all transactions within that block.
- **Cryptographic Linkage**: Block $N$ includes the hash of Block $N-1$, preventing alteration of past blocks without re-mining the entire chain.
- **Canonical Payload Hashing**: Records contain `{source_url, platform, probe_image_hash, similarity_score, timestamp}` hashed via SHA-256.
- **Independent Auditability**: Anyone with `chain_data.json` can run `verify_chain()` or `verify_proof()` to prove data authenticity.

---

## 📹 Screen Recording Walkthrough

For your submission video recording:
1. **Full Pipeline Run**: Run `python main.py --image sample_face.jpg`.
2. **Highlight Step 1**: Face detected, bounding box coords, and HOG vector extracted.
3. **Highlight Step 2 & 3**: Live search results returned with similarity score.
4. **Highlight Step 4**: Block mined with PoW (`0000...`) and Merkle Root generated.
5. **Highlight Step 5**: Independent re-verification confirms ledger integrity and $O(\log N)$ Merkle proof.
6. **(Bonus) 1-to-1 Comparison**: Run `python main.py --compare sample_face.jpg sample_face.jpg`.
7. **(Bonus) Tamper Detection**: Run `python main.py --tamper-demo` to demonstrate instant rejection of altered data.

---

## ⚠️ Known Limitations & Future Enhancements

1. **Face Embedder Precision**: HOG descriptors provide fast, offline, and dependency-free facial feature extraction. Deep learning embedders (such as FaceNet or SFace) can provide higher discriminative power under extreme lighting variations.
2. **Search Rate Limits**: Scripted search endpoints like Yandex may present captchas if flooded with rapid concurrent queries. For high-volume production, commercial search APIs (such as SerpAPI or Google Lens Enterprise) are recommended.
3. **Multi-Chain Finality**: The current PoW implementation operates as a local ledger with persistence. Pushing root Merkle hashes periodically to an EVM testnet (e.g. Base Sepolia or Polygon Amoy) provides cross-network decentralization.
