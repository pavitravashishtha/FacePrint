# 🔍 Face Identification & Blockchain Verification Pipeline

An end-to-end verification pipeline that takes a probe face scan as input, identifies matching content on the web/social media via live reverse image search, and verifies the discovered data using a cryptographic proof-of-work blockchain ledger.

---

## 🚀 Architecture & Pipeline Shape

```
   [ Input Image (Face Scan) ]
                │
                ▼
   ┌───────────────────────────┐
   │ 1. Face Identification    │ (face_engine.py)
   │  - Haar Cascade Detection │
   │  - 64x64 Face Crop & HOG  │
   │  - SHA-256 Fingerprinting │
   └────────────┬──────────────┘
                │
                ▼
   ┌───────────────────────────┐
   │ 2. Web / Social Search    │ (search.py)
   │  - Live Reverse Image API │
   │  - Yandex (Keyless)       │
   │  - SerpAPI / Bing Support │
   └────────────┬──────────────┘
                │
                ▼
   ┌───────────────────────────┐
   │ 3. Match Verification     │ (main.py)
   │  - Cosine Similarity Test │
   │  - Score Thresholding     │
   └────────────┬──────────────┘
                │
                ▼
   ┌───────────────────────────┐
   │ 4. Blockchain Ledger      │ (block-chain.py)
   │  - Proof-of-Work (0000)   │
   │  - SHA-256 Block Chaining │
   │  - JSON Disk Persistence  │
   │  - Re-Verification Check  │
   └───────────────────────────┘
```

---

## 📦 Components

1. **Face Identification (`face_engine.py`)**
   - Detects the largest face using OpenCV Haar Cascades (checks local `models/` directory or bundled `cv2.data.haarcascades`).
   - Crops and aligns the face to `(64, 64)`.
   - Computes a normalized 9-bin HOG (Histogram of Oriented Gradients) feature descriptor.
   - Computes a SHA-256 cryptographic image fingerprint.
   - Provides cosine similarity comparison between face vectors.

2. **Web / Social Media Reverse Search (`search.py`)**
   - **Genuine Live Search**: Executes real reverse image search against live search providers (no hardcoded/pre-picked results).
   - **Supported Providers**:
     - `Yandex Reverse Image Search`: Completely free and keyless; supports direct local file upload via multipart requests.
     - `SerpAPI (Google Lens / Reverse Image)`: Supports direct file upload or image URL (`SERPAPI_KEY`).
     - `Bing Visual Search`: Azure Cognitive Services visual search (`BING_VISUAL_SEARCH_KEY`).
     - `Auto / Chain`: Intelligently tries SerpAPI (if key present) -> Yandex (keyless fallback) -> Bing.

3. **Blockchain Verification (`block-chain.py`)**
   - **Proof-of-Work (PoW)**: Mines blocks with cryptographic difficulty target (`0000...`).
   - **Tamper-Evident Hashing**: Every block hashes `index + timestamp + data + previous_hash + nonce`.
   - **Persistence**: Persists ledger state to `chain_data.json` across restarts.
   - **Re-Verification (`verify_chain` / `verify_record`)**: Recomputes all block hashes from scratch, verifies chain linkage, and validates specific records against the on-chain ledger.

4. **Pipeline Orchestrator (`main.py`)**
   - Connects all modules into a seamless CLI execution flow.
   - Outputs step-by-step progress, similarity scores, block hashes, and verification reports.

---

## 🛠️ Installation & Setup

### 1. Prerequisites
- Python 3.8+
- OpenCV and dependencies

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

*(Optional)* If you have a SerpAPI key or Bing Azure key:
```bash
export SERPAPI_KEY="your_serpapi_key_here"
# or
export BING_VISUAL_SEARCH_KEY="your_azure_key_here"
```
*(Note: If no API keys are provided, the pipeline automatically uses the keyless Yandex reverse image provider out of the box!)*

---

## 🖥️ How to Run

### 1. Run Full End-to-End Pipeline
```bash
python main.py --image path/to/your_face.jpg
```

Specify a particular search provider:
```bash
python main.py --image path/to/your_face.jpg --provider yandex
```

Adjust similarity threshold (default `0.40`):
```bash
python main.py --image path/to/your_face.jpg --threshold 0.50
```

### 2. Verify Existing Blockchain Ledger
Recompute all cryptographic hashes and verify ledger integrity:
```bash
python main.py --verify-chain
```

### 3. Run Adversarial Tamper Detection Demo
Simulate a malicious modification of on-chain data and show cryptographic proof rejection:
```bash
python main.py --tamper-demo
```

---

## ⛓️ Which Blockchain is Used?

This project uses a **Local Proof-of-Work (PoW) Cryptographic Blockchain Ledger** stored persistently on disk in `chain_data.json`.

### Key Blockchain Characteristics:
- **Proof-of-Work Consensus**: Every block undergoes mining requiring nonces that produce a SHA-256 hash starting with `0000`.
- **Cryptographic Linkage**: Block $N$ includes the hash of Block $N-1$, ensuring past records cannot be modified without re-mining the entire chain.
- **Canonical Payload Hashing**: Records contain `{source_url, platform, probe_image_hash, similarity_score, timestamp}` hashed via SHA-256.
- **Independent Auditability**: Anyone with `chain_data.json` can run `verify_chain()` to prove data authenticity.

---

## 📹 Screen Recording Walkthrough

For the submission video recording:
1. **Show Terminal**: Run `python main.py --image sample_face.jpg`.
2. **Highlight Step 1**: Face detected and HOG vector generated with probe SHA-256 hash.
3. **Highlight Step 2 & 3**: Live reverse search executes, candidates returned, and cosine similarity calculated.
4. **Highlight Step 4**: Block is mined with Proof-of-Work and anchored into `chain_data.json`.
5. **Highlight Step 5**: Independent on-chain re-verification confirms matching record and ledger integrity.
6. **(Bonus) Run Tamper Test**: Run `python main.py --tamper-demo` to demonstrate instant detection of altered data.

---

## ⚠️ Known Limitations & Future Enhancements

1. **Face Embedder Precision**: HOG descriptors provide fast, offline, and dependency-free facial feature extraction. For large-scale production identity matching across millions of faces, deep learning embedders (e.g. `InceptionResnetV1` / `FaceNet` / `SFace`) can provide even higher discriminative power under extreme lighting variations.
2. **Search Rate Limits**: Scripted search endpoints like Yandex may present captchas if flooded with rapid concurrent queries. For high-volume production, commercial search APIs (such as SerpAPI or Google Lens Enterprise) are recommended.
3. **Consensus Scaling**: The current PoW implementation operates as an audit ledger with persistence. Pushing root Merkle hashes periodically to an EVM testnet (e.g. Base Sepolia or Polygon Amoy) provides cross-network decentralization.
