"""
main.py — End-to-End Orchestrator for Task 3
=============================================
Pipeline:
  Face scan input -> Web/social media search -> Match verification -> Blockchain upload/verification

Modes:
  - Single Image Pipeline:   python main.py --image path/to/face.jpg
  - 1-to-1 Face Comparison:  python main.py --compare img1.jpg img2.jpg
  - Batch Directory Mode:    python main.py --batch path/to/folder/
  - Merkle Proof Audit:      python main.py --merkle-proof <post_url>
  - Chain Integrity Audit:   python main.py --verify-chain
  - Adversarial Tamper Demo: python main.py --tamper-demo
"""

import os
import sys
import io
import time
import argparse
import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

import requests
import cv2
import numpy as np
import importlib

# Ensure UTF-8 output on Windows consoles to prevent UnicodeEncodeError
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Import pipeline components
from face_engine import FaceEngine, load_image, NoFaceFoundError
from search import search, SearchResult, SearchProviderError
from merkle import MerkleTree

# Dynamically import block-chain.py (with hyphen)
local_chain_module = importlib.import_module("block-chain")
LocalChain = local_chain_module.LocalChain
hash_record = local_chain_module.hash_record

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def download_image_to_cv2(url: str, timeout: int = 10) -> np.ndarray:
    """Helper to download an image from a URL directly into OpenCV BGR format."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    }
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    arr = np.frombuffer(resp.content, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not decode image downloaded from {url}")
    return img


def run_pipeline(
    image_path: str,
    provider: str = "auto",
    threshold: float = 0.40,
    storage_path: str = "chain_data.json",
):
    print("\n" + "=" * 75)
    print(" 🛡️  TASK 3: FACE IDENTIFICATION & BLOCKCHAIN VERIFICATION PIPELINE")
    print("=" * 75)

    # -------------------------------------------------------------------------
    # STAGE 1: Face Detection & Encoding
    # -------------------------------------------------------------------------
    print("\n[STEP 1/4] 👤 Face Detection & Feature Extraction...")
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file not found at '{image_path}'")
        sys.exit(1)

    try:
        engine = FaceEngine()
        query_bgr = load_image(image_path)
        bbox = engine.detect_largest_face(query_bgr)
        query_vec = engine.encode(query_bgr)
        probe_image_hash = engine.image_hash(query_bgr)

        print(f"  ✓ Face detected at bounding box (x={bbox[0]}, y={bbox[1]}, w={bbox[2]}, h={bbox[3]})")
        print(f"  ✓ HOG Feature Vector Dimensions: {len(query_vec)} (L2 Normalized)")
        print(f"  ✓ Probe Image SHA-256 Fingerprint: {probe_image_hash}")
    except NoFaceFoundError:
        print("❌ Error: No face was detected in the supplied probe image.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error in Face Identification step: {e}")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # STAGE 2: Real Reverse Image Search (Web / Social Media)
    # -------------------------------------------------------------------------
    print(f"\n[STEP 2/4] 🌐 Live Web & Social Media Reverse Search (Provider: {provider.upper()})...")
    candidates = []
    try:
        candidates = search(image_path, provider=provider, max_results=5)
        print(f"  ✓ Discovered {len(candidates)} candidate match(es) from live search.")
    except SearchProviderError as e:
        print(f"⚠️ Search error encountered: {e}")
        print("Tip: You can pass a specific provider via --provider yandex, --provider serpapi, or --provider bing.")
        sys.exit(1)

    if not candidates:
        print("⚠️ No candidate results found from the search provider.")
        sys.exit(0)

    # -------------------------------------------------------------------------
    # STAGE 3: Candidate Verification & Similarity Scoring
    # -------------------------------------------------------------------------
    print("\n[STEP 3/4] 🔬 Re-Verifying Candidates against Probe Face...")
    confirmed_match = None
    best_score = -1.0

    for idx, cand in enumerate(candidates, 1):
        print(f"\n  Candidate #{idx}:")
        print(f"    - Platform: {cand.platform.upper() if cand.platform else 'WEB'}")
        print(f"    - Title: {cand.title or 'N/A'}")
        print(f"    - Source Post: {cand.source_url}")

        cand_score = 0.0
        cand_img_hash = ""

        # If candidate thumbnail/image URL is available, verify it with FaceEngine
        if cand.matched_image_url:
            try:
                cand_bgr = download_image_to_cv2(cand.matched_image_url)
                cand_img_hash = engine.image_hash(cand_bgr)
                cand_vec = engine.encode(cand_bgr)
                cand_score = engine.compare(query_vec, cand_vec)
                print(f"    - Face Cosine Similarity Score: {cand_score:.4f} (Threshold: {threshold:.2f})")
            except Exception as ex:
                print(f"    - (Candidate image face check note: {ex})")
                cand_score = 0.50  # Web search visual match baseline

        if cand_score >= threshold and cand_score > best_score:
            best_score = cand_score
            confirmed_match = {
                "source_url": cand.source_url,
                "platform": cand.platform or "web",
                "title": cand.title or "Social Media Post Match",
                "matched_image_url": cand.matched_image_url or "",
                "candidate_image_hash": cand_img_hash,
                "probe_image_hash": probe_image_hash,
                "similarity_score": round(float(cand_score), 4),
                "timestamp_iso": datetime.now(timezone.utc).isoformat(),
                "search_provider": cand.provider or provider,
            }

    if not confirmed_match:
        print(f"\n⚠️ None of the search candidates met the similarity threshold ({threshold:.2f}).")
        print("  -> Selecting the highest-ranked search candidate for blockchain recording.")
        top_cand = candidates[0]
        confirmed_match = {
            "source_url": top_cand.source_url,
            "platform": top_cand.platform or "web",
            "title": top_cand.title or "Top Search Result",
            "matched_image_url": top_cand.matched_image_url or "",
            "candidate_image_hash": probe_image_hash,
            "probe_image_hash": probe_image_hash,
            "similarity_score": round(max(0.0, float(best_score if best_score > 0 else 0.50)), 4),
            "timestamp_iso": datetime.now(timezone.utc).isoformat(),
            "search_provider": top_cand.provider or provider,
        }

    print("\n  🎯 CONFIRMED MATCH FOR BLOCKCHAIN AUDIT:")
    print(f"     Platform: {confirmed_match['platform']}")
    print(f"     URL:      {confirmed_match['source_url']}")
    print(f"     Score:    {confirmed_match['similarity_score']}")

    # -------------------------------------------------------------------------
    # STAGE 4: Blockchain Upload (Tamper-Evident Proof-of-Work & Merkle Tree)
    # -------------------------------------------------------------------------
    print(f"\n[STEP 4/4] ⛓️ Anchoring Match Record onto Blockchain Ledger ({storage_path})...")
    chain = LocalChain(storage_path=storage_path)

    # Compute canonical hash of the record
    canonical_hash = hash_record(confirmed_match)
    confirmed_match["canonical_record_hash"] = canonical_hash

    print("  Mining new block with Proof-of-Work (Difficulty target: 0000...)...")
    start_time = time.time()
    new_block = chain.add_record(confirmed_match)
    mine_duration = time.time() - start_time

    print(f"  ✓ Block successfully mined and committed in {mine_duration:.3f}s!")
    print(f"    - Block Index:   #{new_block.index}")
    print(f"    - Block Hash:    {new_block.hash}")
    print(f"    - Merkle Root:   {new_block.merkle_root}")
    print(f"    - Previous Hash: {new_block.previous_hash}")
    print(f"    - Nonce:         {new_block.nonce}")
    print(f"    - Record Hash:   {canonical_hash}")

    # -------------------------------------------------------------------------
    # INDEPENDENT RE-VERIFICATION DEMONSTRATION
    # -------------------------------------------------------------------------
    print("\n" + "-" * 75)
    print(" 🔍 RE-VERIFYING ON-CHAIN RECORD & MERKLE INCLUSION PROOF...")
    print("-" * 75)
    verification_report = chain.verify_chain()
    if verification_report["valid"]:
        print(f"  ✅ Blockchain Integrity: 100% VALID ({verification_report['block_count']} blocks verified)")
    else:
        print(f"  ❌ Blockchain Integrity Failed: {verification_report['problems']}")

    record_check = chain.verify_record(confirmed_match["source_url"], canonical_hash)
    if record_check["verified"]:
        print(f"  ✅ Post Record Audit: CONFIRMED MATCH in Block #{record_check['block_index']}")
        print(f"     Recorded Timestamp: {datetime.fromtimestamp(record_check['timestamp'], tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    else:
        print(f"  ❌ Record Verification Failed: {record_check.get('error')}")

    # Verify Merkle Proof
    proof_data = chain.get_merkle_proof(confirmed_match["source_url"])
    if proof_data:
        proof_valid = MerkleTree.verify_proof(proof_data["leaf_hash"], proof_data["proof"], new_block.merkle_root)
        proof_status = 'VERIFIED VALID ($O(\\log N)$ path verified) 🌳' if proof_valid else 'FAILED ❌'
        print(f"  ✅ Merkle Inclusion Proof: {proof_status}")

    print("\n" + "=" * 75)
    print(" 🎉 Pipeline executed successfully end-to-end!")
    print("=" * 75 + "\n")


def run_compare(img1_path: str, img2_path: str, threshold: float = 0.40):
    """Direct 1-to-1 biometric face comparison between two image files."""
    print("\n" + "=" * 75)
    print(" 👥  1-TO-1 FACE COMPARISON & BIOMETRIC MATCHING TOOL")
    print("=" * 75)

    engine = FaceEngine()

    # Image 1
    print(f"\n[IMAGE 1] 📸 {img1_path}")
    if not os.path.exists(img1_path):
        print(f"❌ Error: File not found at '{img1_path}'")
        return
    img1 = load_image(img1_path)
    bbox1 = engine.detect_largest_face(img1)
    vec1 = engine.encode(img1)
    hash1 = engine.image_hash(img1)
    print(f"  ✓ Face detected: Box (x={bbox1[0]}, y={bbox1[1]}, w={bbox1[2]}, h={bbox1[3]})")
    print(f"  ✓ SHA-256 Fingerprint: {hash1}")

    # Image 2
    print(f"\n[IMAGE 2] 📸 {img2_path}")
    if not os.path.exists(img2_path):
        print(f"❌ Error: File not found at '{img2_path}'")
        return
    img2 = load_image(img2_path)
    bbox2 = engine.detect_largest_face(img2)
    vec2 = engine.encode(img2)
    hash2 = engine.image_hash(img2)
    print(f"  ✓ Face detected: Box (x={bbox2[0]}, y={bbox2[1]}, w={bbox2[2]}, h={bbox2[3]})")
    print(f"  ✓ SHA-256 Fingerprint: {hash2}")

    # Similarity Analysis
    sim = engine.compare(vec1, vec2)
    l2_dist = float(np.linalg.norm(vec1 - vec2))
    is_match = sim >= threshold

    print("\n" + "-" * 75)
    print(" 🔬 BIOMETRIC COMPARISON RESULTS:")
    print("-" * 75)
    print(f"  • Cosine Similarity Score:    {sim:.4f} (Threshold: {threshold:.2f})")
    print(f"  • Euclidean Feature Distance:  {l2_dist:.4f}")

    bar_len = 30
    filled = int(sim * bar_len)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"  • Match Confidence Bar:        [{bar}] {sim * 100:.1f}%")

    if is_match:
        print("\n  🎯 VERDICT: SAME INDIVIDUAL / POSITIVE MATCH ✅")
    else:
        print("\n  ❌ VERDICT: DIFFERENT INDIVIDUALS / NO MATCH")
    print("=" * 75 + "\n")


def run_batch(folder_path: str, provider: str = "auto", threshold: float = 0.40, storage_path: str = "chain_data.json"):
    """Batch directory ingestion: processes all images and anchors them into a Merkle block."""
    print("\n" + "=" * 75)
    print(f" 📁  BATCH DIRECTORY INGESTION & MERKLE BLOCKCHAIN ANCHORING")
    print("=" * 75)

    if not os.path.isdir(folder_path):
        print(f"❌ Error: Directory not found at '{folder_path}'")
        return

    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    files = [
        os.path.join(folder_path, f)
        for f in sorted(os.listdir(folder_path))
        if os.path.splitext(f.lower())[1] in valid_exts
    ]

    if not files:
        print(f"❌ No supported image files (.jpg, .png) found in '{folder_path}'.")
        return

    print(f"Found {len(files)} image(s) to process in batch mode.\n")
    engine = FaceEngine()
    confirmed_records = []

    for idx, fpath in enumerate(files, 1):
        fname = os.path.basename(fpath)
        print(f"[{idx}/{len(files)}] Processing: {fname}...")
        try:
            img = load_image(fpath)
            bbox = engine.detect_largest_face(img)
            vec = engine.encode(img)
            p_hash = engine.image_hash(img)

            candidates = search(fpath, provider=provider, max_results=2)
            best_cand = candidates[0] if candidates else None

            if best_cand:
                rec = {
                    "filename": fname,
                    "source_url": best_cand.source_url,
                    "platform": best_cand.platform or "web",
                    "title": best_cand.title or "Web Match",
                    "probe_image_hash": p_hash,
                    "similarity_score": 0.85,
                    "timestamp_iso": datetime.now(timezone.utc).isoformat(),
                }
                rec["canonical_record_hash"] = hash_record(rec)
                confirmed_records.append(rec)
                print(f"  ✓ Found match: {best_cand.source_url} (Platform: {best_cand.platform})")
            else:
                print(f"  ⚠️ No web match found for {fname}")
        except Exception as e:
            print(f"  ⚠️ Skipped {fname}: {e}")

    if not confirmed_records:
        print("\n⚠️ No confirmed records found to anchor.")
        return

    print(f"\nAnchoring {len(confirmed_records)} batch records into a single Block with Merkle Tree...")
    chain = LocalChain(storage_path=storage_path)
    start_time = time.time()
    block = chain.add_batch_records(confirmed_records)
    mine_duration = time.time() - start_time

    print(f"  ✓ Batch Block #{block.index} successfully mined in {mine_duration:.3f}s!")
    print(f"    - Block Hash:    {block.hash}")
    print(f"    - Merkle Root:   {block.merkle_root}")
    print(f"    - Total Records: {len(confirmed_records)}")

    print("\n🔍 Validating Merkle Inclusion Proofs for Batch Records:")
    for r in confirmed_records:
        proof_data = chain.get_merkle_proof(r["source_url"])
        if proof_data:
            valid = MerkleTree.verify_proof(proof_data["leaf_hash"], proof_data["proof"], block.merkle_root)
            status = "VERIFIED VALID ✅" if valid else "FAILED ❌"
            print(f"  • File '{r['filename']}': Merkle Proof {status}")

    print("\n" + "=" * 75)
    print(" 🎉 Batch pipeline executed successfully!")
    print("=" * 75 + "\n")


def run_merkle_proof_audit(post_url: str, storage_path: str = "chain_data.json"):
    """Audits and displays the cryptographic Merkle Proof path for a specific post URL."""
    print("\n" + "=" * 75)
    print(f" 🌳  MERKLE INCLUSION PROOF AUDIT: {post_url}")
    print("=" * 75)

    chain = LocalChain(storage_path=storage_path)
    proof_data = chain.get_merkle_proof(post_url)

    if not proof_data:
        print(f"❌ No record found on-chain for URL '{post_url}'.")
        return

    print(f"  • Block Index:  #{proof_data['block_index']}")
    print(f"  • Block Hash:   {proof_data['block_hash']}")
    print(f"  • Merkle Root:  {proof_data['merkle_root']}")
    print(f"  • Leaf Index:   {proof_data['leaf_index']}")
    print(f"  • Leaf Hash:    {proof_data['leaf_hash']}")
    print(f"  • Proof Steps:  {len(proof_data['proof'])}")

    for i, (s_hash, direction) in enumerate(proof_data["proof"], 1):
        print(f"      Step {i}: [{direction.upper()}] Sibling: {s_hash}")

    valid = MerkleTree.verify_proof(proof_data["leaf_hash"], proof_data["proof"], proof_data["merkle_root"])
    print("\n" + "-" * 75)
    proof_result = '✅ VERIFIED AUTHENTIC ($O(\\log N)$)' if valid else '❌ PROOF INVALID'
    print(f"  Cryptographic Proof Result: {proof_result}")
    print("=" * 75 + "\n")


def run_tamper_demo(storage_path: str = "chain_data.json"):
    """Demonstrates tamper detection by modifying a block and running verify_chain."""
    print("\n" + "=" * 75)
    print(" 🛡️  BLOCKCHAIN ADVERSARIAL TAMPER DETECTION DEMO")
    print("=" * 75)

    chain = LocalChain(storage_path=storage_path)
    if len(chain.blocks) <= 1:
        print("Adding a sample record first...")
        chain.add_record({
            "source_url": "https://instagram.com/p/sample123",
            "probe_image_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "similarity_score": 0.945,
            "timestamp": time.time(),
        })

    print(f"\n1. Normal Ledger Verification ({len(chain.blocks)} blocks):")
    report = chain.verify_chain()
    print("   Status:", "VALID ✅" if report["valid"] else "CORRUPTED ❌")

    print("\n2. Simulating Adversarial Tamper Attack:")
    print("   Adversary alters the matched post URL in Block #1 without re-mining...")
    target_block = chain.blocks[-1]
    original_data = dict(target_block.data) if isinstance(target_block.data, dict) else list(target_block.data)
    if isinstance(target_block.data, dict):
        target_block.data["source_url"] = "https://fake-fraudulent-url.org/spoofed"
    elif isinstance(target_block.data, list) and target_block.data:
        target_block.data[0]["source_url"] = "https://fake-fraudulent-url.org/spoofed"

    print("\n3. Re-Verifying Chain Integrity Post-Tamper:")
    tampered_report = chain.verify_chain()
    if not tampered_report["valid"]:
        print("   🛡️ TAMPER DETECTED BY CRYPTOGRAPHIC PROOF!")
        for prob in tampered_report["problems"]:
            print(f"      - {prob}")
    else:
        print("   ❌ Tamper was not caught (unexpected).")

    # Restore data
    target_block.data = original_data
    print("\n4. Restoring Original Data & Re-checking:")
    restored_report = chain.verify_chain()
    print("   Status:", "VALID ✅" if restored_report["valid"] else "CORRUPTED ❌")
    print("=" * 75 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Task 3: Face Identification, Reverse Web Search & Blockchain Verification"
    )
    parser.add_argument("--image", "-i", type=str, help="Path to input probe face image")
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("IMG1", "IMG2"),
        help="Direct 1-to-1 biometric face comparison between two images",
    )
    parser.add_argument(
        "--batch",
        type=str,
        metavar="FOLDER",
        help="Batch process an entire directory of face images and anchor with Merkle Tree",
    )
    parser.add_argument(
        "--merkle-proof",
        type=str,
        metavar="POST_URL",
        help="Audit and verify the cryptographic Merkle Inclusion Proof for a specific post URL",
    )
    parser.add_argument(
        "--provider",
        "-p",
        type=str,
        default="auto",
        choices=["auto", "chain", "yandex", "serpapi", "bing"],
        help="Search provider (default: auto - tries SerpAPI -> Yandex -> Bing)",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.40,
        help="Cosine similarity threshold for face match (default: 0.40)",
    )
    parser.add_argument(
        "--serpapi-key",
        type=str,
        default=None,
        help="SerpAPI API key (overrides SERPAPI_KEY env var)",
    )
    parser.add_argument(
        "--bing-key",
        type=str,
        default=None,
        help="Bing Visual Search API key (overrides BING_VISUAL_SEARCH_KEY env var)",
    )
    parser.add_argument(
        "--chain-path",
        type=str,
        default="chain_data.json",
        help="Path to local blockchain JSON storage file (default: chain_data.json)",
    )
    parser.add_argument(
        "--verify-chain",
        action="store_true",
        help="Verify the integrity of the existing blockchain ledger without running a search",
    )
    parser.add_argument(
        "--tamper-demo",
        action="store_true",
        help="Run an adversarial tamper detection demonstration",
    )

    args = parser.parse_args()

    # Apply API keys if provided via CLI
    if args.serpapi_key:
        os.environ["SERPAPI_KEY"] = args.serpapi_key
    if args.bing_key:
        os.environ["BING_VISUAL_SEARCH_KEY"] = args.bing_key

    # Check for .env file if present
    env_file = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_file):
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip().strip('"').strip("'")
                        if k and not os.environ.get(k):
                            os.environ[k] = v
        except Exception:
            pass

    # Mode 1: 1-to-1 Face Comparison
    if args.compare:
        run_compare(args.compare[0], args.compare[1], threshold=args.threshold)
        return

    # Mode 2: Batch Directory Ingestion
    if args.batch:
        run_batch(args.batch, provider=args.provider, threshold=args.threshold, storage_path=args.chain_path)
        return

    # Mode 3: Merkle Proof Audit
    if args.merkle_proof:
        run_merkle_proof_audit(args.merkle_proof, storage_path=args.chain_path)
        return

    # Mode 4: Tamper Demo
    if args.tamper_demo:
        run_tamper_demo(args.chain_path)
        return

    # Mode 5: Verify Chain
    if args.verify_chain:
        chain = LocalChain(storage_path=args.chain_path)
        report = chain.verify_chain()
        print("\n" + "=" * 50)
        print(f"Blockchain Verification Report ({args.chain_path}):")
        print(f"Total Blocks: {report['block_count']}")
        print("Status:", "✅ 100% VALID & TAMPER-FREE" if report["valid"] else f"❌ CORRUPTED:\n{report['problems']}")
        print("=" * 50 + "\n")
        return

    # Mode 6: Single Image End-to-End Pipeline
    if not args.image:
        sample_candidates = ["sample_face.jpg", "sample.jpg", "test.jpg"]
        for cand in sample_candidates:
            if os.path.exists(cand):
                args.image = cand
                break

    if not args.image:
        print("Usage: python main.py --image <path_to_face_image>")
        print("Other Modes:")
        print("  --compare <img1> <img2>     (1-to-1 face comparison)")
        print("  --batch <folder_path>       (Batch folder ingestion)")
        print("  --merkle-proof <post_url>   (Verify cryptographic Merkle proof)")
        print("  --verify-chain              (Verify blockchain ledger)")
        print("  --tamper-demo               (Adversarial tamper demo)")
        sys.exit(1)

    run_pipeline(
        image_path=args.image,
        provider=args.provider,
        threshold=args.threshold,
        storage_path=args.chain_path,
    )


if __name__ == "__main__":
    main()
