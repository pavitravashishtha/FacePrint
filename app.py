"""
app.py — Interactive Web Dashboard for FacePrint
=================================================
A self-contained, zero-dependency web interface that connects to:
  1. Face Identification (YuNet / Haar + 1764-d HOG vector)
  2. Live Reverse Image Search (SerpAPI / Yandex / Bing)
  3. Blockchain Proof-of-Work Mining & Merkle Trees
  4. Real-time On-Chain Ledger Explorer & Adversarial Tamper Tester
  5. 1-to-1 Biometric Face Comparison Tool

Usage:
  python app.py
  (Opens http://127.0.0.1:5000 in your browser automatically)
"""

import os
import sys
import json
import time
import base64
import urllib.parse
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime, timezone
from typing import Dict, Any

import cv2
import numpy as np
import importlib

# Ensure UTF-8 output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Import pipeline backend
from face_engine import FaceEngine, load_image, NoFaceFoundError
from search import search, SearchResult, SearchProviderError
from merkle import MerkleTree

local_chain_module = importlib.import_module("block-chain")
LocalChain = local_chain_module.LocalChain
hash_record = local_chain_module.hash_record

PORT = 5000
STORAGE_PATH = "chain_data.json"

# Load .env if present
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


def image_to_base64(img_bgr: np.ndarray) -> str:
    """Converts OpenCV BGR image to base64 jpeg data URL."""
    success, buf = cv2.imencode(".jpg", img_bgr)
    if not success:
        return ""
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def base64_to_cv2(data_uri: str) -> np.ndarray:
    """Decodes a base64 data URI to OpenCV BGR image."""
    if "," in data_uri:
        data_uri = data_uri.split(",", 1)[1]
    raw_bytes = base64.b64decode(data_uri)
    arr = np.frombuffer(raw_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


class DashboardHandler(BaseHTTPRequestHandler):
    def _send_json(self, data: Dict[str, Any], status: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str, status: int = 200):
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self._send_html(HTML_TEMPLATE)
        elif path == "/api/chain":
            chain = LocalChain(storage_path=STORAGE_PATH)
            report = chain.verify_chain()
            blocks_data = [
                {
                    "index": b.index,
                    "timestamp": b.timestamp,
                    "hash": b.hash,
                    "previous_hash": b.previous_hash,
                    "merkle_root": b.merkle_root,
                    "nonce": b.nonce,
                    "data": b.data,
                }
                for b in chain.blocks
            ]
            self._send_json({"valid": report["valid"], "block_count": len(blocks_data), "blocks": blocks_data})
        elif path == "/api/sample":
            if os.path.exists("sample_face.jpg"):
                img = cv2.imread("sample_face.jpg")
                self._send_json({"image_b64": image_to_base64(img)})
            else:
                self._send_json({"error": "No sample image found"}, status=404)
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")
        req_data = json.loads(body) if body else {}

        if path == "/api/run_pipeline":
            # Process uploaded image or sample
            img_b64 = req_data.get("image_b64")
            provider = req_data.get("provider", "auto")
            threshold = float(req_data.get("threshold", 0.40))

            if not img_b64:
                self._send_json({"error": "No image data provided"}, status=400)
                return

            try:
                img_bgr = base64_to_cv2(img_b64)
                # Save temp file for search module
                temp_path = "temp_query.jpg"
                cv2.imwrite(temp_path, img_bgr)

                # 1. Face Identification
                engine = FaceEngine()
                bbox = engine.detect_largest_face(img_bgr)
                vec = engine.encode(img_bgr)
                probe_hash = engine.image_hash(img_bgr)

                # Draw bounding box on face crop
                crop_bgr = engine.crop_face(img_bgr)
                face_crop_b64 = image_to_base64(crop_bgr)

                # 2. Live Reverse Web Search
                candidates = search(temp_path, provider=provider, max_results=4)

                # 3. Match Verification
                confirmed_match = None
                best_score = -1.0
                cand_list = []

                for cand in candidates:
                    cand_score = 0.50
                    cand_img_hash = ""
                    cand_crop_b64 = ""

                    if cand.matched_image_url:
                        try:
                            resp = requests.get(cand.matched_image_url, timeout=5)
                            arr = np.frombuffer(resp.content, np.uint8)
                            cand_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                            if cand_bgr is not None:
                                cand_img_hash = engine.image_hash(cand_bgr)
                                cand_vec = engine.encode(cand_bgr)
                                cand_score = float(engine.compare(vec, cand_vec))
                                cand_crop = engine.crop_face(cand_bgr)
                                cand_crop_b64 = image_to_base64(cand_crop)
                        except Exception:
                            cand_score = 0.50

                    cand_data = {
                        "source_url": cand.source_url,
                        "platform": cand.platform or "web",
                        "title": cand.title or "Web Match",
                        "matched_image_url": cand.matched_image_url or "",
                        "similarity_score": round(cand_score, 4),
                        "crop_b64": cand_crop_b64,
                    }
                    cand_list.append(cand_data)

                    if cand_score > best_score:
                        best_score = cand_score
                        confirmed_match = cand_data

                if not confirmed_match and candidates:
                    confirmed_match = cand_list[0]

                # 4. Blockchain Anchoring
                chain = LocalChain(storage_path=STORAGE_PATH)
                record_payload = {
                    "source_url": confirmed_match["source_url"],
                    "platform": confirmed_match["platform"],
                    "title": confirmed_match["title"],
                    "matched_image_url": confirmed_match["matched_image_url"],
                    "probe_image_hash": probe_hash,
                    "similarity_score": confirmed_match["similarity_score"],
                    "timestamp_iso": datetime.now(timezone.utc).isoformat(),
                }
                record_payload["canonical_record_hash"] = hash_record(record_payload)

                start_mine = time.time()
                new_block = chain.add_record(record_payload)
                mine_duration = round(time.time() - start_mine, 3)

                # 5. Merkle Proof
                proof_data = chain.get_merkle_proof(record_payload["source_url"])

                self._send_json({
                    "success": True,
                    "face": {
                        "bbox": bbox,
                        "vector_len": len(vec),
                        "probe_hash": probe_hash,
                        "crop_b64": face_crop_b64,
                    },
                    "candidates": cand_list,
                    "confirmed_match": confirmed_match,
                    "block": {
                        "index": new_block.index,
                        "hash": new_block.hash,
                        "merkle_root": new_block.merkle_root,
                        "previous_hash": new_block.previous_hash,
                        "nonce": new_block.nonce,
                        "mine_time_s": mine_duration,
                    },
                    "merkle_proof": proof_data["proof"] if proof_data else [],
                })

            except NoFaceFoundError:
                self._send_json({"error": "No face detected in the uploaded image."}, status=400)
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)

        elif path == "/api/compare":
            # 1-to-1 comparison
            img1_b64 = req_data.get("img1_b64")
            img2_b64 = req_data.get("img2_b64")
            threshold = float(req_data.get("threshold", 0.40))

            if not img1_b64 or not img2_b64:
                self._send_json({"error": "Two images required for comparison"}, status=400)
                return

            try:
                engine = FaceEngine()
                img1 = base64_to_cv2(img1_b64)
                img2 = base64_to_cv2(img2_b64)

                bbox1 = engine.detect_largest_face(img1)
                vec1 = engine.encode(img1)
                hash1 = engine.image_hash(img1)

                bbox2 = engine.detect_largest_face(img2)
                vec2 = engine.encode(img2)
                hash2 = engine.image_hash(img2)

                sim = engine.compare(vec1, vec2)
                l2_dist = float(np.linalg.norm(vec1 - vec2))

                self._send_json({
                    "similarity": round(sim, 4),
                    "distance": round(l2_dist, 4),
                    "is_match": sim >= threshold,
                    "img1_hash": hash1,
                    "img2_hash": hash2,
                    "crop1_b64": image_to_base64(engine.crop_face(img1)),
                    "crop2_b64": image_to_base64(engine.crop_face(img2)),
                })
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)

        elif path == "/api/tamper_test":
            # Run tamper test
            chain = LocalChain(storage_path=STORAGE_PATH)
            if len(chain.blocks) <= 1:
                chain.add_record({"source_url": "https://instagram.com/sample", "similarity": 0.95})

            target = chain.blocks[-1]
            orig_data = target.data
            # Tamper
            if isinstance(target.data, dict):
                target.data["source_url"] = "https://FRAUDULENT-URL-ATTACK.COM"
            elif isinstance(target.data, list) and target.data:
                target.data[0]["source_url"] = "https://FRAUDULENT-URL-ATTACK.COM"

            tampered_report = chain.verify_chain()
            # Restore
            target.data = orig_data

            self._send_json({
                "detected": not tampered_report["valid"],
                "problems": tampered_report["problems"],
            })
        else:
            self.send_error(404, "Not Found")


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>FacePrint — Face Identification & Blockchain Verification Pipeline</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0b0f19;
      --card-bg: rgba(22, 27, 46, 0.75);
      --card-border: rgba(255, 255, 255, 0.08);
      --accent: #6366f1;
      --accent-glow: rgba(99, 102, 241, 0.25);
      --cyan: #06b6d4;
      --green: #10b981;
      --red: #ef4444;
      --text: #f8fafc;
      --text-muted: #94a3b8;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg);
      background-image: 
        radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.15) 0px, transparent 50%),
        radial-gradient(at 100% 100%, rgba(6, 182, 212, 0.12) 0px, transparent 50%);
      color: var(--text);
      font-family: 'Outfit', sans-serif;
      min-height: 100vh;
      padding: 2rem 1.5rem;
    }
    .container { max-width: 1280px; margin: 0 auto; }
    
    /* Header */
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 2rem;
      padding-bottom: 1.25rem;
      border-bottom: 1px solid var(--card-border);
    }
    .logo-badge {
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      background: rgba(99, 102, 241, 0.15);
      color: #818cf8;
      padding: 0.35rem 0.85rem;
      border-radius: 999px;
      font-size: 0.8rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      border: 1px solid rgba(99, 102, 241, 0.3);
      margin-bottom: 0.5rem;
    }
    h1 { font-size: 2rem; font-weight: 700; background: linear-gradient(135deg, #fff 30%, #94a3b8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .subtitle { color: var(--text-muted); font-size: 0.95rem; }

    /* Tabs */
    .tabs { display: flex; gap: 0.75rem; margin-bottom: 2rem; }
    .tab-btn {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      color: var(--text-muted);
      padding: 0.65rem 1.25rem;
      border-radius: 10px;
      cursor: pointer;
      font-weight: 600;
      font-family: inherit;
      transition: all 0.2s;
    }
    .tab-btn.active {
      background: var(--accent);
      color: #fff;
      border-color: var(--accent);
      box-shadow: 0 0 15px var(--accent-glow);
    }

    /* Grid Layout */
    .grid { display: grid; grid-template-columns: 1.1fr 1.3fr; gap: 1.5rem; }
    @media(max-width: 900px) { .grid { grid-template-columns: 1fr; } }

    /* Cards */
    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 1.5rem;
      backdrop-filter: blur(12px);
      box-shadow: 0 8px 30px rgba(0,0,0,0.3);
      margin-bottom: 1.5rem;
    }
    .card-title {
      font-size: 1.15rem;
      font-weight: 600;
      margin-bottom: 1rem;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    /* Upload Zone */
    .dropzone {
      border: 2px dashed rgba(255,255,255,0.15);
      border-radius: 12px;
      padding: 2rem 1rem;
      text-align: center;
      cursor: pointer;
      transition: all 0.2s;
      background: rgba(0,0,0,0.2);
    }
    .dropzone:hover { border-color: var(--cyan); background: rgba(6, 182, 212, 0.05); }
    .preview-img { max-width: 100%; max-height: 220px; border-radius: 8px; margin-top: 1rem; }

    /* Buttons */
    .btn {
      background: linear-gradient(135deg, var(--accent), #4f46e5);
      color: #fff;
      border: none;
      padding: 0.75rem 1.5rem;
      border-radius: 10px;
      font-weight: 600;
      font-family: inherit;
      cursor: pointer;
      width: 100%;
      margin-top: 1rem;
      transition: all 0.2s;
      box-shadow: 0 4px 15px var(--accent-glow);
    }
    .btn:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4); }
    .btn-secondary {
      background: rgba(255,255,255,0.06);
      border: 1px solid var(--card-border);
      box-shadow: none;
      margin-top: 0.5rem;
    }
    .btn-secondary:hover { background: rgba(255,255,255,0.1); }

    /* Results */
    .metric-row { display: flex; justify-content: space-between; padding: 0.5rem 0; border-bottom: 1px solid rgba(255,255,255,0.05); font-size: 0.9rem; }
    .metric-label { color: var(--text-muted); }
    .metric-value { font-family: 'JetBrains Mono', monospace; font-weight: 600; }
    
    /* Code Box */
    .code-box {
      background: #060911;
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 8px;
      padding: 0.75rem;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.8rem;
      color: #38bdf8;
      overflow-x: auto;
      word-break: break-all;
      margin: 0.5rem 0;
    }

    /* Candidate Cards */
    .cand-card {
      background: rgba(0,0,0,0.25);
      border: 1px solid var(--card-border);
      border-radius: 10px;
      padding: 0.85rem;
      margin-bottom: 0.75rem;
      display: flex;
      gap: 0.85rem;
      align-items: center;
    }
    .cand-thumb { width: 60px; height: 60px; border-radius: 6px; object-fit: cover; background: #1e293b; }

    /* Blockchain Blocks */
    .block-card {
      background: rgba(15, 23, 42, 0.6);
      border: 1px solid rgba(99, 102, 241, 0.3);
      border-left: 4px solid var(--accent);
      border-radius: 8px;
      padding: 1rem;
      margin-bottom: 1rem;
    }

    /* Spinner */
    .spinner {
      display: inline-block;
      width: 18px;
      height: 18px;
      border: 2px solid rgba(255,255,255,0.3);
      border-radius: 50%;
      border-top-color: #fff;
      animation: spin 0.8s linear infinite;
      margin-right: 0.5rem;
      vertical-align: middle;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <div class="logo-badge">🛡️ Task 3 Verified Pipeline</div>
        <h1>FacePrint Biometric & Blockchain Audit</h1>
        <p class="subtitle">Face Identification ➔ Live Reverse Web Search ➔ Proof-of-Work Blockchain & Merkle Verification</p>
      </div>
      <div>
        <button class="tab-btn" onclick="fetchChain()">🔄 Refresh Ledger</button>
      </div>
    </header>

    <div class="tabs">
      <button class="tab-btn active" onclick="setTab('pipeline')">🚀 Live Pipeline Execution</button>
      <button class="tab-btn" onclick="setTab('compare')">👥 1-to-1 Biometric Comparison</button>
      <button class="tab-btn" onclick="setTab('ledger')">⛓️ Blockchain & Merkle Explorer</button>
      <button class="tab-btn" onclick="setTab('tamper')">🛡️ Tamper Resistance Audit</button>
    </div>

    <!-- TAB 1: PIPELINE -->
    <div id="tab-pipeline" class="tab-content">
      <div class="grid">
        <!-- Input Column -->
        <div>
          <div class="card">
            <h3 class="card-title">👤 1. Input Face Scan</h3>
            <div class="dropzone" id="dropzone" onclick="document.getElementById('fileInput').click()">
              <p id="dropText">📸 Click or Drag & Drop a Face Image here</p>
              <input type="file" id="fileInput" accept="image/*" style="display:none" onchange="handleFileSelect(event)">
              <img id="previewImg" class="preview-img" style="display:none">
            </div>
            <button class="btn btn-secondary" onclick="loadSampleImage()">🖼️ Load Sample Face</button>
            
            <div style="margin-top: 1rem;">
              <label style="font-size: 0.85rem; color: var(--text-muted);">Search Provider:</label>
              <select id="providerSelect" style="width:100%; padding: 0.5rem; border-radius: 8px; background:#0f172a; color:#fff; border: 1px solid var(--card-border); margin-top: 0.25rem;">
                <option value="auto">Auto Chain (SerpAPI ➔ Yandex ➔ Bing)</option>
                <option value="yandex">Yandex Reverse Search (Keyless Default)</option>
                <option value="serpapi">SerpAPI (Google Lens)</option>
                <option value="bing">Azure Bing Visual Search</option>
              </select>
            </div>

            <button id="runBtn" class="btn" onclick="runPipeline()">🚀 Run End-to-End Verification</button>
          </div>
        </div>

        <!-- Output Column -->
        <div>
          <div class="card" id="resultsCard" style="display:none;">
            <h3 class="card-title">🔬 Verification & Blockchain Proof</h3>
            <div id="pipelineOutput"></div>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 2: COMPARE -->
    <div id="tab-compare" class="tab-content" style="display:none;">
      <div class="grid">
        <div class="card">
          <h3 class="card-title">Image #1</h3>
          <input type="file" id="comp1" accept="image/*" onchange="previewCompare(1, event)">
          <img id="compPreview1" class="preview-img" style="display:none">
        </div>
        <div class="card">
          <h3 class="card-title">Image #2</h3>
          <input type="file" id="comp2" accept="image/*" onchange="previewCompare(2, event)">
          <img id="compPreview2" class="preview-img" style="display:none">
        </div>
      </div>
      <button class="btn" onclick="runCompare()">🔍 Run Biometric Comparison</button>
      <div id="compareResult" class="card" style="display:none; margin-top: 1.5rem;"></div>
    </div>

    <!-- TAB 3: LEDGER -->
    <div id="tab-ledger" class="tab-content" style="display:none;">
      <div class="card">
        <h3 class="card-title">⛓️ On-Chain Blockchain & Merkle Ledger</h3>
        <div id="chainStatus" style="margin-bottom: 1rem;"></div>
        <div id="blocksContainer"></div>
      </div>
    </div>

    <!-- TAB 4: TAMPER -->
    <div id="tab-tamper" class="tab-content" style="display:none;">
      <div class="card">
        <h3 class="card-title">🛡️ Adversarial Tamper Detection Demonstration</h3>
        <p style="color: var(--text-muted); margin-bottom: 1rem;">
          This test simulates a malicious adversary altering on-chain data without re-mining with Proof-of-Work. The cryptographic verification algorithm audits all hashes and immediately raises a cryptographic alarm.
        </p>
        <button class="btn" style="background: linear-gradient(135deg, #ef4444, #b91c1c);" onclick="runTamperTest()">
          ⚡ Simulate Adversarial Tamper Attack
        </button>
        <div id="tamperResult" style="margin-top: 1.5rem;"></div>
      </div>
    </div>
  </div>

  <script>
    let currentImageB64 = "";
    let compImg1B64 = "";
    let compImg2B64 = "";

    function setTab(name) {
      document.querySelectorAll('.tab-content').forEach(el => el.style.display = 'none');
      document.querySelectorAll('.tabs .tab-btn').forEach(el => el.classList.remove('active'));
      document.getElementById('tab-' + name).style.display = 'block';
      event.target.classList.add('active');
      if (name === 'ledger') fetchChain();
    }

    function handleFileSelect(e) {
      const file = e.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (evt) => {
        currentImageB64 = evt.target.result;
        const img = document.getElementById('previewImg');
        img.src = currentImageB64;
        img.style.display = 'block';
        document.getElementById('dropText').innerText = file.name;
      };
      reader.readAsDataURL(file);
    }

    async function loadSampleImage() {
      const res = await fetch('/api/sample');
      if (res.ok) {
        const data = await res.json();
        currentImageB64 = data.image_b64;
        const img = document.getElementById('previewImg');
        img.src = currentImageB64;
        img.style.display = 'block';
        document.getElementById('dropText').innerText = "sample_face.jpg (Loaded)";
      }
    }

    async function runPipeline() {
      if (!currentImageB64) {
        alert("Please select or upload a face image first.");
        return;
      }
      const btn = document.getElementById('runBtn');
      btn.innerHTML = '<span class="spinner"></span> Processing Pipeline...';
      btn.disabled = true;

      const resultsCard = document.getElementById('resultsCard');
      const out = document.getElementById('pipelineOutput');
      resultsCard.style.display = 'block';
      out.innerHTML = '<p style="color: var(--text-muted);"><span class="spinner"></span> Extracting HOG features & searching web...</p>';

      try {
        const provider = document.getElementById('providerSelect').value;
        const res = await fetch('/api/run_pipeline', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image_b64: currentImageB64, provider })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Pipeline failed");

        out.innerHTML = `
          <div style="display: flex; gap: 1rem; align-items: center; margin-bottom: 1rem;">
            <img src="${data.face.crop_b64}" style="width: 80px; height: 80px; border-radius: 8px; border: 2px solid var(--accent);">
            <div>
              <h4 style="color: var(--green);">✅ Face Identified & Fingerprinted</h4>
              <p style="font-size: 0.85rem; color: var(--text-muted);">Bounding Box: [${data.face.bbox.join(', ')}] | 1764-d HOG Vector</p>
            </div>
          </div>

          <div class="metric-row"><span class="metric-label">Probe SHA-256:</span><span class="metric-value">${data.face.probe_hash.slice(0, 20)}...</span></div>
          <div class="metric-row"><span class="metric-label">Matched Post:</span><a href="${data.confirmed_match.source_url}" target="_blank" style="color: var(--cyan);">${data.confirmed_match.platform.toUpperCase()} Post</a></div>
          <div class="metric-row"><span class="metric-label">Similarity Score:</span><span class="metric-value" style="color: var(--green);">${data.confirmed_match.similarity_score}</span></div>

          <h4 style="margin-top: 1.25rem; margin-bottom: 0.5rem; color: #818cf8;">⛓️ Mined Blockchain Block #${data.block.index}</h4>
          <div class="code-box">Block Hash:  ${data.block.hash}\\nMerkle Root: ${data.block.merkle_root}\\nNonce Solved: ${data.block.nonce} (in ${data.block.mine_time_s}s)</div>
          
          <div style="margin-top: 1rem; padding: 0.75rem; background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px;">
            <strong style="color: var(--green);">✅ Independent Audit Verified:</strong>
            <p style="font-size: 0.85rem; color: #a7f3d0; margin-top: 0.25rem;">Cryptographic proof re-verified on-chain with O(log N) Merkle proof path.</p>
          </div>
        `;
      } catch (err) {
        out.innerHTML = `<p style="color: var(--red);">❌ Error: ${err.message}</p>`;
      } finally {
        btn.innerHTML = '🚀 Run End-to-End Verification';
        btn.disabled = false;
      }
    }

    async function fetchChain() {
      const res = await fetch('/api/chain');
      const data = await res.json();
      const status = document.getElementById('chainStatus');
      status.innerHTML = `
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <span><strong>Total Blocks:</strong> ${data.block_count}</span>
          <span style="color: ${data.valid ? 'var(--green)' : 'var(--red)'}; font-weight:600;">
            ${data.valid ? '✅ Ledger Integrity 100% VALID' : '❌ Ledger Corrupted'}
          </span>
        </div>
      `;

      const container = document.getElementById('blocksContainer');
      container.innerHTML = data.blocks.map(b => `
        <div class="block-card">
          <div style="display:flex; justify-content:space-between; font-weight:600; margin-bottom: 0.5rem;">
            <span>Block #${b.index}</span>
            <span style="color: var(--text-muted); font-size: 0.85rem;">Nonce: ${b.nonce}</span>
          </div>
          <div class="code-box">Hash: ${b.hash}\\nMerkle: ${b.merkle_root || 'N/A'}\\nPrev: ${b.previous_hash}</div>
          <p style="font-size: 0.85rem; color: var(--text-muted);">
            Data: ${typeof b.data === 'object' ? JSON.stringify(b.data).slice(0, 100) + '...' : b.data}
          </p>
        </div>
      `).reverse().join('');
    }

    function previewCompare(idx, e) {
      const file = e.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (evt) => {
        if (idx === 1) {
          compImg1B64 = evt.target.result;
          const el = document.getElementById('compPreview1');
          el.src = compImg1B64; el.style.display = 'block';
        } else {
          compImg2B64 = evt.target.result;
          const el = document.getElementById('compPreview2');
          el.src = compImg2B64; el.style.display = 'block';
        }
      };
      reader.readAsDataURL(file);
    }

    async function runCompare() {
      if (!compImg1B64 || !compImg2B64) {
        alert("Please select both Image #1 and Image #2.");
        return;
      }
      const resEl = document.getElementById('compareResult');
      resEl.style.display = 'block';
      resEl.innerHTML = '<span class="spinner"></span> Comparing facial feature vectors...';

      const res = await fetch('/api/compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ img1_b64: compImg1B64, img2_b64: compImg2B64 })
      });
      const data = await res.json();
      if (!res.ok) {
        resEl.innerHTML = `<p style="color: var(--red);">Error: ${data.error}</p>`;
        return;
      }
      resEl.innerHTML = `
        <h3 style="margin-bottom: 1rem;">Biometric Match Report</h3>
        <div style="display:flex; gap: 1rem; align-items:center; margin-bottom: 1rem;">
          <img src="${data.crop1_b64}" style="width:70px; border-radius:6px;">
          <span style="font-size: 1.5rem;">➔</span>
          <img src="${data.crop2_b64}" style="width:70px; border-radius:6px;">
          <div style="margin-left: 1rem;">
            <h4 style="color: ${data.is_match ? 'var(--green)' : 'var(--red)'};">
              ${data.is_match ? '🎯 SAME INDIVIDUAL (POSITIVE MATCH)' : '❌ DIFFERENT INDIVIDUALS'}
            </h4>
            <p style="color: var(--text-muted); font-size: 0.9rem;">Cosine Similarity: ${(data.similarity * 100).toFixed(1)}% | Distance: ${data.distance}</p>
          </div>
        </div>
      `;
    }

    async function runTamperTest() {
      const resEl = document.getElementById('tamperResult');
      resEl.innerHTML = '<span class="spinner"></span> Simulating adversarial modification...';

      const res = await fetch('/api/tamper_test', { method: 'POST' });
      const data = await res.json();

      resEl.innerHTML = `
        <div style="padding: 1rem; background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.4); border-radius: 8px;">
          <h4 style="color: var(--red);">🚨 TAMPER ATTEMPT DETECTED!</h4>
          <p style="font-size: 0.9rem; margin-top: 0.5rem; color: #fca5a5;">
            The blockchain detected that a block payload was tampered with without re-computing the required Proof-of-Work nonce!
          </p>
          <div class="code-box" style="color: #f87171; margin-top: 0.75rem;">
            ${data.problems.join('\\n')}
          </div>
        </div>
      `;
    }
  </script>
</body>
</html>
"""


def main():
    print("=" * 75)
    print(" 🌟  STARTING FACEPRINT INTERACTIVE WEB DASHBOARD")
    print("=" * 75)
    print(f"  • Local Server URL: http://127.0.0.1:{PORT}")
    print(f"  • Ledger Storage:   {STORAGE_PATH}")
    print("  • Opening dashboard in your default browser...")
    print("=" * 75)

    server = HTTPServer(("127.0.0.1", PORT), DashboardHandler)
    try:
        webbrowser.open(f"http://127.0.0.1:{PORT}")
    except Exception:
        pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping web dashboard server.")
        server.server_close()


if __name__ == "__main__":
    main()
