"""
block-chain.py (Local PoW Blockchain Ledger)
---------------------------------------------
Step 3 of the pipeline: blockchain verification.

The task explicitly allows "a local/simulated chain" as long as you can
demonstrate re-verifying data against the on-chain record — so this
module implements a real, tamper-evident proof-of-work blockchain ledger:

  - Each block cryptographically hashes its own contents + the previous block's hash.
  - A lightweight proof-of-work (adjustable difficulty target) is computed for each block.
  - The chain is persisted to disk (JSON) across program restarts.
  - `verify_chain()` recomputes every hash and checks cryptographic linkage + proof-of-work,
    instantly detecting any tampering or reordering.
  - `verify_record()` verifies whether a specific post hash / record exists and matches on-chain.
"""

import os
import sys
import json
import time
import hashlib
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Any

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


@dataclass
class Block:
    index: int
    timestamp: float
    data: Dict[str, Any]        # Canonical payload: post url, image hash, face hash, similarity, etc.
    previous_hash: str
    nonce: int = 0
    hash: str = field(default="")

    def compute_hash(self) -> str:
        payload = {
            "index": self.index,
            "timestamp": self.timestamp,
            "data": self.data,
            "previous_hash": self.previous_hash,
            "nonce": self.nonce,
        }
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class LocalChain:
    DIFFICULTY_PREFIX = "0000"  # Target proof-of-work difficulty (starts with 4 zeros)

    def __init__(self, storage_path: str = "chain_data.json"):
        self.storage_path = storage_path
        self.blocks: List[Block] = []
        if os.path.exists(storage_path):
            self._load()
        else:
            self._create_genesis_block()
            self._save()

    def _create_genesis_block(self):
        genesis = Block(
            index=0,
            timestamp=time.time(),
            data={"genesis": True, "message": "Genesis Block - Face Blockchain Audit Ledger"},
            previous_hash="0" * 64,
        )
        genesis.hash = self._mine(genesis)
        self.blocks.append(genesis)

    def _mine(self, block: Block) -> str:
        """
        Proof-of-work: iterates nonce until SHA-256 hash satisfies DIFFICULTY_PREFIX.
        """
        block.nonce = 0
        h = block.compute_hash()
        while not h.startswith(self.DIFFICULTY_PREFIX):
            block.nonce += 1
            h = block.compute_hash()
        return h

    def add_record(self, data: Dict[str, Any]) -> Block:
        """
        Anchors a new record (e.g. the discovered social media post + content hash +
        probe face fingerprint + similarity score) as a newly mined block.
        """
        previous = self.blocks[-1]
        block = Block(
            index=previous.index + 1,
            timestamp=time.time(),
            data=data,
            previous_hash=previous.hash,
        )
        block.hash = self._mine(block)
        self.blocks.append(block)
        self._save()
        return block

    def verify_chain(self) -> Dict[str, Any]:
        """
        Recomputes every block's hash from scratch and verifies:
          1. Stored hash matches recomputed hash (no altered payload).
          2. Each block's previous_hash matches prior block's hash (no deletions/insertions).
          3. Proof-of-work difficulty target is satisfied.
        """
        problems = []
        for i, block in enumerate(self.blocks):
            recomputed = block.compute_hash()
            if recomputed != block.hash:
                problems.append(f"Block #{block.index}: stored hash {block.hash[:10]}... != recomputed {recomputed[:10]}... (Tampered Data)")
            if not block.hash.startswith(self.DIFFICULTY_PREFIX):
                problems.append(f"Block #{block.index}: hash does not satisfy PoW target {self.DIFFICULTY_PREFIX}")
            if i > 0:
                prev = self.blocks[i - 1]
                if block.previous_hash != prev.hash:
                    problems.append(f"Block #{block.index}: previous_hash mismatch with Block #{prev.index}")

        return {
            "valid": len(problems) == 0,
            "block_count": len(self.blocks),
            "problems": problems,
        }

    def verify_record(self, post_url: str, expected_hash: Optional[str] = None) -> Dict[str, Any]:
        """
        Verifies a specific post record against the on-chain ledger.
        """
        candidate_found = False
        last_found_block = None

        for block in reversed(self.blocks):
            if block.data.get("source_url") == post_url or block.data.get("matched_post_url") == post_url:
                candidate_found = True
                last_found_block = block
                if expected_hash:
                    matched = (
                        block.data.get("canonical_record_hash") == expected_hash
                        or block.data.get("post_hash") == expected_hash
                        or block.data.get("candidate_image_hash") == expected_hash
                        or block.data.get("probe_image_hash") == expected_hash
                        or block.data.get("image_hash") == expected_hash
                    )
                    if not matched:
                        clean_data = {k: v for k, v in block.data.items() if k != "canonical_record_hash"}
                        if hash_record(clean_data) == expected_hash or hash_record(block.data) == expected_hash:
                            matched = True

                    if matched:
                        return {
                            "exists": True,
                            "verified": True,
                            "block_index": block.index,
                            "block_hash": block.hash,
                            "timestamp": block.timestamp,
                            "data": block.data,
                        }
                else:
                    return {
                        "exists": True,
                        "verified": True,
                        "block_index": block.index,
                        "block_hash": block.hash,
                        "timestamp": block.timestamp,
                        "data": block.data,
                    }

        if candidate_found:
            return {
                "exists": True,
                "verified": False,
                "block_index": last_found_block.index if last_found_block else -1,
                "error": "Record found for URL but cryptographic hash does not match (Tampered).",
            }

        return {"exists": False, "verified": False, "error": "No matching record found on chain."}

    def find_record_by_url(self, source_url: str) -> Optional[Block]:
        for block in self.blocks:
            if block.data.get("source_url") == source_url or block.data.get("matched_post_url") == source_url:
                return block
        return None

    def _save(self):
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump([asdict(b) for b in self.blocks], f, indent=2)

    def _load(self):
        with open(self.storage_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.blocks = [Block(**b) for b in raw]


def hash_record(record_dict: Dict[str, Any]) -> str:
    """Computes canonical SHA-256 hash of a match record dictionary."""
    canonical_json = json.dumps(record_dict, sort_keys=True)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    chain = LocalChain("chain_data.json")
    print(f"Loaded blockchain with {len(chain.blocks)} block(s).")
    report = chain.verify_chain()
    print("Chain integrity status:", "VALID ✅" if report["valid"] else f"CORRUPTED ❌ ({report['problems']})")