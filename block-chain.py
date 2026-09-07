"""
block-chain.py (Local PoW Blockchain Ledger with Merkle Tree Inclusion Proofs)
-------------------------------------------------------------------------------
Step 3 of the pipeline: blockchain verification.

Features:
  - Each block cryptographically hashes its own contents + previous hash + Merkle Root.
  - Merkle Tree generation: computes Merkle Root for all transaction/match records.
  - $O(\\log N)$ Merkle Proofs: proves inclusion of a specific post without exposing the whole block.
  - Proof-of-Work: adjustable difficulty target (0000...) for tamper-evident mining.
  - Disk persistence: JSON ledger state stored in chain_data.json.
  - Auditing: verify_chain() recomputes every block hash and validates linkage.
  - Single-Record & Batch Records support.
"""

import os
import sys
import json
import time
import hashlib
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Any, Tuple, Union

from merkle import MerkleTree, hash_leaf

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
    data: Any                   # Dict (single record) or List[Dict] (batch records)
    previous_hash: str
    merkle_root: str = ""
    nonce: int = 0
    hash: str = field(default="")

    def __post_init__(self):
        if not self.merkle_root:
            self.merkle_root = self.compute_merkle_root()

    def compute_merkle_root(self) -> str:
        """Derives Merkle Root for block's data."""
        if isinstance(self.data, list):
            tree = MerkleTree(self.data)
            return tree.root
        elif isinstance(self.data, dict):
            tree = MerkleTree([self.data])
            return tree.root
        return MerkleTree([str(self.data)]).root

    def compute_hash(self) -> str:
        payload = {
            "index": self.index,
            "timestamp": self.timestamp,
            "merkle_root": self.merkle_root or self.compute_merkle_root(),
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
        genesis_data = {"genesis": True, "message": "Genesis Block - Face Blockchain Audit Ledger"}
        genesis = Block(
            index=0,
            timestamp=time.time(),
            data=genesis_data,
            previous_hash="0" * 64,
            merkle_root=MerkleTree([genesis_data]).root,
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
        Anchors a single record as a newly mined block with Merkle Root.
        """
        previous = self.blocks[-1]
        m_root = MerkleTree([data]).root
        block = Block(
            index=previous.index + 1,
            timestamp=time.time(),
            data=data,
            previous_hash=previous.hash,
            merkle_root=m_root,
        )
        block.hash = self._mine(block)
        self.blocks.append(block)
        self._save()
        return block

    def add_batch_records(self, records: List[Dict[str, Any]]) -> Block:
        """
        Anchors multiple records into a single block using a Merkle Tree.
        """
        if not records:
            raise ValueError("Cannot mine block with empty records list.")

        previous = self.blocks[-1]
        tree = MerkleTree(records)
        block = Block(
            index=previous.index + 1,
            timestamp=time.time(),
            data=records,
            previous_hash=previous.hash,
            merkle_root=tree.root,
        )
        block.hash = self._mine(block)
        self.blocks.append(block)
        self._save()
        return block

    def verify_chain(self) -> Dict[str, Any]:
        """
        Recomputes every block's hash from scratch and verifies:
          1. Stored hash matches recomputed hash.
          2. Stored Merkle Root matches recomputed Merkle Tree root.
          3. Previous_hash links cleanly to prior block.
          4. Proof-of-work difficulty target is satisfied.
        """
        problems = []
        for i, block in enumerate(self.blocks):
            # Check Merkle Root
            expected_merkle = block.compute_merkle_root()
            if block.merkle_root != expected_merkle:
                problems.append(f"Block #{block.index}: Merkle Root mismatch ({block.merkle_root[:10]}... != {expected_merkle[:10]}...)")

            # Check PoW Hash
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
        Verifies a specific post record against the on-chain ledger across single & batch blocks.
        """
        candidate_found = False
        last_found_block = None

        for block in reversed(self.blocks):
            items = block.data if isinstance(block.data, list) else [block.data]
            for item in items:
                if isinstance(item, dict) and (item.get("source_url") == post_url or item.get("matched_post_url") == post_url):
                    candidate_found = True
                    last_found_block = block
                    if expected_hash:
                        matched = (
                            item.get("canonical_record_hash") == expected_hash
                            or item.get("post_hash") == expected_hash
                            or item.get("candidate_image_hash") == expected_hash
                            or item.get("probe_image_hash") == expected_hash
                            or item.get("image_hash") == expected_hash
                        )
                        if not matched:
                            clean_data = {k: v for k, v in item.items() if k != "canonical_record_hash"}
                            if hash_record(clean_data) == expected_hash or hash_record(item) == expected_hash:
                                matched = True

                        if matched:
                            return {
                                "exists": True,
                                "verified": True,
                                "block_index": block.index,
                                "block_hash": block.hash,
                                "merkle_root": block.merkle_root,
                                "timestamp": block.timestamp,
                                "data": item,
                            }
                    else:
                        return {
                            "exists": True,
                            "verified": True,
                            "block_index": block.index,
                            "block_hash": block.hash,
                            "merkle_root": block.merkle_root,
                            "timestamp": block.timestamp,
                            "data": item,
                        }

        if candidate_found:
            return {
                "exists": True,
                "verified": False,
                "block_index": last_found_block.index if last_found_block else -1,
                "error": "Record found for URL but cryptographic hash does not match (Tampered).",
            }

        return {"exists": False, "verified": False, "error": "No matching record found on chain."}

    def get_merkle_proof(self, post_url: str) -> Optional[Dict[str, Any]]:
        """
        Returns the Merkle Proof for a post record in whichever block it resides.
        """
        for block in reversed(self.blocks):
            items = block.data if isinstance(block.data, list) else [block.data]
            for idx, item in enumerate(items):
                if isinstance(item, dict) and (item.get("source_url") == post_url or item.get("matched_post_url") == post_url):
                    tree = MerkleTree(items)
                    proof = tree.get_proof(idx)
                    leaf_hash = tree.leaf_hashes[idx]
                    return {
                        "block_index": block.index,
                        "block_hash": block.hash,
                        "merkle_root": block.merkle_root,
                        "leaf_index": idx,
                        "leaf_hash": leaf_hash,
                        "proof": proof,
                        "record": item,
                    }
        return None

    def _save(self):
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump([asdict(b) for b in self.blocks], f, indent=2)

    def _load(self):
        with open(self.storage_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.blocks = []
        for b in raw:
            # Upgrade legacy blocks if merkle_root was missing
            if "merkle_root" not in b:
                data = b.get("data")
                items = data if isinstance(data, list) else [data]
                b["merkle_root"] = MerkleTree(items).root
            self.blocks.append(Block(**b))


def hash_record(record_dict: Dict[str, Any]) -> str:
    """Computes canonical SHA-256 hash of a match record dictionary."""
    canonical_json = json.dumps(record_dict, sort_keys=True)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    chain = LocalChain("chain_data.json")
    print(f"Loaded blockchain with {len(chain.blocks)} block(s).")
    report = chain.verify_chain()
    print("Chain integrity status:", "VALID ✅" if report["valid"] else f"CORRUPTED ❌ ({report['problems']})")