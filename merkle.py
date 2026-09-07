"""
merkle.py
---------
Cryptographic Merkle Tree implementation for Task 3 blockchain verification.

Features:
- Binary SHA-256 Merkle Tree construction.
- $O(1)$ Root hash verification.
- $O(\\log N)$ Inclusion Proof generation and independent verification.
- Tamper-proof auditing without exposing other records in a block.
"""

import json
import hashlib
from typing import List, Tuple, Any, Optional, Dict


def sha256_hash(data: str) -> str:
    """Computes SHA-256 hash of a string."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def hash_leaf(data: Any) -> str:
    """Computes leaf hash from dict or string payload."""
    if isinstance(data, dict):
        canonical = json.dumps(data, sort_keys=True)
        return sha256_hash(canonical)
    return sha256_hash(str(data))


class MerkleTree:
    def __init__(self, leaves: List[Any]):
        self.raw_leaves = leaves
        self.leaf_hashes: List[str] = [hash_leaf(leaf) for leaf in leaves] if leaves else [sha256_hash("EMPTY_LEAF")]
        self.levels: List[List[str]] = []
        self._build_tree()

    def _build_tree(self):
        """Constructs tree levels from leaves up to the root."""
        current = list(self.leaf_hashes)
        self.levels = [current]

        while len(current) > 1:
            next_level = []
            for i in range(0, len(current), 2):
                left = current[i]
                right = current[i + 1] if i + 1 < len(current) else left  # Duplicate odd leaf
                combined = hashlib.sha256((left + right).encode("utf-8")).hexdigest()
                next_level.append(combined)
            current = next_level
            self.levels.append(current)

    @property
    def root(self) -> str:
        """Returns the Merkle Root hash."""
        return self.levels[-1][0] if self.levels else sha256_hash("EMPTY_ROOT")

    def get_proof(self, index: int) -> List[Tuple[str, str]]:
        """
        Generates Merkle Proof for leaf at given index.
        Returns list of (sibling_hash, direction) where direction is 'left' or 'right'.
        """
        if index < 0 or index >= len(self.leaf_hashes):
            raise IndexError("Leaf index out of bounds.")

        proof = []
        idx = index
        for level in self.levels[:-1]:
            is_right_child = (idx % 2 == 1)
            sibling_idx = idx - 1 if is_right_child else idx + 1
            if sibling_idx < len(level):
                sibling_hash = level[sibling_idx]
            else:
                sibling_hash = level[idx]  # Self-paired odd leaf

            direction = "left" if is_right_child else "right"
            proof.append((sibling_hash, direction))
            idx //= 2

        return proof

    @staticmethod
    def verify_proof(leaf_hash: str, proof: List[Tuple[str, str]], root_hash: str) -> bool:
        """
        Verifies that leaf_hash belongs to the Merkle Tree with root_hash using the given proof.
        """
        current_hash = leaf_hash
        for sibling_hash, direction in proof:
            if direction == "left":
                combined = sibling_hash + current_hash
            else:
                combined = current_hash + sibling_hash
            current_hash = hashlib.sha256(combined.encode("utf-8")).hexdigest()

        return current_hash.lower() == root_hash.lower()
