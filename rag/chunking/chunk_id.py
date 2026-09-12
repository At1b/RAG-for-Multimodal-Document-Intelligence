"""Chunk ID generation.

Isolated in its own module so the ID strategy can be changed later
(e.g. content-hash, deterministic) without modifying chunking logic.
"""

import uuid


def generate_chunk_id() -> str:
    """Return a new unique chunk identifier.

    Uses UUID4 — random, unique, and does not expose filesystem paths
    or user-controlled filenames.
    """
    return str(uuid.uuid4())
