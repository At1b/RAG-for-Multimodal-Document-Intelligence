"""Document ID generation.

Isolated in its own module so the ID strategy can be changed later
(e.g. content-hash, database-assigned) without modifying loaders.
"""

import uuid


def generate_document_id() -> str:
    """Return a new unique document identifier.

    Uses UUID4 — random, unique, and does not expose filesystem paths
    or user-controlled filenames.
    """
    return str(uuid.uuid4())
