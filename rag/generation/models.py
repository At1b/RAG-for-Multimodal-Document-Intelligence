"""Data models for the generation layer.

``GenerationResult`` is the return type of the ``Generator.generate``
method.  It carries the generated answer and optional metadata about
the model that produced it.

This model does NOT duplicate retrieval result information.
Citation formatting belongs to Phase 7.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class GenerationResult(BaseModel):
    """Result of an LLM generation call.

    Attributes:
        answer: The generated answer text.
        model_name: Identifier of the model that produced the answer.
            Empty string when unknown.
        metadata: Arbitrary metadata about the generation
            (e.g. token counts, latency, temperature used).
    """

    answer: str = Field(..., min_length=1)
    model_name: str = Field(default="")
    metadata: dict[str, Any] = Field(default_factory=dict)
