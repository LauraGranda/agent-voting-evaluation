"""Voting subsystem: aggregation of judge-agent scores into a final relevance score."""

from .aggregator import SCHEME_NAME, aggregate

__all__ = ["SCHEME_NAME", "aggregate"]
