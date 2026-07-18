"""variant-confidence: capa de confianza calibrada para predicciones de
variant-effect pathogenicity (AlphaMissense / ESM-1v / EVE).

Ver SDD.md para la especificación completa (AC1-AC9).
"""

from . import calib, data, metrics, split

__all__ = ["calib", "data", "metrics", "split"]
