#!/usr/bin/env python3
"""bandit JSON -> SARIF 2.1.0 converter (memlineage-proven, adapted).

Bandit's native SARIF export is unreliable across versions. This converter
reads bandit's JSON report and emits a SARIF 2.1.0 doc that GitHub Code
Scanning accepts. Paths are validated (no traversal) so a malicious bandit
result cannot inject "../../etc/passwd" into the SARIF uri.

Usage:
    python scripts/bandit2sarif.py bandit.json bandit.sarif
"""
from __future__ import annotations

import json
import os
import sys


def _safe_rel(path: str) -> str:
    """Return a repo-relative, traversal-free path; fall back to basename."""
    norm = os.path.normpath(path)
    if norm.startswith(("..", "/")) or os.path.isabs(norm):
        return os.path.basename(norm)
    return norm


def convert(bandit_json: dict) -> dict:
    results = []
    for finding in bandit_json.get("results", []):
        fname = _safe_rel(finding.get("filename", "unknown.py"))
        line = int(finding.get("line_number", 1))
        issue_id = finding.get("issue_id", "")
        test_id = finding.get("test_id", "")
        rule_id = f"bandit-{test_id}-{issue_id}" if test_id else "bandit"
        results.append({
            "ruleId": rule_id,
            "level": "warning",
            "message": {
                "text": f"{finding.get('issue_text', '')} "
                        f"(CWE: {finding.get('cwe', 'n/a')})",
            },
            "locations": [{
                "physicalLocation": {
                    "artifactLocation": {"uri": fname},
                    "region": {
                        "startLine": line,
                        "endLine": line,
                        "snippet": {"text": finding.get("code", "")[:200]},
                    },
                }
            }],
            "properties": {
                "severity": finding.get("issue_severity", "UNKNOWN"),
                "confidence": finding.get("issue_confidence", "UNKNOWN"),
            },
        })
    return {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [{
            "tool": {
                "driver": {
                    "name": "bandit",
                    "informationUri": "https://bandit.readthedocs.io/",
                    "version": bandit_json.get("version", "unknown"),
                    "rules": [{
                        "id": r["ruleId"],
                        "shortDescription": {"text": "bandit security finding"},
                    } for r in results],
                }
            },
            "results": results,
        }],
    }


def main(argv) -> int:
    if len(argv) != 3:
        print("usage: bandit2sarif.py <in.json> <out.sarif>", file=sys.stderr)
        return 2
    with open(argv[1], "r", encoding="utf-8") as fh:
        data = json.load(fh)
    sarif = convert(data)
    with open(argv[2], "w", encoding="utf-8") as fh:
        json.dump(sarif, fh, indent=2)
    print(f"wrote {argv[2]} ({len(sarif['runs'][0]['results'])} results)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
