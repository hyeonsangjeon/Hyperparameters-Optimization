#!/usr/bin/env python
"""Backward-compatible entry point for the HPO learning lab CLI."""

from __future__ import annotations

import sys

from hpo_lab.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["benchmark", *sys.argv[1:]]))
