#!/usr/bin/env bash
# Prints the body of the CHANGELOG.md section of VERSION (dated or not),
# without its heading, for the GitHub Release.
# Run from the repository root: .github/scripts/release-notes.sh 0.2.0
set -euo pipefail

version="${1:?usage: release-notes.sh VERSION}"
awk -v head="## [$version]" '
    index($0, head) == 1 { inside = 1; next }
    inside && /^## \[/ { exit }
    inside { print }
' CHANGELOG.md | sed -e '/./,$!d'
