#!/usr/bin/env bash
# Checks that the tree is ready to publish VERSION: the crate version, an
# undated changelog section, no existing tag, and the package file list.
# Run from the repository root: .github/scripts/release-check.sh 0.2.0
set -euo pipefail

version="${1:?usage: release-check.sh VERSION}"
fail() {
    echo "release-check: $*" >&2
    exit 1
}

[[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || fail "version '$version' is not MAJOR.MINOR.PATCH"

crate_version="$(cargo metadata --no-deps --format-version 1 \
    | python3 -c 'import json, sys; print(json.load(sys.stdin)["packages"][0]["version"])')"
[[ "$crate_version" == "$version" ]] || fail "Cargo.toml version is $crate_version, not $version"

if git rev-parse -q --verify "refs/tags/v$version" >/dev/null \
    || git ls-remote --exit-code --tags origin "refs/tags/v$version" >/dev/null 2>&1; then
    fail "tag v$version already exists"
fi

heading="## [$version]"
count="$(grep -cxF "$heading" CHANGELOG.md || true)"
[[ "$count" == 1 ]] || fail "CHANGELOG.md needs exactly one undated '$heading' line (found $count)"
if grep -q "^## \[$version\] - " CHANGELOG.md; then
    fail "CHANGELOG.md already dates $version"
fi

# The package holds only the files .cursor/rules/publish.mdc lists.
bad="$(cargo package --list --allow-dirty | grep -Ev \
    '^(src/|examples/|tests/|benches/|compare/goldens/[^/]+\.json$|README\.md$|README\.ja\.md$|CHANGELOG\.md$|LICENSE-MIT$|LICENSE-APACHE$|Cargo\.toml$|Cargo\.toml\.orig$|Cargo\.lock$|\.cargo_vcs_info\.json$)' \
    || true)"
[[ -z "$bad" ]] || fail "files outside the package whitelist: $bad"

echo "release-check: $version is ready"
