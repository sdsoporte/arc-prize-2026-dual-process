#!/usr/bin/env bash
# Rebuild the Kaggle dataset `ser8147/laya-wheel`.
#
#   scripts/kaggle/laya_wheel_dataset.sh            # download + verify, print what it would do
#   scripts/kaggle/laya_wheel_dataset.sh --create   # also create/version the dataset on Kaggle
#
# Why this exists: the ARC-AGI-2 submission kernel runs with `enable_internet: False`, so it cannot
# `pip install laya` from PyPI. It needs the wheel as a declared dataset source. A dataset that was
# created by an undocumented manual command is not reproducible, so this script is the recipe.
#
# The wheel is redistributed unmodified. Upstream is Apache-2.0, and the dataset carries that licence
# rather than a CC0 default, because this is someone else's work.

set -euo pipefail
# shellcheck source=scripts/kaggle/lib.sh
# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

LAYA_VERSION="${LAYA_VERSION:-0.3.11}"
DATASET_SLUG="${DATASET_SLUG:-ser8147/laya-wheel}"
WHEEL="laya-${LAYA_VERSION}-py3-none-any.whl"
# Recorded 2026-09-24. Bumping LAYA_VERSION invalidates it on purpose: a new upstream release must be
# reviewed, not silently absorbed.
EXPECTED_SHA256="${EXPECTED_SHA256:-1ee717dd05a742135869383af9b66b20d5c5f62e62d9e57e3fcf84203ea1e443}"

CREATE=0
[ "${1:-}" = "--create" ] && CREATE=1

WORK="$(mktemp -d -t laya-wheel.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT

require_cmd python3
require_cmd sha256sum

section "1. Fetch laya==${LAYA_VERSION} from PyPI"
note "no deps: every required dependency is already in the Kaggle Python image"
python3 -m pip download "laya==${LAYA_VERSION}" --no-deps --quiet -d "$WORK"
[ -f "$WORK/$WHEEL" ] || die "expected $WHEEL, got: $(ls "$WORK")"

section "2. Verify the artifact hash"
actual="$(sha256sum "$WORK/$WHEEL" | cut -d' ' -f1)"
if [ "$actual" != "$EXPECTED_SHA256" ]; then
  die "sha256 mismatch
     expected $EXPECTED_SHA256
     actual   $actual
     If this is an intentional bump, set EXPECTED_SHA256 and review the new release first."
fi
note "sha256 ok: $actual"
note "$(stat -c%s "$WORK/$WHEEL") bytes"

section "3. Assemble the dataset"
cat > "$WORK/dataset-metadata.json" <<EOF
{
  "title": "Laya ${LAYA_VERSION} wheel (offline install source)",
  "id": "${DATASET_SLUG}",
  "licenses": [
    {
      "name": "Apache 2.0"
    }
  ]
}
EOF

cat > "$WORK/README.md" <<EOF
# Laya ${LAYA_VERSION} wheel — offline install source

The unmodified upstream \`laya\` wheel, republished so a Kaggle kernel running with
\`enable_internet: False\` can install it. PyPI is unreachable in that mode, and \`laya\` is not part of
the Kaggle Python image.

## Provenance

| Field | Value |
|---|---|
| Artifact | \`${WHEEL}\` |
| Source | PyPI, package \`laya\` version \`${LAYA_VERSION}\` |
| SHA-256 | \`${EXPECTED_SHA256}\` |
| License | **Apache-2.0** (upstream; retained unmodified) |

**Why Apache-2.0 and not CC0:** this redistributes someone else's work, so the dataset carries the
upstream licence. The wheel contains the upstream licence text.

## Usage

\`\`\`python
!pip install --no-index --find-links /kaggle/input/laya-wheel laya
\`\`\`

Required dependencies (\`torch>=2.0.0\`, \`transformers>=4.48.0\`, \`safetensors>=0.4.0\`,
\`huggingface_hub>=0.20.0\`, \`numpy>=1.20.0\`) are all already in the Kaggle Python image, so
\`--no-index\` resolves them locally. The remaining declared packages sit behind optional extras.

## Rebuild

\`\`\`bash
scripts/kaggle/laya_wheel_dataset.sh --create
\`\`\`
EOF

note "files prepared in $WORK"

section "4. Publish"
if [ "$CREATE" -eq 1 ]; then
  require_kaggle
  if kaggle datasets status "$DATASET_SLUG" >/dev/null 2>&1; then
    note "dataset exists -> creating a new version"
    kaggle datasets version -p "$WORK" -m "laya ${LAYA_VERSION}" -u
  else
    note "creating the dataset (public)"
    kaggle datasets create -p "$WORK" -u
  fi
  kaggle datasets files "$DATASET_SLUG"
else
  note "dry run. Re-run with --create to publish to $DATASET_SLUG"
  note "contents verified, nothing uploaded."
fi
