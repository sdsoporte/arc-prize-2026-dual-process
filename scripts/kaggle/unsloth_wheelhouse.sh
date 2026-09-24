#!/usr/bin/env bash
# Rebuild the Kaggle dataset `ser8147/arc2-unsloth-wheelhouse`.
#
#   scripts/kaggle/unsloth_wheelhouse.sh            # download + verify, print what it would do
#   scripts/kaggle/unsloth_wheelhouse.sh --create   # also create/version the dataset on Kaggle
#
# Why this exists: the ARC-AGI-2 reference kernel (`notebooks/arc2_reference_kernel/`) runs with
# `enable_internet: False`, and the Kaggle Python image (3.12.13) does not ship what the pipeline
# imports. Measured in the target environment on 2026-09-24 (the kernel's own preflight, `pip list`):
#
#   package          image (before)   required by                                    action here
#   unsloth          absent           arc_solver.py imports it unconditionally        vendored
#   unsloth_zoo      absent           unsloth's own dependency                       vendored
#   trl              absent           UnslothTrainer subclasses trl                   vendored
#   bitsandbytes     absent           Unsloth's quantised path                       vendored
#   xformers         absent           unsloth's hard linux/x86_64 requirement        vendored
#   torchao          0.10.0           unsloth_zoo requires >=0.13.0                  vendored
#   cut_cross_entropy absent          unsloth_zoo dependency                          vendored
#   tyro             absent           unsloth dependency                             vendored
#   msgspec          absent           unsloth_zoo dependency                         vendored
#   structlog        absent           unsloth dependency                             vendored
#   transformers     5.0.0            unsloth excludes 5.0.0 and 5.1.0                downgraded
#   datasets         5.0.0            unsloth requires <4.4.0                        downgraded
#   huggingface_hub  1.11.0           transformers 4.57.6 requires <1.0              downgraded
#   dill             0.4.1            datasets 4.3.0 requires <0.4.1                 downgraded
#
# Everything else the pipeline needs is already in the image at a satisfying version, so it is
# deliberately NOT vendored: tokenizers 0.22.2, safetensors 0.7.0, diffusers 0.37.1, fsspec
# 2025.3.0, importlib-metadata 8.7.1, zipp 3.23.1, typer 0.24.2, shellingham 1.5.4, annotated-doc
# 0.0.4, multiprocess 0.70.16, sentencepiece 0.2.1, nest-asyncio 1.6.0, torchvision 0.25.0+cu128,
# docstring-parser 0.18.0, typeguard 4.5.1, hf-xet 1.4.3. Vendoring those was the first cut of this
# script and it was wrong: pinning fsspec or importlib-metadata needlessly changes a package the
# image already satisfies and trips unrelated pins (gcsfs wants exactly fsspec 2025.3.0;
# opentelemetry-api wants importlib-metadata <8.8.0).
#
# So the kernel needs a declared offline wheel source. This script is the recipe: pinned versions, a
# recorded SHA-256 per wheel, a generated provenance README, a dependency-closure check that fails
# before publishing, dry-run by default, `--create` to publish.
#
# The wheels are redistributed unmodified. They are a licence mixture (LGPL-3.0 for unsloth and
# unsloth_zoo, Apache-2.0, MIT, BSD-3-Clause, ...); the README records the per-package licence, and
# every wheel carries its own upstream licence text.

set -euo pipefail
# shellcheck source=scripts/kaggle/lib.sh
# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

# --- Target environment ------------------------------------------------------
# The Kaggle Python image this wheelhouse is built for. Verified from the kernel's own preflight log
# on 2026-09-24: `python : 3.12.13 | Linux-6.12.90+-x86_64-with-glibc2.35`.
# Bumping either value invalidates every wheel here on purpose.
TARGET_PYTHON="${TARGET_PYTHON:-3.12}"
TARGET_IMPLEMENTATION="${TARGET_IMPLEMENTATION:-cp}"
TARGET_ABI="${TARGET_ABI:-cp312}"
TARGET_PLATFORMS=(
  manylinux_2_28_x86_64
  manylinux_2_24_x86_64
  manylinux_2_17_x86_64
  manylinux2014_x86_64
  linux_x86_64
)

DATASET_SLUG="${DATASET_SLUG:-ser8147/arc2-unsloth-wheelhouse}"
DATASET_TITLE="${DATASET_TITLE:-ARC-AGI-2 unsloth offline wheelhouse (py3.12)}"

# Must be kept in sync with the PINNED list in
# notebooks/arc2_reference_kernel/submission.ipynb; the notebook asserts the two agree.
PINNED=(
  # --- the packages the image does not ship at all ---------------------------
  "unsloth==2026.9.11"
  "unsloth_zoo==2026.9.7"
  "trl==0.24.0"
  "bitsandbytes==0.50.2"
  "xformers==0.0.34"
  # --- packages the image ships, but at a version the pipeline cannot accept --
  "transformers==4.57.6"
  "datasets==4.3.0"
  "huggingface_hub==0.36.2"
  "dill==0.4.0"
  "torchao==0.16.0"
  # --- dependencies of the above that the image does not ship ----------------
  "cut_cross_entropy==25.1.1"
  "tyro==1.0.16"
  "msgspec==0.21.1"
  "structlog==26.1.0"
  "hf_transfer==0.1.9"
)

# Distributions the Kaggle image already provides, at the version MEASURED in the target environment
# on 2026-09-24 (printed by the kernel's own preflight `pip list --format=freeze`). Nothing here is
# guessed: every value was read out of that run. A requirement this map cannot satisfy is a gap that
# the closure check below reports, instead of letting it reach a GPU run.
IMAGE_PROVIDES=$(cat <<'JSON'
{
  "torch": "2.10.0+cu128",
  "torchvision": "0.25.0+cu128",
  "triton": "3.6.0",
  "accelerate": "1.13.0",
  "peft": "0.19.1",
  "tensorflow": "2.20.0",
  "tokenizers": "0.22.2",
  "safetensors": "0.7.0",
  "diffusers": "0.37.1",
  "fsspec": "2025.3.0",
  "importlib-metadata": "8.7.1",
  "zipp": "3.23.1",
  "typer": "0.24.2",
  "shellingham": "1.5.4",
  "annotated-doc": "0.0.4",
  "docstring-parser": "0.18.0",
  "typeguard": "4.5.1",
  "multiprocess": "0.70.16",
  "sentencepiece": "0.2.1",
  "nest-asyncio": "1.6.0",
  "hf-xet": "1.4.3",
  "numpy": "2.0.2",
  "pandas": "2.3.3",
  "pyarrow": "24.0.0",
  "pillow": "11.3.0",
  "regex": "2025.11.3",
  "protobuf": "5.29.5",
  "packaging": "26.1",
  "filelock": "3.29.0",
  "requests": "2.32.4",
  "httpx": "0.28.1",
  "aiohttp": "3.13.5",
  "xxhash": "3.6.0",
  "pyyaml": "6.0.3",
  "click": "8.3.3",
  "rich": "13.9.4",
  "tqdm": "4.67.3",
  "psutil": "5.9.5",
  "typing-extensions": "4.15.0",
  "pydantic": "2.12.3",
  "pydantic-core": "2.41.4",
  "setuptools": "81.0.0",
  "wheel": "0.47.0",
  "six": "1.17.0",
  "python-dateutil": "2.9.0.post0",
  "jinja2": "3.1.6"
}
JSON
)

CREATE=0
[ "${1:-}" = "--create" ] && CREATE=1

WORK="$(mktemp -d -t unsloth-wheelhouse.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
# Wheels land at the dataset ROOT on purpose: the kernel points
# `pip install --no-index --find-links /kaggle/input/<dataset>` straight at this directory, so there
# is no second path to get wrong. pip ignores every non-wheel file here.
WHEELS="$WORK"

require_cmd python3
require_cmd sha256sum

section "1. Download ${#PINNED[@]} pinned wheels for python ${TARGET_PYTHON}"
note "target platforms: ${TARGET_PLATFORMS[*]}"
note "torch itself is NOT vendored: the image's own 2.10.0+cu128 satisfies unsloth's torch<2.13.0,>=2.4.0"
note "and republishing it would clobber the image's CUDA build."

download_args=(
  --no-deps
  --only-binary=:all:
  --python-version "$TARGET_PYTHON"
  --implementation "$TARGET_IMPLEMENTATION"
  --abi "$TARGET_ABI"
  --abi abi3
  --abi none
)
for platform in "${TARGET_PLATFORMS[@]}"; do
  download_args+=(--platform "$platform")
done

python3 -m pip download "${download_args[@]}" "${PINNED[@]}" -d "$WHEELS" --quiet
mapfile -t downloaded < <(find "$WHEELS" -maxdepth 1 -name '*.whl' -printf '%f\n' | sort)
wheel_count="${#downloaded[@]}"
[ "$wheel_count" -eq "${#PINNED[@]}" ] \
  || die "expected ${#PINNED[@]} wheels, downloaded $wheel_count: ${downloaded[*]}"
note "downloaded $wheel_count wheels"

section "2. Verify and record every artifact"
: > "$WORK/WHEELS.sha256"
printf '%s\n' "${PINNED[@]}" > "$WORK/PINNED.txt"
(
  cd "$WHEELS"
  for wheel in *.whl; do
    sha256sum "$wheel" >> "$WORK/WHEELS.sha256"
  done
  sort -o "$WORK/WHEELS.sha256" "$WORK/WHEELS.sha256"
)
note "sha256 recorded for $wheel_count wheels"
note "total size: $(du -sh "$WHEELS" | cut -f1)"

section "3. Dependency-closure check (fails before publishing)"
# Every Requires-Dist of every vendored wheel, with markers evaluated for the target environment,
# must be satisfied by another vendored wheel or by a distribution the image already provides.
# This is what stops a half-vendored dependency set from reaching a GPU run.
python3 - "$WHEELS" "$IMAGE_PROVIDES" "$WORK/LICENSE_TABLE.md" <<'PY'
import json
import sys
import zipfile
from pathlib import Path

from packaging.markers import Marker
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import parse_wheel_filename
from packaging.version import Version

wheels_dir = Path(sys.argv[1])
image_provides = json.loads(sys.argv[2])
license_out = Path(sys.argv[3])

ENV = {
    "python_version": "3.12",
    "python_full_version": "3.12.13",
    "platform_machine": "x86_64",
    "platform_system": "Linux",
    "sys_platform": "linux",
    "os_name": "posix",
    "implementation_name": "cpython",
    "platform_python_implementation": "CPython",
    "extra": "",
}


def canon(name):
    """PEP 503-ish canonicalisation, so `hf_xet`, `hf-xet` and `HF.XET` are one name."""
    return name.lower().replace("_", "-").replace(".", "-")


# The Kaggle dataset carries one aggregate licence; this map is the per-package record that the
# aggregate cannot express. Values are taken from the wheel's own metadata first, and from the
# licence file it bundles last, so nothing here is guessed.
CLASSIFIER_LICENSES = {
    "Apache Software License": "Apache-2.0",
    "BSD License": "BSD-3-Clause",
    "MIT License": "MIT",
    "ISC License (ISCL)": "ISC",
}
LICENSE_ALIASES = {
    "Apache 2.0 License": "Apache-2.0",
    "Apache License": "Apache-2.0",
    "Apache": "Apache-2.0",
    "BSD": "BSD-3-Clause",
    "MIT License": "MIT",
}


def bundled_license(wheel):
    """First non-empty line of the licence file the wheel ships, as a last resort."""
    with zipfile.ZipFile(wheel) as zf:
        names = [
            candidate
            for candidate in zf.namelist()
            if candidate.upper().endswith(("LICENSE", "LICENSE.TXT", "COPYING"))
        ]
        if not names:
            return None, None
        text = zf.read(sorted(names, key=len)[0]).decode("utf-8", "replace")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return (lines[0][:90] if lines else None), sorted(names, key=len)[0]


def license_of(wheel, meta):
    if meta.get("License-Expression"):
        return meta["License-Expression"]
    raw = (meta.get("License") or "").strip().splitlines()
    if raw and raw[0].strip() and len(raw[0].strip()) <= 40:
        value = raw[0].strip().strip('"')
        return LICENSE_ALIASES.get(value, value)
    for classifier in (meta.get("Classifiers") or "").split("; "):
        classifier = classifier.strip()
        if classifier.startswith("License ::"):
            tail = classifier.split("::")[-1].strip()
            return CLASSIFIER_LICENSES.get(tail, tail)
    first_line, path = bundled_license(wheel)
    if first_line:
        return f"see bundled {path} (starts: {first_line!r})"
    return "no licence declared in the wheel metadata or files"


def metadata_of(wheel):
    with zipfile.ZipFile(wheel) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".dist-info/METADATA"))
        raw = zf.read(name).decode("utf-8", "replace")
    fields = {}
    requires = []
    for line in raw.splitlines():
        if line.startswith("Requires-Dist:"):
            requires.append(line.split(":", 1)[1].strip())
        elif line.startswith(("Name:", "Version:", "License:", "License-Expression:")):
            key, _, value = line.partition(":")
            fields[key.strip()] = value.strip()
        elif line.startswith("Classifier: License"):
            fields.setdefault("Classifiers", "")
            fields["Classifiers"] += line.split(":", 1)[1].strip() + "; "
    fields["Requires-Dist"] = requires
    return fields


vendored = {}
licenses = {}
for wheel in sorted(wheels_dir.glob("*.whl")):
    parsed = parse_wheel_filename(wheel.name)
    project = parsed[0]
    meta = metadata_of(wheel)
    vendored[canon(project)] = Version(meta["Version"])
    licenses[canon(project)] = license_of(wheel, meta)

provided = {canon(name): Version(version) for name, version in image_provides.items()}

print("vendored: %d distributions" % len(vendored))
gaps = []
for wheel in sorted(wheels_dir.glob("*.whl")):
    project = canon(parse_wheel_filename(wheel.name)[0])
    meta = metadata_of(wheel)
    for raw in meta["Requires-Dist"]:
        requirement = Requirement(raw)
        if requirement.marker is not None and not requirement.marker.evaluate(ENV):
            continue
        name = canon(requirement.name)
        candidate = vendored.get(name)
        if candidate is None:
            candidate = provided.get(name)
        if candidate is None:
            gaps.append(f"{project}: requires {raw} -- not vendored and not known in the image")
            continue
        if requirement.specifier and candidate not in requirement.specifier:
            gaps.append(
                f"{project}: requires {raw} -- available {name}=={candidate} does not satisfy it"
            )

if gaps:
    print("\nDEPENDENCY CLOSURE BROKEN:", file=sys.stderr)
    for gap in gaps:
        print("  -", gap, file=sys.stderr)
    sys.exit(1)
print("closure OK: every Requires-Dist is satisfied by a vendored wheel or an image package")

rows = sorted(
    "| `%s` | %s | %s |" % (name, version, licenses[name].replace("\n", " "))
    for name, version in vendored.items()
)
license_out.write_text(
    "| distribution | version | declared licence |\n|---|---|---|\n" + "\n".join(rows) + "\n"
)
PY

section "4. Assemble the dataset"
cat > "$WORK/dataset-metadata.json" <<EOF
{
  "title": "${DATASET_TITLE}",
  "id": "${DATASET_SLUG}",
  "licenses": [
    {
      "name": "Other"
    }
  ]
}
EOF

{
  echo "# ${DATASET_TITLE} — offline install source"
  echo
  echo "Wheels for the ARC-AGI-2 reference kernel (\`notebooks/arc2_reference_kernel/\`), which runs with"
  echo "\`enable_internet: False\`. Nothing here is modified: every wheel is the unmodified PyPI artifact,"
  echo "and each one carries its own upstream licence text."
  echo
  echo "## Why this is needed"
  echo
  echo "Measured in the target environment on 2026-09-24 by the kernel's own preflight"
  echo "(\`pip list --format=freeze\`, Kaggle Python 3.12.13, torch 2.10.0+cu128):"
  echo
  echo '```text'
  echo "package           image (before)   required by the pipeline                     action here"
  echo "unsloth           absent           arc_solver.py imports it unconditionally     vendored"
  echo "unsloth_zoo       absent           unsloth's own dependency                    vendored"
  echo "trl               absent           UnslothTrainer subclasses trl                vendored"
  echo "bitsandbytes      absent           Unsloth's quantised path                    vendored"
  echo "xformers          absent           unsloth's hard linux/x86_64 requirement     vendored"
  echo "torchao           0.10.0           unsloth_zoo requires >=0.13.0               vendored"
  echo "cut_cross_entropy absent           unsloth_zoo dependency                       vendored"
  echo "tyro              absent           unsloth dependency                          vendored"
  echo "msgspec           absent           unsloth_zoo dependency                      vendored"
  echo "structlog         absent           unsloth dependency                          vendored"
  echo "transformers      5.0.0            unsloth excludes 5.0.0 and 5.1.0             DOWNGRADED"
  echo "datasets          5.0.0            unsloth requires <4.4.0                     DOWNGRADED"
  echo "huggingface_hub   1.11.0           transformers 4.57.6 requires <1.0           DOWNGRADED"
  echo "dill              0.4.1            datasets 4.3.0 requires <0.4.1              DOWNGRADED"
  echo '```'
  echo
  echo "Everything else the pipeline needs is already in the image at a satisfying version and is"
  echo "deliberately NOT vendored, because pinning a package the image already satisfies only risks"
  echo "breaking an unrelated pin:"
  echo
  echo '```text'
  echo "tokenizers 0.22.2  safetensors 0.7.0  diffusers 0.37.1  fsspec 2025.3.0  importlib-metadata 8.7.1"
  echo "zipp 3.23.1  typer 0.24.2  shellingham 1.5.4  annotated-doc 0.0.4  multiprocess 0.70.16"
  echo "sentencepiece 0.2.1  nest-asyncio 1.6.0  torchvision 0.25.0+cu128  docstring-parser 0.18.0"
  echo "typeguard 4.5.1  hf-xet 1.4.3  plus the scientific stack (numpy, pandas, pyarrow, pillow, ...)"
  echo '```'
  echo
  echo "## Provenance"
  echo
  echo "| Field | Value |"
  echo "|---|---|"
  echo "| Source | PyPI, exact pinned versions below |"
  echo "| Target | CPython ${TARGET_PYTHON}, ${TARGET_PLATFORMS[0]} |"
  echo "| Integrity | \`WHEELS.sha256\` (one SHA-256 per wheel) |"
  echo "| Licences | mixed: LGPL-3.0-or-later (unsloth_zoo), Apple (cut_cross_entropy), Apache-2.0, MIT, BSD |"
  echo "| Rebuild | \`scripts/kaggle/unsloth_wheelhouse.sh --create\` |"
  echo
  echo "**Why the licence is \`Other\`:** the bundle redistributes ${#PINNED[@]} different projects. Two of"
  echo "them cannot be expressed by a single OSI tag: \`unsloth_zoo\` is **LGPL-3.0-or-later**, and"
  echo "\`cut_cross_entropy\` carries Apple's own software licence. The rest are Apache-2.0, MIT, BSD-3-Clause,"
  echo "ISC and similar; the per-package table below is the authoritative record, and every wheel bundles its"
  echo "own upstream licence text. \`torch\` is deliberately NOT vendored — the image's own 2.10.0+cu128"
  echo "satisfies unsloth's \`torch<2.13.0,>=2.4.0\`, and republishing it would clobber the image's CUDA build."
  echo
  cat "$WORK/LICENSE_TABLE.md"
  echo
  echo "## Files"
  echo
  echo '```text'
  (
    cd "$WHEELS"
    for w in ./*.whl; do
      printf '%s  (%s bytes)\n' "${w#./}" "$(stat -c%s "$w")"
    done
  )
  echo '```'
  echo
  echo "## Usage"
  echo
  echo '```python'
  echo '!pip install --no-index --find-links /kaggle/input/arc2-unsloth-wheelhouse -r \'
  echo '    /kaggle/input/arc2-unsloth-wheelhouse/PINNED.txt'
  echo '```'
  echo
  echo "All of these requirements are additionally satisfied by this directory, and everything else the"
  echo "pipeline needs (torch, accelerate, peft, numpy, pandas, pyarrow, pillow, ...) is already in the"
  echo "Kaggle Python image, so \`--no-index\` resolves entirely offline."
} > "$WORK/README.md"

note "files prepared in $WORK"

section "5. Publish"
if [ "$CREATE" -eq 1 ]; then
  require_kaggle
  if kaggle datasets status "$DATASET_SLUG" >/dev/null 2>&1; then
    note "dataset exists -> creating a new version"
    # No -u here: `datasets version` has no --public flag, and a new version inherits the
    # visibility of the dataset it belongs to (this one was created public).
    kaggle datasets version -p "$WORK" -m "unsloth ${PINNED[0]#unsloth==} offline wheelhouse (py${TARGET_PYTHON})"
  else
    note "creating the dataset (public)"
    kaggle datasets create -p "$WORK" -u
  fi
  kaggle datasets files "$DATASET_SLUG"
else
  note "dry run. Re-run with --create to publish to $DATASET_SLUG"
  note "contents verified, nothing uploaded."
fi
