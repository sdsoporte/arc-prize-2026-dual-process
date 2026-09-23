#!/usr/bin/env bash
# Dump the complete live Kaggle state for this repository, in one command.
#
#   scripts/kaggle/kstate.sh
#
# Why this exists: Kaggle artifact state drifts from the repository silently. On 2026-09-23 the
# ARC-AGI-3 and ARC-AGI-2 kernels on Kaggle were older than the local code, and Kaggle Models held
# only v2 of the Laya model while the paper cited a v1 checkpoint that no longer existed. Nothing in
# the repo could have revealed either fact.
#
# This command is READ-ONLY. It never uploads, submits, deletes or edits anything.

set -euo pipefail
# shellcheck source=lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
require_kaggle

# --- 1. Identity and auth ----------------------------------------------------

section "1. Identity and auth"
kaggle config view 2>/dev/null | sed 's/^/   /' || note "kaggle config view failed"

# --- 2. Accelerator quota ----------------------------------------------------

section "2. Accelerator quota (weekly)"
note "GPU hours are the scarce resource for training and for ARC-AGI-3 reruns."
kaggle quota 2>/dev/null | sed 's/^/   /' || note "kaggle quota failed"

# --- 3. Deadlines and submission limits --------------------------------------

section "3. Deadlines and submission limits"
note "maxDailySubmissions is the platform cap. For the Paper Track the binding limit is instead"
note "Competition-Specific Rule 2.2.a: a hackathon team may submit ONE submission total."
python3 - "$PAPER_COMP" "$ARC2_COMP" "$ARC3_COMP" <<'PY' || note "deadline lookup failed"
import sys
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_api_service import ApiGetCompetitionRequest
api = KaggleApi(); api.authenticate()
print(f"   {'competition':<34}{'deadline (UTC)':<22}{'subs/day':>9}{'max team':>10}")
with api.build_kaggle_client() as k:
    for name in sys.argv[1:]:
        req = ApiGetCompetitionRequest(); req.competition_name = name
        d = k.competitions.competition_api_client.get_competition(req).to_dict()
        print(f"   {name:<34}{d.get('deadline',''):<22}{d.get('maxDailySubmissions',''):>9}{d.get('maxTeamSize',''):>10}")
PY

# --- 4. Kernels --------------------------------------------------------------

section "4. Kernels (notebooks)"
note "lastRunTime tells you whether deployed code matches the repository."
kaggle kernels list --user "$KAGGLE_USER" --page-size 30 2>/dev/null | sed 's/^/   /' \
  || note "kernel listing failed"

# --- 5. Datasets -------------------------------------------------------------

section "5. Datasets"
kaggle datasets list --user "$KAGGLE_USER" 2>/dev/null | sed 's/^/   /' \
  || note "dataset listing failed"

# --- 6. Models and published versions ---------------------------------------

section "6. Models and published versions"
note "Compare against models/ in the repo. A repo checkpoint with no published version is not"
note "reproducible by anyone, including the paper's readers."
kaggle models list --owner "$KAGGLE_USER" 2>/dev/null | sed 's/^/   /' \
  || note "model listing failed"
note "'kaggle models instances list' shows only the LATEST version and will hide an older published"
note "version. Use 'variations versions list' to see them all - this is how the v1 checkpoint stayed hidden."
kaggle models variations versions list "$KAGGLE_USER/arc-laya/Transformers/typed-decisions" 2>/dev/null \
  | sed 's/^/   /' || note "model version listing failed"

# --- 7-8. Competition submissions -------------------------------------------

for label in "ARC-AGI-2:$ARC2_COMP" "ARC-AGI-3:$ARC3_COMP"; do
  section "7. ${label%%:*} submissions"
  kaggle competitions submissions -c "${label##*:}" 2>&1 | sed 's/^/   /' || true
done
for label in "ARC-AGI-2:$ARC2_COMP" "ARC-AGI-3:$ARC3_COMP"; do
  printf '   %s remaining today: ' "${label%%:*}"
  kaggle competitions submission-limits -c "${label##*:}" 2>/dev/null \
    | tr '\n' ' ' | sed 's/  */ /g' || printf 'unavailable\n'
  printf '\n'
done

# --- 9. Paper Track writeups -------------------------------------------------

section "9. Paper Track writeups (owner-scoped; anonymous sees none)"
note "The Paper Track submission is a WRITEUP, not a competitions submission. The CLI has no"
note "command for it; this reads /api/v1/competitions/<competition>/hackathon-write-ups."
wlist_tmp="$(mktemp -t kstate-writeups.XXXXXX.json)"
writeups_list > "$wlist_tmp"
python3 - "$wlist_tmp" <<'PY' || note "writeup listing failed"
import json, sys
d = json.load(open(sys.argv[1]))
rows = d.get("hackathonWriteUps") or []
print("   totalCount=%s  returned=%d" % (d.get("totalCount"), len(rows)))
for r in rows:
    w = r.get("writeUp") or {}
    kind = "TEMPLATE" if r.get("template") else "PARTICIPANT"
    print("   - [%s] writeupId=%s writeUpId=%s state=%s" % (kind, r.get("id"), w.get("id"), w.get("contentState")))
    print("     title=%r  updated=%s" % (str(w.get("title"))[:52], w.get("updateTime")))
PY
rm -f "$wlist_tmp"

section "done"
note "Paper Track deadline is the LATEST of the three: code freezes first, then the paper."
