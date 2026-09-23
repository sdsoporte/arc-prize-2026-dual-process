#!/usr/bin/env bash
# Shared configuration and helpers for this repository's Kaggle operations.
#
# Source this file; do not execute it directly.
#   source "$(dirname "$0")/lib.sh"
#
# Everything here exists because the Kaggle CLI 2.2.4 does NOT cover the Paper Track
# writeups, competition deadlines, submission limits or model version listings. Those
# go through the HTTP API with a bearer token instead. See docs/KAGGLE_OPS.md.

set -euo pipefail

# --- Identity / endpoints ----------------------------------------------------

# Kaggle CLI 2.2.4 authenticates from this file. There is no kaggle.json.
: "${KAGGLE_TOKEN_FILE:=$HOME/.kaggle/access_token}"
: "${KAGGLE_USER:=ser8147}"

# Competitions this repository competes in.
: "${PAPER_COMP:=arc-prize-2026-paper-track}"
: "${ARC2_COMP:=arc-prize-2026-arc-agi-2}"
: "${ARC3_COMP:=arc-prize-2026-arc-agi-3}"

# Public Kaggle API host. `api.kaggle.com` answers 404 for these routes; `www.kaggle.com` works.
: "${KAGGLE_API_BASE:=https://www.kaggle.com/api/v1}"

# Paper Track writeup identity, resolved 2026-09-23. Editing the writeup means editing THIS one;
# Competition-Specific Rule 2.2.a allows a hackathon team only one (1) Submission, so a second
# writeup must never be created.
: "${PAPER_WRITEUP_ID:=86160}"
: "${PAPER_WRITEUP_MESSAGE_ID:=114706}"

# --- Output helpers ----------------------------------------------------------

if [ -t 1 ]; then
  C_RESET=$'\033[0m'; C_BOLD=$'\033[1m'; C_DIM=$'\033[2m'
else
  C_RESET=''; C_BOLD=''; C_DIM=''
fi

section() { printf '\n%s== %s ==%s\n' "$C_BOLD" "$1" "$C_RESET"; }
note()    { printf '%s   %s%s\n' "$C_DIM" "$1" "$C_RESET"; }
die()     { printf 'error: %s\n' "$1" >&2; exit 1; }

# --- Preconditions -----------------------------------------------------------

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "'$1' not found on PATH."
}

require_kaggle() {
  require_cmd kaggle
  require_cmd curl
  require_cmd python3
  [ -f "$KAGGLE_TOKEN_FILE" ] || die "Kaggle token not found at $KAGGLE_TOKEN_FILE"
}

# --- Auth --------------------------------------------------------------------

# Prints the bearer token. Never echo its value into logs or docs.
kaggle_token() {
  tr -d '\r\n' < "$KAGGLE_TOKEN_FILE"
}

# Authenticated GET against the Kaggle public API. $1 = path (leading /), $2.. = curl extras.
api_get() {
  local path="$1"; shift
  curl -sS -H "Authorization: Bearer $(kaggle_token)" "$@" "$KAGGLE_API_BASE$path"
}

# Anonymous GET, used to answer "can the public see this?".
api_get_anon() {
  local path="$1"; shift
  curl -sS "$@" "$KAGGLE_API_BASE$path"
}

# Pretty-print JSON from stdin, tolerating an empty or non-JSON body.
json_pretty() {
  python3 -c 'import json,sys
raw = sys.stdin.read()
try:
    print(json.dumps(json.loads(raw), indent=2, ensure_ascii=False))
except Exception:
    sys.stdout.write(raw if raw.strip() else "<empty response>\n")'
}

# --- Writeups (Paper Track only) --------------------------------------------

# The Paper Track submission IS a writeup. The CLI has no command for this; the API route is
# /api/v1/competitions/{competition}/hackathon-write-ups.
# Owner-scoped calls return drafts + published. Anonymous calls return nothing for this
# competition - participant writeups are not publicly readable during the competition.

writeups_list() { api_get "/competitions/$PAPER_COMP/hackathon-write-ups?pageSize=50"; }
writeup_get()   { api_get "/competitions/$PAPER_COMP/hackathon-write-ups/$1"; }
