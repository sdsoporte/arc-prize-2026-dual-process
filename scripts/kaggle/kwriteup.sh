#!/usr/bin/env bash
# Inspect and audit the ARC Prize 2026 Paper Track submission (the writeup).
#
#   scripts/kaggle/kwriteup.sh list
#   scripts/kaggle/kwriteup.sh get    [writeupId]
#   scripts/kaggle/kwriteup.sh dump   [writeupId] [-o FILE]
#   scripts/kaggle/kwriteup.sh diff   [writeupId] [--against FILE]
#   scripts/kaggle/kwriteup.sh check  [writeupId]
#
# Why this exists: the Paper Track submission is a WRITEUP. The Kaggle CLI 2.2.4 has no writeup
# command, `www.kaggle.com/api/v1/writeups/*` returns 404, and the eligibility rules that apply to a
# writeup (word limit, cover image, public notebooks, repository link, license, one submission only)
# are spread across three different competition pages.
#
# This command is READ-ONLY. It never edits or submits the writeup.

set -euo pipefail
# shellcheck source=scripts/kaggle/lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
require_kaggle

DEFAULT_AGAINST="submissions/KAGGLE_WRITEUP.md"
WORD_LIMIT=1500

# One scratch directory for the whole run, removed on exit. Deliberately not per-function:
# a `trap ... RETURN` set inside a called function replaces the caller's trap, which leaks files
# and, under `set -u`, aborts on an unset variable during cleanup.
WORKDIR="$(mktemp -d -t kwriteup.XXXXXX)"
trap 'rm -rf "$WORKDIR"' EXIT

usage() {
  sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

# Fetches writeup $1 into $2 as JSON.
fetch_writeup() {
  writeup_get "${1:-$PAPER_WRITEUP_ID}" > "$2"
  [ -s "$2" ] || die "empty response for writeup ${1:-$PAPER_WRITEUP_ID}"
  grep -q 'writeUp' "$2" || die "response has no writeUp object (bad id or no access)"
}

extract_markdown() {
  python3 - "$1" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
sys.stdout.write(((d.get("writeUp") or {}).get("message") or {}).get("rawMarkdown") or "")
PY
}

# --- list --------------------------------------------------------------------

cmd_list() {
  local f="$WORKDIR/writeups.json"
  writeups_list > "$f"
  section "Paper Track writeups"
  note "Owner-scoped: includes drafts. An anonymous caller sees none for this competition."
  python3 - "$f" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
rows = d.get("hackathonWriteUps") or []
print("totalCount=%s  returned=%d" % (d.get("totalCount"), len(rows)))
for r in rows:
    w = r.get("writeUp") or {}
    kind = "TEMPLATE" if r.get("template") else "PARTICIPANT"
    print("  [%s] writeupId=%s messageId=%s state=%s" % (kind, r.get("id"), w.get("id"), w.get("contentState")))
    print("             title       = %r" % (w.get("title"),))
    print("             url         = https://www.kaggle.com%s" % (w.get("url"),))
    print("             publishTime = %s" % (w.get("publishTime"),))
    print("             updateTime  = %s" % (w.get("updateTime"),))
PY
}

# --- get / dump --------------------------------------------------------------

cmd_get() {
  local f="$WORKDIR/writeup.json"
  fetch_writeup "${1:-$PAPER_WRITEUP_ID}" "$f"
  json_pretty < "$f"
}

cmd_dump() {
  local id="$PAPER_WRITEUP_ID" out=""
  while [ $# -gt 0 ]; do
    case "$1" in
      -o|--out) out="${2:?missing value for $1}"; shift 2 ;;
      -*) die "unknown option: $1" ;;
      *) id="$1"; shift ;;
    esac
  done
  local f="$WORKDIR/dump.json"
  fetch_writeup "$id" "$f"
  if [ -n "$out" ]; then
    extract_markdown "$f" > "$out"
    note "wrote $(wc -c < "$out") bytes of live markdown to $out"
  else
    extract_markdown "$f"
  fi
}

# --- diff --------------------------------------------------------------------

cmd_diff() {
  local id="$PAPER_WRITEUP_ID" against="$DEFAULT_AGAINST"
  while [ $# -gt 0 ]; do
    case "$1" in
      --against) against="${2:?missing value for --against}"; shift 2 ;;
      -*) die "unknown option: $1" ;;
      *) id="$1"; shift ;;
    esac
  done
  [ -f "$against" ] || die "local file not found: $against (run from the repository root)"
  local a="$WORKDIR/local.md" b="$WORKDIR/live.md"
  cp "$against" "$a"
  cmd_dump "$id" -o "$b" >/dev/null
  section "live writeup $id  vs  $against"
  if diff -u "$a" "$b"; then
    note "identical"
  else
    note "'-' is $against (local); '+' is what the judges currently read"
    return 0   # differences are a finding, not a tool failure
  fi
}

# --- check -------------------------------------------------------------------

cmd_check() {
  local id="${1:-$PAPER_WRITEUP_ID}"
  local f="$WORKDIR/check.json"
  fetch_writeup "$id" "$f"
  section "Eligibility and integrity audit -- writeup $id"
  python3 - "$f" "$WORD_LIMIT" <<'PY'
import json, re, subprocess, sys

d = json.load(open(sys.argv[1]))
limit = int(sys.argv[2])
w = d.get("writeUp") or {}
md = ((w.get("message") or {}).get("rawMarkdown")) or ""
links = w.get("writeUpLinks") or []

def check(ok):
    return "PASS" if ok else "FAIL"

stripped = re.sub(r"```.*?```", "", md, flags=re.S)
stripped = re.sub(r"[#*_>`\[\]()|-]", " ", stripped)
words = len([t for t in stripped.split() if any(c.isalnum() for c in t)])

print("  [%s] word count              %d / %d   (Submission Requirements page)" % (check(words <= limit), words, limit))
print("  [%s] contentState            %s" % (check(w.get("contentState") == "published"), w.get("contentState")))
print("  [%s] published               %s" % (check(bool(w.get("publishTime"))), w.get("publishTime")))
print("  [%s] track selected          %s   (required in order to submit)" % (check(bool(d.get("hackathonTrackIds"))), d.get("hackathonTrackIds")))
print("  [%s] cover image             %s" % (check(bool(w.get("coverImageUrl"))), w.get("coverImageUrl")))
lic = (w.get("license") or {}).get("name")
print("  [%s] kaggle license field    %s" % (check(lic == "Attribution 4.0 International (CC BY 4.0)"), lic))
body_cc = bool(re.search(r"CC-?BY", md, re.I))
print("  [%s] body license text       states CC-BY: %s   (Rule 2.5.a requires CC-BY-4.0)" % (check(body_cc), body_cc))

nbs = [l for l in links if l.get("entityType") == "kernels"]
pub = [l for l in nbs if (l.get("resource") or {}).get("isPrivateNullable") is False]
print("  [%s] public notebooks        %d of %d attached are public" % (check(bool(pub)), len(pub), len(nbs)))

repo = [l for l in links if "github.com" in (l.get("url") or "")]
print("  [%s] repository project link %s" % (check(bool(repo)), repo[0].get("url") if repo else "MISSING - Rule 2.8.b requires a code repository link"))
print("  [%s] repository in body      %s" % (check(bool(re.search(r"github\.com", md, re.I))), bool(re.search(r"github\.com", md, re.I))))

print()
print("  [INFO] Rule 2.2.a: a hackathon team may submit ONE submission only.")
print("         Edit this writeup in place. Never create a second one.")

url = "https://www.kaggle.com" + (w.get("url") or "")
code = subprocess.run(["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", url],
                      capture_output=True, text=True).stdout.strip()
print()
print("  [INFO] anonymous GET %s" % url)
print("         -> HTTP %s   (404 means the public cannot read it)" % code)

print()
print("  Attached project links:")
for l in links:
    r = l.get("resource") or {}
    print("    - %-11s private=%-5s upvotes=%s  %r" % (
        l.get("entityType"), r.get("isPrivateNullable"), r.get("upvoteCountNullable"), l.get("title")))
    print("      %s" % (l.get("url"),))
PY
}

# --- dispatch ----------------------------------------------------------------

case "${1:-}" in
  list)  cmd_list ;;
  get)   shift; cmd_get "$@" ;;
  dump)  shift; cmd_dump "$@" ;;
  diff)  shift; cmd_diff "$@" ;;
  check) shift; cmd_check "$@" ;;
  -h|--help|help|"") usage 0 ;;
  *) die "unknown subcommand: $1 (try --help)" ;;
esac
