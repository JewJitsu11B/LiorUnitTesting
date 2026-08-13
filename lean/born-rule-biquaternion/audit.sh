#!/usr/bin/env bash
# No-cheating gate: scan all BornRule source modules for forbidden proof shortcuts.
# A clean pass means no sorry/admit/native_decide, and no `axiom` declarations outside
# an explicitly-allowed cited registry (none expected in the Born core).
# Usage:  bash audit.sh
cd "$(dirname "$0")"
status=0
files=$(ls BornRule/*.lean 2>/dev/null)
echo "=== scanning: $(echo "$files" | tr '\n' ' ') ==="
for tok in '\bsorry\b' '\badmit\b' '\bnative_decide\b'; do
  hits=$(grep -nE "$tok" $files 2>/dev/null | grep -v -- '-- ' )
  if [ -n "$hits" ]; then echo "FORBIDDEN ($tok):"; echo "$hits"; status=1; fi
done
# `axiom` declarations (allow only lines explicitly tagged `-- CITED-AXIOM`)
axhits=$(grep -nE '^\s*axiom\b' $files 2>/dev/null | grep -v 'CITED-AXIOM')
if [ -n "$axhits" ]; then echo "UNCITED AXIOM:"; echo "$axhits"; status=1; fi
if [ "$status" = 0 ]; then echo "NO-CHEATING GATE: clean (no sorry/admit/native_decide/uncited-axiom)"; fi
exit $status
