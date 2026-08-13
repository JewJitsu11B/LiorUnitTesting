#!/usr/bin/env bash
# No-cheating gate: scan ChiralParity source modules for forbidden shortcuts.
cd "$(dirname "$0")"
status=0
files=$(ls ChiralParity/*.lean 2>/dev/null)
echo "=== scanning: $(echo "$files" | tr "\n" " ") ==="
for tok in "\bsorry\b" "\badmit\b" "\bnative_decide\b"; do
  hits=$(grep -nE "$tok" $files 2>/dev/null | grep -v "CITED-AXIOM" | grep -vE "^[^:]*:[0-9]+:\s*--")
  if [ -n "$hits" ]; then echo "FORBIDDEN ($tok):"; echo "$hits"; status=1; fi
done
axhits=$(grep -nE "^\s*axiom\b" $files 2>/dev/null | grep -v "CITED-AXIOM")
if [ -n "$axhits" ]; then echo "UNCITED AXIOM:"; echo "$axhits"; status=1; fi
[ "$status" = 0 ] && echo "NO-CHEATING GATE: clean"
exit $status
