# LiorUnitTesting

Unit-testing suite for the CAL framework.

Contents:
- `cal/` - the package under test: biquaternion arithmetic, the bilocal variable-order
  entropy/detector functionals, the causal tensor, the propagator, and the emergent metric.
- `test_biquaternion.py`, `test_entropy.py`, `test_metric.py`, `test_associator.py`,
  `test_dissolution.py` - the core algebra / entropy / metric / transport test groups.
- `test_entropy_nist_vs_renyi.py`, `test_dual_log_entropy_comparison.py`,
  `test_dual_entropy_channels.py` - the single-log entropy vs dual-log detector comparisons
  (concavity, sign structure, lens/content channel decomposition).
- `test_effective_temperature.py` - the ET1-ET5 superstatistical-form suite (q = 1 + 1/alpha_K).
- `test_superstatistics_bornmax.py` - the Gamma-temperature / q-exponential BornMax identities.
- `test_bornmax_architecture.py` - end-to-end: entropy in the energy slot, detector in the
  temperature slot.

## Running

```
python -m pytest -q
```

`conftest.py` puts this folder on `sys.path` so `import cal...` resolves to the local `cal/`.
Requires `numpy`, `torch`, `scipy`.
