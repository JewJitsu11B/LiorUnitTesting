# Agent N: dynamics by direct iteration

## VERDICT
- Table modes (6 rows) and caption (24000 updates: 0.031570, 2.652e-5): HOLDS. My mpmath roots match every printed digit; my iteration gives the same with one-way past and with zero past.
- Lemma modes (4)/(1): HOLDS. Exact root as history continues as z0^n (relative deviation 2.5e-13 over 5000 updates at alpha=0.999; 1.4e-10 at alpha=0.99, ghat=0.8).
- Prop orientation (1): HOLDS (deviation 1e-11, counter amplitude 1e-14). (3): HOLDS (|A|=|B|).
- Prop orientation (2): HOLDS WITH CONDITIONS. The printed 5.6e-5 ... 1.8e-8 reproduce only with the script's fit (undamped exponentials, ratio counter/co). Fitting the true damped modes gives 2.4e-5, 3.0e-7, 6.0e-8, 1.9e-8, 7.5e-9. Small and vanishing either way.
- Prop orientation (4), "70 to 97 percent of the norm": FAILS as worded. Counter-rotating share of the norm is 40 to 49 percent, the same as a zero past. The ratio counter/co is 0.68 to 0.97, which is evidently the quoted figure.
- Table packet (all five rows): HOLDS WITH CONDITIONS (see Problems 1). Norm loss 0.9 to 1.5 percent at eps=1e-4: HOLDS for the norm (0.89, 1.11, 1.52 percent); norm squared loses 1.8 to 3.0 percent.
- Table closure / Prop born: HOLDS. Script numbers match the table; my 8-seed runs give the same scaling.
- Table readout (nine rows): HOLDS, except row 3 prints 4.0e-16 here against 3.7e-16 (rounding level).
- Tables mass, diffusion, feedback, normalizable: script output matches every printed digit.

## WHAT RAN
- `python N_own.py modes|orient|packet|closure|readout` (own code, written first).
- `python N_work\cal_edge_transducer.py` (unchanged copy, 11.8 s).
- `python N_followup.py` (my iteration with the script's metric definitions): reproduces Table packet exactly: 6.962e-2, 2.631e-2, 4.827e-2, 2.569e-2, 1.459e-2; floors 1.142e-2, 1.238e-2, 4.813e-2, 2.483e-2, 1.259e-2.

## PROBLEMS FOUND
1. Table packet error is not "after aligning the global phase" only: the script also rescales the CAL state to unit norm. Without rescaling the distances are 0.619, 0.104, 0.049, 0.028, 0.021. At eps=1e-2 the packet has lost 62 percent of its norm (10 percent at 1e-3); the paper does not say so. The floor column uses the exact modes with decay removed, also unstated.
2. The script contains no truncated-history run. The 70 to 97 percent claim has no code behind it.
3. Table feedback thresholds b>0.095 and b>0.501 are computed with flat weights (W = 1/(e^lambda - 1)); the simulated power-law weights at alpha=1.01 give b>0.097 and b>0.513. The paper does not say flat.
4. Table closure: script uses <r^2>=0.12 (unstated) and a second packet exp(-(x-3)^2/3 - 1.5ix), displaced and of different width, not just "centred at k0=-1.5". One seed only; my readout drift ranges over a factor 10 across seeds (2e-4 to 2.6e-3 at eps=1e-4).
5. Script's closing line in section D says the norm drift is the uniform (pi/4)(1-alpha) part; its own columns differ by 2 to 4 times at eps=1e-4. The paper's caption (uniform plus diffusive) is right: my prediction 3.02e-2 against measured 3.02e-2 (norm squared, <r^2>=0.06).
6. "Arbitrary-precision agreement to 3e-16" is not in the script. Against mpmath I get up to 5e-15 relative.
7. Minor: growth at alpha=1.001 and 1.01 (no brake) comes from 3000 updates, not 6000; Table readout row 6 uses eta = e1; normalizable kernels are truncated at 400 lags and renormalized.

## FILES
N_own.py, N_followup.py, N_out_*.txt, N_author_output.txt, N_work\cal_edge_transducer.py (copy), results\N.md
