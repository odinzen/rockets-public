# HEC-UHTC oxidation screening — code and figures

Screening code and figures for the study

> **Oxidation, not phase stability, sets high-entropy carbide service life:
> the propellant sets the required hafnium fraction**
> Michael E. Bustamante, Gabriel Bustamante, Kristina Lilova (2026)

For high-entropy carbide ultra-high-temperature ceramics on rocket surfaces, service
life is set by oxidation of the surface, not by bulk single-phase stability. A staged,
inexpensive screening pathway — a closed-form stability screen, equilibrium
thermochemistry for the oxide scale, and a dual-regime recession model anchored to two
real engine throats (the RS-25 and the F-1) — turns a qualitative design rule into an
absolute recession rate. The decisive result: the hafnium fraction a throat surface needs
to stay inside a practical service limit is set by the propellant, about 0.50 on an
oxygen–kerosene engine and 0.91 on the hotter oxygen–hydrogen engine.

High-performance computing touches only the ten DFT metal-pair mixing enthalpies (computed
on the Sol supercomputer at Arizona State University); everything downstream runs on
tabulated thermochemistry and numpy.

## Layout

```
code/       stability screen, volatility, transport, recession, and figure scripts
data/       the ten DFT pair enthalpies computed on ASU Sol, with method provenance
figures/    figures 1–8 (PDF + PNG)
graphical_abstract.{png,pdf}
```

`code/sourced_volatility.py` is the single source of truth for the vaporization
thermochemistry; each value traces to primary measurement or evaluated compilation in its
module docstring, and `p_hfzr` is the coupled shared-oxygen mixed-scale solve.

`data/dft_pair_enthalpies.csv` holds the ten equiatomic pair mixing enthalpies computed
on the Sol supercomputer at Arizona State University (the only first-principles inputs);
they match the `DFT_DHMIX` constants in `code/stability_screen.py`.

## Running

Requires Python with `numpy` and `matplotlib`; `zentropy_lightpath.py` additionally needs
`pyzentropy`. The stability screen and the design-map scripts run standalone:

```
cd code
python stability_screen.py        # spinodals from the ten DFT pair enthalpies
python fig_design_threshold.py    # the hafnium-requirement design map (figs 4, 5)
```

The scripts that evaluate volatile pressures read tabulated Gibbs-energy-function tables
(NIST-JANAF; Barin, *Thermochemical Data of Pure Substances*, 1995). Those tables are from
the public sources cited in each script's header and are **not redistributed here** (Barin
is under copyright). The ten DFT pair enthalpies enter the code as constants; nothing here
re-runs DFT.

## Citing

If you use this code, please cite the manuscript above. The thermodynamic data are from
the cited public sources.

## License and attribution

Copyright 2026 Odinzen LLC. Licensed under the Apache License, Version 2.0 — see
[LICENSE](LICENSE) and [NOTICE](NOTICE). Work by Odinzen LLC (Houston, TX) with the
Navrotsky Eyring Center for Materials of the Universe, School of Molecular Sciences,
Arizona State University.
