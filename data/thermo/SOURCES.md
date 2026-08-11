# Bundled thermochemical inputs

NIST-JANAF tables the volatility scripts read, bundled so the figures reproduce with no
external path.

- `_gases/raw/O-001.txt`, `_gases/raw/O-029.txt` — O(g) and O2(g). US Government work,
  public domain.

HfO(g) thermal functions (Bauschlicher, Kowalski & Jacobson, J. Chem. Phys. 157 (2022)
154302, doi:10.1063/5.0120504) are transcribed inline as `_HFO_G_GEF` in
`code/sourced_volatility.py` and `code/hafnia_coupled.py`, cited at the constant.
