"""
Single source of truth for the oxide vaporisation anchors.

These constants were previously copy-pasted across vapor_equilibrium.py,
fig_strength.py, fig_activity.py and fig_designrule.py. They are consolidated
here so a re-anchor against properly sourced thermochemistry is a one-file edit
rather than four, with no chance of leaving a figure on stale numbers.

Every number carries its own provenance string and a source_type tag. The three
refractory-oxide anchors are currently VENDOR DATASHEET values (Lesker 10^-4
Torr temperatures), not evaluated thermochemistry. That is the honest current
state and the tag is there so it stays obvious until the values are replaced.
"""
import math
from dataclasses import dataclass
from typing import Literal

R = 8.314462618

# vendor_datasheet is weaker than evaluated_compilation, which is weaker than a
# primary measurement. Tagging lets a consumer see what it is standing on.
SourceType = Literal["vendor_datasheet", "evaluated_compilation", "primary_literature"]


@dataclass(frozen=True)
class Anchor:
    """Clausius-Clapeyron anchor for one oxide, plus where the numbers came from."""
    p_ref: float      # Pa
    T_ref: float      # K
    dH: float         # J/mol
    provenance: str
    source_type: SourceType


ANCHORS = {
    "ZrO2": Anchor(
        p_ref=1.33e-2, T_ref=2473.0, dH=640e3,
        provenance="10^-4 Torr at 2200 C, Lesker / Ackermann-Rauh",
        source_type="vendor_datasheet",
    ),
    "HfO2": Anchor(
        p_ref=1.33e-2, T_ref=2773.0, dH=690e3,
        provenance="10^-4 Torr at 2500 C, Lesker",
        source_type="vendor_datasheet",
    ),
    "Ta2O5": Anchor(
        p_ref=1.33e-2, T_ref=2193.0, dH=500e3,
        provenance="10^-4 Torr at 1920 C, Lesker; congruent TaO2(g)",
        source_type="vendor_datasheet",
    ),
}

# Niobium is not a Clausius-Clapeyron anchor. It gets a measured expression per
# environment, so it carries provenance without p_ref/T_ref/dH.
NB_PROVENANCE = ("Kamegashira 1981 (throat, NbO2) + Matsui-Naito 1983 (air, Nb2O5)")
NB_SOURCE_TYPE: SourceType = "primary_literature"

METAL_OXIDE = {"Hf": "HfO2", "Zr": "ZrO2", "Ta": "Ta2O5", "Nb": "Nb2O5"}

# Two published routes to the same reducing-throat pressure. They disagree by
# about 6 percent at the RS-25 throat and by far more when extrapolated well
# above the measurement range (~59 percent at 3873 K), so the choice is explicit
# at the call site rather than assumed.
NbModel = Literal["third_law", "measured_fit"]


def p_nb2o5(T, oxidising, dscale=1.0, model: NbModel = "third_law"):
    """NbO2(g) pressure, Pa. Reducing throat and oxidising air are different phases."""
    if oxidising:
        # NbO2(g) over Nb2O5(l), Matsui & Naito 1983, log p/Pa = -0.889e4/T + 3.700
        # measured 1800-1988 K. NbO and O2 were below detection, so NbO2(g) is
        # the metal-bearing pressure.
        return 10.0 ** (-0.889e4 * dscale / T + 3.700)
    if model == "measured_fit":
        # Direct fit to Kamegashira's measured pressures, x1.4 for the NbO(g) channel
        return 1.4 * 10.0 ** (-2.393e4 * dscale / T + 10.90)
    # THIRD LAW from the JANAF free energy functions (Kamegashira Table 5
    # dHvap,298 = 546.5 kJ/mol; Table 8 d_gef fit), x1.4 for the measured NbO(g)
    # channel. Agrees with the measured-pressure fit to within 6 percent at the throat.
    dgef = -0.012337 * T + 179.710           # J/mol K, gef(g) - gef(l)
    p_NbO2 = math.exp(-546.5e3 * dscale / (R * T) + dgef / R) * 1.0e5
    return 1.4 * p_NbO2


_OXIDE_TO_METAL = {"HfO2": "Hf", "ZrO2": "Zr", "Ta2O5": "Ta", "Nb2O5": "Nb"}


def p_oxide(oxide, T, oxidising=False, dscale=1.0, nb_model: NbModel = "third_law"):
    """Equilibrium volatile metal pressure over the oxide scale, Pa.

    Delegates to sourced_volatility. The vendor anchors above are kept only as the record
    of what the numbers used to be; nothing computes from them any more. Keeping this shim
    means the figure scripts that already import it pick up the sourced thermochemistry
    without each needing a separate edit.

    dscale is retained for signature compatibility and converted to the equivalent
    additive shift. A fractional scaling was meaningful against a 640 kJ/mol vaporisation
    enthalpy; it is not against a 1400 kJ/mol reaction enthalpy.
    """
    from sourced_volatility import P

    dH = 0.0 if dscale == 1.0 else (dscale - 1.0) * 650.0
    return P[_OXIDE_TO_METAL[oxide]](T, dH)


def p_oxide_for_metal(metal, T, oxidising=False, dscale=1.0, nb_model: NbModel = "third_law"):
    """Same as p_oxide but keyed on the metal symbol."""
    return p_oxide(METAL_OXIDE[metal], T, oxidising, dscale, nb_model)


def provenance_rows():
    """(oxide, provenance, source_type) for every anchor, Nb included."""
    rows = [(ox, a.provenance, a.source_type) for ox, a in ANCHORS.items()]
    rows.append(("Nb2O5", NB_PROVENANCE, NB_SOURCE_TYPE))
    return rows
