#@title 2. Run Sinter Burden Optimizer v31.9

# ============================================================================
# SINTER BURDEN OPTIMIZER v31.9
# - Preserves v30 core optimizer
# - Strict IOL/BF total-burden mandates
# - Mill Scale 5–15% when available; 0% when unavailable
# - All iron-bearing materials are ordinary candidates with availability toggles
# - Dynamic iron-ore ranking by current Fe%
# - Table-based manual burden control with kg/t or % change
# ============================================================================

import pandas as pd
import pulp
import numpy as np
from io import BytesIO
import re

print("✅ Libraries loaded. Ready.")

# ============================================================================
# 1. MATERIAL RANKING & COMPENSATION RULES
# ============================================================================

def get_iron_ore_ranking(df):
    """Rank currently available iron-bearing materials by current Fe % (highest first)."""
    ores = [
        m for m in df.index
        if df.loc[m, "Group"] == "Iron_ore"
        and df.loc[m, "Available_Tonnes"] > 0
    ]
    return sorted(ores, key=lambda m: (-float(df.loc[m, "Fe"]), str(m)))
FLUX_RANK = {"LIMESTONE": 1, "DOLOMITE": 2, "QUICKLIME": 3}

MILL_SCALE_MIN_BURDEN_PCT = 0.05
MILL_SCALE_MAX_BURDEN_PCT = 0.15

# Production continuity: search for a quality-compliant blend at higher cost
# when cheaper combinations are unavailable. Only use recovery diagnostics if
# no compliant production blend can be found.
PRODUCTION_CONTINUITY_MODE = True

# --- IRON ORE CAPS ---
IRON_ORE_MAX_PCT_BASE = {"Lloyds_HG": 0.25, "MILL_SCALE": 0.29, "SIOM_MG": 0.29, "KIOM_MG": 0.29, "DIOM_LG": 0.29}
IRON_ORE_MAX_PCT_RELAXED = {"Lloyds_HG": 0.35, "MILL_SCALE": 0.35, "SIOM_MG": 0.40, "KIOM_MG": 0.40, "DIOM_LG": 0.40}
IRON_ORE_MAX_PCT_CRISIS = {"Lloyds_HG": 0.95, "MILL_SCALE": 0.95, "SIOM_MG": 0.95, "KIOM_MG": 0.95, "DIOM_LG": 0.95}
IRON_ORE_MIN_PCT = {"Lloyds_HG": 0.03, "MILL_SCALE": 0.03, "SIOM_MG": 0.03, "KIOM_MG": 0.03, "DIOM_LG": 0.03}
MAX_IRON_ORE_PORTION = 0.80
MAX_IRON_ORE_PORTION_CRISIS = 0.95

# --- FLUX CAPS ---
FLUX_MAX_PCT_BASE = {"LIMESTONE": 0.60, "DOLOMITE": 0.45, "QUICKLIME": 0.60}
FLUX_MAX_PCT_RELAXED = {"LIMESTONE": 0.75, "DOLOMITE": 0.55, "QUICKLIME": 0.80}
FLUX_MAX_PCT_CRISIS = {"LIMESTONE": 0.95, "DOLOMITE": 0.95, "QUICKLIME": 0.95}
FLUX_MIN_PCT = {"LIMESTONE": 0.05, "DOLOMITE": 0.05, "QUICKLIME": 0.02}
FLUX_MIN_PCT_QUALITY_RELAXED = {"LIMESTONE": 0.0, "DOLOMITE": 0.0, "QUICKLIME": 0.0}
MAX_FLUX_PORTION = 0.25
MAX_FLUX_PORTION_CRISIS = 0.40

# --- SiO2/CaO CEILING ESCALATION UNDER IRON ORE SHORTAGE ---
SIO2_MAX_SHORTAGE = 6.2

# --- FE TARGET: TIGHT CONTROL (NEVER RELAXED) ---
FE_TARGET = 54.0
FE_TOLERANCE = 0.3
FE_LOWER = FE_TARGET - FE_TOLERANCE
FE_UPPER = FE_TARGET + FE_TOLERANCE
FE_CENTER_WEIGHT = 2.0

# Dashboard quality targets — single source of truth for app.py.
# These match the v31.9 plant limits used by the optimizer.
TARGETS = {
    "SiO2_max": 5.80,
    "Al2O3_max": 4.50,
    "Al2O3_SiO2_max": 0.98,
    "Basicity_min": 1.90,
    "Basicity_max": 2.00,
    "MgO_min": 2.20,
    "MgO_max": 2.40,
    "CaO_min": 10.50,
    "CaO_max": 11.50,
}


# --- QUALITY DEVIATION WEIGHTS (diagnostic only) ---
DEVIATION_WEIGHTS = {
    "Fe": 5.0, "Basicity": 6.0, "CaO": 5.0, "MgO": 4.0,
    "Al2O3": 3.0, "SiO2": 2.0, "Al2O3_SiO2_ratio": 2.0,
}

PIN_TOLERANCE = 1e-3
FLUX_BASELINE_INCREASE_CAP = 0.05

ADJUSTMENT_RANGES = {
    "Iron_ore": 0.15, "Flux": 0.10, "Recycle": 0.00, "Fuel": 0.10,
    "IOL_Fines_Mandate": 0.00, "BF_Returns_Mandate": 0.00,
}

NUM_ALT_ORE_SLOTS = 0

# ============================================================================
# IOL FINES / BF RETURNS MANDATE PARAMETERS (editable via UI)
# ============================================================================
IOL_FINES_NOMINAL_PCT = 0.08
BF_RETURNS_NOMINAL_PCT = 0.17

# STRICT PLANT MANDATES — no fallback / relaxed bands.
# The optimizer must use exactly 80 kg/t IOL Fines and 170 kg/t BF Returns.
# If either mandate cannot be met because of availability/other constraints,
# the model must report infeasibility rather than silently relaxing the mandate.
IOL_FINES_FALLBACK_MIN = IOL_FINES_NOMINAL_PCT
IOL_FINES_FALLBACK_MAX = IOL_FINES_NOMINAL_PCT
BF_RETURNS_FALLBACK_MIN = BF_RETURNS_NOMINAL_PCT
BF_RETURNS_FALLBACK_MAX = BF_RETURNS_NOMINAL_PCT

PIN_BAND = 0.0

# ============================================================================
# DEFAULT COKE OPTIMISATION PARAMETERS (editable in UI)
# ============================================================================
DEFAULT_OM_COST_RS_T = 1500.0
DEFAULT_COKE_CV_KCAL_KG = 6800.0
DEFAULT_COKE_FC_PCT = 71.35
DEFAULT_HEAT_LATENT_MOISTURE = 540.0
DEFAULT_HEAT_CALCINATION_PER_LOI_KG = 420.0
DEFAULT_HEAT_MELTING_PER_KG_SINTER = 60.0
DEFAULT_HEAT_LOSS_FRACTION = 0.12
DEFAULT_FIRING_RATIO_MAX = 1.10
# Interim model setting: the upper firing-ratio cap is diagnostic only until plant heat-balance coefficients are calibrated.
ENFORCE_FIRING_RATIO_MAX = False

# --- PROVISIONAL COKE / FeO OPERATING WINDOW ---
# Based on published sinter-plant practice; ALL VALUES ARE EDITABLE.
# These are deliberately provisional until plant historical data is used for calibration.
DEFAULT_COKE_MIN_KG_T = 55.0
DEFAULT_COKE_MAX_KG_T = 85.0
DEFAULT_FEO_MIN_PCT = 8.5
DEFAULT_FEO_TARGET_PCT = 9.2
DEFAULT_FEO_MAX_PCT = 10.0
DEFAULT_FEO_REFERENCE_THERMAL_SURPLUS_KCAL = 189180.0
DEFAULT_FEO_REFERENCE_PCT = 8.6
DEFAULT_FEO_THERMAL_SLOPE_PCT_PER_10K_KCAL = 0.35
DEFAULT_REFERENCE_COKE_CV_KCAL_KG = 6800.0
DEFAULT_REFERENCE_COKE_FC_PCT = 71.35

def sanitize_material_name(raw_name):
    name = raw_name.strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^A-Za-z0-9_]", "", name)
    return name

# ============================================================================
# 2. DEFAULT CHEMISTRY + BOUNDS (COKE Tech_Min=0, Tech_Max=999)
# ============================================================================
def get_default_chemistry():
    data = {
        "Material": [
            "MILL_SCALE", "Lloyds_HG", "DIOM_LG", "SIOM_MG", "KIOM_MG",
            "Solid_Waste", "IOL_Fines", "FLUE_DUST", "BF_Returns",
            "DOLOMITE", "LIMESTONE", "QUICKLIME",
            "COKE_BREEZE"
        ],
        "Group": [
            "Iron_ore", "Iron_ore", "Iron_ore", "Iron_ore", "Iron_ore",
            "Recycle", "IOL_Fines_Mandate", "Recycle", "BF_Returns_Mandate",
            "Flux", "Flux", "Flux",
            "Fuel"
        ],
        "Fe":    [68.34, 63.52, 57.17, 59.34, 58.41, 50.0, 60.00, 47.02, 52.5,   0.54, 0.88, 0.01, 0],
        "SiO2":  [2.00,  3.86,  12.39, 6.92,  5.75,  6.00, 5.00,  7.07,  5.62,   4.72, 4.48, 2.50, 2.8],
        "Al2O3": [2.72,  2.27,  2.93,  3.72,  5.48,  4.50, 3.00,  4.50,  3.20,   0.95, 1.19, 0.61, 0],
        "CaO":   [0,     0.022, 0.058, 0.256, 0.157, 1.122,8.79,  1.10,  10.74,  30.02,48.71,89.00,0],
        "MgO":   [0,     0.034, 0.114, 0.331, 0.018, 0.06, 1.52,  0.29,  2.30,   18.75,2.59, 1.57, 0],
        "LOI":   [2.50,  2.29,  4.00,  3.45,  4.62,  3.00, 3.00,  15.00, 3.00,   42.00,40.00,5.00, 70.00],
        "Moisture_Pct": [
            6.0, 5.0, 6.0, 6.0, 6.0, 1.1, 4.13, 9.4, 0.0, 2.0, 2.0, 0.0, 11.27
        ],
        "Tech_Min": [0, 0, 0, 0, 0, 30, 0, 25, 0, 30, 0, 40, 0],
        "Tech_Max": [220, 200, 200, 200, 300, 30, 999, 25, 999, 200, 250, 65, 999],
        "Available_Tonnes": [2000, 10000, 6000, 8000, 5000, 5000, 5000, 3000, 5000, 10000, 15000, 5000, 9999],
        "Price_Rs_t": [7800, 7820, 4600, 4600, 4900, 1000, 5577, 500, 0, 1340, 1355, 9200, 15022],
    }
    df = pd.DataFrame(data).set_index("Material")
    for mat in df[df["Group"] == "Recycle"].index:
        fixed_rate = df.loc[mat, "Tech_Min"]
        df.loc[mat, "Tech_Min"] = fixed_rate
        df.loc[mat, "Tech_Max"] = fixed_rate
    return df

# ============================================================================
# 3. MANUAL RAW-MATERIAL INPUT TABLE
# ============================================================================
# The optimizer can now be run without an Excel master file.
# ALL raw-material inputs are entered/edited manually in the table:
# Material, Group, chemistry, moisture, technical limits, availability and price.
#
# SKME, THAKUR, RBSNN and SMIORE are deliberately included as normal
# Iron_ore rows. Their names and chemistry are editable by the user.

MANUAL_MATERIALS = [
    # name, group, Fe, SiO2, Al2O3, CaO, MgO, LOI, moisture, tech_min, tech_max, availability_t, price
    ("MILL_SCALE", "Iron_ore", 68.34, 2.00, 2.72, 0.00, 0.00, 2.50, 6.00, 0, 220, 2000, 7800),
    ("Lloyds_HG", "Iron_ore", 63.52, 3.86, 2.27, 0.022, 0.034, 2.29, 5.00, 0, 200, 10000, 7820),
    ("DIOM_LG", "Iron_ore", 57.17, 12.39, 2.93, 0.058, 0.114, 4.00, 6.00, 0, 200, 6000, 4600),
    ("SIOM_MG", "Iron_ore", 59.34, 6.92, 3.72, 0.256, 0.331, 3.45, 6.00, 0, 200, 8000, 4600),
    ("KIOM_MG", "Iron_ore", 58.41, 5.75, 5.48, 0.157, 0.018, 4.62, 6.00, 0, 300, 5000, 4900),

    # NEW iron-bearing materials — user enters actual chemistry/stock/price.
    ("SKME", "Iron_ore", 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0, 300, 0, 0),
    ("THAKUR", "Iron_ore", 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0, 300, 0, 0),
    ("RBSNN", "Iron_ore", 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0, 300, 0, 0),
    ("SMIORE", "Iron_ore", 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0, 300, 0, 0),

    ("Solid_Waste", "Recycle", 50.00, 6.00, 4.50, 1.122, 0.06, 3.00, 1.10, 30, 30, 5000, 1000),
    ("IOL_Fines", "IOL_Fines_Mandate", 60.00, 5.00, 3.00, 8.79, 1.52, 3.00, 4.13, 0, 999, 5000, 5577),
    ("FLUE_DUST", "Recycle", 47.02, 7.07, 4.50, 1.10, 0.29, 15.00, 9.40, 25, 25, 3000, 500),
    ("BF_Returns", "BF_Returns_Mandate", 52.50, 5.62, 3.20, 10.74, 2.30, 3.00, 0.00, 0, 999, 5000, 0),

    ("DOLOMITE", "Flux", 0.54, 4.72, 0.95, 30.02, 18.75, 42.00, 2.00, 30, 200, 10000, 1340),
    ("LIMESTONE", "Flux", 0.88, 4.48, 1.19, 48.71, 2.59, 40.00, 2.00, 0, 250, 15000, 1355),
    ("QUICKLIME", "Flux", 0.01, 2.50, 0.61, 89.00, 1.57, 5.00, 0.00, 40, 65, 5000, 9200),
    ("COKE_BREEZE", "Fuel", 0.00, 2.80, 0.00, 0.00, 0.00, 70.00, 11.27, 0, 999, 9999, 15022),
]

MANUAL_COLUMNS = [
    "Material", "Group", "Fe", "SiO2", "Al2O3", "CaO", "MgO", "LOI",
    "Moisture_Pct", "Tech_Min", "Tech_Max", "Available_Tonnes", "Price_Rs_t"
]

def get_manual_input_df():
    return pd.DataFrame(MANUAL_MATERIALS, columns=MANUAL_COLUMNS).set_index("Material")

def build_raw_material_input_table(initial_df=None, on_start_optimizer=None):
    """Editable raw-material master table. Optimizer starts only after user submits inputs."""
    df0 = (initial_df.copy() if initial_df is not None else get_manual_input_df()).copy()

    for mat in ["SKME", "THAKUR", "RBSNN", "SMIORE"]:
        if mat not in df0.index:
            df0.loc[mat] = {
                "Group": "Iron_ore", "Fe": 0, "SiO2": 0, "Al2O3": 0, "CaO": 0,
                "MgO": 0, "LOI": 0, "Moisture_Pct": 0, "Tech_Min": 0,
                "Tech_Max": 300, "Available_Tonnes": 0, "Price_Rs_t": 0
            }

    widgets_by_mat = {}
    output = widgets.Output()
    state = {"df": df0.copy()}

    headers = [
        ("Material / Dealer", 20), ("Group", 16), ("Fe %", 8), ("SiO₂ %", 8),
        ("Al₂O₃ %", 9), ("CaO %", 8), ("MgO %", 8), ("LOI %", 8),
        ("Moisture %", 9), ("Tech Min", 9), ("Tech Max", 9), ("Stock t", 10),
        ("Price Rs/t", 11)
    ]
    ncols = len(headers)
    grid = widgets.GridspecLayout(len(df0) + 1, ncols, width="100%")

    for j, (txt, _) in enumerate(headers):
        grid[0, j] = widgets.HTML(f"<b>{txt}</b>")

    def num(value, step=0.1):
        return widgets.FloatText(value=float(value), step=step,
                                 layout=widgets.Layout(width="100%"))

    for i, mat in enumerate(df0.index, start=1):
        rw = {}
        rw["Material"] = widgets.Text(value=str(mat), layout=widgets.Layout(width="100%"))
        rw["Group"] = widgets.Dropdown(
            options=["Iron_ore", "Flux", "Recycle", "IOL_Fines_Mandate",
                     "BF_Returns_Mandate", "Fuel"],
            value=str(df0.loc[mat, "Group"]), layout=widgets.Layout(width="100%")
        )
        rw["Fe"] = num(df0.loc[mat, "Fe"])
        rw["SiO2"] = num(df0.loc[mat, "SiO2"])
        rw["Al2O3"] = num(df0.loc[mat, "Al2O3"])
        rw["CaO"] = num(df0.loc[mat, "CaO"])
        rw["MgO"] = num(df0.loc[mat, "MgO"])
        rw["LOI"] = num(df0.loc[mat, "LOI"])
        rw["Moisture_Pct"] = num(df0.loc[mat, "Moisture_Pct"])
        rw["Tech_Min"] = num(df0.loc[mat, "Tech_Min"], 1)
        rw["Tech_Max"] = num(df0.loc[mat, "Tech_Max"], 1)
        rw["Available_Tonnes"] = widgets.FloatText(
            value=float(df0.loc[mat, "Available_Tonnes"]), step=100,
            layout=widgets.Layout(width="100%")
        )
        rw["Price_Rs_t"] = widgets.FloatText(
            value=float(df0.loc[mat, "Price_Rs_t"]), step=50,
            layout=widgets.Layout(width="100%")
        )
        widgets_by_mat[mat] = rw

        for j, c in enumerate(MANUAL_COLUMNS):
            grid[i, j] = rw[c]

    use_btn = widgets.Button(
        description="▶ USE INPUTS & START OPTIMIZER",
        button_style="success",
        layout=widgets.Layout(width="250px", height="36px")
    )
    save_btn = widgets.Button(
        description="💾 SAVE INPUT TABLE",
        button_style="info",
        layout=widgets.Layout(width="180px", height="36px")
    )

    def collect(_=None, start_optimizer=False):
        rows, used_names = [], set()
        for old_mat, rw in widgets_by_mat.items():
            name = sanitize_material_name(rw["Material"].value)
            if not name or name in used_names:
                continue
            used_names.add(name)
            rec = {"Material": name}
            for c in MANUAL_COLUMNS[1:]:
                rec[c] = float(rw[c].value) if c != "Group" else rw[c].value
            rows.append(rec)

        new_df = pd.DataFrame(rows).set_index("Material")

        # Basic validation before the optimizer starts.
        errors = []
        for mat in new_df.index:
            if new_df.loc[mat, "Tech_Max"] < new_df.loc[mat, "Tech_Min"]:
                errors.append(f"{mat}: Tech Max < Tech Min")
            if new_df.loc[mat, "Available_Tonnes"] < 0:
                errors.append(f"{mat}: negative stock")
            if new_df.loc[mat, "Price_Rs_t"] < 0:
                errors.append(f"{mat}: negative price")

        state["df"] = new_df
        with output:
            clear_output(wait=True)
            if errors:
                print("❌ INPUT ERRORS — fix these before running:")
                for e in errors:
                    print(" •", e)
                return
            print(f"✅ {len(new_df)} raw materials captured.")
            print("🔒 IOL Fines = 8% | BF Returns = 17% of total burden.")
            print("🔒 Mill Scale = 5–15% of total burden when available.")
            display(HTML(new_df.to_html(float_format="%.2f")))
        if start_optimizer and on_start_optimizer:
            on_start_optimizer(new_df)

    save_btn.on_click(lambda b: collect(b, False))
    use_btn.on_click(lambda b: collect(b, True))

    print("\n" + "=" * 140)
    print("📋 RAW MATERIAL INPUT TABLE — ENTER / EDIT CURRENT PLANT DATA")
    print("=" * 140)
    print("• SKME / THAKUR / RBSNN / SMIORE are included as normal Iron_ore rows.")
    print("• Enter their CURRENT lot chemistry, stock and price. Default stock is 0 until entered.")
    print("• Material / dealer names are editable.")
    print("• Ranking is recalculated from current Fe %.")
    print("• Stock = 0 means unavailable.")
    display(widgets.HBox([use_btn, save_btn]))
    display(grid)
    display(output)

    return state, widgets_by_mat

def get_manual_df_from_state(state):
    return state["df"].copy()

# ============================================================================
# 4. HELPERS (unchanged)
# ============================================================================
def get_mandate_shortfall_triggers(df):
    triggers = 0
    reasons = []
    for mat in ["IOL_Fines", "BF_Returns"]:
        if mat in df.index:
            if df.loc[mat, "Available_Tonnes"] <= 0 or df.loc[mat, "Tech_Max"] == 0:
                triggers += 1
                reasons.append(mat)
    return triggers, reasons

def get_iron_ore_tier(df, iron_ores, extra_missing=0, extra_reasons=None):
    extra_reasons = extra_reasons or []
    unavailable = [m for m in iron_ores if df.loc[m, "Available_Tonnes"] <= 0 or df.loc[m, "Tech_Max"] == 0]
    n = len(unavailable) + extra_missing
    display_list = unavailable + [f"{r}(mandate-shortfall)" for r in extra_reasons]

    # Start from the legacy caps, then assign sensible defaults to every
    # newly added iron-bearing material (SKME/THAKUR/RBSNN/SMIORE/future ores).
    base = {m: IRON_ORE_MAX_PCT_BASE.get(m, 0.29) for m in iron_ores}
    relaxed = {m: IRON_ORE_MAX_PCT_RELAXED.get(m, 0.40) for m in iron_ores}
    crisis = {m: IRON_ORE_MAX_PCT_CRISIS.get(m, 0.95) for m in iron_ores}

    if n == 0:
        return base, unavailable, "✅ All iron ores available - using base caps", "base"
    elif n == 1:
        return relaxed, unavailable, f"⚠️ {n} iron-ore shortfall: {', '.join(display_list)} — RELAXED caps", "relaxed"
    else:
        return crisis, unavailable, (
            f"🔥 {n} iron-ore shortfalls: {', '.join(display_list)} — CRISIS MODE: iron-ore caps loosened."
        ), "crisis"

def get_flux_tier(df, fluxes, extra_missing=0, extra_reasons=None):
    extra_reasons = extra_reasons or []
    unavailable = [m for m in fluxes if df.loc[m, "Available_Tonnes"] <= 0 or df.loc[m, "Tech_Max"] == 0]
    n = len(unavailable) + extra_missing
    display_list = unavailable + [f"{r}(mandate-shortfall)" for r in extra_reasons]
    if n == 0:
        return FLUX_MAX_PCT_BASE.copy(), unavailable, "✅ All fluxes available - using base flux caps", "base"
    elif n == 1:
        return FLUX_MAX_PCT_RELAXED.copy(), unavailable, f"⚠️ {n} flux-equivalent shortfall: {', '.join(display_list)} — RELAXED flux caps", "relaxed"
    else:
        return FLUX_MAX_PCT_CRISIS.copy(), unavailable, (
            f"🔥 {n} flux-equivalent shortfalls: {', '.join(display_list)} — CRISIS MODE: flux caps near-removed."
        ), "crisis"

def check_fuel_gate(df):
    fuels = [m for m in df.index if df.loc[m, "Group"] == "Fuel"]
    problems = []
    for mat in fuels:
        tech_min = df.loc[mat, "Tech_Min"]
        available = df.loc[mat, "Available_Tonnes"]
        tech_max = df.loc[mat, "Tech_Max"]
        if tech_min > 0 and (available <= 0 or tech_max <= 0):
            problems.append(f"{mat} (requires >= {tech_min} kg/t, but Available_Tonnes={available}, Tech_Max={tech_max})")
    if problems:
        return False, problems
    return True, []

def build_bounds(df, production_tonnes):
    bounds = {}
    for mat in df.index:
        tech_min = df.loc[mat, "Tech_Min"]
        tech_max = df.loc[mat, "Tech_Max"]
        available = df.loc[mat, "Available_Tonnes"]
        if available <= 0 or tech_max == 0:
            bounds[mat] = (0, 0)
        else:
            inv_cap = (available / production_tonnes) * 1000
            eff_max = min(tech_max, inv_cap)
            bounds[mat] = (tech_min, eff_max)
    return bounds

# ============================================================================
# 5. MANDATE CONSTRAINTS
# ============================================================================
def add_mandate_constraints(prob, x, df, OUT, mandate_mode="pinned",
                            iol_nominal=IOL_FINES_NOMINAL_PCT,
                            bf_nominal=BF_RETURNS_NOMINAL_PCT,
                            iol_fb_min=IOL_FINES_FALLBACK_MIN,
                            iol_fb_max=IOL_FINES_FALLBACK_MAX,
                            bf_fb_min=BF_RETURNS_FALLBACK_MIN,
                            bf_fb_max=BF_RETURNS_FALLBACK_MAX):
    # Total burden is the complete charged burden, INCLUDING coke.
    # IOL Fines and BF Returns are specified as percentages of this total.
    total_burden = pulp.lpSum(x[m] for m in df.index)

    for mat, nominal, fb_min, fb_max in [
        ("IOL_Fines", iol_nominal, iol_fb_min, iol_fb_max),
        ("BF_Returns", bf_nominal, bf_fb_min, bf_fb_max),
    ]:
        if mat not in x:
            continue

        # STRICT composition mandates:
        # IOL Fines = 8% of total burden, including coke
        # BF Returns = 17% of total burden, including coke
        #
        # This is deliberately NOT nominal * OUT. The total raw burden is
        # variable because LOI/moisture and the 1000 kg dry-product balance
        # make the as-fed burden larger than 1000 kg in many cases.
        prob += x[mat] == nominal * total_burden, f"{mat}_STRICT_BURDEN_PCT"

# ============================================================================
# 6. STRUCTURAL CONSTRAINTS (unchanged)
# ============================================================================
def add_structural_constraints(prob, x, df, bounds, iron_ores, fluxes, iron_ore_max_pct,
                                unavailable_iron, flux_max_pct, unavailable_flux, OUT,
                                baseline_flux_portion=None, iron_tier="base", flux_tier="base",
                                flux_min_pct_override=None, mandate_mode="pinned",
                                iol_nominal=IOL_FINES_NOMINAL_PCT,
                                bf_nominal=BF_RETURNS_NOMINAL_PCT,
                                iol_fb_min=IOL_FINES_FALLBACK_MIN,
                                iol_fb_max=IOL_FINES_FALLBACK_MAX,
                                bf_fb_min=BF_RETURNS_FALLBACK_MIN,
                                bf_fb_max=BF_RETURNS_FALLBACK_MAX):
    non_fuel = [m for m in x if df.loc[m, "Group"] != "Fuel"]
    mass = pulp.lpSum(x[m] * (1 - df.loc[m, "LOI"] / 100) for m in non_fuel)
    prob += mass >= OUT - 2, "Mass_Balance_Lower"
    prob += mass <= OUT + 2, "Mass_Balance_Upper"

    total_iron_ore = pulp.lpSum(x[m] for m in iron_ores)
    total_flux = pulp.lpSum(x[m] for m in fluxes)
    # Total burden includes all charged materials, including coke.
    total_burden = pulp.lpSum(x[m] for m in df.index)

    for mat in iron_ores:
        if mat in unavailable_iron:
            prob += x[mat] == 0, f"{mat}_unavailable"
        else:
            max_pct = iron_ore_max_pct.get(mat, 0.29)
            prob += x[mat] <= max_pct * total_iron_ore + 0.001, f"{mat}_max_pct"
            prob += x[mat] >= IRON_ORE_MIN_PCT.get(mat, 0.03) * total_iron_ore - 0.001, f"{mat}_min_pct"

    if "MILL_SCALE" in x:
        if "MILL_SCALE" in unavailable_iron:
            prob += x["MILL_SCALE"] == 0, "MILL_SCALE_unavailable_dedicated"
        else:
            # When available, Mill Scale MUST be 5–15% of total burden.
            prob += x["MILL_SCALE"] >= MILL_SCALE_MIN_BURDEN_PCT * total_burden, "MILL_SCALE_Burden_Min"
            prob += x["MILL_SCALE"] <= MILL_SCALE_MAX_BURDEN_PCT * total_burden, "MILL_SCALE_Burden_Max"

    iron_ore_portion_cap = MAX_IRON_ORE_PORTION_CRISIS if iron_tier == "crisis" else MAX_IRON_ORE_PORTION
    prob += total_iron_ore <= iron_ore_portion_cap * OUT, "Max_Iron_Ore_Portion"

    min_pct_source = flux_min_pct_override if flux_min_pct_override is not None else FLUX_MIN_PCT
    default_min_pct = 0.0 if flux_min_pct_override is not None else 0.02
    for mat in fluxes:
        if mat in unavailable_flux:
            prob += x[mat] == 0, f"{mat}_unavailable"
        else:
            max_pct = flux_max_pct.get(mat, 0.5)
            min_pct = min_pct_source.get(mat, default_min_pct)
            prob += x[mat] <= max_pct * total_flux + 0.001, f"{mat}_max_pct"
            prob += x[mat] >= min_pct * total_flux - 0.001, f"{mat}_min_pct"

    flux_portion_cap = MAX_FLUX_PORTION_CRISIS if flux_tier == "crisis" else MAX_FLUX_PORTION
    prob += total_flux <= flux_portion_cap * OUT, "Max_Flux_Portion"

    mandate_active = False
    for m in ["IOL_Fines", "BF_Returns"]:
        if m in df.index and df.loc[m, "Available_Tonnes"] > 0 and df.loc[m, "Tech_Max"] > 0:
            mandate_active = True
            break

    if baseline_flux_portion is not None and flux_tier not in ("crisis", "quality_relaxed") and not mandate_active:
        prob += total_flux <= (baseline_flux_portion + FLUX_BASELINE_INCREASE_CAP) * OUT, "Flux_Baseline_Cap"

    add_mandate_constraints(prob, x, df, OUT, mandate_mode=mandate_mode,
                            iol_nominal=iol_nominal, bf_nominal=bf_nominal,
                            iol_fb_min=iol_fb_min, iol_fb_max=iol_fb_max,
                            bf_fb_min=bf_fb_min, bf_fb_max=bf_fb_max)

    return total_iron_ore, total_flux, total_burden

# ============================================================================
# 7. QUALITY & DIAGNOSTIC FUNCTIONS
# ============================================================================
def compute_achieved(blend, df, OUT):
    Fe = sum(blend[m] * df.loc[m, "Fe"] / 100 for m in blend) / OUT * 100
    SiO2 = sum(blend[m] * df.loc[m, "SiO2"] / 100 for m in blend) / OUT * 100
    Al2O3 = sum(blend[m] * df.loc[m, "Al2O3"] / 100 for m in blend) / OUT * 100
    CaO = sum(blend[m] * df.loc[m, "CaO"] / 100 for m in blend) / OUT * 100
    MgO = sum(blend[m] * df.loc[m, "MgO"] / 100 for m in blend) / OUT * 100
    achieved = {"Fe": Fe, "SiO2": SiO2, "Al2O3": Al2O3, "CaO": CaO, "MgO": MgO}
    if SiO2 > 0:
        achieved["Basicity"] = CaO / SiO2
        achieved["Al2O3/SiO2"] = Al2O3 / SiO2
        achieved["B4"] = (CaO + MgO) / (SiO2 + Al2O3)
    else:
        achieved["Basicity"] = 0
        achieved["Al2O3/SiO2"] = 0
        achieved["B4"] = 0
    return achieved

def build_soft_vars_and_constraints(prob, xr, df, OUT, targets, fe_lo, fe_hi, suffix=""):
    Fe_s = pulp.lpSum(xr[m] * df.loc[m, "Fe"] / 100 for m in xr)
    SiO2_s = pulp.lpSum(xr[m] * df.loc[m, "SiO2"] / 100 for m in xr)
    Al2O3_s = pulp.lpSum(xr[m] * df.loc[m, "Al2O3"] / 100 for m in xr)
    CaO_s = pulp.lpSum(xr[m] * df.loc[m, "CaO"] / 100 for m in xr)
    MgO_s = pulp.lpSum(xr[m] * df.loc[m, "MgO"] / 100 for m in xr)

    Fe_under = pulp.LpVariable(f"Fe_under{suffix}", lowBound=0)
    Fe_over = pulp.LpVariable(f"Fe_over{suffix}", lowBound=0)
    SiO2_over = pulp.LpVariable(f"SiO2_over{suffix}", lowBound=0)
    Al2O3_over = pulp.LpVariable(f"Al2O3_over{suffix}", lowBound=0)
    ratio_over = pulp.LpVariable(f"ratio_over{suffix}", lowBound=0)
    Bas_under = pulp.LpVariable(f"Bas_under{suffix}", lowBound=0)
    Bas_over = pulp.LpVariable(f"Bas_over{suffix}", lowBound=0)
    MgO_under = pulp.LpVariable(f"MgO_under{suffix}", lowBound=0)
    MgO_over = pulp.LpVariable(f"MgO_over{suffix}", lowBound=0)
    CaO_under = pulp.LpVariable(f"CaO_under{suffix}", lowBound=0)
    CaO_over = pulp.LpVariable(f"CaO_over{suffix}", lowBound=0)
    Fe_center_dev = pulp.LpVariable(f"Fe_center_dev{suffix}", lowBound=0)

    prob += Fe_s + Fe_under >= fe_lo, f"Fe_lo_soft{suffix}"
    prob += Fe_s - Fe_over <= fe_hi, f"Fe_hi_soft{suffix}"
    prob += SiO2_s - SiO2_over <= targets["SiO2_max"] * OUT / 100, f"SiO2_soft{suffix}"
    prob += Al2O3_s - Al2O3_over <= targets["Al2O3_max"] * OUT / 100, f"Al2O3_soft{suffix}"
    prob += (Al2O3_s - targets["Al2O3_SiO2_max"] * SiO2_s) - ratio_over <= 0, f"Ratio_soft{suffix}"
    prob += (CaO_s - targets["Basicity_min"] * SiO2_s) + Bas_under >= 0, f"Basicity_lo_soft{suffix}"
    prob += (CaO_s - targets["Basicity_max"] * SiO2_s) - Bas_over <= 0, f"Basicity_hi_soft{suffix}"
    prob += MgO_s + MgO_under >= targets["MgO_min"] * OUT / 100, f"MgO_lo_soft{suffix}"
    prob += MgO_s - MgO_over <= targets["MgO_max"] * OUT / 100, f"MgO_hi_soft{suffix}"
    prob += CaO_s + CaO_under >= targets["CaO_min"] * OUT / 100, f"CaO_lo_soft{suffix}"
    prob += CaO_s - CaO_over <= targets["CaO_max"] * OUT / 100, f"CaO_hi_soft{suffix}"

    prob += Fe_s - (FE_TARGET * OUT / 100) <= Fe_center_dev, f"Fe_center_pos{suffix}"
    prob += (FE_TARGET * OUT / 100) - Fe_s <= Fe_center_dev, f"Fe_center_neg{suffix}"

    slacks = {
        "Fe_under": Fe_under, "Fe_over": Fe_over, "SiO2_over": SiO2_over, "Al2O3_over": Al2O3_over,
        "ratio_over": ratio_over, "Bas_under": Bas_under, "Bas_over": Bas_over,
        "MgO_under": MgO_under, "MgO_over": MgO_over, "CaO_under": CaO_under, "CaO_over": CaO_over,
        "Fe_center_dev": Fe_center_dev,
    }
    sums = {"Fe": Fe_s, "SiO2": SiO2_s, "Al2O3": Al2O3_s, "CaO": CaO_s, "MgO": MgO_s}
    return slacks, sums

def weighted_deviation_expr(slacks, targets, OUT):
    W = DEVIATION_WEIGHTS
    expr = (
        W["Fe"] * ((slacks["Fe_under"] + slacks["Fe_over"]) / (FE_TOLERANCE * OUT / 100)) +
        W["SiO2"] * (slacks["SiO2_over"] / (targets["SiO2_max"] * OUT / 100)) +
        W["Al2O3"] * (slacks["Al2O3_over"] / (targets["Al2O3_max"] * OUT / 100)) +
        W["Al2O3_SiO2_ratio"] * (slacks["ratio_over"] / max(targets["Al2O3_SiO2_max"] * OUT / 100, 1e-6)) +
        W["Basicity"] * ((slacks["Bas_under"] + slacks["Bas_over"]) /
                          max((targets["Basicity_max"] - targets["Basicity_min"]) * OUT / 100, 1e-6)) +
        W["MgO"] * ((slacks["MgO_under"] + slacks["MgO_over"]) /
                    max((targets["MgO_max"] - targets["MgO_min"]) * OUT / 100, 1e-6)) +
        W["CaO"] * ((slacks["CaO_under"] + slacks["CaO_over"]) /
                    max((targets["CaO_max"] - targets["CaO_min"]) * OUT / 100, 1e-6)) +
        FE_CENTER_WEIGHT * (slacks["Fe_center_dev"] / (FE_TOLERANCE * OUT / 100))
    )
    return expr

def _report_compensation(blend, df, iron_ores, fluxes, unavailable_iron, unavailable_flux,
                          iron_ore_max_pct, flux_max_pct, iron_tier, flux_tier, OUT, mandate_reasons=None):
    diagnostics = []
    if unavailable_iron or (mandate_reasons and iron_tier != "base"):
        iron_ore_total = sum(blend[m] for m in iron_ores)
        diag_msg = f"\n   ✅ Iron Ore Compensation Result (Tier: {iron_tier}):"
        diag_msg += f"\n   Iron Ore Portion: {iron_ore_total:.1f} kg ({iron_ore_total/OUT*100:.1f}% of burden)"
        for mat in iron_ores:
            if mat in unavailable_iron:
                diag_msg += f"\n      {mat}: UNAVAILABLE"
            else:
                pct = blend[mat] / iron_ore_total * 100 if iron_ore_total > 0 else 0
                max_pct = iron_ore_max_pct.get(mat, 0.29) * 100
                diag_msg += f"\n      {mat}: {blend[mat]:.1f} kg ({pct:.1f}%) [Max {max_pct:.0f}%]"
        diagnostics.append(diag_msg)

    if unavailable_flux or (mandate_reasons and flux_tier != "base"):
        flux_total = sum(blend[m] for m in fluxes)
        diag_msg = f"\n   ✅ Flux Compensation Result (Tier: {flux_tier}):"
        for mat in fluxes:
            if mat in unavailable_flux:
                diag_msg += f"\n      {mat}: UNAVAILABLE"
            else:
                pct = blend[mat] / flux_total * 100 if flux_total > 0 else 0
                max_pct = flux_max_pct.get(mat, 0.5) * 100
                diag_msg += f"\n      {mat}: {blend[mat]:.1f} kg ({pct:.1f}%) [Max {max_pct:.0f}%]"
        diagnostics.append(diag_msg)

    total_burden_actual = sum(blend[m] for m in blend if m in df.index)
    if total_burden_actual > 0:
        iol_actual = blend.get("IOL_Fines", 0.0)
        bf_actual = blend.get("BF_Returns", 0.0)
        diagnostics.append(
            f"\n   🔒 STRICT BURDEN-PERCENT MANDATES: "
            f"IOL Fines = {iol_actual:.2f} kg ({iol_actual/total_burden_actual*100:.2f}% of total burden; target {IOL_FINES_NOMINAL_PCT*100:.2f}%), "
            f"BF Returns = {bf_actual:.2f} kg ({bf_actual/total_burden_actual*100:.2f}% of total burden; target {BF_RETURNS_NOMINAL_PCT*100:.2f}%)."
        )

    if mandate_reasons:
        diagnostics.append(
            f"\n   🔗 NOTE: {', '.join(mandate_reasons)} shortfall detected. Strict burden-percentage mandates are not relaxed."
        )

    if iron_tier == "crisis" or flux_tier == "crisis":
        diagnostics.append(
            "\n   💰 NOTE: CRISIS MODE – usage caps loosened to hit quality; cost may be higher."
        )
    return diagnostics

def _report_fines_loading(blend, df, OUT):
    fine_materials = [m for m in ["IOL_Fines", "BF_Returns", "MILL_SCALE"] if m in blend]
    total_fines = sum(blend.get(m, 0) for m in fine_materials)
    total_burden = sum(blend[m] for m in blend if m in df.index)
    pct = total_fines / total_burden * 100 if total_burden > 0 else 0.0
    msg = f"\n   🧱 COMBINED FINES LOADING (IOL_Fines + BF_Returns + Mill Scale): {total_fines:.1f} kg ({pct:.1f}% of total burden)"
    if pct > 30:
        msg += "\n   ⚠️ HIGH FINES LOADING (>30%) – verify permeability."
    return msg

def _print_dynamic_iron_ore_ranking(df):
    """Display current submitted iron-ore/dealer chemistry ranking by Fe%."""
    if df is None or df.empty:
        return
    rows = []
    for mat in df.index:
        if mat in _eligible_iron_ores(df) and float(df.loc[mat, "Available_Tonnes"]) > 0:
            rows.append({
                "Material / Dealer": mat,
                "Fe %": float(df.loc[mat, "Fe"]),
                "Available (t)": float(df.loc[mat, "Available_Tonnes"])
            })
    if not rows:
        return
    ranking = sorted(rows, key=lambda r: (-r["Fe %"], r["Material / Dealer"]))
    for i, r in enumerate(ranking, 1):
        r["Rank"] = i
    out = pd.DataFrame(ranking)[["Rank", "Material / Dealer", "Fe %", "Available (t)"]]
    print("\n🏷️ CURRENT IRON-ORE QUALITY RANKING (CURRENT SUBMITTED CHEMISTRY — BY Fe %)")
    display(HTML(out.to_html(index=False, float_format="%.2f")))

# ============================================================================
# 8. WET/DRY COSTING TABLES (with TOTAL rows)
# ============================================================================
def compute_dry_cost_table(blend, df, om_cost):
    total_raw_input = sum(blend.values())
    rm_cost = sum(blend[m] * df.loc[m, "Price_Rs_t"] / 1000 for m in blend)
    table = pd.DataFrame({
        "Group": [df.loc[m, "Group"] for m in blend],
        "Dry kg / t sinter": [blend[m] for m in blend],
        "% of Burden": [(blend[m] / total_raw_input) * 100 if total_raw_input else 0 for m in blend],
        "Dry Cost (Rs/t)": [round(blend[m] * df.loc[m, "Price_Rs_t"] / 1000, 2) for m in blend],
    }, index=blend.keys())
    total_row = pd.DataFrame({
        "Group": ["TOTAL"],
        "Dry kg / t sinter": [total_raw_input],
        "% of Burden": [100.0],
        "Dry Cost (Rs/t)": [round(rm_cost, 2)],
    }, index=["TOTAL"])
    table = pd.concat([table, total_row])
    total_sinter_cost = rm_cost + om_cost
    return table, rm_cost, total_sinter_cost

def compute_wet_cost_table(blend, df, om_cost):
    if "Moisture_Pct" not in df.columns:
        print("⚠️ WARNING: 'Moisture_Pct' column not found – defaulting all moisture to 0% for wet costing.")
        df = df.copy()
        df["Moisture_Pct"] = 0.0

    rows = []
    wet_rm_cost = 0
    total_dry_kg = 0
    total_wet_kg = 0
    for m in blend:
        moisture = df.loc[m, "Moisture_Pct"] / 100 if df.loc[m, "Moisture_Pct"] < 100 else 0.0
        dry_kg = blend[m]
        wet_kg = dry_kg / (1 - moisture) if moisture < 1 else dry_kg
        wet_cost = wet_kg * df.loc[m, "Price_Rs_t"] / 1000
        wet_rm_cost += wet_cost
        total_dry_kg += dry_kg
        total_wet_kg += wet_kg
        rows.append({
            "Material": m, "Group": df.loc[m, "Group"],
            "Dry kg / t sinter": dry_kg,
            "Moisture %": round(moisture * 100, 2),
            "Wet (As-Received) kg": round(wet_kg, 2),
            "Wet Cost (Rs/t)": round(wet_cost, 2),
        })

    table = pd.DataFrame(rows).set_index("Material")
    total_row = pd.DataFrame({
        "Group": ["TOTAL"],
        "Dry kg / t sinter": [total_dry_kg],
        "Moisture %": [0.0],
        "Wet (As-Received) kg": [total_wet_kg],
        "Wet Cost (Rs/t)": [round(wet_rm_cost, 2)],
    }, index=["TOTAL"])
    table = pd.concat([table, total_row])
    table["Moisture %"] = table["Moisture %"].astype(float)
    total_sinter_cost_wet = wet_rm_cost + om_cost
    return table, wet_rm_cost, total_sinter_cost_wet

# ============================================================================
# 9. COKE HEAT-BALANCE DIAGNOSTIC (now also used as a constraint in the LP)
# ============================================================================
def compute_coke_heat_balance_diagnostic(blend, df, OUT,
                                         coke_cv=DEFAULT_COKE_CV_KCAL_KG,
                                         coke_fc=DEFAULT_COKE_FC_PCT,
                                         latent_heat=DEFAULT_HEAT_LATENT_MOISTURE,
                                         calcination_heat=DEFAULT_HEAT_CALCINATION_PER_LOI_KG,
                                         melting_heat=DEFAULT_HEAT_MELTING_PER_KG_SINTER,
                                         loss_fraction=DEFAULT_HEAT_LOSS_FRACTION,
                                         feo_min=DEFAULT_FEO_MIN_PCT,
                                         feo_target=DEFAULT_FEO_TARGET_PCT,
                                         feo_max=DEFAULT_FEO_MAX_PCT,
                                         feo_ref_surplus=DEFAULT_FEO_REFERENCE_THERMAL_SURPLUS_KCAL,
                                         feo_ref_pct=DEFAULT_FEO_REFERENCE_PCT,
                                         feo_thermal_slope=DEFAULT_FEO_THERMAL_SLOPE_PCT_PER_10K_KCAL,
                                         ref_coke_cv=DEFAULT_REFERENCE_COKE_CV_KCAL_KG,
                                         ref_coke_fc=DEFAULT_REFERENCE_COKE_FC_PCT):
    """Coke heat-balance + thermal-state FeO diagnostic.

    v30:
        Q_fuel = coke rate × fixed carbon × CV
        Q_required = moisture + calcination + base thermal demand + losses
        Thermal surplus = Q_fuel - Q_required
        FeO = FeO_ref + slope × (thermal surplus - reference surplus) / 10,000

    This keeps the FeO relationship linear/LP-compatible and allows changes
    in raw-material moisture/LOI to change the optimized coke requirement.
    The coefficients are provisional until plant Coke/FeO history is available.
    """
    if "COKE_BREEZE" not in blend:
        return None

    cb_kg = blend["COKE_BREEZE"]
    Q_fuel = cb_kg * (coke_fc / 100) * coke_cv

    total_wet_mass = 0.0
    for m in blend:
        if m == "COKE_BREEZE":
            continue
        moisture = df.loc[m, "Moisture_Pct"] / 100 if "Moisture_Pct" in df.columns else 0.0
        wet_kg = blend[m] / (1 - moisture) if moisture < 1 else blend[m]
        total_wet_mass += wet_kg

    dry_nonfuel = sum(blend[m] for m in blend if m != "COKE_BREEZE")
    moisture_mass = max(total_wet_mass - dry_nonfuel, 0.0)
    Q_moisture = moisture_mass * latent_heat

    total_loi_mass = sum(
        blend[m] * df.loc[m, "LOI"] / 100
        for m in blend if m != "COKE_BREEZE"
    )
    Q_calcination = total_loi_mass * calcination_heat
    Q_melting = OUT * melting_heat

    Q_required_before_loss = Q_moisture + Q_calcination + Q_melting
    Q_required = (
        Q_required_before_loss / (1 - loss_fraction)
        if (1 - loss_fraction) > 0 else 0.0
    )

    firing_ratio = Q_fuel / Q_required if Q_required > 0 else 0.0
    thermal_surplus = Q_fuel - Q_required

    # Effective coke normalized to the reference coke quality.
    # This is a diagnostic quantity only; the LP still uses the actual coke rate.
    effective_coke_kg_t = cb_kg * (coke_fc / ref_coke_fc) * (coke_cv / ref_coke_cv)

    feo_est = feo_ref_pct + feo_thermal_slope * (
        (thermal_surplus - feo_ref_surplus) / 10000.0
    )

    if feo_est < feo_min:
        suggestion = (
            f"FeO {feo_est:.2f}% below minimum {feo_min:.2f}% "
            "→ increase coke / review thermal conditions"
        )
    elif feo_est > feo_max:
        suggestion = (
            f"FeO {feo_est:.2f}% above maximum {feo_max:.2f}% "
            "→ reduce coke / review thermal conditions"
        )
    elif abs(feo_est - feo_target) <= 0.15:
        suggestion = (
            f"FeO {feo_est:.2f}% is close to target {feo_target:.2f}% "
            "– no adjustment suggested"
        )
    elif feo_est < feo_target:
        suggestion = (
            f"FeO {feo_est:.2f}% is below target {feo_target:.2f}% "
            "but within operating band"
        )
    else:
        suggestion = (
            f"FeO {feo_est:.2f}% is above target {feo_target:.2f}% "
            "but within operating band"
        )

    return {
        "CB_kg_LP_chosen": cb_kg,
        "Q_fuel_kcal": Q_fuel,
        "Q_required_kcal": Q_required,
        "Thermal_Surplus_kcal": thermal_surplus,
        "Firing_Ratio": firing_ratio,
        "FeO_Estimate_Pct": feo_est,
        "FeO_Min_Pct": feo_min,
        "FeO_Target_Pct": feo_target,
        "FeO_Max_Pct": feo_max,
        "Reference_Thermal_Surplus_kcal": feo_ref_surplus,
        "Thermal_Slope_Pct_per_10k_kcal": feo_thermal_slope,
        "Effective_Coke_kg_t": effective_coke_kg_t,
        "Controller_Suggestion": suggestion,
        "note": (
            "⚠️ PROVISIONAL FeO thermal-state model – calibrate "
            "coefficients against plant Coke/FeO history."
        )
    }


# ============================================================================
# v31.9 — ROLE-BASED OPTIONAL ALTERNATIVE ORES
# ============================================================================
# Alternative status is determined by Material_Role in the master Excel.
# Material/dealer names are NOT hard-coded.
ROLE_COLUMN = "Material_Role"

def _ensure_material_role(df):
    d = df.copy()
    if ROLE_COLUMN not in d.columns:
        # Backward compatibility for older masters only.
        legacy = {"SKME", "THAKUR", "RBSNN", "SMIORE"}
        d[ROLE_COLUMN] = [
            "Alternative_Iron_Ore" if str(m).strip().upper() in legacy
            else ("Primary_Iron_Ore" if str(d.loc[m, "Group"]).strip() == "Iron_ore" else "Other")
            for m in d.index
        ]
    return d

def _primary_iron_ores(df):
    d = _ensure_material_role(df)
    return [
        m for m in d.index
        if str(d.loc[m, "Group"]).strip() == "Iron_ore"
        and str(d.loc[m, ROLE_COLUMN]).strip() == "Primary_Iron_Ore"
    ]

def _alternative_iron_ores(df):
    d = _ensure_material_role(df)
    return [
        m for m in d.index
        if str(d.loc[m, "Group"]).strip() == "Iron_ore"
        and str(d.loc[m, ROLE_COLUMN]).strip() == "Alternative_Iron_Ore"
    ]

def _eligible_iron_ores(df):
    """Primary ores + alternative ores explicitly enabled through UI availability."""
    d = _ensure_material_role(df)
    return _primary_iron_ores(d) + [
        m for m in _alternative_iron_ores(d)
        if float(d.loc[m, "Available_Tonnes"]) > 0
        and float(d.loc[m, "Tech_Max"]) > 0
    ]

# ============================================================================
# 10. MAIN SOLVER (accepts all parameters + manual override)
# ============================================================================
def solve_blend_with_compensation(df, production_tonnes, targets, baseline_blend=None,
                                  enforce_b4=False, b4_min=1.8, b4_max=2.0,
                                  iol_nominal=IOL_FINES_NOMINAL_PCT,
                                  bf_nominal=BF_RETURNS_NOMINAL_PCT,
                                  iol_fb_min=IOL_FINES_FALLBACK_MIN,
                                  iol_fb_max=IOL_FINES_FALLBACK_MAX,
                                  bf_fb_min=BF_RETURNS_FALLBACK_MIN,
                                  bf_fb_max=BF_RETURNS_FALLBACK_MAX,
                                  coke_cv=DEFAULT_COKE_CV_KCAL_KG,
                                  coke_fc=DEFAULT_COKE_FC_PCT,
                                  latent_heat=DEFAULT_HEAT_LATENT_MOISTURE,
                                  calcination_heat=DEFAULT_HEAT_CALCINATION_PER_LOI_KG,
                                  melting_heat=DEFAULT_HEAT_MELTING_PER_KG_SINTER,
                                  loss_fraction=DEFAULT_HEAT_LOSS_FRACTION,
                                  firing_ratio_max=DEFAULT_FIRING_RATIO_MAX,
                                  enforce_firing_ratio_max=ENFORCE_FIRING_RATIO_MAX,
                                  coke_min_rate=DEFAULT_COKE_MIN_KG_T,
                                  coke_max_rate=DEFAULT_COKE_MAX_KG_T,
                                  feo_min=DEFAULT_FEO_MIN_PCT,
                                  feo_target=DEFAULT_FEO_TARGET_PCT,
                                  feo_max=DEFAULT_FEO_MAX_PCT,
                                  feo_ref_surplus=DEFAULT_FEO_REFERENCE_THERMAL_SURPLUS_KCAL,
                                  feo_ref_pct=DEFAULT_FEO_REFERENCE_PCT,
                                  feo_thermal_slope=DEFAULT_FEO_THERMAL_SLOPE_PCT_PER_10K_KCAL,
                                  ref_coke_cv=DEFAULT_REFERENCE_COKE_CV_KCAL_KG,
                                  ref_coke_fc=DEFAULT_REFERENCE_COKE_FC_PCT,
                                  manual_override=False,
                                  manual_coke_rate=65.0,
                                  fixed_quantities=None):
    OUT = 1000
    df = _ensure_material_role(df)
    iron_ores = _eligible_iron_ores(df)
    fluxes = [m for m in df.index if df.loc[m, "Group"] == "Flux"]

    fuel_ok, fuel_problems = check_fuel_gate(df)
    if not fuel_ok:
        diagnostics = ["🚫 PRODUCTION IMPOSSIBLE: Fuel requirement cannot be met."]
        for p in fuel_problems:
            diagnostics.append(f"   {p}")
        return "No_Production", None, None, None, diagnostics, False

    # Strict mandates are independent of ore/flux compensation tiers.
    # A shortage of IOL Fines or BF Returns must make the optimization
    # infeasible; it must NOT relax ore/flux caps or the mandate itself.
    mandate_reasons = []
    mandate_triggers = 0
    for mat, nominal in [("IOL_Fines", iol_nominal), ("BF_Returns", bf_nominal)]:
        if mat not in df.index:
            mandate_reasons.append(f"{mat} missing from chemistry master")
        elif df.loc[mat, "Available_Tonnes"] <= 0:
            mandate_reasons.append(
                f"{mat} unavailable (strict requirement is {nominal*100:.1f}% of total burden)"
            )
        elif df.loc[mat, "Tech_Max"] <= 0:
            mandate_reasons.append(
                f"{mat} Tech_Max is zero (strict requirement is {nominal*100:.1f}% of total burden)"
            )

    iron_ore_max_pct, unavailable_iron, iron_msg, iron_tier = get_iron_ore_tier(df, iron_ores)
    print(iron_msg)
    flux_max_pct, unavailable_flux, flux_msg, flux_tier = get_flux_tier(df, fluxes)
    print(flux_msg)

    if mandate_reasons:
        diagnostics = [
            "⚠️ MANDATE AVAILABILITY ISSUE — production-continuity mode is active.",
            *[f"   {r}" for r in mandate_reasons],
            "   Normal production optimization will keep IOL=8% and BF=17% whenever those materials are available.",
            "   If a mandated material is unavailable, recovery mode may flag a mandate exception rather than stopping production."
        ]
    else:
        diagnostics = []

    bounds = build_bounds(df, production_tonnes)
    if unavailable_iron:
        diagnostics.append(f"🔄 Compensating for missing ore(s): {', '.join(unavailable_iron)} (Iron tier: {iron_tier})")
    if unavailable_flux:
        diagnostics.append(f"🔄 Compensating for missing flux(es): {', '.join(unavailable_flux)} (Flux tier: {flux_tier})")
    if mandate_reasons:
        diagnostics.append(f"🔗 Mandate shortfall ({', '.join(mandate_reasons)}) escalated BOTH tiers.")

    fe_lo = FE_LOWER * OUT / 100
    fe_hi = FE_UPPER * OUT / 100

    baseline_flux_portion = None
    if baseline_blend:
        baseline_flux_portion = sum(baseline_blend.get(m, 0) for m in fluxes) / OUT

    fixed_quantities = {m: float(v) for m, v in (fixed_quantities or {}).items() if m in df.index}

    def _apply_fixed_quantities(prob, x, suffix=""):
        for mat, qty in fixed_quantities.items():
            prob += x[mat] == qty, f"Manual_Fixed_{re.sub(r'[^A-Za-z0-9_]', '_', mat)}{suffix}"

    shortage_targets = None
    if iron_tier != "base":
        shortage_targets = dict(targets)
        shortage_targets["SiO2_max"] = SIO2_MAX_SHORTAGE
        shortage_targets["CaO_max"] = round(targets["Basicity_max"] * SIO2_MAX_SHORTAGE, 3)

    # Precompute moisture and LOI factors for heat balance
    moisture_factor = {}
    loi_factor = {}
    for m in df.index:
        if m == "COKE_BREEZE":
            continue
        mois = df.loc[m, "Moisture_Pct"] / 100 if "Moisture_Pct" in df.columns else 0.0
        if mois < 1:
            moisture_factor[m] = mois / (1 - mois)
        else:
            moisture_factor[m] = 0.0
        loi_factor[m] = df.loc[m, "LOI"] / 100

    def _add_recovery_mandate_constraints(prob, x):
        """Production-continuity recovery: keep available mandates strict.
        If a mandated material is unavailable, replace its equality with a
        bounded availability exception and report it. This does NOT relax
        quality chemistry targets.
        """
        total_burden = pulp.lpSum(x[m] for m in df.index)
        for mat, nominal in [("IOL_Fines", iol_nominal), ("BF_Returns", bf_nominal)]:
            if mat not in x:
                continue
            available = df.loc[mat, "Available_Tonnes"] > 0 and df.loc[mat, "Tech_Max"] > 0
            if available:
                prob += x[mat] == nominal * total_burden, f"{mat}_Recovery_Strict"
            else:
                prob += x[mat] == 0, f"{mat}_Recovery_Unavailable"

    def _build_and_solve(flux_min_pct_override, flux_tier_label, tag,
                         use_targets=None, mandate_mode="pinned"):
        t = use_targets if use_targets is not None else targets
        prob = pulp.LpProblem(f"Sinter_Burden_Opt_{tag}", pulp.LpMinimize)
        x = {
            m: pulp.LpVariable(
                f"x{tag}_{m}",
                lowBound=bounds[m][0],
                upBound=bounds[m][1]
            )
            for m in df.index
        }

        _apply_fixed_quantities(prob, x, suffix=f"_{tag}")

        add_structural_constraints(
            prob, x, df, bounds, iron_ores, fluxes,
            iron_ore_max_pct, unavailable_iron,
            flux_max_pct, unavailable_flux, OUT,
            baseline_flux_portion,
            iron_tier, flux_tier_label,
            flux_min_pct_override=flux_min_pct_override,
            mandate_mode=mandate_mode,
            iol_nominal=iol_nominal,
            bf_nominal=bf_nominal,
            iol_fb_min=iol_fb_min,
            iol_fb_max=iol_fb_max,
            bf_fb_min=bf_fb_min,
            bf_fb_max=bf_fb_max
        )

        Fe_sum = pulp.lpSum(x[m] * df.loc[m, "Fe"] / 100 for m in x)
        SiO2_sum = pulp.lpSum(x[m] * df.loc[m, "SiO2"] / 100 for m in x)
        Al2O3_sum = pulp.lpSum(x[m] * df.loc[m, "Al2O3"] / 100 for m in x)
        CaO_sum = pulp.lpSum(x[m] * df.loc[m, "CaO"] / 100 for m in x)
        MgO_sum = pulp.lpSum(x[m] * df.loc[m, "MgO"] / 100 for m in x)

        prob += Fe_sum >= fe_lo, "Fe_min_hard"
        prob += Fe_sum <= fe_hi, "Fe_max_hard"
        prob += SiO2_sum <= t["SiO2_max"] * OUT / 100, "SiO2_max_hard"
        prob += Al2O3_sum <= t["Al2O3_max"] * OUT / 100, "Al2O3_max_hard"
        prob += (
            Al2O3_sum - t["Al2O3_SiO2_max"] * SiO2_sum <= 0
        ), "Al2O3_SiO2_max_hard"
        prob += (
            CaO_sum >= t["Basicity_min"] * SiO2_sum
        ), "Basicity_min_hard"
        prob += (
            CaO_sum <= t["Basicity_max"] * SiO2_sum
        ), "Basicity_max_hard"
        prob += MgO_sum >= t["MgO_min"] * OUT / 100, "MgO_min_hard"
        prob += MgO_sum <= t["MgO_max"] * OUT / 100, "MgO_max_hard"
        prob += CaO_sum >= t["CaO_min"] * OUT / 100, "CaO_min_hard"
        prob += CaO_sum <= t["CaO_max"] * OUT / 100, "CaO_max_hard"

        if enforce_b4:
            prob += (
                (CaO_sum + MgO_sum)
                - b4_min * (SiO2_sum + Al2O3_sum) >= 0
            ), "B4_min"
            prob += (
                (CaO_sum + MgO_sum)
                - b4_max * (SiO2_sum + Al2O3_sum) <= 0
            ), "B4_max"

        cost_expr = pulp.lpSum(
            x[m] * df.loc[m, "Price_Rs_t"] / 1000 for m in x
        )

        if "COKE_BREEZE" in x:
            cb = x["COKE_BREEZE"]
            prob += cb >= coke_min_rate, "Coke_Practical_Min"
            prob += cb <= coke_max_rate, "Coke_Practical_Max"

            Q_fuel = cb * (coke_fc / 100) * coke_cv
            Q_moisture = latent_heat * pulp.lpSum(
                x[m] * moisture_factor.get(m, 0.0)
                for m in x if m != "COKE_BREEZE"
            )
            Q_calcination = calcination_heat * pulp.lpSum(
                x[m] * loi_factor.get(m, 0.0)
                for m in x if m != "COKE_BREEZE"
            )
            Q_melting = melting_heat * OUT
            Q_required = (
                Q_moisture + Q_calcination + Q_melting
            ) / (1 - loss_fraction)
            thermal_surplus = Q_fuel - Q_required

            if manual_override:
                prob += cb == manual_coke_rate, "Manual_Coke_Override"
            else:
                prob += Q_fuel >= Q_required, "Heat_Balance_Min"
                if enforce_firing_ratio_max:
                    prob += (
                        Q_fuel <= Q_required * firing_ratio_max
                    ), "Firing_Ratio_Max"

            # IMPORTANT v30:
            # FeO depends on thermal surplus, so changing burden chemistry,
            # moisture or LOI changes the coke requirement.
            FeO_pred = feo_ref_pct + feo_thermal_slope * (
                (thermal_surplus - feo_ref_surplus) / 10000.0
            )

            prob += FeO_pred >= feo_min, "FeO_Min_Hard"
            prob += FeO_pred <= feo_max, "FeO_Max_Hard"

            # Target is lexicographically preferred:
            # 1) get as close as possible to FeO target;
            # 2) among equally good FeO solutions, minimize material cost.
            feo_dev = pulp.LpVariable(
                f"FeO_Target_Deviation_{tag}", lowBound=0
            )
            prob += (
                feo_dev >= FeO_pred - feo_target
            ), f"FeO_Target_Dev_Pos_{tag}"
            prob += (
                feo_dev >= feo_target - FeO_pred
            ), f"FeO_Target_Dev_Neg_{tag}"

            if manual_override:
                prob.setObjective(cost_expr)
            else:
                prob.setObjective(feo_dev)
                prob.solve(pulp.PULP_CBC_CMD(msg=0))

                if pulp.LpStatus[prob.status] != "Optimal":
                    return prob, x, pulp.LpStatus[prob.status]

                min_dev = max(0.0, float(pulp.value(feo_dev) or 0.0))
                prob += (
                    feo_dev <= min_dev + 1e-6
                ), f"FeO_Target_Dev_Pin_{tag}"
                prob.setObjective(cost_expr)
        else:
            prob.setObjective(cost_expr)

        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        return prob, x, pulp.LpStatus[prob.status]

    def _add_diagnostic_coke_constraints(prob, x, suffix=""):
        """Apply practical coke, heat and thermal-state FeO constraints to fallback blends."""
        if "COKE_BREEZE" not in x:
            return

        if manual_override:
            prob += (
                x["COKE_BREEZE"] == manual_coke_rate
            ), f"Manual_Coke_Diagnostic{suffix}"
            return

        cb = x["COKE_BREEZE"]
        prob += cb >= coke_min_rate, f"Coke_Practical_Min{suffix}"
        prob += cb <= coke_max_rate, f"Coke_Practical_Max{suffix}"

        Q_fuel = cb * (coke_fc / 100) * coke_cv
        Q_moisture = latent_heat * pulp.lpSum(
            x[m] * moisture_factor.get(m, 0.0)
            for m in x if m != "COKE_BREEZE"
        )
        Q_calcination = calcination_heat * pulp.lpSum(
            x[m] * loi_factor.get(m, 0.0)
            for m in x if m != "COKE_BREEZE"
        )
        Q_melting = melting_heat * OUT
        Q_required = (
            Q_moisture + Q_calcination + Q_melting
        ) / (1 - loss_fraction)

        prob += Q_fuel >= Q_required, f"Heat_Balance_Min{suffix}"

        if enforce_firing_ratio_max:
            prob += (
                Q_fuel <= Q_required * firing_ratio_max
            ), f"Firing_Ratio_Max{suffix}"

        thermal_surplus = Q_fuel - Q_required
        FeO_pred = feo_ref_pct + feo_thermal_slope * (
            (thermal_surplus - feo_ref_surplus) / 10000.0
        )

        prob += FeO_pred >= feo_min, f"FeO_Min_Hard{suffix}"
        prob += FeO_pred <= feo_max, f"FeO_Max_Hard{suffix}"

    def _finalize(prob, x, tag_label, note=None):
        blend = {m: round(x[m].value(), 2) for m in x}
        total_cost = pulp.value(prob.objective)
        achieved = compute_achieved(blend, df, OUT)
        diag = list(diagnostics)
        if note:
            diag.append(note)
        diag += _report_compensation(blend, df, iron_ores, fluxes, unavailable_iron, unavailable_flux,
                                      iron_ore_max_pct, flux_max_pct, iron_tier, flux_tier, OUT, mandate_reasons)
        diag.append(_report_fines_loading(blend, df, OUT))
        # Add heat balance info (using same coefficients as the solver)
        heat_info = compute_coke_heat_balance_diagnostic(blend, df, OUT, coke_cv, coke_fc,
                                                         latent_heat, calcination_heat, melting_heat, loss_fraction,
                                                         feo_min, feo_target, feo_max, feo_ref_surplus, feo_ref_pct,
                                                         feo_thermal_slope, ref_coke_cv, ref_coke_fc)
        if heat_info:
            if manual_override:
                diag.append(f"\n🔥 MANUAL OVERRIDE: Coke Breeze forced to {manual_coke_rate:.1f} kg/t")
            else:
                diag.append(f"\n🔥 Optimised Coke Breeze: {heat_info['CB_kg_LP_chosen']:.1f} kg/t")
            diag.append(f"   Firing Ratio: {heat_info['Firing_Ratio']:.3f} (max {firing_ratio_max})")
            diag.append(f"   Predicted FeO: {heat_info['FeO_Estimate_Pct']:.2f}% (target {feo_target:.2f}%, band {feo_min:.2f}-{feo_max:.2f}%)")
            diag.append(f"   {heat_info['Controller_Suggestion']}")
        return "Optimal", blend, total_cost, achieved, diag, False

    # ---- PASSES A-F (same as before) ----
    probA, xA, statusA = _build_and_solve(None, flux_tier, "A", mandate_mode="pinned")
    if statusA == "Optimal":
        return _finalize(probA, xA, "A", "✅ Mandates met at nominal.")

    diagnostics.append("⚠️ Base flux floors infeasible with pinned mandates – relaxing flux floors...")
    probB, xB, statusB = _build_and_solve(FLUX_MIN_PCT_QUALITY_RELAXED, "quality_relaxed", "B", mandate_mode="pinned")
    if statusB == "Optimal":
        return _finalize(probB, xB, "B", "✅ Resolved with relaxed flux floors.")

    if shortage_targets is not None:
        diagnostics.append(f"🔄 Iron ore short – widening SiO2/CaO ceilings...")
        probC, xC, statusC = _build_and_solve(FLUX_MIN_PCT_QUALITY_RELAXED, "quality_relaxed", "C", use_targets=shortage_targets, mandate_mode="pinned")
        if statusC == "Optimal":
            return _finalize(probC, xC, "C", "✅ Resolved with widened ceilings – mandates pinned.")

    diagnostics.append("⚠️ Nominal chemistry solution infeasible with STRICT IOL/BF mandates – trying the remaining chemistry-relaxation passes while keeping IOL=8% and BF=17% strictly pinned.")
    probD, xD, statusD = _build_and_solve(None, flux_tier, "D", mandate_mode="pinned")
    if statusD == "Optimal":
        return _finalize(probD, xD, "D", "✅ Solved with chemistry relaxation; IOL/BF mandates remained strict.")

    probE, xE, statusE = _build_and_solve(FLUX_MIN_PCT_QUALITY_RELAXED, "quality_relaxed", "E", mandate_mode="pinned")
    if statusE == "Optimal":
        return _finalize(probE, xE, "E", "✅ Solved with relaxed chemistry/flux floors; IOL/BF mandates remained strict.")

    if shortage_targets is not None:
        probF, xF, statusF = _build_and_solve(FLUX_MIN_PCT_QUALITY_RELAXED, "quality_relaxed", "F", use_targets=shortage_targets, mandate_mode="pinned")
        if statusF == "Optimal":
            return _finalize(probF, xF, "F", "✅ Solved using permitted chemistry relaxations; IOL/BF mandates remained strict.")

    # ============================ PRODUCTION-CONTINUITY RECOVERY ============================
    # First try to preserve ALL quality targets while handling only genuine
    # material-availability exceptions. Cost is still minimized here.
    if PRODUCTION_CONTINUITY_MODE:
        diagnostics.append("🟡 Production-continuity recovery: searching for a quality-compliant blend at higher cost if necessary.")

        probR = pulp.LpProblem("Production_Continuity_Recovery", pulp.LpMinimize)
        xR = {
            m: pulp.LpVariable(f"xR_{m}", lowBound=bounds[m][0], upBound=bounds[m][1])
            for m in df.index
        }

        # Keep structural constraints, but replace the strict mandate equations
        # only for unavailable mandated materials.
        add_structural_constraints(
            probR, xR, df, bounds, iron_ores, fluxes,
            iron_ore_max_pct, unavailable_iron,
            flux_max_pct, unavailable_flux, OUT,
            baseline_flux_portion, iron_tier, "quality_relaxed",
            flux_min_pct_override=FLUX_MIN_PCT_QUALITY_RELAXED,
            mandate_mode="recovery",
            iol_nominal=iol_nominal, bf_nominal=bf_nominal,
            iol_fb_min=iol_fb_min, iol_fb_max=iol_fb_max,
            bf_fb_min=bf_fb_min, bf_fb_max=bf_fb_max
        )

        # The structural function does not interpret "recovery" specially, so
        # remove any strict mandate equalities it may have created and add our
        # availability-aware version.
        for cname in ["IOL_Fines_STRICT_BURDEN_PCT", "BF_Returns_STRICT_BURDEN_PCT"]:
            if cname in probR.constraints:
                del probR.constraints[cname]
        _add_recovery_mandate_constraints(probR, xR)

        Fe_sumR = pulp.lpSum(xR[m] * df.loc[m, "Fe"] / 100 for m in xR)
        SiO2_sumR = pulp.lpSum(xR[m] * df.loc[m, "SiO2"] / 100 for m in xR)
        Al2O3_sumR = pulp.lpSum(xR[m] * df.loc[m, "Al2O3"] / 100 for m in xR)
        CaO_sumR = pulp.lpSum(xR[m] * df.loc[m, "CaO"] / 100 for m in xR)
        MgO_sumR = pulp.lpSum(xR[m] * df.loc[m, "MgO"] / 100 for m in xR)

        probR += Fe_sumR >= fe_lo, "Recovery_Fe_min"
        probR += Fe_sumR <= fe_hi, "Recovery_Fe_max"
        probR += SiO2_sumR <= targets["SiO2_max"] * OUT / 100, "Recovery_SiO2_max"
        probR += Al2O3_sumR <= targets["Al2O3_max"] * OUT / 100, "Recovery_Al2O3_max"
        probR += Al2O3_sumR <= targets["Al2O3_SiO2_max"] * SiO2_sumR, "Recovery_Al2O3_SiO2"
        probR += CaO_sumR >= targets["Basicity_min"] * SiO2_sumR, "Recovery_Basicity_min"
        probR += CaO_sumR <= targets["Basicity_max"] * SiO2_sumR, "Recovery_Basicity_max"
        probR += MgO_sumR >= targets["MgO_min"] * OUT / 100, "Recovery_MgO_min"
        probR += MgO_sumR <= targets["MgO_max"] * OUT / 100, "Recovery_MgO_max"
        probR += CaO_sumR >= targets["CaO_min"] * OUT / 100, "Recovery_CaO_min"
        probR += CaO_sumR <= targets["CaO_max"] * OUT / 100, "Recovery_CaO_max"

        _add_diagnostic_coke_constraints(probR, xR, suffix="_recovery")
        probR += pulp.lpSum(xR[m] * df.loc[m, "Price_Rs_t"] / 1000 for m in xR), "Recovery_Total_Cost"
        probR.solve(pulp.PULP_CBC_CMD(msg=0))

        if pulp.LpStatus[probR.status] == "Optimal":
            blendR = {m: round(xR[m].value(), 2) for m in xR}
            costR = sum(blendR[m] * df.loc[m, "Price_Rs_t"] / 1000 for m in blendR)
            achievedR = compute_achieved(blendR, df, OUT)
            rec_diag = list(diagnostics)
            rec_diag.append("🟢 Production-continuity recovery found a quality-compliant blend.")
            rec_diag.append(f"💰 Recovery cost: Rs {costR:,.2f}/t — cost may be higher because available materials are constrained.")
            if mandate_reasons:
                rec_diag.append("⚠️ Mandate exception: an unavailable IOL/BF material could not be physically loaded.")
            rec_diag += _report_compensation(blendR, df, iron_ores, fluxes, unavailable_iron, unavailable_flux,
                                              iron_ore_max_pct, flux_max_pct, iron_tier, flux_tier, OUT, mandate_reasons)
            return "Recovery", blendR, costR, achievedR, rec_diag, False

    # ============================ PHASE 1 & 2 (diagnostic) ============================
    diagnostics.append("⚠️ All hard constraints infeasible – generating closest achievable reference.")
    prob1 = pulp.LpProblem("Phase1_MinDeviation", pulp.LpMinimize)
    x1 = {m: pulp.LpVariable(f"x1_{m}", lowBound=bounds[m][0], upBound=bounds[m][1]) for m in df.index}
    _apply_fixed_quantities(prob1, x1, suffix="_p1")
    add_structural_constraints(prob1, x1, df, bounds, iron_ores, fluxes, iron_ore_max_pct, unavailable_iron,
                                flux_max_pct, unavailable_flux, OUT, baseline_flux_portion, iron_tier, "quality_relaxed",
                                flux_min_pct_override=FLUX_MIN_PCT_QUALITY_RELAXED, mandate_mode="pinned",
                                iol_nominal=iol_nominal, bf_nominal=bf_nominal,
                                iol_fb_min=iol_fb_min, iol_fb_max=iol_fb_max,
                                bf_fb_min=bf_fb_min, bf_fb_max=bf_fb_max)
    _add_diagnostic_coke_constraints(prob1, x1, suffix="_p1")
    slacks1, sums1 = build_soft_vars_and_constraints(prob1, x1, df, OUT, targets, fe_lo, fe_hi, suffix="_p1")
    obj1 = weighted_deviation_expr(slacks1, targets, OUT)
    prob1 += obj1, "Total_Weighted_Deviation"
    prob1.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[prob1.status] != "Optimal":
        diagnostics.append("❌ No feasible blend even in diagnostic mode. Check inventory.")
        return "Production_Risk", None, None, None, diagnostics, True

    blend1 = {m: round(x1[m].value(), 2) for m in x1}
    achieved1 = compute_achieved(blend1, df, OUT)
    phase1_slack_values = {k: (v.value() or 0.0) for k, v in slacks1.items()}
    dev_report = {
        "Fe": phase1_slack_values["Fe_under"] + phase1_slack_values["Fe_over"],
        "SiO2": phase1_slack_values["SiO2_over"],
        "Al2O3": phase1_slack_values["Al2O3_over"],
        "Al2O3/SiO2": phase1_slack_values["ratio_over"],
        "Basicity": phase1_slack_values["Bas_under"] + phase1_slack_values["Bas_over"],
        "MgO": phase1_slack_values["MgO_under"] + phase1_slack_values["MgO_over"],
        "CaO": phase1_slack_values["CaO_under"] + phase1_slack_values["CaO_over"],
    }
    binding_spec = max(dev_report, key=dev_report.get)
    diagnostics.append(f"📊 Most binding spec: {binding_spec} (deviation = {dev_report[binding_spec]:.3f})")
    diagnostics.append("🏃 Phase 2: minimising cost at same deviation...")

    prob2 = pulp.LpProblem("Phase2_MinCost", pulp.LpMinimize)
    x2 = {m: pulp.LpVariable(f"x2_{m}", lowBound=bounds[m][0], upBound=bounds[m][1]) for m in df.index}
    _apply_fixed_quantities(prob2, x2, suffix="_p2")
    add_structural_constraints(prob2, x2, df, bounds, iron_ores, fluxes, iron_ore_max_pct, unavailable_iron,
                                flux_max_pct, unavailable_flux, OUT, baseline_flux_portion, iron_tier, "quality_relaxed",
                                flux_min_pct_override=FLUX_MIN_PCT_QUALITY_RELAXED, mandate_mode="pinned",
                                iol_nominal=iol_nominal, bf_nominal=bf_nominal,
                                iol_fb_min=iol_fb_min, iol_fb_max=iol_fb_max,
                                bf_fb_min=bf_fb_min, bf_fb_max=bf_fb_max)
    _add_diagnostic_coke_constraints(prob2, x2, suffix="_p2")
    slacks2, sums2 = build_soft_vars_and_constraints(prob2, x2, df, OUT, targets, fe_lo, fe_hi, suffix="_p2")
    for key, var in slacks2.items():
        p1_val = phase1_slack_values.get(key, 0.0)
        prob2 += var <= p1_val + PIN_TOLERANCE, f"Pin_{key}"
    prob2 += pulp.lpSum(x2[m] * df.loc[m, "Price_Rs_t"] / 1000 for m in x2), "Total_Cost"
    prob2.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[prob2.status] == "Optimal":
        blend2 = {m: round(x2[m].value(), 2) for m in x2}
        cost2 = sum(blend2[m] * df.loc[m, "Price_Rs_t"] / 1000 for m in blend2)
        achieved2 = compute_achieved(blend2, df, OUT)
        diagnostics.append(f"✅ Phase 2 complete. Reference cost: Rs {cost2:,.2f}/t")
        return "Production_Risk", blend2, cost2, achieved2, diagnostics, True
    else:
        diagnostics.append("⚠️ Phase 2 issue – returning Phase 1 blend.")
        return "Production_Risk", blend1, None, achieved1, diagnostics, True

# ============================================================================
# PRACTICAL MANUAL SCENARIO
# ============================================================================
def solve_manual_scenario(df, production_tonnes, targets, baseline_blend, fixed_quantities, **kwargs):
    """Re-optimize around user-fixed material quantities.

    The optimizer baseline is never modified. User changes become hard
    constraints and every other eligible material is re-optimized subject to
    availability, chemistry, burden mandates, coke/thermal limits and cost.
    """
    return solve_blend_with_compensation(
        df.copy(), production_tonnes, targets, baseline_blend=baseline_blend,
        fixed_quantities=fixed_quantities, **kwargs
    )

# ============================================================================
# 11. BASELINE BLEND WRAPPER
# ============================================================================
def get_baseline_blend(df, targets, enforce_b4=False):
    result = solve_blend_with_compensation(df, 1000, targets, baseline_blend=None, enforce_b4=enforce_b4)
    status, blend, cost, achieved = result[0], result[1], result[2], result[3]
    if status in ("Optimal", "Recovery", "Production_Risk") and blend is not None:
        return blend, cost, achieved
    return None, None, None

# ============================================================================
# 12. DISPLAY RESULTS (with table-based manual burden control, coke included)
# ============================================================================
def display_results_with_adjustment(status, blend, cost, achieved, df, targets, diagnostics=None,
                                     is_fallback=False, summary_only=False, om_cost=DEFAULT_OM_COST_RS_T,
                                     coke_cv=DEFAULT_COKE_CV_KCAL_KG,
                                     coke_fc=DEFAULT_COKE_FC_PCT,
                                     latent_heat=DEFAULT_HEAT_LATENT_MOISTURE,
                                     calcination_heat=DEFAULT_HEAT_CALCINATION_PER_LOI_KG,
                                     melting_heat=DEFAULT_HEAT_MELTING_PER_KG_SINTER,
                                     loss_fraction=DEFAULT_HEAT_LOSS_FRACTION,
                                     firing_ratio_max=DEFAULT_FIRING_RATIO_MAX,
                                     coke_min_rate=DEFAULT_COKE_MIN_KG_T,
                                     coke_max_rate=DEFAULT_COKE_MAX_KG_T,
                                     feo_min=DEFAULT_FEO_MIN_PCT,
                                     feo_target=DEFAULT_FEO_TARGET_PCT,
                                     feo_max=DEFAULT_FEO_MAX_PCT,
                                     feo_ref_surplus=DEFAULT_FEO_REFERENCE_THERMAL_SURPLUS_KCAL,
                                     feo_ref_pct=DEFAULT_FEO_REFERENCE_PCT,
                                     feo_thermal_slope=DEFAULT_FEO_THERMAL_SLOPE_PCT_PER_10K_KCAL,
                                     ref_coke_cv=DEFAULT_REFERENCE_COKE_CV_KCAL_KG,
                                     ref_coke_fc=DEFAULT_REFERENCE_COKE_FC_PCT,
                                     manual_override=False,
                                     manual_coke_rate=65.0):
    clear_output(wait=True)
    fe_lo = FE_LOWER
    fe_hi = FE_UPPER

    if status == "No_Production":
        print(" STATUS: \033[91m🚫 NO PRODUCTION POSSIBLE\033[0m")
        print("=" * 80)
        if diagnostics:
            for msg in diagnostics:
                print(msg)
        return

    if achieved is None:
        print(" STATUS: \033[91m🟠 PRODUCTION RISK — NO VALID BURDEN FOUND\033[0m")
        print("=" * 80)
        print("⚠️ No mathematically valid burden could be constructed with the current inventory and engineering bounds.\n")
        if diagnostics:
            print("🔍 DIAGNOSTICS:")
            for msg in diagnostics:
                print(f"   {msg}")
        return

    if summary_only and blend is not None:
        print("=" * 80)
        print("📊 EXECUTIVE SUMMARY")
        print("=" * 80)
        if status == "Optimal" and not is_fallback:
            print(" STATUS: \033[92m✅ GREEN — All quality targets met\033[0m")
        elif is_fallback:
            print(" STATUS: \033[91m🔴 RED — Quality/mandate targets NOT met even in crisis mode\033[0m")
        else:
            print(" STATUS: \033[91m🔴 RED — Quality targets not met\033[0m")
        dry_table, rm_cost_dry, total_dry = compute_dry_cost_table(blend, df, om_cost)
        wet_table, rm_cost_wet, total_wet = compute_wet_cost_table(blend, df, om_cost)
        print(f"\n💰 RM Cost (Dry): Rs {rm_cost_dry:,.2f}/t | O&M: Rs {om_cost:,.2f}/t | TOTAL (Dry): Rs {total_dry:,.2f}/t")
        print(f"💰 RM Cost (Wet/Procurement): Rs {rm_cost_wet:,.2f}/t | TOTAL (Wet): Rs {total_wet:,.2f}/t")
        print("\n📊 KEY QUALITY KPIs:")
        for key in ["Fe", "SiO2", "CaO", "MgO", "Basicity"]:
            val = achieved.get(key, 0)
            print(f"   {key}: {val:.2f}")
        if diagnostics:
            violations = [d for d in diagnostics if "⚠️" in d or "🔥" in d]
            if violations:
                print("\n⚠️ TOP DIAGNOSTIC NOTES:")
                for v in violations[:3]:
                    print(f"   {v}")
        print("\n" + "=" * 80)
        return

    print("=" * 80)
    if status == "Optimal" and not is_fallback:
        label = "✅ FEASIBLE — COST OPTIMAL"
    elif status == "Recovery":
        label = "🟡 RECOVERY — QUALITY COMPLIANT / AVAILABILITY CONSTRAINED"
    else:
        label = "🟠 PRODUCTION RISK — CLOSEST AVAILABLE REFERENCE"
    print(f" STATUS: {label}")
    print("=" * 80)

    if diagnostics:
        for msg in diagnostics:
            print(msg)
        print()

    dry_table, rm_cost_dry, total_dry = compute_dry_cost_table(blend, df, om_cost)
    wet_table, rm_cost_wet, total_wet = compute_wet_cost_table(blend, df, om_cost)

    print("📊 DRY BASIS TABLE (chemistry/quality reference — LP's native output):")
    display(HTML(dry_table.to_html(float_format="%.2f")))
    print(f"💰 RM Cost (Dry): Rs {rm_cost_dry:,.2f}/t  |  O&M: Rs {om_cost:,.2f}/t  |  TOTAL (Dry): Rs {total_dry:,.2f}/t\n")

    print("📊 WET / PROCUREMENT BASIS TABLE (as-received tonnage & real purchase cost):")
    display(HTML(wet_table.to_html(float_format="%.2f")))
    print(f"💰 RM Cost (Wet): Rs {rm_cost_wet:,.2f}/t  |  O&M: Rs {om_cost:,.2f}/t  |  TOTAL (Wet): Rs {total_wet:,.2f}/t\n")
    _print_dynamic_iron_ore_ranking(df)

    _print_kpi_table(achieved, targets, fe_lo, fe_hi, strict_mark="⚠" if not is_fallback else "❌")

    # Heat balance diagnostic (using the same coefficients)
    coke_diag = compute_coke_heat_balance_diagnostic(blend, df, 1000, coke_cv, coke_fc,
                                                     latent_heat, calcination_heat, melting_heat, loss_fraction)
    if coke_diag:
        print("\n🔥 COKE HEAT-BALANCE DIAGNOSTIC (using your editable coefficients):")
        print(f"   Coke CV: {coke_cv} kcal/kg  |  Fixed Carbon: {coke_fc}%")
        print(f"   Latent heat: {latent_heat}  |  Calcination: {calcination_heat}  |  Melting: {melting_heat}  |  Loss fraction: {loss_fraction:.3f}")
        if manual_override:
            print(f"   ⚠️ MANUAL OVERRIDE ACTIVE: Coke forced to {manual_coke_rate:.1f} kg/t")
        print(f"   LP-chosen Coke Breeze: {coke_diag['CB_kg_LP_chosen']:.1f} kg/t")
        print(f"   Q_fuel: {coke_diag['Q_fuel_kcal']:,.0f} kcal/t  |  Q_required: {coke_diag['Q_required_kcal']:,.0f} kcal/t")
        print(f"   Firing Ratio: {coke_diag['Firing_Ratio']:.3f} (max: {firing_ratio_max})")
        print(f"   Effective Coke: {coke_diag['Effective_Coke_kg_t']:.2f} kg/t")
        print(f"   Predicted FeO: {coke_diag['FeO_Estimate_Pct']:.2f}% | Target: {feo_target:.2f}% | Band: {feo_min:.2f}-{feo_max:.2f}%")
        print(f"   {coke_diag['Controller_Suggestion']}")
        print(f"   {coke_diag['note']}")

    if blend is not None and achieved is not None:
        _print_adjustment_section(blend, rm_cost_dry, achieved, df, targets)

# ============================================================================
# KPI TABLE
# ============================================================================
def _print_kpi_table(achieved, targets, fe_lo, fe_hi, strict_mark="⚠"):
    print("🧪 SINTER CHEMISTRY (vs. Targets):")
    kpi_data = {
        "KPI": ["Fe (%)", "SiO2 (%)", "Al2O3 (%)", "Al2O3/SiO2", "Basicity", "MgO (%)", "CaO (%)", "B4"],
        "Achieved": [
            f"{achieved['Fe']:.2f}", f"{achieved['SiO2']:.2f}", f"{achieved['Al2O3']:.2f}",
            f"{achieved['Al2O3/SiO2']:.3f}", f"{achieved['Basicity']:.3f}",
            f"{achieved['MgO']:.2f}", f"{achieved['CaO']:.2f}", f"{achieved['B4']:.3f}"
        ],
        "Target": [
            f"{fe_lo:.1f}-{fe_hi:.1f} (54 ±{FE_TOLERANCE:.1f})", f"<= {targets['SiO2_max']}", f"<= {targets['Al2O3_max']}",
            f"<= {targets['Al2O3_SiO2_max']}", f"{targets['Basicity_min']} - {targets['Basicity_max']}",
            f"{targets['MgO_min']} - {targets['MgO_max']}", f"{targets['CaO_min']} - {targets['CaO_max']}", "1.8 - 2.2 (info)"
        ],
        "Status": []
    }
    checks = [
        fe_lo - 0.01 <= achieved['Fe'] <= fe_hi + 0.01,
        achieved['SiO2'] <= targets['SiO2_max'] + 0.01,
        achieved['Al2O3'] <= targets['Al2O3_max'] + 0.01,
        achieved['Al2O3/SiO2'] <= targets['Al2O3_SiO2_max'] + 0.005,
        targets['Basicity_min'] - 0.01 <= achieved['Basicity'] <= targets['Basicity_max'] + 0.01,
        targets['MgO_min'] - 0.01 <= achieved['MgO'] <= targets['MgO_max'] + 0.01,
        targets['CaO_min'] - 0.01 <= achieved['CaO'] <= targets['CaO_max'] + 0.01,
    ]
    for check in checks:
        kpi_data["Status"].append("✅" if check else strict_mark)
    kpi_data["Status"].append("ℹ")
    kpi_df = pd.DataFrame(kpi_data)
    display(HTML(kpi_df.to_html(index=False)))



# ============================================================================
# ============================================================================
# 13. MANUAL BURDEN CONTROL — FULL INTERACTIVE CONTROL
# ============================================================================
def _validate_manual_blend(adjusted, df, targets):
    """Validate a manually edited burden against the same operating rules used by the model."""
    total = sum(float(adjusted.get(m, 0.0)) for m in df.index)
    if total <= 0:
        return {"valid": False, "total": total, "reasons": ["Total burden is zero."]}

    reasons = []

    # Hard chemistry checks
    achieved = compute_achieved(adjusted, df, 1000)
    if not (FE_LOWER - 0.01 <= achieved["Fe"] <= FE_UPPER + 0.01):
        reasons.append(f"Fe {achieved['Fe']:.2f}% outside {FE_LOWER:.1f}–{FE_UPPER:.1f}%")
    if achieved["SiO2"] > targets["SiO2_max"] + 0.01:
        reasons.append(f"SiO2 {achieved['SiO2']:.2f}% > {targets['SiO2_max']:.2f}%")
    if achieved["Al2O3"] > targets["Al2O3_max"] + 0.01:
        reasons.append(f"Al2O3 {achieved['Al2O3']:.2f}% > {targets['Al2O3_max']:.2f}%")
    if achieved["Al2O3/SiO2"] > targets["Al2O3_SiO2_max"] + 0.005:
        reasons.append(
            f"Al2O3/SiO2 {achieved['Al2O3/SiO2']:.3f} > {targets['Al2O3_SiO2_max']:.3f}"
        )
    if not (targets["Basicity_min"] - 0.01 <= achieved["Basicity"] <= targets["Basicity_max"] + 0.01):
        reasons.append(
            f"Basicity {achieved['Basicity']:.3f} outside "
            f"{targets['Basicity_min']:.2f}–{targets['Basicity_max']:.2f}"
        )
    if not (targets["MgO_min"] - 0.01 <= achieved["MgO"] <= targets["MgO_max"] + 0.01):
        reasons.append(f"MgO {achieved['MgO']:.2f}% outside {targets['MgO_min']:.2f}–{targets['MgO_max']:.2f}%")
    if not (targets["CaO_min"] - 0.01 <= achieved["CaO"] <= targets["CaO_max"] + 0.01):
        reasons.append(f"CaO {achieved['CaO']:.2f}% outside {targets['CaO_min']:.2f}–{targets['CaO_max']:.2f}%")

    # Availability / technical bounds
    for m in df.index:
        v = float(adjusted.get(m, 0.0))
        if v < -1e-6:
            reasons.append(f"{m}: negative burden")
        tech_min = float(df.loc[m, "Tech_Min"])
        tech_max = float(df.loc[m, "Tech_Max"])
        if v > 1e-6 and (df.loc[m, "Available_Tonnes"] <= 0 or tech_max <= 0):
            reasons.append(f"{m}: material is unavailable but manual burden is > 0")
        if v + 1e-6 < tech_min and v > 1e-6:
            reasons.append(f"{m}: {v:.2f} kg/t below Tech Min {tech_min:.2f}")
        if v - 1e-6 > tech_max:
            reasons.append(f"{m}: {v:.2f} kg/t above Tech Max {tech_max:.2f}")

    # Strict IOL/BF burden mandates
    for mat, nominal in [("IOL_Fines", IOL_FINES_NOMINAL_PCT), ("BF_Returns", BF_RETURNS_NOMINAL_PCT)]:
        if mat in df.index:
            actual_pct = float(adjusted.get(mat, 0.0)) / total
            if abs(actual_pct - nominal) > 0.0005:
                reasons.append(
                    f"{mat}: {actual_pct*100:.2f}% of burden; strict target is {nominal*100:.2f}%"
                )

    # Mill Scale 5–15% when available
    if "MILL_SCALE" in df.index:
        ms = float(adjusted.get("MILL_SCALE", 0.0))
        if df.loc["MILL_SCALE", "Available_Tonnes"] > 0 and df.loc["MILL_SCALE", "Tech_Max"] > 0:
            ms_pct = ms / total
            if ms_pct < MILL_SCALE_MIN_BURDEN_PCT - 0.0005 or ms_pct > MILL_SCALE_MAX_BURDEN_PCT + 0.0005:
                reasons.append(f"MILL_SCALE: {ms_pct*100:.2f}% of burden; required 5–15% when available")
        elif ms > 0:
            reasons.append("MILL_SCALE: unavailable but manual burden is > 0")

    return {
        "valid": len(reasons) == 0,
        "total": total,
        "achieved": achieved,
        "reasons": reasons
    }


def _print_adjustment_section(blend, cost, achieved, df, targets):
    """
    Interactive manual burden control.

    The user can change any non-mandated material in kg/t or %. The table:
      1. shows the optimized baseline,
      2. calculates the requested final burden,
      3. automatically balances unchanged donor materials to keep total burden
         at the optimized total,
      4. validates chemistry, availability, technical limits, IOL/BF mandates
         and Mill Scale 5–15% rule,
      5. provides Apply / Reset / Re-optimize controls.
    """
    OUT = 1000
    materials = [m for m in df.index if m in blend]
    baseline = {m: float(blend.get(m, 0.0)) for m in materials}
    baseline_total = sum(baseline.values()) or 1.0

    print("\n" + "=" * 125)
    print("🔧 MANUAL BURDEN CONTROL — EDIT → VALIDATE → APPLY")
    print("=" * 125)
    print("• Change is relative to the optimized baseline.")
    print("• Choose kg/t or % change for each material.")
    print("• IOL Fines and BF Returns are locked because their 8% / 17% burden mandates are strict.")
    print("• Total burden is kept at the optimized total by proportionally adjusting unchanged donor materials.")
    print("• APPLY MANUAL BURDEN does not silently accept a bad chemistry — it shows the exact violations.")

    cols = [
        "Material", "Optimized kg/t", "Optimized %", "Mode", "Change",
        "Final kg/t", "Final %", "Validation"
    ]
    grid = widgets.GridspecLayout(len(materials) + 1, len(cols), width="100%")
    for j, c in enumerate(cols):
        grid[0, j] = widgets.HTML(f"<b>{c}</b>")

    modes, changes = {}, {}
    locked_materials = {"IOL_Fines", "BF_Returns"}

    for i, mat in enumerate(materials, start=1):
        base = baseline[mat]
        grid[i, 0] = widgets.Label(mat)
        grid[i, 1] = widgets.Label(f"{base:.2f}")
        grid[i, 2] = widgets.Label(f"{base/baseline_total*100:.2f}%")

        mode = widgets.Dropdown(
            options=["kg/t", "%"], value="kg/t",
            disabled=(mat in locked_materials),
            layout=widgets.Layout(width="100%")
        )
        change = widgets.FloatText(
            value=0.0, step=0.5,
            disabled=(mat in locked_materials),
            layout=widgets.Layout(width="100%")
        )
        modes[mat], changes[mat] = mode, change
        grid[i, 3] = mode
        grid[i, 4] = change
        grid[i, 5] = widgets.Label(f"{base:.2f}")
        grid[i, 6] = widgets.Label(f"{base/baseline_total*100:.2f}%")
        grid[i, 7] = widgets.Label("🔒 MANDATED" if mat in locked_materials else "—")

    auto_balance = widgets.Checkbox(
        value=True,
        description="Auto-balance unchanged materials to preserve total burden",
        indent=False
    )
    apply_btn = widgets.Button(
        description="✅ APPLY MANUAL BURDEN",
        button_style="success",
        layout=widgets.Layout(width="220px", height="36px")
    )
    reset_btn = widgets.Button(
        description="↩ RESET TO OPTIMIZED",
        button_style="warning",
        layout=widgets.Layout(width="190px", height="36px")
    )
    reopt_btn = widgets.Button(
        description="🚀 RE-OPTIMIZE FROM CURRENT DATA",
        button_style="info",
        layout=widgets.Layout(width="250px", height="36px")
    )
    result_out = widgets.Output()
    state = {"adjusted": baseline.copy()}

    def calculate_adjusted():
        requested = {}
        changed = []

        for m in materials:
            if m in locked_materials:
                requested[m] = baseline[m]
                continue

            if modes[m].value == "%":
                requested[m] = max(0.0, baseline[m] * (1.0 + changes[m].value / 100.0))
            else:
                requested[m] = max(0.0, baseline[m] + changes[m].value)

            if abs(requested[m] - baseline[m]) > 1e-9:
                changed.append(m)

        adjusted = requested.copy()

        if auto_balance.value:
            delta = sum(adjusted.values()) - baseline_total
            donors = [
                m for m in materials
                if m not in changed and m not in locked_materials and baseline[m] > 0
            ]
            donor_total = sum(adjusted[m] for m in donors)
            if abs(delta) > 1e-9 and donor_total > 0:
                for m in donors:
                    adjusted[m] = max(
                        0.0,
                        adjusted[m] - delta * adjusted[m] / donor_total
                    )

        return adjusted

    def refresh(_=None):
        adjusted = calculate_adjusted()
        total = sum(adjusted.values()) or 1.0
        validation = _validate_manual_blend(adjusted, df, targets)

        for i, mat in enumerate(materials, start=1):
            final = adjusted[mat]
            grid[i, 5].value = f"{final:.2f}"
            grid[i, 6].value = f"{final/total*100:.2f}%"

            if mat in locked_materials:
                status = "🔒 LOCKED"
            elif final <= 1e-9:
                status = "—"
            else:
                status = "🟢"
            grid[i, 7].value = status

        with result_out:
            clear_output(wait=True)
            print(f"📦 Total Burden: {total:.2f} kg/t")
            print(
                f"🧪 Fe {validation.get('achieved', {}).get('Fe', 0):.2f}% | "
                f"SiO₂ {validation.get('achieved', {}).get('SiO2', 0):.2f}% | "
                f"Al₂O₃ {validation.get('achieved', {}).get('Al2O3', 0):.2f}% | "
                f"CaO {validation.get('achieved', {}).get('CaO', 0):.2f}% | "
                f"MgO {validation.get('achieved', {}).get('MgO', 0):.2f}% | "
                f"Basicity {validation.get('achieved', {}).get('Basicity', 0):.3f}"
            )
            if validation["valid"]:
                print("✅ CURRENT MANUAL SCENARIO IS COMPLIANT")
            else:
                print("⚠️ CURRENT MANUAL SCENARIO HAS VIOLATIONS:")
                for reason in validation["reasons"]:
                    print(" •", reason)

            dry_table, rm_cost_dry, total_dry = compute_dry_cost_table(adjusted, df, DEFAULT_OM_COST_RS_T)
            print(f"💰 Manual RM cost: Rs {rm_cost_dry:,.2f}/t")
            print(f"💰 Manual RM + default O&M: Rs {total_dry:,.2f}/t")

    def apply_manual(_):
        adjusted = calculate_adjusted()
        validation = _validate_manual_blend(adjusted, df, targets)
        state["adjusted"] = adjusted

        with result_out:
            clear_output(wait=True)
            print("📌 MANUAL BURDEN RESULT")
            print("-" * 80)
            if validation["valid"]:
                print("✅ APPLIED — manual burden is within the current model constraints.")
            else:
                print("🔴 NOT APPLIED AS A VALID PRODUCTION BLEND.")
                print("The table remains available for scenario analysis. Fix the violations or re-optimize.")
            print(f"Total burden: {validation['total']:.2f} kg/t")
            if "achieved" in validation:
                a = validation["achieved"]
                print(
                    f"Fe={a['Fe']:.2f}% | SiO₂={a['SiO2']:.2f}% | Al₂O₃={a['Al2O3']:.2f}% | "
                    f"CaO={a['CaO']:.2f}% | MgO={a['MgO']:.2f}% | Basicity={a['Basicity']:.3f}"
                )
            if validation["reasons"]:
                print("\n⚠️ Violations:")
                for reason in validation["reasons"]:
                    print(" •", reason)

    def reset(_):
        for m in materials:
            modes[m].value = "kg/t"
            changes[m].value = 0.0
        state["adjusted"] = baseline.copy()
        refresh()

    def reoptimize(_):
        # Re-run the existing optimizer from the current submitted data.
        # This does not convert the manual scenario into hidden constraints.
        with result_out:
            clear_output(wait=True)
            print("🔄 Re-optimization requested.")
            print("Use the main RUN OPTIMIZER button above to execute the LP with the current input table.")
            print("The manual table itself remains a scenario/control layer.")

    for m in materials:
        modes[m].observe(refresh, names="value")
        changes[m].observe(refresh, names="value")
    auto_balance.observe(refresh, names="value")

    apply_btn.on_click(apply_manual)
    reset_btn.on_click(reset)
    reopt_btn.on_click(reoptimize)

    display(widgets.HBox([auto_balance]))
    display(grid)
    display(widgets.HBox([apply_btn, reset_btn, reopt_btn]))
    display(result_out)
    refresh()


# ============================================================================
# 14. EXCEL MASTER LOADER + IMMEDIATE EDITABLE INPUT INTERFACE
# ============================================================================
# IMPORTANT:
# - Availability is controlled ONLY in the UI.
# - The Excel file supplies chemistry, stock, technical limits and price.
# - Any Availability column in Excel is ignored.
# - Uploading Excel NEVER runs the optimizer.
MASTER_REQUIRED_COLUMNS = [
    "Material", "Group", "Material_Role",
    "Fe_%", "SiO2_%", "Al2O3_%", "CaO_%",
    "MgO_%", "LOI_%", "Moisture_%", "Tech_Min_kg_t", "Tech_Max_kg_t",
    "Available_Stock_t", "Price_Rs_t"
]

def load_master_chemistry_excel(uploaded_file):
    """Load chemistry master. Availability is deliberately NOT read from Excel."""
    if not uploaded_file:
        raise ValueError("No Excel file was uploaded.")

    file_name = next(iter(uploaded_file))
    raw = pd.read_excel(
        BytesIO(uploaded_file[file_name]),
        sheet_name="Raw_Material_Chemistry"
    )

    missing = [c for c in MASTER_REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(
            "Master Excel is missing required columns: " + ", ".join(missing)
        )

    raw = raw[MASTER_REQUIRED_COLUMNS].copy()
    raw["Material"] = raw["Material"].astype(str).str.strip()
    raw = raw[raw["Material"].ne("")].copy()

    numeric_cols = [c for c in MASTER_REQUIRED_COLUMNS if c not in ["Material", "Group", "Material_Role"]]
    for c in numeric_cols:
        raw[c] = pd.to_numeric(raw[c], errors="coerce").fillna(0.0)

    raw["Group"] = raw["Group"].astype(str).str.strip()
    raw["Material_Role"] = raw["Material_Role"].astype(str).str.strip()

    # Excel availability, if present in older masters, is intentionally ignored.
    # Every material starts as Available when stock > 0; the user can override
    # this using the UI toggle.
    raw["Available_Tonnes"] = raw["Available_Stock_t"]
    raw["Price_Rs_t"] = raw["Price_Rs_t"]
    raw["Tech_Min"] = raw["Tech_Min_kg_t"]
    raw["Tech_Max"] = raw["Tech_Max_kg_t"]
    raw["Moisture_Pct"] = raw["Moisture_%"]
    raw["Fe"] = raw["Fe_%"]
    raw["SiO2"] = raw["SiO2_%"]
    raw["Al2O3"] = raw["Al2O3_%"]
    raw["CaO"] = raw["CaO_%"]
    raw["MgO"] = raw["MgO_%"]
    raw["LOI"] = raw["LOI_%"]

    if raw["Material"].duplicated().any():
        dupes = sorted(raw.loc[raw["Material"].duplicated(keep=False), "Material"].unique())
        raise ValueError(f"Duplicate Material names in master: {dupes}")

    # Preserve fixed recycle rates.
    for mat in raw.loc[raw["Group"].eq("Recycle"), "Material"]:
        fixed = float(raw.loc[raw["Material"].eq(mat), "Tech_Min"].iloc[0])
        raw.loc[raw["Material"].eq(mat), "Tech_Max"] = fixed

    df = raw.set_index("Material")
    keep = [
        "Group", "Material_Role", "Fe", "SiO2", "Al2O3", "CaO", "MgO", "LOI",
        "Tech_Min", "Tech_Max", "Available_Tonnes", "Price_Rs_t",
        "Moisture_Pct"
    ]
    return df[keep]


def build_input_editor(initial_df, on_apply):
    """Editable raw-material master shown immediately after Excel upload.

    Uses ordinary HBox/VBox widgets instead of a very wide GridspecLayout,
    because wide Gridspec tables can disappear/clamp in Google Colab.
    Availability is controlled ONLY by the UI toggle.
    """
    df0 = initial_df.copy()
    output = widgets.Output()
    rows = {}

    headers = [
        ("Material / Dealer", 150), ("Group", 125), ("Role", 150), ("Fe %", 70),
        ("SiO2 %", 70), ("Al2O3 %", 75), ("CaO %", 70), ("MgO %", 70),
        ("LOI %", 70), ("Moisture %", 80), ("Tech Min", 80),
        ("Tech Max", 80), ("Stock t", 85), ("Price Rs/t", 90),
        ("Availability", 125)
    ]

    def cell_label(text, width):
        return widgets.HTML(
            f"<div style='font-weight:700;width:{width}px;white-space:nowrap'>{text}</div>"
        )

    header = widgets.HBox(
        [cell_label(text, width) for text, width in headers],
        layout=widgets.Layout(width="1550px", padding="4px")
    )
    table_rows = [header]

    def fnum(v, width, step=0.1):
        return widgets.FloatText(
            value=float(v), step=step,
            layout=widgets.Layout(width=f"{width}px")
        )

    for mat in df0.index:
        rw = {}
        rw["Material"] = widgets.Text(value=str(mat), layout=widgets.Layout(width="150px"))
        rw["Group"] = widgets.Dropdown(
            options=["Iron_ore", "Flux", "Recycle", "IOL_Fines_Mandate",
                     "BF_Returns_Mandate", "Fuel"],
            value=str(df0.loc[mat, "Group"]),
            layout=widgets.Layout(width="125px")
        )
        rw["Material_Role"] = widgets.Dropdown(
            options=["Primary_Iron_Ore", "Alternative_Iron_Ore", "Other"],
            value=str(df0.loc[mat, "Material_Role"]),
            layout=widgets.Layout(width="150px")
        )
        for c, width in [("Fe",70),("SiO2",70),("Al2O3",75),("CaO",70),
                         ("MgO",70),("LOI",70),("Moisture_Pct",80)]:
            rw[c] = fnum(df0.loc[mat, c], width)
        rw["Tech_Min"] = fnum(df0.loc[mat, "Tech_Min"], 80, 1)
        rw["Tech_Max"] = fnum(df0.loc[mat, "Tech_Max"], 80, 1)
        rw["Available_Tonnes"] = widgets.FloatText(
            value=float(df0.loc[mat, "Available_Tonnes"]), step=100,
            layout=widgets.Layout(width="85px")
        )
        rw["Price_Rs_t"] = widgets.FloatText(
            value=float(df0.loc[mat, "Price_Rs_t"]), step=50,
            layout=widgets.Layout(width="90px")
        )
        available = float(df0.loc[mat, "Available_Tonnes"]) > 0
        rw["Available"] = widgets.ToggleButton(
            value=available,
            description="🟢 AVAILABLE" if available else "🔴 UNAVAILABLE",
            button_style="success" if available else "danger",
            layout=widgets.Layout(width="125px", height="30px")
        )

        def toggle(change, rw=rw):
            rw["Available"].description = "🟢 AVAILABLE" if change["new"] else "🔴 UNAVAILABLE"
            rw["Available"].button_style = "success" if change["new"] else "danger"
        rw["Available"].observe(toggle, names="value")
        rows[mat] = rw

        row = widgets.HBox(
            [rw["Material"], rw["Group"], rw["Fe"], rw["SiO2"], rw["Al2O3"],
             rw["CaO"], rw["MgO"], rw["LOI"], rw["Moisture_Pct"], rw["Tech_Min"],
             rw["Tech_Max"], rw["Available_Tonnes"], rw["Price_Rs_t"], rw["Available"]],
            layout=widgets.Layout(width="1400px", padding="2px")
        )
        table_rows.append(row)

    scroll = widgets.Box(
        [widgets.VBox(table_rows)],
        layout=widgets.Layout(width="100%", max_height="520px", overflow_x="auto", overflow_y="auto", border="1px solid #888")
    )

    apply_btn = widgets.Button(
        description="✅ APPLY INPUTS & OPEN OPTIMIZER",
        button_style="success",
        layout=widgets.Layout(width="320px", height="42px")
    )

    def collect(_=None):
        records, names, errors = [], set(), []
        for old_mat, rw in rows.items():
            name = sanitize_material_name(rw["Material"].value)
            if not name:
                errors.append("Blank material name found.")
                continue
            if name in names:
                errors.append(f"Duplicate material name: {name}")
                continue
            names.add(name)
            rec = {"Material": name}
            for c in ["Group","Material_Role","Fe","SiO2","Al2O3","CaO","MgO","LOI","Moisture_Pct",
                      "Tech_Min","Tech_Max","Available_Tonnes","Price_Rs_t"]:
                rec[c] = rw[c].value
            if not rw["Available"].value:
                rec["Available_Tonnes"] = 0.0
            if rec["Tech_Max"] < rec["Tech_Min"]:
                errors.append(f"{name}: Tech Max < Tech Min")
            if rec["Available_Tonnes"] < 0:
                errors.append(f"{name}: negative stock")
            if rec["Price_Rs_t"] < 0:
                errors.append(f"{name}: negative price")
            if rec["Moisture_Pct"] < 0 or rec["Moisture_Pct"] >= 100:
                errors.append(f"{name}: Moisture must be >=0 and <100%")
            records.append(rec)

        if errors:
            with output:
                clear_output(wait=True)
                print("❌ INPUT ERRORS — fix these before opening the optimizer:")
                for e in errors:
                    print(" •", e)
            return

        submitted = pd.DataFrame(records).set_index("Material")
        for mat in submitted[submitted["Group"].eq("Recycle")].index:
            submitted.loc[mat, "Tech_Max"] = submitted.loc[mat, "Tech_Min"]

        with output:
            clear_output(wait=True)
            print(f"✅ {len(submitted)} materials captured from the editable interface.")
            print("🟢/🔴 Availability is controlled by the interface only.")
            print("⛔ Optimizer has NOT run yet.")
        on_apply(submitted)

    apply_btn.on_click(collect)

    title = widgets.HTML("<h3>📥 RAW MATERIAL INPUTS — EDITABLE</h3>")
    help_text = widgets.HTML(
        "Upload your chemistry master and the complete input table appears here. "
        "Edit chemistry, dealer name, stock, price and limits. Use the availability toggle."
    )
    availability_help = widgets.HTML("🟢 Available = optimizer may use it &nbsp;&nbsp; 🔴 Unavailable = optimizer must use 0")
    return widgets.VBox([title, help_text, widgets.HBox([apply_btn]), availability_help, scroll, output],
                        layout=widgets.Layout(width="100%"))


# ============================================================================
# 15. OPTIMIZER CONTROL UI
# ============================================================================
def build_commercial_ui(chem_df):
    chem_df = _ensure_material_role(chem_df)
    """Full optimizer control panel.

    Important UI behaviour:
      - Chemistry was already captured in the editable input table.
      - Availability is still visible and editable HERE as well.
      - Stock, price, moisture and Tech Max can be updated before running.
      - RUN OPTIMIZER is the only control that executes the LP.
      - SUMMARY / WHAT-IF never silently execute an optimizer before RUN.
    """
    print("=" * 110)
    print("🛠 OPTIMIZER CONTROLS — AVAILABILITY + COMMERCIAL INPUTS + LP")
    print("=" * 110)
    print(f"📊 Fe target: {FE_TARGET}%, HARD band {FE_LOWER:.1f}-{FE_UPPER:.1f}%")
    print("🔒 IOL Fines = 8% | BF Returns = 17% — strict burden mandates")
    print("🔒 Availability is controlled by the interface — not Excel")
    print("⛔ The LP runs ONLY when RUN OPTIMIZER is clicked")

    # ------------------------------------------------------------------
    # MATERIAL AVAILABILITY / COMMERCIAL CONTROL
    # ------------------------------------------------------------------
    print("\n📦 MATERIAL AVAILABILITY & COMMERCIAL CONTROL")
    print("These are the final values used when RUN OPTIMIZER is clicked.")

    material_widgets = {}
    def cell_label(text, width):
        return widgets.HTML(f"<div style='font-weight:700;width:{width}px;white-space:nowrap'>{text}</div>")
    headers2 = [("Material",150),("Group",125),("Role",150),("Fe %",70),("Stock t",85),
                ("Price Rs/t",90),("Tech Max",80),("Moisture %",80),("Availability",125)]
    header2 = widgets.HBox([cell_label(t,w) for t,w in headers2], layout=widgets.Layout(width="900px"))
    rows2 = [header2]

    for mat in chem_df.index:
        rw = {}
        rw["Stock"] = widgets.FloatText(value=float(chem_df.loc[mat,"Available_Tonnes"]), step=100, layout=widgets.Layout(width="85px"))
        rw["Price"] = widgets.FloatText(value=float(chem_df.loc[mat,"Price_Rs_t"]), step=50, layout=widgets.Layout(width="90px"))
        rw["TechMax"] = widgets.FloatText(value=float(chem_df.loc[mat,"Tech_Max"]), step=1, layout=widgets.Layout(width="80px"))
        rw["Moisture"] = widgets.FloatText(value=float(chem_df.loc[mat,"Moisture_Pct"]), step=0.1, layout=widgets.Layout(width="80px"))
        avail = float(chem_df.loc[mat,"Available_Tonnes"]) > 0
        rw["Available"] = widgets.ToggleButton(value=avail, description="🟢 AVAILABLE" if avail else "🔴 UNAVAILABLE",
                                                  button_style="success" if avail else "danger", layout=widgets.Layout(width="125px",height="30px"))
        def av_toggle(change, rw=rw):
            rw["Available"].description = "🟢 AVAILABLE" if change["new"] else "🔴 UNAVAILABLE"
            rw["Available"].button_style = "success" if change["new"] else "danger"
        rw["Available"].observe(av_toggle, names="value")
        material_widgets[mat]=rw
        rows2.append(widgets.HBox([
            widgets.Label(str(mat), layout=widgets.Layout(width="150px")),
            widgets.Label(str(chem_df.loc[mat,"Group"]), layout=widgets.Layout(width="125px")),
            widgets.Label(str(chem_df.loc[mat,"Material_Role"]), layout=widgets.Layout(width="150px")),
            widgets.Label(f"{float(chem_df.loc[mat,'Fe']):.2f}", layout=widgets.Layout(width="70px")),
            rw["Stock"],rw["Price"],rw["TechMax"],rw["Moisture"],rw["Available"]
        ], layout=widgets.Layout(width="900px", padding="2px")))

    display(widgets.HTML("<b>🟢/🔴 Availability is controlled here — NOT from Excel.</b>"))
    display(widgets.Box([widgets.VBox(rows2)], layout=widgets.Layout(width="100%", max_height="430px", overflow_x="auto", overflow_y="auto", border="1px solid #888")))
    # ------------------------------------------------------------------
    # RUN BUTTON — visible before the long Coke/FeO parameter section
    # ------------------------------------------------------------------
    run_btn = widgets.Button(description="🚀 RUN OPTIMIZER", button_style="success",
                             layout=widgets.Layout(width="260px", height="48px"))
    summary_btn = widgets.Button(description="📊 SUMMARY VIEW", button_style="warning",
                                 layout=widgets.Layout(width="190px", height="48px"))
    scenario_btn = widgets.Button(description="📊 WHAT-IF MATERIAL LOSS", button_style="info",
                                  layout=widgets.Layout(width="230px", height="48px"))
    output = widgets.Output()
    last = {"result": None}
    display(widgets.HTML("<h3>🚀 OPTIMIZER EXECUTION</h3>"))
    display(widgets.HBox([run_btn, summary_btn, scenario_btn]))
    display(widgets.HTML("<b>Nothing runs until 🚀 RUN OPTIMIZER is clicked.</b>"))
    
    # ------------------------------------------------------------------
    # COMMERCIAL / HEAT PARAMETERS
    # ------------------------------------------------------------------
    om_widget = widgets.FloatText(
        value=DEFAULT_OM_COST_RS_T, description='O&M Cost (Rs/t):', step=50,
        style={'description_width': 'initial'}, layout=widgets.Layout(width='250px')
    )
    display(widgets.HBox([om_widget]))

    print("\n🔥 COKE BREEZE OPTIMISATION PARAMETERS")
    coke_cv_widget = widgets.FloatText(value=DEFAULT_COKE_CV_KCAL_KG, description='Coke CV (kcal/kg):', step=50,
                                        style={'description_width': 'initial'}, layout=widgets.Layout(width='250px'))
    coke_fc_widget = widgets.FloatText(value=DEFAULT_COKE_FC_PCT, description='Fixed Carbon (%):', step=0.5,
                                        style={'description_width': 'initial'}, layout=widgets.Layout(width='200px'))
    latent_heat_widget = widgets.FloatText(value=DEFAULT_HEAT_LATENT_MOISTURE, description='Latent Heat:', step=10,
                                            style={'description_width': 'initial'}, layout=widgets.Layout(width='220px'))
    calc_heat_widget = widgets.FloatText(value=DEFAULT_HEAT_CALCINATION_PER_LOI_KG, description='Calcination Heat:', step=10,
                                          style={'description_width': 'initial'}, layout=widgets.Layout(width='220px'))
    melt_heat_widget = widgets.FloatText(value=DEFAULT_HEAT_MELTING_PER_KG_SINTER, description='Melting Heat:', step=5,
                                          style={'description_width': 'initial'}, layout=widgets.Layout(width='200px'))
    loss_frac_widget = widgets.FloatText(value=DEFAULT_HEAT_LOSS_FRACTION, description='Heat Loss (0-1):', step=0.01,
                                          style={'description_width': 'initial'}, layout=widgets.Layout(width='200px'))
    firing_ratio_widget = widgets.FloatText(value=DEFAULT_FIRING_RATIO_MAX, description='Firing Ratio Max:', step=0.01,
                                             style={'description_width': 'initial'}, layout=widgets.Layout(width='200px'))
    coke_min_widget = widgets.FloatText(value=DEFAULT_COKE_MIN_KG_T, description='Coke Min (kg/t):', step=1,
                                        style={'description_width': 'initial'}, layout=widgets.Layout(width='200px'))
    coke_max_widget = widgets.FloatText(value=DEFAULT_COKE_MAX_KG_T, description='Coke Max (kg/t):', step=1,
                                        style={'description_width': 'initial'}, layout=widgets.Layout(width='200px'))
    feo_min_widget = widgets.FloatText(value=DEFAULT_FEO_MIN_PCT, description='FeO Min (%):', step=0.1,
                                       style={'description_width': 'initial'}, layout=widgets.Layout(width='170px'))
    feo_target_widget = widgets.FloatText(value=DEFAULT_FEO_TARGET_PCT, description='FeO Target (%):', step=0.1,
                                           style={'description_width': 'initial'}, layout=widgets.Layout(width='180px'))
    feo_max_widget = widgets.FloatText(value=DEFAULT_FEO_MAX_PCT, description='FeO Max (%):', step=0.1,
                                       style={'description_width': 'initial'}, layout=widgets.Layout(width='170px'))
    feo_ref_surplus_widget = widgets.FloatText(value=DEFAULT_FEO_REFERENCE_THERMAL_SURPLUS_KCAL,
                                                description='FeO Ref Surplus:', step=5000,
                                                style={'description_width': 'initial'}, layout=widgets.Layout(width='220px'))
    feo_ref_pct_widget = widgets.FloatText(value=DEFAULT_FEO_REFERENCE_PCT, description='FeO Ref (%):', step=0.1,
                                            style={'description_width': 'initial'}, layout=widgets.Layout(width='170px'))
    feo_thermal_slope_widget = widgets.FloatText(value=DEFAULT_FEO_THERMAL_SLOPE_PCT_PER_10K_KCAL,
                                                  description='FeO Slope:', step=0.05,
                                                  style={'description_width': 'initial'}, layout=widgets.Layout(width='180px'))

    manual_override_widget = widgets.Checkbox(value=False, description='Override: fixed coke rate')
    manual_coke_widget = widgets.FloatSlider(value=65.0, min=0, max=200, step=0.5,
                                              description='Fixed Coke (kg/t)',
                                              style={'description_width': 'initial'},
                                              layout=widgets.Layout(width='400px'))

    def toggle_manual(change):
        manual_coke_widget.disabled = not change["new"]

    manual_override_widget.observe(toggle_manual, names="value")
    manual_coke_widget.disabled = True

    display(widgets.HTML("<b>🔥 Coke / FeO</b>"))
    display(widgets.HBox([coke_cv_widget, coke_fc_widget]))
    display(widgets.HBox([latent_heat_widget, calc_heat_widget, melt_heat_widget]))
    display(widgets.HBox([loss_frac_widget, firing_ratio_widget]))
    display(widgets.HBox([coke_min_widget, coke_max_widget]))
    display(widgets.HBox([feo_min_widget, feo_target_widget, feo_max_widget]))
    display(widgets.HBox([feo_ref_surplus_widget, feo_ref_pct_widget, feo_thermal_slope_widget]))
    display(widgets.HBox([manual_override_widget, manual_coke_widget]))

    print("\n📋 STRICT MANDATES")
    iol_nom_widget = widgets.FloatText(value=IOL_FINES_NOMINAL_PCT * 100, description='IOL %:',
                                        disabled=True, layout=widgets.Layout(width='130px'))
    bf_nom_widget = widgets.FloatText(value=BF_RETURNS_NOMINAL_PCT * 100, description='BF %:',
                                      disabled=True, layout=widgets.Layout(width='130px'))
    display(widgets.HBox([iol_nom_widget, bf_nom_widget]))

    # RUN / SUMMARY / WHAT-IF controls were created above the Coke/FeO section.

    def get_df():
        df = chem_df.copy()
        for mat, rw in material_widgets.items():
            df.loc[mat, "Price_Rs_t"] = float(rw["Price"].value)
            df.loc[mat, "Tech_Max"] = float(rw["TechMax"].value)
            df.loc[mat, "Moisture_Pct"] = float(rw["Moisture"].value)
            stock = float(rw["Stock"].value)
            df.loc[mat, "Available_Tonnes"] = stock if rw["Available"].value else 0.0

        for mat in df[df["Group"] == "Recycle"].index:
            fixed = df.loc[mat, "Tech_Min"]
            df.loc[mat, "Tech_Max"] = fixed
        return df

    def solve_current():
        df = get_df()
        return solve_blend_with_compensation(
            df, 1000, targets,
            iol_nominal=iol_nom_widget.value / 100,
            bf_nominal=bf_nom_widget.value / 100,
            coke_cv=coke_cv_widget.value,
            coke_fc=coke_fc_widget.value,
            latent_heat=latent_heat_widget.value,
            calcination_heat=calc_heat_widget.value,
            melting_heat=melt_heat_widget.value,
            loss_fraction=loss_frac_widget.value,
            firing_ratio_max=firing_ratio_widget.value,
            coke_min_rate=coke_min_widget.value,
            coke_max_rate=coke_max_widget.value,
            feo_min=feo_min_widget.value,
            feo_target=feo_target_widget.value,
            feo_max=feo_max_widget.value,
            feo_ref_surplus=feo_ref_surplus_widget.value,
            feo_ref_pct=feo_ref_pct_widget.value,
            feo_thermal_slope=feo_thermal_slope_widget.value,
            manual_override=manual_override_widget.value,
            manual_coke_rate=manual_coke_widget.value
        )

    def show_full(result):
        status, blend, cost, achieved, diagnostics, is_fallback = result
        display_results_with_adjustment(
            status, blend, cost, achieved, get_df(), targets, diagnostics, is_fallback,
            summary_only=False, om_cost=om_widget.value,
            coke_cv=coke_cv_widget.value, coke_fc=coke_fc_widget.value,
            latent_heat=latent_heat_widget.value, calcination_heat=calc_heat_widget.value,
            melting_heat=melt_heat_widget.value, loss_fraction=loss_frac_widget.value,
            firing_ratio_max=firing_ratio_widget.value,
            coke_min_rate=coke_min_widget.value, coke_max_rate=coke_max_widget.value,
            feo_min=feo_min_widget.value, feo_target=feo_target_widget.value,
            feo_max=feo_max_widget.value, feo_ref_surplus=feo_ref_surplus_widget.value,
            feo_ref_pct=feo_ref_pct_widget.value,
            feo_thermal_slope=feo_thermal_slope_widget.value,
            manual_override=manual_override_widget.value,
            manual_coke_rate=manual_coke_widget.value
        )

    def on_run(_):
        with output:
            clear_output(wait=True)
            df_now = get_df()
            print("🔎 AVAILABILITY SNAPSHOT BEFORE OPTIMIZATION")
            av = df_now[["Group", "Fe", "Available_Tonnes", "Price_Rs_t"]].copy()
            av["Status"] = np.where(av["Available_Tonnes"] > 0, "🟢 AVAILABLE", "🔴 UNAVAILABLE")
            display(av)
            print("\n🚀 RUNNING OPTIMIZER WITH CURRENT EDITED INPUTS...")
            result = solve_current()
            last["result"] = result
            show_full(result)

    def on_summary(_):
        with output:
            clear_output(wait=True)
            if last["result"] is None:
                print("⚠️ Optimizer has not been run yet.")
                print("Click 🚀 RUN OPTIMIZER first.")
                return
            result = last["result"]
            status, blend, cost, achieved, diagnostics, is_fallback = result
            display_results_with_adjustment(
                status, blend, cost, achieved, get_df(), targets, diagnostics, is_fallback,
                summary_only=True, om_cost=om_widget.value,
                coke_cv=coke_cv_widget.value, coke_fc=coke_fc_widget.value,
                latent_heat=latent_heat_widget.value, calcination_heat=calc_heat_widget.value,
                melting_heat=melt_heat_widget.value, loss_fraction=loss_frac_widget.value,
                firing_ratio_max=firing_ratio_widget.value,
                coke_min_rate=coke_min_widget.value, coke_max_rate=coke_max_widget.value,
                feo_min=feo_min_widget.value, feo_target=feo_target_widget.value,
                feo_max=feo_max_widget.value, feo_ref_surplus=feo_ref_surplus_widget.value,
                feo_ref_pct=feo_ref_pct_widget.value,
                feo_thermal_slope=feo_thermal_slope_widget.value,
                manual_override=manual_override_widget.value,
                manual_coke_rate=manual_coke_widget.value
            )

    def on_scenario(_):
        with output:
            clear_output(wait=True)
            if last["result"] is None or last["result"][1] is None:
                print("⚠️ Run the optimizer first. What-if analysis uses the current optimized base case.")
                return
            base = last["result"]
            base_blend, base_cost = base[1], base[2]
            df = get_df()
            results = []
            candidates = [
                m for m in df.index
                if (
                    (m in _eligible_iron_ores(df))
                    or df.loc[m, "Group"] in ["Flux", "IOL_Fines", "IOL_Fines_Mandate", "BF_Returns", "BF_Returns_Mandate"]
                )
                and df.loc[m, "Available_Tonnes"] > 0
            ]

            for mat in candidates:
                scenario_df = df.copy()
                scenario_df.loc[mat, "Available_Tonnes"] = 0.0
                result = solve_blend_with_compensation(
                    scenario_df, 1000, targets,
                    iol_nominal=iol_nom_widget.value / 100,
                    bf_nominal=bf_nom_widget.value / 100,
                    coke_cv=coke_cv_widget.value, coke_fc=coke_fc_widget.value,
                    latent_heat=latent_heat_widget.value,
                    calcination_heat=calc_heat_widget.value,
                    melting_heat=melt_heat_widget.value,
                    loss_fraction=loss_frac_widget.value,
                    firing_ratio_max=firing_ratio_widget.value,
                    coke_min_rate=coke_min_widget.value,
                    coke_max_rate=coke_max_widget.value,
                    feo_min=feo_min_widget.value,
                    feo_target=feo_target_widget.value,
                    feo_max=feo_max_widget.value,
                    feo_ref_surplus=feo_ref_surplus_widget.value,
                    feo_ref_pct=feo_ref_pct_widget.value,
                    feo_thermal_slope=feo_thermal_slope_widget.value,
                    manual_override=manual_override_widget.value,
                    manual_coke_rate=manual_coke_widget.value
                )
                st, bl, c, ach, diag, fb = result
                if st == "No_Production":
                    results.append({"Missing": mat, "Status": "🚫 NO PRODUCTION", "Cost": "N/A"})
                elif bl is not None:
                    inc = (c - base_cost) if c is not None and base_cost is not None else None
                    results.append({
                        "Missing": mat,
                        "Status": "✅ Feasible" if st == "Optimal" and not fb else f"🟠 {st}",
                        "Cost": round(c, 2) if c is not None else "N/A",
                        "Cost Increase": round(inc, 2) if inc is not None else "N/A"
                    })
                else:
                    results.append({"Missing": mat, "Status": "❌ INFEASIBLE", "Cost": "N/A", "Cost Increase": "N/A"})

            print("📊 WHAT-IF MATERIAL LOSS RESULTS")
            display(pd.DataFrame(results))

    run_btn.on_click(on_run)
    summary_btn.on_click(on_summary)
    scenario_btn.on_click(on_scenario)

    display(widgets.HTML("<hr>"))
    display(widgets.HTML("<h3>🚀 OPTIMIZER EXECUTION</h3>"))
    display(widgets.HBox([run_btn, summary_btn, scenario_btn]))
    display(widgets.HTML("<b>RUN OPTIMIZER</b> is the only button above that executes the LP."))
    display(output)
    return get_df


# ============================================================================
# STREAMLIT DASHBOARD ANALYSIS HELPERS
# ============================================================================

def quality_checks(achieved, targets=None):
    targets = targets or TARGETS
    if not achieved:
        return {}
    return {
        "Fe": FE_LOWER <= float(achieved.get("Fe", np.nan)) <= FE_UPPER,
        "SiO2": float(achieved.get("SiO2", np.nan)) <= targets["SiO2_max"],
        "Al2O3": float(achieved.get("Al2O3", np.nan)) <= targets["Al2O3_max"],
        "Al2O3/SiO2": float(achieved.get("Al2O3/SiO2", np.nan)) <= targets["Al2O3_SiO2_max"],
        "Basicity": targets["Basicity_min"] <= float(achieved.get("Basicity", np.nan)) <= targets["Basicity_max"],
        "MgO": targets["MgO_min"] <= float(achieved.get("MgO", np.nan)) <= targets["MgO_max"],
        "CaO": targets["CaO_min"] <= float(achieved.get("CaO", np.nan)) <= targets["CaO_max"],
    }


def quality_table(achieved, targets=None):
    targets = targets or TARGETS
    checks = quality_checks(achieved, targets)
    rows = [
        ("Fe", achieved.get("Fe"), f"{FE_LOWER:.1f}–{FE_UPPER:.1f}", checks.get("Fe", False)),
        ("SiO2", achieved.get("SiO2"), f"≤ {targets['SiO2_max']:.2f}", checks.get("SiO2", False)),
        ("Al2O3", achieved.get("Al2O3"), f"≤ {targets['Al2O3_max']:.2f}", checks.get("Al2O3", False)),
        ("Al2O3/SiO2", achieved.get("Al2O3/SiO2"), f"≤ {targets['Al2O3_SiO2_max']:.2f}", checks.get("Al2O3/SiO2", False)),
        ("Basicity", achieved.get("Basicity"), f"{targets['Basicity_min']:.2f}–{targets['Basicity_max']:.2f}", checks.get("Basicity", False)),
        ("MgO", achieved.get("MgO"), f"{targets['MgO_min']:.2f}–{targets['MgO_max']:.2f}", checks.get("MgO", False)),
        ("CaO", achieved.get("CaO"), f"{targets['CaO_min']:.2f}–{targets['CaO_max']:.2f}", checks.get("CaO", False)),
    ]
    return pd.DataFrame([{"KPI": k, "Achieved": v, "Target": t, "Status": "OK" if ok else "OUT"} for k,v,t,ok in rows])


def what_if_analysis(df, targets=None):
    targets = targets or TARGETS
    rows = []
    for material in df.index:
        scenario = df.copy()
        scenario.loc[material, "Available_Tonnes"] = 0.0
        res = solve_blend_with_compensation(scenario, 1000, targets, baseline_blend=None)
        status, blend, cost, achieved = res[0], res[1], res[2], res[3]
        rows.append({
            "Missing Material": material,
            "Status": status,
            "Cost ₹/t": cost,
            "Fe %": achieved.get("Fe") if achieved else np.nan,
            "SiO2 %": achieved.get("SiO2") if achieved else np.nan,
        })
    return pd.DataFrame(rows)