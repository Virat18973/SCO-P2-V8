import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

import optimizer as opt

st.set_page_config(page_title="Sinter Burden Control", page_icon="⚙️", layout="wide", initial_sidebar_state="expanded")

# ----------------------------- CONSTANTS ---------------------------------
TARGETS = opt.TARGETS
CHEM_COLS = ["Fe", "SiO2", "Al2O3", "CaO", "MgO", "LOI", "Moisture_Pct"]
GROUPS = ["Iron_ore", "Flux", "Recycle", "Fuel"]
GROUP_LABEL = {"Iron_ore":"Iron Ore", "Flux":"Flux", "Recycle":"Recycle", "Fuel":"Fuel"}
ALT_NAME_FALLBACK = {"HAEMA", "BAUXA", "BAUXUA"}

# ----------------------------- STYLE -------------------------------------
# Same theme/palette as the original build. Only additions here are for the
# new in-page tab navigation (stTabs) introduced by the consolidated pages.
st.markdown("""
<style>
:root { --bg:#071016; --panel:#0d1a21; --panel2:#111f27; --line:#28404d; --text:#edf5fa; --muted:#8ea6b4; --accent:#2f82b3; --good:#25c481; --warn:#f2b94b; --bad:#ff5555; }
html, body, [data-testid="stAppViewContainer"] { background:var(--bg); color:var(--text); }
[data-testid="stHeader"] { background:transparent; }
[data-testid="stSidebar"] { width:220px !important; min-width:220px !important; background:#09131a; border-right:1px solid var(--line); }
[data-testid="stSidebar"] > div:first-child { width:220px !important; }
.block-container { max-width:none !important; width:100% !important; padding:1.0rem 1.1rem 2.5rem; }
.small { color:var(--muted); font-size:.72rem; }
.eyebrow { color:#73b5d7; font-size:.62rem; letter-spacing:.16em; font-weight:800; text-transform:uppercase; }
h1 { font-size:2rem !important; margin:.05rem 0 .15rem !important; letter-spacing:.01em; }
h2,h3 { letter-spacing:.01em; }
.panel { background:linear-gradient(180deg,#102029,#0d1a21); border:1px solid var(--line); border-radius:8px; padding:.65rem .7rem; margin:.55rem 0; }
.panel-title { color:#8fd0ef; font-size:.67rem; font-weight:900; letter-spacing:.1em; text-transform:uppercase; margin-bottom:.35rem; }
.notice { background:#12242e; border:1px solid #31566a; border-radius:7px; padding:.6rem .75rem; color:#cce6f3; font-size:.76rem; }
.notice-w { border-color:#795b18; background:#241e0e; }
.hero { background:#0d1a21; border:1px solid var(--line); border-radius:8px; padding:.7rem .85rem; margin:.5rem 0 .7rem; }
.kpi { background:#101d24; border:1px solid var(--line); border-radius:8px; padding:.65rem .7rem; min-height:82px; }
.kpi-label { font-size:.6rem; color:#7da4b7; letter-spacing:.11em; font-weight:900; text-transform:uppercase; }
.kpi-value { font-size:1.2rem; font-weight:900; margin-top:.22rem; }
.kpi-sub { font-size:.62rem; color:#6f8998; margin-top:.12rem; }
.kpi-g { border-left:3px solid var(--good); } .kpi-r { border-left:3px solid var(--bad); } .kpi-a { border-left:3px solid var(--warn); } .kpi-s { border-left:3px solid #4ca8df; }
.nav-title { color:#7397a9; font-size:.58rem; font-weight:900; letter-spacing:.16em; margin:.75rem 0 .3rem; }
.sidebar-brand { font-weight:900; font-size:.86rem; letter-spacing:.03em; }
div.stButton > button { border-radius:6px; border:1px solid #2b4c5c; background:#101f27; color:#edf5fa; font-size:.72rem; min-height:2rem; }
div.stButton > button:hover { border-color:#4e94bb; color:white; }
button[kind="primary"] { background:#2c78a5 !important; border-color:#3e91c1 !important; }
[data-testid="stDataEditor"] { border:1px solid var(--line); border-radius:7px; overflow:hidden; }
[data-testid="stDataEditor"] [role="gridcell"] { font-size:10.5px; }
[data-testid="stFileUploader"] { background:#111a21; border-radius:7px; padding:.25rem; }
[data-testid="stMetric"] { background:#101d24; border:1px solid var(--line); border-radius:8px; padding:.45rem; }
.footer { color:#526d7b; font-size:.58rem; text-align:right; margin-top:1rem; }
[data-testid="stDataEditor"] [role="gridcell"], [data-testid="stDataFrame"] [role="gridcell"] {font-size:11px !important;}
.dashboard-paired-table { width:100%; }
 .dashboard-paired-table [data-testid="stDataEditor"],
.dashboard-paired-table [data-testid="stDataFrame"] { width:100% !important; }
.material-chart-note { color:#8ea6b4; font-size:.68rem; margin:.15rem 0 .45rem; }

/* In-page sub-navigation (replaces several stand-alone pages with tabs) */
[data-testid="stTabs"] [data-baseweb="tab-list"] { gap:.25rem; border-bottom:1px solid var(--line); }
[data-testid="stTabs"] [data-baseweb="tab"] {
    background:#0d1a21; border:1px solid var(--line); border-bottom:none;
    border-radius:7px 7px 0 0; color:var(--muted); font-size:.74rem; font-weight:800;
    letter-spacing:.03em; padding:.5rem 1rem;
}
[data-testid="stTabs"] [aria-selected="true"] { background:#12242e; color:var(--text); border-color:var(--accent); }
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background-color:var(--accent); }
[data-testid="stTabs"] [data-baseweb="tab-panel"] { padding-top:.7rem; }
</style>
""", unsafe_allow_html=True)

# ----------------------------- STATE -------------------------------------
def _classify_role(df):
    """Normalize the v31.9 Material_Role field.

    The dashboard never uses material/dealer names to decide whether an ore is
    an alternative. If Material_Role is absent, iron ores default to
    Primary_Iron_Ore so the master remains backward compatible.
    """
    df = df.copy()
    role_col = next((c for c in df.columns if str(c).strip().lower().replace(" ", "_")
                     in {"material_role", "role", "material_type", "type", "category"}), None)
    if role_col is not None:
        vals = df[role_col].astype("string").fillna("").str.strip().str.lower()
        def role(v, group):
            if "alternative" in v or "contingency" in v or v in {"alternative_iron_ore", "alternative"}:
                return "Alternative_Iron_Ore"
            if "primary" in v or v == "primary_iron_ore":
                return "Primary_Iron_Ore"
            return "Other" if str(group) != "Iron_ore" else "Primary_Iron_Ore"
        df["Material_Role"] = [role(v, g) for v,g in zip(vals, df["Group"])]
        if role_col != "Material_Role":
            df.drop(columns=[role_col], inplace=True)
    else:
        df["Material_Role"] = np.where(
            df["Group"].astype(str).eq("Iron_ore"),
            "Primary_Iron_Ore",
            "Other"
        )
    return df

def normalize_master(raw):
    df = raw.copy()
    df.columns = [str(c).strip().replace(" ","_") for c in df.columns]
    aliases = {
        "SiO₂":"SiO2", "SiO₂_%":"SiO2", "Al₂O₃":"Al2O3", "Al₂O₃_%":"Al2O3",
        "CaO_%":"CaO", "MgO_%":"MgO", "Price_₹/t":"Price_Rs_t", "Price_₹_t":"Price_Rs_t",
        "Price":"Price_Rs_t", "RM_Stock":"Available_Tonnes", "RM_Stock_t":"Available_Tonnes",
        "Stock":"Available_Tonnes", "Tech_Max_t/d":"Tech_Max", "Tech_Max_t_d":"Tech_Max",
        "Tech_Min_t/d":"Tech_Min", "Tech_Min_t_d":"Tech_Min", "Moisture":"Moisture_Pct", "Moisture_%":"Moisture_Pct", "Moisture_Pct":"Moisture_Pct"
    }
    df.rename(columns={c:aliases.get(c,c) for c in df.columns}, inplace=True)
    if "Material" in df.columns:
        df["Material"] = df["Material"].astype(str).str.strip()
        df = df.set_index("Material")
    elif df.index.name is None:
        raise ValueError("Master Excel must contain a Material column.")
    required = ["Group", "Fe", "SiO2", "Al2O3", "CaO", "MgO", "LOI", "Tech_Min", "Tech_Max", "Price_Rs_t", "Available_Tonnes"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Master Excel missing: " + ", ".join(missing))
    if "Moisture_Pct" not in df.columns:
        df["Moisture_Pct"] = 0.0
    df = _classify_role(df)
    df["Group"] = df["Group"].astype(str).str.strip()
    for c in CHEM_COLS + ["Tech_Min","Tech_Max","Price_Rs_t","Available_Tonnes"]:
        df[c] = pd.to_numeric(df[c], errors="raise")
    for m in df[df["Group"]=="Recycle"].index:
        df.loc[m,"Tech_Max"] = df.loc[m,"Tech_Min"]
    if df.index.duplicated().any():
        raise ValueError("Duplicate material names found in the master Excel.")
    return df

def initial_df():
    df = opt.get_default_chemistry().copy()
    if "Moisture_Pct" not in df.columns:
        df["Moisture_Pct"] = 0.0
    df["Material_Role"] = np.where(df["Group"].eq("Iron_ore"), "Primary_Iron_Ore", "Other")
    return df

if "master_df" not in st.session_state:
    st.session_state.master_df = initial_df()
    st.session_state.source = "Built-in Master Chemistry"
    st.session_state.production = 1100.0
    st.session_state.available = {m: (float(st.session_state.master_df.loc[m,"Available_Tonnes"]) > 0) for m in st.session_state.master_df.index if st.session_state.master_df.loc[m,"Material_Role"] != "Alternative_Iron_Ore"}
    st.session_state.include_alt = {m: False for m in st.session_state.master_df.index if st.session_state.master_df.loc[m,"Material_Role"] == "Alternative_Iron_Ore"}
    st.session_state.result = None
    st.session_state.manual_base = None
    st.session_state.manual = None
    st.session_state.whatif = None
    st.session_state.runs = 0
    st.session_state.changed = False

def _backend_default(name, fallback):
    return float(getattr(opt, name, fallback))

if "om_cost" not in st.session_state:
    st.session_state.om_cost = _backend_default("DEFAULT_OM_COST_RS_T", 1500.0)
if "coke_cv" not in st.session_state:
    st.session_state.coke_cv = _backend_default("DEFAULT_COKE_CV_KCAL_KG", 6800.0)
if "coke_fc" not in st.session_state:
    st.session_state.coke_fc = _backend_default("DEFAULT_COKE_FC_PCT", 71.35)
if "latent_heat" not in st.session_state:
    st.session_state.latent_heat = _backend_default("DEFAULT_HEAT_LATENT_MOISTURE", 540.0)
if "calcination_heat" not in st.session_state:
    st.session_state.calcination_heat = _backend_default("DEFAULT_HEAT_CALCINATION_PER_LOI_KG", 420.0)
if "melting_heat" not in st.session_state:
    st.session_state.melting_heat = _backend_default("DEFAULT_HEAT_MELTING_PER_KG_SINTER", 60.0)
if "loss_fraction" not in st.session_state:
    st.session_state.loss_fraction = _backend_default("DEFAULT_HEAT_LOSS_FRACTION", 0.12)
if "firing_ratio_max" not in st.session_state:
    st.session_state.firing_ratio_max = _backend_default("DEFAULT_FIRING_RATIO_MAX", 1.10)
if "coke_min_rate" not in st.session_state:
    st.session_state.coke_min_rate = _backend_default("DEFAULT_COKE_MIN_KG_T", 55.0)
if "coke_max_rate" not in st.session_state:
    st.session_state.coke_max_rate = _backend_default("DEFAULT_COKE_MAX_KG_T", 85.0)
if "feo_min" not in st.session_state:
    st.session_state.feo_min = _backend_default("DEFAULT_FEO_MIN_PCT", 8.5)
if "feo_target" not in st.session_state:
    st.session_state.feo_target = _backend_default("DEFAULT_FEO_TARGET_PCT", 9.2)
if "feo_max" not in st.session_state:
    st.session_state.feo_max = _backend_default("DEFAULT_FEO_MAX_PCT", 10.0)
if "manual_coke_override" not in st.session_state:
    st.session_state.manual_coke_override = False
if "manual_coke_rate" not in st.session_state:
    st.session_state.manual_coke_rate = 65.0
if "nav" not in st.session_state: st.session_state.nav = "Dashboard"


# ------------------------- REDESIGN HELPERS -------------------------
def display_material_sequence(df):
    return [str(m) for m in df.index]

def aligned_result_table(blend, df, include_total=True):
    mats = display_material_sequence(df)
    total = float(sum(float(blend.get(m, 0.0)) for m in mats))
    total_cost = float(sum(float(blend.get(m, 0.0))*float(df.loc[m,"Price_Rs_t"])/1000 for m in mats))
    rows = []
    for m in mats:
        q = float(blend.get(m, 0.0))
        cost = q * float(df.loc[m,"Price_Rs_t"]) / 1000
        rows.append({
            "Material": m,
            "Group": GROUP_LABEL.get(df.loc[m,"Group"], df.loc[m,"Group"]),
            "kg/t": q,
            "% Burden": q/total*100 if total else 0.0,
            "Cost ₹/t": cost,
            "% Cost": cost/total_cost*100 if total_cost else 0.0
        })
    if include_total:
        rows.append({"Material":"TOTAL","Group":"","kg/t":total,"% Burden":100.0 if total else 0.0,
                     "Cost ₹/t":total_cost,"% Cost":100.0 if total_cost else 0.0})
    return pd.DataFrame(rows)

def paired_table_height(n):
    return max(320, 31*(n+1)+40)

def wet_result_table(blend, df, include_total=False):
    """Same shape as aligned_result_table but on a wet / as-received basis.

    Used for the Dashboard's Dry/Wet toggle on the optimized-output table so
    it stays row-aligned with the raw material input table.
    """
    mats = display_material_sequence(df)
    qty = {}
    for m in mats:
        q = float(blend.get(m, 0.0))
        moisture = float(df.loc[m, "Moisture_Pct"]) if "Moisture_Pct" in df.columns else 0.0
        moisture = max(0.0, min(99.9, moisture))
        qty[m] = q/(1.0-moisture/100.0) if q and moisture < 100 else q
    total = float(sum(qty.values()))
    total_cost = float(sum(qty[m]*float(df.loc[m,"Price_Rs_t"])/1000 for m in mats))
    rows = []
    for m in mats:
        q = qty[m]
        cost = q*float(df.loc[m,"Price_Rs_t"])/1000
        rows.append({
            "Material": m,
            "Group": GROUP_LABEL.get(df.loc[m,"Group"], df.loc[m,"Group"]),
            "kg/t": q,
            "% Burden": q/total*100 if total else 0.0,
            "Cost ₹/t": cost,
            "% Cost": cost/total_cost*100 if total_cost else 0.0
        })
    if include_total:
        rows.append({"Material":"TOTAL","Group":"","kg/t":total,"% Burden":100.0 if total else 0.0,
                     "Cost ₹/t":total_cost,"% Cost":100.0 if total_cost else 0.0})
    return pd.DataFrame(rows), total, total_cost

# ----------------------------- DATA HELPERS -------------------------------
def primary_names():
    return [m for m in st.session_state.master_df.index if st.session_state.master_df.loc[m,"Material_Role"]=="Primary_Iron_Ore"]
def alt_names():
    return [m for m in st.session_state.master_df.index if st.session_state.master_df.loc[m,"Material_Role"]=="Alternative_Iron_Ore"]

def editable_chemistry():
    df=st.session_state.master_df.copy()
    cols=["Material_Role","Group"]+CHEM_COLS
    view=df[cols].reset_index().rename(columns={"Material":"Material","Material_Role":"Role","Group":"Group","SiO2":"SiO₂","Al2O3":"Al₂O₃", "Moisture_Pct":"Moisture %"})
    return view

def save_chemistry(view):
    v=view.copy().set_index("Material")
    v.rename(columns={"Role":"Material_Role","SiO₂":"SiO2","Al₂O₃":"Al2O3", "Moisture %":"Moisture_Pct"}, inplace=True)
    for m in v.index:
        if m in st.session_state.master_df.index:
            for c in CHEM_COLS:
                st.session_state.master_df.loc[m,c]=float(v.loc[m,c])
    st.session_state.changed=True

def active_df():
    df=st.session_state.master_df.copy()
    keep=[]
    for m in df.index:
        role=df.loc[m,"Material_Role"]
        if role=="Alternative_Iron_Ore":
            if st.session_state.include_alt.get(m,False): keep.append(m)
        else:
            keep.append(m)
    df=df.loc[keep].copy()
    for m in df.index:
        role=df.loc[m,"Material_Role"]
        if role=="Alternative_Iron_Ore" and not st.session_state.include_alt.get(m,False):
            df.loc[m,"Available_Tonnes"]=0.0
        elif role!="Alternative_Iron_Ore" and not st.session_state.available.get(m,True):
            df.loc[m,"Available_Tonnes"]=0.0
    return df

def run_optimizer():
    df=active_df()
    kwargs={
        "coke_cv": st.session_state.coke_cv,
        "coke_fc": st.session_state.coke_fc,
        "latent_heat": st.session_state.latent_heat,
        "calcination_heat": st.session_state.calcination_heat,
        "melting_heat": st.session_state.melting_heat,
        "loss_fraction": st.session_state.loss_fraction,
        "firing_ratio_max": st.session_state.firing_ratio_max,
        "coke_min_rate": st.session_state.coke_min_rate,
        "coke_max_rate": st.session_state.coke_max_rate,
        "feo_min": st.session_state.feo_min,
        "feo_target": st.session_state.feo_target,
        "feo_max": st.session_state.feo_max,
        "manual_override": st.session_state.manual_coke_override,
        "manual_coke_rate": st.session_state.manual_coke_rate,
    }
    try:
        x=opt.solve_blend_with_compensation(df, float(st.session_state.production), TARGETS, baseline_blend=None, **kwargs)
    except TypeError:
        x=opt.solve_blend_with_compensation(df, float(st.session_state.production), TARGETS, baseline_blend=None)
    st.session_state.result={"status":x[0],"blend":x[1],"cost":x[2],"achieved":x[3],"diagnostics":x[4],"fallback":x[5],"df":df.copy()}
    st.session_state.manual_base=x[1].copy() if x[1] else None
    st.session_state.manual=x[1].copy() if x[1] else None
    st.session_state.runs += 1
    st.session_state.changed=False
    st.session_state.whatif=None

def quality_ok(a):
    if not a: return False
    checks = [
        opt.FE_LOWER <= a["Fe"] <= opt.FE_UPPER,
        a["SiO2"] <= TARGETS["SiO2_max"], a["Al2O3"] <= TARGETS["Al2O3_max"],
        a["Al2O3/SiO2"] <= TARGETS["Al2O3_SiO2_max"],
        TARGETS["Basicity_min"] <= a["Basicity"] <= TARGETS["Basicity_max"],
        TARGETS["MgO_min"] <= a["MgO"] <= TARGETS["MgO_max"],
        TARGETS["CaO_min"] <= a["CaO"] <= TARGETS["CaO_max"],
    ]
    return all(checks)

def result_table(blend,df, include_zero=False):
    rows=[]
    if not blend: return pd.DataFrame(columns=["Material","Group","kg/t","% Burden","Cost ₹/t","% Cost"])
    total=sum(float(v) for v in blend.values())
    total_cost=sum(float(q)*float(df.loc[m,"Price_Rs_t"])/1000 for m,q in blend.items())
    sequence=list(df.index)
    for m in sequence:
        q=float(blend.get(m,0))
        if not include_zero and q<=1e-8: continue
        cost=q*float(df.loc[m,"Price_Rs_t"])/1000
        rows.append({"Material":m,"Group":GROUP_LABEL.get(df.loc[m,"Group"],df.loc[m,"Group"]),"kg/t":q,"% Burden":(q/total*100 if total else 0),"Cost ₹/t":cost,"% Cost":(cost/total_cost*100 if total_cost else 0)})
    rows.append({"Material":"TOTAL","Group":"","kg/t":total,"% Burden":100.0,"Cost ₹/t":total_cost,"% Cost":100.0})
    return pd.DataFrame(rows)

def kpi(label,value,sub="",kind="s"):
    return f'<div class="kpi kpi-{kind}"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>'

def material_contribution_chart(blend, df, kind="burden", height=430):
    rows = []
    for m, q in blend.items():
        q = float(q)
        if q <= 1e-9 or m not in df.index:
            continue
        price = float(df.loc[m, "Price_Rs_t"])
        value = q if kind == "burden" else q * price / 1000
        rows.append({
            "Material": str(m),
            "Group": str(df.loc[m, "Group"]),
            "Value": value
        })

    if not rows:
        return go.Figure()

    chart_df = pd.DataFrame(rows)
    total = float(chart_df["Value"].sum())
    chart_df["Pct"] = chart_df["Value"] / total * 100 if total else 0.0
    chart_df = chart_df.sort_values("Value", ascending=True)

    group_colors = {
        "Iron_ore": "#6072f5",
        "Flux": "#f05a45",
        "Recycle": "#00c99b",
        "Fuel": "#a45bea",
    }

    fig = go.Figure()
    for group in GROUPS:
        part = chart_df[chart_df["Group"] == group]
        if part.empty:
            continue
        fig.add_trace(go.Bar(
            x=part["Value"],
            y=part["Material"],
            orientation="h",
            name=GROUP_LABEL.get(group, group),
            marker_color=group_colors.get(group, "#6f8794"),
            text=[f"{p:.1f}%" for p in part["Pct"]],
            textposition="outside",
            textfont=dict(size=11, color="#edf5fa"),
            customdata=np.column_stack([part["Pct"].to_numpy(), part["Group"].to_numpy()]),
            hovertemplate=(
                "<b>%{y}</b><br>"
                + ("Burden: %{x:.2f} kg/t" if kind == "burden" else "Cost: ₹%{x:.2f}/t")
                + "<br>Contribution: %{customdata[0]:.2f}%"
                + "<br>Group: %{customdata[1]}<extra></extra>"
            ),
        ))

    title = "MATERIAL CONTRIBUTION • % OF TOTAL BURDEN" if kind == "burden" else "MATERIAL CONTRIBUTION • % OF TOTAL COST"
    unit = "kg/t" if kind == "burden" else "₹/t"

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=max(height, 120 + 32 * len(chart_df)),
        margin=dict(l=105, r=65, t=38, b=45),
        barmode="stack",
        title=dict(text=title, x=0, xanchor="left", font=dict(size=12, color="#8fd0ef")),
        xaxis=dict(title=unit, gridcolor="#1e333f", zeroline=False,
                    tickfont=dict(size=10, color="#9db1bc"), title_font=dict(size=10, color="#8ea6b4")),
        yaxis=dict(title="", tickfont=dict(size=11, color="#edf5fa"), gridcolor="rgba(0,0,0,0)", automargin=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=1, xanchor="right", font=dict(size=9, color="#cbd8df")),
        showlegend=True,
    )
    return fig

def quality_cards(ach):
    items=[("Fe",ach.get("Fe",np.nan),f"{opt.FE_LOWER:.1f}–{opt.FE_UPPER:.1f}","%"),("SiO₂",ach.get("SiO2",np.nan),f"≤ {TARGETS['SiO2_max']}","%"),("Al₂O₃",ach.get("Al2O3",np.nan),f"≤ {TARGETS['Al2O3_max']}","%"),("Al₂O₃/SiO₂",ach.get("Al2O3/SiO2",np.nan),f"≤ {TARGETS['Al2O3_SiO2_max']}",""),("Basicity",ach.get("Basicity",np.nan),f"{TARGETS['Basicity_min']}–{TARGETS['Basicity_max']}",""),("MgO",ach.get("MgO",np.nan),f"{TARGETS['MgO_min']}–{TARGETS['MgO_max']}","%"),("CaO",ach.get("CaO",np.nan),f"{TARGETS['CaO_min']}–{TARGETS['CaO_max']}","%"),("B4",ach.get("B4",np.nan),"1.8–2.2 info","")]
    cols=st.columns(8)
    for c,(lab,val,tgt,unit) in zip(cols,items):
        c.markdown(kpi(lab,f"{val:.3f}{unit}",tgt,"g" if lab=="B4" or (lab in ["Fe","SiO₂","Al₂O₃","Al₂O₃/SiO₂","Basicity","MgO","CaO"] and quality_ok(ach)) else "a"),unsafe_allow_html=True)

def page_header(title, subtitle):
    st.markdown('<div class="eyebrow">HOSPET ALLOY STEEL PLANT</div>',unsafe_allow_html=True)
    st.markdown(f"<h1>{title}</h1><div class='small'>{subtitle}</div>",unsafe_allow_html=True)

# ----------------------------- SIDEBAR -----------------------------------
# Consolidated to 8 destinations (was 13). Pages that used to duplicate the
# same tables/charts now live together as tabs on one page instead of as
# separate sidebar entries.
with st.sidebar:
    st.markdown('<div class="sidebar-brand">HOSPET STEELS LIMITED</div><div class="small">Kalyani Steels × Mukand • Hospet</div>',unsafe_allow_html=True)
    st.markdown("---")
    nav_groups=[
        ("WORKSPACE",["Dashboard"]),
        ("OPERATIONS",["RM Stock & Materials","Recipe & Composition","Manual Burden Control"]),
        ("ANALYSIS",["Process & Cost Parameters","Scenario Analysis"]),
        ("REPORTING",["Reports"]),
        ("SYSTEM",["Upload & Settings"]),
    ]
    for head,items in nav_groups:
        st.markdown(f'<div class="nav-title">{head}</div>',unsafe_allow_html=True)
        for item in items:
            if st.button(item,key="nav_"+item,use_container_width=True,type="primary" if st.session_state.nav==item else "secondary"):
                st.session_state.nav=item; st.rerun()
    st.markdown("---")
    st.markdown(f'<div class="small"><b>DATA</b><br>{st.session_state.source}<br>{len(primary_names())} primary<br>{len(alt_names())} alternative<br><br><b>MODEL</b><br>v30 • Ready</div>',unsafe_allow_html=True)


# ----------------------------- COST / THERMAL HELPERS -----------------------
def dry_wet_tables(blend, df, om_cost):
    rows_dry=[]; rows_wet=[]
    total_dry=sum(float(blend.get(m,0.0)) for m in blend if m in df.index)
    total_wet=0.0; rm_dry=0.0; rm_wet=0.0
    for m in df.index:
        q=float(blend.get(m,0.0))
        if q < 0: q=0.0
        moisture=float(df.loc[m,"Moisture_Pct"]) if "Moisture_Pct" in df.columns else 0.0
        moisture=max(0.0,min(99.9,moisture))
        wet_q=q/(1.0-moisture/100.0) if q and moisture<100 else q
        price=float(df.loc[m,"Price_Rs_t"])
        dc=q*price/1000.0; wc=wet_q*price/1000.0
        rm_dry += dc; rm_wet += wc; total_wet += wet_q
        rows_dry.append({"Material":m,"Burden kg/t":q,"Burden %":q/total_dry*100 if total_dry else 0.0,"Cost ₹/t":dc})
        rows_wet.append({"Material":m,"Burden kg/t":wet_q,"Burden %":wet_q/total_wet*100 if total_wet else 0.0,"Cost ₹/t":wc})
    dry_total=rm_dry+float(om_cost); wet_total=rm_wet+float(om_cost)
    for rows, rm, total in [(rows_dry,rm_dry,dry_total),(rows_wet,rm_wet,wet_total)]:
        for r in rows: r["Cost %"]=r["Cost ₹/t"]/total*100 if total else 0.0
        rows.append({"Material":"O&M","Burden kg/t":np.nan,"Burden %":np.nan,"Cost ₹/t":float(om_cost),"Cost %":float(om_cost)/total*100 if total else 0.0})
        rows.append({"Material":"TOTAL","Burden kg/t":total_dry if rows is rows_dry else total_wet,"Burden %":100.0,"Cost ₹/t":total,"Cost %":100.0})
    return pd.DataFrame(rows_dry), pd.DataFrame(rows_wet), rm_dry, rm_wet, dry_total, wet_total

def coke_diagnostic(blend, df):
    fn=getattr(opt,"compute_coke_heat_balance_diagnostic",None)
    if not callable(fn) or not blend: return None
    try:
        return fn(blend,df,1000,st.session_state.coke_cv,st.session_state.coke_fc,st.session_state.latent_heat,
                  st.session_state.calcination_heat,st.session_state.melting_heat,st.session_state.loss_fraction,
                  st.session_state.feo_min,st.session_state.feo_target,st.session_state.feo_max,
                  getattr(opt,"DEFAULT_FEO_REFERENCE_THERMAL_SURPLUS_KCAL",189180.0),
                  getattr(opt,"DEFAULT_FEO_REFERENCE_PCT",8.6),
                  getattr(opt,"DEFAULT_FEO_THERMAL_SLOPE_PCT_PER_10K_KCAL",0.35),
                  getattr(opt,"DEFAULT_REFERENCE_COKE_CV_KCAL_KG",6800.0),
                  getattr(opt,"DEFAULT_REFERENCE_COKE_FC_PCT",71.35))
    except Exception:
        return None

# ----------------------------- SHARED TAB CONTENT -------------------------
# These render exactly the same tables/charts/text as the original standalone
# pages. They are now called from inside st.tabs() so the same content shows
# up once per data source instead of being rebuilt on several separate pages.

def dry_wet_composition_content(r):
    # O&M cost is editable right here too (same session_state.om_cost used on
    # Process & Cost Parameters, so edits stay in sync across both pages).
    om1, om2 = st.columns([1, 3])
    with om1:
        st.session_state.om_cost = st.number_input(
            "O&M Cost ₹/t", min_value=0.0, value=float(st.session_state.om_cost), step=50.0, key="om_cost_dry_wet"
        )
    with om2:
        st.caption("Editing O&M cost here updates it everywhere else in the app, including Process & Cost Parameters.")

    dry, wet, rm_dry, rm_wet, total_dry, total_wet = dry_wet_tables(
        r["blend"], r["df"], st.session_state.om_cost
    )
    df = r["df"]
    wet = wet.copy()
    wet.insert(2, "Moisture %", wet["Material"].map(
        lambda m: float(df.loc[m, "Moisture_Pct"]) if m in df.index and "Moisture_Pct" in df.columns else 0.0
    ))
    wet.loc[wet["Material"].isin(["O&M", "TOTAL"]), "Moisture %"] = 0.0

    dry_total_cost = float(dry.loc[dry["Material"]=="TOTAL","Cost ₹/t"].iloc[0])
    wet_burden_total = float(wet.loc[wet["Material"]=="TOTAL","Burden kg/t"].iloc[0])
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi("DRY BURDEN", f"{total_dry:,.2f} kg/t", "Optimizer / chemistry basis", "s"), unsafe_allow_html=True)
    c2.markdown(kpi("WET BURDEN", f"{wet_burden_total:,.2f} kg/t", "As-received basis", "g"), unsafe_allow_html=True)
    c3.markdown(kpi("DRY TOTAL COST", f"₹{dry_total_cost:,.2f}/t", "RM + O&M", "a"), unsafe_allow_html=True)
    c4.markdown(kpi("WET TOTAL COST", f"₹{total_wet:,.2f}/t", "RM + O&M", "a"), unsafe_allow_html=True)

    st.markdown(
        '<div class="notice"><b>DRY BASIS</b> = optimizer / chemistry basis &nbsp; • &nbsp; '
        '<b>WET BASIS</b> = as-received procurement basis &nbsp; • &nbsp; '
        'Cost includes the editable O&M cost shown as a separate row.</div>',
        unsafe_allow_html=True
    )

    a, b = st.columns(2, gap="small")
    with a:
        st.markdown('<div class="panel"><div class="panel-title">DRY BASIS • BURDEN & COST COMPOSITION</div>'
                     '<div class="small">Price ₹/t is editable — cost recalculates immediately for both dry and wet basis.</div></div>', unsafe_allow_html=True)
        dry_edit = dry.copy()
        dry_edit.insert(1, "Price ₹/t", dry_edit["Material"].map(
            lambda m: float(df.loc[m, "Price_Rs_t"]) if m in df.index else np.nan
        ))
        edited_dry = st.data_editor(
            dry_edit, hide_index=True, use_container_width=True, height=max(520, 34 * len(dry_edit) + 45),
            key="dry_basis_editor",
            disabled=["Material", "Burden kg/t", "Burden %", "Cost ₹/t", "Cost %"],
            column_config={
                "Price ₹/t": st.column_config.NumberColumn("Price ₹/t", min_value=0, step=1, format="%.0f"),
                "Burden kg/t": st.column_config.NumberColumn("Burden kg/t", format="%.2f"),
                "Burden %": st.column_config.NumberColumn("Burden %", format="%.2f"),
                "Cost ₹/t": st.column_config.NumberColumn("Cost ₹/t", format="₹ %.2f"),
                "Cost %": st.column_config.NumberColumn("Cost %", format="%.2f"),
            },
        )
        if not edited_dry["Price ₹/t"].equals(dry_edit["Price ₹/t"]):
            for _, row in edited_dry.iterrows():
                m = row["Material"]
                if m in df.index and pd.notna(row["Price ₹/t"]):
                    st.session_state.result["df"].loc[m, "Price_Rs_t"] = float(row["Price ₹/t"])
                    st.session_state.master_df.loc[m, "Price_Rs_t"] = float(row["Price ₹/t"])
            st.rerun()
        st.markdown(f'<div class="small">RM cost: ₹{rm_dry:,.2f}/t • O&M: ₹{st.session_state.om_cost:,.2f}/t • '
                     f'<b>Total dry cost: ₹{dry_total_cost:,.2f}/t</b></div>', unsafe_allow_html=True)

    with b:
        st.markdown('<div class="panel"><div class="panel-title">WET / AS-RECEIVED • BURDEN & COST COMPOSITION</div></div>', unsafe_allow_html=True)
        st.table(wet.round(3))
        st.markdown(f'<div class="small">RM procurement cost: ₹{rm_wet:,.2f}/t • O&M: ₹{st.session_state.om_cost:,.2f}/t • '
                     f'<b>Total wet cost: ₹{total_wet:,.2f}/t</b></div>', unsafe_allow_html=True)

def composition_content(r, kind):
    df=r["df"]; blend=r["blend"]
    total=sum(float(v) for v in blend.values())
    cost=sum(float(blend[m])*float(df.loc[m,"Price_Rs_t"])/1000 for m in blend)

    vals = (
        {GROUP_LABEL[g]:sum(float(blend[m]) for m in blend if df.loc[m,"Group"]==g) for g in GROUPS}
        if kind=="burden"
        else {GROUP_LABEL[g]:sum(float(blend[m])*float(df.loc[m,"Price_Rs_t"])/1000 for m in blend if df.loc[m,"Group"]==g) for g in GROUPS}
    )
    center=total if kind=="burden" else cost
    a,b=st.columns([1.7,1])
    with a:
        st.markdown('<div class="panel"><div class="panel-title">MATERIAL CONTRIBUTION</div>', unsafe_allow_html=True)
        st.plotly_chart(material_contribution_chart(blend, df, kind, height=max(400, 125 + 32 * len(blend))),
                         use_container_width=True, config={"displayModeBar":False})
        st.markdown('</div>',unsafe_allow_html=True)
    with b:
        rows=[{"Group":g,"Value":v,"% of Burden" if kind=="burden" else "% of Cost":(v/center*100 if center else 0)} for g,v in vals.items()]
        st.markdown('<div class="panel"><div class="panel-title">GROUP SUMMARY</div>',unsafe_allow_html=True)
        st.table(pd.DataFrame(rows).round(3))
        st.markdown('</div>',unsafe_allow_html=True)

    st.markdown('<div class="panel"><div class="panel-title">MATERIAL BREAKDOWN</div></div>',unsafe_allow_html=True)
    st.table(result_table(blend,df).round(3))

def recipe_content(r):
    bd=aligned_result_table(r["blend"],r["df"]); ach=r["achieved"]
    cost=float(bd.iloc[-1]["Cost ₹/t"]); total=float(bd.iloc[-1]["kg/t"])
    c=st.columns(4)
    for col,(l,v,s,k) in zip(c,[("TOTAL COST",f"₹{cost:,.2f}/t","Optimized","s"),
                                 ("BURDEN",f"{total:,.2f} kg/t","Optimized","g"),
                                 ("Fe",f"{ach['Fe']:.3f}%",f"{opt.FE_LOWER:.1f}–{opt.FE_UPPER:.1f}","a"),
                                 ("QUALITY","PASS" if quality_ok(ach) else "REVIEW","All mandatory targets","g" if quality_ok(ach) else "r")]):
        col.markdown(kpi(l,v,s,k),unsafe_allow_html=True)
    st.markdown('<div class="panel"><div class="panel-title">QUALITY</div></div>',unsafe_allow_html=True)
    quality_cards(ach)
    st.markdown('<div class="panel"><div class="panel-title">RECIPE</div></div>',unsafe_allow_html=True)
    st.table(bd.round(3))

def whatif_content():
    r = st.session_state.result
    if not r or not r.get("blend"):
        st.info("Run the optimizer first.")
        return
    if st.button("▶ RUN MATERIAL SHORTAGE SCENARIOS",type="primary",key="run_whatif"):
        with st.spinner("Evaluating scenarios…"):
            scenarios = opt.what_if_analysis(active_df(), TARGETS)
            base_cost = float(r.get("cost") or 0)
            if "Cost ₹/t" in scenarios.columns:
                scenarios["Cost Impact ₹/t"] = scenarios["Cost ₹/t"].apply(
                    lambda x: round(float(x) - base_cost, 2) if pd.notna(x) else np.nan
                )
            st.session_state.whatif = scenarios
    if st.session_state.whatif is not None:
        st.dataframe(
            st.session_state.whatif.round(3),
            hide_index=True,
            use_container_width=True,
            height=max(320, 38 * (len(st.session_state.whatif) + 1) + 30)
        )
    else:
        st.info("Run the scenario analysis.")

def bottleneck_content():
    r=st.session_state.result
    if not r or not r["achieved"]:
        st.info("Run optimizer first."); return
    st.table(opt.quality_table(r["achieved"],TARGETS).round(3))

def rm_stock_content():
    st.markdown('<div class="notice">Availability, price, RM stock and Tech Max are editable here. Chemistry is maintained on the Dashboard master table.</div>',unsafe_allow_html=True)
    inp=st.session_state.master_df.reset_index()[["Material","Material_Role","Group","Price_Rs_t","Available_Tonnes","Tech_Max"]].copy()
    inp.rename(columns={"Material_Role":"Role","Price_Rs_t":"Price ₹/t","Available_Tonnes":"RM Stock t","Tech_Max":"Tech Max t/d"},inplace=True)
    inp["Availability / Include"]=[st.session_state.include_alt.get(m,False) if st.session_state.master_df.loc[m,"Material_Role"]=="Alternative_Iron_Ore" else st.session_state.available.get(m,True) for m in inp.Material]
    ed=st.data_editor(inp,key="rm_editor",hide_index=True,use_container_width=True,height=max(250,42*len(inp)+45),disabled=["Material","Type","Group"])
    if not ed.equals(inp):
        for _,row in ed.iterrows():
            m=row.Material; isalt=st.session_state.master_df.loc[m,"Material_Role"]=="Alternative_Iron_Ore"
            st.session_state.master_df.loc[m,"Price_Rs_t"]=float(row["Price ₹/t"])
            st.session_state.master_df.loc[m,"Available_Tonnes"]=float(row["RM Stock t"])
            st.session_state.master_df.loc[m,"Tech_Max"]=float(row["Tech Max t/d"])
            (st.session_state.include_alt if isalt else st.session_state.available)[m]=bool(row["Availability / Include"])
        st.session_state.changed=True

def alternative_content():
    alts=alt_names()
    if not alts:
        st.info("No alternative rows were detected in the active Master Excel. Add a Type/Material_Role value such as Alternative in the same workbook and re-upload it.")
        return
    st.markdown('<div class="notice">OFF = completely excluded. ON = eligible for the optimizer; ON does not force usage. Chemistry, price, stock and Tech Max are editable from the Dashboard master/input tables.</div>',unsafe_allow_html=True)
    rows=[]
    for m in alts:
        r=st.session_state.master_df.loc[m]
        rows.append({"Material":m,"Fe":r.Fe,"SiO₂":r.SiO2,"Al₂O₃":r.Al2O3,"CaO":r.CaO,"MgO":r.MgO,"LOI":r.LOI,"Price ₹/t":r.Price_Rs_t,"RM Stock t":r.Available_Tonnes,"Tech Max t/d":r.Tech_Max,"Include in Mix":st.session_state.include_alt.get(m,False)})
    ed=st.data_editor(pd.DataFrame(rows),key="alt_editor",hide_index=True,use_container_width=True,height=max(220,42*len(rows)+45),disabled=["Material"])
    if not ed.equals(pd.DataFrame(rows)):
        for _,row in ed.iterrows():
            m=row.Material
            for src,dst in [("Fe","Fe"),("SiO₂","SiO2"),("Al₂O₃","Al2O3"),("CaO","CaO"),("MgO","MgO"),("LOI","LOI"),("Price ₹/t","Price_Rs_t"),("RM Stock t","Available_Tonnes"),("Tech Max t/d","Tech_Max")]:
                st.session_state.master_df.loc[m,dst]=float(row[src])
            st.session_state.include_alt[m]=bool(row["Include in Mix"])
        st.session_state.changed=True

# ----------------------------- DASHBOARD ---------------------------------
def dashboard():
    page_header("SINTER BURDEN CONTROL","Cost optimization • quality assurance • raw material decision support")
    st.markdown('<div class="hero"><div class="panel-title">DATA CONTROL CENTER</div><div class="small">Upload one Master Chemistry Excel. Chemistry, moisture, price, stock and Tech Min/Max are displayed in one table. Availability controls decide which materials enter the optimizer; alternative ores remain OFF until the user enables them.</div></div>',unsafe_allow_html=True)

    up1,up2=st.columns([1.7,1])
    with up1:
        f=st.file_uploader("MASTER EXCEL • XLSX",type=["xlsx"],key="dash_master")
        if f is not None:
            try:
                newdf=normalize_master(pd.read_excel(f))
                if st.button("ACTIVATE MASTER",type="primary",key="activate_dash"):
                    st.session_state.master_df=newdf; st.session_state.source=f.name
                    st.session_state.available={m:(float(newdf.loc[m,"Available_Tonnes"])>0) for m in newdf.index if newdf.loc[m,"Material_Role"]!="Alternative_Iron_Ore"}
                    st.session_state.include_alt={m:False for m in newdf.index if newdf.loc[m,"Material_Role"]=="Alternative_Iron_Ore"}
                    st.session_state.result=None; st.session_state.changed=False; st.session_state.runs=0; st.session_state.manual_scenario_result=None
                    st.rerun()
                st.markdown(f'<div class="small">Detected {int((newdf["Material_Role"]=="Primary_Iron_Ore").sum())} primary iron ore • {int((newdf["Material_Role"]=="Alternative_Iron_Ore").sum())} alternative iron ore • {int((newdf["Material_Role"]=="Other").sum())} other</div>',unsafe_allow_html=True)
            except Exception as e: st.error(str(e))
    with up2:
        st.markdown(f'<div class="notice"><b>ACTIVE MASTER</b><br>{st.session_state.source}<br>{len(primary_names())} primary • {len(alt_names())} alternative</div>',unsafe_allow_html=True)

    c1,c2,c3=st.columns([1,1.6,1])
    with c1: st.session_state.production=st.number_input("Production (t)",min_value=1.0,value=float(st.session_state.production),step=10.0,key="prod")
    with c2:
        if st.button("🚀 RUN OPTIMIZER",type="primary",use_container_width=True):
            with st.spinner("Optimizing v31.9…"): run_optimizer()
            st.rerun()
    with c3: st.markdown(f'<div class="notice" style="text-align:center"><b>RUN #{st.session_state.runs}</b><br>{sum(st.session_state.include_alt.values())} alternative enabled</div>',unsafe_allow_html=True)
    if st.session_state.changed: st.markdown('<div class="notice notice-w">Inputs changed — run optimizer to apply.</div>',unsafe_allow_html=True)

    r=st.session_state.result
    if r and r["blend"]:
        bd=aligned_result_table(r["blend"],r["df"]); total=float(bd.iloc[-1]["kg/t"]); cost=float(bd.iloc[-1]["Cost ₹/t"]); ach=r["achieved"]; ok=quality_ok(ach); used_alt=[m for m in alt_names() if r["blend"].get(m,0)>0]
        cols=st.columns(7)
        cards=[("OPTIMIZED COST",f"₹{cost:,.2f}/t","Material cost","s"),("TOTAL BURDEN",f"{total:,.1f} kg/t","All charged materials","g"),("Fe",f"{ach['Fe']:.3f}%",f"{opt.FE_LOWER:.1f}–{opt.FE_UPPER:.1f}","a"),("QUALITY","PASS" if ok else "REVIEW","Backend targets","g" if ok else "r"),("IOL FINES",f"{r['blend'].get('IOL_Fines',0)/total*100:.2f}%","Target 8%","g"),("BF RETURNS",f"{r['blend'].get('BF_Returns',0)/total*100:.2f}%","Target 17%","g"),("ALT ORE","USED" if used_alt else "NOT USED","Optional","a")]
        for c,(l,v,s,k) in zip(cols,cards): c.markdown(kpi(l,v,s,k),unsafe_allow_html=True)
    else:
        cols=st.columns(5)
        for c,(l,v,s,k) in zip(cols,[("OPTIMIZED COST","—","Run optimizer","s"),("TOTAL BURDEN","—","Run optimizer","g"),("Fe","—","Target band","a"),("QUALITY","READY","Awaiting run","g"),("ALT ORE","NOT USED","OFF by default","a")]): c.markdown(kpi(l,v,s,k),unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # CHEMISTRY CONSTRAINTS — immediately below the top KPI row
    # ------------------------------------------------------------------
    st.markdown(
        '<div class="panel"><div class="panel-title">CHEMISTRY CONSTRAINTS</div>'
        '<div class="small">Achieved values are shown against the active optimizer target bands.</div></div>',
        unsafe_allow_html=True
    )
    if r and r.get("achieved"):
        quality_cards(r["achieved"])
    else:
        st.info("Chemistry achievement will appear after optimization.")

    # ------------------------------------------------------------------
    # FULL-WIDTH MASTER INPUT TABLE
    # ------------------------------------------------------------------
    st.markdown(
        '<div class="panel"><div class="panel-title">RAW MATERIAL INPUTS — FULL WIDTH</div>'
        '<div class="small">Chemistry, moisture, price, stock, Tech Min/Max and availability all come from the same active Master Excel. '
        'Switch Availability / Include ON or OFF before running the optimizer.</div></div>',
        unsafe_allow_html=True
    )

    material_count = len(st.session_state.master_df.index)
    # Height is based on the number of materials, so every row is visible and
    # there is no internal vertical scrollbar.
    input_h = max(300, 31 * material_count + 55)

    inp = st.session_state.master_df.reset_index()[
        ["Material","Material_Role","Group"] + CHEM_COLS +
        ["Price_Rs_t","Available_Tonnes","Tech_Min","Tech_Max"]
    ].copy()
    inp.rename(columns={
        "Material_Role":"Role",
        "SiO2":"SiO₂",
        "Al2O3":"Al₂O₃",
        "Moisture_Pct":"Moisture %",
        "Price_Rs_t":"Price ₹/t",
        "Available_Tonnes":"RM Stock t",
        "Tech_Min":"Tech Min",
        "Tech_Max":"Tech Max"
    }, inplace=True)

    inp["Availability / Include"] = [
        st.session_state.include_alt.get(m, False)
        if st.session_state.master_df.loc[m,"Material_Role"] == "Alternative_Iron_Ore"
        else st.session_state.available.get(m, False)
        for m in inp.Material
    ]

    ed = st.data_editor(
        inp,
        key="merged_master_editor",
        hide_index=True,
        use_container_width=True,
        height=input_h,
        disabled=["Material","Role","Group"],
        column_config={
            "Material": st.column_config.TextColumn("Raw Material", width="small"),
            "Role": st.column_config.TextColumn("Role", width="small"),
            "Group": st.column_config.TextColumn("Group", width="small"),
            "Fe": st.column_config.NumberColumn("Fe %", format="%.2f", width="small"),
            "SiO₂": st.column_config.NumberColumn("SiO₂ %", format="%.2f", width="small"),
            "Al₂O₃": st.column_config.NumberColumn("Al₂O₃ %", format="%.2f", width="small"),
            "CaO": st.column_config.NumberColumn("CaO %", format="%.2f", width="small"),
            "MgO": st.column_config.NumberColumn("MgO %", format="%.2f", width="small"),
            "LOI": st.column_config.NumberColumn("LOI %", format="%.2f", width="small"),
            "Moisture %": st.column_config.NumberColumn("Moisture %", format="%.2f", width="small"),
            "Price ₹/t": st.column_config.NumberColumn("Price ₹/t", min_value=0, step=1, format="%.0f", width="small"),
            "RM Stock t": st.column_config.NumberColumn("RM Stock t", min_value=0, step=100, format="%.0f", width="small"),
            "Tech Min": st.column_config.NumberColumn("Tech Min", min_value=0, step=1, format="%.0f", width="small"),
            "Tech Max": st.column_config.NumberColumn("Tech Max", min_value=0, step=1, format="%.0f", width="small"),
            "Availability / Include": st.column_config.CheckboxColumn("Available / Include", width="small"),
        }
    )

    if not ed.equals(inp):
        for _, row in ed.iterrows():
            m = row["Material"]
            role = st.session_state.master_df.loc[m, "Material_Role"]
            for src, dst in [
                ("Fe","Fe"),("SiO₂","SiO2"),("Al₂O₃","Al2O3"),
                ("CaO","CaO"),("MgO","MgO"),("LOI","LOI"),
                ("Moisture %","Moisture_Pct"),("Tech Min","Tech_Min"),
                ("Tech Max","Tech_Max"),("Price ₹/t","Price_Rs_t"),
                ("RM Stock t","Available_Tonnes")
            ]:
                st.session_state.master_df.loc[m, dst] = float(row[src])
            (
                st.session_state.include_alt
                if role == "Alternative_Iron_Ore"
                else st.session_state.available
            )[m] = bool(row["Availability / Include"])
        st.session_state.changed = True

    # ------------------------------------------------------------------
    # OPTIMIZED BURDEN + COST — DRY AND WET SIDE BY SIDE
    # ------------------------------------------------------------------
    st.markdown(
        '<div class="panel"><div class="panel-title">OPTIMIZED BURDEN & COST COMPOSITION</div>'
        '<div class="small">Dry = optimizer / chemistry basis. Wet = as-received procurement basis. '
        'Both tables use the same material sequence and include a TOTAL row.</div></div>',
        unsafe_allow_html=True
    )

    if r and r.get("blend"):
        dry_out = aligned_result_table(r["blend"], r["df"], include_total=True)
        wet_out, wet_total_burden, wet_total_cost = wet_result_table(
            r["blend"], r["df"], include_total=True
        )

        # One shared height, derived from material count, so neither side
        # introduces an internal vertical scrollbar.
        result_h = max(300, 31 * (material_count + 1) + 55)

        left, right = st.columns(2, gap="small")

        with left:
            st.markdown(
                '<div class="panel-title">DRY BASIS — BURDEN & COST</div>',
                unsafe_allow_html=True
            )
            st.dataframe(
                dry_out.round(3),
                hide_index=True,
                use_container_width=True,
                height=result_h,
                column_config={
                    "Material": st.column_config.TextColumn("Material", width="small"),
                    "Group": st.column_config.TextColumn("Group", width="small"),
                    "kg/t": st.column_config.NumberColumn("kg/t", format="%.2f", width="small"),
                    "% Burden": st.column_config.NumberColumn("% Burden", format="%.2f", width="small"),
                    "Cost ₹/t": st.column_config.NumberColumn("Cost ₹/t", format="₹ %.2f", width="small"),
                    "% Cost": st.column_config.NumberColumn("% Cost", format="%.2f", width="small"),
                }
            )
            dry_total_burden = float(dry_out.iloc[-1]["kg/t"])
            dry_total_cost = float(dry_out.iloc[-1]["Cost ₹/t"])
            st.markdown(
                f'<div class="small"><b>TOTAL:</b> {dry_total_burden:,.2f} kg/t '
                f'&nbsp; • &nbsp; ₹{dry_total_cost:,.2f}/t</div>',
                unsafe_allow_html=True
            )

        with right:
            st.markdown(
                '<div class="panel-title">WET / AS-RECEIVED — BURDEN & COST</div>',
                unsafe_allow_html=True
            )
            st.dataframe(
                wet_out.round(3),
                hide_index=True,
                use_container_width=True,
                height=result_h,
                column_config={
                    "Material": st.column_config.TextColumn("Material", width="small"),
                    "Group": st.column_config.TextColumn("Group", width="small"),
                    "kg/t": st.column_config.NumberColumn("kg/t", format="%.2f", width="small"),
                    "% Burden": st.column_config.NumberColumn("% Burden", format="%.2f", width="small"),
                    "Cost ₹/t": st.column_config.NumberColumn("Cost ₹/t", format="₹ %.2f", width="small"),
                    "% Cost": st.column_config.NumberColumn("% Cost", format="%.2f", width="small"),
                }
            )
            st.markdown(
                f'<div class="small"><b>TOTAL:</b> {wet_total_burden:,.2f} kg/t '
                f'&nbsp; • &nbsp; ₹{wet_total_cost:,.2f}/t</div>',
                unsafe_allow_html=True
            )

        # Specific consumption is intentionally NOT shown as KPI cards.
        # Keep it as a compact line below the two composition tables.
        st.markdown(
            f'<div class="panel"><div class="panel-title">SPECIFIC CONSUMPTION</div>'
            f'<div class="small"><b>Dry:</b> {dry_total_burden:,.2f} kg/t'
            f' &nbsp;&nbsp; | &nbsp;&nbsp; '
            f'<b>Wet / As-Received:</b> {wet_total_burden:,.2f} kg/t</div></div>',
            unsafe_allow_html=True
        )
    else:
        st.info("Run the optimizer to populate the dry and wet burden/cost tables.")

    st.markdown(
        '<div class="small" style="margin-top:.5rem">'
        'Edit chemistry, price, stock, technical limits and availability directly in the merged master table. '
        'Alternative ores are OFF by default and become eligible only when enabled. '
        'Run the optimizer after changing inputs.</div>',
        unsafe_allow_html=True
    )

# ----------------------------- CONSOLIDATED PAGES --------------------------
def rm_stock_materials():
    page_header("RM Stock & Materials","Daily material availability, price, stock, technical maximums, and contingency ores — all from the same master workbook.")
    tabs = st.tabs(["PRIMARY STOCK", "ALTERNATIVE ORES"])
    with tabs[0]:
        rm_stock_content()
    with tabs[1]:
        alternative_content()

def recipe_composition():
    page_header("Recipe & Composition","Optimized recipe, burden/cost composition, and dry vs wet basis — one page, one source of truth.")
    r=st.session_state.result
    if not r or not r["blend"]:
        st.info("Run the optimizer first."); return
    tabs = st.tabs(["RECIPE", "BURDEN COMPOSITION", "COST COMPOSITION", "DRY vs WET"])
    with tabs[0]:
        recipe_content(r)
    with tabs[1]:
        composition_content(r, "burden")
    with tabs[2]:
        composition_content(r, "cost")
    with tabs[3]:
        dry_wet_composition_content(r)

def process_cost_parameters():
    page_header("Process & Cost Parameters","O&M cost and coke/thermal controls passed to the v30 optimizer, with live FeO and firing-ratio diagnostics.")
    r=st.session_state.result
    c1,c2=st.columns([1,2])
    with c1:
        st.session_state.om_cost=st.number_input("O&M Cost ₹/t",min_value=0.0,value=float(st.session_state.om_cost),step=50.0,key="om_cost_input")
        if r and r.get("blend"):
            bd_tot=aligned_result_table(r["blend"],r["df"])
            rm_cost=float(bd_tot.iloc[-1]["Cost ₹/t"])
            st.markdown(kpi("TOTAL DRY COST incl. O&M", f"₹{rm_cost+st.session_state.om_cost:,.2f}/t", f"RM ₹{rm_cost:,.2f}/t + O&M ₹{st.session_state.om_cost:,.2f}/t", "s"), unsafe_allow_html=True)
    with c2:
        st.caption("O&M is added to total sinter cost; it does not change the raw-material optimizer objective. See Recipe & Composition → Dry vs Wet for the full dry and wet cost tables — O&M can be edited from there too.")

    st.markdown('<div class="panel"><div class="panel-title">🔥 COKE & THERMAL PARAMETERS</div><div class="small">These controls are passed to the v30 optimizer when available. Re-run the optimizer after changing them.</div></div>',unsafe_allow_html=True)
    q1,q2,q3,q4=st.columns(4)
    with q1: st.session_state.coke_cv=st.number_input("Coke CV (kcal/kg)",min_value=1000.0,value=float(st.session_state.coke_cv),step=50.0)
    with q2: st.session_state.coke_fc=st.number_input("Fixed Carbon (%)",min_value=1.0,max_value=100.0,value=float(st.session_state.coke_fc),step=.1)
    with q3: st.session_state.coke_min_rate=st.number_input("Coke Min (kg/t)",min_value=0.0,value=float(st.session_state.coke_min_rate),step=1.0)
    with q4: st.session_state.coke_max_rate=st.number_input("Coke Max (kg/t)",min_value=0.0,value=float(st.session_state.coke_max_rate),step=1.0)
    q5,q6,q7,q8=st.columns(4)
    with q5: st.session_state.feo_min=st.number_input("FeO Min (%)",min_value=0.0,value=float(st.session_state.feo_min),step=.1)
    with q6: st.session_state.feo_target=st.number_input("FeO Target (%)",min_value=0.0,value=float(st.session_state.feo_target),step=.1)
    with q7: st.session_state.feo_max=st.number_input("FeO Max (%)",min_value=0.0,value=float(st.session_state.feo_max),step=.1)
    with q8: st.session_state.manual_coke_override=st.checkbox("Manual Coke Override",value=bool(st.session_state.manual_coke_override))
    if st.session_state.manual_coke_override:
        st.session_state.manual_coke_rate=st.number_input("Fixed Coke Rate (kg/t)",min_value=float(st.session_state.coke_min_rate),max_value=float(st.session_state.coke_max_rate),value=float(st.session_state.manual_coke_rate),step=.5)

    if r and r.get("blend"):
        diag=coke_diagnostic(r["blend"],r["df"])
        if diag:
            st.markdown('<div class="panel"><div class="panel-title">DIAGNOSTICS</div></div>', unsafe_allow_html=True)
            x,y,z=st.columns(3)
            x.metric("Predicted FeO",f"{diag.get('FeO_Estimate_Pct',np.nan):.2f}%")
            y.metric("Thermal Surplus",f"{diag.get('Thermal_Surplus_kcal',diag.get('Thermal_Surplus',np.nan)):,.0f}")
            z.metric("Firing Ratio",f"{diag.get('Firing_Ratio',np.nan):.3f}")
            st.caption(str(diag.get("Controller_Suggestion", "")))
    else:
        st.info("Run the optimizer to see live FeO / firing-ratio diagnostics here.")

def scenario_analysis():
    page_header("Scenario Analysis","Material shortage stress-testing and quality-constraint pressure, side by side.")
    tabs = st.tabs(["MATERIAL SHORTAGE", "CONSTRAINT PRESSURE"])
    with tabs[0]:
        st.caption("Test one-at-a-time material unavailability against the current model.")
        whatif_content()
    with tabs[1]:
        st.caption("Identify the constraints closest to their limits.")
        bottleneck_content()

def manual():
    page_header("Manual Burden Control","Practical scenario analysis — the optimized recipe remains the frozen theoretical baseline.")
    r=st.session_state.result
    if not r or not r.get("blend"):
        st.info("Run the optimizer first."); return

    df=r["df"]
    base=dict(st.session_state.get("manual_base") or r["blend"])
    st.session_state.manual_base=base

    st.markdown('<div class="notice"><b>THEORETICAL BASELINE → PRACTICAL SCENARIO</b><br>Change one or more raw materials. The changed quantities become fixed practical constraints. The optimizer automatically re-optimizes all remaining eligible materials while preserving availability, chemistry, Tech Min/Max, coke/thermal limits and the IOL/BF burden mandates.</div>',unsafe_allow_html=True)

    mode=st.radio("Adjustment mode",["kg/t","%"],horizontal=True,key="manual_mode")
    rows=[]
    for m in df.index:
        b=float(base.get(m,0.0))
        if b <= 1e-9 and float(r["blend"].get(m,0.0)) <= 1e-9: continue
        total=sum(float(v) for v in base.values())
        opct=b/total*100 if total else 0.0
        role=str(df.loc[m,"Material_Role"])
        locked=role in {"Other"} or str(df.loc[m,"Group"]) in {"IOL_Fines_Mandate","BF_Returns_Mandate"}
        if m == "COKE_BREEZE": locked=False
        rows.append({
            "Raw Material":m,
            "Optimized kg/t":b,
            "Optimized %":opct,
            "Change":0.0,
            "Locked":locked,
        })
    base_table=pd.DataFrame(rows)
    ed=st.data_editor(
        base_table, hide_index=True, use_container_width=True,
        key="manual_scenario_editor",
        disabled=["Raw Material","Optimized kg/t","Optimized %","Locked"],
        column_config={
            "Optimized kg/t":st.column_config.NumberColumn("Optimized kg/t",format="%.2f"),
            "Optimized %":st.column_config.NumberColumn("Optimized %",format="%.2f%%"),
            "Change":st.column_config.NumberColumn("Change kg/t" if mode=="kg/t" else "Change %",step=0.5,format="%+.2f" if mode=="kg/t" else "%+.2f%%"),
            "Locked":st.column_config.CheckboxColumn("Locked"),
        },
        height=max(180, 38*len(base_table)+45)
    )

    fixed={}
    for _,row in ed.iterrows():
        m=row["Raw Material"]
        if bool(row["Locked"]):
            continue
        b=float(row["Optimized kg/t"])
        ch=float(row["Change"] or 0.0)
        q=b+ch if mode=="kg/t" else b*(1+ch/100.0)
        fixed[m]=max(0.0,q)

    if "manual_scenario_result" not in st.session_state:
        st.session_state.manual_scenario_result=None

    if st.button("🔄 RECALCULATE PRACTICAL SCENARIO",type="primary",use_container_width=True,key="manual_reopt"):
        with st.spinner("Re-optimizing around the practical constraints…"):
            kwargs={
                "coke_cv":st.session_state.coke_cv,
                "coke_fc":st.session_state.coke_fc,
                "latent_heat":st.session_state.latent_heat,
                "calcination_heat":st.session_state.calcination_heat,
                "melting_heat":st.session_state.melting_heat,
                "loss_fraction":st.session_state.loss_fraction,
                "firing_ratio_max":st.session_state.firing_ratio_max,
                "coke_min_rate":st.session_state.coke_min_rate,
                "coke_max_rate":st.session_state.coke_max_rate,
                "feo_min":st.session_state.feo_min,
                "feo_target":st.session_state.feo_target,
                "feo_max":st.session_state.feo_max,
                "manual_override":st.session_state.manual_coke_override,
                "manual_coke_rate":st.session_state.manual_coke_rate,
            }
            try:
                x=opt.solve_manual_scenario(df,float(st.session_state.production),TARGETS,base,fixed,**kwargs)
            except TypeError:
                x=opt.solve_manual_scenario(df,float(st.session_state.production),TARGETS,base,fixed)
            st.session_state.manual_scenario_result={"status":x[0],"blend":x[1],"cost":x[2],"achieved":x[3],"diagnostics":x[4],"df":df.copy()}
        st.rerun()

    practical=st.session_state.get("manual_scenario_result")
    if practical and practical.get("blend"):
        pbd=aligned_result_table(practical["blend"],df,include_total=True)
        base_cost=sum(float(base[m])*float(df.loc[m,"Price_Rs_t"])/1000 for m in base if m in df.index)
        practical_cost=float(pbd.iloc[-1]["Cost ₹/t"])
        base_total=sum(float(v) for v in base.values()); practical_total=sum(float(v) for v in practical["blend"].values())
        a,b,c,d=st.columns(4)
        a.markdown(kpi("BASELINE COST",f"₹{base_cost:,.2f}/t","Theoretical","s"),unsafe_allow_html=True)
        b.markdown(kpi("PRACTICAL COST",f"₹{practical_cost:,.2f}/t",f"Δ ₹{practical_cost-base_cost:+,.2f}/t","a"),unsafe_allow_html=True)
        c.markdown(kpi("BASELINE BURDEN",f"{base_total:,.2f} kg/t","Theoretical","g"),unsafe_allow_html=True)
        d.markdown(kpi("PRACTICAL BURDEN",f"{practical_total:,.2f} kg/t",f"Δ {practical_total-base_total:+,.2f}","g"),unsafe_allow_html=True)

        compare=[]
        total_b=base_total; total_p=practical_total
        for m in df.index:
            ov=float(base.get(m,0)); pv=float(practical["blend"].get(m,0))
            if ov==0 and pv==0: continue
            compare.append({"Raw Material":m,"Optimized kg/t":ov,"Optimized %":ov/total_b*100 if total_b else 0,
                            "Practical kg/t":pv,"Practical %":pv/total_p*100 if total_p else 0,
                            "Change kg/t":pv-ov,"Change %":(pv-ov)/ov*100 if ov else np.nan})
        comp=pd.DataFrame(compare)
        comp.loc[len(comp)]={"Raw Material":"TOTAL","Optimized kg/t":total_b,"Optimized %":100.0,
                             "Practical kg/t":total_p,"Practical %":100.0,"Change kg/t":total_p-total_b,"Change %":(total_p-total_b)/total_b*100 if total_b else 0}
        st.subheader("Theoretical vs Practical")
        st.table(comp.round(3))

        ach=practical["achieved"]
        st.subheader("Practical chemistry / compliance")
        quality_cards(ach)
        st.table(opt.quality_table(ach,TARGETS).round(3))
        st.markdown("### Practical diagnostics")
        for msg in practical.get("diagnostics",[]): st.write(msg)
    else:
        st.info("Enter a change and press RE-CALCULATE PRACTICAL SCENARIO. The original optimized recipe remains unchanged.")

def reports():
    page_header("Reports & Export","Export the latest optimized recipe with reconciled burden and cost contributions.")
    r=st.session_state.result
    if not r or not r["blend"]: st.info("Run optimizer first."); return
    bd=aligned_result_table(r["blend"],r["df"])
    st.table(bd.round(3))
    st.download_button("⬇ DOWNLOAD OPTIMIZATION REPORT",bd.to_csv(index=False).encode(),"sinter_optimization_report.csv","text/csv",use_container_width=True)

def settings():
    page_header("Upload & Settings","Single master workbook management.")
    f=st.file_uploader("UPLOAD MASTER CHEMISTRY EXCEL",type=["xlsx"],key="settings_master")
    if f:
        try:
            df=normalize_master(pd.read_excel(f)); st.success(f"Validated: {len(df)} materials • {int((df['Material_Role']=='Primary_Iron_Ore').sum())} primary • {int((df['Material_Role']=='Alternative_Iron_Ore').sum())} alternative")
            if st.button("ACTIVATE MASTER",type="primary"):
                st.session_state.master_df=df; st.session_state.source=f.name; st.session_state.available={m:(float(df.loc[m,"Available_Tonnes"])>0) for m in df.index if df.loc[m,"Material_Role"]!="Alternative_Iron_Ore"}; st.session_state.include_alt={m:False for m in alt_names()}; st.session_state.result=None; st.session_state.runs=0; st.session_state.manual_scenario_result=None; st.rerun()
        except Exception as e: st.error(str(e))
    if st.button("↺ RESTORE BUILT-IN MASTER",use_container_width=True):
        st.session_state.master_df=initial_df(); st.session_state.source="Built-in Master Chemistry"; st.session_state.available={m:True for m in primary_names()}; st.session_state.include_alt={}; st.session_state.result=None; st.session_state.runs=0; st.rerun()
    st.markdown('<div class="notice">The same workbook can carry both Primary and Alternative rows. Use a Material_Role column with Primary_Iron_Ore / Alternative_Iron_Ore / Other. If it is absent, iron-ore rows default to Primary_Iron_Ore for backward compatibility.</div>',unsafe_allow_html=True)

# ----------------------------- ROUTING -----------------------------------
pages={
    "Dashboard":dashboard,
    "RM Stock & Materials":rm_stock_materials,
    "Recipe & Composition":recipe_composition,
    "Manual Burden Control":manual,
    "Process & Cost Parameters":process_cost_parameters,
    "Scenario Analysis":scenario_analysis,
    "Reports":reports,
    "Upload & Settings":settings,
}
pages[st.session_state.nav]()
st.markdown('<div class="footer">Sinter Burden Control • Hospet Alloy Steel Plant • Production decision-support interface</div>',unsafe_allow_html=True)
