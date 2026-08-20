
"""Hospet Steels Sinter Burden Optimizer v32 - Streamlit-safe backend.

Core rules retained from the supplied v31.9 model:
- strict IOL Fines = 8% of total charged burden, including coke
- strict BF Returns = 17% of total charged burden, including coke
- total burden is not forced to 1000 kg/t
- 1000 kg/t is the dry-product reference for chemistry/mass balance
- cost = sum(kg/t * Rs/t / 1000)
- availability is a hard ON/OFF gate
- alternatives are eligible only when explicitly ON
- fresh LP problem per solve/scenario
"""
import re
import numpy as np
import pandas as pd
import pulp

TARGETS={"Fe_min":53.7,"Fe_max":54.3,"SiO2_max":5.8,"Al2O3_max":4.5,"Al2O3_SiO2_max":0.98,
         "Basicity_min":1.9,"Basicity_max":2.0,"MgO_min":2.2,"MgO_max":2.4,"CaO_min":10.5,"CaO_max":11.5}

IOL_PCT=.08
BF_PCT=.17
MILL_MIN=.05
MILL_MAX=.15

DEFAULTS=[
("MILL_SCALE","Primary_Iron_Ore","Iron_ore",68.34,2,2.72,0,0,2.5,6,0,220,2000,7800),
("Lloyds_HG","Primary_Iron_Ore","Iron_ore",63.52,3.86,2.27,.02,.03,2.29,5,0,200,10000,7820),
("DIOM_LG","Primary_Iron_Ore","Iron_ore",57.17,12.39,2.93,.05,.11,4,6,0,200,6000,4600),
("SIOM_MG","Primary_Iron_Ore","Iron_ore",59.34,6.92,3.72,.25,.33,3.45,6,0,200,8000,4600),
("KIOM_MG","Primary_Iron_Ore","Iron_ore",58.41,5.75,5.48,.15,.02,4.62,6,0,300,5000,4900),
("Solid_Waste","Other","Recycle",50,6,4.5,1.12,.06,3,1.1,30,30,5000,1000),
("IOL_Fines","Other","IOL_Fines_Mandate",60,5,3,8.79,1.52,3,4.13,0,999,5000,5577),
("FLUE_DUST","Other","Recycle",47.02,7.07,4.5,1.10,.29,15,9.4,25,25,3000,500),
("BF_Returns","Other","BF_Returns_Mandate",52.5,5.62,3.2,10.74,2.30,3,0,0,999,5000,0),
("DOLOMITE","Other","Flux",.54,4.72,.95,30.02,18.75,42,2,30,200,10000,1340),
("LIMESTONE","Other","Flux",.88,4.48,1.19,48.71,2.59,40,2,0,250,15000,1355),
("QUICKLIME","Other","Flux",.01,2.5,.61,89,1.57,5,0,40,65,5000,9200),
("COKE_BREEZE","Other","Fuel",0,2.8,0,0,0,70,11.27,0,85,9999,15022),
]
COLS=["Material","Material_Role","Group","Fe","SiO2","Al2O3","CaO","MgO","LOI","Moisture_Pct","Tech_Min","Tech_Max","Available_Tonnes","Price_Rs_t"]

def safe_name(x): return re.sub(r"[^A-Za-z0-9_]+","_",str(x))

def get_default_chemistry():
    return pd.DataFrame(DEFAULTS,columns=COLS).set_index("Material")

def _normalise_role(v,group,material):
    s=str(v).strip().lower()
    if "alternative" in s or "contingency" in s: return "Alternative_Iron_Ore"
    if group=="Iron_ore": return "Primary_Iron_Ore"
    return "Other"

def load_master_excel(uploaded):
    if hasattr(uploaded,"read"):
        raw=pd.read_excel(uploaded)
    else:
        raw=pd.read_excel(uploaded)
    raw.columns=[str(c).strip() for c in raw.columns]
    aliases={
      "Material":"Material","material":"Material",
      "Material_Type":"Material_Role","Material Role":"Material_Role","Role":"Material_Role",
      "Group":"Group","Fe":"Fe","Fe_%":"Fe","Fe %":"Fe",
      "SiO2":"SiO2","SiO2_%":"SiO2","SiO₂ %":"SiO2",
      "Al2O3":"Al2O3","Al2O3_%":"Al2O3","Al₂O₃ %":"Al2O3",
      "CaO":"CaO","CaO_%":"CaO","MgO":"MgO","MgO_%":"MgO",
      "LOI":"LOI","LOI_%":"LOI","Moisture_Pct":"Moisture_Pct","Moisture %":"Moisture_Pct",
      "Tech_Min":"Tech_Min","Tech_Min_kg_t":"Tech_Min","Tech Min":"Tech_Min",
      "Tech_Max":"Tech_Max","Tech_Max_kg_t":"Tech_Max","Tech Max":"Tech_Max",
      "Available_Tonnes":"Available_Tonnes","Available_Stock_t":"Available_Tonnes","RM Stock t":"Available_Tonnes",
      "Price_Rs_t":"Price_Rs_t","Price ₹/t":"Price_Rs_t"
    }
    raw=raw.rename(columns={c:aliases.get(c,c) for c in raw.columns})
    required=["Material","Group","Fe","SiO2","Al2O3","CaO","MgO","LOI","Tech_Min","Tech_Max","Available_Tonnes","Price_Rs_t"]
    missing=[c for c in required if c not in raw.columns]
    if missing: raise ValueError("Missing columns: "+", ".join(missing))
    if "Moisture_Pct" not in raw: raw["Moisture_Pct"]=0.0
    if "Material_Role" not in raw: raw["Material_Role"]="Other"
    raw=raw[required+["Moisture_Pct","Material_Role"]].copy()
    raw["Material"]=raw["Material"].astype(str).str.strip()
    if raw["Material"].duplicated().any(): raise ValueError("Duplicate material names.")
    for c in required[2:]+["Moisture_Pct"]:
        raw[c]=pd.to_numeric(raw[c],errors="coerce").fillna(0.0)
    raw["Material_Role"]=[_normalise_role(v,g,m) for v,g,m in zip(raw["Material_Role"],raw["Group"],raw["Material"])]
    raw.loc[raw["Group"].eq("Recycle"),"Tech_Max"]=raw.loc[raw["Group"].eq("Recycle"),"Tech_Min"]
    return raw.set_index("Material")[["Material_Role","Group","Fe","SiO2","Al2O3","CaO","MgO","LOI","Moisture_Pct","Tech_Min","Tech_Max","Available_Tonnes","Price_Rs_t"]]

def quality_checks(a,targets=TARGETS):
    return {
      "Fe":targets["Fe_min"]-1e-6<=a["Fe"]<=targets["Fe_max"]+1e-6,
      "SiO2":a["SiO2"]<=targets["SiO2_max"]+1e-6,
      "Al2O3":a["Al2O3"]<=targets["Al2O3_max"]+1e-6,
      "Al2O3_SiO2":a["Al2O3_SiO2"]<=targets["Al2O3_SiO2_max"]+1e-6,
      "Basicity":targets["Basicity_min"]-1e-6<=a["Basicity"]<=targets["Basicity_max"]+1e-6,
      "MgO":targets["MgO_min"]-1e-6<=a["MgO"]<=targets["MgO_max"]+1e-6,
      "CaO":targets["CaO_min"]-1e-6<=a["CaO"]<=targets["CaO_max"]+1e-6
    }

def compute_achieved(blend,df,out=1000):
    total=sum(float(blend.get(m,0)) for m in df.index)
    dry=sum(float(blend.get(m,0))*(1-float(df.loc[m,"Moisture_Pct"])/100-float(df.loc[m,"LOI"])/100) for m in df.index)
    def avg(col): return sum(float(blend.get(m,0))*float(df.loc[m,col])/100 for m in df.index)/out
    fe=avg("Fe"); sio=avg("SiO2"); al=avg("Al2O3"); cao=avg("CaO"); mgo=avg("MgO")
    ratio=al/sio if sio else np.inf
    bas=cao/sio if sio else np.nan
    b4=(cao+mgo)/(sio+al) if sio+al else np.nan
    return {"Fe":fe,"SiO2":sio,"Al2O3":al,"Al2O3_SiO2":ratio,"Basicity":bas,"MgO":mgo,"CaO":cao,"B4":b4,"Total_Burden":total,"Dry_Solids":dry}

def _bounds(df,production_tonnes):
    b={}
    for m in df.index:
        avail=float(df.loc[m,"Available_Tonnes"]); tmax=float(df.loc[m,"Tech_Max"]); tmin=float(df.loc[m,"Tech_Min"])
        if avail<=0 or tmax<=0: b[m]=(0,0); continue
        inv=avail/production_tonnes*1000
        b[m]=(max(0,tmin),min(tmax,inv))
    return b

def _add_constraints(prob,x,df,bounds,targets,tag=""):
    total=pulp.lpSum(x[m] for m in df.index)
    # Strict mandates: total charged burden INCLUDING coke.
    if "IOL_Fines" in x: prob += x["IOL_Fines"] == IOL_PCT*total, f"IOL_STRICT_{tag}"
    if "BF_Returns" in x: prob += x["BF_Returns"] == BF_PCT*total, f"BF_STRICT_{tag}"
    if "MILL_SCALE" in x and bounds["MILL_SCALE"][1]>0:
        prob += x["MILL_SCALE"] >= MILL_MIN*total, f"MILL_MIN_{tag}"
        prob += x["MILL_SCALE"] <= MILL_MAX*total, f"MILL_MAX_{tag}"
    iron=[m for m in df.index if df.loc[m,"Group"]=="Iron_ore"]
    flux=[m for m in df.index if df.loc[m,"Group"]=="Flux"]
    for m in iron:
        cap=.25 if str(df.loc[m,"Material_Role"])=="Primary_Iron_Ore" else .29
        prob += x[m] <= cap*total, f"IRON_CAP_{safe_name(m)}_{tag}"
    if iron: prob += pulp.lpSum(x[m] for m in iron) <= .80*total, f"IRON_TOTAL_{tag}"
    for m in flux:
        cap={"LIMESTONE":.60,"DOLOMITE":.45,"QUICKLIME":.60}.get(m,.60)
        prob += x[m] <= cap*total, f"FLUX_CAP_{safe_name(m)}_{tag}"
    # dry-solids balance to 1 tonne product
    dry=pulp.lpSum(x[m]*(1-df.loc[m,"Moisture_Pct"]/100-df.loc[m,"LOI"]/100) for m in df.index if m!="COKE_BREEZE")
    prob += dry == 1000, f"DRY_SOLIDS_BALANCE_{tag}"
    # chemistry constraints
    Fe=pulp.lpSum(x[m]*df.loc[m,"Fe"]/100 for m in df.index)
    Si=pulp.lpSum(x[m]*df.loc[m,"SiO2"]/100 for m in df.index)
    Al=pulp.lpSum(x[m]*df.loc[m,"Al2O3"]/100 for m in df.index)
    Ca=pulp.lpSum(x[m]*df.loc[m,"CaO"]/100 for m in df.index)
    Mg=pulp.lpSum(x[m]*df.loc[m,"MgO"]/100 for m in df.index)
    # Chemistry is expressed per tonne of finished sinter (1000 kg dry-product reference).
    out=1000.0
    prob += Fe >= targets["Fe_min"]*out/100, f"FE_MIN_{tag}"; prob += Fe <= targets["Fe_max"]*out/100, f"FE_MAX_{tag}"
    prob += Si <= targets["SiO2_max"]*out/100, f"SI_MAX_{tag}"
    prob += Al <= targets["Al2O3_max"]*out/100, f"AL_MAX_{tag}"
    prob += Al <= targets["Al2O3_SiO2_max"]*Si, f"RATIO_{tag}"
    prob += Ca >= targets["Basicity_min"]*Si, f"BAS_MIN_{tag}"; prob += Ca <= targets["Basicity_max"]*Si, f"BAS_MAX_{tag}"
    prob += Mg >= targets["MgO_min"]*out/100, f"MG_MIN_{tag}"; prob += Mg <= targets["MgO_max"]*out/100, f"MG_MAX_{tag}"
    prob += Ca >= targets["CaO_min"]*out/100, f"CA_MIN_{tag}"; prob += Ca <= targets["CaO_max"]*out/100, f"CA_MAX_{tag}"

def solve_blend_with_compensation(df,production_tonnes=1000,targets=TARGETS,baseline_blend=None,scenario_tag="base",**kwargs):
    df=df.copy()
    bounds=_bounds(df,production_tonnes)
    prob=pulp.LpProblem(f"Sinter_{safe_name(scenario_tag)}",pulp.LpMinimize)
    x={m:pulp.LpVariable(f"x_{safe_name(m)}_{safe_name(scenario_tag)}",lowBound=bounds[m][0],upBound=bounds[m][1]) for m in df.index}
    _add_constraints(prob,x,df,bounds,targets,safe_name(scenario_tag))
    # zero price is legitimate for BF returns if plant price is unknown.
    prob += pulp.lpSum(x[m]*float(df.loc[m,"Price_Rs_t"])/1000 for m in df.index), "TOTAL_MATERIAL_COST"
    status=prob.solve(pulp.PULP_CBC_CMD(msg=0))
    st=pulp.LpStatus[status]
    if st!="Optimal":
        return st,None,None,None,[f"Solver status: {st}"],True
    blend={m:round(float(x[m].value() or 0),4) for m in df.index}
    cost=sum(blend[m]*float(df.loc[m,"Price_Rs_t"])/1000 for m in df.index)
    achieved=compute_achieved(blend,df,1000)
    return "Optimal",blend,cost,achieved,[],False

def calculate_cost_breakdown(blend,df):
    rows=[]; total=sum(blend.values()); tc=0
    for m in df.index:
        q=float(blend.get(m,0)); c=q*float(df.loc[m,"Price_Rs_t"])/1000; tc+=c
        rows.append({"Material":m,"Group":df.loc[m,"Group"],"kg/t":q,"% Burden":q/total*100 if total else 0,"Cost ₹/t":c,"% Cost":0})
    for r in rows: r["% Cost"]=r["Cost ₹/t"]/tc*100 if tc else 0
    return rows,tc,total

def build_result_table(blend,df):
    rows,tc,total=calculate_cost_breakdown(blend,df)
    out=pd.DataFrame(rows)
    out.loc[len(out)]=["TOTAL","",total,100,tc,100]
    return out

def build_wet_table(blend,df):
    rows,tc,total=calculate_cost_breakdown(blend,df)
    out=[]
    wt=0
    for m in df.index:
        q=float(blend.get(m,0)); moist=float(df.loc[m,"Moisture_Pct"]); wet=q/(1-moist/100) if moist<100 else q
        c=q*float(df.loc[m,"Price_Rs_t"])/1000; wt+=wet
        out.append({"Material":m,"Group":df.loc[m,"Group"],"Burden kg/t":wet,"Moisture %":moist,"Burden %":0,"Cost ₹/t":c,"Cost %":0})
    tc=sum(r["Cost ₹/t"] for r in out)
    for r in out:
        r["Burden %"]=r["Burden kg/t"]/wt*100 if wt else 0; r["Cost %"]=r["Cost ₹/t"]/tc*100 if tc else 0
    out.append({"Material":"TOTAL","Group":"","Burden kg/t":wt,"Moisture %":0,"Burden %":100,"Cost ₹/t":tc,"Cost %":100})
    return pd.DataFrame(out)

def redistribute_manual(base,df,requested):
    adj={m:float(v) for m,v in base.items()}
    changed=set(requested)
    for m,v in requested.items(): adj[m]=max(0,float(v))
    total0=sum(base.values()); delta=sum(adj.values())-total0
    # Never compensate through strict mandates or fixed recycle/fuel.
    compensators=[m for m in base if m not in changed and m not in ("IOL_Fines","BF_Returns") and df.loc[m,"Group"] not in ("Recycle","Fuel")]
    if abs(delta)>1e-9 and compensators:
        denom=sum(base[m] for m in compensators)
        for m in compensators:
            adj[m]=max(0,base[m]-delta*(base[m]/denom))
    return {m:round(v,4) for m,v in adj.items()}

def coke_diagnostic(blend,df):
    cb=float(blend.get("COKE_BREEZE",0)); cv=6800; fc=.7135
    moisture=sum(blend.get(m,0)*df.loc[m,"Moisture_Pct"]/100 for m in df.index if m!="COKE_BREEZE")
    loi=sum(blend.get(m,0)*df.loc[m,"LOI"]/100 for m in df.index if m!="COKE_BREEZE")
    qfuel=cb*fc*cv; qreq=540*moisture+420*loi+60*1000
    return {"FeO":8.6+.35*((qfuel-qreq-189180)/10000),"Thermal Surplus":qfuel-qreq,"Firing Ratio":qfuel/qreq if qreq else 0}
