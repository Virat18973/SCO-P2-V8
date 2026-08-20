
import io, math
import pandas as pd
import numpy as np
import streamlit as st
import optimizer as opt

st.set_page_config(page_title="Sinter Burden Control", page_icon="🏭", layout="wide", initial_sidebar_state="expanded")

CSS = """
<style>
:root{--bg:#071017;--panel:#0d1b23;--panel2:#101f28;--line:#284654;--text:#edf5fa;--muted:#7895a5;--blue:#2f82b3;--good:#27c48b;--warn:#f0b64b;--bad:#ff5d61}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg);color:var(--text)}
[data-testid="stSidebar"]{background:#09131a;border-right:1px solid var(--line)}
h1{font-size:2rem!important;margin:.1rem 0 .2rem!important}
h2{font-size:1.25rem!important}
.small,.caption{color:var(--muted);font-size:.72rem}
.eyebrow{color:#65c8f5;font-size:.58rem;font-weight:900;letter-spacing:.18em}
.panel,.hero,.notice{background:var(--panel);border:1px solid var(--line);border-radius:7px;padding:.65rem .8rem;margin:.35rem 0}
.hero{background:#0d2230}.notice{background:#102638}
.panel-title{color:#69c9f4;font-size:.64rem;font-weight:900;letter-spacing:.12em}
.kpi{background:var(--panel);border:1px solid var(--line);border-radius:7px;padding:.55rem .65rem;min-height:88px}
.kpi-label{font-size:.56rem;color:#72a1b5;font-weight:900;letter-spacing:.08em}.kpi-value{font-size:1.05rem;font-weight:900;margin-top:.3rem}.kpi-sub{font-size:.58rem;color:var(--muted)}
.good{border-left:3px solid var(--good)}.warn{border-left:3px solid var(--warn)}.bad{border-left:3px solid var(--bad)}.blue{border-left:3px solid var(--blue)}
[data-testid="stDataFrame"],[data-testid="stDataEditor"]{border:1px solid var(--line);border-radius:6px}
[data-testid="stDataFrame"] [role="gridcell"],[data-testid="stDataEditor"] [role="gridcell"]{font-size:11px}
div.stButton>button{border:1px solid #315466;border-radius:6px;background:#10212b;color:#edf5fa}
button[kind="primary"]{background:#2d7fac!important;border-color:#4aa2d0!important}
.footer{color:#4f6977;font-size:.58rem;text-align:right;margin-top:1rem}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

TARGETS=opt.TARGETS

def init_state():
    if "df" not in st.session_state:
        st.session_state.df=opt.get_default_chemistry()
        st.session_state.source="Built-in master"
        st.session_state.availability={m:True for m in st.session_state.df.index}
        st.session_state.result=None
        st.session_state.manual_base=None
        st.session_state.manual_values=None
        st.session_state.runs=0
        st.session_state.nav="Dashboard"
        st.session_state.om_cost=1500.0
        st.session_state.changed=False
init_state()

def active_df():
    df=st.session_state.df.copy()
    for m in df.index:
        if not st.session_state.availability.get(m,True):
            df.loc[m,"Available_Tonnes"]=0.0
    return df

def kpi(label,value,sub="",kind="blue"):
    return f'<div class="kpi {kind}"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>'

def quality_status(a):
    return opt.quality_checks(a,TARGETS)

def quality_table(a):
    rows=[]
    specs=[
        ("Fe","Fe_min","Fe_max","range"),
        ("SiO₂","SiO2_max",None,"max"),
        ("Al₂O₃","Al2O3_max",None,"max"),
        ("Al₂O₃/SiO₂","Al2O3_SiO2_max",None,"max"),
        ("Basicity","Basicity_min","Basicity_max","range"),
        ("MgO","MgO_min","MgO_max","range"),
        ("CaO","CaO_min","CaO_max","range"),
        ("B4",None,None,"info"),
    ]
    for label,lo,hi,typ in specs:
        key={"SiO₂":"SiO2","Al₂O₃":"Al2O3","Al₂O₃/SiO₂":"Al2O3_SiO2"}.get(label,label)
        val=float(a.get(key,np.nan))
        if typ=="range": target=f"{TARGETS[lo]:g}–{TARGETS[hi]:g}"; ok=TARGETS[lo]-1e-6<=val<=TARGETS[hi]+1e-6
        elif typ=="max": target=f"≤ {TARGETS[lo]:g}"; ok=val<=TARGETS[lo]+1e-6
        else: target="1.8–2.2 info"; ok=True
        rows.append({"KPI":label,"Achieved":val,"Target":target,"Status":"PASS" if ok else "REVIEW",
                     "Margin": (min(val-TARGETS[lo],TARGETS[hi]-val) if typ=="range" else TARGETS[lo]-val) if typ!="info" else np.nan})
    return pd.DataFrame(rows)

def result_tables(r):
    return opt.build_result_table(r["blend"],r["df"]), opt.build_wet_table(r["blend"],r["df"])

def run():
    df=active_df()
    with st.spinner("Optimizing burden..."):
        res=opt.solve_blend_with_compensation(df,1000,TARGETS)
    st.session_state.result={"status":res[0],"blend":res[1],"cost":res[2],"achieved":res[3],
                             "diagnostics":res[4],"fallback":res[5],"df":df}
    st.session_state.manual_base=dict(res[1]) if res[1] else None
    st.session_state.manual_values=dict(res[1]) if res[1] else None
    st.session_state.runs+=1
    st.session_state.changed=False

# sidebar
with st.sidebar:
    st.markdown("### HOSPET STEELS LIMITED")
    st.markdown('<div class="small">Kalyani Steels × Mukand • Hospet</div>',unsafe_allow_html=True)
    st.markdown("---")
    groups=[("WORKSPACE",["Dashboard"]),("OPERATIONS",["RM Stock & Materials","Recipe & Composition","Manual Burden Control"]),
            ("ANALYSIS",["Process & Cost Parameters","Scenario Analysis"]),("REPORTING",["Reports"]),("SYSTEM",["Upload & Settings"])]
    for head,items in groups:
        st.markdown(f'<div class="panel-title" style="margin-top:.7rem">{head}</div>',unsafe_allow_html=True)
        for item in items:
            if st.button(item,key="nav_"+item,use_container_width=True,type="primary" if st.session_state.nav==item else "secondary"):
                st.session_state.nav=item; st.rerun()
    st.markdown("---")
    d=st.session_state.df
    st.markdown(f'<div class="small"><b>DATA</b><br>{st.session_state.source}<br>{sum(d.Material_Role.astype(str).str.contains("Primary").fillna(False)) if "Material_Role" in d else len(d)} materials<br><br><b>MODEL</b><br>v32 • Streamlit</div>',unsafe_allow_html=True)

def header(title,subtitle):
    st.markdown('<div class="eyebrow">HOSPET ALLOY STEEL PLANT</div>',unsafe_allow_html=True)
    st.title(title); st.markdown(f'<div class="small">{subtitle}</div>',unsafe_allow_html=True)

def dashboard():
    header("SINTER BURDEN CONTROL","Cost optimization • quality assurance • raw material decision support")
    u1,u2,u3=st.columns([2.2,1.2,1])
    with u1:
        f=st.file_uploader("MASTER CHEMISTRY EXCEL",type=["xlsx"],key="dash_upload")
        if f:
            try:
                newdf=opt.load_master_excel(f)
                st.success(f"{len(newdf)} materials loaded from {f.name}")
                if st.button("ACTIVATE MASTER",type="primary",use_container_width=True):
                    st.session_state.df=newdf
                    st.session_state.source=f.name
                    st.session_state.availability={m:False if str(newdf.loc[m,"Material_Role"])=="Alternative_Iron_Ore" else True for m in newdf.index}
                    st.session_state.result=None; st.session_state.manual_base=None; st.session_state.manual_values=None; st.session_state.changed=False
                    st.rerun()
            except Exception as e: st.error(f"Excel validation failed: {e}")
    with u2:
        st.markdown(kpi("ACTIVE MASTER",st.session_state.source,"single source of truth","good"),unsafe_allow_html=True)
    with u3:
        if st.button("🚀 RUN OPTIMIZER",type="primary",use_container_width=True):
            run(); st.rerun()
    if st.session_state.changed: st.warning("Inputs changed. Run optimizer to apply the current availability/prices/limits.")

    r=st.session_state.result
    if r and r["blend"]:
        total=sum(r["blend"].values()); cost=sum(r["blend"][m]*r["df"].loc[m,"Price_Rs_t"]/1000 for m in r["blend"])
        a=r["achieved"]; qok=all(quality_status(a).values())
        iol=100*r["blend"].get("IOL_Fines",0)/total if total else 0
        bf=100*r["blend"].get("BF_Returns",0)/total if total else 0
        cards=st.columns(7)
        vals=[("OPTIMIZED COST",f"₹{cost:,.2f}/t","raw material","blue"),
              ("TOTAL BURDEN",f"{total:,.2f} kg/t","actual charged burden","good"),
              ("Fe",f"{a['Fe']:.3f}%","53.7–54.3","warn"),
              ("QUALITY","PASS" if qok else "REVIEW","chemistry gate","good" if qok else "bad"),
              ("IOL FINES",f"{iol:.2f}%","target 8%","good" if abs(iol-8)<.01 else "bad"),
              ("BF RETURNS",f"{bf:.2f}%","target 17%","good" if abs(bf-17)<.01 else "bad"),
              ("ALT ORE","USED" if any(r["blend"].get(m,0)>1e-6 and "Alternative" in str(r["df"].loc[m,"Material_Role"]) for m in r["df"].index) else "NOT USED","contingency","blue")]
        for c,(l,v,s,k) in zip(cards,vals): c.markdown(kpi(l,v,s,k),unsafe_allow_html=True)

        st.markdown('<div class="panel"><div class="panel-title">CHEMISTRY CONSTRAINTS / ACHIEVED</div></div>',unsafe_allow_html=True)
        st.dataframe(quality_table(a),hide_index=True,use_container_width=True,height=310)

    # full width input table
    st.markdown('<div class="panel"><div class="panel-title">RAW MATERIAL INPUTS — FULL WIDTH</div><div class="small">All chemistry, moisture, price, stock and technical limits come from the same uploaded master. Availability is controlled here. Alternative ores are OFF by default.</div></div>',unsafe_allow_html=True)
    df=st.session_state.df.reset_index()
    if "Material_Role" not in df: df["Material_Role"]="Other"
    view=df[["Material","Material_Role","Group","Fe","SiO2","Al2O3","CaO","MgO","LOI","Moisture_Pct","Price_Rs_t","Available_Tonnes","Tech_Min","Tech_Max"]].copy()
    view.rename(columns={"Material_Role":"Role","SiO2":"SiO₂ %","Al2O3":"Al₂O₃ %","Price_Rs_t":"Price ₹/t","Available_Tonnes":"RM Stock t","Tech_Min":"Tech Min","Tech_Max":"Tech Max","Moisture_Pct":"Moisture %"},inplace=True)
    view["Available / Include"]=[st.session_state.availability.get(m,True) for m in view.Material]
    edited=st.data_editor(view,key="master_editor",hide_index=True,use_container_width=True,height=min(720,max(360,46*len(view)+45)),
                          disabled=["Material","Role","Group","Fe","SiO₂ %","Al₂O₃ %","CaO","MgO","LOI","Moisture %"],
                          column_config={"Available / Include":st.column_config.CheckboxColumn("Available / Include",width="small"),
                                         "Price ₹/t":st.column_config.NumberColumn(format="₹ %.0f"),
                                         "RM Stock t":st.column_config.NumberColumn(format="%.0f"),
                                         "Tech Min":st.column_config.NumberColumn(format="%.0f"),
                                         "Tech Max":st.column_config.NumberColumn(format="%.0f")})
    if not edited.equals(view):
        for _,row in edited.iterrows():
            m=row.Material
            st.session_state.df.loc[m,"Price_Rs_t"]=float(row["Price ₹/t"])
            st.session_state.df.loc[m,"Available_Tonnes"]=float(row["RM Stock t"])
            st.session_state.df.loc[m,"Tech_Min"]=float(row["Tech Min"])
            st.session_state.df.loc[m,"Tech_Max"]=float(row["Tech Max"])
            st.session_state.availability[m]=bool(row["Available / Include"])
        st.session_state.changed=True

    # optimized dry/wet below input
    if r and r["blend"]:
        dry,wet= result_tables(r)
        st.markdown('<div class="panel"><div class="panel-title">OPTIMIZED BURDEN & COST — DRY / WET</div></div>',unsafe_allow_html=True)
        a,b=st.columns(2,gap="medium")
        with a:
            st.markdown('<div class="panel"><div class="panel-title">DRY BASIS — OPTIMIZED BURDEN & COST</div></div>',unsafe_allow_html=True)
            st.dataframe(dry,hide_index=True,use_container_width=True,height=min(600,42*(len(dry)+1)))
            st.markdown(kpi("TOTAL DRY",f"{dry.iloc[-1]['kg/t']:,.2f} kg/t",f"₹{dry.iloc[-1]['Cost ₹/t']:,.2f}/t raw material","good"),unsafe_allow_html=True)
        with b:
            st.markdown('<div class="panel"><div class="panel-title">WET / AS-RECEIVED BASIS — BURDEN & COST</div></div>',unsafe_allow_html=True)
            st.dataframe(wet,hide_index=True,use_container_width=True,height=min(600,42*(len(wet)+1)))
            st.markdown(kpi("TOTAL WET",f"{wet.iloc[-1]['Burden kg/t']:,.2f} kg/t",f"₹{wet.iloc[-1]['Cost ₹/t']:,.2f}/t procurement","good"),unsafe_allow_html=True)
        s1,s2=st.columns(2)
        s1.markdown(kpi("DRY SPECIFIC CONSUMPTION",f"{dry.iloc[-1]['kg/t']:,.2f} kg/t","optimizer / dry-basis","blue"),unsafe_allow_html=True)
        s2.markdown(kpi("WET SPECIFIC CONSUMPTION",f"{wet.iloc[-1]['Burden kg/t']:,.2f} kg/t","as-received / wet-basis","blue"),unsafe_allow_html=True)

def rm_stock():
    header("RM Stock & Materials","Single master table: chemistry + price + availability + technical limits.")
    df=st.session_state.df.reset_index()
    cols=["Material","Material_Role","Group","Price_Rs_t","Available_Tonnes","Tech_Min","Tech_Max"]
    v=df[cols].copy().rename(columns={"Material_Role":"Role","Price_Rs_t":"Price ₹/t","Available_Tonnes":"RM Stock t","Tech_Min":"Tech Min","Tech_Max":"Tech Max"})
    v["Available / Include"]=[st.session_state.availability.get(m,True) for m in v.Material]
    ed=st.data_editor(v,hide_index=True,use_container_width=True,height=650,disabled=["Material","Role","Group"])
    if not ed.equals(v):
        for _,row in ed.iterrows():
            m=row.Material; st.session_state.df.loc[m,"Price_Rs_t"]=float(row["Price ₹/t"]); st.session_state.df.loc[m,"Available_Tonnes"]=float(row["RM Stock t"]); st.session_state.df.loc[m,"Tech_Min"]=float(row["Tech Min"]); st.session_state.df.loc[m,"Tech_Max"]=float(row["Tech Max"]); st.session_state.availability[m]=bool(row["Available / Include"])
        st.session_state.changed=True

def recipe():
    header("Recipe & Composition","Optimized recipe and material contribution on one page.")
    r=st.session_state.result
    if not r or not r["blend"]: st.info("Run the optimizer first."); return
    dry,wet=result_tables(r)
    st.dataframe(dry,hide_index=True,use_container_width=True,height=620)
    c1,c2=st.columns(2)
    with c1: st.markdown(kpi("DRY BURDEN",f"{dry.iloc[-1]['kg/t']:,.2f} kg/t","100% actual burden","good"),unsafe_allow_html=True)
    with c2: st.markdown(kpi("DRY MATERIAL COST",f"₹{dry.iloc[-1]['Cost ₹/t']:,.2f}/t","raw materials only","blue"),unsafe_allow_html=True)

def manual():
    header("Manual Burden Control","Table-based practical simulation. Optimized baseline is frozen; changing one material automatically redistributes the remaining adjustable burden.")
    r=st.session_state.result
    if not r or not r["blend"]: st.info("Run optimizer first."); return
    base=dict(st.session_state.manual_base or r["blend"]); df=r["df"]
    if st.button("↩ RESET TO OPTIMIZED BASELINE",use_container_width=True):
        st.session_state.manual_values=dict(base); st.rerun()
    vals=dict(st.session_state.manual_values or base)
    total=sum(base.values())
    rows=[]
    for m,q in base.items():
        pct=q/total*100 if total else 0
        role=str(df.loc[m].get("Material_Role",""))
        fixed=m in ("IOL_Fines","BF_Returns") or df.loc[m,"Group"] in ("Recycle","Fuel")
        rows.append({"Raw Material":m,"Group":df.loc[m,"Group"],"Optimized kg/t":q,"Optimized %":pct,
                     "Change kg/t":0.0,"Change %":0.0,"New kg/t":vals.get(m,q),"New %":vals.get(m,q)/total*100 if total else 0,
                     "Fixed":fixed})
    ed=st.data_editor(pd.DataFrame(rows),hide_index=True,use_container_width=True,height=650,
                      disabled=["Raw Material","Group","Optimized kg/t","Optimized %","New kg/t","New %","Fixed"],
                      column_config={"Change kg/t":st.column_config.NumberColumn("Change kg/t",step=.1,format="%.2f"),
                                     "Change %":st.column_config.NumberColumn("Change %",step=.1,format="%.2f")})
    # Apply only explicit changes; one/multiple changed rows are compensated by unchanged adjustable materials.
    requested={}; changed=[]
    for _,row in ed.iterrows():
        m=row["Raw Material"]
        if bool(row["Fixed"]): continue
        dk=float(row["Change kg/t"]); dp=float(row["Change %"])
        if abs(dp)>1e-9: dk += base[m]*dp/100
        if abs(dk)>1e-9: requested[m]=base[m]+dk; changed.append(m)
    if requested:
        adjusted=opt.redistribute_manual(base,df,requested)
        st.session_state.manual_values=adjusted
    else: adjusted=vals
    adj_df=opt.build_result_table(adjusted,df)
    a=opt.compute_achieved(adjusted,df,1000)
    cost=sum(adjusted[m]*df.loc[m,"Price_Rs_t"]/1000 for m in adjusted)
    c=st.columns(4)
    c[0].markdown(kpi("BASE COST",f"₹{sum(base[m]*df.loc[m,'Price_Rs_t']/1000 for m in base):,.2f}/t","optimized","blue"),unsafe_allow_html=True)
    c[1].markdown(kpi("PRACTICAL COST",f"₹{cost:,.2f}/t","manual scenario","good"),unsafe_allow_html=True)
    c[2].markdown(kpi("BURDEN",f"{sum(adjusted.values()):,.2f} kg/t","should remain at baseline","blue"),unsafe_allow_html=True)
    c[3].markdown(kpi("QUALITY","PASS" if all(opt.quality_checks(a,TARGETS).values()) else "REVIEW","manual scenario","good" if all(opt.quality_checks(a,TARGETS).values()) else "bad"),unsafe_allow_html=True)
    st.dataframe(adj_df,hide_index=True,use_container_width=True,height=620)

def process():
    header("Process & Cost Parameters","O&M and coke/FeO controls.")
    st.session_state.om_cost=st.number_input("O&M Cost ₹/t",min_value=0.0,value=float(st.session_state.om_cost),step=50.0)
    st.info("O&M is added for reporting and does not change the raw-material optimizer objective.")
    r=st.session_state.result
    if r and r["blend"]:
        diag=opt.coke_diagnostic(r["blend"],r["df"])
        c=st.columns(3); c[0].metric("Predicted FeO",f"{diag['FeO']:.2f}%"); c[1].metric("Thermal Surplus",f"{diag['Thermal Surplus']:,.0f}"); c[2].metric("Firing Ratio",f"{diag['Firing Ratio']:.3f}")

def scenario():
    header("Scenario Analysis","One-at-a-time material shortage stress testing and constraint pressure.")
    r=st.session_state.result
    if not r or not r["blend"]: st.info("Run optimizer first."); return
    if st.button("▶ RUN MATERIAL SHORTAGE SCENARIOS",type="primary"):
        results=[]
        base_cost=sum(r["blend"][m]*r["df"].loc[m,"Price_Rs_t"]/1000 for m in r["blend"])
        for mat in r["df"].index:
            if r["df"].loc[mat,"Available_Tonnes"]<=0: continue
            sdf=r["df"].copy(); sdf.loc[mat,"Available_Tonnes"]=0
            # use a fresh solve; optimizer creates a fresh LP and unique constraint names
            res=opt.solve_blend_with_compensation(sdf,1000,TARGETS,scenario_tag=f"shortage_{opt.safe_name(mat)}")
            results.append({"Missing Material":mat,"Group":sdf.loc[mat,"Group"],"Status":res[0],
                            "Cost ₹/t":round(res[2],2) if res[2] is not None else None,
                            "Cost Impact ₹/t":round(res[2]-base_cost,2) if res[2] is not None else None,
                            "Fe %":round(res[3].get("Fe",np.nan),3) if res[3] else None,
                            "SiO₂ %":round(res[3].get("SiO2",np.nan),3) if res[3] else None})
        st.dataframe(pd.DataFrame(results),hide_index=True,use_container_width=True,height=650)
    st.markdown('<div class="panel"><div class="panel-title">CONSTRAINT PRESSURE</div></div>',unsafe_allow_html=True)
    if r["achieved"]: st.dataframe(quality_table(r["achieved"]),hide_index=True,use_container_width=True)

def reports():
    header("Reports & Export","Latest optimized recipe with dry and wet cost contribution.")
    r=st.session_state.result
    if not r or not r["blend"]: st.info("Run optimizer first."); return
    dry,wet=result_tables(r)
    st.dataframe(dry,hide_index=True,use_container_width=True,height=620)
    st.download_button("⬇ DOWNLOAD DRY REPORT",dry.to_csv(index=False).encode(),"sinter_optimization_dry.csv","text/csv",use_container_width=True)

def settings():
    header("Upload & Settings","Single master workbook management.")
    f=st.file_uploader("UPLOAD MASTER CHEMISTRY EXCEL",type=["xlsx"],key="settings_upload")
    if f:
        try:
            df=opt.load_master_excel(f); st.success(f"Validated {len(df)} materials.")
            if st.button("ACTIVATE MASTER",type="primary"):
                st.session_state.df=df; st.session_state.source=f.name
                st.session_state.availability={m:False if str(df.loc[m,"Material_Role"])=="Alternative_Iron_Ore" else True for m in df.index}
                st.session_state.result=None; st.session_state.changed=False; st.rerun()
        except Exception as e: st.error(str(e))
    st.info("One workbook is the single source of truth. Uploading never runs the optimizer automatically.")

pages={"Dashboard":dashboard,"RM Stock & Materials":rm_stock,"Recipe & Composition":recipe,"Manual Burden Control":manual,"Process & Cost Parameters":process,"Scenario Analysis":scenario,"Reports":reports,"Upload & Settings":settings}
pages[st.session_state.nav]()
st.markdown('<div class="footer">Sinter Burden Control • Hospet Alloy Steel Plant • Production decision-support interface</div>',unsafe_allow_html=True)
