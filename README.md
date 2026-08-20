# Sinter Cost Optimisation — Streamlit Deployment

Run:
streamlit run app.py

Files:
- app.py — Streamlit dashboard
- optimizer.py — v31.9 optimizer backend
- requirements.txt — deployment dependencies

Dashboard layout:
- KPI cards at top
- Chemistry constraint KPI cards immediately below
- Full-width raw-material input table with no inner vertical scrollbar
- Dry burden & cost table on the left
- Wet burden & cost table on the right
- Dry/Wet specific consumption shown as a compact section, not KPI cards
- Manual Burden Control remains table-based and independent from the theoretical baseline
- Scenario Analysis includes material shortage and constraint pressure

Model rules retained:
- IOL Fines = 8% of total burden including Coke
- BF Returns = 17% of total burden including Coke
- Mill Scale = 5–15% when available; 0% when unavailable
- Alternative_Iron_Ore is OFF by default; ON means eligible, not forced
