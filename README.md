# Hospet Steels — Sinter Cost Optimisation v32

Run:
```bash
pip install -r requirements.txt
streamlit run app.py
```

Key changes:
- One uploaded Excel is the single raw-material source.
- Full-width raw-material input table.
- Chemistry constraints immediately below KPI cards.
- Dry and wet optimized burden/cost tables below inputs, side-by-side.
- Dry and wet specific consumption shown below the tables.
- IOL Fines = exactly 8% of total charged burden including coke.
- BF Returns = exactly 17% of total charged burden including coke.
- Total burden is not forced to 1000 kg/t.
- Material cost uses `sum(kg/t × ₹/t / 1000)`.
- Availability toggles gate optimization; alternatives are OFF by default.
- Manual burden control is table-based, not sliders.
- Manual changes redistribute remaining adjustable materials while keeping the optimized baseline separate.
- Scenario analysis creates a fresh LP for every shortage scenario, avoiding overlapping PuLP constraint names.
- No matplotlib / google.colab imports in the Streamlit backend.
