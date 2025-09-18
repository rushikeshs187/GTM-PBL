# GTM Global Trade & Logistics Dashboard — Sri Lanka Focus (Redesigned, No Sidebar)
# • Live UN Comtrade+ (imports/exports) by HS-6
# • FX (USD→LKR) with override
# • In-depth landed cost (Incoterms, insurance base, duty/VAT, brokerage, drayage)
# • Scenario shocks, multi-origin compare
# • Route mapping (folium + Nominatim), packing/ULD calculator
# • Presets & tariff helper
# NOTE: No direct assignment to session_state keys that are bound to widgets.

import io
import json
import math
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

st.set_page_config(page_title="GTM — Global Trade & Logistics (Sri Lanka)", layout="wide", page_icon="📦")

# ============ Styles ============
st.markdown("""
<style>
:root { --bg:#0b0f14; --panel:#101621; --muted:#9fb0c4; --ink:#e7edf7; --border:#1e2b3c; --primary:#60a5fa; --accent:#a78bfa; }
section.main > div { padding-top: 0.8rem !important }
body { background: var(--bg) }
.card { background: linear-gradient(180deg, rgba(21,30,48,.9), rgba(12,17,28,.9)); border:1px solid var(--border); padding:16px; border-radius:14px; box-shadow:0 12px 40px rgba(0,0,0,.35) }
.kpi { display:grid; grid-template-columns: repeat(5, minmax(0,1fr)); gap:.6rem }
.kpi .box { background: #0f172a; border:1px solid #1e293b; border-radius:12px; padding:.9rem 1rem }
.kpi h3 { margin:0; font-size:1.2rem }
.kpi p { margin:0; font-size:.8rem; color:#94a3b8 }
hr.soft { border:0; border-top:1px solid var(--border); margin:.75rem 0 }
.topbar { display:grid; grid-template-columns: 1.5fr 1fr 1fr .8fr; gap:.6rem }
.subbar { display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap:.6rem; margin-top:.6rem }
label { font-size:.8rem; color:var(--muted); margin-bottom:4px; display:block }
input, select, textarea { width:100%; padding:.55rem .7rem; border-radius:10px; border:1px solid var(--border); background:#0d1422; color:var(--ink) }
button, .stButton>button { background: linear-gradient(135deg, var(--primary), var(--accent)); color:white; border:none; font-weight:700; padding:.55rem .9rem; border-radius:10px }
.small { font-size:.85rem; color:#8ca0b4 }
</style>
""", unsafe_allow_html=True)

# ============ Constants ============
UN_COMTRADE = "https://comtradeplus.un.org/api/get"
FX_URL = "https://api.exchangerate.host/latest"
DEFAULT_HS = "300431"
REPORTERS = {
    "Sri Lanka (144)": "144",
    "India (356)": "356",
    "Denmark (208)": "208",
    "United Arab Emirates (784)": "784",
    "Singapore (702)": "702",
    "World (000)": "0",
}
PRESETS = {
    "Insulin pens (retail) — HS 300431": {
        "hs": "300431", "incoterm": "CIF", "fob": 20000.0, "freight": 2500.0, "insurance_pct": 1.0,
        "ins_base": "FOB", "duty_pct": 0.0, "vat_pct": 8.0, "broker": 300.0, "dray": 120.0,
        "note": "ISFTA concession likely for India→Sri Lanka pharma (verify on MACMAP)."
    },
    "Pharma APIs (bulk) — HS 293721 (example)": {
        "hs": "293721", "incoterm": "FOB", "fob": 35000.0, "freight": 1800.0, "insurance_pct": 0.6,
        "ins_base": "FOB", "duty_pct": 2.0, "vat_pct": 8.0, "broker": 350.0, "dray": 150.0,
        "note": "APIs may have different tariff lines/NTMs; confirm exact subheading on MACMAP."
    },
    "Medical devices (misc.) — HS 901890 (example)": {
        "hs": "901890", "incoterm": "CIF", "fob": 25000.0, "freight": 3200.0, "insurance_pct": 1.0,
        "ins_base": "CIF", "duty_pct": 5.0, "vat_pct": 8.0, "broker": 320.0, "dray": 140.0,
        "note": "Devices can face MFN duties unless FTA/GSP applies; check serial/UDI requirements."
    },
}

def ensure_defaults():
    # Keep defaults separate from widget keys to avoid collisions.
    st.session_state.setdefault("d_incoterm", "CIF")
    st.session_state.setdefault("d_fob", 20000.0)
    st.session_state.setdefault("d_freight", 2500.0)
    st.session_state.setdefault("d_ins_pct", 1.0)
    st.session_state.setdefault("d_ins_base", "FOB")
    st.session_state.setdefault("d_duty_pct", 0.0)
    st.session_state.setdefault("d_vat_pct", 8.0)
    st.session_state.setdefault("d_broker", 300.0)
    st.session_state.setdefault("d_dray", 120.0)
    st.session_state.setdefault("d_hs", DEFAULT_HS)
    st.session_state.setdefault("d_fx_note", "ISFTA concession may apply — verify on MACMAP")

ensure_defaults()

# ============ Data helpers ============
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_fx(base="USD", symbols=("LKR","EUR")):
    try:
        r = requests.get(FX_URL, params={"base": base, "symbols": ",".join(symbols)}, timeout=20)
        r.raise_for_status()
        return r.json().get("rates", {})
    except Exception:
        return {}

@st.cache_data(show_spinner=True, ttl=3600)
def fetch_comtrade(reporter="144", flow="1", years="2019,2020,2021,2022,2023", hs=DEFAULT_HS):
    params = {"type":"C","freq":"A","px":"HS","ps":years,"r":reporter,"p":"all","rg":flow,"cc":hs}
    try:
        r = requests.get(UN_COMTRADE, params=params, timeout=60)
        r.raise_for_status()
        return pd.DataFrame(r.json().get("dataset", []))
    except Exception:
        # Fallback demo
        data = [
            {"period":2019,"ptTitle":"India","TradeValue":12000000,"NetWeight":100000},
            {"period":2019,"ptTitle":"Denmark","TradeValue":6000000,"NetWeight":40000},
            {"period":2020,"ptTitle":"India","TradeValue":13000000,"NetWeight":110000},
            {"period":2020,"ptTitle":"Denmark","TradeValue":5000000,"NetWeight":38000},
            {"period":2021,"ptTitle":"India","TradeValue":16000000,"NetWeight":120000},
            {"period":2021,"ptTitle":"Denmark","TradeValue":7000000,"NetWeight":46000},
            {"period":2022,"ptTitle":"India","TradeValue":20000000,"NetWeight":140000},
            {"period":2022,"ptTitle":"Denmark","TradeValue":9000000,"NetWeight":52000},
            {"period":2023,"ptTitle":"India","TradeValue":24000000,"NetWeight":160000},
            {"period":2023,"ptTitle":"Denmark","TradeValue":11000000,"NetWeight":60000},
        ]
        return pd.DataFrame(data)

# Geocoding
geolocator = Nominatim(user_agent="gtm_dashboard/1.0 (edu)")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1, swallow_exceptions=True)

@st.cache_data
def geocode_point(q: str):
    if not q:
        return None
    loc = geocode(q)
    if not loc:
        return None
    return (loc.latitude, loc.longitude, loc.address)

@st.cache_data
def haversine_km(a, b):
    R = 6371.0
    lat1, lon1 = np.radians(a[0]), np.radians(a[1])
    lat2, lon2 = np.radians(b[0]), np.radians(b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(h))

# ============ Header ============
st.markdown("<h2 style='margin:0'>GTM Global Trade & Logistics Dashboard — Sri Lanka Focus</h2>", unsafe_allow_html=True)
st.markdown("<div class='small'>Live trade • FX • Landed cost • Routes • Packing • Presets</div>", unsafe_allow_html=True)

# ============ TOP CONTROL RIBBON (no sidebar) ============
fx_live = fetch_fx()
fx_rate = float(fx_live.get("LKR", 0) or 0)

with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    # Row 1
    colA, colB, colC, colD = st.columns([1.5, 1, 1, 0.8])
    with colA:
        st.markdown("<label>HS code (6-digit, HS2017)</label>", unsafe_allow_html=True)
        hs_val = st.text_input("hs6", value=st.session_state["d_hs"], key="w_hs", label_visibility="collapsed")
    with colB:
        st.markdown("<label>Reporter</label>", unsafe_allow_html=True)
        reporter_name = st.selectbox("reporter", list(REPORTERS.keys()), index=0, key="w_reporter", label_visibility="collapsed")
        reporter = REPORTERS[reporter_name]
    with colC:
        st.markdown("<label>Flow</label>", unsafe_allow_html=True)
        flow = st.selectbox("flow", ["Imports","Exports"], index=0, key="w_flow", label_visibility="collapsed")
        flow_code = "1" if flow == "Imports" else "2"
    with colD:
        st.markdown("<label>Years</label>", unsafe_allow_html=True)
        years = st.selectbox("years", ["2019,2020,2021,2022,2023","2020,2021,2022,2023,2024","2018,2019,2020,2021,2022"],
                             index=0, key="w_years", label_visibility="collapsed")

    # Row 2
    colE, colF, colG, colH = st.columns([1, 1, 1, 0.6])
    with colE:
        st.markdown("<label>USD→LKR override (optional)</label>", unsafe_allow_html=True)
        fx_override = st.number_input("fx", min_value=0.0, step=0.01, value=0.0, key="w_fx", label_visibility="collapsed")
        fx_use = fx_override if fx_override > 0 else fx_rate
    with colF:
        st.markdown("<label>Origin (city/airport/port)</label>", unsafe_allow_html=True)
        origin_q = st.text_input("origin", value="Bengaluru BLR", key="w_origin", label_visibility="collapsed")
    with colG:
        st.markdown("<label>Destination (city/port)</label>", unsafe_allow_html=True)
        dest_q = st.text_input("dest", value="Colombo CMB", key="w_dest", label_visibility="collapsed")
    with colH:
        st.markdown("<label>Mode</label>", unsafe_allow_html=True)
        mode = st.selectbox("mode", ["Air","Sea","Road"], index=0, key="w_mode", label_visibility="collapsed")

    colBTN1, colBTN2 = st.columns([0.25, 0.25])
    with colBTN1: st.button("Refresh trade data", key="btn_refresh")
    with colBTN2: st.button("Plot / Update Route", key="btn_route")

    st.markdown("</div>", unsafe_allow_html=True)

# ============ Fetch trade data ============
df = fetch_comtrade(reporter=reporter, flow=flow_code, years=years, hs=hs_val)
period = df.get("period") or df.get("yr") or df.get("Time")
partner = df.get("ptTitle") or df.get("partner") or df.get("Partner")
value = df.get("TradeValue") or df.get("PrimaryValue") or df.get("value")
kg = df.get("NetWeight") or df.get("netWgt")

ndf = pd.DataFrame({
    "year": pd.to_numeric(period, errors="coerce"),
    "partner": partner.astype(str) if partner is not None else "World",
    "value_usd": pd.to_numeric(value, errors="coerce"),
    "kg": pd.to_numeric(kg, errors="coerce"),
}).dropna(subset=["year","value_usd"]).fillna(0)

trend = ndf.groupby("year")["value_usd"].sum().reset_index()
partners = ndf.groupby("partner")["value_usd"].sum().reset_index().sort_values("value_usd", ascending=False)
unit_vals = ndf.groupby("year").apply(lambda g: (g["value_usd"].sum() / max(1.0, g["kg"].sum()))).reset_index(name="usd_per_kg")

total_trade = trend["value_usd"].sum() if not trend.empty else 0
_top = partners.iloc[0] if not partners.empty else pd.Series({"partner":"—","value_usd":0})
years_sorted = sorted(trend["year"].tolist())
yoy = 0.0
if len(years_sorted) >= 2:
    growths = []
    for i in range(1, len(years_sorted)):
        prev = float(trend.loc[trend["year"]==years_sorted[i-1], "value_usd"].values[0])
        cur = float(trend.loc[trend["year"]==years_sorted[i], "value_usd"].values[0])
        growths.append((cur - prev) / max(1.0, prev))
    yoy = float(np.mean(growths))

# KPIs
with st.container():
    st.markdown('<div class="kpi">', unsafe_allow_html=True)
    st.markdown(f'<div class="box"><p>Total Trade (USD)</p><h3>{total_trade:,.0f}</h3></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="box"><p>Top Partner</p><h3>{_top["partner"]} ({_top["value_usd"]:,.0f})</h3></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="box"><p># Partners</p><h3>{partners.shape[0]}</h3></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="box"><p>Avg YoY Growth</p><h3>{yoy*100:.1f}%</h3></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="box"><p>FX USD→LKR</p><h3>{(fx_use or 0):.2f}</h3></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Charts + Map row
cLeft, cRight = st.columns([1.2, 1])
with cLeft:
    tabs = st.tabs(["Trend", "Partner Share", "Unit Values", "Data"])
    with tabs[0]:
        if not trend.empty:
            st.plotly_chart(px.line(trend, x="year", y="value_usd", markers=True, title="Total Trade (USD)"), use_container_width=True)
        else:
            st.info("No data.")
    with tabs[1]:
        if not partners.empty:
            st.plotly_chart(px.bar(partners.head(12), x="value_usd", y="partner", orientation="h", title="Top Partners (USD)"),
                            use_container_width=True)
            st.caption("Tip: Use the route panel to map a partner country's main gateway.")
        else:
            st.info("No partner data.")
    with tabs[2]:
        if not unit_vals.empty:
            st.plotly_chart(px.line(unit_vals, x="year", y="usd_per_kg", markers=True, title="Unit Value (USD/kg)"),
                            use_container_width=True)
        else:
            st.info("No unit values.")
    with tabs[3]:
        st.dataframe(ndf, use_container_width=True)
        csv_raw = io.StringIO(); ndf.to_csv(csv_raw, index=False)
        st.download_button("Download raw dataset (CSV)", data=csv_raw.getvalue(), file_name="trade_raw.csv", mime="text/csv")

with cRight:
    st.markdown("<div class='card'><b>Route & Lead Time</b>", unsafe_allow_html=True)
    lead_time_days = 0.0
    if st.session_state.get("btn_route"):
        o = geocode_point(origin_q)
        d = geocode_point(dest_q)
        if o and d:
            a, b = (o[0], o[1]), (d[0], d[1])
            dist_km = float(haversine_km(a, b))
            speed = 800 if st.session_state["w_mode"] == "Air" else (35*24 if st.session_state["w_mode"] == "Sea" else 60)
            handling = 0.8 if st.session_state["w_mode"] == "Air" else (1.5 if st.session_state["w_mode"] == "Sea" else 0.2)
            clearance = 0.8 if st.session_state["w_mode"] == "Air" else (2.0 if st.session_state["w_mode"] == "Sea" else 0.5)
            local = 0.3 if st.session_state["w_mode"] == "Air" else (0.5 if st.session_state["w_mode"] == "Sea" else 0.2)
            lead_time_days = (dist_km / speed) / 24 + handling + clearance + local

            fmap = folium.Map(location=[(a[0]+b[0])/2, (a[1]+b[1])/2], zoom_start=4, control_scale=True)
            folium.Marker(a, tooltip=f"Origin: {origin_q}").add_to(fmap)
            folium.Marker(b, tooltip=f"Destination: {dest_q}").add_to(fmap)
            folium.PolyLine([a, b], color="#60a5fa", weight=4).add_to(fmap)
            st_folium(fmap, height=420, use_container_width=True)
            st.caption(f"Distance ≈ {dist_km:,.0f} km • Estimated lead time: {lead_time_days:.1f} days")
        else:
            st.info("Enter clearer locations (e.g., 'Bengaluru BLR', 'Colombo CMB').")
    else:
        st.caption("Click “Plot / Update Route” above to render the map.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<hr class='soft'/>", unsafe_allow_html=True)

# ============ LANDED COST CALC ============

with st.container():
    st.markdown("<div class='card'><b>In-Depth Landed Cost Calculator & Scenario Analysis</b>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        incoterm = st.selectbox("Incoterm", ["FOB","CIF","DAP","DDP"],
                                index=["FOB","CIF","DAP","DDP"].index(st.session_state["d_incoterm"]), key="w_inc")
        fob = st.number_input("FOB value (USD)", min_value=0.0, value=st.session_state["d_fob"], step=100.0, key="w_fob")
        insurance_pct = st.number_input("Insurance %", min_value=0.0, value=st.session_state["d_ins_pct"], step=0.1, key="w_ins_pct")
        ins_base = st.selectbox("Insurance base", ["FOB","CIF"], index=["FOB","CIF"].index(st.session_state["d_ins_base"]), key="w_ins_base")
    with c2:
        freight = st.number_input("Freight (USD)", min_value=0.0, value=st.session_state["d_freight"], step=50.0, key="w_freight")
        broker = st.number_input("Brokerage & Handling (USD)", min_value=0.0, value=st.session_state["d_broker"], step=10.0, key="w_broker")
        dray = st.number_input("Last-mile / Drayage (USD)", min_value=0.0, value=st.session_state["d_dray"], step=10.0, key="w_dray")
        vat_pct = st.number_input("VAT / GST %", min_value=0.0, value=st.session_state["d_vat_pct"], step=0.5, key="w_vat")
    with c3:
        duty_pct = st.number_input("Duty %", min_value=0.0, value=st.session_state["d_duty_pct"], step=0.5, key="w_duty")
        shock_freight = st.slider("Freight shock %", 0, 200, 0, key="w_shock_f")
        shock_tariff = st.slider("Tariff shock (Δ duty %)", 0, 20, 0, key="w_shock_t")

def landed_cost(fob, freight, insurance_pct, ins_base, duty_pct, vat_pct, broker, dray, shock_freight, incoterm):
    freight_final = freight * (1 + shock_freight/100)
    ins_base_val = fob if ins_base == "FOB" else (fob + freight_final)
    insurance = ins_base_val * (insurance_pct/100)
    cif = fob + freight_final + insurance
    duty = cif * (duty_pct/100)
    taxable = cif + duty
    vat = taxable * (vat_pct/100)
    total = taxable + vat + broker + dray
    if incoterm == "FOB":
        total = fob + freight_final + insurance + duty + vat + broker + dray
    return {"freight_final":freight_final, "insurance":insurance, "cif":cif, "duty":duty, "vat":vat, "total":total}

res = landed_cost(fob, freight, insurance_pct, ins_base, duty_pct+shock_tariff, vat_pct, broker, dray, shock_freight, incoterm)

m1, m2, m3, m4, m5, m6 = st.columns(6)
with m1: st.metric("Freight (after shock)", f"${res['freight_final']:,.0f}")
with m2: st.metric("Insurance", f"${res['insurance']:,.0f}")
with m3: st.metric("CIF", f"${res['cif']:,.0f}")
with m4: st.metric("Duty", f"${res['duty']:,.0f}")
with m5: st.metric("VAT", f"${res['vat']:,.0f}")
with m6: st.metric("Total Landed Cost", f"${res['total']:,.0f}")

st.markdown("</div>", unsafe_allow_html=True)

# ============ Multi-origin compare ============
st.markdown("<div class='card'><b>Compare Origins (quick what-if)</b>", unsafe_allow_html=True)
comp_df = pd.DataFrame([
    {"Origin":"India", "FOB": fob,           "Freight": 2500, "Duty%": 0.0},
    {"Origin":"Denmark","FOB": fob*1.05,    "Freight": 5500, "Duty%": max(duty_pct, 2.0)},
    {"Origin":"Singapore","FOB": fob*1.02,  "Freight": 3200, "Duty%": duty_pct},
])
rows=[]
for _, r in comp_df.iterrows():
    rr = landed_cost(r.FOB, r.Freight, insurance_pct, ins_base, r["Duty%"], vat_pct, broker, dray, shock_freight, incoterm)
    rows.append({"Origin":r.Origin, "TLC_USD":rr["total"], "CIF":rr["cif"], "Duty":rr["duty"], "VAT":rr["vat"]})
comp_out = pd.DataFrame(rows)
st.dataframe(comp_out, use_container_width=True)
st.plotly_chart(px.bar(comp_out, x="Origin", y="TLC_USD", title="Total Landed Cost by Origin (USD)"), use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ============ Presets & Tariff Helper & Packing ============
st.markdown("<div class='card'><b>Presets • Tariff Helper • Packing</b>", unsafe_allow_html=True)

cP1, cP2 = st.columns([1.2, 1])
with cP1:
    st.markdown("**Preset selector**")
    preset_name = st.selectbox("Choose preset", list(PRESETS.keys()), index=0, key="w_preset")
    if st.button("Apply preset"):
        p = PRESETS[preset_name]
        # Update default-state keys (NOT widget keys directly) then rerun.
        st.session_state["d_hs"] = p["hs"]
        st.session_state["d_incoterm"] = p["incoterm"]
        st.session_state["d_fob"] = p["fob"]
        st.session_state["d_freight"] = p["freight"]
        st.session_state["d_ins_pct"] = p["insurance_pct"]
        st.session_state["d_ins_base"] = p["ins_base"]
        st.session_state["d_duty_pct"] = p["duty_pct"]
        st.session_state["d_vat_pct"] = p["vat_pct"]
        st.session_state["d_broker"] = p["broker"]
        st.session_state["d_dray"] = p["dray"]
        st.session_state["d_fx_note"] = p["note"]
        st.success("Preset applied. Re-run to see inputs updated.")
        st.rerun()

with cP2:
    st.markdown("**Tariff & NTM helper**")
    st.markdown("- 🧭 [MACMAP — Tariffs & Measures](https://www.macmap.org/)\n- 📊 [TradeMap — Flows](https://www.trademap.org/)\n- 🏛️ [Sri Lanka Customs](http://www.customs.gov.lk/)")
    st.caption("Use HS-6 to start; verify MFN vs FTA (e.g., ISFTA) vs GSP+ on national lines.")
    st.text_area("Your tariff/NTM notes", value=st.session_state.get("d_fx_note",""), key="w_notes", height=100)

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='card'><b>Packing • ULD & Container Capacity</b>", unsafe_allow_html=True)
pc1, pc2, pc3 = st.columns(3)
with pc1:
    carton_l = st.number_input("Carton length (cm)", 1.0, 200.0, 40.0, key="w_pl")
    carton_w = st.number_input("Carton width (cm)", 1.0, 200.0, 30.0, key="w_pw")
    carton_h = st.number_input("Carton height (cm)", 1.0, 200.0, 25.0, key="w_ph")
with pc2:
    carton_kg = st.number_input("Carton weight (kg)", 0.1, 200.0, 8.0, key="w_pkg")
    layer_gap = st.number_input("Layer gap (cm)", 0.0, 10.0, 0.0, key="w_gap")
    max_stack_h = st.number_input("Max stack height (cm)", 50.0, 250.0, 140.0, key="w_maxh")
with pc3:
    use_pmc = st.checkbox("Air PMC pallet (243x318×160 cm)", value=True, key="w_pmc")
    use_20 = st.checkbox("Sea 20' (589×235×239 cm)", value=False, key="w_20")
    use_40 = st.checkbox("Sea 40' (1203×235×239 cm)", value=False, key="w_40")

def pack_on(base_l, base_w, base_h):
    per_row = math.floor(base_l // carton_l) * math.floor(base_w // carton_w)
    layers = math.floor((min(base_h, max_stack_h)) // (carton_h + layer_gap))
    boxes = max(0, per_row) * max(0, layers)
    kg_total = boxes * carton_kg
    return boxes, kg_total

results = []
if use_pmc:
    b,k = pack_on(243.0, 318.0, 160.0); results.append(("PMC pallet", b, k))
if use_20:
    b,k = pack_on(589.0, 235.0, 239.0); results.append(("20' container", b, k))
if use_40:
    b,k = pack_on(1203.0, 235.0, 239.0); results.append(("40' container", b, k))

if results:
    pk_df = pd.DataFrame(results, columns=["Unit","Max cartons","Total kg"])
    st.dataframe(pk_df, use_container_width=True)
    st.plotly_chart(px.bar(pk_df, x="Unit", y="Max cartons", title="Packing capacity"), use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# ============ Export scenario pack ============
exp = {
    "hs": hs_val,
    "reporter": reporter_name,
    "flow": flow,
    "years": years,
    "fx_rate_usd_lkr": fx_use,
    "route": {"origin": origin_q, "dest": dest_q, "mode": mode},
    "inputs": {"incoterm": incoterm, "fob": fob, "freight": freight, "insurance_pct": insurance_pct,
               "ins_base": ins_base, "duty_pct": duty_pct, "vat_pct": vat_pct, "broker": broker,
               "dray": dray, "shock_freight": shock_freight, "shock_tariff": shock_tariff},
    "outputs": res,
}
row = {}
row.update(exp["inputs"])
if not trend.empty:
    for _, r in trend.iterrows():
        row[f"trend_{int(r.year)}"] = float(r.value_usd)
if not partners.empty:
    for _, r in partners.head(10).iterrows():
        row[f"partner_{r.partner}"] = float(r.value_usd)
row["tlc_usd"] = res["total"]; row["cif"] = res["cif"]; row["duty"] = res["duty"]; row["vat"] = res["vat"]
csv_buf = io.StringIO(); pd.DataFrame([row]).to_csv(csv_buf, index=False)

cE1, cE2 = st.columns(2)
with cE1:
    st.download_button("Download scenario CSV", data=csv_buf.getvalue(), file_name="gtm_scenario.csv", mime="text/csv")
with cE2:
    st.download_button("Download scenario JSON", data=json.dumps(exp, indent=2), file_name="gtm_scenario.json", mime="application/json")
