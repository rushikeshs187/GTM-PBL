# GTM Global Trade & Logistics Dashboard — Sri Lanka Focus (Refined + Integrated Map)
# Live trade (UN Comtrade), FX, Landed Cost, Route & Lead Time (map), Presets, Tariff Helper, Packing
# Design: No sidebar. Top control ribbon. KPI row. Charts + Map side-by-side. Error-hardened.

import io
import json
import math
import requests
import pandas as pd
import numpy as np
import streamlit as st
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# Optional chart library (fallback to built-ins if missing)
try:
    import plotly.express as px
except Exception:
    px = None

st.set_page_config(page_title="GTM — Global Trade & Logistics (Sri Lanka)", layout="wide", page_icon="📦")

# =================== Styles ===================
st.markdown("""
<style>
:root { --bg:#0b0f14; --panel:#101621; --muted:#9fb0c4; --ink:#e7edf7; --border:#1e2b3c; --primary:#60a5fa; --accent:#a78bfa; }
section.main > div { padding-top: .75rem !important }
.card { background: linear-gradient(180deg, rgba(21,30,48,.94), rgba(12,17,28,.92));
        border:1px solid var(--border); padding:16px; border-radius:14px; box-shadow:0 12px 40px rgba(0,0,0,.35) }
.kpi { display:grid; grid-template-columns: repeat(6, minmax(0,1fr)); gap:.6rem }
.kpi .box { background:#0f172a; border:1px solid #1e293b; border-radius:12px; padding:.9rem 1rem; height:100% }
.kpi h3 { margin:0; font-size:1.15rem }
.kpi p { margin:0; font-size:.8rem; color:#94a3b8 }
label { font-size:.8rem; color:var(--muted); margin-bottom:4px; display:block }
input, select, textarea { width:100%; padding:.55rem .7rem; border-radius:10px; border:1px solid var(--border);
                          background:#0d1422; color:#e7edf7 }
.stButton>button { width:100%; background:linear-gradient(135deg, var(--primary), var(--accent));
                   color:white; border:none; font-weight:700; padding:.6rem .9rem; border-radius:10px }
hr.soft { border:0; border-top:1px solid var(--border); margin:.8rem 0 }
</style>
""", unsafe_allow_html=True)

# =================== Constants & Defaults ===================
UN_COMTRADE = "https://comtradeplus.un.org/api/get"
FX_URL = "https://api.exchangerate.host/latest"
DEFAULT_HS = "300431"  # Insulin pens (retail)

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

def set_defaults():
    st.session_state.setdefault("d_hs", DEFAULT_HS)
    st.session_state.setdefault("d_incoterm", "CIF")
    st.session_state.setdefault("d_fob", 20000.0)
    st.session_state.setdefault("d_freight", 2500.0)
    st.session_state.setdefault("d_ins_pct", 1.0)
    st.session_state.setdefault("d_ins_base", "FOB")
    st.session_state.setdefault("d_duty_pct", 0.0)
    st.session_state.setdefault("d_vat_pct", 8.0)
    st.session_state.setdefault("d_broker", 300.0)
    st.session_state.setdefault("d_dray", 120.0)
    st.session_state.setdefault("d_fx_note", "ISFTA concession may apply — verify on MACMAP")
set_defaults()

# =================== Helpers ===================
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
    """UN Comtrade+: type=C, freq=A, px=HS, rg: 1=imports / 2=exports."""
    params = {"type":"C","freq":"A","px":"HS","ps":years,"r":reporter,"p":"all","rg":flow,"cc":hs}
    try:
        r = requests.get(UN_COMTRADE, params=params, timeout=60)
        r.raise_for_status()
        return pd.DataFrame(r.json().get("dataset", []))
    except Exception:
        # Safe fallback demo (typo fixed)
        data = [
            {"period": 2019, "ptTitle": "India",   "TradeValue": 12000000, "NetWeight": 100000},
            {"period": 2019, "ptTitle": "Denmark", "TradeValue":  6000000, "NetWeight":  40000},
            {"period": 2020, "ptTitle": "India",   "TradeValue": 13000000, "NetWeight": 110000},
            {"period": 2020, "ptTitle": "Denmark", "TradeValue":  5000000, "NetWeight":  38000},
            {"period": 2021, "ptTitle": "India",   "TradeValue": 16000000, "NetWeight": 120000},
            {"period": 2021, "ptTitle": "Denmark", "TradeValue":  7000000, "NetWeight":  46000},
            {"period": 2022, "ptTitle": "India",   "TradeValue": 20000000, "NetWeight": 140000},
            {"period": 2022, "ptTitle": "Denmark", "TradeValue":  9000000, "NetWeight":  52000},
            {"period": 2023, "ptTitle": "India",   "TradeValue": 24000000, "NetWeight": 160000},
            {"period": 2023, "ptTitle": "Denmark", "TradeValue": 11000000, "NetWeight":  60000},
        ]
        return pd.DataFrame(data)

def pick_col(df: pd.DataFrame, names, fill=None):
    """Return the first existing column in names; else a Series filled with `fill` or empty Series."""
    for n in names:
        if n in df.columns:
            return df[n]
    if fill is None:
        return pd.Series(dtype=float)
    return pd.Series([fill] * len(df))

# Geocoding (cached + rate limited)
geolocator = Nominatim(user_agent="gtm_dashboard/1.0 (edu)")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1, swallow_exceptions=True)

@st.cache_data
def geocode_point(q: str):
    if not q: return None
    loc = geocode(q)
    if not loc: return None
    return (loc.latitude, loc.longitude, loc.address)

@st.cache_data
def haversine_km(a, b):
    R = 6371.0
    lat1, lon1 = np.radians(a[0]), np.radians(a[1])
    lat2, lon2 = np.radians(b[0]), np.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(h))

def cagr(first, last, years_count):
    if first <= 0 or years_count <= 0:
        return 0.0
    return (last/first)**(1/years_count) - 1

def safe_line(df, x, y, title):
    if px is not None:
        return st.plotly_chart(px.line(df, x=x, y=y, markers=True, title=title), use_container_width=True)
    st.subheader(title); st.line_chart(df.set_index(x)[y])

def safe_bar(df, x, y, title, horizontal=False):
    if px is not None:
        fig = px.bar(df, x=x, y=y, title=title, orientation="h" if horizontal else "v")
        return st.plotly_chart(fig, use_container_width=True)
    st.subheader(title)
    st.bar_chart(df.set_index(y if horizontal else x)[x if horizontal else y])

# =================== Header ===================
st.markdown("<h2 style='margin:0'>GTM Global Trade & Logistics Dashboard — Sri Lanka Focus</h2>", unsafe_allow_html=True)
st.caption("Live trade • FX • Landed cost • Routes (map) • Packing • Presets")

# =================== Top Control Ribbon (no sidebar) ===================
fx_live = fetch_fx()
fx_rate_live = float(fx_live.get("LKR", 0) or 0)

st.markdown("<div class='card'>", unsafe_allow_html=True)
# Row 1 controls
r1c1, r1c2, r1c3, r1c4 = st.columns([1.6, 1, 1, .9])
with r1c1:
    st.markdown("<label>HS code (6-digit)</label>", unsafe_allow_html=True)
    hs_val = st.text_input("hs6", value=st.session_state["d_hs"], key="w_hs", label_visibility="collapsed")
with r1c2:
    st.markdown("<label>Reporter</label>", unsafe_allow_html=True)
    reporter_name = st.selectbox("reporter", list(REPORTERS.keys()), index=0, key="w_reporter", label_visibility="collapsed")
    reporter = REPORTERS[reporter_name]
with r1c3:
    st.markdown("<label>Flow</label>", unsafe_allow_html=True)
    flow = st.selectbox("flow", ["Imports","Exports"], index=0, key="w_flow", label_visibility="collapsed")
    flow_code = "1" if flow == "Imports" else "2"
with r1c4:
    st.markdown("<label>Years</label>", unsafe_allow_html=True)
    years = st.selectbox("years", ["2019,2020,2021,2022,2023","2020,2021,2022,2023,2024","2018,2019,2020,2021,2022"],
                         index=0, key="w_years", label_visibility="collapsed")

# Row 2 controls
r2c1, r2c2, r2c3, r2c4, r2c5 = st.columns([1, 1, 1, .8, .8])
with r2c1:
    st.markdown("<label>USD→LKR override (optional)</label>", unsafe_allow_html=True)
    fx_override = st.number_input("fx", min_value=0.0, step=0.01, value=0.0, key="w_fx", label_visibility="collapsed")
    fx_use = fx_override if fx_override > 0 else fx_rate_live
with r2c2:
    st.markdown("<label>Origin (city/airport/port)</label>", unsafe_allow_html=True)
    origin_q = st.text_input("origin", value="Bengaluru BLR", key="w_origin", label_visibility="collapsed")
with r2c3:
    st.markdown("<label>Destination (city/port)</label>", unsafe_allow_html=True)
    dest_q   = st.text_input("dest", value="Colombo CMB", key="w_dest", label_visibility="collapsed")
with r2c4:
    st.markdown("<label>Mode</label>", unsafe_allow_html=True)
    mode = st.selectbox("mode", ["Air","Sea","Road"], index=0, key="w_mode", label_visibility="collapsed")
with r2c5:
    st.markdown("<label>Preset</label>", unsafe_allow_html=True)
    preset_name = st.selectbox("Preset", list(PRESETS.keys()), index=0, key="w_preset", label_visibility="collapsed")
    if st.button("Apply preset"):
        p = PRESETS[preset_name]
        # Update default keys, then rerun (never write to widget keys directly)
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
        st.success("Preset applied.")
        st.rerun()
st.markdown("</div>", unsafe_allow_html=True)

# =================== Fetch & Normalize Trade Data ===================
df = fetch_comtrade(reporter=reporter, flow=flow_code, years=years, hs=hs_val)

period  = pick_col(df, ["period", "yr", "Time"])
partner = pick_col(df, ["ptTitle", "partner", "Partner"], fill="World")
value   = pick_col(df, ["TradeValue", "PrimaryValue", "value"], fill=0)
kg      = pick_col(df, ["NetWeight", "netWgt"], fill=0)

if df.empty:
    ndf = pd.DataFrame(columns=["year","partner","value_usd","kg"])
else:
    ndf = pd.DataFrame({
        "year": pd.to_numeric(period, errors="coerce"),
        "partner": partner.astype(str),
        "value_usd": pd.to_numeric(value, errors="coerce"),
        "kg": pd.to_numeric(kg, errors="coerce"),
    }).dropna(subset=["year","value_usd"]).fillna(0)

trend = ndf.groupby("year")["value_usd"].sum().reset_index() if not ndf.empty else pd.DataFrame(columns=["year","value_usd"])
partners_df = (ndf.groupby("partner")["value_usd"].sum()
               .reset_index().sort_values("value_usd", ascending=False)) if not ndf.empty else pd.DataFrame(columns=["partner","value_usd"])
unit_vals = (ndf.groupby("year").apply(lambda g: (g["value_usd"].sum() / max(1.0, g["kg"].sum())))
             .reset_index(name="usd_per_kg")) if not ndf.empty else pd.DataFrame(columns=["year","usd_per_kg"])

# Stats
total_trade = float(trend["value_usd"].sum()) if not trend.empty else 0.0
_top = partners_df.iloc[0] if not partners_df.empty else pd.Series({"partner":"—","value_usd":0})
years_sorted = sorted(trend["year"].tolist()) if not trend.empty else []
yoy = 0.0; cagr_val = 0.0
if len(years_sorted) >= 2:
    first = float(trend.loc[trend["year"]==years_sorted[0], "value_usd"].values[0])
    last  = float(trend.loc[trend["year"]==years_sorted[-1], "value_usd"].values[0])
    diffs = []
    for i in range(1, len(years_sorted)):
        prev = float(trend.loc[trend["year"]==years_sorted[i-1], "value_usd"].values[0])
        cur  = float(trend.loc[trend["year"]==years_sorted[i],   "value_usd"].values[0])
        diffs.append((cur - prev) / max(1.0, prev))
    yoy = float(np.mean(diffs)) if diffs else 0.0
    cagr_val = cagr(first, last, len(years_sorted)-1)

# =================== Route & Lead Time (always integrated) ===================
lead_time_days = 0.0
o_pt = geocode_point(origin_q)
d_pt = geocode_point(dest_q)
dist_km = None
if o_pt and d_pt:
    a, b = (o_pt[0], o_pt[1]), (d_pt[0], d_pt[1])
    dist_km = float(haversine_km(a, b))
    speed = 800 if mode == "Air" else (35*24 if mode == "Sea" else 60)  # km/h
    handling = 0.8 if mode == "Air" else (1.5 if mode == "Sea" else 0.2)
    clearance = 0.8 if mode == "Air" else (2.0 if mode == "Sea" else 0.5)
    local = 0.3 if mode == "Air" else (0.5 if mode == "Sea" else 0.2)
    lead_time_days = (dist_km / speed) / 24 + handling + clearance + local

# =================== KPI Row ===================
st.markdown('<div class="kpi">', unsafe_allow_html=True)
st.markdown(f'<div class="box"><p>Total Trade (USD)</p><h3>{total_trade:,.0f}</h3></div>', unsafe_allow_html=True)
st.markdown(f'<div class="box"><p>Top Partner</p><h3>{_top["partner"]} ({float(_top["value_usd"]):,.0f})</h3></div>', unsafe_allow_html=True)
st.markdown(f'<div class="box"><p># Partners</p><h3>{partners_df.shape[0]}</h3></div>', unsafe_allow_html=True)
st.markdown(f'<div class="box"><p>Avg YoY Growth</p><h3>{yoy*100:.1f}%</h3></div>', unsafe_allow_html=True)
st.markdown(f'<div class="box"><p>CAGR (period)</p><h3>{cagr_val*100:.1f}%</h3></div>', unsafe_allow_html=True)
st.markdown(f'<div class="box"><p>FX USD→LKR</p><h3>{(fx_use or 0):.2f}</h3></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# =================== Charts + Map ===================
left, right = st.columns([1.1, .9], gap="small")

with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    tabs = st.tabs(["Trend", "Partner Share", "Unit Values", "Data"])
    with tabs[0]:
        if not trend.empty:
            safe_line(trend, "year", "value_usd", "Total Trade (USD)")
        else:
            st.info("No data for selected filters.")
    with tabs[1]:
        if not partners_df.empty:
            safe_bar(partners_df.head(12), "value_usd", "partner", "Top Partners (USD)", horizontal=True)
        else:
            st.info("No partner data.")
    with tabs[2]:
        if not unit_vals.empty:
            safe_line(unit_vals, "year", "usd_per_kg", "Unit Value (USD/kg)")
        else:
            st.info("No unit value data.")
    with tabs[3]:
        st.dataframe(ndf, use_container_width=True)
        buf = io.StringIO(); ndf.to_csv(buf, index=False)
        st.download_button("Download raw dataset (CSV)", data=buf.getvalue(), file_name="trade_raw.csv", mime="text/csv")
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='card'><b>Route & Lead Time (Live Map)</b>", unsafe_allow_html=True)
    if o_pt and d_pt:
        fmap = folium.Map(location=[(o_pt[0]+d_pt[0])/2, (o_pt[1]+d_pt[1])/2], zoom_start=4, control_scale=True)
        folium.Marker((o_pt[0], o_pt[1]), tooltip=f"Origin: {origin_q}").add_to(fmap)
        folium.Marker((d_pt[0], d_pt[1]), tooltip=f"Destination: {dest_q}").add_to(fmap)
        folium.PolyLine([(o_pt[0], o_pt[1]), (d_pt[0], d_pt[1])], color="#60a5fa", weight=4).add_to(fmap)
        st_folium(fmap, height=440, use_container_width=True)
        if dist_km is not None:
            st.caption(f"Distance ≈ {dist_km:,.0f} km • Estimated lead time: {lead_time_days:.1f} days ({mode})")
    else:
        st.info("Enter clear locations (e.g., 'Bengaluru BLR', 'Colombo CMB').")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<hr class='soft'/>", unsafe_allow_html=True)

# =================== Landed Cost ===================
st.markdown("<div class='card'><b>In-Depth Landed Cost Calculator & Scenario Analysis</b>", unsafe_allow_html=True)
lc1, lc2, lc3 = st.columns(3)
with lc1:
    incoterm = st.selectbox("Incoterm", ["FOB","CIF","DAP","DDP"],
                            index=["FOB","CIF","DAP","DDP"].index(st.session_state["d_incoterm"]), key="w_inc")
    fob = st.number_input("FOB value (USD)", min_value=0.0, value=st.session_state["d_fob"], step=100.0, key="w_fob")
    insurance_pct = st.number_input("Insurance %", min_value=0.0, value=st.session_state["d_ins_pct"], step=0.1, key="w_ins_pct")
    ins_base = st.selectbox("Insurance base", ["FOB","CIF"],
                            index=["FOB","CIF"].index(st.session_state["d_ins_base"]), key="w_ins_base")
with lc2:
    freight = st.number_input("Freight (USD)", min_value=0.0, value=st.session_state["d_freight"], step=50.0, key="w_freight")
    broker = st.number_input("Brokerage & Handling (USD)", min_value=0.0, value=st.session_state["d_broker"], step=10.0, key="w_broker")
    dray = st.number_input("Last-mile / Drayage (USD)", min_value=0.0, value=st.session_state["d_dray"], step=10.0, key="w_dray")
    vat_pct = st.number_input("VAT / GST %", min_value=0.0, value=st.session_state["d_vat_pct"], step=0.5, key="w_vat")
with lc3:
    duty_pct = st.number_input("Duty %", min_value=0.0, value=st.session_state["d_duty_pct"], step=0.5, key="w_duty")
    shock_freight = st.slider("Freight shock %", 0, 200, 0, key="w_shock_f")
    shock_tariff  = st.slider("Tariff shock (Δ duty %)", 0, 20, 0, key="w_shock_t")

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

if lead_time_days:
    st.caption(f"Lead time currently estimated at {lead_time_days:.1f} days for {mode} route shown above.")
st.markdown("</div>", unsafe_allow_html=True)

# =================== Compare origins ===================
st.markdown("<div class='card'><b>Compare Origins (quick what-if)</b>", unsafe_allow_html=True)
comp_df = pd.DataFrame([
    {"Origin":"India",    "FOB": fob,          "Freight": 2500, "Duty%": 0.0},
    {"Origin":"Denmark",  "FOB": fob*1.05,     "Freight": 5500, "Duty%": max(duty_pct, 2.0)},
    {"Origin":"Singapore","FOB": fob*1.02,     "Freight": 3200, "Duty%": duty_pct},
])
rows=[]
for _, r in comp_df.iterrows():
    rr = landed_cost(r.FOB, r.Freight, insurance_pct, ins_base, r["Duty%"], vat_pct, broker, dray, shock_freight, incoterm)
    rows.append({"Origin":r.Origin, "TLC_USD":rr["total"], "CIF":rr["cif"], "Duty":rr["duty"], "VAT":rr["vat"]})
out = pd.DataFrame(rows)
st.dataframe(out, use_container_width=True)
if px is not None:
    st.plotly_chart(px.bar(out, x="Origin", y="TLC_USD", title="Total Landed Cost by Origin (USD)"), use_container_width=True)
else:
    st.bar_chart(out.set_index("Origin")["TLC_USD"])
st.markdown("</div>", unsafe_allow_html=True)

# =================== Presets / Tariff helper / Packing ===================
boxL, boxR = st.columns([1.1, .9], gap="small")

with boxL:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("**Tariff & NTM helper**")
    st.markdown("- 🧭 [MACMAP — Tariffs & Measures](https://www.macmap.org/)\n- 📊 [TradeMap — Flows](https://www.trademap.org/)\n- 🏛️ [Sri Lanka Customs](http://www.customs.gov.lk/)")
    st.caption("Use HS-6 to start; verify MFN vs FTA (e.g., ISFTA) vs GSP+ on national lines.")
    st.text_area("Tariff/NTM notes", value=st.session_state.get("d_fx_note",""), key="w_notes", height=110)
    st.markdown("</div>", unsafe_allow_html=True)

with boxR:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("**Packing • ULD & Container Capacity**")
    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        carton_l = st.number_input("Carton length (cm)", 1.0, 200.0, 40.0, key="w_pl")
        carton_w = st.number_input("Carton width (cm)",  1.0, 200.0, 30.0, key="w_pw")
        carton_h = st.number_input("Carton height (cm)", 1.0, 200.0, 25.0, key="w_ph")
    with pc2:
        carton_kg = st.number_input("Carton weight (kg)", 0.1, 200.0, 8.0, key="w_pkg")
        layer_gap = st.number_input("Layer gap (cm)", 0.0, 10.0, 0.0, key="w_gap")
        max_stack_h = st.number_input("Max stack height (cm)", 50.0, 250.0, 140.0, key="w_maxh")
    with pc3:
        use_pmc = st.checkbox("Air PMC pallet (243×318×160 cm)", value=True, key="w_pmc")
        use_20  = st.checkbox("Sea 20' (589×235×239 cm)", value=False, key="w_20")
        use_40  = st.checkbox("Sea 40' (1203×235×239 cm)", value=False, key="w_40")

    def pack_on(base_l, base_w, base_h):
        per_row = math.floor(base_l // carton_l) * math.floor(base_w // carton_w)
        layers  = math.floor((min(base_h, max_stack_h)) // (carton_h + layer_gap))
        boxes   = max(0, per_row) * max(0, layers)
        kg_total= boxes * carton_kg
        return boxes, kg_total

    results = []
    if use_pmc: results.append(("PMC pallet", *pack_on(243.0, 318.0, 160.0)))
    if use_20:  results.append(("20' container", *pack_on(589.0, 235.0, 239.0)))
    if use_40:  results.append(("40' container", *pack_on(1203.0, 235.0, 239.0)))
    if results:
        pk_df = pd.DataFrame(results, columns=["Unit","Max cartons","Total kg"])
        st.dataframe(pk_df, use_container_width=True)
        if px is not None:
            st.plotly_chart(px.bar(pk_df, x="Unit", y="Max cartons", title="Packing capacity"), use_container_width=True)
        else:
            st.bar_chart(pk_df.set_index("Unit")["Max cartons"])
    st.markdown("</div>", unsafe_allow_html=True)

# =================== Export Scenario ===================
exp = {
    "hs": hs_val, "reporter": reporter_name, "flow": flow, "years": years,
    "fx_rate_usd_lkr": fx_use,
    "route": {"origin": origin_q, "dest": dest_q, "mode": mode, "distance_km": (None if dist_km is None else round(dist_km,1)),
              "lead_time_days": (None if not lead_time_days else round(lead_time_days,1))},
    "inputs": {"incoterm": incoterm, "fob": fob, "freight": freight, "insurance_pct": insurance_pct,
               "ins_base": ins_base, "duty_pct": duty_pct, "vat_pct": vat_pct, "broker": broker,
               "dray": dray, "shock_freight": shock_freight, "shock_tariff": shock_tariff},
    "outputs": res,
}

row = {}
row.update(exp["inputs"])
if not trend.empty:
    for _, r in trend.iterrows(): row[f"trend_{int(r.year)}"] = float(r.value_usd)
if not partners_df.empty:
    for _, r in partners_df.head(10).iterrows(): row[f"partner_{r.partner}"] = float(r.value_usd)
row["tlc_usd"] = res["total"]; row["cif"] = res["cif"]; row["duty"] = res["duty"]; row["vat"] = res["vat"]
csv_buf = io.StringIO(); pd.DataFrame([row]).to_csv(csv_buf, index=False)

dl1, dl2 = st.columns(2)
with dl1: st.download_button("Download scenario CSV", data=csv_buf.getvalue(), file_name="gtm_scenario.csv", mime="text/csv")
with dl2: st.download_button("Download scenario JSON", data=json.dumps(exp, indent=2), file_name="gtm_scenario.json", mime="application/json")

st.caption("Note: Always verify tariffs/NTMs on official sources (MACMAP, Sri Lanka Customs). Models here are educational.")
