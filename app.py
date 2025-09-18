# app.py — GTM Dashboard (Sri Lanka • HS 3004.31)
# v5.0 — No external APIs. Clean UI, correct Incoterms math, routes & map.

import math, io, json
import pandas as pd
import numpy as np
import streamlit as st
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="GTM — Sri Lanka (HS 3004.31)", layout="wide", page_icon="📦")

# ===================== THEME =====================
def apply_css(theme="Light", compact=False):
    if theme == "Light":
        bg, panel, ink, muted, border = "#f7f9fc", "#ffffff", "#0f172a", "#526581", "#e6ebf2"
        primary, accent = "#2563eb", "#7c3aed"
        card_grad = "linear-gradient(180deg, rgba(255,255,255,.98), rgba(255,255,255,.98))"
        hero_grad  = ("radial-gradient(1200px 400px at 0% -10%, rgba(37,99,235,.10), transparent 60%),"
                      "radial-gradient(1200px 400px at 100% 110%, rgba(124,58,237,.08), transparent 60%),"
                      "linear-gradient(180deg, rgba(255,255,255,.98), rgba(255,255,255,.98))")
        kpi_bg, kpi_bd, input_bg = "#f3f6fb", "#e6ebf2", "#fbfdff"
    else:
        bg, panel, ink, muted, border = "#0b0f14", "#0f1521", "#e7edf7", "#9fb0c4", "#1e2b3c"
        primary, accent = "#60a5fa", "#a78bfa"
        card_grad = "linear-gradient(180deg, rgba(21,30,48,.94), rgba(12,17,28,.92))"
        hero_grad  = ("radial-gradient(1200px 400px at 0% -10%, rgba(96,165,250,.15), transparent 60%),"
                      "radial-gradient(1200px 400px at 100% 110%, rgba(167,139,250,.12), transparent 60%),"
                      "linear-gradient(180deg, rgba(18,26,40,.92), rgba(10,15,25,.92))")
        kpi_bg, kpi_bd, input_bg = "#0f172a", "#1e293b", "#0d1422"

    density = ".5rem .7rem" if compact else ".62rem .8rem"
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    :root {{ --bg:{bg}; --panel:{panel}; --ink:{ink}; --muted:{muted}; --border:{border};
            --primary:{primary}; --accent:{accent}; --kpi-bg:{kpi_bg}; --kpi-bd:{kpi_bd}; --input-bg:{input_bg}; }}
    html, body, .stApp {{ background: var(--bg) !important; }}
    .hero {{ border-radius:18px; padding:26px 24px 18px; background:{hero_grad}; border:1px solid var(--border); box-shadow:0 18px 40px rgba(0,0,0,.07); margin-bottom:12px; }}
    .hero-title {{ font-family:Inter,system-ui; letter-spacing:-.02em; font-weight:800; font-size:clamp(26px,4vw,36px); margin:0; color:var(--ink) }}
    .hero-sub {{ margin-top:6px; color:var(--muted); font-size:14px; }}
    .card {{ background:{card_grad}; border:1px solid var(--border); padding:16px; border-radius:14px; box-shadow:0 10px 34px rgba(0,0,0,.06) }}
    .grid-3 {{ display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap:.6rem }}
    .box {{ background:var(--kpi-bg); border:1px solid var(--kpi-bd); border-radius:12px; padding:.9rem 1rem; height:100% }}
    label {{ font-size:.8rem; color:var(--muted); margin-bottom:4px; display:block }}
    input, select, textarea {{ width:100%; padding:{density}; border-radius:10px; border:1px solid var(--border); background:var(--input-bg); color:var(--ink) }}
    .stButton>button {{ width:100%; background:linear-gradient(135deg, var(--primary), var(--accent)); color:white; border:none; font-weight:700; padding:.6rem .9rem; border-radius:10px }}
    header {{ border-bottom:none !important; }}
    [data-testid="stMetric"] div {{ gap: .15rem; }}
    </style>
    """, unsafe_allow_html=True)

if "theme" not in st.session_state: st.session_state["theme"] = "Light"
if "compact" not in st.session_state: st.session_state["compact"] = False
apply_css(st.session_state["theme"], st.session_state["compact"])

# ===================== HERO =====================
st.markdown("""
<div class="hero">
  <div class="hero-title">GTM — Sri Lanka Trade & Logistics (HS 3004.31 Insulin Pens)</div>
  <div class="hero-sub">No API keys required • Clean visuals • Correct Incoterms engine • Routes & map</div>
</div>
""", unsafe_allow_html=True)

# ===================== TOP BAR (style toggles) =====================
st.markdown("<div class='card'>", unsafe_allow_html=True)
c1, c2, c3 = st.columns([.45,.4,.15])
with c1:
    tsel = st.selectbox("Theme", ["Light","Dark"], index=["Light","Dark"].index(st.session_state["theme"]))
with c2:
    csel = st.checkbox("Compact mode", value=st.session_state["compact"])
with c3:
    st.write("")
    if st.button("Apply"):
        st.session_state["theme"] = tsel
        st.session_state["compact"] = csel
        apply_css(tsel, csel)
st.markdown("</div>", unsafe_allow_html=True)

# ===================== SAMPLE DATA (embedded) =====================
# You can replace this with your CSV later; keep the same columns.
DATA = [
    {"year":2019,"partner":"India","value_usd":12_000_000,"kg":100_000},
    {"year":2019,"partner":"Denmark","value_usd":6_000_000,"kg":40_000},
    {"year":2020,"partner":"India","value_usd":13_000_000,"kg":110_000},
    {"year":2020,"partner":"Denmark","value_usd":5_000_000,"kg":38_000},
    {"year":2021,"partner":"India","value_usd":16_000_000,"kg":120_000},
    {"year":2021,"partner":"Denmark","value_usd":7_000_000,"kg":46_000},
    {"year":2022,"partner":"India","value_usd":20_000_000,"kg":140_000},
    {"year":2022,"partner":"Denmark","value_usd":9_000_000,"kg":52_000},
    {"year":2023,"partner":"India","value_usd":24_000_000,"kg":160_000},
    {"year":2023,"partner":"Denmark","value_usd":11_000_000,"kg":60_000},
]
df = pd.DataFrame(DATA)

# ===================== CONTROLS =====================
st.markdown("<div class='card'>", unsafe_allow_html=True)
left, right = st.columns([1.6, 1.4])

with left:
    c = st.columns([.9,.9,.9])
    with c[0]:
        hs6 = st.text_input("HS code (6-digit)", value="300431")
    with c[1]:
        flow = st.selectbox("Flow", ["Imports","Exports"], index=0)
    with c[2]:
        years = st.multiselect("Years", sorted(df["year"].unique().tolist()), default=sorted(df["year"].unique().tolist()))
with right:
    c = st.columns([1,1,1])
    with c[0]:
        mode = st.selectbox("Mode", ["Air","Sea","Road"], index=0)
    with c[1]:
        origin = st.selectbox("Origin (preset)", [
            "Bengaluru BLR (12.949, 77.668)",
            "Chennai MAA (12.994, 80.170)",
            "Copenhagen CPH (55.629, 12.651)",
            "Mumbai BOM (19.089, 72.865)",
        ], index=0)
    with c[2]:
        dest = st.selectbox("Destination (preset)", [
            "Colombo Port (6.949, 79.844)",
            "Bandaranaike CMB (7.180, 79.884)",
        ], index=0)
st.markdown("</div>", unsafe_allow_html=True)

# ===================== BASIC AGG =====================
use = df[df["year"].isin(years)] if years else df.copy()
trend = use.groupby("year")["value_usd"].sum().reset_index()
partners = use.groupby("partner")["value_usd"].sum().reset_index().sort_values("value_usd", ascending=False)
unit_vals = use.groupby("year").apply(lambda g: g["value_usd"].sum()/max(1.0, g["kg"].sum())).reset_index(name="usd_per_kg")

total_trade = float(trend["value_usd"].sum()) if not trend.empty else 0.0
_top = partners.iloc[0] if not partners.empty else pd.Series({"partner":"—","value_usd":0})
def cagr(a,b,n): return 0.0 if a<=0 or n<=0 else (b/a)**(1/n)-1
ys = sorted(trend["year"].tolist())
avg_yoy = 0.0; cagr_val = 0.0
if len(ys)>=2:
    deltas=[(float(trend.loc[trend["year"]==ys[i],"value_usd"])-float(trend.loc[trend["year"]==ys[i-1],"value_usd"]))/
            max(1.0, float(trend.loc[trend["year"]==ys[i-1],"value_usd"])) for i in range(1,len(ys))]
    avg_yoy = float(np.mean(deltas))
    cagr_val = cagr(float(trend.loc[trend["year"]==ys[0],"value_usd"]), float(trend.loc[trend["year"]==ys[-1],"value_usd"]), len(ys)-1)

# ===================== LAYOUT TABS =====================
ov_tab, trade_tab, cost_tab, map_tab = st.tabs(["Overview", "Trade", "Landed Cost", "Routes & Map"])

# ---------- OVERVIEW ----------
with ov_tab:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"### Snapshot — HS {hs6} ({flow})")
    k1,k2,k3 = st.columns(3)
    with k1: st.metric("Total trade (USD)", f"{total_trade:,.0f}")
    with k2: st.metric("Avg YoY growth", f"{avg_yoy*100:.1f}%")
    with k3: st.metric("CAGR", f"{cagr_val*100:.1f}%")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    colA, colB = st.columns(2)
    with colA:
        if not trend.empty:
            st.bar_chart(trend.set_index("year")["value_usd"], height=260, use_container_width=True)
        else:
            st.info("No trend data for selected years.")
    with colB:
        if not partners.empty:
            st.bar_chart(partners.set_index("partner")["value_usd"], height=260, use_container_width=True)
        else:
            st.info("No partner data for selected years.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- TRADE ----------
with trade_tab:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    tabs = st.tabs(["Unit values (USD/kg)", "Raw data", "Export"])
    with tabs[0]:
        if not unit_vals.empty:
            st.line_chart(unit_vals.set_index("year")["usd_per_kg"], height=260, use_container_width=True)
        else:
            st.info("No unit value series.")
    with tabs[1]:
        st.dataframe(use, use_container_width=True, height=300)
    with tabs[2]:
        buf = io.StringIO(); use.to_csv(buf, index=False)
        st.download_button("Download CSV", data=buf.getvalue(), file_name="hs300431_sample.csv", mime="text/csv")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- LANDED COST (correct Incoterms math) ----------
with cost_tab:
    st.markdown("<div class='card'><b>Incoterms® 2020 calculator — Customs value & total landed cost</b>", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns([1.1,1.1,1.1,1.1])
    with c1:
        incoterm = st.selectbox("Incoterm",
            ["EXW","FCA","FAS","FOB","CFR","CIF","CPT","CIP","DAP","DPU","DDP"], index=5)
        invoice_value = st.number_input(f"Seller invoice value ({incoterm}) — USD", min_value=0.0, value=150000.0, step=100.0)
        qty_units = st.number_input("Units / pieces (optional)", min_value=0, value=10000)
    with c2:
        main_freight = st.number_input("Main carriage freight (USD)", min_value=0.0, value=2000.0, step=50.0)
        origin_charges = st.number_input("Origin charges — not dutiable (USD)", min_value=0.0, value=250.0, step=10.0)
        dest_charges = st.number_input("Destination charges — not dutiable (USD)", min_value=0.0, value=400.0, step=10.0)
    with c3:
        brokerage = st.number_input("Brokerage & regulatory (USD)", min_value=0.0, value=350.0, step=10.0)
        inland = st.number_input("Inland to warehouse (USD)", min_value=0.0, value=300.0, step=10.0)
        other_local = st.number_input("Other local charges (USD)", min_value=0.0, value=0.0, step=10.0)
    with c4:
        ins_mode = st.radio("Insurance input", ["Percent","Amount"], index=0, horizontal=True)
        ins_base = st.selectbox("If %: base", ["Invoice","Invoice+Freight"], index=1)
        ins_pct  = st.number_input("Insurance %", min_value=0.0, value=1.0, step=0.1)
        ins_amt  = st.number_input("Insurance amount (USD)", min_value=0.0, value=0.0, step=10.0)
        seller_pays_import_taxes = st.checkbox("Seller pays import taxes (DDP)", value=(incoterm=="DDP"))

    # Freight/Insurance inclusion by term
    INCOTERM_INCLUDES = {
        # (freight_included_in_seller_quote, insurance_included_in_seller_quote)
        "EXW": (False, False), "FCA": (False, False), "FAS": (False, False), "FOB": (False, False),
        "CFR": (True,  False), "CIF": (True,  True),  "CPT": (True,  False), "CIP": (True,  True),
        "DAP": (True,  False), "DPU": (True,  False), "DDP": (True,  False),
    }
    freight_included, insurance_included = INCOTERM_INCLUDES.get(incoterm, (False, False))

    # Customs value (CIF-equivalent for ad valorem): invoice + (freight if not already included) + (insurance if not already included)
    freight_add_for_customs = 0.0 if freight_included else main_freight
    if insurance_included:
        insurance_add_for_customs = 0.0
    else:
        if ins_mode == "Percent":
            base = invoice_value + (0.0 if freight_included else main_freight) if ins_base=="Invoice+Freight" else invoice_value
            insurance_add_for_customs = base * (ins_pct/100.0)
        else:
            insurance_add_for_customs = ins_amt

    customs_value = invoice_value + freight_add_for_customs + insurance_add_for_customs

    # Taxes
    duty_pct = st.number_input("Import duty % (on customs value)", min_value=0.0, value=0.0, step=0.1)
    other_tax_pct = st.number_input("Other tariff/levy % (on customs value)", min_value=0.0, value=0.0, step=0.1)
    vat_pct = st.number_input("VAT / GST % (on customs value + duty + other)", min_value=0.0, value=0.0, step=0.1)

    duty = customs_value * (duty_pct/100.0)
    other_tax = customs_value * (other_tax_pct/100.0)
    vat_base = customs_value + duty + other_tax
    vat = vat_base * (vat_pct/100.0)

    taxes_payable_by_buyer = 0.0 if seller_pays_import_taxes else (duty + other_tax + vat)

    # Total landed cost (what buyer actually spends)
    total_landed = (
        invoice_value
        + (0.0 if freight_included else main_freight)
        + (0.0 if insurance_included else (insurance_add_for_customs if ins_mode=="Percent" else ins_amt))
        + origin_charges + dest_charges + brokerage + inland + other_local
        + taxes_payable_by_buyer
    )

    # KPIs
    m1,m2,m3,m4 = st.columns(4)
    with m1: st.metric("Customs value (CIF-equiv.)", f"${customs_value:,.0f}")
    with m2: st.metric("Duty", f"${duty:,.0f}")
    with m3: st.metric("VAT / GST", f"${vat:,.0f}")
    with m4: st.metric("Other tariff/levy", f"${other_tax:,.0f}")
    x1,x2,x3,x4 = st.columns(4)
    with x1: st.metric("Buyer insurance outlay", f"${0.0 if insurance_included else (insurance_add_for_customs if ins_mode=='Percent' else ins_amt):,.0f}")
    with x2: st.metric("Buyer freight outlay", f"${0.0 if freight_included else main_freight:,.0f}")
    with x3: st.metric("Taxes payable by buyer", f"${taxes_payable_by_buyer:,.0f}" + (" (DDP seller-paid)" if seller_pays_import_taxes else ""))
    with x4: st.metric("Total Landed Cost (USD)", f"${total_landed:,.0f}")
    if qty_units and qty_units>0:
        st.caption(f"Per-unit landed cost: **${(total_landed/qty_units):,.2f}**")

    # Snapshot export
    snap = {
        "incoterm":incoterm,"invoice_value":invoice_value,"freight_included":freight_included,"insurance_included":insurance_included,
        "customs_value":customs_value,"duty_pct":duty_pct,"other_tax_pct":other_tax_pct,"vat_pct":vat_pct,
        "duty":duty,"other_tax":other_tax,"vat":vat,
        "origin_charges":origin_charges,"dest_charges":dest_charges,"brokerage":brokerage,"inland":inland,"other_local":other_local,
        "taxes_payable_by_buyer":taxes_payable_by_buyer,"total_landed":total_landed,"qty_units":qty_units,
        "per_unit": (total_landed/qty_units if qty_units else None)
    }
    cdl, cdr = st.columns(2)
    with cdl:
        buf_json = json.dumps(snap, indent=2)
        st.download_button("Download snapshot (JSON)", data=buf_json, file_name="landed_cost_snapshot.json", mime="application/json")
    with cdr:
        buf_csv = io.StringIO(); pd.DataFrame([snap]).to_csv(buf_csv, index=False)
        st.download_button("Download snapshot (CSV)", data=buf_csv.getvalue(), file_name="landed_cost_snapshot.csv", mime="text/csv")

# ---------- ROUTES & MAP (no APIs; presets & manual coords) ----------
with map_tab:
    st.markdown("<div class='card'><b>Route & Map (no keys)</b>", unsafe_allow_html=True)

    def parse_point(label: str):
        # label example: "Colombo Port (6.949, 79.844)"
        if "(" in label and ")" in label:
            inside = label.split("(")[1].split(")")[0]
            lat_s, lon_s = [x.strip() for x in inside.split(",")]
            return float(lat_s), float(lon_s)
        return 0.0, 0.0

    a = parse_point(origin); b = parse_point(dest)

    # haversine distance (km)
    def haversine_km(p1, p2):
        R = 6371.0
        lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
        lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
        dlat, dlon = lat2-lat1, lon2-lon1
        h = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
        return 2*R*math.asin(math.sqrt(h))

    dist_km = haversine_km(a, b)
    speed = 800 if mode=="Air" else (35*24 if mode=="Sea" else 60)  # km/h
    handling, clearance, local = (0.8,0.8,0.3) if mode=="Air" else ((1.5,2.0,0.5) if mode=="Sea" else (0.2,0.5,0.2))
    lead_days = (dist_km/speed)/24 + handling + clearance + local

    fmap = folium.Map(location=[(a[0]+b[0])/2, (a[1]+b[1])/2], zoom_start=5, control_scale=True)
    folium.Marker(a, tooltip=f"Origin: {origin}").add_to(fmap)
    folium.Marker(b, tooltip=f"Destination: {dest}").add_to(fmap)
    folium.PolyLine([a,b], color="#2563eb", weight=4).add_to(fmap)
    st_folium(fmap, height=420, use_container_width=True)

    st.caption(f"Distance ≈ {dist_km:,.0f} km • Estimated lead time: {lead_days:.1f} days ({mode})")

st.caption("Educational tool — verify tariffs/NTM rules with official sources (e.g., Sri Lanka Customs).")
