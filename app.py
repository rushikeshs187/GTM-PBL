# app.py — GTM Global Trade & Logistics Dashboard (Sri Lanka Focus)
# v2 — Precise Incoterms & Landed-Cost engine (EXW, FCA, FAS, FOB, CFR, CIF, CPT, CIP, DAP, DPU, DDP)
# - Insurance: % or fixed amount; selectable base (Invoice or Invoice+Freight)
# - Accurate customs value (CIF-equivalent) -> duty -> VAT; DDP logic (seller pays import taxes)
# - Live signals: traffic ETA (Google/HERE), cargo flights (AeroDataBox), cargo AIS (MarineTraffic)
# - Trade/FX/weather charts, packing, scenarios, exports
# Secrets: GOOGLE_MAPS_KEY (opt), HERE_API_KEY (opt), AERODATABOX_KEY (opt), MARINETRAFFIC_KEY (opt)

import io, os, json, math
from typing import Optional, Tuple

import requests, pandas as pd, numpy as np, streamlit as st
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

try:
    import plotly.express as px
except Exception:
    px = None

st.set_page_config(page_title="GTM — Global Trade & Logistics (Sri Lanka)", layout="wide", page_icon="📦")

# ========================= Theming =========================
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
    .hero {{ border-radius:18px; padding:26px 24px 20px; background:{hero_grad}; border:1px solid var(--border); box-shadow:0 18px 40px rgba(0,0,0,.07); margin-bottom:12px; }}
    .hero-title {{ font-family:Inter,system-ui; letter-spacing:-.02em; font-weight:800; font-size:clamp(28px,4vw,40px); margin:0; color:var(--ink); }}
    .hero-sub {{ margin-top:6px; color:var(--muted); font-size:14px; }}
    .card {{ background:{card_grad}; border:1px solid var(--border); padding:16px; border-radius:14px; box-shadow:0 12px 40px rgba(0,0,0,.06) }}
    .kpi {{ display:grid; grid-template-columns: repeat(6, minmax(0,1fr)); gap:.6rem }}
    .kpi .box {{ background:var(--kpi-bg); border:1px solid var(--kpi-bd); border-radius:12px; padding:.9rem 1rem; height:100% }}
    .kpi h3 {{ margin:0; font-size:1.05rem; color:var(--ink) }}
    .kpi p  {{ margin:0; font-size:.8rem; color:var(--muted) }}
    label {{ font-size:.8rem; color:var(--muted); margin-bottom:4px; display:block }}
    input, select, textarea {{ width:100%; padding:{density}; border-radius:10px; border:1px solid var(--border); background:var(--input-bg); color:var(--ink) }}
    .stButton>button {{ width:100%; background:linear-gradient(135deg, var(--primary), var(--accent)); color:white; border:none; font-weight:700; padding:.6rem .9rem; border-radius:10px }}
    hr.soft {{ border:0; border-top:1px solid var(--border); margin:.8rem 0 }}
    header {{ border-bottom:none !important; }}
    </style>
    """, unsafe_allow_html=True)

# defaults
if "theme" not in st.session_state: st.session_state["theme"] = "Light"
if "compact" not in st.session_state: st.session_state["compact"] = False
apply_css(st.session_state["theme"], st.session_state["compact"])

# style bar
st.markdown("<div class='card'>", unsafe_allow_html=True)
c1, c2, c3 = st.columns([.4,.4,.2])
with c1:
    tsel = st.selectbox("Theme", ["Light","Dark"], index=["Light","Dark"].index(st.session_state["theme"]))
with c2:
    csel = st.checkbox("Compact mode", value=st.session_state["compact"])
with c3:
    st.write("")
    if st.button("Apply style"):
        st.session_state["theme"] = tsel
        st.session_state["compact"] = csel
        apply_css(tsel, csel)
st.markdown("</div>", unsafe_allow_html=True)

# ========================= Data helpers =========================
UN_COMTRADE = "https://comtradeplus.un.org/api/get"
FX_URL      = "https://api.exchangerate.host/latest"
WEATHER_API = "https://api.open-meteo.com/v1/forecast"

REPORTERS = {
    "Sri Lanka (144)": "144",
    "India (356)": "356",
    "Denmark (208)": "208",
    "UAE (784)": "784",
    "Singapore (702)": "702",
    "World (000)": "0",
}

def get_secret(name: str) -> Optional[str]:
    try:
        v = st.secrets.get(name)
        if v: return str(v)
    except Exception:
        pass
    return os.getenv(name)

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_fx(base="USD", symbols=("LKR","EUR")):
    try:
        r = requests.get(FX_URL, params={"base":base,"symbols":",".join(symbols)}, timeout=20)
        r.raise_for_status()
        return r.json().get("rates", {})
    except Exception:
        return {}

@st.cache_data(show_spinner=True, ttl=3600)
def fetch_comtrade(reporter="144", flow="1", years="2019,2020,2021,2022,2023", hs="300431"):
    params = {"type":"C","freq":"A","px":"HS","ps":years,"r":reporter,"p":"all","rg":flow,"cc":hs}
    try:
        r = requests.get(UN_COMTRADE, params=params, timeout=60)
        r.raise_for_status()
        return pd.DataFrame(r.json().get("dataset", []))
    except Exception:
        # fallback synthetic
        data = [
            {"period":2019,"ptTitle":"India","TradeValue":12000000,"NetWeight":100000},
            {"period":2020,"ptTitle":"India","TradeValue":13000000,"NetWeight":110000},
            {"period":2021,"ptTitle":"India","TradeValue":16000000,"NetWeight":120000},
            {"period":2022,"ptTitle":"India","TradeValue":20000000,"NetWeight":140000},
            {"period":2023,"ptTitle":"India","TradeValue":24000000,"NetWeight":160000},
        ]
        return pd.DataFrame(data)

# geocoding & weather
geolocator = Nominatim(user_agent="gtm_dashboard/2.0 (edu)")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1, swallow_exceptions=True)

@st.cache_data
def geocode_point(q: str):
    if not q: return None
    loc = geocode(q)
    if not loc: return None
    return (loc.latitude, loc.longitude, loc.address)

@st.cache_data(show_spinner=False, ttl=900)
def fetch_weather(lat, lon):
    try:
        r = requests.get(WEATHER_API, params={"latitude":lat,"longitude":lon,"current":"temperature_2m,precipitation,wind_speed_10m"}, timeout=12)
        r.raise_for_status()
        return r.json().get("current", {})
    except Exception:
        return {}

def weather_risk(cur):
    if not cur: return "N/A", "—"
    wind = float(cur.get("wind_speed_10m", 0) or 0)
    precip = float(cur.get("precipitation", 0) or 0)
    if wind >= 12 or precip >= 5: return "High", f"Wind {wind} m/s, precip {precip} mm"
    if wind >= 8 or precip >= 2: return "Moderate", f"Wind {wind} m/s, precip {precip} mm"
    return "OK", f"Wind {wind} m/s, precip {precip} mm"

# ========================= Live providers =========================
def parse_iata(text: str) -> Optional[str]:
    if not text: return None
    tok = (text.strip().split() or [""])[-1]
    return tok if len(tok)==3 and tok.isalpha() and tok.isupper() else None

def google_driving_eta(o: Tuple[float,float], d: Tuple[float,float]) -> Optional[int]:
    key = get_secret("GOOGLE_MAPS_KEY")
    if not key: return None
    url = "https://maps.googleapis.com/maps/api/distancematrix/json"
    params = {"origins":f"{o[0]},{o[1]}", "destinations":f"{d[0]},{d[1]}", "departure_time":"now", "key":key}
    try:
        r = requests.get(url, params=params, timeout=12); r.raise_for_status()
        elem = r.json()["rows"][0]["elements"][0]
        secs = (elem.get("duration_in_traffic") or elem.get("duration") or {}).get("value")
        return int(secs) if secs is not None else None
    except Exception:
        return None

def here_driving_eta(o: Tuple[float,float], d: Tuple[float,float]) -> Optional[int]:
    key = get_secret("HERE_API_KEY")
    if not key: return None
    url = "https://router.hereapi.com/v8/routes"
    params = {"transportMode":"car","origin":f"{o[0]},{o[1]}","destination":f"{d[0]},{d[1]}","return":"summary","departureTime":"now","apikey":key}
    try:
        r = requests.get(url, params=params, timeout=12); r.raise_for_status()
        return int(r.json()["routes"][0]["sections"][0]["summary"]["duration"])
    except Exception:
        return None

def best_live_road_eta(o, d) -> Optional[int]:
    return google_driving_eta(o, d) or here_driving_eta(o, d)

def aerodatabox_board(endpoint: str, iata: str, limit: int = 10) -> pd.DataFrame:
    key = get_secret("AERODATABOX_KEY")
    if not key or not iata: return pd.DataFrame()
    headers = {"X-RapidAPI-Key": key, "X-RapidAPI-Host": "aerodatabox.p.rapidapi.com"}
    url = f"https://aerodatabox.p.rapidapi.com/airports/iata/{iata}/{endpoint}/now"
    try:
        r = requests.get(url, headers=headers, params={"withLeg":"true","withCancelled":"false"}, timeout=14)
        r.raise_for_status()
        js = r.json()
        arr = js.get("arrivals") if "arrivals" in js else js.get("departures") if "departures" in js else js
        rows=[]
        # cargo heuristic
        cargo_hints = {"cargo","freighter","fx","5x","qr cargo","ek skycargo","ups","dhl","cv","lx cargo","ey cargo","sq cargo","tk cargo","qr","ru"}
        cargo_prefix = {"FX","5X","5Y","CV","QY","RU","TK","QR","LH","SQ","EY","EK"}
        for it in (arr or [])[:limit*3]:
            fl = (it.get("number") or it.get("callSign") or "").upper()
            al = ((it.get("airline") or {}).get("name")) or ""
            text = f"{fl} {al}".lower()
            is_cargo = bool(it.get("isCargo") is True or any(h in text for h in cargo_hints) or (fl[:2] in cargo_prefix))
            if not is_cargo: continue
            rows.append({
                "flight": fl,
                "from": (((it.get("departure") or {}).get("airport") or {}).get("iata")) or "",
                "to": (((it.get("arrival") or {}).get("airport") or {}).get("iata")) or "",
                "sched_local": ((it.get("arrival") or {}).get("scheduledTimeLocal")) or ((it.get("departure") or {}).get("scheduledTimeLocal")) or "",
                "status": it.get("status") or "",
                "airline": al,
            })
            if len(rows) >= limit: break
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()

def marinetraffic_cargo_bbox(lat: float, lon: float, box_km: float = 30) -> pd.DataFrame:
    key = get_secret("MARINETRAFFIC_KEY")
    if not key or lat is None or lon is None: return pd.DataFrame()
    dlat = box_km/111.0
    dlon = box_km/(111.0*max(0.1, math.cos(math.radians(lat))))
    bbox = f"{lon-dlon},{lat-dlat},{lon+dlon},{lat+dlat}"
    url = f"https://services.marinetraffic.com/api/exportvessel/v:5/{key}/timespan:20/protocol:json/bbox:{bbox}"
    try:
        r = requests.get(url, timeout=12); r.raise_for_status()
        data = r.json() if r.headers.get("content-type","").startswith("application/json") else []
        rows=[]
        for v in data:
            try: t = int(v.get("SHIPTYPE", -1))
            except Exception: t = -1
            if 70 <= t <= 79:
                rows.append({"shipname":v.get("SHIPNAME"), "type":t, "lat":v.get("LAT"), "lon":v.get("LON"),
                             "speed_kn":v.get("SPEED"), "course":v.get("COURSE"), "ts":v.get("TIMESTAMP")})
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()

# ========================= Hero =========================
st.markdown("""
<div class="hero">
  <div class="hero-title">GTM Global Trade & Logistics Dashboard — Sri Lanka Focus</div>
  <div class="hero-sub">Live trade • FX • Routes & map • Weather • Cargo flights • Cargo vessels • <b>Exact Incoterms & Landed Cost</b> • Packing • Scenarios</div>
</div>
""", unsafe_allow_html=True)

# ========================= Global controls =========================
fx_live = fetch_fx(); fx_rate_live = float(fx_live.get("LKR", 0) or 0)

st.markdown("<div class='card'>", unsafe_allow_html=True)
r1, r2 = st.columns([1.65, 1.35])

with r1:
    c = st.columns([1.1,.9,.9,.9])
    with c[0]:
        st.markdown("<label>HS code (6-digit)</label>", unsafe_allow_html=True)
        hs_val = st.text_input("hs6", value="300431", label_visibility="collapsed")
    with c[1]:
        st.markdown("<label>Reporter</label>", unsafe_allow_html=True)
        rep_name = st.selectbox("reporter", list(REPORTERS.keys()), index=0, label_visibility="collapsed")
        reporter = REPORTERS[rep_name]
    with c[2]:
        st.markdown("<label>Flow</label>", unsafe_allow_html=True)
        flow = st.selectbox("flow", ["Imports","Exports"], index=0, label_visibility="collapsed")
        flow_code = "1" if flow=="Imports" else "2"
    with c[3]:
        st.markdown("<label>Years</label>", unsafe_allow_html=True)
        years = st.selectbox("years", ["2019,2020,2021,2022,2023","2020,2021,2022,2023,2024"], index=0, label_visibility="collapsed")

with r2:
    c = st.columns([1,1,1,1])
    with c[0]:
        st.markdown("<label>USD→LKR override</label>", unsafe_allow_html=True)
        fx_override = st.number_input("fx", 0.0, value=0.0, step=0.01, label_visibility="collapsed")
        fx_use = fx_override if fx_override>0 else fx_rate_live
    with c[1]:
        st.markdown("<label>Origin (City IATA/Port)</label>", unsafe_allow_html=True)
        origin_q = st.text_input("origin", value="Bengaluru BLR", label_visibility="collapsed")
    with c[2]:
        st.markdown("<label>Destination (City IATA/Port)</label>", unsafe_allow_html=True)
        dest_q   = st.text_input("dest", value="Colombo CMB", label_visibility="collapsed")
    with c[3]:
        st.markdown("<label>Mode</label>", unsafe_allow_html=True)
        mode = st.selectbox("mode", ["Air","Sea","Road"], index=0, label_visibility="collapsed")
st.markdown("</div>", unsafe_allow_html=True)

# ========================= Trade data / KPIs =========================
df = fetch_comtrade(reporter=reporter, flow=flow_code, years=years, hs=hs_val)
def pick(df, cols, fill=None):
    for c in cols:
        if c in df.columns: return df[c]
    return pd.Series([fill]*len(df)) if fill is not None else pd.Series(dtype=float)

period  = pick(df, ["period","yr","Time"])
partner = pick(df, ["ptTitle","partner","Partner"], fill="World")
value   = pick(df, ["TradeValue","PrimaryValue","value"], fill=0)
kg      = pick(df, ["NetWeight","netWgt"], fill=0)
if df.empty:
    ndf = pd.DataFrame(columns=["year","partner","value_usd","kg"])
else:
    ndf = pd.DataFrame({"year":pd.to_numeric(period, errors="coerce"),
                        "partner": partner.astype(str),
                        "value_usd":pd.to_numeric(value, errors="coerce"),
                        "kg":pd.to_numeric(kg, errors="coerce")}).dropna(subset=["year","value_usd"]).fillna(0)

trend = ndf.groupby("year")["value_usd"].sum().reset_index() if not ndf.empty else pd.DataFrame(columns=["year","value_usd"])
partners_df = (ndf.groupby("partner")["value_usd"].sum().reset_index().sort_values("value_usd", ascending=False)) if not ndf.empty else pd.DataFrame(columns=["partner","value_usd"])
unit_vals = (ndf.groupby("year").apply(lambda g: (g["value_usd"].sum() / max(1.0, g["kg"].sum()))).reset_index(name="usd_per_kg")) if not ndf.empty else pd.DataFrame(columns=["year","usd_per_kg"])

total_trade = float(trend["value_usd"].sum()) if not trend.empty else 0.0
_top = partners_df.iloc[0] if not partners_df.empty else pd.Series({"partner":"—","value_usd":0})
def cagr(a,b,n): return 0.0 if a<=0 or n<=0 else (b/a)**(1/n)-1
years_sorted = sorted(trend["year"].tolist()) if not trend.empty else []
avg_yoy=0.0; cagr_val=0.0
if len(years_sorted)>=2:
    diffs=[]
    for i in range(1,len(years_sorted)):
        prev=float(trend.loc[trend["year"]==years_sorted[i-1], "value_usd"].values[0])
        cur =float(trend.loc[trend["year"]==years_sorted[i],   "value_usd"].values[0])
        diffs.append((cur-prev)/max(1.0,prev))
    avg_yoy=float(np.mean(diffs)) if diffs else 0.0
    cagr_val=cagr(float(trend.loc[trend["year"]==years_sorted[0],"value_usd"]), float(trend.loc[trend["year"]==years_sorted[-1],"value_usd"]), len(years_sorted)-1)

st.markdown('<div class="kpi">', unsafe_allow_html=True)
st.markdown(f'<div class="box"><p>Total Trade (USD)</p><h3>{total_trade:,.0f}</h3></div>', unsafe_allow_html=True)
st.markdown(f'<div class="box"><p>Top Partner</p><h3>{_top["partner"]} ({float(_top["value_usd"]):,.0f})</h3></div>', unsafe_allow_html=True)
st.markdown(f'<div class="box"><p># Partners</p><h3>{partners_df.shape[0]}</h3></div>', unsafe_allow_html=True)
st.markdown(f'<div class="box"><p>Avg YoY Growth</p><h3>{avg_yoy*100:.1f}%</h3></div>', unsafe_allow_html=True)
st.markdown(f'<div class="box"><p>CAGR</p><h3>{cagr_val*100:.1f}%</h3></div>', unsafe_allow_html=True)
st.markdown(f'<div class="box"><p>FX USD→LKR</p><h3>{fx_use:.2f}</h3></div>', unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ========================= Live signals =========================
o_pt = geocode_point(origin_q); d_pt = geocode_point(dest_q)

t1,t2,t3,t4 = st.columns(4)
with t1:
    if o_pt and d_pt and mode=="Road":
        secs = best_live_road_eta((o_pt[0],o_pt[1]), (d_pt[0],d_pt[1]))
        if secs: st.metric("🚦 Road ETA (traffic)", f"{secs/3600:.1f} h")
with t2:
    oiata=parse_iata(origin_q)
    if oiata and get_secret("AERODATABOX_KEY"):
        deps = aerodatabox_board("departures", oiata, limit=5)
        st.metric("🛫 Cargo departures (origin)", "0" if deps.empty else str(len(deps)))
with t3:
    diata=parse_iata(dest_q)
    if diata and get_secret("AERODATABOX_KEY"):
        arrs = aerodatabox_board("arrivals", diata, limit=5)
        st.metric("🛬 Cargo arrivals (dest)", "0" if arrs.empty else str(len(arrs)))
with t4:
    if d_pt and get_secret("MARINETRAFFIC_KEY"):
        mt = marinetraffic_cargo_bbox(d_pt[0], d_pt[1], box_km=30)
        st.metric("⚓ Cargo vessels near dest", "0" if mt.empty else str(len(mt)))

# ========================= Charts & Map =========================
left, right = st.columns([1.12,.88], gap="small")

with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    tabs = st.tabs(["Trade trend","Partner share","Unit values","Raw"])
    with tabs[0]:
        if not trend.empty and px is not None: st.plotly_chart(px.line(trend, x="year", y="value_usd", markers=True, title="Total Trade (USD)"), use_container_width=True)
        elif not trend.empty: st.line_chart(trend.set_index("year")["value_usd"])
        else: st.info("No data for selected filters.")
    with tabs[1]:
        if not partners_df.empty and px is not None: st.plotly_chart(px.bar(partners_df.head(12), x="value_usd", y="partner", orientation="h", title="Top partners (USD)"), use_container_width=True)
        elif not partners_df.empty: st.bar_chart(partners_df.set_index("partner")["value_usd"])
        else: st.info("No partner data.")
    with tabs[2]:
        if not unit_vals.empty and px is not None: st.plotly_chart(px.line(unit_vals, x="year", y="usd_per_kg", markers=True, title="Unit Value (USD/kg)"), use_container_width=True)
        elif not unit_vals.empty: st.line_chart(unit_vals.set_index("year")["usd_per_kg"])
        else: st.info("No unit value data.")
    with tabs[3]:
        st.dataframe(ndf, use_container_width=True)
        buf = io.StringIO(); ndf.to_csv(buf, index=False)
        st.download_button("Download raw dataset (CSV)", data=buf.getvalue(), file_name="trade_raw.csv", mime="text/csv")
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='card'><b>Route • Lead Time • Weather • Cargo Boards • Cargo AIS • Emissions</b>", unsafe_allow_html=True)

    # simple distance + lead time
    dist_km=None; lead_days=0.0
    if o_pt and d_pt:
        a=(o_pt[0],o_pt[1]); b=(d_pt[0],d_pt[1])
        R=6371.0
        from math import radians, sin, cos, asin, sqrt
        lat1,lon1,lat2,lon2 = map(radians,[a[0],a[1],b[0],b[1]])
        dlat, dlon = lat2-lat1, lon2-lon1
        h = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
        dist_km = 2*R*asin(sqrt(h))
        speed = 800 if mode=="Air" else (35*24 if mode=="Sea" else 60)  # km/h
        handling, clearance, local = (0.8,0.8,0.3) if mode=="Air" else ((1.5,2.0,0.5) if mode=="Sea" else (0.2,0.5,0.2))
        lead_days = (dist_km/speed)/24 + handling + clearance + local

        fmap = folium.Map(location=[(a[0]+b[0])/2, (a[1]+b[1])/2], zoom_start=4, control_scale=True)
        folium.Marker(a, tooltip=f"Origin: {origin_q}").add_to(fmap)
        folium.Marker(b, tooltip=f"Destination: {dest_q}").add_to(fmap)
        folium.PolyLine([a,b], color="#2563eb", weight=4).add_to(fmap)
        st_folium(fmap, height=420, use_container_width=True)

        st.caption(f"Distance ≈ {dist_km:,.0f} km • Estimated lead time: {lead_days:.1f} days ({mode})")

        ow = fetch_weather(a[0],a[1]); dw = fetch_weather(b[0],b[1])
        orisk, omsg = weather_risk(ow); drisk, dmsg = weather_risk(dw)
        st.write(f"🌤️ Origin: **{orisk}** ({omsg}) · Destination: **{drisk}** ({dmsg})")

        if mode=="Road":
            secs = best_live_road_eta(a,b)
            if secs: st.write(f"🚚 **Live road ETA (traffic)**: ~{secs/3600:.1f} h")

        oiata=parse_iata(origin_q); diata=parse_iata(dest_q)
        if get_secret("AERODATABOX_KEY") and (oiata or diata):
            with st.expander("✈️ Cargo flight boards"):
                ac1,ac2 = st.columns(2)
                with ac1:
                    st.caption(f"Origin cargo departures — {oiata or '—'}")
                    dep = aerodatabox_board("departures", oiata, limit=10) if oiata else pd.DataFrame()
                    if not dep.empty: st.dataframe(dep, use_container_width=True, height=240)
                with ac2:
                    st.caption(f"Destination cargo arrivals — {diata or '—'}")
                    arr = aerodatabox_board("arrivals", diata, limit=10) if diata else pd.DataFrame()
                    if not arr.empty: st.dataframe(arr, use_container_width=True, height=240)

        if d_pt and get_secret("MARINETRAFFIC_KEY"):
            with st.expander("🚢 Cargo vessels near destination"):
                mt = marinetraffic_cargo_bbox(d_pt[0], d_pt[1], box_km=30)
                if not mt.empty:
                    st.dataframe(mt[["shipname","type","speed_kn","course","ts"]], use_container_width=True, height=260)
                    st.caption("AIS window ≈ 20 min; bbox ≈ 30 km.")
    else:
        st.info("Enter clear locations (e.g., 'Bengaluru BLR', 'Colombo CMB').")

    # emissions
    ship_kg = st.number_input("Shipment weight (kg)", min_value=0.0, value=200.0, step=10.0)
    EF = {"Air":600.0,"Sea":15.0,"Road":120.0}
    if dist_km is not None:
        co2e = dist_km*(ship_kg/1000.0)*(EF.get(mode,120.0)/1000.0)
        st.metric("Estimated emissions (kg CO₂e)", f"{co2e:,.0f}")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<hr class='soft'/>", unsafe_allow_html=True)

# ========================= Landed Cost — precise Incoterms =========================
st.markdown("<div class='card'><b>Landed Cost • Exact Incoterms</b>", unsafe_allow_html=True)

# 1) Inputs
lc1, lc2, lc3, lc4 = st.columns([1.1,1.1,1.1,1.1])
with lc1:
    incoterm = st.selectbox("Incoterm",
        ["EXW","FCA","FAS","FOB","CFR","CIF","CPT","CIP","DAP","DPU","DDP"], index=5)
    invoice_value = st.number_input(f"Seller invoice value ({incoterm}) — USD", min_value=0.0, value=150000.0, step=100.0)
    qty_units = st.number_input("Units / pieces (optional)", min_value=0, value=10000)
with lc2:
    main_freight = st.number_input("Main carriage freight (USD)", min_value=0.0, value=0.0, step=50.0)
    origin_charges = st.number_input("Origin charges (pre-carriage/export/handling) — not dutiable (USD)", min_value=0.0, value=0.0, step=10.0)
    dest_charges = st.number_input("Destination charges (terminal/cold storage) — not dutiable (USD)", min_value=0.0, value=400.0, step=10.0)
with lc3:
    brokerage = st.number_input("Brokerage & regulatory (USD)", min_value=0.0, value=350.0, step=10.0)
    inland = st.number_input("Inland transport to warehouse (USD)", min_value=0.0, value=300.0, step=10.0)
    other_local = st.number_input("Other local charges (USD)", min_value=0.0, value=0.0, step=10.0)
with lc4:
    ins_mode = st.radio("Insurance input", ["Percent","Amount"], index=0, horizontal=True)
    ins_base = st.selectbox("If %: base", ["Invoice","Invoice+Freight"], index=1)
    ins_pct  = st.number_input("Insurance %", min_value=0.0, value=1.0, step=0.1)
    ins_amt  = st.number_input("Insurance amount (USD)", min_value=0.0, value=0.0, step=10.0)
    seller_pays_import_taxes = st.checkbox("Seller pays import taxes (DDP)", value=(incoterm=="DDP"))

# 2) Incoterm matrix — what’s already included in invoice_value?
# For customs value (CIF-equivalent) we must ensure freight/insurance are included exactly once.
INCOTERM_INCLUDES = {
    # freight, insurance
    "EXW": (False, False),
    "FCA": (False, False),
    "FAS": (False, False),
    "FOB": (False, False),
    "CFR": (True,  False),
    "CIF": (True,  True),
    "CPT": (True,  False),
    "CIP": (True,  True),
    "DAP": (True,  False),   # delivered to place, seller bears main carriage; insurance may or may not be sold, treat as not guaranteed
    "DPU": (True,  False),
    "DDP": (True,  False),   # seller bears transport and import taxes; we handle taxes with checkbox above
}
freight_included, insurance_included = INCOTERM_INCLUDES.get(incoterm, (False, False))

# 3) Compute insurance_add and freight_add for CUSTOMS value
freight_add_for_customs = 0.0 if freight_included else main_freight
if insurance_included:
    insurance_add_for_customs = 0.0   # already inside invoice_value
else:
    if ins_mode == "Percent":
        base = invoice_value + (0.0 if freight_included else main_freight) if ins_base=="Invoice+Freight" else invoice_value
        insurance_add_for_customs = base * (ins_pct/100.0)
    else:
        insurance_add_for_customs = ins_amt

customs_value = invoice_value + freight_add_for_customs + insurance_add_for_customs  # CIF-equivalent for duty

# 4) Taxes
duty_pct = st.number_input("Import duty % (on customs value)", min_value=0.0, value=0.0, step=0.1)
other_tax_pct = st.number_input("Other tariff/levy % (on customs value, optional)", min_value=0.0, value=0.0, step=0.1)
vat_pct = st.number_input("VAT / GST % (on customs value + duty + other)", min_value=0.0, value=0.0, step=0.1)

duty = customs_value * (duty_pct/100.0)
other_tax = customs_value * (other_tax_pct/100.0)
vat_base = customs_value + duty + other_tax
vat = vat_base * (vat_pct/100.0)

# 5) Buyer-pay total
# If DDP (or the checkbox is ticked), seller already paid import taxes — do NOT add duty/other/vat to buyer total.
taxes_payable_by_buyer = 0.0 if seller_pays_import_taxes else (duty + other_tax + vat)

# Buyer pays: invoice_value (whatever incoterm includes) + local charges + taxes they owe
total_landed = (
    invoice_value
    + (0.0 if freight_included else main_freight)   # if freight not in invoice, buyer actually pays it to carrier
    + (0.0 if insurance_included else (insurance_add_for_customs if ins_mode=="Percent" else ins_amt))  # buyer’s insurance outlay if not included
    + origin_charges + dest_charges + brokerage + inland + other_local
    + taxes_payable_by_buyer
)

# 6) Outputs
o1,o2,o3,o4 = st.columns(4)
with o1: st.metric("Customs value (CIF-equiv.)", f"${customs_value:,.0f}")
with o2: st.metric("Duty", f"${duty:,.0f}")
with o3: st.metric("VAT / GST", f"${vat:,.0f}")
with o4: st.metric("Other tariff/levy", f"${other_tax:,.0f}")

m1,m2,m3,m4 = st.columns(4)
with m1: st.metric("Buyer insurance outlay", f"${0.0 if insurance_included else (insurance_add_for_customs if ins_mode=='Percent' else ins_amt):,.0f}")
with m2: st.metric("Buyer freight outlay", f"${0.0 if freight_included else main_freight:,.0f}")
with m3: st.metric("Taxes payable by buyer", f"${taxes_payable_by_buyer:,.0f}" + (" (DDP seller-paid)" if seller_pays_import_taxes else ""))
with m4: st.metric("Total Landed Cost (USD)", f"${total_landed:,.0f}")

if qty_units and qty_units>0:
    st.caption(f"Landed cost per unit: **${(total_landed/qty_units):,.2f}**")

st.markdown("</div>", unsafe_allow_html=True)

# ========================= Origin compare (what-if) =========================
st.markdown("<div class='card'><b>Compare Origins (What-if)</b>", unsafe_allow_html=True)
comp = pd.DataFrame([
    {"Origin":"India (ISFTA)","Invoice":invoice_value,"Incoterm":incoterm,"Freight":main_freight,"Duty%":duty_pct},
    {"Origin":"Denmark (MFN)","Invoice":invoice_value*1.05,"Incoterm":incoterm,"Freight":main_freight*1.8,"Duty%":max(duty_pct,2.0)},
    {"Origin":"Singapore","Invoice":invoice_value*1.02,"Incoterm":incoterm,"Freight":main_freight*1.2,"Duty%":duty_pct},
])
rows=[]
for _, r in comp.iterrows():
    fr_in, ins_in = INCOTERM_INCLUDES.get(r.Incoterm, (False,False))
    # reuse same insurance rule for comparison
    fr_add = 0.0 if fr_in else r.Freight
    if ins_in:
        ins_add = 0.0
    else:
        ins_add = ( (r.Invoice + (0.0 if fr_in else r.Freight)) * (ins_pct/100.0) ) if ins_mode=="Percent" else ins_amt
    cv = r.Invoice + fr_add + ins_add
    duty_c = cv * (r["Duty%"]/100.0)
    vat_c  = (cv + duty_c) * (vat_pct/100.0)
    other_c= cv * (other_tax_pct/100.0)
    taxes_buyer = 0.0 if seller_pays_import_taxes else (duty_c + vat_c + other_c)
    tlc = (r.Invoice + (0.0 if fr_in else r.Freight) + (0.0 if ins_in else ins_add) +
           origin_charges + dest_charges + brokerage + inland + other_local + taxes_buyer)
    rows.append({"Origin":r.Origin,"TLC_USD":tlc})
out=pd.DataFrame(rows)
st.dataframe(out, use_container_width=True)
if px is not None: st.plotly_chart(px.bar(out, x="Origin", y="TLC_USD", title="Total Landed Cost by Origin (USD)"), use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ========================= Packing & Scenarios (unchanged) =========================
st.markdown("<div class='card'><b>Packing • ULD & Container Capacity</b>", unsafe_allow_html=True)
pc1,pc2,pc3 = st.columns(3)
with pc1:
    carton_l = st.number_input("Carton length (cm)", 1.0, 200.0, 40.0)
    carton_w = st.number_input("Carton width (cm)",  1.0, 200.0, 30.0)
    carton_h = st.number_input("Carton height (cm)", 1.0, 200.0, 25.0)
with pc2:
    carton_kg = st.number_input("Carton weight (kg)", 0.1, 200.0, 8.0)
    layer_gap = st.number_input("Layer gap (cm)", 0.0, 10.0, 0.0)
    max_stack_h = st.number_input("Max stack height (cm)", 50.0, 250.0, 140.0)
with pc3:
    use_pmc = st.checkbox("Air PMC pallet (243×318×160 cm)", value=True)
    use_20  = st.checkbox("Sea 20' (589×235×239 cm)", value=False)
    use_40  = st.checkbox("Sea 40' (1203×235×239 cm)", value=False)
def pack_on(base_l, base_w, base_h):
    per_row = math.floor(base_l // carton_l) * math.floor(base_w // carton_w)
    layers  = math.floor((min(base_h, max_stack_h)) // (carton_h + layer_gap))
    boxes   = max(0, per_row) * max(0, layers)
    return boxes, boxes*carton_kg
rows=[]
if use_pmc: rows.append(("PMC pallet", *pack_on(243.0,318.0,160.0)))
if use_20:  rows.append(("20' container", *pack_on(589.0,235.0,239.0)))
if use_40:  rows.append(("40' container", *pack_on(1203.0,235.0,239.0)))
if rows:
    pk = pd.DataFrame(rows, columns=["Unit","Max cartons","Total kg"])
    st.dataframe(pk, use_container_width=True)
    if px is not None: st.plotly_chart(px.bar(pk, x="Unit", y="Max cartons", title="Packing capacity"), use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ========================= Save / Export =========================
st.markdown("<div class='card'><b>Save & Export Scenario</b>", unsafe_allow_html=True)
sc_name = st.text_input("Scenario name", value="My scenario")
if st.button("Save current scenario"):
    if "scenarios" not in st.session_state: st.session_state["scenarios"]=[]
    st.session_state["scenarios"].append({
        "name":sc_name,"incoterm":incoterm,"invoice_value":invoice_value,"freight":main_freight,
        "insurance_add":insurance_add_for_customs,"customs_value":customs_value,"duty":duty,"vat":vat,"other":other_tax,
        "origin_charges":origin_charges,"dest_charges":dest_charges,"brokerage":brokerage,"inland":inland,"other_local":other_local,
        "seller_pays_import_taxes":seller_pays_import_taxes,"total_landed":total_landed
    })
    st.success("Saved.")

if st.session_state.get("scenarios"):
    sdf=pd.DataFrame(st.session_state["scenarios"])
    st.dataframe(sdf[["name","incoterm","customs_value","duty","vat","total_landed"]], use_container_width=True)
    if px is not None and len(sdf)>0:
        st.plotly_chart(px.bar(sdf, x="name", y="total_landed", title="Scenario TLC (USD)"), use_container_width=True)
    b=io.StringIO(); sdf.to_csv(b, index=False)
    st.download_button("Download scenarios CSV", data=b.getvalue(), file_name="gtm_scenarios.csv", mime="text/csv")

# quick export of this run
snap = {
  "incoterm": incoterm, "invoice_value": invoice_value, "freight_included": freight_included, "insurance_included": insurance_included,
  "freight_add_for_customs": freight_add_for_customs, "insurance_add_for_customs": insurance_add_for_customs,
  "customs_value": customs_value, "duty_pct": duty_pct, "other_tax_pct": other_tax_pct, "vat_pct": vat_pct,
  "duty": duty, "other_tax": other_tax, "vat": vat, "taxes_payable_by_buyer": taxes_payable_by_buyer,
  "origin_charges": origin_charges, "dest_charges": dest_charges, "brokerage": brokerage, "inland": inland, "other_local": other_local,
  "total_landed": total_landed, "qty_units": qty_units, "per_unit": (total_landed/qty_units if qty_units else None)
}
buf_json = json.dumps(snap, indent=2)
buf_csv  = io.StringIO(); pd.DataFrame([snap]).to_csv(buf_csv, index=False)
d1,d2 = st.columns(2)
with d1: st.download_button("Download snapshot JSON", data=buf_json, file_name="gtm_snapshot.json", mime="application/json")
with d2: st.download_button("Download snapshot CSV",  data=buf_csv.getvalue(), file_name="gtm_snapshot.csv",  mime="text/csv")

st.caption("Educational tool. Always verify tariff & NTM rules with official sources (MACMAP, Sri Lanka Customs).")
