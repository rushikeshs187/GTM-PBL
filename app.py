# app.py — GTM Global Trade & Logistics Dashboard (Sri Lanka Focus)
# Adds LIVE CARGO views (AeroDataBox cargo flights, MarineTraffic cargo vessels) and removes "Tip:" captions.

import io, os, json, math, datetime as dt
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

# ---------- defaults ----------
def ensure_defaults():
    st.session_state.setdefault("theme", "Light")
    st.session_state.setdefault("compact", False)
    st.session_state.setdefault("d_hs", "300431")
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
    st.session_state.setdefault("scenarios", [])
ensure_defaults()

# ---------- theme ----------
def render_css(theme="Light", compact=False):
    if theme == "Light":
        bg, panel, ink, muted, border = "#f7f9fc", "#ffffff", "#0f172a", "#526581", "#e6ebf2"
        primary, accent = "#2563eb", "#7c3aed"
        card_grad = "linear-gradient(180deg, rgba(255,255,255,.98), rgba(255,255,255,.98))"
        hero_grad = ("radial-gradient(1200px 400px at 0% -10%, rgba(37,99,235,.10), transparent 60%),"
                     "radial-gradient(1200px 400px at 100% 110%, rgba(124,58,237,.08), transparent 60%),"
                     "linear-gradient(180deg, rgba(255,255,255,.98), rgba(255,255,255,.98))")
        kpi_bg, kpi_bd, input_bg = "#f3f6fb", "#e6ebf2", "#fbfdff"
    else:
        bg, panel, ink, muted, border = "#0b0f14", "#0f1521", "#e7edf7", "#9fb0c4", "#1e2b3c"
        primary, accent = "#60a5fa", "#a78bfa"
        card_grad = "linear-gradient(180deg, rgba(21,30,48,.94), rgba(12,17,28,.92))"
        hero_grad = ("radial-gradient(1200px 400px at 0% -10%, rgba(96,165,250,.15), transparent 60%),"
                     "radial-gradient(1200px 400px at 100% 110%, rgba(167,139,250,.12), transparent 60%),"
                     "linear-gradient(180deg, rgba(18,26,40,.92), rgba(10,15,25,.92))")
        kpi_bg, kpi_bd, input_bg = "#0f172a", "#1e293b", "#0d1422"

    density_pad = ".5rem .7rem" if compact else ".62rem .8rem"
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    :root {{ --bg:{bg}; --panel:{panel}; --muted:{muted}; --ink:{ink}; --border:{border};
            --primary:{primary}; --accent:{accent}; --kpi-bg:{kpi_bg}; --kpi-bd:{kpi_bd}; --input-bg:{input_bg}; }}
    html, body, .stApp {{ background: var(--bg) !important; }}
    h1, h2, h3, .hero-title {{ letter-spacing:-.02em; font-feature-settings:"liga" 1, "calt" 1; font-family:Inter,system-ui; color:var(--ink); }}
    .hero {{ border-radius:18px; padding:26px 24px 20px; background:{hero_grad}; border:1px solid var(--border); box-shadow:0 18px 40px rgba(0,0,0,.07); margin-bottom:12px; }}
    .hero-title {{ font-size:clamp(28px,4vw,40px); line-height:1.1; font-weight:800; margin:0; }}
    .hero-sub {{ margin-top:6px; color:var(--muted); font-size:14px; }}
    .card {{ background:{card_grad}; border:1px solid var(--border); padding:16px; border-radius:14px; box-shadow:0 12px 40px rgba(0,0,0,.06) }}
    .kpi {{ display:grid; grid-template-columns: repeat(6, minmax(0,1fr)); gap:.6rem }}
    .kpi .box {{ background:var(--kpi-bg); border:1px solid var(--kpi-bd); border-radius:12px; padding:.9rem 1rem; height:100% }}
    .kpi h3 {{ margin:0; font-size:1.05rem }}
    .kpi p  {{ margin:0; font-size:.8rem; color:var(--muted) }}
    label {{ font-size:.8rem; color:var(--muted); margin-bottom:4px; display:block }}
    input, select, textarea {{ width:100%; padding:{density_pad}; border-radius:10px; border:1px solid var(--border); background:var(--input-bg); color:var(--ink) }}
    .stButton>button {{ width:100%; background:linear-gradient(135deg, var(--primary), var(--accent)); color:white; border:none; font-weight:700; padding:.6rem .9rem; border-radius:10px }}
    .warn {{ background:#fff7ed; color:#9a3412; border:1px solid #fed7aa; padding:.5rem .7rem; border-radius:10px }}
    hr.soft {{ border:0; border-top:1px solid var(--border); margin:.8rem 0 }}
    header {{ border-bottom:none !important; }}
    </style>
    """, unsafe_allow_html=True)

st.markdown("<div class='card'>", unsafe_allow_html=True)
s1, s2, s3 = st.columns([.4,.4,.2])
with s1:
    theme_choice = st.selectbox("Theme", ["Light","Dark"], index=["Light","Dark"].index(st.session_state["theme"]))
with s2:
    compact_choice = st.checkbox("Compact mode", value=st.session_state["compact"])
with s3:
    st.markdown("&nbsp;")
    if st.button("Apply style"):
        st.session_state["theme"] = theme_choice
        st.session_state["compact"] = compact_choice
        st.rerun()
st.markdown("</div>", unsafe_allow_html=True)
render_css(st.session_state["theme"], st.session_state["compact"])

# ---------- constants ----------
UN_COMTRADE = "https://comtradeplus.un.org/api/get"
FX_URL = "https://api.exchangerate.host/latest"
OPEN_METEO = "https://api.open-meteo.com/v1/forecast"
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
    }
}

# ---------- helpers ----------
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

def pick_col(df: pd.DataFrame, names, fill=None):
    for n in names:
        if n in df.columns:
            return df[n]
    return pd.Series([fill] * len(df)) if fill is not None else pd.Series(dtype=float)

geolocator = Nominatim(user_agent="gtm_dashboard/1.4 (edu)")
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

def cagr(first, last, n):
    if first <= 0 or n <= 0: return 0.0
    return (last/first)**(1/n) - 1

def safe_line(df, x, y, title):
    if px is not None: st.plotly_chart(px.line(df, x=x, y=y, markers=True, title=title), use_container_width=True)
    else: st.subheader(title); st.line_chart(df.set_index(x)[y])

def safe_bar(df, x, y, title, horizontal=False):
    if px is not None: st.plotly_chart(px.bar(df, x=x, y=y, title=title, orientation="h" if horizontal else "v"), use_container_width=True)
    else: st.subheader(title); st.bar_chart(df.set_index(y if horizontal else x)[x if horizontal else y])

@st.cache_data(show_spinner=False, ttl=900)
def fetch_weather(lat, lon):
    try:
        r = requests.get(OPEN_METEO, params={"latitude":lat, "longitude":lon, "current":"temperature_2m,precipitation,wind_speed_10m"}, timeout=12)
        r.raise_for_status()
        return r.json().get("current", {})
    except Exception:
        return {}

def weather_risk(cur):
    if not cur: return "N/A", "—"
    wind = float(cur.get("wind_speed_10m", 0) or 0)
    precip = float(cur.get("precipitation", 0) or 0)
    if wind >= 12 or precip >= 5: return "High", f"Wind {wind} m/s, precip {precip} mm"
    if wind >= 8 or precip >= 2:  return "Moderate", f"Wind {wind} m/s, precip {precip} mm"
    return "OK", f"Wind {wind} m/s, precip {precip} mm"

# ---------- LIVE providers (with cargo filters) ----------
def parse_iata(text: str) -> Optional[str]:
    if not text: return None
    token = (text.strip().split() or [""])[-1]
    return token if len(token)==3 and token.isalpha() and token.isupper() else None

# (1) Google/HERE traffic-aware road ETA
def google_driving_eta(origin_latlon: Tuple[float,float], dest_latlon: Tuple[float,float]) -> Optional[int]:
    key = get_secret("GOOGLE_MAPS_KEY")
    if not key: return None
    base = "https://maps.googleapis.com/maps/api/distancematrix/json"
    params = {"origins":f"{origin_latlon[0]},{origin_latlon[1]}",
              "destinations":f"{dest_latlon[0]},{dest_latlon[1]}",
              "departure_time":"now","key":key}
    try:
        r = requests.get(base, params=params, timeout=12); r.raise_for_status()
        elem = r.json()["rows"][0]["elements"][0]
        secs = (elem.get("duration_in_traffic") or elem.get("duration") or {}).get("value")
        return int(secs) if secs is not None else None
    except Exception:
        return None

def here_driving_eta(origin_latlon: Tuple[float,float], dest_latlon: Tuple[float,float]) -> Optional[int]:
    key = get_secret("HERE_API_KEY")
    if not key: return None
    base = "https://router.hereapi.com/v8/routes"
    params = {"transportMode":"car", "origin":f"{origin_latlon[0]},{origin_latlon[1]}",
              "destination":f"{dest_latlon[0]},{dest_latlon[1]}", "return":"summary",
              "departureTime":"now", "apikey":key}
    try:
        r = requests.get(base, params=params, timeout=12); r.raise_for_status()
        return int(r.json()["routes"][0]["sections"][0]["summary"]["duration"])
    except Exception:
        return None

def best_live_road_eta(o, d) -> Optional[int]:
    return google_driving_eta(o,d) or here_driving_eta(o,d)

# (2) AeroDataBox — cargo flight boards (heuristic + isCargo if present)
CARGO_AIRLINE_HINTS = {"cargo","freighter","fx","5x","qr cargo","ek skycargo","ups","dhl","qx","cv","lx cargo","ey cargo","sq cargo","tk cargo","qr"}
CARGO_IATA_PREFIXES = {"FX","5X","5Y","CV","QY","RU","TK","ET","QR","LH","SQ","EY","EK","ULC"}  # includes common cargo ops; heuristic

def is_cargo_row(row: dict) -> bool:
    try:
        # explicit flag if exists
        if isinstance(row.get("isCargo"), bool):
            return row["isCargo"]
    except Exception:
        pass
    fl = (row.get("number") or row.get("flight") or "").upper()
    al = ((row.get("airline") or {}).get("name") if isinstance(row.get("airline"), dict) else row.get("airline")) or ""
    text = f"{fl} {al}".lower()
    if any(h in text for h in CARGO_AIRLINE_HINTS): return True
    if fl[:2] in CARGO_IATA_PREFIXES: return True
    return False

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
        for it in (arr or [])[:limit*3]:  # fetch more, then filter
            row = {
                "flight": it.get("number") or it.get("callSign") or "",
                "from": (((it.get("departure") or {}).get("airport") or {}).get("iata")) or "",
                "to": (((it.get("arrival") or {}).get("airport") or {}).get("iata")) or "",
                "sched_local": ((it.get("arrival") or {}).get("scheduledTimeLocal")) or ((it.get("departure") or {}).get("scheduledTimeLocal")) or "",
                "status": it.get("status") or "",
                "airline": ((it.get("airline") or {}).get("name")) or "",
                "isCargo": it.get("isCargo"),
            }
            if is_cargo_row(row):
                rows.append(row)
            if len(rows) >= limit:
                break
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()

def aerodatabox_arrivals_cargo(iata: str, limit: int = 10) -> pd.DataFrame:
    return aerodatabox_board("arrivals", iata, limit)

def aerodatabox_departures_cargo(iata: str, limit: int = 10) -> pd.DataFrame:
    return aerodatabox_board("departures", iata, limit)

# (3) MarineTraffic — cargo vessels only (type 70–79)
def marinetraffic_cargo_bbox(lat: float, lon: float, box_km: float = 30) -> pd.DataFrame:
    key = get_secret("MARINETRAFFIC_KEY")
    if not key or lat is None or lon is None:
        return pd.DataFrame()
    dlat = box_km / 111.0
    dlon = box_km / (111.0 * max(0.1, math.cos(math.radians(lat))))
    bbox = f"{lon-dlon},{lat-dlat},{lon+dlon},{lat+dlat}"
    url = f"https://services.marinetraffic.com/api/exportvessel/v:5/{key}/timespan:20/protocol:json/bbox:{bbox}"
    try:
        r = requests.get(url, timeout=12); r.raise_for_status()
        data = r.json() if r.headers.get("content-type","").startswith("application/json") else []
        rows=[]
        for v in data:
            t = v.get("SHIPTYPE")
            if t is None: continue
            try:
                t = int(t)
            except Exception:
                continue
            if 70 <= t <= 79:  # cargo
                rows.append({
                    "shipname": v.get("SHIPNAME"),
                    "type": t,
                    "lat": v.get("LAT"), "lon": v.get("LON"),
                    "speed_kn": v.get("SPEED"),
                    "course": v.get("COURSE"),
                    "ts": v.get("TIMESTAMP"),
                })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()

# ---------- hero ----------
st.markdown("""
<div class="hero">
  <div class="hero-title">GTM Global Trade & Logistics Dashboard — Sri Lanka Focus</div>
  <div class="hero-sub">Live trade • FX • Routes & map • Weather • <b>Cargo flights</b> • <b>Cargo vessels</b> • Landed cost • Packing • Scenarios</div>
</div>
""", unsafe_allow_html=True)

# ---------- controls ----------
fx_live = fetch_fx(); fx_rate_live = float(fx_live.get("LKR", 0) or 0)

st.markdown("<div class='card'>", unsafe_allow_html=True)
r1c1, r1c2, r1c3, r1c4 = st.columns([1.5, 1, 1, .9])
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
    years = st.selectbox("years", ["2019,2020,2021,2022,2023","2020,2021,2022,2023,2024","2018,2019,2020,2021,2022"], index=0, key="w_years", label_visibility="collapsed")

r2c1, r2c2, r2c3, r2c4, r2c5 = st.columns([1,1,1,.8,.9])
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
        for k, v in {"d_hs":"hs","d_incoterm":"incoterm","d_fob":"fob","d_freight":"freight",
                     "d_ins_pct":"insurance_pct","d_ins_base":"ins_base","d_duty_pct":"duty_pct",
                     "d_vat_pct":"vat_pct","d_broker":"broker","d_dray":"dray"}.items():
            st.session_state[k] = p[v]
        st.session_state["d_fx_note"] = p["note"]
        st.rerun()
st.markdown("</div>", unsafe_allow_html=True)

# ---------- trade data ----------
df = fetch_comtrade(reporter=reporter, flow=flow_code, years=years, hs=hs_val)
period  = pick_col(df, ["period","yr","Time"])
partner = pick_col(df, ["ptTitle","partner","Partner"], fill="World")
value   = pick_col(df, ["TradeValue","PrimaryValue","value"], fill=0)
kg      = pick_col(df, ["NetWeight","netWgt"], fill=0)

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
partners_df = (ndf.groupby("partner")["value_usd"].sum().reset_index().sort_values("value_usd", ascending=False)) if not ndf.empty else pd.DataFrame(columns=["partner","value_usd"])
unit_vals = (ndf.groupby("year").apply(lambda g: (g["value_usd"].sum() / max(1.0, g["kg"].sum()))).reset_index(name="usd_per_kg")) if not ndf.empty else pd.DataFrame(columns=["year","usd_per_kg"])

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

# ---------- route ----------
o_pt = geocode_point(origin_q); d_pt = geocode_point(dest_q)
dist_km = None; lead_time_days = 0.0
if o_pt and d_pt:
    a, b = (o_pt[0], o_pt[1]), (d_pt[0], d_pt[1])
    dist_km = float(haversine_km(a, b))
    speed = 800 if mode == "Air" else (35*24 if mode == "Sea" else 60)
    handling, clearance, local = (0.8,0.8,0.3) if mode=="Air" else ((1.5,2.0,0.5) if mode=="Sea" else (0.2,0.5,0.2))
    lead_time_days = (dist_km / speed) / 24 + handling + clearance + local

EF = {"Air":600.0, "Sea":15.0, "Road":120.0}

# ---------- KPIs ----------
st.markdown('<div class="kpi">', unsafe_allow_html=True)
st.markdown(f'<div class="box"><p>Total Trade (USD)</p><h3>{total_trade:,.0f}</h3></div>', unsafe_allow_html=True)
st.markdown(f'<div class="box"><p>Top Partner</p><h3>{_top["partner"]} ({float(_top["value_usd"]):,.0f})</h3></div>', unsafe_allow_html=True)
st.markdown(f'<div class="box"><p># Partners</p><h3>{partners_df.shape[0]}</h3></div>', unsafe_allow_html=True)
st.markdown(f'<div class="box"><p>Avg YoY Growth</p><h3>{yoy*100:.1f}%</h3></div>', unsafe_allow_html=True)
st.markdown(f'<div class="box"><p>CAGR (period)</p><h3>{cagr_val*100:.1f}%</h3></div>', unsafe_allow_html=True)
st.markdown(f'<div class="box"><p>FX USD→LKR</p><h3>{fx_use:.2f}</h3></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------- LIVE SIGNALS (now cargo-focused) ----------
tile1, tile2, tile3, tile4 = st.columns(4)
with tile1:
    if o_pt and d_pt and mode == "Road":
        secs = best_live_road_eta((o_pt[0], o_pt[1]), (d_pt[0], d_pt[1]))
        if secs: st.metric("🚦 Road ETA (traffic)", f"{secs/3600:.1f} h")
with tile2:
    oiata = parse_iata(origin_q)
    if oiata and get_secret("AERODATABOX_KEY"):
        deps_cargo = aerodatabox_departures_cargo(oiata, limit=5)
        st.metric("🛫 Cargo departures (origin)", str(len(deps_cargo)) if not deps_cargo.empty else "0")
with tile3:
    diata = parse_iata(dest_q)
    if diata and get_secret("AERODATABOX_KEY"):
        arrs_cargo = aerodatabox_arrivals_cargo(diata, limit=5)
        st.metric("🛬 Cargo arrivals (dest)", str(len(arrs_cargo)) if not arrs_cargo.empty else "0")
with tile4:
    if d_pt and get_secret("MARINETRAFFIC_KEY"):
        mt_cargo = marinetraffic_cargo_bbox(d_pt[0], d_pt[1], box_km=30)
        st.metric("⚓ Cargo vessels near dest", str(len(mt_cargo)) if not mt_cargo.empty else "0")

# ---------- charts + map ----------
left, right = st.columns([1.12, .88], gap="small")

with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    tabs = st.tabs(["Trade trend", "Partner share", "Unit values", "Raw"])
    with tabs[0]:
        if not trend.empty: safe_line(trend, "year","value_usd","Total Trade (USD)")
        else: st.info("No data for selected filters.")
    with tabs[1]:
        if not partners_df.empty: safe_bar(partners_df.head(12), "value_usd","partner","Top partners (USD)", horizontal=True)
        else: st.info("No partner data.")
    with tabs[2]:
        if not unit_vals.empty: safe_line(unit_vals, "year","usd_per_kg","Unit Value (USD/kg)")
        else: st.info("No unit value data.")
    with tabs[3]:
        st.dataframe(ndf, use_container_width=True)
        b = io.StringIO(); ndf.to_csv(b, index=False)
        st.download_button("Download raw dataset (CSV)", data=b.getvalue(), file_name="trade_raw.csv", mime="text/csv")
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='card'><b>Route • Lead Time • Weather • Cargo Boards • Cargo AIS • Emissions</b>", unsafe_allow_html=True)
    w1, w2 = st.columns(2)
    with w1:
        ship_kg = st.number_input("Shipment weight (kg)", min_value=0.0, value=200.0, step=10.0, key="w_shipkg")
    with w2:
        temp_ctrl = st.checkbox("Temperature-controlled (cold chain)", value=True, key="w_cold")

    if o_pt and d_pt:
        fmap = folium.Map(location=[(o_pt[0]+d_pt[0])/2, (o_pt[1]+d_pt[1])/2], zoom_start=4, control_scale=True)
        folium.Marker((o_pt[0], o_pt[1]), tooltip=f"Origin: {origin_q}").add_to(fmap)
        folium.Marker((d_pt[0], d_pt[1]), tooltip=f"Destination: {dest_q}").add_to(fmap)
        folium.PolyLine([(o_pt[0], o_pt[1]), (d_pt[0], d_pt[1])], color="#2563eb", weight=4).add_to(fmap)
        st_folium(fmap, height=420, use_container_width=True)

        st.caption(f"Distance ≈ { (0 if dist_km is None else dist_km):,.0f} km • Estimated lead time: {lead_time_days:.1f} days ({mode})")
        ow = fetch_weather(o_pt[0], o_pt[1]); dw = fetch_weather(d_pt[0], d_pt[1])
        orisk, omsg = weather_risk(ow); drisk, dmsg = weather_risk(dw)
        st.write(f"🌤️ Origin weather risk: **{orisk}** ({omsg}) · Destination: **{drisk}** ({dmsg})")

        if mode == "Road":
            secs_live = best_live_road_eta((o_pt[0], o_pt[1]), (d_pt[0], d_pt[1]))
            if secs_live: st.write(f"🚚 **Live road ETA (traffic)**: ~{secs_live/3600:.1f} h")

        oiata = parse_iata(origin_q); diata = parse_iata(dest_q)
        if get_secret("AERODATABOX_KEY") and (oiata or diata):
            with st.expander("✈️ Cargo flight boards (AeroDataBox)"):
                ac1, ac2 = st.columns(2)
                with ac1:
                    st.caption(f"Origin cargo departures — {oiata or '—'}")
                    deps = aerodatabox_departures_cargo(oiata, limit=10) if oiata else pd.DataFrame()
                    if not deps.empty: st.dataframe(deps, use_container_width=True, height=240)
                with ac2:
                    st.caption(f"Destination cargo arrivals — {diata or '—'}")
                    arrs = aerodatabox_arrivals_cargo(diata, limit=10) if diata else pd.DataFrame()
                    if not arrs.empty: st.dataframe(arrs, use_container_width=True, height=240)

        if d_pt and get_secret("MARINETRAFFIC_KEY"):
            with st.expander("🚢 Cargo vessels near destination (MarineTraffic)"):
                mtc = marinetraffic_cargo_bbox(d_pt[0], d_pt[1], box_km=30)
                if not mtc.empty:
                    st.dataframe(mtc[["shipname","type","speed_kn","course","ts"]], use_container_width=True, height=260)
                    st.caption("AIS window ~20 min; bbox ~30 km around destination.")

        tonnes = (ship_kg or 0) / 1000.0
        co2e_kg = (dist_km or 0) * tonnes * (EF.get(mode,120.0) / 1000.0)
        st.metric("Estimated emissions (kg CO₂e)", f"{co2e_kg:,.0f}")
        if temp_ctrl and lead_time_days and lead_time_days > 3 and mode != "Air":
            st.markdown("<div class='warn'>⚠️ Cold chain risk: long lead time. Consider passive packaging upgrades or Air.</div>", unsafe_allow_html=True)
    else:
        st.info("Enter clear locations (e.g., 'Bengaluru BLR', 'Colombo CMB').")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<hr class='soft'/>", unsafe_allow_html=True)

# ---------- landed cost ----------
st.markdown("<div class='card'><b>Landed Cost • Sensitivity</b>", unsafe_allow_html=True)
lc1, lc2, lc3, lc4 = st.columns([1,1,1,.9])
with lc1:
    incoterm = st.selectbox("Incoterm", ["FOB","CIF","DAP","DDP"],
                            index=["FOB","CIF","DAP","DDP"].index(st.session_state["d_incoterm"]), key="w_inc")
    fob = st.number_input("FOB value (USD)", min_value=0.0, value=st.session_state["d_fob"], step=100.0, key="w_fob")
    insurance_pct = st.number_input("Insurance %", min_value=0.0, value=st.session_state["d_ins_pct"], step=0.1, key="w_ins_pct")
with lc2:
    ins_base = st.selectbox("Insurance base", ["FOB","CIF"], index=["FOB","CIF"].index(st.session_state["d_ins_base"]), key="w_ins_base")
    freight = st.number_input("Freight (USD)", min_value=0.0, value=st.session_state["d_freight"], step=50.0, key="w_freight")
    broker = st.number_input("Brokerage & Handling (USD)", min_value=0.0, value=st.session_state["d_broker"], step=10.0, key="w_broker")
with lc3:
    dray = st.number_input("Last-mile / Drayage (USD)", min_value=0.0, value=st.session_state["d_dray"], step=10.0, key="w_dray")
    duty_pct = st.number_input("Duty %", min_value=0.0, value=st.session_state["d_duty_pct"], step=0.5, key="w_duty")
    vat_pct = st.number_input("VAT / GST %", min_value=0.0, value=st.session_state["d_vat_pct"], step=0.5, key="w_vat")
with lc4:
    fx_sens = st.slider("FX shock (USD→LKR) %", -20, 20, 0, key="w_fx_shock")
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
with m6:
    st.metric("Total Landed Cost", f"${res['total']:,.0f}")
    if lead_time_days: st.caption(f"Lead time est. {lead_time_days:.1f} days ({mode}).")
st.markdown("</div>", unsafe_allow_html=True)

# ---------- origin compare ----------
st.markdown("<div class='card'><b>Compare Origins (What-if)</b>", unsafe_allow_html=True)
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
if px is not None: st.plotly_chart(px.bar(out, x="Origin", y="TLC_USD", title="Total Landed Cost by Origin (USD)"), use_container_width=True)
else: st.bar_chart(out.set_index("Origin")["TLC_USD"])
st.markdown("</div>", unsafe_allow_html=True)

# ---------- guides / packing ----------
boxL, boxR = st.columns([1.05, .95], gap="small")
with boxL:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    with st.expander("Incoterms quick help"):
        st.markdown("- **FOB**: Buyer pays freight/insurance from origin port; seller handles export.")
        st.markdown("- **CIF**: Seller covers cost+insurance+freight to destination port.")
        st.markdown("- **DAP**: Delivered at place (unloaded not included); buyer handles import clearance/taxes.")
        st.markdown("- **DDP**: Seller covers everything including import duties/taxes.")
    st.markdown("**Tariff & NTM helper**")
    st.markdown("- 🧭 [MACMAP — Tariffs & Measures](https://www.macmap.org/)\n- 📊 [TradeMap — Flows](https://www.trademap.org/)\n- 🏛️ [Sri Lanka Customs](http://www.customs.gov.lk/)")
    st.caption("Use HS-6 to start; verify MFN vs FTA (e.g., ISFTA) vs GSP+ on national lines.")
    st.text_area("Notes", value=st.session_state.get("d_fx_note",""), key="w_notes", height=120)
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
        if px is not None: st.plotly_chart(px.bar(pk_df, x="Unit", y="Max cartons", title="Packing capacity"), use_container_width=True)
        else: st.bar_chart(pk_df.set_index("Unit")["Max cartons"])
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- scenarios ----------
st.markdown("<div class='card'><b>Save & Compare Scenarios</b>", unsafe_allow_html=True)
sc1, sc2 = st.columns([.44,.56])
with sc1:
    scenario_name = st.text_input("Scenario name", value="My scenario")
    if st.button("Save this scenario"):
        entry = {
            "name": scenario_name, "hs": hs_val, "reporter": reporter_name, "flow": flow, "years": years,
            "mode": mode, "origin": origin_q, "dest": dest_q, "dist_km": dist_km,
            "incoterm": incoterm, "fob": fob, "freight": freight, "insurance_pct": insurance_pct,
            "ins_base": ins_base, "duty_pct": duty_pct, "vat_pct": vat_pct, "broker": broker, "dray": dray,
            "shock_freight": shock_freight, "shock_tariff": shock_tariff,
            "tlc_usd": res["total"],
            "emissions_kg": ((dist_km or 0)*(st.session_state.get("w_shipkg",0)/1000.0)*(EF.get(mode,120.0)/1000.0))
        }
        st.session_state["scenarios"].append(entry)
        st.success(f"Saved: {scenario_name}")
with sc2:
    if st.session_state["scenarios"]:
        sdf = pd.DataFrame(st.session_state["scenarios"])
        st.dataframe(sdf[["name","mode","origin","dest","tlc_usd","emissions_kg"]], use_container_width=True)
        if px is not None and len(sdf) > 0:
            st.plotly_chart(px.bar(sdf, x="name", y="tlc_usd", title="Scenario TLC (USD)"), use_container_width=True)
            st.plotly_chart(px.bar(sdf, x="name", y="emissions_kg", title="Scenario Emissions (kg CO₂e)"), use_container_width=True)
        b2 = io.StringIO(); sdf.to_csv(b2, index=False)
        st.download_button("Download scenarios CSV", data=b2.getvalue(), file_name="gtm_scenarios.csv", mime="text/csv")
    else:
        st.info("No saved scenarios yet.")
st.markdown("</div>", unsafe_allow_html=True)

# ---------- export snapshot ----------
exp = {
    "hs": hs_val, "reporter": reporter_name, "flow": flow, "years": years,
    "fx_rate_usd_lkr": fx_use*(1+st.session_state.get("w_fx_shock",0)/100),
    "route": {"origin": origin_q, "dest": dest_q, "mode": mode,
              "distance_km": (None if dist_km is None else round(dist_km,1)),
              "lead_time_days": (None if not lead_time_days else round(lead_time_days,1))},
    "inputs": {"incoterm": incoterm, "fob": fob, "freight": freight, "insurance_pct": insurance_pct,
               "ins_base": ins_base, "duty_pct": duty_pct, "vat_pct": vat_pct, "broker": broker,
               "dray": dray, "shock_freight": shock_freight, "shock_tariff": shock_tariff},
    "outputs": res,
}
row = {}; row.update(exp["inputs"])
if not trend.empty:
    for _, r in trend.iterrows(): row[f"trend_{int(r.year)}"] = float(r.value_usd)
if not partners_df.empty:
    for _, r in partners_df.head(10).iterrows(): row[f"partner_{r.partner}"] = float(r.value_usd)
row["tlc_usd"] = res["total"]; row["cif"] = res["cif"]; row["duty"] = res["duty"]; row["vat"] = res["vat"]
csv_buf = io.StringIO(); pd.DataFrame([row]).to_csv(csv_buf, index=False)
dl1, dl2 = st.columns(2)
with dl1: st.download_button("Download scenario CSV", data=csv_buf.getvalue(), file_name="gtm_scenario.csv", mime="text/csv")
with dl2: st.download_button("Download scenario JSON", data=json.dumps(exp, indent=2), file_name="gtm_scenario.json", mime="application/json")

st.caption("Educational tool. Verify tariffs/NTMs with official sources (MACMAP, Sri Lanka Customs).")
