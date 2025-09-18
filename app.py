# app.py — GTM Global Trade & Logistics Dashboard (Sri Lanka Focus)
# v4.3 — Robust AeroDataBox (time window + fallback + cargo-only toggle),
#        Port Congestion (AISstream) + colored markers, Diagnostics, refined UI.

import io, os, json, math, datetime as dt, asyncio
from typing import Optional, Tuple

import requests, pandas as pd, numpy as np, streamlit as st
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import websockets

try:
    import plotly.express as px
except Exception:
    px = None

st.set_page_config(page_title="GTM — Global Trade & Logistics (Sri Lanka)", layout="wide", page_icon="📦")

# ===================== Theming =====================
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
    label {{ font-size:.8rem; color:var(--muted); margin-bottom:4px; display:block }}
    input, select, textarea {{ width:100%; padding:{density}; border-radius:10px; border:1px solid var(--border); background:var(--input-bg); color:var(--ink) }}
    .stButton>button {{ width:100%; background:linear-gradient(135deg, var(--primary), var(--accent)); color:white; border:none; font-weight:700; padding:.6rem .9rem; border-radius:10px }}
    .badge-ok {{ padding:.15rem .45rem; border-radius:999px; border:1px solid #16a34a; color:#16a34a; font-size:.75rem }}
    .badge-warn {{ padding:.15rem .45rem; border-radius:999px; border:1px solid #ef4444; color:#ef4444; font-size:.75rem }}
    header {{ border-bottom:none !important; }}
    [data-testid="stMetric"] div {{ gap: .15rem; }}
    </style>
    """, unsafe_allow_html=True)

if "theme" not in st.session_state: st.session_state["theme"] = "Light"
if "compact" not in st.session_state: st.session_state["compact"] = False
apply_css(st.session_state["theme"], st.session_state["compact"])

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

# ===================== Helpers & APIs =====================
UN_COMTRADE = "https://comtradeplus.un.org/api/get"
FX_URL      = "https://api.exchangerate.host/latest"
FX_TS_URL   = "https://api.exchangerate.host/timeseries"
WEATHER_API = "https://api.open-meteo.com/v1/forecast"
MARINE_API  = "https://marine-api.open-meteo.com/v1/marine"

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

def has_key(name: str) -> bool:
    try:
        if st.secrets.get(name): return True
    except Exception:
        pass
    return bool(os.getenv(name))

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_fx(base="USD", symbols=("LKR","EUR")):
    try:
        r = requests.get(FX_URL, params={"base":base,"symbols":",".join(symbols)}, timeout=20)
        r.raise_for_status(); return r.json().get("rates", {})
    except Exception:
        return {}

@st.cache_data(show_spinner=False, ttl=1800)
def fetch_fx_timeseries(base="USD", symbol="LKR", days=30):
    end = dt.date.today()
    start = end - dt.timedelta(days=days+1)
    try:
        r = requests.get(FX_TS_URL, params={"base":base,"symbols":symbol,"start_date":start.isoformat(),"end_date":end.isoformat()}, timeout=20)
        r.raise_for_status(); js = r.json().get("rates", {})
        rows=[{"date": d, "rate": float(v.get(symbol))} for d, v in sorted(js.items()) if v.get(symbol)]
        df = pd.DataFrame(rows).sort_values("date")
        if df.empty: return df, None
        df["ret"] = df["rate"].pct_change()
        vol = float(df["ret"].dropna().std()*100.0) if df["ret"].dropna().shape[0] else None
        return df, vol
    except Exception:
        return pd.DataFrame(), None

@st.cache_data(show_spinner=True, ttl=3600)
def fetch_comtrade(reporter="144", flow="1", years="2019,2020,2021,2022,2023", hs="300431"):
    params = {"type":"C","freq":"A","px":"HS","ps":years,"r":reporter,"p":"all","rg":flow,"cc":hs}
    try:
        r = requests.get(UN_COMTRADE, params=params, timeout=60)
        r.raise_for_status(); return pd.DataFrame(r.json().get("dataset", []))
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

# Geocoding & weather
geolocator = Nominatim(user_agent="gtm_dashboard/4.3 (edu)")
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
        r.raise_for_status(); return r.json().get("current", {})
    except Exception:
        return {}

def weather_risk(cur):
    if not cur: return "N/A", "—"
    wind = float(cur.get("wind_speed_10m", 0) or 0)
    precip = float(cur.get("precipitation", 0) or 0)
    if wind >= 12 or precip >= 5: return "High", f"Wind {wind} m/s, precip {precip} mm"
    if wind >= 8 or precip >= 2:  return "Moderate", f"Wind {wind} m/s, precip {precip} mm"
    return "OK", f"Wind {wind} m/s, precip {precip} mm"

# Marine (Open-Meteo Marine — free)
@st.cache_data(show_spinner=False, ttl=1800)
def fetch_marine(lat, lon):
    try:
        params = {
            "latitude": lat, "longitude": lon,
            "hourly": "wave_height,wave_period,wind_wave_height,wind_speed_10m",
            "length_unit": "metric", "wind_speed_unit": "ms", "timezone": "UTC"
        }
        r = requests.get(MARINE_API, params=params, timeout=14)
        r.raise_for_status()
        js = r.json().get("hourly", {})
        if not js: return None
        times = js.get("time", [])
        if not times: return None
        idx = 0
        return {
            "time": times[idx],
            "wave_height_m": float((js.get("wave_height") or [None])[idx] or 0),
            "wave_period_s": float((js.get("wave_period") or [None])[idx] or 0),
            "wind_wave_height_m": float((js.get("wind_wave_height") or [None])[idx] or 0),
            "wind_speed_ms": float((js.get("wind_speed_10m") or [None])[idx] or 0),
        }
    except Exception:
        return None

# ----------------- Live providers -----------------
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
    params = {"transportMode":"car","origin":f"{o[0]},{o[1]}", "destination":f"{d[0]},{d[1]}", "return":"summary", "departureTime":"now", "apikey":key}
    try:
        r = requests.get(url, params=params, timeout=12); r.raise_for_status()
        return int(r.json()["routes"][0]["sections"][0]["summary"]["duration"])
    except Exception:
        return None

def best_live_road_eta(o, d) -> Optional[int]:
    return google_driving_eta(o, d) or here_driving_eta(o, d)

# -------- Robust AeroDataBox (time window + fallback) --------
def aerodatabox_board(endpoint: str, iata: str, limit: int = 10, cargo_only: bool = True) -> pd.DataFrame:
    key = get_secret("AERODATABOX_KEY")
    if not key or not iata:
        return pd.DataFrame()

    headers = {"X-RapidAPI-Key": key, "X-RapidAPI-Host": "aerodatabox.p.rapidapi.com"}

    # Prefer a time window: now-12h → now+24h
    now = dt.datetime.utcnow()
    frm = (now - dt.timedelta(hours=12)).strftime("%Y-%m-%dT%H:%M")
    to  = (now + dt.timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M")
    url_window = f"https://aerodatabox.p.rapidapi.com/airports/iata/{iata}/{endpoint}/{frm}/{to}"

    def _parse(js):
        arr = js.get("arrivals") if "arrivals" in js else js.get("departures") if "departures" in js else js
        rows=[]
        cargo_hints = {"cargo","freighter","fx","5x","ups","dhl","cv","qf cargo","qr cargo","ek skycargo","tk cargo","sq cargo","ey cargo","ru"}
        cargo_prefix = {"FX","5X","5Y","CV","QY","RU","TK","QR","LH","SQ","EY","EK","UPS","DHL"}
        for it in (arr or []):
            fl = (it.get("number") or it.get("callSign") or "").upper()
            al = ((it.get("airline") or {}).get("name")) or ""
            text = f"{fl} {al}".lower()
            is_cargo = (it.get("isCargo") is True) or any(h in text for h in cargo_hints) or (fl[:2] in cargo_prefix)
            if cargo_only and not is_cargo:
                continue
            rows.append({
                "flight": fl,
                "from": (((it.get("departure") or {}).get("airport") or {}).get("iata")) or "",
                "to": (((it.get("arrival") or {}).get("airport") or {}).get("iata")) or "",
                "sched_local": ((it.get("arrival") or {}).get("scheduledTimeLocal")) or ((it.get("departure") or {}).get("scheduledTimeLocal")) or "",
                "status": it.get("status") or "",
                "airline": al,
                "cargo_detected": is_cargo,
            })
            if len(rows) >= limit:
                break
        return pd.DataFrame(rows)

    try:
        r = requests.get(url_window, headers=headers, params={"withLeg":"true","withCancelled":"false"}, timeout=14)
        if r.ok:
            df = _parse(r.json())
            if not df.empty:
                return df
        # Fallback to “now”
        url_now = f"https://aerodatabox.p.rapidapi.com/airports/iata/{iata}/{endpoint}/now"
        r2 = requests.get(url_now, headers=headers, params={"withLeg":"true","withCancelled":"false"}, timeout=14)
        if r2.ok:
            return _parse(r2.json())
    except Exception:
        pass
    return pd.DataFrame()

# -------- AISstream snapshot (for congestion & markers) --------
@st.cache_data(show_spinner=False, ttl=60)
def aisstream_snapshot(lat: float, lon: float, box_km: float = 30, seconds: int = 8) -> pd.DataFrame:
    key = get_secret("AISSTREAM_KEY")
    if not key or lat is None or lon is None:
        return pd.DataFrame()

    dlat = box_km / 111.0
    dlon = box_km / (111.0 * max(0.1, math.cos(math.radians(lat))))
    bbox = [[lat + dlat, lon - dlon], [lat - dlat, lon + dlon]]

    def _norm_nav_status(raw) -> str:
        if raw is None: return "other"
        try:
            code = int(raw)
            if code == 1: return "anchored"
            if code == 5: return "moored"
            if code in (0,7): return "underway"
            return "other"
        except Exception:
            s = str(raw).strip().lower()
            if "anchor" in s: return "anchored"
            if "moored" in s or "bert" in s: return "moored"
            if "underway" in s: return "underway"
            return "other"

    async def _grab():
        import json as _json
        rows = []
        try:
            async with websockets.connect("wss://stream.aisstream.io/v0/stream") as ws:
                sub = {
                    "APIKey": key,
                    "BoundingBoxes": [bbox],
                    "FilterMessageTypes": [
                        "PositionReport", "ExtendedClassBPositionReport", "StandardClassBPositionReport"
                    ]
                }
                await ws.send(_json.dumps(sub))
                stop = dt.datetime.utcnow() + dt.timedelta(seconds=seconds)
                while dt.datetime.utcnow() < stop:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=seconds)
                    except Exception:
                        break
                    try:
                        m = _json.loads(msg)
                        md = m.get("MetaData", {}) or {}
                        body = m.get("Message", {}) or {}
                        lat_v = md.get("latitude"); lon_v = md.get("longitude")
                        if lat_v is None or lon_v is None: continue
                        nav_raw = (body.get("NavigationalStatus") or body.get("NavigationStatus")
                                   or body.get("navStatus") or body.get("NavStatus"))
                        sog = body.get("Sog") or body.get("SpeedOverGround") or body.get("speedOverGround")
                        rows.append({
                            "shipname": md.get("ShipName") or "",
                            "mmsi": md.get("MMSI"),
                            "lat": float(lat_v), "lon": float(lon_v),
                            "time_utc": md.get("time_utc"),
                            "type": m.get("MessageType"),
                            "nav_status_raw": nav_raw,
                            "nav_status_norm": _norm_nav_status(nav_raw),
                            "sog_kn": float(sog) if sog not in (None, "") else None
                        })
                    except Exception:
                        continue
        except Exception:
            pass
        return rows

    try:
        data = asyncio.run(_grab())
    except RuntimeError:
        loop = asyncio.get_event_loop()
        data = loop.run_until_complete(_grab())

    df = pd.DataFrame(data)
    if not df.empty:
        df = df.sort_values("time_utc").drop_duplicates(subset=["mmsi"], keep="last").reset_index(drop=True)
    return df

# ===================== Hero =====================
st.markdown("""
<div class="hero">
  <div class="hero-title">GTM Global Trade & Logistics Dashboard — Sri Lanka Focus</div>
  <div class="hero-sub">Overview • Trade • Live Ops • Costs • Tools — free live signals (AeroDataBox, AISstream, Marine weather)</div>
</div>
""", unsafe_allow_html=True)

# ===================== Global controls =====================
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
        fx_live = fetch_fx(); fx_rate_live = float(fx_live.get("LKR", 0) or 0)
        fx_override = st.number_input("fx", min_value=0.0, step=0.01, value=0.0, key="w_fx", label_visibility="collapsed")
        fx_use = fx_override if fx_override > 0 else (fx_rate_live if fx_rate_live > 0 else None)
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

# ===================== Data prep =====================
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

o_pt = geocode_point(origin_q); d_pt = geocode_point(dest_q)
fx_df, fx_vol = fetch_fx_timeseries(base="USD", symbol="LKR", days=30)

# ===================== Tabs =====================
o_tab, t_tab, live_tab, cost_tab, tools_tab = st.tabs(["Overview", "Trade", "Live Ops", "Costs", "Tools"])

# ----- Overview -----
with o_tab:
    s_google = has_key("GOOGLE_MAPS_KEY") or has_key("HERE_API_KEY")
    s_aero   = has_key("AERODATABOX_KEY")
    s_ais    = has_key("AISSTREAM_KEY")
    s_mt     = has_key("MARINETRAFFIC_KEY")
    s_fx_ok  = fx_use is not None
    s_geo    = bool(o_pt and d_pt)
    badge = lambda ok, lbl: f"<span class='{'badge-ok' if ok else 'badge-warn'}'>{'✅' if ok else '⚠️'} {lbl}</span>"
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(
        f"**Connections:** {badge(s_fx_ok,'FX')} &nbsp; {badge(s_geo,'Geocoding')} &nbsp; "
        f"{badge(s_google,'Road ETA/Incidents')} &nbsp; {badge(s_aero,'Air cargo (AeroDataBox)')} &nbsp; "
        f"{badge(s_ais,'Cargo AIS (AISstream)')} &nbsp; {badge(s_mt,'Cargo AIS (MarineTraffic)')}",
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

    tp_val = float(_top["value_usd"]) if not partners_df.empty else 0.0
    tp_name = _top["partner"] if not partners_df.empty else "—"
    partners_count = partners_df["partner"].nunique() if not partners_df.empty else 0
    fx_txt = f"{fx_use:.2f}" if fx_use else "—"
    period_lbl = f"{years.split(',')[0]}→{years.split(',')[-1]}"

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"""
    ### Snapshot ({period_lbl})
    - **Total trade:** {total_trade:,.0f} USD for HS **{hs_val}** ({flow} · {rep_name}).
    - **Top partner:** **{tp_name}** with **{tp_val:,.0f} USD** across {partners_count} partner(s).
    - **Growth:** Avg YoY **{avg_yoy*100:.1f}%** · CAGR **{cagr_val*100:.1f}%**.
    - **FX USD→LKR:** **{fx_txt}** {'(override)' if fx_override>0 else ''}.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    k = st.columns(3)
    with k[0]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.metric("Total Trade (USD)", f"{total_trade:,.0f}")
        st.metric("Partners shown", f"{partners_count}")
        st.markdown("</div>", unsafe_allow_html=True)
    with k[1]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.metric("Avg YoY growth", f"{avg_yoy*100:.1f}%")
        st.metric("CAGR", f"{cagr_val*100:.1f}%")
        st.markdown("</div>", unsafe_allow_html=True)
    with k[2]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.metric("USD→LKR", fx_txt)
        if fx_vol is not None: st.metric("FX Vol (30d stdev)", f"{fx_vol:.2f}%")
        if px is not None and not fx_df.empty:
            st.plotly_chart(px.line(fx_df, x="date", y="rate", title="USD→LKR (last 30 days)"), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ----- Trade -----
with t_tab:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    tabs = st.tabs(["Trend", "Partners", "Unit values", "Raw"])
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

# ----- Live Ops -----
with live_tab:
    st.markdown("<div class='card'><b>Route & Operations</b>", unsafe_allow_html=True)
    if not (o_pt and d_pt):
        st.info("Enter clear locations (e.g., 'Bengaluru BLR' → 'Colombo CMB'). For air boards include IATA codes.")
    else:
        a=(o_pt[0],o_pt[1]); b=(d_pt[0],d_pt[1])
        # distance & lead time
        R=6371.0
        from math import radians, sin, cos, asin, sqrt
        lat1,lon1,lat2,lon2 = map(radians,[a[0],a[1],b[0],b[1]])
        dlat, dlon = lat2-lat1, lon2-lon1
        h = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
        dist_km = 2*R*asin(sqrt(h))
        speed = 800 if mode=="Air" else (35*24 if mode=="Sea" else 60)  # km/h
        handling, clearance, local = (0.8,0.8,0.3) if mode=="Air" else ((1.5,2.0,0.5) if mode=="Sea" else (0.2,0.5,0.2))
        lead_days = (dist_km/speed)/24 + handling + clearance + local

        fmap = folium.Map(location=[((a[0]+b[0])/2), ((a[1]+b[1])/2)], zoom_start=4, control_scale=True)
        folium.Marker(a, tooltip=f"Origin: {origin_q}").add_to(fmap)
        folium.Marker(b, tooltip=f"Destination: {dest_q}").add_to(fmap)
        folium.PolyLine([a,b], color="#2563eb", weight=4).add_to(fmap)

        # cargo-only toggle for boards
        cargo_only = st.toggle("Cargo flights only", value=True, help="Turn off if the airport returns few/zero labelled cargo flights.")

        # ---- AIS snapshot once, reuse for markers + congestion ----
        ais_df = pd.DataFrame()
        if has_key("AISSTREAM_KEY"):
            ais_df = aisstream_snapshot(d_pt[0], d_pt[1], box_km=30, seconds=8)
            color_map = {"anchored":"#f59e0b", "moored":"#ef4444", "underway":"#10b981", "other":"#64748b"}
            if not ais_df.empty:
                for _, r in ais_df.iterrows():
                    color = color_map.get(r["nav_status_norm"], "#64748b")
                    folium.CircleMarker(
                        [r["lat"], r["lon"]], radius=4, color=color, fill=True, fill_opacity=0.9,
                        tooltip=f"{r['shipname'] or 'Vessel'} (MMSI {r['mmsi']}) • {r['nav_status_norm']}"
                    ).add_to(fmap)

        st_folium(fmap, height=420, use_container_width=True)
        st.caption(f"Distance ≈ {dist_km:,.0f} km • Estimated lead time: {lead_days:.1f} days ({mode})")

        # weather (origin/dest)
        ow = fetch_weather(a[0],a[1]); dw = fetch_weather(b[0],b[1])
        orisk, omsg = weather_risk(ow); drisk, dmsg = weather_risk(dw)
        st.write(f"🌤️ Origin: **{orisk}** ({omsg}) · Destination: **{drisk}** ({dmsg})")

        # marine (destination)
        marine = fetch_marine(d_pt[0], d_pt[1])
        if marine:
            st.markdown(f"🌊 **Port marine conditions (Open-Meteo Marine)** — Wave **{marine['wave_height_m']:.1f} m**, "
                        f"Wind **{marine['wind_speed_ms']:.1f} m/s**, Period **{marine['wave_period_s']:.0f} s** (UTC {marine['time']}).")

        # live road ETA (if Road)
        if mode=="Road":
            secs = best_live_road_eta(a,b)
            if secs: st.metric("🚚 Live road ETA (traffic)", f"{secs/3600:.1f} h")

        # traffic incidents (HERE)
        if has_key("HERE_API_KEY"):
            def here_traffic_incidents(a: Tuple[float,float], b: Tuple[float,float]) -> pd.DataFrame:
                key = get_secret("HERE_API_KEY")
                min_lat, max_lat = min(a[0],b[0]), max(a[0],b[0])
                min_lon, max_lon = min(a[1],b[1]), max(a[1],b[1])
                pad_lat = (max_lat - min_lat) * 0.2 + 0.1
                pad_lon = (max_lon - min_lon) * 0.2 + 0.1
                top = max_lat + pad_lat; left = min_lon - pad_lon
                bottom = min_lat - pad_lat; right = max_lon + pad_lon
                url = "https://traffic.ls.hereapi.com/traffic/6.2/incidents.json"
                params = {"bbox": f"{top},{left};{bottom},{right}", "criticality": "critical,major,minor", "apiKey": key, "locationreferences": "none"}
                try:
                    r = requests.get(url, params=params, timeout=12); r.raise_for_status()
                    data = r.json()
                    items = (((data or {}).get("TRAFFICITEMS") or {}).get("TRAFFICITEM")) or []
                    rows=[]
                    for it in items:
                        crit = (it.get("CRITICALITY") or [{}])[0].get("DESCRIPTION","").lower()
                        cat  = it.get("TRAFFICITEMTYPEDESC","")
                        desc = (it.get("TRAFFICITEMDESCRIPTION") or [{}])[0].get("content","")
                        rows.append({"criticality": crit, "category": cat, "description": desc})
                    return pd.DataFrame(rows)
                except Exception:
                    return pd.DataFrame()
            inc = here_traffic_incidents(a,b)
            if not inc.empty:
                total = len(inc); majors = inc[inc["criticality"].str.contains("critical|major", na=False)].shape[0]
                st.markdown(f"🛑 **Road incidents (HERE):** {total} total · {majors} critical/major")
                st.dataframe(inc.head(40), use_container_width=True, height=240)

        # air cargo boards
        oiata = parse_iata(origin_q); diata = parse_iata(dest_q)
        if has_key("AERODATABOX_KEY") and (oiata or diata):
            st.markdown("### ✈️ Air cargo boards (AeroDataBox)")
            c1,c2 = st.columns(2)
            with c1:
                st.caption(f"Origin departures — {oiata or '—'}")
                dep = aerodatabox_board("departures", oiata, limit=12, cargo_only=cargo_only) if oiata else pd.DataFrame()
                if not dep.empty: st.dataframe(dep, use_container_width=True, height=260)
                else: st.info("No departures returned for the current window.")
            with c2:
                st.caption(f"Destination arrivals — {diata or '—'}")
                arr = aerodatabox_board("arrivals", diata, limit=12, cargo_only=cargo_only) if diata else pd.DataFrame()
                if not arr.empty: st.dataframe(arr, use_container_width=True, height=260)
                else: st.info("No arrivals returned for the current window.")

        # -------- Port Congestion tile --------
        if has_key("AISSTREAM_KEY") and not ais_df.empty:
            counts = ais_df["nav_status_norm"].value_counts().to_dict()
            anchored = int(counts.get("anchored", 0))
            moored   = int(counts.get("moored",   0))
            underway = int(counts.get("underway", 0))
            congestion_index = anchored + moored

            if "ais_trend" not in st.session_state: st.session_state["ais_trend"] = []
            st.session_state["ais_trend"].append({
                "t": dt.datetime.utcnow().strftime("%H:%M:%S"),
                "anchored": anchored, "moored": moored, "underway": underway,
                "congestion": congestion_index
            })
            st.session_state["ais_trend"] = st.session_state["ais_trend"][-60:]

            st.markdown("### ⚓ Port congestion (live, AISstream)")
            cc1, cc2, cc3, cc4 = st.columns(4)
            with cc1: st.metric("Congestion index", f"{congestion_index}")
            with cc2: st.metric("Anchored", f"{anchored}")
            with cc3: st.metric("Moored/berth", f"{moored}")
            with cc4: st.metric("Underway", f"{underway}")

            trend_df = pd.DataFrame(st.session_state["ais_trend"])
            if px is not None and not trend_df.empty:
                fig = px.line(trend_df, x="t", y=["congestion","anchored","moored"],
                              title="Port congestion trend (last ~60 samples)")
                fig.update_layout(legend_title_text="")
                st.plotly_chart(fig, use_container_width=True)
            elif not trend_df.empty:
                st.line_chart(trend_df.set_index("t")[["congestion","anchored","moored"]])

            with st.expander("Latest AIS snapshot (normalized)"):
                st.dataframe(
                    ais_df[["shipname","mmsi","nav_status_norm","sog_kn","lat","lon","time_utc"]].sort_values("nav_status_norm"),
                    use_container_width=True, height=260
                )

    st.markdown("</div>", unsafe_allow_html=True)

# ----- Costs (Incoterms engine + scenarios) -----
with cost_tab:
    st.markdown("<div class='card'><b>Landed Cost • Exact Incoterms</b>", unsafe_allow_html=True)
    lc1, lc2, lc3, lc4 = st.columns([1.1,1.1,1.1,1.1])
    with lc1:
        incoterm = st.selectbox("Incoterm",
            ["EXW","FCA","FAS","FOB","CFR","CIF","CPT","CIP","DAP","DPU","DDP"], index=5, key="incoterm_sel")
        invoice_value = st.number_input(f"Seller invoice value ({incoterm}) — USD", min_value=0.0, value=150000.0, step=100.0, key="inv")
        qty_units = st.number_input("Units / pieces (optional)", min_value=0, value=10000, key="qty")
    with lc2:
        main_freight = st.number_input("Main carriage freight (USD)", min_value=0.0, value=0.0, step=50.0, key="freight")
        origin_charges = st.number_input("Origin charges — not dutiable (USD)", min_value=0.0, value=0.0, step=10.0, key="orgchg")
        dest_charges = st.number_input("Destination charges — not dutiable (USD)", min_value=0.0, value=400.0, step=10.0, key="destchg")
    with lc3:
        brokerage = st.number_input("Brokerage & regulatory (USD)", min_value=0.0, value=350.0, step=10.0, key="broker")
        inland = st.number_input("Inland transport to warehouse (USD)", min_value=0.0, value=300.0, step=10.0, key="inland")
        other_local = st.number_input("Other local charges (USD)", min_value=0.0, value=0.0, step=10.0, key="otherlocal")
    with lc4:
        ins_mode = st.radio("Insurance input", ["Percent","Amount"], index=0, horizontal=True, key="insmode")
        ins_base = st.selectbox("If %: base", ["Invoice","Invoice+Freight"], index=1, key="insbase")
        ins_pct  = st.number_input("Insurance %", min_value=0.0, value=1.0, step=0.1, key="inspct")
        ins_amt  = st.number_input("Insurance amount (USD)", min_value=0.0, value=0.0, step=10.0, key="insamt")
        seller_pays_import_taxes = st.checkbox("Seller pays import taxes (DDP)", value=(incoterm=="DDP"), key="ddpflag")

    INCOTERM_INCLUDES = {
        "EXW": (False, False), "FCA": (False, False), "FAS": (False, False), "FOB": (False, False),
        "CFR": (True,  False), "CIF": (True,  True),  "CPT": (True,  False), "CIP": (True,  True),
        "DAP": (True,  False), "DPU": (True,  False), "DDP": (True,  False),
    }
    freight_included, insurance_included = INCOTERM_INCLUDES.get(incoterm, (False, False))

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

    duty_pct = st.number_input("Import duty % (on customs value)", min_value=0.0, value=0.0, step=0.1, key="dutyp")
    other_tax_pct = st.number_input("Other tariff/levy % (on customs value, optional)", min_value=0.0, value=0.0, step=0.1, key="othp")
    vat_pct = st.number_input("VAT / GST % (on customs value + duty + other)", min_value=0.0, value=0.0, step=0.1, key="vatp")

    duty = customs_value * (duty_pct/100.0)
    other_tax = customs_value * (other_tax_pct/100.0)
    vat_base = customs_value + duty + other_tax
    vat = vat_base * (vat_pct/100.0)

    taxes_payable_by_buyer = 0.0 if seller_pays_import_taxes else (duty + other_tax + vat)
    total_landed = (
        invoice_value
        + (0.0 if freight_included else main_freight)
        + (0.0 if insurance_included else (insurance_add_for_customs if ins_mode=="Percent" else ins_amt))
        + origin_charges + dest_charges + brokerage + inland + other_local
        + taxes_payable_by_buyer
    )

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
    if qty_units and qty_units>0: st.caption(f"Landed cost per unit: **${(total_landed/qty_units):,.2f}**")
    st.markdown("</div>", unsafe_allow_html=True)

    # Compare origin (what-if)
    st.markdown("<div class='card'><b>Compare Origins (What-if)</b>", unsafe_allow_html=True)
    comp = pd.DataFrame([
        {"Origin":"India (ISFTA)","Invoice":invoice_value,"Incoterm":incoterm,"Freight":main_freight,"Duty%":duty_pct},
        {"Origin":"Denmark (MFN)","Invoice":invoice_value*1.05,"Incoterm":incoterm,"Freight":main_freight*1.8,"Duty%":max(duty_pct,2.0)},
        {"Origin":"Singapore","Invoice":invoice_value*1.02,"Incoterm":incoterm,"Freight":main_freight*1.2,"Duty%":duty_pct},
    ])
    rows=[]
    for _, r in comp.iterrows():
        fr_in, ins_in = INCOTERM_INCLUDES.get(r.Incoterm, (False,False))
        fr_add = 0.0 if fr_in else r.Freight
        if ins_in:
            ins_add = 0.0
        else:
            ins_add = ( (r.Invoice + (0.0 if fr_in else r.Freight)) * (ins_pct/100.0) ) if ins_mode=="Percent" else ins_amt
        cv = r.Invoice + fr_add + ins_add
        duty_c = cv * (r["Duty%"]/100.0)
        other_c= cv * (other_tax_pct/100.0)
        vat_c  = (cv + duty_c + other_c) * (vat_pct/100.0)
        taxes_buyer = 0.0 if seller_pays_import_taxes else (duty_c + vat_c + other_c)
        tlc = (r.Invoice + (0.0 if fr_in else r.Freight) + (0.0 if ins_in else ins_add) +
               origin_charges + dest_charges + brokerage + inland + other_local + taxes_buyer)
        rows.append({"Origin":r.Origin,"TLC_USD":tlc})
    out=pd.DataFrame(rows)
    st.dataframe(out, use_container_width=True)
    if px is not None: st.plotly_chart(px.bar(out, x="Origin", y="TLC_USD", title="Total Landed Cost by Origin (USD)"), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Save/Export
    st.markdown("<div class='card'><b>Save & Export Scenario</b>", unsafe_allow_html=True)
    sc_name = st.text_input("Scenario name", value="My scenario")
    if st.button("Save current scenario"):
        if "scenarios" not in st.session_state: st.session_state["scenarios"]=[]
        st.session_state["scenarios"].append({
            "name":sc_name,"incoterm":incoterm,"invoice_value":invoice_value,"freight":main_freight,
            "insurance_add":(0.0 if INCOTERM_INCLUDES.get(incoterm,(False,False))[1] else ( (invoice_value + (0.0 if INCOTERM_INCLUDES.get(incoterm,(False,False))[0] else main_freight)) * (ins_pct/100.0) if ins_mode=="Percent" else ins_amt )),
            "customs_value":customs_value,"duty":duty,"vat":vat,"other":other_tax,
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

    snap = {
      "incoterm": incoterm, "invoice_value": invoice_value, "freight_included": INCOTERM_INCLUDES.get(incoterm,(False,False))[0],
      "insurance_included": INCOTERM_INCLUDES.get(incoterm,(False,False))[1],
      "customs_value": customs_value, "duty_pct": duty_pct, "other_tax_pct": other_tax_pct, "vat_pct": vat_pct,
      "duty": duty, "other_tax": other_tax, "vat": vat, "taxes_payable_by_buyer": (0.0 if (incoterm=='DDP' or st.session_state.get('ddpflag')) else (duty+other_tax+vat)),
      "origin_charges": origin_charges, "dest_charges": dest_charges, "brokerage": brokerage, "inland": inland, "other_local": other_local,
      "total_landed": total_landed, "qty_units": qty_units, "per_unit": (total_landed/qty_units if qty_units else None)
    }
    buf_json = json.dumps(snap, indent=2)
    buf_csv  = io.StringIO(); pd.DataFrame([snap]).to_csv(buf_csv, index=False)
    d1,d2 = st.columns(2)
    with d1: st.download_button("Download snapshot JSON", data=buf_json, file_name="gtm_snapshot.json", mime="application/json")
    with d2: st.download_button("Download snapshot CSV",  data=buf_csv.getvalue(), file_name="gtm_snapshot.csv",  mime="text/csv")
    st.markdown("</div>", unsafe_allow_html=True)

# ----- Tools -----
with tools_tab:
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

# ================= Diagnostics (connectivity & keys) =================
diag_tab = st.tabs(["Diagnostics"])[0]
with diag_tab:
    st.markdown("<div class='card'><b>Live integrations — self test</b>", unsafe_allow_html=True)

    def _mask(v: Optional[str], show_last=4):
        if not v: return "—"
        v = str(v)
        return "•"*(max(0,len(v)-show_last)) + v[-show_last:]

    keys_row = pd.DataFrame([{
        "AERODATABOX_KEY": _mask(get_secret("AERODATABOX_KEY")),
        "AISSTREAM_KEY":   _mask(get_secret("AISSTREAM_KEY")),
        "HERE_API_KEY":    _mask(get_secret("HERE_API_KEY")),
        "GOOGLE_MAPS_KEY": _mask(get_secret("GOOGLE_MAPS_KEY")),
    }]).T
    keys_row.columns = ["visible_to_app"]
    st.write("**Secrets detected (masked):**")
    st.dataframe(keys_row, use_container_width=True, height=150)

    st.write("**Choose a simple test route and airport** (used for pings)")
    colA, colB, colC = st.columns(3)
    with colA:
        test_origin = st.text_input("Test origin (geocode)", value="Negombo, Sri Lanka")
    with colB:
        test_dest   = st.text_input("Test destination (geocode)", value="Colombo Port, Sri Lanka")
    with colC:
        test_iata   = st.text_input("Test IATA (AeroDataBox)", value="CMB")

    o_geo = geocode_point(test_origin) or (7.2008,79.8737,"Negombo")
    d_geo = geocode_point(test_dest)   or (6.9497,79.8440,"Colombo Port")
    a=(o_geo[0], o_geo[1]); b=(d_geo[0], d_geo[1])

    st.markdown("### Run pings")
    b1, b2, b3, b4, b5, b6 = st.columns(6)
    go_fx      = b1.button("FX (exchangerate.host)")
    go_com     = b2.button("UN Comtrade")
    go_marine  = b3.button("Marine (Open-Meteo)")
    go_aero    = b4.button("AeroDataBox (arrivals)")
    go_eta     = b5.button("Road ETA (Google/HERE)")
    go_ais     = b6.button("AISstream snapshot")

    def box(ok, title, meta, payload):
        color = "#16a34a" if ok else "#ef4444"
        st.markdown(f"<div class='card' style='border-color:{color}'>"
                    f"<b style='color:{color}'>{'✅' if ok else '⚠️'} {title}</b><br>"
                    f"<span style='color:#526581'>{meta}</span></div>", unsafe_allow_html=True)
        if payload is not None:
            with st.expander("Show response"):
                if isinstance(payload, (dict,list)):
                    st.json(payload)
                else:
                    st.code(str(payload))

    if go_fx:
        try:
            r = requests.get(FX_URL, params={"base":"USD","symbols":"LKR"}, timeout=12)
            ok = (r.status_code==200 and "rates" in r.json())
            box(ok, "FX ping", f"HTTP {r.status_code}", r.json() if ok else r.text)
        except Exception as e:
            box(False, "FX ping", f"Error: {e}", None)

    if go_com:
        try:
            r = requests.get(UN_COMTRADE, params={"type":"C","freq":"A","px":"HS","ps":"2023","r":"144","p":"all","rg":"1","cc":"300431"}, timeout=25)
            js = r.json() if r.headers.get("content-type","").startswith("application/json") else {"text": r.text}
            ok = (r.status_code==200 and ("dataset" in js))
            meta = f"HTTP {r.status_code}, rows={len(js.get('dataset',[])) if isinstance(js,dict) else 'n/a'}"
            box(ok, "UN Comtrade ping", meta, js if ok else r.text)
        except Exception as e:
            box(False, "UN Comtrade ping", f"Error: {e}", None)

    if go_marine:
        try:
            js = fetch_marine(b[0], b[1])
            ok = bool(js)
            box(ok, "Open-Meteo Marine ping", f"Point=({b[0]:.4f},{b[1]:.4f})", js or {})
        except Exception as e:
            box(False, "Open-Meteo Marine ping", f"Error: {e}", None)

    if go_aero:
        try:
            df_ping = aerodatabox_board("arrivals", test_iata.strip().upper(), limit=5, cargo_only=True)
            ok = not df_ping.empty
            meta = f"Rows={len(df_ping)} • Key={_mask(get_secret('AERODATABOX_KEY'))}"
            box(ok, "AeroDataBox arrivals ping", meta, df_ping.head(5).to_dict(orient="records") if ok else "No rows")
        except Exception as e:
            box(False, "AeroDataBox arrivals ping", f"Error: {e}", None)

    if go_eta:
        secs = None
        prov = None
        try:
            if has_key("GOOGLE_MAPS_KEY"):
                secs = google_driving_eta(a,b); prov = "Google"
            if secs is None and has_key("HERE_API_KEY"):
                secs = here_driving_eta(a,b); prov = "HERE"
            ok = secs is not None
            meta = f"{prov or '—'} ETA = {secs/60:.1f} min" if ok else "No provider key available or no ETA"
            box(ok, "Road ETA ping", meta, None)
        except Exception as e:
            box(False, "Road ETA ping", f"Error: {e}", None)

    if go_ais:
        try:
            df_ping = aisstream_snapshot(b[0], b[1], box_km=30, seconds=8)
            ok = not df_ping.empty
            counts = df_ping["nav_status_norm"].value_counts().to_dict() if ok else {}
            meta = f"Rows={len(df_ping)} • anchored={counts.get('anchored',0)} moored={counts.get('moored',0)} underway={counts.get('underway',0)}"
            box(ok, "AISstream snapshot ping", meta, df_ping.head(10).to_dict(orient="records") if ok else "No messages received (try again)")
        except Exception as e:
            box(False, "AISstream snapshot ping", f"Error: {e}", None)

st.caption("Educational tool. Verify tariffs/NTM rules with official sources (MACMAP, Sri Lanka Customs).")
