"""Weather from energy: a plain-English prototype (Great Britain).

Reads wind and solar generation, works out roughly what the weather is
doing, explains what that means for electricity, and checks itself against
independent weather data so visitors can judge how far to trust it.

Setup:  pip install flask pandas requests pvlive-api
Run:    python app.py            live data
        python app.py --demo     made-up data, works offline
Open:   http://127.0.0.1:5000
"""
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, render_template_string

DEMO = "--demo" in sys.argv
log = logging.getLogger("weather")

# ---- Settings you can tune --------------------------------------------------
HISTORY_DAYS = 7                 # history used to work out what is "normal"
CACHE_SECONDS = 600              # call the data sources at most every 10 minutes
MIN_WIND_REFERENCE_MW = 10_000   # placeholder: stops a calm week making modest wind look strong
NIGHT_BASELINE_MW = 300          # below this, solar says nothing about cloud (night or dusk)
MIN_HOURS_FOR_STATS = 24         # don't quote accuracy figures from less data than this
RECENT_ROWS = 12                 # hours shown in the "how did it do?" table

ELEXON_URL = "https://data.elexon.co.uk/bmrs/api/v1/generation/actual/per-type"
LONDON = ZoneInfo("Europe/London")

# Independent weather data for the accuracy check: Open-Meteo, which blends national
# weather-service models. This is MODELLED weather, not station readings. To use Met Office
# station observations instead, replace fetch_reference() so it returns the same three
# hourly series. The free Open-Meteo API is for non-commercial use only; check its terms.
REFERENCE_URL = "https://api.open-meteo.com/v1/forecast"
# Rough, hand-picked spots near where wind farms and solar farms are concentrated.
# Refine these with real farm locations.
WIND_POINTS = [(53.9, 1.8), (58.0, -2.8), (55.7, -4.2), (54.0, -3.5), (51.6, 1.5), (52.4, -3.6)]
SUN_POINTS = [(50.9, -1.4), (51.5, -2.0), (52.0, -0.5), (52.6, 0.5), (53.3, -1.0), (50.6, -4.5)]

# Each band: (upper limit, label, headline word, plain-English sentence). Checked in order.
WIND_BANDS = [
    (0.10, "calm", "Calm", "It is very calm. Hardly any wind is reaching the wind farms."),
    (0.25, "light", "Light winds", "Winds are light."),
    (0.50, "moderate", "Moderate winds", "There is a moderate breeze across the wind farms."),
    (0.75, "windy", "Windy", "It is windy across the wind farms."),
    (9.99, "very windy", "Very windy",
     "It is very windy. Wind farms are producing close to their recent peak."),
]
SOLAR_BANDS = [
    (0.25, "overcast", "overcast",
     "Skies look heavily overcast. Solar farms are making less than a quarter of what a clear day gives."),
    (0.55, "cloudy", "cloudy", "It looks mostly cloudy."),
    (0.85, "partly cloudy", "some cloud", "There is some cloud about."),
    (9.99, "clear", "sunny", "Skies look clear."),
]
WIND_ORDER = [b[1] for b in WIND_BANDS]    # calm ... very windy
SOLAR_ORDER = [b[1] for b in SOLAR_BANDS]  # overcast ... clear
TREND_TEXT = {"rising": " It is getting stronger.", "falling": " It is easing off.", "steady": ""}


# ---- Getting the data -------------------------------------------------------
def fetch_wind(start, end):
    """Wind output in MW (onshore + offshore), one value per half hour.

    Uses Elexon's actual-generation-per-type data. It covers wind farms
    connected to the main grid only, not small "embedded" wind farms.
    Timestamps are the END of each half hour, to match PV_Live.
    """
    values = {}
    day = start
    while day < end:
        next_day = min(day + timedelta(days=1), end)
        response = requests.get(
            ELEXON_URL,
            params={
                "from": day.strftime("%Y-%m-%dT%H:%MZ"),
                "to": next_day.strftime("%Y-%m-%dT%H:%MZ"),
                "format": "json",
            },
            timeout=30,
        )
        response.raise_for_status()
        for period in response.json()["data"]:
            wind = [d["quantity"] for d in period["data"] if d["psrType"].startswith("Wind")]
            if wind:
                period_end = pd.Timestamp(period["startTime"]) + pd.Timedelta(minutes=30)
                values[period_end] = float(sum(wind))
        day = next_day
    return pd.Series(values, name="wind_mw").sort_index()


def fetch_solar(start, end):
    """Estimated GB solar output in MW, half-hourly, from Sheffield Solar's PV_Live."""
    from pvlive_api import PVLive

    data = PVLive().between(start=start, end=end, dataframe=True)
    index = pd.to_datetime(data["datetime_gmt"], utc=True)
    return pd.Series(data["generation_mw"].values, index=index, name="solar_mw").sort_index()


def fetch_reference(days):
    """Independent hourly weather for the same period, averaged over several spots.

    Returns three hourly UTC series: wind speed at turbine height (m/s),
    cloud cover (%) and sunlight reaching the ground (W/m2).
    """
    def average_over(points, variables):
        response = requests.get(
            REFERENCE_URL,
            params={
                "latitude": ",".join(str(lat) for lat, _ in points),
                "longitude": ",".join(str(lon) for _, lon in points),
                "hourly": ",".join(variables),
                "wind_speed_unit": "ms",
                "timezone": "UTC",
                "past_days": days,
                "forecast_days": 1,
            },
            timeout=30,
        )
        response.raise_for_status()
        places = response.json()
        if isinstance(places, dict):  # one location comes back as a single object
            places = [places]
        frames = [
            pd.DataFrame({v: place["hourly"][v] for v in variables},
                         index=pd.to_datetime(place["hourly"]["time"], utc=True))
            for place in places
        ]
        return pd.concat(frames).groupby(level=0).mean()

    wind = average_over(WIND_POINTS, ["wind_speed_120m"])
    sun = average_over(SUN_POINTS, ["cloud_cover", "shortwave_radiation"])
    return {
        "wind_speed": wind["wind_speed_120m"],
        "cloud": sun["cloud_cover"],
        "radiation": sun["shortwave_radiation"],
    }


def demo_data():
    """Made-up but realistic-looking data, so the site can be tried offline."""
    n = HISTORY_DAYS * 48
    index = pd.date_range(end=pd.Timestamp.now(tz="UTC").floor("30min"), periods=n, freq="30min")
    rng = np.random.default_rng(7)
    hours = index.hour + index.minute / 60
    sun = np.clip(np.sin((hours - 6) / 12 * np.pi), 0, None)
    day_number = np.arange(n) // 48
    clearness = rng.uniform(0.5, 1.0, day_number.max() + 1)
    clearness[-1] = 0.35  # make the latest day a cloudy one
    solar = 9000 * sun * clearness[day_number] * rng.uniform(0.9, 1.0, n)
    wind = np.clip(6000 + np.cumsum(rng.normal(0, 350, n)), 500, 16000)
    return pd.Series(wind, index), pd.Series(solar, index)


def demo_reference(wind, solar):
    """Made-up 'official' weather that roughly agrees with the demo data, plus noise."""
    rng = np.random.default_rng(11)
    wind_h, solar_h = hourly(wind), hourly(solar)
    share = (wind_h / max(float(wind.max()), MIN_WIND_REFERENCE_MW)).clip(0, 1)
    speed = (3 + 9 * share ** (1 / 3) + rng.normal(0, 1.3, len(share))).clip(lower=0)
    best = solar_h.groupby(solar_h.index.hour).transform("max")
    ratio = (solar_h / best).fillna(0).clip(0, 1)
    cloud = (100 * (1 - ratio) + rng.normal(0, 18, len(ratio))).clip(0, 100)
    radiation = solar_h / 9000 * 800 * rng.uniform(0.9, 1.1, len(solar_h))
    return {"wind_speed": speed, "cloud": cloud, "radiation": radiation}


def load_all():
    """Return (wind, solar, reference). Reference is None if it could not be loaded."""
    if DEMO:
        wind, solar = demo_data()
        return wind, solar, demo_reference(wind, solar)
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = end - timedelta(days=HISTORY_DAYS)
    wind, solar = fetch_wind(start, end), fetch_solar(start, end)
    try:
        reference = fetch_reference(HISTORY_DAYS)
    except Exception as error:  # the site still works without the accuracy check
        log.warning("Independent weather data unavailable: %s", error)
        reference = None
    return wind, solar, reference


# ---- Turning numbers into weather -------------------------------------------
def band(value, bands):
    for upper, label, word, sentence in bands:
        if value < upper:
            return label, word, sentence
    return bands[-1][1:]


def trend(series, reference, hours=2):
    steps = hours * 2  # half-hourly data
    if len(series) <= steps:
        return "steady"
    change = (series.iloc[-1] - series.iloc[-1 - steps]) / reference
    if change > 0.10:
        return "rising"
    if change < -0.10:
        return "falling"
    return "steady"


def electricity_meaning(wind_label, solar_label):
    """Return (what it means for electricity, optional tip). Rules of thumb, not predictions."""
    wind_strong = wind_label in ("windy", "very windy")
    wind_weak = wind_label in ("calm", "light")
    sun_good = solar_label in ("clear", "partly cloudy")
    sun_poor = solar_label in ("cloudy", "overcast", "night")
    flexible_now = "If you can be flexible, now is a good time to use power."
    if wind_strong and sun_good:
        return ("Wind and sun are both producing well, so less gas is usually needed "
                "and wholesale prices tend to be lower.", flexible_now)
    if wind_strong:
        return ("Wind is doing most of the work. When it is strong, less gas is usually "
                "needed and prices tend to be lower.", flexible_now)
    if wind_weak and sun_poor:
        return ("Little wind and little sun, so gas power stations and imports usually "
                "fill the gap, and prices tend to be higher.",
                "If you can be flexible, it may be greener to wait.")
    if wind_weak and sun_good:
        return ("Sunshine is helping, but there is little wind, so gas usually still "
                "plays a big part.", None)
    return ("A mixed picture: wind and solar are both contributing, with other "
            "sources topping up the rest.", None)


def build_summary(wind, solar):
    """Work out the weather and its meaning from the latest wind and solar series."""
    # Wind: compare with the recent peak (a rough stand-in for "full output")
    wind_ref = max(float(wind.max()), MIN_WIND_REFERENCE_MW)
    wind_now = float(wind.iloc[-1])
    wind_share = wind_now / wind_ref
    w_label, w_word, w_text = band(wind_share, WIND_BANDS)
    w_text += TREND_TEXT[trend(wind, wind_ref)]

    # Solar: compare with the best recent day at the same time of day (a rough clear-sky guide)
    solar_now = float(solar.iloc[-1])
    latest = solar.index[-1]
    same_time = solar[(solar.index.hour == latest.hour) & (solar.index.minute == latest.minute)]
    same_time = same_time.iloc[:-1]  # earlier days only
    baseline = float(same_time.max()) if len(same_time) else float("nan")

    s_ratio = None
    if np.isnan(baseline):
        s_label, s_word, s_text = "unknown", None, "Not enough history yet to judge cloud cover."
    elif baseline < NIGHT_BASELINE_MW:
        s_label, s_word = "night", None
        s_text = "The sun is down or very low, so solar output says nothing about cloud right now."
    else:
        s_ratio = solar_now / baseline
        s_label, s_word, s_text = band(s_ratio, SOLAR_BANDS)

    meaning, tip = electricity_meaning(w_label, s_label)

    def local(ts):
        return ts.tz_convert(LONDON).strftime("%H:%M on %d %b")

    return {
        "headline": f"{w_word} and {s_word}" if s_word else w_word,
        "wind_text": w_text,
        "solar_text": s_text,
        "meaning": meaning,
        "tip": tip,
        "wind_gw": round(wind_now / 1000, 1),
        "wind_percent_of_recent_peak": round(wind_share * 100),
        "solar_gw": round(solar_now / 1000, 1),
        "solar_percent_of_clear": None if s_ratio is None else round(s_ratio * 100),
        "wind_as_of": local(wind.index[-1]),
        "solar_as_of": local(latest),
    }


# ---- Checking ourselves against independent weather data ----------------------
def hourly(series):
    """Half-hourly -> hourly mean, labelled by the END of the hour.

    This matches how the half-hours are labelled and how Open-Meteo labels its
    sunlight values (the average of the hour before). Hours with only one
    half-hour of data are dropped.
    """
    rolled = series.resample("1h", closed="right", label="right")
    return rolled.mean()[rolled.count() == 2]


def expected_from_ranks(official, our_values, more_means_higher_output):
    """Translate an official reading into the output we would expect, by matching ranks.

    The windiest official hour is paired with our highest output, the calmest with our
    lowest, and so on (for cloud it runs the other way round). This avoids guessing a
    turbine power curve or a cloud-to-output rule. It tests whether the ORDER of conditions
    agrees, which is what the labels on the main page rely on.
    """
    if len(official) == 0:
        return pd.Series(dtype=float)
    ranks = official.rank(ascending=more_means_higher_output)  # ties share an average rank
    position = np.rint(ranks.values / len(official) * len(our_values)).astype(int) - 1
    matched = np.sort(our_values)[np.clip(position, 0, len(our_values) - 1)]
    return pd.Series(matched, index=official.index)


def compare_labels(ours, official, order):
    """'match' if the labels agree, 'close' if one step apart, otherwise 'differs'."""
    gap = abs(order.index(ours) - order.index(official))
    return "match" if gap == 0 else "close" if gap == 1 else "differs"


def verdict(correlation):
    """Plain-English reading of a rank correlation. Placeholder thresholds."""
    if correlation >= 0.85:
        return "very closely"
    if correlation >= 0.65:
        return "fairly closely"
    if correlation >= 0.40:
        return "only loosely"
    return "poorly"


def agreement(table, output_col, official_col, unit, intro, call):
    """Summarise how well our readings agree with the independent data, or None if too little data."""
    if len(table) < MIN_HOURS_FOR_STATS:
        return None
    # Spearman rank correlation = Pearson correlation of the ranks. It copes with output not
    # rising in a straight line with wind speed. 1 means they rise and fall together perfectly.
    correlation = table[output_col].rank().corr(table[official_col].rank())
    if pd.isna(correlation):
        return None
    exact = int(round(100 * float((table["status"] == "match").mean())))
    close = int(round(100 * float(table["status"].isin(["match", "close"]).mean())))
    reading = verdict(correlation)
    return {
        "hours": int(len(table)),
        "correlation": round(float(correlation), 2),
        "verdict": reading,
        "exact_percent": exact,
        "within_one_percent": close,
        "text": (f"Over the last {len(table)} {unit}, {intro} {reading}. Our {call} was "
                 f"exactly right {exact}% of the time and within one step {close}% of the time."),
    }


def recent_cell(table, stamp):
    if stamp not in table.index:
        return None
    row = table.loc[stamp]
    return {"ours": row["ours"], "official": row["theirs"], "status": row["status"]}


def build_accuracy(wind, solar, reference):
    """Compare what we inferred from power output with independent weather data, hour by hour."""
    wind_hourly, solar_hourly = hourly(wind), hourly(solar)

    # Wind: our label comes from output as a share of the recent peak (as on the main page)
    wind_ref = max(float(wind.max()), MIN_WIND_REFERENCE_MW)
    w = pd.DataFrame({"output": wind_hourly, "official": reference["wind_speed"]}).dropna()
    share = w["output"] / wind_ref
    w["ours"] = share.map(lambda x: band(x, WIND_BANDS)[0])
    expected_share = expected_from_ranks(w["official"], share.values, True)
    w["theirs"] = expected_share.map(lambda x: band(x, WIND_BANDS)[0])
    w["status"] = [compare_labels(a, b, WIND_ORDER) for a, b in zip(w["ours"], w["theirs"])]

    # Solar: our label comes from output vs the best recent day at the same hour, daylight only
    best = solar_hourly.groupby(solar_hourly.index.hour).transform("max")
    s = pd.DataFrame({"output": solar_hourly, "best": best,
                      "cloud": reference["cloud"], "sunlight": reference["radiation"]}).dropna()
    s = s[s["best"] >= NIGHT_BASELINE_MW].copy()
    ratio = s["output"] / s["best"]
    s["ours"] = ratio.map(lambda x: band(x, SOLAR_BANDS)[0])
    expected_ratio = expected_from_ranks(s["cloud"], ratio.values, False)
    s["theirs"] = expected_ratio.map(lambda x: band(x, SOLAR_BANDS)[0])
    s["status"] = [compare_labels(a, b, SOLAR_ORDER) for a, b in zip(s["ours"], s["theirs"])]

    wind_stats = agreement(w, "output", "official", "hours",
                           "wind output rose and fell with official wind speeds", "wind call")
    solar_stats = agreement(s, "output", "sunlight", "daylight hours",
                            "solar output followed official sunlight levels", "cloud call")

    recent = [
        {"time": stamp.tz_convert(LONDON).strftime("%H:%M"),
         "wind": recent_cell(w, stamp), "sun": recent_cell(s, stamp)}
        for stamp in w.index[-RECENT_ROWS:]
    ]

    parts = []
    if wind_stats:
        parts.append(f"{wind_stats['within_one_percent']}% of wind calls")
    if solar_stats:
        parts.append(f"{solar_stats['within_one_percent']}% of cloud calls")
    trust_line = ("Recent accuracy: " + " and ".join(parts)
                  + " were within one step of independent weather data.") if parts else None

    stats = [x for x in (wind_stats, solar_stats) if x]
    caution = any(x["correlation"] < 0.65 or x["within_one_percent"] < 70 for x in stats)
    return {"wind": wind_stats, "solar": solar_stats, "recent": recent,
            "trust_line": trust_line, "caution": caution}


# ---- The website ------------------------------------------------------------
app = Flask(__name__)
_cache = {"time": 0.0, "data": None}


def refresh():
    wind, solar, reference = load_all()
    summary = build_summary(wind, solar)
    summary["accuracy"] = None
    if reference is not None:
        try:
            summary["accuracy"] = build_accuracy(wind, solar, reference)
        except Exception as error:  # a problem here should never take the page down
            log.warning("Accuracy check failed: %s", error)
    return summary


def get_summary():
    if _cache["data"] is None or time.time() - _cache["time"] > CACHE_SECONDS:
        try:
            _cache.update(time=time.time(), data=refresh())
        except Exception as error:
            if _cache["data"] is None:
                raise
            # Keep showing the older result and try again in about a minute
            app.logger.warning("Refresh failed, showing older data: %s", error)
            _cache["time"] = time.time() - CACHE_SECONDS + 60
    return _cache["data"]


PAGE = """<!doctype html>
<html lang="en-GB">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="600">
<title>Weather and electricity</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 640px; margin: 2rem auto;
         padding: 0 1rem; line-height: 1.5; color: #1b1f24; }
  h1 { font-size: 1.1rem; color: #555; font-weight: 600; margin: 0; }
  .headline { font-size: 2rem; font-weight: 700; margin: .2rem 0 .4rem; }
  .card { border: 1px solid #d8dde3; border-radius: 10px; padding: 1rem; margin: 1rem 0; }
  .card h2 { margin: 0 0 .3rem; font-size: 1rem; }
  .tip { background: #eef6ee; border-radius: 8px; padding: .6rem .8rem; }
  .warn { background: #fff4d6; border-radius: 8px; padding: .6rem .8rem; }
  .small { color: #666; font-size: .85rem; }
  .demo { background: #fff4d6; border-radius: 8px; padding: .5rem .8rem; }
  table { border-collapse: collapse; width: 100%; font-size: .85rem; margin: .5rem 0; }
  th, td { text-align: left; padding: .25rem .4rem; border-bottom: 1px solid #e6e9ed; }
  th { color: #555; font-weight: 600; }
</style>
</head>
<body>
{% if demo %}<p class="demo">Demo mode: this page is showing made-up data.</p>{% endif %}
<h1>What is the weather doing to our electricity?</h1>
<p class="headline">{{ s.headline }}</p>
{% if s.accuracy and s.accuracy.trust_line %}
<p class="small">{{ s.accuracy.trust_line }} <a href="#accuracy">See how we checked</a></p>
{% endif %}

<div class="card">
  <h2>Wind</h2>
  <p>{{ s.wind_text }}</p>
  <p class="small">{{ s.wind_gw }} GW now, {{ s.wind_percent_of_recent_peak }}% of the recent peak
     (as of {{ s.wind_as_of }})</p>
</div>

<div class="card">
  <h2>Sun and cloud</h2>
  <p>{{ s.solar_text }}</p>
  <p class="small">{{ s.solar_gw }} GW now{% if s.solar_percent_of_clear is not none %},
     {{ s.solar_percent_of_clear }}% of a clear day{% endif %}
     (as of {{ s.solar_as_of }})</p>
</div>

<div class="card">
  <h2>What this means for electricity</h2>
  <p>{{ s.meaning }}</p>
  {% if s.tip %}<p class="tip">{{ s.tip }}</p>{% endif %}
</div>

{% set a = s.accuracy %}
{% set marks = {'match': '✓ match', 'close': '~ close', 'differs': '✗ differs'} %}
<div class="card" id="accuracy">
  <h2>How far can you trust this?</h2>
  {% if not a %}
    <p>The accuracy check is not available right now because the independent weather data
       could not be loaded. Until it is back, treat the readings above with extra care.</p>
  {% else %}
    {% if a.wind %}<p><strong>Wind.</strong> {{ a.wind.text }}</p>{% endif %}
    {% if a.solar %}<p><strong>Sun and cloud.</strong> {{ a.solar.text }}</p>{% endif %}
    {% if not a.wind and not a.solar %}
      <p>There is not yet enough overlapping data to measure accuracy.</p>
    {% endif %}
    {% if a.caution %}
      <p class="warn">Some of these figures are weak, so treat the readings above with extra care.</p>
    {% endif %}
    {% if a.recent %}
      <p class="small">The last few hours, hour by hour. "Ours" is what the power output suggests;
         "official" is what the independent weather data implies, put into the same labels by
         ranking each hour against the rest of the week (so no turbine or cloud rules are
         assumed). Times are UK time (hour ending).</p>
      <table>
        <tr><th>Time</th><th>Wind: ours / official</th><th></th>
            <th>Sun: ours / official</th><th></th></tr>
        {% for row in a.recent %}
        <tr>
          <td>{{ row.time }}</td>
          {% for cell in (row.wind, row.sun) %}
            {% if cell %}<td>{{ cell.ours }} / {{ cell.official }}</td><td>{{ marks[cell.status] }}</td>
            {% else %}<td>&ndash;</td><td></td>{% endif %}
          {% endfor %}
        </tr>
        {% endfor %}
      </table>
    {% endif %}
    <p class="small">Why they sometimes disagree: farms can be switched off or turned down, thin cloud
       barely changes solar output, we compare national output with weather at a handful of spots,
       and the independent data is itself a model rather than a set of station readings.
       Correlation here means 1 is a perfect match in how they rise and fall, and 0 is no link.</p>
  {% endif %}
</div>

<p class="small">These are rough guides to the weather worked out from power output, not a
forecast. Wind figures cover large wind farms connected to the main grid, and solar is an
estimate. Contains BMRS data &copy; Elexon Limited. Solar estimates from Sheffield Solar's
PV_Live. Weather comparison data by Open-Meteo.com (CC BY 4.0), which blends national
weather-service models.</p>
</body>
</html>
"""


@app.route("/")
def home():
    try:
        return render_template_string(PAGE, s=get_summary(), demo=DEMO)
    except Exception as error:
        app.logger.error("Could not load data: %s", error)
        return "Sorry, the latest data could not be loaded. Please try again in a few minutes.", 503


@app.route("/api/summary")
def api_summary():
    try:
        return jsonify(get_summary())
    except Exception as error:
        app.logger.error("Could not load data: %s", error)
        return jsonify(error="data unavailable"), 503


if __name__ == "__main__":
    app.run()
