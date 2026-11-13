# SLR-GAT — Delay Forecaster

Operational front end for *Graph Neural Networks for Delay Propagation in a
Low-Density Railway Corridor* (Sri Lanka Main Line UP, Colombo Fort →
Rambukkana).

FastAPI backend serving the trained GAT model, single-page frontend with no
build step and no CDN dependency — it runs offline, which matters for a viva
on unreliable wifi.

---

## Two modes

**Manual** — the operator picks a service, types the first three observed
delays, and gets a forecast for the rest of the route with an uncertainty band.
This is the seed-then-forecast framing the whole thesis is built on.

**Replay** — pick a past date and service, seed the three real observed delays,
forecast the rest, and compare against what actually happened. Reports MAE and
skill against persistence (carry the last observed delay forward).

*Intervention was deliberately not built.* The injection experiment measured
zero lateral propagation — 0 of 4 other services moved by more than 0.01 min
when 25 minutes was injected into train 1168, and only 7 of 90 other-train
stops were connected by a headway edge at all. There is no cross-service
cascade to display on this corridor.

---

## Running it

```bash
pip install -r requirements.txt

# put the real model in place
#   copy MyDrive/SLRGAT_LDC_v1/09_deploy/*  ->  bundle/
#   then delete bundle/SYNTHETIC.txt

cd backend
uvicorn main:app --reload --port 8000
```

Open <http://127.0.0.1:8000>.

The bundle path can be overridden with `SLRGAT_BUNDLE=/path/to/09_deploy`.

### Running from Colab

```python
!pip -q install fastapi "uvicorn[standard]" pyngrok
!cd /content/slrgat-app/backend && nohup uvicorn main:app --port 8000 &
from pyngrok import ngrok; print(ngrok.connect(8000))
```

---

## The bundle

Produced by **NB08 Cell 4**. The app refuses to start without all six files
and says which are missing.

| file | contents |
|---|---|
| `config.json` | architecture, TCFG, FEATURES, SEED_LEN, normalisation stats |
| `scaler.json` | per-fold standardisation (mu / sd per column) |
| `model.pt` | weights — loaded with **strict** state-dict matching |
| `band.csv` | measured per-step MAE from the horizon study |
| `panel.parquet` | timetables and node features |
| `edges.parquet` | running (etype 0) and headway (etype 1) edges |

A bundle carrying `"synthetic": true` renders a warning banner in the UI. The
one shipped here is synthetic — random weights, meaningless numbers — so the
app can be run before the real model is in place.

---

## API

| endpoint | purpose |
|---|---|
| `GET /api/info` | deployed model metadata, or the load error |
| `GET /api/dates` | dates present in the bundle |
| `GET /api/services?date=YYYY-MM-DD` | services running that day |
| `POST /api/forecast` | `{train_id, date, seeds[], others_on_time}` → Manual |
| `POST /api/replay` | `{train_id, date}` → forecast, actual, MAE, skill |

```bash
curl -X POST localhost:8000/api/forecast -H 'Content-Type: application/json' \
  -d '{"train_id":"1168","date":"2026-07-26","seeds":[5,9,14]}'
```

---

## Decisions taken from the measurements, not from taste

Each of these would misrepresent the research if changed.

**No trust cutoff.** The forecast runs to Rambukkana. Across all seven folds
the model beats persistence at every horizon on average (skill 0.133 at 1–5
stops rising to 0.450 at 26–30), so truncating it would discard a measured
advantage. The widening band carries the honesty instead.

**The band is measured, not invented.** Per-step mean absolute error from 2,967
historical forecasts, smoothed for display and made non-decreasing. It is a
*mean* absolute error, not a 95% interval — roughly half of actual values fall
outside it, and the UI footer says so.

**Manual mode assumes other services are on time.** The operator knows their
own train's delays, not every other service's; seeding others from history
would use information unavailable at forecast time. Because lateral propagation
measured zero, this costs nothing. `others_on_time: false` exists to
demonstrate exactly that.

**The rollout is identical to training and evaluation.** Same causal
seed-then-forecast procedure, same headway rewiring, same delta
parameterisation. That is what makes the app's numbers comparable to the
thesis tables.

---

## A limitation the UI surfaces honestly

The model's per-stop increment is **near independent of the train's current
delay**: a punctual service and one 14 minutes late are both forecast to gain
about 0.55 min per stop. It encodes a *delay will accumulate* prior — correct
on average, because 71.3% of services depart late and delay grows along the
route, and wrong on quiet runs.

Measured across 23 Tier A runs:

| run's actual mean delay | runs | model MAE | persistence MAE | model wins |
|---|---|---|---|---|
| < 2 min (punctual) | 5 | 2.477 | **0.850** | 1/5 |
| 2–5 min | 2 | 1.949 | **1.000** | 0/2 |
| 5–10 min | 8 | **3.225** | 4.669 | 6/8 |
| > 10 min (delayed) | 8 | **7.069** | 8.632 | 6/8 |

When Replay shows negative skill the UI explains why rather than hiding it.
This cannot be turned into an operating rule: the seed does not predict which
regime a run belongs to — two runs of service 1169 with identical zero seeds
gave skill −39.3 and +0.47.

---

## Layout

```
backend/
  model_def.py   GNN + Head, ported verbatim from NB06 Cell 3
  engine.py      bundle loading, graph build, causal rollout
  main.py        FastAPI
frontend/
  index.html     single page
  app.js         inline SVG chart, no dependencies
bundle/          ← replace with 09_deploy/
```

`model_def.py` must not be tidied — any change to layer names or ordering
breaks strict weight loading.
