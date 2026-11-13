"""Forecast engine: loads the deployment bundle and runs seed-then-forecast.

The rollout here is the same causal procedure used in training and evaluation
(NB06 Cell 7 / NB08 Cell 2). Keeping them identical is what makes the app's
numbers comparable to the thesis.

Bundle layout (produced by NB08 Cell 4):
    config.json     architecture, TCFG, FEATURES, SEED_LEN, stats, ...
    scaler.json     per-fold standardisation (mu / sd per column)
    model.pt        weights, strict state-dict match with model_def.build_model
    band.csv        measured per-step MAE from the horizon study
    panel.parquet   timetables and node features
    edges.parquet   running (etype 0) and headway (etype 1) edges
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model_def import build_model


class Engine:
    def __init__(self, bundle: str | Path):
        b = Path(bundle)
        missing = [f for f in ("config.json", "scaler.json", "model.pt",
                               "band.csv", "panel.parquet", "edges.parquet")
                   if not (b / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"bundle at {b} is missing: {', '.join(missing)}. "
                "Run NB08 Cell 4 and copy 09_deploy/ here.")

        self.cfg = json.load(open(b / "config.json"))
        sc = json.load(open(b / "scaler.json"))
        self.sc_cols = sc["cols"]
        self.sc_mu = pd.Series(sc["mu"])
        self.sc_sd = pd.Series(sc["sd"])

        self.FEATURES = self.cfg["features"]
        self.LAG1 = self.cfg["lag1_col"]
        self.LAG2 = self.cfg["lag2_col"]
        self.SEED_LEN = self.cfg["seed_len"]
        self.REWIRE = self.cfg["rewire"]
        self.MAX_DELAY = self.cfg["max_delay"]
        self.MAX_DELTA = self.cfg["max_delta"]
        self.TARGET_MODE = self.cfg["target_mode"]
        self.ST = self.cfg["stats"]

        self.model = build_model(self.cfg)
        self.model.load_state_dict(torch.load(b / "model.pt", map_location="cpu"))
        self.model.eval()

        band = pd.read_csv(b / "band.csv", index_col=0)["mae"]
        band = band.reindex(range(1, int(band.index.max()) + 1)).interpolate()
        # display smoothing only: the raw per-step MAE is jagged because later
        # steps have few observations, and a band that narrows with distance
        # reads as a bug. The underlying measurement is unchanged.
        self.band = band.rolling(3, min_periods=1, center=True).mean().cummax()

        self.d = pd.read_parquet(b / "panel.parquet")
        self.d = (self.d.sort_values(["record_date", "train_id", "stop_sequence"])
                        .reset_index(drop=True))
        self.E = pd.read_parquet(b / "edges.parquet")

    # ---------------- preprocessing -------------------------------------
    def _scale(self, df):
        out = df.copy()
        out[self.sc_cols] = (out[self.sc_cols] - self.sc_mu) / self.sc_sd
        return out

    def _to_minutes(self, reg, lag1):
        if self.TARGET_MODE == "delta":
            p = (lag1 + reg.clamp(-self.MAX_DELTA, self.MAX_DELTA))
        else:
            p = torch.expm1(reg.clamp(0.0, float(np.log1p(self.MAX_DELAY))))
        return p.clamp(0.0, self.MAX_DELAY)

    def _day_tensors(self, df_day):
        df_day = df_day.reset_index(drop=True)
        local = {n: i for i, n in enumerate(df_day.node_id.values)}
        e = self.E[self.E.src.isin(local) & self.E.dst.isin(local)]
        sc = self._scale(df_day)
        N = len(df_day)

        prev1 = np.full(N, -1, dtype=np.int64)
        for _, gg in df_day.groupby(["record_date", "train_id"], sort=False):
            idx = gg.index.values
            prev1[idx[1:]] = idx[:-1]

        succ1 = np.full(N, -1, dtype=np.int64)
        ok = prev1 >= 0
        succ1[prev1[ok]] = np.nonzero(ok)[0]
        succ2 = np.where(succ1 >= 0, succ1[np.clip(succ1, 0, None)], -1)
        succ2 = np.where(succ1 >= 0, succ2, -1)

        kpos = df_day.k.to_numpy("int64")
        return dict(
            x=torch.tensor(sc[self.FEATURES].to_numpy("float32")),
            y=torch.tensor(df_day.delay_min.to_numpy("float32")),
            lag1=torch.tensor(df_day.lag1.to_numpy("float32")),
            succ1=torch.tensor(succ1),
            succ2=torch.tensor(succ2),
            kpos=torch.tensor(kpos),
            max_len=int(kpos.max()) + 1,
            ei=torch.tensor([[local[s] for s in e.src],
                             [local[t] for t in e.dst]], dtype=torch.long),
            ea=torch.tensor(e[["gap", "log_gap"]].to_numpy("float32")),
            et=torch.tensor(e.etype.to_numpy("int64")),
        )

    # ---------------- rollout -------------------------------------------
    @torch.no_grad()
    def _rollout(self, g):
        live, live1 = g["x"].clone(), g["lag1"].clone()
        ea0, hw, src = g["ea"], g["et"] == 1, g["ei"][0]
        seed = g["kpos"] < self.SEED_LEN
        out = torch.where(seed, g["y"], torch.full_like(g["y"], float("nan")))
        cur_delay = torch.where(seed, g["y"], torch.zeros_like(g["y"]))
        cur = torch.nonzero(g["kpos"] == self.SEED_LEN - 1, as_tuple=True)[0]

        for _ in range(g["max_len"] + 1):
            nxt = g["succ1"][cur]
            nxt = nxt[nxt >= 0]
            if len(nxt) == 0:
                break
            ea = ea0
            if self.REWIRE and hw.any():
                # a late train ahead shrinks the gap to the train behind
                gap = (ea0[hw, 0] - cur_delay[src[hw]]).clamp(min=0.0)
                ea = ea0.clone()
                ea[hw, 0] = gap
                ea[hw, 1] = torch.log1p(gap)
            reg, _ = self.model(live, g["ei"], ea, g["et"], g)
            chosen = self._to_minutes(reg, live1)[nxt]
            out[nxt] = chosen
            cur_delay = cur_delay.clone(); cur_delay[nxt] = chosen
            live, live1 = live.clone(), live1.clone()
            s1 = g["succ1"][nxt]; o1 = s1 >= 0
            if o1.any():
                live[s1[o1], self.LAG1] = (chosen[o1] - self.ST["mu1"]) / self.ST["sd1"]
                live1[s1[o1]] = chosen[o1]
            s2 = g["succ2"][nxt]; o2 = s2 >= 0
            if o2.any():
                live[s2[o2], self.LAG2] = (chosen[o2] - self.ST["mu2"]) / self.ST["sd2"]
            cur = nxt
        return out

    def _apply_seeds(self, dd, train_id, seeds, others_on_time):
        dd = dd.copy()
        if others_on_time:
            # the operator knows their own train's delays, not every other
            # service's. Seeding others from history would use information
            # unavailable at forecast time.
            dd.loc[dd.train_id != train_id, "delay_min"] = 0.0
        m = dd.train_id == train_id
        for kk, v in enumerate(list(seeds)[: self.SEED_LEN]):
            dd.loc[m & (dd.k == kk), "delay_min"] = float(v)
        dd["obs_delay"] = dd.delay_min * dd.is_seed
        for lag, col in ((1, "lag1"), (2, "lag2")):
            sh = (dd.sort_values(["train_id", "k"])
                    .groupby("train_id").delay_min.shift(lag))
            dd[col] = sh.reindex(dd.index).fillna(0.0).values
        return dd

    # ---------------- public API ----------------------------------------
    def dates(self):
        return [str(x)[:10] for x in sorted(self.d.record_date.unique())]

    def services(self, date):
        dd = self.d[self.d.record_date == pd.Timestamp(date)]
        if not len(dd):
            return []
        g = (dd.groupby("train_id")
               .agg(stops=("k", "size"), tier=("tier", "first"),
                    train_type=("train_type", "first"),
                    first_dep=("sched_departure_ts", "min"),
                    origin=("station", "first"),
                    terminus=("station", "last"))
               .reset_index())
        g["first_dep"] = g.first_dep.dt.strftime("%H:%M")
        return g.sort_values("first_dep").to_dict("records")

    def forecast(self, train_id, date, seeds=None, others_on_time=True):
        dd = self.d[self.d.record_date == pd.Timestamp(date)].reset_index(drop=True)
        if not len(dd):
            raise ValueError(f"no data for {date}")
        if train_id not in set(dd.train_id):
            raise ValueError(f"service {train_id} does not run on {date}")
        if seeds is not None:
            dd = self._apply_seeds(dd, train_id, seeds, others_on_time)

        g = self._day_tensors(dd)
        dd["pred"] = self._rollout(g).numpy()
        dd["steps"] = dd.k - self.SEED_LEN + 1

        r = dd[dd.train_id == train_id].sort_values("stop_sequence")
        stops = []
        for _, z in r.iterrows():
            fc = bool(z.steps >= 1) and not pd.isna(z.pred)
            band = float(self.band.get(int(z.steps), np.nan)) if fc else None
            stops.append(dict(
                stop=int(z.stop_sequence),
                station=str(z.station),
                departure=z.sched_departure_ts.strftime("%H:%M"),
                is_seed=bool(z.is_seed),
                forecast=round(float(z.pred), 1) if fc else None,
                band=round(band, 1) if band == band and band is not None else None,
                actual=None if pd.isna(z.delay_min) else round(float(z.delay_min), 1),
            ))
        return dict(train_id=train_id, date=str(date)[:10],
                    seed_len=self.SEED_LEN, stops=stops)

    def replay(self, train_id, date):
        """Seed the three real observed delays, forecast the rest, score it."""
        res = self.forecast(train_id, date, seeds=None, others_on_time=False)
        ev = [s for s in res["stops"]
              if s["forecast"] is not None and s["actual"] is not None]
        if not ev:
            res.update(mae=None, persistence_mae=None, skill=None, n=0)
            return res
        err = [abs(s["forecast"] - s["actual"]) for s in ev]
        seed_val = [s["actual"] for s in res["stops"] if s["is_seed"]][-1]
        herr = [abs(seed_val - s["actual"]) for s in ev]
        mae, mh = float(np.mean(err)), float(np.mean(herr))
        res.update(n=len(ev), mae=round(mae, 3), persistence_mae=round(mh, 3),
                   seed_value=seed_val,
                   skill=round(1 - mae / mh, 3) if mh > 0 else None)
        return res

    def info(self):
        c = self.cfg
        return dict(synthetic=bool(c.get("synthetic", False)),
            architecture=c["arch"], parameters=sum(
            p.numel() for p in self.model.parameters()),
            hidden=c["tcfg"]["HIDDEN"], heads=c["tcfg"]["HEADS"],
            layers=c["tcfg"]["LAYERS"], seed_len=c["seed_len"],
            trained_on_days=len(c.get("trained_on", [])),
            val_rollout_mae=round(c.get("val_mae", float("nan")), 3),
            rewire=c["rewire"], target_mode=c["target_mode"],
            n_dates=len(self.dates()),
            n_services=int(self.d.train_id.nunique()),
            n_stations=int(self.d.station.nunique()))
