"""สร้างดัชนีรายภาคจากคะแนนรายบันทึก (บทที่ 4.3)

raw      : diffusion index (net balance) = 100 x (สัดส่วนตอบเพิ่ม - สัดส่วนตอบลด)
weighted : ถ่วงน้ำหนักตามสัดส่วนสาขาใน GRP ของภาค
adj      : weighted two-way fixed effects (บริษัท + ไตรมาส) เพื่อแก้ composition bias
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import ACTIVITY_TOPICS


def attach_metadata(scored: pd.DataFrame, notes: pd.DataFrame) -> pd.DataFrame:
    meta = notes[["note_id", "firm_id", "region", "sector", "quarter"]]
    return scored.merge(meta, on="note_id", how="left").dropna(subset=["score"])


def diffusion_index(long: pd.DataFrame, by=("region", "quarter", "topic")) -> pd.DataFrame:
    g = long.assign(up=long["score"] > 0, down=long["score"] < 0).groupby(list(by))
    out = g.agg(n=("score", "size"), share_up=("up", "mean"), share_down=("down", "mean")).reset_index()
    out["di"] = 100.0 * (out["share_up"] - out["share_down"])
    return out


def sector_weighted_index(long: pd.DataFrame, weights: dict[str, dict[str, float]]) -> pd.DataFrame:
    di = diffusion_index(long, by=("region", "sector", "quarter", "topic"))
    di["w"] = [weights[r].get(s, 0.0) for r, s in zip(di["region"], di["sector"])]
    di["wdi"] = di["w"] * di["di"]
    out = di.groupby(["region", "quarter", "topic"]).agg(wdi=("wdi", "sum"), w=("w", "sum")).reset_index()
    out["di_w"] = out["wdi"] / out["w"]          # normalize ใหม่เมื่อบางสาขาไม่มีตัวอย่างในไตรมาสนั้น
    return out[["region", "quarter", "topic", "di_w"]]


def _two_way_fe(y, w, firm_codes, time_codes, n_firm, n_time, max_iter=2000, tol=1e-10):
    """Weighted least squares ของ y = alpha_firm + lambda_time ด้วย alternating projections"""
    alpha, lam = np.zeros(n_firm), np.zeros(n_time)
    w_time = np.bincount(time_codes, w, minlength=n_time)
    w_firm = np.bincount(firm_codes, w, minlength=n_firm)
    for _ in range(max_iter):
        lam_new = np.divide(np.bincount(time_codes, w * (y - alpha[firm_codes]), minlength=n_time), w_time,
                            out=np.full(n_time, np.nan), where=w_time > 0)
        resid = y - np.nan_to_num(lam_new)[time_codes]
        alpha_new = np.divide(np.bincount(firm_codes, w * resid, minlength=n_firm), w_firm,
                              out=np.zeros(n_firm), where=w_firm > 0)
        delta = max(np.nanmax(np.abs(lam_new - lam)), np.max(np.abs(alpha_new - alpha)))
        alpha, lam = alpha_new, lam_new
        if delta < tol:
            break
    return lam


def composition_adjusted_index(long: pd.DataFrame, weights: dict[str, dict[str, float]] | None = None) -> pd.DataFrame:
    """ดัชนีที่ตัดผลจากการเปลี่ยนตัวบริษัทออก: lambda_t จาก score_it = alpha_i + lambda_t (+ น้ำหนักสาขา)

    normalize ให้ค่าเฉลี่ยตลอดช่วงของ lambda เท่ากับค่าเฉลี่ยของดัชนีดิบ เพื่อให้อ่านเป็น net balance ได้
    """
    rows = []
    for (region, topic), g in long.groupby(["region", "topic"]):
        if weights is not None:
            n_st = g.groupby(["sector", "quarter"])["score"].transform("size").to_numpy(float)
            w = g["sector"].map(weights[region]).fillna(0.0).to_numpy(float) / n_st
        else:
            w = np.ones(len(g))
        firm_codes, firms = pd.factorize(g["firm_id"])
        time_codes, times = pd.factorize(g["quarter"])
        y = g["score"].to_numpy(float)
        lam = _two_way_fe(y, w, firm_codes, time_codes, len(firms), len(times))
        w_time = np.bincount(time_codes, w, minlength=len(times))
        raw_t = np.divide(np.bincount(time_codes, w * y, minlength=len(times)), w_time,
                          out=np.full(len(times), np.nan), where=w_time > 0)
        lam = lam - np.nanmean(lam) + np.nanmean(raw_t)
        rows += [{"region": region, "topic": topic, "quarter": q, "di_adj": 100.0 * v} for q, v in zip(times, lam)]
    return pd.DataFrame(rows)


def dispersion_index(long: pd.DataFrame, topic: str = "sales") -> pd.DataFrame:
    """ดัชนีความไม่แน่นอน = ส่วนเบี่ยงเบนมาตรฐานของคะแนนข้ามบริษัท (x100)"""
    sub = long[long["topic"] == topic]
    out = sub.groupby(["region", "quarter"])["score"].std().mul(100.0).rename("UNCERTAINTY").reset_index()
    return out


def build_indicators(long: pd.DataFrame, weights: dict[str, dict[str, float]]) -> pd.DataFrame:
    """ตารางกว้างราย ภาค x ไตรมาส พร้อมใช้ใน Study 2 และ AI assistant"""
    raw = diffusion_index(long).pivot_table(index=["region", "quarter"], columns="topic", values="di")
    wtd = sector_weighted_index(long, weights).pivot_table(index=["region", "quarter"], columns="topic", values="di_w")
    adj = composition_adjusted_index(long, weights).pivot_table(index=["region", "quarter"], columns="topic", values="di_adj")
    out = pd.concat([raw.add_suffix("_raw"), wtd.add_suffix("_w"), adj.add_suffix("_adj")], axis=1)
    out.columns.name = None
    act = list(ACTIVITY_TOPICS)
    out["BLP_RAI_raw"] = out[[f"{t}_raw" for t in act]].mean(axis=1)
    out["BLP_RAI_w"] = out[[f"{t}_w" for t in act]].mean(axis=1)
    out["BLP_RAI"] = out[[f"{t}_adj" for t in act]].mean(axis=1)
    out["OUTLOOK"], out["PRICE"], out["COST"] = out["outlook_adj"], out["price_adj"], out["cost_adj"]
    out = out.reset_index().merge(dispersion_index(long), on=["region", "quarter"], how="left")
    n = long.groupby(["region", "quarter"])["note_id"].nunique().rename("N_NOTES").reset_index()
    return out.merge(n, on=["region", "quarter"], how="left").sort_values(["region", "quarter"]).reset_index(drop=True)
