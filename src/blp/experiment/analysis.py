"""วิเคราะห์ผล Study 4 ตามแผนที่ลงทะเบียนล่วงหน้า (บทที่ 7.3)

H5 เวลา   : LMM บน log(นาที), H0: beta >= ln(0.8) -> ทดสอบด้านเดียวว่าเวลาลดลงเกิน 20%
H6 คุณภาพ : non-inferiority, H0: ผลต่างคะแนน <= -margin
H7 ความผิดพลาด: Poisson GLM + cluster-robust SE ระดับบุคคล, offset = log(จำนวนคำ/1000)
simulate_experiment() ใช้ซ้อมโค้ดวิเคราะห์กับข้อมูลจำลองก่อนเก็บข้อมูลจริง (dry run สำหรับ pre-registration)
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


def simulate_experiment(schedule: pd.DataFrame, true_reduction: float = 0.30, base_minutes: float = 180.0,
                        sd_person: float = 0.30, sd_task: float = 0.15, sd_resid: float = 0.30,
                        sd_ai_slope: float = 0.15, quality_effect: float = 0.0, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pids = schedule["participant_id"].unique()
    u = dict(zip(pids, rng.normal(0, sd_person, len(pids))))
    w = dict(zip(pids, rng.normal(0, sd_ai_slope, len(pids))))
    tasks = sorted(schedule["task"].unique())
    v = dict(zip(tasks, rng.normal(0, sd_task, len(tasks))))
    df = schedule.copy()
    beta = np.log(1 - true_reduction)
    junior = (df["experience"] == "junior").to_numpy(float)
    log_t = (np.log(base_minutes) + df["ai"] * (beta - 0.05 * junior) + df["participant_id"].map(w) * df["ai"]
             + df["task"].map(v) + df["participant_id"].map(u) - 0.03 * (df["session"] - 1)
             + 0.15 * junior + rng.normal(0, sd_resid, len(df)))
    df["minutes"] = np.exp(log_t).round(1)
    df["quality"] = np.clip(7.0 + quality_effect * df["ai"] - 0.4 * junior + rng.normal(0, 1.0, len(df)), 1, 10).round(1)
    df["words"] = rng.integers(900, 1400, len(df))
    df["errors"] = rng.poisson(1.2 * df["words"] / 1000 * np.where(df["ai"] == 1, 0.9, 1.0))
    return df


def _fit_lmm(formula: str, df: pd.DataFrame):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return smf.mixedlm(formula, df, groups=df["participant_id"]).fit(reml=True)


def analyze_time(df: pd.DataFrame, threshold: float = 0.20, alpha: float = 0.05) -> dict:
    d = df.assign(log_time=np.log(df["minutes"]))
    m = _fit_lmm("log_time ~ ai + C(session) + C(task)", d)
    b, se = float(m.params["ai"]), float(m.bse["ai"])
    z = (b - np.log(1 - threshold)) / se
    z_a = stats.norm.ppf(1 - alpha)
    return {"beta": b, "se": se, "reduction": 1 - np.exp(b),
            "reduction_ci95": (1 - np.exp(b + 1.96 * se), 1 - np.exp(b - 1.96 * se)),
            "reduction_lower_bound_one_sided": 1 - np.exp(b + z_a * se),
            "p_reduction_gt_threshold": float(stats.norm.cdf(z)), "threshold": threshold,
            "supports_H5": bool(stats.norm.cdf(z) < alpha)}


def carryover_test(df: pd.DataFrame) -> dict:
    """first-order carryover: session ก่อนหน้าใช้ AI หรือไม่ (ai x sequence ใช้ไม่ได้เพราะ collinear กับ session ใน ABAB/BABA)"""
    d = df.sort_values(["participant_id", "session"]).assign(log_time=lambda x: np.log(x["minutes"]))
    d["prev_ai"] = d.groupby("participant_id")["ai"].shift(1).fillna(0)
    m = _fit_lmm("log_time ~ ai + prev_ai + C(session) + C(task)", d)
    return {"term": "prev_ai", "coef": float(m.params["prev_ai"]), "p": float(m.pvalues["prev_ai"])}


def participant_level_test(df: pd.DataFrame, threshold: float = 0.20) -> dict:
    """robustness: Wilcoxon signed-rank บน d_i - ln(0.8) (ด้านเดียว)"""
    lt = df.assign(lt=np.log(df["minutes"])).groupby(["participant_id", "ai"])["lt"].mean().unstack()
    d = (lt[1] - lt[0]).to_numpy()
    res = stats.wilcoxon(d - np.log(1 - threshold), alternative="less")
    return {"n": int(len(d)), "median_reduction": float(1 - np.exp(np.median(d))), "p_wilcoxon": float(res.pvalue)}


def analyze_quality(df: pd.DataFrame, margin: float = 0.5, alpha: float = 0.05) -> dict:
    m = _fit_lmm("quality ~ ai + C(session) + C(task)", df)
    b, se = float(m.params["ai"]), float(m.bse["ai"])
    p = float(1 - stats.norm.cdf((b + margin) / se))
    return {"diff": b, "se": se, "ci95": (b - 1.96 * se, b + 1.96 * se), "margin": margin,
            "p_noninferiority": p, "supports_H6": bool(p < alpha)}


def analyze_errors(df: pd.DataFrame) -> dict:
    d = df.assign(offset=np.log(df["words"] / 1000))
    m = smf.glm("errors ~ ai + C(session) + C(task)", d, family=sm.families.Poisson(), offset=d["offset"]).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(d["participant_id"])[0]})
    b, se = float(m.params["ai"]), float(m.bse["ai"])
    return {"irr": float(np.exp(b)), "irr_ci95": (float(np.exp(b - 1.96 * se)), float(np.exp(b + 1.96 * se))),
            "p": float(m.pvalues["ai"])}


def heterogeneity(df: pd.DataFrame) -> dict:
    """ผลแยก junior / senior (สมมติฐานสำรวจ: junior ได้ประโยชน์มากกว่า)"""
    return {g: analyze_time(sub)["reduction"] for g, sub in df.groupby("experience")}
