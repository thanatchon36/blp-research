"""ทดสอบคุณค่าเชิงพยากรณ์ของดัชนี BLP แบบ pseudo real-time (บทที่ 5)

ณ สิ้นไตรมาส t ใช้ได้เฉพาะ: ดัชนี BLP ของไตรมาส t และข้อมูลทางการถึงไตรมาส t-1
ประเมินแบบ expanding window นอกตัวอย่าง เทียบกับ AR benchmark
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

DEFAULT_MODELS: dict[str, list[str]] = {
    "M0_AR": ["y_lag1"],
    "M1_AR+BLP": ["y_lag1", "BLP_RAI"],
    "M2_AR+BLP+OUTLOOK": ["y_lag1", "BLP_RAI", "OUTLOOK_lag1"],
}


def make_dataset(ind: pd.DataFrame, hard: pd.DataFrame) -> pd.DataFrame:
    df = hard[["region", "quarter", "y"]].merge(ind, on=["region", "quarter"], how="inner")
    df = df.sort_values(["region", "quarter"]).reset_index(drop=True)
    grp = df.groupby("region")
    df["y_lag1"] = grp["y"].shift(1)
    df["OUTLOOK_lag1"] = grp["OUTLOOK"].shift(1)
    return df


def _ols_predict(train: pd.DataFrame, test: pd.DataFrame, x: list[str], fe: bool) -> float:
    def design(d: pd.DataFrame) -> np.ndarray:
        parts = [np.ones((len(d), 1)), d[x].to_numpy(float)]
        if fe:
            parts.append(pd.get_dummies(d["region"]).reindex(columns=regions[1:], fill_value=0).to_numpy(float))
        return np.hstack(parts)

    regions = sorted(train["region"].unique())
    beta, *_ = np.linalg.lstsq(design(train), train["y"].to_numpy(float), rcond=None)
    return float((design(test) @ beta)[0])


def expanding_window_oos(df: pd.DataFrame, models: dict[str, list[str]] | None = None,
                         start: str = "2018Q1", pooled: bool = True) -> pd.DataFrame:
    """pooled=True ใช้ panel ข้ามภาค + region fixed effects (เพิ่มจำนวนข้อมูล) / False = ประมาณรายภาค"""
    models = models or DEFAULT_MODELS
    needed = sorted({c for cols in models.values() for c in cols} | {"y"})
    data = df.dropna(subset=needed)
    rows = []
    for q in sorted(data.loc[data["quarter"] >= pd.Period(start, freq="Q"), "quarter"].unique()):
        for region in sorted(data["region"].unique()):
            test = data[(data["quarter"] == q) & (data["region"] == region)]
            if test.empty:
                continue
            train = data[data["quarter"] < q] if pooled else data[(data["quarter"] < q) & (data["region"] == region)]
            row = {"region": region, "quarter": q, "y": float(test["y"].iloc[0])}
            for name, cols in models.items():
                row[name] = _ols_predict(train, test, cols, fe=pooled)
            rows.append(row)
    return pd.DataFrame(rows)


def clark_west(y, f_bench, f_model, lags: int = 1) -> tuple[float, float]:
    """Clark & West (2007) สำหรับโมเดลซ้อนกัน: H1 = โมเดลใหญ่แม่นกว่า (ทดสอบด้านเดียว)"""
    y, f1, f2 = map(np.asarray, (y, f_bench, f_model))
    adj = (y - f1) ** 2 - ((y - f2) ** 2 - (f1 - f2) ** 2)
    res = sm.OLS(adj, np.ones(len(adj))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    t = float(res.tvalues[0])
    return t, float(1.0 - stats.norm.cdf(t))


def diebold_mariano(e1, e2, h: int = 1) -> tuple[float, float]:
    """Diebold & Mariano (1995) + Harvey-Leybourne-Newbold correction, loss = squared error, สองด้าน"""
    d = np.asarray(e1) ** 2 - np.asarray(e2) ** 2
    T = len(d)
    dc = d - d.mean()
    var = dc @ dc / T + 2 * sum((dc[k:] @ dc[:-k]) / T for k in range(1, h))
    dm = d.mean() / np.sqrt(var / T)
    stat = dm * np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    return float(stat), float(2 * (1 - stats.t.cdf(abs(stat), df=T - 1)))


def evaluate(preds: pd.DataFrame, benchmark: str = "M0_AR") -> pd.DataFrame:
    names = [c for c in preds.columns if c not in ("region", "quarter", "y")]
    rmse_b = np.sqrt(np.mean((preds["y"] - preds[benchmark]) ** 2))
    rows = []
    for name in names:
        e = preds["y"] - preds[name]
        row = {"model": name, "rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(np.abs(e)))}
        row["rel_rmse"] = row["rmse"] / rmse_b
        if name != benchmark:
            row["cw_t"], row["cw_p"] = clark_west(preds["y"], preds[benchmark], preds[name])
            row["dm_stat"], row["dm_p"] = diebold_mariano(preds["y"] - preds[benchmark], e)
        rows.append(row)
    return pd.DataFrame(rows)


def lead_lag_corr(df: pd.DataFrame, x: str, y: str = "y", lags=range(-2, 3)) -> pd.DataFrame:
    """corr(x_t, y_{t+k}) ราย k: k > 0 แปลว่า x นำ y"""
    out = []
    for k in lags:
        shifted = df.groupby("region")[y].shift(-k)
        out.append({"lead_k": k, "corr": float(df[x].corr(shifted))})
    return pd.DataFrame(out)
