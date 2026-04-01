
import re
import ast
import json
from math import sqrt

import numpy as np
import pandas as pd
import statsmodels.api as sm

from scipy.stats import chi2_contingency, chi2
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LogisticRegression


# =========================================================
# CONFIG
# =========================================================

INPUT_CSV = "Final_Grouping2.csv"
TARGET_ORDER = ["Yes", "No", "Maybe"]
MULTINOM_ORDER = ["No", "Yes", "Maybe"]   # baseline = No for statsmodels
MIN_COUNT_PER_LEVEL = 10
MIN_SUPPORT_SENSITIVITY = 8
KEEP_TOP_CONCERNS = None
KEEP_TOP_MAIN_CONCERNS = None


# =========================================================
# HELPERS
# =========================================================

_slug = re.compile(r"\W+").sub


def slugify(s: str) -> str:
    return _slug("_", str(s)).strip("_").lower()


def to_list_safe(x):
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
        return [p.strip() for p in re.split(r"[;,]", s) if p.strip()]
    return []


def norm_item(s):
    s = re.sub(r"\s+", " ", str(s)).strip()
    return s.title()


def parse_list_cell(cell):
    if pd.isna(cell):
        return []
    s = str(cell).strip()
    if not s:
        return []
    try:
        val = ast.literal_eval(s)
        if isinstance(val, (list, tuple)):
            return [str(x).strip().lower() for x in val if str(x).strip()]
        if isinstance(val, str) and val.strip():
            return [val.strip().lower()]
    except Exception:
        pass
    if "," in s:
        return [t.strip().lower() for t in s.split(",") if t.strip()]
    return [s.lower()]


def cramers_v(chi2_stat, n, r, c):
    k = min(r, c)
    return sqrt(chi2_stat / (n * (k - 1))) if k > 1 and n > 0 else np.nan


def one_hot_presence(df_in: pd.DataFrame, list_col: str, prefix: str, keep_top=None) -> pd.DataFrame:
    ex = df_in[list_col].explode()
    if keep_top is not None:
        top = set(ex.value_counts().head(keep_top).index)
    else:
        top = None

    def row_to_series(L):
        if not isinstance(L, list):
            return pd.Series(dtype=int)
        items = L if top is None else [x for x in L if x in top]
        cols = {f"{prefix}__{slugify(x)}": 1 for x in items}
        return pd.Series(cols, dtype=int)

    X = df_in[list_col].apply(row_to_series).fillna(0).astype(int)
    return X


def one_hot_cat(series: pd.Series, prefix: str, min_count: int = 10, drop_first: bool = True) -> pd.DataFrame:
    s = series.astype("string")
    counts = s.value_counts(dropna=True)
    rare_levels = counts[counts < min_count].index
    s = s.where(~s.isin(rare_levels), other="Other")
    dummies = pd.get_dummies(s, prefix=prefix, dtype=int)
    if drop_first and dummies.shape[1] > 1:
        dummies = dummies.iloc[:, 1:]
    return dummies


def run_multiclass_chi_square(df_in: pd.DataFrame, col: str, target_col: str = "Target",
                              min_count_per_level: int = 10, targets=None):
    if targets is None:
        targets = ["Yes", "No", "Maybe"]

    d = df_in.dropna(subset=[target_col, col]).copy()
    d[target_col] = d[target_col].astype("string").str.title()

    counts = d[col].value_counts(dropna=False)
    rare_levels = counts[counts < min_count_per_level].index
    collapsed_col = f"{col}_collapsed"

    if len(rare_levels) > 0:
        d[collapsed_col] = d[col].where(~d[col].isin(rare_levels), other="Other")
    else:
        d[collapsed_col] = d[col]

    ct = pd.crosstab(d[collapsed_col], d[target_col]).reindex(columns=targets, fill_value=0)
    ct = ct.loc[ct.sum(axis=1) > 0]

    chi2_stat, p_value, dof, expected = chi2_contingency(ct.values, correction=False)
    n = ct.values.sum()
    v = cramers_v(chi2_stat, n, ct.shape[0], ct.shape[1])

    residuals = (ct.values - expected) / np.sqrt(expected)
    resid_df = pd.DataFrame(residuals, index=ct.index, columns=ct.columns)
    row_props = ct.div(ct.sum(axis=1), axis=0) * 100.0

    d["Target_yes"] = (d[target_col] == "Yes").astype(int)
    ct2 = pd.crosstab(d[collapsed_col], d["Target_yes"]).reindex(columns=[0, 1], fill_value=0)
    chi2_bi, p_bi, dof_bi, _ = chi2_contingency(ct2.values, correction=False)
    v_bi = cramers_v(chi2_bi, ct2.values.sum(), ct2.shape[0], ct2.shape[1])

    return {
        "column": col,
        "table": ct,
        "chi2": chi2_stat,
        "p_value": p_value,
        "dof": dof,
        "cramers_v": v,
        "residuals": resid_df,
        "row_props": row_props.round(1),
        "binary_yes_notyes": {
            "table": ct2,
            "chi2": chi2_bi,
            "p_value": p_bi,
            "dof": dof_bi,
            "cramers_v": v_bi,
        },
    }


def concern_association_table(df_in: pd.DataFrame, list_col: str, target_col: str = "Target"):
    data = df_in.dropna(subset=[target_col]).copy()
    targets = ["Yes", "No", "Maybe"]

    all_items = sorted({c for L in data[list_col] for c in L})
    results = []

    for item in all_items:
        present_mask = data[list_col].apply(lambda L: item in L)
        present_counts = data.loc[present_mask, target_col].value_counts().reindex(targets, fill_value=0)
        absent_counts = data.loc[~present_mask, target_col].value_counts().reindex(targets, fill_value=0)

        table = np.vstack([present_counts.values, absent_counts.values])

        if table.sum() == 0 or table[0].sum() == 0 or table[1].sum() == 0:
            continue

        chi2_stat, p_value, dof, _ = chi2_contingency(table, correction=False)
        v = cramers_v(chi2_stat, table.sum(), table.shape[0], table.shape[1])

        results.append({
            "concern": item,
            "present_counts": dict(zip(targets, present_counts.values)),
            "absent_counts": dict(zip(targets, absent_counts.values)),
            "chi2": chi2_stat,
            "df": dof,
            "p_value": p_value,
            "cramers_v": v,
            "support_present": int(table[0].sum()),
            "support_absent": int(table[1].sum()),
            "total": int(table.sum())
        })

    assoc_df = pd.DataFrame(results).sort_values("p_value", ascending=True).reset_index(drop=True)
    if not assoc_df.empty:
        reject, p_adj, _, _ = multipletests(assoc_df["p_value"], method="fdr_bh", alpha=0.05)
        assoc_df["p_adj_bh"] = p_adj
        assoc_df["reject_fdr_bh@0.05"] = reject
    return assoc_df


def build_model_matrix(df_in: pd.DataFrame):
    X_main = one_hot_presence(df_in, "main_concerns_list", prefix="MAIN", keep_top=KEEP_TOP_MAIN_CONCERNS)
    X_layer = one_hot_cat(df_in["layer"], "LAYER", min_count=10, drop_first=True) if "layer" in df_in.columns else pd.DataFrame(index=df_in.index)
    X_dtype = one_hot_cat(df_in["decision_type"], "DTYPE", min_count=10, drop_first=True) if "decision_type" in df_in.columns else pd.DataFrame(index=df_in.index)
    X_scope = one_hot_cat(df_in["scope"], "SCOPE", min_count=10, drop_first=True) if "scope" in df_in.columns else pd.DataFrame(index=df_in.index)

    X = pd.concat([X_main, X_layer, X_dtype, X_scope], axis=1).fillna(0)
    X = sm.add_constant(X, has_constant="add")
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    return X


# =========================================================
# LOAD + TARGET
# =========================================================

df = pd.read_csv(INPUT_CSV, low_memory=False)

df["Target"] = (
    df["comparison_result"]
    .astype("string")
    .str.extract(r"^\s*(Yes|No|Maybe)\b", flags=re.IGNORECASE)[0]
    .str.title()
)

# Parse concerns
df["concerns_list"] = df["concerns"].apply(to_list_safe).apply(
    lambda L: [norm_item(x) for x in L if str(x).strip()]
)

if "main_concerns" in df.columns:
    df["main_concerns_list"] = df["main_concerns"].apply(to_list_safe).apply(
        lambda L: [norm_item(x) for x in L if str(x).strip()]
    )
else:
    df["main_concerns_list"] = [[] for _ in range(len(df))]

print("Loaded rows:", len(df))
print("\nTarget distribution:")
print(df["Target"].value_counts(dropna=False))


# =========================================================
# 1) CHI-SQUARE FOR concerns
# =========================================================

assoc_df_concerns = concern_association_table(df, "concerns_list", target_col="Target")
print("\nTop concern associations:")
print(
    assoc_df_concerns.sort_values("p_adj_bh" if "p_adj_bh" in assoc_df_concerns.columns else "p_value")
    [["concern", "p_value", "p_adj_bh", "reject_fdr_bh@0.05", "cramers_v", "support_present"]]
    .head(15)
    .to_string(index=False)
    if not assoc_df_concerns.empty else "No concern associations computed."
)

assoc_df_main_concerns = concern_association_table(df, "main_concerns_list", target_col="Target")
print("\nTop main_concern associations:")
print(
    assoc_df_main_concerns.sort_values("p_adj_bh" if "p_adj_bh" in assoc_df_main_concerns.columns else "p_value")
    [["concern", "p_value", "p_adj_bh", "reject_fdr_bh@0.05", "cramers_v", "support_present"]]
    .head(15)
    .to_string(index=False)
    if not assoc_df_main_concerns.empty else "No main_concern associations computed."
)


# =========================================================
# 2) CHI-SQUARE FOR SINGLE-LABEL CATEGORICALS
# =========================================================

categorical_results = {}
for col in ["layer", "family_kind8", "decision_type", "scope", "ambiguity_level"]:
    if col in df.columns:
        res = run_multiclass_chi_square(
            df,
            col=col,
            target_col="Target",
            min_count_per_level=MIN_COUNT_PER_LEVEL,
            targets=TARGET_ORDER,
        )
        categorical_results[col] = res
        print(f"\n=== {col} ===")
        print(f"Chi-square: {res['chi2']:.4f} (df={res['dof']}), p-value={res['p_value']:.6f}, Cramér's V={res['cramers_v']:.3f}")
        print("Standardized residuals:")
        print(res["residuals"].round(2).to_string())
        print("\nRow-wise %:")
        print(res["row_props"].to_string())
        print(f"\n(Yes vs Not-Yes) Chi-square: {res['binary_yes_notyes']['chi2']:.4f} "
              f"(df={res['binary_yes_notyes']['dof']}), "
              f"p-value={res['binary_yes_notyes']['p_value']:.6f}, "
              f"Cramér's V={res['binary_yes_notyes']['cramers_v']:.3f}")


# =========================================================
# 3) MULTINOMIAL LOGISTIC REGRESSION (statsmodels)
# =========================================================

X = build_model_matrix(df)
y = df["Target"].astype("category")
y = y.cat.reorder_categories(MULTINOM_ORDER, ordered=False)

mask = y.notna() & X.notna().all(axis=1)
X_m = X.loc[mask].copy()
y_m = y[mask].astype("category")

mn = sm.MNLogit(y_m, X_m)
mn_res = mn.fit(method="newton", maxiter=200, disp=False)

null_X = sm.add_constant(pd.DataFrame(index=X_m.index), has_constant="add")
null_model = sm.MNLogit(y_m, null_X).fit(disp=False)

LR = 2 * (mn_res.llf - null_model.llf)
df_diff = mn_res.df_model - null_model.df_model
p_lr = 1 - chi2.cdf(LR, df_diff)

print("\n=== Multinomial logistic regression (statsmodels) ===")
print(mn_res.summary())
print("\nMcFadden R^2:", mn_res.prsquared)
print(f"LR test (full vs null): chi2={LR:.3f}, df={df_diff}, p={p_lr:.6f}")

# FDR-adjusted coefficient tables
pvals = mn_res.pvalues
params = mn_res.params
cats = list(y_m.cat.categories)
code_to_label = {code: cats[code] for code in pvals.columns}

tables = {}
for k in pvals.columns:
    label = code_to_label.get(k, str(k))
    reject, qvals, _, _ = multipletests(pvals[k], method="fdr_bh", alpha=0.05)
    out = pd.DataFrame({
        "coef": params[k],
        "OR": np.exp(params[k]),
        "p": pvals[k],
        "q_bh": qvals,
        "reject@0.05": reject
    }).sort_values("q_bh")
    tables[label] = out

print("\nTop multinomial coefficients by class contrast:")
for label, table in tables.items():
    print(f"\n--- {label} vs baseline ({cats[0]}) ---")
    print(table.head(15).to_string())


# =========================================================
# 4) REGULARIZED MULTINOMIAL LOGISTIC REGRESSION (sklearn)
# =========================================================

d = df.dropna(subset=["Target"]).copy()

class_order = ["No", "Maybe", "Yes"]
y_reg = pd.Categorical(d["Target"], categories=class_order, ordered=True)
y_codes = y_reg.codes

cat_cols = ["layer", "family_kind8", "decision_type", "scope", "lifecycle_stage", "ambiguity_level"]
cat_cols = [c for c in cat_cols if c in d.columns]

sig_tags = ["modifiability", "usability", "time_to_market", "cost", "interoperability"]

X_tags = pd.DataFrame(index=d.index)
if "main_concerns" in d.columns:
    tag_lists = d["main_concerns"].apply(parse_list_cell)
    for t in sig_tags:
        X_tags[f"mc__{t}"] = tag_lists.apply(lambda lst: int(t in set(lst)))

X_cat = pd.get_dummies(d[cat_cols], drop_first=True, dtype=int) if cat_cols else pd.DataFrame(index=d.index)
X_reg = pd.concat([X_cat, X_tags], axis=1)

constant_cols = [c for c in X_reg.columns if X_reg[c].nunique(dropna=True) <= 1]
if constant_cols:
    X_reg = X_reg.drop(columns=constant_cols)

keep = (~X_reg.isna().any(axis=1)) & (y_codes >= 0)
X_reg = X_reg.loc[keep].copy()
y_fit = y_codes[keep]

clf = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    penalty="l2",
    C=1.0,
    max_iter=200
)
clf.fit(X_reg, y_fit)

print("\n=== Regularized multinomial logistic regression (sklearn) ===")
print("Rows used:", X_reg.shape[0])
print("Predictors:", X_reg.shape[1])

classes = list(clf.classes_)
baseline_label = "Yes" if "Yes" in classes else classes[-1]
baseline_idx = classes.index(baseline_label)

coef_full = pd.DataFrame(clf.coef_, columns=X_reg.columns, index=classes)
coef_vs_base = coef_full.subtract(coef_full.iloc[baseline_idx], axis=1)
coef_vs_base = coef_vs_base.drop(index=baseline_label)

rrr = np.exp(coef_vs_base)

rrr_long = (
    rrr.reset_index()
       .melt(id_vars="index", var_name="predictor", value_name="RRR")
       .rename(columns={"index": "contrast"})
       .sort_values(["contrast", "RRR"], ascending=[True, False])
       .reset_index(drop=True)
)

coef_long = (
    coef_vs_base.reset_index()
                .melt(id_vars="index", var_name="predictor", value_name="coef_vs_baseline")
                .rename(columns={"index": "contrast"})
)

rrr_out = rrr_long.merge(coef_long, on=["contrast", "predictor"])
rrr_out["RRR"] = rrr_out["RRR"].map(lambda x: float(f"{x:.3f}"))
rrr_out["coef_vs_baseline"] = rrr_out["coef_vs_baseline"].map(lambda x: float(f"{x:.4f}"))

print("\nTop RRR-style contrasts from regularized multinomial model:")
print(rrr_out.head(30).to_string(index=False))
print(f"\nBaseline class for contrast table: '{baseline_label}'")


# =========================================================
# 5) BINARY ROBUSTNESS LOGIT + MARGINAL EFFECTS
# =========================================================

y_bin = (df.loc[X_reg.index, "Target"] == "Yes").astype(int)
X_bin = sm.add_constant(X_reg, has_constant="add")

logit = sm.Logit(y_bin, X_bin).fit(disp=False)

print("\n=== Binary robustness logit: Yes vs others ===")
print(logit.summary())

mfx = logit.get_margeff(at="overall", method="dydx").summary_frame()
mc_idx = [c for c in mfx.index if c.startswith("mc__")]
if mc_idx:
    print("\nTop marginal effects for main_concern predictors:")
    print(mfx.loc[mc_idx].sort_values("dy/dx", ascending=False).head(10).to_string())


# =========================================================
# 6) SENSITIVITY MULTINOMIAL AFTER DROPPING SPARSE FEATURES
# =========================================================

X_sens = X_reg.copy()
y_str_full = df.loc[X_sens.index, "Target"].astype("string").str.title()

drop_cols = []
for col in X_sens.columns:
    supp = int((X_sens[col] > 0).sum())
    if supp < MIN_SUPPORT_SENSITIVITY:
        drop_cols.append(col)
        continue

    sub = y_str_full[X_sens[col] == 1]
    vc = sub.value_counts()
    if not all(k in vc for k in ["No", "Yes", "Maybe"]):
        drop_cols.append(col)

X_sens = X_sens.drop(columns=drop_cols, errors="ignore").copy()
X_sens = X_sens.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
X_sens = sm.add_constant(X_sens, has_constant="add")

y_sub = df.loc[X_sens.index, "Target"].astype("string").str.title()
y_cat = pd.Categorical(y_sub, categories=["No", "Yes", "Maybe"], ordered=False)
y_codes_sens = pd.Series(y_cat.codes, index=X_sens.index)

mn_sens = sm.MNLogit(y_codes_sens, X_sens).fit(method="newton", maxiter=500, disp=False)
null_sens = sm.MNLogit(y_codes_sens, sm.add_constant(pd.DataFrame(index=X_sens.index), has_constant="add")).fit(disp=False)

LR_sens = 2 * (mn_sens.llf - null_sens.llf)
df_diff_sens = mn_sens.df_model - null_sens.df_model
p_lr_sens = 1 - chi2.cdf(LR_sens, df_diff_sens)

print("\n=== Sensitivity multinomial after sparse-feature filtering ===")
print(f"Dropped {len(drop_cols)} columns (support < {MIN_SUPPORT_SENSITIVITY} or separation risk).")
print(mn_sens.summary())
print("\nMcFadden R^2 (sensitivity):", mn_sens.prsquared)
print(f"LR test (sensitivity, full vs null): chi2={LR_sens:.3f}, df={df_diff_sens}, p={p_lr_sens:.6f}")