import re
import numpy as np
import pandas as pd
import patsy as pt
import statsmodels.api as sm

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, f1_score, brier_score_loss


# =========================================================
# CONFIG
# =========================================================

INPUT_CSV = "Final_Grouping2.csv"
OUTPUT_CSV = "Final_Grouping2_with_counts.csv"


# =========================================================
# 1) EXTRACT NUMBER OF DRIVERS / OPTIONS
# =========================================================

# Accept common heading variants
DRIVER_HEADS = r"(Decision\s+Drivers?)"
OPTION_HEADS = r"((Options\s+Considered)|(Considered\s+Options)|Options|Alternatives?)"

# Anchor headings at start of line
HEAD_TMPL = r"(?im)^\s{{0,3}}#{{1,6}}\s*{title}[^\n]*\n(?P<body>.*?)(?=^\s{{0,3}}#{{1,6}}\s|\Z)"

DRIVER_RE = re.compile(HEAD_TMPL.format(title=DRIVER_HEADS), re.DOTALL | re.MULTILINE)
OPTION_RE = re.compile(HEAD_TMPL.format(title=OPTION_HEADS), re.DOTALL | re.MULTILINE)

# Top-level list items only
ITEM_RE = re.compile(r"(?m)^[ \t]{0,3}(?:[-*+]|(?:\d{1,3})[.)])(?:\s|\s*\[[ xX]\]\s)")


def count_drivers_and_options(text):
    """Extract counts from a markdown ADR string in df['topic']."""
    if not isinstance(text, str) or not text.strip():
        return pd.Series({"n_drivers": 0, "n_options": 0})

    d_match = DRIVER_RE.search(text)
    o_match = OPTION_RE.search(text)

    drivers_body = d_match.group("body") if d_match else ""
    options_body = o_match.group("body") if o_match else ""

    n_drivers = len(ITEM_RE.findall(drivers_body))
    n_options = len(ITEM_RE.findall(options_body))

    return pd.Series({"n_drivers": n_drivers, "n_options": n_options})


# =========================================================
# 2) LOAD DATA + CREATE TARGET
# =========================================================

df = pd.read_csv(INPUT_CSV, low_memory=False)

df[["n_drivers", "n_options"]] = df["topic"].apply(count_drivers_and_options)

# Primary: look only before the '//' separator
head = (
    df["comparison_result"]
      .astype("string")
      .str.split("//", n=1)
      .str[0]
)

df["Target"] = (
    head.str.extract(r"\b(Yes|No|Maybe)\b", flags=re.IGNORECASE)[0]
        .str.title()
)

# Fallback: if still missing, search the whole string
mask = df["Target"].isna()
if mask.any():
    df.loc[mask, "Target"] = (
        df.loc[mask, "comparison_result"]
          .astype("string")
          .str.extract(r"\b(Yes|No|Maybe)\b", flags=re.IGNORECASE)[0]
          .str.title()
    )

df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved dataset with counts to: {OUTPUT_CSV}")


# =========================================================
# 3) MULTINOMIAL MODEL WITH COUNTS + OPTIONAL CONTROLS
# =========================================================

d = df.copy()
d = d[d["Target"].isin(["No", "Maybe", "Yes"])].dropna(subset=["n_drivers", "n_options"])

d["n_drivers_c"] = d["n_drivers"]
d["n_options_c"] = d["n_options"]
d["Target"] = pd.Categorical(d["Target"], categories=["No", "Maybe", "Yes"])

controls = ["layer", "family_kind8", "decision_type", "scope", "lifecycle_stage", "ambiguity_level"]
controls = [c for c in controls if c in d.columns]
ctrl_terms = " + ".join([f"C({c})" for c in controls])

base_terms = "n_drivers_c + n_options_c"
formula = f"Target ~ {base_terms}" + (f" + {ctrl_terms}" if ctrl_terms else "")

y, X = pt.dmatrices(formula, data=d, return_type="dataframe")
mn = sm.MNLogit(y, X).fit(method="newton", maxiter=300, disp=False)

print("\n=== Multinomial model with counts ===")
print(mn.summary())

rrr = np.exp(mn.params).round(3)
print("\nRRR (exp(beta)):")
print(rrr)


# =========================================================
# 4) CENTER COUNTS FOR CV / ROBUSTNESS ANALYSIS
# =========================================================

cat_cols = [c for c in [
    "layer", "family_kind8", "decision_type", "scope",
    "lifecycle_stage", "ambiguity_level"
] if c in df.columns]

num_cols = [c for c in ["n_drivers", "n_options"] if c in df.columns]

for c in num_cols:
    df[c + "_c"] = df[c] - df[c].mean()

y_multi = pd.Categorical(df["Target"], categories=["No", "Maybe", "Yes"])
y_codes = y_multi.codes

y_bin = (df["Target"] == "Yes").astype(int)

X_cat = pd.get_dummies(df[cat_cols], drop_first=True, dtype=int) if cat_cols else pd.DataFrame(index=df.index)
X_num = df[[c + "_c" for c in num_cols]]

X = pd.concat([X_num, X_cat], axis=1)

keep = (~X.isna().any(axis=1)) & (y_codes >= 0)
X = X.loc[keep].copy()
y_codes = y_codes[keep]
y_bin_fit = y_bin.loc[keep]


# =========================================================
# 5) CROSS-VALIDATION: BINARY + MULTINOMIAL
# =========================================================

pipe_bin = make_pipeline(
    StandardScaler(with_mean=False),
    LogisticRegression(solver="liblinear", penalty="l2", max_iter=200)
)

pipe_mult = make_pipeline(
    StandardScaler(with_mean=False),
    LogisticRegression(multi_class="multinomial", solver="lbfgs", C=1.0, max_iter=400)
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def binary_scores(pipe, X, y):
    acc = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    f1 = cross_val_score(pipe, X, y, cv=cv, scoring="f1")

    ll, br = [], []
    for tr, te in cv.split(X, y):
        pipe.fit(X.iloc[tr], y.iloc[tr])
        p = pipe.predict_proba(X.iloc[te])[:, 1]
        ll.append(log_loss(y.iloc[te], p, labels=[0, 1]))
        br.append(brier_score_loss(y.iloc[te], p))

    return {
        "acc": acc,
        "f1": f1,
        "logloss": np.array(ll),
        "brier": np.array(br),
    }


def multinomial_scores(pipe, X, y_codes):
    acc = cross_val_score(pipe, X, y_codes, cv=cv, scoring="accuracy")
    f1m = cross_val_score(pipe, X, y_codes, cv=cv, scoring="f1_macro")

    ll = []
    for tr, te in cv.split(X, y_codes):
        pipe.fit(X.iloc[tr], y_codes[tr])
        P = pipe.predict_proba(X.iloc[te])
        ll.append(log_loss(y_codes[te], P, labels=[0, 1, 2]))

    return {
        "acc": acc,
        "f1_macro": f1m,
        "logloss": np.array(ll),
    }


bin_scores = binary_scores(pipe_bin, X, y_bin_fit)

print("\n=== Binary (Yes vs Not-Yes) 5-fold CV ===")
for k, arr in bin_scores.items():
    print(f"{k:8s}: {arr.mean():.3f} (± {arr.std():.3f})")

mult_scores = multinomial_scores(pipe_mult, X, y_codes)

print("\n=== Multinomial (No/Maybe/Yes) 5-fold CV ===")
for k, arr in mult_scores.items():
    print(f"{k:8s}: {arr.mean():.3f} (± {arr.std():.3f})")