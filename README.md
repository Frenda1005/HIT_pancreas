1) Suggested repo structure
pancreas-HIT-dosimetry/
├─ README.md
├─ LICENSE
├─ .gitignore
├─ requirements.txt           # or environment.yml (conda)
├─ configs/
│  └─ example_paths.yaml
├─ data/                      # (empty, gitignored) put patient folders here
│  └─ .keep
├─ outputs/                   # auto-created by scripts; gitignored
│  └─ .keep
├─ src/
│  └─ hit_dosimetry/
│     ├─ __init__.py
│     ├─ io_geometry.py       # load CT/RS/RD, FORUID check, resampling
│     ├─ masks_dose.py        # mask retrieval, overlays, dose resample
│     ├─ dvh_metrics.py       # DVH curve + metrics
│     ├─ radiomics_wrap.py    # pyradiomics wrapper
│     ├─ merge_master.py      # build master_table.csv
│     ├─ survival_models.py   # KM, Cox, RSF
│     ├─ toxicity_models.py   # logistic, RF, NTCP helpers
│     └─ lkb_ntcp.py          # EUD + LKB (TD50,m) fitting
├─ scripts/
│  ├─ 01_extract_dosimetry.py
│  ├─ 02_build_master.py
│  ├─ 03_survival_analysis.py
│  ├─ 04_toxicity_analysis.py
│  └─ 05_ntcp_lkb_fit.py
└─ notebooks/                 # optional demo notebooks
   └─ 00_quickstart.ipynb
Do NOT commit DICOM/PHI. Keep all raw data in data/ (gitignored).
All figures/CSVs go to outputs/ (gitignored).
Code lives in src/hit_dosimetry.
2) Initialize & push to GitHub (commands)
# create folder and init git
mkdir pancreas-HIT-dosimetry && cd pancreas-HIT-dosimetry
git init

# create an empty main branch (optional if default is main)
git checkout -b main

# add files (paste the snippets below)
# then:
git add .
git commit -m "Initial commit: HIT dosimetry pipeline"

# create a GitHub repo (choose one):

# Option A: via GitHub CLI (gh)
# gh auth login   # if not logged in
gh repo create pancreas-HIT-dosimetry --public --source=. --remote=origin --push

# Option B: manually create on github.com, then:
git remote add origin https://github.com/<YOUR_USER>/pancreas-HIT-dosimetry.git
git push -u origin main
3) Pasteable files
README.md
# Pancreatic Cancer HIT Dosimetry Study

End-to-end research pipeline for Heavy Ion Therapy (HIT) dosimetry in pancreatic cancer:
- **Dosimetric characterization** (DVH metrics, overlays, radiomics)
- **Dose–response** (KM, Cox, RSF)
- **Normal-tissue toxicity** (ROC/Youden thresholds, NTCP logistic)
- **LKB NTCP** (EUD, TD50, m, probit MLE)
- **RBE vs physical dose** comparisons
- **Biomarker integration** (e.g., CA19-9)

> ⚠️ **PHI & DICOM**: do **not** commit patient data. This repo is code-only. All raw data stay in `data/` (gitignored).

---

## Quick start

### 1) Environment
```bash
# pip
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
(Or conda env create -f environment.yml if you prefer conda.)
2) Data layout (example)
data/
  49275345/
    CT/ ... DICOM series ...
    RS.dcm
    RD.dcm
    radiomics_params.yaml
  49275346/
    ...
3) Run pipeline
# 01: extract DVH metrics, overlays, radiomics per patient
python scripts/01_extract_dosimetry.py --base data --out outputs

# 02: build master table (merge DVH + radiomics + clinical outcomes)
python scripts/02_build_master.py \
  --base data \
  --out outputs \
  --clinical data/clinical_outcomes.csv

# 03: survival analyses (KM, Cox, RSF)
python scripts/03_survival_analysis.py --master outputs/master_table.csv --out outputs

# 04: toxicity analyses (logistic, RF, thresholds)
python scripts/04_toxicity_analysis.py --master outputs/master_table.csv --out outputs

# 05: LKB NTCP (EUD, TD50, m)
python scripts/05_ntcp_lkb_fit.py \
  --dvh outputs/DVH_curves_long_allpatients.csv \
  --master outputs/master_table.csv \
  --structures Duodenum Stomach SmallBowel \
  --n 0.1 \
  --out outputs
Files produced
outputs/DVH_metrics_summary.csv — per-structure DVH metrics (Dmean, Dmax, D95, D98, D0.03cc, etc.)
outputs/DVH_curves_long_allpatients.csv — cumulative DVHs (dose vs %volume)
outputs/radiomics_features.csv — pyradiomics features per structure
outputs/master_table.csv — merged DVH + radiomics + clinical outcomes
Figures: KM plots, Cox summaries, RSF importances, ROC curves, NTCP calibration, LKB curves
Clinical outcomes template
See data/clinical_outcomes.csv (not committed). Minimum columns:
PatientID,OS_time,OS_event,PFS_time,PFS_event,Toxicity_GI_Grade,CA19-9_baseline,Age,Sex,Performance_Status,Total_Dose_GyRBE,Fractions,Concurrent_Chemo
Safety & privacy
Keep all DICOM/PHI out of git.
Consider Git LFS only for synthetic test NIfTI if needed.
License
MIT (see LICENSE).

## `.gitignore`
```gitignore
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
*.egg-info/
.venv/
env/
.ipynb_checkpoints/

# OS / IDE
.DS_Store
.vscode/
.idea/

# Data & outputs (PHI safety)
data/
!data/.keep
outputs/
!outputs/.keep

# Logs
*.log

# Jupyter
*.ipynb_checkpoints
LICENSE (MIT)
MIT License

Copyright (c) 2025 ...

Permission is hereby granted, free of charge, to any person obtaining a copy
...
requirements.txt (pip)
numpy
pandas
matplotlib
seaborn
pydicom
SimpleITK
rt-utils
pyradiomics
lifelines
scikit-survival
scikit-learn
scipy
patsy
(If you prefer conda, I can generate environment.yml instead.)
configs/example_paths.yaml
base_dir: "data"
outputs_dir: "outputs"
roi_filter: ["SpinalCord","Duodenum","Stomach","Liver","Kidney_R","Kidney_L","GTV","CTV","PTV"]
radiomics_yaml: "radiomics_params.yaml"
4) Minimal script stubs (paste & go)
These wrap the code we already developed into runnable CLI scripts. You can expand them as needed.
scripts/01_extract_dosimetry.py
import os, argparse, pandas as pd
from glob import glob
from src.hit_dosimetry.io_geometry import load_ct, load_dose, check_foruid
from src.hit_dosimetry.masks_dose import load_rtstruct_mask, resample_dose_to_ct, save_overlays
from src.hit_dosimetry.dvh_metrics import dvh_metrics_for_roi, save_dvh_curve
from src.hit_dosimetry.radiomics_wrap import extract_radiomics

def main(base, out):
    os.makedirs(out, exist_ok=True)
    all_metrics, all_dvh, all_rads = [], [], []

    patient_dirs = sorted([d for d in glob(os.path.join(base, "*")) if os.path.isdir(d)])
    for pdir in patient_dirs:
        pid = os.path.basename(pdir)
        ct_dir = os.path.join(pdir, "CT")
        rs_path = sorted(glob(os.path.join(pdir, "RS*.dcm")) + glob(os.path.join(pdir, "RS.*.dcm")))[0]
        rd_path = sorted(glob(os.path.join(pdir, "RD*.dcm")) + glob(os.path.join(pdir, "RD.*.dcm")))[0]
        rad_yaml = os.path.join(pdir, "radiomics_params.yaml")

        ct_img, ct_arr, ct_first = load_ct(ct_dir)
        dose_img, dose_arr, units = load_dose(rd_path)
        check_foruid(ct_first, rs_path, rd_path)

        dose_ct_img, dose_ct = resample_dose_to_ct(dose_img, ct_img)
        voxel_mm3 = float(ct_img.GetSpacing()[0]*ct_img.GetSpacing()[1]*ct_img.GetSpacing()[2])

        # Get ROIs from RS
        roi_names, get_mask = load_rtstruct_mask(ct_dir, rs_path)

        for roi in roi_names:
            mask = get_mask(roi, ct_arr.shape)
            if mask is None: continue
            # QC overlays & GIF
            save_overlays(ct_arr, dose_ct, mask, roi, units, out)
            # DVH curve + metrics
            curve = save_dvh_curve(dose_ct, mask, roi, units, out)
            for d,v in curve:
                all_dvh.append({"PatientID": pid, "Structure": roi, "Dose": d, "VolumePct": v})
            met = dvh_metrics_for_roi(dose_ct, mask, voxel_mm3)
            met.update({"PatientID": pid, "Structure": roi})
            all_metrics.append(met)
            # Radiomics
            feats = extract_radiomics(ct_img, mask, rad_yaml)
            feats.update({"PatientID": pid, "Structure": roi})
            all_rads.append(feats)

    pd.DataFrame(all_metrics).to_csv(os.path.join(out, "DVH_metrics_summary.csv"), index=False)
    pd.DataFrame(all_dvh).to_csv(os.path.join(out, "DVH_curves_long_allpatients.csv"), index=False)
    pd.DataFrame(all_rads).to_csv(os.path.join(out, "radiomics_features.csv"), index=False)
    print("Saved metrics, curves, radiomics to", out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main(args.base, args.out)
scripts/02_build_master.py
import os, argparse, pandas as pd
from glob import glob

def main(base, out, clinical):
    os.makedirs(out, exist_ok=True)
    dvh = pd.read_csv(os.path.join(out, "DVH_metrics_summary.csv"))
    rads = pd.read_csv(os.path.join(out, "radiomics_features.csv"))
    clin = pd.read_csv(clinical)

    dvh_wide = dvh.pivot_table(index="PatientID",
                               columns="Structure",
                               values=["Dmean","Dmax","D95","D98","V20Gy_pct","V30Gy_pct","V40Gy_pct","D1cc","D2cc","D0.03cc"])
    dvh_wide.columns = [f"{roi}_{metric}" for metric, roi in dvh_wide.columns]
    dvh_wide = dvh_wide.reset_index()

    rads_wide = rads.drop(columns=[c for c in rads.columns if "diagnostics" in c], errors="ignore") \
                    .pivot_table(index="PatientID", columns="Structure", aggfunc="first")
    rads_wide.columns = [f"{roi}_{feat}" for feat, roi in rads_wide.columns]
    rads_wide = rads_wide.reset_index()

    master = clin.merge(dvh_wide, on="PatientID", how="left").merge(rads_wide, on="PatientID", how="left")
    master.to_csv(os.path.join(out, "master_table.csv"), index=False)
    print("Saved master_table.csv to", out)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clinical", required=True)
    args = ap.parse_args()
    main(args.base, args.out, args.clinical)
scripts/03_survival_analysis.py
import os, argparse, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from lifelines import KaplanMeierFitter, CoxPHFitter
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored

def main(master_path, out):
    os.makedirs(out, exist_ok=True)
    df = pd.read_csv(master_path)

    # KM overall OS
    km = KaplanMeierFitter()
    plt.figure(figsize=(6,5))
    km.fit(df["OS_time"], df["OS_event"], label="All")
    km.plot_survival_function(ci_show=True)
    plt.title("OS (All)"); plt.xlabel("Months"); plt.ylabel("Survival"); plt.grid(True)
    plt.tight_layout(); plt.savefig(os.path.join(out, "KM_OS_all.png"), dpi=200); plt.close()

    # Cox example
    cph = CoxPHFitter()
    cols = ["OS_time","OS_event","Duodenum_D0.03cc","CA19-9_baseline","Age"]
    sub = df.dropna(subset=cols)[cols]
    cph.fit(sub, duration_col="OS_time", event_col="OS_event")
    cph.print_summary()

    # RSF example
    import numpy as np
    y = np.array(list(zip(df["OS_event"].astype(bool), df["OS_time"])),
                 dtype=[("event", bool), ("time", float)])
    X = df[["Duodenum_D0.03cc","Stomach_D1cc","Liver_Dmean","Age","CA19-9_baseline"]].fillna(0)
    rsf = RandomSurvivalForest(n_estimators=500, min_samples_split=5, min_samples_leaf=2,
                               max_features="sqrt", n_jobs=-1, random_state=42)
    rsf.fit(X, y)
    cidx = concordance_index_censored(y["event"], y["time"], rsf.predict(X))[0]
    print("RSF C-index:", cidx)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main(args.master, args.out)
scripts/04_toxicity_analysis.py
import os, argparse, pandas as pd, matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

def main(master_path, out):
    os.makedirs(out, exist_ok=True)
    df = pd.read_csv(master_path)
    df["Toxicity_High"] = (df["Toxicity_GI_Grade"] >= 3).astype(int)

    X = df[["Duodenum_D0.03cc","Stomach_D1cc","Liver_Dmean","Age","CA19-9_baseline"]].fillna(0)
    y = df["Toxicity_High"]

    # Logistic
    logit = LogisticRegression(max_iter=1000, class_weight="balanced")
    logit.fit(X, y); p = logit.predict_proba(X)[:,1]
    auc = roc_auc_score(y, p); fpr,tpr,_ = roc_curve(y, p)
    plt.figure(figsize=(5,5))
    plt.plot(fpr,tpr,label=f"AUC={auc:.2f}"); plt.plot([0,1],[0,1],"--")
    plt.title("Logistic NTCP ROC"); plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(out, "NTCP_logistic_ROC.png"), dpi=200); plt.close()

    # Random Forest
    rf = RandomForestClassifier(n_estimators=500, max_depth=5, class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X, y); pr = rf.predict_proba(X)[:,1]
    auc2 = roc_auc_score(y, pr)
    print("RF AUC:", auc2)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main(args.master, args.out)
scripts/05_ntcp_lkb_fit.py
import os, argparse, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, roc_curve

def cumulative_to_differential(dose, volpct_cum):
    order = np.argsort(dose)
    d = np.asarray(dose)[order]; vcum = np.asarray(volpct_cum)[order]/100.0
    vdiff = np.maximum(vcum[:-1] - vcum[1:], 0.0)
    dmids = 0.5*(d[:-1] + d[1:])
    total = vdiff.sum()
    if total > 0: vdiff = vdiff/total
    return dmids, vdiff

def eud_from_dvh(dose, volpct_cum, n=0.1):
    a = 1.0/float(n)
    dmids, vfrac = cumulative_to_differential(dose, volpct_cum)
    dmids = np.clip(dmids, 1e-6, None)
    return float((np.sum(vfrac*(dmids**a)))**(1.0/a))

def fit_lkb(eud, y, TD50_init=None, m_init=0.15):
    def nll(params):
        TD50, m = params
        if TD50<=0 or m<=0 or m>5: return 1e6
        t = (eud - TD50)/(m*TD50); p = norm.cdf(t)
        p = np.clip(p, 1e-9, 1-1e-9)
        return -(y*np.log(p) + (1-y)*np.log(1-p)).sum()
    if TD50_init is None: TD50_init = np.median(eud)
    res = minimize(nll, x0=[TD50_init, m_init], method="Nelder-Mead")
    return res.x[0], res.x[1], res

def youden(y_true, scores):
    fpr,tpr,thr = roc_curve(y_true, scores); j = tpr - fpr; k = np.argmax(j)
    return thr[k], tpr[k], fpr[k], roc_auc_score(y_true, scores)

def main(dvh_path, master_path, structures, n, out):
    os.makedirs(out, exist_ok=True)
    dvh = pd.read_csv(dvh_path)
    clin = pd.read_csv(master_path)
    rows = []

    for roi in structures:
        if roi not in dvh["Structure"].unique(): 
            print("skip", roi); continue
        euds = []
        for pid, g in dvh[dvh["Structure"]==roi].groupby("PatientID"):
            euds.append({"PatientID": pid, "EUD": eud_from_dvh(g["Dose"].values, g["VolumePct"].values, n=n)})
        eud_df = pd.DataFrame(euds)
        data = clin[["PatientID","Toxicity_GI_Grade"]].merge(eud_df, on="PatientID").dropna()
        data["Toxicity_High"] = (data["Toxicity_GI_Grade"] >= 3).astype(int)
        if data.empty: continue

        y = data["Toxicity_High"].values
        e = data["EUD"].values
        TD50, m, res = fit_lkb(e, y)
        t = (e - TD50)/(m*TD50); p = norm.cdf(t)
        auc = roc_auc_score(y, p)
        thr, tpr, fpr, auc_eud = youden(y, e)

        rows.append({"Structure": roi, "n": n, "TD50": TD50, "m": m, "AUC_NTCP": auc, "EUD_cutoff": thr, "AUC_EUD": auc_eud})

        xs = np.linspace(max(0.001, e.min()*0.8), e.max()*1.2, 200)
        pc = norm.cdf((xs - TD50)/(m*TD50))
        plt.figure(figsize=(6,5))
        plt.scatter(e, y, alpha=0.7)
        plt.plot(xs, pc, label=f"LKB TD50={TD50:.1f}, m={m:.2f}")
        plt.axvline(thr, ls="--", label=f"Youden {thr:.1f}")
        plt.ylim(-0.05,1.05); plt.xlabel(f"{roi} EUD (Gy or Gy[RBE])"); plt.ylabel("NTCP/Toxicity prob")
        plt.title(f"{roi} — LKB (n={n})"); plt.grid(True); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(out, f"{roi}_LKB_n{n}.png"), dpi=200); plt.close()

    pd.DataFrame(rows).to_csv(os.path.join(out, f"LKB_summary_n{n}.csv"), index=False)
    print("Saved LKB summary to", out)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dvh", required=True)
    ap.add_argument("--master", required=True)
    ap.add_argument("--structures", nargs="+", required=True)
    ap.add_argument("--n", type=float, default=0.1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main(args.dvh, args.master, args.structures, args.n, args.out)
5) Last useful bits
Badges/credits: add citations to pyradiomics, lifelines, scikit-survival in README.
Issues/PR templates: optional, for collaboration.
Releases: tag a v0.1.0 once your first run completes.
