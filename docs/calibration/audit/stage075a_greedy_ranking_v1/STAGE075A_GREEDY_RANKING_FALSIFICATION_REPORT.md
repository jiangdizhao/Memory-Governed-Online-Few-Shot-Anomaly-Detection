# Stage 0.75-A — Native Greedy Ranking Falsification Report

Status: **EXECUTED EXPLORATORY DEVELOPMENT AUDIT; INDEPENDENT RESEARCH REVIEW REQUIRED**.

The four development categories were already exposed in Stage 0.5. No confirmation category, mask, Stage 1 or Stage 2 result was accessed here. Dynamic full-stream scores are prequential under changing memory states and do not describe one fixed learned model.

## Full 20 trajectory results

| Category | Seed | N | Defects | Static AUROC | Greedy AUROC | Δ AUROC | Static AP | Greedy AP | Δ AP | Memory changes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| audiojack | 0 | 3815 | 1296 | 0.618178 | 0.707632 | 0.089453 | 0.459562 | 0.603744 | 0.144181 | 73 |
| audiojack | 1 | 3815 | 1296 | 0.579581 | 0.667164 | 0.087583 | 0.391815 | 0.539819 | 0.148003 | 74 |
| audiojack | 2 | 3815 | 1296 | 0.608218 | 0.687669 | 0.079451 | 0.447000 | 0.569925 | 0.122924 | 74 |
| audiojack | 3 | 3815 | 1296 | 0.604476 | 0.644578 | 0.040102 | 0.428443 | 0.520392 | 0.091949 | 74 |
| audiojack | 4 | 3815 | 1296 | 0.702689 | 0.752423 | 0.049734 | 0.575924 | 0.658037 | 0.082113 | 73 |

### audiojack

Five-seed ΔAUROC: mean=0.069265, SD=0.022797, min=0.040102, median=0.079451, max=0.089453, n=5; ΔAUPRC: mean=0.117834, SD=0.029901, min=0.082113, median=0.122924, max=0.148003, n=5.

Per seed, as-operated, conservative group-clean, resident-source-clean, and previously admitted same-group strata are in the machine-readable metrics. Single-class strata mark AUROC and balanced accuracy undefined.
| regulator | 0 | 3855 | 1080 | 0.602720 | 0.761128 | 0.158408 | 0.330169 | 0.587874 | 0.257704 | 80 |
| regulator | 1 | 3855 | 1080 | 0.606260 | 0.765898 | 0.159638 | 0.355743 | 0.573097 | 0.217354 | 78 |
| regulator | 2 | 3855 | 1080 | 0.626414 | 0.762850 | 0.136435 | 0.371293 | 0.620751 | 0.249457 | 77 |
| regulator | 3 | 3855 | 1080 | 0.571461 | 0.745669 | 0.174208 | 0.322443 | 0.531982 | 0.209538 | 78 |
| regulator | 4 | 3855 | 1080 | 0.720835 | 0.860588 | 0.139753 | 0.539887 | 0.748376 | 0.208489 | 76 |

### regulator

Five-seed ΔAUROC: mean=0.153689, SD=0.015577, min=0.136435, median=0.158408, max=0.174208, n=5; ΔAUPRC: mean=0.228508, SD=0.023326, min=0.208489, median=0.217354, max=0.257704, n=5.

Per seed, as-operated, conservative group-clean, resident-source-clean, and previously admitted same-group strata are in the machine-readable metrics. Single-class strata mark AUROC and balanced accuracy undefined.
| rolled_strip_base | 0 | 3805 | 2525 | 0.900086 | 0.973195 | 0.073108 | 0.946924 | 0.986063 | 0.039139 | 33 |
| rolled_strip_base | 1 | 3805 | 2525 | 0.914205 | 0.964042 | 0.049837 | 0.956309 | 0.980965 | 0.024655 | 33 |
| rolled_strip_base | 2 | 3805 | 2525 | 0.911952 | 0.959274 | 0.047322 | 0.955430 | 0.978377 | 0.022947 | 33 |
| rolled_strip_base | 3 | 3805 | 2525 | 0.898308 | 0.967547 | 0.069239 | 0.947205 | 0.983159 | 0.035954 | 33 |
| rolled_strip_base | 4 | 3805 | 2525 | 0.929802 | 0.977942 | 0.048140 | 0.964826 | 0.988958 | 0.024131 | 33 |

### rolled_strip_base

Five-seed ΔAUROC: mean=0.057529, SD=0.012563, min=0.047322, median=0.049837, max=0.073108, n=5; ΔAUPRC: mean=0.029365, SD=0.007578, min=0.022947, median=0.024655, max=0.039139, n=5.

Per seed, as-operated, conservative group-clean, resident-source-clean, and previously admitted same-group strata are in the machine-readable metrics. Single-class strata mark AUROC and balanced accuracy undefined.
| switch | 0 | 3820 | 1994 | 0.793275 | 0.873928 | 0.080653 | 0.832200 | 0.892101 | 0.059901 | 53 |
| switch | 1 | 3820 | 1994 | 0.710252 | 0.881128 | 0.170876 | 0.747620 | 0.891032 | 0.143412 | 54 |
| switch | 2 | 3820 | 1994 | 0.719897 | 0.845738 | 0.125841 | 0.762438 | 0.869828 | 0.107391 | 54 |
| switch | 3 | 3820 | 1994 | 0.661700 | 0.820770 | 0.159070 | 0.712682 | 0.845026 | 0.132344 | 54 |
| switch | 4 | 3820 | 1994 | 0.791541 | 0.895351 | 0.103810 | 0.828147 | 0.905681 | 0.077534 | 53 |

### switch

Five-seed ΔAUROC: mean=0.128050, SD=0.037534, min=0.080653, median=0.125841, max=0.170876, n=5; ΔAUPRC: mean=0.104116, SD=0.035402, min=0.059901, median=0.107391, max=0.143412, n=5.

Per seed, as-operated, conservative group-clean, resident-source-clean, and previously admitted same-group strata are in the machine-readable metrics. Single-class strata mark AUROC and balanced accuracy undefined.

## Fixed snapshots

For each seed, snapshots at t=0, 400, 800, 1600, 2400 and 3200 use writes strictly before t. The next-256 and all-remaining cohorts are the same frozen group-clean positions for static and snapshot scores. Metrics and cohort denominators are in `ranking_metrics.json`.

## Runtime and scope

Four class trajectories completed; full matrix count: 20. Runtime, CUDA peak allocation, per-snapshot C64 thresholds, memory identities and output provenance are recorded in `online_completion_matrix.json`.

No superiority, population-risk guarantee, or independent validation claim is made. Five camera views are not treated as five independent manufactured products.
