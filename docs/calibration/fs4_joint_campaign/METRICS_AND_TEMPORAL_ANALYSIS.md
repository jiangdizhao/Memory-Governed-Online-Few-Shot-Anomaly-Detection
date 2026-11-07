# Metrics and temporal analysis

All Stage B metrics use sealed pre-event scores, thresholds and predictions. For a binary label with defect as positive, `FPR=FP/(FP+TN)`, `FNR=FN/(FN+TP)`, `BER=(FPR+FNR)/2`, and `BalancedAccuracy=1-BER`. A missing class denominator yields NA with an explicit reason. AUROC uses average ranks for tied scores. Average Precision uses the stepwise precision-recall sum over distinct descending score thresholds; it is NA when there are no defect positives.

Per category and registered readout report TP/TN/FP/FN, FPR, FNR, BER, Balanced Accuracy, AUROC and Average Precision. Four-category macros give equal weight to each category and are NA if any required category metric is undefined. H1 is the paired category-macro `E−L` BER contrast for S/GC. H2 is the separate category-macro `CE/E/GC−CL/L/GC` BER contrast. Shared L/E score rankings must match exactly for each scorer; no shared AUROC/AP improvement claim is permitted.

Within each category, the segment unit is Source Group. For `n` groups, use the frozen integer bounds `[0,floor(n/3))`, `[floor(n/3),floor(2n/3))`, `[floor(2n/3),n)`. Report event counts, BER/FPR/FNR, query rate/count, verified-normal feedback, ESS, active threshold updates and holds.

Sliding windows are 50 adjacent groups (250 events) with stride 10. Report BER, FPR, FNR, class counts, executed queries/rate, normal feedback, ESS sufficiency, E rescore count/time/cache bytes and longest query-free image interval. One-class BER is NA, with class counts and excluded-window count. Worst-window BER is the maximum valid BER; worst FPR and FNR are separate maxima with their valid counts.

Query diagnostics include image and group query rates, candidate and event ordinal distributions, selection-probability distribution and clipping count, cost/utility history, distinct normal evidence groups, effective sample size, earliest accepted update, holds by reason, threshold movement, query deserts, E cache occupancy/rescore cost, and native Greedy/C1 versions. Worker telemetry reports elapsed time, host working set and CUDA allocated/reserved peaks. No mask or pixel metric is opened.

The unit of dependence is the five-view Source Group. Four categories and one stochastic seed do not support broad population claims. No inferential uncertainty procedure is registered. Any future descriptive cluster bootstrap needs its own frozen seed and must not be described as exact uncertainty for a counterfactual adaptive policy.
