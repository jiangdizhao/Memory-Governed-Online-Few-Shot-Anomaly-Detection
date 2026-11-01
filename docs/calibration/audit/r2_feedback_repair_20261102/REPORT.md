# R2-FEEDBACK-REPAIR-1 — Feedback-Path Attribution and Residual-Error Repair

**Classification:** `RETROSPECTIVE_DEVELOPMENT_DIAGNOSTIC`
**Result:** Complete CPU replay and post-seal evaluation; C1-Full parity passed for all four categories.
**Final namespace:** `r2_feedback_repair_20261102_a6` (Stage A reused from its sealed source `r2_feedback_repair_20261102_a4`).

Historical development outcome summaries had already been observed by the researcher/developer. This is not blind, independent, external, or confirmation evidence. The run used only the previously cached four development prefixes, the original 67 legal query disclosures, and the authorized first-512 outcome projection after every new score was sealed.

## Executive result

C1-Full (`G_NA`) increased AUROC over Greedy-only (`G`) in each class-bearing prefix: switch +0.03605 (paired Source Group bootstrap 95% CI +0.00635 to +0.06942), regulator +0.00747 (−0.01908 to +0.03719), and rolled_strip_base +0.01249 (−0.00440 to +0.03125). The strict four-category AUROC macro is undefined because audiojack is normal-only; the three-category descriptive means are 0.79750 for G and 0.81617 for G_NA (difference +0.01867).

At the frozen C64 threshold, C1-Full lowered missed anomalies in all three class-bearing categories, with false-alarm costs that varied by category. The observed gains therefore combine ranking changes and operating-point trade-offs. G_NA exceeded the geometry-free control in AUROC in all three categories, but the regulator bootstrap interval includes zero. The normal-only audiojack prefix contributes false-alarm evidence only.

## Protocol and causal separation

- Exactly 2,048 events: first 512 each for switch, audiojack, regulator, rolled_strip_base; support seed 0; original four Supports plus additional C64 normals. This is not strict four-shot-total.
- The recorded C0 trajectory, Query Tape, Top-32 descriptors, thresholds, dynamic Greedy-memory decisions, and Source Group exclusions were held fixed. C1 banks were independent by arm and category. No query, admission, threshold, or event-window changes were made.
- Stage A scored each event from its cached C0 map and Top-32 features, durably appended all current scores, then read at most the current event’s queried feedback and applied the recorded C1-Full gate/compact-beta/features through the native `SignedCorrectionCodebook.update()`. No offline evaluator labels entered scoring.
- G_GLOBAL used the eligible pre-event G_NA bank and the fixed scalar `sum(kappa*beta)/(1e-6+sum(kappa))` on the same Top-32 positions. It owned no update bank; temporal and same-source exclusions were retained.
- Only after the complete Stage A score seal did a separate supervised Stage B project the authorized first-512 outcomes for these categories. No masks or images were read/decoded. The fixed unqueried subset is diagnostic and is not an unbiased alternative benchmark.
- All 67 original queries were preserved: 40 normal feedback (17 audiojack, 13 regulator, 3 rolled_strip_base, 7 switch) and 27 anomalous feedback (3 regulator, 14 rolled_strip_base, 10 switch). Audiojack had no anomalous feedback.

## C1-Full parity and source audit

Numerical tolerances were fixed at `atol=1e-5`, `rtol=1e-4`. All four categories passed C0 identity, C1-Full patch-delta/image-score checks, exact query/update counts, native update statistics, occupancy checks, and Source Group exclusion checks. The maxima below are actual CPU replay differences; the non-bitwise counts reflect CPU/GPU float ordering, not tolerance failures.

|Category|Status|Events|Delta checks|Max |Δ patch||Score checks|Max |Δ score||Non-bitwise Δ|Non-bitwise score|Update mismatches|Occupancy mismatches|Source-exclusion mismatches|CPU/state SHA mismatch|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|switch|PASS|512|512|0.00000015|512|0.00000012|13115|98|0|0|0|1|
|audiojack|PASS|512|512|0.00000008|512|0.00000006|12871|88|0|0|0|1|
|regulator|PASS|512|512|0.00000022|512|0.00000018|13497|182|0|0|0|1|
|rolled_strip_base|PASS|512|512|0.00000012|512|0.00000006|12471|69|0|0|0|1|

Reference AUROC checks passed: switch 0.84308264 vs 0.843083; regulator 0.69462328 vs 0.694623; rolled_strip_base 0.91079812 vs 0.910798. Audiojack AUROC/AP were correctly undefined. Final CPU state SHA differed from the historical execution digest in each category; that digest is diagnostic across CPU/GPU update paths, while the numerical score, update-stat, occupancy, and exclusion checks passed.

The old EXP2A manifest’s initial status is `INCOMPLETE`, but its completion record and Stage A seal say `COMPLETE` / `SEALED_COMPLETE`; those sealed receipts governed input acceptance. The EXP2A runner-source SHA differs from the old manifest’s expected runner SHA (`4b798d…` actual vs `5f3696…` expected). The frozen native C1 dependencies match, cached rows/shards are hash-verified, and the replay passed the empirical C1-Full parity checks; the historical runner-hash discrepancy remains disclosed rather than concealed.

## Ranking and frozen-threshold results

Confusion matrices are `TN/FP/FN/TP`; FPR/FNR/BA use the original frozen C64 threshold for each category. `—` denotes undefined for the normal-only audiojack class mix.

|Category|Arm|Anomaly prevalence|AUROC|AP|FPR|FNR|BA|TN/FP/FN/TP|
|---|---|---|---|---|---|---|---|---|
|switch|G|0.701|0.8070|0.9152|0.0261|0.6017|0.6861|149/4/216/143|
|switch|G_N|0.701|0.8178|0.9206|0.0131|0.6490|0.6690|151/2/233/126|
|switch|G_A|0.701|0.8259|0.9250|0.1373|0.3760|0.7434|132/21/135/224|
|switch|G_NA|0.701|0.8431|0.9322|0.0654|0.4318|0.7514|143/10/155/204|
|switch|G_GLOBAL|0.701|0.7959|0.9107|0.1176|0.4735|0.7044|135/18/170/189|
|audiojack|G|0.000|—|—|0.0059|—|—|509/3/0/0|
|audiojack|G_N|0.000|—|—|0.0039|—|—|510/2/0/0|
|audiojack|G_A|0.000|—|—|0.0059|—|—|509/3/0/0|
|audiojack|G_NA|0.000|—|—|0.0039|—|—|510/2/0/0|
|audiojack|G_GLOBAL|0.000|—|—|0.0000|—|—|512/0/0/0|
|regulator|G|0.389|0.6872|0.5888|0.0096|0.9246|0.5329|310/3/184/15|
|regulator|G_N|0.389|0.6914|0.6059|0.0064|0.9397|0.5270|311/2/187/12|
|regulator|G_A|0.389|0.6274|0.5181|0.1150|0.8040|0.5405|277/36/160/39|
|regulator|G_NA|0.389|0.6946|0.6219|0.0128|0.8492|0.5690|309/4/169/30|
|regulator|G_GLOBAL|0.389|0.6844|0.5736|0.0064|0.9347|0.5295|311/2/186/13|
|rolled_strip_base|G|0.861|0.8983|0.9824|0.0704|0.3129|0.8083|66/5/138/303|
|rolled_strip_base|G_N|0.861|0.8779|0.9786|0.0423|0.3923|0.7827|68/3/173/268|
|rolled_strip_base|G_A|0.861|0.9044|0.9836|0.2113|0.1338|0.8275|56/15/59/382|
|rolled_strip_base|G_NA|0.861|0.9108|0.9849|0.1831|0.1383|0.8393|58/13/61/380|
|rolled_strip_base|G_GLOBAL|0.861|0.8964|0.9821|0.2113|0.1542|0.8173|56/15/68/373|

### Secondary cached references

C1-NoBoth and historical C2(a=0.02) are shown only as fixed secondary references; no sweep or winner selection was performed.

|Category|Reference|AUROC|AP|FPR|FNR|BA|
|---|---|---|---|---|---|---|
|switch|C1_NO_BOTH|0.8525|0.9369|0.0588|0.4791|0.7310|
|switch|C2_A002|0.8410|0.9305|0.0588|0.4958|0.7227|
|audiojack|C1_NO_BOTH|—|—|0.0039|—|—|
|audiojack|C2_A002|—|—|0.0000|—|—|
|regulator|C1_NO_BOTH|0.7180|0.6387|0.0096|0.9347|0.5279|
|regulator|C2_A002|0.7041|0.6089|0.0064|0.9447|0.5244|
|rolled_strip_base|C1_NO_BOTH|0.9069|0.9841|0.1831|0.1701|0.8234|
|rolled_strip_base|C2_A002|0.9050|0.9836|0.1549|0.1950|0.8250|

## Residual-error repair ledger

Rates use the correct G baseline denominators: FN repairs / G false negatives; FP repairs / G false positives; introduced FN / G true positives; introduced FP / G true negatives. Empty denominators remain undefined. Every candidate confusion matrix reconstructed from the four transition counts.

|Category|Arm|FN→TP repaired|FP→TN repaired|TP→FN introduced|TN→FP introduced|Net FN reduction|Net FP reduction|FN repair rate|FP repair rate|New-FN rate|New-FP rate|Confusion verified|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|switch|G_N|0|2|17|0|-17|2|0.00%|50.00%|11.89%|0.00%|yes|
|switch|G_A|81|0|0|17|81|-17|37.50%|0.00%|0.00%|11.41%|yes|
|switch|G_NA|62|2|1|8|61|-6|28.70%|50.00%|0.70%|5.37%|yes|
|switch|G_GLOBAL|47|0|1|14|46|-14|21.76%|0.00%|0.70%|9.40%|yes|
|audiojack|G_N|0|1|0|0|0|1|—|33.33%|—|0.00%|yes|
|audiojack|G_A|0|0|0|0|0|0|—|0.00%|—|0.00%|yes|
|audiojack|G_NA|0|1|0|0|0|1|—|33.33%|—|0.00%|yes|
|audiojack|G_GLOBAL|0|3|0|0|0|3|—|100.00%|—|0.00%|yes|
|regulator|G_N|0|1|3|0|-3|1|0.00%|33.33%|20.00%|0.00%|yes|
|regulator|G_A|24|0|0|33|24|-33|13.04%|0.00%|0.00%|10.65%|yes|
|regulator|G_NA|16|1|1|2|15|-1|8.70%|33.33%|6.67%|0.65%|yes|
|regulator|G_GLOBAL|2|1|4|0|-2|1|1.09%|33.33%|26.67%|0.00%|yes|
|rolled_strip_base|G_N|0|2|35|0|-35|2|0.00%|40.00%|11.55%|0.00%|yes|
|rolled_strip_base|G_A|79|0|0|10|79|-10|57.25%|0.00%|0.00%|15.15%|yes|
|rolled_strip_base|G_NA|78|1|1|9|77|-8|56.52%|20.00%|0.33%|13.64%|yes|
|rolled_strip_base|G_GLOBAL|70|0|0|10|70|-10|50.72%|0.00%|0.00%|15.15%|yes|

Observed examples: switch G_NA repaired 62 missed anomalies and 2 false alarms, introduced 1 missed anomaly and 8 false alarms (net FN reduction 61; FP count increased by 6). Regulator repaired 16 FN and 1 FP, introduced 1 FN and 2 FP (net FN reduction 15; FP count increased by 1). Rolled_strip_base repaired 78 FN and 1 FP, introduced 1 FN and 9 FP (net FN reduction 77; FP count increased by 8). Audiojack’s only measurable result is reduction from 3 to 2 false alarms for G_NA; anomaly-side rates are undefined.

## Paired effects and Source Group uncertainty

Paired effects below are candidate minus reference. The 1,000 Source Group bootstrap replicates (seed 20261101) reuse each group draw across arms and condition on the recorded trajectories.

|Category|Contrast vs G|ΔAUROC|ΔAP|ΔBA|ΔFPR|ΔFNR|
|---|---|---|---|---|---|---|
|switch|G_N|0.0107|0.0054|-0.0171|-0.0131|0.0474|
|switch|G_A|0.0189|0.0099|0.0573|0.1111|-0.2256|
|switch|G_NA|0.0360|0.0170|0.0654|0.0392|-0.1699|
|switch|G_GLOBAL|-0.0112|-0.0045|0.0183|0.0915|-0.1281|
|audiojack|G_N|—|—|—|-0.0020|—|
|audiojack|G_A|—|—|—|0.0000|—|
|audiojack|G_NA|—|—|—|-0.0020|—|
|audiojack|G_GLOBAL|—|—|—|-0.0059|—|
|regulator|G_N|0.0043|0.0171|-0.0059|-0.0032|0.0151|
|regulator|G_A|-0.0598|-0.0708|0.0076|0.1054|-0.1206|
|regulator|G_NA|0.0075|0.0331|0.0361|0.0032|-0.0754|
|regulator|G_GLOBAL|-0.0027|-0.0152|-0.0034|-0.0032|0.0101|
|rolled_strip_base|G_N|-0.0204|-0.0038|-0.0256|-0.0282|0.0794|
|rolled_strip_base|G_A|0.0061|0.0011|0.0191|0.1408|-0.1791|
|rolled_strip_base|G_NA|0.0125|0.0025|0.0310|0.1127|-0.1746|
|rolled_strip_base|G_GLOBAL|-0.0019|-0.0003|0.0089|0.1408|-0.1587|

### Primary comparison: G_NA − G, paired bootstrap 95% percentile intervals

|Category|ΔAUROC|AUROC 95% CI|ΔAP|AP 95% CI|ΔBA|BA 95% CI|ΔFPR|FPR 95% CI|ΔFNR|FNR 95% CI|
|---|---|---|---|---|---|---|---|---|---|---|
|switch|0.0360|[0.0063, 0.0694]|0.0170|[0.0035, 0.0319]|0.0654|[0.0341, 0.0952]|0.0392|[0.0000, 0.0759]|-0.1699|[-0.2151, -0.1264]|
|regulator|0.0075|[-0.0191, 0.0372]|0.0331|[0.0017, 0.0646]|0.0361|[0.0177, 0.0556]|0.0032|[-0.0065, 0.0139]|-0.0754|[-0.1132, -0.0386]|
|rolled_strip_base|0.0125|[-0.0044, 0.0312]|0.0025|[-0.0005, 0.0064]|0.0310|[-0.0267, 0.0848]|0.1127|[0.0156, 0.2181]|-0.1746|[-0.2210, -0.1297]|

### Incremental-feedback and geometry-free AUROC contrasts

|Category|Contrast|ΔAUROC|95% Source Group bootstrap CI|
|---|---|---|---|
|switch|G_NA_minus_G_N|0.0253|[-0.0060, 0.0607]|
|switch|G_NA_minus_G_A|0.0172|[0.0009, 0.0337]|
|switch|G_NA_minus_G_GLOBAL|0.0472|[0.0215, 0.0746]|
|regulator|G_NA_minus_G_N|0.0032|[-0.0267, 0.0352]|
|regulator|G_NA_minus_G_A|0.0672|[0.0351, 0.0976]|
|regulator|G_NA_minus_G_GLOBAL|0.0102|[-0.0156, 0.0366]|
|rolled_strip_base|G_NA_minus_G_N|0.0329|[0.0039, 0.0651]|
|rolled_strip_base|G_NA_minus_G_A|0.0064|[-0.0012, 0.0163]|
|rolled_strip_base|G_NA_minus_G_GLOBAL|0.0144|[0.0055, 0.0255]|

Full bootstrap fields for AUROC, AP, BA, FPR and FNR are preserved for all contrasts in `aggregate_results.json`. The strict four-category ranking macro is undefined; reported three-category means are descriptive only.

## Feedback, bank evolution, and Source Group exclusion

The corrected category-specific accounting is a post-seal replay of only queried update inputs; it did not recompute scores. It validates G_NA native update statistics and occupancy against the recorded C1-Full diagnostics. The Stage A score/seal files were reused byte-for-byte from a4.

|Category|Bank|Offered normal/anomalous queries|Inserted|Merged|Replaced|Skipped|Final occupancy|Max occupancy|Excluded slot-events|Validation|
|---|---|---|---|---|---|---|---|---|---|---|
|switch|G_N|7/10|107|0|0|117|107|107|175|PASS|
|switch|G_A|7/10|201|0|0|119|201|201|507|PASS|
|switch|G_NA|7/10|256|0|34|254|256|256|656|PASS|
|audiojack|G_N|17/0|147|0|0|397|147|147|350|PASS|
|audiojack|G_A|17/0|0|0|0|0|0|0|0|PASS|
|audiojack|G_NA|17/0|147|0|0|397|147|147|350|PASS|
|regulator|G_N|13/3|256|7|46|107|256|256|564|PASS|
|regulator|G_A|13/3|71|0|0|25|71|71|103|PASS|
|regulator|G_NA|13/3|256|7|101|148|256|256|640|PASS|
|rolled_strip_base|G_N|3/14|17|0|0|79|17|17|39|PASS|
|rolled_strip_base|G_A|3/14|252|6|0|190|252|252|527|PASS|
|rolled_strip_base|G_NA|3/14|256|6|13|269|256|256|566|PASS|

G_N accepts normal feedback only; G_A accepts anomalous feedback only; G_NA accepts both. Normal-memory decisions remain on the common recorded C0 trajectory. Source Group exclusion is applied at each pre-event bank snapshot. Audiojack G_A remains empty because it received no anomalous feedback.

## Feature-dependent correction, sign decomposition, and pair ordering

At the same G_NA snapshot, positive and negative Gaussian contributions used the complete shared denominator. Maximum absolute decomposition error was ≤2.24×10⁻⁸. Dense-map maxima were recomputed after patch correction; image-score changes are not additive after max aggregation.

|Category|Mean |positive patch contribution||Mean |negative patch contribution||Patch cancellation fraction|Mean full image-score effect vs G|Max decomposition error|
|---|---|---|---|---|---|
|switch|0.0162|0.0067|0.1471|0.0149|0.00000001|
|audiojack|0.0000|0.0164|0.0000|-0.0132|0.00000000|
|regulator|0.0098|0.0180|0.2360|0.0097|0.00000002|
|rolled_strip_base|0.0202|0.0015|0.0639|0.0214|0.00000001|

Pair credits assign 0.5 to score ties. Cross-Source-Group diagnostics use their own denominators and do not reuse the all-pair denominator.

|Category|Arm|Improved all pairs|Damaged all pairs|All-pair N|All-pair ΔAUC|Improved cross-group|Damaged cross-group|Cross-group N|Cross-group ΔAUC|
|---|---|---|---|---|---|---|---|---|---|
|switch|G_N|2242|1653|54927|0.0107|2227|1645|54591|0.0107|
|switch|G_A|3644|2607|54927|0.0189|3621|2598|54591|0.0187|
|switch|G_NA|4951|2971|54927|0.0360|4915|2956|54591|0.0359|
|switch|G_GLOBAL|1201|1814|54927|-0.0112|1201|1814|54591|-0.0112|
|regulator|G_N|1876|1609|62287|0.0043|1857|1606|61865|0.0041|
|regulator|G_A|3919|7642|62287|-0.0598|3900|7612|61865|-0.0600|
|regulator|G_NA|5209|4744|62287|0.0075|5173|4714|61865|0.0074|
|regulator|G_GLOBAL|1617|1787|62287|-0.0027|1617|1787|61865|-0.0027|
|rolled_strip_base|G_N|386|1026|31311|-0.0204|386|1024|31271|-0.0204|
|rolled_strip_base|G_A|610|420|31311|0.0061|609|420|31271|0.0060|
|rolled_strip_base|G_NA|774|383|31311|0.0125|774|382|31271|0.0125|
|rolled_strip_base|G_GLOBAL|280|340|31311|-0.0019|280|340|31271|-0.0019|

G_NA increased cross-group AUC by +0.03589 for switch, +0.00742 for regulator, and +0.01254 for rolled_strip_base. G_GLOBAL reduced AUROC relative to G for switch and rolled_strip_base and was nearly unchanged for regulator; the paired G_NA−G_GLOBAL intervals are positive for switch/rolled and include zero for regulator.

## Fixed unqueried-event subset

This is the original unqueried-event subset under the recorded causal trajectory, not a newly selected test set.

|Category|Arm|Events|Prevalence|AUROC|AP|FPR|FNR|BA|
|---|---|---|---|---|---|---|---|---|
|switch|G|495|0.705|0.8099|0.9186|0.0205|0.6017|0.6889|
|switch|G_NA|495|0.705|0.8470|0.9359|0.0548|0.4269|0.7591|
|switch|G_GLOBAL|495|0.705|0.7993|0.9140|0.1096|0.4699|0.7102|
|audiojack|G|495|0.000|—|—|0.0040|—|—|
|audiojack|G_NA|495|0.000|—|—|0.0020|—|—|
|audiojack|G_GLOBAL|495|0.000|—|—|0.0000|—|—|
|regulator|G|496|0.395|0.6882|0.5956|0.0100|0.9235|0.5333|
|regulator|G_NA|496|0.395|0.7003|0.6294|0.0133|0.8520|0.5673|
|regulator|G_GLOBAL|496|0.395|0.6870|0.5822|0.0067|0.9337|0.5298|
|rolled_strip_base|G|495|0.863|0.9004|0.9829|0.0735|0.3091|0.8087|
|rolled_strip_base|G_NA|495|0.863|0.9136|0.9855|0.1765|0.1382|0.8427|
|rolled_strip_base|G_GLOBAL|495|0.863|0.8990|0.9827|0.2059|0.1546|0.8198|

G_NA AUROC exceeded G on the unqueried subsets for switch (0.84704 vs 0.80989), regulator (0.70029 vs 0.68821), and rolled_strip_base (0.91356 vs 0.90043), with Source Group exclusion active. These retrospective, historically exposed outcomes do not constitute independent or blind confirmation. Audiojack remains ranking-undefined and supplies normal-side FPR only.

## Answers to the seven research questions

1. **Which Greedy-only errors does C1-Full repair, and at what cost?** It repairs missed anomalies across all three class-bearing categories, while also changing false alarms. Net FN reductions are 61/15/77 for switch/regulator/rolled_strip_base. Net FP changes are −6/+1/−8 respectively (positive means fewer FPs in the ledger). Thus switch and rolled trade increased false alarms for lower FNR; regulator has a smaller FPR cost and lower FNR. Audiojack repaired one false alarm but has no anomaly-side evidence.
2. **Incremental anomalous feedback?** Relative to G_N, adding anomaly feedback raised AUROC by +0.02532 switch, +0.00318 regulator, +0.03293 rolled_strip_base. The intervals include zero for switch/regulator and exclude zero for rolled. At the frozen threshold this lowers FNR but raises FPR in all three; the false-alarm cost is largest for rolled_strip_base.
3. **Incremental normal feedback already available to Greedy?** Relative to G_A, adding normal feedback raised AUROC by +0.01717 switch, +0.06724 regulator, +0.00642 rolled_strip_base. Its bootstrap interval excludes zero for switch/regulator and crosses zero for rolled. It lowers FPR substantially but can increase FNR; this is evidence about these replayed C1 banks, not a claim that normal memory itself cannot make such repairs.
4. **Does feature-dependent correction beat the geometry-free control?** G_NA AUROC exceeds G_GLOBAL by +0.04721 switch, +0.01019 regulator, +0.01440 rolled_strip_base. Paired intervals exclude zero for switch and rolled, but include zero for regulator. The three-category pattern favors feature-conditioned retrieval in two categories, not a universal conclusion.
5. **Are improvements present on unqueried events under source exclusion?** Yes, G_NA has higher AUROC than G on all three class-bearing unqueried subsets, and source exclusion remained active. The subset is fixed and diagnostic; historical outcomes were already exposed, so this is not blind future confirmation.
6. **Ranking or operating-point changes?** Both. G_NA raises AUROC in all three class-bearing categories; only switch has a 95% Source Group interval for G_NA−G wholly above zero. It also lowers FNR at the frozen threshold, while FPR and balanced-accuracy effects vary. AUC gain is not a substitute for the threshold ledger.
7. **What remains unsupported?** Query feedback is sparse (67 total, 27 anomalous); audiojack has no anomalies; only four development categories were used; historical results were known; bootstrap intervals condition on recorded trajectories; and CPU state hashes differ from the historical execution path despite passing score/update numerical checks. There is no Static+Correction arm, so complete Greedy/Correction synergy is not established. No claim is made that all normal-memory methods are incapable of these repairs.

## Run history and resource bounds

Every retry used a new namespace; no incomplete run was resumed or overwritten, and no automatic restart occurred.

|Namespace|Phase/result|Retained evidence and note|
|---|---|---|
|r2_feedback_repair_20261102_a1|Failed before supervisor|JSONL append attempted before first-row creation; manifest retained; no child, score, or outcome evaluation. Failure metadata added; immutable.|
|r2_feedback_repair_20261102_a2|Failed before supervisor|Direct child import lacked repository root on sys.path; manifest retained; no child, score, or outcome evaluation. Failure metadata added; immutable.|
|r2_feedback_repair_20261102_a3|Stage A stopped at switch:35|36 score rows retained; current queried record was read only after its score append, then its diagnostic identity was rejected because the schema omits event_id. No Stage B evaluation. Immutable.|
|r2_feedback_repair_20261102_a4|Full Stage A + Stage B complete; 201.406 s|All scores sealed and metrics computed. A later report review found the category-bank telemetry summary was overwritten with the final category counters; original namespace left unchanged.|
|r2_feedback_repair_20261102_a5|Stage B reuse stopped before inputs|Sealed A4 scores copied into a fresh namespace; the guard rejected PyTorch's availability indicator. A separate CPU probe showed `is_available=True` with `device_count=0`; the corrected guard checks visible device count and CUDA-context initialization. The failed run stopped before query/evaluator inputs; immutable.|
|r2_feedback_repair_20261102_a6|Stage B-only reuse complete; 30.172 s|Reused A4 sealed scores; post-seal query-update replay corrected category-specific bank telemetry. Scientific metrics are byte-for-byte equal to A4 results; only reporting/accounting metadata differs.|

Final A4 full run: Stage A 143.485s; original Stage B 50.484s; supervised wall time 201.406s, within the 600s Job Object deadline (remaining time was parent/supervisor overhead). Final A6 Stage B-only reconciliation: 30.172s, also within 600s. A6 reused Stage A algorithm SHA `0501200ba92c38fd556b7acb39fc7299a845ec7cc460201777cebd9a8120a36c` and score SHA `98f7bd26218336bed38923674f5b4551d259904c1df059728285e67642d55632` after checking the five native Stage A dependency fingerprints; Stage B/report code differences are recorded in the reuse contract.

A6 output usage: 1,640,782 bytes before the complete seal, 10 files after completion; the run is bounded at 67,108,864 bytes / 20 files with 65,536 bytes reserved for failure metadata. No output namespace exceeded the bound. This is an application-level pre-write bound; no OS-level filesystem quota was used.

## Tests, provenance, and changed files

- `py_compile` passed for the runner and focused test module.
- Focused CPU tests: **18 passed, 0 failed, 0 skipped**; pytest completed in 6.538s inside the 180s Windows Job Object supervisor. The tests cover native Gaussian parity, independent correction banks, class-conditional updates, score-before-update, source exclusion, geometry-free control, fixed-denominator decomposition, dense-map maxima, ledger conservation, ties/cross-group denominators, unqueried subsets, evaluator separation, output budget, versioned Stage A reuse, CPU containment, and process-tree termination.
- No unrelated repository-wide tests were run. No GPU, DINO, image decode, mask access, parameter sweep, new category, or new feature extraction occurred.

Execution branch: `calibration-stage0-contract-20261029`. HEAD at execution: `cae6cb14822c9e41d697f7cc3d086e717d68b071`. Python: `cpu` via the specified Anaconda interpreter. Stage A algorithm SHA-256: `0501200ba92c38fd556b7acb39fc7299a845ec7cc460201777cebd9a8120a36c`. Stage A seal SHA-256: `b2d8f5567d95290432f685194b7b9b884697f89ca3dc7d7e18bb1a0bbc81caa4`. Stage B seal SHA-256: `0472e4f6dab454fde4fc7d3d450c8710f9a99e4100dee9bfdd804458adfb1f79`. Final aggregate SHA-256: `eb96fa3ca4580441245648c84c260cba0799acb2d692da3053db38a388d77885`.

|Source|SHA-256|
|---|---|
|experiments/calibration/r2_feedback_repair.py|92fe290c4c7ea5cbb4da53b925b97ae58b8fa6c66e0ab98ca86895bee48fb0f2|
|tests/calibration/test_r2_feedback_repair.py|c9d37130b43576dc40bc491f25ea118c22f2a7a526917962c5c9fdf3a200f997|
|experiments/calibration/r1_matched_correction_pilot.py|3208e2c52277ad76351eb86968e362aa315effe9e3c734bfe2b6ff7bab0db448|
|experiments/calibration/r2_exp2a_c1_strength.py|4b798dcb1472e6dd630f58b3e62f3782dfb302f7579e942fcdb399885a5f33d3|
|experiments/calibration/r2_exp2b_memory_readout.py|13e84ec8e2d84816c9bb0e9224c04f881ce0a2ec22dc4b4fdfcfe6e5cc943deb|
|test_phase3_AIF_simplified_v22.py|bfc871d5a9a32b2f5bdb9d7aab5177e702dec2e1110a6632f20bd13c20fcba30|
|vmb_visionad_new.py|bd6545bf5ff63acce01792e5d42aa964c88e73fb59670febeb55f2d1139a409a|

The A6 run manifest records the runner SHA before final diff-check normalization as `e6f2976f752fdf8301039a6396aee89598e79698a1df7ccb3409c4f6924526f3`. Before staging, only one extra trailing blank line was removed; the final deliverable SHA is the value above. No executable statement changed, and the Stage A algorithm SHA is unchanged.

Input receipts: EXP2A experiment seal `8a93dc52b1d26f660241ac85fc2132ac83678873d358ec96190f283ebf1384c0`; EXP2A Stage A seal `b23f552e4c18cccba12d52e3a978f2fe6973b03b58704d44fe0295ad8f5b74c1`. Cached score-row SHA-256 `d6c59dfef488d2a0a9becea6f77713c75bd60390d19b61e3b9509b93a7e0dfec`; transition-row SHA-256 `8bba12ee2ff0f38d7620ccf784e95ba2734768ad8d64a7a7530b63a5157099dc`; queried-feedback SHA-256 `bea1a17f1d90308d82199392e7dbd2f2df0035605a869897bde1f0cdc545a3a2`; strength-diagnostic SHA-256 `820f80f0876f871cc5714e9f584a90a239df8ba037a9e413c93a24bc896d6b65`. The optional EXP2B Stage A reference was `SEALED_COMPLETE` and its receipt matched; it was not used to select or tune results. The authorized frozen evaluator metadata SHA-256 was `48b80202e4b5e02a85b8a950431b386199ee59046d220956062eef25804d0166`. The unchanged `stage05_runner.py` helper was pinned at execution commit HEAD and has SHA-256 `1d110e1ca813a064767ebeab597b609b3d863f8afd4c8085b06948ad88347e14`.

Deliverables are the runner, focused tests, this report, and the aggregate-only JSON. The final Git commit is local only; no push was performed.

## Aggregate artifact privacy

`aggregate_results.json` contains category-level counts, metrics, confidence intervals, source-group bootstrap count, and bank statistics only. It contains no raw labels, event IDs, Source Group IDs, paths, feature arrays, masks, images, or private feedback rows.
