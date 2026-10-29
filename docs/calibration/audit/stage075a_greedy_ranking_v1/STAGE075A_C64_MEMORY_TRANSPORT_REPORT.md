# Stage 0.75-A — C64 Memory-Transport Audit

The analysis pairs scores on the same 64 protected training-normal C64 images at every actual dynamic prototype-content change. C64 images remain immutable and never enter the mutable Greedy bank.

Actual content changes: 1190.
Median paired score shift summary: mean=-0.000004, SD=0.000133, min=-0.004604, median=0.000000, max=0.000000, n=1190.
Shift heterogeneity H=Q75−Q25: mean=0.000246, SD=0.002138, min=0.000000, median=0.000000, max=0.055154, n=1190.
Normalized translation versus direct higher-q90 disagreement: mean=0.036976, SD=0.129296, min=0.000000, median=0.000000, max=2.044486, n=1190.
Paired C64 rescoring time in seconds: mean=0.332434, SD=0.060487, min=0.300759, median=0.323632, max=1.075723, n=1190.
Numeric-uniform shifts: 87.
Descriptive 1%, 5%, 10% of pre-update C64 IQR heterogeneity counts: {'0.01': 1139, '0.05': 1161, '0.1': 1172}.
Tie-aware C64 rank reorderings: {'strict_discordant_pair_count': 52782, 'newly_tied_pair_count': 1, 'newly_ordered_pair_count': 1, 'mean_anchor_rank_movement': {'n': 1190, 'mean': 1.2573004201680673, 'sd_sample': 1.7362616570802234, 'min': 0.0, 'median': 0.71875, 'max': 17.53125, 'quantiles_linear': {'0.0': 0.0, '0.25': 0.21875, '0.5': 0.71875, '0.75': 1.59375, '1.0': 17.53125}}}.

The q90 convention is the higher order statistic at index ceil(0.90×63)=57. H uses linearly interpolated quartiles. Median is NumPy's even-sample median. The diagnostic scale floor is 1e-12. Uniformity bands are descriptive sensitivity cutoffs, not B8 hyperparameters.

A scalar threshold translation and direct q90 coincide under a pure translation when the starting threshold is the same higher-q90 order statistic of the pre-update C64 scores. Heterogeneity, rank reorderings, and a starting threshold that does not equal that q90 produce disagreement.

These measurements concern normal C64 scores only. They cannot establish that physical-defect scores underwent the same transformation. B8 remains unimplemented and unvalidated.

Per category and support seed results are in `c64_transport_summary.json`; paired anchor scores are retained only in local raw outputs.
