# Shortcut check

Trains on the stroke erased and inpainted with the surrounding
background, so a model can only use background or scanner cues,
never the drawing itself. Compared against normal preprocessing
and the skeleton_rerender mode, which strips stroke-thickness and
pressure cues while keeping the drawing's shape.

- normal: mean ROC-AUC 0.600 (mean CI lower bound 0.100), 2 folds
- background_only: mean ROC-AUC 0.100 (mean CI lower bound 0.000), 2 folds
- skeleton_rerender: mean ROC-AUC 0.900 (mean CI lower bound 0.700), 2 folds

## Finding: no shortcut detected

The background-only model's AUC confidence interval does
not clearly exceed chance (0.5). No evidence that the normal
model is relying on paper or scanner background cues rather
than the drawing.