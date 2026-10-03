# Explainability check

Development run (--fast): the model was trained for one epoch from random initialization on a tiny sample, so the numbers below only prove the pipeline runs. They say nothing about what a trained model attends to.

Explained 5 test images with Grad-CAM.

- Mean heatmap mass on stroke pixels relative to the strokes' area share: 1.02 (1.0 means no preference for strokes over blank paper).
- Mean heatmap mass in the outer 10% border margin: 0.37.
- Share of heatmaps that fell faster than a random deletion order: 1.00.

Finding: on average the model attends to stroke pixels at least in proportion to their area.
