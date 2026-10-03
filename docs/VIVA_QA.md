# Viva Q&A

Short, honest answers based on how this project was actually built. Where a result does not exist
yet, the answer says so.

**1. What does the system do, in one sentence?**
It reads a hand-drawn spiral or wave and a recording of how it was drawn, returns a screening label
("lower concern", "higher concern" or "uncertain") plus tremor measurements, and always says it is a
screening aid, not a diagnosis.

**2. Why "screening" and never "diagnosis"?**
Nothing here is clinically validated. The data are 158 people drawing on paper, the shipped image
model is a development model, and the tremor model is trained on simulated data. Claiming a diagnosis
would be wrong and could harm someone, so the disclaimer appears on every result, in the README, the
model card and every API response.

**3. How did you stop data leakage?**
Each subject draws the same template several times, so near-identical drawings are common (246
duplicate groups by perceptual hash). I extracted subject IDs from the filenames and used
`StratifiedGroupKFold` by subject, so no person appears in both training and test. Augmentation is
applied on the fly to training folds only, never saved before splitting, and a test checks folds have
no subject or duplicate overlap. Had IDs been missing, the fallback was grouping by duplicate hash,
documented as a weaker control.

**4. Why optimize for sensitivity?**
For a screening tool a missed case costs more than a false alarm that leads to a clinic visit. On a
validation split only, I choose the threshold with the highest specificity at a target sensitivity of
0.9. The test fold is never used for any tuning. The trade-off is more false alarms, which is why an
"uncertain" band exists near the threshold.

**5. What are your results?**
I have no real accuracy numbers yet. Full 5-fold, 3-seed GPU runs are not done, and every metric in
the model card is marked `TODO(run-on-gpu)`. The `--fast` smoke runs prove the pipeline end to end on
a tiny subset and must not be quoted as performance. I would rather say that than invent a table.

**6. Why many backbones, and how will you compare them honestly?**
ResNet50, EfficientNet-B0, ConvNeXt-Tiny, ViT-Small and DINOv2-Small cover CNN, modern CNN,
supervised transformer and self-supervised transformer families, with a HOG + SVM classical baseline.
With about 158 subjects many differences will be within noise, so I report mean and standard deviation
over folds and seeds, a bootstrap interval on AUC, and a paired bootstrap between the top two models,
stating plainly when the gap is not significant.

**7. What do the ablations test?**
Pretrained versus scratch, augmentation versus none, spiral only versus wave only versus fused
(average probabilities and logistic-regression stacking), full fine-tuning versus a frozen backbone
versus a DINOv2 linear probe, and a learning curve at 25, 50 and 100 percent of the data. Fusion is
reported even if it does not help.

**8. What is the shortcut check and what did it find?**
It erases the stroke pixels and fills them with the local background, then trains on the background
only. If that beats chance, the dataset leaks paper or scanner cues. I also compare against a
skeleton re-render that removes stroke thickness. The only run so far is a tiny `--fast` run whose
confidence intervals are far too wide to conclude anything, so the honest finding is "not yet tested
properly". It must be rerun on the full data.

**9. Domain shift: scans versus canvas drawings?**
This is the biggest threat to any real-world claim. The model learned from scanned paper; the app
captures a canvas drawn with a mouse, finger or stylus. I reduce the gap with shared preprocessing
(crop to the drawing, aspect-preserving resize, fixed stroke look on the canvas) and an optional
skeleton re-render, but I have not measured performance on canvas drawings, so I state it as a
limitation.

**10. Why is there one shared `preprocess.py`?**
Training and serving must see identical inputs or accuracy silently drops. The backend copies that one
file at build time, and a test feeds the same image through the training path (from a file) and the
serving path (from base64 bytes) and asserts the tensors are identical.

**11. Why ONNX and onnxruntime instead of PyTorch in the backend?**
The free hosting tier has 512 MB of RAM. PyTorch alone is gigabytes on disk and hundreds of MB in
memory. An ONNX model on onnxruntime runs on CPU in a small process; I measured about 300 MB peak for
the whole backend. An export parity test requires PyTorch and ONNX outputs to agree within 1e-4 (they
agree to 6e-8). INT8 quantization was tried and was slower on this CPU, so it is not used.

**12. Why an occlusion map and not Grad-CAM in the app?**
Grad-CAM needs gradients, so it needs PyTorch, which the backend does not have. Occlusion only needs
forward passes: slide a white patch over the image and measure how much the probability drops. It is
slower (65 forward passes, about 1.3 s) but model-agnostic. It is labeled "occlusion map" everywhere.
Grad-CAM is used offline in the research code, with two sanity checks: how much heatmap mass falls on
stroke pixels, and a deletion test.

**13. Do explanation maps prove the model is right?**
No. They show where the score is sensitive, not why it is correct. The offline checks can show a model
attends to the border or background, which would be a red flag, and I report that if it happens. The
current Grad-CAM numbers come from a development run and say nothing about a trained model.

**14. How does the tremor analysis work, and how do you know it is right?**
The stroke is resampled to 100 Hz, smoothed, and turned into speed, acceleration and jerk. Tremor is
measured as the share of power in 4 to 12 Hz and the dominant frequency in 3 to 12 Hz, from Welch
spectra of the velocity signals. I validate it against a synthetic spiral generator where the true
frequency and amplitude are known: recovered frequency stays within 0.5 Hz across the range, and the
band ratio rises with amplitude. That checks the measurement, not any clinical meaning.

**15. Why is the LightGBM model trained on synthetic data, and is it meaningful?**
No labeled recording of strokes with diagnoses was available. So the model is trained on simulated
spirals, label "tremor injected or not", and the UI says "Feature pipeline demo, trained on simulated
data". Its AUC of 0.998 is near 1 by construction. It demonstrates the feature pipeline and exact
per-feature contributions, not a clinical tremor classifier.

**16. How do you get SHAP-style bars without the shap package?**
LightGBM's `predict(..., pred_contrib=True)` returns exact TreeSHAP values per feature plus a bias, and
a test checks they sum to the raw score (error about 4e-14). The `shap` library is used only offline to
draw summary figures, so it is not in the backend image.

**17. What data does the app store, and what about privacy?**
Only features and scores, tied to a random device ID kept in the browser. Raw drawings and raw strokes
are never stored. The About page has a "delete my data" button that removes every row for that device.
There are no accounts. The history chart compares a person only against their own first sessions, not
against population norms, because I have no valid norms.

**18. How is the API made safe and robust?**
Pydantic limits (at most 20,000 points, 2 MB images, valid value ranges), restricted CORS, a request
size limit, per-client rate limiting, one consistent error format, and structured logs. Models load
once at startup; if a file is missing, `/health` says so and the prediction endpoints return 503
instead of crashing.

**19. What would make a mouse recording unreliable, and how do you handle it?**
Mice sample at lower, irregular rates and add their own jitter, which looks like tremor. The feature
code returns warnings for short, sparse or low-rate input instead of failing, the UI warns about mouse
input, and I recommend a stylus or touchscreen. I tested feature stability under sampling jitter and
mouse-level noise with the synthetic generator.

**20. What would you do next with more time or access?**
Run the full GPU cross-validation and the shortcut check properly; evaluate on canvas drawings from
volunteers (and test-retest reliability); obtain a labeled kinematic dataset such as PaHaW to replace
the synthetic LightGBM; calibrate the probabilities; and, only after that, discuss any clinical study
with clinicians. Until then, the claims stay at "screening aid".
