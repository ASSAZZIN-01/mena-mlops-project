# AraBERT distillation evidence

The course distillation evidence uses a 12-layer AraBERT teacher and a
6-layer student. The student keeps the teacher tokenizer/vocabulary and
rebuilds the encoder configuration with `student_layers: 6`. The distillation
objective combines supervised cross-entropy with temperature-scaled teacher
KL divergence.

## Reproducible configuration

The reference implementation and configuration are committed in the Raay
course reference repository:

- Configuration: `configs/distill.yaml`
- Teacher: `models/baseline/final`
- Student layer count: `6`
- Evaluation report: `reports/eval_distilled.json`
- Distillation evaluation commit: `15d168d`

The construction logic is implemented in
`src/raay/training/distill_models.py`, where the student configuration sets
`num_hidden_layers` to the configured six-layer value.

## Recorded evaluation

The committed evaluation report records:

| Metric | Student result |
| --- | ---: |
| Accuracy | 0.8303509502 |
| Macro-F1 | 0.5974942692 |
| Weighted-F1 | 0.8144238819 |
| Evaluation rows | 7,209 |

The exact student checkpoint path and its artifact checksum must be attached
to the final submission bundle. Large model binaries remain external/DVC
artifacts rather than being duplicated in this repository.
