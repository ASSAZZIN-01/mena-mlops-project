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

## Project Vertex experiment

The project now contains a reproducible Vertex AI experiment using the shared
CUDA training image:

- Vertex job: `projects/922483082863/locations/us-central1/customJobs/2307246063360147456`
- MLflow run: `00d77b334f054a388511118f135bad8a`
- Training lineage commit: `2c048d5b1b8b4980ce28424514547053d590b438`
- Configuration: `configs/distillation.yaml`
- Student artifact: `gs://mena-mlops-project-dvc/vertex-staging/distillation-six-layer-20261017-083444/output/student/model.safetensors`
- Student SHA-256: `b33b039f33297c40c6423da357d5b220ed4935ff4a3775f795193da3802a29c0`
- Teacher: 12-layer `models/arabert-debug`
- Student: 6 encoder layers
- Maximum epochs: 5, with validation early stopping
- Alpha: 0.4
- Temperature: 4.0
- Learning rate: `3e-5`

The authoritative local re-evaluation of the downloaded Vertex checkpoint on
the canonical 1,225-row test split produced:

| Representation | Accuracy | Macro-F1 | Neutral F1 |
| --- | ---: | ---: | ---: |
| Distilled PyTorch | 0.7820 | 0.5246 | 0.0000 |
| Distilled ONNX FP32 | 0.7820 | 0.5246 | 0.0000 |
| Distilled ONNX INT8 | 0.7829 | 0.5250 | 0.0000 |

The test split contains 415 negative, 55 neutral, and 755 positive examples.
The student predicted no neutral examples and therefore fails the configured
neutral-class floor. The student also fails the stable macro-F1 floor of
0.55 and does not improve over the stable baseline of 0.5523.

The formal gate decision is intentionally recorded as rejected:

- Selected representation: distilled ONNX INT8.
- Optimization gate: accepted.
- Quality gate: rejected.
- Deployment baseline gate: rejected because this benchmark candidate has no
  deployed baseline comparison.
- No registry promotion, candidate deployment, canary, or rollback was
  performed for this student.

The Vertex run now records each DVC split as a first-class MLflow dataset
input, with the split SHA-256, DVC path, DVC lock checksum, and source commit.
The student model remains available under the run's `student-model` artifact
path. It is deliberately not a Model Registry version because the quality
gate rejected it; the run tag `candidate_registry_status` records this
decision rather than presenting a rejected benchmark as deployable.

The Vertex report showed a zero-row neutral class because of the worker-side
evaluation output; the downloaded checkpoint was re-evaluated against the
canonical local split, which confirmed the actual 55 neutral test rows and is
the result used for the promotion decision.
