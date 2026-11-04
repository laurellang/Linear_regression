# Machine Learning HW1: Regression

The assignment is in [`docs/机器学习-作业1-回归算法（作业要求）-2026 (1).docx`](docs/机器学习-作业1-回归算法（作业要求）-2026%20(1).docx).
It requires data preparation, at least two regression algorithms, at least two
evaluation metrics, visualization, and an optimization comparison. Keep an
independent test set for final evaluation only.

## Dataset

The selected dataset is [UCI Lattice-physics (ID 1091)](https://archive.ics.uci.edu/dataset/1091/lattice-physics+(pwr+fuel+assembly+neutronics+simulation+results)),
licensed CC BY 4.0. The downloaded files are `dataset_UCI/raw.csv` (24,000 rows)
and `dataset_UCI/test.csv` (360 rows). Each row has 41 whitespace-separated
numeric values and no header. UCI describes 39 fuel enrichment features and
two continuous outputs, `k-inf` and `PPPF`. The working column convention is
column 1 = `k-inf`, column 2 = `PPPF`, and columns 3-41 = enrichment features
(1-based indexing). This is strongly supported by the value ranges, but UCI
does not publish an explicit file schema. The regression target is `k-inf`;
exclude both output columns from the input features.

Split `raw.csv` into 80% training and 20% validation with seed 42. Use the
same split for every model. Reserve `test.csv` for one final evaluation after
model choices and parameters are fixed. Report MAE, RMSE, and R².

The dataset does have regression targets. `ucimlrepo.fetch_ucirepo(id=1091)`
currently cannot import it because UCI does not provide a standardized data URL
for this dataset. Use the downloaded files and document the parsing convention.

## Collaboration

GitHub Issues holds the specification and implementation tasks. Claim an
unblocked issue, then commit directly to `main`. Each contributor records the
command used to verify their experiment and summarizes its results in the
issue. The shared framework contains data, evaluation, plotting, result-saving,
and command-line utilities; model and robustness experiments live in separate
runnable files. Start with [`AGENTS.md`](AGENTS.md) for repository conventions.

Status: repository setup is complete; the task specification and implementation
issues have not yet been created.
