# Machine Learning HW1: Regression

The assignment is in [`docs/机器学习-作业1-回归算法（作业要求）-2026 (1).docx`](docs/机器学习-作业1-回归算法（作业要求）-2026%20(1).docx).
It requires data preparation, at least two regression algorithms, at least two
evaluation metrics, visualization, and an optimization comparison. Keep an
independent test set for final evaluation only.

## Dataset

The candidate dataset is [UCI Lattice-physics (ID 1091)](https://archive.ics.uci.edu/dataset/1091/lattice-physics+(pwr+fuel+assembly+neutronics+simulation+results)),
licensed CC BY 4.0. The downloaded files are `dataset_UCI/raw.csv` (24,000 rows)
and `dataset_UCI/test.csv` (360 rows). Each row has 41 whitespace-separated
numeric values and no header. UCI describes 39 fuel enrichment features and
two continuous outputs, `k-inf` and `PPPF`. The first two columns appear to be
the outputs; confirm their order from the dataset source before implementation.
Choose one output as the regression target and exclude both outputs from the
input features unless the prediction scenario explicitly justifies otherwise.

The dataset does have regression targets. `ucimlrepo.fetch_ucirepo(id=1091)`
currently cannot import it because UCI does not provide a standardized data URL
for this dataset. Use the downloaded files and document the parsing convention.

## Collaboration

GitHub Issues holds the specification and implementation tasks. Claim an
unblocked issue, work on a branch, and submit a pull request linked to it with
the command used to verify the result. Start with [`AGENTS.md`](AGENTS.md) for
the repository's task and documentation conventions.

Status: repository setup is complete; the task specification and implementation
issues have not yet been created.
