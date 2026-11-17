NumCompute
A lightweight scientific computing toolkit built with Python and NumPy. NumCompute includes common numerical operations, data preprocessing, evaluation metrics, and a decision tree classifier that can learn from incoming batches of data.

The project was developed to explore how numerical algorithms and machine learning models work internally, without relying on high-level machine learning libraries.
Features
- Data handling: Read CSV files and process data in chunks.
- Preprocessing: Standard scaling, min-max scaling, missing-value imputation, and one-hot encoding.
- Numerical utilities: Sorting, top-k selection, ranking, percentiles, descriptive statistics, and finite-difference gradients.
- Evaluation: Classification and regression metrics, including streaming updates.
- Decision tree: Classification using Gini impurity or entropy, with configurable depth and buffered retraining.
- Streaming pipeline: Update preprocessing and model components as new data arrives.
- Visualisation: Plot accuracy over time, compare models, and view predictions against actual labels.
- Testing and benchmarks: Automated tests and performance comparisons.
Requirements
- Python 3.9 or newer
- NumPy
- matplotlib
No pandas or scikit-learn is required. Python's standard library is used for file handling, testing, and other utilities.

Project structure (WBS)
NumCompute/
├── numcompute/
│   ├── __init__.py
│   ├── io.py
│   ├── preprocessing.py
│   ├── sort_search.py
│   ├── rank.py
│   ├── stats.py
│   ├── metrics.py
│   ├── optim.py
│   ├── pipeline.py
│   ├── utils.py
│   ├── benchmarking.py
│   ├── tree.py
│   ├── stream.py
│   └── visualise.py
├── data/
│   ├── make_data.py
│   └── stream_data.csv
├── demo/
│   ├── quickstart.ipynb
│   ├── stream_demo.ipynb
│   ├── run_spyder.py
│   └── benchmarks.py
├── tests/
│   └── test_numcompute.py
├── results/
│   ├── accuracy.png
│   ├── predictions.png
│   └── buffer_comparison.png
├── pyproject.toml
└── README.md
