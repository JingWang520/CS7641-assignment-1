# CS7641-assignment-1

This project contains the code for CS7641-assignment-1, including two experiments: the heart disease dataset experiment and the mobile device dataset experiment. Below is the directory structure and detailed instructions.

## Directory Structure

```
data
    ├── documentation.pdf
    ├── heart_statlog_cleveland_hungary_final.csv
    └── mobile_train.csv
result
    ├── image
src
    ├── heart.py
    ├── heart_add.py
    ├── mobile.py
    └── mobile_add.py
```

- `data/`: Contains datasets and related documentation.
  - `documentation.pdf`: Documentation for the datasets.
  - `heart_statlog_cleveland_hungary_final.csv`: Heart disease dataset.
  - `mobile_train.csv`: Mobile device dataset.
- `result/`: Contains the results of the experiments.
- `src/`: Contains source code.
  - `heart.py`: Main experiment code for the heart disease dataset.
  - `heart_add.py`: Additional experiment code for the heart disease dataset.
  - `mobile.py`: Main experiment code for the mobile device dataset.
  - `mobile_add.py`: Additional experiment code for the mobile device dataset.

## Installation

Before running the code, ensure you have the following Python libraries installed:

- seaborn
- pandas
- numpy
- matplotlib
- scikit-learn

You can install these libraries using the following command:

```bash
pip install seaborn pandas numpy matplotlib scikit-learn
```

## Running the Code

### Heart Disease Dataset Experiment

To run the experiment code for the heart disease dataset, execute:

```bash
cd src
python heart.py
python heart_add.py
```

### Mobile Device Dataset Experiment

To run the experiment code for the mobile device dataset, execute:

```bash
cd src
python mobile.py
python mobile_add.py
```

## Results

The results of the experiments will be saved in the `result/` directory. Please check the respective files for detailed experiment results and analysis.