# Figures for the paper "An empirical model to calculate snow depth from daily snow water equivalent: SWE2HS 1.0"

The code in this repository can be used to reproduce the figures (except of the map of station locations, Figure 2) in the following publication:

Aschauer, J., Michel, A., Jonas, T., & Marty, C. (2023). An empirical model to calculate snow depth from daily snow water equivalent: SWE2HS 1.0. Geoscientific Model Development.

To recreate the figures, clone the repository and install the requirements from `requirements.txt` by running

```
pip install reqirements.txt
```

Additionally, a working latex installation is required because some labels in the figures are rendered by latex.

When everything is set up, the figures can be recreated by running:

```
python create_all_figures.py
```

This will also download the input data used for calibration and validation from https://doi.org/10.16904/envidat.394 and store it in a subfolder. 