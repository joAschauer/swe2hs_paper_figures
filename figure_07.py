"""
The main function of this module creates plot for Figure 7.
"""

from pathlib import Path
import pickle
from types import MethodType

import matplotlib.pyplot as plt
import pandas as pd

from SALib.analyze import sobol

DATA_DIR = DATA_DIR = Path(__file__).parent.resolve() / 'data'

parameter_labels = {
    'rho_max_dry': r'$\rho_{max,init}$',
    'rho_max_wet': r'$\rho_{max,end}$',
    'rho_new': r'$\rho_{new}$',
    'R' : r'$R$',
    'max_overburden': r'$\sigma_{max}$',
    'wetting_speed': r'$v_{melt}$',
}

def nested_barplot(vars, Sis):
    var_labels = {
        'rmse': 'RMSE',
        'bias': 'BIAS',
        'r2': '$R^2$'
    }
    # sort variables based on senistivity for first variable:
    sorting = Sis[vars[0]].to_df()[0].sort_values(by='ST', ascending=False).index
    total_indices = []
    confidences = []
    for variable in vars:
        si_df = (
            Sis[variable]
            .to_df()[0]
            .reindex(sorting)
            .rename(index={c: m for c, m in parameter_labels.items()})
        )
        total_indices.append(
            si_df['ST'].rename(var_labels[variable])
        )
        confidences.append(
            si_df['ST_conf'].rename(var_labels[variable])
        )

    total_indices = pd.concat(total_indices, axis=1)
    confidences = pd.concat(confidences, axis=1)
    colors = ['dodgerblue','lightblue', 'lightcyan']
    fig, ax = plt.subplots(1, 1,figsize=[5, 3.3], sharey=True, sharex=True)
    total_indices.plot.bar(
        yerr=confidences,
        ax=ax,
        color=colors[:len(vars)],
        edgecolor='dimgrey',
        width=0.7,
    )
    
    ax.tick_params(bottom=False, left=True, right=True, top=False, direction='in')
    ax.set_ylabel(r'$S_{Ti}$', fontsize=13, usetex=True)

    x_labs = ax.get_xticklabels()
    ax.set_xticklabels(
        x_labs, fontsize=13, usetex=True, rotation=0, va='baseline', ha='center'
    )
    ax.tick_params(axis='x', pad=15)
    ax.legend(frameon=False)
    return fig

def main(filepath):
    print(f"- creating figure_06 and saving to\n  {filepath}")
    with open(DATA_DIR / 'sensitivity_indices_rmse_bias_r2.pkl', 'rb') as f:
        Sis = pickle.load(f)

    for i, variable in enumerate(['rmse', 'bias', 'r2']):
        Sis[variable].to_df = MethodType(sobol.to_df, Sis[variable]) # overriding `to_df`method for Sobol

    fig = nested_barplot(['r2', 'bias'], Sis)
    fig.savefig(filepath, dpi=300, bbox_inches='tight', pad_inches=0.05)
    fig.clf()
    plt.close()
    return None
    

