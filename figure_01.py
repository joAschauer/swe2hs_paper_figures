from copy import copy
import pandas as pd
import swe2hs as jopack
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import matplotlib.ticker as ticker
import numpy as np


def simple_exponential_function_for_one_layer(swe2hs_result):
    R = jopack._default_model_parameters.R
    rho = copy(jopack._default_model_parameters.RHO_NEW)
    layer_swes = swe2hs_result.layer_heights * swe2hs_result.layer_densities / 1000
    swe = layer_swes.sel(layers=1).to_pandas()
    hs = np.zeros(len(swe))
    rho_max = float(swe2hs_result.layer_max_densities.sel(layers=1).isel(time=3))
    for i, s in enumerate(swe.to_numpy()):
        hs[i] = s*1000 / rho
        if s > 0:
            rho = rho_max - (rho_max-rho) * np.exp(-1/R)    
    return pd.Series(hs, index=swe.index), swe


def create_synthetic_swe():
    p1 = 30
    p2 = 60
    swe = [0.,0.] + 5*[p1] + 10*[p2] + [i for i in np.linspace(p2, 0, 19)]
    return pd.Series(swe, index=pd.date_range(start="2020-01-01", periods=len(swe)))


def calculate_models(swe):
    hs_swe2hs = jopack.convert_1d(swe, swe_input_unit='mm', hs_output_unit='cm', return_layers=True)
    hs_swe2hs = hs_swe2hs.assign_coords(time=np.arange(len(hs_swe2hs.time)))
    hs_simple, swe_layer = simple_exponential_function_for_one_layer(hs_swe2hs)
    hs_top = hs_swe2hs.hs.to_pandas()
    hs_swe2hs = hs_swe2hs.where(hs_swe2hs.any(dim = 'time'),drop=True)
    return hs_swe2hs, hs_simple, hs_top


def make_plot(hs_swe2hs, hs_simple, hs_top, swe):
    fig, axs = plt.subplots(2, 1, sharex=True, height_ratios=[5,1], gridspec_kw={'hspace':0.2})
    jopack.visualization.layer_plot(
        axs[0], 
        hs_swe2hs,
        cmap='Blues',
        color_variable=None,
        top_line_kwargs={'lw': 2, },
        layer_line_kwargs={'lw':2},
    )
    axs[0].plot(hs_simple.index, hs_simple, c='k', ls='--', lw=0.8)
    axs[1].plot(hs_simple.index, swe)
    axs[0].fill_between(
        hs_simple.index, 
        hs_swe2hs.layer_heights.sel(layers=1).to_pandas(), 
        hs_simple, 
        facecolor='r', 
        alpha=0.5)

    bottom = axs[0].hlines(y=0, xmin=0, xmax=35, lw=2, colors='k')

    axs[0].fill_between(
        x=hs_simple.index, 
        y1=hs_top,
        y2=hs_simple,
        facecolor='lightgrey', 
        where=hs_top>=hs_simple,
    )

    axs[0].fill_between(
        x=hs_simple.index, 
        y1=hs_swe2hs.layer_heights.sel(layers=1).to_pandas(),
        y2=0,
        facecolor='lightgrey', 
    )

    txt_layer1 = axs[0].text(
        x=8, y=8, s='Layer 1',
        ha='left', va='center',
        transform=axs[0].transData,
        bbox={'boxstyle':'square',
                'fc':'w',
                'ec':'k',
                'alpha':1,
                },
    )

    txt_layer2 = axs[0].text(
        x=11, y=23, s='Layer 2',
        ha='left', va='center',
        transform=axs[0].transData,
        bbox={'boxstyle':'square',
                'fc':'w',
                'ec':'k',
                'alpha':1,
                },
    )

    for p, l in zip([1,6,17,26],('(a)','(b)','(c)','(d)')):
        con = ConnectionPatch(
            xyA=(p, hs_top[p]), coordsA=axs[0].transData,
            xyB=(p, swe[p]), coordsB=axs[1].transData,
            arrowstyle="-", shrinkB=0)
        
        fig.add_artist(con)
        
        txt = fig.text(
            x=p, y=-4.5, s=l,
            ha='center', va='center',
            transform=axs[0].transData,
            bbox={'boxstyle':'square',
                'fc':'w',
                'ec':'w',
                'alpha':1,
                },
        )
        txt.set_zorder(100000)
        con.set_zorder(1)

    txt_layer1.set_zorder(100000)    

    for ax in axs:
        ax.spines[['bottom', "top", "right"]].set_visible(False)

    axs[0].xaxis.set_visible(False)
    axs[1].spines[['bottom']].set_visible(True)
    axs[0].set_ylabel('HS [cm]')
    axs[1].set_ylabel('SWE [mm]')
    axs[1].set_xlabel('n days')
    axs[1].set_xticks(hs_simple.index)
    axs[1].xaxis.set_major_locator(ticker.MultipleLocator(5))
    axs[1].xaxis.set_minor_locator(ticker.MultipleLocator(1))
    axs[1].set_xlim(-0.5, 35)
    axs[0].set_ylim(-0.5, 60)
    return fig


def main(filepath):
    print(f"creating figure_01 and saving to\n {filepath}")
    swe = create_synthetic_swe()
    hs_swe2hs, hs_simple, hs_top = calculate_models(swe)
    fig = make_plot(hs_swe2hs, hs_simple, hs_top, swe)
    fig.savefig(filepath, bbox_inches='tight', dpi=300)