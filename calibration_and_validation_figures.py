from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd
import numpy as np

import swe2hs as jopack
from swe2hs.visualization import layer_plot, groupby_hydroyear


DATA_DIR = Path(__file__).parent.resolve() / '.data_cache'

def drop_allzero_swe_chunks(df):
    """
    Trim the dataset for faster computation.
    """
    dfi = df.copy()
    swe = dfi['SWE_[m]'].to_numpy()
    to_drop = (swe == 0) & (np.roll(swe, -1) == 0) & (np.roll(swe, 1) == 0)
    return dfi[~to_drop]

DATA_MANUAL_CALIB = (pd
    .read_csv(
        DATA_DIR / 'manual_stations_calibration_data.csv',
        parse_dates=['date'],
        index_col='date')
    .pipe(drop_allzero_swe_chunks)
    )

DATA_MANUAL_VALID = (pd
    .read_csv(
        DATA_DIR / 'manual_stations_validation_data.csv',
        parse_dates=['date'],
        index_col='date')
    .pipe(drop_allzero_swe_chunks)
    )

DATA_AUTOMATIC_VALID = (pd
    .read_csv(
        DATA_DIR / 'automatic_stations_validation_data.csv',
        parse_dates=['date'],
        index_col='date')
    .pipe(drop_allzero_swe_chunks)
    )

m = {k: v for k, v in vars(jopack._default_model_parameters).items() if not k.startswith('_')}
MODEL_PARAMS = {k.lower(): v for k, v in m.items() if not k=='R'}
MODEL_PARAMS['R'] = m['R']


def predict_with_params(swe_data, params, return_layers, hs_output_unit='m'):
    result = jopack.convert_1d(
        swe_data,
        ignore_zeropadded_gaps=True,
        interpolate_small_gaps=True,
        max_gap_length=3,
        return_layers=return_layers,
        hs_output_unit=hs_output_unit,
        **params)
    return result


def layer_plots_6_example_years(
    data,
    params,
    stationyears,
):
    """
    data : pd.DataFrame
        data source
    params : dict
        model_parameters
    stationyears : list of tuples of len=6
        [(stat_name, hyear), (stat_name, hyear), ...]
    """
    def align_axis_x(ax, ax_target):
        """Make x-axis of `ax` aligned with `ax_target` in figure"""
        posn_old, posn_target = ax.get_position(), ax_target.get_position()
        ax.set_position([posn_target.x0, posn_old.y0,
                        posn_target.width, posn_old.height])
        return None

    def inset_colorbar(
        ax,
        vmin,
        vmax,
        cmap,
        cbar_label,
    ):
        color_norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
        color_mapper = mpl.cm.ScalarMappable(norm=color_norm, cmap=cmap)
        axins = inset_axes(
            ax,
            width="3%",  # width: 5% of parent_bbox width
            height="65%",  # height: 50%
            loc="upper left",
            bbox_to_anchor=(0.77, -0.05, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )
        cbar = plt.colorbar(color_mapper, axins, ax)
        cbar.set_label(cbar_label)
        return cbar
    
    station_lookup = pd.read_csv(DATA_DIR / 'coordinates_swe2hs_paper_stations.csv')
    
    station_descr = {
        'CDP_aws': ['Col de Porte', 1325],
        'DAV_aws': ['Davos', 1563],
        'FEL_aws': ['Fellhorn', 1610],
        'KUR_aws': ['Kühroint', 1420],
        'KUT_aws': ['Kühtai', 1920],
        'SPI_aws': ['Spitzingsee', 1100],
        'WAL_aws': ['Wattener Lizum', 1994],
        'WFJ_aws': ['Weissfluhjoch', 2536],
        'ZUG_aws': ['Zugspitze', 2420],
        'LAR_aws': ['Laret', 1513]
    }

    fig, ((ax01, ax31),
          (ax02, ax32),
          (dummy1, dummy3),
          (ax11, ax41),
          (ax12, ax42),
          (dummy2, dummy4),
          (ax21, ax51),
          (ax22, ax52)) = plt.subplots(
        8, 2, figsize=[10, 15],
        gridspec_kw={'height_ratios': [
            4, 1, 0.3, 4, 1, 0.3, 4, 1], 'hspace': 0.03, 'wspace': 0.05},
        sharex=False,
        sharey=False,
    )
    panel_axes = {
        0: [ax01, ax02],
        1: [ax11, ax12],
        2: [ax21, ax22],
        3: [ax31, ax32],
        4: [ax41, ax42],
        5: [ax51, ax52],
    }
    parallel_axes = [
        (ax01, ax31), 
        (ax02, ax32),
        (ax11, ax41), 
        (ax12, ax42),
        (ax21, ax51), 
        (ax22, ax52),
    ]
    dummy1.remove()
    dummy2.remove()
    dummy3.remove()
    dummy4.remove()
    panel = 0
    
    for stnyear in stationyears:
        stn = stnyear[0]
        year = stnyear[1]
        for s, stn_df in data.groupby('site_id'):
            if s ==stn:
                result = predict_with_params(
                    stn_df['SWE_[m]'],
                    params,
                    return_layers=True,
                    hs_output_unit='cm',
                )
                result['HS_measured'] = ('time', stn_df['HS_[m]'].to_numpy())
                result['SWE_input'] = ('time', stn_df['SWE_[m]'].to_numpy())

                for wy, wy_ds in groupby_hydroyear(result):
                    if wy == year:
                        for ax in panel_axes[panel]:
                            ax.xaxis.set_major_locator(
                                mdates.MonthLocator(bymonth=(1, 3, 5, 7, 9, 11)))
                            ax.xaxis.set_minor_locator(mdates.MonthLocator())
                            formatter = mdates.ConciseDateFormatter(
                                ax.xaxis.get_major_locator())
                            ax.xaxis.set_major_formatter(formatter)
                        if panel < 3:
                            colorbar=False
                        else:
                            colorbar=False
                        layer_plot(
                            panel_axes[panel][0],
                            wy_ds,
                            color_variable='layer_densities',
                            cmap='Blues',
                            vmin=50,
                            vmax=550,
                            cbar_label='Density [kg m$^{-3}$]',
                            top_line_kwargs={'lw': 1},
                            layer_line_kwargs={'lw': 0.25},
                            colorbar=colorbar,
                        )
                        panel_axes[panel][0].plot(
                            wy_ds['time'].to_index(),
                            wy_ds['HS_measured'].to_pandas()*100,
                            ls='--', c='r', lw=1, label='HS measured'
                        )
                        panel_axes[panel][1].plot(
                            wy_ds['time'].to_index(),
                            wy_ds['SWE_input'].to_pandas()*1000,
                            ls='-', c='b', lw=1, label='SWE measured'
                        )

                        # Station name and altitude in subplot:
                        try:
                            station_label = f'{station_descr[stn][0]}, {station_descr[stn][1]} m a.s.l.'
                        except KeyError:
                            altitude = int((station_lookup
                                .loc[station_lookup['site_id']==stn, 'elevation_[m]']
                                .iloc[0]
                            ))
                            name = str((station_lookup
                                .loc[station_lookup['site_id']==stn, 'location_name']
                                .iloc[0]
                            ))
                            station_label =f'{name}, {altitude} m a.s.l.'
                        
                        panel_axes[panel][0].text(
                            0.03, 0.95,
                            station_label,
                            fontsize=10, ha='left', va='top',
                            transform=panel_axes[panel][0].transAxes)
                        if panel == 0:
                            panel_axes[panel][0].legend(bbox_to_anchor=(
                                0.003, 0.9), loc='upper left', frameon=False)
                            panel_axes[panel][1].legend(
                                loc='upper left', frameon=False)
                        if panel < 3:
                            panel_axes[panel][0].set_ylabel('HS [cm]')
                            panel_axes[panel][1].set_ylabel('SWE [mm]')
                        
                        if panel < 3:
                            labelleft = True
                        else:
                            labelleft = False
                        panel_axes[panel][0].tick_params(
                            which='both', bottom=True, left=True, right=True, top=False,
                            direction='in', labelbottom=False, labelleft=labelleft
                        )
                        panel_axes[panel][1].tick_params(
                            which='both', bottom=True, left=True, right=True, top=True, 
                            direction='in', labelleft=labelleft
                        )
                        if panel < 3:
                            align_axis_x(panel_axes[panel][1], ax01)
                        else:
                            align_axis_x(panel_axes[panel][1], ax31)
                        panel += 1
    
    ax31.set_ylim(ax31.get_ylim()[0], 280)
    # alignig ylims:
    for axs in parallel_axes:
        ymin = min([ax.get_ylim()[0] for ax in axs])
        ymax = max([ax.get_ylim()[1] for ax in axs])
        for ax in axs:
            ax.set_ylim((ymin, ymax))
    
    # alignig xlims:
    for axs in panel_axes.values():
        xmin = min([ax.get_xlim()[0] for ax in axs])
        xmax = max([ax.get_xlim()[1] for ax in axs])
        for ax in axs:
            ax.set_xlim((xmin, xmax))

    cbar = inset_colorbar(
        ax=ax31,
        vmin=50,
        vmax=550,
        cmap='Blues',
        cbar_label='Density [kg m$^{-3}$]',
        )

    return fig


def figure_03(filepath):
    print(f"- creating figure_03 and saving to\n  {filepath}")
    fig = layer_plots_6_example_years(
        data=DATA_AUTOMATIC_VALID,
        params=MODEL_PARAMS,
        stationyears=[
            ('WFJ_aws', 2014),
            ('KUR_aws', 2021),
            ('CDP_aws', 2007),
            ('FEL_aws', 2013),
            ('KUT_aws', 2002),
            ('WAL_aws', 2014),
        ],
    )
    fig.savefig(filepath, dpi=300,
                bbox_inches='tight', pad_inches=0.05)
    fig.clf()
    plt.close()
    return None


def figure_A01(filepath):
    print(f"- creating figure_A01 and saving to\n  {filepath}")
    fig = layer_plots_6_example_years(
        data=DATA_MANUAL_VALID,
        params=MODEL_PARAMS,
        stationyears=[
            ('7MA', 1994),
            ('7ST', 1971),
            ('7MA', 2017),
            ('7ZU', 2001),
            ('5SP', 2019),
            ('2ST', 1990),
        ],
    )
    fig.savefig(filepath, dpi=300,
                bbox_inches='tight', pad_inches=0.05)
    fig.clf()
    plt.close()
    return None


def figure_A02(filepath):
    print(f"- creating figure_A02 and saving to\n  {filepath}")
    fig = layer_plots_6_example_years(
        data=DATA_MANUAL_CALIB,
        params=MODEL_PARAMS,
        stationyears=[
            ('1GT', 1999),
            ('5AR', 1999),
            ('1GB', 1988),
            ('1AD', 2000),
            ('4MS', 2006),
            ('1LS', 1970),
        ],
    )
    fig.savefig(filepath, dpi=300,
                bbox_inches='tight', pad_inches=0.05)
    fig.clf()
    plt.close()
    return None


