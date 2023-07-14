from pathlib import Path
import calendar

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
from sklearn.metrics import r2_score

import pandas as pd
import numpy as np

import swe2hs as jopack
from swe2hs.visualization import layer_plot, groupby_hydroyear


DATA_DIR = Path(__file__).parent.resolve() / "data" / ".model_input_data_cache"

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


def _nonzero_scorer(scorefunc):
    """
    Only calculate score where at least one of y_true or y_pred is nonzero.
    
    Can be used to decorate a scoring function which accepts two arguments
    y_true and y_predicted.
    """
    def _removed_zero_score(*args):
        data = pd.DataFrame({'y_true': args[0],
                             'y_pred': args[1]}
                            )
        data = data.loc[data.any(axis=1)]
        y_true = data['y_true'].to_numpy()
        y_pred = data['y_pred'].to_numpy()
        return scorefunc(y_true, y_pred)
    return _removed_zero_score


@_nonzero_scorer
def _bias_score(y_true, y_hat):
    
    assert(len(y_true) == len(y_hat))
    error = y_hat-y_true
    return np.average(error)


@_nonzero_scorer
def _r2_nonzero(y_true, y_pred):
    return r2_score(y_true, y_pred)


def _get_month_abbr(month_int):
    return calendar.month_abbr[month_int]


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
    
    # plain readable text names for stations in AWS dataset.
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
                bbox_inches='tight', 
                # pad_inches=0.05,
                )
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
                bbox_inches='tight', 
                # pad_inches=0.05,
                )
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
                bbox_inches='tight', 
                # pad_inches=0.05,
                )
    fig.clf()
    plt.close()
    return None


def scatterplot_true_pred(
    ax, 
    true,
    pred,
    style='scatter',
    fitline=True,
    fitline_kwargs=None,
    identity_line=True,
    identity_line_kwargs=None,
    **plot_kwargs
):
    """
    Scatterplot for modeled vs. measured values.

    Parameters
    ----------
    ax : mpl.axes.Axes
        Axes which is filled by the function.
    true : array, pd.Series or list like
        true values (will be plotted on x-axis).
    pred : array, pd.Series or list like
        predicted values (will be plotted on y-axis).

    Returns
    -------
    ax : mpl.axes.Axes

    """
    plot_kwargs.setdefault('s', 1)
    plot_kwargs.setdefault('marker', 'o')
    plot_kwargs.setdefault('facecolor', 'k')
    plot_kwargs.setdefault('lw', 0)
    
    data = pd.DataFrame({'true': true, 'pred':pred}, index=true.index)
    data = data.dropna()
    
    true = data['true']
    pred = data['pred']
    gmin = min([true.min(), pred.min()])
    gmax = max([true.max(), pred.max()])
    
    if style=='scatter':
        sns.scatterplot(
            x=true,
            y=pred,
            ax=ax,
            **plot_kwargs,
            )
    elif style=='density':
        data = data.reset_index(drop=True)
        sns.histplot(
            data=data,
            x='true',
            y='pred',
            binwidth=(gmax-gmin)/100,
            thresh=0,
            pmax=0.3,
            cmap='RdPu',
            ax=ax,
            )
    
    if identity_line:
        i_line_kwargs = {
            'linestyle': '-', 
            'color': 'k', 
            'lw': 0.8
        }
        if identity_line_kwargs is not None:
            i_line_kwargs.update(identity_line_kwargs)
        # 1/1 line limits, extended by 1%
        gmin -= 0.01*gmin
        gmax += 0.01*gmax
        # draw 1/1 line
        ax.axline([gmin, gmin], [gmax, gmax], **i_line_kwargs)

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.set_ylim((min(xmin,ymin),max(xmax,ymax)))
    ax.set_xlim((min(xmin,ymin),max(xmax,ymax)))
    ax.set_aspect('equal')
    ax.set_xlabel(r"measured")
    ax.set_ylabel(r"modeled")

    if fitline:
        line_kwargs = {
                'linestyle': '--',
                'color': 'k',
                'lw':0.8,
            }
        if fitline_kwargs is not None:
            line_kwargs.update(fitline_kwargs)
        # linear fit to the scatterplot:
        # obtain m (slope) and b(intercept) of linear regression line
        m2, b2 = np.polyfit(true, pred, 1)
        # new x-vector
        x_fitline = np.linspace(true.min(), true.max())
        #add linear regression line to scatterplot 
        ax.plot(
            x_fitline,
            m2*x_fitline+b2,
            **line_kwargs)
    return ax


def plot_density_inset_scatter(ax, data, params):
    result = predict_with_params(
        data['SWE_[m]'],
        params,
        return_layers=False
    )
    # main plot
    scatterplot_true_pred(
        ax, 
        data['HS_[m]']*100, 
        result*100,
        style='scatter',
        hue=None,
        fitline_kwargs={'color': 'r', 'lw': 1.5}
    )
    # inset density plot
    axins = ax.inset_axes([0.55, 0.02, 0.43, 0.43], transform=ax.transAxes)
    scatterplot_true_pred(axins, data['HS_[m]']*100, result*100, style='density',
                          fitline=False, identity_line_kwargs={'lw': 0.5})
    axins.set(xlabel=None, ylabel=None, xticklabels=[], yticklabels=[])
    ax.set_xlabel(r"HS reference [cm]")
    ax.set_ylabel(r"HS model [cm]")
    return ax, axins


def figure_04(filepath):
    """
    Layout of figure:
    ax0  ax1
    ax2  ax3
    """
    print(f"- creating figure_04 and saving to\n  {filepath}")
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=[8, 8])

    ax0, axins0 = plot_density_inset_scatter(ax0, DATA_MANUAL_CALIB, MODEL_PARAMS)
    ax1.set_axis_off()
    ax2, axins2 = plot_density_inset_scatter(ax2, DATA_MANUAL_VALID, MODEL_PARAMS)
    ax3, axins3 = plot_density_inset_scatter(ax3, DATA_AUTOMATIC_VALID, MODEL_PARAMS)

    ax3.set(ylabel=None, yticklabels=[])
    ax0.set(xlabel=None, xticklabels=[])
    ax0.text(0.05, 0.95, '(a)', fontsize='x-large',
             ha='left', va='top', transform=ax0.transAxes)
    ax2.text(0.05, 0.95, '(b)', fontsize='x-large',
             ha='left', va='top', transform=ax2.transAxes)
    ax3.text(0.05, 0.95, '(c)', fontsize='x-large',
             ha='left', va='top', transform=ax3.transAxes)

    axs = [ax0, axins0, ax2, axins2, ax3, axins3]
    gmin = min([min([ax.get_ylim()[0] for ax in axs]),
               min([ax.get_xlim()[0] for ax in axs])])
    gmax = max([max([ax.get_ylim()[1] for ax in axs]),
               max([ax.get_xlim()[1] for ax in axs])])
    ticks = list(range(0, 800, 100))
    for ax in axs:
        ax.set_ylim((gmin, gmax))
        ax.set_xlim((gmin, gmax))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.tick_params(bottom=True, left=True, right=True,
                       top=True, direction='in')
        ax.set_aspect('equal')

    plt.tight_layout()
    fig.savefig(filepath, dpi=300,
                bbox_inches='tight', pad_inches=0.05)
    fig.clf()
    plt.close()
    return None


def monthly_comparison_boxplot(ax, true, pred, start_month=9, plot_kwargs=None):
    assert start_month in range(1,13)
    # reshape data to long format:
    true = pd.DataFrame({
        'values': true.values, 
        'source': 'reference', 
        'Month': true.index.month.map(_get_month_abbr)}, 
        index=true.index)
    pred = pd.DataFrame({
        'values': pred.values, 
        'source': 'model', 
        'Month': pred.index.month.map(_get_month_abbr)}, 
        index=pred.index)
    # arrange data in long format
    df = pd.concat([true, pred], axis=0)
    
    pkwargs = {
        'palette': 'Set2',
    }
    if plot_kwargs is not None:
        pkwargs.update(plot_kwargs)
    sns.boxplot(
        x='Month',
        y='values',
        hue='source',
        data=df,
        ax=ax,
        **pkwargs
    )
    return ax


def calculate_swe2hs_on_data_and_plot_boxes(
    ax,
    data,
    params,
):
    result = predict_with_params(
        data['SWE_[m]'],
        params,
        return_layers=False
    )
    kwargs = {
        'palette': ['lightblue', 'dodgerblue'],
        'order': [_get_month_abbr(i) for i in [10, 11, 12, 1, 2, 3, 4, 5, 6]],
        'linewidth': 1.0,
        'fliersize': 1.0,
        'flierprops': {'marker': '.'}
    }

    df = pd.DataFrame({'ref': data['HS_[m]'], 'mod': result})
    df = df.loc[~df.index.month.isin([7, 8, 9]), :]

    ax = monthly_comparison_boxplot(
        ax, df['ref']*100, df['mod']*100, plot_kwargs=kwargs)
    return ax


def dropzero_r2(true, predicted):
    any_nan = np.isnan(true) | np.isnan(predicted)
    both_zero = (true == 0) & (predicted == 0)
    to_drop = any_nan | both_zero
    if np.all(to_drop):
        result = np.nan
    else:
        result = r2_score(true[~to_drop], predicted[~to_drop])
    return result

def calculate_and_plot_errors(ax, data, params):
    result = predict_with_params(
        data['SWE_[m]'],
        params,
        return_layers=False
    )

    df = pd.DataFrame({'ref': data['HS_[m]'], 'mod': result})
    df = df.loc[~df.index.month.isin([7, 8, 9]), :]

    monthly_error = (df.groupby(df.index.month).apply(lambda x: dropzero_r2(x['ref'].to_numpy(), x['mod'].to_numpy()))
                        .rename('r2')
                        .to_frame()
                        .assign(r2=lambda x: x['r2'].where(~x.index.isin([7, 8, 9]), np.nan))
                        .assign(month=lambda x: x.index.map(_get_month_abbr))
                        )

    ax = sns.pointplot(
        data=monthly_error,
        x='month',
        y='r2',
        order=[_get_month_abbr(i) for i in [10, 11, 12, 1, 2, 3, 4, 5, 6]],
        linestyles=':',
        ax=ax
    )
    return ax


def figure_05(filepath):
    print(f"- creating figure_05 and saving to\n  {filepath}")
    fig, ((ax00, ax01, ax02), (ax10, ax11, ax12)) = plt.subplots(
        2, 3, figsize=[8, 5.5], gridspec_kw={'height_ratios': [2.5, 1]})
    
    ax00 = calculate_swe2hs_on_data_and_plot_boxes(ax00, DATA_MANUAL_CALIB, MODEL_PARAMS)
    ax01 = calculate_swe2hs_on_data_and_plot_boxes(ax01, DATA_MANUAL_VALID, MODEL_PARAMS)  # ax01
    ax02 = calculate_swe2hs_on_data_and_plot_boxes(ax02, DATA_AUTOMATIC_VALID, MODEL_PARAMS)
    ax10 = calculate_and_plot_errors(ax10, DATA_MANUAL_CALIB, MODEL_PARAMS)  # ax10
    ax11 = calculate_and_plot_errors(ax11, DATA_MANUAL_VALID, MODEL_PARAMS)  # ax11
    ax12 = calculate_and_plot_errors(ax12, DATA_AUTOMATIC_VALID, MODEL_PARAMS)

    for ax in [ax01, ax11, ax02, ax12]:
        ax.set(ylabel=None, yticklabels=[])
    ax00.text(0.05, 0.95, '(a)', fontsize='x-large',
              ha='left', va='top', transform=ax00.transAxes)
    ax01.text(0.05, 0.95, '(b)', fontsize='x-large',
              ha='left', va='top', transform=ax01.transAxes)
    ax02.text(0.05, 0.95, '(c)', fontsize='x-large',
              ha='left', va='top', transform=ax02.transAxes)
    ax00.set_ylabel('HS [cm]')
    ax10.set_ylabel('$R^2$', usetex=True)
    axs = [ax00, ax01, ax02]
    ymin = min([ax.get_ylim()[0] for ax in axs])
    ymax = max([ax.get_ylim()[1] for ax in axs])
    for ax in axs:
        ax.set_ylim((ymin, ymax))
        ax.tick_params(bottom=True, left=True, right=True,
                       top=True, direction='in')
        ax.set(xlabel=None, xticklabels=[])

    axs = [ax10, ax11, ax12]
    ymin = min([ax.get_ylim()[0] for ax in axs])
    ymax = max([ax.get_ylim()[1] for ax in axs])
    for ax in axs:
        ax.set_ylim((ymin, 1))
        ax.tick_params(bottom=True, left=True, right=True, top=True,
                       direction='in', labelsize=9)
        ax.set_xlabel('Month')

    # legend
    ax00.get_legend().remove()
    ax01.get_legend().remove()
    handles, labels = ax02.get_legend_handles_labels()
    ax02.legend(
        handles,
        ['Reference', 'Model'],
        frameon=False,
        loc='upper right',
    )

    plt.tight_layout()
    fig.savefig(filepath, dpi=300,
                bbox_inches='tight', pad_inches=0.05)
    fig.clf()
    plt.close()
    return None


def predict_and_score(data, params):
    result = predict_with_params(
        data['SWE_[m]'],
        params,
        return_layers=False
    )

    data = pd.DataFrame(
        {'true': data['HS_[m]'], 'pred': result}, index=data.index).dropna()

    true = data['true']*100
    pred = data['pred']*100
    r2 = _r2_nonzero(true, pred)
    bias = _bias_score(true, pred)

    scores = {
        "r2": r2,
        "bias": bias,
    }
    return scores


def predict_and_score_per_station(data, params):
    scores = []
    for stn, df in data.groupby('site_id'):
        scores.append(pd.DataFrame(predict_and_score(df, params), index=[stn]))
    return pd.concat(scores, axis=0)


def figure_06(filepath):
    print(f"- creating figure_06 and saving to\n  {filepath}")
    latex_labels = {
        'r2': '$R^2$',
        'bias': 'bias',
    }
    aws = (predict_and_score_per_station(DATA_AUTOMATIC_VALID, MODEL_PARAMS)
           .assign(dataset=lambda x: "Automatic stations")
           .assign(dummy=lambda x: 1)  # used for y location of the bars
           )
    manu_calib = (predict_and_score_per_station(DATA_MANUAL_CALIB, MODEL_PARAMS)
          .assign(dataset=lambda x: "Manual stations calibration data")
          .assign(dummy=lambda x: 1)
          )
    manu_valid = (predict_and_score_per_station(DATA_MANUAL_VALID, MODEL_PARAMS)
                .assign(dataset=lambda x: "Manual stations validation data")
                .assign(dummy=lambda x: 1)
                )
    df = pd.concat([manu_calib, manu_valid, aws], axis=0)
    fig, axs = plt.subplots(2, 1, figsize=[5, 3.3])
    for score, ax in zip(['r2', 'bias'], axs):
        if score == 'r2':
            usetex = True
        else:
            usetex = False
        sns.boxplot(
            data=df,
            x=score,
            y='dummy',
            hue='dataset',
            orient='h',
            ax=ax,
            palette=['lightblue', 'lightcyan', 'dodgerblue'],
            showfliers=False)
        sns.stripplot(
            data=df,
            x=score,
            y="dummy",
            hue='dataset',
            orient='h',
            dodge=True,
            size=4,
            color=".3",
            linewidth=0,
            ax=ax)

        ax.tick_params(bottom=True, left=False, right=False,
                       top=True, direction='in')
        ax.set_yticklabels([])
        ax.set_ylabel(None)
        ax.set_xlabel(latex_labels[score], usetex=usetex)
        ax.get_legend().remove()

    handles, labels = axs[0].get_legend_handles_labels()
    axs[0].legend(
        handles[:3],
        labels[:3],
        frameon=False,
        loc='upper left',
    )
    plt.tight_layout()
    fig.savefig(filepath, dpi=300,
                bbox_inches='tight', pad_inches=0.05)
    fig.clf()
    plt.close()
    return None