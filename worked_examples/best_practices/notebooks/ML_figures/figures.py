import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm

from matplotlib.ticker import AutoMinorLocator
from matplotlib.colors import Normalize

from .composition import _element_composition
from scipy import stats

plt.rcParams.update({'font.size': 14})


# %%
def act_pred(y_act, y_pred,
             name='example',
             x_hist=True,
             y_hist=True,
             reg_line=True,
             save_dir=None):

    mec = '#2F4F4F'
    mfc = '#C0C0C0'

    plt.figure(1, figsize=(4, 4))
    left, width = 0.1, 0.65

    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.15]
    rect_histy = [left_h, bottom, 0.15, height]

    ax2 = plt.axes(rect_scatter)
    ax2.tick_params(direction='in',
                    length=7,
                    top=True,
                    right=True)

    # add minor ticks
    minor_locator_x = AutoMinorLocator(2)
    minor_locator_y = AutoMinorLocator(2)
    ax2.get_xaxis().set_minor_locator(minor_locator_x)
    ax2.get_yaxis().set_minor_locator(minor_locator_y)
    plt.tick_params(which='minor',
                    direction='in',
                    length=4,
                    right=True,
                    top=True)

    # feel free to change the colors here.
    ax2.plot(y_act, y_pred, 'o', mfc=mfc, alpha=0.5, label=None,
             mec=mec, mew=1.2, ms=5.2)

    ax2.plot([-10**9, 10**9], [-10**9, 10**9], 'k--', alpha=0.8,
             label='ideal')

    ax2.set_ylabel('Predicted value (Units)')
    ax2.set_xlabel('Actual value (Units)')
    x_range = max(y_act) - min(y_act)
    ax2.set_xlim(max(y_act) - x_range*1.05,
                 min(y_act) + x_range*1.05)

    ax2.set_ylim(max(y_act) - x_range*1.05,
                 min(y_act) + x_range*1.05)

    ax1 = plt.axes(rect_histx)
    ax1_n, ax1_bins, ax1_patches = ax1.hist(y_act,
                                            bins=31,
                                            density=True,
                                            color=mfc,
                                            edgecolor=mec,
                                            alpha=0)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xlim(ax2.get_xlim())
    ax1.axis('off')

    if x_hist:
        [p.set_alpha(1.0) for p in ax1_patches]

    ax3 = plt.axes(rect_histy)
    ax3_n, ax3_bins, ax3_patches = ax3.hist(y_pred,
                                            bins=31,
                                            density=True,
                                            color=mfc,
                                            edgecolor=mec,
                                            orientation='horizontal',
                                            alpha=0)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_ylim(ax2.get_ylim())
    ax3.axis('off')

    if y_hist:
        [p.set_alpha(1.0) for p in ax3_patches]

    if reg_line:
        polyfit = np.polyfit(y_act, y_pred, deg=1)
        reg_ys = np.poly1d(polyfit)(np.unique(y_act))
        ax2.plot(np.unique(y_act), reg_ys, alpha=0.8, label='linear fit')

    ax2.legend(loc=2, framealpha=0.35, handlelength=1.5)
    plt.draw()

    if save_dir is not None:
        fig_name = f'{save_dir}/{name}_act_pred.png'
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(fig_name, bbox_inches='tight', dpi=300)
    plt.draw()
    plt.pause(0.001)
    plt.close()


def residual(y_act, y_pred,
             name='example',
             save_dir=None):

    mec = '#2F4F4F'
    mfc = '#C0C0C0'

    y_act = np.array(y_act)
    y_pred = np.array(y_pred)

    xmin = np.min([y_act]) * 0.9
    xmax = np.max([y_act]) / 0.9

    y_err = y_pred - y_act
    ymin = np.min([y_err]) * 0.9
    ymax = np.max([y_err]) / 0.9

    fig, ax = plt.subplots(figsize=(4, 4))

    ax.plot(y_act, y_err, 'o', mec=mec, mfc=mfc,
            alpha=0.5, label=None, mew=1.2, ms=5.2)
    ax.plot([xmin, xmax], [0, 0], 'k--', alpha=0.8, label='ideal')

    ax.set_ylabel('Residual error (Units)')
    ax.set_xlabel('Actual value (Units)')
    ax.legend(loc='lower right')

    minor_locator_x = AutoMinorLocator(2)
    minor_locator_y = AutoMinorLocator(2)
    ax.get_xaxis().set_minor_locator(minor_locator_x)
    ax.get_yaxis().set_minor_locator(minor_locator_y)

    ax.tick_params(right=True,
                   top=True,
                   direction='in',
                   length=7)
    ax.tick_params(which='minor',
                   right=True,
                   top=True,
                   direction='in',
                   length=4)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    if save_dir is not None:
        fig_name = f'{save_dir}/{name}_residual.png'
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(fig_name, bbox_inches='tight', dpi=300)
    plt.draw()
    plt.pause(0.001)
    plt.close()


def residual_hist(y_act, y_pred,
                  name='example',
                  save_dir=None):

    mec = '#2F4F4F'
    mfc = '#C0C0C0'

    fig, ax = plt.subplots(figsize=(4, 4))
    y_err = y_pred - y_act
    kde_act = stats.gaussian_kde(y_err)
    x_range = np.linspace(min(y_err), max(y_err), 1000)

    ax.hist(y_err, color=mfc, bins=35, alpha=1, edgecolor=mec, density=True)
    ax.plot(x_range, kde_act(x_range), '-', lw=1.2, color='k', label='kde')

    ax.set_xlabel('Residual error (Units)')
    plt.legend(loc=2, framealpha=0.35, handlelength=1.5)

    ax.tick_params(direction='in',
                   length=7,
                   top=True,
                   right=True)

    minor_locator_x = AutoMinorLocator(2)
    minor_locator_y = AutoMinorLocator(2)
    ax.get_xaxis().set_minor_locator(minor_locator_x)
    ax.get_yaxis().set_minor_locator(minor_locator_y)
    plt.tick_params(which='minor',
                    direction='in',
                    length=4,
                    right=True,
                    top=True)

    if save_dir is not None:
        fig_name = f'{save_dir}/{name}_residual_hist.png'
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(fig_name, bbox_inches='tight', dpi=300)

    plt.draw()
    plt.pause(0.001)
    plt.close()


def loss_curve(x_data,
               train_err, val_err,
               name='example',
               save_dir=None):

    mec1 = '#2F4F4F'
    mfc1 = '#C0C0C0'
    mec2 = 'maroon'
    mfc2 = 'pink'

    fig, ax = plt.subplots(figsize=(4, 4))

    ax.plot(x_data, train_err, '-', color=mec1, marker='o',
            mec=mec1, mfc=mfc1, ms=4, alpha=0.5, label='train')
    ax.plot(x_data, val_err, '--', color=mec2, marker='s',
            mec=mec2, mfc=mfc2, ms=4, alpha=0.5, label='validation')

    max_val_err = max(val_err)
    ax.axhline(max_val_err, color='b', linestyle='--', alpha=0.3)

    ax.set_xlabel('Number of training epochs')
    ax.set_ylabel('Loss (Units)')
    ax.set_ylim(0, 2 * np.mean(val_err))

    ax.legend(loc=1, framealpha=0.35, handlelength=1.5)

    minor_locator_x = AutoMinorLocator(2)
    minor_locator_y = AutoMinorLocator(2)
    ax.get_xaxis().set_minor_locator(minor_locator_x)
    ax.get_yaxis().set_minor_locator(minor_locator_y)

    ax.tick_params(right=True,
                   top=True,
                   direction='in',
                   length=7)
    ax.tick_params(which='minor',
                   right=True,
                   top=True,
                   direction='in',
                   length=4)

    if save_dir is not None:
        fig_name = f'{save_dir}/{name}_loss_curve.png'
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(fig_name, bbox_inches='tight', dpi=300)
    plt.draw()
    plt.pause(0.001)
    plt.close()


def element_prevalence(formulae,
                       name='example',
                       save_dir=None,
                       log_scale=False,
                       ptable_fig=True):
    ptable = pd.read_csv('ML_figures/element_properties/ptable.csv')
    ptable.index = ptable['symbol'].values
    elem_tracker = ptable['count']
    n_row = ptable['row'].max()
    n_column = ptable['column'].max()

    for formula in formulae:
        formula_dict = _element_composition(formula)
        elem_count = pd.Series(formula_dict, name='count')
        elem_tracker = elem_tracker.add(elem_count, fill_value=0)

    if ptable_fig:
        fig, ax = plt.subplots(figsize=(n_column, n_row))
        rows = ptable['row']
        columns = ptable['column']
        symbols = ptable['symbol']
        rw = 0.9  # rectangle width (rw)
        rh = rw  # rectangle height (rh)
        for row, column, symbol in zip(rows, columns, symbols):
            row = ptable['row'].max() - row
            cmap = cm.YlGn
            count_min = elem_tracker.min()
            count_max = elem_tracker.max()
            norm = Normalize(vmin=count_min, vmax=count_max)
            count = elem_tracker[symbol]
            if log_scale:
                norm = Normalize(vmin=np.log(1), vmax=np.log(count_max))
                if count != 0:
                    count = np.log(count)
            color = cmap(norm(count))
            if count == 0:
                color = 'silver'
            if row < 3:
                row += 0.5
            rect = patches.Rectangle((column, row), rw, rh,
                                     linewidth=1.5,
                                     edgecolor='gray',
                                     facecolor=color,
                                     alpha=1)

            plt.text(column+rw/2, row+rw/2, symbol,
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontsize=20,
                     fontweight='semibold', color='k')

            ax.add_patch(rect)

        granularity = 20
        for i in range(granularity):
            value = int(round((i) * count_max/(granularity-1)))
            if log_scale:
                if value != 0:
                    value = np.log(value)
            color = cmap(norm(value))
            if value == 0:
                color = 'silver'
            length = 9
            x_offset = 3.5
            y_offset = 7.8
            x_loc = i/(granularity) * length + x_offset
            width = length / granularity
            height = 0.35
            rect = patches.Rectangle((x_loc, y_offset), width, height,
                                     linewidth=1.5,
                                     edgecolor='gray',
                                     facecolor=color,
                                     alpha=1)

            if i in [0, 4, 9, 14, 19]:
                text = f'{value:0.0f}'
                if log_scale:
                    text = f'{np.exp(value):0.1e}'.replace('+', '')
                plt.text(x_loc+width/2, y_offset-0.4, text,
                         horizontalalignment='center',
                         verticalalignment='center',
                         fontweight='semibold',
                         fontsize=20, color='k')

            ax.add_patch(rect)

        plt.text(x_offset+length/2, y_offset+0.7,
                 'log(Element Count)' if log_scale else 'Element Count',
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontweight='semibold',
                 fontsize=20, color='k')

        ax.set_ylim(-0.15, n_row+.1)
        ax.set_xlim(0.85, n_column+1.1)

        # fig.patch.set_visible(False)
        ax.axis('off')

        if save_dir is not None:
            fig_name = f'{save_dir}/{name}_ptable.png'
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(fig_name, bbox_inches='tight', dpi=300)
        plt.draw()
        plt.pause(0.001)
        plt.close()

    if not ptable_fig:
        fig, ax = plt.subplots(figsize=(15, 6))
        non_zero = elem_tracker[elem_tracker != 0].sort_values(ascending=False)
        if log_scale:
            non_zero = np.log(non_zero)
        non_zero.plot.bar(width=0.7, edgecolor='k')

        minor_locator_y = AutoMinorLocator(2)
        ax.get_yaxis().set_minor_locator(minor_locator_y)

        ax.set_ylabel('Element Count')
        if log_scale:
            ax.set_ylabel('log(Element Count)')

        ax.tick_params(right=True,
                       top=True,
                       direction='in',
                       length=7)
        ax.tick_params(which='minor',
                       right=True,
                       top=True,
                       direction='in',
                       length=4)

        if save_dir is not None:
            fig_name = f'{save_dir}/{name}_elem_hist.png'
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(fig_name, bbox_inches='tight', dpi=300)
        plt.draw()
        plt.pause(0.001)
        plt.close()


# %%
if __name__ == '__main__':
    # read in example act vs. pred data
    df_act_pred = pd.read_csv('example_data/act_pred.csv')
    y_act, y_pred = df_act_pred.iloc[:, 1], df_act_pred.iloc[:, 2]

    act_pred(y_act, y_pred,
             reg_line=True,
             save_dir='example_figures')

    act_pred(y_act, y_pred,
             name='example_no_hist',
             x_hist=False, y_hist=False,
             reg_line=True,
             save_dir='example_figures')

    residual(y_act, y_pred,
             save_dir='example_figures')

    residual_hist(y_act, y_pred,
                  save_dir='example_figures')

    # read in loss curve data
    df_lc = pd.read_csv('example_data/training_progress.csv')
    epoch = df_lc['epoch']
    train_err, val_err = df_lc['mae_train'], df_lc['mae_val']

    loss_curve(epoch, train_err, val_err,
               save_dir='example_figures')

    # element prevalence
    formula = df_act_pred.iloc[:, 0]

    element_prevalence(formula, save_dir='example_figures',
                       log_scale=False)
    element_prevalence(formula, save_dir='example_figures', name='example_log',
                       log_scale=True)

    plt.rcParams.update({'font.size': 12})
    element_prevalence(formula, save_dir='example_figures',
                       ptable_fig=False, log_scale=False)
    element_prevalence(formula, save_dir='example_figures', name='example_log',
                       ptable_fig=False, log_scale=True)
