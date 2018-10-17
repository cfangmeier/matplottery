from __future__ import print_function

from enum import Enum, auto
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


import matplottery.utils as utils


def set_defaults():
    from matplotlib import rcParams
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = 'helvetica, Helvetica, Arial, Nimbus Sans L, Mukti Narrow, FreeSans, Liberation Sans'
    rcParams['legend.fontsize'] = 'large'
    rcParams['axes.labelsize'] = 'x-large'
    rcParams['axes.titlesize'] = 'x-large'
    rcParams['xtick.labelsize'] = 'large'
    rcParams['ytick.labelsize'] = 'large'
    rcParams['figure.subplot.hspace'] = 0.1
    rcParams['figure.subplot.wspace'] = 0.1


def add_cms_info(ax, typ="Simulation", lumi="75.0", energy='13', xtype=0.1):
    ax.text(0.0, 1.01, "CMS", horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes,
            weight="bold", size="x-large")
    ax.text(xtype, 1.01, typ, horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes,
            style="italic", size="x-large")
    ax.text(0.99, 1.01, "%s fb${}^{-1}$ (%s TeV)" % (lumi, energy), horizontalalignment='right',
            verticalalignment='bottom', transform=ax.transAxes, size="large")


class RatioType(Enum):
    NONE = None
    DATAOVERBG = auto()
    DATAOVERBGPLUSSIGNAL = auto()
    SIGNALOVERBG = auto()
    SIGNALOVERBGPLUSSIGNAL = auto()
    SIGNALOVERSQRTBG = auto()
    SIGNALOVERSQRTBGPLUSSIGNAL = auto()

def plot_stack(bgs=None, data=None, sigs=None,
               title="", xlabel="", ylabel="",
               mpl_hist_params=None,
               mpl_data_params=None,
               mpl_ratio_params=None,
               mpl_figure_params=None,
               mpl_legend_params=None,
               mpl_sig_params=None,
               cms_type=None,
               lumi="-1",
               energy="13",
               xticks=None,
               ratio_type=None,
               ratio_range=None,
               do_bkg_syst=False,
               do_bkg_errors=False):
    if bgs is None: bgs = []
    if sigs is None: sigs = []
    if mpl_hist_params is None: mpl_hist_params = {}
    if mpl_data_params is None: mpl_data_params = {}
    if mpl_ratio_params is None: mpl_ratio_params = {}
    if mpl_figure_params is None: mpl_figure_params = {}
    if mpl_legend_params is None: mpl_legend_params = {}
    if mpl_sig_params is None: mpl_sig_params = {}

    set_defaults()

    colors = [bg.get_attr("color") for bg in bgs]
    labels = [bg.get_attr("label") for bg in bgs]
    if not all(colors):
        # print("Not enough colors specified, so using automatic colors")
        colors = None

    if bgs:
        bins = bgs[0].edges
    elif data:
        bins = data.edges
    else:
        print("What are you even trying to plot?")
        return

    centers = [h.bin_centers for h in bgs]
    weights = [h.counts for h in bgs]

    sbgs = sum(bgs)
    total_integral = sbgs.integral

    def percent(bg):
        return 100.0*bg.integral/total_integral if total_integral > 0 else 0
    label_map = {bg.get_attr("label"): "{:.0f}%".format(percent(bg)) for bg in bgs}

    mpl_bg_hist = {"alpha": 0.9,
                   "histtype": "stepfilled",
                   "stacked": True}
    mpl_bg_hist.update(mpl_hist_params)
    mpl_data_hist = {
        "color": "k",
        "linestyle": "",
        "marker": "o",
        "markersize": 3,
        "linewidth": 1.5
    }
    mpl_data_hist.update(mpl_data_params)

    if ratio_type is not None:
        fig, (ax_main, ax_ratio) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [9, 2], "top": 0.94},
                                                **mpl_figure_params)
    else:
        fig, ax_main = plt.subplots(1, 1, **mpl_figure_params)

    _, _, patches = ax_main.hist(centers, bins=bins, weights=weights, label=labels, color=colors, **mpl_bg_hist)
    if do_bkg_errors:
        for bg, patch in zip(bgs, patches):
            patch = patch[0]
            ax_main.errorbar(
                bg.bin_centers,
                bg.counts,
                yerr=bg.errors,
                markersize=patch.get_linewidth(),
                marker="o",
                linestyle="",
                linewidth=patch.get_linewidth(),
                color=patch.get_edgecolor(),
            )

    if do_bkg_syst:
        tot_vals = sbgs.counts
        tot_errs = sbgs.errors
        double_edges = np.repeat(sbgs.edges, 2, axis=0)[1:-1]
        his = np.repeat(tot_vals+tot_errs, 2)
        los = np.repeat(tot_vals-tot_errs, 2)
        ax_main.fill_between(double_edges, his, los, step="mid",
                             alpha=0.4, facecolor='#cccccc', edgecolor='#aaaaaa', linewidth=0.5, linestyle='-',
                             zorder=5)

    if data:
        data_xerr = None
        select = data.counts != 0
        ax_main.errorbar(
                data.bin_centers[select],
                data.counts[select],
                yerr=data.errors[select],
                xerr=data_xerr,
                label=data.get_attr("label", "Data"),
                zorder=6, **mpl_data_hist)
    if sigs:
        for sig in sigs:
            if mpl_sig_params.get("hist",True):
                ax_main.hist(sig.bin_centers, bins=bins, weights=sig.counts, color="r", histtype="step",
                             label=sig.get_attr("label","sig"))
                ax_main.errorbar(sig.bin_centers, sig.counts, yerr=sig.errors, xerr=None,
                                 markersize=1, linewidth=1.5, linestyle="",marker="o",color=sig.get_attr("color"))
            else:
                select = sig.counts != 0
                ax_main.errorbar(sig.bin_centers[select], sig.counts[select], yerr=sig.errors[select], xerr=None,
                                 markersize=3, linewidth=1.5, linestyle="",marker="o", color=sig.get_attr("color"),
                                 label=sig.get_attr("label","sig"))

    ax_main.set_ylabel(ylabel, horizontalalignment="right", y=1.)
    ax_main.set_title(title)
    legend = ax_main.legend(
        handler_map={matplotlib.patches.Patch: utils.TextPatchHandler(label_map)},
        loc='upper right',
        **mpl_legend_params
        )
    legend.set_zorder(10)
    ylims = ax_main.get_ylim()
    ax_main.set_ylim([0.0,ylims[1]])

    if cms_type is not None:
        add_cms_info(ax_main, cms_type, lumi, energy)

    # ax_main.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

    if ratio_type is not None:

        if ratio_type == RatioType.DATAOVERBG:
            ratios = data/sum(bgs)
            label = 'Data/BG'
        elif ratio_type == RatioType.DATAOVERBGPLUSSIGNAL:
            ratios = data/(sum(bgs) + sum(sigs))
            label = 'Data/(BG+SIG)'
        elif ratio_type == RatioType.SIGNALOVERBG:
            ratios = sum(sigs)/sum(bgs)
            label = 'SIG/BG'
        elif ratio_type == RatioType.SIGNALOVERBGPLUSSIGNAL:
            ratios = sum(sigs)/(sum(bgs) + sum(sigs))
            label = 'SIG/(BG+SIG)'
        elif ratio_type == RatioType.SIGNALOVERSQRTBG:
            ratios = sum(sigs)/np.sqrt(sum(bgs))
            label = 'SIG/sqrt(BG)'
        elif ratio_type == RatioType.SIGNALOVERSQRTBGPLUSSIGNAL:
            ratios = sum(sigs)/np.sqrt(sum(bgs) + sum(sigs))
            label = 'SIG/sqrt(BG+SIG)'
        else:
            raise ValueError('Unknown RatioType: {ratio_type}')

        mpl_opts_ratio = {
                "yerr": ratios.errors,
                "label": label,
                # "xerr": data_xerr,
                }
        if ratios.errors_up is not None:
            mpl_opts_ratio["yerr"] = [ratios.errors_down, ratios.errors_up]

        mpl_opts_ratio.update(mpl_data_hist)
        mpl_opts_ratio.update(mpl_ratio_params)

        ax_ratio.errorbar(ratios.bin_centers, ratios.counts, **mpl_opts_ratio)
        ax_ratio.set_autoscale_on(False)
        ylims = ax_ratio.get_ylim()
        ax_ratio.plot([ax_ratio.get_xlim()[0], ax_ratio.get_xlim()[1]], [1, 1], color="gray", linewidth=1., alpha=0.5)
        ax_ratio.set_ylim(ylims)
        # ax_ratio.legend()
        if ratio_range is not None:
            ax_ratio.set_ylim(ratio_range)

        if do_bkg_syst:
            double_edges = np.repeat(ratios.edges,2,axis=0)[1:-1]
            his = np.repeat(1.+np.abs(sbgs.relative_errors),2)
            los = np.repeat(1.-np.abs(sbgs.relative_errors),2)
            ax_ratio.fill_between(double_edges, his, los, step="mid",
                                  alpha=0.4, facecolor='#cccccc', edgecolor='#aaaaaa', linewidth=0.5, linestyle='-')

        ax_ratio.set_ylabel(mpl_opts_ratio["label"], horizontalalignment="right", y=1.)
        ax_ratio.set_xlabel(xlabel, horizontalalignment="right", x=1.)

        if xticks is not None:
            ax_ratio.xaxis.set_ticks(ratios.bin_centers)
            ax_ratio.set_xticklabels(xticks, horizontalalignment='center', rotation=45)
    else:
        ax_main.set_xlabel(xlabel, horizontalalignment="right", x=1.)
    plt.sca(ax_main)


def plot_2d(hist,
            title="", xlabel="", ylabel="",
            mpl_hist_params={}, mpl_2d_params={}, mpl_ratio_params={},
            mpl_figure_params={}, mpl_legend_params={},
            cms_type=None, lumi="-1",
            do_log=False, do_projection=False, do_profile=False,
            cmap="PuBu_r", colz_fmt=None,
            logx=False, logy=False,
            xticks=None, yticks=None,
            xlim=None, ylim=None,
            zrange=[]):
    set_defaults()

    projx, projy = None, None
    if do_projection:
        projx = hist.x_projection
        projy = hist.y_projection
    elif do_profile:
        projx = hist.x_profile
        projy = hist.y_profile

    fig = plt.gcf()
    ax = plt.gca()
    fig.subplots_adjust(left=0.14, right=1.0, top=0.92)
    do_marginal = do_projection or do_profile

    if do_marginal:
        gs = matplotlib.gridspec.GridSpec(2, 3, width_ratios=[4, 1, 0.1], height_ratios=[1, 4], wspace=0.05, hspace=0.05,
                                          left=0.1, top=0.94, right=0.92)
        ax = plt.subplot(gs[1, 0])
        axz = plt.subplot(gs[1, 2])
        axx = plt.subplot(gs[0, 0], sharex=ax)  # top x projection
        axy = plt.subplot(gs[1, 1], sharey=ax)  # right y projection
        axx.label_outer()
        axy.label_outer()

        col = matplotlib.cm.get_cmap(cmap)(0.4)
        lw = 1.5
        axx.hist(projx.bin_centers, bins=projx.edges, weights=np.nan_to_num(projx.counts), histtype="step",
                 color=col, linewidth=lw)
        axx.errorbar(projx.bin_centers, projx.counts, yerr=projx.errors, linestyle="", marker="o", markersize=0,
                     linewidth=lw, color=col)
        axy.hist(projy.bin_centers, bins=projy.edges, weights=np.nan_to_num(projy.counts), histtype="step",
                 color=col, orientation="horizontal", linewidth=lw)
        axy.errorbar(projy.counts, projy.bin_centers, xerr=projy.errors, linestyle="", marker="o", markersize=0,
                     linewidth=lw, color=col)

    if do_profile:
        axx.set_ylabel('<' + ylabel + '>')
        axy.set_xlabel('<' + xlabel + '>')
    if do_projection:
        axx.set_ylabel('<count>')
        axy.set_xlabel('<count>')

    ax.set_xlabel(xlabel, horizontalalignment="right", x=1.)
    ax.set_ylabel(ylabel, horizontalalignment="right", y=1.)

    mpl_2d_hist = {"cmap": cmap}
    mpl_2d_hist.update(mpl_2d_params)
    if zrange:
        mpl_2d_hist["vmin"] = zrange[0]
        mpl_2d_hist["vmax"] = zrange[1]

    H = hist.counts
    X, Y = np.meshgrid(*hist.edges)
    if do_log:
        mpl_2d_hist["norm"] = matplotlib.colors.LogNorm(vmin=H[H>H.min()].min(), vmax=H.max())
        if do_marginal:
            axx.set_yscale("log", nonposy='clip')
            axy.set_xscale("log", nonposx='clip')
    mappable = ax.pcolorfast(X, Y, H, **mpl_2d_hist)
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)

    if logx:
        ax.set_xscale("log", nonposx='clip')
    if logy:
        ax.set_yscale("log", nonposy='clip')

    if colz_fmt:
        xedges, yedges = hist.edges
        xcenters, ycenters = hist.bin_centers
        counts = hist.counts.flatten()
        errors = hist.errors.flatten()
        pts = np.array([
            xedges,
            np.zeros(len(xedges))+yedges[0]
            ]).T
        x = ax.transData.transform(pts)[:,0]
        y = ax.transData.transform(pts)[:,1]
        fxwidths = (x[1:] - x[:-1]) / (x.max() - x.min())

        info = np.c_[
                np.tile(xcenters,len(ycenters)),
                np.repeat(ycenters,len(xcenters)),
                np.tile(fxwidths,len(ycenters)),
                counts,
                errors
                ]
        norm = mpl_2d_hist.get("norm",
                matplotlib.colors.Normalize(
                    mpl_2d_hist.get("vmin",H.min()),
                    mpl_2d_hist.get("vmax",H.max()),
                    ))
        val_to_rgba = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap).to_rgba
        fs = min(int(30.0/min(len(xcenters),len(ycenters))),15)

        def val_to_text(bv,be):
            if bv == be == 0.0:
                return "0\n($\pm$0%)"
            else:
                return ("{:%s}\n($\pm${:.1f}%%)" % colz_fmt).format(bv,100.0*be/bv)

        do_autosize = True
        for x, y, fxw, bv, be in info:
            if do_autosize:
                fs_ = min(5.5*fxw*fs,14)
            else:
                fs_ = 1.0*fs
            color = "w" if (utils.compute_darkness(*val_to_rgba(bv)) > 0.45) else "k"
            ax.text(x, y, val_to_text(bv, be),
                    color=color, ha="center", va="center", fontsize=fs_,
                    wrap=True)

    if do_marginal:
        plt.colorbar(mappable, cax=axz)
    else:
        plt.colorbar(mappable)

    if do_marginal:
        if cms_type is not None:
            add_cms_info(axx, cms_type, lumi, xtype=0.12)
        axx.set_title(title)
    else:
        if cms_type is not None:
            add_cms_info(ax, cms_type, lumi, xtype=0.12)
        ax.set_title(title)

    if xticks is not None:
        ax.xaxis.set_ticks(xticks)
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    if yticks is not None:
        ax.yaxis.set_ticks(yticks)
        ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())

