"""Core data containers and plotting helpers.

The module deliberately has no dependency on the QTLab-specific layer.  This
keeps the reusable array operations available to projects that use other data
acquisition systems and avoids the circular import present in the original
codebase.
"""

from __future__ import annotations

import copy
import logging
import operator as op
import os
import re
from collections import OrderedDict
from inspect import signature

import matplotlib as mpl
import matplotlib.axes
import matplotlib.cm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import scipy.interpolate

# from matplotlib.cm import get_cmap
from matplotlib.cm import ScalarMappable
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from scipy.signal import medfilt, savgol_filter

try:
    from lmfit import Parameters, minimize, report_fit
except ImportError:  # Optional dependency used only by Dataset.global_fit.
    minimize = None
    Parameters = None
    report_fit = None
from scipy.io import loadmat

logger = logging.getLogger(__name__)

# std_colors = [
#     "#e41a1c",
#     "#377eb8",
#     "#4daf4a",
#     "#984ea3",
#     "#ff7f00",
#     "#ffff33",
#     "#a65628",
#     "#f781bf",
# ]
# _colornames = ["red", "blue", "green", "purple", "orange", "yellow", "brown", "pink"]
# color_dct = {}
# for key, value in zip(_colornames, std_colors):
#     color_dct[key] = value


# plt.rcParams["svg.fonttype"] = "none"
# plt.rcParams["mathtext.fontset"] = "cm"

# plt.rcParams["mathtext.fontset"] = "stix"
# print(plt.rcParams["datapath"])
# #
# import matplotlib

# matplotlib.font_manager._rebuild()
# for f in matplotlib.font_manager.fontManager.ttflist:
#     print(f.name)


# plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["font.sans-serif"] = "Times New Roman"
# plt.rcParams["mathtext.fontset"] = "custom"
# plt.rcParams["mathtext.rm"] = "Helvetica"
# plt.rcParams["mathtext.it"] = "Helvetica:italic"
# plt.rcParams["mathtext.bf"] = "Helvetica:bold"
# plt.rcParams["mathtext.sf"] = "Helvetica"
# plt.rcParams["text.usetex"] = True
# plt.rcParams["text.latex.unicode"] = True
# plt.rcParams["text.latex.preamble"] = [
#     r"\usepackage{siunitx}",  # i need upright \micro symbols, but you need...
#     r"\sisetup{detect-all}",  # ...this to force siunitx to actually use your fonts
#     r"\usepackage{helvet}",  # set the normal font here
#     r"\usepackage{sansmath}",  # load up the sansmath so that math -> helvet
#     r"\sansmath",  # <- tricky! -- gotta actually tell tex to use!
# ]


def generate_color_list(cmap="gnuplot", length=1):
    """Return ``length`` evenly spaced colors from a Matplotlib colormap."""
    if length < 0:
        raise ValueError("length must be non-negative")
    if length == 0:
        return []
    color_map = mpl.colormaps[cmap]
    denominator = max(length - 1, 1)
    return [color_map(index / denominator) for index in range(length)]


def add_metric_prefix(d):
    """
    The function takes a floating point number, and outputs the value as a number between 1-999 with order of
    magnitude added.

    Parameters
    ----------
    d : float

    Returns
    -------
    string

    """
    unit_list = ["f", "p", "n", "u", "m", "", "k", "M", "G"]
    unit = 5
    if d == 0:
        return "0"
    while abs(d) >= 1000 and unit < len(unit_list) - 1:
        d = d / 1000
        unit += 1
    while abs(d) < 1 and unit > 0:
        d = d * 1000
        unit -= 1
    return f"{d:.2f} {unit_list[unit]}"


def _read_delimited_file(filename, axes, options):
    """Read a delimited file and separate pandas options from Data metadata."""
    options = dict(options)
    valid_reader_options = set(signature(pd.read_csv).parameters)
    reader_options = {
        key: options.pop(key)
        for key in tuple(options)
        if key in valid_reader_options and key != "filepath_or_buffer"
    }
    if "sep" not in reader_options and "delimiter" not in reader_options:
        reader_options["delimiter"] = "\t"
    reader_options.setdefault("comment", "#")
    reader_options.setdefault("header", None)
    reader_options.setdefault("names", list(axes))

    frame = pd.read_csv(filename, **reader_options)
    columns = {
        str(column): frame[column].to_numpy()
        for column in frame
        if not frame[column].empty
    }
    return columns, options


class Subplot:
    _default = {
        "type": "2d",
        "bins": 50,
        "axis_scales": ("linear", "linear"),
        "crange": (0, 0),
        "zrange": (0, 0),
        "cmap": "seismic",
        "legend": False,
        "invert_yaxis": False,
        "invert_xaxis": False,
        "background": "white",
        "legend_border": False,
        "font": "Arial",
        "title": "",
        "yticks": 0,
        "xticks": 0,
        "title_align": "center",
        "scilimits": (-3, 3),
    }

    def __init__(self, *datasets, **settings):
        self.datasets = list(datasets)
        self._settings = settings

    def __delitem__(self, key):
        del self._settings[key]

    def __contains__(self, item):
        return item in self._settings

    def __setitem__(self, key, value):
        self._settings[key] = value

    def __getitem__(self, item):
        ret = self._settings.get(item)
        if ret is None:
            ret = self._default.get(item)
        if item == "axis_labels" and not ret:
            return self.datasets[0].axes
        return ret

    def settings(self, *args, **settings):
        for key in settings:
            self._settings[key] = settings[key]
        if args:
            dct = {}
            for arg in args:
                if arg in self._settings:
                    dct[arg] = self._settings[arg]
            return dct
        else:
            return self._settings

    def get_data(self):
        return self.datasets

    def set_data(self, *datasets):
        self.datasets = list(datasets)
        if Figure.cf:
            try:
                Figure.cf.plot_subplot(self)
            except:
                pass

    def add_data(self, *datasets):
        for dataset in datasets:
            self.datasets.append(dataset)
        if Figure.cf:
            try:
                Figure.cf.plot_subplot(self)
            except:
                pass


class Figure:
    def __init__(
        self,
        aspect_ratio=1.5,
        rows=1,
        font=None,
        dpi=150,
        title="",
        labels=None,
        size=1.0,
        sharex=False,
        sharey=False,
        pad=0.2,
    ):
        if labels is None:
            labels = []
        self._subplots = []
        self._aspect_ratio = aspect_ratio
        self._rows = rows
        self._title = title
        self._dpi = dpi
        self._pad = pad
        self._font = font if font else {"font.size": 12}
        self._labels = labels
        self._size = size
        self._share = (sharex, sharey)
        Figure.cf = self

    def __bool__(self):
        return bool(len(self._subplots))

    def add_subplot(self, subplot, subplot_twin=None):
        if not subplot_twin:
            self._subplots.append(subplot)
        else:
            subplot["twin"] = subplot_twin
            self._subplots.append(subplot)

    def transpose(self):
        self._subplots = list(
            np.reshape(self._subplots, (self._rows, -1), order="F").flatten()
        )

    def _get_axis_metric_prefix(self, dec, lim, reverse=False):
        decades = [1e-15, 1e-12, 1e-9, 1e-6, 1e-3, 1, 1e3, 1e6, 1e9, 1e12]
        prefix = ["f", "p", "n", r"\mu ", "m", "", "k", "M", "G", "T"]
        if reverse:
            decades = [1e15, 1e12, 1e-9, 1e6, 1e3, 1, 1e-3, 1e-6, 1e-9, 1e-12]

        scale = 1
        # print('before: dec: {}, lim: {}'.format(dec,lim))
        while lim > 100:
            lim /= 1e3
            scale *= 1e3
            dec *= 1e3
        while lim <= 0.1:
            lim *= 1e3
            scale /= 1e3
            dec /= 1e3
        if dec < 1e-15:
            dec = 1
            scale = 1
        if reverse:
            for i, d in enumerate(decades):
                if int(np.log10(dec)) > int(np.log10(d)):
                    # print('after: dec: {}, lim: {}, scale: {}, prefix: {}'.format(dec, lim, scale, prefix[i-1]))
                    return prefix[i - 1], scale
        else:
            for i, d in enumerate(decades):
                if int(np.log10(dec)) < int(np.log10(d)):
                    # print('after: dec: {}, lim: {}, scale: {}, prefix: {}'.format(dec, lim, scale, prefix[i-1]))
                    return prefix[i - 1], scale
        return "", 1

    def plot_subplot(self, subplot):
        # plt.rcParams["font.family"] = subplot['font']
        plt.sca(subplot["ax"])
        subplot["ax"].cla()
        if subplot["type"] == "image":
            for dataset in subplot.datasets:
                subplot["ax"].imshow(dataset)
                subplot["ax"].spines["top"].set_visible(False)
                subplot["ax"].spines["left"].set_visible(False)
                subplot["ax"].spines["bottom"].set_visible(False)
                subplot["ax"].spines["right"].set_visible(False)
                subplot["ax"].xaxis.set_visible(False)
                subplot["ax"].yaxis.set_visible(False)
                return
        if subplot["type"] == "2d":
            handles = []
            for i, dataset in enumerate(subplot.datasets):
                if dataset:
                    dataset.plot_2d(handles, index=i, **subplot.settings("cmap"))
            if subplot["legend"]:
                tdct = subplot.settings("legend_loc")
                if "legend_loc" in tdct:
                    tdct = {"loc": tdct["legend_loc"]}
                plt.legend(handles=handles, frameon=subplot["legend_border"], **tdct)
        if subplot["type"] == "scatter":
            handles = []
            for i, dataset in enumerate(subplot.datasets):
                if dataset:
                    dataset.plot_scatter(handles)
            subplot["lines"] = handles
            if subplot["legend"]:
                plt.legend(handles=handles, frameon=subplot["legend_border"])
        if subplot["type"] == "hist2d":
            for dataset in subplot.datasets:
                if dataset:
                    dataset.plot_hist2d(**subplot.settings("cmap", "bins"))
        if subplot["type"] == "hist":
            for i, dataset in enumerate(subplot.datasets):
                if dataset:
                    dataset.plot_hist(
                        index=i, **subplot.settings("bins", "color", "normed")
                    )
            if subplot["legend"]:
                plt.legend(frameon=subplot["legend_border"])
        if subplot["type"] == "color":
            for dataset in subplot.datasets:
                if subplot["cbar"] and subplot["cbar"] != "None":
                    subplot["cbar"].remove()
                if dataset:
                    dataset.plot_color(**subplot.settings("crange", "cmap", "calpha"))
                if subplot["cbar"] != "None":
                    # divider = make_axes_locatable(subplot['ax'])
                    # cax1 = divider.append_axes("right", size="10%", pad=0.05)
                    # subplot['cbar'] = plt.colorbar(format='%.2g', cax=cax1)
                    subplot["cbar"] = plt.colorbar(shrink=0.9)
                    subplot["cbar"].set_label(subplot["axis_labels"][2])
        if subplot["type"] == "3d":
            for dataset in subplot.datasets:
                if subplot["cbar"]:
                    subplot["cbar"].remove()
                if dataset:
                    p = dataset.plot_3d(**subplot.settings("zrange", "cmap"))
                subplot["cbar"] = plt.colorbar(p)
                subplot["cbar"].set_clim(*subplot["crange"])
                subplot["cbar"].set_label(subplot["axis_labels"][2])
                tick_locator = ticker.MaxNLocator(nbins=5)
                subplot["cbar"].locator = tick_locator
                subplot["cbar"].update_ticks()
        plt.gca().set_facecolor(subplot["background"])
        if subplot["invert_xaxis"]:
            plt.gca().invert_xaxis()
        if subplot["invert_yaxis"]:
            plt.gca().invert_yaxis()
        plt.xscale(subplot["axis_scales"][0])
        plt.yscale(subplot["axis_scales"][1])
        plt.xlabel(subplot["axis_labels"][0])
        plt.ylabel(subplot["axis_labels"][1])
        if subplot["xrange"]:
            xlim = plt.xlim()
            x0, x1 = subplot["xrange"]
            if x0 == -np.inf or x0 == np.inf:
                x0 = xlim[0]
            if x1 == -np.inf or x1 == np.inf:
                x1 = xlim[1]
            plt.xlim(x0, x1)
        if subplot["yrange"]:
            ylim = plt.ylim()
            y0, y1 = subplot["yrange"]
            if y0 == -np.inf or y0 == np.inf:
                y0 = ylim[0]
            if y1 == -np.inf or y1 == np.inf:
                y1 = ylim[1]
            plt.ylim(y0, y1)
        if subplot["yticks"]:
            plt.locator_params(axis="y", nticks=subplot["yticks"])
        if subplot["xticks"]:
            plt.locator_params(axis="x", nticks=subplot["xticks"])
        plt.gca().ticklabel_format(axis="x", useOffset=False)
        if subplot["hide_border"]:
            ax = plt.gca()
            if subplot["hide_xlabels"]:
                ax.spines["bottom"].set_visible(False)
            if subplot["hide_ylabels"]:
                ax.spines["left"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        if subplot["hide_ylabels"]:
            plt.gca().yaxis.set_visible(False)
        if subplot["hide_xlabels"]:
            plt.gca().xaxis.set_visible(False)
        plt.title(subplot["title"], loc=subplot["title_align"])

        if subplot["twin"]:
            self.plot_subplot(subplot["twin"])
        plt.gcf().canvas.draw()

    def _change_ticks(self, subplot):
        # change labels
        if (
            "{metric_prefix}" in subplot["axis_labels"][1]
            or "{metric_prefix_reverse}" in subplot["axis_labels"][1]
        ):
            pf_y, scale_y = self._get_axis_metric_prefix(
                subplot.settings().get("scale_factor_y", 1),
                np.max(np.abs(subplot["ax"].get_ylim())),
                reverse="{metric_prefix_reverse}" in subplot["axis_labels"][1],
            )
            ticks = subplot["ax"].get_yticks() / scale_y
            subplot["ax"].set_yticklabels([f"${s:g}$" for s in ticks])
            if subplot["ax"].get_ylabel():
                subplot["ax"].set_ylabel(
                    subplot["axis_labels"][1]
                    .replace("{metric_prefix}", pf_y)
                    .replace("{metric_prefix_reverse}", pf_y)
                )
        if (
            "{metric_prefix}" in subplot["axis_labels"][0]
            or "{metric_prefix_reverse}" in subplot["axis_labels"][0]
        ):
            pf_x, scale_x = self._get_axis_metric_prefix(
                subplot.settings().get("scale_factor_x", 1),
                np.max(np.abs(subplot["ax"].get_xlim())),
                reverse="{metric_prefix_reverse}" in subplot["axis_labels"][0],
            )
            ticks = subplot["ax"].get_xticks() / scale_x
            subplot["ax"].set_xticklabels([f"${s:g}$" for s in ticks])
            if subplot["ax"].get_xlabel():
                subplot["ax"].set_xlabel(
                    subplot["axis_labels"][0]
                    .replace("{metric_prefix}", pf_x)
                    .replace("{metric_prefix_reverse}", pf_x)
                )
        if (
            len(subplot["axis_labels"]) > 2
            and (
                "{metric_prefix}" in subplot["axis_labels"][2]
                or "{metric_prefix_reverse}" in subplot["axis_labels"][2]
            )
            and subplot["cbar"] != "None"
        ):
            pf_z, scale_z = self._get_axis_metric_prefix(
                subplot.settings().get("scale_factor_z", 1),
                np.max(np.abs(subplot["cbar"].get_clim())),
                reverse="{metric_prefix_reverse}" in subplot["axis_labels"][2],
            )
            # pf_z, scale_z = self._get_axis_metric_prefix(
            #     subplot.settings().get("scale_factor_z", 1),
            #     np.max(subplot["cbar"]),
            #     reverse="{metric_prefix_reverse}" in subplot["axis_labels"][2],
            # )
            ticks = np.array(
                [
                    float(t.get_text().replace("−", "-").replace("$", ""))
                    for t in subplot["cbar"].ax.get_yticklabels()
                ]
            )
            try:
                ticks *= (
                    float(
                        subplot["cbar"]
                        .ax.yaxis.get_major_formatter()
                        .get_offset()
                        .replace("−", "-")
                    )
                    / scale_z
                )
            except:
                ticks /= scale_z
            ticklabels = np.array([f"${s:g}$" for s in ticks], dtype=str)
            # set the in between values to 0 (we don't generally need to see em)
            ticklabels[1 : (len(ticklabels) // 2)] = " "
            ticklabels[(len(ticklabels) // 2 + 1) : -1] = " "
            subplot["cbar"].ax.set_yticklabels(ticklabels)
            subplot["cbar"].set_label(
                subplot["axis_labels"][2]
                .replace("{metric_prefix}", pf_z)
                .replace("{metric_prefix_reverse}", pf_z)
            )

    def _keydown(self, event):
        if event.key == "control":
            self._ctrl = True

    def _keyup(self, event):
        if event.key == "control":
            self._ctrl = False

    def _onpick(self, event):
        pc = event.artist
        cont = False
        for subplot in self._subplots:
            if pc in subplot["lines"]:
                cont = True
                break
        if not cont:
            return True
        N = len(event.ind)
        if not N:
            return True
        func = subplot["onpick"]
        # the click locations
        mx = event.mouseevent.xdata
        my = event.mouseevent.ydata

        # pc.set_offset_position('data')
        try:
            xy = pc.get_data()
        except:
            xy = pc.get_offsets()
        # xy = line.get_data()
        distances = np.hypot(mx - xy[event.ind][:, 0], my - xy[event.ind][:, 1])
        indmin = distances.argmin()
        dataind = event.ind[indmin]

        highlighter = subplot["highlighter"]
        if not highlighter:
            (subplot["highlighter"],) = subplot["ax"].plot(
                xy[dataind][0], xy[dataind][1], "o", ms=12, alpha=0.4, color="yellow"
            )
        else:
            if self._ctrl:
                d = subplot["highlighter"].get_data()
                subplot["highlighter"].set_data(
                    np.append(d[0], xy[dataind][0]), np.append(d[1], xy[dataind][1])
                )
            else:
                subplot["highlighter"].set_data(xy[dataind][0], xy[dataind][1])
        plt.gcf().canvas.draw()
        if func:
            func(subplot, dataind, self._ctrl)

    def _onclick(self, event):
        for spl in self._subplots:
            if spl["ax"] == event.inaxes:
                if spl["onclick"]:
                    spl["onclick"](spl, event)
                    plt.gcf().canvas.draw()

    def visualise(
        self,
        save_as="",
        block=True,
        position=None,
        name="Figure",
        change_func=None,
        tight=True,
        labels=None,
        show=True,
    ):
        if labels is None:
            labels = []
        Figure.cf = self
        params = {"legend.fontsize": 6, "legend.handlelength": 1}
        plt.rcParams.update(params)
        if self._font:
            plt.rcParams.update(self._font)
        plt.close()

        fig = plt.figure(
            name,
            figsize=(
                4.2  # was 3.3
                * self._aspect_ratio
                * self._size
                * int((len(self._subplots)) / self._rows),
                self._rows * 3.3 * self._size,
            ),
            dpi=self._dpi,
        )
        self._fig = fig
        # get the events
        fig.canvas.mpl_connect("pick_event", self._onpick)
        fig.canvas.mpl_connect("key_press_event", self._keydown)
        fig.canvas.mpl_connect("key_release_event", self._keyup)
        fig.canvas.mpl_connect("button_press_event", self._onclick)
        self._ctrl = False

        subplots = np.empty(
            (int((len(self._subplots)) / self._rows), self._rows),
            dtype=matplotlib.cm.ScalarMappable,
        )

        for index, subplot in enumerate(self._subplots):
            i, j = divmod(index, self._rows)

            if subplot["type"] == "3d":
                subplot["ax"] = fig.add_subplot(
                    self._rows,
                    int((len(self._subplots)) / self._rows),
                    index + 1,
                    projection="3d",
                )
                subplots[i, j] = subplot["ax"]
            else:
                subplot["ax"] = fig.add_subplot(
                    self._rows, int((len(self._subplots)) / self._rows), index + 1
                )
                subplots[i, j] = subplot["ax"]
                if subplot["twin"]:
                    subplot["twin"]["ax"] = subplot["ax"].twinx()

            self.plot_subplot(subplot)
            subplot["id"] = index
            if subplot["onload"]:
                subplot["onload"](subplot)

            if labels:
                self._labels = labels
            if self._labels:
                tot = len(self._subplots)
                maxx, b = divmod(tot, self._rows)
                a, b = divmod(index, maxx)
                x = b / maxx
                y = 1 - (a / self._rows)
                plt.gcf().text(
                    x,
                    y,
                    self._labels[index],
                    weight="bold",
                    fontsize=6,
                    horizontalalignment="left",
                    verticalalignment="top",
                )
            # subplot['ax'].ticklabel_format(style='sci',scilimits=subplot['scilimits'])
            subplot["ax"].ticklabel_format(style="plain", useOffset=False)

        if self._title:
            fig.suptitle(self._title)
        if tight:
            plt.tight_layout(pad=self._pad)

        if self._share[0]:
            fig.subplots_adjust(hspace=0)
        if self._share[1]:
            fig.subplots_adjust(wspace=0)
        for i in range(subplots.shape[0]):
            for j in range(subplots.shape[1]):
                # i is x direction
                # j is y direction
                if i < subplots.shape[0] - 1 and self._share[1]:
                    index = i + j * subplots.shape[0]
                    cbar = self._subplots[index]["cbar"]
                    if cbar:
                        cbar.remove()
                if i > 0 and self._share[1]:
                    index = i + j * subplots.shape[0]
                    subplot = self._subplots[index]["ax"]
                    subplot.tick_params(
                        axis="y", which="both", left=False, labelleft=False
                    )
                    subplot.set_ylabel("")

                if j < subplots.shape[1] - 1 and self._share[0]:
                    index = i + j * subplots.shape[0]
                    subplot = self._subplots[index]["ax"]
                    subplot.tick_params(
                        axis="x", which="both", bottom=False, labelbottom=False
                    )
                    subplot.set_xlabel("")
        # if self._share[0]:
        #     fig.subplots_adjust(hspace=0)
        # if self._share[1]:
        #     fig.subplots_adjust(wspace=0)
        for subplot in self._subplots:
            self._change_ticks(subplot)
            if subplot["twin"]:
                self._change_ticks(subplot["twin"])

        if change_func:
            change_func(self)
        if position:
            mngr = plt.get_current_fig_manager()
            # get the QTCore PyRect object
            # geom = mngr.window.geometry()
            # print(get_backend())
            # mngr.window.Move(0,0)
            mngr.window.wm_geometry("+%d+%d" % position)
            # thismanager.window.SetPosition(position)
        if show:
            if not save_as:
                plt.show(block=block)
            else:
                file_path = save_as
                directory = os.path.dirname(file_path)
                if not os.path.exists(directory):
                    try:
                        os.makedirs(directory)
                    except:
                        pass
                plt.savefig(file_path)
                plt.close()


class Data:
    """Collection of equally shaped, named NumPy arrays.

    String indexing selects a column as another :class:`Data` object. Boolean
    masks and slices select samples across every column. Arithmetic and
    comparison operators are available on one-column objects.
    """

    def __getitem__(self, item):
        if isinstance(item, str):
            if item not in self._data:
                raise KeyError(item)
            dat = self.copy()
            dat._data = {item: self._data[item]}
            return dat
        if isinstance(item, (np.ndarray, list, tuple)):
            d = {key: values[item] for key, values in self._data.items()}
            dat = self.copy()
            dat._data = d
            return dat
        if isinstance(item, slice):
            d = {key: values[item] for key, values in self._data.items()}
            dat = self.copy()
            dat._data = d
            return dat
        if isinstance(item, (int, np.integer)):
            if len(self._data) != 1:
                raise IndexError(
                    f"Incorrect indexing, cannot find a single value for index {item}."
                )
            return self._data[self.onlykey()][item]
        raise TypeError(f"unsupported index type: {type(item).__name__}")

    def __setitem__(self, key, value):
        if isinstance(key, str):
            array = value.values if isinstance(value, Data) else np.asarray(value)
            if self._data and array.shape != next(iter(self._data.values())).shape:
                raise ValueError("assigned column must match the existing data shape")
            self._data[key] = array
            return
        if isinstance(key, np.ndarray):
            for col in self._data:
                self._data[col][key] = value[col]
            return
        raise TypeError("key must be a column name or NumPy mask")

    ####################################################################################################################
    #  data operation methods
    ####################################################################################################################

    # useful in limiting data in such a way: d = d[d['x'] > 0], or multiplying columns: d['Isd']*=1e9
    def operate(self, other, operator):
        if len(self._data) > 1:
            raise RuntimeError(
                "Incorrect operator, cannot operate on multiple columns at the same time."
            )
        # get numpy arrays
        key = self.onlykey()
        if isinstance(other, Data):
            if key not in other or len(other._data[key]) != len(self._data[key]):
                raise RuntimeError("Incorrect operation.")
            return operator(self._data[key], other._data[key])
        else:
            return operator(self._data[key], other)

    def __eq__(self, other):
        return self.operate(other, op.eq)

    def __gt__(self, other):
        return self.operate(other, op.gt)

    def __ge__(self, other):
        return self.operate(other, op.ge)

    def __lt__(self, other):
        return self.operate(other, op.lt)

    def __le__(self, other):
        return self.operate(other, op.le)

    def __add__(self, other):
        return self.operate(other, op.add)

    def __sub__(self, other):
        return self.operate(other, op.sub)

    def __mul__(self, other):
        return self.operate(other, op.mul)

    def __truediv__(self, other):
        return self.operate(other, op.truediv)

    def __str__(self, *args, **kwargs):
        msg = ""
        for key in self._data:
            msg += f"{key}:\n{self._data[key]}\n"
        return msg

    def __contains__(self, key):
        return key in self._data

    def __iter__(self):
        if len(self._data) != 1:
            return self._data.__iter__()
        return self._data[self.onlykey()].__iter__()

    def __len__(self):
        return len(next(iter(self._data.values()))) if self._data else 0

    def __bool__(self):
        return bool(self._data)

    def __delitem__(self, key):
        del self._data[key]

    def onlykey(self):
        if len(self._data) != 1:
            raise RuntimeError(
                "Multiple columns present, data not ready for single index."
            )
        return next(iter(self._data))

    ####################################################################################################################
    #  initialise methods
    ####################################################################################################################

    def copy(self, new_data=None):
        if new_data is not None:
            d = copy.deepcopy(self)
            d._data = new_data
            return d
        return copy.deepcopy(self)
        # return eval('{}(new_data,**self.ps())'.format(self.__class__.__name__))  # this is very ugly, but I don't know how to do it otherwise... subclasses should be called correctly!

    def __init__(self, dat, **kwargs):
        if not isinstance(dat, (dict, OrderedDict)) or not dat:
            raise ValueError("dat must be a non-empty mapping of NumPy arrays")
        if any(not isinstance(value, np.ndarray) for value in dat.values()):
            raise TypeError("all data columns must be NumPy arrays")
        if len({value.shape for value in dat.values()}) != 1:
            raise ValueError("all data columns must have the same shape")

        self._data = dict(dat)
        self._plt_settings = {}
        self.ps(**kwargs)
        if not kwargs.get("axes"):
            self.ps(axes=list(dat))
        # set methods to instance methods
        self.reshape = self._reshape
        self.flatten = self._flatten
        self.interpolate = self._interpolate
        self.resample = self._resample
        self.smooth = self._smooth
        self.derive = self._derive

    def rename_axis(self, old_name, new_name):
        if new_name in self._data:
            raise RuntimeError("New name already present in data!")
        self._data[new_name] = self._data[old_name]
        del self._data[old_name]
        if old_name in self.axes:
            axes = list(self.axes)
            axes[axes.index(old_name)] = new_name
            self.axes = axes

    ####################################################################################################################
    #  load from file
    ####################################################################################################################

    @classmethod
    def load_from_file(cls, filename, **kwargs):
        path = os.fspath(filename)
        axes = tuple(kwargs.pop("axes", ("x", "y")))
        extension = os.path.splitext(path)[1].lower()
        if extension in {".dat", ".csv", ".txt", ".tsv", ".asc"}:
            if (
                extension == ".csv"
                and "sep" not in kwargs
                and "delimiter" not in kwargs
            ):
                kwargs["sep"] = ","
            data, metadata = _read_delimited_file(path, axes, kwargs)
        elif extension == ".mat":
            raw = loadmat(path)
            data = {
                axis: np.asarray(raw[axis]).squeeze()
                for axis in axes
                if axis in raw and np.asarray(raw[axis]).size
            }
            metadata = kwargs
        else:
            raise ValueError(
                f"unsupported data file extension: {extension or '<none>'}"
            )
        if not data:
            raise ValueError(f"no data found in {path}")
        return cls(data, axes=list(axes), **metadata)

    ####################################################################################################################
    #  properties
    ####################################################################################################################

    # just a little shortcut for playing around with axes
    @property
    def axes(self):
        return self.ps("axes")["axes"]

    @axes.setter
    def axes(self, lst):
        if hasattr(lst, "__len__") and len(lst) > 0:
            self.ps(axes=lst)
        else:
            raise ValueError(
                "Wrong type supplied to axes; expecting a list of strings."
            )

    # the correct way of getting the values out
    @property
    def values(self):
        return self._data[self.onlykey()]

    ####################################################################################################################
    #  plot settings
    ####################################################################################################################

    # this is used to communicate all variables on the data object. (e.g., the axes, the color of the plot, whatever!)
    def ps(self, *args, **kwargs):
        return self.plot_settings(*args, **kwargs)

    def plot_settings(self, *args, **kwargs):
        self._plt_settings.update(kwargs)
        if args:
            return {
                arg: self._plt_settings[arg]
                for arg in args
                if arg in self._plt_settings
            }
        return self._plt_settings

    ####################################################################################################################
    #  plot methods
    ####################################################################################################################

    def plot_scatter(self, handles=None):
        if handles is None:
            handles = []
        plt_set = self.plot_settings()
        picker = plt_set.get("picker", 5)
        line = plt.scatter(
            self[self.axes[0]],
            self[self.axes[1]],
            picker=picker,
            **self.plot_settings("color", "label"),
        )
        handles.append(line)
        return handles

    def plot_2d(self, handles=None, cmap="gist_rainbow", index=0, **kwargs):
        if handles is None:
            handles = []
        try:
            ns = np.unique(self[self.axes[2]].values)
        except:
            ns = [0]
        datasets = len(ns)
        ps = self.plot_settings()
        # get color(s)
        c = ps.get("color")
        if datasets > 1:
            if type(c) is list:
                colors = c
            else:
                # o_cmap = get_cmap(cmap)
                o_cmap = ScalarMappable(cmap)
                colors = [o_cmap(x / datasets) for x in range(datasets)]
        for i, v in enumerate(ns):
            # select subset of data
            try:
                d = self[self[self.axes[2]] == v]
            except:
                d = self
            # get the label of the dataset
            try:
                nlbl = ps.get("n_labels", [])[i]
            except:
                nlbl = v
            lbl = ps.get("label", "{name}").format(
                name=ps.get("name", ""), index=index, n=nlbl
            )
            lw = ps.get("linewidth", 0.5)
            zo = ps.get("zorder", 2)
            mark = ps.get("marker", "")
            if mark:
                ln = ps.get("linestyle", "")
            else:
                ln = ps.get("linestyle", "-")
            if type(lw) is list:
                lw = lw[index]
            if type(zo) is list:
                zo = zo[i]
            if type(mark) is list:
                mark = mark[i]
            if datasets > 1:
                if (i == 0 and lbl) or ps.get("n_label", False):
                    (line,) = plt.plot(
                        d[self.axes[0]].values,
                        d[self.axes[1]].values,
                        label=lbl,
                        linewidth=lw,
                        color=colors[i],
                        linestyle=ln,
                        marker=mark,
                        zorder=zo,
                        **self.ps("markersize", "mec", "mew", "mfc", "path_effects"),
                    )
                    handles.append(line)
                else:
                    plt.plot(
                        d[self.axes[0]].values,
                        d[self.axes[1]].values,
                        linewidth=lw,
                        color=colors[i],
                        linestyle=ln,
                        marker=mark,
                        zorder=zo,
                        **self.ps("markersize", "mec", "mew", "mfc", "path_effects"),
                    )
            else:
                c = ps.get("color")
                if not (type(c) is str or type(c) is np.str_ or type(c) is tuple):
                    try:
                        c = c[index]
                    except:
                        pass
                if c:
                    if lbl:
                        (line,) = plt.plot(
                            d[self.axes[0]].values,
                            d[self.axes[1]].values,
                            label=lbl,
                            linewidth=lw,
                            color=c,
                            linestyle=ln,
                            marker=mark,
                            zorder=zo,
                            **self.ps(
                                "markersize", "mec", "mew", "mfc", "path_effects"
                            ),
                        )
                        handles.append(line)
                    else:
                        plt.plot(
                            d[self.axes[0]].values,
                            d[self.axes[1]].values,
                            linewidth=lw,
                            color=c,
                            linestyle=ln,
                            marker=mark,
                            zorder=zo,
                            **self.ps(
                                "markersize", "mec", "mew", "mfc", "path_effects"
                            ),
                        )
                else:
                    if lbl:
                        (line,) = plt.plot(
                            d[self.axes[0]].values,
                            d[self.axes[1]].values,
                            label=lbl,
                            linewidth=lw,
                            linestyle=ln,
                            marker=mark,
                            zorder=zo,
                            **self.ps(
                                "markersize", "mec", "mew", "mfc", "path_effects"
                            ),
                        )
                        handles.append(line)
                    else:
                        plt.plot(
                            d[self.axes[0]].values,
                            d[self.axes[1]].values,
                            linewidth=lw,
                            linestyle=ln,
                            marker=mark,
                            zorder=zo,
                            **self.ps(
                                "markersize", "mec", "mew", "mfc", "path_effects"
                            ),
                        )

    def plot_hist2d(self, bins=50, cmap="gnuplot"):
        return plt.hist2d(
            self[self.axes[0]].values, self[self.axes[1]].values, bins=bins, cmap=cmap
        )

    def plot_hist(self, bins=50, axis=-1, index=0, **kwargs):
        ps = self.ps()
        # lbl = ps.get('label', '{name}').format(name=self.name, index=index)
        lbl = ps.get("label", "{name}").format(name=ps.get("name", ""), index=index)
        c = ps.get("color")
        if not (type(c) is str or type(c) is np.str_ or type(c) is tuple):
            try:
                c = c[index]
            except:
                pass
        if c:
            kwargs["color"] = c

        try:
            axis = self.axes[axis]
        except:
            pass
        return plt.hist(self[axis].values, bins=bins, label=lbl, **kwargs)

    def plot_color(self, crange="auto_max", cmap="seismic", calpha=1):
        if crange == "auto_max":
            max = np.max(np.abs(self[self.axes[2]].values))
            crange = (-max, max)
        elif crange == "auto_mean":
            mean = np.mean(np.abs(self[self.axes[2]].values)) * 5
            crange = (-mean, mean)
        if np.inf in crange or -np.inf in crange:
            if abs(crange[0]) == np.inf:
                low = np.min(self[self.axes[2]].values)
            else:
                low = crange[0]
            if abs(crange[1]) == np.inf:
                high = np.max(self[self.axes[2]].values)
            else:
                high = crange[1]
            crange = (low, high)
        p = plt.pcolormesh(
            self[self.axes[0]].values,
            self[self.axes[1]].values,
            self[self.axes[2]].values,
            cmap=cmap,
            vmin=crange[0],
            vmax=crange[1],
            alpha=calpha,
            rasterized=True,
        )
        try:
            return p
        except:
            return None

    def plot_3d(self, zrange=(0, 0), cmap="seismic"):
        if zrange[0] == 0 and zrange[1] == 0:
            max = np.max(np.abs(self[self.axes[2]].values))
            zrange = (-max, max)
        p = plt.gca().plot_surface(
            self[self.axes[0]].values,
            self[self.axes[1]].values,
            self[self.axes[2]].values,
            cmap=cmap,
        )
        plt.gca().set_zlim3d(zrange[0], zrange[1])
        try:
            return p
        except:
            return None

    ####################################################################################################################
    #  data manipulations methods
    ####################################################################################################################

    @staticmethod
    def reshape(dat, newshape, order="C"):
        return dat.copy()._reshape(newshape, order=order)

    def _reshape(self, newshape, order="C"):
        for key in self._data:
            self._data[key] = np.reshape(self._data[key], newshape, order=order)
        return self

    @staticmethod
    def flatten(dat):
        return dat.copy()._flatten()

    def _flatten(self):
        for key in self._data:
            self._data[key] = self._data[key].flatten()
        return self

    def to_matrix(self, split_on=""):
        if split_on not in self._data:
            split_on = self.axes[0]
        uv = np.unique(self._data[split_on])
        return self.reshape((-1, int(len(self._data[split_on]) / len(uv))))

    @staticmethod
    def interpolate(dat, points):
        return dat.copy()._interpolate(points)

    def _interpolate(self, points):
        if len(self.axes) > 2:
            x = self[self.axes[0]].values[:, 0]
            y = self[self.axes[1]].values[0, :]
            z = self[self.axes[2]].values
            if np.max(points[0]) > np.max(x) or np.min(points[0]) < np.min(x):
                logger.warning("interpolation points exceed the x-axis range")
            if np.max(points[1]) > np.max(y) or np.min(points[1]) < np.min(y):
                logger.warning("interpolation points exceed the y-axis range")
            inter = RegularGridInterpolator((x, y), z)
            try:
                dct = {
                    self.axes[0]: points[0],
                    self.axes[1]: points[1],
                    self.axes[2]: inter(np.array(points).T),
                }
            except:
                dct = {
                    self.axes[0]: points[0],
                    self.axes[1]: points[1],
                    self.axes[2]: np.zeros(len(points[0])),
                }
            self._data = dct
            return self
        elif len(self.axes) == 2:
            x = self[self.axes[0]].values
            y = self[self.axes[1]].values
            dct = {
                self.axes[0]: points,
                self.axes[1]: np.interp(points, x, y),
            }
            self._data = dct
            return self

    @staticmethod
    def resample(d, *args):
        return d.copy()._resample(*args)

    def _resample(self, *args):
        if len(self.axes) > 2:
            if len(args) == 1:
                grid = args[0]
            elif len(args) == 2:
                grid = args
            min = {}
            max = {}
            space = {}
            for index, col in enumerate(self.axes):
                if index >= len(grid):
                    break
                min[col] = np.min(self[col].values)
                max[col] = np.max(self[col].values)
                space[col] = np.linspace(min[col], max[col], grid[index])
            y, x = np.meshgrid(space[self.axes[1]], space[self.axes[0]])

            data_locations = np.vstack(
                (self._data[self.axes[0]].ravel(), self._data[self.axes[1]].ravel())
            ).T
            grid_locations = np.vstack((x.ravel(), y.ravel())).T
            grid_data = scipy.interpolate.griddata(
                data_locations,
                self._data[self.axes[2]].ravel(),
                grid_locations,
                method="nearest",
            )

            dct = {
                self.axes[0]: x,
                self.axes[1]: y,
                self.axes[2]: np.reshape(grid_data, grid),
            }
            self._data = dct
        else:
            size = args[0]
            x = np.linspace(
                np.min(self._data[self.axes[0]]), np.max(self._data[self.axes[0]]), size
            )
            y = np.interp(x, self._data[self.axes[0]], self._data[self.axes[1]])
            dct = {
                self.axes[0]: x,
                self.axes[1]: y,
            }
            self._data = dct
        return self

    @staticmethod
    def smooth(d, *args, **kwargs):
        return d.copy()._smooth(**kwargs)

    def _smooth(self, method="gaussian"):
        for key in self._data:
            d = self._data[key]
            m = re.match(r"^medfilt((\d)+_(\d)+)?$", method)
            if m:
                try:
                    d = medfilt(d, (int(m.group(2)), int(m.group(3))))
                except:
                    d = medfilt(d)
            m = re.match(r"^gaussian(\d)*$", method)
            if m:
                try:
                    d = gaussian_filter(d, sigma=float(m.group(1)))  #
                except:
                    d = gaussian_filter(d, sigma=1)  #
            m = re.match(r"^savgol((\d+)_(\d+))?$", method)
            if m:
                if len(d.shape) > 1:
                    try:
                        d = np.array(
                            list(
                                map(
                                    lambda x: savgol_filter(
                                        x, int(m.group(2)), int(m.group(3))
                                    ),
                                    d,
                                )
                            )
                        )
                    except:
                        d = np.array(list(map(lambda x: savgol_filter(x, 9, 3), d)))
                else:
                    try:
                        d = savgol_filter(d, int(m.group(2)), int(m.group(3)))
                    except:
                        d = savgol_filter(d, 9, 3)
            self._data[key] = d
        return self

    @staticmethod
    def derive(d, **kwargs):
        return d.copy()._derive(**kwargs)

    def _derive(self, x=None, method="gradient"):
        if hasattr(x, "values"):
            x = x.values
        if not hasattr(x, "__len__") or not len(x):
            x = np.array([1, 0])
        if len(x.shape) > 1:
            try:
                dx = x[0, :][1] - x[0, :][0]
            except:
                dx = 0.0
            axis = 1
            if dx == 0.0:
                dx = x[:, 0][1] - x[:, 0][0]
                axis = 0
            dct = {}
            for key in self._data:
                if method == "gradient":
                    dct[key] = np.gradient(self._data[key], axis=axis) / dx
                elif "savgol" in method:
                    m = re.match(r"^savgol((\d+)_(\d+))$", method)
                    if m:
                        dct[key] = savgol_filter(
                            self._data[key],
                            int(m.group(2)),
                            int(m.group(3)),
                            deriv=1,
                            delta=dx,
                            axis=axis,
                        )
                    else:
                        dct[key] = savgol_filter(
                            self._data[key], 9, 3, deriv=1, delta=dx, axis=axis
                        )
            self._data = dct
        elif x.shape != self._data[next(iter(self._data))].shape:
            raise RuntimeError("Shape mismatch between x and y")
        else:
            dx = x[1] - x[0]
            dct = {}
            for key in self._data:
                if method == "gradient":
                    dct[key] = np.gradient(self._data[key]) / dx
                elif "savgol" in method:
                    m = re.match(r"^savgol((\d+)_(\d+))$", method)
                    if m:
                        dct[key] = savgol_filter(
                            self._data[key],
                            int(m.group(2)),
                            int(m.group(3)),
                            deriv=1,
                            delta=dx,
                        )
                    else:
                        dct[key] = savgol_filter(
                            self._data[key], 9, 3, deriv=1, delta=dx
                        )

            self._data = dct
        return self

    def fit(
        self,
        func,
        p0,
        bounds=None,
        ignore_error=True,
        return_dict=False,
        fix=None,
        followprogress=False,
    ):
        # get the parameter list from the function. It is important that the function has a signature. C functions will
        # not work and require a lambda wrap: lambda x,a,b: cfunc(x,a,b)
        param_list = list(signature(func).parameters)[1:]
        return_keys = param_list.copy()
        # transfer p0 from a dict to an array
        if type(p0) is dict:
            tp0 = []
            for key in param_list:
                if p0 and key in p0:
                    tp0.append(p0[key])
                else:
                    tp0.append(1)
            p0 = tp0
        # initialise the fixed variables
        if not fix:
            fix = []
        param_list_fixed = param_list.copy()
        # loop through the fixed parameters and create argument lists
        irm = []
        for arg in fix:
            # remove from the fixed list (we don't want fixed arguments as parameters to our fit function)
            param_list_fixed.remove(arg)
            # and replace it by its p0 value in the parameter list that is passed to the original function
            i = param_list.index(arg)
            param_list[i] = str(p0[i])
            irm.append(i)
        # remove the fixed values from p0
        irm = np.sort(irm)[::-1]
        for i in irm:
            del p0[i]

        # create a new fit function by exec (ugly, but I don't know another way)
        if followprogress:
            ldct = locals()
            exec(
                "def g(fnc,xd,yd): \n"
                '\tl1, = plt.plot(xd,yd,marker="o")\n'
                "\tl2, = plt.plot(xd,np.zeros(len(xd)))\n"
                # '\tprint(xd,yd)\n'
                "\tdef f(x," + ", ".join(param_list_fixed) + "):\n"
                "\t\ty = fnc(x," + ", ".join(param_list) + ")\n"
                # \t\tplt.ion()\n'
                # '\t\tplt.close()\n'
                # '\t\tplt.clear\n'
                "\t\tl2.set_ydata(y)\n"
                "\t\tplt.draw()\n"
                "\t\tplt.pause(0.1)\n"
                "\t\ttime.sleep(0.1)\n"
                "\t\treturn y\n"
                "\treturn f\t",
                globals(),
                ldct,
            )
            xdata = self._data[self.axes[0]]
            ydata = self._data[self.axes[1]]
            fitfunc = ldct["g"](func, xdata, ydata)
        else:
            ldct = locals()
            exec(
                "def g(fnc): return lambda x,"
                + ", ".join(param_list_fixed)
                + ": fnc(x,"
                + ", ".join(param_list)
                + ")",
                globals(),
                ldct,
            )
            fitfunc = ldct["g"](func)

        if bounds:
            # bounds can also be supplied as a dictionary instead of a list
            if type(bounds) is dict:
                tbounds = [[], []]
                for i, key in enumerate(param_list_fixed):
                    tbounds[0].append(
                        bounds[key][0] if bounds[key][0] != "p0" else p0[i]
                    )
                    tbounds[1].append(
                        bounds[key][1] if bounds[key][1] != "p0" else p0[i]
                    )
                bounds = tbounds
        if len(self.axes) == 2:
            xdata = self._data[self.axes[0]]
            ydata = self._data[self.axes[1]]
            if ignore_error:
                try:
                    if bounds:
                        params, _ = curve_fit(
                            fitfunc, xdata, ydata, bounds=bounds, p0=p0
                        )
                        r2 = np.corrcoef(ydata, fitfunc(xdata, *params))[0][1] ** 2
                    else:
                        params, _ = curve_fit(fitfunc, xdata, ydata, p0=p0)
                        r2 = np.corrcoef(ydata, fitfunc(xdata, *params))[0][1] ** 2
                except RuntimeError:
                    params = np.zeros(len(p0))
                    r2 = 0.0
                except ZeroDivisionError:
                    params = np.zeros(len(p0))
                    r2 = 0.0
            else:
                if bounds:
                    params, _ = curve_fit(fitfunc, xdata, ydata, bounds=bounds, p0=p0)
                    r2 = np.corrcoef(ydata, fitfunc(xdata, *params))[0][1] ** 2
                else:
                    params, _ = curve_fit(fitfunc, xdata, ydata, p0=p0)
                    r2 = np.corrcoef(ydata, fitfunc(xdata, *params))[0][1] ** 2
            try:
                dct = {
                    self.axes[0]: self._data[self.axes[0]],
                    self.axes[1]: np.array(fitfunc(self._data[self.axes[0]], *params)),
                }
            except:
                dct = {
                    self.axes[0]: self._data[self.axes[0]],
                    self.axes[1]: np.array(np.zeros(len(self._data[self.axes[0]]))),
                }
            fit_curve = self.copy(dct)
            return_dct = {}
            return_lst = []
            i = 0
            for key, val in zip(return_keys, param_list, strict=False):
                if key in param_list_fixed:
                    return_lst.append(params[i])
                    return_dct[key] = params[i]
                    i += 1
                else:
                    return_lst.append(float(val))
                    return_dct[key] = float(val)
            if return_dict:
                return return_dct, r2, fit_curve
            return return_lst, r2, fit_curve

        elif len(self.axes) == 3:
            xdata = np.array(self._data[self.axes[0]]).flatten()
            ydata = np.array(self._data[self.axes[1]]).flatten()
            zdata = np.array(self._data[self.axes[2]]).flatten()
            if ignore_error:
                try:
                    if bounds:
                        params, _ = curve_fit(
                            fitfunc, (xdata, ydata), zdata, bounds=bounds, p0=p0
                        )
                        r2 = (
                            np.corrcoef(zdata, fitfunc((xdata, ydata), *params))[0][1]
                            ** 2
                        )
                    else:
                        params, _ = curve_fit(fitfunc, (xdata, ydata), zdata, p0=p0)
                        r2 = (
                            np.corrcoef(zdata, fitfunc((xdata, ydata), *params))[0][1]
                            ** 2
                        )
                except RuntimeError:
                    params = np.zeros(len(p0))
                    r2 = 0.0
            else:
                if bounds:
                    logger.debug(
                        "fitting %d x-values and %d y-values", len(xdata), len(ydata)
                    )
                    params, _ = curve_fit(
                        fitfunc, (xdata, ydata), zdata, bounds=bounds, p0=p0
                    )
                    r2 = np.corrcoef(zdata, fitfunc((xdata, ydata), *params))[0][1] ** 2
                else:
                    params, _ = curve_fit(fitfunc, (xdata, ydata), zdata, p0=p0)
                    r2 = np.corrcoef(zdata, fitfunc((xdata, ydata), *params))[0][1] ** 2
            dct = {
                self.axes[0]: np.array(self._data[self.axes[0]]),
                self.axes[1]: np.array(self._data[self.axes[1]]),
                self.axes[2]: np.array(fitfunc((xdata, ydata), *params)),
            }
            fit_curve = self.copy(dct)
            return_dct = {}
            return_lst = []
            i = 0
            for key, val in zip(return_keys, param_list, strict=False):
                if key in param_list_fixed:
                    return_lst.append(params[i])
                    return_dct[key] = params[i]
                    i += 1
                else:
                    return_lst.append(float(val))
                    return_dct[key] = float(val)
            if return_dict:
                return return_dct, r2, fit_curve
            return return_lst, r2, fit_curve
        else:
            raise RuntimeError("Cannot fit data, axes are ambiguous.")

    def save(self, filename, sep="\t"):
        """Write the selected axes to a delimited text file."""
        path = os.fspath(filename)
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        columns = {axis: self._data[axis].flatten() for axis in self.axes}
        pd.DataFrame(columns).to_csv(path, sep=sep, index=False)


class Cyclic_Data(Data):
    def __init__(self, dat, **kwargs):
        self.cycle_to_trace = self._cycle_to_trace
        self.average_cycles = self._average_cycles
        self.label_cycles = self._label_cycles
        super().__init__(dat, **kwargs)

    @staticmethod
    def cycle_to_trace(d, **kwargs):
        return d.copy()._cycle_to_trace(**kwargs)

    def _cycle_to_trace(self, cyclic_axis="", method="average"):
        if cyclic_axis not in self.axes:
            cyclic_axis = self.axes[0]
        shape = self._data[cyclic_axis].shape
        if len(shape) > 1:
            dct = {}
            l = int(shape[1] / 2)
            if "forward" in method or "backward" in method:
                l += 1
            for key in self._data:
                dct[key] = np.empty((shape[0], l))
            for i, d in enumerate(self._data[cyclic_axis]):
                mn = np.argmin(d)
                for key in self._data:
                    d = np.roll(self._data[key][i, :], -mn)
                    # print(d)
                    if method == "average":
                        dct[key][i, :] = (d[:l] + d[l:][::-1]) / 2
                    elif "forward" in method:
                        dct[key][i, :] = d[:l]
                    elif "backward" in method:
                        dct[key][i, :] = d[::-1][:l]
                    elif "min" in method:
                        dct[key][i, :] = np.min(np.array([d[:l], d[l:][::-1]]), axis=0)
                    elif "max" in method:
                        dct[key][i, :] = np.max(np.array([d[:l], d[l:][::-1]]), axis=0)
                    elif "rise" in method:
                        dct[key][i, :] = np.append(
                            d[int(l / 2) : int(l)], d[int(l * 1.5) :][::-1]
                        )
                    elif "fall" in method:
                        dct[key][i, :] = np.append(
                            d[: int(l / 2)][::-1], d[l : int(l * 1.5)]
                        )
            self._data = dct
        else:
            mn = np.argmin(self._data[cyclic_axis])
            for key in self._data:
                d = np.roll(self._data[key], -mn)
                l = int(len(d) / 2)
                if method == "average":
                    self._data[key] = (d[:l] + d[l:][::-1]) / 2
                if "forward" in method:
                    self._data[key] = d[:l]
                if "backward" in method:
                    self._data[key] = d[l:][::-1]
                if "min" in method:
                    self._data[key] = np.min(np.array([d[:l], d[l:][::-1]]), axis=0)
                if "max" in method:
                    self._data[key] = np.max(np.array([d[:l], d[l:][::-1]]), axis=0)
        return self

    @staticmethod
    def average_cycles(d, **kwargs):
        return d.copy()._average_cycles(**kwargs)

    def _average_cycles(self, cyclic_axis="", ignore_first=False):
        if cyclic_axis not in self.axes:
            cyclic_axis = self.axes[0]
        shape = self._data[cyclic_axis].shape
        if len(shape) > 1:
            dct = {}
            for i in range(shape[0]):
                d = {}
                for key in self._data:
                    d[key] = self._data[key][i, :]
                cd = d[cyclic_axis]
                l = (
                    np.argmin(cd[(np.argmin(cd) + 2) :]) + 2
                )  # find the cycle length by finding the minimum of 2 consecutive cycles (hopefully this works well for stupid data too...)
                for key in d:
                    rs = np.reshape(d[key], (-1, l))
                    if ignore_first:
                        rs = rs[1:, :]
                    if i == 0:
                        # print(np.reshape(d[key], (-1, l)).mean(axis=0))
                        dct[key] = rs.mean(axis=0)
                    else:
                        dct[key].append(rs.mean(axis=0))
            for key in dct:
                self._data[key] = np.array(dct[key])
            # raise RuntimeError('Averaging the cycles in a matrix is not implemented.')
        else:
            d = self._data[cyclic_axis]
            l = (
                np.argmin(d[(np.argmin(d) + 1) :]) + 1
            )  # find the cycle length by finding the minimum of 2 consecutive cycles (hopefully this works well for stupid data too...)
            dct = {}
            if l > 2:
                for key in self._data:
                    rs = np.reshape(self._data[key], (-1, l))
                    if ignore_first:
                        rs = rs[1:, :]
                    dct[key] = rs.mean(axis=0)
                self._data = dct

        return self

    def split_to_dataset(self, split_row=""):
        if not split_row:
            split_row = self.axes[0]
        un = np.unique(self._data[split_row])
        lst = []
        for u in un:
            dat = self[self[split_row] == u]
            dat.axes = [d for d in self.axes]
            dat.axes.remove(split_row)
            del dat._data[split_row]
            dct = {split_row: u, "data": dat}
            lst.append(dct)
        return Dataset(*lst)

    @staticmethod
    def label_cycles(d, **kwargs):
        return d.copy()._label_cycles(**kwargs)

    def _label_cycles(self, cyclic_axis="", to_axis="n"):
        if cyclic_axis not in self.axes:
            cyclic_axis = self.axes[0]
        shape = self._data[cyclic_axis].shape
        if len(shape) > 1:
            raise RuntimeError("Labelling the cycles in a matrix is not implemented.")
        else:
            d = self._data[cyclic_axis]
            l = (
                np.argmin(d[(np.argmin(d) + 1) :]) + 1
            )  # find the cycle length by finding the minimum of 2 consecutive cycles (hopefully this works well for stupid data too...)
            n = np.repeat(np.arange(int(len(d) / l)), l)  # create the n label
            self._data[to_axis] = n
            if (
                len(self.axes) < 3
            ):  # if we only have two axes, then add the n label as a third
                self.axes = [*self.axes, to_axis]
        return self


class Dataset:
    def __str__(self):
        s = ""
        s += "\t".join([key for key in self._dct]) + "\n"
        key = next(iter(self._dct))
        # print([self._dct[key][0] for key in self._dct])
        for ln in range(len(self._dct[key])):
            s += "\t".join([str(self._dct[key][ln]) for key in self._dct]) + "\n"
        # for key in self._dct:
        #     str+='{}\t{}\n'.format(key,self._dct[key])
        return s

    def __len__(self):
        return len(self._dct[next(iter(self._dct))])

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __contains__(self, item):
        return item in self._dct

    def __add__(self, other):
        if type(other) is type(self):
            dct = {}
            # get keys on both objects
            keys = []
            for key in self._dct:
                if key not in keys:
                    keys.append(key)
            try:
                l1 = len(self._dct[key])
            except:
                l1 = 0
            for key in other._dct:
                if key not in keys:
                    keys.append(key)
            try:
                l2 = len(other._dct[key])
            except:
                l2 = 0
            # add the keys to the new object
            for key in keys:
                if key in self._dct:
                    dct[key] = self._dct[key]
                else:
                    dct[key] = [0] * l1
                if type(dct[key]) is np.ndarray:
                    if key in other._dct:
                        dct[key] = np.append(dct[key], other._dct[key])
                    else:
                        dct[key] = np.append(dct[key], np.zeros(l2))
                else:
                    if key in other._dct:
                        dct[key].extend(other._dct[key])
                    else:
                        dct[key].extend([0] * l2)
            dset = copy.deepcopy(self)
            dset._dct = dct
            return dset

    def __setitem__(self, key, value):
        if isinstance(key, str) and (
            isinstance(value, list) or isinstance(value, np.ndarray)
        ):
            ln = len(self._dct[next(iter(self._dct))])
            if ln == len(value):
                self._dct[key] = value
            else:
                raise RuntimeError("Erroneous item assignment on dataset.")
        else:
            raise RuntimeError("Erroneous item assignment on dataset.")

    def __getitem__(self, item):
        if type(item) is bool:
            if item:
                return self
            else:
                return None

        if type(item) is str:
            r = self._dct[item]
            if not r:
                return r
            if not isinstance(self._dct[item][0], Data):
                r = np.array(r)
            if len(r) == 1:
                return r[0]
            return r

        if type(item) is np.ndarray:
            dct = {}
            for key in self._dct:
                dct[key] = list(np.array(self._dct[key], dtype=object)[item])
            dset = copy.deepcopy(
                self
            )  # if you make changes to self, dset completely behaves independetly
            dset._dct = dct
            return dset

        if type(item) is slice:
            dct = {}
            for key in self._dct:
                dct[key] = list(np.array(self._dct[key], dtype=object)[item])
            dset = copy.deepcopy(self)
            dset._dct = dct
            return dset

        if type(item) is int or np.issubdtype(item, int):
            dct = {}
            for key in self._dct:
                dct[key] = [self._dct[key][item]]
            dset = copy.deepcopy(self)
            dset._dct = dct
            return dset

    def __init__(self, *args, **kwargs):
        self._dct = {}
        for arg in args:
            if type(arg) is dict:
                for key in arg:
                    try:
                        self._dct[key].append(arg[key])
                    except:
                        self._dct[key] = [arg[key]]

    @classmethod
    def find(
        cls, directory=".", extensions=("dat", "txt", "csv", "mat", "asc"), pattern=""
    ):
        lst = []
        id = 0
        for root, _dir, files in os.walk(directory):
            for file in files:
                filename = os.path.join(root, file).replace("\\", "/")
                ext = filename.split(".")[-1]
                if ext not in extensions:
                    continue
                m = re.match(pattern, filename)
                if m:
                    dct = m.groupdict()
                    dct["timestamp"] = os.path.getctime(filename)
                    dct["filename"] = filename
                    dct["id"] = id
                    lst.append(dct)
                    id += 1
        return cls(*lst)

    def load(self, cls=Cyclic_Data, **kwargs):
        return_lst = []
        for file in self._dct["filename"]:
            return_lst.append(cls.load_from_file(file, **kwargs))
        self._dct["data"] = return_lst
        return return_lst

    @staticmethod
    def _residual(params, dataset, func):
        x = np.array([])
        y = np.array([])
        for index, dat in enumerate(dataset["data"]):
            if not len(x) and not len(y):
                x = np.empty((len(dataset), len(dat[dat.axes[0]].values)))
                y = np.empty((len(dataset), len(dat[dat.axes[0]].values)))
                resid = np.empty(x.shape)
            x[index, :] = dat[dat.axes[0]].values.flatten()
            y[index, :] = dat[dat.axes[-1]].values.flatten()

        # make residual per data set
        sig = list(signature(func).parameters)[1:]
        param_list = list(params)

        try:
            lines = dataset["fitlines"]
        except:
            lines = []
        for i in range(len(dataset)):
            p = []
            for key in sig:
                if key in param_list:
                    p.append(params[key])
                else:
                    p.append(params[f"{key}_{i}"])
            yfunc = func(x[i, :], *p)
            if len(lines):
                lines[i].set_data(x[i, :], yfunc)
                plt.draw()
                plt.pause(0.1)

            resid[i, :] = y[i, :] - yfunc
        return resid.flatten()

    def global_fit(
        self, func, p0, bounds, return_dict=True, fix=None, followprogress=False
    ):
        # create 5 sets of parameters, one per data set
        if not minimize:
            raise RuntimeError(
                "lmfit package not installed, cannot perform global fit."
            )
        sig = list(signature(func).parameters)[1:]

        if followprogress:
            lns = []
            for d in self["data"]:
                plt.plot(
                    d[d.axes[0]].values, d[d.axes[1]].values, marker="o", linewidth=0
                )
                (ln,) = plt.plot(0, 0)
                lns.append(ln)
            plt.draw()
            plt.pause(0.1)
            self["fitlines"] = lns

        if not fix:
            fix = []
        if type(p0) is dict:
            tp0 = []
            for key in sig:
                tp0.append(p0[key])
            p0 = tp0
        if type(bounds) is dict:
            tbounds = [[], []]
            for key in sig:
                tbounds[0].append(bounds[key][0])
                tbounds[1].append(bounds[key][1])
            bounds = tbounds
        fit_params = Parameters()
        for i, p in enumerate(p0):
            if hasattr(p, "__len__"):
                # non-shared parameter
                for j, p_ in enumerate(p):
                    min = bounds[0][i]
                    if hasattr(min, "__len__"):
                        min = min[j]
                    max = bounds[1][i]
                    if hasattr(max, "__len__"):
                        max = max[j]
                    fit_params.add(
                        f"{sig[i]}_{j}",
                        value=p_,
                        min=min,
                        max=max,
                        vary=False if sig[i] in fix else True,
                    )
            else:
                # shared parameter
                fit_params.add(
                    sig[i],
                    value=p,
                    min=bounds[0][i],
                    max=bounds[1][i],
                    vary=False if sig[i] in fix else True,
                )
        out = minimize(Dataset._residual, fit_params, args=(self, func))
        if "fit" in self._dct:
            logger.info("overwriting existing fit data")
        self._dct["fit"] = []
        param_list = list(out.params)
        for i, data in enumerate(self["data"]):
            # make residual per data set
            p = []
            for key in sig:
                if key in param_list:
                    p.append(out.params[key])
                else:
                    p.append(out.params[f"{key}_{i}"])
            dat = data.copy()
            dat[dat.axes[-1]] = func(dat[dat.axes[0]].values, *p)
            dat.ps(linewidth=1, color="k", marker=None, label="")
            self._dct["fit"].append(dat)
        return_dct = {}
        for key in sig:
            if key in param_list:
                return_dct[key] = out.params[key].value
            else:
                return_dct[key] = [
                    out.params[f"{key}_{i}"].value for i in range(len(self))
                ]
        # run the global fit to all the data sets
        return return_dct if return_dict else out


# PEP 8 name for new code; the original name remains available to notebooks.
CyclicData = Cyclic_Data


__all__ = [
    "CyclicData",
    "Cyclic_Data",
    "Data",
    "Dataset",
    "Figure",
    "Subplot",
    "add_metric_prefix",
    "generate_color_list",
]
