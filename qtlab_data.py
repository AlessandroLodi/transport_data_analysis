import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.pylab as pl
from matplotlib.cm import get_cmap
import matplotlib.mlab as mlab
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
import re
import scipy.interpolate
from scipy.optimize import curve_fit
from matplotlib.colors import to_rgba
import re
from collections import OrderedDict

from .dataclass import *
from .physics_models import *

# from dataclass import *
# from physics_models import *
import calendar


class Subplot_IVg(Subplot):
    def __init__(self, *args, **kwargs):
        kwargs["axis_labels"] = kwargs.get(
            "axis_labels",
            (
                r"$V_{\rm{g}}$ $\rm{({metric_prefix}V)}$",
                r"$I_{\rm{b}}$ $\rm{({metric_prefix}A)}$",
            ),
        )
        super().__init__(*args, **kwargs)


class Subplot_IV(Subplot):
    def __init__(self, *args, **kwargs):
        kwargs["axis_labels"] = kwargs.get(
            "axis_labels",
            (
                r"$V_{\rm{b}}$ $\rm{({metric_prefix}V)}$",
                r"$I_{\rm{b}}$ $\rm{({metric_prefix}A)}$",
            ),
        )
        super().__init__(*args, **kwargs)


class Subplot_GVg(Subplot):
    def __init__(self, *args, **kwargs):
        kwargs["axis_labels"] = kwargs.get(
            "axis_labels",
            (
                r"$V_{\rm{g}}$ $\rm{({metric_prefix}V)}$",
                r"$G_{\rm{b}}$ $\rm{({metric_prefix}S)}$",
            ),
        )
        super().__init__(*args, **kwargs)


class Subplot_GV(Subplot):
    def __init__(self, *args, **kwargs):
        kwargs["axis_labels"] = kwargs.get(
            "axis_labels",
            (
                r"$V_{\rm{b}}$ $\rm{({metric_prefix}V)}$",
                r"$G_{\rm{b}}$ $\rm{({metric_prefix}S)}$",
            ),
        )
        super().__init__(*args, **kwargs)


class Subplot_IVsVg(Subplot):
    def __init__(self, *args, **kwargs):
        kwargs["type"] = kwargs.get("type", "color")
        kwargs["axis_labels"] = kwargs.get(
            "axis_labels",
            (
                r"$V_{\rm{g}}$ $\rm{({metric_prefix}V)}$",
                r"$V_{\rm{b}}$ $\rm{({metric_prefix}V)}$",
                r"$I_{\rm{b}}$ $\rm{({metric_prefix}A)}$",
            ),
        )
        kwargs["crange"] = kwargs.get("crange", "auto_mean")
        super().__init__(*args, **kwargs)


class Subplot_GVsVg(Subplot):
    def __init__(self, *args, **kwargs):
        datasets = []
        if kwargs.get("derive_data", True):
            for arg in args:
                d = arg.copy()
                try:
                    del d["Gsd"]
                except:
                    pass
                d["Gsd"] = d["Isd"].derive(
                    x=d["Vsd"], method=kwargs.get("method", "savgol")
                )
                d.axes = ("Vg", "Vsd", "Gsd")
                d.ps(parent=arg)
                datasets.append(d)
        else:
            datasets = args
        try:
            del kwargs["derive_data"]
        except:
            pass
        kwargs["type"] = kwargs.get("type", "color")
        kwargs["axis_labels"] = kwargs.get(
            "axis_labels",
            (
                r"$V_{\rm{g}}$ $\rm{({metric_prefix}V)}$",
                r"$V_{\rm{b}}$ $\rm{({metric_prefix}V)}$",
                r"$G_{\rm{b}}$ $\rm{({metric_prefix}S)}$",
            ),
        )
        kwargs["crange"] = kwargs.get("crange", "auto_mean")
        super().__init__(*datasets, **kwargs)


class Subplot_FitAlpha(Subplot_GVsVg):
    def __init__(self, *args, **kwargs):
        def add_lines(subplot):
            isd = subplot.get_data()[0]["Vsd"]
            Vs = [np.min(isd), np.max(isd)]
            Vg1 = list(map(self._s2g, Vs))  # mu = Vs
            Vg1 = np.array(Vg1) + self.vc
            Vg2 = list(map(self._d2g, Vs))  # mu = 0
            Vg2 = np.array(Vg2) + self.vc
            plt.plot(Vg1, Vs, label="mu = Vs")
            plt.plot(Vg2, Vs, label="mu = 0 V")
            plt.legend()

        def onload(subplot):
            dat = subplot.get_data()[0]
            ps = dat.ps("Vc", "alpha_gate", "alpha_source")
            vc = ps["Vc"]
            subplot["ax"].plot(vc, 0, "x", color="black")
            if "alpha_gate" in ps and "alpha_source" in ps:
                Vs = np.array([np.min(dat["Vsd"].values), np.max(dat["Vsd"].values)])
                Vg_s = (Vs * (1.0 - ps["alpha_source"]) / ps["alpha_gate"]) + vc
                Vg_d = (-Vs * ps["alpha_source"] / ps["alpha_gate"]) + vc
                (self._drainline,) = subplot["ax"].plot(Vg_d, Vs)
                (self._sourceline,) = subplot["ax"].plot(Vg_s, Vs)

        def onclick(subplot, event):
            try:
                dat = subplot.get_data()[0].ps("parent")["parent"]
            except KeyError:
                dat = subplot.get_data()[0]
            vc = dat.ps("Vc")["Vc"]
            x, y = (event.xdata, event.ydata)
            x -= vc
            isdrain = 0 if x < 0 else 1
            isdrain ^= 0 if y < 0 else 1
            ymax = np.max(dat["Vsd"].values)
            ymin = np.min(dat["Vsd"].values)
            if isdrain:
                self._drainslope = y / x
                try:
                    self._drainline.set_data(
                        [
                            vc + (ymin / self._drainslope),
                            vc + (ymax / self._drainslope),
                        ],
                        [ymin, ymax],
                    )
                except:
                    (self._drainline,) = subplot["ax"].plot(
                        [
                            vc + (ymin / self._drainslope),
                            vc + (ymax / self._drainslope),
                        ],
                        [ymin, ymax],
                    )
            else:
                self._sourceslope = y / x
                try:
                    self._sourceline.set_data(
                        [
                            vc + (ymin / self._sourceslope),
                            vc + (ymax / self._sourceslope),
                        ],
                        [ymin, ymax],
                    )
                except:
                    (self._sourceline,) = subplot["ax"].plot(
                        [
                            vc + (ymin / self._sourceslope),
                            vc + (ymax / self._sourceslope),
                        ],
                        [ymin, ymax],
                    )
            try:
                dat.ps(alpha_source=1 / (1 - self._drainslope / self._sourceslope))
                dat.ps(
                    alpha_gate=-self._drainslope
                    / (1 - self._drainslope / self._sourceslope)
                )
            except:
                pass

        kwargs["onclick"] = kwargs.get("onclick", onclick)
        kwargs["onload"] = kwargs.get("onload", onload)
        if len(args) != 1:
            return
        super().__init__(args[0], **kwargs)


class Subplot_FitVc(Subplot_GVsVg):
    def __init__(self, *args, **kwargs):
        def onload(subplot):
            dat = subplot.get_data()[0]
            try:
                vc = dat.ps("Vc")["Vc"]
                (self.vcline,) = subplot["ax"].plot(vc, 0, "x", color="black")
            except:
                pass

        def onclick(subplot, event):
            try:
                dat = subplot.get_data()[0].ps("parent")["parent"]
            except KeyError:
                dat = subplot.get_data()[0]
            try:
                self.vcline.set_data(event.xdata, 0)
            except:
                (self.vcline,) = subplot["ax"].plot(event.xdata, 0, "x", color="black")
            dat.ps(Vc=event.xdata)

        kwargs["onclick"] = kwargs.get("onclick", onclick)
        kwargs["onload"] = kwargs.get("onload", onload)
        if len(args) != 1:
            return
        super().__init__(args[0], **kwargs)


class Subplot_ResonanceBiasTrace(Subplot_IV):
    def __init__(self, *args, **kwargs):
        datasets = []
        for arg in args:
            try:
                datasets.append(arg._resonant_trace)
            except:
                datasets.append(arg.resonance_bias_trace())
        super().__init__(*datasets, **kwargs)


class Subplot_ZeroBiasGateTrace(Subplot):
    def __init__(self, *args, **kwargs):
        datasets = []
        if kwargs.get("subdata", True):
            for arg in args:
                try:
                    datasets.append(arg._gatetrace)
                except:
                    datasets.append(arg.zero_bias_gate_trace())
                try:
                    datasets.append(arg._gatetrace_fit)
                except:
                    pass
        else:
            for arg in args:
                datasets.append(arg)
        kwargs["axis_labels"] = kwargs.get(
            "axis_labels",
            ("$V_g$ ({metric_prefix}V)", r"$G_{b} \rm{({metric_prefix}S)}$"),
        )
        super().__init__(*datasets, **kwargs)


class Subplot_ExcitedStateSpectrum(Subplot):
    def __init__(self, *args, **kwargs):
        peak_threshold = kwargs.get("peak_threshold", 1e-8)

        def peaks(subplot):
            for i, dat in enumerate(subplot.get_data()):
                if kwargs.get("correct_offset", True):
                    dat["G"] += 2e-8 * i
                    peaks = peakdet(
                        dat["G"].values, x=dat["E"].values, delta=peak_threshold
                    )
                    offset = peaks[0][0][0]
                    dat["E"] -= offset
            subplot.set_data(*subplot.get_data())
            for dat in subplot.get_data():
                peaks = peakdet(
                    dat["G"].values, x=dat["E"].values, delta=peak_threshold
                )
                for peak in peaks[0]:
                    if peak[0] < 0.03:
                        subplot["ax"].plot(peak[0], peak[1], "x", color="black")
                        subplot["ax"].text(
                            peak[0],
                            peak[1] * 1.05,
                            "{:.1f}".format(peak[0] * 1e3),
                            horizontalalignment="center",
                            verticalalignment="bottom",
                        )

        kwargs["axis_labels"] = kwargs.get(
            "axis_labels",
            (
                r"$ϵ \/\/ \mathrm{({metric_prefix}eV)}$",
                r"$G_{\mathrm{b}} \/\/ \mathrm{({metric_prefix}S)}$",
            ),
        )
        kwargs["onload"] = kwargs.get("onload", peaks)
        super().__init__(*args, **kwargs)


class Subplot_ShowExcitationLines(Subplot_GVsVg):
    def __init__(self, *args, **kwargs):
        offset = kwargs.get("offset", 0.01)
        polarity = kwargs.get("polarity", "positive")
        direction = kwargs.get("direction", "source")

        def add_lines(subplot):
            dat = subplot.get_data()[0].ps("parent")["parent"]
            o = offset
            if polarity != "positive":
                o = -o
            if direction != "source":
                o = -o
            lines = dat._get_parallel_lines(
                direction=direction, polarity=polarity, num_points=50, interpolation=5
            )
            ps = dat.ps("Vc", "alpha_source", "alpha_gate")
            a = ps["alpha_gate"] / (1 - ps["alpha_source"])
            b = -ps["alpha_gate"] / ps["alpha_source"]
            if direction == "source":
                x1 = o / np.sqrt(a ** 2 + 1)
                x2 = (a / b) * x1
                x3 = x2 - x1
            else:
                x1 = o / np.sqrt(b ** 2 + 1)
                x2 = (b / a) * x1
                x3 = x2 - x1

            for line in zip(lines[0], lines[1]):
                x = line[0] + x3
                y = line[1]
                subplot["ax"].plot(
                    [x[0], x[-1]], [y[0], y[-1]], color=to_rgba("k", 0.5)
                )

        kwargs["onload"] = kwargs.get("onload", add_lines)
        super().__init__(*args, **kwargs)


class QTLab_Data(Cyclic_Data):
    @classmethod
    # this override the methods you can find in the Data class
    def load_from_file(cls, filename, **kwargs):
        dat = None
        ps = {}
        if filename and (type(filename) is str or type(filename) is np.str_):
            if filename.split(".")[-1] in ("dat", "csv", "txt"):
                print(f">>> Loading {filename}")
                # read the header of the filename
                axes = ("x", "y")
                if kwargs.get("readheader", True):
                    # read the entire reader
                    header = ""
                    with open(filename, "r") as f:
                        i = 0
                        while True:
                            i += 1
                            ln = f.readline()
                            if i == 2:
                                ps["timestamp"] = calendar.timegm(
                                    time.strptime(ln[17:37], "%b %d %H:%M:%S %Y")
                                )
                            if len(ln) > 0 and (ln[0] == "#" or ln[0] == "\n"):
                                header += ln
                            else:
                                break
                    axes = re.findall("# Column \d+:[\s\S]+?name: (.+)\n", header)
                    comments = re.findall(
                        "# ([a-zA-Z0-9_]+): ([a-zA-Z0-9_\.]+)\n", header
                    )
                    for comment in comments:
                        try:
                            ps[comment[0]] = float(comment[1])
                        except:
                            ps[comment[0]] = comment[1]

                # set the axes (which might have been read from the file)
                axes = kwargs.get("axes", axes)
                ps["axes"] = axes
                # add qtlab standards to the kwargs for pandas
                # these two if lines add two dic elements
                # {'comment': '#', 'delimiter', '\t'}
                if "comment" not in kwargs:
                    kwargs["comment"] = "#"
                if "delimiter" not in kwargs:
                    kwargs["delimiter"] = "\t"
                # remove non pandas keys:
                pandas_dct = dict(
                    sep=", ",
                    delimiter=None,
                    header="infer",
                    names=None,
                    index_col=None,
                    usecols=None,
                    squeeze=False,
                    prefix=None,
                    mangle_dupe_cols=True,
                    dtype=None,
                    engine=None,
                    converters=None,
                    true_values=None,
                    false_values=None,
                    skipinitialspace=False,
                    skiprows=None,
                    nrows=None,
                    na_values=None,
                    keep_default_na=True,
                    na_filter=True,
                    verbose=False,
                    skip_blank_lines=True,
                    parse_dates=False,
                    infer_datetime_format=False,
                    keep_date_col=False,
                    date_parser=None,
                    dayfirst=False,
                    iterator=False,
                    chunksize=None,
                    compression="infer",
                    thousands=None,
                    decimal=b".",
                    lineterminator=None,
                    quotechar='"',
                    quoting=0,
                    escapechar=None,
                    comment=None,
                    encoding=None,
                    dialect=None,
                    # tupleize_cols=None, deprecated
                    error_bad_lines=True,
                    warn_bad_lines=True,
                    skipfooter=0,
                    # skip_footer=0, deprecated
                    doublequote=True,
                    delim_whitespace=False,
                    low_memory=True,
                    # buffer_lines=None, deprecated
                    memory_map=False,
                    float_precision=None,
                )
                for key in kwargs:
                    # print(f'key in kwargs are {key}')
                    if key in pandas_dct:
                        # print(f'kwargs[key] is: {kwargs[key]}')
                        pandas_dct[key] = kwargs[key]
                for key in pandas_dct:
                    # print(f'key in pandas_dct is {key}')
                    if key in kwargs:
                        del kwargs[key]
                del pandas_dct["names"]
                d = pd.read_csv(filename, names=axes, **pandas_dct)
                dat = {}
                for key in d:
                    if not d[key].empty:
                        dat[key] = np.array(d[key])
                if not dat:
                    print(">> Warning: no data found for {}".format(filename))
            elif filename.split(".")[-1] in ("mat"):
                print(">>>>> Loading {}".format(filename))
                # set the axes (which might have been read from the file)
                axes = kwargs.get("axes", ("x", "y"))
                ps["axes"] = axes
                dct = loadmat(filename)
                dat = {}
                for key in axes:
                    if key in dct and len(dct[key]) > 0:
                        dat[key] = np.array(dct[key][0])
                if not dat:
                    print(
                        ">> Warning: no data found in axes {} for {}. Possible axes: {}".format(
                            axes, filename, [key for key in dct]
                        )
                    )
            else:
                raise RuntimeError(
                    "Currently only supporting .txt, .dat, .csv files as raw data."
                )
        if dat:
            return cls(dat, **{**ps, **kwargs})
        return None


class Stability_Diagram(QTLab_Data):
    def __init__(self, dat, cyclic_method="None", **kwargs):
        super().__init__(dat, **kwargs)
        if "n" not in dat:
            w = np.where(
                dat[kwargs.get("Vg", "Vg")] < np.roll(dat[kwargs.get("Vg", "Vg")], 1)
            )[0]
            print(f"########### {kwargs}")
            c = 0
            n = []
            l = len(n)
            for ind in w[1:]:
                try:
                    l = len(n)
                    n = np.hstack((n, np.ones(ind - l) * c))
                except:
                    n = np.ones(ind - l) * c
                c += 1
            try:
                l = len(n)
                n = np.hstack((n, np.ones(len(dat[kwargs.get("Vg", "Vg")]) - l) * c))
            except:
                n = np.ones(len(dat[kwargs.get("Vg", "Vg")])) * c
            dat["n"] = np.array(n)
        if not self.ps("initialised"):
            self.to_matrix(kwargs.get("Vg", "Vg"))
            if cyclic_method != "None":
                try:
                    self.average_cycles(cyclic_axis="Vsd")
                except:
                    pass
                self.cycle_to_trace(
                    cyclic_axis=kwargs.get("Vsd", "Vsd"), method=cyclic_method
                )
            self.axes = (
                kwargs.get("Vg", "Vg"),
                kwargs.get("Vsd", "Vsd"),
                kwargs.get("Isd", "Isd"),
            )
            self.initialised = True
            self.ps(initialised=True)

    def correct_offset(self, zero=0.1):
        vgs = np.unique(self._data["Vg"])
        for vg in vgs:
            d = self[self["Vg"] == vg]
            imean = np.mean(d[np.abs(d["Vsd"].values) < zero]["Isd"].values)
            d["Isd"] -= imean
            self[self["Vg"].values == vg] = d._data

    def correct_offset_vsd(self, zero=0.01):
        vgs = np.unique(self._data["Vg"])
        for vg in vgs:
            d = self[self["Vg"] == vg]
            # imean = np.mean(d[np.abs(d['Vsd'].values) < zero]['Isd'].values)
            vsd_zero = np.mean(d[np.abs(d["Isd"].values) < zero]["Vsd"].values)
            d["Isd"] -= vsd_zero
            self[self["Vg"].values == vg] = d._data

    def gatetrace(self, vs):
        ddat = self.copy()  # make a copy of the data to play with
        # delete any extra columns.
        try:
            del ddat._data["TimeStamp_s"]
            del ddat._data["MC_Temp_K"]
            del ddat._data["n"]
        except:
            pass

        ddat.flatten()
        # find all vsd voltages
        vsds = np.unique(ddat["Vsd"].values)
        # find the difference between them and the one you want to measure
        diff = vsds - vs
        # pick out the vsd at the minimum
        vsd = vsds[np.argmin(np.abs(diff))]
        # take out only the GT at that vsd
        ddat = ddat[(ddat["Vsd"].values == vsd)]
        vg = ddat["Vg"].values
        # get the unique Vg values, and the inverse array which contains indices for the Vsd elements
        un, inverse, weigths = np.unique(vg, return_inverse=True, return_counts=True)
        # create the multiplication matrix
        matrix = np.zeros((len(un), len(vg)))
        matrix[inverse, np.arange(len(vg))] = 1
        # perform the dot product for all the keys
        for k in ddat._data:
            print(k)
            ddat._data[k] = (
                matrix.dot(np.reshape(ddat[k].values, (-1, 1))).T.flatten() / weigths
            )
        del ddat._data["Vsd"]  # remove averaged column
        ddat.axes = ("Vg", "Isd")
        # self._gatetrace = ddat
        return ddat

    def shift_gatetrace(self, smooth: bool = None):
        ddat = self.copy()
        # print(f"ddat[Isd] = {ddat['Isd']}\n ddat_data['Isd']={ddat._data['Isd']}\n")
        if smooth:
            ddat["Isd"] = savgol_filter(ddat["Isd"].values, 51, 3)
        ddat._data["Isd"] = np.roll(
            ddat["Isd"].values,
            (
                np.argmin(np.abs(ddat["Vg"].values))
                - np.argmin(np.abs(ddat["Isd"].values))
            ),
        )
        # print(f"ddat[Isd] = {ddat['Isd']}\n ddat_data['Isd']={ddat._data['Isd']}\n")
        ddat.axes = ("Vg", "Isd")
        self._gatetrace = ddat
        return ddat

    def gate_trace_derivative(self, vs, x_shift: int):
        ddat = self.copy()  # make a copy of the data to play with
        ddat["Gsd"] = ddat["Isd"].derive(x=ddat["Vsd"])
        ddat.flatten()
        # find all vsd voltages
        vsds = np.unique(ddat["Vsd"].values)
        # find the difference between them and the one you want to measure
        diff = vsds - vs
        # pick out the vsd at the minimum
        vsd = vsds[np.argmin(np.abs(diff))]
        # take out only the GT at that vsd
        ddat = ddat[(ddat["Vsd"].values == vsd)]
        vg = ddat["Vg"].values
        # get the unique Vg values, and the inverse array which contains indices for the Vsd elements
        un, inverse, weigths = np.unique(vg, return_inverse=True, return_counts=True)
        # create the multiplication matrix
        print(f"un = {un}\n vg= {vg}\n")
        matrix = np.zeros((len(un), len(vg)))
        matrix[inverse, np.arange(len(vg))] = 1
        # perform the dot product for all the keys
        for k in ddat._data:
            if k == "Vg":
                # shift Vg so when you take the log it won't cause problem
                ddat._data[k] += x_shift
            ddat._data[k] = (
                matrix.dot(np.reshape(ddat[k].values, (-1, 1))).T.flatten() / weigths
            )
            ddat._data[k] = np.log10(ddat._data[k])

        del ddat._data["Vsd"]  # remove averaged column
        ddat.axes = ("Vg", "Gsd")
        self._gate_trace_derivative = ddat
        return ddat

    def do_derivative(self, axis: str):
        """
        Return the derivative of the dataset
        """
        ddat = self.copy()
        if axis.lower() in "Vg":
            pass

    def bias_trace(self, v_gate):
        ddat = self.copy()
        try:
            del ddat._data["TimeStamp_s"]
            del ddat._data["MC_Temp_K"]
            del ddat._data["n"]
        except:
            pass

        ddat.flatten()
        vg_vals = np.unique(ddat["Vg"].values)
        diff = vg_vals - v_gate
        vg = vg_vals[np.argmin(np.abs(diff))]
        ddat = ddat[(ddat["Vg"].values == vg)]
        vsd = ddat["Vsd"].values

        un, inverse, weights = np.unique(vsd, return_inverse=True, return_counts=True)
        matrix = np.zeros((len(un), len(vsd)))
        print(matrix)
        matrix[inverse, np.arange(len(vsd))] = 1

        for k in ddat._data:
            print(f"this is the key: {k}")
            ddat._data[k] = (
                matrix.dot(np.reshape(ddat[k].values, (-1, 1))).T.flatten() / weights
            )

        del ddat._data["Vg"]
        ddat.axes = ("Vsd", "Isd")
        self._biastrace = ddat
        return ddat

    def resonance_bias_trace(self, width=0.0, centre=37):
        dat = self.copy()
        print(centre)
        u = np.unique(dat._data["Vg"])
        if not width:
            print("hello")
            dat.ps(Vc=centre)
            cvc = u[np.argmin(np.abs(u - dat.ps("Vc")["Vc"]))]
            dat = dat[dat["Vg"].values == cvc]
        else:
            print("world")
            dat = dat[np.abs(dat["Vg"].values - dat.ps("Vc")["Vc"]) < width]
            # average Vsd values for each Vg {
            x = dat["Vsd"].values
            # get the unique Vg values, and the inverse array which contains indices for the Vsd elements
            un, inverse, weights = np.unique(x, return_inverse=True, return_counts=True)
            # create the multiplication matrix (matrix dot x / weights will give un)
            matrix = np.zeros((len(un), len(x)))
            matrix[inverse, np.arange(len(x))] = 1
            # perform the dot product for all keys
            for key in dat._data:
                dat._data[key] = (
                    matrix.dot(np.reshape(dat[key].values, (-1, 1))).T.flatten()
                    / weights
                )
        self._resonant_trace = dat
        dat.axes = ("Vsd", "Isd")
        return dat

    def zero_bias_gate_trace(self, zero=0.003):
        ddat = self.copy()
        ddat["Gsd"] = ddat["Isd"].derive(x=ddat["Vsd"])
        ddat.flatten()
        ddat = ddat[np.abs(ddat["Vsd"].values) < zero]
        # average Vsd values for each Vg {
        x = ddat["Vg"].values
        # get the unique Vg values, and the inverse array which contains indices for the Vsd elements
        un, inverse, weights = np.unique(x, return_inverse=True, return_counts=True)
        # create the multiplication matrix (matrix dot x / weights will give un)
        matrix = np.zeros((len(un), len(x)))
        matrix[inverse, np.arange(len(x))] = 1
        # perform the dot product for all keys
        for key in ddat._data:
            ddat._data[key] = (
                matrix.dot(np.reshape(ddat[key].values, (-1, 1))).T.flatten() / weights
            )
        # }
        del ddat._data["Vsd"]  # remove the averaged column
        ddat.axes = ("Vg", "Gsd")  # set the correct axes
        self._gatetrace = ddat  # save it for later use
        return ddat

    def fit_subthreshold_swing(self, func=None, p0=None, bounds=None, return_dict=True):
        dato = self.copy()
        # ddat.flatten()
        dat = self._data
        Isd = dat["Isd"]
        Vg = dat["Vg"]
        print("**************")
        if not func:
            func = lambda Isd, Vg: np.gradient(np.log10(Isd)) / (Vg[1] - Vg[0])

        if not p0:
            p0 = {}
        if type(p0) is dict:
            if "Gmax" in p0:
                p0["Gmax"] /= np.max(Isd.derive(x=Vg))
            else:
                p0["Gmax"] = 1
        if not bounds:
            params, r2, fit = dato.fit(
                func, p0=p0, bounds=bounds, return_dict=return_dict
            )

        fit.ps(linewidth=1, color="k")
        self._gatetrace_fit = fit
        return params, r2

    def fit_coulomb_peak(self, func=None, p0=None, bounds=None, return_dict=True):
        try:
            dat = self._gatetrace
        except:
            dat = self.zero_bias_gate_trace()
        if not func:
            func = lambda Vg, T, Vc, Gmax, alpha: physics_models.thermal_broadening(
                Vg, self.ps("T")["T"], Vc, Gmax, alpha
            )

        gmax = np.max(dat._data["Gsd"])
        vc = dat._data["Vg"][np.argmax(dat._data["Gsd"])]

        # p0 can be supplied as a dict
        if not p0:
            p0 = {}
        if type(p0) is dict:
            if "Gmax" in p0:
                p0["Gmax"] /= gmax
            if not "Vc" in p0:
                p0["Vc"] = vc
            if not "Gmax" in p0:
                p0["Gmax"] = 1

        dat = dat.copy()
        dat["Gsd"] /= gmax
        if type(bounds) is dict:
            if "Gmax" in bounds:
                bounds["Gmax"][0] /= gmax
                bounds["Gmax"][1] /= gmax
            else:
                bounds["Gmax"] = [0.9, 1.1]
        if bounds:
            params, r2, fit = dat.fit(
                func, p0=p0, bounds=bounds, return_dict=return_dict,
            )
        else:
            params, r2, fit = dat.fit(func, p0=p0, return_dict=return_dict)
        fit["Gsd"] *= gmax
        fit.ps(linewidth=1, color="k")
        self._gatetrace.ps(marker="o")
        self._gatetrace_fit = fit

        try:
            self.ps(Vc=params["Vc"])
            self.ps(alpha_gate=params["alpha"])
        except:
            pass
        return params, r2

    def fit_alphas(self, T=77.0, side="both", return_dict=True):
        try:
            vc = self.ps("Vc")["Vc"]
        except:
            self.fit_coulomb_peak()
            vc = self.ps("Vc")["Vc"]
        dat = self.copy()
        window = np.mean(np.abs(dat["Isd"].values))
        steepness = np.mean(np.abs(dat["Isd"].values)) / 5
        dat["Isd"] = 1 / (1 + np.exp(-(dat["Isd"].values - window) / steepness)) + (
            1 / (1 + np.exp(-(dat["Isd"].values + window) / steepness)) - 1
        )
        self._confined = dat
        sd = lambda data, Vc, alpha_gate, alpha_source: physics_models.stabdiag(
            data, 1, Vc, alpha_gate, alpha_source, T
        )
        alpha = self.ps("alpha_gate")["alpha_gate"]
        if side == "neg":
            dat = dat[dat["Vsd"].values < 0]
        if side == "pos":
            dat = dat[dat["Vsd"].values > 0]
        params, r2, fit = dat.fit(
            sd,
            p0=[vc, alpha, 0.6],
            bounds=([vc - 0.1, 0, 0], [vc + 0.1, 1, 1],),
            ignore_error=False,
            return_dict=False,
        )
        if side != "both":
            fit = dat.copy()
            fit["Isd"] = np.array(sd((fit["Vg"].values, fit["Vsd"].values), *params))
        self.ps(Vc=params[0], alpha_gate=params[1], alpha_source=params[2])
        self._confined_fit = fit
        return params, r2

    def fit_curved_diagram(
        self,
        T=77.0,
        side="both",
        sigma_window=1.0,
        sigma_steepness=5,
        p0=None,
        bounds=None,
        return_dict=True,
    ):
        try:
            vc = self.ps("Vc")["Vc"]
        except:
            self.fit_coulomb_peak()
            vc = self.ps("Vc")["Vc"]
        dat = self.copy()
        window = np.mean(np.abs(dat["Isd"].values)) * sigma_window
        steepness = np.mean(np.abs(dat["Isd"].values)) / sigma_steepness

        vsd = dat["Vsd"].values
        ind = np.where(np.abs(vsd) < 0.005)
        vsd[ind] = 0.005
        vsd = np.abs(vsd)

        dat["Isd"] = 1 / (
            1 + np.exp(-(dat["Isd"].values / vsd - window) / steepness)
        ) + (1 / (1 + np.exp(-(dat["Isd"].values / vsd + window) / steepness)) - 1)
        self._confined = dat
        sd = lambda data, Vc, alpha_gate, alpha_source, alpha_J, J: physics_models.curved_stabdiag(
            data, 1, Vc, alpha_gate, alpha_source, alpha_J, T, J
        )
        alpha = self.ps("alpha_gate")["alpha_gate"]
        if side == "neg":
            dat = dat[dat["Vsd"].values < 0]
        if side == "pos":
            dat = dat[dat["Vsd"].values > 0]
        if not bounds and not p0:
            params, r2, fit = dat.fit(
                sd,
                p0=[vc, alpha, 0.5, 0.5, 0.5],
                bounds=([vc - 40, 0, 0, 0, 0.05], [vc + 40, 1, 1, 1, 1],),
                ignore_error=False,
            )
        else:
            params, r2, fit = dat.fit(sd, p0=p0, bounds=bounds, ignore_error=False,)
        # params = [vc, alpha, 0.8, 0.1]
        # fit=dat.copy()
        print("Fitting data with R^2 = {:.3g}".format(r2))
        if side != "both":
            fit = dat.copy()
            fit["Isd"] = np.array(sd((fit["Vg"].values, fit["Vsd"].values), *params))
        self.ps(Vc=params[0], alpha_gate=params[1], alpha_source=params[2], J=params[3])
        self._confined_fit = fit
        return params, r2, fit

    def _get_parallel_lines(
        self, direction="source", polarity="positive", num_points=10, interpolation=200
    ):
        vsd = self._data["Vsd"]
        ps = self.ps("alpha_source", "alpha_gate", "Vc")
        if polarity == "positive":
            Vs1 = np.linspace(0, np.max(vsd), num_points)
        else:
            Vs1 = np.linspace(np.min(vsd), 0, num_points)
        if direction == "source":
            Vg1 = Vs1 * (1 - ps["alpha_source"]) / ps["alpha_gate"]
        else:
            Vg1 = -Vs1 * ps["alpha_source"] / ps["alpha_gate"]
        if polarity == "positive":
            Vs2 = np.linspace(np.max(vsd), np.max(vsd), num_points)
        else:
            Vs2 = np.linspace(np.min(vsd), np.min(vsd), num_points)
        if direction == "source":
            Vg2 = Vg1 - (Vs2 - Vs1) * ps["alpha_source"] / ps["alpha_gate"]
        else:
            Vg2 = Vg1 + (Vs2 - Vs1) * (1 - ps["alpha_source"]) / ps["alpha_gate"]
        Vg1 += ps["Vc"]
        Vg2 += ps["Vc"]
        Vs = []
        Vg = []
        for x1, y1, x2, y2 in zip(Vg1, Vs1, Vg2, Vs2):
            Vg.append(np.linspace(x1, x2, interpolation))
            Vs.append(np.linspace(y1, y2, interpolation))
        return (Vg, Vs)

    def excited_state_spectrum(
        self,
        direction="source",
        polarity="positive",
        num_points=400,
        interpolation=200,
        offset=0.01,
        reject_outliers=True,
        method="gradient",
    ):
        lines = self._get_parallel_lines(
            direction=direction,
            polarity=polarity,
            num_points=num_points,
            interpolation=interpolation,
        )
        E = []
        G = []
        if direction != "source":
            offset = -offset
        if polarity != "positive":
            offset = -offset
        ps = self.ps("Vc", "alpha_source", "alpha_gate")
        a = ps["alpha_gate"] / (1 - ps["alpha_source"])
        b = -ps["alpha_gate"] / ps["alpha_source"]
        if direction == "source":
            x1 = offset / np.sqrt(a ** 2 + 1)
            y = a * x1
            x2 = (a / b) * x1
            x3 = x2 - x1
        else:
            x1 = offset / np.sqrt(b ** 2 + 1)
            y = b * x1
            x2 = (b / a) * x1
            x3 = x2 - x1
        for line in zip(lines[0], lines[1]):
            line = [line[0] + x3, line[1]]
            E.append(line[1][0] - y)
            dat = self.copy()
            dat["Gsd"] = dat["Isd"].derive(x=dat["Vsd"], method=method)
            dat.axes = ("Vg", "Vsd", "Gsd")
            if reject_outliers:

                # line = (line[1][:-1], line[0][:-1])
                dat.interpolate(line)
                dat = dat[
                    np.abs(dat["Gsd"].values - np.mean(dat["Gsd"].values))
                    < 4 * np.std(dat["Gsd"].values)
                ]
                G.append(dat["Gsd"].values.mean())
            else:
                G.append(np.mean(dat.interpolate(line)["Gsd"].values))
        dct = {
            "E": np.array(E),
            "G": np.array(G),
        }
        dat = Data(dct, axes=("E", "G"))
        self._excited_state_data = dat
        return dat

    def manual_fit_Vc(self, **kwargs):
        fig = Figure()
        fig.add_subplot(Subplot_FitVc(self, **kwargs))
        fig.visualise()
        return self.ps("Vc")["Vc"]

    def manual_fit_alpha(self, **kwargs):
        vc = self.ps("Vc")
        if not vc:
            a, _ = self.fit_coulomb_peak()
            print("mi sto incazzando")
            vc = a["Vc"]
        fig = Figure()
        fig.add_subplot(Subplot_FitAlpha(self, **kwargs))
        fig.visualise()
        return self.ps("Vc", "alpha_source", "alpha_gate")


class ADWin_Stability_Diagram(Stability_Diagram):
    def __init__(
        self,
        dat,
        cyclic_method="None",
        digitise_adwin=True,
        Vsd="Vsd",
        Vg="Vg",
        **kwargs,
    ):
        super().__init__(dat, cyclic_method=cyclic_method, **kwargs)
        if digitise_adwin:

            self.flatten()
            self._data[Vsd] = (
                np.array(list(map(int, ((self._data[Vsd] + 10) * 3276.8))))
                * 0.00030517578125
                - 10
            )

            df = pd.DataFrame(self._data)
            df = df.groupby((Vg, Vsd), as_index=False).mean()
            for key in self._data:
                self._data[key] = np.array(df[key])
            self.to_matrix(Vg)


class QTLab_Dataset(Dataset):
    @classmethod
    def find(
        cls,
        directory=".",
        extensions=("dat", "txt", "csv"),
        pattern=".*\d{6,6}_(?P<exp>[a-zA-Z0-9_]+)_(?P<type>[a-zA-Z0-9\-_]+?)_(?P<device>[a-zA-Z]+[0-9]+)\.(?:dat|csv|txt)",
    ):
        dset = super().find(directory, extensions, pattern)
        dset.inspect_files()
        return dset

    def inspect_files(self):
        infolst = []
        keys = []
        for filename in self._dct["filename"]:
            header = ""
            info = {}
            with open(filename, "r") as f:
                i = 0
                while True:
                    i += 1
                    ln = f.readline()
                    if i == 2:
                        info["timestamp"] = calendar.timegm(
                            time.strptime(ln[17:37], "%b %d %H:%M:%S %Y")
                        )
                    if len(ln) > 0 and (ln[0] == "#" or ln[0] == "\n"):
                        header += ln
                    else:
                        break

            info["axes"] = re.findall("# Column \d+:[\s\S]+?name: (.+)\n", header)
            comments = re.findall("# ([a-zA-Z0-9_]+): ([a-zA-Z0-9_\.]+)\n", header)
            for comment in comments:
                try:
                    info[comment[0]] = float(comment[1])
                except:
                    info[comment[0]] = comment[1]

            for key in info:
                if not key in keys:
                    keys.append(key)

            infolst.append(info)
        for key in keys:
            if key in self._dct:
                print("> Updating info {} in dataset.".format(key))
                del self._dct[key]
        for info in infolst:
            for key in keys:
                if key in self._dct:
                    try:
                        self._dct[key].append(info[key])
                    except:
                        self._dct[key].append(np.nan)
                else:
                    try:
                        self._dct[key] = [info[key]]
                    except:
                        self._dct[key] = [np.nan]
