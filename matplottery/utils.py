from __future__ import print_function, division

import sys
import array
import matplotlib
import numpy as np
import copy

PY2 = True
if sys.version_info[0] >= 3:
    PY2 = False

MET_LATEX = "E$\\!\\!\\! \\backslash{}_\\mathrm{T}$"


def compute_darkness(r, g, b, a=1.0):
    # darkness = 1 - luminance
    return a*(1.0 - (0.299*r + 0.587*g + 0.114*b))


def clopper_pearson_error(passed, total, level=0.6827):
    """
    matching TEfficiency::ClopperPearson()
    """
    import scipy.stats
    alpha = 0.5*(1.-level)
    low = scipy.stats.beta.ppf(alpha, passed, total-passed+1)
    high = scipy.stats.beta.ppf(1 - alpha, passed+1, total-passed)
    return low, high


def fill_fast(hist, xvals, yvals=None, weights=None):
    """
    partially stolen from root_numpy implementation
    using for loop with TH1::Fill() is slow, so use
    numpy to convert array to C-style array, and then FillN
    """
    two_d = False
    if yvals is not None:
        two_d = True
        yvals = array.array("d", yvals)
    if weights is None:
        weights = np.ones(len(xvals))
    xvals = array.array("d", xvals)
    weights = array.array("d", weights)
    if not two_d:
        hist.FillN(len(xvals), xvals, weights)
    else:
        hist.FillN(len(xvals), xvals, yvals, weights)


class TextPatchHandler(object):
    def __init__(self, label_map=None):
        self.label_map = label_map if label_map is not None else {}
        super(TextPatchHandler, self).__init__()

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        label = orig_handle.get_label()
        fc = orig_handle.get_facecolor()
        ec = orig_handle.get_edgecolor()
        lw = orig_handle.get_linewidth()
        color = "w" if (compute_darkness(*fc) > 0.45) else "k"
        text = self.label_map.get(label,"")
        patch1 = matplotlib.patches.Rectangle([x0, y0], width, height, facecolor=fc, edgecolor=ec, linewidth=lw,
                                              transform=handlebox.get_transform())
        patch2 = matplotlib.text.Text(x0+0.5*width, y0+0.45*height, text, transform=handlebox.get_transform(),
                                      fontsize=0.55*fontsize, color=color, ha="center",va="center")
        handlebox.add_artist(patch1)
        handlebox.add_artist(patch2)
        return patch1


class Hist1D(object):

    def __init__(self, obj=None, **kwargs):
        tstr = str(type(obj))

        self._counts = None
        self._edges = None
        self._errors = None
        self._errors_up = None    # only handled when dividing with binomial errors
        self._errors_down = None  # only handled when dividing with binomial errors
        self._extra = {}
        kwargs = self.init_extra(**kwargs)
        if "ROOT." in tstr:
            self.init_root(obj, **kwargs)
        elif "uproot" in tstr:
            self.init_uproot(obj, **kwargs)
        elif "ndarray" in tstr or "list" in tstr:
            self.init_numpy(obj, **kwargs)

    def copy(self):
        hnew = self.__class__()
        hnew.__dict__.update(copy.deepcopy(self.__dict__))
        return hnew

    def init_numpy(self, obj, **kwargs):
        if "errors" in kwargs:
            self._errors = kwargs["errors"]
            del kwargs["errors"]

        self._counts, self._edges = np.histogram(obj,**kwargs)
        self._counts = self._counts.astype(np.float64)

        # poisson defaults if not specified
        if self._errors is None:
            if "weights" not in kwargs:
                self._errors = np.sqrt(self._counts)
            else:
                # if weighted entries, need to get sum of sq. weights per bin
                # and sqrt of that is bin error
                kwargs["weights"] = kwargs["weights"]**2.
                counts, _ = np.histogram(obj,**kwargs)
                self._errors = np.sqrt(counts)
        self._errors = self._errors.astype(np.float64)

    def init_root(self, obj, **kwargs):
        nbins = obj.GetNbinsX()
        if not kwargs.get("no_overflow",False):
            # move under and overflow into first and last visible bins
            # set bin error before content because setting the content updates the error?
            obj.SetBinError(1, (obj.GetBinError(1)**2.+obj.GetBinError(0)**2.)**0.5)
            obj.SetBinError(nbins, (obj.GetBinError(nbins)**2.+obj.GetBinError(nbins+1)**2.)**0.5)
            obj.SetBinContent(1, obj.GetBinContent(1)+obj.GetBinContent(0))
            obj.SetBinContent(nbins, obj.GetBinContent(nbins)+obj.GetBinContent(nbins+1))
        edges = np.array([1.0*obj.GetBinLowEdge(ibin) for ibin in range(1,nbins+2)])
        self._counts = np.array([1.0*obj.GetBinContent(ibin) for ibin in range(1,nbins+1)],dtype=np.float64)
        self._errors = np.array([1.0*obj.GetBinError(ibin) for ibin in range(1,nbins+1)],dtype=np.float64)
        self._edges = edges

    def init_uproot(self, obj, **kwargs):
        self._edges = np.array(obj.fXaxis.fXbins)
        if len(self._edges) == 0:
            self._edges = np.linspace(obj.low, obj.high, obj.numbins+1)
        self._counts = np.array(obj.values)
        self._errors = np.sqrt(obj.fSumw2)[1:-1]

        if not kwargs.get("no_overflow", False):
            # under and overflow
            # if no sumw2, then we'll let the errors=sqrt(counts)
            # handle the error properly (since we move in the counts at least)
            underflow, overflow = obj[0], obj[-1]
            self._counts[0] += underflow
            self._counts[-1] += overflow
            if obj.fSumw2:
                eunderflow2, eoverflow2 = obj.fSumw2[0], obj.fSumw2[-1]
                self._errors[0] = (self._errors[0]**2.+eunderflow2)**0.5
                self._errors[-1] = (self._errors[-1]**2.+eoverflow2)**0.5

        if len(self._errors) == 0:
            self._errors = self._counts**0.5

    def init_extra(self, **kwargs):
        if "color" in kwargs:
            self._extra["color"] = kwargs["color"]
            del kwargs["color"]
        if "label" in kwargs:
            self._extra["label"] = kwargs["label"]
            del kwargs["label"]
        return kwargs

    def fill_random(self, pdf="gaus", N=1000):
        if pdf not in ["gaus", "uniform"]:
            print("Warning: {} not a supported function.".format(pdf))
            return
        low, high = self._edges[0], self._edges[-1]
        cent = 0.5*(self._edges[0] + self._edges[-1])
        width = high-low
        if pdf == "gaus":
            vals = np.random.normal(cent, 0.2*width, N)
        elif pdf == "uniform":
            vals = np.random.uniform(low, high, N)
        else:
            raise ValueError("Unsupported pdf: "+pdf)
        counts, _ = np.histogram(vals, bins=self._edges)
        self._counts += counts
        self._errors = np.sqrt(self._errors**2. + counts)

    @property
    def errors(self):
        return self._errors

    @property
    def errors_up(self):
        return self._errors_up

    @property
    def errors_down(self):
        return self._errors_down

    @property
    def relative_errors(self):
        return self._errors / self._counts

    @errors.setter
    def errors(self, errors):
        self._errors = errors

    @property
    def counts(self):
        return self._counts

    @property
    def edges(self):
        return self._edges

    @property
    def counts_errors(self):
        return self._counts, self._errors

    @property
    def bin_centers(self):
        return 0.5*(np.array(self._edges[1:])+np.array(self._edges[:-1]))

    @property
    def bin_widths(self):
        return self._edges[1:]-self._edges[:-1]

    @property
    def integral(self):
        return float(np.sum(self._counts))

    @property
    def integral_and_error(self):
        return float(np.sum(self._counts)), float(np.sum(self._errors**2.0)**0.5)

    def _check_consistency(self, other):
        if len(self._edges) != len(other.edges):
            raise ValueError("These histograms cannot be combined due to different binning")
        return True

    def _rebin(self, edges_new):
        low_edges = self.edges[:-1]
        high_edges = self.edges[1:]
        vals = self.counts
        vars = self.errors**2
        widths = self.bin_widths

        nbins = len(edges_new) - 1
        vals_new = np.zeros(nbins, dtype=vals.dtype)
        vars_new = np.zeros(nbins, dtype=vals.dtype)
        # edges_new = np.linspace(min_, max_, nbins+1, dtype=vals.dtype)

        for i, (low, high) in enumerate(zip(edges_new[:-1], edges_new[1:])):
            # wholly contained bins
            b_idx = ((low_edges >= low) * (high_edges <= high)).nonzero()[0]
            bin_sum = np.sum(vals[b_idx])
            vars_new = np.sum(vars[b_idx])
            # internally contained
            b_idx = ((low_edges < low) * (high_edges > high)).nonzero()[0]
            bin_sum += np.sum(vals[b_idx])
            vars_new += np.sum(vars[b_idx])
            # left edge
            b_idx = ((low_edges < high) * (low_edges >= low) * (high_edges > high)).nonzero()[0]
            if len(b_idx) != 0:
                idx = b_idx[0]
                frac = (high - low_edges[idx])/widths[idx]
                bin_sum += vals[idx]*frac
                vars_new += vars[idx]*frac**2
            # right edge
            b_idx = ((high_edges > low) * (low_edges < low) * (high_edges <= high)).nonzero()[0]
            if len(b_idx) != 0:
                idx = b_idx[0]
                frac = (high_edges[idx] - low)/widths[idx]
                bin_sum += vals[idx]*frac
                vars_new += vars[idx]*frac**2

            vals_new[i] = bin_sum

        self._counts = vals_new
        self._edges = edges_new
        self._errors = np.sqrt(vars_new)

    def rebin(self, bins, min_=None, max_=None):
        """
        Rebins the contents of the histogram into either `bins` uniform bins from `min_` to `max_` or into variable
        sized bins with edges defined by `bins`.
        :param bins: either an int for the number of bins or an numpy array defining the bin edges
        :param min_: the lower limit for uniform binning
        :param max_: the upper limit for uniform binning
        """
        if type(bins) is int:
            if min_ is None:
                min_ = self.edges[0]
            if max_ is None:
                max_ = self.edges[-1]
            self._rebin(np.linspace(min_, max_, bins))
        else:
            self._rebin(np.array(bins))

    def __eq__(self, other):
        if not self._check_consistency(other): return False
        eps = 1.e-6
        return np.all(np.abs(self._counts - other.counts) < eps) \
            and np.all(np.abs(self._edges - other.edges) < eps) \
            and np.all(np.abs(self._errors - other.errors) < eps)

    def __add__(self, other):
        if type(other) == int and other == 0:
            return self
        if self._counts is None:
            return other
        self._check_consistency(other)
        hnew = self.__class__()
        hnew._counts = self._counts + other.counts
        hnew._errors = (self._errors**2. + other.errors**2.)**0.5
        hnew._edges = self._edges
        hnew._extra = self._extra
        return hnew

    __radd__ = __add__

    def __sub__(self, other):
        self._check_consistency(other)
        hnew = self.__class__()
        hnew._counts = self._counts - other.counts
        hnew._errors = (self._errors**2. + other.errors**2.)**0.5
        hnew._edges = self._edges
        hnew._extra = self._extra
        return hnew

    def __div__(self, other):
        if type(other) in [float,int]:
            return self.__mul__(1.0/other)
        else:
            return self.divide(other)

    __truediv__ = __div__

    def __floordiv__(self, other):
        """ Shorthand for division with binomial stats """
        return self.divide(other, binomial=True)

    def divide(self, other, binomial=False):
        self._check_consistency(other)
        hnew = self.__class__()
        hnew._edges = self._edges
        hnew._extra = self._extra
        with np.errstate(divide="ignore",invalid="ignore"):
            if not binomial:
                hnew._counts = self._counts / other.counts
                hnew._errors = (
                        (self._errors/other.counts)**2 +
                        (other.errors*self.counts/other.counts**2)**2
                        )**0.5
            else:
                hnew._errors_down, hnew._errors_up = clopper_pearson_error(self._counts,other._counts)
                hnew._counts = self._counts/other._counts
                hnew._errors = 0.*hnew._counts
                # these are actually the positions for down and up, but we want the errors
                # wrt to the central value
                hnew._errors_up = hnew._errors_up - hnew._counts
                hnew._errors_down = hnew._counts - hnew._errors_down
        return hnew

    def __mul__(self, fact):
        if type(fact) in [float,int]:
            hnew = self.copy()
            hnew._counts *= fact
            hnew._errors *= fact
            return hnew
        else:
            raise Exception("Can't multiply histogram by non-scalar")

    __rmul__ = __mul__

    def __pow__(self, expo):
        if type(expo) in [float,int]:
            hnew = self.copy()
            with np.errstate(divide="ignore",invalid="ignore"):
                hnew._counts = hnew._counts ** expo
                hnew._errors *= hnew._counts**(expo-1) * expo
            return hnew
        else:
            raise Exception("Can't multiply histogram by non-scalar")

    def __repr__(self):
        use_ascii = False
        if use_ascii: sep = "+-"
        else:
            if PY2:
                sep = u"\u00B1".encode("utf-8")
            else:
                sep = u"\u00B1"
        # trick: want to use numpy's smart formatting (truncating,...) of arrays
        # so we convert value,error into a complex number and format that 1D array :)
        formatter = {"complex_kind": lambda x:"%5.2f {} %4.2f".format(sep) % (np.real(x),np.imag(x))}
        a2s = np.array2string(self._counts+self._errors*1j,formatter=formatter, suppress_small=True, separator="   ")
        # return "<{}:\n{}\n>".format(self.__class__.__name__,a2s)
        return "<{}:{}>".format(self.__class__.__name__,a2s)

    def set_attr(self, attr, val):
        self._extra[attr] = val

    def get_attr(self, attr, default=None):
        return self._extra.get(attr, default)

    def get_attrs(self):
        return self._extra


class Hist2D(Hist1D):

    def init_numpy(self, obj, **kwargs):
        if "errors" in kwargs:
            self._errors = kwargs["errors"]
            del kwargs["errors"]

        if len(obj) == 0:
            xs, ys = [],[]
        else:
            xs, ys = obj[:,0], obj[:,1]
        counts, edgesx, edgesy = np.histogram2d(xs, ys, **kwargs)
        # each row = constant y, lowest y on top
        self._counts = counts.T
        self._edges = edgesx, edgesy
        self._counts = self._counts.astype(np.float64)

        # poisson defaults if not specified
        if self._errors is None:
            if "weights" not in kwargs:
                self._errors = np.sqrt(self._counts)
            else:
                # if weighted entries, need to get sum of sq. weights per bin
                # and sqrt of that is bin error
                kwargs["weights"] = kwargs["weights"]**2.
                counts, _, _ = np.histogram2d(obj[:, 0], obj[:, 1], **kwargs)
                self._errors = np.sqrt(counts.T)
        self._errors = self._errors.astype(np.float64)

    def init_root(self, obj, **kwargs):
        xaxis = obj.GetXaxis()
        yaxis = obj.GetYaxis()
        edges_x = np.array([1.0*xaxis.GetBinLowEdge(ibin) for ibin in range(1,xaxis.GetNbins()+2)])
        edges_y = np.array([1.0*yaxis.GetBinLowEdge(ibin) for ibin in range(1,yaxis.GetNbins()+2)])
        counts, errors = [], []
        for iy in range(1,obj.GetNbinsY()+1):
            counts_y, errors_y = [], []
            for ix in range(1,obj.GetNbinsX()+1):
                cnt = obj.GetBinContent(ix,iy)
                err = obj.GetBinError(ix,iy)
                counts_y.append(cnt)
                errors_y.append(err)
            counts.append(counts_y[:])
            errors.append(errors_y[:])
        self._counts = np.array(counts, dtype=np.float64)
        self._errors = np.array(errors, dtype=np.float64)
        self._edges = edges_x, edges_y

    def init_uproot(self, obj, **kwargs):
        # these arrays are (Nr+2)*(Nc+2) in size
        # note that we can't use obj.values because
        # uproot chops off the first and last elements
        # https://github.com/scikit-hep/uproot/blob/master/uproot/hist.py#L79
        err2 = np.array(obj.fSumw2)
        vals = np.array(obj)
        x_ax, y_ax = obj.fXaxis, obj.fYaxis
        xedges = x_ax.fXbins
        if not xedges:
            xedges = np.linspace(x_ax.fXmin, x_ax.fXmax, x_ax.fNbins+1)
        yedges = y_ax.fXbins
        if not yedges:
            yedges = np.linspace(y_ax.fXmin, y_ax.fXmax, y_ax.fNbins+1)
        self._counts = vals.reshape(len(yedges)+1,len(xedges)+1)[1:-1, 1:-1]
        if err2:
            self._errors = np.sqrt(err2.reshape(len(yedges)+1,len(xedges)+1)[1:-1, 1:-1])
        else:
            self._errors = np.sqrt(self.counts)
        self._edges = np.array(xedges), np.array(yedges)

    def _check_consistency(self, other):
        if len(self._edges[0]) != len(other._edges[0]) \
                or len(self._edges[1]) != len(other._edges[1]):
            raise Exception("These histograms cannot be combined due to different binning")
        return True

    def __eq__(self, other):
        if not self._check_consistency(other): return False
        eps = 1.e-6
        return np.all(np.abs(self._counts - other.counts) < eps) \
            and np.all(np.abs(self._edges[0] - other.edges[0]) < eps) \
            and np.all(np.abs(self._edges[1] - other.edges[1]) < eps) \
            and np.all(np.abs(self._errors - other.errors) < eps)

    @property
    def bin_centers(self):
        xcenters = 0.5*(self._edges[0][1:]+self._edges[0][:-1])
        ycenters = 0.5*(self._edges[1][1:]+self._edges[1][:-1])
        return xcenters, ycenters

    @property
    def bin_widths(self):
        xwidths = self._edges[0][1:]-self._edges[0][:-1]
        ywidths = self._edges[1][1:]-self._edges[1][:-1]
        return xwidths, ywidths

    @property
    def x_projection(self):
        hnew = Hist1D()
        hnew._counts = self._counts.sum(axis=0)
        hnew._errors = np.sqrt((self._errors**2).sum(axis=0))
        hnew._edges = self._edges[0]
        return hnew

    @property
    def y_projection(self):
        hnew = Hist1D()
        hnew._counts = self._counts.sum(axis=1)
        hnew._errors = np.sqrt((self._errors**2).sum(axis=1))
        hnew._edges = self._edges[1]
        return hnew

    def _calculate_profile(self, counts, errors, edges_to_sum, edges):
        centers = 0.5*(edges_to_sum[:-1]+edges_to_sum[1:])
        num = np.matmul(counts.T, centers)
        den = np.sum(counts, axis=0)
        num_err = np.matmul(errors.T**2, centers**2)**0.5
        den_err = np.sum(errors**2, axis=0)**0.5
        with np.errstate(divide="ignore", invalid="ignore"):
            r_val = num/den
            r_err = ((num_err/den)**2 + (den_err*num/den**2.0)**2.0)**0.5
        hnew = Hist1D()
        hnew._counts = r_val
        hnew._errors = r_err
        hnew._edges = edges
        return hnew

    @property
    def x_profile(self):
        xedges = self._edges[0]
        yedges = self._edges[1]
        return self._calculate_profile(self._counts, self._errors, yedges, xedges)

    @property
    def y_profile(self):
        xedges = self._edges[0]
        yedges = self._edges[1]
        return self._calculate_profile(self._counts.T, self._errors.T, xedges, yedges)


def to_html_table(rows, col_labels, row_labels, table_class):
    header = '<thead><tr>'+ ''.join(f'<th nowrap>{label}</th>' for label in col_labels) + '</tr></thead>'

    body = []
    for label, row in zip(row_labels, rows):
        body.append(f'<tr><td nowrap>{label}</td>' + ''.join(f'<td nowrap>{val}</td>' for val in row))
    return f'<table class="table {table_class}">' + header + ''.join(body) + '</table>'


def hist_to_table(data: Hist1D, bgs: [Hist1D], column_labels=(), format="{:.2f}",
                  table_class="table-condensed"):

    def row(hist, label=None):
        table.append('<tr>')
        label = hist.get_attr('label') if label is None else label
        table.append(f"<td nowrap><strong>{label}</strong></td>")
        table.extend(f'<td>{format.format(count)}</td>' for count in hist.counts)
        table.append(f'<td>{format.format(sum(hist.counts))}</td></tr>\n')

    table = [f'<table class="table {table_class}"><thead><tr><th/>']
    if column_labels:
        table.extend(f'<th nowrap>{label}</th>' for label in column_labels)
    else:
        table.extend(f'<th nowrap>[{low:g}, {high:g})</th>'
                     for low, high in zip(data.edges[:-1], data.edges[1:]))
    table.append('<th nowrap>Total</th></tr></thead><tbody>\n')
    for hist in bgs:
        row(hist)
    row(sum(bgs), label='Total BG')
    row(data)
    table.append('</tbody></table>')
    return ''.join(table)


def register_root_palettes():
    import matplotlib.pyplot as plt
    # RGB stops taken from
    # https://github.com/root-project/root/blob/9acb02a9524b2d9d5edb57c519aea4f4ab8022ac/core/base/src/TColor.cxx#L2523

    palettes = {
            "kBird": {
                "reds": [0.2082, 0.0592, 0.0780, 0.0232, 0.1802, 0.5301, 0.8186, 0.9956, 0.9764],
                "greens": [0.1664, 0.3599, 0.5041, 0.6419, 0.7178, 0.7492, 0.7328, 0.7862, 0.9832],
                "blues": [0.5293, 0.8684, 0.8385, 0.7914, 0.6425, 0.4662, 0.3499, 0.1968, 0.0539],
                "stops": np.linspace(0.,1.,9),
                },
            "kRainbow": {
                "reds": [0./255., 5./255., 15./255., 35./255., 102./255., 196./255., 208./255., 199./255., 110./255.],
                "greens": [0./255., 48./255., 124./255., 192./255., 206./255., 226./255., 97./255., 16./255., 0./255.],
                "blues": [99./255., 142./255., 198./255., 201./255., 90./255., 22./255., 13./255., 8./255., 2./255.],
                "stops": np.linspace(0., 1., 9),
                },
            "SUSY": {
                "reds": [0.50, 0.50, 1.00, 1.00, 1.00],
                "greens": [0.50, 1.00, 1.00, 0.60, 0.50],
                "blues": [1.00, 1.00, 0.50, 0.40, 0.50],
                "stops": [0.00, 0.34, 0.61, 0.84, 1.00],
                },
            }

    for key in palettes:
        stops = palettes[key]["stops"]
        reds = palettes[key]["reds"]
        greens = palettes[key]["greens"]
        blues = palettes[key]["blues"]
        cdict = {
            "red": zip(stops, reds, reds),
            "green": zip(stops, greens, greens),
            "blue": zip(stops, blues, blues)
        }
        plt.register_cmap(name=key, data=cdict)

