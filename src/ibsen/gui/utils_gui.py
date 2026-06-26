import numpy as np
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from astropy.table import Table
from astropy.io import ascii

from pathlib import Path

try:
    from gammapy.estimators import FluxPoints
except:
    print("Impossible to import Gammapy.")
    FluxPoints = None
    
def read_fits_write_txt(fits_name, txt_name, to_save=True,
                        x_axis_offset=0, to_return=False,
                        flux_type='flux', **plot_kwargs):
    """
    Reads the fits file obtaned by 
    lightcurve.write(fits_name, sed_type=<'flux' OR 'eflux'>)
    and exctracts the data: t+-dt, flux+- df.
    You can provide x_axis_offset to subtract from time.
    
        - if to_save: saves the data into text file txt_name.txt
        - if to_plot: plots the data. Use **plot_kwargs
        - `flux_type` species which format of the flux is in the .fits file:
            -- 'flux' [1 / s / cm2], default, OR
            -- 'eflux' [TeV / s / cm2] (the physical flux) which will be transformed 
                into [erg / s / cm2].
        - if to_return: returns a dict of np.arrays 
        {"t", "flux", "t_mjd", "dt", "df_p", "df_m"}

    """


    # read FLUXPOINTS table
    tbl = Table.read(fits_name, hdu="FLUXPOINTS")

    # columns: time vs flux (works whether or not they have units)
    time_min = getattr(tbl["time_min"], "value", tbl["time_min"])
    time_max = getattr(tbl["time_max"], "value", tbl["time_max"])
    time_mjd = 0.5 * (time_min + time_max)
    time = time_mjd - x_axis_offset
    dt = 0.5 * (time_max-time_min)
    
    if flux_type == 'flux':
        flux = np.squeeze(np.asarray(getattr(tbl["flux"], "value", tbl["flux"])))
        try:
            dfp = np.squeeze(np.asarray(getattr(tbl["flux_errp"], "value", tbl["flux_errp"])))
            dfm = np.squeeze(np.asarray(getattr(tbl["flux_errn"], "value", tbl["flux_errn"])))
        except:
            dfp = np.squeeze(np.asarray(getattr(tbl["flux_err"], "value", tbl["flux_err"])))
            dfm = np.squeeze(np.asarray(getattr(tbl["flux_err"], "value", tbl["flux_err"])))
            
    elif flux_type == 'eflux':
        flux = np.squeeze(np.asarray(getattr(tbl["eflux"], "value", tbl["eflux"])))
        try:
            dfp = np.squeeze(np.asarray(getattr(tbl["eflux_errp"], "value", tbl["eflux_errp"])))
            dfm = np.squeeze(np.asarray(getattr(tbl["eflux_errn"], "value", tbl["eflux_errn"])))
        except:
            dfp = np.squeeze(np.asarray(getattr(tbl["flux_err"], "value", tbl["flux_err"])))
            dfm = np.squeeze(np.asarray(getattr(tbl["flux_err"], "value", tbl["flux_err"])))
        ### manually transform TeV/s/cm2 --> erg/s/cm2
        flux, dfp, dfm = [ar_ * 1.6 for ar_ in (flux, dfp, dfm)]
    else: raise ValueError('flux_type can be only `flux` or `eflux`')

    if to_save:
        data_ = np.array([time, flux, time_mjd, dt, dfp, dfm]).T
        header_ = "t flux t_mjd dt df_p df_m"
        np.savetxt(txt_name+'.txt', data_, header=header_, delimiter=' ',
                   fmt='%s')
        np.savetxt(txt_name+'.dat', data_, header=header_, delimiter=' ',
                   fmt='%s')
        

        
    if to_return:
        return {"t": time, "flux":flux, "t_mjd":time_mjd, "dt":dt, 
                "df_p":dfp, "df_m": dfm}
    
def read_lightcurve_columns(
    path: str,
    *,
    # candidate names are checked case-insensitively; underscores/spaces ignored
    t_names=("t", "time", "mjd", "jd", "bjd", "tbjd"),
    dt_names=("dt", "deltat", "time_err", "timeerr", "sigma_t", "sigmat", "err_t"),
    f_names=("f", "flux", "flx", "rate", "counts", "count_rate"),
    df_names=("df", "dflux", "flux_err", "fluxerr", "sigma_f", "sigmaflux", 
              "err_f", "eflux", "df_p", "flux_errp", "eflux_errp"),
    add_cols=None,
        ):
    """
    Read an ASCII table (with optional '#' commented header) and extract columns for:
      - t (time)
      - dt (time uncertainty) [optional -> NaNs if missing]
      - f (flux)
      - df (flux uncertainty) [optional -> NaNs if missing]

    Returns:
        t, dt, f, df, meta
    where meta includes:
        - 'table': the astropy Table
        - 'colmap': mapping of {'t': name|None, 'dt':..., 'f':..., 'df':...}
    """
    # Astropy can usually guess the format/delimiter; comment="#" handles typical headers.
    tbl: Table = ascii.read(
        path,
        guess=True,
        comment="#",
        fast_reader=True,
    )

    if len(tbl.colnames) == 0:
        raise ValueError("No columns detected. Is the file empty or not a table?")

    # Normalization helper for robust matching
    def norm(s: str) -> str:
        s = s.strip().lower()
        s = s.replace(" ", "").replace("_", "").replace("-", "")
        return s

    colnames = list(tbl.colnames)
    norm_map = {name: norm(name) for name in colnames}

    def find_col(candidates) -> str | None:
        cand_norm = {norm(c) for c in candidates}
        # exact normalized match first
        for name, nn in norm_map.items():
            if nn in cand_norm:
                return name
        # looser: contains candidate token (e.g. "flux" in "raw_flux")
        for name, nn in norm_map.items():
            for c in cand_norm:
                if c and (c in nn):
                    return name
        return None

    t_col = find_col(t_names)
    dt_col = find_col(dt_names)
    f_col = find_col(f_names)
    df_col = find_col(df_names)
    
    if add_cols is not None:
        other_cols = []
        for other_name in add_cols:
            other_cols.append(find_col((other_name,)))
    # print(other_cols)

    # helpers to convert to float numpy arrays (handle masked columns)
    def to_float_array(colname: str) -> np.ndarray:
        col = tbl[colname]
        arr = np.array(col, dtype=float)
        # If masked, fill masked entries with NaN
        if hasattr(col, "mask"):
            arr = np.array(col.filled(np.nan), dtype=float)
        return arr

    if t_col is None:
        raise ValueError(f"Could not find time column. Tried names: {t_names}. Found: {colnames}")
    if f_col is None:
        raise ValueError(f"Could not find flux column. Tried names: {f_names}. Found: {colnames}")

    # Required (or possibly missing if require_* = False)
    t = to_float_array(t_col) if t_col is not None else np.full(len(tbl), np.nan)
    f = to_float_array(f_col) if f_col is not None else np.full(len(tbl), np.nan)

    # Optional -> NaNs
    dt = to_float_array(dt_col) if dt_col is not None else np.full(len(tbl), np.nan)
    df = to_float_array(df_col) if df_col is not None else np.full(len(tbl), np.nan)

    if add_cols is not None:
        other_values = []
        for other_name in other_cols:
            other_values.append(to_float_array(other_name))
    meta = {
        "table": tbl,
        "colmap": {"t": t_col, "dt": dt_col, "f": f_col, "df": df_col},
        "all_columns": colnames,
    }
    if add_cols is None:
        return t, dt, f, df, meta
    return t, dt, f, df, other_values, meta
        
def read_lightcurve_special_case(path):
    da = np.genfromtxt(path,
                   delimiter=' ', usecols=[0, 1, 4, 5, 3], 
                   names=['t', 'f', 'df_p', 'df_m', 'dt'])
    t, f, df_p, df_m, dt = [da[key] for key in ('t', 'f', 'df_p', 'df_m', 'dt')]
    df_p = np.nan_to_num(df_p)
    df_m = np.nan_to_num(df_m)
    df = 0.5 * (df_m + df_p)
    return t, f, df

def read_lightcurve(path):
    ext = Path(path).suffix.lower()
    if ext in ('.txt', '.dat', '.csv'):
        t, dt, f, df, _ = read_lightcurve_columns(path=path)
    elif ext == '.fits':
        res = read_fits_write_txt(fits_name = path, txt_name=None, plot_kwargs=None,
                                  to_save=False, x_axis_offset=0, to_return=True,
                                  flux_type='eflux')
        t, dt, f, df = [res[keyw] for keyw in ("t", "dt", "flux", "df_p")]
    else:
        raise ValueError("The format of the file not recognized.")
    _sort = np.argsort(t)
    t, dt, f, df = [ar[_sort] for ar in (t, dt, f, df)]
    return t, dt, f, df

def read_sed_txt_dat(path):
    """
    Reads the SED called sed_{key}{suffix}.txt

    returns : dict['e', 'sed', 'dsed', 'dsed_minus', 'dsed_plus',
                     'de', 'de_minus', 'de_plus']
    e in TeV
    seds in erg/cm2/s
    """

    da = np.genfromtxt(path,
                       delimiter=' ', usecols=[0, 1, 2, 3, 4, 5], 
                       names=['e', 'sed', 'de_p', 'de_m', 'dsed_p', 'dsed_m'])

   
    e, sed, de_p, de_m, dsed_p, dsed_m = [da[key] for key in ('e', 'sed',
                                                              'de_p', 'de_m',
                                                'dsed_p', 'dsed_m')]
    de = 0.5 * (de_m + de_p)
    dsed = 0.5 * (dsed_m + dsed_p)
    
    res = {'e': e*1e12, 'sed': sed, 'dsed': dsed,
              'dsed_minus': dsed_m, 'dsed_plus': dsed_p,
        'de': de*1e12, 'de_minus': de_m*1e12, 'de_plus': de_p*1e12,}
    return res    

def get_sed_values(sed_gammapy, return_de=False):
    """
    From a FluxPoints gammapy object extracts e [TeV], sed [erg s-1 cm-2],
    dsed [erg s-1 cm-2].
        """
    tab  = sed_gammapy.to_table(sed_type='e2dnde')
    e, sed, dsed = tab['e_ref'].value, \
        (tab['e2dnde'].to("erg s-1 cm-2")).value, \
            (tab['e2dnde_err'].to("erg s-1 cm-2")).value
    de = (tab['e_max'].value - tab['e_min'].value) / 2.0
    if return_de:
        return e, de, sed, dsed
    return e, sed, dsed

def read_sed_gammapy(path):
    if FluxPoints is not None:
        sed = FluxPoints.read(path)
        e, sed, dsed = get_sed_values(sed)
        return {'e': e*1e12, 'sed':sed, 'dsed': dsed}
    raise ImportError("Gammapy cannot be imported.")
    
class GradientPlot:
    def __init__(self, fig, ax, *, cmap="coolwarm", lw=2, ls="-", alpha=1.0,
                 scatter=False, marker="o", s=20, colorbar=False, cbar_label="grad"):
        self.fig = fig
        self.ax = ax
        self.cmap = cmap
        self.lw = lw
        self.ls = ls
        self.alpha = alpha
        self.scatter = scatter
        self.marker = marker
        self.s = s
        self.want_colorbar = colorbar
        self.cbar_label = cbar_label

        self.artist = None   # LineCollection or PathCollection
        self.cbar = None

    def _clean_data(self, x, y, p):
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()
        p = np.asarray(p).ravel()
        m = np.isfinite(x) & np.isfinite(y) & np.isfinite(p)
        return x[m], y[m], p[m]

    def update(self, x, y, p, *, vmin=None, vmax=None, autoscale=True):
        x, y, p = self._clean_data(x, y, p)
        if x.size == 0:
            return

        vmin_here = np.min(p) if vmin is None else vmin
        vmax_here = np.max(p) if vmax is None else vmax
        norm = Normalize(vmin=vmin_here, vmax=vmax_here)

        if self.scatter or x.size < 2:
            # --- scatter mode ---
            if self.artist is None or self.artist.__class__.__name__ != "PathCollection":
                # first time: create
                if self.artist is not None:
                    self.artist.remove()
                self.artist = self.ax.scatter(
                    x, y, c=p, cmap=self.cmap, norm=norm,
                    s=self.s, marker=self.marker, linewidths=0, alpha=self.alpha
                )
            else:
                # update existing scatter
                self.artist.set_offsets(np.c_[x, y])
                self.artist.set_array(p)
                self.artist.set_norm(norm)

        else:
            # --- gradient line via LineCollection ---
            points = np.column_stack([x, y]).reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            if self.artist is None or not isinstance(self.artist, LineCollection):
                if self.artist is not None:
                    self.artist.remove()
                lc = LineCollection(segments, cmap=self.cmap, norm=norm)
                lc.set_array(p[:-1])
                lc.set_linewidth(self.lw)
                lc.set_linestyle(self.ls)
                lc.set_alpha(self.alpha)
                self.artist = self.ax.add_collection(lc)
            else:
                self.artist.set_segments(segments)
                self.artist.set_array(p[:-1])
                self.artist.set_norm(norm)
                # linewidth/linestyle/alpha can also be updated if you want:
                # self.artist.set_linewidth(self.lw)

        # Axis limits
        if autoscale:
            # For LineCollection, autoscale needs the collection to be considered:
            self.ax.relim()
            self.ax.autoscale_view()

        # Colorbar: create once, then update
        if self.want_colorbar:
            if self.cbar is None:
                self.cbar = self.fig.colorbar(self.artist, ax=self.ax, label=self.cbar_label)
            else:
                self.cbar.update_normal(self.artist)


def plot_surface_quads_local(ax, coords, param, cmap="coolwarm",
                       vmin=None, vmax=None, alpha=1.0,
                       edgecolor="none", linewidth=0.0,
                       close_phi=False):
    """
    Draw quads on a 3D axis.

    Returns:
        poly (Poly3DCollection or None),
        mappable (ScalarMappable or None),
        norm (Normalize or None)
    """
    P = np.asarray(coords, float)
    if P.ndim != 3 or P.shape[-1] != 3:
        raise ValueError("coords must have shape (Ntheta, Nphi, 3)")

    S = None
    if param is not None:
        S = np.asarray(param, float)
        if S.shape != P.shape[:-1]:
            raise ValueError("param must have shape (Ntheta, Nphi)")

    if close_phi:
        P = np.concatenate([P, P[:1, :, :]], axis=0)
        if S is not None:
            S = np.concatenate([S, S[:1, :]], axis=0)

    # Build quads
    A = P[:-1, :-1]
    B = P[ 1:, :-1]
    C = P[ 1:,  1:]
    D = P[:-1,  1:]
    quads = np.stack([A, B, C, D], axis=2).reshape(-1, 4, 3)

    mappable = None
    norm = None
    colors = None

    if S is not None:
        s_quad = 0.25 * (S[:-1, :-1] + S[1:, :-1] + S[1:, 1:] + S[:-1, 1:]).ravel()

        good = np.isfinite(s_quad)
        good &= np.isfinite(quads).all(axis=(1, 2))
        quads = quads[good]
        s_quad = s_quad[good]
        if s_quad.size == 0:
            return None, None, None

        vmin = np.nanmin(s_quad) if vmin is None else vmin
        vmax = np.nanmax(s_quad) if vmax is None else vmax
        norm = Normalize(vmin=vmin, vmax=vmax)

        cmap_obj = plt.get_cmap(cmap)
        colors = cmap_obj(norm(s_quad))

        # this is what the colorbar will use
        mappable = ScalarMappable(norm=norm, cmap=cmap_obj)
        mappable.set_array(s_quad)

    poly = Poly3DCollection(
        quads,
        facecolors=colors,
        edgecolors=edgecolor,
        linewidths=linewidth,
        alpha=alpha
    )
    ax.add_collection3d(poly)

    # Autoscale
    xyz = P.reshape(-1, 3)
    m = np.isfinite(xyz).all(axis=1)
    xyz = xyz[m]
    if xyz.size:
        ax.auto_scale_xyz(xyz[:, 0], xyz[:, 1], xyz[:, 2])

    return poly, mappable, norm