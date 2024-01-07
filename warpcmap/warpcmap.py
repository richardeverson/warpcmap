import numpy as np
from scipy.special import betainc
from matplotlib import cm, rcParams
from matplotlib.colors import ListedColormap
from matplotlib.pyplot import gca
from scipy.optimize import root_scalar

def warp_colormap(basemap, z, beta=1, Nentries=256):
    """
    Construct a new colormap by warping basemap so that the colour
    in "middle" of the basemap is (ie, corresponding to a value of 0.5)
    corresponds to the value z and the rate of change of colours around z
    is given by beta.

    Parameters
    ----------

    basemap:  Matplotlib ColorMap or string naming one.
        The colormap to warp

    z:  float (0 < z < 1)

        The location that the middle of the basemap is warped to.
        For instance, if the basemap is the 'jet' colourmap, so that
        the middle of the map is green, then in the new colourmap,
        green corresponds to the value z.

    beta: float (beta > 0)
        Beta controls the rate of change of colours close to z.  It is
        approximately the gradient of the mapping between the new colours
        and the old colours.  Values of beta in the range 1 to 5 are usual
        as they give more resolution to the data values close to z.
        beta < 1 compresses the colours close to z.

    Nentries: int
        Number of entries in the new colourmap.

    Returns
    -------

    newmap:  ListedColormap
        The warped colourmap.


    Example
    -------

    Emphasise the data values around 0.8
    >>> cmap = warp_colormap('jet', z=0.8, beta=3)
    >>> imshow(X, cmap=cmap)
    >>> colorbar()

    Note
    ----

    If the range of the data is not [0, 1], then z is linearly mapped
    to the data range so if the data range is (100, 120) and z = 0.8,
    then the emphasised values will be around 116.
    """
    if not (0 < z < 1):
        raise ValueError('z must be between 0 and 1')
    if isinstance(basemap, str):
        basemap = cm.get_cmap(basemap)

    def objective(alpha):
        return betainc(alpha, beta, z) - 0.5

    soln = root_scalar(objective, bracket=(1e-10, 1e2))
    alpha = soln.root

    rgba = np.zeros((Nentries, 4))
    for n, y in enumerate(np.linspace(0, 1, Nentries)):
        x = betainc(alpha, beta, y)
        rgba[n,:] = basemap(x)
        newmap = ListedColormap(rgba)
    return newmap

def wimshow(X,
            cmap=None,
            vmin=None, vmax=None, vmid=None, beta=1, Nentries=256,
            ax=None,
            **kwargs):
    """
    Convenience wrapper for `imshow` for displaying scalar data that allows
    setting of the range of the data values that the coloramp covers and
    how it is warped.

    Parameters
    ----------
    X : array-like
        The image with scalar data.  The data is visualized
        using a colormap. If RGB or RGBA data is to be displayed, just
        use `imshow` directly.

        The two dimensions (M, N) define the rows and columns of
        the image.

    cmap : str or `~matplotlib.colors.Colormap`, optional
        The Colormap instance or registered colormap name used to map
        scalar data to colors.
        Defaults to :rc:`image.cmap`.

    vmin, vmax : scalar, optional
        When using scalar data and no explicit *norm*, *vmin* and *vmax*
        define the data range that the colormap covers. By default,
        the colormap covers the complete value range of the supplied
        data. *vmin*, *vmax* are ignored if the *norm* parameter is used.

    vmid: scalar, optional
        The data value that the middle of the colormap is warped to.
        For instance, if the colormap is the 'jet' colourmap, so that
        the middle of the map is green, then in the new colourmap,
        green corresponds to the value vmid.
        Default is the middle of the range of the data.

    beta:  scalar > 0, optional
        Beta controls the rate of change of colours close to *vmid*.
        Larger values of beta give a more rapid change of colour with
        data value.  Values of beta in the range 1 to 5 are usual
        as they give more resolution to the data values close to *vmid*.
        beta < 1 compresses the colours close to *vmid*.

    Nentries: int, optional
        Number of entries in the warped colourmap.
        Default: 256

    ax: matplotlib.axes.Axes, optional
        The Matplotlib axes in which to plot.
        Default:  The current axes.

    Returns
    -------
    image : `~matplotlib.image.AxesImage`

    Other Parameters
    ----------------

    All other parameters are passed directly to `imshow`.
    """
    assert len(X.shape) == 2, "wimshow only supports scalar data; use imshow for RGB and RGBA data"
    if cmap is None:
        cmap = rcParams['image.cmap']

    if vmin is None:
        vmin = X.min()

    if vmax is None:
        vmax = X.max()

    if vmid is None:
        vmid = (vmin + vmax)/2

    if ax is None:
        ax = gca()
    z = (vmid - vmin)/(vmax-vmin)
    warped = warp_colormap(cmap, z, beta=beta, Nentries=Nentries)
    return ax.imshow(X, cmap=warped, vmin=vmin, vmax=vmax, **kwargs)


def wpcolormesh(*args,
            cmap=None,
            vmin=None, vmax=None, vmid=None, beta=1, Nentries=256,
            ax=None,
            **kwargs):
    """
    Convenience wrapper for `pcolormesh` for displaying scalar data that allows
    setting of the range of the data values that the coloramp covers and
    how it is warped.

    Call signature::

    wpcolormesh([X, Y,] C, **kwargs)

    *X* and *Y* can be used to specify the corners of the rectangles.


    Parameters
    ----------
    C : 2D array-like
        The values to be color-mapped.

    X, Y : array-like, optional
        The coordinates of the corners of quadrilaterals of a pcolormesh.
        See `pcolormesh` for details.

    cmap : str or `~matplotlib.colors.Colormap`, optional
        The Colormap instance or registered colormap name used to map
        scalar data to colors.
        Defaults to :rc:`image.cmap`.

    vmin, vmax : scalar, optional
        When using scalar data and no explicit *norm*, *vmin* and *vmax*
        define the data range that the colormap covers. By default,
        the colormap covers the complete value range of the supplied
        data. *vmin*, *vmax* are ignored if the *norm* parameter is used.

    vmid: scalar, optional
        The data value that the middle of the colormap is warped to.
        For instance, if the colormap is the 'jet' colourmap, so that
        the middle of the map is green, then in the new colourmap,
        green corresponds to the value vmid.
        Default is the middle of the range of the data or (*vmin* + *vmax*)/2
        if they are specified.

    beta:  scalar > 0, optional
        Beta controls the rate of change of colours close to *vmid*.
        Larger values of beta give a more rapid change of colour with
        data value.  Values of beta in the range 1 to 5 are usual
        as they give more resolution to the data values close to *vmid*.
        beta < 1 compresses the colours close to *vmid*.

    Nentries: int, optional
        Number of entries in the warped colourmap.
        Default: 256

    ax: matplotlib.axes.Axes, optional
        The Matplotlib axes in which to plot.
        Default:  The current axes.

    Returns
    -------
    `matplotlib.collections.QuadMesh`

    Other Parameters
    ----------------

    All other parameters are passed directly to `pcolormesh`.
    """
    if len(args) == 1:
        C = args[0]
    elif len(args) == 3:
        C = args[2]
    else:
        raise TypeError(f'wpcolormesh() takes 1 or 3 positional arguments '
                        f'but {len(args)} were given')
    if cmap is None:
        cmap = rcParams['image.cmap']

    if vmin is None:
        vmin = C.min()

    if vmax is None:
        vmax = C.max()

    if vmid is None:
        vmid = (vmin + vmax)/2

    if ax is None:
        ax = gca()
    z = (vmid - vmin)/(vmax-vmin)
    warped = warp_colormap(cmap, z, beta=beta, Nentries=Nentries)
    return ax.pcolormesh(*args, cmap=warped, vmin=vmin, vmax=vmax, **kwargs)



def wpcolor(*args,
            cmap=None,
            vmin=None, vmax=None, vmid=None, beta=1, Nentries=256,
            ax=None,
            **kwargs):
    """
    Convenience wrapper for `pcolor` for displaying scalar data that allows
    setting of the range of the data values that the coloramp covers and
    how it is warped.

    Call signature::

    wpcolor([X, Y,] C, **kwargs)

    *X* and *Y* can be used to specify the corners of the rectangles.


    Parameters
    ----------
    C : 2D array-like
        The values to be color-mapped.

    X, Y : array-like, optional
        The coordinates of the corners of quadrilaterals of a pcolor.
        See `pcolor` for details.

    cmap : str or `~matplotlib.colors.Colormap`, optional
        The Colormap instance or registered colormap name used to map
        scalar data to colors.
        Defaults to :rc:`image.cmap`.

    vmin, vmax : scalar, optional
        When using scalar data and no explicit *norm*, *vmin* and *vmax*
        define the data range that the colormap covers. By default,
        the colormap covers the complete value range of the supplied
        data. *vmin*, *vmax* are ignored if the *norm* parameter is used.

    vmid: scalar, optional
        The data value that the middle of the colormap is warped to.
        For instance, if the colormap is the 'jet' colourmap, so that
        the middle of the map is green, then in the new colourmap,
        green corresponds to the value vmid.
        Default is the middle of the range of the data or (*vmin* + *vmax*)/2
        if they are specified.

    beta:  scalar > 0, optional
        Beta controls the rate of change of colours close to *vmid*.
        Larger values of beta give a more rapid change of colour with
        data value.  Values of beta in the range 1 to 5 are usual
        as they give more resolution to the data values close to *vmid*.
        beta < 1 compresses the colours close to *vmid*.

    Nentries: int, optional
        Number of entries in the warped colourmap.
        Default: 256

    ax: matplotlib.axes.Axes, optional
        The Matplotlib axes in which to plot.
        Default:  The current axes.

    Returns
    -------
    `matplotlib.collections.QuadMesh`

    Other Parameters
    ----------------

    All other parameters are passed directly to `pcolor`.
    """
    if len(args) == 1:
        C = args[0]
    elif len(args) == 3:
        C = args[2]
    else:
        raise TypeError(f'wpcolor() takes 1 or 3 positional arguments '
                        f'but {len(args)} were given')
    if cmap is None:
        cmap = rcParams['image.cmap']

    if vmin is None:
        vmin = C.min()

    if vmax is None:
        vmax = C.max()

    if vmid is None:
        vmid = (vmin + vmax)/2

    if ax is None:
        ax = gca()
    z = (vmid - vmin)/(vmax-vmin)
    warped = warp_colormap(cmap, z, beta=beta, Nentries=Nentries)
    return ax.pcolor(*args, cmap=warped, vmin=vmin, vmax=vmax, **kwargs)
