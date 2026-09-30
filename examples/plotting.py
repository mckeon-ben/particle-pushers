'''
Reads JSON data files, turning the stored final states into error
estimates and observed orders, prints the tables and draws the figure.

File contract
-------------
Required: 'schema' and 'families'. Each family carries 'label',
'order', 'method_order', 'methods' -- the last mapping each display
name to 'time', the seconds that method took, and 'x' and 'u' arrays
holding one final state per entry of dt --
and its own 'dt', the step sizes those states were computed at. Steps
are per family because a fourth-order method's error falls four
decades faster than a second-order one over the same range, so a
single shared list tends to leave one family pre-asymptotic while the
other has already reached the round-off floor.

Optional 'experiment' names the problem in the figure title of the
screen layout, and 'parameters' may carry 'T', the final lab time, which
is appended to it. Anything else in the file is ignored. Display names
not listed in STYLES get a style from a fallback cycle rather than
raising.

From final states to errors
---------------------------
The file stores the final state y(dt) for each method at each step size.
The successive-difference norm

    delta(dt) = || y(dt) - y(dt/2) ||

is the error up to a known constant. Writing e(dt) for the error,
delta(dt) = e(dt) - e(dt/2) = e(dt) (1 - 2**-p), so

    e(dt) ~ delta(dt) / (1 - 2**-p),

a factor of 4/3 at second order and 16/15 at fourth. This is the error
at the coarser step of each pair, which is the step the differences are
indexed by; the Richardson correction delta/(2**p - 1) is the error at
the finer step, smaller by exactly 2**p.

The observed order is taken from the ratio of consecutive differences
against the ratio of their step sizes,

    p = log(delta_i / delta_{i+1}) / log(dt_i / dt_{i+1}),

rather than as log2 of the difference ratio. The two agree when the step
is exactly halved, but the general form does not silently misreport when
it is not: a step list that doubles everywhere except once, as a typo
easily produces, would otherwise show an order error of about half a per
cent at the affected pair and look like real behavior.
'''

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCHEMA = 1

# Where data files live when they are not given by an explicit path:
# the data folder beside this script, wherever it is run from.
# A bare name on the command line is resolved against the working
# directory first and then here, and passing no name at all plots every
# data file in this folder.
DATA_DIR = Path(__file__).resolve().parent / 'data'

# Where figures are written, kept apart from the data folder so one
# directory holds only inputs and the other only outputs.
PLOT_DIR = Path(__file__).resolve().parent / 'plots'

# Default output format. Vector, so the figure stays sharp at any zoom
# and at whatever size a paper puts it. Pass -o with another extension
# to override; matplotlib picks the writer from the suffix.
FIGURE_SUFFIX = '.pdf'

# Print layout for the Journal of Scientific Computing, used with
# --print. The figure is drawn at the full text width, 174 mm, so no text
# or line is scaled down: lettering of 8 to 12 pt in Helvetica, lines of
# at least 0.3 pt, and vector EPS with the fonts embedded. At 10 pt the
# superscripts in the tick labels stay close to the 8 pt minimum.
PRINT_WIDTH = 174 / 25.4
PRINT_FONT_SIZE = 10
PRINT_LINE_WIDTH = 1.0
PRINT_SUFFIX = '.eps'

# Here, matplotlib takes the first entry actually installed, so a machine
# with the real fonts uses them and one without still produces the same
# metrics.
FONT_STACK = ['Helvetica', 'Arial', 'TeX Gyre Heros', 'Nimbus Sans',
              'Liberation Sans', 'FreeSans', 'DejaVu Sans']

# Typeset through a local LaTeX installation instead of matplotlib's own
# engine.
USETEX = True

LATEX_PREAMBLE = r'''
\usepackage[T1]{fontenc}
\usepackage{helvet}
\renewcommand{\familydefault}{\sfdefault}
\usepackage{sansmath}
\sansmath
\DeclareSymbolFont{sansletters}{OT1}{phv}{m}{n}
\DeclareMathSymbol{-}{\mathbin}{sansletters}{"2D}
\DeclareSymbolFont{sansgreek}{OT1}{cmss}{m}{n}
\DeclareMathSymbol{\Delta}{\mathord}{sansgreek}{"01}
'''

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': FONT_STACK,
    # Set the maths in the same family.
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'sans',
    'mathtext.it': 'sans:italic',
    'mathtext.bf': 'sans:bold',
    'mathtext.cal': 'sans',
    'axes.grid': True,
    # Opaque rather than translucent, which EPS cannot store.
    'grid.color': '0.9',
    'figure.dpi': 150,
    'legend.fontsize': 8,
    # Embed TrueType rather than the Type 3 fonts matplotlib writes by
    # default.
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})


def use_latex(enabled=USETEX):
    '''
    Switch LaTeX typesetting on or off.

    Under usetex the mathtext and font.sans-serif settings above are
    ignored, since the preamble decides both; they stay in place as the
    fallback for when this is off.

    Parameters
    ----------
    enabled : bool, optional
        Whether to typeset through LaTeX.
    '''
    plt.rcParams.update({
        'text.usetex': enabled,
        'text.latex.preamble': LATEX_PREAMBLE if enabled else '',
    })


def use_print_layout():
    '''
    Switch to the journal's print sizes for text and lines.

    All text is set at PRINT_FONT_SIZE and every line, including the
    axes, ticks and grid, at PRINT_LINE_WIDTH.
    '''
    size, width = PRINT_FONT_SIZE, PRINT_LINE_WIDTH
    plt.rcParams.update({
        'font.size': size,
        'axes.titlesize': size,
        'axes.labelsize': size,
        'xtick.labelsize': size,
        'ytick.labelsize': size,
        'legend.fontsize': size,
        'lines.linewidth': width,
        'axes.linewidth': width,
        'grid.linewidth': width,
        'xtick.major.width': width,
        'ytick.major.width': width,
        'xtick.minor.width': width,
        'ytick.minor.width': width,
        'lines.markersize': 3.5,
    })


# Style overrides by display name, so a method keeps its marker across
# experiments. Names not listed fall back to the markers below.
STYLES = {
    'Boris': ('-', 'o'),
    'Vay': ('-', 's'),
    'Higuera-Cary': ('-', '^'),
    'Gordon-Hafizi (quadratic)': ('-', 'v'),
    'Gordon-Hafizi (exact)': ('-', 'D'),
}

# Okabe-Ito, the standard colorblind-safe qualitative palette.
PALETTE = [
    '#0072B2',  # blue
    '#D55E00',  # vermillion
    '#009E73',  # bluish green
    '#CC79A7',  # reddish purple
    '#E69F00',  # orange
    '#56B4E9',  # sky blue
]

FALLBACK_MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '<', '>']

SECTORS = [('x', 'position'), ('u', 'velocity')]


def resolve(name):
    '''
    Find a data file given a path, a filename, or a bare stem.

    Tries the working directory before DATA_DIR so an explicit path
    always wins, and appends the .json suffix only when the name does
    not already carry one -- appending unconditionally would mangle a
    name that has dots in it for other reasons.

    Parameters
    ----------
    name : str
        Path, filename or bare stem of the data file.

    Returns
    -------
    Path
        The data file found.

    Raises
    ------
    SystemExit
        If no matching file exists.
    '''
    given = Path(name)
    names = ([given] if given.suffix == '.json'
             else [given, given.with_name(given.name + '.json')])
    for base in (Path('.'), DATA_DIR):
        for candidate in names:
            full = candidate if candidate.is_absolute() else base / candidate
            if full.is_file():
                return full
    raise SystemExit(f'{name}: no such data file in . or {DATA_DIR}/')


def find_all():
    '''
    Every data file in DATA_DIR, sorted by name.

    Returns
    -------
    list of Path
        The data files found.

    Raises
    ------
    SystemExit
        If DATA_DIR holds no .json files.
    '''
    files = sorted(DATA_DIR.glob('*.json'))
    if not files:
        raise SystemExit(f'no .json files found in {DATA_DIR}/')
    return files


def load(filename):
    '''
    Read a data file, checking the schema and the required keys.

    Parameters
    ----------
    filename : str or Path
        Data file to read.

    Returns
    -------
    dict
        The record.

    Raises
    ------
    ValueError
        If the schema is wrong, or a required key is missing.
    '''
    with open(filename) as fh:
        record = json.load(fh)
    if record.get('schema') != SCHEMA:
        raise ValueError(f'{filename}: schema {record.get("schema")!r}, '
                         f'expected {SCHEMA}')
    if 'families' not in record:
        raise ValueError(f"{filename}: missing required key 'families'")
    for family in record['families']:
        if 'dt' not in family:
            raise ValueError(f'{filename}: family {family.get("label")!r} '
                             f'has no step sizes')
    return record


def assign_styles(panels):
    '''
    Line style, marker and color for every display name.

    Assigned across all families at once, because matplotlib restarts
    its color cycle on each new axes: without this a method would
    change color between columns as soon as two families stopped
    listing the same names in the same order.

    Parameters
    ----------
    panels : list of dict
        Per-family results, as returned by analyze.

    Returns
    -------
    dict
        Display name -> (line style, marker, color).
    '''
    names = []
    for panel in panels:
        for name in panel['names']:
            if name not in names:
                names.append(name)
    styles = {}
    for i, name in enumerate(names):
        ls, marker = STYLES.get(
            name, ('-', FALLBACK_MARKERS[i % len(FALLBACK_MARKERS)]))
        styles[name] = (ls, marker, PALETTE[i % len(PALETTE)])
    return styles


def place_label(ax, text, x, y, renderer, side='below'):
    '''
    Label a guide line where the label overlaps no line.

    Tries the preferred side of the guide at points along it, from its
    middle outwards, before the other side. Below the guide the label
    goes to the right of the point, then directly under it; above, to
    the left, then directly over it. Offsetting it sideways first keeps
    it off a rising guide however steep. The first placement that lies
    inside the axes, clear of the frame by the same gap as the guide,
    and touches no line or marker drawn on them is kept; if none does,
    the label takes the first placement tried. Call once the layout is
    final, since the test is made in display coordinates.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes holding the guide.
    text : str
        Label text.
    x, y : np.ndarray
        Points of the guide, on log-log axes.
    renderer : matplotlib.backend_bases.RendererBase
        Renderer used to measure the label.
    side : {'below', 'above'}, optional
        Side of the guide tried first; the one away from the data, so
        the label cannot be read as labeling a curve. Default 'below'.

    Returns
    -------
    matplotlib.text.Annotation
        The placed label.
    '''
    lx, ly = np.log(x), np.log(y)
    order = np.argsort(lx)
    lines = ax.get_lines()
    paths = [line.get_transform().transform_path(line.get_path())
             for line in lines]
    points = [line.get_transform().transform(line.get_xydata())
              for line in lines if line.get_marker() not in (None, 'None')]
    pad = renderer.points_to_pixels(plt.rcParams['lines.markersize'])
    gap = 4
    # The label keeps the same gap from the frame as from the guide.
    frame = ax.get_window_extent(renderer).padded(
        -renderer.points_to_pixels(gap))
    label = ax.annotate(text, xy=(x[0], y[0]), xytext=(0, 0),
                        textcoords='offset points')
    # Offset in points, then horizontal and vertical alignment.
    placements = {
        'below': [((gap, -gap), 'left', 'top'), ((0, -gap), 'center', 'top')],
        'above': [((-gap, gap), 'right', 'bottom'),
                  ((0, gap), 'center', 'bottom')],
    }
    other = 'above' if side == 'below' else 'below'
    candidates = []
    for sides in (placements[side], placements[other]):
        for fraction in (0.5, 0.3, 0.7, 0.1, 0.9):
            at = lx.min() + fraction * (lx.max() - lx.min())
            xy = (np.exp(at), np.exp(np.interp(at, lx[order], ly[order])))
            candidates += [(xy, placement) for placement in sides]

    def put(xy, placement):
        offset, ha, va = placement
        label.xy = xy
        label.set_position(offset)
        label.set_ha(ha)
        label.set_va(va)

    for xy, placement in candidates:
        put(xy, placement)
        box = label.get_window_extent(renderer)
        inside = (frame.x0 <= box.x0 and box.x1 <= frame.x1
                  and frame.y0 <= box.y0 and box.y1 <= frame.y1)
        clear = (not any(path.intersects_bbox(box, filled=False)
                         for path in paths)
                 and not any(box.padded(pad).contains(px, py)
                             for pts in points for px, py in pts))
        if inside and clear:
            return label
    put(*candidates[0])
    return label


def differences(states, dt, order):
    '''
    Estimated errors and observed orders for one sector.

    The returned errors are indexed by the coarser step of each pair, so
    they align with dt[:-1]; the orders are shorter again by one.

    Parameters
    ----------
    states : array_like
        Final states, shape (len(dt), 3), one per step size.
    dt : array_like
        Step sizes.
    order : int
        Nominal order, used to rescale the successive differences into
        error estimates.

    Returns
    -------
    delta : np.ndarray
        Estimated errors, shape (len(dt) - 1,).
    orders : np.ndarray
        Observed orders, shape (len(dt) - 2,).
    '''
    states = np.asarray(states, dtype=float)
    dt = np.asarray(dt, dtype=float)
    scale = 1.0 / (1.0 - 2.0 ** -order)
    delta = scale * np.linalg.norm(np.diff(states, axis=0), axis=1)
    orders = (np.log(delta[:-1] / delta[1:])
              / np.log(dt[:-2] / dt[1:-1]))
    return delta, orders


def analyze(record):
    '''
    Errors and orders for every method in every family.

    Parameters
    ----------
    record : dict
        Record read from a data file.

    Returns
    -------
    list of dict
        One dict per family, carrying the display label, the nominal
        order, the step sizes the errors correspond to, and a per-method
        dict of 'x'/'u' errors and orders.
    '''
    panels = []
    for family in record['families']:
        dt = np.asarray(family['dt'], dtype=float)
        methods = {}
        for name in family['method_order']:
            entry = family['methods'][name]
            methods[name] = {}
            for key, _ in SECTORS:
                delta, orders = differences(entry[key], dt, family['order'])
                methods[name][key] = delta
                methods[name][key + '_order'] = orders
        panels.append({'label': family['label'], 'order': family['order'],
                       'names': family['method_order'],
                       'dt': dt[:-1], 'methods': methods})
    return panels


def print_tables(record, panels):
    '''
    Print the estimated errors and observed orders for every family.

    Parameters
    ----------
    record : dict
        Record read from a data file; supplies the heading.
    panels : list of dict
        Per-family results, as returned by analyze.
    '''
    print(f'\n{record.get("experiment", "Richardson self-convergence")}')
    for panel in panels:
        print('#' * 78)
        print(f'# {panel["label"]} methods')
        print('#' * 78)
        header = (f'{"Method":>28} | '
                  + ' '.join(f'{f"dt={d:.5f}":>11}' for d in panel['dt']))
        for key, sector in SECTORS:
            print(f'Estimated {sector} error and observed orders')
            print(header)
            for name in panel['names']:
                m = panel['methods'][name]
                row = ' '.join(f'{d:>11.3e}' for d in m[key])
                orders = ','.join(f'{o:.2f}' for o in m[key + '_order'])
                print(f'{name:>28} | {row}   order: {orders}')
            print()


def plot(record, panels, filename, layout='screen'):
    '''
    Log-log convergence plots: sectors down the rows, families across.

    Parameters
    ----------
    record : dict
        Record read from a data file; supplies the title.
    panels : list of dict
        Per-family results, as returned by analyze.
    filename : str or Path
        Output figure path.
    layout : {'screen', 'print'}, optional
        'screen' (default) draws the panels large under a title; 'print'
        draws them at the journal width, labeled (a), (b), ..., with one
        legend below and no figure title, since the caption belongs to
        the paper.

    Returns
    -------
    matplotlib.figure.Figure
        The figure, already saved to filename.
    '''
    styles = assign_styles(panels)
    n_fam = len(panels)
    figsize = ((PRINT_WIDTH, 0.8 * PRINT_WIDTH) if layout == 'print'
               else (6.2 * n_fam, 9.6))
    fig, axes = plt.subplots(2, n_fam, figsize=figsize,
                             squeeze=False, sharex='col', sharey='row')

    # Guide at the nominal order, set below every curve so it reads as a
    # guide rather than overplotting a method.
    guide_lines = {}
    for col, panel in enumerate(panels):
        dt = panel['dt']
        for key, _ in SECTORS:
            lowest = min(panel['methods'][n][key][-1] for n in panel['names'])
            guide_lines[col, key] = (
                0.2 * lowest * (dt / dt[-1]) ** panel['order'])

    # A common vertical range per row keeps the two families on the same
    # scale, so the gap between second and fourth order reads directly
    # off the figure. The range covers the guides as well as the data,
    # with three times matplotlib's default headroom, so the guide labels
    # have room below the guides.
    limits = {}
    for key, _ in SECTORS:
        vals = np.log10(np.concatenate(
            [p['methods'][n][key] for p in panels for n in p['names']]
            + [guide_lines[col, key] for col in range(n_fam)]))
        pad = 0.15 * (vals.max() - vals.min())
        limits[key] = (10 ** (vals.min() - pad), 10 ** (vals.max() + pad))

    guides = []
    for col, panel in enumerate(panels):
        dt, slope = panel['dt'], panel['order']
        for row, (key, sector) in enumerate(SECTORS):
            ax = axes[row][col]
            for name in panel['names']:
                ls, marker, color = styles[name]
                ax.loglog(dt, panel['methods'][name][key], ls=ls,
                          marker=marker, color=color, label=name)
            guide = guide_lines[col, key]
            guide_label = rf'$O\left(\Delta t^{{{slope}}}\right)$'
            ax.plot(dt, guide, ls='--', color='0.5', label=guide_label)
            guides.append((ax, guide_label, dt, guide))
            ax.set_xscale('log', base=2)
            ax.set_yscale('log')
            ax.set_ylim(*limits[key])
            if col == 0:
                ax.set_ylabel(rf'Estimated $\|{key}_N - {key}(T)\|_2$')
            ax.set_title(f'{sector.capitalize()} ({panel["label"]})')
            if layout != 'print':
                ax.legend(loc='lower right')

    if layout == 'print':
        # Panels are labeled by letter alone; the caption describes them.
        for i, ax in enumerate(axes.flat):
            ax.set_title(f'({chr(ord("a") + i)})')
        for ax in axes[-1]:
            ax.set_xlabel(r'$\Delta t$')
        # One legend for the methods under the figure rather than one per
        # panel, which would cover the data at print size. Entries are
        # gathered from every panel, in case the families list different
        # methods.
        entries = {}
        for ax in axes.flat:
            for handle, label in zip(*ax.get_legend_handles_labels()):
                if label in styles:
                    entries.setdefault(label, handle)
        labels = [name for name in styles if name in entries]
        handles = [entries[name] for name in labels]
        legend_height = 0.5
        fig.tight_layout(rect=(0, legend_height / fig.get_figheight(), 1, 1))
        fig.legend(handles, labels, loc='lower center', ncol=3,
                   frameon=False)
        # Each guide is labeled on its panel, since its slope differs
        # between columns and so cannot share one legend entry.
        renderer = fig.canvas.get_renderer()
        for ax, text, x, y in guides:
            place_label(ax, text, x, y, renderer)
    else:
        experiment = record.get('experiment')
        # 'parameters' is optional in the file contract, like
        # 'experiment', so reach for T defensively rather than
        # indexing: a record without it still plots, just untitled.
        final_time = record.get('parameters', {}).get('T')
        title = 'Richardson self-convergence'
        parts = [experiment] if experiment else [title]
        if final_time is not None:
            parts.append(rf'$(T_{{\mathrm{{end}}}} = {final_time:g})$')
        fig.suptitle(' '.join(parts))
        fig.supxlabel(r'$\Delta t$')
        fig.tight_layout()
    # Print figures keep their exact width; screen ones are cropped.
    fig.savefig(filename,
                bbox_inches=None if layout == 'print' else 'tight')
    print(f'Saved {filename}')
    return fig


def main():
    '''
    Plot the data files named on the command line, or all of them.

    Raises
    ------
    SystemExit
        If -o is given with several data files, or if any file is
        skipped.
    '''
    parser = argparse.ArgumentParser(
        description=' '.join(__doc__.split('\n\n')[0].split()))
    parser.add_argument('results', nargs='*',
                        help=f'data files, by path or bare name; '
                             f'omit to plot every .json in '
                             f'{DATA_DIR}/')
    parser.add_argument('-o', '--output', default=None,
                        help=f'output figure (default: {PLOT_DIR}/<name>'
                             f'{FIGURE_SUFFIX}, or {PRINT_SUFFIX} with '
                             f'--print)')
    parser.add_argument('--print', action='store_true', dest='print_layout',
                        help='draw at the print size of the journal')
    args = parser.parse_args()

    use_latex()
    layout, suffix = 'screen', FIGURE_SUFFIX
    if args.print_layout:
        use_print_layout()
        layout, suffix = 'print', PRINT_SUFFIX

    files = ([resolve(name) for name in args.results] if args.results
             else find_all())
    if args.output and len(files) > 1:
        raise SystemExit('-o takes a single data file; with several, '
                         'each figure is named after its own input')

    if not args.output:
        PLOT_DIR.mkdir(parents=True, exist_ok=True)

    failed = 0
    for path in files:
        output = args.output or str(
            PLOT_DIR / (path.stem + suffix))
        try:
            record = load(path)
        except (ValueError, KeyError) as exc:
            # Report and carry on rather than abandoning the batch: a
            # results folder may hold unrelated JSON, and one bad file
            # should not cost the figures for the good ones.
            print(f'{path}: skipped ({exc})')
            failed += 1
            continue
        print(f'\n{path}')
        panels = analyze(record)
        print_tables(record, panels)
        plt.close(plot(record, panels, output, layout))
    if failed:
        raise SystemExit(f'{failed} file(s) skipped')


if __name__ == '__main__':
    main()
