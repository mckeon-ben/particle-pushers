'''
Reads JSON data files, turning the stored final states into error
estimates and observed orders, prints the tables and draws the figure.

File contract
-------------
Required: 'schema' and 'families'. Each family carries 'label',
'order', 'dt' (its step sizes), 'method_order' and 'methods', which
maps each display name to 'time' (seconds taken) and to 'x' and 'u',
one final state per entry of 'dt'. Each family has its own 'dt'
because a fourth-order error falls twice as many decades as a
second-order one over the same range of steps.

Optional: 'experiment', which heads the printed tables, and
'parameters', whose 'T' (the final lab time) is shown on the figure.
'''

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCHEMA = 1

# Inputs and outputs, beside this script wherever it is run from.
DATA_DIR = Path(__file__).resolve().parent / 'data'
PLOT_DIR = Path(__file__).resolve().parent / 'plots'

# Page widths in inches, drawn at final size so nothing is rescaled:
# the journal text width (174 mm), an A4 page with 25 mm margins
# (160 mm), and matplotlib's default. The layouts differ only in width,
# lettering and renderer; only the default needs no LaTeX.
JOURNAL_WIDTH = 174 / 25.4
THESIS_WIDTH = 160 / 25.4
DEFAULT_WIDTH = 6.4

# The journal sets its figures in Helvetica, math included.
SANS_PREAMBLE = r'''
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

# Fallbacks when LaTeX is off; matplotlib takes the first installed.
SANS_FONTS = ['Helvetica', 'Arial', 'TeX Gyre Heros', 'Nimbus Sans',
              'Liberation Sans', 'FreeSans', 'DejaVu Sans']

# Computer Modern is LaTeX's default, so the thesis needs no package.
CM_PREAMBLE = ''
CM_FONTS = ['cmr10', 'DejaVu Serif']

DEFAULT_FONTS = ['DejaVu Sans']

# Distinct suffixes, so drawing for two pages leaves two files.
LAYOUTS = {
    'journal': {'width': JOURNAL_WIDTH, 'font_size': 10, 'line_width': 1.0,
                'suffix': '-journal.eps', 'usetex': True,
                'preamble': SANS_PREAMBLE, 'family': 'sans-serif',
                'fonts': SANS_FONTS, 'mathtext': 'custom',
                # Set the math in the same family as the text.
                'rc': {'mathtext.rm': 'sans', 'mathtext.it': 'sans:italic',
                       'mathtext.bf': 'sans:bold', 'mathtext.cal': 'sans'}},
    'thesis': {'width': THESIS_WIDTH, 'font_size': 10, 'line_width': 1.0,
               'suffix': '-thesis.pdf', 'usetex': True,
               'preamble': CM_PREAMBLE, 'family': 'serif',
               'fonts': CM_FONTS, 'mathtext': 'cm'},
    'default': {'width': DEFAULT_WIDTH, 'font_size': 10, 'line_width': 1.0,
                'suffix': '.pdf', 'usetex': False, 'preamble': '',
                'family': 'sans-serif', 'fonts': DEFAULT_FONTS,
                'mathtext': 'dejavusans'},
}

# Height over width, the same on every page.
FIGURE_RATIO = 0.8

# Settings shared by every layout.
plt.rcParams.update({
    'axes.grid': True,
    'grid.color': '0.9',
    'figure.dpi': 150,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})


def use_layout(name):
    '''
    Set the lettering, line widths and renderer of one layout.

    The font list and mathtext set matter only when LaTeX is off.

    Parameters
    ----------
    name : str
        Key of LAYOUTS.

    Returns
    -------
    dict
        The layout, whose width and suffix the caller still needs.
    '''
    layout = LAYOUTS[name]
    size, width = layout['font_size'], layout['line_width']
    plt.rcParams.update({
        'text.usetex': layout['usetex'],
        'text.latex.preamble': layout['preamble'],
        'font.family': layout['family'],
        f'font.{layout["family"]}': layout['fonts'],
        'mathtext.fontset': layout['mathtext'],
        'axes.formatter.use_mathtext': True,
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
    plt.rcParams.update(layout.get('rc', {}))
    return layout


# Fixed markers by display name; other names use FALLBACK_MARKERS.
STYLES = {
    'Boris': ('-', 'o'),
    'Vay': ('-', 's'),
    'Higuera-Cary': ('-', '^'),
    'Gordon-Hafizi (quadratic)': ('-', 'v'),
    'Gordon-Hafizi (exact)': ('-', 'D'),
}

# Okabe-Ito colorblind-safe palette.
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

    Looks in the working directory, then DATA_DIR, adding .json if
    the name lacks it.

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

    Assigned across all families at once, so a method keeps its color
    in every panel.

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

    Tries points along the guide from its middle outwards, on the
    preferred side first, and keeps the first placement inside the
    axes that touches no line or marker; failing that, the first one
    tried. Call once the layout is final, since the test is made in
    display coordinates.

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
        Side of the guide tried first. Default 'below'.

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

    Errors align with dt[:-1], the coarser step of each pair.

    Parameters
    ----------
    states : array_like
        Final states, shape (len(dt), 3), one per step size.
    dt : array_like
        Step sizes.
    order : int
        Nominal order, used to rescale differences into errors.

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
        One per family: label, order, step sizes, and per-method
        'x'/'u' errors and orders.
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


def step_label(h):
    '''
    Label a step size, as a power of two where it is one.

    Parameters
    ----------
    h : float
        Step size.

    Returns
    -------
    str
        Label such as '2^-9', or two significant figures otherwise.
    '''
    k = np.log2(h)
    return f'2^{round(k):d}' if abs(k - round(k)) < 1e-9 else f'{h:.2e}'


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
        names = panel['names']
        width = max(len(name) for name in names)
        header = (f'{"Method":>{width}} |'
                  + ''.join(f'{step_label(d):>9}' for d in panel['dt']))
        print('#' * len(header))
        print(f'# {panel["label"]} methods')
        print('#' * len(header))
        for key, sector in SECTORS:
            print(f'Estimated {sector} error and observed orders')
            print(header)
            for name in names:
                m = panel['methods'][name]
                print(f'{name:>{width}} |'
                      + ''.join(f'{d:>9.2e}' for d in m[key]))
                # Each order sits under the finer step of its pair.
                print(f'{"order":>{width}} |' + ' ' * 9
                      + ''.join(f'{o:>9.2f}' for o in m[key + '_order']))
            print()


def plot(record, panels, filename, layout='default'):
    '''
    Log-log convergence plots: sectors down the rows, families across.

    Panels are labeled (a), (b), ... with one shared legend below.

    Parameters
    ----------
    record : dict
        Record read from a data file.
    panels : list of dict
        Per-family results, as returned by analyze.
    filename : str or Path
        Output figure path.
    layout : str, optional
        Key of LAYOUTS, giving the figure width; lettering is already
        set by use_layout.

    Returns
    -------
    matplotlib.figure.Figure
        The figure, already saved to filename.
    '''
    styles = assign_styles(panels)
    n_fam = len(panels)
    page = LAYOUTS[layout]
    fig, axes = plt.subplots(
        2, n_fam, figsize=(page['width'], FIGURE_RATIO * page['width']),
        squeeze=False, sharex='col', sharey='row')

    # Nominal-order guides, set below every curve.
    guide_lines = {}
    for col, panel in enumerate(panels):
        dt = panel['dt']
        for key, _ in SECTORS:
            lowest = min(panel['methods'][n][key][-1] for n in panel['names'])
            guide_lines[col, key] = (
                0.2 * lowest * (dt / dt[-1]) ** panel['order'])

    # One vertical range per row, covering data and guides, with extra
    # headroom for the guide labels.
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
    for i, ax in enumerate(axes.flat):
        ax.set_title(f'({chr(ord("a") + i)})')
    for ax in axes[-1]:
        ax.set_xlabel(r'$\Delta t$')
    # Final time, bottom left of each row's first panel. The box is
    # opaque because EPS cannot store transparency.
    final_time = record.get('parameters', {}).get('T')
    if final_time is not None:
        for row in axes:
            row[0].annotate(rf'$T = {final_time:.4g}$',
                            xy=(0, 0), xycoords='axes fraction',
                            xytext=(10, 10), textcoords='offset points',
                            ha='left', va='bottom',
                            bbox={'boxstyle': 'round,pad=0.4',
                                  'facecolor': 'white', 'edgecolor': '0.7',
                                  'linewidth': plt.rcParams['axes.linewidth']})
    # One legend under the figure, gathered from every panel.
    entries = {}
    for ax in axes.flat:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label in styles:
                entries.setdefault(label, handle)
    labels = [name for name in styles if name in entries]
    handles = [entries[name] for name in labels]
    # Three columns, a point smaller than the text, to fit the
    # narrowest page.
    ncol = 3
    legend_height = 0.2 * -(-len(labels) // ncol)
    fig.tight_layout(rect=(0, legend_height / fig.get_figheight(), 1, 1))
    fig.legend(handles, labels, loc='lower center', ncol=ncol,
               frameon=False, fontsize=page['font_size'] - 1,
               handlelength=1.5, columnspacing=1.0, handletextpad=0.4)
    # Guides are labeled on their panels, as their slopes differ.
    renderer = fig.canvas.get_renderer()
    for ax, text, x, y in guides:
        place_label(ax, text, x, y, renderer)
    fig.savefig(filename)
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
                             f', with the suffix the layout calls for)')
    # At most one page; the default needs no LaTeX.
    page = parser.add_mutually_exclusive_group()
    page.add_argument('--journal', action='store_const', dest='layout',
                      const='journal',
                      help='draw at the text width and in the typeface '
                           'of the journal; needs LaTeX')
    page.add_argument('--thesis', action='store_const', dest='layout',
                      const='thesis',
                      help='draw at the text width of an A4 thesis, in '
                           'Computer Modern; needs LaTeX')
    parser.set_defaults(layout='default')
    args = parser.parse_args()

    layout = args.layout
    suffix = use_layout(layout)['suffix']

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
            # Skip a bad file rather than abandon the batch.
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
