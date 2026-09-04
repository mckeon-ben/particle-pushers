'''
Reads JSON data files turning the stored final states into error
estimates and observed orders, prints the tables and draws the figure.

File contract
-------------
Required: 'schema' and 'families'. Each family carries 'label',
'order', 'method_order', 'methods' -- the last mapping each display
name to 'x' and 'u' arrays holding one final state per entry of dt --
and its own 'dt', the step sizes those states were computed at. Steps
are per family because a fourth-order method's error falls four
decades faster than a second-order one over the same range, so a
single shared list tends to leave one family pre-asymptotic while the
other has already reached the round-off floor.

Optional 'experiment' names the problem in the figure title, and
'parameters' may carry 'T', the final lab time, which is appended to
it. Anything else in the file is ignored. Display names not listed in
STYLES get a style from a fallback cycle rather than raising.

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
cent at the affected pair and look like real behaviour.

'''

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


SCHEMA = 1

# Where results files live when they are not given by an explicit path.
# A bare name on the command line is resolved against the working
# directory first and then here, and passing no name at all plots every
# results file in this folder.
DATA_DIR = Path('data')

# Where figures are written, kept apart from the data folder so one
# directory holds only inputs and the other only outputs.
PLOT_DIR = Path('plots')

# Default output format. Vector, so the figure stays sharp at any zoom
# and at whatever size a paper puts it. Pass -o with another extension
# to override; matplotlib picks the writer from the suffix.
FIGURE_SUFFIX = '.pdf'

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
    # Embed TrueType rather than the Type 3 fonts matplotlib writes into
    # PDFs by default.
    'pdf.fonttype': 42,
})


def use_latex(enabled=USETEX):
    '''Switch LaTeX typesetting on or off.

    Under usetex the mathtext and font.sans-serif settings above are
    ignored, since the preamble decides both; they stay in place as the
    fallback for when this is off.
    '''
    plt.rcParams.update({
        'text.usetex': enabled,
        'text.latex.preamble': LATEX_PREAMBLE if enabled else '',
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

# Okabe-Ito, the standard colourblind-safe qualitative palette.
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
    '''Find a data file given a path, a filename, or a bare stem.

    Tries the working directory before DATA_DIR so an explicit path
    always wins, and appends the .json suffix only when the name does
    not already carry one -- appending unconditionally would mangle a
    name that has dots in it for other reasons.
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
    '''Every data file in DATA_DIR, sorted by name.'''
    files = sorted(DATA_DIR.glob('*.json'))
    if not files:
        raise SystemExit(f'no .json files found in {DATA_DIR}/')
    return files


def load(filename):
    '''Read a data file, checking the schema and the required keys.'''
    with open(filename) as fh:
        record = json.load(fh)
    if record.get('schema') != SCHEMA:
        raise ValueError(f'{filename}: schema {record.get("schema")!r}, '
                         f'expected {SCHEMA}')
    if 'families' not in record:
        raise ValueError(f'{filename}: missing required key {"families"!r}')
    for family in record['families']:
        if 'dt' not in family:
            raise ValueError(f'{filename}: family {family.get("label")!r} '
                             f'has no step sizes')
    return record


def assign_styles(panels):
    '''Line style, marker and colour for every display name.

    Assigned across all families at once, because matplotlib restarts
    its colour cycle on each new axes: without this a method would
    change colour between columns as soon as two families stopped
    listing the same names in the same order.
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


def differences(states, dt, order):
    '''Estimated errors and observed orders for one sector.

    states has shape (len(dt), 3), one final state per step size. The
    returned errors are indexed by the coarser step of each pair, so
    they align with dt[:-1]; the orders are shorter again by one.
    '''
    states = np.asarray(states, dtype=float)
    dt = np.asarray(dt, dtype=float)
    scale = 1.0 / (1.0 - 2.0 ** -order)
    delta = scale * np.linalg.norm(np.diff(states, axis=0), axis=1)
    orders = (np.log(delta[:-1] / delta[1:])
              / np.log(dt[:-2] / dt[1:-1]))
    return delta, orders


def analyse(record):
    '''Errors and orders for every method in every family.

    Returns a list of per-family dicts carrying the display label, the
    nominal order, the step sizes the errors correspond to, and a
    per-method dict of 'x'/'u' errors and orders.
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


def print_tables(panels):
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


def plot(panels, record, filename):
    '''Log-log convergence plots: sectors down the rows, families across.'''
    styles = assign_styles(panels)
    n_fam = len(panels)
    fig, axes = plt.subplots(2, n_fam, figsize=(6.2 * n_fam, 9.6),
                             squeeze=False, sharex='col', sharey='row')

    # A common vertical range per row keeps the two families on the same
    # scale, so the gap between second and fourth order reads directly
    # off the figure.
    limits = {}
    for key, _ in SECTORS:
        vals = np.concatenate([p['methods'][n][key]
                               for p in panels for n in p['names']])
        limits[key] = (0.1 * vals.min(), 5.0 * vals.max())

    for col, panel in enumerate(panels):
        dt, slope = panel['dt'], panel['order']
        for row, (key, sector) in enumerate(SECTORS):
            ax = axes[row][col]
            for name in panel['names']:
                ls, marker, colour = styles[name]
                ax.loglog(dt, panel['methods'][name][key], ls=ls,
                          marker=marker, color=colour, label=name)
            # Guide at the nominal order, set below every curve so it
            # reads as a guide rather than overplotting a method.
            lowest = min(panel['methods'][n][key][-1] for n in panel['names'])
            guide = 0.2 * lowest * (dt / dt[-1]) ** slope
            ax.plot(dt, guide, ls='--', color='k',
                    label=rf'$O\left(\Delta t^{{{slope}}}\right)$')
            ax.set_xscale('log', base=2)
            ax.set_yscale('log')
            ax.set_ylim(*limits[key])
            if col == 0:
                ax.set_ylabel(f'Estimated {sector} error')
            ax.set_title(f'{sector.capitalize()} ({panel["label"]})')
            ax.grid(True, which='both', alpha=0.3)
            ax.legend(fontsize=8, loc='lower right')

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
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    print(f'Saved convergence plot to {filename}')
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('results', nargs='*',
                        help=f'data files, by path or bare name; '
                             f'omit to plot every .json in '
                             f'{DATA_DIR}/')
    parser.add_argument('-o', '--output', default=None,
                        help=f'output figure (default: {PLOT_DIR}/<name>'
                             f'{FIGURE_SUFFIX})')
    args = parser.parse_args()

    use_latex()

    files = ([resolve(name) for name in args.results] if args.results
             else find_all())
    if args.output and len(files) > 1:
        raise SystemExit('-o takes a single results file; with several, '
                         'each figure is named after its own input')

    if not args.output:
        PLOT_DIR.mkdir(parents=True, exist_ok=True)

    failed = 0
    for path in files:
        output = args.output or str(
            PLOT_DIR / (path.stem + FIGURE_SUFFIX))
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
        panels = analyse(record)
        print()
        print_tables(panels)
        plt.close(plot(panels, record, output))
    if failed:
        raise SystemExit(f'{failed} file(s) skipped')


if __name__ == '__main__':
    main()
