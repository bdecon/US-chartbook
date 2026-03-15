"""
Build chartbook_sources.json for the website landing page.

Reads each release's representative CSV, extracts the latest data date,
formats it for display, and writes JSON to the website repo.

Run from: uschartbook/notebooks/
Output:   bdecon.github.io/files/chartbook_sources.json
"""

import csv
import json
import datetime
from pathlib import Path

from chartbook_releases import RELEASES, ST_SERIES

DATA_DIR = Path('../chartbook/data')
import os as _os
_web = _os.getenv('WEBSITE_DIR')
OUTPUT = Path(_web) / 'chartbook_sources.json' if _web else None
SNAPSHOT = Path('../sources_snapshot.json')

MONTH_NAMES = [
    '', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
]

QUARTER_MAP = {1: 'Q1', 2: 'Q1', 3: 'Q1',
               4: 'Q2', 5: 'Q2', 6: 'Q2',
               7: 'Q3', 8: 'Q3', 9: 'Q3',
               10: 'Q4', 11: 'Q4', 12: 'Q4'}


def parse_date(s):
    """Parse a date string from CSV. Handles YYYY-MM-DD and YYYY-MM-DD HH:MM:SS."""
    s = s.strip().split(' ')[0]  # drop time portion
    return datetime.date.fromisoformat(s)


def last_date_from_headers(csv_path):
    """Extract the latest date from column headers like 'Dec 2025', 'Jan 2026'."""
    with open(csv_path) as f:
        headers = next(csv.reader(f))
    dates = []
    for h in headers:
        h = h.strip()
        # Match patterns like "Dec 2025", "Jan 2026" (ignore suffixed ones like "Dec 2025 3M")
        parts = h.split()
        if len(parts) == 2:
            try:
                dt = datetime.datetime.strptime(h, '%b %Y').date()
                dates.append(dt)
            except ValueError:
                continue
    return max(dates) if dates else None


def last_date_from_csv(csv_path, date_col=0, cutoff=None):
    """Read the last date from a CSV file.

    Skips rows where the first value column after the date is empty
    (placeholder rows with no data yet).
    If cutoff is given, returns the latest date <= cutoff.
    """
    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        dates = []
        for row in reader:
            try:
                d = parse_date(row[date_col])
                # Skip placeholder rows with empty data
                val_col = date_col + 1
                if val_col < len(row) and row[val_col].strip() == '':
                    continue
                dates.append(d)
            except (ValueError, IndexError):
                continue

    if not dates:
        return None

    if cutoff:
        dates = [d for d in dates if d <= cutoff]

    return max(dates) if dates else None


def format_period(dt, freq):
    """Format a date into a human-readable reference period."""
    if freq == 'monthly':
        return f'{MONTH_NAMES[dt.month]} {dt.year}'
    elif freq == 'quarterly':
        q = QUARTER_MAP[dt.month]
        return f'{q} {dt.year}'
    elif freq == 'weekly':
        return f'{MONTH_NAMES[dt.month]} {dt.day}, {dt.year}'
    elif freq in ('annual', 'triennial'):
        return str(dt.year)
    return str(dt)


def get_gdp_period(release):
    """Special handling for GDP: read from estimate log."""
    log_path = Path(release['estimate_log']).resolve()

    if not log_path.exists():
        return None

    with open(log_path) as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return None

    latest = rows[-1]
    q = latest['quarter']  # e.g. '2025Q4'
    estimate = latest['estimate']  # e.g. 'Advance'
    quarter_label = q[4:]  # 'Q4'
    year = q[:4]  # '2025'
    return {'period': f'{quarter_label} {year}', 'estimate': estimate}


def build_sources():
    today = datetime.date.today()
    results = []

    for rel in RELEASES:
        name = rel['name']
        freq = rel['freq']

        # GDP has special handling
        if rel.get('estimate_log'):
            gdp = get_gdp_period(rel)
            if gdp is None:
                results.append({'name': name, 'period': '—', 'freq': freq})
            else:
                gdp_name = f'{name} ({gdp["estimate"]})'
                results.append({'name': gdp_name, 'period': gdp['period'], 'freq': freq})
            continue

        # Static override (no date column)
        if rel.get('date_col') is None and 'override' in rel:
            results.append({'name': name, 'period': rel['override'], 'freq': freq})
            continue

        # Read from CSV
        csv_path = DATA_DIR / rel['csv']
        if not csv_path.exists():
            results.append({'name': name, 'period': '—', 'freq': freq})
            continue

        # Dates in column headers (e.g., msa_unemp_rate.csv)
        if rel.get('date_from_headers'):
            dt = last_date_from_headers(csv_path)
        else:
            date_col = rel.get('date_col', 0)
            # SEP has future dates — filter to today
            cutoff = today if rel['csv'] == 'sep.csv' else None
            dt = last_date_from_csv(csv_path, date_col, cutoff)
        if dt is None:
            results.append({'name': name, 'period': '—', 'freq': freq})
            continue

        period = format_period(dt, freq)
        results.append({'name': name, 'period': period, 'freq': freq})

    return results


def detect_freshness(results):
    """Compare current results to previous snapshot to detect changes.

    Returns results with 'last_changed' dates added.
    """
    # Load previous snapshot
    prev = {}
    if SNAPSHOT.exists():
        with open(SNAPSHOT) as f:
            for item in json.load(f).get('releases', []):
                prev[item['name']] = item

    today_str = datetime.date.today().isoformat()

    for r in results:
        name = r['name']
        if name in prev and prev[name].get('period') == r['period']:
            # Period unchanged — keep previous last_changed
            r['last_changed'] = prev[name].get('last_changed', today_str)
        else:
            # New or changed — mark as fresh today
            r['last_changed'] = today_str

    return results


def main():
    results = build_sources()
    results = detect_freshness(results)

    # Sort by last_changed descending for "most recent" selection
    by_freshness = sorted(results, key=lambda r: r['last_changed'], reverse=True)
    recent = by_freshness[:10]

    output = {
        'generated': datetime.date.today().isoformat(),
        'releases': results,
        'recent': recent,
        'st_series': ST_SERIES,
    }

    # Write JSON to website
    if OUTPUT:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT, 'w') as f:
            json.dump(output, f, indent=2)

    # Save snapshot for next comparison
    snapshot = {
        'generated': output['generated'],
        'releases': results,
    }
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    with open(SNAPSHOT, 'w') as f:
        json.dump(snapshot, f, indent=2)

    if OUTPUT:
        print(f'Wrote {len(results)} releases to {OUTPUT}')
    else:
        print(f'Processed {len(results)} releases (WEBSITE_DIR not set, skipped website export)')
    print(f'Recent: {", ".join(r["name"] for r in recent)}')


if __name__ == '__main__':
    main()
