"""
One-time bootstrap: walk local git history of the uschartbook repo to find
when each release's representative CSV last had its final data date change.

This seeds sources_snapshot.json so the freshness rotation works immediately.
"""

import csv
import json
import subprocess
import datetime
import tempfile
from pathlib import Path

from chartbook_releases import RELEASES
from build_sources_json import (
    parse_date, last_date_from_csv, last_date_from_headers,
    format_period, get_gdp_period, QUARTER_MAP, MONTH_NAMES,
)

REPO = Path(__file__).resolve().parent.parent
DATA_REL = 'chartbook/data'
SNAPSHOT = Path('../sources_snapshot.json')


def git_log_dates(n=50):
    """Get the last N commit hashes and dates."""
    result = subprocess.run(
        ['git', 'log', f'-{n}', '--format=%H %aI'],
        capture_output=True, text=True, cwd=REPO,
    )
    commits = []
    for line in result.stdout.strip().split('\n'):
        if not line:
            continue
        parts = line.split(' ', 1)
        commits.append((parts[0], parts[1][:10]))  # hash, YYYY-MM-DD
    return commits


def git_show_file(commit, path):
    """Get file contents at a given commit. Returns None if file doesn't exist."""
    result = subprocess.run(
        ['git', 'show', f'{commit}:{path}'],
        capture_output=True, text=True, cwd=REPO,
    )
    if result.returncode != 0:
        return None
    return result.stdout


def extract_last_date_from_content(content, rel, commit_date):
    """Extract the formatted period from CSV content string."""
    freq = rel['freq']

    if rel.get('date_from_headers'):
        lines = content.split('\n')
        if not lines:
            return None
        headers = next(csv.reader([lines[0]]))
        dates = []
        for h in headers:
            h = h.strip()
            parts = h.split()
            if len(parts) == 2:
                try:
                    dt = datetime.datetime.strptime(h, '%b %Y').date()
                    dates.append(dt)
                except ValueError:
                    continue
        if not dates:
            return None
        return format_period(max(dates), freq)

    # Normal row-based date
    lines = content.strip().split('\n')
    if len(lines) < 2:
        return None

    reader = csv.reader(lines[1:])  # skip header
    date_col = rel.get('date_col', 0)
    if date_col is None:
        return rel.get('override')

    dates = []
    for row in reader:
        try:
            d = parse_date(row[date_col])
            dates.append(d)
        except (ValueError, IndexError):
            continue

    if not dates:
        return None

    # SEP: filter to commit date
    if rel['csv'] == 'sep.csv':
        cutoff = datetime.date.fromisoformat(commit_date)
        dates = [d for d in dates if d <= cutoff]
        if not dates:
            return None

    return format_period(max(dates), freq)


def main():
    commits = git_log_dates(50)
    print(f'Found {len(commits)} commits')

    # For each release, walk history to find when its period last changed
    results = {}

    for rel in RELEASES:
        name = rel['name']

        # Skip GDP (uses estimate log) and overrides (static)
        if rel.get('estimate_log'):
            results[name] = {
                'name': name,
                'period': get_gdp_period(rel) or '—',
                'freq': rel['freq'],
                'last_changed': commits[0][1],  # use latest commit date
            }
            continue

        if rel.get('date_col') is None and 'override' in rel:
            results[name] = {
                'name': name,
                'period': rel['override'],
                'freq': rel['freq'],
                'last_changed': commits[-1][1],  # old — static data
            }
            continue

        csv_rel_path = f'{DATA_REL}/{rel["csv"]}'

        # Get current period from HEAD
        current_content = git_show_file(commits[0][0], csv_rel_path)
        if current_content is None:
            results[name] = {
                'name': name, 'period': '—', 'freq': rel['freq'],
                'last_changed': commits[0][1],
            }
            continue

        current_period = extract_last_date_from_content(
            current_content, rel, commits[0][1])

        # Walk back through history to find when it changed
        last_changed = commits[0][1]
        for commit_hash, commit_date in commits[1:]:
            old_content = git_show_file(commit_hash, csv_rel_path)
            if old_content is None:
                break
            old_period = extract_last_date_from_content(
                old_content, rel, commit_date)
            if old_period != current_period:
                break
            last_changed = commit_date

        results[name] = {
            'name': name,
            'period': current_period or '—',
            'freq': rel['freq'],
            'last_changed': last_changed,
        }
        print(f'  {name:40s} {current_period or "—":20s} changed: {last_changed}')

    # Write snapshot
    snapshot = {
        'generated': datetime.date.today().isoformat(),
        'releases': list(results.values()),
    }
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    with open(SNAPSHOT, 'w') as f:
        json.dump(snapshot, f, indent=2)
    print(f'\nWrote snapshot with {len(results)} entries')


if __name__ == '__main__':
    main()
