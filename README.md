# US Chartbook

### Open source notes on US economic activity

[View the chartbook](https://www.bd-econ.com/chartbook.pdf)

Brian Dew, @bd_econ

----

## About

The US Chartbook is a set of notes describing social and economic developments in the United States since 1989. The goal is to strike a balance between data resources like FRED and professional analysis such as the IMF Article IV report. The chartbook is more curated than FRED, with structure, descriptive text, and added context. At the same time, it intends to be more detailed, stable, and up to date than the US Article IV report.

The chartbook is open source, so people are welcome to use it for their own projects or suggest edits. If you find an error, please let me know by e-mail or by creating an issue on GitHub.

The chartbook is currently version 0.1.8.

## Sections

- Summary and High-Frequency Indicators
- Overall Economic Activity
- Financial Accounts
- Households
- Businesses
- Government
- External Sector
- Labor Markets
- Financial Markets
- Prices
- Sources / References / Acknowledgments

## Project Structure

```
uschartbook/
├── chartbook/                 # LaTeX build directory
│   ├── chartbook.tex          #   Main document (~13,000 lines)
│   ├── preamble.tex           #   Packages, settings, custom commands
│   ├── glossary.tex           #   Glossary definitions
│   ├── data/                  #   CSV and TEX files read by chartbook.tex
│   └── text/                  #   Generated LaTeX text fragments
├── notebooks/                 # Jupyter notebooks (data retrieval and processing)
│   ├── raw/                   #   Static reference data (manually sourced CSVs)
│   └── shapefiles/            #   Geographic boundary files for map charts
├── src/uschartbook/           # Shared Python package
│   ├── config.py              #   Paths, API keys, HTTP headers
│   └── utils.py               #   Data retrieval and text generation utilities
├── gdp_estimate_log.csv       # Tracks GDP revision estimates
└── sources_snapshot.json      # Data freshness tracking
```

## Data Pipeline

The chartbook is built in two stages: data collection and document compilation.

**Stage 1: Data collection.** Jupyter notebooks in `notebooks/` fetch data from public APIs (BEA, BLS, Census, FRED, EIA, and others) and process it into chart-ready form. Each notebook writes CSV files to `chartbook/data/` and LaTeX text fragments to `chartbook/text/`. All notebooks import shared configuration and utilities from `src/uschartbook/`.

**Stage 2: Compilation.** The main LaTeX file (`chartbook/chartbook.tex`) reads the CSV and text files using `\input` and `pgfplotstableread` commands, producing the final PDF. The document is compiled with `lualatex` from the `chartbook/` directory.

```
Public APIs (BEA, BLS, Census, FRED, EIA, ...)
    │
    ▼
Jupyter Notebooks (notebooks/)
    │
    ├──▶ chartbook/data/   (CSV, TEX)
    ├──▶ chartbook/text/   (LaTeX fragments)
    │
    ▼
lualatex (chartbook/chartbook.tex)
    │
    ▼
chartbook.pdf
```

## To Do

See [list](https://github.com/bdecon/US-chartbook/issues) of open issues.
