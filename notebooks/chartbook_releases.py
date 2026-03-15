"""
Chartbook release configuration.

Maps ~45 data releases to their representative CSV files.
Used by Run_Chartbook.ipynb to generate chartbook_sources.json
for the website landing page.

Each entry:
  name     — short display name for the table
  csv      — representative CSV in chartbook/data/
  freq     — monthly | quarterly | annual | weekly
  date_col — column index for the date (default 0); None if no date column
  override — static string for releases with no parseable date
"""

RELEASES = [
    # === Monthly ===
    {"name": "Employment Situation", "csv": "nfp_monthly.csv", "freq": "monthly"},
    {"name": "Consumer Price Index", "csv": "cpi_monthly.csv", "freq": "monthly"},
    {"name": "Personal Income & Outlays", "csv": "pce_pi.csv", "freq": "monthly"},
    {"name": "Producer Price Index", "csv": "ppi.csv", "freq": "monthly"},
    {"name": "Import/Export Prices", "csv": "mxpi.csv", "freq": "monthly"},
    {"name": "JOLTS", "csv": "jolts.csv", "freq": "monthly"},
    {"name": "International Trade", "csv": "tradelt.csv", "freq": "monthly"},
    {"name": "Retail Sales", "csv": "rs_ind.csv", "freq": "monthly"},
    {"name": "Industrial Production", "csv": "indpro.csv", "freq": "monthly"},
    {"name": "Construction Spending", "csv": "construction_spending.csv", "freq": "monthly"},
    {"name": "Housing Starts", "csv": "hhform.csv", "freq": "monthly"},
    {"name": "New Home Sales", "csv": "nhs.csv", "freq": "monthly"},
    {"name": "House Prices", "csv": "hpi.csv", "freq": "monthly"},
    {"name": "Consumer Sentiment", "csv": "umichsoc.csv", "freq": "monthly"},
    {"name": "Capital Goods Orders", "csv": "dgno.csv", "freq": "monthly"},
    {"name": "Wage Tracker", "csv": "atl_wgt2.csv", "freq": "monthly"},
    {"name": "Consumer Expectations", "csv": "sce_job_separation.csv", "freq": "monthly"},
    {"name": "CPS Microdata", "csv": "uwe_cps.csv", "freq": "monthly"},
    {"name": "State Unemployment", "csv": "state_unrate.csv", "freq": "monthly"},
    {"name": "Metro Unemployment", "csv": "msa_unemp_rate.csv", "freq": "monthly",
     "date_from_headers": True},
    {"name": "TIC Capital Flows", "csv": "tic_bond.csv", "freq": "monthly"},
    {"name": "Treasury Statement", "csv": "tmb_rec.csv", "freq": "monthly"},
    {"name": "Money Stock (M2)", "csv": "m2.csv", "freq": "monthly"},
    {"name": "Exchange Rate (REER)", "csv": "reer.csv", "freq": "monthly"},
    {"name": "Electricity", "csv": "elec_prod.csv", "freq": "monthly"},
    {"name": "Oil Production", "csv": "oil_prodq.csv", "freq": "monthly"},
    {"name": "Mortgage Rates", "csv": "mortgage.csv", "freq": "weekly"},
    {"name": "Initial Jobless Claims", "csv": "icsa.csv", "freq": "weekly"},
    {"name": "Fed Balance Sheet", "csv": "fed_assets.csv", "freq": "weekly"},

    # === Quarterly ===
    {"name": "GDP", "csv": "gdp_rec2.csv", "freq": "quarterly",
     "estimate_log": "../gdp_estimate_log.csv"},
    {"name": "GDP by Industry", "csv": "gdpva_pc.csv", "freq": "quarterly"},
    {"name": "GDP by State", "csv": "gdpva_pc.csv", "freq": "quarterly"},
    {"name": "Corporate Profits", "csv": "cprof.csv", "freq": "quarterly"},
    {"name": "Employment Costs", "csv": "eci_yy.csv", "freq": "quarterly"},
    {"name": "Financial Accounts", "csv": "hhdebt.csv", "freq": "quarterly"},
    {"name": "Current Account", "csv": "cab.csv", "freq": "quarterly"},
    {"name": "International Investment", "csv": "iip.csv", "freq": "quarterly"},
    {"name": "Public Debt", "csv": "pubdebt.csv", "freq": "quarterly"},
    {"name": "Labor Productivity", "csv": "lprod_hrsa.csv", "freq": "quarterly"},
    {"name": "Homeownership", "csv": "homeown.csv", "freq": "quarterly"},
    {"name": "FOMC Projections", "csv": "sep.csv", "freq": "quarterly"},

    # === Annual / Infrequent ===
    {"name": "CPS ASEC", "csv": "poor_age_latest.csv", "freq": "annual",
     "date_col": None, "override": "2024"},
    {"name": "American Community Survey", "csv": "acs_cz_age.csv", "freq": "annual",
     "date_col": None, "override": "2023"},
    {"name": "Consumer Expenditures", "csv": "ce_table.csv", "freq": "annual"},
    {"name": "Life Expectancy", "csv": "life_exp.csv", "freq": "annual"},
    {"name": "Productivity (TFP)", "csv": "tfp.csv", "freq": "annual"},
    {"name": "Survey of Consumer Finances", "csv": "wealth_dist.csv", "freq": "triennial",
     "date_col": None, "override": "2022"},
]

# Daily/weekly series mentioned in text, not tabled
ST_SERIES = [
    "interest rates", "Treasury yields", "yield curve",
    "S&P 500", "VIX", "gold", "oil prices",
    "exchange rates", "GDPNow", "inflation nowcasts",
    "Fed funds projections",
]
