# Data Notes

## `latamstocks.csv`

This file is a static local dataset used by the repository’s structured-data demo examples, especially `examples/example5.py`.

### Columns

- `Date` — trading date in `YYYY-MM-DD` format
- `Ticker` — equity ticker symbol
- `Close` — daily closing price
- `Volume` — daily trading volume

### Current shape

- Format: CSV
- Delimiter: comma
- Header: `Date,Ticker,Close,Volume`

### Intended use

This dataset is included for educational and demonstration purposes only. It is used to illustrate:
- local CSV / SQL-style querying
- agent interaction with structured tabular data
- simple equity data lookups and aggregations

### Limitations

- This is a static local file, not a live market feed.
- The repository does not treat this dataset as audited, authoritative, or production-grade market data.
- The file should be understood as an example dataset for demos and experiments.
- Any downstream analysis remains dependent on the quality and completeness of the source file.
