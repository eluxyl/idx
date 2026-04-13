# idx

For Mortgage Rate Enrichment, use enrich_real_estate_data function from mortgage.py.

What the function does:
(1) fetches the MORTGAGE30US series directly from FRED, (url/path should be passed as an parameter)
(2) resamples it to monthly averages, 
(3) merges it onto both the combined sold and listings datasets using a year_month key, (url/path passed as parameters)
(4) includes a validation check confirming no null rate values exist after the merge, 
(5) saves both enriched datasets as new CSVs.
