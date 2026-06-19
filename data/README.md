# Data

## Source

**Bike Sharing Dataset** — UCI Machine Learning Repository.
Hadi Fanaee-T and João Gama, *Event labeling combining ensemble detectors and
background knowledge*, Progress in Artificial Intelligence (2013).

- Original page: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
- The full attribute description is in [`UCI_original_readme.txt`](UCI_original_readme.txt).

## File used

`hour.csv` — hourly counts of rented bikes for the Capital Bikeshare system
(Washington D.C.), 2011–2012. 17,379 rows, one per hour.

## Target and key columns

| Column | Meaning |
|--------|---------|
| `cnt` | **Target** — total bikes rented that hour (`casual` + `registered`) |
| `dteday`, `yr`, `mnth`, `hr`, `weekday` | Date / time fields |
| `season` | 1: spring, 2: summer, 3: fall, 4: winter |
| `weathersit` | Weather situation (1 clear → 4 severe) |
| `holiday`, `workingday` | Calendar flags |
| `temp`, `atemp`, `hum`, `windspeed` | Weather, normalized to `[0, 1]` |
| `casual`, `registered` | Rentals by user type (**excluded** — they leak the target) |

> Note: `casual + registered = cnt`, so both are dropped from the feature set to
> avoid target leakage.
