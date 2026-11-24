# Electricity demand for capacity planning

The five annual planning tables combine 12 forecast-based residual shapes with common electricity profiles for road charging, space heating and household hot water, and space cooling. Every model supplies the same annual electricity in a given planning year. The hourly shapes use the 2023 weather and calendar so that each model can be compared under the same conditions. [The consolidated planning case](#consolidated-planning-case) uses one thermal profile family and the seasonal TYNDP passenger profiles.

For the fixed charging experiment, start with the [H2RES demand package](h2res-demand.md) and its [published hourly inputs](../results/planning/h2res/).

## Annual electricity

All values in the table are TWh of electricity. The reference deduction is applied before any scaling. The planning total equals final electricity delivered to consumers plus network losses, using the electricity column of the [Croatian NECP energy balances](https://commission.europa.eu/document/download/3dabd910-2681-4b95-9657-e17309c795fd_en?filename=Croatia+-+Final+Updated+NECP+(Energy+balances)+2021+-+2030.pdf). This defines the accounting boundary for the scenario. It excludes separately reported transformation and energy-sector uses. NECP values are interpolated linearly between 2030, 2040 and 2050. TYNDP end-use values for 2045 are interpolated between 2040 and 2050.

| Planning year | Residual base | Road charging | Heating and hot water | Cooling | Total |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2030 | 14.939 | 0.611 | 3.835 | 1.154 | 20.539 |
| 2035 | 15.223 | 1.717 | 4.068 | 1.154 | 22.161 |
| 2040 | 15.671 | 2.822 | 4.136 | 1.154 | 23.783 |
| 2045 | 16.081 | 3.836 | 3.984 | 1.149 | 25.050 |
| 2050 | 16.491 | 4.850 | 3.833 | 1.143 | 26.317 |

- **2023 deduction.** Eurostat provides 0.0185 TWh of road electricity and the household space heating, hot water and cooling entries. JRC-IDEES-2023 supplies the modelled service-sector electricity for space heating and space cooling. The resulting deductions are 0.0185 TWh for road, 3.3934 TWh for heating and hot water, and 0.7091 TWh for cooling. Service hot water and circulation auxiliaries stay in the residual. Applying the full deduction assumes the corresponding electricity falls within the ENTSO-E load boundary.
- **Road.** The NECP provides annual electricity for road transport. TYNDP National Trends separates passenger cars from vans, trucks and buses. Its vehicle shares divide the Croatian NECP road total. The 2023 deduction uses the passenger shape for all road electricity.
- **Heating and cooling.** The [TYNDP 2026 ETM dashboard](https://2026-data.entsos-tyndp-scenarios.eu/main-documents/TYNDP2026_ETM_DASHBOARD.xlsx) supplies household and service-sector electricity for space heating and cooling. Household hot water is held at its 2023 Eurostat value of 1.23639 TWh. This continuity assumption avoids importing the sharp step down to 0.298 TWh in the TYNDP 2030 household hot-water row and is a scenario choice.

## Hourly profiles

| Component | Source and allocation |
| --- | --- |
| Passenger charging | Add the fixed market and prosumer files for TYNDP 2030, `HR00`, `WS003`. Average the recurring weekday template at source labels 8–12 January and the weekend template at 13–14 January. The 2030 label specifies the demand scenario. Map both patterns to Europe/Zagreb and apply the weekend pattern on public holidays. The source files provide 98.027 GWh of fixed charging. The model uses their shape to allocate the full passenger electricity budget as fixed load. Hours 1–24 are interpreted as consecutive local hours beginning at midnight. |
| Commercial charging | Use the NLR warehouse fleet's 100 vehicle-days. Allocate each day's mileage evenly across its recorded off-shift seconds, aggregate by hour and repeat the profile. This is a depot-charging proxy for the combined van, truck and bus energy budget. |
| Space heating | Compute hourly degrees below 18 °C at Zagreb, Osijek, Rijeka and Split from the benchmark's ERA5 temperatures. Add the four degree-hour profiles using county population shares. |
| Household hot water | Use the two populated normalized Croatian When2Heat profiles for single- and multi-family dwellings. Average each by local hour for 2022, normalize it and combine the two at 77% and 23%, using Eurostat's rounded population share living in a house. The archive omits the second 02:00 at the autumn clock change. The available 8,759 source hours determine the 24-hour means. |
| Space cooling | Compute degrees above 22 °C after taking the trailing six-hour ERA5 mean for each region. Add the four regional profiles using the same county weights. This models the timing of cooling as a temperature response. |

The Croatia county groups use the [2023 DZS population table](https://podaci.dzs.hr/2024/en/76805). Rijeka represents Primorje-Gorski kotar, Istria and Lika-Senj. Split represents Zadar, Šibenik-Knin, Split-Dalmatia and Dubrovnik-Neretva. Osijek represents Virovitica-Podravina, Požega-Slavonia, Slavonski Brod-Posavina, Osijek-Baranja and Vukovar-Srijem. Zagreb represents the remaining counties and the City of Zagreb. The derived weights are approximately 49.33%, 16.83%, 13.13% and 20.71% in Zagreb, Osijek, Rijeka and Split order. Population weights approximate where these loads occur.

For a one-hour interval, MW numerically equals the MWh delivered in that hour. Each nonnegative shape is divided by its annual sum and multiplied by its MWh target. Space heat and hot water are scaled separately before combining. So are passenger and commercial charging. The shared 2023 sector estimates are subtracted from each archived forecast hour by hour. The generator rejects a negative residual. It then scales each model's residual to the common annual base and adds the three future components.

## Degree-hour reference case

Using Python 3.12 with the [training dependencies](../requirements-training.txt):

```sh
python src/planning.py --output work/planning
```

The [degree-hour reference tables](../results/planning/) have 8,760 UTC rows. `reference-2023.csv.gz` contains the three deductions and the 12 residuals. Each planning-year file contains the three shared sector columns followed by `<model>__baseline_mw` and `<model>__total_mw` for each model. Use the total column for the selected model. Keep `reference_timestamp_utc` aligned with the corresponding weather and renewable availability series. Planning years label the annual scenario parameters. The timestamps identify the shared reference chronology.

The [degree-hour reference manifest](../results/planning/manifest.json) records annual MWh, scale factors and file hashes. The [source register](../data/planning/sources.json) identifies the source files, workbook cells and revisions used in the [scenario configuration](../configs/planning.json). The compact [local-hour templates](../data/planning/local-hour-weights.csv) are included so the main generator runs offline. To rebuild them from the original downloads, install `openpyxl==3.1.5` and run:

```sh
python src/planning_sources.py --tyndp-demand Demand.zip --nlr-schedules veh_schedules.csv --when2heat when2heat.csv --output work/local-hour-weights.csv
```

For the figure, install [the plotting dependency](../requirements-figures.txt) and run `python src/planning_figure.py`.

## Profile sensitivities

Each command writes a separate case to its output directory. The original command reproduces the published reference tables.

```sh
python src/planning.py --thermal-profile hotmaps-cooling --output work/hotmaps-cooling
python src/planning.py --thermal-profile hotmaps-thermal --rebase-thermal --output work/hotmaps-thermal
python src/planning.py --thermal-profile hotmaps-all --passenger-shape tyndp-seasonal --output work/paper-fixed
python src/planning.py --hot-water tyndp --output work/tyndp-hot-water
python src/planning.py --passenger-shape tyndp-seasonal --ev-case tyndp-flex --output work/tyndp-ev
```

**Heating and cooling.** Hotmaps defines one weight per local hour, day type and rounded local-day mean temperature. Its Croatian data distinguish coastal HR03 and continental HR04 and separately cover household and service buildings. We apply the lookups to the same four ERA5 cities, regional weights and 2023 UTC clock as the reference case. Sundays and holidays form one day type. Saturdays and the day before a holiday form another. Rounding to the 18 °C shoulder bin assigns zero weight to heating and cooling in this sensitivity. Lookups outside the published active range fail. The [Hotmaps source register](../data/planning/hotmaps-sources.json) records attribution, checksums and the four input tables.

`hotmaps-cooling` changes cooling alone. `hotmaps-thermal` also changes space heating. Each household and service profile is scaled separately to its annual electricity budget before adding the sectors. The hot-water shape still comes from When2Heat. `--rebase-thermal` applies the same thermal method to the 2023 deduction. With the 2050 budgets and 2023 weather, Hotmaps gives a cooling peak of 1,236 MW versus 1,922 MW from degree hours. Combined heating and hot-water peaks are 1,714 and 1,553 MW respectively with the continuity hot-water budget. These comparisons test two models of load timing. Validation against independently observed end-use load would require measurements. Technology-specific efficiency and national heat-pump shares are needed for a bottom-up electric heating case. [Hotmaps methodology](https://wiki.hotmaps.eu/en/Hotmaps-open-data-repositories) · [PyPSA-Eur heating representation](https://pypsa-eur.readthedocs.io/en/latest/design/heat/)

As a separate weather check, the same 2050 electricity budgets on the full 8,784-hour 2024 weather record yield cooling peaks of 1,060 MW for Hotmaps and 1,532 MW for degree hours. Recalculating the 2023 deduction with Hotmaps heating and cooling leaves only 46 MW of residual load for Chronos-2 in one hour, versus a minimum of 243 MW in the reference case. That hour merits review before adopting the alternative historical subtraction.

**Annual hot water.** `--hot-water tyndp` uses the Croatian TYNDP household electricity series for this end use, including interpolation in 2045. The common annual total stays fixed. Electricity released from the lower hot-water allocation goes to the residual base. The 2023 deduction keeps its Eurostat budget.

**Passenger charging shape.** `--passenger-shape tyndp-seasonal` remaps all six distinct 24-hour patterns in the Croatian TYNDP fixed load archive. The profiles distinguish weekdays, weekends and holidays in May–October and November–April. The source day labels act as template locators. The model maps them to the Croatian reference calendar and applies them only to future charging. All four TYNDP planning-year files inspected have the same normalized shape. In the 2050 fully fixed case, national road charging peaks at 1,567 MW with the seasonal templates and 1,062 MW with the reference templates. The annual road budget is 4.850 TWh in both. [TYNDP 2026 demand inputs](https://2026.entsos-tyndp-scenarios.eu/download/)

**Flexible passenger charging.** `--ev-case tyndp-flex` assigns the Croatian TYNDP survey share to the flexible fleet. This is 50% at every published planning anchor and 50% after interpolation in 2045. The 2050 flexible grid budget is 2.017 TWh. The files in this case use `<model>__fixed_demand_mw`. Their annual energy plus `flexible_passenger_grid_mwh` in the manifest closes the full annual budget. Read the [Croatian EV parameters](../data/planning/tyndp-hr-ev.json) for driving, availability, minimum state of charge, fleet and chargers. A H2RES adapter needs to set grid charging efficiency, reconcile fleet and capacity with the NECP energy budget, enforce battery state continuity, and insert the flexible energy exactly once in the electricity balance. V2G requires an explicit operating case. [TYNDP EV methodology](https://2026.entsos-tyndp-scenarios.eu/scenarios-methodology-report/annex-i-ev-properties/)

**Commercial road transport.** The reference warehouse schedule remains available for the combined fixed commercial segment. ZET's schedules include trip blocks, stop times and route shapes for the Zagreb network. They can inform bus operating duties. Power-sector use requires assumptions on vehicle assignment, electrification, depot or layover charging and charger constraints. VESUVIO medium-case data for Croatian trucks yield an energy-weighted en-route share of about 11.5% in 2050. The public file supplies site-level energy and peak statistics. An hourly depot and corridor schedule remains a separate input. [ZET GTFS](https://www.zet.hr/odredbe/datoteke-u-gtfs-formatu/669) · [VESUVIO data](https://doi.org/10.5281/zenodo.22918386)

**Climate sensitivity.** For a climate case, supply and thermal demand should use matched weather. Croatian SECURES-Energy input files contain complete 2050 temperatures. The four thermal demand columns share gaps at the 18 °C bin, covering 425 hours in RCP 4.5 and 370 hours in RCP 8.5. Use the weather columns with an explicit demand method or validate and document a gap treatment before using the demand columns. [SECURES-Energy](https://doi.org/10.5281/zenodo.14615500)

To reproduce the compact source files, use the file names and URLs in the source registers. Use `python src/planning_hotmaps.py --source-dir DIR --output work/hotmaps-hr.csv.gz` for Hotmaps and `python src/planning_ev.py Electric-Vehicle.zip --output work/tyndp-hr-ev.json` for the Croatian EV parameters. The `planning_sources.py` command above accepts `--seasonal-passenger-output work/tyndp-hr-passenger-shapes.json` to regenerate the six fixed templates.

## Consolidated planning case

`hotmaps-all` uses Hotmaps for space heating, space cooling and household hot water. The hot-water table selects a shape by local hour, type of day and season, with the Hotmaps seasonal dates. At the same 1.236 TWh annual electricity use, the Hotmaps and When2Heat household hot-water profiles peak at 445 MW and 279 MW. For an isolated test of hot-water timing, compare `hotmaps-all` with `hotmaps-thermal` using otherwise identical settings. [Hotmaps report, section 2.7](https://www.hotmaps-project.eu/wp-content/uploads/2018/03/D2.3-Hotmaps_for-upload_revised-final_.pdf) · [Hot-water input DOI](https://doi.org/10.5281/zenodo.4687273)

The consolidated fixed case combines `hotmaps-all` with `tyndp-seasonal` and uses the published 2023 deduction. This holds the residual shapes fixed in the first comparison. The alternative with `--rebase-thermal` rebuilds the historical thermal deduction as well. For 2050, the fixed case retains 26.317 TWh and yields total peaks of 4.548–4.963 GW across the 12 models on the 2023 weather clock. Adding `--ev-case tyndp-flex` removes 2.017 TWh from the fixed tables and reserves it for the flexible charging module. Its fixed demand peaks are not total system peaks.

The [H2RES demand package](h2res-demand.md) provides the five consolidated tables, their manifest and plots from the same files.

TYNDP supplies modelled passenger charging shapes. The NLR commercial proxy uses 100 operating days from an American warehouse-delivery fleet. Its hourly shares reproduce the expected load shape of the published minimum-power charging strategy, computed across the complete sample instead of resampling fleet days. Hotmaps supplies useful heating and cooling shapes based on empirical schedules and regional temperature. Applying those shapes to electricity budgets assumes a proportional relation between useful demand and electricity use within each component. The data support a reproducible scenario with cited assumptions. Direct validation of commercial charging and temperature-dependent conversion efficiency requires additional measurements or technology data. [Borlaug et al. (2021)](https://doi.org/10.1038/s41560-021-00855-0) · [NLR input schedules](https://github.com/NatLabRockies/hdev-depot-charging-2021/tree/61530df51f6cd7d64f014ebbf8714cef24c8ca70/data/fleet-schedules/fleet2-warehouse-delivery) · [Hotmaps report](https://www.hotmaps-project.eu/wp-content/uploads/2018/03/D2.3-Hotmaps_for-upload_revised-final_.pdf)

## Interpretation and use in H2RES

The scenarios compare investment sensitivity to the hourly residual shape under equal annual electricity. They assume fixed charging and preserve the 2023 weather chronology. The commercial charging proxy is based on a US fleet. The [VESUVIO release](https://zenodo.org/records/22918386) covers Croatian trucks and reports energy and peak statistics for depot and en-route charging. The Croatian [H2RES transport study](https://www.mdpi.com/1996-1073/13/24/6516) provides vehicle availability and driving profiles for a flexible charging case. The [Croatian cooling study](https://doi.org/10.1016/j.enconman.2022.116374) motivates testing separate coastal and continental shapes. The available inputs support sensitivity cases for a one-hour charging shift, a different commercial charging schedule, the TYNDP road and hot-water trajectories, and the change in estimated service cooling from 0.174 TWh in 2023 to 0.652 TWh in 2030. These alternatives can change peak timing or total demand.

When H2RES uses the total profile, its electricity balance must account for the three end uses and network losses once. An H2RES run that optimizes charging or conversion for one of these same end uses should instead receive a demand series that omits the corresponding fixed component. H2RES configuration and investment results are outside this data export. The same model and third-party terms documented in [data](../data/README.md) apply to the derived series. TimesFM-3 series are included for academic, non-commercial research use.

Cooling is particularly sensitive to the assumed response. Keeping annual 2050 cooling at 1.143 TWh, thresholds of 20, 22 and 24 °C produce cooling-only peaks of 1.38, 1.92 and 2.88 GW on the same weather year. The 22 °C threshold is consistent with one [Croatian building case study](https://thermalscience.rs/pdfs/papers-2020/TSCI200324279F.pdf). The threshold and six-hour averaging window are scenario parameters. The peak sensitivity should be carried into any interpretation of storage and firm capacity decisions.
