from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import requests

from src.gbif import fetch_unfiltered_regional_gbif
from src.contract import CANONICAL_COLUMNS, normalize_observations
from src.regions import REGIONS

ROOT = Path(__file__).parents[1]
OUTPUT_PATH = ROOT / "data" / "raw" / "five_regions_gbif_unfiltered.csv"
EXAMPLE_PATH = ROOT / "data" / "examples" / "historical_migration_example.csv"

st.set_page_config(page_title="Historical Datasets", page_icon="H", layout="wide")
st.title("Historical Wildlife Datasets")
st.caption("Collect unfiltered historical GBIF wildlife records from the five-region Indian pilot scope")

st.info("This page is for historical dataset extraction. The main dashboard remains the live iNaturalist backend extraction page.")

st.info("Only geographic bounds are applied. No date, taxon, insect, image, quality, dataset, or species filter is applied.")
st.caption("Regions: " + ", ".join(REGIONS))
records_per_region = st.number_input("Records to download per region", min_value=300, max_value=10000, value=1000, step=300)

if st.button("Fetch all five regions", type="primary"):
    try:
        with st.spinner("Fetching paginated GBIF records for all five regions..."):
            extracted, available = fetch_unfiltered_regional_gbif(REGIONS, int(records_per_region))
    except requests.RequestException as error:
        st.error(f"GBIF is temporarily unavailable: {error}. Try again later.")
    else:
        if extracted.empty:
            st.warning("No GBIF observations were returned for the five regions.")
        else:
            OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            extracted.to_csv(OUTPUT_PATH, index=False)
            st.session_state["historical_frame"] = extracted
            st.session_state["available_counts"] = available
            st.success(f"Downloaded {len(extracted):,} observations and saved {OUTPUT_PATH.name}.")

historical = st.session_state.get("historical_frame")
if historical is not None and not historical.empty:
    historical = normalize_observations(historical)
    st.session_state["historical_frame"] = historical
    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric("Observations", f"{len(historical):,}")
    metric_b.metric("Species", f"{historical['species'].nunique(dropna=True):,}")
    metric_c.metric("States", f"{historical['state'].nunique(dropna=True):,}")
    available_counts = st.session_state.get("available_counts", {})
    if available_counts:
        st.caption("GBIF records available in source: " + "; ".join(f"{region}: {count:,}" for region, count in available_counts.items()))
    st.subheader("Occurrence census by region and type")
    st.caption("This is a census of downloaded GBIF occurrence records, not an estimate of animal population size.")
    census = (
        historical.assign(
            taxonomic_group=historical["taxonomic_group"].fillna("Unclassified"),
            has_date=historical["observed_on"].notna(),
            has_coordinates=historical[["latitude", "longitude"]].notna().all(axis=1),
        )
        .groupby(["monitoring_region", "taxonomic_group"], as_index=False)
        .agg(
            records=("occurrence_id", "size"),
            unique_species=("species", "nunique"),
            dated_records=("has_date", "sum"),
            coordinate_records=("has_coordinates", "sum"),
        )
        .sort_values(["monitoring_region", "records"], ascending=[True, False])
    )
    st.dataframe(census, use_container_width=True, hide_index=True)
    display_columns = [column for column in CANONICAL_COLUMNS if column in historical.columns]
    display_frame = historical[display_columns]
    st.caption(f"Showing all {len(display_columns)} observation fields. Scroll horizontally to inspect the full record.")
    st.dataframe(display_frame, use_container_width=True, hide_index=True)
    st.download_button(
        "Download historical CSV",
        display_frame.to_csv(index=False),
        "historical_wildlife_dataset.csv",
        "text/csv",
    )
else:
    if OUTPUT_PATH.exists():
        try:
            saved_historical = pd.read_csv(OUTPUT_PATH)
        except (OSError, pd.errors.ParserError) as error:
            st.warning(f"The saved historical CSV could not be loaded: {error}")
        else:
            if not saved_historical.empty:
                st.session_state["historical_frame"] = saved_historical
                st.rerun()
    st.write("Fetch the five-region collection. The result is kept separate from the live observation database.")