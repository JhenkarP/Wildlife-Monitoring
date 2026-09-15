from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.contract import CANONICAL_COLUMNS, load_csv, normalize_observations
from src.database import append_observations, read_observations, replace_observations
from src.inat import fetch_recent_observations, observations_to_frame
from src.regions import REGIONS

ROOT = Path(__file__).parent
CSV_PATH = ROOT / "data" / "raw" / "observations.csv"
DB_PATH = ROOT / "data" / "processed" / "wildlife.sqlite"

st.set_page_config(page_title="Wildlife Monitoring", page_icon="W", layout="wide")
st.title("Wildlife Monitoring and Conservation")
st.caption("Historical observation analysis, migration signals, and conservation decision support")

with st.sidebar:
    st.header("Data sources")
    uploaded_csv = st.file_uploader("Historical migration CSV", type=["csv"])
    if uploaded_csv is not None and st.button("Import uploaded CSV"):
        imported = normalize_observations(pd.read_csv(uploaded_csv))
        count = replace_observations(DB_PATH, imported)
        st.success(f"Imported {count:,} observations")
        st.rerun()
    if CSV_PATH.exists() and st.button("Import historical CSV", type="primary"):
        imported = load_csv(str(CSV_PATH))
        count = replace_observations(DB_PATH, imported)
        st.success(f"Imported {count:,} observations")
    elif not CSV_PATH.exists():
        st.info("Historical CSV not found. Use Live Explorer or add data/raw/observations.csv.")

    selected_region = st.selectbox("iNaturalist region", list(REGIONS))
    st.caption(REGIONS[selected_region]["description"])
    live_days = st.slider("Recent days", 1, 30, 7)
    if st.button("Fetch recent observations"):
        with st.spinner("Fetching iNaturalist observations..."):
            results = fetch_recent_observations(REGIONS[selected_region]["bounds"], days=live_days)
            live_frame = observations_to_frame(results)
            if not live_frame.empty:
                new_count = append_observations(DB_PATH, live_frame)
            else:
                new_count = 0
            st.session_state["live_count"] = new_count
            st.rerun()

observations = read_observations(DB_PATH)
tab_dashboard, tab_live, tab_detection = st.tabs(["Dashboard", "Live Explorer", "Image Detection"])

with tab_dashboard:
    if observations.empty:
        st.info("No observation dataset is loaded yet. Use Live Explorer or import the historical CSV.")
        st.code(",".join(CANONICAL_COLUMNS), language="text")
    else:
        observations["observed_on"] = pd.to_datetime(observations["observed_on"], errors="coerce")
        species_count = observations["species"].nunique(dropna=True)
        location_count = observations["protected_area"].nunique(dropna=True)
        date_count = observations["observed_on"].notna().sum()
        metric_a, metric_b, metric_c, metric_d = st.columns(4)
        metric_a.metric("Observations", f"{len(observations):,}")
        metric_b.metric("Species", f"{species_count:,}")
        metric_c.metric("Protected areas", f"{location_count:,}")
        metric_d.metric("Dated records", f"{date_count:,}")
        states = sorted(observations["state"].dropna().unique())
        selected_states = st.multiselect("State / region", states)
        species = sorted(observations["species"].dropna().unique())
        selected_species = st.multiselect("Species", species)
        filtered = observations.copy()
        if selected_states:
            filtered = filtered[filtered["state"].isin(selected_states)]
        if selected_species:
            filtered = filtered[filtered["species"].isin(selected_species)]
        left, right = st.columns(2)
        with left:
            counts = filtered["species"].value_counts().head(15).rename_axis("species").reset_index(name="observations")
            st.subheader("Most observed species")
            st.plotly_chart(px.bar(counts, x="observations", y="species", orientation="h"), use_container_width=True)
        with right:
            trend = filtered.dropna(subset=["observed_on"]).assign(month=lambda frame: frame["observed_on"].dt.to_period("M").astype(str))
            trend = trend.groupby("month", as_index=False).size().rename(columns={"size": "observations"})
            st.subheader("Observation trend")
            st.plotly_chart(px.line(trend, x="month", y="observations", markers=True), use_container_width=True)
        map_data = filtered.dropna(subset=["latitude", "longitude"])
        st.subheader("Observation map")
        if map_data.empty:
            st.info("No valid coordinates are available for mapping.")
        else:
            st.map(map_data[["latitude", "longitude"]], use_container_width=True)

        st.subheader("Migration and conservation analysis")
        analysis_columns = ["migration_role", "season", "seasonal_density", "iucn_status"]
        available_analysis = [column for column in analysis_columns if filtered[column].notna().any()]
        if not available_analysis:
            st.info("Migration role, season, seasonal density, and IUCN status will appear after the historical migration dataset is imported.")
        else:
            analysis_left, analysis_right = st.columns(2)
            with analysis_left:
                if filtered["migration_role"].notna().any():
                    role_counts = filtered["migration_role"].value_counts().rename_axis("migration_role").reset_index(name="observations")
                    st.plotly_chart(px.bar(role_counts, x="migration_role", y="observations"), use_container_width=True)
                if filtered["season"].notna().any():
                    season_counts = filtered["season"].value_counts().rename_axis("season").reset_index(name="observations")
                    st.plotly_chart(px.bar(season_counts, x="season", y="observations"), use_container_width=True)
            with analysis_right:
                if filtered["iucn_status"].notna().any():
                    iucn_counts = filtered["iucn_status"].value_counts().rename_axis("iucn_status").reset_index(name="observations")
                    st.plotly_chart(px.bar(iucn_counts, x="iucn_status", y="observations"), use_container_width=True)
                if filtered["seasonal_density"].notna().any():
                    st.plotly_chart(px.histogram(filtered, x="seasonal_density"), use_container_width=True)

        st.subheader("Observation records")
        display_columns = [column for column in CANONICAL_COLUMNS if column in filtered.columns]
        report_data = filtered[display_columns]
        st.download_button("Download filtered CSV report", report_data.to_csv(index=False), "wildlife_observations.csv", "text/csv")
        st.caption(f"Showing all {len(display_columns)} observation fields. Scroll horizontally to inspect the full record.")
        st.dataframe(report_data, use_container_width=True, hide_index=True)

with tab_live:
    st.subheader("Recent iNaturalist observations")
    st.write("Use the sidebar to fetch real photo observations for a selected Indian protected-area region.")
    if st.session_state.get("live_count") is not None:
        st.success(f"Loaded {st.session_state['live_count']:,} recent observations into SQLite.")
    st.warning("iNaturalist records are community observations, not camera-trap detections. Treat them as live supplemental data.")

with tab_detection:
    st.subheader("Wildlife image detection")
    st.write("Upload an image to classify it as one of the four trained Indian mammals.")
    image_file = st.file_uploader("Upload a wildlife image", type=["jpg", "jpeg", "png"])
    if image_file:
        from PIL import Image
        from src.detector import classify_with_custom_model, classify_with_speciesnet

        image = Image.open(image_file).convert("RGB")
        detection_model = st.selectbox(
            "Detection model",
            ["Fine-tuned ResNet18 (four animals)", "SpeciesNet (comparison)"],
        )
        if st.button("Run detection", type="primary"):
            try:
                spinner_text = "Running fine-tuned ResNet18..." if detection_model.startswith("Fine-tuned") else "Running SpeciesNet..."
                with st.spinner(spinner_text):
                    if detection_model.startswith("Fine-tuned"):
                        annotated, detections = classify_with_custom_model(image)
                    else:
                        annotated, detections = classify_with_speciesnet(image)
            except Exception as error:
                st.error(f"{detection_model} could not run: {error}")
            else:
                st.image(annotated, caption="Detected objects", use_container_width=True)
                if detections:
                    st.dataframe(pd.DataFrame(detections), use_container_width=True, hide_index=True)
                    st.json(detections)
                else:
                    st.info("No supported animal object was detected above the confidence threshold.")