from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.classifier import CatClassifier
from src.taxonomy import grouped_species, load_taxonomy


def complete_taxonomy_graph(taxonomy: dict, subfamilies: set[str] | None = None) -> str:
    lines = [
        'digraph {',
        '  graph [rankdir=LR, bgcolor="transparent", pad="0.2", nodesep="0.12", ranksep="0.7"];',
        '  node [shape=box, style="rounded,filled", fontname="Arial", color="#173b3f", fillcolor="#e5f1ed", margin="0.12,0.08"];',
        '  edge [color="#527174"];',
    ]
    for subfamily, lineages in grouped_species(taxonomy).items():
        if subfamilies is not None and subfamily not in subfamilies:
            continue
        subfamily_id = subfamily.lower()
        lines.append(f'  {subfamily_id} [label="{subfamily}", fillcolor="#f8e6c4"];')
        for lineage, species in lineages.items():
            lineage_id = lineage.lower().replace(" ", "_")
            lines.append(f'  {lineage_id} [label="{lineage}"];')
            lines.append(f'  {subfamily_id} -> {lineage_id};')
            genera = sorted({item["genus"] for item in species})
            for genus in genera:
                genus_id = f"{lineage_id}_{genus.lower()}"
                lines.append(f'  {genus_id} [label="Genus {genus}"];')
                lines.append(f'  {lineage_id} -> {genus_id};')
                for item in [item for item in species if item["genus"] == genus]:
                    species_id = item["id"]
                    label = f'{item["common_name"]}\\n{item["scientific_name"]}'
                    lines.append(f'  {species_id} [label="{label}", fillcolor="#d8e8f4"];')
                    lines.append(f'  {genus_id} -> {species_id};')
    lines.append('}')
    return "\\n".join(lines)


st.set_page_config(page_title="Felidae Classifier", page_icon="🐈", layout="wide")
taxonomy = load_taxonomy()
groups = grouped_species(taxonomy)
availability_path = PROJECT_ROOT / "taxonomy" / "inat_research_observations.json"
with availability_path.open(encoding="utf-8") as file:
    online_availability = json.load(file)["counts"]

st.markdown("# Felidae classifier")
st.caption("A taxonomy-first classification workspace for the 41 extant Felidae species.")

metric_columns = st.columns(4)
metric_columns[0].metric("Family", taxonomy["family"])
metric_columns[1].metric("Species labels", taxonomy["species_count"])
metric_columns[2].metric("Wild species", taxonomy["wild_species_count"])
metric_columns[3].metric("Subfamilies", len(groups))

st.divider()
st.subheader("Classification flow")
st.graphviz_chart("""
digraph {
  graph [rankdir=LR, bgcolor="transparent", pad="0.2"];
  node [shape=box, style="rounded,filled", fontname="Arial", color="#173b3f", fillcolor="#e5f1ed", margin="0.16,0.10"];
  edge [fontname="Arial", color="#527174"];
  image [label="Input image", fillcolor="#f8e6c4"];
  detector [label="Animal detector"];
  cat [label="Cat / non-cat?"];
  reject [label="Reject: not a cat", fillcolor="#f9dddd"];
  felidae [label="Family: Felidae"];
  subfamily [label="Subfamily\\nPantherinae or Felinae"];
  lineage [label="Lineage"];
  genus [label="Genus"];
  species [label="Species\\ncommon + scientific name", fillcolor="#d8e8f4"];
  unknown [label="Unknown / insufficient evidence", fillcolor="#f9dddd"];
  image -> detector -> cat;
  cat -> reject [label="no"];
  cat -> felidae [label="yes"];
  felidae -> subfamily -> lineage -> genus -> species;
  species -> unknown [label="low confidence"];
}
""")

st.subheader("Complete species classification")
st.caption("Every accepted label is expanded below with its common and scientific name.")
pantherinae, felinae = st.tabs(["Pantherinae · 7 species", "Felinae · 34 species"])
with pantherinae:
    st.graphviz_chart(complete_taxonomy_graph(taxonomy, {"Pantherinae"}), use_container_width=True)
with felinae:
    st.graphviz_chart(complete_taxonomy_graph(taxonomy, {"Felinae"}), use_container_width=True)

st.subheader("Taxonomy explorer")
st.caption("Online availability is a dated iNaturalist research-grade observation snapshot, not a guaranteed image-download count.")
selected_subfamily = st.selectbox("Subfamily", ["All"] + list(groups))
selected_lineage = st.selectbox(
    "Lineage",
    ["All"] + sorted({species["lineage"] for species in taxonomy["species"]
                       if selected_subfamily == "All" or species["subfamily"] == selected_subfamily}),
)

species = [item for item in taxonomy["species"]
           if (selected_subfamily == "All" or item["subfamily"] == selected_subfamily)
           and (selected_lineage == "All" or item["lineage"] == selected_lineage)]
st.dataframe(
    [{
        "Common name": item["common_name"],
        "Scientific name": item["scientific_name"],
        "Genus": item["genus"],
        "Subfamily": item["subfamily"],
        "Lineage": item["lineage"],
        "Wild species": "Yes" if item["wild_species"] else "No",
        "iNaturalist observations": online_availability.get(item["id"], 0),
    } for item in species],
    use_container_width=True,
    hide_index=True,
)

st.divider()
st.subheader("Image classification")
uploaded = st.file_uploader("Upload a cat image", type=["jpg", "jpeg", "png", "webp"])
classifier = CatClassifier(PROJECT_ROOT / "models" / "panthera_five_species_full_correction.pt")
if uploaded:
    from PIL import Image

    image = Image.open(uploaded)
    result = classifier.predict_with_localization(image, taxonomy)
    if result.status == "model_not_trained":
        st.info("The trained classifier checkpoint is not available yet.")
    else:
        preview_columns = st.columns(2)
        with preview_columns[0]:
            st.image(image, caption="Uploaded image", use_container_width=True)
        with preview_columns[1]:
            st.image(result.localized_image, caption="Grad-CAM localization", use_container_width=True)

        st.write({"label": result.label, "scientific_name": result.scientific_name, "confidence": result.confidence})
        if result.bounding_box is None:
            st.warning("The classifier did not find a strong localized region. The box is only an approximation, not a trained detector result.")
        else:
            st.write({
                "bounding_box_pixels": result.bounding_box,
                "animal_region_width_pixels": result.pixel_width,
                "animal_region_height_pixels": result.pixel_height,
            })
            reference_columns = st.columns(2)
            with reference_columns[0]:
                reference_length_cm = st.number_input("Known reference length (cm)", min_value=0.0, value=0.0, step=1.0)
            with reference_columns[1]:
                reference_length_pixels = st.number_input("Reference length (pixels)", min_value=0.0, value=0.0, step=1.0)
            if reference_length_cm > 0 and reference_length_pixels > 0:
                centimeters_per_pixel = reference_length_cm / reference_length_pixels
                st.write({
                    "scale_cm_per_pixel": centimeters_per_pixel,
                    "estimated_region_width_cm": result.pixel_width * centimeters_per_pixel,
                    "estimated_region_height_cm": result.pixel_height * centimeters_per_pixel,
                })
            else:
                st.caption("Add a visible reference object with known size to convert pixels into centimeters.")
            st.info("Weight estimation is not enabled: it requires images paired with measured animal weights for regression training.")

st.caption("Taxonomy source: IUCN/SSC Cat Specialist Group Living Species, cross-checked against the Wikipedia List of felids. Taxonomy can change; review the source before training a new model.")
