"""Versioned visual feature schema for assisted wildlife annotation."""

from __future__ import annotations

from typing import Any


SCHEMA_VERSION = "1.2"
FEATURE_SCHEMA: dict[str, tuple[tuple[str, str], ...]] = {
    "asian elephant": (("trunk", "Visible trunk on the elephant."), ("fan_shaped_ears", "Large fan-shaped ears on the elephant."), ("tusks", "Visible tusks on the elephant.")),
    "asiatic lion": (("sparse_mane_exposed_ears", "Sparse mane with exposed ears on the lion."), ("longitudinal_belly_fold", "Longitudinal belly fold on the lion."), ("dark_tail_tuft", "Dark tuft at the end of the lion's tail.")),
    "barasingha": (("multi_tined_antlers", "Broad antlers with multiple tines on the barasingha."), ("white_throat_patch", "White throat patch on the barasingha."), ("reddish_brown_coat", "Reddish-brown coat on the barasingha.")),
    "bengal tiger": (("orange_white_facial_fur", "Contrasting orange and white facial fur on the tiger."), ("vertical_body_stripes", "Dark vertical stripes on the tiger's orange body."), ("dark_banded_tail", "Dark bands along the tiger's tail.")),
    "chital": (("white_body_spots", "White spots across the chital's body."), ("three_tined_antlers", "Antlers with three branching tines on the chital."), ("dark_dorsal_stripe", "Dark dorsal stripe along the chital's back.")),
    "dhole": (("reddish_coat", "Reddish or rust-colored coat on the dhole."), ("rounded_ears", "Short rounded ears on the dhole."), ("bushy_dark_tipped_tail", "Bushy dark-tipped tail on the dhole.")),
    "gaur": (("shoulder_hump", "Pronounced shoulder hump on the gaur."), ("white_lower_leg_stockings", "White stockings on the gaur's lower legs."), ("curved_horns", "Short curved horns on the gaur.")),
    "greater one horned rhino": (("single_horn", "Small single horn at the front of the rhino's snout."), ("armor_like_skin_folds", "Horizontal skin folds on the rhino's neck and shoulders behind the head."), ("rounded_ears", "Small rounded ear above the rhino's head.")),
    "hanuman langur": (("black_face", "Black face and muzzle on the langur."), ("grey_silver_coat", "Grey-silver coat on the langur."), ("long_tail", "Long tail on the langur.")),
    "indian leopard": (("body_rosette_spots", "Open-centered rosette spots across the leopard's torso or flanks."), ("long_white_whiskers", "Long white whiskers around the leopard's muzzle or cheeks."), ("spotted_paws", "Dark spots on the leopard's lower legs or paws.")),
    "nilgai": (("blue_grey_male_coat", "Blue-grey coat on the adult male nilgai."), ("white_throat_patch", "White throat patch on the nilgai."), ("short_straight_horns", "Short mostly straight horns on the nilgai.")),
    "sambar": (("dark_shaggy_coat", "Dark shaggy coat on the sambar."), ("large_ears", "Large ears on the sambar."), ("antlers", "Large branching antlers on the sambar.")),
    "sloth bear": (("shaggy_black_coat", "Shaggy black coat on the sloth bear."), ("pale_muzzle", "Broad pale muzzle on the sloth bear."), ("white_chest_mark", "Pale or white chest mark on the sloth bear.")),
    "striped hyena": (("vertical_dark_stripes", "Dark vertical stripes on the hyena's body."), ("sloping_back", "Sloping back on the hyena."), ("dorsal_mane", "Mane along the hyena's neck or back.")),
}

ALLOWED_STATUSES = {"visible", "not_visible", "uncertain"}


def feature_names(species: str) -> tuple[str, ...]:
    return tuple(name for name, _ in FEATURE_SCHEMA[species])


def validate_label_record(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    species = record.get("species")
    if species not in FEATURE_SCHEMA:
        return [f"unknown species: {species!r}"]
    features = record.get("features")
    expected = set(feature_names(species))
    if not isinstance(features, dict):
        return ["features must be an object"]
    if set(features) != expected:
        errors.append(f"feature keys must be exactly {sorted(expected)}")
    for name in sorted(expected & set(features)):
        value = features[name]
        if not isinstance(value, dict):
            errors.append(f"{name}: value must be an object")
            continue
        status = value.get("status")
        if status not in ALLOWED_STATUSES:
            errors.append(f"{name}: status must be one of {sorted(ALLOWED_STATUSES)}")
        if not isinstance(value.get("evidence"), str) or not value["evidence"].strip():
            errors.append(f"{name}: evidence must be a non-empty string")
    return errors