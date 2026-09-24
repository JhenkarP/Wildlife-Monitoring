"""Versioned visual feature schema for assisted wildlife annotation."""

from __future__ import annotations

from typing import Any


SCHEMA_VERSION = "1.1"
FEATURE_SCHEMA: dict[str, tuple[tuple[str, str], ...]] = {
    "asian elephant": (("trunk", "The trunk is visible."), ("fan_shaped_ears", "The large fan-shaped ears are visible."), ("tusks", "One or more tusks are visible.")),
    "asiatic lion": (("reduced_mane_exposing_ears", "A reduced or open mane leaves the ears visibly exposed."), ("belly_fold", "The characteristic longitudinal belly fold is visible."), ("dark_tail_tuft", "A dark tuft is visible at the end of the tail.")),
    "barasingha": (("multi_tined_antlers", "Broad antlers with multiple tines are visible."), ("pale_underside", "A pale underside or rump is visible."), ("reddish_brown_coat", "The coat is visibly reddish brown.")),
    "bengal tiger": (("facial_stripes", "Dark vertical stripes are visible on the face."), ("vertical_body_stripes", "Dark vertical stripes cross an orange body coat."), ("white_cheek_chest", "White fur is visible on the cheek or chest.")),
    "chital": (("white_body_spots", "Many small white spots are visible across the body."), ("three_tined_antlers", "Antlers with the characteristic branching tines are visible."), ("dark_dorsal_stripe", "A dark stripe is visible along the back.")),
    "dhole": (("reddish_coat", "The coat is distinctly reddish or rust colored."), ("rounded_ears", "Short rounded ears are clearly visible."), ("bushy_dark_tipped_tail", "A bushy tail with a darker tip is visible.")),
    "gaur": (("shoulder_hump", "A pronounced shoulder hump is visible."), ("white_lower_leg_stockings", "The lower legs have distinct white stockings."), ("curved_horns", "Short curved horns are visible.")),
    "greater one horned rhino": (("single_horn", "One horn is visible on the snout."), ("armor_like_skin_folds", "Thick armor-like folds of skin are visible."), ("prehensile_upper_lip", "The pointed prehensile upper lip is visible.")),
    "hanuman langur": (("black_face", "The face and muzzle are distinctly black."), ("grey_silver_coat", "The body coat is grey or silvery."), ("long_tail", "A long tail is visible.")),
    "indian leopard": (("rosette_spots", "Rosette-shaped spot patterns are visible."), ("pale_underside", "A pale underside is visible."), ("long_ringed_tail", "A long tail with visible bands or rings is present.")),
    "nilgai": (("blue_grey_male_coat", "The adult male has a blue-grey coat."), ("white_throat_patch", "A distinct white throat patch is visible."), ("short_straight_horns", "Short mostly straight horns are visible.")),
    "sambar": (("dark_shaggy_coat", "The coat is dark and visibly shaggy."), ("large_ears", "Large ears are clearly visible."), ("antlers", "Large branching antlers are visible.")),
    "sloth bear": (("shaggy_black_coat", "A shaggy black coat is visible."), ("pale_muzzle", "A broad pale muzzle is visible."), ("white_chest_mark", "A pale or white chest mark is visible.")),
    "striped hyena": (("vertical_dark_stripes", "Dark vertical stripes are visible on the body."), ("sloping_back", "The characteristic sloping back is visible."), ("dorsal_mane", "A mane is visible along the neck or back.")),
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
        confidence = value.get("confidence")
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0 <= confidence <= 1:
            errors.append(f"{name}: confidence must be a number from 0 to 1")
        if not isinstance(value.get("evidence"), str) or not value["evidence"].strip():
            errors.append(f"{name}: evidence must be a non-empty string")
    return errors