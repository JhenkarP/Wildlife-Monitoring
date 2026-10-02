"""Versioned visual feature schema for assisted wildlife annotation."""

from __future__ import annotations

from typing import Any


SCHEMA_VERSION = "1.2"
FEATURE_SCHEMA: dict[str, tuple[tuple[str, str], ...]] = {
    "asian elephant": (("trunk", "Visible trunk on the elephant."), ("fan_shaped_ears", "Large fan-shaped ears on the elephant."), ("tusks", "Visible tusks on the elephant.")),
    "asiatic lion": (("eyes", "Visible eyes of the Asiatic lion."), ("ears", "Visible ears of the Asiatic lion."), ("tail", "Visible tail of the Asiatic lion."), ("body_coat", "Short tawny coat covering the lion's torso, shoulders, and flanks; exclude the head, legs, tail, and background.")),
    "barasingha": (("multi_tined_antlers", "Broad antlers with multiple tines on the barasingha."), ("white_throat_patch", "White throat patch on the barasingha."), ("reddish_brown_coat", "Reddish-brown coat on the barasingha.")),
    "bengal tiger": (("head", "Head of the Bengal tiger."), ("abdomen", "Abdomen of the Bengal tiger."), ("legs", "Legs of the Bengal tiger."), ("tail", "Tail of the Bengal tiger.")),
    "chital": (("white_body_spots", "White spots across the chital's body."), ("three_tined_antlers", "Antlers with three branching tines on the chital."), ("dark_dorsal_stripe", "Dark dorsal stripe along the chital's back.")),
    "dhole": (("reddish_coat", "Reddish or rust-colored coat on the dhole."), ("rounded_ears", "Short rounded ears on the dhole."), ("bushy_dark_tipped_tail", "Bushy dark-tipped tail on the dhole.")),
    "gaur": (("shoulder_hump", "Pronounced shoulder hump on the gaur."), ("white_lower_leg_stockings", "White stockings on the gaur's lower legs."), ("curved_horns", "Short curved horns on the gaur.")),
    "greater one horned rhino": (("single_horn", "Small single horn at the front of the rhino's snout."), ("armor_like_skin_folds", "Horizontal skin folds on the rhino's neck and shoulders behind the head."), ("rounded_ears", "Small rounded ear above the rhino's head.")),
    "hanuman langur": (("black_face", "Black face and muzzle on the langur."), ("grey_silver_coat", "Grey-silver coat on the langur."), ("long_tail", "Long tail on the langur.")),
    "indian leopard": (("body_rosette_spots", "Open-centered rosette spots across the leopard's torso or flanks."), ("long_white_whiskers", "Long white whiskers around the leopard's muzzle or cheeks."), ("spotted_paws", "Dark spots on the leopard's lower legs or paws.")),
    "nilgai": (("blue_grey_male_coat", "Blue-grey coat on the adult male nilgai."), ("white_throat_patch", "White throat patch on the nilgai."), ("short_straight_horns", "Short mostly straight horns on the nilgai.")),
    "sambar": (("antlers", "Large branching antlers on the Sambar; include the visible antler branches and exclude the head and background."), ("ears", "Visible ears of the Sambar; exclude the head and background."), ("eyes", "Visible eyes of the Sambar; exclude the head and background."), ("body", "Visible torso and body of the Sambar; exclude the head, legs, background, and surrounding scenery."), ("legs", "Visible legs of the Sambar; exclude the body, head, background, and surrounding scenery.")),
    "sloth bear": (("shaggy_black_coat", "Shaggy black coat on the sloth bear."), ("pale_muzzle", "Broad pale muzzle on the sloth bear."), ("white_chest_mark", "Pale or white chest mark on the sloth bear.")),
    "striped hyena": (("vertical_dark_stripes", "Dark vertical stripes on the hyena's body."), ("sloping_back", "Sloping back on the hyena."), ("dorsal_mane", "Mane along the hyena's neck or back.")),
}

FEATURE_QUERY_OVERRIDES: dict[str, dict[str, tuple[str, ...]]] = {
    "bengal tiger": {
        "legs": ("front legs of tiger", "hind legs of tiger"),
    },
    "asiatic lion": {
        "eyes": ("only lion eyes, not the whole lion",),
        "ears": ("only lion ears, not the whole lion",),
        "tail": ("only lion tail, not the whole lion",),
        "body_coat": ("only the lion's body coat on the torso, shoulders, and flanks; exclude the head, legs, tail, and background",),
    },
    "sambar": {
        "antlers": ("only sambar antlers, not the whole animal",),
        "ears": ("only sambar ears, not the whole animal",),
        "eyes": ("only sambar eyes, not the whole animal",),
        "body": ("only sambar torso and body, not the head, legs, background, or scenery",),
        "legs": ("only sambar legs, not the body, head, background, or scenery",),
    },
}

ALLOWED_STATUSES = {"visible", "not_visible", "uncertain"}


def feature_names(species: str) -> tuple[str, ...]:
    return tuple(name for name, _ in FEATURE_SCHEMA[species])


def feature_queries(species: str, feature_name: str) -> tuple[str, ...]:
    override = FEATURE_QUERY_OVERRIDES.get(species, {}).get(feature_name)
    if override:
        return override
    animal_name = species.rsplit(" ", 1)[-1]
    return (f"{animal_name} {feature_name.replace('_', ' ')}",)


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