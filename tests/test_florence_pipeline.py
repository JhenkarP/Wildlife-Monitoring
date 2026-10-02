import unittest
from unittest.mock import patch

from PIL import Image
from PIL import ImageColor

from annotation.propose_feature_labels import (
    _feature_proposal,
    _feature_query,
    _feature_queries,
    _normalized_box,
    _polygon_boxes,
)
from src.feature_renderer import render_feature_boxes
from src.feature_schema import feature_queries


class FlorencePipelineTests(unittest.TestCase):
    def test_feature_query_comes_from_schema_name(self):
        self.assertEqual(
            _feature_query("greater one horned rhino", "armor_like_skin_folds"),
            "rhino armor like skin folds",
        )
        self.assertEqual(
            _feature_query("bengal tiger", "tail"),
            "tiger tail",
        )
        self.assertEqual(
            _feature_query("bengal tiger", "head"),
            "tiger head",
        )
        self.assertEqual(
            _feature_query("bengal tiger", "abdomen"),
            "tiger abdomen",
        )
        self.assertEqual(
            _feature_query("bengal tiger", "legs"),
            "front legs of tiger",
        )
        self.assertEqual(
            _feature_queries("bengal tiger", "legs"),
            ["front legs of tiger", "hind legs of tiger"],
        )
        self.assertEqual(
            feature_queries("bengal tiger", "legs"),
            ("front legs of tiger", "hind legs of tiger"),
        )

    def test_normalized_box_converts_and_clamps_coordinates(self):
        image = Image.new("RGB", (100, 200))
        self.assertEqual(
            _normalized_box([-10, 20, 60, 240], image),
            [0.0, 0.1, 0.6, 0.9],
        )

    def test_polygon_fallback_returns_bounding_box(self):
        result = _polygon_boxes({"polygons": [[[10, 20, 30, 20, 30, 50, 10, 50]]]})
        self.assertEqual(result, [[10, 20, 30, 50]])

    @patch("annotation.propose_feature_labels.detect")
    def test_feature_proposal_preserves_all_candidates(self, detect):
        detect.return_value = {
            "<OPEN_VOCABULARY_DETECTION>": {
                "bboxes": [[0, 0, 10, 10], [10, 10, 50, 50]],
                "polygons": [],
            }
        }
        proposal, _ = _feature_proposal(
            None,
            None,
            "cpu",
            None,
            Image.new("RGB", (100, 100)),
            "greater one horned rhino",
            "single_horn",
            "A single horn.",
        )
        self.assertEqual(len(proposal["candidate_bboxes"]), 2)
        self.assertEqual(proposal["selected_bbox_index"], 1)
        self.assertEqual(proposal["bbox"], proposal["candidate_bboxes"][1])
        self.assertEqual(proposal["status"], "visible")

    def test_detected_boxes_render_green_without_confidence_label(self):
        image = Image.new("RGB", (100, 100), "white")
        marked = render_feature_boxes(
            image,
            {
                "features": {
                    "single_horn": {
                        "status": "visible",
                        "bbox": [0.1, 0.1, 0.5, 0.5],
                    }
                }
            },
        )
        green = ImageColor.getrgb("#16803c")
        pixels = (
            marked.getpixel((x, y))
            for x in range(marked.width)
            for y in range(marked.height)
        )
        self.assertTrue(any(pixel == green for pixel in pixels))

    @patch("annotation.propose_feature_labels.detect")
    def test_rejects_whole_animal_candidate(self, detect):
        detect.return_value = {
            "<OPEN_VOCABULARY_DETECTION>": {
                "bboxes": [[0, 0, 100, 80]],
                "polygons": [],
            }
        }
        proposal, _ = _feature_proposal(
            None,
            None,
            "cpu",
            None,
            Image.new("RGB", (100, 100)),
            "bengal tiger",
            "stripes",
            "Dark vertical stripes on the tiger's orange body.",
        )
        self.assertEqual(proposal["status"], "uncertain")
        self.assertIsNotNone(proposal["bbox"])
        self.assertEqual(proposal["rejected_candidate_bboxes"], [[0.0, 0.0, 1.0, 0.8]])


if __name__ == "__main__":
    unittest.main()
