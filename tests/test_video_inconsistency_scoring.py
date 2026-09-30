"""Tests for detector-output parsing and the scoring algorithm (contract C6)."""

from __future__ import annotations

import math

import pytest

from validator.modules.video_inconsistency.errors import VideoSubmissionError
from validator.modules.video_inconsistency.manifest import ClipSpec, IssueLabel
from validator.modules.video_inconsistency.predictions import (
    PredictedIssue,
    parse_detector_output,
)
from validator.modules.video_inconsistency.scoring import (
    ScoringSettings,
    score_predictions,
)


FPS = 10.0
FRAMES = 100  # 10 s clips


def label(type_: str, start: float, end: float, bbox: list[float] | None = None) -> IssueLabel:
    if type_ in ("inserted_object", "blurred_region") and bbox is None:
        bbox = [0.1, 0.1, 0.5, 0.5]
    return IssueLabel(
        type=type_,
        start_time=start,
        end_time=end,
        start_frame=int(round(start * FPS)),
        end_frame=int(round(end * FPS)),
        bbox=bbox,
    )


def clip(
    clip_id: str, issues: list[IssueLabel], difficulty: str = "medium"
) -> ClipSpec:
    return ClipSpec(
        clip_id=clip_id,
        video_path=f"videos/{clip_id}.mp4",
        fps=FPS,
        num_frames=FRAMES,
        width=64,
        height=48,
        difficulty=difficulty,
        issues=issues,
    )


def pred(
    type_: str,
    start: float,
    end: float,
    confidence: float = 1.0,
    bbox: list[float] | None = None,
) -> PredictedIssue:
    return PredictedIssue(
        type=type_, start_time=start, end_time=end, confidence=confidence, bbox=bbox
    )


def exact(issue: IssueLabel) -> PredictedIssue:
    return pred(issue.type, issue.start_time, issue.end_time, 1.0, issue.bbox)


# --------------------------------------------------------------------------
# parse_detector_output
# --------------------------------------------------------------------------


def test_parse_accepts_dict_and_bare_list():
    item = {"type": "zoom_jump", "start_time": 1, "end_time": 2.5, "confidence": 0.7}
    from_dict = parse_detector_output({"issues": [item]}, 10.0)
    from_list = parse_detector_output([item], 10.0)
    assert from_dict == from_list
    assert from_dict[0].type == "zoom_jump"
    assert from_dict[0].start_time == 1.0
    assert from_dict[0].confidence == 0.7
    assert from_dict[0].bbox is None


def test_parse_defaults_and_ignored_keys():
    (issue,) = parse_detector_output(
        [
            {
                "type": "inserted_object",
                "start_time": 1.0,
                "end_time": 2.0,
                "bbox": [0.1, 0.2, 0.3, 0.4],
                "description": "a duck",
                "extra": {"anything": 1},
            }
        ],
        10.0,
    )
    assert issue.confidence == 1.0
    assert issue.bbox == [0.1, 0.2, 0.3, 0.4]
    assert not hasattr(issue, "description")


def test_parse_empty_outputs():
    assert parse_detector_output([], 5.0) == []
    assert parse_detector_output({"issues": []}, 5.0) == []


def test_parse_clamps_times_into_clip():
    (issue,) = parse_detector_output(
        [{"type": "frozen_frames", "start_time": -3.0, "end_time": 99.0}], 6.0
    )
    assert (issue.start_time, issue.end_time) == (0.0, 6.0)


def test_parse_point_event_and_null_bbox():
    (issue,) = parse_detector_output(
        [{"type": "dropped_frames", "start_time": 2.0, "end_time": 2.0, "bbox": None}], 6.0
    )
    assert issue.start_time == issue.end_time == 2.0


def test_parse_tiny_negative_span_within_tolerance_is_accepted():
    (issue,) = parse_detector_output(
        [{"type": "dropped_frames", "start_time": 2.0, "end_time": 2.0 - 1e-9}], 6.0
    )
    assert issue.end_time == issue.start_time


@pytest.mark.parametrize(
    "raw, needle",
    [
        ("not a list", "list"),
        (42, "list"),
        (None, "list"),
        ({"issues": "nope"}, "list"),
        ({"other": []}, "issues"),
        ([5], "item 0"),
        ([{"type": "nope", "start_time": 0, "end_time": 1}], "type"),
        ([{"type": 3, "start_time": 0, "end_time": 1}], "type"),
        ([{"start_time": 0, "end_time": 1}], "type"),
        ([{"type": "zoom_jump", "end_time": 1}], "start_time"),
        ([{"type": "zoom_jump", "start_time": 0}], "end_time"),
        ([{"type": "zoom_jump", "start_time": True, "end_time": 1}], "start_time"),
        ([{"type": "zoom_jump", "start_time": 0, "end_time": False}], "end_time"),
        ([{"type": "zoom_jump", "start_time": "0", "end_time": 1}], "start_time"),
        ([{"type": "zoom_jump", "start_time": math.nan, "end_time": 1}], "start_time"),
        ([{"type": "zoom_jump", "start_time": 0, "end_time": math.inf}], "end_time"),
        ([{"type": "zoom_jump", "start_time": 3, "end_time": 2}], "end_time"),
        ([{"type": "zoom_jump", "start_time": 0, "end_time": 1, "confidence": 1.5}], "confidence"),
        ([{"type": "zoom_jump", "start_time": 0, "end_time": 1, "confidence": -0.1}], "confidence"),
        ([{"type": "zoom_jump", "start_time": 0, "end_time": 1, "confidence": math.nan}], "confidence"),
        ([{"type": "zoom_jump", "start_time": 0, "end_time": 1, "confidence": "high"}], "confidence"),
        ([{"type": "zoom_jump", "start_time": 0, "end_time": 1, "bbox": [0, 0, 1]}], "bbox"),
        ([{"type": "zoom_jump", "start_time": 0, "end_time": 1, "bbox": "box"}], "bbox"),
        ([{"type": "zoom_jump", "start_time": 0, "end_time": 1, "bbox": [0, 0, 2, 1]}], "bbox"),
        ([{"type": "zoom_jump", "start_time": 0, "end_time": 1, "bbox": [0.5, 0, 0.2, 1]}], "bbox"),
        ([{"type": "zoom_jump", "start_time": 0, "end_time": 1, "bbox": [0, 0, "a", 1]}], "bbox"),
        ([{"type": "zoom_jump", "start_time": 0, "end_time": 1, "description": 5}], "description"),
        (
            [{"type": "zoom_jump", "start_time": 0, "end_time": 1, "description": "x" * 501}],
            "description",
        ),
    ],
)
def test_parse_rejects_invalid_output(raw, needle):
    with pytest.raises(VideoSubmissionError) as excinfo:
        parse_detector_output(raw, 10.0)
    assert excinfo.value.failure_mode == "detector_output_invalid"
    assert excinfo.value.fatal is False
    assert needle in str(excinfo.value)


def test_parse_error_names_the_offending_index():
    good = {"type": "zoom_jump", "start_time": 0, "end_time": 1}
    bad = {"type": "zoom_jump", "start_time": 0, "end_time": 1, "confidence": 3}
    with pytest.raises(VideoSubmissionError, match=r"item 2 field 'confidence'"):
        parse_detector_output([good, good, bad], 10.0)


def test_parse_accepts_description_at_limit_and_rejects_over_1000_items():
    ok = {"type": "zoom_jump", "start_time": 0, "end_time": 1, "description": "x" * 500}
    assert len(parse_detector_output([ok], 10.0)) == 1
    item = {"type": "zoom_jump", "start_time": 0, "end_time": 1}
    assert len(parse_detector_output([item] * 1000, 10.0)) == 1000
    with pytest.raises(VideoSubmissionError) as excinfo:
        parse_detector_output([item] * 1001, 10.0)
    assert excinfo.value.failure_mode == "detector_output_invalid"


# --------------------------------------------------------------------------
# score_predictions
# --------------------------------------------------------------------------


def test_perfect_predictions_score_one():
    a = label("frozen_frames", 2.0, 4.0)
    b = label("inserted_object", 5.0, 7.0)
    c = label("dropped_frames", 3.0, 3.0)
    clips = [clip("a", [a, b]), clip("b", [c], "hard"), clip("clean", [], "easy")]
    preds = [[exact(a), exact(b)], [exact(c)], []]
    result = score_predictions(clips, preds)
    assert result.score == pytest.approx(1.0)
    assert result.loss == pytest.approx(0.0)
    assert result.macro_f1 == pytest.approx(1.0)
    assert result.micro_f1 == pytest.approx(1.0)
    assert result.localization_score == pytest.approx(1.0)
    assert result.clip_accuracy == pytest.approx(1.0)
    assert result.mean_ap == pytest.approx(1.0)
    assert set(result.per_type_f1) == {"frozen_frames", "inserted_object", "dropped_frames"}
    assert set(result.per_type_ap) == set(result.per_type_f1)


def test_no_predictions_on_edited_clips():
    g = label("frozen_frames", 2.0, 4.0)
    result = score_predictions([clip("a", [g]), clip("clean", [])], [[], []])
    # AP = 0, F1 = 0, localisation = 0, balanced accuracy = (0 + 1) / 2.
    assert result.macro_f1 == 0.0
    assert result.mean_ap == 0.0
    assert result.localization_score == 0.0
    assert result.clip_accuracy == pytest.approx(0.5)
    assert result.score == pytest.approx(0.05 * 0.5)
    assert result.loss == pytest.approx(1.0 - 0.025)
    assert result.per_type_counts["frozen_frames"]["fn"] == pytest.approx(1.5)


def test_hallucinations_on_clean_clips():
    g = label("frozen_frames", 2.0, 4.0)
    clips = [clip("a", [g]), clip("clean", [])]
    preds = [[exact(g)], [pred("zoom_jump", 1.0, 2.0)]]
    result = score_predictions(clips, preds)
    # frozen_frames F1 = 1, zoom_jump has an FP and no GT -> F1 = 0, macro = 0.5.
    assert result.per_type_f1["zoom_jump"] == 0.0
    assert result.macro_f1 == pytest.approx(0.5)
    # The confident FP joins the localisation denominator: 1 TP (q=1) over 1 GT + 1 FP.
    assert result.localization_score == pytest.approx(0.5)
    assert result.clip_accuracy == pytest.approx(0.5)  # TPR 1, TNR 0
    # mAP: frozen_frames AP 1, zoom_jump (predicted, no ground truth) AP 0.
    assert result.mean_ap == pytest.approx(0.5)
    assert result.score == pytest.approx(0.75 * 0.5 + 0.2 * 0.5 + 0.05 * 0.5)
    assert result.precision == pytest.approx(0.5)
    assert result.recall == pytest.approx(1.0)


def test_all_clean_and_no_predictions_has_perfect_f1():
    result = score_predictions([clip("a", []), clip("b", [])], [[], []])
    assert result.macro_f1 == result.micro_f1 == 1.0
    assert result.clip_accuracy == 1.0
    # Nothing to localise and nothing hallucinated: a silent detector is perfect.
    assert result.localization_score == 1.0
    assert result.mean_ap == 1.0
    assert result.per_type_ap == {}
    assert result.score == pytest.approx(1.0)
    assert result.per_type_f1 == {}


def test_all_clean_with_hallucination_scores_below_one():
    result = score_predictions([clip("a", []), clip("b", [])], [[pred("zoom_jump", 1, 2)], []])
    assert result.macro_f1 == 0.0
    assert result.mean_ap == 0.0
    assert result.clip_accuracy == pytest.approx(0.5)


def test_types_without_gt_or_predictions_are_excluded_from_macro():
    g = label("mirrored_segment", 1.0, 3.0)
    result = score_predictions([clip("a", [g])], [[exact(g)]])
    assert list(result.per_type_f1) == ["mirrored_segment"]
    assert result.macro_f1 == 1.0


def test_tiou_threshold_boundary():
    settings = ScoringSettings(tiou_threshold=0.5)
    g = label("reversed_segment", 0.0, 4.0)
    at_threshold = score_predictions([clip("a", [g])], [[pred("reversed_segment", 0.0, 2.0)]], settings)
    counts = at_threshold.per_type_counts["reversed_segment"]
    assert counts["tp"] == pytest.approx(1.5) and counts["fp"] == 0.0 and counts["fn"] == 0.0
    assert at_threshold.localization_score == pytest.approx(0.5)

    below = score_predictions([clip("a", [g])], [[pred("reversed_segment", 0.0, 1.5)]], settings)
    counts = below.per_type_counts["reversed_segment"]
    assert counts["tp"] == 0.0 and counts["fp"] == pytest.approx(1.5) and counts["fn"] == pytest.approx(1.5)


def test_wrong_type_is_fp_and_fn():
    g = label("frozen_frames", 2.0, 4.0)
    result = score_predictions([clip("a", [g])], [[pred("zoom_jump", 2.0, 4.0)]])
    assert result.per_type_counts["frozen_frames"]["fn"] > 0
    assert result.per_type_counts["zoom_jump"]["fp"] > 0
    assert result.macro_f1 == 0.0


def test_min_event_padding_matches_point_event_within_a_fifth_of_a_second():
    g = label("dropped_frames", 5.0, 5.0)
    near = score_predictions([clip("a", [g])], [[pred("dropped_frames", 5.15, 5.15)]])
    assert near.per_type_counts["dropped_frames"]["tp"] > 0
    assert near.macro_f1 == pytest.approx(1.0)
    # GT padded to [4.8, 5.2], prediction to [4.95, 5.35]: tIoU = 0.25 / 0.55.
    assert near.localization_score == pytest.approx(0.25 / 0.55)

    far = score_predictions([clip("a", [g])], [[pred("dropped_frames", 5.3, 5.3)]])
    assert far.per_type_counts["dropped_frames"]["tp"] == 0.0


def test_padding_is_shifted_back_inside_the_clip():
    # Point event at the very start: padded to [0, 0.4], not [-0.2, 0.2].
    g = label("dropped_frames", 0.0, 0.0)
    result = score_predictions([clip("a", [g])], [[pred("dropped_frames", 0.0, 0.0)]])
    assert result.localization_score == pytest.approx(1.0)
    # And at the end of the clip.
    g_end = label("dropped_frames", 10.0, 10.0)
    result = score_predictions([clip("a", [g_end])], [[pred("dropped_frames", 10.0, 10.0)]])
    assert result.localization_score == pytest.approx(1.0)


def test_duplicate_predictions_give_one_tp_and_one_fp():
    g = label("zoom_jump", 2.0, 4.0)
    result = score_predictions([clip("a", [g])], [[exact(g), exact(g)]])
    counts = result.per_type_counts["zoom_jump"]
    assert counts["tp"] == pytest.approx(1.5)
    assert counts["fp"] == pytest.approx(1.5)
    assert counts["fn"] == 0.0
    assert result.per_type_f1["zoom_jump"] == pytest.approx(2 * 0.5 * 1.0 / 1.5)


def test_greedy_matching_prefers_highest_tiou_and_confidence_order():
    g1 = label("frozen_frames", 1.0, 3.0)
    g2 = label("frozen_frames", 5.0, 7.0)
    clips = [clip("a", [g1, g2])]
    # Each prediction should pair with its own GT regardless of list order.
    preds = [[pred("frozen_frames", 5.0, 7.0, 0.6), pred("frozen_frames", 1.0, 3.0, 0.9)]]
    result = score_predictions(clips, preds)
    counts = result.per_type_counts["frozen_frames"]
    assert counts["tp"] == pytest.approx(3.0) and counts["fp"] == 0.0 and counts["fn"] == 0.0
    assert result.localization_score == pytest.approx(1.0)


def test_cap_overflow_counts_as_false_positives():
    settings = ScoringSettings(max_predictions_per_clip=2)
    g1 = label("frozen_frames", 1.0, 2.0)
    g2 = label("frozen_frames", 5.0, 6.0)
    preds = [
        [
            pred("frozen_frames", 1.0, 2.0, 0.9),
            pred("frozen_frames", 5.0, 6.0, 0.8),
            pred("frozen_frames", 8.0, 9.0, 0.7),
        ]
    ]
    result = score_predictions([clip("a", [g1, g2])], preds, settings)
    counts = result.per_type_counts["frozen_frames"]
    assert counts["tp"] == pytest.approx(3.0)
    assert counts["fp"] == pytest.approx(1.5)


def test_cap_drops_lowest_confidence_even_if_correct():
    settings = ScoringSettings(max_predictions_per_clip=2)
    g = label("frozen_frames", 1.0, 2.0)
    preds = [
        [
            pred("frozen_frames", 8.0, 9.0, 0.9),
            pred("frozen_frames", 6.0, 7.0, 0.8),
            pred("frozen_frames", 1.0, 2.0, 0.6),  # correct, but ranked third
        ]
    ]
    result = score_predictions([clip("a", [g])], preds, settings)
    counts = result.per_type_counts["frozen_frames"]
    assert counts["tp"] == 0.0
    assert counts["fp"] == pytest.approx(4.5)  # three FPs at weight 1.5
    assert counts["fn"] == pytest.approx(1.5)


def test_confidence_threshold_filters_predictions():
    g = label("frozen_frames", 1.0, 2.0)
    clips = [clip("a", [g]), clip("clean", [])]
    preds = [[pred("frozen_frames", 1.0, 2.0, 0.49)], [pred("zoom_jump", 1.0, 2.0, 0.1)]]
    result = score_predictions(clips, preds)
    counts = result.per_type_counts["frozen_frames"]
    assert counts["tp"] == 0.0 and counts["fp"] == 0.0 and counts["fn"] == pytest.approx(1.5)
    assert result.per_type_counts["zoom_jump"]["fp"] == 0.0
    assert result.clip_accuracy == pytest.approx(0.5)  # edited clip not flagged, clean clip fine

    # At exactly the threshold the prediction counts.
    preds_at = [[pred("frozen_frames", 1.0, 2.0, 0.5)], []]
    assert score_predictions(clips, preds_at).macro_f1 == pytest.approx(1.0)

    lenient = ScoringSettings(confidence_threshold=0.0)
    assert score_predictions(clips, preds, lenient).per_type_counts["zoom_jump"]["fp"] > 0


def test_difficulty_weighting_changes_pooled_counts():
    g = label("zoom_jump", 2.0, 4.0)
    clips = [clip("easy", [g], "easy"), clip("hard", [g], "hard")]
    preds = [[exact(g)], []]  # easy clip hit, hard clip missed
    result = score_predictions(clips, preds)
    counts = result.per_type_counts["zoom_jump"]
    assert counts["tp"] == pytest.approx(1.0)
    assert counts["fn"] == pytest.approx(2.0)
    assert counts["support"] == pytest.approx(3.0)
    assert result.recall == pytest.approx(1.0 / 3.0)  # unweighted would be 0.5
    assert result.precision == pytest.approx(1.0)
    assert result.localization_score == pytest.approx(1.0 / 3.0)

    flat = ScoringSettings(
        difficulty_weights=(("easy", 1.0), ("medium", 1.0), ("hard", 1.0), ("expert", 1.0))
    )
    assert score_predictions(clips, preds, flat).recall == pytest.approx(0.5)


def test_spatial_bbox_affects_localization_but_not_f1():
    g = label("inserted_object", 2.0, 4.0, bbox=[0.1, 0.1, 0.5, 0.5])
    clips = [clip("a", [g])]
    same = score_predictions(clips, [[pred("inserted_object", 2.0, 4.0, 1.0, [0.1, 0.1, 0.5, 0.5])]])
    missing = score_predictions(clips, [[pred("inserted_object", 2.0, 4.0)]])
    disjoint = score_predictions(
        clips, [[pred("inserted_object", 2.0, 4.0, 1.0, [0.6, 0.6, 0.9, 0.9])]]
    )
    half = score_predictions(clips, [[pred("inserted_object", 2.0, 4.0, 1.0, [0.1, 0.1, 0.5, 0.3])]])

    for result in (same, missing, disjoint, half):
        assert result.macro_f1 == pytest.approx(1.0)
        assert result.micro_f1 == pytest.approx(1.0)
    assert same.localization_score == pytest.approx(1.0)
    assert missing.localization_score == pytest.approx(0.5)
    assert disjoint.localization_score == pytest.approx(0.5)
    assert half.localization_score == pytest.approx(0.5 + 0.5 * 0.5)
    assert same.score > half.score > missing.score


def test_bbox_ignored_for_non_spatial_types():
    g = label("zoom_jump", 2.0, 4.0)
    result = score_predictions(
        [clip("a", [g])], [[pred("zoom_jump", 2.0, 4.0, 1.0, [0.6, 0.6, 0.9, 0.9])]]
    )
    assert result.localization_score == pytest.approx(1.0)


def test_partial_tiou_localization_weighting():
    g = label("color_grade_jump", 0.0, 4.0)
    result = score_predictions([clip("a", [g])], [[pred("color_grade_jump", 0.0, 3.0)]])
    assert result.localization_score == pytest.approx(0.75)
    # tIoU 0.75 clears every mAP threshold (0.3 ... 0.7), so AP is 1.
    assert result.mean_ap == pytest.approx(1.0)
    assert result.score == pytest.approx(0.75 * 1.0 + 0.2 * 0.75 + 0.05 * 1.0)


def test_clip_accuracy_uses_present_class_only():
    g = label("frozen_frames", 1.0, 2.0)
    assert score_predictions([clip("a", [g])], [[exact(g)]]).clip_accuracy == pytest.approx(1.0)
    assert score_predictions([clip("a", [g])], [[]]).clip_accuracy == 0.0
    assert score_predictions([clip("a", [])], [[]]).clip_accuracy == 1.0


def test_score_predictions_input_validation():
    with pytest.raises(ValueError):
        score_predictions([clip("a", [])], [])
    with pytest.raises(ValueError):
        score_predictions([], [])


def test_score_is_clipped_and_result_is_finite():
    g = label("frozen_frames", 1.0, 2.0)
    result = score_predictions([clip("a", [g])], [[exact(g)]])
    assert 0.0 <= result.score <= 1.0
    assert math.isfinite(result.loss)


# --------------------------------------------------------------------------
# ScoringSettings validation
# --------------------------------------------------------------------------


def test_default_settings_are_valid_and_hashable():
    settings = ScoringSettings()
    total = (
        settings.weight_map
        + settings.weight_f1
        + settings.weight_localization
        + settings.weight_clip_accuracy
    )
    assert total == pytest.approx(1.0)
    assert (settings.weight_map, settings.weight_f1) == (0.75, 0.0)
    assert settings.tiou_thresholds == (0.3, 0.4, 0.5, 0.6, 0.7)
    assert settings.bbox_iou_threshold == 0.3
    assert settings.weight_for("expert") == 2.5
    hash(settings)


def test_difficulty_weights_accept_mapping():
    settings = ScoringSettings(
        difficulty_weights={"easy": 1, "medium": 2, "hard": 3, "expert": 4}  # type: ignore[arg-type]
    )
    assert settings.weight_for("hard") == 3.0
    assert settings.weight_for("expert") == 4.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tiou_threshold": 0.0},
        {"tiou_threshold": 1.5},
        {"min_event_seconds": -0.1},
        {"min_event_seconds": math.nan},
        {"confidence_threshold": -0.1},
        {"confidence_threshold": 1.1},
        {"max_predictions_per_clip": 0},
        {"weight_f1": 0.5},
        {"weight_map": 0.8},
        {"weight_map": 1.2, "weight_localization": -0.1, "weight_clip_accuracy": -0.1},
        {"difficulty_weights": (("easy", 1.0), ("medium", 1.0))},
        # "expert" is required (every manifest difficulty needs a weight).
        {"difficulty_weights": (("easy", 1.0), ("medium", 1.5), ("hard", 2.0))},
        {
            "difficulty_weights": (
                ("easy", 1.0),
                ("medium", 0.0),
                ("hard", 1.0),
                ("expert", 1.0),
            )
        },
        {"tiou_thresholds": ()},
        {"tiou_thresholds": (0.0, 0.5)},
        {"tiou_thresholds": (0.5, 1.5)},
        {"tiou_thresholds": (0.5, 0.5)},
        {"tiou_thresholds": (math.nan,)},
        {"bbox_iou_threshold": 0.0},
        {"bbox_iou_threshold": 1.2},
    ],
)
def test_invalid_settings_rejected(kwargs):
    with pytest.raises(ValueError):
        ScoringSettings(**kwargs)


def test_score_term_weights_may_be_zero():
    # weight_map = 0 with the v1 weights reproduces the v1 (F1-based) score.
    v1 = ScoringSettings(
        weight_map=0.0, weight_f1=0.6, weight_localization=0.25, weight_clip_accuracy=0.15
    )
    g = label("color_grade_jump", 0.0, 4.0)
    result = score_predictions([clip("a", [g])], [[pred("color_grade_jump", 0.0, 3.0)]], v1)
    assert result.score == pytest.approx(0.6 * 1.0 + 0.25 * 0.75 + 0.15 * 1.0)


# --------------------------------------------------------------------------
# Mean average precision
# --------------------------------------------------------------------------

THRESHOLD_KEYS = ["0.3", "0.4", "0.5", "0.6", "0.7"]


def test_single_perfect_prediction_has_ap_one():
    g = label("frozen_frames", 2.0, 4.0)
    result = score_predictions([clip("a", [g])], [[exact(g)]])
    assert result.mean_ap == pytest.approx(1.0)
    assert result.per_type_ap == {"frozen_frames": pytest.approx(1.0)}
    assert list(result.ap_by_tiou) == THRESHOLD_KEYS
    assert all(v == pytest.approx(1.0) for v in result.ap_by_tiou.values())


def test_higher_confidence_false_positive_halves_ap():
    g = label("frozen_frames", 2.0, 4.0)
    # Ranked: FP (0.9), TP (0.6).  precision 0 -> 1/2 at recall 1; AP = 1 * 0.5.
    preds = [[pred("frozen_frames", 6.0, 8.0, 0.9), pred("frozen_frames", 2.0, 4.0, 0.6)]]
    result = score_predictions([clip("a", [g])], preds)
    assert result.per_type_ap["frozen_frames"] == pytest.approx(0.5)
    assert result.mean_ap == pytest.approx(0.5)

    # Same predictions with the confidences swapped: the FP trails, AP is 1.
    swapped = [[pred("frozen_frames", 6.0, 8.0, 0.6), pred("frozen_frames", 2.0, 4.0, 0.9)]]
    assert score_predictions([clip("a", [g])], swapped).mean_ap == pytest.approx(1.0)


def test_ap_ignores_confidence_threshold_but_f1_does_not():
    g = label("frozen_frames", 2.0, 4.0)
    result = score_predictions([clip("a", [g])], [[pred("frozen_frames", 2.0, 4.0, 0.05)]])
    assert result.mean_ap == pytest.approx(1.0)  # low confidence is still ranked
    assert result.macro_f1 == 0.0  # F1 path discards it


def test_partial_recall_and_duplicate_predictions():
    g1 = label("frozen_frames", 1.0, 2.0)
    g2 = label("frozen_frames", 6.0, 7.0)
    one_hit = score_predictions([clip("a", [g1, g2])], [[exact(g1)]])
    assert one_hit.mean_ap == pytest.approx(0.5)  # recall 0.5 at precision 1

    # A trailing duplicate is an FP after the only TP: it cannot lower AP.
    dup = score_predictions(
        [clip("a", [g1])],
        [[pred("frozen_frames", 1.0, 2.0, 0.9), pred("frozen_frames", 1.0, 2.0, 0.8)]],
    )
    assert dup.mean_ap == pytest.approx(1.0)


def test_precision_envelope_interpolation():
    g1 = label("frozen_frames", 1.0, 2.0)
    g2 = label("frozen_frames", 6.0, 7.0)
    preds = [
        [
            pred("frozen_frames", 3.5, 4.5, 0.9),  # FP
            pred("frozen_frames", 1.0, 2.0, 0.8),  # TP
            pred("frozen_frames", 6.0, 7.0, 0.7),  # TP
        ]
    ]
    # Raw precision 0, 1/2, 2/3 at recall 0, 1/2, 1. The envelope lifts the whole
    # curve to 2/3, so AP = 1.0 * 2/3 (not 0.5 * 0.5 + 0.5 * 2/3).
    result = score_predictions([clip("a", [g1, g2])], preds)
    assert result.mean_ap == pytest.approx(2.0 / 3.0)


def test_boundary_quality_counts_at_lower_thresholds_only():
    g = label("reversed_segment", 0.0, 4.0)
    # Prediction [0, 2.2]: tIoU = 2.2 / 4 = 0.55 -> TP at 0.3/0.4/0.5, FP at 0.6/0.7.
    result = score_predictions([clip("a", [g])], [[pred("reversed_segment", 0.0, 2.2)]])
    assert result.ap_by_tiou == {
        "0.3": pytest.approx(1.0),
        "0.4": pytest.approx(1.0),
        "0.5": pytest.approx(1.0),
        "0.6": pytest.approx(0.0),
        "0.7": pytest.approx(0.0),
    }
    assert result.per_type_ap["reversed_segment"] == pytest.approx(0.6)
    assert result.mean_ap == pytest.approx(0.6)

    exact_result = score_predictions([clip("a", [g])], [[exact(g)]])
    assert exact_result.mean_ap > result.mean_ap


def test_custom_tiou_thresholds_and_keys():
    g = label("reversed_segment", 0.0, 4.0)
    settings = ScoringSettings(tiou_thresholds=(0.5, 0.75))
    result = score_predictions([clip("a", [g])], [[pred("reversed_segment", 0.0, 3.2)]], settings)
    # tIoU = 0.8 clears 0.5 and 0.75.
    assert list(result.ap_by_tiou) == ["0.5", "0.75"]
    assert result.mean_ap == pytest.approx(1.0)
    worse = score_predictions([clip("a", [g])], [[pred("reversed_segment", 0.0, 2.4)]], settings)
    assert worse.ap_by_tiou == {"0.5": pytest.approx(1.0), "0.75": pytest.approx(0.0)}
    assert worse.mean_ap == pytest.approx(0.5)


def test_spatial_type_needs_a_box_to_count_for_map():
    g = label("inserted_object", 2.0, 4.0, bbox=[0.1, 0.1, 0.5, 0.5])
    clips = [clip("a", [g])]
    with_box = score_predictions(
        clips, [[pred("inserted_object", 2.0, 4.0, 1.0, [0.1, 0.1, 0.5, 0.5])]]
    )
    no_box = score_predictions(clips, [[pred("inserted_object", 2.0, 4.0)]])
    assert with_box.mean_ap == pytest.approx(1.0)
    assert no_box.mean_ap == 0.0
    # F1 (v1 semantics) still credits the temporal match without a box.
    assert no_box.macro_f1 == pytest.approx(1.0)
    assert with_box.score > no_box.score


def test_bbox_iou_below_threshold_is_a_false_positive():
    g = label("blurred_region", 2.0, 4.0, bbox=[0.1, 0.1, 0.5, 0.5])
    clips = [clip("a", [g])]
    # IoU = 0.04 / 0.28 = 0.143 (< 0.3).
    far = [[pred("blurred_region", 2.0, 4.0, 1.0, [0.4, 0.1, 0.8, 0.5])]]
    assert score_predictions(clips, far).mean_ap == 0.0
    assert score_predictions(clips, far, ScoringSettings(bbox_iou_threshold=0.1)).mean_ap == (
        pytest.approx(1.0)
    )
    # IoU = 0.5 (>= 0.3).
    half = [[pred("blurred_region", 2.0, 4.0, 1.0, [0.1, 0.1, 0.5, 0.3])]]
    assert score_predictions(clips, half).mean_ap == pytest.approx(1.0)
    assert score_predictions(
        clips, half, ScoringSettings(bbox_iou_threshold=0.6)
    ).mean_ap == pytest.approx(0.0)


def test_bbox_is_ignored_for_non_spatial_types_in_map():
    g = label("zoom_jump", 2.0, 4.0)
    result = score_predictions(
        [clip("a", [g])], [[pred("zoom_jump", 2.0, 4.0, 1.0, [0.6, 0.6, 0.9, 0.9])]]
    )
    assert result.mean_ap == pytest.approx(1.0)


def test_cross_clip_ranking_rewards_calibrated_confidence():
    g = label("frozen_frames", 2.0, 4.0)
    clips = [clip("a", [g]), clip("b", [g])]
    hit = pred("frozen_frames", 2.0, 4.0, 0.6)
    miss = pred("frozen_frames", 6.0, 8.0, 0.9)
    # Ranked FP (b), TP (a): precision 1/2 at recall 1/2 -> AP = 0.5 * 0.5.
    assert score_predictions(clips, [[hit], [miss]]).mean_ap == pytest.approx(0.25)
    # Calibrated: TP first (precision 1 at recall 1/2), then an FP -> AP = 0.5.
    calibrated = [[pred("frozen_frames", 2.0, 4.0, 0.9)], [pred("frozen_frames", 6.0, 8.0, 0.6)]]
    assert score_predictions(clips, calibrated).mean_ap == pytest.approx(0.5)


def test_matching_is_restricted_to_the_same_clip():
    g = label("frozen_frames", 2.0, 4.0)
    clips = [clip("a", [g]), clip("clean", [])]
    # The clean clip's prediction overlaps a's ground truth in time but must not match it.
    preds = [[pred("frozen_frames", 2.0, 4.0, 0.6)], [pred("frozen_frames", 2.0, 4.0, 0.9)]]
    # Ranked FP (clean), TP (a): precision 1/2 (equal weights) at recall 1.
    assert score_predictions(clips, preds).mean_ap == pytest.approx(0.5)


def test_wrong_type_never_matches():
    g = label("frozen_frames", 2.0, 4.0)
    result = score_predictions([clip("a", [g])], [[pred("zoom_jump", 2.0, 4.0)]])
    assert result.per_type_ap == {"frozen_frames": 0.0, "zoom_jump": 0.0}
    assert result.mean_ap == 0.0


def test_difficulty_weights_affect_ap():
    g = label("zoom_jump", 2.0, 4.0)
    clips = [clip("easy", [g], "easy"), clip("expert", [g], "expert")]
    easy_only = score_predictions(clips, [[exact(g)], []])
    expert_only = score_predictions(clips, [[], [exact(g)]])
    # Recall is weighted: 1.0 / 3.5 vs 2.5 / 3.5 at precision 1.
    assert easy_only.mean_ap == pytest.approx(1.0 / 3.5)
    assert expert_only.mean_ap == pytest.approx(2.5 / 3.5)

    flat = ScoringSettings(
        difficulty_weights=(("easy", 1.0), ("medium", 1.0), ("hard", 1.0), ("expert", 1.0))
    )
    assert score_predictions(clips, [[exact(g)], []], flat).mean_ap == pytest.approx(0.5)


def test_difficulty_weights_scale_false_positives_in_the_curve():
    g = label("zoom_jump", 2.0, 4.0)
    clips = [clip("easy", [g], "easy"), clip("expert", [], "expert")]
    # Ranked FP (expert clip, weight 2.5) then TP (easy clip, weight 1.0), recall 1:
    # precision = 1 / 3.5.
    preds = [[pred("zoom_jump", 2.0, 4.0, 0.5)], [pred("zoom_jump", 2.0, 4.0, 0.9)]]
    assert score_predictions(clips, preds).mean_ap == pytest.approx(1.0 / 3.5)


def test_type_with_predictions_but_no_ground_truth_scores_zero_ap():
    g = label("frozen_frames", 2.0, 4.0)
    clips = [clip("a", [g]), clip("clean", [])]
    # Even a 0.1-confidence hallucination counts for mAP.
    preds = [[exact(g)], [pred("zoom_jump", 1.0, 2.0, 0.1)]]
    result = score_predictions(clips, preds)
    assert result.per_type_ap == {"frozen_frames": pytest.approx(1.0), "zoom_jump": 0.0}
    assert result.mean_ap == pytest.approx(0.5)
    assert all(v == pytest.approx(0.5) for v in result.ap_by_tiou.values())


def test_types_without_gt_or_predictions_are_excluded_from_map():
    g = label("mirrored_segment", 1.0, 3.0)
    result = score_predictions([clip("a", [g])], [[exact(g)]])
    assert list(result.per_type_ap) == ["mirrored_segment"]


def test_all_clean_without_predictions_is_perfect():
    result = score_predictions([clip("a", []), clip("b", [], "expert")], [[], []])
    assert result.mean_ap == 1.0
    assert result.per_type_ap == {}
    assert result.ap_by_tiou == {key: 1.0 for key in THRESHOLD_KEYS}
    assert result.score == pytest.approx(1.0)


def test_cap_drops_lowest_ranked_predictions_from_ap():
    g = label("frozen_frames", 1.0, 2.0)
    preds = [
        [
            pred("frozen_frames", 8.0, 9.0, 0.9),
            pred("frozen_frames", 6.0, 7.0, 0.8),
            pred("frozen_frames", 1.0, 2.0, 0.6),  # correct, but ranked third
        ]
    ]
    capped = score_predictions(
        [clip("a", [g])], preds, ScoringSettings(max_predictions_per_clip=2)
    )
    assert capped.mean_ap == 0.0
    uncapped = score_predictions(
        [clip("a", [g])], preds, ScoringSettings(max_predictions_per_clip=3)
    )
    assert uncapped.mean_ap == pytest.approx(1.0 / 3.0)  # TP at rank 3: precision 1/3


def test_cap_is_applied_per_clip_before_pooling():
    g = label("frozen_frames", 1.0, 2.0)
    clips = [clip("a", [g]), clip("b", [g])]
    settings = ScoringSettings(max_predictions_per_clip=1)
    preds = [
        [pred("frozen_frames", 1.0, 2.0, 0.9), pred("frozen_frames", 5.0, 6.0, 0.8)],
        [pred("frozen_frames", 5.0, 6.0, 0.7), pred("frozen_frames", 1.0, 2.0, 0.6)],
    ]
    # Clip a keeps its TP, clip b keeps only its FP: precision 1 at recall 1/2.
    assert score_predictions(clips, preds, settings).mean_ap == pytest.approx(0.5)


def test_ties_break_by_clip_order_then_prediction_order():
    g = label("frozen_frames", 2.0, 4.0)
    hit = pred("frozen_frames", 2.0, 4.0, 0.5)
    miss = pred("frozen_frames", 6.0, 8.0, 0.5)
    clips = [clip("a", [g]), clip("b", [g])]
    # Equal confidence across clips: clip a first.
    assert score_predictions(clips, [[miss], [hit]]).mean_ap == pytest.approx(0.25)
    assert score_predictions(clips, [[hit], [miss]]).mean_ap == pytest.approx(0.5)
    # Equal confidence within a clip: the detector's order decides.
    one = [clip("a", [g])]
    assert score_predictions(one, [[miss, hit]]).mean_ap == pytest.approx(0.5)
    assert score_predictions(one, [[hit, miss]]).mean_ap == pytest.approx(1.0)


def test_ap_is_deterministic():
    g1 = label("frozen_frames", 1.0, 2.0)
    g2 = label("inserted_object", 4.0, 6.0)
    clips = [clip("a", [g1, g2]), clip("b", [g1], "expert"), clip("c", [])]
    preds = [
        [exact(g1), exact(g2), pred("zoom_jump", 3.0, 4.0, 0.5)],
        [pred("frozen_frames", 1.2, 2.2, 0.5), pred("frozen_frames", 1.0, 2.0, 0.5)],
        [pred("frozen_frames", 1.0, 2.0, 0.5)],
    ]
    first = score_predictions(clips, preds)
    for _ in range(3):
        assert score_predictions(clips, preds) == first


def test_final_score_formula_combines_all_terms():
    g1 = label("color_grade_jump", 0.0, 4.0)
    g2 = label("inserted_object", 5.0, 7.0)
    clips = [clip("a", [g1, g2], "hard"), clip("clean", [])]
    preds = [
        [pred("color_grade_jump", 0.0, 3.0, 0.9), pred("inserted_object", 5.0, 7.0, 0.8)],
        [pred("zoom_jump", 1.0, 2.0, 0.7)],
    ]
    result = score_predictions(clips, preds)
    expected = (
        0.75 * result.mean_ap
        + 0.0 * result.macro_f1
        + 0.2 * result.localization_score
        + 0.05 * result.clip_accuracy
    )
    assert result.score == pytest.approx(expected)
    assert result.loss == pytest.approx(1.0 - expected)
    # color_grade_jump AP 1, inserted_object (no box) 0, zoom_jump hallucination 0.
    assert result.mean_ap == pytest.approx(1.0 / 3.0)


def test_confident_junk_is_not_free_for_localization():
    g = label("frozen_frames", 2.0, 4.0)
    clean_preds = [[exact(g)]]
    junk = [exact(g)] + [pred("zoom_jump", float(t), float(t) + 0.5) for t in range(0, 8, 2)]
    tidy = score_predictions([clip("a", [g])], clean_preds)
    spammed = score_predictions([clip("a", [g])], [junk])
    assert tidy.localization_score == pytest.approx(1.0)
    assert spammed.localization_score == pytest.approx(1.0 / 5.0)
    assert spammed.score < tidy.score
