from grid_motion.aggregation import evaluate_intervals


def test_reference_evaluation_reports_confusion_and_missing_keys():
    predictions = {
        "intervals": [
            {
                "interval_index": 0,
                "cells": [
                    {"cell_id": 1, "moving": True},
                    {"cell_id": 2, "moving": True},
                ],
            }
        ]
    }
    reference = {
        "intervals": [
            {
                "interval_index": 0,
                "cells": [
                    {"cell_id": 1, "moving": True},
                    {"cell_id": 2, "moving": False},
                    {"cell_id": 3, "moving": True},
                ],
            }
        ]
    }

    metrics = evaluate_intervals(predictions, reference)

    assert metrics["confusion"] == {
        "true_positive": 1,
        "false_positive": 1,
        "false_negative": 1,
        "true_negative": 0,
    }
    assert metrics["missing_prediction_keys"] == [{"interval_index": 0, "cell_id": 3}]
