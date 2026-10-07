from mena_mlops.load_testing import valid_prediction_response


def test_prediction_response_contract_example() -> None:
    body = {
        "label": "positive",
        "probabilities": {
            "negative": 0.1,
            "neutral": 0.2,
            "positive": 0.7,
        },
        "model_version": "stable-v1",
    }

    assert valid_prediction_response(body)


def test_invalid_prediction_response_is_rejected() -> None:
    assert not valid_prediction_response({"label": "positive"})
