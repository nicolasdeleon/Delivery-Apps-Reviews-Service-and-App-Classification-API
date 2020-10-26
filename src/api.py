import os

import flask
from flask import Flask, request

from models import BianryModelInterface, ReviewClassifier

APP = Flask(__name__)


@APP.route("/predict")
def predict():
    """predict with a target set to review"""
    review = request.args.get("review")
    prediction = MODEL.predict(review)
    response = {}
    response["response"] = {
        'prediction': prediction
    }
    return flask.jsonify(response)


if __name__ == "__main__":
    MODEL = BianryModelInterface(
        model_bin=os.path.join(
            os.path.abspath('models'), 'binary_service_model.bin'
        ),
        device='cpu',
        tokenizer_state=os.path.abspath('tokenizer_state'),
        cat='Service'
    )
    MODEL.model_ramp_up()
    APP.run()

