import os

import flask
from flask import Flask, request

from models import ModelInterface, ReviewClassifier

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
    MODEL = ModelInterface(
        model_bin=os.path.join(
            os.path.abspath('models'), 'model.bin'
        ),
        device='cpu',
        tokenizer_state=os.path.abspath('tokenizer_state'),
    )
    MODEL.model_ramp_up()
    APP.run(debug=True, host='0.0.0.0', port=5000)
