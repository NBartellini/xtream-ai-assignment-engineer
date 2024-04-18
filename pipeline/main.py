import os
import flask
import logging
import argparse
from flask_restful import Api

from pipeline.errors import errors
from pipeline.pipeline import pipeline
from pipeline.schemas import schema_dict
from pipeline.controllers import Home, Controller

app = flask.Flask(__name__)

api = Api(app)
api.add_resource(Home, "/")
api.add_resource(Controller, "/controller",resource_class_kwargs={'pipeline_func':pipeline,'errors_definitions':errors,'schema_dict':schema_dict})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)