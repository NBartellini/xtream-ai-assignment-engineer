import os
import json
import flask
import traceback

from flask_restful import Resource
from marshmallow import Schema
from typing import Any, Dict, List, Callable, Union

app = flask.Flask(__name__)

class Home(Resource):

    def __init__(self):
        super(Home, self).__init__()

    def get(self)->flask.Response:
        message =  f'Hello from Xtream Diamonds\' price prediction!'
        result = {'status_code':200, 'data':{'message':message}}
        response = self.handle_response(result)
        return response
    
    def handle_response(self, result: Dict[str, Any]):
        data, status_code = json.dumps(result['data']), result['status_code']
        response = app.response_class(response=data,status=status_code,mimetype='application/json')
        return response


class Controller(Resource):
    def __init__(self, schema_dict: Dict[str,Any], errors_definitions: Dict[str,Dict[str,Union[int,str]]], pipeline_func: Callable):
        self.schema = Schema.from_dict(schema_dict)
        self.errors = errors_definitions
        self.pipeline_func = pipeline_func
        super(Controller, self).__init__()

    def handle_response(self, result: Dict[str, Any])->flask.Response:
        data, status_code = json.dumps(result['data']), result['status_code']
        response = app.response_class(response=data,status=status_code,mimetype='application/json')
        return response
    
    def handle_error(self, error_name:str,error_call:Exception)->Dict:
        error_info = self.errors.get(error_name)
        if error_info is None:
            error_code = None
            error_message = None
        else:
            error_code = error_info.get('status_code')
            error_message = error_info.get('message')
        if error_code is None:
            error_code = 500
        if error_message is None:
            error_message = "Undefined Error"
        if error_name == 'ValidationError':
            error_message=error_call
            trace_info = "Schemas validation"
        elif isinstance(error_call,Exception):
            trace_info = traceback.extract_tb(error_call.__traceback__)
        else:
            trace_info = ""
        error_dict = {'status_code':error_code,
            'data':{
            'error':error_name,
            'message':error_message,
            'traceback': str(trace_info)
        }}
        return error_dict

    
    def post(self)->flask.Response:
        if flask.request.is_json:
            data = flask.request.get_json(force=True)
            validation_result = self.schema().validate(data)
            if bool(validation_result):
                result = self.handle_error("ValidationError", validation_result)
            else:
                output = self.pipeline_func(**data)
                if isinstance(output,Exception):
                    error_name = type(output).__name__
                    result = self.handle_error(error_name, output)
                else:
                    result = {'status_code':200, 'data':{'input':data,'output':output}}
        else:
            result = self.handle_error('NoJSONError', Exception())   
        response = self.handle_response(result)
        return response