class FailedCategoricalEncodingError(Exception):
    def __init__(self, message):
        super().__init__(message)

class NoJSONError(Exception):
    def __init__(self, message):
        super().__init__(message)

errors = {
     "ValidationError": {
        "status_code": 400,
        "message": "$validation_message"
    },
    "NoJSONError": {
        "status_code":400,
        "message": "To run this pipeline you must give a json."
    },
     "FailedCategoricalEncodingError": {
        "status_code": 400,
        "message": "Couldn't encode categorical value."
    }
    
    }