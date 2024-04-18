from marshmallow import fields, ValidationError

def validate_text(text):
    is_space = text.isspace()
    is_empty = not(text)
    if is_space or is_empty:
        raise ValidationError("Text must not be empty or contain only spaces")

def validate_dimensions(number):
    is_less_zero = number <=0
    if is_less_zero:
        raise ValidationError("Dimensions must be greater than 0.")

schema_dict = {
    'carat': fields.Float(required=True, validate=validate_dimensions),
    'cut': fields.String(required=True, validate=validate_text),
    'color': fields.String(required=True, validate=validate_text),
    'clarity': fields.String(required=True, validate=validate_text),
    'depth': fields.Float(required=True, validate=validate_dimensions),
    'table': fields.Float(required=True, validate=validate_dimensions),
    'x': fields.Float(required=True, validate=validate_dimensions),
    'y': fields.Float(required=True, validate=validate_dimensions),
    'z': fields.Float(required=True, validate=validate_dimensions)
}