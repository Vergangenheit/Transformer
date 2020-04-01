from wtforms import (Form, TextField, validators, SubmitField)


class ReusableForm(Form):
    """User entry form for entering questions for generation of answer"""
    seed = TextField("Enter a seed string or 'random' : ", validators=[validators.InputRequired()])

    # submit button
    submit = SubmitField("Enter")
