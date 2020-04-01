from flask import Flask, render_template, request
from forms import ReusableForm

app = Flask(__name__)


# @app.route("/")
# def hello():
#     return "<h1>This is the start</h1>"


# home page
@app.route("/", methods=['GET', 'POST'])
def home():
    form = ReusableForm(request.form)
    return render_template('index.html', form=form)


app.run(host='0.0.0.0', port=50000, debug=True)
