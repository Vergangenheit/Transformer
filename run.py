from flask import Flask, render_template, request
from deployment.forms import ReusableForm
from deployment.utils import generate_from_seed, load_tf_model

app = Flask(__name__)


# @app.route("/")
# def hello():
#     return "<h1>This is the start</h1>"


# home page
@app.route("/", methods=['GET', 'POST'])
def home():
    form = ReusableForm(request.form)

    if request.method == 'POST' and form.validate():
        # extract information
        seed = request.form['seed']
        return render_template('seeded.html', input=generate_from_seed(model=model, seed=seed))
    return render_template('index.html', form=form)


if __name__ == "__main__":
    load_tf_model()
    app.run(host='0.0.0.0', port=50000, debug=True)
