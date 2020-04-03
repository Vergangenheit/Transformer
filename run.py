from flask import Flask, render_template, request
from forms import ReusableForm
from utils import generate_from_seed, load_tf_model

app = Flask(__name__)


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
    model = load_tf_model()
    app.run(host='0.0.0.0', port=50000, debug=True)
