from flask import Flask, render_template, request, redirect
from survey import gen_survey_result
from flask_cors import CORS
import matplotlib

app = Flask(__name__, static_url_path='')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
CORS(app)

matplotlib.use('Agg')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/vectors')
def vectors():
    print("AAA vectors")
    return app.send_static_file("vectors.html")


@app.route('/survey', methods=['POST', 'GET'])
def survey():
    if request.method == 'POST':
        # print(request.form)
        result = gen_survey_result(request.form)
        # print(result)
        return render_template('survey_result.html', result=result)
    else:
        return render_template('survey.html')


if __name__ == "__main__":
    app.run(debug=True)
