from flask import Flask, render_template, request
from predict import predict_category

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        input_text = request.form['text']
        result = predict_category(input_text)
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
