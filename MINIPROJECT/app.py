from flask import Flask, request, render_template
import pandas as pd

app = Flask(__name__, static_folder='static/css')

database = {'helloworld@gmail.com': '123',
            'hello@gmail.com': '345', 'mean@gmail.com': 'average'}


@app.route('/')
def hello_world():
    return render_template('home.html')


@app.route('/index1', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        # print(request.form)
        name1 = request.form['email']
        pwd = request.form['password']
        # print(name1, pwd)
        if name1 not in database:
            return render_template('home.html', info='Invalid User')
        else:
            if database[name1] != pwd:
                return render_template('home.html', info='Invalid Password')
            else:
                return render_template('index1.html', username=name1)
    else:
        return "Method not allowed. Please use the POST method to access this route."


@app.route('/file_accepted', methods=['POST', 'GET'])
def file_accept():
    if request.method == 'POST':
        # print(request.form)
        inp_file = request.form['file']
        # print(name1, pwd
        return render_template('result.html', file_name=inp_file)
    else:
        return "Method not allowed. Please use the POST method to access this route."


@app.route('/about', methods=['POST', 'GET'])
def about():
    return render_template('about.html')


@app.route('/help', methods=['POST'])
def help():
    return render_template('help.html')


@app.route('/file', methods=['POST', 'GET'])
def upload():
    file = request.files['file']
    df = pd.read_csv(file)
    return render_template('result.html')

@app.route('/file', methods=['POST', 'GET'])
def get_result():
    result="Here's your result!!!"
    return result

if __name__ == '__main__':
    app.run(debug=True, port=5000)
