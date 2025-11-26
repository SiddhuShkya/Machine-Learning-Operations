from flask import Flask, render_template, request, redirect, url_for
"""
The code 'app = Flask()' creates an instance of the Flask class,
which will be your WSGI (Web Server Gateway Interface) application.
"""
# WSGI Application
app = Flask(__name__)

@app.route("/")
def welcome():
    return f"<html><h1>Welcome to flask app<h1></html>"


@app.route("/index", methods=['GET'])
def index():
    return render_template('index.html')


@app.route("/about")
def about():
    return render_template('about.html')

@app.route('/success/<int:score>')
def success(score):
    if score >= 40:
        res="Passed"
    else:
        res="Failed"
    return render_template('result.html', result=res)

@app.route('/successres/<int:score>')
def successres(score):
    if score >= 40:
        res="Passed"
    else:
        res="Failed"
    exp = {
        'score': score,
        'result': res
    }
    return render_template('result1.html', result=exp)

@app.route('/successif/<int:score>')
def successif(score):
    return render_template('result.html', result=score)

@app.route('/fail/<int:score>')
def fail(score):
    return render_template('result.html', result=score)


@app.route('/submit', methods=['POST', 'GET'])
def submit():
    total = 0
    if request.method == 'POST':
        math = request.form['math']
        science = request.form['science']
        english = request.form['english']
        social = request.form['social']
        computer = request.form['computer']
        total = (int(math) + int(science) + int(english) + int(social) + int(computer)) / 5 
    else:
        return render_template('getresult.html') 
    return redirect(url_for('successres', score=int(total)))

print("Creating Flask app instance...")
if __name__ == "__main__":
    app.run(debug=True)
