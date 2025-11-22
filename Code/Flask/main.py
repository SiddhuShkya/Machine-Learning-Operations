from flask import Flask, render_template

"""
The code 'app = Flask()' creates an instance of the Flask class,
which will be your WSGI (Web Server Gateway Interface) application.
"""
# WSGI Application
app = Flask(__name__)

@app.route("/")
def welcome():
    return f"<html><h1>Welcome to flask app<h1></html>"


@app.route("/index")
def index():
    return render_template('index.html')


@app.route("/about")
def about():
    return render_template('about.html')


print("Creating Flask app instance...")
if __name__ == "__main__":
    app.run(debug=True)
