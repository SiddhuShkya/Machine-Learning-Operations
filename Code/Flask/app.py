from flask import Flask

"""
The code 'app = Flask()' creates an instance of the Flask class,
which will be your WSGI (Web Server Gateway Interface) application.
"""
# WSGI Application
app = Flask(__name__)


@app.route("/")
def welcome():
    return "Hello, Flask!"


@app.route("/index")
def index():
    return "This is the index page."


print("Creating Flask app instance...")
if __name__ == "__main__":
    app.run(debug=True)
