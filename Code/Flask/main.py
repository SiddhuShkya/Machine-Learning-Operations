from flask import Flask, render_template, request

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


@app.route("/form", methods=['GET', 'POST'])
def form():
    if request.method == 'POST':
        # Capture form data
        personal_info = {
            'name': request.form.get('name', ''),
            'age': request.form.get('age', ''),
            'country': request.form.get('country', '')
        }
        # Render the form with submitted data
        return render_template('form.html', submitted=True, info=personal_info)
    # Render empty form for GET request
    return render_template('form.html', submitted=False, info=None)

@app.route("/submit", methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        # Capture form data
        personal_info = {
            'name': request.form.get('name', ''),
            'age': request.form.get('age', ''),
            'country': request.form.get('country', '')
        }
        # Render the form with submitted data
        return f"Hello {personal_info['name']}. Your age is {personal_info['age']} and you are from {personal_info['country']}."
    # Render empty form for GET request
    return f"Hello {personal_info['name']}. Your age is {personal_info['age']} and you are from {personal_info['country']}."

print("Creating Flask app instance...")
if __name__ == "__main__":
    app.run(debug=True)
