from flask import Flask, jsonify, request, render_template

app = Flask(__name__)

items = [
    {"id": 1, "name": "Item 1", "description": "This is item 1"},
    {"id": 2, "name": "Item 2", "description": "This is item 2"},
]


@app.route("/")
def home():
    return "Welcome To the Sample To Do List App"


# Demo page showing how to use GET, POST, PUT and DELETE against this API
@app.route("/demo")
def demo():
    return render_template("http_verbs_demo.html")


# Get: Retrieve all the items
@app.route("/items", methods=["GET"])
def get_items():
    return jsonify(items)


# Get: Retrieve a specific item by Id
@app.route("/items/<int:item_id>", methods=["GET"])
def get_item(item_id):
    item = next((item for item in items if item["id"] == item_id), None)
    if item is None:
        return jsonify({"Error": "Item not found!"}), 404
    return jsonify(item)


## Post: Create a new task
@app.route("/items", methods=["POST"])
def create_item():
    if not request.json or "name" not in request.json:
        return jsonify({"Error": "Please provide a name for the item!"}), 400
    new_item = {
        "id": items[-1]["id"] + 1 if items else 1,
        "name": request.json["name"],
        "description": request.json.get("description", ""),
    }
    items.append(new_item)
    return jsonify(new_item), 201


# Put: Update an Existing item
@app.route("/items/<int:item_id>", methods=["PUT"])
def update_item(item_id):
    item = next((item for item in items if item["id"] == item_id), None)
    if item is None:
        return jsonify({"Error": "Item not found!"}), 404
    item["name"] = request.json.get("name", item["name"])
    item["description"] = request.json.get("description", item["description"])
    return jsonify(item)


# Delete: Delete an item
@app.route("/items/<int:item_id>", methods=["DELETE"])
def delete_item(item_id):
    global items
    item = next((item for item in items if item["id"] == item_id), None)
    if item is None:
        return jsonify({"Error": "Item not found!"}), 404
    items = [item for item in items if item["id"] != item_id]
    return jsonify({"result": "Item deleted!"})


if __name__ == "__main__":
    app.run(debug=True)
