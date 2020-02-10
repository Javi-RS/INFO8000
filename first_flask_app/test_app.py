from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def fun():
    name = request.args.get("name", "World")
    return jsonify("response":"Hello " + name)