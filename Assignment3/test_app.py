from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def fun():
    name = request.args.get("name", "World")
    #return jsonify({"response":"Hello " + name})
    return "Please, specify a route"
    

@app.route('/get')
def create():
    name = request.args.get("name", "World")
    return jsonify({"response":"Hello " + name})

@app.route('/post')
def create():
    name = request.args.get("name", "World")
    
    return jsonify({"result":"Success"})