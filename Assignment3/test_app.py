import sqlite3
from flask import Flask, request, jsonify

app = Flask(__name__)

def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d

@app.route('/', methods=['GET'])
def fun():
#  name = request.args.get("name", "World")
#  return jsonify({"response":"Hola " + name})
  return "Please, specify a route: read or write"

@app.route('/read/all', methods=['GET'])
def read_all():

  query_parameters = request.args
  table = query_parameters.get('table')

  conn = sqlite3.connect('Assignment2.db')
  conn.row_factory = dict_factory
  cur = conn.cursor()

  if table == 'researchers':
    all = cur.execute('SELECT * FROM researchers;').fetchall()
  elif table == 'projects':
    all = cur.execute('SELECT * FROM projects;').fetchall()
  elif table == 'departments':
    all = cur.execute('SELECT * FROM departments;').fetchall()
  elif table == 'tasks':
    all = cur.execute('SELECT * FROM tasks;').fetchall()
  elif table == 'equipment':
     all = cur.execute('SELECT * FROM equipment;').fetchall()
  else:
    return "Please, specify the table to show"
  return jsonify(all)

@app.errorhandler(404)
def page_not_found(e):
  return "<h1>404</h1><p>The resource could not be found.</p>", 404

@app.route('/read_researchers', methods=['GET'])
def read_filter():
  query_parameters = request.args

  id = query_parameters.get('id')
  first_name = query_parameters.get('first_name')
  last_name = query_parameters.get('last_name')
  phone = query_parameters.get('phone')
  department_id = query_parameters.get('department_id')
  

  query = "SELECT * FROM researchers WHERE"
  to_filter = []

  if id:
    query += ' id=? AND'
    to_filter.append(id)

  if first_name:
    query += ' first_name=? AND'
    to_filter.append(published)

  if last_name:
    query += ' last_name=? AND'
    to_filter.append(author)

  if phone:
    query += ' phone=? AND'
    to_filter.append(phone)

  if department_id:
    query += ' department_id=? AND'
    to_filter.append(department_id)

  if not (id or first_name or last_name or phone or department_id):
    return page_not_found(404)

  query = query[:-4] + ';'

  conn = sqlite3.connect('Assignment2.db')
  conn.row_factory = dict_factory
  cur = conn.cursor()

  results = cur.execute(query, to_filter).fetchall()

  return jsonify(results)
