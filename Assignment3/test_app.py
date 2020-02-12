import sqlite3
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d

@app.route('/', methods=['GET'])
def fun(secret_key=None):
  return "Please, specify a route: read/all, readresearcher or addresearcher"

@app.route('/read/all', methods=['GET'])
def read_all(secret_key=None):

  query_parameters = request.args
  table = query_parameters.get('table')

  conn = sqlite3.connect('Assignment2.db')
  conn.row_factory = dict_factory
  cur = conn.cursor()

  if table == 'researchers':
    all = cur.execute('SELECT * FROM researchers;').fetchall()
#    all = pd.read_sql_query('SELECT * FROM researchers;',conn)
#    all = all.to_json()
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

@app.route('/readresearcher', methods=['GET'])
def read_filter(secret_key=None):
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

@app.route('/addresearcher', methods=['POST', 'GET'])
def addresearcher():
  #app.cofig['SECRET_KEY'] = '12345678'
  headers = request.headers
  auth = headers.get("Api-Key")
  if auth == '12345678':
    if request.method == 'POST':
      try:
        query_parameters = request.args
        first_name = query_parameters.get('first_name')
        last_name = query_parameters.get('last_name')
        id = query_parameters.get('id')
        department_id = query_parameters.get('department_id')
        phone = query_parameters.get('phone')

        conn = sqlite3.connect('Assignment2.db')
        cur = conn.cursor()
        results = cur.execute("INSERT INTO researchers (department_id,first_name,id,last_name,phone) VALUES (?,?,?,?,?)",(int(department_id),first_name,int(id),last_name,phone))

        conn.commit()
        msg = 'Record succesfully added'
        results_read = cur.execute('SELECT * FROM researchers;').fetchall()
#        results_read = pd.read_sql_query('SELECT * FROM researchers;',conn)
#        results_read = results_read.to_json()
#        result_read = pd.json_normalize(results_read)

      except:
        results_read = conn.rollback()
        msg = 'Error adding data'
#        return jsonify(message = msg)

      conn.close()
      return jsonify(message = msg, new_table = results_read)

    else:
      return "No a POST method"
  else:
    return "UNAUTHORIZED"

