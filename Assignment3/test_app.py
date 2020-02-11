#import sqlite3
from flask import Flask, request, jsonify

app = Flask(__name__)

# Create some test data for our catalog in the form of a list of dictionaries.
books = [
    {'id': 0,
     'title': 'A Fire Upon the Deep',
     'author': 'Vernor Vinge',
     'first_sentence': 'The coldsleep itself was dreamless.',
     'year_published': '1992'},
    {'id': 1,
     'title': 'The Ones Who Walk Away From Omelas',
     'author': 'Ursula K. Le Guin',
     'first_sentence': 'With a clamor of bells that set the swallows soaring, the Festival of Summer came to the city Omelas, bright-towered by the sea.',
     'published': '1973'},
    {'id': 2,
     'title': 'Dhalgren',
     'author': 'Samuel R. Delany',
     'first_sentence': 'to wound the autumnal city.',
     'published': '1975'}
]

@app.route('/', methods=['GET'])
def fun():
#  name = request.args.get("name", "World")
#  return jsonify({"response":"Hola " + name})
  return "Please, specify a route"

@app.route('/read/all', methods=['GET']))
def read_all():
  return jsonify(books)

#  DATABASE = os.path.join(app.root_path, 'Assignment2.db')
#  table = request.args.get("table")
#  conn = sqlite3.connect('DATABASE')
#  cur = conn.cursor()
#  cur.execute('SELECT * FROM "table"  ORDER BY name')
#  people = cur.fetchall()
#  for id in table:
#    print(f'{name} {}')

#@app.route('/update/', methods=['GET, 'POST']))
#def updatee():
#    if validate_on_submit():
#      flash('Data introduced correctly!', 'success')
#    else:
#      flash('wrong data. Please check data introduced')
#  return render_template('write.html',title='Writing')

#@app.route('/delete/')
#def delete():

#@app.route('/create/')
#def create():
