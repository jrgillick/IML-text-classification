from flask import Flask, render_template, request
from flask import session, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
import secrets

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


class Artwork(db.Model):
  id = db.Column(db.Integer, primary_key=True)
  description = db.Column(db.String(256), index=True)
  labelA = db.Column(db.String(256), index=True)
  labelB = db.Column(db.String(256), index=True)
  predicted = db.Column(db.Integer, index=True)
  hide = db.Column(db.Integer, index=True)

  def to_dict(self):
    return {
      'id': self.id,
      'description': self.description,
      'labelA': self.labelA,
      'labelB': self.labelB,
      'predicted': self.predicted,
      'hide': self.hide
    }

class Session(db.Model):
  sessionID = db.Column(db.String(256), index=True)

db.create_all()

@app.route('/')
def index():
    sessionID = secrets.token_hex(16)
    return render_template('iml_table.html',
                           title='Art Description Classification with Interactive Machine Learning',
                           sessionID=sessionID)

@app.route('/api/data')
def data():
    query = Artwork.query

    # search filter
    search = request.args.get('search[value]')
    if search:
        query = query.filter(db.or_(
            Artwork.description.like(f'%{search}%')
        ))
    total_filtered = query.count()

    # sorting
    order = []
    i = 0
    while True:
        col_index = request.args.get(f'order[{i}][column]')
        if col_index is None:
            break
        col_name = request.args.get(f'columns[{col_index}][data]')
        if col_name not in ['id', 'description', 'labelA', 'labelB', 'predicted']:
            col_name = 'name'
        descending = request.args.get(f'order[{i}][dir]') == 'desc'
        col = getattr(Artwork, col_name)
        if descending:
            col = col.desc()
        order.append(col)
        i += 1
    if order:
        query = query.order_by(*order)

    # pagination
    start = request.args.get('start', type=int)
    length = request.args.get('length', type=int)
    query = query.offset(start).limit(length)

    # response
    return {
        'data': [artwork.to_dict() for artwork in query],
        'recordsFiltered': total_filtered,
        'recordsTotal': Artwork.query.count(),
        'draw': request.args.get('draw', type=int),
    }

if __name__ == '__main__':
    app.run()
