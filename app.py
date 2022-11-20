from flask import Flask, render_template, request
from flask import redirect, url_for
from flask_sqlalchemy import SQLAlchemy
import secrets

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

from sqlalchemy import Column, ForeignKey, Integer, Table
from sqlalchemy.orm import relationship




class Artwork(db.Model):
  #__tablename__ = "artwork"
  id = db.Column(db.Integer, primary_key=True)
  description = db.Column(db.String(512), index=True)
  labelA = db.Column(db.String(256), index=True)
  labelB = db.Column(db.String(256), index=True)
  predicted = db.Column(db.Integer, index=True)
  hide = db.Column(db.Integer, index=True)
  classifier_code = db.Column(db.String(64), nullable=False)
  #labels = db.relationship("Label", backref='artwork', lazy=True)

  def to_dict(self):
    #classifier = db.session.query(Classifier).filter(Artwork.classifier_code == self.classifier_code)
    h = {
      'id': self.id,
      'description': self.description,
      'labelA': self.labelA,
      'labelB': self.labelB,
      'predicted': self.predicted,
      'hide': self.hide,
      'classifier_code': self.classifier_code
    }
    return h

class Classifier(db.Model):
  #__tablename__ = "classifier"
  id = db.Column(db.Integer, primary_key=True)
  classifier_code = db.Column(db.String(256), index=True)
  categoryA = db.Column(db.String(256), index=True)
  categoryB = db.Column(db.String(256), index=True)
  #labels = db.relationship("Label", backref='classifier', lazy=True)

#class Label(db.Model):
#  #__tablename__ = "label"
#  id = db.Column(db.Integer, primary_key=True)
#  classifier_id = db.Column(db.Integer, db.ForeignKey('classifier.id'), nullable=False)
#  artwork_id = db.Column(db.Integer, db.ForeignKey('artwork.id'), nullable=False)


db.create_all()

@app.route('/')
def index():
  classifier_code = secrets.token_hex(16)
  classifier = Classifier(classifier_code = classifier_code, categoryA='AAAA', categoryB='BBBB')
  db.session.add(classifier)
  db.session.commit()
  #query = Artwork.query.filter(classifier_code=='0')
  query = db.session.query(Artwork).filter(Artwork.classifier_code == '0')
  initial_artworks = [artwork.to_dict() for artwork in query]

  # Create data for this new classifier
  for i, art in enumerate(initial_artworks):
    new_artwork = Artwork(description = art['description'], labelA = None, labelB = None, predicted = None, hide = None, classifier_code = classifier_code)
    db.session.add(new_artwork)
    db.session.commit()
  return redirect(url_for('build_classifier', classifier_code = classifier_code))
  

@app.route('/build_classifier')
def build_classifier():
  classifier_code = request.args['classifier_code']
  return render_template('iml_table.html',
                           title='Build a Classifier',
                           classifier_code = classifier_code)


@app.route('/api/data')
def data():
    classifier_code = request.args['classifier_code']
    #query = Artwork.query
    query = db.session.query(Artwork).filter(Artwork.classifier_code == classifier_code)
    #query = Artwork.query.outerjoin(Artwork.labels)
    #query = query.filter(Label.classifier_code == classifier_code)

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
        'recordsTotal': db.session.query(Artwork).filter(Artwork.classifier_code == classifier_code).count(), #Artwork.query.count(),
        'draw': request.args.get('draw', type=int),
    }



if __name__ == '__main__':
    app.run()
