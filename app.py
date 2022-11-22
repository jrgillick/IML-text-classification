from flask import Flask, render_template, request
from flask import redirect, url_for, jsonify
from flask_sqlalchemy import SQLAlchemy
import secrets

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

from sqlalchemy import Column, ForeignKey, Integer, Table
from sqlalchemy.orm import relationship

import pickle, os
import modeling
if os.path.exists('bert_sentence_embeddings.pkl'):
  with open('bert_sentence_embeddings.pkl', 'rb') as f:
    bert_sentence_embeddings = pickle.load(f)
  print(len(bert_sentence_embeddings))

import pdb; pdb.set_trace()

class Artwork(db.Model):
  #__tablename__ = "artwork"
  unique_id = db.Column(db.Integer, primary_key=True)
  id = db.Column(db.Integer, index=True)
  description = db.Column(db.String(512), index=True)
  labelA = db.Column(db.String(256), index=True)
  labelB = db.Column(db.String(256), index=True)
  predicted = db.Column(db.Integer, index=True)
  hide = db.Column(db.Integer, index=True)
  classifier_code = db.Column(db.String(64), nullable=False)
  #labels = db.relationship("Label", backref='artwork', lazy=True)

  def flip_labelA(self):
    unset = None
    if self.labelA is None:
      self.labelA = 1 # Set labelA
      if self.labelB is not None:
        self.labelB = None # Unset labelB if set
        unset = 'B'
    else:
        self.labelA = None # Unset labelA if set
        unset = 'A'
    return unset

  def flip_labelB(self):
    unset = None
    if self.labelB is None:
      self.labelB = 1 # Set labelB
      if self.labelA is not None:
        self.labelA = None # Unset labelA if set
        unset = 'A'
    else:
        self.labelB = None # Unset labelB if set
        unset = 'B'
    return unset

  def to_dict(self):
    #classifier = db.session.query(Classifier).filter(Artwork.classifier_code == self.classifier_code)
    h = {
      'id': self.id,
      'description': self.description,
      'labelA': self.labelA,
      'labelB': self.labelB,
      'predicted': self.predicted,
      #'hide': self.hide, TODO add this back later
      'classifier_code': self.classifier_code
    }
    return h

class Classifier(db.Model):
  #__tablename__ = "classifier"
  id = db.Column(db.Integer, primary_key=True)
  classifier_code = db.Column(db.String(256), index=True)
  categoryA = db.Column(db.String(256), index=True)
  categoryB = db.Column(db.String(256), index=True)
  custom_count = db.Column(db.Integer, index=True)
  #labels = db.relationship("Label", backref='classifier', lazy=True)

  def self.sentencesA(self):
    # return desc for all artworks with labelA and this classifier code
  def self.sentencesB(self):
    # return desc for all artworks with labelB and this classifier code

#class Label(db.Model):
#  #__tablename__ = "label"
#  id = db.Column(db.Integer, primary_key=True)
#  classifier_id = db.Column(db.Integer, db.ForeignKey('classifier.id'), nullable=False)
#  artwork_id = db.Column(db.Integer, db.ForeignKey('artwork.id'), nullable=False)


db.create_all()

@app.route('/')
def index():
  classifier_code = secrets.token_hex(16)
  classifier = Classifier(classifier_code = classifier_code, categoryA='A', categoryB='B', custom_count=0)
  db.session.add(classifier)
  db.session.commit()
  #query = Artwork.query.filter(classifier_code=='0')
  query = db.session.query(Artwork).filter(Artwork.classifier_code == '0')
  initial_artworks = [artwork.to_dict() for artwork in query]

  # Create data for this new classifier
  for i, art in enumerate(initial_artworks):
    new_artwork = Artwork(id=art['id'], description = art['description'], labelA = None, labelB = None, predicted = None, hide = None, classifier_code = classifier_code)
    db.session.add(new_artwork)
    db.session.commit()
  return redirect(url_for('build_classifier', classifier_code = classifier_code))

@app.route('/add_custom_text', methods=['GET', 'POST'])
def add_custom_text():
  data = request.get_json()
  sentence = data['custom_text']
  if sentence is '' or sentence is None: return jsonify({})

  query = db.session.query(Classifier).filter(Classifier.classifier_code == data['classifier_code'])
  classifier = [classifier for classifier in query][0]
  #create a new Artwork object with this text
  embedding = modeling.get_bert_text_embedding(sentence).squeeze().detach().numpy()
  bert_sentence_embeddings[sentence] = embedding
  new_id = -1*classifier.custom_count - 1
  new_artwork = Artwork(id=new_id, description = data['custom_text'], predicted = None, hide = None, classifier_code = data['classifier_code'])
  if data['label'] == 'A':
    new_artwork.labelA = 1
  else:
    new_artwork.labelB = 1
  classifier.custom_count += 1
  db.session.add(new_artwork)
  db.session.commit()
  h = new_artwork.to_dict()
  return jsonify(h)

  
@app.route('/check_label', methods=['GET', 'POST'])
def check_label():
  if request.method == 'POST':
    data = request.get_json()
    classifier_code = data['classifier_code']
    artwork_id = int(data['id'][1:])
    label = data['id'][0]

    # Load the Artwork object that was labeled
    query = db.session.query(Artwork).filter(Artwork.classifier_code == classifier_code, Artwork.id == artwork_id)
    artwork =  [artwork for artwork in query][0]
    print(f"Artwork ID: {artwork.id}")

    # flip the label that was clicked on
    # Then check the other label and turn it off if they're both on now (can't both be on)
    if label == 'A':
      unset = artwork.flip_labelA()
      db.session.commit()
    if label == 'B':
      unset = artwork.flip_labelB()
      db.session.commit()

    # Update that artwork in the database
    #db.session.update(artwork)

    h = artwork.to_dict()
    if unset is not None:
      h['unset'] = unset
    return jsonify(h)
    #return redirect(url_for('build_classifier', classifier_code = classifier_code))


@app.route('/build_classifier')
def build_classifier():
  classifier_code = request.args['classifier_code']
  return render_template('iml_table.html',
                           title='Classifying Artwork Descriptions',
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
    app.run(debug=True)
