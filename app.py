from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy

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

db.create_all()

if __name__ == '__main__':
    app.run()
