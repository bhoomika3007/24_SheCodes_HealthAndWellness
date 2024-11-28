from flask import Flask
from flask_sqlalchemy import SQLAlchemy

# Initialize Flask app
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'  # SQLite database file
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # Disable modification tracking
app.secret_key = 'supersecretkey'  # Used for sessions

# Initialize the database
db = SQLAlchemy(app)

# Import routes after the app is initialized
from app import routes

# Create database tables if they don't exist already
with app.app_context():
    db.create_all()  # This will create the database and the tables from the model

# Now you can use the app to run your application
