from flask import Flask

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'

# Import routes
from app import routes
