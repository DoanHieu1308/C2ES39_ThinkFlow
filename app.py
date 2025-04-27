from flask import Flask
from routes.ai_speed_to_text_routes import summary_routes

app = Flask(__name__)
app.register_blueprint(summary_routes)

if __name__ == '__main__':
    app.run(debug=True)
