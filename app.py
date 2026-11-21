import os
from flask import Flask
from routes import api_bp, load_digit_model


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB upload cap

    model_path = os.path.join("models", "mnist_cnn_model.keras")
    if os.path.exists(model_path):
        load_digit_model(model_path)

    app.register_blueprint(api_bp)
    return app


if __name__ == "__main__":
    application = create_app()
    port = int(os.environ.get("PORT", 5000))
    application.run(host="0.0.0.0", port=port, debug=False)
