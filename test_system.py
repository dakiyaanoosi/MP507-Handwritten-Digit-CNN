import ast
import io
import unittest
import numpy as np
from PIL import Image, ImageDraw


class TestMP507System(unittest.TestCase):
    def test_01_ast_syntax(self):
        """Verify AST syntax of all Python source files."""
        files = ["app.py", "routes.py", "train_cnn.py"]
        for f in files:
            with open(f, "r", encoding="utf-8") as src_file:
                code = src_file.read()
            # If ast.parse fails, an exception is raised
            ast.parse(code, filename=f)
            print(f"[OK] AST syntax valid for {f}")

    def test_02_preprocessing_blank(self):
        """Verify that a completely blank/white image produces an all-zero tensor."""
        from routes import preprocess_digit_image
        # White 280x280 canvas image
        blank_img = Image.new("RGB", (280, 280), (255, 255, 255))
        tensor, canvas = preprocess_digit_image(blank_img, invert=True)
        self.assertEqual(tensor.shape, (1, 28, 28, 1))
        self.assertEqual(float(np.max(tensor)), 0.0)
        print("[OK] Blank image produces all-zero (1, 28, 28, 1) tensor")

    def test_03_preprocessing_digit(self):
        """Verify preprocessing on a synthetic digit."""
        from routes import preprocess_digit_image
        # Draw a black stroke on a white canvas
        img = Image.new("RGB", (280, 280), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        # Draw a synthetic digit '7'
        draw.line([(80, 60), (200, 60)], fill=(0, 0, 0), width=18)
        draw.line([(200, 60), (100, 220)], fill=(0, 0, 0), width=18)

        tensor, canvas = preprocess_digit_image(img, invert=True)
        self.assertEqual(tensor.shape, (1, 28, 28, 1))
        self.assertGreater(float(np.max(tensor)), 0.0)
        self.assertLessEqual(float(np.max(tensor)), 1.0)
        self.assertGreaterEqual(float(np.min(tensor)), 0.0)
        print(f"[OK] Synthetic digit preprocessed successfully. Max={np.max(tensor):.2f}, Nonzero pixels={np.count_nonzero(tensor)}")

    def test_04_flask_health_and_index(self):
        """Verify / and /api/health routes."""
        from app import create_app
        app = create_app()
        app.testing = True
        client = app.test_client()

        res_index = client.get("/")
        self.assertEqual(res_index.status_code, 200)
        self.assertIn(b"MP507: Handwritten Digit Recognition", res_index.data)

        res_health = client.get("/api/health")
        self.assertEqual(res_health.status_code, 200)
        data = res_health.get_json()
        self.assertEqual(data["status"], "ok")
        print(f"[OK] Flask index and health endpoints verified. Model loaded: {data.get('model_loaded')}")

    def test_05_predict_blank_rejection(self):
        """Verify /api/predict rejects blank canvas with HTTP 400."""
        from app import create_app
        import base64
        app = create_app()
        app.testing = True
        client = app.test_client()

        # Blank white image data URL
        blank_img = Image.new("RGB", (280, 280), (255, 255, 255))
        buf = io.BytesIO()
        blank_img.save(buf, format="PNG")
        b64_str = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

        res = client.post("/api/predict", json={"image": b64_str})
        # If model is loaded, it should return 400 Blank image detected
        # If model is not loaded yet, 503
        self.assertIn(res.status_code, [400, 503])
        print(f"[OK] Prediction validation for blank input returned status {res.status_code}: {res.get_json()}")


if __name__ == "__main__":
    unittest.main()
