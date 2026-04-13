from flask import Flask, render_template, redirect, url_for
import os

app = Flask(__name__)

@app.route('/')
def index():
    """Serve the index page or redirect to main app."""
    # Check if the main Flask app exists in subfolder
    main_app_path = os.path.join(os.path.dirname(__file__), 'Trend-Analysis-NLP')
    if os.path.exists(main_app_path):
        return render_template('index.html')
    else:
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Social Media Analytics</title>
            <style>
                body { font-family: Arial, sans-serif; text-align: center; padding: 50px; }
                .error { color: red; }
            </style>
        </head>
        <body>
            <h1>Social Media Analytics Dashboard</h1>
            <p class="error">Main application folder not found.</p>
            <p>Please ensure the Trend-Analysis-NLP folder exists with the complete Flask application.</p>
        </body>
        </html>
        """

@app.route('/launch')
def launch():
    """Redirect to the main Flask application."""
    return redirect('http://localhost:5000', code=302)

if __name__ == '__main__':
    # Run on a different port to avoid conflicts
    app.run(host='0.0.0.0', port=8080, debug=True)
