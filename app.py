from flask import Flask, render_template, request, redirect, flash, url_for
from werkzeug.utils import secure_filename
from main import getPrediction
import os

# Save images to the 'static' folder as Flask serves images from this directory
UPLOAD_FOLDER = 'static/images/'

# Create app object using Flask class
app = Flask(__name__, static_folder='static')

# Add reference fingerprint.
# Cookies travel with a signature that they claim to be legit.
# Legitimacy here means that the signature was issued by the owner of the cookie.
# Others cannot change this cookie as it needs the secret key.
# Its used as the key to encrypt the session - which can be sotred in a cookie.
# Cokkies should be encrypted if they contain potentially sensitive information.
# app.secrect_key = b'_5#y2L"F4Q8z\n\xec]/'

# Define the upload folder to save images uploaded by the user
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define the route to home.
# The decorator below links the relative route of the URL to the function it is decorator
# Here, index function is with '/', the home directory.
# Running the app sends us to index.html
# Note that render_template means it looks for the file in the templates folder.
@app.route('/')
def index():
    return render_template('index.html')

# Add Post method to the decorator to allow for form submission.
@app.route('/', methods=['POST'])
def submit_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected for uploading')
        return redirect(request.url)
    
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        try:
            # Get prediction on the uploaded file
            label, image_with_boxes_path = getPrediction(filename)

            # Display the result and image path
            flash(f'The object in the image is: {label}')
            flash(image_with_boxes_path)

            return redirect('/')
        except Exception as e:
            flash(f'Error processing image: {str(e)}')
            return redirect(request.url)

if __name__ == "__main__":
    key = os.urandom(12)
    app.secret_key = key
    app.run(debug=True)