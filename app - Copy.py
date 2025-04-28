from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
import os
from werkzeug.utils import secure_filename
from pymongo import MongoClient
from flask_pymongo import PyMongo
from bson import ObjectId
from datetime import datetime  # Import datetime module
app = Flask(__name__)

app.config["MONGO_URI"] = "mongodb://localhost:27017/Researchersedst"  # Update with your MongoDB URI
mongo = PyMongo(app)
app.secret_key = "secret_key"

# Configuration for file uploads
EXPERIENCE_FOLDER = os.path.join('static', 'experiences')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'pdf', 'mp4', 'avi', 'mkv'}

# Ensure the experiences folder exists
if not os.path.exists(EXPERIENCE_FOLDER):
    os.makedirs(EXPERIENCE_FOLDER)
UPLOAD_FOLDER = "static/experiences"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


app.config['UPLOAD_FOLDER'] = EXPERIENCE_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route to display all uploaded experiences
@app.route('/')
def index():
    experiences = []
    for folder in os.listdir(UPLOAD_FOLDER):
        experience_folder = os.path.join(UPLOAD_FOLDER, folder)
        if os.path.isdir(experience_folder):
            files = os.listdir(experience_folder)
            experiences.append({
                'name': folder,
                'files': files
            })
    return render_template('index.html', experiences=experiences)
@app.route('/submit_researcher', methods=['POST'])
def submit():
    # Retrieve data from the form
    researcher = request.form.get('researcher')
    email = request.form.get('email')
    experience = request.form.get('title')

    title = request.form.get('title')
    summary = request.form.get('summary')
    start_date = request.form.get('start_date')
    end_date = request.form.get('end_date')
    field = request.form.get('field')
    methods = request.form.getlist('methods')
    tools = request.form.getlist('tools')
    years_of_experience = request.form.get("years_of_experience")
    file_paths = []
    # Handle missing or empty values
    if years_of_experience is None or years_of_experience.strip() == "":
        years_of_experience = 0
    else:
        years_of_experience = int(years_of_experience)  # Convert to int
    #metadata files
    experience_folder = os.path.join(UPLOAD_FOLDER, secure_filename(experience))
    os.makedirs(experience_folder, exist_ok=True)

    # Save uploaded files
    files = request.files.getlist('files')
    for file in files:
        if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath=os.path.join(experience_folder, filename)
                file.save(filepath)
                # Store file metadata
                file_paths.append({
                   'filename': filename,
                   'path': filepath,
                   'upload_date': datetime.now().strftime('%Y-%m-%d')
                })  


    # Insert data into MongoDB
    if researcher and email and experience:
        researcher_data = {
            "researcher": researcher,
            "email": email,
            
            "title": title,
            "summary": summary,
            "start_date": start_date,
            "end_date": end_date,
            "attributes": {
            	"field": field,
            	"methods": methods,
            	"tools": tools,
                 # Convert number input to an integer
                "years_of_experience": years_of_experience,
                'files': file_paths
        },
            
             
        }
        collection.insert_one(researcher_data)
       
        return render_template("result.html", researcher_data=researcher_data)
    else:
        return "All fields are required!", 400



# Route to add a new experience
# Route to serve uploaded files
@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)
@app.route('/experience/<name>')
def experience_detail(name):
 # Retrieve the specific experience document by title (or another field) from the 'experiences' collection
    experience = mongo.db.experiences.find_one({"title": name})  # Querying the 'experiences' collection

    
    # Get the path to the specific experience folder
    experience_path = os.path.join(EXPERIENCE_FOLDER, name)
    
    if os.path.exists(experience_path) and os.path.isdir(experience_path):
        # List all files inside this folder
        files = os.listdir(experience_path)
        return render_template(
            'experience_detail.html',
            experience_name=name,
            files=files,
            experience_path=experience_path
            experience=experience # Pass the full data for display
        )
    else:
        return "Experience not found", 404
@app.route('/form')
def form():
    return render_template('form.html')

# Route to search for experiences
@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        query = request.form['query'].lower()
        results = []
        for folder in os.listdir(UPLOAD_FOLDER):
            if query in folder.lower():
                results.append(folder)
        return render_template('search.html', query=query, results=results)
    return render_template('search.html')


if __name__ == '__main__':
    app.run(debug=True,port=5004)
