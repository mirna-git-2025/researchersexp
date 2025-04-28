from pymongo import MongoClient
from datetime import datetime, timedelta
import random

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")  # Change if needed
db = client["Researchersedst"]  # Change to your database name
collection = db["experiences"]  # Change to your collection name

# Delete all existing records
collection.delete_many({})
print("All records deleted successfully.")

# Researchers with their domains and suitable methods/tools
researchers = [
    ("Dr. Mohamad Khalil", "Biomedical Spectrum", ["Genomic Sequencing", "Bioinformatics Analysis", "Proteomics"], ["Bioconductor", "GenePattern", "Galaxy"]),
    ("Dr. Ahmad Shahine", "Computer Science", ["Machine Learning", "Algorithm Optimization", "Data Mining"], ["TensorFlow", "PyTorch", "Scikit-learn"]),
    ("Dr. Monzer Hamzeh", "Microbiology", ["Bacterial Genomics", "Metagenomics", "Antibiotic Resistance Analysis"], ["QIIME", "SPAdes", "MEGA"]),
    ("Dr. Hiba Mawlawi", "Biotechnology", ["CRISPR Gene Editing", "Synthetic Biology", "Molecular Cloning"], ["SnapGene", "Geneious", "Benchling"])
]

# Generate 100 records
new_records = []
start_base = datetime(2020, 1, 1)

for i in range(100):
    researcher, field, methods, tools = random.choice(researchers)
    start_date = start_base + timedelta(days=random.randint(0, 1800))
    end_date = start_date + timedelta(days=random.randint(30, 365))
    
    record = {
        "researcher": researcher,
        "email": researcher.lower().replace(" ", "_") + "@university.edu",
        "title": f"Research in {field}",
        "summary": f"A comprehensive study on {field} using advanced AI and statistical tools.",
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "attributes": {
            "field": field,
            "methods": random.sample(methods, 2),
            "tools": random.sample(tools, 2),
            "years_of_experience": 12,
            "files": [
                {
                    "filename": f"file_{j}.pdf",
                    "path": f"/research_data/{researcher.lower().replace(' ', '_')}/file_{j}.pdf"
                } for j in range(6)
            ]
        }
    }
    
    # Print one record to verify the structure
    if i == 0:
        print("Sample record:\n", record)

    new_records.append(record)

# Insert new records
collection.insert_many(new_records)
print("100 new records inserted successfully.")
