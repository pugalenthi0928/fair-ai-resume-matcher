import torch
from sentence_transformers import SentenceTransformer, util
import pandas as pd

# Load BERT model for sentence embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

# Sample job descriptions
job_descriptions = [
    "Looking for an experienced Data Scientist with strong Python and machine learning skills.",
    "Software Engineer needed with expertise in Java, Spring Boot, and cloud deployment.",
    "Hiring an NLP researcher with deep learning and transformer experience."
]

# Sample resumes
resumes = [
    "Machine learning expert with Python, TensorFlow, and cloud computing experience.",
    "Java developer with Spring Boot, microservices, and AWS deployment skills.",
    "NLP scientist with experience in BERT, GPT, and transformers."
]

# Encode job descriptions and resumes
job_embeddings = model.encode(job_descriptions, convert_to_tensor=True)
resume_embeddings = model.encode(resumes, convert_to_tensor=True)

# Compute cosine similarity between resumes and job descriptions
similarity_matrix = util.pytorch_cos_sim(resume_embeddings, job_embeddings)

# Display results
df = pd.DataFrame(similarity_matrix.numpy(), index=resumes, columns=job_descriptions)
print("\nResume Matching Scores:\n")
print(df)

# Find best matches
for i, resume in enumerate(resumes):
    best_match_idx = torch.argmax(similarity_matrix[i]).item()
    print(f"\nBest match for resume: '{resume}'")
    print(f"→ Matched Job: '{job_descriptions[best_match_idx]}'")
    print(f"→ Similarity Score: {similarity_matrix[i][best_match_idx].item():.2f}")
