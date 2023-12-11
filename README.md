# Resume Recommendation System with Deeplake Vector DB and Langchain
This repository contains a Resume Recommendation System that leverages Deeplake Vector DB and Langchain RAG (Retrieval Augmented Generation) to match job descriptions with stored resumes. The system identifies similar resumes based on job descriptions using Langchain and selects top candidates using a custom similarity function. Finally, the system sends the selected resumes back to the hiring manager.

## Overview

The goal of this project is to streamline the hiring process by automating the matching of job descriptions with stored resumes. Utilizing Langchain, the system identifies resumes similar to provided job descriptions. The custom similarity function further refines the selection process to recommend top candidates whose resumes closely align with the job requirements.

## Features

- Deeplake Vector DB Integration: Stores resumes in a vector database for efficient retrieval and comparison.
- Langchain RAG Integration: Uses Langchain RAG to find resumes similar to provided job descriptions.
- Custom Similarity Function: Filters and ranks resumes based on a custom similarity metric tailored to the specific job requirements.
- Deployment: Complete solution is deployed using Flask. It can be hosted locally or for production setup in AWS ec2 by exposing port 5000 (given in main.py)
- GMail Integration: Automatically creates a mail and sends the top candidate resumes back to the hiring manager for review.

## Installation

1. Clone the repository: 

    ```
    git clone https://github.com/yourusername/resume-recommendation.git
    ```

2. Install dependencies:
    ```
    pip install -r requirements.txt
    ```
3. Set Up Deeplake Vector DB:
    
    - We have used deeplake vector db. Setup free deeplake account from here https://www.deeplake.ai/

4. Setup AWS/GCP/Azure Cloud:
    - To store candidate resumes, create a bucket in any cloud provider and store data in csv/pdf format.

## Usage (locally)
1. Provide Job Description:
Input the job description for which you seek resumes into the system.
2. Run Recommendation System:

    ```
    cd resume_recommendation
    python main.py
    ```
3. Send request:
    ```
    curl -X POST \
            -H "Content-Type: application/json; charset=utf-8" \
            -d @request.json \
            "localhost:5000/process-job-email"
    ```
4. Response format:
```
{
  "responseStatus": "success/failure",
  "data": {
    "replyMessage": "string",
    "resume": "URL/string", 
    "jobRole": "string",
    "skillsHighlighted": ["string"],
    "suitableCandidates": [
      {
        "name": "string",
        "contactInfo": "string",
        "skills": ["string"],
        "resumeLink": "URL/string"
      }
    ]
  },
  "error": {
    "code": "string",
    "message": "string"
  }
}
```


