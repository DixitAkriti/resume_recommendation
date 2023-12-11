#import necessary libraries
from flask import Flask, request
import pandas as pd
import openai
import os
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import DeepLake
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from utils import *

os.environ["OPENAI_API_KEY"] = "sk-xxxxxxxxxxxxxxxx"

# Create Retriever
activeloop_org_id= "test_resume"
activelooop_dataset_name = "resumes_dataset_with_out_splitting_v2"
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
dataset_path = f"hub://{activeloop_org_id}/{activelooop_dataset_name}"
os.environ['ACTIVELOOP_TOKEN'] = "xxxxxxxxxxxxxxxxxxx"
db = DeepLake(dataset_path=dataset_path, embedding=embeddings)
retriever = db.as_retriever()


# create template to query from vector db
prompt_template = """As a recruiter, you need to recommend candidates from context. Give only candidate names present in document. Do not give names not present in context, don't hallucinate.
{context}
Question: {question}
Give answer in below format:
["candidate 1", "candidate 2", "candidate 3", "candidate 4"]
-------
"""

def query_from_vector_db(prompt_template, job_requirements, retriever):
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    query_template = "Given this Job requirements:{job_requirements}, give only the Name of 4 candidates from given data similar to job requirements in less than 20 words"

    llm=ChatOpenAI(model="gpt-3.5-turbo")

    chain_type_kwargs = {"prompt": prompt}
    qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs=chain_type_kwargs,)
    
    query = query_template.format(
            job_requirements=job_requirements
        )
    response = qa_chain.run({"query": query})
    return eval(response)


bucket_name = 'genericbucket123'
source_blob_name = 'resume_data/extracted_entities.csv' 
entities_df=download_from_s3(bucket_name, source_blob_name)




def jd_entities(resume_text, model="gpt-3.5-turbo"):
    
    api_key = "sk-xxxxxxxxxxxxxxxx"
    openai.api_key = api_key
    seed=123

    messages = []
    task_message = f"Task: Be precise in your answers, extract as json list of only technical skills, total job experience in months, specific job position role name, location from given text: {resume_text}\n"
    task_message += "Instruction1: Always answer in lower case  \n"
    task_message += "Instruction2: json keys should be Skills, Experience in months, Job Role  \n"
    task_message += "Instruction3: Experience in months can be rounded off to approximate months\n"
    
    messages = [{"role": "user", "content": task_message}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
        seed=seed
    )
    return response.choices[0].message["content"]




custom_weights={'Skills':0.4, 'Job Role':0.4, 'Experience in months':0.2}

def similarity_score(item1, item2):
    common_elements = len(list(set(item1).intersection(set(item2))))
    total_elements=  len(list(set(item1).union(set(item2))))
    return common_elements/total_elements




def mailing_agent(info, receiver_name, sender_name, role_name, model="gpt-3.5-turbo"):
    api_key = "sk-xxxxxxxxxxxxxxxx"
    os.environ["OPENAI_API_KEY"] = api_key
    messages = []
    task_message = ""
    
    # Mention the Task
    task_message = f"candidate info: {info}\ntake this template: Thank you for reaching out to me regarding the position. I have attached resumes for your reference.\nCandidate name: [Candidate name]\nCandidate skills: [Candidate skills]\nCandidate experience: [Experience in months]\nThank you, and have a great day.\nBest regards,\ntask: you are an organisation recommending candidates for {role_name}, be precise, use only information given in prompt, length of output should be less than 100 words, create a mail to {receiver_name} from {sender_name} using only candidate info and template to a hiring manager."
    # print(task_message)


    
    messages = [{"role": "user", "content": task_message}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
        
    )
    return response.choices[0].message["content"]

def getanswer(sender_name, query):
    result_jd = jd_entities(query)
    result_jd = eval(result_jd)
    def custom_similarity(dict1, dict2=result_jd, weights=custom_weights):
    
        skill1=eval(dict1['Skills'])
        skill2=dict2['Skills']
        role1=dict1['Job Role'].replace(',', '').split()
        role2=dict2['Job Role'].replace(',', '').split()
        exp1=dict1['Experience in months']
        exp2=dict2['Experience in months']
        skill_similarity=similarity_score(skill1, skill2)
        role_similarity=similarity_score(role1, role2)
        if exp1>=exp2:
            exp_similarity=1
        else:
            exp_similarity=0
        combined_score= weights['Skills']*skill_similarity + weights['Job Role']*role_similarity + weights['Experience in months']*exp_similarity
        return combined_score
    
    response = query_from_vector_db(prompt_template, query, retriever)
    new_df=entities_df[entities_df['Name'].isin(response)]
    new_df['Combined Score']=new_df[['Skills', 'Experience in months', 'Job Role']].apply(custom_similarity, axis=1)
    result_df=new_df.sort_values('Combined Score', ascending=False)
    result_df=result_df.iloc[:1]
    result_df.to_csv('final.csv', index=False)
    
    response = mailing_agent(result_df.iloc[:1,:4].values, sender_name, 'Rohit', role_name=result_jd["Job Role"])
    output={"Answer":response}
    return output, result_jd, result_df

app = Flask(__name__)

@app.route('/process-job-email', methods=['POST'])
def processclaim():
    
    # try:
    input_json = request.json
    query = input_json["emailData"]["content"]
    response, result_jd, result_df = getanswer(input_json["emailData"]["sender"]["name"], query)
    
    output = {
  "responseStatus": "success/failure",
  "data": {
    "replyMessage": response,
    "resume": query, 
    "jobRole": result_jd['Job Role'],
    "skillsHighlighted": result_jd['Skills'],
    "suitableCandidates": [
      {
        "name": result_df['Name'].iloc[0],
        "contactInfo": "candidate@gmail.com",
        "skills": result_df['Skills'].iloc[0],
        "resumeLink": result_df['Resume'].iloc[0]
      }
    ]
  },
}
    
    return output
    # except:
    #     return jsonify({"Status":"Failure --- some error occured"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
