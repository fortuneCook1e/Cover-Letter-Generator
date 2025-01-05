import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

class Llm:
    def __init__(self):
        self.llm = ChatGroq(
            model="llama-3.2-90b-vision-preview",
            temperature=0,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
    
    def scrap_web(self, url):
        loader = WebBaseLoader(url)
        page_data = loader.load().pop().page_content
        return page_data
    
    def extract_job_info(self, page_data):
        prompt_extract = PromptTemplate.from_template(
            """
            This is a piece of scraped text from a job posting website. I want you to extract the following information from this scraped text:
            - Job Title
            - Description
            - Requirements of the role
            
            Return the extracted data in valid JSON format, with each field clearly labeled. Only return the JSON object without any additional text or explanations.
            
            Scraped Job Data:
            {page_data}
                               
            Output:
            """                   
        )
        chain_extract = prompt_extract | self.llm  # Use self.llm instead of passing llm as an argument
        res = chain_extract.invoke(input={'page_data': page_data})
        
        # Parse JSON with error handling
        json_parser = JsonOutputParser()
        try:
            json_res = json_parser.parse(res.content)
        except Exception as e:
            print("Error parsing JSON:", e)
            json_res = None
        return json_res
    
    def generate_job_query(self, json_res):
        if json_res is None:
            return None  # Early exit if json_res is invalid
        # Combine job information into a single query string
        job_query = f"ROLE: {json_res['Job Title']}. \nDESCRIPTION: {json_res['Description']} \nREQUIREMENTS: " + ", ".join(json_res['Requirements'])
        return job_query
    
    def generate_cover_letter(self, job_query, profile_info):
        if job_query is None:
            return "Error: Invalid job information provided."
        
        prompt_extract = PromptTemplate.from_template(
            """
            Based on the following job description and my profile information, write a tailored cover letter for the job. 

            Adhere STRICTLY to these rules:
            RULES 1: Only include skills, tools, and experiences that are directly mentioned in the profile information. Do not infer or add any skills that are not explicitly stated.
            RULES 2: Focus primarily on my project experience that demonstrate relevant skills and results for the job. Mention specific projects and achievements whenever possible, showing how they relate to the job requirements.
            RULES 3: Do not include any placeholders like '[company name]'. If specific company details are needed, simply state 'the company' or 'your organization' instead.
            RULES 4: Only output the cover letter. Do not include any additional text or explanations.
            RULES 5: Keep the cover letter concise (in about 300 words)
            
            Job Description:
            {job_description}
            
            Profile Information:
            {profile_info}
            
            Cover Letter:
            """
        )

        chain_extract = prompt_extract | self.llm  # Use self.llm instead of passing llm as an argument
        res = chain_extract.invoke(input={'job_description': job_query, 'profile_info': profile_info})
        return res.content
