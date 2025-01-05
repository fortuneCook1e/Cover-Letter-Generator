import streamlit as st
from profile import Profile
from llm import Llm

# Initialize the Profile and Llm classes
profile = Profile()
llm = Llm()

# Streamlit app UI
st.title("📧 Cover Letter Generator")

# Input field for job posting URL
job_url = st.text_input("Enter the URL of the job posting", placeholder="https://example.com/job-posting")

# Button to start the process
if st.button("Generate Cover Letter"):
    if job_url:
        with st.spinner("Scraping job posting and generating cover letter..."):
            # Step 1: Scrape job information
            page_data = llm.scrap_web(job_url)
            if page_data:
                # Step 2: Extract job information in JSON format
                job_info = llm.extract_job_info(page_data)
                if job_info:
                    # Step 3: Generate the query from the job information
                    job_query = llm.generate_job_query(job_info)
                    
                    # Step 4: Retrieve relevant profile information
                    profile_info = profile.retrieve_from_vectorstore(job_query)
                    print("PROFILE INFO: " + profile_info)
                    
                    # Step 5: Generate the cover letter
                    cover_letter = llm.generate_cover_letter(job_query, profile_info)
                    
                    # Display the cover letter with copy-to-clipboard functionality
                    st.subheader("Generated Cover Letter")
                    st.code(cover_letter, language='markdown', wrap_lines=True)
                else:
                    st.error("Failed to extract job information. Please check the URL or try again.")
            else:
                st.error("Failed to scrape the job posting. Please check the URL or try again.")
    else:
        st.warning("Please enter a valid job posting URL.")
