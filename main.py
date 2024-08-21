import streamlit as st

from Services.document_process import DocumentProcessor
from Services.embedding_client import EmbeddingClient
from Services.chroma_collection_creator import ChromaCollectionCreator
from Services.search_with_context import SearchResults

st.title("Hello World")

if __name__ == "__main__":
    
    embed_config = {
        "model_name": "textembedding-gecko@003",
        # "project": "contextawaresearch",
        "project": "gemini-quizify-428916",
        "location": "us-central1"
    }
        
    # Step 1: init the question bank list in st.session_state
    st.session_state['responses'] = []
    ##### YOUR CODE HERE #####

    screen = st.empty()
    with screen.container():
        st.header("Context aware search with AI")
        
        # Create a new st.form flow control for Data Ingestion
        with st.form("Load Data to Chroma"):
            st.write("Select PDFs for Ingestion, the topic and query")
            
            processor = DocumentProcessor()
            processor.ingest_documents()
        
            embed_client = EmbeddingClient(**embed_config) 
        
            chroma_creator = ChromaCollectionCreator(processor, embed_client)

            topic_input = st.text_input("Topic (Should be a name or keyword)", placeholder="Enter the topic - Keyword")

            query = st.text_input("Query for your topic", placeholder="Enter your question in detail")

                
            submitted = st.form_submit_button("Submit")
            
            if submitted:
                chroma_creator.create_chroma_collection()
                    
                if len(processor.pages) > 0:
                    st.write(f"Generating response for topic: {topic_input}")
                
                generator = SearchResults(topic_input, query, chroma_creator) # Step 3: Initialize a QuizGenerator class using the topic, number of questrions, and the chroma collection
                response = generator.generate_response_with_vectorstore()
                st.write(response)
