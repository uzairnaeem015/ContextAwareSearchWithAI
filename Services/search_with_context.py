from langchain_core.prompts import PromptTemplate
from langchain_google_vertexai import VertexAI
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

class SearchResults:
    def __init__(self, topic=None, query = None, vectorstore=None):

        if not topic:
            self.topic = "General Knowledge"
        else:
            self.topic = topic

        self.query = query
        self.vectorstore = vectorstore
        self.llm = None
        self.responses = [] 
        self.system_template = """
            Please response on this topic: {topic} by using the context only, if context not found or irrelevent, then ask the user to update the context file.
            

            Follow the instructions to answer the query in detail: 
            1. Generate a response based on the topic provided and context, and answer the query.
            2. The response should be easy to to understand.
            3. The response should be in a respectable way.
            4. Provide references as much as you can using the context only.
            5. After explanation, use the exact same words as mentioned in the context.

            You must respond in simple text, where as the references will be highlighted in bold 
            
            Query: {query}
            
            Context: {context}
            """
    
    def init_llm(self):
        """
        Initializes and configures the Large Language Model (LLM) for generating responses.

        This method should handle any setup required to interact with the LLM, including authentication,
        setting up any necessary parameters, or selecting a specific model.

        :return: An instance or configuration for the LLM.
        """
        self.llm = VertexAI(
            model_name = "gemini-pro",
            temperature = 0.5, # Increased for less deterministic results 
            max_output_tokens = 2000
        )

    def generate_response_with_vectorstore(self):
        """
        Generates a response based on the topic provided using a vectorstore

        :return: text.
        """
        if not self.llm:
            self.init_llm()
        if not self.vectorstore:
            raise ValueError("Vectorstore not provided.")
        
        

        # Enable a Retriever
        retriever = self.vectorstore.db.as_retriever()
        
        # Function to combine topic and query into a single string
        def combine_topic_and_query(topic, query):
            return f"Topic: {topic}\nQuery: {query}"

        # Combine topic and query into a formatted string
        formatted_input = combine_topic_and_query(self.topic, self.query)

        # RunnableParallel allows Retriever to get relevant documents
        #setup_and_retrieval = RunnableParallel(
        #    {"context": retriever, "formatted_input": RunnablePassthrough()}
        #)
                
        # RunnableParallel allows Retriever to get relevant documents
        # RunnablePassthrough allows chain.invoke to send self.topic to LLM
        setup_and_retrieval = RunnableParallel(
            {"context": retriever, "topic": RunnablePassthrough(), "query": RunnablePassthrough()}
        )
        # Use the system template to create a PromptTemplate
        prompt = PromptTemplate.from_template(self.system_template)
        # Create a chain with the Retriever, PromptTemplate, and LLM.
        chain = setup_and_retrieval | prompt | self.llm  

        response = chain.invoke(formatted_input)
        return response
