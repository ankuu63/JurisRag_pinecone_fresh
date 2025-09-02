import os 
from dotenv import load_dotenv


from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_cohere import CohereRerank

from langchain_core.prompts import PromptTemplate


from langchain_community.document_loaders import PyPDFLoader

from langchain.text_splitter import CharacterTextSplitter


load_dotenv()

#1. llm provider 
class llmsetup:
    def __init__(self):
        api_key=os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("api key not found in env")
        
        self.llm= ChatOpenAI(
            api_key=api_key,
            model="gpt-3.5-turbo",
            temperature=0.7
        )

    def __call__(self, prompt):

        response = self.llm.invoke(prompt)
        return response.content
        

class Embedder:

    def __init__(self):
        self.Emodel= OpenAIEmbeddings(
                    model='text-embedding-3-small',
                    dimensions=200
        )



class VectorStore:
    def __init__(self, embedder: Embedder, index_name="legal-assistant-index"):
        
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise ValueError("Pinecone API key missing from .env")

        self.pinecone = Pinecone(api_key=pinecone_api_key) #object of pinecone

        # Create index if not exists
        if index_name not in [i["name"] for i in self.pinecone.list_indexes()]:
            self.pinecone.create_index(
                name=index_name,
                dimension=200,  # must match embedding dimension
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )

        self.index_name = index_name

        #creating vectorstore
        self.vectorstore = PineconeVectorStore(
            index=self.pinecone.Index(index_name),
            embedding=embedder.Emodel
        )




    def add_documents(self, docs):
        self.vectorstore.add_documents(docs)

    #retriever

    def retriever(self):
        return self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "lambda_mult": 1}
        )




class Preprocessor:

    def __init__(self):

        self.chunker= CharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=80,
            separator="\n"
            
        )    

    def process_pdf(self,pdf_path: str):

        #1 loading pdf into doc obj
        loader= PyPDFLoader(pdf_path)
        docs= loader.load()

        texts= "\n".join(doc.page_content for doc in docs)


        #2. chunkings the text of pdf
        try:
            chunk= self.chunker.split_text(texts)
            return chunk
        except Exception as e:
            print("chunk not chunked by chunker",e)

        

class Reranker:

    def __init__(self):
        key= os.getenv("COHERE_API_KEY")
        if not key:
            raise ValueError("cohere key not found")
    

        self.reranker= CohereRerank(
            cohere_api_key=key,
            model="rerank-english-v3.0",
            top_n=3
        )
    
    

###
###
###callable having main function as run 
    
class RAGPipeline:

    def __init__  (self):
        print("initializing the pipeline")

        self.llm0=llmsetup()
        self.embedder0= Embedder()
        self.vectordb=VectorStore(self.embedder0)
        self.preprocessor= Preprocessor()
        #self.retriever= self.vectordb.retriever()
        self.reranker= Reranker()

        print("pipeline initialized succsefully")




            #######
            #######
    
    def run(self, query: str, pdf_path: str =None)->str:

        #1. process the pdf 
        if pdf_path:
                try:
                    chunks= self.preprocessor.process_pdf(pdf_path)
                    if len(chunks)==0:
                        return "chunks size is 0"
                         
                except Exception as e:
                    print("pdf processing error ")

        else: 
            raise ValueError("pdf_path is currupt")
    


        #2 add chunks into the pineconestore
        
        try:
            self.vectordb.add_documents(chunks)
        except Exception as e:
            print("doc-> vectorstore error:",e)

        try:
            self.retriever= self.vectordb.retriever()
        except Exception as e:
            print("retriever creation error :",e)
        

        retrieved_docs=[]
        reranked_retrieved_docs=[]
        
        try:
            retrieved_docs= self.retriever.get_relevant_documents(query)
        except Exception as e:
            print("retrieving error:",e)


        #reranking
        try:
            reranked_retrieved_docs= self.reranker.rerank(query,retrieved_docs)
        except Exception as e:
            print("reranking error :",e)
            reranked_retrieved_docs=retrieved_docs
        

        context= "\n\n".join([i.page_content for i in reranked_retrieved_docs])

        #3 prompting and generating final prompt

        prompt= PromptTemplate(
                    template= """ You are a helpful Legal assistant , act as an authentic indian legal scholar.
                                  Answer ONLY from the provided data content , you are allowed to answer authentic and genuine facts also .
                                  If the context is insuficient , just say you dnt know it and context is not availabel in doc.

                                context: {context}  question is {question}

                               """,

                    input_variables= ['context', 'question']
        )

        final_prompt= prompt.invoke({
            "context":context,
            "question":query
        })

        
        answer=self.llm0(final_prompt)
        return answer
     
       