from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import nltk
nltk.download('punkt', quiet=True)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  
    chunk_overlap=100, 
    length_function=len,
    separators=["\n\n", "\n"]  
)

urls = input("Enter , seperated urls").split(",")
loader = UnstructuredURLLoader(urls=urls)
docs = loader.load()



unique_content = list({doc.page_content.strip(): doc for doc in docs}.values())  
split_docs = text_splitter.split_documents(unique_content)



embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(split_docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})



MODEL_NAME = "google/flan-t5-large" 
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


prompt_template = ChatPromptTemplate.from_messages([
    ("system", "Generate a clear concise most simplest understanding language answer in about 3-5 bullet or more if you need more to explain points, using ONLY the context below.\n\nContext: {context}"),
    ("human", "{input}")
])

pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=800,
    temperature=0.6,
    do_sample=True,
    top_k=50,
    num_beams=3
)

def generate_answer(inputs):
    try:
        system_msg = next(m for m in inputs.messages if m.type == "system")
        human_msg = next(m for m in inputs.messages if m.type == "human")


        formatted_prompt = prompt_template.format(
            context=system_msg.content,
            input=human_msg.content
        )

        return pipe(formatted_prompt)[0]['generated_text']
        
    except Exception as e:
        return f"Error: {str(e)}"


rag_chain = create_retrieval_chain(
    retriever,
    create_stuff_documents_chain(
        llm=generate_answer,
        prompt=prompt_template
    )
)


try:
    user_input = input("Enter question: ")
    while user_input != "exit":
        response = rag_chain.invoke({"input":user_input })
        answer = response["answer"]
        print(f"User Question",user_input)
        print("\nFinal Answer:", answer)
        user_input = input("Enter question: ")
        
    

except Exception as e:
    print(f"Error: {str(e)}")