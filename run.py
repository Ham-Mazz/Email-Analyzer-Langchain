
from langchain.llms import GPT4All
from langchain.document_loaders import UnstructuredEmailLoader
from langchain.embeddings import GPT4AllEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.text_splitter import CharacterTextSplitter
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever

import logging
import asyncio


callbacks = [StreamingStdOutCallbackHandler()]


embeddings = GPT4AllEmbeddings()
loader = DirectoryLoader(
        path = "/Users/Hmazz/OneDrive/Desktop/Email LLM/Emails",
		show_progress=True,
		use_multithreading=True,
		loader_cls=UnstructuredEmailLoader,
		loader_kwargs = {"process_attachments"	 : True}, 
    )
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=500)
splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0, separator=". ")
redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)
pipeline_compressor = DocumentCompressorPipeline(
    transformers=[splitter, redundant_filter, relevant_filter]
)

splits = text_splitter.split_documents(data) 

vectordb = Chroma.from_documents(documents=splits, embedding=embeddings)
vectorstore =  Chroma (collection_name="split_parents", embedding_function=embeddings)
store = InMemoryStore()

llm = GPT4All(
    model="/Users/Hmazz/OneDrive\Desktop/Email LLM/Models/gpt4all-falcon-q4_0.gguf", callbacks=callbacks, verbose=True,
)
question = "What are these emails discussing?"

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)
retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=vectordb.as_retriever(), llm=llm
)

retriever.add_documents(data)
compression_retriever = ContextualCompressionRetriever(base_compressor=pipeline_compressor, base_retriever=retriever)

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

#unique_docs = retriever.get_relevant_documents(question)
ensemble_retriever = EnsembleRetriever(
    retrievers=[retriever, compression_retriever], weights=[0.4, 0.6]
)
template = """Use the following emails (context) to build a comprehensive summary about the content of emails, as well as build a task/objectives list from these emails while keeping track of the parties involved in the emails. use the emails to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. End with asking the user if they any questions about the emails. 
{context}
Question: {question}
Helpful Answer:"""
rag_prompt_custom = PromptTemplate.from_template(template)

rag_chain = {"context": retriever, "question": RunnablePassthrough()} | rag_prompt_custom | llm


print(rag_chain.invoke(question))

