__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import time

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# create chromadb client
client_db = chromadb.PersistentClient('./db')


# define embedding models
embedding_models = [
	'bge-large',
	'snowflake-arctic-embed2',
	'paraphrase-multilingual',
	'all-minilm',
	'snowflake-arctic-embed',
	'bge-m3',
	'mxbai-embed-large',
	'nomic-embed-text',
	'embeddinggemma',
	'nomic-embed-text-v2-moe',
	'openai-text-embedding-3-small',
	'openai-text-embedding-3-large',
	'openai-text-embedding-ada-002',
	'qwen3-embedding'
]

# for each embedding model
for embedding_model in embedding_models:

	if 'openai' in embedding_model:
		embedding_function = OpenAIEmbeddingFunction(
					api_key=os.getenv("OPENAI_API_KEY"),
					model_name=embedding_model[7:]
				)

	else:
		embedding_function = OllamaEmbeddingFunction(
				url="http://localhost:11434",
				model_name=embedding_model,
				timeout=180
			)
	#   create chromadb collection
	collection = client_db.create_collection(
		name=f"documents_{embedding_model}",
		embedding_function=embedding_function

	)

	#   split document into text chunks
	loader = DirectoryLoader('./data', glob='*.txt')
	documents = loader.load()

	text_splitter = RecursiveCharacterTextSplitter(
		chunk_size=300,
		chunk_overlap=0,
		length_function=len,
		add_start_index=True
	)

	chunks = text_splitter.split_documents(documents)

	#   add chunks to documents collection
	tic = time.time()
	collection.add(
		ids=[str(j) for j in range(len(chunks))],
		documents=[chunks[j].page_content for j in range(len(chunks))],
		metadatas=[chunks[j].metadata for j in range(len(chunks))]
	)
	print(embedding_model, time.time() - tic)
