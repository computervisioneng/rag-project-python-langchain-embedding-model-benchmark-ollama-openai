import openai

__import__('pysqlite3')
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from openai import OpenAI

# create chromadb client
client_db = chromadb.PersistentClient('./db')

# create openai client
openai_client = OpenAI()

# get collection (any)
collection = client_db.get_collection(name="documents_bge-large")

# get all items in the collection
items = collection.get()

writer = open('./results', 'w')

# for each chunk
for text_chunk_index, text_chunk in enumerate(items['documents']):
	print(text_chunk_index, text_chunk)

	#   create a question for which the text chunk is the answer
	prompt = f"Create a question that can be answered with this text: {text_chunk}"
	response = openai.responses.create(model='gpt-5.2', input=prompt)
	output = response.output_text

	#   save results
	writer.write(f"{str(text_chunk_index)}---{output}\n")
