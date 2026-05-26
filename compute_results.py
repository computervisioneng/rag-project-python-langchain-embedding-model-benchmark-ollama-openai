__import__('pysqlite3')
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import time

import chromadb


# create chromadb client
client_db = chromadb.PersistentClient('./db')

# define text embedding models
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
	'openai-text-embedding-ada-002'
]

# read questions from file
file_ = open('./results', 'r')

lines = [l[:-1] for l in file_.readlines() if len(l) > 1]

accuracies_top1 = []
accuracies_top3 = []
execution_times = []

# for each embedding model
for embedding_model in embedding_models:
	accuracy_top1 = 0
	accuracy_top3 = 0
	execution_time = 0

	#   get collection
	collection = client_db.get_collection(name=f"documents_{embedding_model}")

	#   for each question
	for line in lines:
		question_index, question = line.split('---')

		#       get most similar text chunks to question
		tic = time.time()
		results = collection.query(
			query_texts=[question],
			n_results=3
		)
		exec_time = time.time() - tic

		#       compute top1_accuracy, top3_accuracy, execution_time
		accuracy_top1 += 1 if str(question_index) in results['ids'][0][:1] else 0
		accuracy_top3 += 1 if str(question_index) in results['ids'][0] else 0
		execution_time += exec_time

	accuracies_top3.append(round(accuracy_top3 / len(lines), 3))
	accuracies_top1.append(round(accuracy_top1 / len(lines), 3))
	execution_times.append(round(execution_time / len(lines), 3))

print(accuracies_top3)
print(accuracies_top1)
print(execution_times)
