import plotly.express as px


# define data
top3_accuracy = [1.0, 1.0, 1.0, 1.0, 0.95, 1.0, 1.0, 0.921, 1.0, 1.0, 1.0, 1.0, 1.0]
top1_accuracy = [0.98, 0.96, 0.97, 0.95, 0.881, 0.98, 0.99, 0.861, 0.99, 0.98, 0.99, 0.99, 0.98]
execution_time = [416, 632, 372, 59, 359, 576, 337, 156, 298, 462, 356, 308, 235]

embedding_models = [
	'ollama-bge-large',
	'ollama-snowflake-arctic-embed2',
	'ollama-paraphrase-multilingual',
	'ollama-all-minilm',
	'ollama-snowflake-arctic-embed',
	'ollama-bge-m3',
	'ollama-mxbai-embed-large',
	'ollama-nomic-embed-text',
	'ollama-embeddinggemma',
	'ollama-nomic-embed-text-v2-moe',
	'openai-text-embedding-3-small',
	'openai-text-embedding-3-large',
	'openai-text-embedding-ada-002'
]

provider = ['ollama' if 'openai' not in j else 'openai' for j in embedding_models]

# accuracy vs execution_time
fig = px.scatter(x=execution_time, y=top1_accuracy, color=provider, size=[10 for j in embedding_models],
                 hover_data=[embedding_models])

fig.update_layout(
	xaxis_title=dict(text="Execution time (ms)", font=dict(size=30)),
	yaxis_title=dict(text="Top1 accuracy", font=dict(size=30)),
	legend=dict(font=dict(size=30))
)

fig.show()

# accuracy vs number_parameters
