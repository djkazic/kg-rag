# KG-RAG: LLM Generated Knowledge Graphs and Retrieval

This demo reads plaintext data from `data/` and generates a Knowledge Graph Index (using `llamaindex`). It does this by creating `subject, predicate, object` triplets.

It then opens an interactive CLI prompt for you to ask questions in natural language. So far, I have gotten the best results with the `dolphin-mistral` model.

At inference time, Colbert reranking is done to ensure the most topical graph nodes are fetched.

## Requirements:

- Ollama (must pull the model you want to use)
- All `requirements.txt` dependencies
- bge embedding model (will be downloaded automatically)

**Note:** this software is in beta. You may get unexpected responses, or your knowledge graph may be incomplete. Finally, a reminder: garbage in, garbage out. Make sure your data is structured and easy to parse into triplets, or else you may have missing segments in your final knowledge graph.
