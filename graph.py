import torch

from pathlib import Path

import logging
import sys

from llama_index.core import (
    load_index_from_storage,
    SimpleDirectoryReader,
)

from llama_index.core import PromptTemplate
from llama_index.core import KnowledgeGraphIndex, Settings, StorageContext
from llama_index.core.graph_stores import SimpleGraphStore

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.postprocessor.colbert_rerank import ColbertRerank

from pyvis.network import Network


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

documents = SimpleDirectoryReader("./data").load_data()

print("=> loading llm")
Settings.llm = Ollama(
    model="dolphin-mistral",
    request_timeout=30.0,
    temperature=0.2,
)
print("=> setting up embeddings")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.chunk_size = 512
found_graph = (Path.cwd() / 'persist').exists()
if found_graph:
    storage_context = StorageContext.from_defaults(
        persist_dir="./persist",
    )
    index = load_index_from_storage(storage_context)
else:
    graph_store = SimpleGraphStore()
    storage_context = StorageContext.from_defaults(graph_store=graph_store)
    print("=> generating graph...")
    extract_tmpl_str =  (
        "Given the text, extract as many " # "{max_knowledge_triplets} "
        "concise triplets in the form of (subject, predicate, object) as you can. Avoid stopwords. Remove prefixes like \"The\".\n"
        "Preserve the capitalization of the source text.\n"
        "The triplets you output will be used to build a knowledge graph, so ensure that the subject is referred to in a consistent manner.\n"
        "---------------------\n"
        "Example:"
        "Text: Alice is Bob's mother.\n"
        "Triplets:\n(Alice, is mother of, Bob)\n\n"
        "Example:\n"
        "Text: The Lightning Network's transactions can be settled instantly.\n"
        "Triplets:\n(Lightning Network's transactions, can be settled instantly)\n"
        "---------------------\n"
        "Remember, only output triplets.\n"
        "Text: {text}\n"
        "Triplets:\n"
    )
    extract_tmpl = PromptTemplate(extract_tmpl_str)
    index = KnowledgeGraphIndex.from_documents(
        documents,
        max_triplets_per_chunk=10,
        storage_context=storage_context,
        include_embeddings=True,
        kg_triple_extract_template=extract_tmpl,
    )
    print("=> persisting graph...")
    storage_context.persist(persist_dir="./persist")
    print("=> generating graph viz")
    g = index.get_networkx_graph()
    net = Network(notebook=True, cdn_resources="remote", directed=True)
    net.from_nx(g)
    net.save_graph("networkx-pyvis.html")
print("=> loading colbert")
colbert_reranker = ColbertRerank(
    top_n=5,
    model="colbert-ir/colbertv2.0",
    tokenizer="colbert-ir/colbertv2.0",
    keep_retrieval_score=True,
)
new_summary_tmpl_str = (
    "Context:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "think carefully and then answer the query. Do not mention the context in your answer.\n"
    "Also, you do not need to justify your answer.\n"
    "Query: {query_str}\n"
    "Answer: "
)
query_engine = index.as_query_engine(
    temperature=0.7,
    # include_text=False,
    response_mode="tree_summarize",
    streaming=True,
    similarity_top_k=6,
    node_postprocessors=[colbert_reranker],
    # explore_global_knowledge=True,
)
new_summary_tmpl = PromptTemplate(new_summary_tmpl_str)
query_engine.update_prompts(
    {"response_synthesizer:summary_template": new_summary_tmpl}
)
# prompts_dict = query_engine.get_prompts()
# print(prompts_dict)
print("=> ready")
while True:
    user_query = input("Enter query: ")
    response = query_engine.query(
        user_query,
    )
    print(response)
    print()
