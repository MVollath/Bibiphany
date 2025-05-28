from typing import Callable

import chromadb
from chromadb.utils import embedding_functions
import ulid

import re
import unicodedata


def generate_paper_id(first_author: str, title: str, year: int, max_length: int = 63) -> str:
    if not first_author or not isinstance(first_author, str):
        raise ValueError("Authors must be a non-empty list of strings.")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("Title must be a non-empty string.")
    if not isinstance(year, int) or year < 1000 or year > 2200:  # Basic year validation, futureproof lol
        raise ValueError("Year must be a valid integer.")

    first_author_name = first_author.strip()
    last_name_parts = first_author_name.split(' ')
    first_author_last_name = last_name_parts[-1] if last_name_parts else ""
    clean_last_name = re.sub(r'[^a-zA-Z0-9]', '', first_author_last_name).lower()

    # Calculate max_length for title size
    title_size = max_length - 4 - len(clean_last_name)

    normalized_title = unicodedata.normalize('NFKD', title).encode('ascii', 'ignore').decode('utf-8')
    sanitized_title = re.sub(r'[^a-zA-Z0-9\s]', '', normalized_title)
    sanitized_title = "_".join(sanitized_title.lower().strip().split())
    title_snippet = sanitized_title[:title_size]

    if title_snippet.endswith('_') and len(title_snippet) > 1:
        title_snippet = title_snippet.rstrip('_')

    final_id = f"{clean_last_name}_{title_snippet}_{year}"

    return final_id


class Database:
    _client: chromadb.PersistentClient
    _embedding_function: Callable
    _collection: str

    def __init__(self, persistence_path: str = "database/storage", collection: str = "bibiphany",
                 embedding_function: Callable = None):
        self._client = chromadb.PersistentClient(path=persistence_path)
        self._embedding_function = embedding_function or embedding_functions.DefaultEmbeddingFunction()

    def store(self, to_store: list[str], document_title: str, document_year: int, document_author: str,
              meta_information: list[dict] = None):
        collection_title = generate_paper_id(document_author, document_title, document_year)
        collie = self._client.get_or_create_collection(collection_title, embedding_function=self._embedding_function)
        ids = [ulid.new().str for document in to_store]
        if meta_information and len(meta_information) != len(to_store):
            raise ValueError("If meta information is passed, all entries require meta information!")
        collie.add(ids=ids, documents=to_store, metadatas=meta_information)

    def delete(self, document_title: str, document_year: int, document_author: str):
        self._client.delete_collection(generate_paper_id(document_author, document_title, document_year))

    def add_to_result(self, query_results: dict, index: int, collection_name: str, result_structure: list):
        result_structure.append({
            "document": query_results["documents"][0][index],
            "meta": query_results["metadatas"][0][index],
            "distance": query_results["distances"][0][index],
            "id": query_results["ids"][0][index],
            "collection": collection_name
        })
        result_structure = sorted(result_structure, key=lambda x: x["distance"])
        return min(result_structure, key=lambda x: x["distance"]), result_structure

    def include_query_result(self, query_results: dict, collection_name: str, result_structure: list, max_size: int):
        min_sim = min(result_structure, key=lambda x: x["distance"]) if len(result_structure) > 0 else None
        for i, result_dist in enumerate(query_results["distances"][0]):
            if len(result_structure) < max_size or result_dist > min_sim:
                min_sim, result_structure = self.add_to_result(query_results, i, collection_name, result_structure)
                if len(result_structure) > max_size:
                    result_structure = result_structure[:-1]
        return result_structure

    def retrieve(self, query: str):
        embedded_query = self._embedding_function([query])
        results = []
        for collection in self._client.list_collections():
            query_results = collection.query(query_embeddings=embedded_query, n_results=10)
            results = self.include_query_result(query_results, collection.name, results, 10)
        return results
