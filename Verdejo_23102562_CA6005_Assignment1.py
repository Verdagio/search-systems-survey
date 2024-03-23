# DANIEL VEDEJO - 23102562

import subprocess
import re
import torch
import re
import nltk
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



nltk.download('stopwords')
STOPWORDS = set(nltk.corpus.stopwords.words('english'))

def clean_text(text):
    """
    A function to clean the input text by removing non-alphanumeric characters, removing stopwords, converting it to lowercase.
    
    Parameters:
    text (str): The input text to be cleaned.
    
    Returns:
    str: The cleaned text.
    """

    text = re.sub(r'[^\w\s]', '', text.lower()) if text else str(text).lower()
    return " ".join(word for word in text.split() if word not in STOPWORDS)


def use_gpu():
    """
    Check if GPU is available and return the appropriate device type.
    """
    return 'cuda' if torch.cuda.is_available() else 'cpu'

class InfoRetrieval:
    def __init__(self, documents, top_n=25):
        """
        Initialize the class with the given documents.

        Parameters: 
            documents (list): A list of documents to initialize the class with.

        Returns:
            None
        """
        self.documents = self.ingest_documents(documents)
        self.index = self.__index_documents()
        self.vectorizer = TfidfVectorizer()
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModel.from_pretrained("bert-base-uncased").to(use_gpu())
        self.top_n = top_n

    def __index_documents(self):
        """
        Index the given documents into an inverted index.

        Returns:
            dict: An inverted index of the documents with the word as the key and a list of documents that contain that word as the value.
        """
        index = {}
        for document in self.documents:
            if document['full_text'] is not None:
                for word in document['full_text'].split():
                    if word not in index:
                        index[word] = []
                    index[word].append(document)
        return index
    
    def ingest_documents(self, documents):

        docs = []
        for document in documents:
            if all(key in document and document[key] is not None for key in ['title', 'author', 'text']):
                docs.append({**document, 'full_text':  clean_text("".join([document.get('title', ''), document.get('author', ''), document.get('text', '')])) })
        return docs
    
    def get_relevant_docs(self, query_text):
        """
        Retrieves a list of relevant documents based on the given query text.

        Parameters:
            query_text (str): The text query to search for relevant documents.

        Returns:
            list: A list of relevant documents matching the query text.
        """
        relevant_docs = []
        for word in query_text.split():
            for doc in self.index.get(word, []):
                if doc not in relevant_docs:
                    relevant_docs.append(doc)
        return relevant_docs
    
    def top_n_vals(self, ls):
        return ls[:self.top_n]

    def bm25_search(self, queries):
        """
        Perform a search using the BM25 model using the given query and return a list of unique results.
        
        Parameters:
            query (str): The search query to be used.
            
        Returns:
            list: A list of unique search results.
        """
        results = []
        
        for run, query in enumerate(queries):
            query_id, query_terms = query['num'], clean_text(query['title'])
            run_id = run
            documents = self.get_relevant_docs(query_terms)
            num_docs = len(documents)
            avg_doc_len = sum(len(doc['full_text'].split()) for doc in documents) / num_docs
            k1 = 1.2 # k1 & b are hyperparameters that can be tuned for different search results 
            b = 0.75
            scores = []
            for doc in documents:
                doc_len = len(doc['full_text'].split())
                score = 0
                for word in query_terms.split():
                    df = len(self.index.get(word, [])) # calculate the number of documents that contain the words
                    tf = doc['full_text'].split().count(word) # calculate the number of times the word appears in the document (term frequency)
                    idf = math.log((num_docs - df + 0.5) / (df + 0.5) + 1) # calculate the inverse document frequencw
                    score += idf * (tf * (k1 + 1)) / (tf + 1.5 * (1 - b + b * (doc_len / avg_doc_len)))
                    
                scores.append(score)

            sub_ls = []
            for i, (doc, score) in enumerate(zip(documents, scores), start=1):
                sub_ls.append((query_id,  i, doc['docno'], i, round(score, 4), run_id))
            
            results.extend(self.top_n_vals(sub_ls))
        
        return results


    def vsm_search(self, queries):
        """
        Perform a search using the vector space model using the given query and return a list of unique results.
        
        Parameters:
            query (str): The search query to be used.
            
        Returns:
            list: A list of unique search results.
        """
        results = []
        for run_id, query in enumerate(queries):
            query_id, query_text = query['num'], clean_text(query['title'])
            
            doc_vect = self.vectorizer.fit_transform([doc['full_text'] for doc in self.get_relevant_docs(query_text)])
            query_vect = self.vectorizer.transform([query_text])
            similarity = cosine_similarity(query_vect, doc_vect)
            
            top_indices = similarity.argsort()[0]
            sub_ls = []
            for i, index in enumerate(top_indices, start=1):
                document_id = self.documents[index]['docno']
                sim = similarity[0, index]
                sub_ls.append((query_id,  i, document_id,  i, round(sim, 4), run_id))
                
            results.extend(self.top_n_vals(sub_ls))
        return results            
 
    def _compute_document_embeddings(self):
        """
        Compute the document embeddings for the given documents.

        Parameters:
            documents (list): A list of documents to compute the document embeddings for.

        Returns:
            list: A list of document embeddings.
        """
        document_embeddings = []
        for doc in self.documents:
            if doc['text'] is not None:
                inputs = self.tokenizer(doc['full_text'], return_tensors="pt", padding=True, truncation=True).to(use_gpu())
                with torch.no_grad():
                    outputs = self.model(**inputs)
                # Use the last hidden state for each token as document embedding
                last_hidden_state = outputs.last_hidden_state
                doc_embedding = torch.mean(last_hidden_state, dim=1)  # Perform mean pooling
                document_embeddings.append(doc_embedding)
        return document_embeddings

    def lang_model_retrieval(self, queries):
        """
        Generates embeddings for the given queries and computes similarity between query embeddings and document embeddings.
        
        Parameters:
            queries (list): A list of dictionaries representing queries. Each dictionary contains a 'num' key for the query ID and a 'title' key for the query text.
            top_n (int, optional): The number of top similar documents to retrieve for each query. Defaults to 25.
        
        Returns:
            list: A list of tuples representing the results. Each tuple contains the query ID, rank, document ID, similarity score, and the run number.
        """
        results = []
        doc_embeddings = self._compute_document_embeddings()
        for run, query in enumerate(queries):
            
            query_id, query_text = query['num'], clean_text(query['title'])

            inputs = self.tokenizer(query_text, return_tensors="pt", padding=True, truncation=True).to(use_gpu()) # force using the gpu to speed stuff up if its available
            with torch.no_grad():
                outputs = self.model(**inputs)

            last_hidden_state_query = outputs.last_hidden_state
            query_embedding = torch.mean(last_hidden_state_query, dim=1)

            document_similarities = np.array([torch.dot(query_embedding.squeeze(), doc_emb.squeeze()).item() for doc_emb in doc_embeddings])
            sorted_document_indices = np.argsort(document_similarities)[::-1]
            sub_res = []
            for i, index in enumerate(sorted_document_indices, start=1):
                document_id = self.documents[index]['docno']
                sim = document_similarities[index]
                sub_res.append((query_id, i, document_id, i, round(sim, 4), run))
            results.extend(self.top_n_vals(sub_res))
        return results

    
    def search(self, queries, method):
        if method == 'bm25':
            results = self.bm25_search(queries)
        elif method == 'vsm':
            results = self.vsm_search(queries)
        elif method == 'lm':
            results = self.lang_model_retrieval(queries)

        columns = ['query_id', 'iter', 'document_id', 'rank', 'similarity', 'run_id']
        results = pd.DataFrame(results, columns=columns)
        return results

if __name__ == "__main__":
    queries = pd.read_xml('./data/cran.qry.xml')
    queries = queries.to_dict('records')
    df = pd.read_xml('./data/cran.all.1400.xml')
    documents = df.to_dict('records')

        
    print(f'Using {use_gpu()}')
    top_n = int(input('Enter top n (default=1400): ') or len(documents))

    ir = InfoRetrieval(documents, top_n)
    
    for method in ['lm', 'bm25', 'vsm']:
        res = ir.search(queries, method)
        res.to_csv('results/' + method + '_res.txt', index=False, header=False, sep=' ')
        with open(f'judge/{method}_judge.txt', 'w') as f:
            subprocess.run(['trec_eval', 'data/cranqrel.trec.txt', f'results/{method}_res.txt'], stdout=f, shell=True)

    
    # methods = ['Language Model', 'BM25', 'Vector Space Model']
    # map_values = [0.0028, 0.0030, 0.0024]  # Mean Average Precision
    # Rprec_values = [0.0054, 0.0062, 0.0053]  # R-precision
    # P_5_values = [0.0066, 0.0013, 0.0053]  # Precision at 5 documents
    # P_1000_values = [0.0003, 0.0005, 0.0003]  # Precision at 1000 documents

    # # Bar width 
    # barWidth = 0.2

    # # Set position of bar on X axis
    # r1 = range(len(methods))
    # r2 = [x + barWidth for x in r1]
    # r3 = [x + barWidth for x in r2]
    # r4 = [x + barWidth for x in r3]

    # # Make the plot
    # plt.figure(figsize=(10, 8))

    # plt.bar(r1, map_values, color='b', width=barWidth, edgecolor='grey', label='MAP')
    # plt.bar(r2, Rprec_values, color='r', width=barWidth, edgecolor='grey', label='Rprec')
    # plt.bar(r3, P_5_values, color='g', width=barWidth, edgecolor='grey', label='P@5')
    # plt.bar(r4, P_1000_values, color='y', width=barWidth, edgecolor='grey', label='P@1000')

    # # Add xticks on the middle of the group bars
    # plt.xlabel('Method', fontweight='bold', fontsize=15)
    # plt.xticks([r + barWidth for r in range(len(methods))], methods, rotation=45)
    # plt.ylabel('Score', fontweight='bold', fontsize=15)
    # plt.title('Comparison of Retrieval Methods', fontweight='bold', fontsize=16)

    # # Create legend & Show graphic
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    print('Done')
