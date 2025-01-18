
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# # Load the pre-trained model
# model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
import re
class retrival_evalution():
    def __init__(self):
        # self.precision_at_k(actual_response,expected_response=expected)
        print('start') 

    def precision_at_k(self,actual_response,expected_response,k=2):

        actual_response_page_content = []
        expected_response_page_content = []

        for doc in expected_response:
            page_content = re.search(r"page_content='(.*?)'", doc)
            if page_content:
                expected_response_page_content.append(page_content)
        

        for doc in actual_response:
            actual_response_page_content.append(doc.page_content)
        
        relevent_k = 0
        k = 0
        for i in actual_response_page_content:
            if i in expected_response_page_content:
                relevent_k += 1
            k +=1 
        
        return relevent_k, k

        

            

        return precision




