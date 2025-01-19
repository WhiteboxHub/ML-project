


import re
class retrival_evalution():
    def __init__(self):
        
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
    