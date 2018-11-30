
import math

class Retrieve:
    # Create new Retrieve object storing index and termWeighting scheme
    def __init__(self,index, termWeighting):
        self.index = index
        self.termWeighting = termWeighting
        self.document = {}

        # Change to another form of doctionary: {docid: {term: count, term: count}}
        for term, value in self.index.items():
            for docid, count in value.items():
                if not docid in self.document.keys():
                    self.document[docid] = {}
                self.document[docid][term] = count

    # Method performing retrieval for specified query
    def forQuery(self, query):
        result = {} # Store the result: {docid: similarity}
        result_list = [] # Store the ordered document id

        # Binary term weighting scheme
        if self.termWeighting == 'binary':
            q2 = len(query) # q^2 is equal to the length of each query
            for docid, d_term in self.document.items():
                d2 = len(self.document[docid]) # d^2 is equal to the length of each document
                sim = 0
                qd = 0
                for q_term, count in query.items():
                    # q*d is equal to 1 when the term appearing in query also appears in document
                    if q_term in self.document[docid].keys():
                        qd += 1
                # Calculate the similarity and store it in result dictionary
                sim = qd/(math.sqrt(q2)*math.sqrt(d2))
                result[docid] = sim

        # Frequency of term weighting scheme
        elif self.termWeighting == 'tf':
            q2 = 0
            for term, count in query.items():
                q2 += math.pow(query[term], 2) # q^2
            for docid, value in self.document.items():
                d2 = 0
                qd = 0
                for term, count in value.items():
                    d2 += math.pow(self.document[docid][term], 2) # d^2
                    if term in query.keys():
                        qd += query[term] * count # q*d when the term appears both query and document
                # Calculate the similarity and store it in result dictionary
                sim = qd/(math.sqrt(q2)*math.sqrt(d2))
                result[docid] = sim

        # TF.IDF term weighting scheme
        elif self.termWeighting == 'tfidf':
            D = len(self.document) # collection (set) of documents
            for docid, value in self.document.items():
                d_tfidf = {} # Store the idf and tfidf of this term: {term: [idf, tfidf]}
                q2 = 0
                d2 = 0
                qd = 0
                idf = 0
                for term, count in value.items():
                    tf = self.document[docid][term] # Number of times the term occurs in document
                    df = len(self.index[term]) # Number of documents containing this term
                    idf = math.log(D/df)
                    tfidf = tf * idf
                    d_tfidf[term] = [idf, tfidf]
                    d2 += math.pow(tfidf, 2) # d^2
                for term, count in query.items():
                    tf = query[term]
                    if term in d_tfidf.keys():
                        idf = d_tfidf[term][0] # idf is equal to it which already got above
                    else:
                        if term in self.index.keys():
                            df = len(self.index[term])
                            idf = math.log(D/df)
                        else: # The term appearing in query doesn't appear in any documents
                            idf = 0 # idf is equal to 0
                    tfidf = tf * idf
                    q2 += math.pow(tfidf, 2) # q^2
                    if term in d_tfidf.keys():
                        qd += tfidf * d_tfidf[term][1]
                # Calculate the similarity and store it in result dictionary
                sim = qd/(math.sqrt(q2)*math.sqrt(d2))
                result[docid] = sim
        # Sort the result dictionary through the value in descending order
        sorted_result = sorted(result.items(), key=lambda item:item[1], reverse=True)
        # Only store the docid in result list from sorted_result tuple
        for tuple_result in sorted_result:
            result_list.append(tuple_result[0])

        return result_list
