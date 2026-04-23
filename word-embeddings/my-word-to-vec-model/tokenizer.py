
class Tokenizer: 
    
    def tokens(self, corpus):
        return [sentence.split() for sentence in corpus]