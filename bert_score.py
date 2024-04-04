
from transformers import CamembertTokenizer, CamembertModel
from transformers import FlaubertTokenizer, FlaubertModel
from transformers import BertTokenizer, BertModel
from scipy.spatial.distance import cosine
import numpy as np 
import torch.nn as nn 
import torch 

bertscorer_models = ['flaubert/flaubert-small-cased','flaubert/flaubert-base-uncased',
                    'flaubert/flaubert-base-cased','flaubert/flaubert-large-cased',
                    'camembert-base','camembert/camembert-large','camembert/camembert-base-ccnet',
                    'camembert/camembert-base-wikipedia-4gb','camembert/camembert-base-oscar-4gb',
                    'camembert/camembert-base-ccnet-4gb','bert-base-uncased','bert-large-uncased',
                    'bert-base-cased','bert-large-cased','bert-base-chinese','bert-base-multilingual-cased',
                    'bert-large-uncased-whole-word-masking','bert-large-cased-whole-word-masking' ] 


class BERTScorer: 
    def __init__(self,model_name,device,layer_id=None,batch_size = 16,use_importance_weighting=False,idf_r=None,idf_c=None,b_r=None,b_p=None,b_f=None):
        '''
            Inputs : 
                -model_name 
                -device 
                -layer_id : idx of the hidden layer from which the embedding will be extracted, if = None, extract from the last hidden layer 
                -batch_size 
                -use_importance_weighting (bool)
                -idf_c List[int]: importance weights for tokens in candidate sequences 
                -idf_r List[int]: importance weights for tokens in reference sequences 
                -b_r : rescaling baseline for recall 
                -b_p : rescaling baseline for precision  
                -b_f : rescaling baseline for f_score  
        '''
        assert model_name not in bertscorer_models, 'Model not supported'
        # FlauBert model 
        if model_name in ['flaubert/flaubert-small-cased','flaubert/flaubert-base-uncased','flaubert/flaubert-base-cased',
                          'flaubert/flaubert_base_uncased','flaubert/flaubert-large-cased']:
            self.tokenizer = FlaubertTokenizer.from_pretrained(model_name)
            self.model = FlaubertModel.from_pretrained(model_name)
        
        # CamemBERT model 
        if model_name in ['camembert-base','camembert/camembert-large','camembert/camembert-base-ccnet','camembert/camembert-base-wikipedia-4gb',
                         'camembert/camembert-base-oscar-4gb','camembert/camembert-base-ccnet-4gb']:
            self.tokenizer = CamembertTokenizer.from_pretrained(model_name)
            self.model = CamembertModel.from_pretrained(model_name)

        # BERT model 
        if model_name in ['bert-base-uncased','bert-large-uncased','bert-base-cased','bert-large-cased','bert-base-chinese',
                            'bert-base-multilingual-cased','bert-large-uncased-whole-word-masking','bert-large-cased-whole-word-masking' ]:
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertModel.from_pretrained(model_name)

        self.layer_id = layer_id 
        if self.layer_id != None :
            assert self.layer_id < self.model.config.num_hidden_layers, 'Layer index must be < number of hidden layers in the model'
        
        self.device = device 
        self.model.to(self.device)

        self.idf_r = idf_r
        self.idf_c = idf_c
        self.batch_size = batch_size
        self.use_importance_weighting = use_importance_weighting
        self.b_r = b_r
        self.b_p = b_p
        self.b_f = b_f


    def __call__(self,candidates,references):
        '''
            Input :
                - candidates : List[str]
                - references : List[str]
            Output: 
                - dict with the keys : 
                    - recall : list of bert recall for each elements in the data 
                    - precision : list of bert precision for each elements in the data 
                    - f_score : list of f_score for each elements in the data 
                    - avg_recall
                    - avg_precision 
                    - avg_f_score 
        '''
        m = len(candidates)
        
        assert references !=m, 'candidates and references must be of the same size'

        recall = []
        precision = []
        f_score = []
        
        for j in range(0, m, self.batch_size):  
            candidate_batch= candidates[j:j+self.batch_size]
            reference_batch= references[j:j+self.batch_size]

            #Tokenization 
            candidate_tokens = self.tokenizer(candidate_batch, return_tensors='pt', padding=True, truncation=True).to(self.device) # size m,max_seq_len_c
            reference_tokens = self.tokenizer(reference_batch, return_tensors='pt', padding=True, truncation=True).to(self.device) # size m,max_seq_len_r
            
            candidate_mask = candidate_tokens['attention_mask'].to(self.device) # size m,max_seq_len_c
            reference_mask = reference_tokens['attention_mask'].to(self.device) # size m,max_seq_len_r

            # Forward through the model to calculat the contextual embedding 
            with torch.no_grad():
                if self.layer_id!=None:
                    candidate_output = self.model(**candidate_tokens, output_hidden_states=True)
                    candidate_embedding = candidate_output.hidden_states[self.layer_id] # size m,max_seq_len_c,embd_size  
                    reference_output = self.model(**reference_tokens, output_hidden_states=True)
                    reference_embedding = reference_output.hidden_states[self.layer_id] # size m,max_seq_len_r,embd_size  
                    
                else:
                    candidate_output = self.model(**candidate_tokens)
                    candidate_embedding = candidate_output.last_hidden_state # size m,max_seq_len_c,embd_size  
                    reference_output = self.model(**reference_tokens)
                    reference_embedding = reference_output.last_hidden_state # size m,max_seq_len_r,embd_size  

            
            for i in range(len(candidate_batch)): 
                # Removing the embedding of pad , <s> and <\s> tokens from the candidates and references 

                candidate_mask[i][candidate_tokens['input_ids'][i] == self.tokenizer.eos_token_id] = 0
                candidate_mask[i][candidate_tokens['input_ids'][i] == self.tokenizer.bos_token_id] = 0
                candidate_mask[i][candidate_tokens['input_ids'][i] == self.tokenizer.sep_token_id] = 0
                reference_mask[i][reference_tokens['input_ids'][i] == self.tokenizer.eos_token_id] = 0
                reference_mask[i][reference_tokens['input_ids'][i] == self.tokenizer.bos_token_id] = 0
                reference_mask[i][reference_tokens['input_ids'][i] == self.tokenizer.sep_token_id] = 0
                
                emb_c = candidate_embedding[i][candidate_mask[i]==1] # remove padding tokens from the exampple
                emb_r = reference_embedding[i][reference_mask[i]==1]
            
                # Pairwise cosine similarity 
                cos_sim = nn.functional.cosine_similarity(emb_r[:, None, :], emb_c[None, :, :], dim=2).to(self.device)
                
                ## Greedy matching
                row_max_elems, _ = torch.max(cos_sim, dim=1)
                col_max_elems, _ = torch.max(cos_sim, dim=0)
            
                row_max_elems = row_max_elems.to(self.device)
                col_max_elems = col_max_elems.to(self.device)
                
        
                
                if not self.use_importance_weighting:
                    r = torch.sum(row_max_elems) / row_max_elems.size(0)
                    p = torch.sum(col_max_elems) / col_max_elems.size(0)
                    f = 2*(r*p) / (r+p)
                else : 
                    # idfs for tokens in the ref/cand 
                    weights_c = self.idf_c[candidate_tokens['input_ids'][i][candidate_mask[i].bool()]]
                    weights_r = self.idf_r[reference_tokens['input_ids'][i][reference_mask[i].bool()]]
                    
                    r = torch.sum(row_max_elems*weights_r) / torch.sum(weights_r)
                    p = torch.sum(col_max_elems*weights_c) / torch.sum(weights_c)
                    f = 2*(r*p) / (r+p)
            
                recall.append(r)
                precision.append(p)
                f_score.append(f)
        
        # Baseline rescaling 
        if self.b_r != None: 
            recall = ((np.array(recall) - self.b_r)/ (1-self.b_r)).tolist()
        if self.b_p != None: 
            precision = ((np.array(precision) - self.b_p)/ (1-self.b_p)).tolist()
        if self.b_f != None: 
            f_score = ((np.array(f_score) - self.b_f)/ (1-self.b_f)).tolist()
        
        return {'avg_recall':sum(recall)/len(recall),
            'avg_precision':sum(precision)/len(precision),
            'avg_f_score':sum(f_score)/len(f_score),      
            'recall':recall,'precision':precision,'f_score':f_score }

