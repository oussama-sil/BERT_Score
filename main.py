from bert_score import BERTScorer
import torch



if __name__=='__main__':


    device = "cuda" if torch.cuda.is_available() else "cpu"

    bert_scorer = BERTScorer(
        model_name='flaubert/flaubert_base_uncased' ,
        device=device,
        layer_id=-1,
        batch_size = 16,
        use_importance_weighting=False,
        idf_r=None,
        idf_c=None,
        b_r=None,
        b_p=None,
        b_f=None)
    

    candidates = ['Testing BERTScorer']
    references = ['BERTScorer Testing']

    out = bert_scorer(candidates,references)

    print(f"BERT scores : P={out['precision']}, R ={out['recall']}, F1 score={out['f_score']}")
