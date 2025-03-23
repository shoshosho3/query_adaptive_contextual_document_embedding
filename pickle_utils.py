from consts import *
import pickle

def get_queries(args, stage="train", embedding_type=NO_TYPE):
    with open(f'{args.dataset}/query_embeddings_{stage}_{args.dataset}_{embedding_type}{args.index}.pkl', 'rb') as f:
        train_query_embeddings = pickle.load(f)

    return train_query_embeddings


def open_pickles(args, ):
    with open(f'{args.dataset}/doc_embeddings_{args.dataset}_{args.index}.pkl', 'rb') as f:
        doc_embeddings = pickle.load(f)

    train_query_embeddings = get_queries(args, stage="train")

    train_query_embeddings_bert = get_queries(args, stage="train", embedding_type=BERT)

    train_query_embeddings_tfidf = get_queries(args, stage="train", embedding_type=TFIDF)

    test_query_embeddings = get_queries(args, stage="test")

    test_query_embeddings_bert = get_queries(args, stage="test", embedding_type=BERT)

    test_query_embeddings_tfidf = get_queries(args, stage="test", embedding_type=TFIDF)

    return doc_embeddings, train_query_embeddings, train_query_embeddings_bert, train_query_embeddings_tfidf, \
        test_query_embeddings, test_query_embeddings_bert, test_query_embeddings_tfidf