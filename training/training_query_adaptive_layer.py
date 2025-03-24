from torch.utils.data import DataLoader


def initial_prints(model):
    print("*" * 40 + f" {model} " + "*" * 40)
    print("Training the model...")
    print()


def _prepare_multi_embeddings_training(doc_embeddings, train_query_embeddings, train_query_embeddings_bert,
                                       train_query_embeddings_tfidf, train_queries, train_qrels, corpus, more_positive):
    """Helper function to prepare multi embeddings dataset and dataloader"""
    dataset = more_positive.MultiEmbeddingsQueryDataset(
        doc_embeddings, train_query_embeddings,
        train_query_embeddings_bert, train_query_embeddings_tfidf,
        list(train_queries.keys()), train_qrels,
        list(corpus.keys()), num_negatives=500, max_positives=5)

    dataloader = DataLoader(
        dataset, batch_size=32, shuffle=True,
        collate_fn=more_positive.multi_custom_collate_fn)
    return dataset, dataloader


def _prepare_single_embeddings_training(doc_embeddings, train_query_embeddings, train_queries,
                                        train_qrels, corpus, more_positive):
    """Helper function to prepare single embeddings dataset and dataloader"""
    dataset = more_positive.QueryDataset(
        doc_embeddings, train_query_embeddings, list(train_queries.keys()),
        train_qrels, list(corpus.keys()), num_negatives=500, max_positives=5
    )
    dataloader = DataLoader(
        dataset, batch_size=32, shuffle=True,
        collate_fn=more_positive.custom_collate_fn
    )
    return dataset, dataloader


def _train_multi_embeddings_model(adaptive_model, dataloader, criterion, optimizer, more_positive):
    """Helper function to train multi embeddings model"""
    more_positive.train_multi_embeddings_query_adaptive_model(
        adaptive_model, dataloader, criterion, optimizer, num_epochs=1
    )


def _train_single_embeddings_model(adaptive_model, dataloader, criterion, optimizer, more_positive):
    """Helper function to train single embeddings model"""
    more_positive.train_query_adaptive_model(
        adaptive_model, dataloader, criterion, optimizer, num_epochs=1)


def _evaluate_multi_embeddings_model(adaptive_model, doc_embeddings, test_query_embeddings,
                                     test_query_embeddings_bert, test_query_embeddings_tfidf,
                                     test_queries, test_qrels, corpus, calculate_map):
    """Helper function to evaluate multi embeddings model"""
    return calculate_map(
        adaptive_model, doc_embeddings, test_query_embeddings,
        test_query_embeddings_bert, test_query_embeddings_tfidf,
        list(test_queries.keys()), test_qrels, list(corpus.keys())
    )


def _evaluate_single_embeddings_model(adaptive_model, doc_embeddings, test_query_embeddings,
                                      test_queries, test_qrels, corpus, calculate_map):
    """Helper function to evaluate single embeddings model"""
    return calculate_map(
        adaptive_model, doc_embeddings, test_query_embeddings,
        list(test_queries.keys()), test_qrels, list(corpus.keys())
    )


def train_adaptive_cde(doc_embeddings, train_query_embeddings, test_query_embeddings,
                       train_queries, test_queries, train_qrels, test_qrels, corpus,
                       num_epochs, adaptive_model, criterion, optimizer, dataset_name,
                       more_positive, calculate_map, save_model, model_name="QACDE",
                       is_multi_embeddings=False, train_query_embeddings_bert=None,
                       train_query_embeddings_tfidf=None, test_query_embeddings_bert=None,
                       test_query_embeddings_tfidf=None):
    model_type = 'Multi Embeddings Query Adaptive CDE' if is_multi_embeddings else 'Query Adaptive CDE'
    initial_prints(model_type)

    # initializations
    best_map_score = 0.0

    # loop over the epochs
    for i in range(num_epochs):
        print(f"Epoch {i + 1}:")

        # Prepare dataset and dataloader based on model type
        if is_multi_embeddings:
            _, dataloader = _prepare_multi_embeddings_training(
                doc_embeddings, train_query_embeddings, train_query_embeddings_bert,
                train_query_embeddings_tfidf, train_queries, train_qrels, corpus, more_positive
            )

            # Train the model
            _train_multi_embeddings_model(adaptive_model, dataloader, criterion, optimizer, more_positive)

            # Evaluate the model
            test_map_score = _evaluate_multi_embeddings_model(
                adaptive_model, doc_embeddings, test_query_embeddings,
                test_query_embeddings_bert, test_query_embeddings_tfidf,
                test_queries, test_qrels, corpus, calculate_map
            )
        else:
            _, dataloader = _prepare_single_embeddings_training(
                doc_embeddings, train_query_embeddings, train_queries,
                train_qrels, corpus, more_positive
            )

            # Train the model
            _train_single_embeddings_model(adaptive_model, dataloader, criterion, optimizer, more_positive)

            # Evaluate the model
            test_map_score = _evaluate_single_embeddings_model(
                adaptive_model, doc_embeddings, test_query_embeddings,
                test_queries, test_qrels, corpus, calculate_map
            )

        # Save the model if the MAP score is better
        if test_map_score >= best_map_score:
            save_model(adaptive_model, model_name, dataset_name)
            best_map_score = test_map_score

        # Print the MAP score
        print(f"Mean Average Precision (MAP) On Test Set: {test_map_score:.4f}\n")

    print('*' * 40)
    print('End of training, best MAP score:', best_map_score)
    print()