tfseqestimator
==============
Sequence estimators for TensorFlow


Available estimators
--------------------
1. FullSequenceClassifier: one class for a whole sequence
2. FullSequenceRegressor: one value for a whole sequence
3. SequenceItemsClassifier: one class for each sequence item
4. SequenceItemsRegressor: one value for each sequence item


Usage
-----

.. code:: python
    from tfseqestimator import FullSequenceClassifier, RnnType

    token_sequence = sequence_categorical_column_with_hash_bucket(...)
    token_emb = embedding_column(categorical_column=token_sequence, ...)

    estimator = FullSequenceClassifier(
        sequence_feature_columns=[token_emb],
        rnn_type=RnnType.REGULAR_STACKED_LSTM, rnn_layers=[32, 16])

    # Input builders
    def input_fn_train: # returns x, y
      pass
    estimator.train(input_fn=input_fn_train, steps=100)

    def input_fn_eval: # returns x, y
      pass
    metrics = estimator.evaluate(input_fn=input_fn_eval, steps=10)

    def input_fn_predict: # returns x, None
      pass
    predictions = estimator.predict(input_fn=input_fn_predict)
