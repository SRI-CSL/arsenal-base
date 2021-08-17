Some evaluation of different models:

With the new dataset (2021-07-30T0004):
- 07-30-2021: trained one epoch on full trainig set
- 08-04-2021: trained one epoch on first 50% of trainig set
- 08-05-2021: trained two epochs on first 50% of training set

For reference, 05-12-2021 was trained on the previous dataset (2021-04-14T1922)
08-13-2021 is another model trained on the previous dataset with two different evaluations:
- eval_08-12-2021_ngram_restrictions.png shows model performance with `no_repeat_ngram_size=3` instances, i.e., the generation procedure prevented any instances where subsequents of length >=3 were repeated (this setting was also used for earlier evaluations)
- eval_08-12-2021_NO_ngram_restrictions.png shows model performance with `no_repeat_ngram_size=0` instances. This corresponds to the same setting used for newer model evaluations (>=07-30-2021) and shows significantly better performance even for the older models.



Results:
- confusions_[model_id] give an overview about incorrectly generated tokens (this only records the first token mismatch in each sequence, subsequent differences are ignored)
- eval_[model_id] show plots to compare some statistics for true and predicted "sentences" (i.e., CSTs in polish notation)
- lm_scores_[dataset]_[model] show scores for "real sentences" (collected from the actual standards documents and entity-processed), based on models trained on the respective dataset and the given model architecture (bert or gpt-2).
- train_loss shows plots of the training progress for the three different training attemps with the new dataset.