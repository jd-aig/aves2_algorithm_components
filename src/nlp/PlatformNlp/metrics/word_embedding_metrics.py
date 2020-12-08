import logging
from PlatformNlp.metrics import register_metrices
from PlatformNlp.metrics.platform_metrics import PlatformMetrice
from PlatformNlp.tokenization import load_vocab
import json
import numpy as np

logger = logging.getLogger(__name__)


@register_metrices('word_embedding_metrics')
class WordEmbeddingMetrics(PlatformMetrice):
    """
    Sentence (or sentence pair) prediction (classification or regression) task.

    Args:
        dictionary (Dictionary): the dictionary for the input of the task
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('--topk', type=int, default=10, help='topk to show the similarity between words')
        parser.add_argument('--vocab_file', type=str, help='vocab file')

    def __init__(self, args, input_ids, label_ids, predict_scores, label_mapping):
        super().__init__(args, input_ids, label_ids, predict_scores, label_mapping)
        self.args = self.args
        self.input_ids = input_ids
        self.label_ids = label_ids
        self.predict_scores = predict_scores
        self.label_mapping = label_mapping

    def compute_metrices(self):
        json_dict = dict()
        vocab = load_vocab(self.args.vocab_file)
        inv_vocab = {v: k for k, v in vocab.items()}
        for (index, input_id) in enumerate(self.input_ids):
            similarity = self.predict_scores[index]
            similarity = np.array(similarity).reshape(-1, self.args.vocab_size)
            for (i, id_word) in enumerate(input_id):
                id_word = int(id_word)
                similarity_i = list(similarity[i])
                similarity_i_ = [-x for x in similarity_i]
                similarity_i_ = np.array(similarity_i_)
                char_words = inv_vocab.get(id_word, "UNK")
                nearst_id = (similarity_i_).argsort()[1:self.args.topk + 1]
                nearst_words = [inv_vocab.get(id_n, "UNK") for id_n in nearst_id]
                json_dict[char_words] = nearst_words

        json_data = json.dumps(json_dict)
        return json_data


