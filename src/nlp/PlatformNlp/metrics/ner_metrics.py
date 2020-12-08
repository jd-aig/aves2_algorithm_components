from PlatformNlp.metrics import register_metrices
from PlatformNlp.metrics.platform_metrics import PlatformMetrice
from PlatformNlp.tokenization import load_vocab
from PlatformNlp import conlleval
import logging
import json
import os

logger = logging.getLogger(__name__)


@register_metrices('ner_metrics')
class NerMetrics(PlatformMetrice):
    """
    Sentence (or sentence pair) prediction (classification or regression) task.

    Args:
        dictionary (Dictionary): the dictionary for the input of the task
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('--num_classes', type=int, default=11, help='ner classes')

    def __init__(self, args, input_ids, label_ids, predict_scores, label_mapping):
        super().__init__(args, input_ids, label_ids, predict_scores, label_mapping)
        self.args = self.args
        self.input_ids = input_ids
        self.label_ids = label_ids
        self.predict_scores = predict_scores
        self.label_mapping = label_mapping

    def compute_metrices(self):
        predict_ids = self.predict_scores
        label_ids = self.label_ids
        vocab = load_vocab(self.args.vocab_file)
        output_predict_file = os.path.join(self.args.output_dir, "ner_eval.txt")
        inv_vocab = {v: k for k, v in vocab.items()}
        inv_label = {v: k for k, v in self.label_mapping.items()}
        for (index, input_id) in enumerate(self.input_ids):
            line = ""
            label_id = list(label_ids[index])
            pred_id = list(predict_ids[index])
            for (i, id) in enumerate(input_id):
                char_words = inv_vocab.get(id, "UNK")
                label = inv_label.get(label_id[i])
                pred = inv_label.get(pred_id[i])
                if pred is None:
                    pred = "O"
                line += char_words + ' ' + label + ' ' + pred + '\n'
            line += "\n"

            with open(output_predict_file, 'a', encoding='utf-8') as writer:
                writer.write(line)
                writer.flush()

        eval_result = conlleval.return_report(output_predict_file)
        json_data = json.dumps(eval_result)
        return json_data

