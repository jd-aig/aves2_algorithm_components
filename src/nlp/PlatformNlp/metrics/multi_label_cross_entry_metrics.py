import logging
from PlatformNlp.metrics import register_metrices
from PlatformNlp.metrics.platform_metrics import PlatformMetrice
from PlatformNlp.utils import calculate_multi_label
import json


logger = logging.getLogger(__name__)


@register_metrices('multi_label')
class MultiLabelMetrics(PlatformMetrice):
    """
    Sentence (or sentence pair) prediction (classification or regression) task.

    Args:
        dictionary (Dictionary): the dictionary for the input of the task
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('--label_file', type=str, help='label_file for mapping json')
        parser.add_argument('--threshold', type=float, help='threshold for classify label')

    def __init__(self, args, input_ids, label_ids, predict_scores, label_mapping):
        super().__init__(args, input_ids, label_ids, predict_scores, label_mapping)
        self.args = self.args
        self.input_ids = input_ids
        self.label_ids = label_ids
        self.predict_scores = predict_scores
        self.label_mapping = label_mapping

    def compute_metrices(self):
        multi_label_pred = []
        for i in range(len(self.predict_scores)):
            decsion_score = self.predict_scores[i]
            multi_label_pred_line = [1 if j > self.args.threshold else 0 for j in decsion_score]
            multi_label_pred.append(multi_label_pred_line)

        exact_accu, less_accu = calculate_multi_label(multi_label_pred, self.label_ids)

        json_dict = dict()
        json_dict["exact_accu"] = exact_accu
        json_dict["less_accu"] = less_accu
        json_data = json.dumps(json_dict)
        return json_data


