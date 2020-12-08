import logging
from PlatformNlp.metrics import register_metrices
from PlatformNlp.metrics.platform_metrics import PlatformMetrice
from PlatformNlp.utils import calculate_label
import json


logger = logging.getLogger(__name__)


@register_metrices('multi_class')
class MultiClassMetrics(PlatformMetrice):
    """
    Sentence (or sentence pair) prediction (classification or regression) task.

    Args:
        dictionary (Dictionary): the dictionary for the input of the task
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('--label_file', type=str, help='label_file for mapping json')

    def __init__(self, args, input_ids, label_ids, predict_scores, label_mapping):
        super().__init__(args, input_ids, label_ids, predict_scores, label_mapping)
        self.args = args
        self.input_ids = input_ids
        self.label_ids = label_ids
        self.predict_scores = predict_scores
        self.label_mapping = label_mapping

    def compute_metrices(self):
        predict_labels = []
        for score in self.predict_scores:
            predict_labels.append(list(score).index(max(list(score))))
        precision, recall, accuracy, f1_score, auc_micro, auc_macro, classify_report = calculate_label(predict_labels,
                                                                                                       self.label_ids,
                                                                                                       list(self.label_mapping.values()))
        classification_report_dict = {}
        classify_report = str(classify_report).split('\n')
        for i in range(len(classify_report)):
            x = classify_report[i]
            x = str(x).split(' ')
            xx = []
            for j in x:
                try:
                    assert len(j) > 0
                    xx.append(j)
                except:
                    continue
            if len(xx) == 4:
                classification_report_dict['evaluation_index'] = xx
            elif len(xx) == 7:
                classification_report_dict['avg_all'] = xx[3:]
            elif len(xx) > 0:
                classification_report_dict[xx[0]] = xx[1:]
        json_dict = dict()
        json_dict["precision"] = precision
        json_dict["recall_score"] = recall
        json_dict["accuracy"] = accuracy
        json_dict["f1_score"] = f1_score
        json_dict["auc_micro"] = auc_micro
        json_dict["auc_macro"] = auc_macro
        json_dict["classification_report"] = classification_report_dict
        json_data = json.dumps(json_dict)
        return json_data


