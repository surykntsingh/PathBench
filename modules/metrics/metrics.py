import atexit

from modules.metrics.pycocoevalcap.bleu.bleu import Bleu
from modules.metrics.pycocoevalcap.meteor.meteor import Meteor
from modules.metrics.pycocoevalcap.rouge.rouge import Rouge

_COCO_SCORERS = None


def _get_coco_scorers():
    global _COCO_SCORERS
    if _COCO_SCORERS is None:
        _COCO_SCORERS = [
            (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L")
        ]
    return _COCO_SCORERS


def _close_coco_scorers():
    global _COCO_SCORERS
    if _COCO_SCORERS is None:
        return
    for scorer, _ in _COCO_SCORERS:
        close_fn = getattr(scorer, 'close', None)
        if callable(close_fn):
            close_fn()
    _COCO_SCORERS = None


atexit.register(_close_coco_scorers)



def compute_coco_scores(eval_lists):
    """
    Performs the MS COCO evaluation using the Python 3 implementation (https://github.com/salaniz/pycocoevalcap)

    :param gts: Dictionary with the image ids and their gold captions,
    :param res: Dictionary with the image ids ant their generated captions
    :print: Evaluation score (the mean of the scores of all the instances) for each measure
    """

    gts= {i:[eval_list[1]] for i,eval_list in enumerate(eval_lists)}
    preds = {i: [eval_list[0]] for i, eval_list in enumerate(eval_lists)}

    # print(f'preds: {preds} gts: {gts}')

    scorers = _get_coco_scorers()
    eval_res = {}
    # Compute score for each metric
    for scorer, method in scorers:
        try:
            score, scores = scorer.compute_score(gts, preds)
        except TypeError:
            score, scores = scorer.compute_score(gts, preds)
        if type(method) == list:
            for sc, m in zip(score, method):
                eval_res[m] = sc
        else:
            eval_res[method] = score

    return eval_res


def compute_scores(gts, res):
    """
    Performs the MS COCO evaluation using the Python 3 implementation (https://github.com/salaniz/pycocoevalcap)

    :param gts: Dictionary with the image ids and their gold captions,
    :param res: Dictionary with the image ids ant their generated captions
    :print: Evaluation score (the mean of the scores of all the instances) for each measure
    """

    # Set up scorers
    scorers = _get_coco_scorers()
    eval_res = {}
    # Compute score for each metric
    for scorer, method in scorers:
        try:
            score, scores = scorer.compute_score(gts, res)
        except TypeError:
            score, scores = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, m in zip(score, method):
                eval_res[m] = sc
        else:
            eval_res[method] = score
    return eval_res
