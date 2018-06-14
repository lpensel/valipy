"""
Contains the base class for evaluations.
"""
from .basicevaluation import BasicEvaluation


class RankEvaluation(BasicEvaluation):
    """
    Class used for the evaluation of Grid-Search results based on a ranking 
    approach.
    """

    def eval(self, results, scoring, no_splits, scoring_weight=None):
        """
        TODO
        """
        if scoring is None:
            no_scoring = 1
        else:
            no_scoring = len(scoring)

        params = results["params"]

        if scoring_weight is None or len(scoring_weight) != no_scoring:
            scoring_weight = [1.0 for i in range(no_scoring)]

        scoring_names = ["score"]

        if no_scoring > 1:
            scoring_names = [i for i in scoring.keys()]


        param_scores = [0.0 for i in range(len(params))]

        for split in range(no_splits):
            for no in range(no_scoring):
                key = "split{}_test_{}".format(split,scoring_names[no])
                values = results[key]
                for p in range(len(params)):
                    target = values[p]
                    points = len(params)
                    for v in values:
                        if v > target:
                            points -= 1
                    points *= scoring_weight[no]
                    param_scores[p] += points

        return_value = [(str(params[i]),param_scores[i]) 
                        for i in range(len(params))]

        return_value.sort(key=lambda para: para[1], reverse=True)

        return return_value, param_scores.index(max(param_scores))