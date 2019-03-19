def compute_profitability(A, S, y_pred):
    """

    Computes a buy_or_sell trading stategy based on the predictions.

    Arguments:

        y_pred -- predictions for the relative performance.
        S -- real SPY price
        A -- real Apple price

    Returns:

        score - the profitability

    """

    length_y = len(y_true)
    strategy_earnings = 0
    tran_cost_rate = 0.0001

    for i in range(length_y - 1):
        if y_pred[i + 1] > 0:
            difference = A[i+1] - A[i] + S[i] - S[i + 1]
        elif y_pred[i + 1] < 0:
            difference = -(A[i+1] - A[i] + S[i] - S[i + 1])

        strategy_earnings += (difference + tran_cost_rate * (A[i+1] + S[i+1] + A[i] + S[i])) / (A[i] + S[i])

    strategy_earnings = 100 * strategy_earnings

    return strategy_earnings