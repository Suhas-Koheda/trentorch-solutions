def roc_threshold_audit(
    probs: list[float], labels: list[int], tau: float
) -> tuple[int, int, int, int, float, float]:
    """
    probs: predicted probabilities.
    labels: binary true labels (0 or 1), same length as probs.
    tau: decision threshold. Predict positive iff p_i >= tau.

    Returns (TP, FP, TN, FN, precision, recall).
    precision = TP / (TP + FP), or 0 if that denominator is 0.
    recall = TP / (TP + FN), or 0 if that denominator is 0.
    """
    # TODO: tally the confusion matrix, then compute precision and recall.
    results=[1 if prob>=tau else 0 for prob in probs]
    tp=sum(p==1 and l==1 for p,l in zip(results,labels))
    fp=sum(p==1 and l==0 for p,l in zip(results,labels))
    tn=sum(p==0 and l==0 for p,l in zip(results,labels))
    fn=sum(p==0 and l==1 for p,l in zip(results,labels))
    return (tp,fp,tn,fn,(tp/(tp+fp)) if (tp+fp) != 0 else 0,tp/(tp+fn) if (tp+fn)!= 0 else 0)
