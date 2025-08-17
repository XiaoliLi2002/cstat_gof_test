from Simulations.utilities.utilities import chi2

def p_value_chi(x,df,alpha=0.1):
    """
    Chi-squared test
    Compute the one-sided and two-sided p-values of a chi-squared test

    Args:
        x: data
        df: degree of freedom
        alpha: significance level (=0.1)

    Returns:
        chi2.sf(x,df): One-sided p-value
        2*min(chi2.sf(x,df),1-chi2.sf(x,df)): two-sided p-value

        chi2.isf(alpha,df): one-sided critical value
        chi2.isf(alpha/2,df)-chi2.isf(1-alpha/2,df): two-sided width
    """
    return chi2.sf(x,df), 2*min(chi2.sf(x,df),1-chi2.sf(x,df)), chi2.isf(alpha,df), chi2.isf(alpha/2,df)-chi2.isf(1-alpha/2,df)