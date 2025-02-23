from utilities import chi2

def p_value_chi(x,df): #Alg.1
    return chi2.sf(x,df), 2*min(chi2.sf(x,df),1-chi2.sf(x,df))