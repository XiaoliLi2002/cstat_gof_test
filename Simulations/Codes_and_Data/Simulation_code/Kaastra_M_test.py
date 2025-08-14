from utilities import math, p_value_norm
def Max_mean_unit(mu):
    A=-0.56709
    B= -2.7336
    C= -2.3603
    D= 0.52816
    E= 0.33133
    F=1.0174
    alpha=3.9375
    beta= 0.48446
    return (A+B*mu+C*(mu-D)**2)*math.e**(-alpha*mu)+E*math.e**(-beta*mu)+F

def Max_mean(s):
    mean=0
    for x in s:
        mean+=Max_mean_unit(x)
    return mean

def Max_var_unit(mu):
    A=-3.1971
    B= 1.5118
    C= -1.5118
    D= 0.79384
    E= 1.9294
    F=6.1740
    G= 22.360/1000
    H= -7.2981
    I= 2.08378
    alpha= 0.750315
    beta=4.49654
    return (A+B*mu**2+C*(mu-D)**2)*math.e**(-alpha*mu)+(E+F*mu+G*(mu-H)**2)*math.e**(-beta*mu)+I

def Max_var(s):
    var=0
    for x in s:
        var+=Max_var_unit(x)
    return var

def KMtest(x,s):  #Alg.2a
    return p_value_norm(x,Max_mean(s),math.sqrt(Max_var(s)))
