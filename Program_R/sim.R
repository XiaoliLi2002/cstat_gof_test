
Cstat_func <- function(svec, Nvec){
  sum(svec - Nvec * log(svec) - Nvec + Nvec * log(Nvec+1e-10))
}

thetastar = 1
sitheta <- function(theta, n) {sapply(1:n, function(i) {exp(-1-i*theta/n)*2})}
# mu=2/e, exp model

n <- 1000

onesim <- function(thetastar, n){
  obs <- rpois(n, sitheta(thetastar, n))
  mle <- optim(1, function(theta) Cstat_func(sitheta(theta, n), obs), method = "Brent", lower = 0, upper = 4)$par
  return(c(mle, Cstat_func(thetastar, obs)-Cstat_func(mle, obs)))
}


result <- sapply(1:1000, function(k) onesim(thetastar, n))
plot(result[1,], jitter(result[2,]))
#plot(result[1,], jitter(result[2,]), xlim = c(0, 5), ylim = c(0, 4*n))


#resultCmarginal=c(1:1000)
#for (i in 1:1000) {
#  obs <- rpois(n, sitheta(thetastar, n))
#  resultCmarginal[i]=Cstat_func(sitheta(thetastar, n), obs)
#}
#hist(resultCmarginal)
