# Generated by using Rcpp::compileAttributes() -> do not edit by hand
# Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

pga <- function(phi, resp, penalty, zeta, c, lambda, nlambda, makelamb, lambdaminratio, penaltyfactor, reltol, maxiter, steps, btmax, mem, tau, nu, alg, array, ll, Lmin, nthreads) {
    .Call(`_SMME_pga`, phi, resp, penalty, zeta, c, lambda, nlambda, makelamb, lambdaminratio, penaltyfactor, reltol, maxiter, steps, btmax, mem, tau, nu, alg, array, ll, Lmin, nthreads)
}

