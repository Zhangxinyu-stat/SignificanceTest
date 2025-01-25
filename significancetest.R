
library(MASS)
library(xgboost)
library(splines)


# Testing procedure for H0:E(Y|X,Z) = E(Y|X).
Significancetest <- function(x, z, w, y, L){
  D <- nrow(x)
  p1 <- ncol(x)
  p2 <- ncol(z)
  p <- p1+p2
  
  omega <- mvrnorm(L, mu=rep(0, p), Sigma=diag(p)/p)
  hw <- sin(w%*%t(omega))
  
  ## data split
  D1 <- sample(D, 0.5*D)
  D2 <- (1:(D))[-D1]
 
  x1 <- x[D1,]; x2 <- x[D2,]
  z1 <- z[D1,]; z2 <- z[D2,]
  w1 <- w[D1,]; w2 <- w[D2,]
  y1 <- y[D1]; y2 <- y[D2]
  hw1 <- hw[D1,]; hw2 <- hw[D2,]
  
  param_m <- list(max_depth = 3, eta = 0.2, 
                  min_child_weight = 1,
                  subsample = 0.5,
                  objective = "reg:squarederror", eval_metric = "rmse")
  data1_m <- xgb.DMatrix(x1, label = y1, nthread = 40)
  data2_m <- xgb.DMatrix(x2, label =  y2, nthread = 40)
  
  watchlist1_m <- list(train = data1_m, eval = data2_m)
  XGB1 <- xgb.train(param = param_m, data = data1_m, nrounds = 100, watchlist = watchlist1_m,
                    early_stopping_rounds = 20, verbose = 0)
  m_test2 <- predict(XGB1, x2)
  
  watchlist2_m <- list(train = data2_m, eval = data1_m)
  XGB2 <- xgb.train(param = param_m, data = data2_m, nrounds = 100, watchlist = watchlist2_m,
                    early_stopping_rounds = 20, verbose = 0)
  m_test1 <- predict(XGB2, x1)
  
  param_phi <- list(max_depth = 3, eta = 0.2, 
                    min_child_weight = 1,
                    subsample = 0.5,
                    objective = "reg:squarederror", eval_metric = "rmse")
  
  phi_temp2 <- matrix(NA, length(D2), L)
  for (l in 1:L) {
    data1_phi <- xgb.DMatrix(x1, label = hw1[,l], nthread = 40)
    data2_phi <- xgb.DMatrix(x2, label = hw2[,l], nthread = 40)
    watchlist1_m <- list(train = data1_phi, eval = data2_phi)
    XGB1_phi <- xgb.train(param = param_phi, data = data1_phi, nrounds = 100, watchlist = watchlist1_m,
                          early_stopping_rounds = 20, verbose = 0)
    phi_test2 <- predict(XGB1_phi, x2)
    
    phi_temp2[,l] <- phi_test2
  }
  
  phi_temp1 <- matrix(NA, length(D1), L)
  for (l in 1:L) {
    data1_phi <- xgb.DMatrix(x1, label = hw1[,l], nthread = 40)
    data2_phi <- xgb.DMatrix(x2, label = hw2[,l], nthread = 40)
    watchlist2_m <- list(train = data2_phi, eval = data1_phi)
    XGB1_phi <- xgb.train(param = param_phi, data = data2_phi, nrounds = 100, watchlist = watchlist2_m,
                          early_stopping_rounds = 20, verbose = 0)
    phi_test1 <- predict(XGB1_phi, x1)
    
    phi_temp1[,l] <- phi_test1
  }

  T_temp2 <- matrix(NA, L, 1)
  Omega_temp2 <- matrix(NA, L, L)
  for (l in 1:L) {
    for (k in 1:L) {
      T_temp2[l,] <- sum((y2-m_test2)*(hw2-phi_temp2)[,l])/sqrt(length(D2))
      Omega_temp2[k,l] <- mean((y2-m_test2)^2*(hw2-phi_temp2)[,k]*(hw2-phi_temp2)[,l])
    }
  }
  
  T_temp1 <- matrix(NA, L, 1)
  Omega_temp1 <- matrix(NA, L, L)
  for (l in 1:L) {
    for (k in 1:L) {
      T_temp1[l,] <- sum((y1-m_test1)*(hw1-phi_temp1)[,l])/sqrt(length(D1))
      Omega_temp1[k,l] <- mean((y1-m_test1)^2*(hw1-phi_temp1)[,k]*(hw1-phi_temp1)[,l])
    }
  }
 
  V2 <- t(T_temp2)%*%solve(Omega_temp2)%*%T_temp2
  V1 <- t(T_temp1)%*%solve(Omega_temp1)%*%T_temp1
  V <- V1+V2
  pval <- 1-pchisq(V, df = 2*L)
  return(pval)
}

# Testing procedure for H0:E(Y|X,Z) = E(Y|X) by Selecting of transformation functions.
Significancetest_LargeLL <- function(x, z, w, y, LL, L_bar){
  D <- nrow(x)
  p1 <- ncol(x)
  p2 <- ncol(z)
  p <- p1+p2
 
  omega <- mvrnorm(LL, mu=rep(0, p), Sigma=diag(p)/p)
  hw <- sin(w%*%t(omega))
  
  ## data split
  D1 <- sample(D, 0.5*D)
  D2 <- (1:(D))[-D1]
  
  x1 <- x[D1,]; x2 <- x[D2,]
  y1 <- y[D1]; y2 <- y[D2]
  hw1 <- hw[D1,]; hw2 <- hw[D2,]

  paramh1_m <- list(max_depth = 3, eta = 0.2, 
                    min_child_weight = 1,
                    subsample = 0.5,
                    objective = "reg:squarederror", eval_metric = "rmse")
  datah1_m <- xgb.DMatrix(x1, label = y1, nthread = 40)
  datah2_m <- xgb.DMatrix(x2, label = y2, nthread = 40)
  
  watchlisth1_m <- list(train = datah1_m, eval = datah2_m)
  XGBh1 <- xgb.train(param = paramh1_m, data = datah1_m, nrounds = 100, watchlist = watchlisth1_m,
                     early_stopping_rounds = 20, verbose = 0)
  mh1_test <- predict(XGBh1, x1)
 
  Th1_temp <- matrix(NA, LL, 1)
  for (l in 1:LL) {
    Th1_temp[l,] <- mean((y1-mh1_test)*hw1[,l])
  }
  
  index1 <- order(abs(Th1_temp), decreasing = T)[1:L_bar]
  hw1_index1 <- as.matrix(hw1[,index1])
  hw2_index1 <- as.matrix(hw2[,index1])
 
  watchlisth1_m <- list(train = datah2_m, eval = datah1_m)
  XGBh2 <- xgb.train(param = paramh1_m, data = datah2_m, nrounds = 100, watchlist = watchlisth1_m,
                     early_stopping_rounds = 20, verbose = 0)
  mh2_test <- predict(XGBh2, x2)
  
  
  Th2_temp <- matrix(NA, LL, 1)
  for (l in 1:LL) {
    Th2_temp[l,] <- mean((y2-mh2_test)*hw2[,l])
  }
  
  index2 <- order(abs(Th2_temp), decreasing = T)[1:L_bar]
  hw1_index2 <- as.matrix(hw1[,index2])
  hw2_index2 <- as.matrix(hw2[,index2])

  param_m <- list(max_depth = 3, eta = 0.2, 
                  min_child_weight = 1,
                  subsample = 0.5,
                  objective = "reg:squarederror", eval_metric = "rmse")
  data1_m <- xgb.DMatrix(x1, label = y1, nthread = 40)
  data2_m <- xgb.DMatrix(x2, label =y2, nthread = 40)
  
  watchlist1_m <- list(train = data1_m, eval = data2_m)
  XGB1 <- xgb.train(param = param_m, data = data1_m, nrounds = 100, watchlist = watchlist1_m,
                    early_stopping_rounds = 20, verbose = 0)
  m_test2 <- predict(XGB1, x2)
  
  watchlist1_m <- list(train = data2_m, eval = data1_m)
  XGB2 <- xgb.train(param = param_m, data = data2_m, nrounds = 100, watchlist = watchlist1_m,
                    early_stopping_rounds = 20, verbose = 0)
  m_test1 <- predict(XGB2, x1)
  
  param_phi <- list(max_depth = 3, eta = 0.2,   
                    min_child_weight = 1,
                    subsample = 0.5,
                    objective = "reg:squarederror", eval_metric = "rmse")
 
  phi_temp2 <- matrix(NA, length(D2), L_bar)
  for (l in 1:L_bar) {
    data1_phi <-xgb.DMatrix(x1, label = hw1_index1[,l], nthread = 40)
    data2_phi <- xgb.DMatrix(x2, label =hw2_index1[,l], nthread = 40)
    watchlist1_m <- list(train = data1_phi, eval = data2_phi)
    XGB1_phi <- xgb.train(param = param_phi, data = data1_phi, nrounds = 100, watchlist = watchlist1_m,
                          early_stopping_rounds = 20, verbose = 0)
    phi_test2 <- predict(XGB1_phi, x2)
    phi_temp2[,l] <- phi_test2
  }
  
  phi_temp1 <- matrix(NA, length(D1), L_bar)
  for (l in 1:L_bar) {
    data1_phi <- xgb.DMatrix(x1, label = hw1_index2[,l], nthread = 40)
    data2_phi <- xgb.DMatrix(x2, label = hw2_index2[,l], nthread = 40)
    watchlist1_m <- list(train = data2_phi, eval = data1_phi)
    XGB2_phi <- xgb.train(param = param_phi, data = data2_phi, nrounds = 100, watchlist = watchlist1_m,
                          early_stopping_rounds = 20, verbose = 0)
    phi_test2 <- predict(XGB2_phi, x1)
    
    phi_temp1[,l] <- phi_test2
  }
  
  
  T_temp2 <- matrix(NA, L_bar, 1)
  Omega_temp2 <- matrix(NA, L_bar, L_bar)
  for (l in 1:L_bar) {
    for (k in 1:L_bar) {
      T_temp2[l,] <- sum((y2-m_test2)*(hw2_index1-phi_temp2)[,l])/sqrt(length(D2))
      Omega_temp2[k,l] <- mean((y2-m_test2)^2*(hw2_index1-phi_temp2)[,k]*(hw2_index1-phi_temp2)[,l])
    }
  }
  
  T_temp1 <- matrix(NA, L_bar, 1)
  Omega_temp1 <- matrix(NA, L_bar, L_bar)
  for (l in 1:L_bar) {
    for (k in 1:L_bar) {
      T_temp1[l,] <- sum((y1-m_test1)*(hw1_index2-phi_temp1)[,l])/sqrt(length(D1))
      Omega_temp1[k,l] <- mean((y1-m_test1)^2*(hw1_index2-phi_temp1)[,k]*(hw1_index2-phi_temp1)[,l])
    }
  }
  
  V2 <- t(T_temp2)%*%solve(Omega_temp2)%*%T_temp2
  V1 <- t(T_temp1)%*%solve(Omega_temp1)%*%T_temp1
  V <- V1+V2
  pval <- 1-pchisq(V, df = L_bar*2)
  return(pval)
}


#Testing procedure for conditional independence test.
Conditionalindependencetest <- function(x, z, w, y, K, L){
  D <- nrow(w)
  p <- ncol(w)
  
  y <- ns(y, df = K)
  omega <- mvrnorm(L, mu = rep(0, p), Sigma = diag(p)/p)
  hw <- sin(w%*%t(omega))
  
  param_m <- list(max_depth = 1, eta = 0.1, 
                  min_child_weight = 1,
                  subsample = 0.5,
                  objective = "reg:squarederror", eval_metric = "rmse")
  
  m_temp <- matrix(NA, D, K)
  for (k in 1:K) {
    data1_m <- xgb.DMatrix(x, label = y[,k], nthread = 40)
    data2_m <- xgb.DMatrix(x, label = y[,k], nthread = 40)
    watchlist1_m <- list(train = data1_m, eval = data2_m)
    XGB1 <- xgb.train(param = param_m, data = data1_m, nrounds = 100, watchlist = watchlist1_m,
                      early_stopping_rounds = 20, verbose = 0)
    m <- predict(XGB1, x)
    m_temp[,k] <- m
  }
  
  param_phi <- list(max_depth = 1, eta = 0.1, 
                    min_child_weight = 1,
                    subsample = 0.5,
                    objective = "reg:squarederror", eval_metric = "rmse")
  
  phi_temp <- matrix(NA, D, L)
  for (l in 1:L) {
    data1_phi <- xgb.DMatrix(x, label = hw[,l], nthread = 40)
    data2_phi <- xgb.DMatrix(x, label = hw[,l], nthread = 40)
    watchlist1_m <- list(train = data1_phi, eval = data2_phi)
    XGB1_phi <- xgb.train(param = param_phi, data = data1_phi, nrounds = 100, watchlist = watchlist1_m,
                          early_stopping_rounds = 20, verbose = 0)
    phi <- predict(XGB1_phi, x)
    phi_temp[,l] <- phi
  }
  
  S <- numeric(K * L)
  for (k in 1:K) {
    for (l in 1:L) {
      S[l + (k-1)*L] <- sum((y-m_temp)[, k]*(hw-phi_temp)[, l])/sqrt(D)
    }
  }
  
  Sigma <- matrix(0, nrow = K*L, ncol = K*L)
  for (k1 in 1:K) {
    for (l1 in 1:L) {
      for (k2 in 1:K) {
        for (l2 in 1:L) {
          Sigma[((k1-1)*L+l1), ((k2-1)*L+l2)] <- mean((y-m_temp)[, k1]*(y-m_temp)[, k2]*(hw-phi_temp)[, l1]*(hw-phi_temp)[, l2]) 
        }
      }
    }
  }
 
  V <- t(S)%*%solve(Sigma)%*%S
  pval <- 1-pchisq(V, df = K*L)
  return(pval)
}

