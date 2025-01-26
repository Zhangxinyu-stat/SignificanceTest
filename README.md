# SignificanceTest
R code for "Model-free variable importance testing with machine learning methods". 
## Usage
- `Significancetest(x, z, w, y, L)`
- `Significancetest_LargeLL(x, z, w, y, LL, L_bar)`
- `Conditionalindependencetest(x, z, w, y, K, L)`
## Required Packages
- `MASS`
- `xgboost`
- `splines`
## Inputs
- `w`: A matrix of $N*p$, where $N$ is the sample size and $p$ is the dimension of predictor.
- `x`: A matrix of $N*p_1$, where $p_1$ is the dimension of subset of predictor.
- `z`: A matrix of $N*p_2$, where $p_2=p-p_1$ is the dimension of subset of predictor.
- `y`: The response variable.
- `L`: The number of fixed transformation functions for predictor `w`.
- `LL`: The large number of transformation functions for predictor `w`.
- `L_bar`: The number of the selected transformation functions for predictor `w`.
- `K`: The number of transformation functions for response `y`.
## Examples
```
library(MASS)
library(xgboost)
library(splines)

# setting
p <- 50; N <- 200; p1 <- (1/2)*p; p2 <- p-p1
L <- 13; LL <- 100; L_bar <- 6; K <- 4
Sig <- toeplitz((1/2)^seq(0, p-1))
beta <- c(rep(1/sqrt(2),2), rep(0, p1-2))*sqrt(2)
theta <- c(rep(1/sqrt(3),3), rep(0, p2-3))/sqrt(2)

# generate x, z, w, and y
w <- mvrnorm(N, mu=rep(0, p), Sigma=Sig)
x <- w[,1:p1]; z <- w[,(p1+1):p]
error <- rnorm(N, 0, 0.5)
y <- x%*%beta+z%*%theta+error

# p-value
Significancetest(x, z, w, y, L)
Significancetest_LargeLL(x, z, w, y, LL, L_bar)
Conditionalindependencetest(x, z, w, y, K, L)
```
