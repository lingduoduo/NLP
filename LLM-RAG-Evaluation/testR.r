library(glmnet)
X <- as.matrix(cbind(x1, x2, x3))
y <- your_simulated_y
fit <- glmnet(X, y)
plot(fit)