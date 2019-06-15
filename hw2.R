data_set <- read.csv("C:\\Users\\Goko\\Desktop\\Undergraduate\\COMP421\\hw2\\hw02_data_set_images.csv", header=FALSE)
label_set <- read.csv("C:\\Users\\Goko\\Desktop\\Undergraduate\\COMP421\\hw2\\hw02_data_set_labels.csv", header=FALSE)

train_a <- data_set[c(1:25),]
train_b <- data_set[c(40:64),]
train_c <- data_set[c(79:103),]
train_d <- data_set[c(118:142),]
train_e <- data_set[c(157:181),]

test_a <- data_set[c(26:39),]
test_b <- data_set[c(65:78),]
test_c <- data_set[c(104:117),]
test_d <- data_set[c(143:156),]
test_e <- data_set[c(182:195),]

label_a <- as.numeric(label_set[c(1:25),])
label_b <- as.numeric(label_set[c(40:64),])
label_c <- as.numeric(label_set[c(79:103),])
label_d <- as.numeric(label_set[c(118:142),])
label_e <- as.numeric(label_set[c(157:181),])

label_test_a <- as.numeric(label_set[c(26:39),])
label_test_b <- as.numeric(label_set[c(65:78),])
label_test_c <- as.numeric(label_set[c(104:117),])
label_test_d <- as.numeric(label_set[c(143:156),])
label_test_e <- as.numeric(label_set[c(182:195),])

X <- as.matrix(rbind(train_a, train_b, train_c, train_d, train_e))

X_test <- as.matrix(rbind(test_a, test_b, test_c, test_d, test_e))

y_truth <- c(label_a, label_b, label_c, label_d, label_e)

y_truth_test <- c(label_test_a, label_test_b, label_test_c, label_test_d, label_test_e)

K <- max(y_truth)
N <- length(y_truth)


Y_truth <- matrix(0, N, K)
Y_truth[cbind(1:N, y_truth)] <- 1


safelog <- function(x) {
  return (log(x + 1e-100))
}
sigmoid <- function(X, w, w0) {
  return (1 / (1 + exp(-(X %*% w + w0))))
}


gradient_W <- function(X, Y_truth, Y_predicted) {
  return (-sapply(X = 1:ncol(Y_truth), function(c) colSums(matrix(Y_truth[,c] - Y_predicted[,c], nrow = nrow(X), ncol = ncol(X), byrow = FALSE) * X)))
}

gradient_w0 <- function(Y_truth, Y_predicted) {
  return (-colSums(Y_truth - Y_predicted))
}



eta <- 0.01
epsilon <- 1e-3
set.seed(521)
W <- matrix(runif(ncol(X) * K, min = -0.01, max = 0.01), ncol(X), K)
w0 <- runif(K, min = -0.01, max = 0.01)
iteration <- 1
objective_values <- c()

while (1) {
  Y_predicted <- sigmoid(X, W, w0)

  objective_values <- c(objective_values, sum((Y_truth - Y_predicted)^2)/2)

  W_old <- W
  w0_old <- w0

  W <- W - eta * gradient_W(X, Y_truth, Y_predicted)
  w0 <- w0 - eta * gradient_w0(Y_truth, Y_predicted)

  if (sum((Y_truth - Y_predicted)^2)/2 < epsilon) {
    break
  }

  iteration <- iteration + 1
}

y_predicted <- apply(Y_predicted, 1, which.max)
confusion_matrix <- table(y_predicted, y_truth)


plot(1:iteration, objective_values,
     type = "l", lwd = 2, las = 1,
     xlab = "Iteration", ylab = "Error")

y_predicted_test <- sigmoid(X_test, W, w0)

y_predicted_test <- apply(y_predicted_test, 1, which.max)
confusion_matrix1 <- table(y_predicted_test, y_truth_test)

print(confusion_matrix)
print(confusion_matrix1)

