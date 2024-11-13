#' Generate data for simulation
#'
#' This function generates simulated data in multiple time series with heterogeneity and non-stationarity.
#' It includes 3 settings in Setion 5.3.
#'
#' @param n_patients the number of patients
#' @param n_var the number of X variables
#' @param T maximum time
#' @param idx_x indices for the x data, a sparse matrix
#' @param idx_y indices for the y data, a sparse matrix
#' @param rank rank for the random matrices
#' @param k spline smoothness
#' @param N number of knots in the splineS
#' @return A list is returned, containing output_x and output_y as sparse matrices of x_data and y_data, spline knots, individualized dynamic latent factor, shared latent factor for X and Y.
#' @export
#' @importFrom stats rnorm
#' @importFrom splines splineDesign
#' @importFrom SparseArray COO_SparseArray SVT_SparseArray randomSparseArray nzcoo nzdata
#' @importFrom methods as
#' @references Zhang, J., F. Xue, Q. Xu, J. Lee, and A. Qu. "Individualized dynamic latent factor model for multi-resolutional data with application to mobile health." Biometrika (2024): asae015.
#' @examples
#' library(splines)
#' #if (!require("BiocManager", quietly = TRUE))
#' #install.packages("BiocManager")
#' #BiocManager::install("SparseArray")
#' library(SparseArray)
#'
#' I <- 3
#' J <- 10
#' T <- 100
#' R <- 3
#' k <- 3
#' N <- 300
#' idx_x <- randomSparseArray(c(I, J, T), density=0.8)
#' idx_y <- randomSparseArray(c(I, 1, T), density=0.2)
#' idx_x <- array(runif(length(idx_x), 0, 1), dim = dim(idx_x))
#' idx_y <- array(runif(length(idx_y), 0, 1), dim = dim(idx_y))
#' generate_data(I, J, T, idx_x, idx_y, R, k, N)
generate_data <- function(n_patients, n_var, T, idx_x, idx_y, rank, k, N) {
  D <- N - k
  Fx <- matrix(rnorm(n_var * rank), nrow = n_var, ncol = rank)
  Fy <- rnorm(rank)

  knots <- seq(1, N - 2 * k - 1) / (N - 2 * k - 1) * T
  knots <- c(rep(-1, k + 1), knots, rep(T + 1, k + 1))

  weights <- array(rnorm(n_patients * rank * D), dim = c(n_patients, rank, D))

  x_data <- numeric(0)
  y_data <- numeric(0)
  # Setting 2.3
  spl <- list(
    function(t, i) 0.02 * i * log(t + 1),
    function(t, i) 2 * exp(-(t - 60 + 10 * i) / 50 * (t - 60 + 10 * i)) + 4 * exp(-(t - 70 + 10 * i) / 20 * (t - 70 + 10 * i)),
    function(t, i) cos(0.12 * pi * t) + 1
  )

  # Setting 2.2
  # spl <- list(
  # function(t, i) 0.2 * log(t + 1),
  # function(t, i) 2 * exp(-(t - 60 + 10 * i) / 50 * (t - 60 + 10 * i)) + 4 * exp(-(t - 70 + 10 * i) / 20 * (t - 70 + 10 * i)),
  # function(t, i) cos(0.12 * pi * t) + 1
  # )

  # Setting 2.1
  # spl <- list(
  # function(t, i) 0.2 * log(t + 1),
  # function(t, i) 2 * exp(-(t - 60) / 50 * (t - 60)) + 4 * exp(-(t - 70) / 20 * (t - 70)),
  # function(t, i) cos(0.12 * pi * t) + 1
  # )
  for (i in 1:n_patients) {
    for (j in 1:n_var) {
      spline_values <- sapply(1:rank, function(r) spl[[r]](idx_x[i, j, ], i))
      tmp <- Fx[j, ] %*% t(spline_values) + 0.5 * rnorm(length(idx_x[i, j, ]))
      x_data <- c(x_data, tmp)
    }

      spline_values <- sapply(1:rank, function(r) spl[[r]](idx_y[i, 1, ], i))
      tmp <- Fy %*% t(spline_values) + 0.5 * rnorm(length(idx_y[i, 1, ]))
      y_data <- c(y_data, tmp)
  }
  idx_x_coords <- which(idx_x != 0, arr.ind = TRUE)
  idx_y_coords <- which(idx_y != 0, arr.ind = TRUE)
  output_x <- COO_SparseArray(c(n_patients, n_var, T), idx_x_coords, x_data)
  output_y <- COO_SparseArray(c(n_patients, 1, T), idx_y_coords, y_data)

  list(output_x, output_y, knots, weights, Fx, Fy)
}

#' Individualized Dynamic Latent Factor Model
#'
#' This function implements the individualized dynamic latent factor model.
#'
#' @param X a sparse matrix for predictor variables
#' @param Y a sparse matrix for response variables
#' @param n_patients the number of patients
#' @param n_var the number of X variables
#' @param T maximum time
#' @param idx_x indices for the X data, a sparse matrix
#' @param idx_y indices for the Y data, a sparse matrix
#' @param rank rank for the random matrices
#' @param k spline smoothness
#' @param N number of knots in the spline
#' @param lambda1 regularization parameter for fused lasso, with the default value 1
#' @param lambda2 regularization parameter for total variation, with the default value 1
#' @param Niter number of iterations for the Adam optimizer, with the default value 100
#' @param alpha learning rate for the Adam optimizer, with the default value 0.001
#' @param ebs convergence threshold, with the default value 0.0001
#' @param l regularization parameter, with the default value 1
#' @return A list is returned, containing the model weights, factor matrix, spline knots, predicted X and Y.
#' @export
#' @importFrom SparseArray COO_SparseArray extract_sparse_array nzcoo nzdata
#' @references Zhang, J., F. Xue, Q. Xu, J. Lee, and A. Qu. "Individualized dynamic latent factor model for multi-resolutional data with application to mobile health." Biometrika (2024): asae015.
#' @examples
#' library(splines)
#' #if (!require("BiocManager", quietly = TRUE))
#' #install.packages("BiocManager")
#' #BiocManager::install("SparseArray")
#' library(SparseArray)
#'
#' I <- 3
#' J <- 10
#' T <- 100
#' R <- 3
#' k <- 3
#' N <- 300
#' idx_x <- randomSparseArray(c(I, J, T), density=0.8)
#' idx_x <- array(runif(length(idx_x), 0, 1), dim = dim(idx_x))
#' idx_y_train <- randomSparseArray(c(I, 1, T), density=0.2)
#' idx_y_train <- array(runif(length(idx_y_train), 0, 1), dim = dim(idx_y_train))
#' idx_y_test <- randomSparseArray(c(I, 1, T), density=0.2)
#' idx_y_test <- array(runif(length(idx_y_test), 0, 1), dim = dim(idx_y_test))
#' data <- generate_data(I, J, T, idx_x, idx_y_train, R, k, N)
#' output_x <- data[[1]]
#' output_y <- data[[2]]
#' knots <- data[[3]]
#' weights <- data[[4]]
#' Fx <- data[[5]]
#' Fy <- data[[6]]
#' IDLFM(X = output_x, Y = output_y, n_patients = I, n_var = J, T = T,
#' idx_x = idx_x, idx_y = idx_y_train, rank = R, k = k, N = N)
IDLFM <- function(X, Y, n_patients, n_var, T, idx_x, idx_y, rank, k, N, lambda1=1, lambda2=1, Niter=100, alpha=0.001, ebs=0.0001, l=1) {
  beta1 <- 0.9
  beta2 <- 0.999
  m_w <- 0
  m_F <- 0
  v_w <- 0
  v_F <- 0

  D <- N - k - 1
  F <- matrix(rnorm((n_var + 1) * rank), nrow = n_var + 1, ncol = rank)

  coords_Y <- nzcoo(Y)
  coords_Y[, 2] <- n_var
  coords_X <- nzcoo(X)
  coords <- rbind(coords_X, coords_Y)
  data <- c(nzdata(X), nzdata(Y))
  xy <- COO_SparseArray(c(n_patients, n_var + 1, T), nzcoo = coords, nzdata = data, dimnames=NULL, check=TRUE)

  knots <- seq(1, N - 2 * k - 2) / (N - 2 * k - 2) * T
  knots <- c(rep(-1, k + 1), knots, rep(T + 1, k + 1))

  idx_x_coo <- COO_SparseArray(c(dim(idx_x)), nzcoo = which(idx_x != 0, arr.ind = TRUE), nzdata = idx_x[idx_x != 0])
  idx_y_coo <- COO_SparseArray(c(dim(idx_y)), nzcoo = which(idx_y != 0, arr.ind = TRUE), nzdata = idx_y[idx_y != 0])
  coords_idx_y <- nzcoo(idx_y_coo)
  coords_idx_y[, 2] <- n_var
  coords_idx_x <- nzcoo(idx_x_coo)
  coords_idx <- rbind(coords_idx_x, coords_idx_y)
  data_idx <- c(nzdata(idx_x_coo), nzdata(idx_y_coo))
  idx <- COO_SparseArray(c(n_patients, n_var + 1, T), nzcoo = coords_idx, nzdata = data_idx)

  weights <- array(rnorm(n_patients * rank * D), dim = c(n_patients, rank, D))

  # Define xy_pred
  xy_pred <- function(weights, knots, F, n_patients, n_var, idx, rank, k) {
    thetas <- vector("list", n_patients)
    for (i in 1:n_patients) {
      thetas[[i]] <- vector("list", rank)
      for (r in 1:rank) {
        thetas[[i]][[r]] <- function(x) {
          if (length(x) > 0) {
            spline_mat <- splineDesign(knots, x, ord = k + 1, outer.ok = TRUE)
            weight_mat <- matrix(weights[i, r, ])
            result <- spline_mat %*% weight_mat
            return(result)
          } else {
            return(rep(0, D))
          }
        }
      }
    }
    data <- numeric()
    for (i in 1:n_patients) {
      for (j in 1:(n_var + 1)) {
        ## Calculate B-spline results for each variable for each patient
        subset_idx <- list(i, j, TRUE)
        subset_data <- extract_sparse_array(idx, subset_idx)
        thetas_subset <- sapply(1:rank, function(r) {
          thetas[[i]][[r]](subset_data)
        })
        tmp <- F[j, ] %*% thetas_subset
        data <- c(data, tmp)
      }
    }
    idx_coords <- nzcoo(idx)
    output <- COO_SparseArray(c(n_patients, n_var + 1, dim(idx)[3]), idx_coords, data)
    return(output)
  }

  xy_hat <- xy_pred(weights, knots, F, n_patients, n_var, idx, rank, k)

  nobs <- length(data)
  S <- sum((nzdata(xy) - nzdata(xy_hat))^2) / nobs
  S_record <- c(S)
  print(S)

  K <- matrix(0, nrow = D, ncol = T, byrow = TRUE)
  for (t in 1:T) {
    xval <- t
    if (xval <= knots[k]) {
      left <- k
    } else {
      left <- which(knots > xval)[1] - 1
    }
    # Fill a row
    bb <- splineDesign(knots * 1.0, xval, ord = k + 1, outer.ok = TRUE)
    if (length(bb) >= left - k + 1) {
      K[(left - k + 1):left, t] <- bb[(left - k + 1):left]
    } else {
      K[(left - k + 1):left, t] <- bb
    }
  }

  # Define tensordot
  tensordot <- function(A, B, axes) {
    axesA <- axes[[1]]
    axesB <- axes[[2]]

    permA <- c(setdiff(1:length(dim(A)), axesA), axesA)
    permB <- c(axesB, setdiff(1:length(dim(B)), axesB))
    A_perm <- aperm(A, permA)
    B_perm <- aperm(B, permB)

    dimA <- dim(A_perm)
    dimB <- dim(B_perm)

    sum_dim <- prod(dimA[(length(dimA) - length(axesA) + 1):length(dimA)])

    A_mat <- matrix(A_perm, nrow = prod(dimA[1:(length(dimA) - length(axesA))]), ncol = sum_dim)
    B_mat <- matrix(B_perm, nrow = sum_dim, ncol = prod(dimB[(length(axesB) + 1):length(dimB)]))

    result_mat <- A_mat %*% B_mat

    result_dim <- c(dimA[1:(length(dimA) - length(axesA))], dimB[(length(axesB) + 1):length(dimB)])
    result <- array(result_mat, dim = result_dim)

    return(result)
  }

  for (itr in 1:Niter) {
    unique_t <- sort(unique(nzdata(idx)))
    theta <- tensordot(weights, K, axes = c(3, 1))
    xy_hat_dense <- as.array(xy_hat)
    xy_dense <- as.array(xy)
    grad_weights <- tensordot(2 * (xy_hat_dense - xy_dense), F, axes = c(2, 1))
    grad_weights <- tensordot(grad_weights, K, axes = c(2, 2))
    # Fused lasso for theta_i
    grad_pen <- array(0, dim = dim(weights))
    trans_m <- -diag(n_patients - 1)

    insertCols <- function(mat, col, index) {
      if (index == 0) {
        return(cbind(col, mat))
      } else if (index == ncol(mat)) {
        return(cbind(mat, col))
      } else {
        return(cbind(mat[, 1:index, drop=FALSE], col, mat[, (index+1):ncol(mat), drop=FALSE]))
      }
    }

    for (i in 0:(n_patients - 1)) {
      tmp <- insertCols(trans_m, rep(1, n_patients - 1), i)
      tmp <- tensordot(tmp, weights, axes = c(2, 1))
      tmp <- sign(tmp)
      grad_pen[i+1, , ] <- apply(tmp, 2, sum)
    }
    # Total variation for bspline
    diag1 <- cbind(matrix(0, D - 1, 1), -diag(D - 1))
    jump <- rbind(diag1, matrix(0, 1, D))
    jump <- jump + diag(D)
    jump_m <- jump
    if (k > 1) {
      for (i in 1:k) {
        jump_m <- jump_m %*% jump
      }
    }
    grad_pen1 <- tensordot(weights, t(jump_m), axes = c(3, 1))
    grad_pen1 <- sign(grad_pen1)
    grad_pen1 <- tensordot(grad_pen1[, , 1:(D - k)], jump_m[1:(D - k), ], axes = c(3, 1))
    grad_weights <- grad_weights + l * weights + lambda1 * grad_pen + lambda2 * grad_pen1
    grad_F <- tensordot(2 * (xy_hat_dense - xy_dense), theta, axes = list(c(1, 3), c(1, 3)))
    grad_F <- grad_F + l * F

    m_w <- beta1 * m_w + (1 - beta1) * grad_weights
    m_F <- beta1 * m_F + (1 - beta1) * grad_F
    v_w <- beta2 * v_w + (1 - beta2) * grad_weights^2
    v_F <- beta2 * v_F + (1 - beta2) * grad_F^2
    mhat_w <- m_w / (1 - beta1^itr)
    mhat_F <- m_F / (1 - beta1^itr)
    vhat_w <- v_w / (1 - beta2^itr)
    vhat_F <- v_F / (1 - beta2^itr)

    weights <- weights - alpha * mhat_w / (sqrt(vhat_w) + 1e-8)
    F <- F - alpha * mhat_F / (sqrt(vhat_F) + 1e-8)

    xy_hat <- xy_pred(weights, knots, F, n_patients, n_var, idx, rank, k)
    S <- sum((nzdata(xy) - nzdata(xy_hat))^2) / nobs
    t <- abs((S_record[length(S_record)] - S) / S_record[length(S_record)])

    if (itr > 10 && S >= max(S_record)) {
      print('Diverge')
      break
    }
    if (t < ebs) {
      print(itr)
      print('Converge')
      break
    }

    if (itr %% 100 == 0) {
      print(c(itr, S))
    }
  }
  print('Max iteration')

  index_list <- list(1:n_patients, 1:n_var, 1:dim(xy_hat)[3])
  X_hat <- extract_sparse_array(xy_hat, index_list)
  index_list <- list(1:n_patients, n_var, 1:dim(xy_hat)[3])
  Y_hat <- extract_sparse_array(xy_hat, index_list)
  Y_hat_array <- as.array(Y_hat)
  dim(Y_hat_array) <- c(n_patients, 1, dim(xy_hat)[3])
  Y_hat <- as(Y_hat_array, "COO_SparseArray")

  return(list(weights, F, knots, X_hat, Y_hat))
}
