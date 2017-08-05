# [0] read in data ------------------------------------------------------------
path <- c("D:\\Google Drive\\NCTU\\106\\下學期\\機器學習\\HW\\HW1\\")

library(magrittr)
library(dplyr)
train_X <- read.csv(file = paste0(path,"X_train.csv"), header = F)
train_T <- read.csv(file = paste0(path,"T_train.csv"), header = F)

colnames(train_X) <- c("X1","X2")
colnames(train_T) <- c("Y")

train_raw <- cbind.data.frame(train_X,train_T)

# seperate data for validation
D123 <- train_raw[    1:30000, ]
D4   <- train_raw[30001:40000, ]


# [1] Model construction -------------------------------------------------------
#  1. Model : Bivariate Normal Distribution ------------------------------------
# seperate to 59*59 subspaces (-59~-10,-39~10,-19~30,1~50,..., 1061~1110,1081~1130,1101~1150) 
# mean of X1, X2 in each subspaces
u_X1 <- matrix(0,59,59)
u_X2 <- matrix(0,59,59)
# sd of X1, X2 in each subspaces
sigma_X1 <- matrix(0,59,59)
sigma_X2 <- matrix(0,59,59)

start.time <- Sys.time()

for(i in 1:59){
  for(j in 1:59){
    
    tmp <- D123 %>% filter(X1 >= 20*j-79 , X1 < 20*j-30 , X2 >= 20*i-79 , X2 < 20*i-30 )
    if(mean(tmp$X1) %>% is.na()){u_X1[i,j] = 20*j-55} 
    else{u_X1[i,j] = mean(tmp$X1)}
    if(mean(tmp$X2) %>% is.na()){u_X2[i,j] = 20*i-55} 
    else{u_X2[i,j] = mean(tmp$X2)}  
    
    if(sd(tmp$X1) %>% is.na()){sigma_X1[i,j] = 15} 
    else{sigma_X1[i,j] = sd(tmp$X1)}
    if(sd(tmp$X2) %>% is.na()){sigma_X2[i,j] = 15} 
    else{sigma_X2[i,j] = sd(tmp$X2)}            
  }
}
sigma_X1[sigma_X1==0] <- 5     # if sd < 0, then set to be 5
sigma_X2[sigma_X2==0] <- 5     


Sys.time() - start.time  # Time difference of 26.95075 secs



#  2. basic function (phi) --------------------------------------------------
phi <- matrix(1,30000,1) # phi_0 set to be 1

start.time <- Sys.time()
for(i in 1:59){
  for(j in 1:59){
    X1 <- D123[,1] 
    X2 <- D123[,2]
    
    tmp <- exp(-1*((X1-u_X1[i,j])*(X1-u_X1[i,j])) / (2*sigma_X1[i,j]*sigma_X1[i,j]) 
               -1*((X2-u_X2[i,j])*(X2-u_X2[i,j])) / (2*sigma_X2[i,j]*sigma_X2[i,j]))
    
    phi <- cbind(phi,tmp) %>% as.matrix()
  }
}
Sys.time() - start.time  # Time difference of 13.8039 mins


# [2] Bayesian ------------------------------------

# parameters setting [lamda = alpha / beta] 
alpha <- 1
beta  <- 1000000

start.time <- Sys.time()

t <- as.matrix(D123[,c(3)])

SN <- solve( alpha*diag(3482) + beta*(t(phi) %*% phi) )

m_N <- beta*SN %*% t(phi) %*% t


y_hat_Bay <- matrix(0,30000,1)
for(i in 1:30000){
  mean_curve  <- t(m_N) %*% phi[i,]
  SD_N <- 1/beta + t(phi[i,]) %*% SN %*% phi[i,]
  y_hat_Bay[i,1] <- rnorm(1,mean_curve,SD_N)
}

y_hat_Bay[y_hat_Bay<0] <- 0     # if y_hat_Bay < 0, then set to be 0

Sys.time() - start.time # Time difference of 45.26474 mins

# [3] validation data --------------------------------------------------------------------
phi_valid <- matrix(1,10000,1) # phi_0 set to be 1

for(i in 1:59){
  for(j in 1:59){
    X1 <- D4[,1] 
    X2 <- D4[,2]
    
    tmp <- exp(-1*((X1-u_X1[i,j])*(X1-u_X1[i,j])) / (2*sigma_X1[i,j]*sigma_X1[i,j]) 
               -1*((X2-u_X2[i,j])*(X2-u_X2[i,j])) / (2*sigma_X2[i,j]*sigma_X2[i,j]))
    
    phi_valid <- cbind(phi_valid,tmp) %>% as.matrix()
  }
}

t_valid <- as.matrix(D4[,c(3)])

y_hat_Bay_valid <- matrix(0,10000,1)
for(i in 1:10000){
  mean_curve_valid  <- t(m_N) %*% phi_valid[i,]
  SD_N_valid <- 1/beta + t(phi_valid[i,]) %*% SN %*% phi_valid[i,]
  y_hat_Bay_valid[i,1] <- rnorm(1,mean_curve_valid,SD_N_valid)
}

y_hat_Bay_valid[y_hat_Bay_valid<0] <- 0     # if y_hat_Bay < 0, then set to be 0

# [4] Mean Square Error -------------------------------------------------------------
# for training data
tmp <- cbind(y_hat_Bay,t) %>% as.data.frame() %>% mutate(V3 = (V1-V2)*(V1-V2))
MSE <- mean(tmp$V3/2)
MSE # 69.03021 #61.55996
train_123 <- cbind.data.frame(D123,y_hat_Bay)

# for validation data
tmp <- cbind(y_hat_Bay_valid,t_valid) %>% as.data.frame() %>% mutate(V3 = (V1-V2)*(V1-V2))
MSE <- mean(tmp$V3/2)
MSE # 79.49177
valid_4 <- cbind.data.frame(D4,y_hat_Bay_valid)


# [5] 3D plot ----------------------------------------------------------------------------
library(plotly)

# (raw 40000 training data)
#write.csv(train_raw, file = paste0(path,"train_raw.csv"),row.names = F)
plot_ly(data = train_raw, x = ~X1, y = ~X2, z = ~Y,
        marker = list(color = ~Y, colorscale = c('#FFE1A1', '#683531'), showscale = TRUE)) %>%
  add_markers()


# (30000 training data)  
#write.csv(train_123, file = paste0(path,"train_123_C.csv"),row.names = F)
plot_ly(data = train_123, x = ~X1, y = ~X2, z = ~y_hat_Bay,
        marker = list(color = ~y_hat_Bay, colorscale = c('#FFE1A1', '#683531'), showscale = TRUE)) %>%
  add_markers()


# (10000 validation data)  
plot_ly(data = valid_4, x = ~X1, y = ~X2, z = ~y_hat_Bay_valid,
        marker = list(color = ~y_hat_Bay_valid, colorscale = c('#FFE1A1', '#683531'), showscale = TRUE)) %>%
  add_markers()

# [6] testing data ------------------------------------------------------------------------------------
test_X <- read.csv(file = paste0(path,"X_test.csv"), header = F)

colnames(test_X) <- c("X1","X2")

phi_test <- matrix(1,10000,1) # phi_0 set to be 1

for(i in 1:59){
  for(j in 1:59){
    X1 <- test_X[,1] 
    X2 <- test_X[,2]
    
    tmp <- exp(-1*((X1-u_X1[i,j])*(X1-u_X1[i,j])) / (2*sigma_X1[i,j]*sigma_X1[i,j]) 
               -1*((X2-u_X2[i,j])*(X2-u_X2[i,j])) / (2*sigma_X2[i,j]*sigma_X2[i,j]))
    
    phi_test <- cbind(phi_test,tmp) %>% as.matrix()
  }
}

y_hat_Bay_test <- matrix(0,10000,1)
for(i in 1:10000){
  mean_curve_test  <- t(m_N) %*% phi_test[i,]
  SD_N_test <- 1/beta + t(phi_test[i,]) %*% SN %*% phi_test[i,]
  y_hat_Bay_test[i,1] <- rnorm(1,mean_curve_test,SD_N_test)
}

y_hat_Bay_test[y_hat_Bay_test<0] <- 0     # if y_hat_Bay < 0, then set to be 0


test_set <- cbind.data.frame(test_X,y_hat_Bay_test)

# plot (test set)  
plot_ly(data = test_set, x = ~X1, y = ~X2, z = ~y_hat_Bay_test,
        marker = list(color = ~y_hat_Bay_test, colorscale = c('#FFE1A1', '#683531'), showscale = TRUE)) %>%
  add_markers()

# save output
write.table(y_hat_Bay_test, file = paste0(path,"Bay.csv"),row.names = F,col.names = F, sep = ",")

