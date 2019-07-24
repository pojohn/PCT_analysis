#https://www.jamleecute.com/gradient-boosting-machines-gbm/
library(showtext)
#讀入csv檔
library(readr)
#表格設定
library(knitr)
library(kableExtra)
library(ellipse)
library(pheatmap)
library(DT)
#資料整理
library(dplyr)
#繪圖設定
library(ggplot2)
#KW檢定
library(jmv)
#變數重要性繪圖框架
library(vip)

#讀入資料
dfg<-read_csv("E:/Rtest/PCT analysis/20190715PCT(gbm).csv",
             col_types = cols(
               'Time' = col_factor(levels = c(30,40,50), ordered = FALSE, include_na = FALSE),
               'Power' = col_factor(levels = c(45,60,75), ordered = FALSE, include_na = FALSE),
               'Force' = col_factor(levels = c(30,40,50), ordered = FALSE, include_na = FALSE),
               'Tail Length' = col_factor(levels = c(200,250,300), ordered = FALSE, include_na = FALSE),
               'EFO Current' = col_factor(levels = c(25,35,45), ordered = FALSE, include_na = FALSE),
               'Temp' = col_factor(levels = c(140,160,180), ordered = FALSE, include_na = FALSE),
               'Ball-area' = col_factor(levels = c(5,15,25), ordered = FALSE, include_na = FALSE),
               'Ball-Version' = col_factor(levels = c(0,1), ordered = FALSE, include_na = FALSE),
               'Ball-Shear' = col_double(),
               'Ball-Size(X)' = col_double(),
               'Ball-Size(Y)' = col_double(),
               'Ball-Size(Z)' = col_double()
             )
)

#檢查資料
str(dfg)

###########################################
#1.radient Boosting Machines GBM
###########################################


#套件與資料準備
library(rsample)      # data splitting 
library(gbm)          # basic implementation
library(xgboost)      # a faster implementation of gbm
library(caret)        # an aggregator package for performing many machine learning models
library(h2o)          # a java-based platform
library(pdp)          # model visualization
library(ggplot2)      # model visualization
library(lime)         # model visualization
library(vtreat)


#將數據切分為7:3的訓練測試比例
set.seed(123)
ames_split <- initial_split(dfg, prop = .7)
ames_train <- training(ames_split)
ames_test  <- testing(ames_split)


#
h2o.init()
h2o.no_progress()

# create feature names
y <- "Ball-Version"
x <- setdiff(names(ames_train), y)

# turn training set into h2o object
train.h2o <- as.h2o(ames_train)

# training basic GBM model with defaults
h2o.fit1 <- h2o.gbm(
  x = x,
  y = y,
  training_frame = train.h2o,
  nfolds = 5
)

# assess model results
h2o.fit1


# training basic GBM model with defaults
h2o.fit2 <- h2o.gbm(
  x = x,
  y = y,
  training_frame = train.h2o,
  nfolds = 5,
  ntrees = 5000,
  stopping_rounds = 10,
  stopping_tolerance = 0,
  seed = 123
)

# model stopped after xx trees
h2o.fit2@parameters$ntrees
# cross validated RMSE
h2o.rmse(h2o.fit2, xval = TRUE)



#以下便採用Random Discrete Path法來執行和剛剛一模一樣的hyperparameter grid。不過，在此我們會加入新的search條件：當連續有10個模型效果都無法超越目前最佳的模型獲得0.5%的MSE改善時，則停止。如果持續有在獲得改善，但超過360秒(60分鐘)時，也停止程序。

# random grid search criteria
search_criteria <- list(
  strategy = "RandomDiscrete",
  stopping_metric = "mse",
  stopping_tolerance = 0.005,
  stopping_rounds = 10,
  max_runtime_secs = 60*60
)

#Full grid search

# create training & validation sets
split <- h2o.splitFrame(train.h2o, ratios = 0.75)
train <- split[[1]]
valid <- split[[2]]


# perform grid search 
system.time(
  grid <- h2o.grid(
    algorithm = "gbm",
    grid_id = "gbm_grid1",
    x = x, 
    y = y, 
    training_frame = train.h2o,
    ntrees = 5000,
    stopping_rounds = 10,
    stopping_tolerance = 0
  )
)

# train final model
h2o.final <- h2o.gbm(
  x = x,
  y = y,
  training_frame = train.h2o,
  nfolds = 8,
  ntrees = 100000,
  learn_rate = 0.001,
  learn_rate_annealing = 1,
  max_depth = 10,
  min_rows = 5,
  sample_rate = 0.75,
  col_sample_rate = 1,
  stopping_rounds = 10,
  stopping_tolerance = 0,
  seed = 3213
)

#Visualization
h2o.varimp_plot(h2o.final, num_of_features = 10)

# get a few observations to perform local interpretation on
local_obs <- ames_train[1:4, ]
explainer <- lime(ames_train, h2o.final)
explanation <- explain(local_obs, explainer, n_features = 5)
plot_features(explanation)






# convert test set to h2o object
test.h2o <- as.h2o(ames_test)

# evaluate performance on new data
h2o.performance(model = h2o.final, newdata = test.h2o)
# predict with h2o.predict
h2o.predict(h2o.final, newdata = test.h2o)
# predict values with predict
predict(h2o.final, test.h2o)
###############################################################################
###############################################################################
###############################################################################
###############################################################################
#GBM
# 使抽樣結果可以重複
set.seed(123)

# train GBM model
system.time(
  gbm.fit <- gbm(
    formula = `Ball-Version` ~ .,
    distribution = "gaussian",
    data = ames_train,
    n.trees = 3000, # 總迭代次數
    interaction.depth = 1, # 弱模型的切割數
    shrinkage = 0.001, # 學習步伐
    cv.folds = 3, # cross validation folds
    n.cores = NULL, # will use all cores by default
    verbose = FALSE
  )  
)
#我們也可以透過以下方式將GBMs找尋最佳迭代數的過程繪出：(其中黑線的為訓練誤差(train.error)，綠線為cv.error, 若method使用“test”，則會有紅線表示valid.error)
gbm.perf(object = gbm.fit, plot.it = TRUE,method = "cv")
gbm.perf(object = gbm.fit, method = "cv")
#因為手動調整參數是沒效率的，我們來建立hyperparameters grid並自動套用grid search。
# create hyperparameter grid
hyper_grid <- expand.grid(
  shrinkage = c(.01, .1, .3), # 學習步伐
  interaction.depth = c(1, 3, 5), # 模型切割數
  n.minobsinnode = c(5, 10, 15), # 節點最小觀測值個數
  bag.fraction = c(.65, .8, 1), # 使用隨機梯度下降(<1)
  optimal_trees = 0,               # 儲存最適模型樹的欄位
  min_RMSE = 0                     # 儲存最小均方差的欄位
)
#####################################################################
# randomize data
random_index <- sample(1:nrow(ames_train), nrow(ames_train))
random_ames_train <- ames_train[random_index, ]
# grid search 
for(i in 1:nrow(hyper_grid)) {
  
  # reproducibility
  set.seed(123)
  
  # train model
  gbm.tune <- gbm(
    formula = `Ball-Version` ~ .,
    distribution = "gaussian",
    data = random_ames_train,
    n.trees = 5000, # 使用5000個樹模型
    interaction.depth = hyper_grid$interaction.depth[i],
    shrinkage = hyper_grid$shrinkage[i],
    n.minobsinnode = hyper_grid$n.minobsinnode[i],
    bag.fraction = hyper_grid$bag.fraction[i],
    train.fraction = .75, # 使用75%的訓練資料，並用剩餘資料做OOB成效評估/驗證
    n.cores = NULL, # will use all cores by default
    verbose = FALSE
  )
  
  # 將每個GBM模型最小的模型代號和對應的驗證均方誤差(valid RMSE)回傳到
  hyper_grid$optimal_trees[i] <- which.min(gbm.tune$valid.error)
  hyper_grid$min_RMSE[i] <- sqrt(min(gbm.tune$valid.error))
}


hyper_grid %>% 
  dplyr::arrange(min_RMSE) %>%
  head(10)



#############
# 根據上一部的結果，調整參數區間與數值
hyper_grid_2 <- expand.grid(
  shrinkage = c(.01, .05, .1), # 聚焦更小的學習步伐
  interaction.depth = c(3, 5, 7), #聚焦>1的切割數
  n.minobsinnode = c(5, 7, 10), # 聚焦更小的節點觀測值數量
  bag.fraction = c(.65, .8, 1), # 不變
  optimal_trees = 0,               # a place to dump results
  min_RMSE = 0                     # a place to dump results
)
#我們再一次的用for loop迴圈執行以上81種超參數組合的模型，找出每一次最適的模型與對應的最小誤差
# grid search 
for(i in 1:nrow(hyper_grid_2)) {
  
  # reproducibility
  set.seed(123)
  
  # train model
  gbm.tune <- gbm(
    formula = `Ball-Version` ~ .,
    distribution = "gaussian",
    data = random_ames_train,
    n.trees = 6000,
    interaction.depth = hyper_grid_2$interaction.depth[i],
    shrinkage = hyper_grid_2$shrinkage[i],
    n.minobsinnode = hyper_grid_2$n.minobsinnode[i],
    bag.fraction = hyper_grid_2$bag.fraction[i],
    train.fraction = .75, # 使用剩餘的25%資料估計OOB誤差
    n.cores = NULL, # will use all cores by default
    verbose = FALSE
  )
  
  # add min training error and trees to grid
  hyper_grid_2$optimal_trees[i] <- which.min(gbm.tune$valid.error)
  hyper_grid_2$min_RMSE[i] <- sqrt(min(gbm.tune$valid.error))
}

#檢視結果
hyper_grid_2 %>% 
  dplyr::arrange(min_RMSE) %>%
  head(10)
#我們也可以透過以下方式將GBMs找尋最佳迭代數的過程繪出：(其中黑線的為訓練誤差(train.error)，綠線為cv.error, 若method使用“test”，則會有紅線表示valid.error)
# train GBM model
system.time(
  gbm.fit3 <- gbm(
    formula = `Ball-Version` ~ .,
    distribution = "gaussian",
    data = ames_train,
    n.trees = 1250, # 總迭代次數
    interaction.depth = 7, # 弱模型的切割數
    n.minobsinnode = 5,
    bag.fraction = 1,
    shrinkage = 0.001, # 學習步伐
    cv.folds = 3, # cross validation folds
    n.cores = NULL, # will use all cores by default
    verbose = FALSE
  )  
)
gbm.perf(object = gbm.fit3, plot.it = TRUE,method = "cv")
# for reproducibility
set.seed(123)

system.time(
  # train GBM model
  gbm.fit.final <- gbm(
    formula = `Ball-Version` ~ .,
    distribution = "gaussian",
    data = ames_train,
    n.trees = 1250,
    interaction.depth = 7,
    shrinkage = 0.01,
    n.minobsinnode = 7,
    bag.fraction = 1, 
    train.fraction = 1, # 如果使用<1(xx%)的訓練比例，就會用剩餘的(1-XX%)資料估計OOB誤差
    cv.folds = 4, # 有別於使用OOB估計誤差，我們估計更穩健的CV誤差
    n.cores = NULL, # will use all cores by default
    verbose = FALSE
  )
)
#最佳模型的cv誤差如下
sqrt(min(gbm.fit.final$cv.error))
#視覺化
vip::vip(gbm.fit.final)
#Partial dependence plots(部份依賴圖)
#一旦識別出最重要的幾個變數後，下一步就是去了解當解釋變數變動時，目標變數是如何變動的(即marginal effects，邊際效果，每變動一單位解釋變數時，對目標變數的影響)。我們可以使用partial dependence plots(PDPs)和individual conditional expectation(ICE)曲線。

gbm.fit.final %>%
  partial(object = .,# A fitted model object of appropriate class (e.g., "gbm", "lm", "randomForest", "train", etc.).
          pred.var = "Ball-Size(Y)", 
          n.trees = gbm.fit.final$n.trees, # 如果是gbm的話，需指定模型所使用樹個數
          grid.resolution = 100) %>%
  # The autplot function can be used to produce graphics based on ggplot2
  autoplot(rug = TRUE, train = ames_train) + # plotPartial()不支援gbm
  scale_y_continuous(labels = scales::dollar) # 因為是使用ggplot基礎繪圖，故可以使用ggplot相關圖層來調整

####################################
####################################
####################################
####################################
#LIME
#LIME是一種新程序幫助我們了解，單一觀察值的預測目標值是如何產生的。對gbm物件使用lime套件，我們需要定義模型類型model type和預測方法prediction methods。

model_type.gbm <- function(x, ...) {
  return("regression")
}

predict_model.gbm <- function(x, newdata, ...) {
  pred <- predict(x, newdata, n.trees = x$n.trees)
  return(as.data.frame(pred))
}
# get a few observations to perform local interpretation on
local_obs <- ames_test[1:2, ]

# apply LIME
explainer <- lime(
  x=ames_train, # The training data used for training the model that should be explained.
  model = gbm.fit.final # The model whose output should be explained
)
# 一旦使用lime()創建好了explainer，則可將explainer用作解釋模型作用在新觀察值的結果
explanation <- explain(x = local_obs, # New observations to explain
                       explainer = explainer,
                       n_features = 5 # The number of features to use for each explanation.
)
plot_features(explanation = explanation)
#####################################################
#####################################################
#####################################################
#####################################################
# predict values for test data
pred <- predict(gbm.fit.final, n.trees = gbm.fit.final$n.trees, ames_test)


