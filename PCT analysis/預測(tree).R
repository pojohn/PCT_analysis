#https://www.jamleecute.com/decision-tree-cart-%e6%b1%ba%e7%ad%96%e6%a8%b9/
#讀入csv檔
library(readr)
library(rpart)
library(rpart.plot)
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
fit1 <- rpart(formula = `Ball-Version` ~ ., data = dfg, method = 'class')

rpart.plot(fit1, extra= 106)

fit2 <- rpart(formula = `Ball-Version` ~ `Ball-Size(X)`+`Ball-Size(Y)`+`Ball-Size(Z)`, data = BV, method = 'class')

rpart.plot(fit2, extra= 106)
