library(readr)
library(magrittr) # needs to be run every time you start R and want to use %>%
library(lubridate)
library(tidyverse)
library(dplyr)

set.seed(5178)

mypredict = function(){
  my_train <- preprocess(train)
  
  start_date <- ymd("2011-03-01") %m+% months(2 * (t - 1))
  end_date <- ymd("2011-05-01") %m+% months(2 * (t - 1))
  test_current <- test %>%
    filter(Date >= start_date & Date < end_date) #%>%
  #select(-IsHoliday)
  
  start_last_year = min(test_current$Date) - 375
  end_last_year = max(test_current$Date) - 350
  tmp_train <- my_train %>%
    filter(Date > start_last_year & Date < end_last_year) %>%
    mutate(Wk = ifelse(year(Date) == 2010, week(Date)-1, week(Date))) %>%
    rename(Weekly_Pred = Weekly_Sales) %>%
    select(-Date)
  
  test_current <- test_current %>%
    mutate(Wk = week(Date))
  
  # find the unique pairs of (Store, Dept) combo that appeared in both training and test sets
  train_pairs <- my_train[, 1:2] %>% count(Store, Dept) %>% filter(n != 0)
  test_pairs <- test_current[, 1:2] %>% count(Store, Dept) %>% filter(n != 0)
  unique_pairs <- intersect(train_pairs[, 1:2], test_pairs[, 1:2])
  
  # pick out the needed training samples, convert to dummy coding, then put them into a list
  train_split <- unique_pairs %>% 
    left_join(my_train, by = c('Store', 'Dept')) %>% 
    mutate(Wk = factor(ifelse(year(Date) == 2010, week(Date) - 1, week(Date)), levels = 1:52)) %>% 
    mutate(Yr = year(Date))
  train_split = as_tibble(model.matrix(~ Weekly_Sales + Store + Dept + Yr + Wk, train_split)) %>% group_split(Store, Dept)
  
  # do the same for the test set
  test_split <- unique_pairs %>% 
    left_join(test_current, by = c('Store', 'Dept')) %>% 
    mutate(Wk = factor(ifelse(year(Date) == 2010, week(Date) - 1, week(Date)), levels = 1:52)) %>% 
    mutate(Yr = year(Date))
  test_split = as_tibble(model.matrix(~ Store + Dept + Yr + Wk, test_split)) %>% mutate(Date = test_split$Date) %>% group_split(Store, Dept)
  
  # pre-allocate a list to store the predictions
  test_pred <- vector(mode = "list", length = nrow(unique_pairs))
  
  # perform regression for each split, note we used lm.fit instead of lm
  for (i in 1:nrow(unique_pairs)) {
    tmp_train <- train_split[[i]]
    tmp_test <- test_split[[i]]
    
    # shift for fold 5
    if (t==5){
      shift = 1/7
      tmp_test[, "Wk51"] = tmp_test[, "Wk51"] * (1-shift) + 
        tmp_test[, "Wk52"] * shift
    } 
    
    
    mycoef <- lm.fit(as.matrix(tmp_train[, -(2:4)]), tmp_train$Weekly_Sales)$coefficients
    mycoef[is.na(mycoef)] <- 0
    tmp_pred <- mycoef[1] + as.matrix(tmp_test[, 4:55]) %*% mycoef[-1]
    
    test_pred[[i]] <- cbind(tmp_test[, 2:3], Date = tmp_test$Date, Weekly_Pred = tmp_pred[,1])
  }
  
  # turn the list into a table at once, 
  # this is much more efficient then keep concatenating small tables
  test_pred <- bind_rows(test_pred)
  return(test_pred)
}

preprocess <- function(df){
  #group by dept, unstack date
  my_df <- df %>% select(Dept, Store, Date, Weekly_Sales)
  my_df <- my_df %>% spread(Date, Weekly_Sales) %>% arrange(Dept) 
  my_df[is.na(my_df)] <- 0
  
  #put it in a loop and perform svd on every dept (1 to 99)
  df_total = data.frame()
  n_comp <- 8
  
  for (i in 1:99){
    dept <- my_df %>% filter(Dept == i)
    
    if (nrow(dept) > n_comp){
      dept <- as.matrix(dept[-c(1, 2)])
      a <- rowMeans(dept)
      dept <- dept - a
      svd <- svd(dept)
      mat <- svd$u[,1:n_comp] %*% diag(svd$d[1:n_comp]) %*% t(svd$v[,1:n_comp]) + a
    } else {
      mat <- as.matrix(dept[-c(1, 2)])
    }
    df_add <- as.data.frame(mat)
    colnames(df_add) <- colnames(my_df)[-c(1, 2)]
    df_total <- rbind(df_total, df_add)
  }
  df_total <- cbind(Dept = my_df$Dept, Store = my_df$Store, df_total)
  
  
  train_temp <- df_total %>% gather(Date, Weekly_Sales, -Store, -Dept) %>% mutate(Date = as.Date(Date))
  return(train_temp)
}
