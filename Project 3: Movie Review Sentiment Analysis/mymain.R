library('text2vec')
library('glmnet')

set.seed(5178)

#####################################
# Load libraries
# Load your vocabulary and training data
#####################################
myvocab <- scan(file = "myvocab.txt", what = character())
train <- read.table("train.tsv", stringsAsFactors = FALSE,
                    header = TRUE)

#####################################
# Train a binary classification model
#####################################
train$review <- gsub('<.*?>', ' ', train$review)
it_train = itoken(train$review,
                  preprocessor = tolower, 
                  tokenizer = word_tokenizer)
vectorizer = vocab_vectorizer(create_vocabulary(myvocab, 
                                                ngram = c(1L, 2L)))
dtm_train = create_dtm(it_train, vectorizer)

# cross validation to get best lambda
cv <- cv.glmnet(x = dtm_train, 
          y = train$sentiment, 
          family = 'binomial', 
          alpha = 1,
          type.measure = "auc",
          nfolds = 10)

# train
model <- glmnet(x = dtm_train, 
               y = train$sentiment,
               family = 'binomial',
               alpha = 1,
               type.measure = "auc",
               lambda = cv$lambda.min)


#####################################
# Load test data, and 
# Compute prediction
#####################################
test <- read.table("test.tsv", stringsAsFactors = FALSE,
                   header = TRUE)
test$review <- gsub('<.*?>', ' ', test$review)
it_train = itoken(test$review,
                  preprocessor = tolower, 
                  tokenizer = word_tokenizer)
vectorizer = vocab_vectorizer(create_vocabulary(myvocab, 
                                                ngram = c(1L, 2L)))
dtm_test = create_dtm(it_train, vectorizer)

preds <- predict(model,
                 dtm_test,
                 s = model$lambda,
                 type = 'response')


#####################################
# Store your prediction for test data in a data frame
# "output": col 1 is test$id
#           col 2 is the predicted probs
#####################################
output <- data.frame(test$id, preds)
names(output) <- c('id', 'prob')
write.table(output, file = "mysubmission.txt", 
            row.names = FALSE, sep='\t')
