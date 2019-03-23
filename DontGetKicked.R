
options(java.parameters = "-Xmx4128m")

require(pacman)
p_load(
  streamlineR,
  infer,
  janitor,
  drake,
  esquisse,
  DataExplorer,
  woeR,
  woeBinning,
  woe,
  caret,
  h2o,
  rsparkling,
  sparklyr,
  rattle,
  mlr,
  ranger,
  automl,
  gamlr,
  bartMachine,
  vtreat,
  glmnet,
  elasticnet,
  DMwR,
  smotefamily,
  lars,
  xgboost,
  greybox, 
  lime,
  ggplot2,
  elasticnet,
  tidyverse,
  RJDBC,
  DBI,
  catboost,
  wrapr,
  Hmisc,
  GAMBoost,
  ade4
)

set.seed(1123)

# Adress to zip data
zipfile <- "C:/Users/Owner/Documents/Data Science/Project1_Dont_Get_Kicked/training.zip"
# Create a name for the dir where we'll unzip
zipdir <- tempfile()
# Create the dir using that name
dir.create(zipdir)
# Unzip the file into the dir
unzip(zipfile, exdir = zipdir)
# Get the files into the dir
files <- list.files(zipdir)
# Notify if there's more than one file
if (length(files) > 1)
  stop("More than one data file inside zip")
# Get the full name of the file
file <- paste(zipdir, files[1], sep = "/")
# Read the file
DGK_df <- data.table::fread(file, 
                            sep = "auto")

## What does the table look like
DGK_df %>% glimpse()

## What lets create some time factors and convert numerics read as strings 
## back to numeric

DGK_df <- DGK_df %>% 
  dplyr::mutate(PurchDate2 = lubridate::mdy(PurchDate),
                purchDay = lubridate::day(PurchDate2),
                PurchMon = lubridate::month(PurchDate2),
                PurchYear = lubridate::year(PurchDate2)) %>% 
  dplyr::select(-PurchDate, -PurchDate2, -RefId) %>% 
  mutate(MMRAcquisitionAuctionAveragePrice2 = as.numeric(MMRAcquisitionAuctionAveragePrice),
         MMRAcquisitionAuctionAveragePrice2 = as.integer(MMRAcquisitionAuctionAveragePrice),
         MMRAcquisitionAuctionCleanPrice2 = as.integer(MMRAcquisitionAuctionCleanPrice),
         MMRAcquisitionRetailAveragePrice2 = as.integer(MMRAcquisitionRetailAveragePrice),
         MMRAcquisitonRetailCleanPrice2 = as.integer(MMRAcquisitonRetailCleanPrice),
         MMRCurrentAuctionAveragePrice2 = as.integer(MMRCurrentAuctionAveragePrice),
         MMRCurrentAuctionCleanPrice2 = as.integer(MMRCurrentAuctionCleanPrice),
         MMRCurrentRetailAveragePrice2 = as.integer(MMRCurrentRetailAveragePrice),
         MMRCurrentRetailCleanPrice2 = as.integer(MMRCurrentRetailCleanPrice),
         VNZIP2 = as.factor(VNZIP1),
         IsBadBuy_factor = as.factor(IsBadBuy)) %>% 
  dplyr::select(-MMRAcquisitionAuctionAveragePrice:-MMRCurrentRetailCleanPrice) %>% 
  mutate_if(is.character, as.factor) %>% 
  select(-VNZIP1)


## Investigate outcome balance
## <12% in binomial classification should indicate rare event
DGK_df %>% 
  dplyr::group_by(IsBadBuy) %>% 
  dplyr::summarise(rcnt = n()) %>% 
  dplyr::mutate(prop = rcnt/sum(rcnt)) 

## Another nice summary of dataset
Hmisc::contents(DGK_df)

## summary of dataframe
DGK_df %>% summary()


## Train/validation data split 80-20
obs_cnt <- base::nrow(DGK_df)
train_split <- 0.8
validate_split <- 1-train_split
training_int <- sample(obs_cnt, train_split*obs_cnt)

## Training dataset
training_df <- DGK_df[training_int,] %>% 
               dplyr::select(-IsBadBuy)
training_df %>% glimpse()

training_df %>% 
  group_by(IsBadBuy_factor) %>% 
  summarise(rcnt = n()) %>% 
  mutate(prop = rcnt/sum(rcnt)) 

## Validation dataset
validate_df <- DGK_df[-training_int,]%>% 
  dplyr::select(-IsBadBuy)
validate_df %>% glimpse()

validate_df %>% 
  group_by(IsBadBuy_factor) %>% 
  summarise(rcnt = n()) %>% 
  mutate(prop = rcnt/sum(rcnt)) 

## Parameterize outcome column, position in column
## dataset and input columns as well
## H2O requires factor outcome variable
outcome_col <- "IsBadBuy_factor"

output_posn <- which(outcome_col == base::colnames(training_df))

input_cols <- base::colnames(training_df)[-output_posn]

## Desired output
# categorical_var  num_levels
#              w   20
#              y   2
#              z   4

## Dataframe containing all categorical variables 
## and the number of levels in each

cat_levels_train_df <- purrr::map_dfr(1:ncol(DGK_df), function(a) {
  if (class(DGK_df[, a]) %in% c("factor"))
    data_frame(categorical_var = names(DGK_df)[a],
               num_levels = length(table(DGK_df[, a])))
}) %>% arrange(desc(num_levels))

# DGK_df %>% glimpse()

## Discretize numeric variable using
## woe decision tree
tree_binning <- woe.tree.binning(training_df,
                                 outcome_col,
                                 input_cols)

## Quality of bins
woe.binning.plot(tree_binning)
woe.binning.table(tree_binning)

## Implement discretization tree on training dataset
training_woe_bin_df <- woe.binning.deploy(training_df,
                                          min.iv.total = 0.02, ## Exclude columes with total IV < 0.02
                                          binning = tree_binning) %>% 
dplyr::select(outcome_col, dplyr::contains("binned"))

# training_woe_bin_df2 <- training_woe_bin_df %>% 
#   mutate_if(is.numeric, as.factor)


## Compare post-discretization categorical table with levels
cat_levels_train_woe_bin_df <- purrr::map_dfr(1:ncol(training_woe_bin_df), function(a) {
  if (class(training_woe_bin_df[, a]) %in% c("factor"))
    data_frame(categorical_var = names(training_woe_bin_df)[a],
               num_levels = length(table(training_woe_bin_df[, a])))
}) %>% arrange(desc(num_levels))

## Implement discretization tree on validation dataset
validate_woe_bin_df <- woe.binning.deploy(validate_df,
                                          min.iv.total = 0.02,
                                          binning = tree_binning) %>% 
  dplyr::select(outcome_col, dplyr::contains("binned"))


## Compare post-discretization categorical table with levels
cat_levels_valid_woe_bin_df <- purrr::map_dfr(1:ncol(validate_woe_bin_df), function(a) {
  if (class(validate_woe_bin_df[, a]) %in% c("factor"))
    data_frame(categorical_var = names(validate_woe_bin_df)[a],
               num_levels = length(table(validate_woe_bin_df[, a])))
}) %>% arrange(desc(num_levels))


## Parametrization of oucome and
## imput column w/ new woe tree
output_woe_posn <- which(outcome_col == base::colnames(training_woe_bin_df))

input_woe_cols <- base::colnames(training_woe_bin_df)[-output_woe_posn]

## initialize local h2o cluster
## 
h2o.init(nthreads = 2) ## nthreads = -1 uses all compute cores on machine

## Convert local r dataframe to hex dataset for h2o
h2o_training_woe_df <- h2o::as.h2o(training_woe_bin_df)

summary(h2o_training_woe_df)

h2o_validate_woe_df <- h2o::as.h2o(validate_woe_bin_df)

glm_hyper_params <- list(
  alpha = seq(from = 0, to = 1, by = 0.001),
  lambda = seq(from = 0, to = 1, by = 0.000001)
)

search_criteria <- list(
  strategy = "RandomDiscrete",
  max_runtime_secs = 28800,
  max_models = 50,
  stopping_metric = "AUC",
  stopping_tolerance = 0.001,
  stopping_rounds = 5,
  seed = 1234
)

glm_grid <-
  h2o.grid(
    algorithm = "glm",
    grid_id = "grd_glm",
    x = input_woe_cols,
    y = outcome_col,
    training_frame = h2o_training_woe_df,
    # validation_frame = ,
    nfolds = 5,
    keep_cross_validation_predictions = TRUE,
    family = "binomial",
    hyper_params = glm_hyper_params,
    search_criteria = search_criteria,
    stopping_metric = "AUC",
    stopping_tolerance = 1e-5,
    stopping_rounds = 5,
    seed = 1234
  )

summary(glm_grid)

glm_model_ids <- glm_grid@model_ids
glm_models <- lapply(glm_model_ids, function(id) {h2o.getModel(id)})

# models[[1]]
glm_models_sort <- h2o.getGrid(grid_id = "grd_glm", sort_by = "auc", decreasing = TRUE)

glm_models_best <- h2o.getModel(glm_models_sort@model_ids[[1]])

glm_models_best@allparameters

glm_models_best@model$validation_metrics@metrics

perf_glm_best <- h2o.performance(glm_models_best, h2o_validate_woe_df)

plot(perf_glm_best, type="roc", main="ROC Curve for Best Logistic Regression Model")

glm_models_best@allparameters


gbm_hyper_params <- list(
  ntrees = 100,
  ## early stopping
  max_depth = 5:10,
  min_rows = c(20, 50, 100), #c(1, 5, 10, 20, 50, 100),
  learn_rate = c(0.001, 0.01, 0.1),
  learn_rate_annealing = c(0.99, 0.999, 1),
  sample_rate = c(0.7, 1),
  col_sample_rate = c(0.7, 1),
  nbins = c(30, 100, 300),
  nbins_cats = c(64, 256, 1024)
)

models_gbm <-
  h2o.grid(
    algorithm = "gbm",
    grid_id = "grd_gbm",
    x = input_woe_cols,
    y = outcome_col,
    training_frame = h2o_training_woe_df,
    # validation_frame = income_h2o_valid,
    nfolds = 5,
    keep_cross_validation_predictions = TRUE,
    hyper_params = gbm_hyper_params,
    search_criteria = search_criteria,
    stopping_metric = "AUC",
    stopping_tolerance = 1e-3,
    stopping_rounds = 0,
    seed = 1234
  )

gbm_models_sort <- h2o.getGrid(grid_id = "grd_gbm", sort_by = "auc", decreasing = TRUE)
gbm_models_best <- h2o.getModel(gbm_models_sort@model_ids[[1]])
perf_gbm_best <- h2o.performance(gbm_models_best, h2o_validate_woe_df)

plot(perf_glm_best, type="roc", main="ROC Curve for Best GBM Model")


rf_hyper_params <- list(
  ntrees = 100, ## early stopping
  max_depth = 5:10,
  min_rows = c(50, 100),
  nbins = c(30, 100, 300),
  nbins_cats = c(64, 256, 1024),
  sample_rate = c(0.7, 1),
  mtries = c(-1, 2, 6)
)


models_rf <-
  h2o.grid(
    algorithm = "randomForest",
    grid_id = "grd_rf",
    x = input_woe_cols,
    y = outcome_col,
    training_frame = h2o_training_woe_df, # validation_frame = income_h2o_valid,
    nfolds = 5,
    keep_cross_validation_predictions = TRUE,
    hyper_params = rf_hyper_params,
    search_criteria = search_criteria,
    stopping_metric = "AUC",
    stopping_tolerance = 1e-3,
    stopping_rounds = 2,
    seed = 1234
  )

rf_models_sort <- h2o.getGrid(grid_id = "grd_rf", sort_by = "auc", decreasing = TRUE)
rf_models_best <- h2o.getModel(rf_models_sort@model_ids[[1]])
perf_rf_best <- h2o.performance(rf_models_best, h2o_validate_woe_df)

plot(perf_glm_best, type="roc", main="ROC Curve for Best Random Forest Model")


NNet_hyper_params <- list(activation = c("Rectifier", "Maxout", "Tanh", "RectifierWithDropout", "MaxoutWithDropout", "TanhWithDropout"), 
                     hidden = list(c(50, 50, 50, 50), c(200, 200), c(200, 200, 200), c(200, 200, 200, 200)), 
                     epochs = c(50, 100, 200), 
                     l1 = c(0, 0.00001, 0.0001), 
                     l2 = c(0, 0.00001, 0.0001), 
                     adaptive_rate = c(TRUE, FALSE), 
                     rate = c(0, 0.1, 0.005, 0.001), 
                     rate_annealing = c(1e-8, 1e-7, 1e-6), 
                     rho = c(0.9, 0.95, 0.99, 0.999), 
                     epsilon = c(1e-10, 1e-8, 1e-6, 1e-4), 
                     momentum_start = c(0, 0.5),
                     momentum_stable = c(0.99, 0.5, 0), 
                     input_dropout_ratio = c(0, 0.1, 0.2)
)

models_NNet <-
  h2o.grid(
    algorithm = "deeplearning",
    grid_id = "grd_dl",
    x = input_woe_cols,
    y = outcome_col,
    training_frame = h2o_training_woe_df,
    # validation_frame = income_h2o_valid,
    nfolds = 5,
    hyper_params = NNet_hyper_params,
    search_criteria = search_criteria,
    stopping_metric = "AUC",
    stopping_tolerance = 1e-3,
    stopping_rounds = 2,
    seed = 1234
  )



glm_dodel_impl <-
  h2o.glm(
    x = input_woe_cols,
    y = outcome_col,
    training_frame = h2o_training_woe_df,
    nfolds = 10,
    fold_assignment = "Modulo",
    keep_cross_validation_predictions = TRUE,
    seed = 1234,
    family = "binomial",
    alpha = glm_models_best@allparameters$alpha,
    lambda = glm_models_best@allparameters$lambda
  )

rf_dodel_impl <-
  h2o.randomForest(
    x = input_woe_cols,
    y = outcome_col,
    training_frame = h2o_training_woe_df,
    nfolds = 10,
    fold_assignment = "Modulo",
    keep_cross_validation_predictions = TRUE,
    ntrees = rf_models_best@allparameters$ntrees,
    max_depth = rf_models_best@allparameters$max_depth,
    min_rows = rf_models_best@allparameters$min_rows,
    nbins = rf_models_best@allparameters$nbins,
    nbins_cats = rf_models_best@allparameters$nbins_cats,
    mtries = rf_models_best@allparameters$mtries,
    sample_rate = rf_models_best@allparameters$sample_rate,
    stopping_metric = "AUC",
    stopping_tolerance = rf_models_best@allparameters$stopping_tolerance,
    stopping_rounds = rf_models_best@allparameters$stopping_rounds,
    seed = rf_models_best@allparameters$seed
  )

gbm_model_impl <-
  h2o.gbm(
    x = input_woe_cols,
    y = outcome_col,
    training_frame = h2o_training_woe_df,
    nfolds = 10,
    fold_assignment = "Modulo",
    keep_cross_validation_predictions = TRUE,
    ntrees = gbm_models_best@allparameters$ntrees,
    max_depth = gbm_models_best@allparameters$max_depth,
    min_rows = gbm_models_best@allparameters$min_rows,
    nbins = gbm_models_best@allparameters$nbins,
    nbins_cats = gbm_models_best@allparameters$nbins_cats,
    learn_rate = gbm_models_best@allparameters$learn_rate,
    learn_rate_annealing = gbm_models_best@allparameters$learn_rate_annealing,
    sample_rate = gbm_models_best@allparameters$sample_rate,
    col_sample_rate = gbm_models_best@allparameters$col_sample_rate,
    stopping_metric = "AUC",
    stopping_tolerance = gbm_models_best@allparameters$stopping_tolerance,
    stopping_rounds = gbm_models_best@allparameters$stopping_rounds,
    seed = gbm_models_best@allparameters$seed
  )

Enstack_model_impl <-
  h2o.stackedEnsemble(
    x = input_woe_cols,
    y = outcome_col,
    training_frame = h2o_training_woe_df,
    base_models = list(
      glm_dodel_impl@model_id,
      rf_dodel_impl@model_id,
      gbm_model_impl@model_id
    )
  )

h2o.auc(h2o.performance(glm_dodel_impl, h2o_validate_woe_df))

h2o.auc(h2o.performance(rf_dodel_impl, h2o_validate_woe_df))

h2o.auc(h2o.performance(gbm_model_impl, h2o_validate_woe_df))

h2o.auc(h2o.performance(Enstack_model_impl, h2o_validate_woe_df))

h2o.varimp_plot(glm_dodel_impl, num_of_features = 20)
h2o.varimp_plot(rf_dodel_impl, num_of_features = 20)
h2o.varimp_plot(gbm_model_impl, num_of_features = 20)

###------------------------------------

# Adress to zip data
zipfile2 <- "C:/Users/Owner/Documents/Data Science/Project1_Dont_Get_Kicked/test.zip"
# Create a name for the dir where we'll unzip
zipdir2 <- tempfile()
# Create the dir using that name
dir.create(zipdir2)
# Unzip the file into the dir
unzip(zipfile2, exdir = zipdir2)
# Get the files into the dir
files2 <- list.files(zipdir2)
# Notify if there's more than one file
if (length(files2) > 1)
  stop("More than one data file inside zip")
# Get the full name of the file
file2 <- paste(zipdir2, files2[1], sep = "/")
# Read the file
test_df <- data.table::fread(file2, 
                             sep = "auto")

test_df %>% glimpse()


test_df <- test_df %>% 
  dplyr::mutate(PurchDate2 = lubridate::mdy(PurchDate),
                purchDay = lubridate::day(PurchDate2),
                PurchMon = lubridate::month(PurchDate2),
                PurchYear = lubridate::year(PurchDate2)) %>% 
  dplyr::select(-PurchDate, -PurchDate2, -RefId) %>% 
  mutate(MMRAcquisitionAuctionAveragePrice2 = as.numeric(MMRAcquisitionAuctionAveragePrice),
         MMRAcquisitionAuctionAveragePrice2 = as.integer(MMRAcquisitionAuctionAveragePrice),
         MMRAcquisitionAuctionCleanPrice2 = as.integer(MMRAcquisitionAuctionCleanPrice),
         MMRAcquisitionRetailAveragePrice2 = as.integer(MMRAcquisitionRetailAveragePrice),
         MMRAcquisitonRetailCleanPrice2 = as.integer(MMRAcquisitonRetailCleanPrice),
         MMRCurrentAuctionAveragePrice2 = as.integer(MMRCurrentAuctionAveragePrice),
         MMRCurrentAuctionCleanPrice2 = as.integer(MMRCurrentAuctionCleanPrice),
         MMRCurrentRetailAveragePrice2 = as.integer(MMRCurrentRetailAveragePrice),
         MMRCurrentRetailCleanPrice2 = as.integer(MMRCurrentRetailCleanPrice) ) %>% 
  dplyr::select(-MMRAcquisitionAuctionAveragePrice:-MMRCurrentRetailCleanPrice) %>% 
  mutate_if(is.character, as.factor) %>% 
  mutate(VNZIP2 = as.factor(VNZIP1)) %>% 
  select(-VNZIP1)

## Implement discretization tree on training dataset
test_woe_bin_df <- woe.binning.deploy(test_df,
                                          min.iv.total = 0.02, ## Exclude columes with total IV < 0.02
                                          binning = tree_binning) %>% 
  dplyr::select(dplyr::contains("binned"))

test_woe_bin_df %>% glimpse()

h2o_test_woe_df <- h2o::as.h2o(test_woe_bin_df)

test_pred_glm <- h2o.predict(object = glm_dodel_impl , newdata = h2o_test_woe_df) %>% as.data.frame()
test_pred_glm %>% glimpse()

test_pred_rf <- h2o.predict(object = rf_dodel_impl, newdata = h2o_test_woe_df) %>% as.data.frame()

test_pred_gbm <- h2o.predict(object = gbm_model_impl, newdata = h2o_test_woe_df) %>% as.data.frame()

test_pred_enstack <- h2o.predict(object = Enstack_model_impl, newdata = h2o_test_woe_df) %>% as.data.frame()


data.frame(RefId = test_df$RefId,  IsBadBuy = test_pred_glm$predict) %>% data.table::fwrite("C:/Users/Owner/Documents/Data Science/Project1_Dont_Get_Kicked/glm_submission_unbal.csv")

data.frame(RefId = test_df$RefId, IsBadBuy = test_pred_rf$predict) %>% data.table::fwrite("C:/Users/Owner/Documents/Data Science/Project1_Dont_Get_Kicked/rf_submission_unbal.csv")

data.frame(RefId = test_df$RefId, IsBadBuy = test_pred_gbm$predict) %>% data.table::fwrite("C:/Users/Owner/Documents/Data Science/Project1_Dont_Get_Kicked/gbm_submission_unbal.csv")
  
data.frame(RefId = test_df$RefId, IsBadBuy = test_pred_enstack$predict) %>% data.table::fwrite("C:/Users/Owner/Documents/Data Science/Project1_Dont_Get_Kicked/enstack_submission_unbal.csv") 




# test_df %>% 
#   group_by(IsBadBuy) %>% 
#   summarise(rcnt = n()) %>% 
#   mutate(prop = rcnt/sum(rcnt)) 

# Hmisc::contents(test_df)

# test_df %>% summary()

