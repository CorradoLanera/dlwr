install.packages(
  c("tensorflow", "keras", "tidyverse", "fs", "reticulate", "zip"),
  dependencies = TRUE
)

library(reticulate)
virtualenv_create("r-reticulate", python = install_python())

library(keras)
install_keras(envname = "r-reticulate")

library(fs)
dir_create("~/.kaggle")
file_move("~/../Downloads/kaggle.json", "~/.kaggle")

py_install("kaggle", pip = TRUE)
