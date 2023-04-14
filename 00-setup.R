install.packages(
  c("tensorflow", "keras", "tidyverse", "fs", "reticulate", "zip"),
  dependencies = TRUE
)


reticulate::install_miniconda()
keras::install_keras("conda", tensorflow = "gpu")

library(fs)
dir_create("~/.kaggle")
file_move("~/../Downloads/kaggle.json", "~/.kaggle")

py_install("kaggle", pip = TRUE)
