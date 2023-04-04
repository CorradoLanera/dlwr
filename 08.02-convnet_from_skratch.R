library(reticulate)
library(tensorflow)
library(keras)
library(fs)
library(glue)

# accept competition rules at http://www.kaggle.com/c/dogs-vs-cats/rules
system("kaggle competitions download -c dogs-vs-cats")
zip::unzip(
  "dogs-vs-cats.zip",
  exdir = "dogs-vs-cats",
  files = "train.zip"
)
zip::unzip("dogs-vs-cats/train.zip", exdir = "dogs-vs-cats")

original_dir <- path("dogs-vs-cats/train")
new_base_dir <- path("dogs-vs-cats_small")

make_subset <- function(subset_name, start_index, end_index) {
  for (category in c("dog", "cat")) {
    file_name <- glue::glue("{category}.{start_index:end_index}.jpg")
    dir_create(new_base_dir / subset_name / category)
    file_copy(
      original_dir / file_name,
      new_base_dir / subset_name / category /  file_name
    )
  }
}

make_subset("train", start_index = 1, end_index = 1000)
make_subset("validation", start_index = 1001, end_index = 1500)
make_subset("test", start_index = 1501, end_index = 2500)
