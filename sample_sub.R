library(readr)
library(triangle)

sample_sub <- read_csv("sample_submission.csv")

preds <- rtriangle(n = nrow(sample_sub), a = 0, b = 1, c = 0.5)

ret <- data.frame(id = sample_sub$id, target = preds)

write_csv(ret, "random_guess.csv")
