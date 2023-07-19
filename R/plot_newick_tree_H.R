#!/usr/bin/env Rscript
library(ggtree)
args <- commandArgs(trailingOnly = TRUE)

tree <- read.tree(text = args[1])

p <- ggtree(tree, layout = "rectangular") +
  geom_tippoint() +
  geom_tiplab(size = 4, angle = 135, offset = 0.01, hjust = 0.5) +
  vexpand(0.01, direction = -1)

ggsave(args[2], device = "png", width = 9, height = 10,  limitsize = FALSE)