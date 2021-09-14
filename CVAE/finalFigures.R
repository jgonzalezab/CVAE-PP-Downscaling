# Load libraries
library(loadeR)
library(transformeR) 
library(downscaleR)
library(visualizeR)
library(climate4R.value)
library(magrittr)
library(gridExtra)
library(grid)
library(RColorBrewer)
library(sp)
library(downscaleR.keras)
library(loadeR.2nc)
library(reticulate)

# Load numpy
np <- import("numpy", convert = FALSE)

# Number of random samples to plot for each model
nGenerations <- 8

# Load data y_test data
load(file = './data/y_test_template.RData')

obsIndex <- 415
cb <- brewer.pal(n = 11, 'BuPu')
cb <- cb %>% colorRampPalette()
at <- seq(min(subsetDimension(y_test, 'time', obsIndex)$Data, na.rm = TRUE),
		  max(subsetDimension(y_test, 'time', obsIndex)$Data, na.rm = TRUE), 0.01)

# cVAE Figure
figures <- list()

figures[[1]] <- spatialPlot(subsetDimension(y_test, 'time', obsIndex),
							backdrop.theme = 'coastline',
							main = list('Observation', cex = 5),
							col.regions = cb,
							at = at,
                			set.min = at[1], set.max = at[length(at)],
                			colorkey = FALSE)

for (i in 1:nGenerations) {

	y_pred_cVAE <- y_test
	y_pred_cVAE$Data <- py_to_r(np$load(paste0('./data/yPred_', i, '.npy')))
	attributes(y_pred_cVAE$Data)$dimensions <- c('time', 'lat', 'lon')

	figures[[i+1]] <- spatialPlot(subsetDimension(y_pred_cVAE, 'time', obsIndex),
							    backdrop.theme = 'coastline',
							    main = list('Stochastic Downscaled Field', cex = 5),
							    col.regions = cb,
							    at = at,
                			    set.min = at[1], set.max = at[length(at)], colorkey = FALSE)

}

file_name <- paste0('./figures/CVAE.pdf')

pdf(file = file_name, width = 50, height = 50)
lay <- rbind(c(1, NA, NA, NA),
			 c(2, 3, 4, 5),
			 c(6, 7, 8, 9))
grid.arrange(grobs = figures, ncol = 4, nrow = 4,
             layout_matrix = lay)

dev.off()


