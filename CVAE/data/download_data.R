system("export _JAVA_OPTIONS='-Xmx3g'")

# Login into UDG http://meteo.unican.es/udg-tap/home
library(loadeR)
library(transformeR)
library(magrittr)

loginUDG('***', '***')

# Location and temporal selection
longitude <- c(-10, 32)
latitude <- c(36, 72)
time <- 1979:2008

# Predictor's variables
variables <- c('z@500','z@700','z@850','z@1000',
               'hus@500','hus@700','hus@850','hus@1000',
               'ta@500','ta@700','ta@850','ta@1000',
               'ua@500','ua@700','ua@850','ua@1000',
               'va@500','va@700','va@850','va@1000')

# Download and save predictor (ERA-Interim)
x <- lapply(variables, function(x) {
                loadGridData(dataset = 'ECMWF_ERA-Interim-ESD',
                     var = x,
                     lonLim = longitude,
                     latLim = latitude,
                     years = time)
      }) %>% makeMultiGrid()

save(x, file = paste0('./x.rda'))

# Download and save predictand (EWEMBI)
y <- loadGridData(dataset = 'E-OBS_v14_0.50regular',
                  var = 'pr',
                  lonLim = longitude,
                  latLim = latitude,
                  years = time)

save(y, file = paste0('./y.rda'))

