library(riem)
library(foreign)

# Get stations
nets = riem_networks()
riem_stations("LT__ASOS")

# Get data for 2016
data = riem_measures("EYVI", date_start = "2016-01-01", date_end="2016-12-31")
head(data)

# Write to csv
write.csv(data, file = "weather.csv")
