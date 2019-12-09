library(corrplot)
df = read.csv("../../data/db_dump/SanFranciscoFire.csv",
              na.strings = "(null)")

df = subset(na.omit(df), select = -c(WeatherDescription, Year, HM,
                                     DOY, Fire))
pdf("../plot/corr_sf.pdf")
corrplot(cor(df), method="color")
invisible(dev.off())

df = read.csv("../../data/db_dump/NewYorkFire.csv", na.strings = "(null)")

df = subset(na.omit(df), select = -c(WeatherDescription, Year, HM,
                                     DOY, Fire))
pdf("../plot/corr_ny.pdf")
corrplot(cor(df), method="color")
invisible(dev.off())

df = read.csv("../../data/db_dump/AllFire.csv", na.strings = "(null)")

df = subset(na.omit(df), select = -c(WeatherDescription, City, Year, HM,
                                     DOY, Fire))
pdf("../plot/corr_all.pdf")
corrplot(cor(df), method="color")
invisible(dev.off())
