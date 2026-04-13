# ============================================================
# Predicting Weekly Respiratory Disease Burden Using Random Forest
# Module: CS5812 Predictive Data Analysis
#
# Description:
# This script uses processed weather, air quality, traffic,
# and respiratory disease datasets to build a Random Forest
# regression model in R.
#
# Target:
# Weekly total respiratory disease burden
# = Adenovirus + hMPV + Influenza + RSV
# ============================================================


# -----------------------------
# 1. Install packages (run once)
# -----------------------------
# install.packages("tidyverse")
# install.packages("lubridate")
# install.packages("randomForest")
# install.packages("Metrics")


# -----------------------------
# 2. Load libraries
# -----------------------------
library(tidyverse)
library(lubridate)
library(randomForest)
library(Metrics)


# -----------------------------
# 3. Set working directory
# -----------------------------
setwd("C:/Users/poorn/OneDrive/Desktop/New Data")


# -----------------------------
# 4. Read processed datasets
# -----------------------------
weather <- read.csv("Processed_Weather.csv")
air <- read.csv("Processed_Air_Quality.csv")
traffic <- read.csv("Processed_Traffic.csv")
resp <- read.csv("Processed_Respiratory_Diseases.csv")


# -----------------------------
# 5. Convert date columns
# -----------------------------
weather$time <- as.Date(weather$time)
air$time <- as.Date(air$time)
traffic$date <- as.Date(traffic$date)
resp$date <- as.Date(resp$date)


# -----------------------------
# 6. Aggregate region-level daily data
#    to national daily level
# -----------------------------
weather_daily <- weather %>%
  group_by(time) %>%
  summarise(
    Temp_Max = mean(temperature_2m_max, na.rm = TRUE),
    Temp_Min = mean(temperature_2m_min, na.rm = TRUE),
    Precipitation = mean(precipitation_sum, na.rm = TRUE),
    WindSpeed = mean(windspeed_10m_max, na.rm = TRUE),
    .groups = "drop"
  )

air_daily <- air %>%
  group_by(time) %>%
  summarise(
    PM10 = mean(pm10, na.rm = TRUE),
    PM2_5 = mean(pm2_5, na.rm = TRUE),
    NO2 = mean(nitrogen_dioxide, na.rm = TRUE),
    Ozone = mean(ozone, na.rm = TRUE),
    SO2 = mean(sulphur_dioxide, na.rm = TRUE),
    .groups = "drop"
  )

traffic_daily <- traffic %>%
  group_by(date) %>%
  summarise(
    Traffic_Volume = sum(vehicle_count, na.rm = TRUE),
    .groups = "drop"
  )


# -----------------------------
# 7. Merge daily predictors
# -----------------------------
daily_data <- weather_daily %>%
  left_join(air_daily, by = c("time" = "time")) %>%
  left_join(traffic_daily, by = c("time" = "date")) %>%
  rename(Date = time)


# -----------------------------
# 8. Create week start variable
#    so daily predictors can be
#    converted to weekly level
# -----------------------------
daily_data <- daily_data %>%
  mutate(week_start = floor_date(Date, unit = "week", week_start = 1))

resp <- resp %>%
  mutate(week_start = date)


# -----------------------------
# 9. Convert daily predictors
#    to weekly predictors
# -----------------------------
weekly_predictors <- daily_data %>%
  group_by(week_start) %>%
  summarise(
    Temp_Max = mean(Temp_Max, na.rm = TRUE),
    Temp_Min = mean(Temp_Min, na.rm = TRUE),
    Precipitation = mean(Precipitation, na.rm = TRUE),
    WindSpeed = mean(WindSpeed, na.rm = TRUE),
    PM10 = mean(PM10, na.rm = TRUE),
    PM2_5 = mean(PM2_5, na.rm = TRUE),
    NO2 = mean(NO2, na.rm = TRUE),
    Ozone = mean(Ozone, na.rm = TRUE),
    SO2 = mean(SO2, na.rm = TRUE),
    Traffic_Volume = mean(Traffic_Volume, na.rm = TRUE),
    .groups = "drop"
  )


# -----------------------------
# 10. Create target variable
#     total weekly respiratory burden
# -----------------------------
resp <- resp %>%
  mutate(
    Respiratory_Burden = Adenovirus + hMPV + Influenza + RSV
  )


# -----------------------------
# 11. Merge predictors with target
# -----------------------------
model_data <- weekly_predictors %>%
  inner_join(
    resp %>%
      select(
        week_start,
        Adenovirus,
        hMPV,
        Influenza,
        RSV,
        Respiratory_Burden
      ),
    by = "week_start"
  )


# -----------------------------
# 12. Check final dataset
# -----------------------------
print(dim(model_data))
print(head(model_data))
print(summary(model_data))


# -----------------------------
# 13. Optional:
#     choose target variable here
# -----------------------------
# You can change the target to:
# "Adenovirus", "hMPV", "Influenza", "RSV", or "Respiratory_Burden"

target_var <- "Respiratory_Burden"


# -----------------------------
# 14. Remove missing values
# -----------------------------
model_data <- na.omit(model_data)


# -----------------------------
# 15. Time-based train-test split
#     80% train, 20% test
# -----------------------------
split_index <- floor(0.8 * nrow(model_data))

train_data <- model_data[1:split_index, ]
test_data  <- model_data[(split_index + 1):nrow(model_data), ]


# -----------------------------
# 16. Build formula dynamically
# -----------------------------
predictor_vars <- c(
  "Temp_Max", "Temp_Min", "Precipitation", "WindSpeed",
  "PM10", "PM2_5", "NO2", "Ozone", "SO2", "Traffic_Volume"
)

rf_formula <- as.formula(
  paste(target_var, "~", paste(predictor_vars, collapse = " + "))
)


# -----------------------------
# 17. Train Random Forest model
# -----------------------------
set.seed(123)

rf_model <- randomForest(
  formula = rf_formula,
  data = train_data,
  ntree = 500,
  mtry = 3,
  importance = TRUE
)


# -----------------------------
# 18. Print model summary
# -----------------------------
print(rf_model)


# -----------------------------
# 19. Make predictions
# -----------------------------
rf_pred <- predict(rf_model, newdata = test_data)


# -----------------------------
# 20. Evaluate model
# -----------------------------
actual_values <- test_data[[target_var]]

rmse_value <- rmse(actual_values, rf_pred)
mae_value  <- mae(actual_values, rf_pred)
r2_value   <- cor(actual_values, rf_pred)^2

cat("====================================\n")
cat("Random Forest Model Performance\n")
cat("====================================\n")
cat("Target Variable :", target_var, "\n")
cat("RMSE :", round(rmse_value, 3), "\n")
cat("MAE  :", round(mae_value, 3), "\n")
cat("R²   :", round(r2_value, 3), "\n")


# -----------------------------
# 21. Variable importance
# -----------------------------
print(importance(rf_model))

varImpPlot(
  rf_model,
  main = paste("Variable Importance -", target_var)
)


# -----------------------------
# 22. Actual vs Predicted scatter plot
# -----------------------------
plot(
  actual_values,
  rf_pred,
  xlab = paste("Actual", target_var),
  ylab = paste("Predicted", target_var),
  main = paste("Actual vs Predicted -", target_var),
  pch = 19
)

abline(0, 1, col = "red", lwd = 2)


# -----------------------------
# 23. Save prediction results
# -----------------------------
results <- data.frame(
  Week = test_data$week_start,
  Actual = actual_values,
  Predicted = rf_pred
)

print(head(results))

write.csv(
  results,
  "RF_Weekly_Respiratory_Predictions.csv",
  row.names = FALSE
)


# -----------------------------
# 24. Plot actual vs predicted over time
# -----------------------------
ggplot(results, aes(x = Week)) +
  geom_line(aes(y = Actual, linetype = "Actual")) +
  geom_line(aes(y = Predicted, linetype = "Predicted")) +
  labs(
    title = paste("Actual vs Predicted Weekly", target_var),
    x = "Week",
    y = target_var,
    linetype = "Legend"
  ) +
  theme_minimal()


# -----------------------------
# 25. Example future prediction
#     Replace with real values
# -----------------------------
future_data <- data.frame(
  Temp_Max = c(14.5),
  Temp_Min = c(6.8),
  Precipitation = c(1.2),
  WindSpeed = c(18.4),
  PM10 = c(17.5),
  PM2_5 = c(10.8),
  NO2 = c(19.6),
  Ozone = c(31.4),
  SO2 = c(4.9),
  Traffic_Volume = c(16000000)
)

future_prediction <- predict(rf_model, newdata = future_data)

future_results <- data.frame(
  Predicted_Value = future_prediction
)

print(future_results)

write.csv(
  future_results,
  "RF_Future_Weekly_Respiratory_Prediction.csv",
  row.names = FALSE
)
write.csv(
  results,
  "RF_Weekly_Respiratory_Predictions.csv",
  row.names = FALSE
)

write.csv(
  future_results,
  "RF_Future_Weekly_Respiratory_Prediction.csv",
  row.names = FALSE
)
pdf("RF_Variable_Importance.pdf")
varImpPlot(
  rf_model,
  main = paste("Variable Importance -", target_var)
)
dev.off()
pdf("RF_Actual_vs_Predicted.pdf")
plot(
  actual_values,
  rf_pred,
  xlab = paste("Actual", target_var),
  ylab = paste("Predicted", target_var),
  main = paste("Actual vs Predicted -", target_var),
  pch = 19
)
abline(0, 1, col = "red", lwd = 2)
dev.off()
pdf("RF_Predicted_Over_Time.pdf")
ggplot(results, aes(x = Week)) +
  geom_line(aes(y = Actual, linetype = "Actual")) +
  geom_line(aes(y = Predicted, linetype = "Predicted")) +
  labs(
    title = paste("Actual vs Predicted Weekly", target_var),
    x = "Week",
    y = target_var,
    linetype = "Legend"
  ) +
  theme_minimal()
dev.off()
metrics_results <- data.frame(
  Target = target_var,
  RMSE = rmse_value,
  MAE = mae_value,
  R2 = r2_value
)

print(metrics_results)

write.csv(
  metrics_results,
  "RF_Model_Metrics.csv",
  row.names = FALSE
)
# -----------------------------
# 26. Save model metrics
# -----------------------------
metrics_results <- data.frame(
  Target = target_var,
  RMSE = rmse_value,
  MAE = mae_value,
  R2 = r2_value
)

print(metrics_results)

write.csv(
  metrics_results,
  "RF_Model_Metrics.csv",
  row.names = FALSE
)


# -----------------------------
# 27. Save variable importance plot
# -----------------------------
pdf("RF_Variable_Importance.pdf")
varImpPlot(
  rf_model,
  main = paste("Variable Importance -", target_var)
)
dev.off()


# -----------------------------
# 28. Save actual vs predicted scatter plot
# -----------------------------
pdf("RF_Actual_vs_Predicted.pdf")
plot(
  actual_values,
  rf_pred,
  xlab = paste("Actual", target_var),
  ylab = paste("Predicted", target_var),
  main = paste("Actual vs Predicted -", target_var),
  pch = 19
)
abline(0, 1, col = "red", lwd = 2)
dev.off()


# -----------------------------
# 29. Save actual vs predicted over time plot
# -----------------------------
pdf("RF_Predicted_Over_Time.pdf")
ggplot(results, aes(x = Week)) +
  geom_line(aes(y = Actual, linetype = "Actual")) +
  geom_line(aes(y = Predicted, linetype = "Predicted")) +
  labs(
    title = paste("Actual vs Predicted Weekly", target_var),
    x = "Week",
    y = target_var,
    linetype = "Legend"
  ) +
  theme_minimal()
dev.off()
dir.create("C:/Users/poorn/Desktop/New Data")
file.create("C:/Users/poorn/Desktop/New Data/RF_Weekly_Respiratory_Model.R")
