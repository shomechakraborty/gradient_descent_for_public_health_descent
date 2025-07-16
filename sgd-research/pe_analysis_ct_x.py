from sklearn.model_selection import TimeSeriesSplit
from neural_network_class import NeuralNetworkModel
from sklearn.neural_network import MLPRegressor
import math, statistics as stats, copy, random, numpy

def main():
    input_data_file_names = ["business_applications_processed.csv", "manufacturing_and_trade_inventories_and_sales_processed.csv", "construction_spending_rate_processed.csv", "advance_retail_and_food_sales_processed.csv", "new_manufacturer_shipments_inventories_and_orders_processed.csv", "international_goods_and_services_trade_processed.csv"]
    target_data_file_names = ["all_employees_health_care_and_social_assitance.csv", "business_applications_health_care_and_social_assistance.csv", "total_construction_spending_health_care.csv"]
    input_data, _ = process_data(input_data_file_names, target_data_file_names[0], 2005, 2025)
    _, employee_target_data = process_data(input_data_file_names, target_data_file_names[0], 2005, 2025)
    _, business_applications_target_data = process_data(input_data_file_names, target_data_file_names[1], 2005, 2025)
    _, total_construction_spending_target_data = process_data(input_data_file_names, target_data_file_names[2], 2005, 2025)
    print("TARGET DATA: EMPLOYEES")
    run_evaluation(input_data, employee_target_data)
    print("TARGET DATA: EMPLOYEES - END")
    print("TARGET DATA: BUSINESS APPLICATIONS")
    run_evaluation(input_data, business_applications_target_data)
    print("TARGET DATA: BUSINESS APPLICATIONS - END")
    print("TARGET DATA: TOTAL CONSTRUCTION SPENDING")
    run_evaluation(input_data, total_construction_spending_target_data)
    print("TARGET DATA: TOTAL CONSTRUCTION SPENDING - END")

"""This function runs the Stochastic Gradient Descent (SGD) regression neural network along with a standard
MLP neural network Economic testing data for benchmark testing and reports each model's RMSE for 10 time series splits
on a dataset of a given target variable, along well as the total weighted feature importances of each trial SGD model."""
def run_evaluation(input_data, target_data):
    testing_targets_across_splits = []
    model_testing_predictions_across_splits = []
    lbfgs_model_testing_predictions_across_splits = []
    adam_model_testing_predictions_across_splits = []
    model_rmse_across_splits = []
    lbfgs_model_rmse_across_splits = []
    adam_model_rmse_across_splits = []
    model_nrmse_across_splits = []
    lbfgs_model_nrmse_across_splits = []
    adam_model_nrmse_across_splits = []
    model_total_weighted_feature_importances_across_splits = []
    model_training_costs_across_splits = []
    model_validation_mses_across_splits = []
    tscv = TimeSeriesSplit(n_splits = 50, test_size = 4, max_train_size = 12)
    i = 1
    for feed_indices, testing_indices in tscv.split(input_data):
        print("Split " + str(i))
        print("Feed Indices - " + str(feed_indices))
        print("Testing Indices - " + str(testing_indices))
        feed_input_data = [input_data[i] for i in feed_indices]
        feed_target_data = [target_data[i] for i in feed_indices]
        testing_input_data = [input_data[i] for i in testing_indices]
        testing_target_data = [target_data[i] for i in testing_indices]
        print("Feed Input Data - " + str(feed_input_data))
        print("Feed Target Data - " + str(feed_target_data))
        print("Testing Input Data - " + str(testing_input_data))
        print("Testing Target Data - " + str(testing_target_data))
        feed_data_reference = {
        "input_data": feed_input_data,
        "target_data": feed_target_data
        }

        model = NeuralNetworkModel(**feed_data_reference, hidden_layers = 3, random_state = 1)

        lbfgs_model = MLPRegressor(solver = "lbfgs", hidden_layer_sizes = (8, 4, 2,), random_state = 1)
        lbfgs_model.fit(feed_input_data, feed_target_data)

        adam_model = MLPRegressor(solver = "adam", hidden_layer_sizes = (8, 4, 2), random_state = 1)
        adam_model.fit(feed_input_data, feed_target_data)

        model_testing_predictions = model.run_model(testing_input_data)

        lbfgs_model_testing_predictions = lbfgs_model.predict(testing_input_data)

        adam_model_testing_predictions = adam_model.predict(testing_input_data)

        model_ses = []
        lbfgs_model_ses = []
        adam_model_ses = []
        for j in range(len(testing_target_data)):
            model_ses.append(pow(testing_target_data[j] - model_testing_predictions[j], 2))
            lbfgs_model_ses.append(pow(testing_target_data[j] - lbfgs_model_testing_predictions[j], 2))
            adam_model_ses.append(pow(testing_target_data[j] - adam_model_testing_predictions[j], 2))
        model_rmse = math.sqrt(stats.mean(model_ses))
        lbfgs_model_rmse = math.sqrt(stats.mean(lbfgs_model_ses))
        adam_model_rmse = math.sqrt(stats.mean(adam_model_ses))
        model_nrmse = model_rmse / stats.mean(testing_target_data)
        lbfgs_model_nrmse = lbfgs_model_rmse / stats.mean(testing_target_data)
        adam_model_nrmse = adam_model_rmse / stats.mean(testing_target_data)
        for j in range(len(testing_target_data)):
            testing_targets_across_splits.append(testing_target_data[j])
            model_testing_predictions_across_splits.append(model_testing_predictions[j])
            lbfgs_model_testing_predictions_across_splits.append(lbfgs_model_testing_predictions[j])
            adam_model_testing_predictions_across_splits.append(adam_model_testing_predictions[j])
        model_total_weighted_feature_importances = calculate_total_weighted_feature_importances(model)
        model_training_costs = model.get_training_cost_values_over_epochs()
        model_validation_mses = model.get_validation_mses_over_epochs()
        model_rmse_across_splits.append(model_rmse)
        lbfgs_model_rmse_across_splits.append(lbfgs_model_rmse)
        adam_model_rmse_across_splits.append(adam_model_rmse)
        model_nrmse_across_splits.append(model_nrmse)
        lbfgs_model_nrmse_across_splits.append(lbfgs_model_nrmse)
        adam_model_nrmse_across_splits.append(adam_model_nrmse)
        model_total_weighted_feature_importances_across_splits.append(model_total_weighted_feature_importances)
        model_training_costs_across_splits.append(model_training_costs)
        model_validation_mses_across_splits.append(model_validation_mses)
        print("Split " + str(i) + " Results")
        print("Model Predictions - " + str(model_testing_predictions))
        print("LBFGS Model Predictions - " + str(lbfgs_model_testing_predictions))
        print("Adam Model Predictions - " + str(adam_model_testing_predictions))
        print("Model RMSE - " + str(model_rmse))
        print("LBFGS Model RMSE - " + str(lbfgs_model_rmse))
        print("Adam Model RMSE - " + str(adam_model_rmse))
        print("Model N-RMSE - " + str(model_nrmse))
        print("LBFGS Model N-RMSE - " + str(lbfgs_model_nrmse))
        print("Adam Model N-RMSE - " + str(adam_model_nrmse))
        print("Model Feature Importances - " + str(model_total_weighted_feature_importances))
        print("Model Training Costs - " + str(model_training_costs))
        print("Model Validation MSEs - " + str(model_validation_mses))
        i += 1
    model_mean_rmse = stats.mean(model_rmse_across_splits)
    lbfgs_model_mean_rmse = stats.mean(lbfgs_model_rmse_across_splits)
    adam_model_mean_rmse = stats.mean(adam_model_rmse_across_splits)
    model_mean_nrmse = stats.mean(model_nrmse_across_splits)
    lbfgs_model_mean_nrmse = stats.mean(lbfgs_model_nrmse_across_splits)
    adam_model_mean_nrmse = stats.mean(adam_model_nrmse_across_splits)
    print("Summary")
    print("Testing Targets - " + str(testing_targets_across_splits))
    print("Model Testing Predictions - " + str(model_testing_predictions_across_splits))
    print("LBFGS Model Testing Predictions - " + str(lbfgs_model_testing_predictions_across_splits))
    print("Adam Model Testing Predictions - " + str(adam_model_testing_predictions_across_splits))
    print("Model RMSE Across Splits - " + str(model_rmse_across_splits))
    print("LBFGS Model RMSE Across Splits - " + str(lbfgs_model_rmse_across_splits))
    print("Adam Model RMSE Across Splits - " + str(adam_model_rmse_across_splits))
    print("Model N-RMSE Across Splits - " + str(model_nrmse_across_splits))
    print("LBFGS Model N-RMSE Across Splits - " + str(lbfgs_model_nrmse_across_splits))
    print("Adam Model N-RMSE Across Splits - " + str(adam_model_nrmse_across_splits))
    print("Model Total Weighted Feature Importances Across Splits - " + str(model_total_weighted_feature_importances_across_splits))
    print("Model Training Costs Across Splits - " + str(model_training_costs_across_splits))
    print("Model Validations MSEs Across Splits - " + str(model_validation_mses_across_splits))
    print("Model Mean RMSE - " + str(model_mean_rmse))
    print("LBFGS Model Mean RMSE - " + str(lbfgs_model_mean_rmse))
    print("Adam Model Mean RMSE - " + str(adam_model_mean_rmse))
    print("Model Mean N-RMSE - " + str(model_mean_nrmse))
    print("LBFGS Model Mean N-RMSE - " + str(lbfgs_model_mean_nrmse))
    print("Adam Model Mean N-RMSE - " + str(adam_model_mean_nrmse))

"""This model processes Economic datasets and returns the respective input and target
datasets to be used for testing between the neural network models."""
def process_data(input_data_file_names, target_data_file_name, start_year, end_year):
    random.seed(42)
    input_data = []
    target_data = []
    months = ((end_year - start_year) * 12) + 1
    for i in range(months):
        input_data.append([])
    for i in range(len(input_data_file_names)):
        file = open(input_data_file_names[i])
        j = 0
        for line in file:
            data_line = line.strip().split(",")
            if data_line[0] == "" or data_line[0] == "Period":
                continue
            year = int(data_line[0][-4:])
            month = data_line[0][:-5]
            if year < start_year or year > end_year or (year == end_year and month != "Jan"):
                continue
            value = float(data_line[1])
            input_data[j].append(value)
            j += 1
    file = open(target_data_file_name)
    j = 0
    for line in file:
        data_line = line.strip().split(",")
        if data_line[0] == "observation_date":
            continue
        year = int(data_line[0][:4])
        month = int(data_line[0][5:7])
        if year < start_year or year > end_year or (year == start_year and month == 1) or (year == end_year and month != 1 and month != 2):
            continue
        target_data.append(float(data_line[1]))
        j += 1
    return input_data, target_data

"""This function determines the total weighted feature weights of a given Bayesian Optimized SGD model using a
backward weighting propagation method."""
def calculate_total_weighted_feature_importances(model):
    parameters = model.get_parameters()
    total_weighted_input_importances_across_layers = []
    for i in range(len(parameters)):
        total_weighted_input_importances_across_layers.append([])
    for i in reversed(range(len(parameters))):
        layer_total_weighted_input_importances_across_neurons = []
        for j in range(len(parameters[i])):
            if i != len(parameters) - 1:
                neuron_total_weighted_input_importances = [parameters[i][j][0][k] * total_weighted_input_importances_across_layers[i + 1][j] for k in range(len(parameters[i][j][0]))]
            else:
                neuron_total_weighted_input_importances = [parameters[i][j][0][k] for k in range(len(parameters[i][j][0]))]
            layer_total_weighted_input_importances_across_neurons.append(neuron_total_weighted_input_importances)
        layer_total_weighted_input_importances = [sum([layer_total_weighted_input_importances_across_neurons[k][j] for k in range(len(layer_total_weighted_input_importances_across_neurons))]) for j in range(len(layer_total_weighted_input_importances_across_neurons[0]))]
        total_weighted_input_importances_across_layers[i] = layer_total_weighted_input_importances
    return total_weighted_input_importances_across_layers[0]

main()
