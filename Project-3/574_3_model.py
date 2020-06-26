from sklearn.naive_bayes import CategoricalNB
import numpy as np
from Preprocessing import preprocess
from Report_Results import report_results
from utils import *
from Postprocessing import *



from sklearn.naive_bayes import CategoricalNB
from Preprocessing import preprocess
from Postprocessing import *
from utils import *

metrics = ["race", "sex", "age", 'c_charge_degree', 'priors_count', 'c_charge_desc']
training_data, training_labels, test_data, test_labels, categories, mappings = preprocess(metrics)

NBC = CategoricalNB()
NBC.fit(training_data, training_labels)

training_class_predictions = NBC.predict_proba(training_data)
training_predictions = []
test_class_predictions = NBC.predict_proba(test_data)
test_predictions = []

for i in range(len(training_labels)):
    training_predictions.append(training_class_predictions[i][1])

for i in range(len(test_labels)):
    test_predictions.append(test_class_predictions[i][1])

training_race_cases = get_cases_by_metric(training_data, categories, "race", mappings, training_predictions, training_labels)
test_race_cases = get_cases_by_metric(test_data, categories, "race", mappings, test_predictions, test_labels)

training_race_cases, thresholds = enforce_equal_opportunity(training_race_cases, 0.01)

for group in test_race_cases.keys():
    test_race_cases[group] = apply_threshold(test_race_cases[group], thresholds[group])


# ADD MORE PRINT LINES HERE - THIS ALONE ISN'T ENOUGH
# YOU NEED ACCURACY AND COST FOR TRAINING AND TEST DATA
# PLUS WHATEVER RELEVANT METRICS ARE USED IN YOUR POSTPROCESSING METHOD, TO ENSURE EPSILON WAS ENFORCED
print("--------------------USING EQUAL OPPORTUNITY--------------------")
print("Accuracy on training data:")
print(get_total_accuracy(training_race_cases))
print("")

print("Cost on training data:")
print('${:,.0f}'.format(apply_financials(training_race_cases)))
print("")

print("Accuracy on testing data:")
print(get_total_accuracy(test_race_cases))
print("")

print("Cost on testing data:")
print('${:,.0f}'.format(apply_financials(test_race_cases)))
print("")

print("Thresholds:")
for group in thresholds.keys():
    print("{}: {}".format(group, thresholds[group]))
print()

print("TPR should be within", 0.01)
for group in training_race_cases.keys():
    TPR = get_true_positive_rate(training_race_cases[group])
    print("Training TPR for " + group + ": " + str(TPR))
    