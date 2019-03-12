import pandas
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# The path of the file that will be executed 
iowa_file_path = 'C:/Users/natha/OneDrive - University of Essex/Team Project/home-data-for-ml-course/train.csv'
test_file_path = "C:/Users/natha/OneDrive - University of Essex/Team Project/home-data-for-ml-course/test.csv"

# The data found at the sepcified path is being read into the variable home_data
home_data = pandas.read_csv(iowa_file_path) 
test_data = pandas.read_csv(test_file_path)
#Using One Hot Encoding to be able to use categorical data
home_data = pandas.get_dummies(home_data)
test_data = pandas.get_dummies(test_data)

y = home_data.SalePrice

#printing all the attributes of the houses
print("House attributes: \n")
for attribute in home_data.columns:
    print(attribute)

#choosing from the list of attributes the most relevant ones, these being the features of the house
# Create the list of features below
house_features = ["LotArea","YearBuilt","1stFlrSF","2ndFlrSF","FullBath","BedroomAbvGr", "GrLivArea", "Exterior1st_CBlock"] #"TotRmsAbvGrd"

# select data corresponding to features in feature_names
X = home_data[house_features]
test_X = test_data[house_features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

def get_mean_absolute_error(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes = max_leaf_nodes, random_state = 0)
    model.fit(train_X, train_y)
    predictions_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, predictions_val)
    return(mae)
    
def get_best_no_of_leaves():
    min = None
    for max_leaf_nodes in range(5,100,5):
        my_mae = get_mean_absolute_error(max_leaf_nodes, train_X, val_X, train_y, val_y)
        #print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
        if min == None or my_mae < min:
            min = my_mae
            best_no_of_leaves = max_leaf_nodes
    return best_no_of_leaves

print()
print("\n The optimal number of leaves in the Decision Tree should be ", get_best_no_of_leaves())

#specify the model - we will be using the decision tree model, splitting the data into leaves
iowa_model = DecisionTreeRegressor(max_leaf_nodes = get_best_no_of_leaves(), random_state = 1)

# Fit model
iowa_model.fit(train_X, train_y)

print()
print("Making predictions for the following houses:")
print(val_X.head())
print()
print("The predictions are: ")
predicted_home_prices = iowa_model.predict(val_X.head())
for p in predicted_home_prices:
    print(p)
print()
#use val_X.head() to use only the first 5 houses

print("Actual target values for those homes:", y.head().tolist())
print()

#model validation - finding the errors in prediction
predicted_home_prices_train = iowa_model.predict(val_X)
predicted_home_prices_test = iowa_model.predict(test_X)
print("The mean absolute error in the house prices is(training file): " + str(mean_absolute_error(val_y, predicted_home_prices_train)))
print()

predictions = list(predicted_home_prices_test)
ids = list(test_data.Id)

a = open("output.csv", 'w')

#a.write("ID"+ "SalePrice" + "\n")
for i in range(len(predictions)):
    a.write(str(ids[i])+ ", " + str(predictions[i]) + "\n")
    
a.close()
