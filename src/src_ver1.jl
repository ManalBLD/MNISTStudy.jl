using Images
using MLDatasets
using ImageView
using Flux 

using Base.Iterators: repeated
using Images
using Flux: onehotbatch, onecold, crossentropy , params

#Data preparation : 
function load_prepare_data(train_X, train_y, test_X, test_y)

    #Normalize between 0 and 1  : 
    train_X = Float32.(train_X) ./ 255.0
    test_X = Float32.(test_X) ./ 255.0

    #Add a dimension for the channels, as it is useful afterwards for our neural network model.
    train_X = reshape(train_X, 28, 28, 1, :)
    test_X = reshape(test_X, 28, 28, 1, :)

    # One-hot label encoding: Transform labels into binary vectors for easy learning
    train_y = onehotbatch(train_y, 0:9)
    test_y = onehotbatch(test_y, 0:9)

    return train_X, train_y, test_X, test_y
end

#Building the CNN (convolutional neural network) model with functions from the Flux package: 
function build_cnn_model()

    model = Chain(
        Conv((3, 3), 1 => 16, relu),    # Convolution 3x3, 1 input channel (grayscale image) and 16 filters
        MaxPool((2, 2)),                # Max pooling to reduce image size
        Conv((3, 3), 16 => 32, relu),   # Second convolutional layer
        MaxPool((2, 2)),                # Second max pooling
        Flux.flatten,                   #Flatten convolution output to pass through dense layers
        Dense(32 * 5 * 5 => 128, relu), # Dense layer after flatten, adjusted for image size
        Dense(128 => 10),               # Output layer with 10 neurons for the 10 classes
        softmax                         # Softmax to obtain class probabilities
    )
    return model
end


#************* Training ******************

# Function to calculate accuracy
function compute_accuracy(loader, model)
    accuracy = 0.0
    total = 0
    for (x_batch, y_batch) in loader
        predict = model(x_batch)                        # Predicting classes
        predicted_classes = onecold(predict, 0:9)       # Convert predictions into classes
        true_labels = onecold(y_batch, 0:9)             # Convert real labels into classes
        accuracy += sum(predicted_classes .== true_labels)
        total += length(true_labels)
    end
    return accuracy / total
end

#the testing data should be used to measure the accuracy of the model.

function train_model(train_loader, test_loader, model, epochs, optimizer)

    loss(x, y) = crossentropy(model(x), y)
    for epoch in 1:epochs
        for (x_batch, y_batch) in train_loader
            Flux.train!(loss, params(model), [(x_batch, y_batch)], optimizer)
        end
        test_accuracy = compute_accuracy(test_loader, model)
        println("Epoch $epoch finished. Test accuracy : $test_accuracy")
    end
       
end

"""
Recall: The higher it is, the more the Machine Learning model maximizes the number of True Positives. 
it won't miss any positives.

Precision: When precision is high, it means that the majority of the model's positive predictions 
are well-predicted positives, minimizing the number of false positives.
The higher the precision, the fewer false positives the model predicts. 
F1- SCORE: The higher the F1 Score, the better the model's performance.

True Negative (TN) : the prediction and the actual value are negative. 
True Positive (TP) : prediction and actual value are positive. 
False Positive (FP) : the prediction is positive while the actual value is negative.
False Negative (FN) : the prediction is negative while the actual value is negative.

"""
function metriques(predicted_classes, true_labels)
    tp = sum((predicted_classes .== true_labels) .& (true_labels .== 1))  # True positive
    fp = sum((predicted_classes .== 1) .& (true_labels .== 0))            # False positive
    fn = sum((predicted_classes .== 0) .& (true_labels .== 1))            # False negative

    precision = tp / (tp + fp + 1e-10)  # Add a small constant to avoid division by zero
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

    return precision, recall, f1
end

#These 2 functions have the same name but different parameters, that's what we call multiple dispatching
#Here, the 2nd “metrics” function calculates the confusion matrix, also known as a contingency table, 
#It allows you to visualize the model's ability to classify cases correctly or incorrectly, by comparing the model's predictions with the real truth.

function metriques(predicted_classes, true_labels, num_classes)
    cm = zeros(Int, num_classes, num_classes)
    for (pred, true_label) in zip(predicted_classes, true_labels)
        cm[true_label + 1, pred + 1] += 1  # +1 because Julia indexes start at 1
    end
    return cm
end


function display_metriques(test_loader, model)

    for (x_batch, y_batch) in test_loader  # Uses test data
        predicted_classes = onecold(model(x_batch), 0:9)  # Predicting classes for this mini-lot
        true_labels = onecold(y_batch, 0:9)               # Convert real labels into classes

        # Calculate the metrics for this mini-lot
        precision, recall, f1 = metriques(predicted_classes, true_labels)
        println("Précision: ", precision)
        println("Recall: ", recall)
        println("F1-score: ", f1)

        # Calculate the confusion matrix for this mini-lot
        cm = metriques(predicted_classes, true_labels, 10)  # 10 classes for MNIST
        println("Matrice de confusion:")
        println(cm)

        break  # Delete this line to browse all test_loader batches
    end
end

"""
REMARKS : 
    Interpretation of the confusion matrix: 
    * 6 samples from class 0 were correctly classified as 0 
    * 10 samples from class 1 were correctly classified as 1
    * And so on ...
    Analysis of metrics ( precision, recall , f1-score):
    * we have 0.9999999 for precision , recall and also f1-score, which shows that our model performs well
    The model does an excellent job of distinguishing the different classes, with accuracy close to 100% for several classes.

    There is virtually no confusion between classes. This matrix reflects the good results we obtained for other metrics such as F1-score, precision and recall.
    This suggests that our model performs very well for the task of classifying MNIST images.

"""