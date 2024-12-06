using Images
using MLDatasets
using ImageView
using Flux 

using Base.Iterators: repeated
using Images
using Flux: onehotbatch, onecold, crossentropy , params

using Zygote: @nograd
using Base.Threads

#Data preparation :
function load_prepare_data(train_X, train_y, test_X, test_y; num_train, num_test)
    
    train_X, train_y = train_X[:, :, 1:num_train], train_y[1:num_train]
    test_X, test_y = test_X[:, :, 1:num_test], test_y[1:num_test]

    #Normalize between 0 and 1 : 
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


function convolution2d(input, kernel)

    if ndims(input) == 3
        h, w, c = size(input)
        n = 1  # Number of images (batch size) = 1 if input is 3D
    else
        h, w, c, n = size(input)
    end 

    kh, kw, kc = size(kernel)
    # Check that input and kernel channel numbers match
    if c != kc
        throw(DimensionMismatch("The number of channels in the kernel and the input do not match."))
    end
    output_h = h - kh + 1
    output_w = w - kw + 1
    output = zeros(output_h, output_w, n)
    
    for img in 1:n
        for i in 1:output_h
            for j in 1:output_w
                # Sum all channels in a vectorized way
                region = view(input, i:i+kh-1, j:j+kw-1, :, img)
                output[i, j, img] = sum(dot(region[:, :, k], kernel[:, :, k]) for k in 1:c)
            end
        end
    end
    return output
end

# Function to generate random kernels 
function generate_kernels(kernel_size, input_channels, num_filters)
    return rand(kernel_size, kernel_size, input_channels, num_filters)
end


# Convolution function for multiple filters and padding-free batch images

function convolution_layer(input, kernels)
    h, w, c, n = size(input)
    kh, kw, kc, num_filters = size(kernels)

    # Check that kernel and input channels match
    if c != kc
        throw(DimensionMismatch("The number of channels in the kernel and the input do not match."))
    end

    output_h = h - kh + 1
    output_w = w - kw + 1
    output = zeros(output_h, output_w, num_filters, n)

    # Apply each filter to each image in the batch
    @threads for f in 1:num_filters
        for img in 1:n
            output[:, :, f, img] = convolution2d(input[:, :, :, img], kernels[:, :, :, f])
        end
    end
    return output
end


function max_pooling(input, pool_size)
    h, w, num_filters, n = size(input)
    ph, pw = pool_size
    output_h = div(h, ph)
    output_w = div(w, pw)
    output = zeros(Float32, output_h, output_w, num_filters, n)

    # Use slicing and maximum over each region in a more efficient way
    @threads for k in 1:num_filters
        for b in 1:n
            for i in 1:output_h
                for j in 1:output_w
                    # “view” is a data structure that acts like an array (it is a subtype of AbstractArray in Julia)
                    output[i, j, k, b] = maximum(view(input, (i-1)*ph+1:i*ph, (j-1)*pw+1:j*pw, k, b))
                end
            end
        end
    end
    return output
end


#Declare functions for which automatic differentiation is not required
#The ability of Zygote is to work with callable structs that can be used for implementing ML models.
@nograd convolution2d
@nograd convolution_layer
@nograd max_pooling


function build_cnn_model()
    model = Chain(

        # First convolution layer with our manual functions
        x -> convolution_layer(x, generate_kernels(3, 1, 16)),  # 16 filters  3x3
        x -> max_pooling(x, (2, 2)),         # Max pooling with size 2x2

        # Second convolution layer
        x -> convolution_layer(x, generate_kernels(3, 16, 32)), # 32 filters 3x3, assuming 16 input channels
        x -> max_pooling(x, (2, 2)),         # Max pooling with size 2x2

        # Flattening and dense layers for classification
        x -> Flux.flatten(x),                                   # Merge filters and flatten
        Dense(5 * 5 * 32 => 128, relu),                         # Adjust the size here according to the size after pooling
        Dense(128 => 10),                                       # Output layer for 10 classes
        softmax                                                 # Softmax for probabilities
    )
    return model
end


#************* Training ******************

#Function to calculate accuracy

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
    tp = sum((predicted_classes .== true_labels) .& (true_labels .== 1))  # True Positive
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


""" So,if you're interested in the implementation of convolution layers with padding, you'll find the code below."""

#= function creer_kernel(kernel_size::Tuple{Int, Int})
    kernel=rand(DiscreteUniform(-10 , 10), kernel_size[1] , kernel_size[2])
    return kernel
end =#

#= function convolution_kernel(image::Matrix{Float32}, kernel_size::Tuple{Int, Int})
    n, p = size(image)
    kn = kernel_size[1] 
    kp=kernel_size[2]
    kernel=creer_kernel(kernel_size)

    padded_image = zeros(Float32, n + kn - 1, p + kp - 1)

    padded_image[(1 + (kn - 1) ÷ 2):(n + (kn - 1) ÷ 2), (1 + (kp - 1) ÷ 2):(p + (kp - 1) ÷ 2)] .= image

    padded_image[1:(1 + (kn - 1) ÷ 2), :] .= padded_image[1 + (kn - 1) ÷ 2, :][1]  # Haut
    padded_image[(n + (kn - 1) ÷ 2 + 1):end, :] .= padded_image[n + (kn - 1) ÷ 2, :][end]  # Bas

    padded_image[:, 1:(1 + (kp - 1) ÷ 2)] .= padded_image[:, 1 + (kp - 1) ÷ 2][:, 1]  # Gauche
    padded_image[:, (p + (kp - 1) ÷ 2 + 1):end] .= padded_image[:, p + (kp - 1) ÷ 2][:, end]  # Droit

    output_height = n
    output_width = p
    output = zeros(Float32, output_height, output_width)

    for i in 1:output_height
        for j in 1:output_width
            sub_matrix = padded_image[(i):(i + kn - 1), (j):(j + kp - 1)]
            output[i, j] = sum(sub_matrix .* kernel)
        end
    end
    println("padding:")
    println(padded_image)

    return output 
end =#
