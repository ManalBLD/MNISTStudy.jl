
# Load the source file containing all functions
include("src_ver2.jl")

# Load dataset :
train_X, train_y = MNIST.traindata()
test_X, test_y = MNIST.testdata()

# Calling up the function with a dynamic number of images
train_X, train_y, test_X, test_y = load_prepare_data(train_X, train_y, test_X, test_y; num_train=500, num_test=500)

# Building the model
model = build_cnn_model()

# Define loaders with adjusted batch size 
#NB: if you test with a mini-batch of images < 64, you absolutely must change the batchsize because the batchsize must be <= chosen image batch. 
train_loader = Flux.DataLoader((train_X, train_y), batchsize = 64, shuffle=true)
test_loader = Flux.DataLoader((test_X, test_y), batchsize = 64)

# Optimizer and drive parameters
optimizer = Adam()
epochs = 10

# Training the model
train_model(train_loader, test_loader, model, epochs, optimizer)

# Post-training metrics calculation
total_accuracy = compute_accuracy(test_loader, model)
println("Total accuracy of test set: $total_accuracy")

display_metriques(test_loader, model)
