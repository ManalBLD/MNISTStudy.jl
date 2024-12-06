include("src_ver1.jl")

# Load MNIST dataset
train_X, train_y = MNIST.traindata()
test_X, test_y = MNIST.testdata()

train_X, train_y, test_X, test_y = load_prepare_data(train_X, train_y, test_X, test_y)

# Building the model
model = build_cnn_model()

# Define loaders
train_loader = Flux.DataLoader((train_X, train_y), batchsize=64, shuffle=true)
test_loader = Flux.DataLoader((test_X, test_y), batchsize=64)

# Optimizer and drive parameters
optimizer = Adam()
epochs = 10

# Training the model
train_model(train_loader, test_loader, model, epochs, optimizer)

# Post-training metrics calculation
total_accuracy = compute_accuracy(test_loader, model)
println("Total accuracy of test set: $total_accuracy")

display_metriques(test_loader, model)
