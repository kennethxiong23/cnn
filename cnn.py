import tensorflow.keras.utils as utils
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.losses as losses
import tensorflow.keras.optimizers.legacy as optimizers

train = utils.image_dataset_from_directory(
        "images",
        label_mode = 'categorical',
        batch_size = 32,
        image_size = (596,596),
        seed = 23,
        validation_split = 0.3,
        subset = "training",
)

test = utils.image_dataset_from_directory(
        "images",
        label_mode = 'categorical',
        batch_size = 32,
        image_size = (596, 596),
        seed = 23,
        validation_split = 0.3,
        subset = "validation",

)

class Net():
        def __init__(self, input_shape):
                self.model = models.Sequential()
                self.model.add(layers.ZeroPadding2D(
                        padding = ((1,2),(1,2)),
                        input_shape = input_shape))
                self.model.add(layers.Conv2D(
                        8, # filters
                        29, # size
                        strides = 10, #step size
                        activation = 'relu',
                )) #output 58x58x8
                self.model.add(layers.MaxPool2D(pool_size=2))
                #output 29x29x8
                self.model.add(layers.Conv2D(
                8, # filters
                3, # size
                strides = 1, #step size
                activation = 'relu',
                )) #output 28x28x8
                self.model.add(layers.MaxPool2D(pool_size=2))
                #output 14x14x8
                self.model.add(layers.Flatten())
                #output: 1568
                self.model.add(layers.Dense(256, activation = 'relu'))
                self.model.add(layers.Dense(64, activation = 'relu'))
                self.model.add(layers.Dense(16, activation = 'relu'))
                self.model.add(layers.Dense(5, activation = 'softmax'))
                self.loss = losses.CategoricalCrossentropy()
                self.optimizer = optimizers.SGD(learning_rate = 0.0001)
                self.model.compile(
                        loss = self.loss,
                        optimizer = self.optimizer,
                        metrics = ["accuracy"]
                )
        def __str__(self):
                self.model.summary()
                return ""

net =  Net((596,596,3))

net.model.fit(
        train,
        batch_size = 32,
        epochs = 1000,
        verbose  = 2,
        validation_data = test,
        validation_batch_size = 32,
)

# net.model.save("face_model_save")