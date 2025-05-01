We have a target vector with highly imbalanced classes, and you want to make adjustments so that you
can handle the class imbalance, Collect more data. If that isn’t possible, change the metrics used to evaluate your model. If that doesn’t
work, consider using a model’s built-in class weight parameters (if available), downsampling, or
upsampling. We cover evaluation metrics in a later chapter, so for now let us focus on class weight
parameters, downsampling, and upsampling.
To demonstrate our solutions, we need to create some data with imbalanced classes. Fisher’s Iris dataset
contains three balanced classes of 50 observations, each indicating the species of flower (Iris setosa, Iris
virginica, and Iris versicolor). To unbalance the dataset, we remove 40 of the 50 Iris setosa observations
and then merge the Iris virginica and Iris versicolor classes. The end result is a binary target vector
indicating if an observation is an Iris setosa flower or not. The result is 10 observations of Iris setosa
(class 0) and 100 observations of not Iris setosa (class 1), Many algorithms in scikit-learn offer a parameter to weight classes during training to counteract the
effect of their imbalance. While we have not covered it yet, RandomForestClassifier is a popular
classification algorithm and includes a class_weight parameter. You can pass an argument specifying
the desired class weights explicitly, Or you can pass balanced, which automatically creates weights inversely proportional to class
frequencies, Alternatively, we can downsample the majority class or upsample the minority class. In downsampling,
we randomly sample without replacement from the majority class (i.e., the class with more
observations) to create a new subset of observations equal in size to the minority class. For example, if
the minority class has 10 observations, we will randomly select 10 observations from the majority class
and use those 20 observations as our data. Here we do exactly that using our unbalanced Iris data, Our other option is to upsample the minority class. In upsampling, for every observation in the majority
class, we randomly select an observation from the minority class with replacement. The end result is the
same number of observations from the minority and majority classes. Upsampling is implemented very
similarly to downsampling, just in reverse, In the real world, imbalanced classes are everywhere—most visitors don’t click the buy button and
many types of cancer are thankfully rare. For this reason, handling imbalanced classes is a common
activity in machine learning.
Our best strategy is simply to collect more observations—especially observations from the minority
class. However, this is often just not possible, so we have to resort to other options.
A second strategy is to use a model evaluation metric better suited to imbalanced classes. Accuracy is
often used as a metric for evaluating the performance of a model, but when imbalanced classes are
present accuracy can be ill suited. For example, if only 0.5% of observations have some rare cancer,
then even a naive model that predicts nobody has cancer will be 99.5% accurate. Clearly this is not
ideal. Some better metrics we discuss in later chapters are confusion matrices, precision, recall, F1
scores, and ROC curves.
A third strategy is to use the class weighing parameters included in implementations of some models.
This allows us to have the algorithm adjust for imbalanced classes. Fortunately, many scikit-learn
classifiers have a class_weight parameter, making it a good option.
The fourth and fifth strategies are related: downsampling and upsampling. In downsampling we create a
random subset of the majority class of equal size to the minority class. In upsampling we repeatedly
sample with replacement from the minority class to make it of equal size as the majority class. The
decision between using downsampling and upsampling is context-specific, and in general we should try
both to see which produces better results
