# Methodology

![alt text](figures/methodology-overview_sketch.png)

## Data Set

 We use the "FabWave" dataset (==reference==) for benchmarking our models. It provides a wide variaty of objects, but mainly mechanical components. It contains more than 4000 parts in step file format, belonging to 44 categories. 

Before appliying the data set to our models a thorough preprocessing had to be performed.

First, any empty or corrupted files are removed.


 
Unfortuantly not all parts form the dataset can be used for all three of our models. This is because not all step files could be converted into all three representaitons formats.For better comparison between the models we prepare a training, validation and test split of the data. Each ML model uses the same data splits. That means that the "FabWave" dataset has been filtered for STEP files for which all three representation types (invariants, images, graphs) can be generated.

The "FabWave" dataset categorizes the CAD models into 43 types of parts. These we use as given. For the regression task we determine we labels for each part grammatically. For each CAD model these regression labels are calculated. 

- Volume
- Amount of Faces, Edges and Vertices
- (Dimensions of the bounding box  ==Do we need to rotate all models according to their main distribution in space?==)

==summary table of the final dataset==

## Representation of CAD data

## Machine Learning Models

### Tree-LSTM for trees representaiton

### RotationNet for image representation

### MLP for Invariants
Our models are tested with a classification tasks, as well as a multi-regression task. The classification task aims at the identification of the *type* a given part is categorized as; e.g. gears, bearing, etc. The regression tasks is set up to determine several basic attributes of a CAD model, like its volume, its amount of vertices, edges and faces, and bounding box dimensions.


### Model Training

For each ML model a hyperparameter tuning is performed; followed by a final training session of the tuned ML model.  Only by optimizing all ML models fair and meaningful comparisons can be drawn.

ML Models

Three different machine learning models are compared. The models have been chosen, because they state different approaches in anlysising CAD models. The central difference between these approaches is the *representation* of the CAD models information. It is hypothesised that not all representations are able to convey the same amount of information yielding in different performances for the models. The models themselfs have merely been chosen as they are capable of processing the coressponding CAD model representations.

## Comparision of Representation approaches

We expected no model or representation type to outperform all others in all tasks.
This is because each approach has its own strengths and weaknesses as layout in section 3.
To test this hypothesis we compare the performance of the three models in terms of a classification and regression tasks in as abstract way if possible. The regression tasks aims to determine universial features of the CAD models. We hope this serves as a good indicator for the applicability of the representation type for other more specific tasks.

In terms of the classification task we compare our approaches as generaly as possible, despite the fact that the classes of parts are only relevant in an industrial context. 
For each approach abstract metrics like accuracy, precision, recall and F1-score are calculated.
More abstract classes would be suited to recommed a representation type for a new task, but such classes were not available.

Comparing our models in terms of no task specific metrics is beneficial here, because it allows us to draw conclusions about the general applicability of the representation types. Instead we focus on more general regression metrics like the mean absolute error (MAE) and the root mean square error (RMSE).

In order to determine the representation type that can be expected to yield good results for a wide and unknow variety tasks, we applied the ML models to learn abtstract part features.

To allow for a good comparability all models are trained on the same data set. Note that it would be also reasonable to use different preprocessing techniques like data augmentation for each approach, without changing the comparability. But in order to support the argument that one representation type can generally be expected to yield better results than another, we use the same preprocessing for all models.
