(Purpose)
In this paper we show wheter or not a certain data representation approach for CAD models is generaly perferable other some altertatives.

(Context for the field)
CAD models are used accros a wide range of industries, including mechanical and civil engineering, multimedia production and ... .
They all share a need to design, change and plan for produciton or integration of virtual or real life objects.
Modern CAD-Programs are making this possible, without the need for expensive reallife prototypical objects.
Onward we will focus on CAD models representing meanichal parts in the STEP file format.

The STEP file format is widely used for exchange of 3D CAD Models between different CAD/CAM systems.
A standard exchange formate became necessary as more and more CAD-Systems were developed and used by different companies for different purposes, that nonetheless had to able to cooparated with each other effectively.
Since STEP files are common it makes sense to focus on automatic processing of CAD models in this format.
That is even though STEP files can not necessarily used as input to a machine leraning model directly.

Automatic processing of CAD files promisses gains in terms of time saving and accuracy of a wide variety of tasks across all phases of the product life cycle.
Tools for generating CAD models are expected to speed up the drafting process; feature detection systems allow for quicker development of G-Code and ...
For most of these tasks it is necessary to develop a machine learning because it is not feasible to define an algorithm manualy that is suffciently generalsized while also achiving high quailty output.
Developing such a machine learing application is a non-trival tasks. One aspect of it is choosing an appropriate representation of the CAD models data.

A wide variaty of ML archicetures has been developed to match the many possible representation types 3D models can be represented as. Each representation requires its custom ML archiceture to be processible in the first place. This lead to the fact that a variaty of representations and its corresponding ML architecures have to be compared when a new 3D model analysis application is to be developed in order to determined the most suitable one.

A CAD models data can be conveyed using a variaty of representations.
These include 2D-imaging, voxels, meshes and graphs, among others.
These different representations of essentialy the same data makes it possible to use a variety of ML model types for CAD model anlaysis.
Unfortnatly each representation comes with individual advantages and disadvantages.
This makes it difficult to choose the best representation form and ML model for a given task.

(Summary of previous research)
In order to help guiding the decision on which data representation form may be used, several previous work has been done... They all lack...

(Objectives and methods)
In this paper we compare three promising machine learning approaches, each using a different typ of CAD model data representation.
By comparing these approaches in terms of *general* CAD model attributes, i.e. not specific for most production use cases, we show which ML model along with its input data representation can be expected to perform good for any specific application. 

All ML models will be trained and tested on the same data sets. As such we utilized the "FabWave" dataset and pre-processed its data for a fair comparision between our ml models.

(List of contributions)
Our contributions include:

- Comparsion of dimensionless invariants to trees and multiple images as input for ML tasks
- Performance comparison of three data representation types (invariants, trees, images) in terms of classification of real engineering parts.
- Performance comparison of three data representaion types in terms of regression tasks aiming at predicting basic properties of real engineering parts.

(Paper outline)
The Rest of this paper is structured as follows. In Section 2 we review works related to our own, including the foundations for the ML models which are examined. Section 3 describes the problem statement, before the detail methodology used is explained in section 4. The following sections 5 and 6 first list our results, followed by an interpretation of the results and their implication for futur works. The paper ends with conclusing remarks.