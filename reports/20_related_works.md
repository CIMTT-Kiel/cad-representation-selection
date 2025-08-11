(*Introducing basic research area of 3D computer vision*)

Motivated by the success of deep learning on 2D images *Computer vision of 3D data* has seen extensive research recent years.
Today it plays a role in serveral domains including engineering, media production and autonomous driving; with many potential applications in each of them. Here we will focus on engineering applictations as the dataset we use (==ref fabwave==) contains mechanical parts in the STEP file format commonly used engineering desing and manufacturing workflows.

A wide variety of application have been developed analysing mechanical parts in terms of classification or regression tasks.

*Pernot* looked into the possibility to tell thin parts apart from parts with thin features and from other parts. They suggested that such a differentiation could deliver insights on how to prepare a CAD model for an FEM simulation. ([[202503241416|pernot2015thin]])

==must be checked==
Kim et. al. (A_survey_of_content_based_3D_shape_retrieval_methods) discusses serval approach of matching 3D CAD models to images such object may appear in.

Miles tried to identify a set of machining features hoping to support manufacturing purposes (==ref==). This application also touched on regressing feature dimensions (ref miles).

(==must be checked==)
When Yeo tried to identify machining features they set out to derive a feature vector describing each face of a part. Then calculating a score indicating wether or not this face points to the existence of a certain type of machining feature. ([[202309011725|yeo2021machining]]) 

([[202503241416|pernot2015thin]], [[202503251111|ahmed2019survey]]). 

Because of the complex nature of these applications much work remains in this field of research.

Where are two main reasons why *Computer Vision on 3D data* is still an active field of research:

1. 3D models can be represented in a variaty of ways, including *2D images*, *points clouds*, *meshes*, *voxels*, *descriptors* ([[202503251111|ahmed2019survey]]) and *trees* ([[202309011731|miles2022recursive]]).

As each representation type comes with its own set of drawbacks, like inefficient storage and computing, insufficient resolution or missing indifference for rotation and translation, it is a non-trivial tasks to choose the best representation type for any given application.

2. Applications of 3D computer vision vary a lot regarding their requirements.

This makes it necessary to develop custom solution for most new applications. For example, model retrieval from a database requires comparing models in terms of similarity. Meanwhile automatic CAD model adaptation requires the detection and differentiation of functional features of a part.

3. ML models often lack robustness regarding new input data. ([[202503251111|ahmed2019survey]])

In order to applicable the ML models have to be sufficiently accurate and robust even when facing unseen data.


# Comparison of approaches to analyse 3D data

In several other works have compared different approaches for analysing CAD models. To the best of our knowledeg no such paper yet considered all representation types possible. Not to mention all different algorthims applicable.

Different approaches have been used for a varitety of applictations.


When Miles et. al. utilized their tree representation of CAD models to a multi-feature detection, they compared it to a multiview-approach and single shot multibox detection. The goal here was to detected multiple features on simple CAD models. Comparison to other types of 3D data representation were not performed.







Work on CAD analysis is mainon going due to the complex nature of these tasks and the large potential thaly t lies in their application. To mention a few these include *model retrieval form a database* ([[202503241416|pernot2015thin]]), *automatic adaptation for simulation purposes* ([[202503241416|pernot2015thin]]) and *generation of manufacturing workflows*. Depending on the appliction and its specific context the CAD analysis approach has to be 

To the best of our knowledge the

## Representation of 3D-Data

This work mainly builds on the work of Miles, Kaiser and \<Autor> (==add references==) as they intorduce the three different concepts of CAD data representation we aim to compare. 

**Tree Data Structure**

Miles et. al. pointed out that B-Rep models are inherently structured hierarchicly. Thus topological entities, like vertices, edges and faces could be represented by a graph, along with geometrical information in the form of coordinate values. ([[202309011731|miles2022recursive]])

Representing the CAD models information in form of a tree allowed the application of a *Child-Sum-Tree-LSTM* (==reference==). The Tree-LSTM outputs a single feature vector that can be used as input for a variety of ML model. 

(Advantages and Disadvantages of trees and LSTMs)

**CAD Model description via dimensionless Invariants**

Another approach to obtain a feature vector was introduced by Kaiser et. al. Based on the Pi-Theorem (==reference==) they argue that any three-dimensional body can be described by a set of dimensionless invariants. 

Using a feature vector of invariants brings with it a loss of information about the represented CAD model. This is mainly because the calculation of these invariants requires meshing the CAD model using tetrahedrons, thus approximating its shape (==too harsh?==).  ([[202503101725|kaiser2018automated]])

**Images**

As image processing via convolutional neural networks its widely used, it is the most straight forward approach looked into. 

But apart from most other CNN application, here multiple images of one CAD model have to be processed per classification or regression output. This is because a single 2D image will always cover parts of a CAD model. Thus multiple images, showing several the model from different perspectives are required. None the less this does not guarante that all features of a CAD model will be represented, especially those lying on the inside of a part.