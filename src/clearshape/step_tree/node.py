"""
This module provides a definition for the `Node` class, used to represent elements within a hierarchical tree structure, particularly for managing STEP file data. Each node can represent various types of data entities, ranging from geometric components to atomic elements defining characters in a number or boolean values defined within STEP files.

The `Node` class supports creating nodes directly from STEP entity instances or atomic classes, ensuring nodes are correctly labeled and referenced for further processing and analysis in tree structures.
"""

__author__ = "Daniel Mansfeldt"

# standard libary

# Third party
from steputils import p21
import numpy as np


class Node:
    """
    A class to represent a node in a hierarchical tree structure for STEP file data.

    Nodes are designed to encapsulate information about STEP file entities or atomic classes, including their identifiers, labels, and STEP instance references when applicable.

    Attributes
    ----------
    id : int
        A unique identifier for the node.
    label : int
        A numeric label representing the node's type, encoded as an index based on `Node.valid_classes`.
    instance_reference : str, optional
        The STEP instance reference (e.g. "#100") of the node, defaulting to None if not applicable.

    Methods
    -------
    from_step_instance(id, instance)
        Class method to create a `Node` from a `p21.SimpleEntityInstance`.
    from_atomic(id, atomic_class)
        Class method to create a `Node` from an atomic class string.
    _encode_node_class(node_class)
        Class method to encode the node class into its corresponding numeric label.
    """

    valid_classes = [
        "*",  # 0
        "+",
        "-",
        ".",
        ".F.",
        ".T.",# 5
        "0",
        "1",
        "2",
        "3",
        "4",  # 10
        "5",
        "6",
        "7",
        "8",
        "9",  # 15
        "ADVANCED_FACE",
        "AXIS2_PLACEMENT_3D",
        "CARTESIAN_POINT",
        "CIRCLE",
        "CLOSED_SHELL", # 20
        "CYLINDRICAL_SURFACE",
        "DIRECTION",
        "EDGE_CURVE",
        "EDGE_LOOP",
        "FACE_BOUND", # 25
        "FACE_OUTER_BOUND",
        "LINE",
        "ORIENTED_EDGE",
        "PLANE",
        "VECTOR", # 30
        "VERTEX_POINT",
    ]

    def __init__(self, id: int, label: int, instance_reference: str = None) -> None:
        if type(id) != int: raise TypeError
        if type(label) != int: raise TypeError
        self.id = id
        self.label = label
        self.instance_reference = instance_reference

    @classmethod
    def from_step_instance(cls, id: int, instance: p21.SimpleEntityInstance):
        """
        Creates a `Node` object from a STEP entity instance.

        Parameters
        ----------
        id : int
            A unique identifier for the node.
        instance : p21.SimpleEntityInstance
            The STEP entity instance from which to create the node.

        Returns
        -------
        Node
            An instance of the `Node` class representing the given STEP instance.

        Raises
        ------
        ValueError
            If the entity's name is not in the list of valid classes.
        """
        if instance.entity.name not in cls.valid_classes:
            raise ValueError(
                f"STEP entity instance '{instance.entity.name}' is no valid node class!"
            )

        # label corresponds to the index of the entity class in cls.classes
        label = cls._encode_node_class(instance.entity.name)

        return cls(id=id, label=label, instance_reference=instance.ref)

    @classmethod
    def from_atomic(cls, id: int, atomic_class: str):
        """
        Creates a `Node` object from an atomic element given as a string.

        Parameters
        ----------
        id : int
            A unique identifier for the node.
        atomic_class : str
            The atomic class string to be represented by the node.

        Returns
        -------
        Node
            An instance of the `Node` class with the specified atomic class label.

        Raises
        ------
        TypeError
            If the `id` is not an integer.
        ValueError
            If the atomic class is not in the list of valid classes.
        """
        if not isinstance(id, int):
            raise TypeError("ID must be an integer!")
        
        if not cls._node_class_is_atomic(atomic_class):
            raise ValueError(f"Node class must refer to atomic. Node class is {atomic_class}")

        label = cls._encode_node_class(atomic_class)

        return cls(
            id=id,
            label=label,
        )

    @classmethod
    def _encode_node_class(cls, node_class: str):
        """
        Encodes a node class into its corresponding numeric label.

        Parameters
        ----------
        node_class : str
            The class of the node to encode.

        Returns
        -------
        int
            The index of the node class in `valid_classes`, serving as its numeric label.

        Raises
        ------
        ValueError
            If the `node_class` is not in the list of valid classes.
        """
        if node_class not in cls.valid_classes:
            raise ValueError(f"`{node_class}` is not a valid node class.")
        return int(np.where(np.isin(cls.valid_classes, node_class))[0])

    @classmethod
    def _node_class_is_atomic(cls, node_class:str):
        """
        Return True only if node class string refers to atomic. False otherwise.
        """
        atomic_classes = ["*","+","-",".",".F.",".T."] + [str(i) for i in range(10)]
        return node_class in atomic_classes